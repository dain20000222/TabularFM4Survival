# =============================================================================
# Landmark Random Survival Forest Survival Analysis Implementation
# =============================================================================
# This script performs landmark survival analysis using Random Survival Forest
# models with longitudinal data. Landmark analysis involves making survival
# predictions at specific time points (landmarks) by conditioning on patients
# being event-free up to that time. RSF is an ensemble method that extends
# random forests to survival data with censoring.
# =============================================================================

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import warnings
import random
import itertools
import torch

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration and Setup
# =============================================================================

# ============================================================
# Parse command line arguments
# ============================================================
parser = argparse.ArgumentParser(description='Run landmark RSF with different k values for landmark quantiles')
parser.add_argument('--k', type=float, default=20, help='Step size for landmark time percentiles (default: 20)')
args = parser.parse_args()

# ============================================================
# Reproducibility
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# Paths / IO
# ============================================================
data_dir = os.path.join("data")
dataset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

surv_probs_out_path = f"landmark_rsf_risk_{args.k}.csv"

print(f"Total datasets: {len(dataset_files)}")
print(f"Datasets to process: {dataset_files if dataset_files else 'None'}")
print(f"Using k={args.k} for landmark quantiles")

# ============================================================
# Helper functions
# ============================================================
def build_patient_outcome(df):
    """Get final follow-up time (t_event) and status_event for each pid."""
    followup_col = "time2" if "time2" in df.columns else "time"
    outcome = (
        df.sort_values(followup_col)
          .groupby("pid")
          .agg(
              t_event=(followup_col, "max"),
              status_event=("event", "last")
          )
    )
    return outcome

def apply_locf_landmark(df, landmark_time, covariates, patient_outcome, default_values):
    """
    Apply LOCF landmarking for a given landmark_time = tau_k.
    - Keep only patients with t_event > landmark_time (at risk at tau_k)
    - Residual time = t_event - landmark_time
    - Covariates taken from last observation at or before landmark_time (LOCF).
    """
    rows = []
    for pid, patient in df.groupby("pid"):
        patient = patient.sort_values("time")

        t_event = patient_outcome.loc[pid, "t_event"]
        status_event = patient_outcome.loc[pid, "status_event"]

        if t_event <= landmark_time:
            continue

        residual_time = float(t_event - landmark_time)
        event_indicator = int(status_event)

        valid = patient[patient["time"] <= landmark_time]
        row = {"pid": pid, "time": residual_time, "event": event_indicator}

        if not valid.empty:
            latest = valid.iloc[-1]
            for c in covariates:
                row[c] = latest[c]
        else:
            for c in covariates:
                row[c] = default_values[c]

        rows.append(row)

    return pd.DataFrame(rows)

# ============================================================
# Global accumulators
# ============================================================
all_surv_rows = []

# ============================================================
# Hyperparameter grid
# ============================================================
rsf_grid = {
    "n_estimators": [100, 200, 500, 1000],
    "max_depth": [3, 10, 50, None],
    "min_samples_split": [10, 50]
}
_rsfg_keys = list(rsf_grid.keys())
hp_grid = [dict(zip(_rsfg_keys, vals)) for vals in itertools.product(*(rsf_grid[k] for k in _rsfg_keys))]

# ============================================================
# Main loop per dataset
# ============================================================
for i, file_name in enumerate(dataset_files, start=1):
    dataset_name = file_name.replace(".csv", "")
    file_path = os.path.join(data_dir, file_name)

    print("=" * 70)
    print(f"\nProcessing dataset: {dataset_name} ({i}/{len(dataset_files)})")

    # Load dataset
    original_df = pd.read_csv(file_path)
    print(f"Original dataset shape: {original_df.shape}")

    # Determine event-time column for selecting landmark quantiles
    has_time2 = "time2" in original_df.columns
    time_col_for_tau = "time2" if has_time2 else "time"

    # Build patient outcome 
    patient_outcome = build_patient_outcome(original_df)

    # Split patients into train/val/test (70:15:15 by pid)
    original_pids = original_df["pid"].unique()
    pid_train, pid_temp = train_test_split(original_pids, test_size=0.3, random_state=SEED)
    pid_val, pid_test = train_test_split(pid_temp, test_size=0.5, random_state=SEED)
    print(f"Train patients: {len(pid_train)}, Val patients: {len(pid_val)}, Test patients: {len(pid_test)}")

    # Create split dataframes
    train_df = original_df[original_df["pid"].isin(pid_train)].copy()
    val_df   = original_df[original_df["pid"].isin(pid_val)].copy()
    test_df  = original_df[original_df["pid"].isin(pid_test)].copy()

    # --- Landmarks (taus) computed from TRAIN ONLY ---
    event_times_train = train_df.loc[train_df["event"] == 1, time_col_for_tau].values
    if len(event_times_train) == 0:
        print(f"Skipping {dataset_name}: No events observed in TRAIN split")
        continue

    # Define landmarks using k parameter
    percentiles = np.arange(args.k, 100, args.k)

    # Drop percentiles that effectively round to 100 (floating-point artifact)
    percentiles = [p for p in percentiles if round(p, 2) < 100]

    times = np.percentile(event_times_train, percentiles)

    taus = list(times)
    num_eval_times = len(taus)

    print(f"Landmarks (taus) from TRAIN: {[f'{t:.6g}' for t in taus]}")
    print(f"Number of evaluation times: {num_eval_times} (k={args.k})")

    # Choose raw feature columns
    num_cols = [c for c in original_df.columns if c.startswith("num_")]
    fac_cols = [c for c in original_df.columns if c.startswith("fac_")]
    raw_feature_cols = num_cols + fac_cols

    # One-hot encode categorical features
    x_train_raw = train_df[raw_feature_cols]
    x_val_raw = val_df[raw_feature_cols]
    x_test_raw = test_df[raw_feature_cols]

    x_train_ohe = pd.get_dummies(x_train_raw, drop_first=True)
    x_val_ohe = pd.get_dummies(x_val_raw, drop_first=True)
    x_test_ohe = pd.get_dummies(x_test_raw, drop_first=True)

    all_columns = x_train_ohe.columns.union(x_val_ohe.columns).union(x_test_ohe.columns)
    x_train_ohe = x_train_ohe.reindex(columns=all_columns, fill_value=0)
    x_val_ohe = x_val_ohe.reindex(columns=all_columns, fill_value=0)
    x_test_ohe = x_test_ohe.reindex(columns=all_columns, fill_value=0)

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    x_train_proc = pd.DataFrame(
        imputer.fit_transform(x_train_ohe),
        columns=all_columns,
        index=x_train_ohe.index,
    )
    x_val_proc = pd.DataFrame(
        imputer.transform(x_val_ohe),
        columns=all_columns,
        index=x_val_ohe.index,
    )
    x_test_proc = pd.DataFrame(
        imputer.transform(x_test_ohe),
        columns=all_columns,
        index=x_test_ohe.index,
    )

    # Rebuild processed longitudinal frames
    non_feature_cols = [c for c in original_df.columns if c not in raw_feature_cols]

    train_processed = pd.concat(
        [train_df[non_feature_cols].reset_index(drop=True),
         x_train_proc.reset_index(drop=True)],
        axis=1,
    )
    val_processed = pd.concat(
        [val_df[non_feature_cols].reset_index(drop=True),
         x_val_proc.reset_index(drop=True)],
        axis=1,
    )
    test_processed = pd.concat(
        [test_df[non_feature_cols].reset_index(drop=True),
         x_test_proc.reset_index(drop=True)],
        axis=1,
    )

    # Default feature values from processed TRAIN set
    feature_cols_processed = all_columns.tolist()
    default_values = {col: float(train_processed[col].mean()) for col in feature_cols_processed}

    # ============================================================
    # Hyperparameter tuning loop
    # ============================================================
    for hp_config in hp_grid:
        print(f"\n{'='*70}")
        print(f"Testing hyperparameter configuration: {hp_config}")
        print(f"{'='*70}")

        # Process both validation and test sets
        for eval_set_name, eval_processed in [("val", val_processed), ("test", test_processed)]:
            print(f"\nEvaluating on {eval_set_name} set...")

            # Get patient IDs for this evaluation set
            if eval_set_name == "val":
                eval_pids = pid_val
            else:
                eval_pids = pid_test

            # Store predictions for each landmark
            patient_predictions = {}

            # Process each landmark tau_j
            for cond_tau_idx in range(len(taus)):
                cond_tau = taus[cond_tau_idx]

                print(f"\nProcessing landmark tau{cond_tau_idx} (t={cond_tau:.4g})")

                # Patients at risk at this landmark
                at_risk_pids_train = patient_outcome.loc[patient_outcome.index.isin(pid_train) & (patient_outcome['t_event'] > cond_tau)].index.values
                at_risk_pids_eval = patient_outcome.loc[patient_outcome.index.isin(eval_pids) & (patient_outcome['t_event'] > cond_tau)].index.values

                print(f"Train patients at risk: {len(at_risk_pids_train)}, Eval patients at risk: {len(at_risk_pids_eval)}")

                if len(at_risk_pids_train) < 10 or len(at_risk_pids_eval) < 1:
                    print(f"Skipping tau{cond_tau_idx}: insufficient data")
                    continue

                # Apply LOCF landmarking
                train_at_risk = train_processed[train_processed['pid'].isin(at_risk_pids_train)].copy()
                eval_at_risk = eval_processed[eval_processed['pid'].isin(at_risk_pids_eval)].copy()

                train_df_tau = apply_locf_landmark(
                    train_at_risk,
                    landmark_time=cond_tau,
                    covariates=feature_cols_processed,
                    patient_outcome=patient_outcome,
                    default_values=default_values
                )

                eval_df_tau = apply_locf_landmark(
                    eval_at_risk,
                    landmark_time=cond_tau,
                    covariates=feature_cols_processed,
                    patient_outcome=patient_outcome,
                    default_values=default_values
                )

                print(f"Train: {len(train_df_tau)} patients, Events: {train_df_tau['event'].sum()}")
                print(f"Eval: {len(eval_df_tau)} patients, Events: {eval_df_tau['event'].sum()}")

                if len(train_df_tau) < 10 or train_df_tau['event'].sum() < 5 or len(eval_df_tau) < 1:
                    print(f"Skipping tau{cond_tau_idx}: insufficient training data")
                    continue

                # Prepare data for RSF model
                x_train_tau = train_df_tau[feature_cols_processed].copy()
                x_eval_tau = eval_df_tau[feature_cols_processed].copy()

                # Remove constant columns
                constant_cols = x_train_tau.columns[x_train_tau.std() == 0]
                if len(constant_cols) > 0:
                    x_train_tau = x_train_tau.drop(columns=constant_cols)
                    x_eval_tau = x_eval_tau.drop(columns=constant_cols)

                y_train_tau = Surv.from_dataframe('event', 'time', train_df_tau)

                try:
                    # Train RSF model
                    rsf_model = RandomSurvivalForest(
                        random_state=SEED,
                        n_jobs=-1,
                        **hp_config
                    )
                    rsf_model.fit(x_train_tau, y_train_tau)

                    # Risk scores (higher = higher risk)
                    risk_scores = rsf_model.predict(x_eval_tau)

                    # Survival functions
                    surv_funcs = rsf_model.predict_survival_function(x_eval_tau)

                    # Store predictions
                    for eval_idx, pid in enumerate(eval_df_tau['pid'].values):
                        if pid not in patient_predictions:
                            patient_predictions[pid] = {}

                        risk = float(risk_scores[eval_idx])
                        surv_func = surv_funcs[eval_idx]

                        # Calculate survival probabilities at future landmarks
                        surv_probs = {}
                        for future_idx in range(cond_tau_idx + 1, len(taus)):
                            future_tau = taus[future_idx]
                            residual_time = future_tau - cond_tau
                            try:
                                surv_prob = float(surv_func(residual_time))
                            except:
                                surv_prob = np.nan
                            surv_probs[future_idx] = surv_prob

                        patient_predictions[pid][cond_tau_idx] = {
                            'risk': risk,
                            'surv_probs': surv_probs
                        }

                except Exception as e:
                    print(f"Error fitting RSF model at tau{cond_tau_idx}: {e}")
                    continue

            # Convert predictions to output format
            for pid in patient_predictions:
                for cond_tau_idx in patient_predictions[pid]:
                    cond_tau = taus[cond_tau_idx]
                    pred = patient_predictions[pid][cond_tau_idx]

                    # Risk row (EXCLUDE last landmark)
                    if cond_tau_idx < len(taus) - 1:
                        all_surv_rows.append({
                            "pid": pid,
                            "tau": float(cond_tau),
                            "t": np.nan,
                            "risk": pred['risk'],
                            "surv_prob": np.nan,
                            "dataset": dataset_name,
                            "hyperparameter": str(hp_config),
                            "eval_set": eval_set_name
                        })

                    # Survival probability rows
                    # Skip survival probs for last landmark (only need risk)
                    if cond_tau_idx < len(taus) - 1:
                        for future_idx in pred['surv_probs']:
                            future_tau = taus[future_idx]
                            all_surv_rows.append({
                                "pid": pid,
                                "tau": float(cond_tau),
                                "t": float(future_tau),
                                "risk": np.nan,
                                "surv_prob": pred['surv_probs'][future_idx],
                                "dataset": dataset_name,
                                "hyperparameter": str(hp_config),
                                "eval_set": eval_set_name
                            })

# ============================================================
# Write output CSV
# ============================================================
if len(all_surv_rows) > 0:
    all_surv_df = pd.DataFrame(
        all_surv_rows,
        columns=["pid", "tau", "t", "risk", "surv_prob", "dataset", "hyperparameter", "eval_set"]
    )

    # Add k column
    all_surv_df["k"] = args.k

    # Sort
    all_surv_df = (
        all_surv_df.sort_values(["k", "hyperparameter", "eval_set", "dataset", "pid", "tau", "t"], na_position="last")
                   .reset_index(drop=True)
    )

    all_surv_df.to_csv(surv_probs_out_path, index=False)
    print(f"\nSaved ALL datasets survival probabilities to: {surv_probs_out_path}")
else:
    print("\nNo survival rows generated; nothing saved.")

print("\n" + "=" * 70)
print("Done.")

