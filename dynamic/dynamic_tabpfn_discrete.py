# =============================================================================
# Dynamic TabPFN Survival Analysis Implementation  
# =============================================================================
# This script performs dynamic survival analysis using TabPFN (Tabular Prior-data
# Fitted Networks) with longitudinal data. It implements landmark analysis where
# survival predictions are made at multiple time points using Last Observation
# Carried Forward (LOCF) for missing covariate values. TabPFN leverages 
# transformer-based architectures with prior-data fitted networks.
# =============================================================================

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from tabpfn import TabPFNClassifier
import warnings
import random
import csv
import torch
from sksurv.nonparametric import kaplan_meier_estimator

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration and Setup
# =============================================================================

# =============================================================================
# Configuration and Setup
# =============================================================================

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run dynamic TabPFN with different landmark binning strategies')
parser.add_argument('--k', type=float, default=20, help='Step size for landmark time percentiles (default: 20)')
parser.add_argument('--binning', type=str, default='km_quantile', choices=['quantile', 'km_quantile', 'pooled'],
                    help='Landmark binning strategy: quantile or km_quantile (default: km_quantile)')
parser.add_argument('--oof_folds', type=int, default=0, help='DOOF: folds of the out-of-fold predictions of the training patients (0: none)')
args = parser.parse_args()

# =============================================================================
# Reproducibility Settings
# =============================================================================

# Set random seeds for reproducibility across all libraries
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# Data Configuration and I/O Setup
# =============================================================================

# Directory containing longitudinal datasets with time-varying covariates
data_dir = os.path.join("data_mean")
dataset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# Output file path for risk prediction results
binning_suffix = f"_{args.binning}" if args.binning != "quantile" else ""
k_token = f"{float(args.k):.8g}"
risk_out_path = f"dynamic_tabpfn_discrete_risk_{k_token}{binning_suffix}.csv"

print(f"Total datasets: {len(dataset_files)}")
print(f"Datasets to process: {dataset_files if dataset_files else 'None'}")
print(f"Using k={args.k} with binning={args.binning} for landmark generation")

# =============================================================================
# Hyperparameter Configuration
# =============================================================================

# Define hyperparameter combinations for TabPFN feature engineering:
# - quantile_info: Include landmark time index as a feature
# - time_since: Include time since last observation as a feature
hyperparameters = [
    {'quantile_info': False, 'time_since': False},  # Baseline: covariates + eval_time only
    {'quantile_info': False, 'time_since': True},   # Add time since last observation
    {'quantile_info': True,  'time_since': False},  # Add landmark index information
    {'quantile_info': True,  'time_since': True},   # Full feature set
]

# =============================================================================
# Helper Functions
# =============================================================================
def k_to_percentile_grid(k: float, tol: float = 1e-6):
    """Return (percentiles, n_bins, K_int_or_None) for a step size k."""
    kf = float(k)
    if not (0.0 < kf < 100.0):
        raise ValueError(f"k must be in (0, 100), got {k}")

    K = 100.0 / kf
    K_int = int(round(K))
    if np.isclose(K, K_int, rtol=0.0, atol=tol) and K_int >= 2:
        percentiles = (np.arange(1, K_int) * (100.0 / K_int))
        return percentiles, (K_int - 1), K_int

    percentiles = np.arange(kf, 100.0, kf)
    return percentiles, int(len(percentiles)), None


def build_landmarks_from_train(train_df, time_col_for_tau, k, binning):
    """Build landmark times from train data using quantile or KM-quantile bins."""
    if binning == "pooled":  # POOLED (11-06): pooled measurement and event times, km_landmarks.pooled_taus
        from km_landmarks import pooled_taus
        return pooled_taus(train_df, time_col_for_tau, k)
    patient_outcome = (
        train_df.sort_values(time_col_for_tau)
        .groupby("pid")
        .agg(
            t_event=(time_col_for_tau, "max"),
            status_event=("event", "max"),
        )
    )

    event_times = patient_outcome.loc[patient_outcome["status_event"] == 1, "t_event"].to_numpy(dtype=float)
    if len(event_times) == 0:
        return np.array([], dtype=float)

    percentiles, n_bins, _ = k_to_percentile_grid(k)

    if binning == "quantile":
        taus = np.percentile(event_times, percentiles)
    else:
        km_time, km_surv = kaplan_meier_estimator(
            patient_outcome["status_event"].astype(bool).to_numpy(),
            patient_outcome["t_event"].to_numpy(dtype=float),
        )
        if len(km_time) == 0:
            taus = np.percentile(event_times, percentiles)
        else:
            km_range = (float(km_surv[0]), float(km_surv[-1]))
            surv_levels = np.linspace(km_range[0], km_range[1], n_bins + 2)[1:-1]
            taus = []
            for s in surv_levels:
                idx = np.searchsorted(-km_surv, -s)
                if idx < len(km_time):
                    taus.append(float(km_time[idx]))
            taus = np.array(taus) if taus else np.percentile(event_times, percentiles)

    return np.unique(np.asarray(taus, dtype=float))


def get_locf_covariates(df, pid, tau, feature_cols, default_values, return_time_since=False):
    """
    Get Last Observation Carried Forward (LOCF) covariates for a patient at time tau.
    
    Parameters:
    - df: DataFrame with longitudinal data
    - pid: patient ID
    - tau: landmark time
    - feature_cols: list of feature column names
    - default_values: dict of default values for features
    - If return_time_since=True, also return time_since_last. If no observation exists before tau, defaults are used and time_since_last = tau.
    
    Returns:
    - array of covariate values
    """
    patient_data = df[df["pid"] == pid]
    eligible = patient_data[patient_data["time"] <= tau]
    
    if not eligible.empty:
        latest = eligible.loc[eligible["time"].idxmax()]
        last_time = float(latest["time"])
        values = []
        for c in feature_cols:
            val = latest[c] if c in latest.index else np.nan
            values.append(default_values[c] if pd.isna(val) else val)
        x = np.array(values, dtype=float)

        if return_time_since:
            return x, float(tau - last_time)
        return x
    else:
        # No observation before tau → defaults + time_since_last = tau
        x = np.array([default_values[c] for c in feature_cols], dtype=float)
        if return_time_since:
            return x, float(tau)
        return x

def get_positive_class_probs(probs):
    """Extract positive-class (hazard=1) probabilities."""
    if hasattr(probs, "columns"):
        if 1 in probs.columns:
            return probs[1].to_numpy(dtype=float)
        if "1" in probs.columns:
            return probs["1"].to_numpy(dtype=float)
        if probs.shape[1] >= 2:
            return probs.iloc[:, 1].to_numpy(dtype=float)
        return probs.iloc[:, 0].to_numpy(dtype=float)

    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.shape[1] >= 2:
        return arr[:, 1]
    return arr[:, 0]

def hazards_to_survival_df(pids, eval_times, hazard_probs):
    """Convert row-wise hazard predictions into monotone survival by pid."""
    tmp = pd.DataFrame({
        "pid": np.asarray(pids),
        "t": np.asarray(eval_times, dtype=float),
        "hazard_prob": np.clip(np.asarray(hazard_probs, dtype=float), 0.0, 1.0),
    })
    tmp = tmp.sort_values(["pid", "t"], kind="mergesort")
    # Conditional survival from landmark tau: S_j = prod_{m<=j}(1 - h_m)
    tmp["surv_prob"] = tmp.groupby("pid")["hazard_prob"].transform(
        lambda x: np.cumprod(1.0 - x.to_numpy(dtype=float))
    )
    return tmp


def dh_oof_rows(train_processed, taus, feature_cols, default_values, quantile_info, time_since, time_col, fit_fn, predict_fn,
                dataset_name, hyperparameter, n_folds):
    """DOOF (11-03): out-of-fold survival predictions of the TRAINING patients, for post-hoc calibration
    (../tmlr_ipcw_loss_10_25/calibrate_dynamic.py). Training patients split into n_folds folds (KFold by patient, seed 42,
    as dynamic_cf_w.py); each fold predicted by the model fitted on the hazard rows of the other folds (same row
    construction and the same features as the real fit), at every landmark, for the patients at risk at tau."""
    from sklearn.model_selection import KFold
    pids = np.unique(train_processed["pid"])
    t_event = train_processed.groupby("pid")[time_col].max()
    rows = []
    for i_fit, i_pred in KFold(n_folds, shuffle=True, random_state=42).split(pids):
        fit_df = train_processed[train_processed["pid"].isin(pids[i_fit])]
        pred_df = train_processed[train_processed["pid"].isin(pids[i_pred])]
        Xs, ys = [], []
        for ti in range(len(taus) - 1):
            X, y = construct_binary_trainset_for_landmark(fit_df, taus[ti], taus[ti:], feature_cols, default_values, taus,
                                                          quantile_info, time_since, time_col)
            if len(X) == 0:
                continue
            X = X.copy()
            X["tau"] = float(taus[ti])
            Xs.append(X)
            ys.append(y)
        if not Xs:
            continue
        model = fit_fn(pd.concat(Xs, axis=0, ignore_index=True), pd.concat(ys, axis=0, ignore_index=True))
        for ti in range(len(taus) - 1):
            tau = float(taus[ti])
            sub = pred_df[pred_df["pid"].isin(t_event.index[t_event > tau])]
            X, ppids, times = construct_binary_testset_for_landmark(sub, taus[ti], taus[ti:], feature_cols, default_values, taus,
                                                                    quantile_info, time_since, time_col)
            if len(X) == 0:
                continue
            X = X.copy()
            X["tau"] = tau
            tmp = hazards_to_survival_df(ppids, times, predict_fn(model, X))
            for p, t, sv in zip(tmp["pid"].to_numpy(), tmp["t"].to_numpy(), tmp["surv_prob"].to_numpy()):
                if not np.isclose(float(t), tau):
                    rows.append({"pid": int(p), "tau": tau, "t": float(t), "risk": np.nan, "surv_prob": float(sv),
                                 "dataset": dataset_name, "hyperparameter": str(hyperparameter), "eval_set": "oof", "k": args.k})
    print(f"  [DOOF] {len(rows)} out-of-fold survival predictions of {len(pids)} training patients")
    return rows

def construct_binary_trainset_for_landmark(df, tau, future_times, feature_cols, default_values, taus, quantile_info, time_since, followup_col="time2"):
    """
    Construct binary training dataset for a specific landmark tau.
    
    For each patient with T_i > tau OR delta_i = 1:
    - Get LOCF covariates at tau
    - For each future time t in future_times (where t >= tau):
      - Create row: (X_tau, t, [t_idx], label) - t_idx optional
    - Label = 1 only for the event interval (discrete hazard), else 0
    
    Parameters:
    - df: processed longitudinal DataFrame
    - tau: landmark time
    - future_times: list of future evaluation times (>= tau)
    - feature_cols: list of feature column names
    - default_values: dict of default values
    - taus: list of all landmark times (for finding t_idx)
    - quantile_info: whether to include t_idx as a feature
    - followup_col: column name for follow-up time
    
    Returns:
    - X_train: DataFrame with features (covariates + eval_time [+ t_idx])
    - y_train: Series with binary labels
    """
    # Get patient-level outcomes
    patient_outcome = (
        df.sort_values(followup_col)
          .groupby("pid")
          .agg(
              t_event=(followup_col, "max"),
              status_event=("event", "max"),
          )
    )

    at_risk_pids = patient_outcome.index[
        (patient_outcome["t_event"] > tau)
    ]
    
    rows = []
    labels = []
    
    for pid in at_risk_pids:
        t_i = float(patient_outcome.loc[pid, "t_event"])
        d_i = int(patient_outcome.loc[pid, "status_event"])
        
        # Get LOCF covariates at tau
        if time_since:
            x_tau, tsl = get_locf_covariates(
                df, pid, tau, feature_cols, default_values, return_time_since=True
            )
        else:
            x_tau = get_locf_covariates(df, pid, tau, feature_cols, default_values)
                
        # For each future evaluation interval (t_start, t]
        prev_t = float(tau)
        for t in future_times:
            if t <= tau:
                continue  # Skip times before landmark

            t_end = float(t)
            t_start = float(prev_t)

            # Not at risk at interval start
            if t_i <= t_start:
                break

            # Event occurs in this interval: add one positive row then stop
            if d_i == 1 and (t_i <= t_end):
                label = int(t_i > t_start)
                if label == 1:
                    extras = [t_end]  # eval_time

                    if quantile_info:
                        t_idx = next((i for i, tau_val in enumerate(taus) if np.isclose(tau_val, t_end)), -1)
                        extras.append(float(t_idx))

                    if time_since:
                        extras.append(float(tsl))  # time_since_last

                    row = np.concatenate([x_tau, extras])
                    rows.append(row)
                    labels.append(1)
                break

            # Survived this whole interval -> negative label
            if t_i > t_end:
                label = 0
            else:
                # Censored inside interval -> interval label is undefined
                break

            extras = [t_end]  # eval_time

            if quantile_info:
                t_idx = next((i for i, tau_val in enumerate(taus) if np.isclose(tau_val, t_end)), -1)
                extras.append(float(t_idx))

            if time_since:
                extras.append(float(tsl))  # time_since_last

            row = np.concatenate([x_tau, extras])
            
            rows.append(row)
            labels.append(label)
            prev_t = t_end
    
    if len(rows) == 0:
        feature_names = list(feature_cols) + ["eval_time"] \
                + (["t_idx"] if quantile_info else []) \
                + (["time_since_last"] if time_since else [])
        return pd.DataFrame(columns=feature_names), pd.Series(dtype=int)
    
    feature_names = list(feature_cols) + ["eval_time"] \
                + (["t_idx"] if quantile_info else []) \
                + (["time_since_last"] if time_since else [])
    X_train = pd.DataFrame(rows, columns=feature_names)
    y_train = pd.Series(labels, dtype=int)
    
    return X_train, y_train


def construct_binary_testset_for_landmark(df, tau, future_times, feature_cols, default_values, taus, quantile_info, time_since, followup_col="time2"):
    """
    Construct binary test dataset for a specific landmark tau.
    
    For each patient with T_i > tau OR delta_i = 1:
    - Get LOCF covariates at tau
    - For each future time t in future_times (where t >= tau):
      - Create row: (X_tau, t, [t_idx]) - t_idx optional
    
    Returns:
    - X_test: DataFrame with features
    - pids: array of patient IDs corresponding to each row
    - eval_times: array of evaluation times corresponding to each row
    """
    # Get patient-level outcomes
    patient_outcome = (
        df.sort_values(followup_col)
          .groupby("pid")
          .agg(
              t_event=(followup_col, "max"),
              status_event=("event", "max"),
          )
    )
    
    # Patients at risk at tau: T_i > tau OR delta_i = 1
    # at_risk_pids = patient_outcome.index[
    #     (patient_outcome["t_event"] > tau) | (patient_outcome["status_event"] == 1)
    # ]

    at_risk_pids = patient_outcome.index[
        (patient_outcome["t_event"] > tau)
    ]
    
    rows = []
    pids = []
    eval_times = []
    
    for pid in at_risk_pids:
        # Get LOCF covariates at tau
        if time_since:
            x_tau, tsl = get_locf_covariates(
                df, pid, tau, feature_cols, default_values, return_time_since=True
            )
        else:
            x_tau = get_locf_covariates(df, pid, tau, feature_cols, default_values)

        # For each future evaluation time
        for t in future_times:
            if t <= tau:
                continue
            
            extras = [float(t)]

            if quantile_info:
                t_idx = next((i for i, tau_val in enumerate(taus) if np.isclose(tau_val, t)), -1)
                extras.append(float(t_idx))

            if time_since:
                extras.append(float(tsl))

            row = np.concatenate([x_tau, extras])
            
            rows.append(row)
            pids.append(pid)
            eval_times.append(t)
    
    if len(rows) == 0:
        feature_names = list(feature_cols) + ["eval_time"] \
                + (["t_idx"] if quantile_info else []) \
                + (["time_since_last"] if time_since else [])
        return pd.DataFrame(columns=feature_names), np.array([]), np.array([])
    
    feature_names = list(feature_cols) + ["eval_time"] \
                + (["t_idx"] if quantile_info else []) \
                + (["time_since_last"] if time_since else [])
    X_test = pd.DataFrame(rows, columns=feature_names)
    
    return X_test, np.array(pids), np.array(eval_times)


# ============================================================
# Global accumulators for CSV output
# ============================================================
all_risk_rows = []
all_oof_rows = []  # DOOF


# ============================================================
# Hyperparameter loop
# ============================================================
for hyperparameter in hyperparameters:
    quantile_info = hyperparameter['quantile_info']
    time_since = hyperparameter['time_since']
    print("\n" + "="*70)
    
    # ============================================================
    # Main loop per dataset
    # ============================================================
    for i, file_name in enumerate(dataset_files, start=1):
        dataset_name = file_name.replace(".csv", "")
        file_path = os.path.join(data_dir, file_name)

        print("=" * 70)
        print(f"\nProcessing dataset: {dataset_name} ({i}/{len(dataset_files)})")

        # -----------------------------
        # Load dataset
        # -----------------------------
        original_df = pd.read_csv(file_path)
        print(f"Original dataset shape: {original_df.shape}")

        # Determine event-time column
        has_time2 = "time2" in original_df.columns
        time_col_for_tau = "time2" if has_time2 else "time"

        # -----------------------------
        # Split patients into train/val/test (70:15:15 by pid)
        # -----------------------------
        original_pids = original_df["pid"].unique()
        pid_train, pid_temp = train_test_split(original_pids, test_size=0.3, random_state=SEED)
        pid_val, pid_test = train_test_split(pid_temp, test_size=0.5, random_state=SEED)
        print(f"Train patients: {len(pid_train)}, Val patients: {len(pid_val)}, Test patients: {len(pid_test)}")

        # -----------------------------
        # Choose raw feature columns (num_*, fac_*)
        # -----------------------------
        num_cols = [c for c in original_df.columns if c.startswith("num_")]
        fac_cols = [c for c in original_df.columns if c.startswith("fac_")]
        raw_feature_cols = num_cols + fac_cols

        # Split to train/val/test rows
        train_df = original_df[original_df["pid"].isin(pid_train)].copy()
        val_df = original_df[original_df["pid"].isin(pid_val)].copy()
        test_df = original_df[original_df["pid"].isin(pid_test)].copy()

        # --- Define landmarks using TRAIN ONLY ---
        taus = build_landmarks_from_train(
            train_df=train_df,
            time_col_for_tau=time_col_for_tau,
            k=args.k,
            binning=args.binning,
        )

        if len(taus) == 0:
            print(f"Skipping {dataset_name}: No events observed in TRAIN split")
            continue

        num_eval_times = len(taus)

        if num_eval_times < 2:
            print(f"Skipping {dataset_name}: Need at least 2 landmark times, got {num_eval_times}")
            continue

        print(f"Landmarks (taus) from TRAIN: {[f'{t:.6g}' for t in taus]}")
        print(f"Number of evaluation times: {num_eval_times} (k={args.k}, binning={args.binning})")

        # -----------------------------
        # One-hot encode categorical (fac_) and align train/val/test columns
        # -----------------------------
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

        # -----------------------------
        # Impute missing values (fit on train only)
        # -----------------------------
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

        # -----------------------------
        # Rebuild processed longitudinal frames
        # -----------------------------
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

        # -----------------------------
        # Default feature values from processed TRAIN set only
        # -----------------------------
        feature_cols_processed = all_columns.tolist()
        default_values = {col: float(train_processed[col].mean()) for col in feature_cols_processed}

        # -----------------------------
        # Build ONE combined training set across ALL landmarks (taus),
        # then train ONE TabPFN model per dataset per hyperparameter.
        # -----------------------------
        combined_X_parts = []
        combined_y_parts = []

        for tau_idx in range(len(taus) - 1):
            tau = taus[tau_idx]

            # Future evaluation times: all taus from current tau onwards (including last tau)
            future_times = taus[tau_idx:]

            print(f"\n{'='*70}")
            print(f"Adding training data for landmark tau{tau_idx} (t={tau:.6g})")
            print(f"Future eval times: {[f'{t:.6g}' for t in future_times]}")

            # -----------------------------
            # Build training data for this landmark
            # -----------------------------
            X_train, y_train = construct_binary_trainset_for_landmark(
                train_processed, tau, future_times, feature_cols_processed, default_values, taus,
                quantile_info, time_since, time_col_for_tau
            )

            if len(X_train) == 0:
                print(f"No training data for tau{tau_idx}; skipping")
                continue

            # Add tau as a feature so a single model can learn across landmarks
            X_train = X_train.copy()
            X_train["tau"] = float(tau)

            combined_X_parts.append(X_train)
            combined_y_parts.append(y_train)

        if len(combined_X_parts) == 0:
            print(f"No training data across any tau; skipping dataset {dataset_name}")
            continue

        X_train_all = pd.concat(combined_X_parts, axis=0, ignore_index=True)
        y_train_all = pd.concat(combined_y_parts, axis=0, ignore_index=True)

        # Print class distribution
        n_ones = y_train_all.sum()
        n_zeros = len(y_train_all) - n_ones
        print(f"Training samples (train only, all taus): {len(y_train_all)} (event: {n_ones}, no event: {n_zeros})")

        # Sample if too large (TabPFN limit: 50000 rows)
        if len(X_train_all) > 50000:
            print(f"Sampling 50000 rows from {len(X_train_all)} total rows...")
            sample_idx = np.random.choice(len(X_train_all), size=50000, replace=False)
            X_train_all = X_train_all.iloc[sample_idx].reset_index(drop=True)
            y_train_all = y_train_all.iloc[sample_idx].reset_index(drop=True)

        # -----------------------------
        # Train TabPFN model (single model for all taus)
        # -----------------------------
        X_train_array = X_train_all.to_numpy(dtype=float)
        y_train_array = y_train_all.to_numpy(dtype=int)

        tabpfn_model = TabPFNClassifier(random_state=SEED)
        tabpfn_model.fit(X_train_array, y_train_array)
        if args.oof_folds:  # DOOF
            def _fit(X, y):
                if len(X) > 50000:
                    idx = np.random.choice(len(X), size=50000, replace=False)
                    X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
                m = TabPFNClassifier(random_state=SEED)
                m.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=int))
                return m
            def _pred(m, X):
                p = m.predict_proba(X.to_numpy(dtype=float))
                return p[:, 1] if p.shape[1] > 1 else p[:, 0]
            all_oof_rows += dh_oof_rows(train_processed, taus, feature_cols_processed, default_values, quantile_info, time_since,
                                        time_col_for_tau, _fit, _pred, dataset_name, hyperparameter, args.oof_folds)

        # -----------------------------
        # For each landmark tau (excluding last tau), run inference using the same model
        # -----------------------------
        for tau_idx in range(len(taus) - 1):
            tau = taus[tau_idx]

            # Future evaluation times: all taus from current tau onwards (including last tau)
            future_times = taus[tau_idx:]

            print(f"\n{'='*70}")
            print(f"Inference for landmark tau{tau_idx} (t={tau:.6g})")
            print(f"Future eval times: {[f'{t:.6g}' for t in future_times]}")

            # Get val patients at risk at tau (T_i > tau) - for prediction filtering only
            val_patient_outcome = (
                val_processed.sort_values(time_col_for_tau)
                  .groupby("pid")
                  .agg(
                      t_event=(time_col_for_tau, "max"),
                      status_event=("event", "max"),
                  )
            )
            val_at_risk_pids = val_patient_outcome.index[
                val_patient_outcome["t_event"] > tau
            ]

            # Get test patients at risk at tau (T_i > tau) - for prediction filtering only
            test_patient_outcome = (
                test_processed.sort_values(time_col_for_tau)
                  .groupby("pid")
                  .agg(
                      t_event=(time_col_for_tau, "max"),
                      status_event=("event", "max"),
                  )
            )
            test_at_risk_pids = test_patient_outcome.index[
                test_patient_outcome["t_event"] > tau
            ]

            # -----------------------------
            # Evaluate on validation set (only at-risk patients)
            # -----------------------------
            val_processed_at_risk = val_processed[val_processed["pid"].isin(val_at_risk_pids)]
            X_val, val_pids, val_eval_times = construct_binary_testset_for_landmark(
                val_processed_at_risk, tau, future_times, feature_cols_processed, default_values, taus,
                quantile_info, time_since, time_col_for_tau
            )

            if len(X_val) > 0:
                # Add tau as a feature (must match training columns)
                X_val = X_val.copy()
                X_val["tau"] = float(tau)

                X_val_array = X_val.to_numpy(dtype=float)
                probs = tabpfn_model.predict_proba(X_val_array)
                hazard_probs = get_positive_class_probs(probs)
                tmp = hazards_to_survival_df(val_pids, val_eval_times, hazard_probs)

                # -----------------------------
                # RISK = average over future timepoints of (1 - surv_prob)
                # Future timepoints here are strictly after tau (exclude t == tau)
                # -----------------------------
                tau_float = float(tau)
                tmp_future = tmp[~np.isclose(tmp["t"].to_numpy(), tau_float)].copy()
                tmp_future["risk_component"] = 1.0 - tmp_future["surv_prob"]

                # Mean risk per pid (one risk score per pid per tau)
                risk_by_pid = tmp_future.groupby("pid")["risk_component"].mean()

                # Store risk score row (t is NaN) once per pid
                for pid, risk in risk_by_pid.items():
                    all_risk_rows.append({
                        "pid": int(pid),
                        "tau": float(tau),
                        "t": np.nan,
                        "risk": float(risk),
                        "surv_prob": np.nan,
                        "dataset": dataset_name,
                        "hyperparameter": str(hyperparameter),
                        "eval_set": "val",
                        "k": args.k
                    })

                # Store survival probability for strictly future times (exclude tau)
                for pid, t, surv_prob in zip(tmp["pid"].to_numpy(), tmp["t"].to_numpy(), tmp["surv_prob"].to_numpy()):
                    if not np.isclose(float(t), tau_float):
                        all_risk_rows.append({
                            "pid": int(pid),
                            "tau": float(tau),
                            "t": float(t),
                            "risk": np.nan,
                            "surv_prob": float(surv_prob),
                            "dataset": dataset_name,
                            "hyperparameter": str(hyperparameter),
                            "eval_set": "val",
                            "k": args.k
                        })
           
            # -----------------------------
            # Evaluate on test set (using same model, only at-risk patients)
            # -----------------------------
            test_processed_at_risk = test_processed[test_processed["pid"].isin(test_at_risk_pids)]
            X_test, test_pids, test_eval_times = construct_binary_testset_for_landmark(
                test_processed_at_risk, tau, future_times, feature_cols_processed, default_values, taus,
                quantile_info, time_since, time_col_for_tau
            )

            if len(X_test) > 0:
                # Add tau as a feature (must match training columns)
                X_test = X_test.copy()
                X_test["tau"] = float(tau)

                X_test_array = X_test.to_numpy(dtype=float)
                probs = tabpfn_model.predict_proba(X_test_array)
                hazard_probs = get_positive_class_probs(probs)
                tmp = hazards_to_survival_df(test_pids, test_eval_times, hazard_probs)
                
                # -----------------------------
                # RISK = average over future timepoints of (1 - surv_prob)
                # Future timepoints here are strictly after tau (exclude t == tau)
                # -----------------------------
                tau_float = float(tau)
                tmp_future = tmp[~np.isclose(tmp["t"].to_numpy(), tau_float)].copy()
                tmp_future["risk_component"] = 1.0 - tmp_future["surv_prob"]

                # Mean risk per pid (one risk score per pid per tau)
                risk_by_pid = tmp_future.groupby("pid")["risk_component"].mean()

                # Store risk score row (t is NaN) once per pid
                for pid, risk in risk_by_pid.items():
                    all_risk_rows.append({
                        "pid": int(pid),
                        "tau": float(tau),
                        "t": np.nan,
                        "risk": float(risk),
                        "surv_prob": np.nan,
                        "dataset": dataset_name,
                        "hyperparameter": str(hyperparameter),
                        "eval_set": "test",
                        "k": args.k
                    })

                # Store survival probability for strictly future times (exclude tau)
                for pid, t, surv_prob in zip(tmp["pid"].to_numpy(), tmp["t"].to_numpy(), tmp["surv_prob"].to_numpy()):
                    if not np.isclose(float(t), tau_float):
                        all_risk_rows.append({
                            "pid": int(pid),
                            "tau": float(tau),
                            "t": float(t),
                            "risk": np.nan,
                            "surv_prob": float(surv_prob),
                            "dataset": dataset_name,
                            "hyperparameter": str(hyperparameter),
                            "eval_set": "test",
                            "k": args.k
                        })

# ============================================================
# Write results to CSV file
# ============================================================
print("\n" + "="*70)
print("Writing results to CSV file...")

if len(all_risk_rows) > 0:
    results_df = pd.DataFrame(all_risk_rows)
    results_df = results_df.sort_values(["hyperparameter", "eval_set", "k", "dataset", "pid", "tau", "t"], na_position="last").reset_index(drop=True)
    results_df.to_csv(risk_out_path, index=False)
    print(f"Results saved to: {risk_out_path}")
    print(f"Total rows: {len(results_df)}")
    val_rows = (results_df['eval_set'] == 'val').sum()
    test_rows = (results_df['eval_set'] == 'test').sum()
    print(f"  Val rows: {val_rows} ({(~results_df[results_df['eval_set']=='val']['risk'].isna()).sum()} risk scores, {(~results_df[results_df['eval_set']=='val']['surv_prob'].isna()).sum()} surv probs)")
    print(f"  Test rows: {test_rows} ({(~results_df[results_df['eval_set']=='test']['risk'].isna()).sum()} risk scores, {(~results_df[results_df['eval_set']=='test']['surv_prob'].isna()).sum()} surv probs)")
else:
    print("No results to save")

if all_oof_rows:  # DOOF
    oof_path = risk_out_path.replace("_risk_", "_oof_")
    pd.DataFrame(all_oof_rows).to_csv(oof_path, index=False)
    print(f"Out-of-fold predictions saved to: {oof_path}")

print("\nProcessing complete!")
