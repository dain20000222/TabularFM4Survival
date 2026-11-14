# =============================================================================
# Dynamic DeepHit Survival Analysis Implementation
# =============================================================================
# This script performs dynamic survival analysis using DeepHit with longitudinal
# data. It implements landmark analysis where survival predictions are made at
# multiple time points (landmarks) using historical covariate information.
# Dynamic DeepHit is a deep learning method for survival analysis that handles competing
# risks and time-varying effects.
# =============================================================================

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
import warnings
import random
from ddh.ddh_api import DynamicDeepHit
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
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
parser = argparse.ArgumentParser(description='Run dynamic DeepHit with different landmark binning strategies')
parser.add_argument('--k', type=float, default=20, help='Step size for landmark time percentiles (default: 20)')
parser.add_argument('--binning', type=str, default='km_quantile', choices=['quantile', 'km_quantile', 'pooled'],
                    help='Landmark binning strategy: quantile or km_quantile (default: km_quantile)')
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

# Output file path for survival probability results
binning_suffix = f"_{args.binning}" if args.binning != "quantile" else ""
k_token = f"{float(args.k):.8g}"
surv_probs_out_path = f"dynamic_deephit_risk_{k_token}{binning_suffix}.csv"

print(f"Total datasets: {len(dataset_files)}")
print(f"Datasets to process: {dataset_files if dataset_files else 'None'}")
print(f"Using k={args.k} with binning={args.binning} for landmark generation")

# =============================================================================
# Global Data Accumulators
# =============================================================================

# Collect survival probability results across all datasets
all_surv_rows = []

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


def convert_to_sequential_format(df, covariates):
    """
    Convert longitudinal data to sequential format for Dynamic DeepHit.
    
    This function transforms the longitudinal dataset from a long format 
    (one row per observation) to a sequential format where each patient
    has variable-length sequences of features, times, and events.
    
    Parameters:
    -----------
    df : DataFrame
        Longitudinal dataset with columns 'pid', 'time', 'event' and covariates
    covariates : list
        List of covariate column names to include as features
    
    Returns:
    --------
    tuple : (x_data, t_data, e_data, pid_data)
        - x_data: np.array of patient feature sequences (object array)
        - t_data: np.array of patient time sequences (object array) 
        - e_data: np.array of patient event sequences (object array)
        - pid_data: np.array of patient IDs
    """
    x_data = []
    t_data = []
    e_data = []
    pid_data = []
    
    # Process each patient individually, sorting by time for temporal order
    for pid in sorted(df['pid'].unique()):  
        patient_data = df[df['pid'] == pid].sort_values('time')
        
        # Extract features, times, and events for this patient
        patient_features = []
        patient_times = []
        patient_events = []
        
        for _, row in patient_data.iterrows():
            # Extract and clean features for this time point
            features = []
            for cov in covariates:
                val = row[cov]
                if pd.isna(val):
                    val = 0.0  # Handle missing values with zero imputation
                features.append(float(val))
            
            patient_features.append(features)
            patient_times.append(float(row['time']))
            patient_events.append(float(row['event']))
        
        # Skip patients with no valid observations
        if len(patient_features) == 0:
            continue
            
        # Convert to numpy arrays with float64 precision for numerical stability
        x_data.append(np.array(patient_features, dtype=np.float64))
        t_data.append(np.array(patient_times, dtype=np.float64))
        e_data.append(np.array(patient_events, dtype=np.float64))
        pid_data.append(pid)

    # Return as object arrays to handle variable-length sequences
    # Note: Dynamic DeepHit's _get_padded_features will handle conversion to proper numeric arrays
    # 11-07: build 1-d object arrays explicitly. np.array(list, dtype=object) turns sequences of EQUAL length into a 3-d
    # array (FRTCS: 3 rows for every patient), which broke the padding ("can't convert np.ndarray of type numpy.object_").
    # For variable lengths (the paper's datasets) the result is unchanged.
    def _seq(a):
        out = np.empty(len(a), dtype=object)
        out[:] = a
        return out
    return (_seq(x_data), _seq(t_data), _seq(e_data), np.array(pid_data))

# =============================================================================
# Main Dataset Processing Loop
# =============================================================================
for i, file_name in enumerate(dataset_files, start=1):
    # =================================================================
    # Dataset Loading and Initial Setup
    # =================================================================
    
    dataset_name = file_name.replace(".csv", "")
    file_path = os.path.join(data_dir, file_name)
    print("=" * 70)
    print(f"\nProcessing dataset: {dataset_name} ({i}/{len(dataset_files)})")

    # Load longitudinal dataset
    df = pd.read_csv(file_path)
    print(f"Original dataset shape: {df.shape}")

    # =================================================================
    # Column Identification and Data Exploration
    # =================================================================
    
    # Define standard longitudinal survival analysis columns
    time_col = "time"        # Observation time (start of interval)
    time2_col = "time2"      # Event/censoring time (end of interval)  
    event_col = "event"      # Event indicator (1=event, 0=censored)

    # =================================================================
    # Patient-Level Train/Validation/Test Split
    # =================================================================
    
    # Split patients (not observations) into train/val/test (70%/15%/15%)
    original_pids = df["pid"].unique()
    pid_train, pid_temp = train_test_split(original_pids, test_size=0.3, random_state=SEED)
    pid_val, pid_test = train_test_split(pid_temp, test_size=0.5, random_state=SEED)
    print(f"Train patients: {len(pid_train)}, Val patients: {len(pid_val)}, Test patients: {len(pid_test)}")

    # =================================================================
    # Feature Selection and Preprocessing Setup
    # =================================================================
    
    # Identify covariate columns (exclude time, event, and ID columns)
    covariates = df.columns.difference([time_col, time2_col, event_col, 'pid']).tolist()
    
    # Categorize features for appropriate preprocessing
    categorical_cols = []
    numerical_cols = []
    
    for col in covariates:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)

    # Create patient-specific datasets
    df_train = df[df['pid'].isin(pid_train)]
    df_val = df[df['pid'].isin(pid_val)]
    df_test = df[df['pid'].isin(pid_test)]

    # =================================================================
    # Landmark Time Point Selection (Training Set Only)
    # =================================================================
    
    # Build landmarks from training set only to avoid data leakage
    taus = build_landmarks_from_train(
        train_df=df_train,
        time_col_for_tau=time2_col,
        k=args.k,
        binning=args.binning,
    )

    if len(taus) == 0:
        print(f"Skipping {dataset_name}: No events observed in TRAIN split")
        continue

    taus = list(taus)
    num_eval_times = len(taus)

    if num_eval_times < 2:
        print(f"Skipping {dataset_name}: Need at least 2 landmark times, got {num_eval_times}")
        continue

    t_max = float(df["time2"].max())
    eval_times = list(taus) + [t_max]

    print(f"Landmarks (taus) from TRAIN: {[f'{t:.6g}' for t in taus]}")
    print(f"Number of evaluation times: {num_eval_times} (k={args.k}, binning={args.binning})")

    # =================================================================
    # Feature Engineering and Preprocessing
    # =================================================================
    
    # One-hot encode categorical variables (fit transformation on training data only)
    x_train = pd.get_dummies(df_train[covariates], drop_first=True)
    x_val = pd.get_dummies(df_val[covariates], drop_first=True)
    x_test = pd.get_dummies(df_test[covariates], drop_first=True)

    # Align feature columns across all splits (union of all columns)
    all_columns = x_train.columns.union(x_val.columns).union(x_test.columns)
    x_train = x_train.reindex(columns=all_columns, fill_value=0)
    x_val = x_val.reindex(columns=all_columns, fill_value=0)
    x_test = x_test.reindex(columns=all_columns, fill_value=0)

    # =================================================================
    # Missing Value Imputation
    # =================================================================
    
    # Fit imputer on training data only to prevent data leakage
    imputer = SimpleImputer(strategy='median')

    x_train = pd.DataFrame(imputer.fit_transform(x_train),
                        columns=x_train.columns,
                        index=x_train.index)

    x_val = pd.DataFrame(imputer.transform(x_val),
                        columns=x_val.columns,
                        index=x_val.index)

    x_test = pd.DataFrame(imputer.transform(x_test),
                        columns=x_test.columns,
                        index=x_test.index)

    # =================================================================
    # Dataset Reconstruction with Processed Features
    # =================================================================
    
    # Combine processed features with time/event/ID columns
    df_train = pd.concat([df_train[['pid','time','time2','event']].reset_index(drop=True),
                                    x_train.reset_index(drop=True)],
                                    axis=1)
    
    df_val = pd.concat([df_val[['pid','time','time2','event']].reset_index(drop=True),
                                x_val.reset_index(drop=True)],
                                axis=1)
    
    df_test = pd.concat([df_test[['pid','time','time2','event']].reset_index(drop=True),
                                x_test.reset_index(drop=True)],
                                axis=1)

    # Update covariates after encoding
    covariates = df_train.columns.difference(['pid','time','time2','event']).tolist()

    # Calculate default values from TRAIN set only for imputation
    default_values = {}
    for col in covariates:
        default_values[col] = df_train[col].mean()
    print(f"Default values calculated from train set")

    # Get the first landmark time
    first_landmark = taus[0]
    
    # Add default rows for all train/val/test patients who:
    # 1. Are at risk at first landmark (event time > first landmark)
    # 2. Have no observation before first landmark (first observation >= first landmark)
    
    # TRAIN set
    default_rows_train = []
    for pid in pid_train:
        patient_data = df_train[df_train['pid'] == pid]
        min_visit_time = patient_data['time'].min()
        max_event_time = patient_data['time2'].max()  # Last follow-up time
        
        # Check if patient is at risk at first landmark AND no observation before it
        if max_event_time > first_landmark and min_visit_time >= first_landmark:
            default_row = default_values.copy()
            default_row["pid"] = pid
            default_row["time"] = 0.0
            default_row["time2"] = min_visit_time
            default_row["event"] = 0.0
            default_rows_train.append(default_row)
    
    if len(default_rows_train) > 0:
        default_df = pd.DataFrame(default_rows_train)
        df_train = pd.concat([df_train, default_df], ignore_index=True)
        df_train = df_train.sort_values(['pid', 'time2']).reset_index(drop=True)
        print(f"Added {len(default_rows_train)} default rows to train set (at risk at first landmark, no prior observation)")
    
    # VAL set
    default_rows_val = []
    for pid in pid_val:
        patient_data = df_val[df_val['pid'] == pid]
        min_visit_time = patient_data['time'].min()
        max_event_time = patient_data['time2'].max()  # Last follow-up time
        
        # Check if patient is at risk at first landmark AND no observation before it
        if max_event_time > first_landmark and min_visit_time >= first_landmark:
            default_row = default_values.copy()
            default_row["pid"] = pid
            default_row["time"] = 0.0
            default_row["time2"] = min_visit_time
            default_row["event"] = 0.0
            default_rows_val.append(default_row)
    
    if len(default_rows_val) > 0:
        default_df = pd.DataFrame(default_rows_val)
        df_val = pd.concat([df_val, default_df], ignore_index=True)
        df_val = df_val.sort_values(['pid', 'time2']).reset_index(drop=True)
        print(f"Added {len(default_rows_val)} default rows to val set (at risk at first landmark, no prior observation)")
    
    # TEST set
    default_rows_test = []
    for pid in pid_test:
        patient_data = df_test[df_test['pid'] == pid]
        min_visit_time = patient_data['time'].min()
        max_event_time = patient_data['time2'].max()  # Last follow-up time
        
        # Check if patient is at risk at first landmark AND no observation before it
        if max_event_time > first_landmark and min_visit_time >= first_landmark:
            default_row = default_values.copy()
            default_row["pid"] = pid
            default_row["time"] = 0.0
            default_row["time2"] = min_visit_time
            default_row["event"] = 0.0
            default_rows_test.append(default_row)
    
    if len(default_rows_test) > 0:
        default_df = pd.DataFrame(default_rows_test)
        df_test = pd.concat([df_test, default_df], ignore_index=True)
        df_test = df_test.sort_values(['pid', 'time2']).reset_index(drop=True)
        print(f"Added {len(default_rows_test)} default rows to test set (at risk at first landmark, no prior observation)")

    # Convert to sequential format for each split
    x_train, t_train, e_train, train_pids = convert_to_sequential_format(df_train, covariates)
    x_val, t_val, e_val, val_pids = convert_to_sequential_format(df_val, covariates)
    x_test, t_test, e_test, test_pids = convert_to_sequential_format(df_test, covariates)
    
    print(f"Train shape: {len(x_train)} patients, Val shape: {len(x_val)} patients, Test shape: {len(x_test)} patients")

    # Build patient-level outcomes for VAL and TEST sets (last follow-up time)
    patient_outcome_val = (
        df_val.sort_values("time2")
               .groupby("pid")
               .agg(
                   t_event=("time2", "max"),
                   status_event=("event", "max")
               )
    )
    patient_outcome_test = (
        df_test.sort_values("time2")
               .groupby("pid")
               .agg(
                   t_event=("time2", "max"),
                   status_event=("event", "max")
               )
    )

    # ============================================================
    # Hyperparameter tuning loop
    # ============================================================
    # Parameter grid - 32 combinations: 2 hidden_rnn × 2 hidden_layer × 2 sigma × 4 learning_rate

    param_grid = {
        'layers_rnn': [2],
        'hidden_rnn': [50, 100],
        'hidden_layer': [[100, 100], [100, 100, 100]],
        'sigma': [1, 3],
        'learning_rate': [1e-2, 5e-3, 1e-3, 1e-4],
    }
            
    params_list = list(ParameterGrid(param_grid))
    print(f"\nHyperparameter grid: {len(params_list)} combinations")
    
    for param_idx, param in enumerate(params_list):
        hyperparam_str = f"layers_rnn={param['layers_rnn']},hidden_rnn={param['hidden_rnn']},hidden_layer={param['hidden_layer']},sigma={param['sigma']},lr={param['learning_rate']}"
        print(f"\n{'='*70}")
        print(f"[{param_idx+1}/{len(params_list)}] Training with hyperparameters:")
        print(f"  {hyperparam_str}")
        print(f"{'='*70}")
        
        try:
            # Train the model
            model = DynamicDeepHit(
                layers_rnn = param['layers_rnn'],
                hidden_rnn = param['hidden_rnn'], 
                long_param = {'layers': param['hidden_layer'], 'dropout': 0.3}, 
                att_param = {'layers': param['hidden_layer'], 'dropout': 0.3}, 
                cs_param = {'layers': param['hidden_layer'], 'dropout': 0.3},
                sigma = param['sigma'],
                split = [0.0] + taus + [float(df["time2"].max())]
            )
            
            # Train the model
            model.fit(
                x_train, t_train, e_train,
                iters = 10, 
                learning_rate = param['learning_rate']
            )
            
            print(f"Model trained successfully!")
            
        except Exception as e:
            print(f"Error training model: {e}")
            continue
        
        # Evaluate on both val and test sets
        for eval_set_name, (x_eval, t_eval, e_eval, eval_pids, patient_outcome_eval) in [
            ("val", (x_val, t_val, e_val, val_pids, patient_outcome_val)),
            ("test", (x_test, t_test, e_test, test_pids, patient_outcome_test))
        ]:
            print(f"\n{'='*50}")
            print(f"Evaluating on {eval_set_name} set ({len(eval_pids)} patients)")
            print(f"{'='*50}")
            
            # Dictionary to store predictions for all eval patients
            patient_predictions = {pid: {} for pid in eval_pids}

            # For each landmark time
            for idx, tau in enumerate(taus[:-1]):
                print(f"Processing landmark time tau{idx} = {tau:.6g}")
                
                # Define risk set at this landmark: patients with T_i > tau
                at_risk_mask = patient_outcome_eval["t_event"] > tau
                at_risk_pids = set(patient_outcome_eval.index[at_risk_mask])
                
                # Predict risk at all future landmark times using information only up to tau
                future_times = [taus[i] for i in range(idx + 1, len(taus))]
                
                # Get risk predictions - shape: (n_eval, n_future_times, max_seq_len)
                try:
                    out_risk = model.predict_risk(x_eval, future_times, all_step=True)
                except Exception as e:
                    print(f"Error predicting risk at tau{idx}: {e}")
                    continue
                
                # For each eval patient, extract the risk score at tau
                for patient_idx, pid in enumerate(eval_pids):
                    # Skip patients who are not at risk at tau (T_i <= tau)
                    if pid not in at_risk_pids:
                        continue
                    
                    patient_visit_times = t_eval[patient_idx]
                    
                    # Find the most recent visit at or before tau
                    valid_visits_mask = patient_visit_times <= tau
                    
                    if not np.any(valid_visits_mask):
                        continue  # No visits before tau
                    
                    valid_visit_indices = np.where(valid_visits_mask)[0]
                    most_recent_visit_idx = valid_visit_indices[-1]
                    
                    # Extract risk score at tau (first future time)
                    risk_to_next = out_risk[patient_idx, 0, most_recent_visit_idx]
                    
                    # Extract survival probabilities for future landmarks
                    surv_probs = {}
                    for future_idx in range(idx + 1, len(taus)):
                        # Index in out_risk for this future time
                        risk_idx = future_idx - (idx + 1)
                        if risk_idx < out_risk.shape[1]:
                            risk_future = out_risk[patient_idx, risk_idx, most_recent_visit_idx]
                            # Survival probability = 1 - risk
                            surv_probs[f't_{future_idx}'] = 1.0 - risk_future
                    
                    # Store predictions
                    if f'tau{idx}' not in patient_predictions[pid]:
                        patient_predictions[pid][f'tau{idx}'] = {}
                    patient_predictions[pid][f'tau{idx}'] = {
                        'risk': risk_to_next,
                        **surv_probs
                    }
                    
                print(f"  Predictions extracted for {len([p for p in patient_predictions.values() if f'tau{idx}' in p])} patients")

            # ========== Save all predictions for this hyperparameter configuration ==========
            for pid in sorted(eval_pids):
                pred = patient_predictions[pid]
                
                # First, write all risk scores (one per conditioning landmark) - EXCLUDE last landmark
                for idx in range(len(taus) - 1):
                    if f'tau{idx}' in pred and 'risk' in pred[f'tau{idx}']:
                        tau = float(taus[idx])  
                        risk_val = pred[f'tau{idx}']['risk']
                        all_surv_rows.append({
                            'pid': pid,
                            'tau': tau,
                            't': np.nan,
                            'risk': risk_val if np.isfinite(risk_val) else np.nan,
                            'surv_prob': np.nan,
                            'dataset': dataset_name,
                            'hyperparameter': hyperparam_str,
                            'eval_set': eval_set_name,
                            'k': args.k
                        })
                
                # Then, write all survival probabilities
                for idx, tau in enumerate(taus):
                    if f'tau{idx}' in pred:
                        # Write survival probabilities for all future timepoints
                        for future_idx in range(idx + 1, len(taus)):
                            future_tau = taus[future_idx]
                            surv_key = f't_{future_idx}'
                            if surv_key in pred[f'tau{idx}']:
                                surv_val = pred[f'tau{idx}'][surv_key]
                                all_surv_rows.append({
                                    'pid': pid,
                                    'tau': tau,
                                    't': future_tau,
                                    'risk': np.nan,
                                    'surv_prob': surv_val if np.isfinite(surv_val) else np.nan,
                                    'dataset': dataset_name,
                                    'hyperparameter': hyperparam_str,
                                    'eval_set': eval_set_name,
                                    'k': args.k
                                })
            
            print(f"  Saved predictions for {eval_set_name} set")

# ============================================================
# Save all predictions to CSV at the end
# ============================================================
if all_surv_rows:
    out_df = pd.DataFrame(all_surv_rows)
    out_df.to_csv(surv_probs_out_path, index=False)
    print(f"\n{'='*70}")
    print(f"All predictions saved to {surv_probs_out_path}")
    print(f"Total rows: {len(out_df)}")
    print(f"{'='*70}")
else:
    print("\nNo predictions to save.")

print("\n" + "="*70)
print("Processing complete!")
print("="*70)

