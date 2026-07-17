# =============================================================================
# MITRA (Model-agnostic Individual Treatment Response Analysis) 
# Survival Analysis Implementation
# =============================================================================
# This script performs survival analysis using MITRA, a model-agnostic approach
# that transforms survival prediction into binary classification problems by
# considering multiple time horizons for each patient.
# =============================================================================

import pandas as pd
import numpy as np
import os
import csv
import argparse
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored, integrated_brier_score, concordance_index_ipcw
from sksurv.nonparametric import kaplan_meier_estimator
import torch
import warnings
from autogluon.tabular import TabularDataset, TabularPredictor
import random
import tempfile, shutil
warnings.filterwarnings("ignore")

# =============================================================================
# Configuration and Setup
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

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run MITRA survival analysis with different evaluation time configurations')
parser.add_argument('--k', type=float, default=10.0, help='Step size for evaluation time percentiles (default: 10.0)')
parser.add_argument('--binning', type=str, default='quantile', choices=['quantile', 'equal_width', 'km_quantile'],
                    help='Binning strategy: quantile (default), equal_width, or km_quantile (KM-weighted)')
args = parser.parse_args()

# =============================================================================
# Data Configuration
# =============================================================================

# Directory containing the datasets
data_dir = os.path.join("static_data")

# List all CSV files in the directory
dataset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# Output CSV path based on k parameter
binning_suffix = f"_{args.binning}" if args.binning != "quantile" else ""
k_token = f"{float(args.k):.8g}"
csv_path = f"static_mitra_evaluation_final_{k_token}{binning_suffix}.csv"

# =============================================================================
# Utility Functions
# =============================================================================

def get_valid_eval_times(base_eval_times, test_times, buffer=0.01):
    """
    Filter evaluation times to be within valid data follow-up period.
    
    This function ensures that evaluation time points are within the range
    of observed data to avoid extrapolation issues in survival prediction.
    
    Parameters:
    -----------
    base_eval_times : array-like
        Initial set of evaluation time points
    test_times : array-like
        Observed time values from test/validation data
    buffer : float, optional (default=0.01)
        Small buffer to ensure times are strictly within bounds
    
    Returns:
    --------
    valid_times : array-like
        Filtered evaluation times within valid range
    """
    # Get min and max time values from test data
    min_test_time = test_times.min()
    max_test_time = test_times.max()
    
    # Apply time constraints with buffer to avoid boundary issues
    valid_times = base_eval_times[
        (base_eval_times > min_test_time + buffer) & 
        (base_eval_times < max_test_time - buffer)
    ]
    
    return valid_times


def k_to_percentile_grid(k: float, tol: float = 1e-6):
    """Return (percentiles, n_bins, K_int_or_None) for a step size k.

    If k corresponds to an integer number of intervals K=100/k (within tol), we
    build the grid exactly as (1/K, 2/K, ..., (K-1)/K)*100 to avoid float-step
    drift (important for K=15 -> k=6.6666667).
    """
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

def construct_mitra_binary_trainset(x_train, y_train, times):
    """
    Construct MITRA training dataset using binary classification approach.
    
    The MITRA method transforms survival prediction into multiple binary 
    classification problems by including each patient's own event/censoring
    time alongside predefined evaluation time points. This creates multiple
    observations per patient, each asking: "Did the event occur by time t?"
    
    Parameters:
    -----------
    x_train : DataFrame
        Training covariates with shape (n_patients, n_features)
    y_train : structured array  
        Training survival data with 'time' and 'event' fields
    times : array-like
        Predefined time points to consider for binary classification
    
    Returns:
    --------
    X_mitra : DataFrame
        Features for MITRA including original covariates and evaluation time
    y_mitra : Series
        Binary labels indicating event occurrence by evaluation time
    """
    # Handle edge case of no valid evaluation times
    if len(times) == 0:
        print("Warning: No valid evaluation times for training set construction")
        return pd.DataFrame(), pd.Series(dtype=int)

    # Extract survival data components
    T = y_train["time"]      # Event/censoring times
    delta = y_train["event"] # Event indicators (1=event, 0=censored)
    n_train = len(T)
    
    dataset_rows = []
    binary_labels = []
    
    # Process each patient
    for i in range(n_train):
        T_i = T[i]        # Patient's observed time
        delta_i = delta[i] # Patient's event status
        x_i = x_train.iloc[i].values  # Patient's covariates
        
        # Combine predefined time points with patient's own time
        horizons = np.append(times.copy(), T_i)
        horizons.sort()
        
        # For censored patients, only consider times up to censoring time
        if delta_i == 0:  # Censored case
            horizons = horizons[horizons <= T_i]

        # Create binary classification samples for each valid time horizon
        for t in horizons:
            # Features: original covariates + evaluation time
            row = np.concatenate([x_i, [t]])
            # Label: 1 if event occurred by time t, 0 otherwise
            label = int(delta_i and T_i <= t)
            dataset_rows.append(row)
            binary_labels.append(label)
    
    # Handle edge case of no valid samples
    if len(dataset_rows) == 0:
        print("Warning: No valid training samples created")
        return pd.DataFrame(), pd.Series(dtype=int)
    
    # Construct final dataset
    feature_cols = list(x_train.columns) + ["eval_time"]
    X_mitra = pd.DataFrame(dataset_rows, columns=feature_cols)
    y_mitra = pd.Series(binary_labels)
    
    # Display class distribution for monitoring data balance
    n_ones = sum(binary_labels)
    n_zeros = len(binary_labels) - n_ones
    print(f"\nTraining set class distribution:")
    print(f"Class 0 (no event): {n_zeros} ({n_zeros/len(binary_labels)*100:.1f}%)")
    print(f"Class 1 (event):    {n_ones} ({n_ones/len(binary_labels)*100:.1f}%)")
    print(f"Total samples:      {len(binary_labels)}")

    return X_mitra, y_mitra

def construct_mitra_binary_testset(x_test, times):
    """Construct MITRA test dataset using evaluation times for binary classification.
    
    Parameters:
    - x_test: DataFrame of imputed covariates for test set
    - times: Array of time points to consider for binary classification

    Returns:
    - X_mitra_test: DataFrame of features for MITRA test set
    - patient_ids: Array mapping each row to the original patient index
    """
    if len(times) == 0:
        # If no valid evaluation times, return empty dataset
        print("Warning: No valid evaluation times for test set construction")
        feature_cols = list(x_test.columns) + ["eval_time"]
        return pd.DataFrame(columns=feature_cols), np.array([])
        
    n_test = x_test.shape[0]

    test_rows = []
    patient_ids = []
    
    for i in range(n_test):
        x_i = x_test.iloc[i].values

        # Create one row per evaluation time
        for t in times:
            row = np.concatenate([x_i, [t]])
            test_rows.append(row)
            patient_ids.append(i)

    feature_cols = list(x_test.columns) + ["eval_time"]
    X_mitra_test = pd.DataFrame(test_rows, columns=feature_cols)

    return X_mitra_test, np.array(patient_ids)

# =============================================================================
# Main Dataset Processing Loop  
# =============================================================================

for file_name in dataset_files:
    try:
        # =================================================================
        # Dataset Loading and Initial Setup
        # =================================================================
        
        dataset_name = file_name.replace(".csv", "")
        file_path = os.path.join(data_dir, file_name)
        print("="*50)
        print(f"\nProcessing dataset: {dataset_name}")

        # Track best results per dataset
        best_row = None

        # Load datasets
        df = pd.read_csv(file_path)
        df.drop(columns=['pid'], inplace=True)
        shape = df.shape
        print(f"Dataset shape: {shape}")

        # Define columns
        time_col = "time"
        event_col = "event"
        censored = (df["event"] == 0).sum()
        censored_percent = (censored)/len(df)*100
        print(f"Percentage of censored data: {censored_percent}%")
        covariates = df.columns.difference([time_col, event_col])

        # Define covariates and target variable
        x = df[covariates].copy()
        y = Surv.from_arrays(event=df[event_col].astype(bool), time=df[time_col])
        
        # Split into train/val/test (70%/15%/15%) with stratification on event
        x_trainval, x_test, y_trainval, y_test = train_test_split(
            x, y, test_size=0.15, stratify=y["event"], random_state=SEED
        )

        x_train, x_val, y_train, y_val = train_test_split(
            x_trainval, y_trainval, test_size=0.1765, stratify=y_trainval["event"], random_state=SEED
        )

        # One-hot encode using get_dummies (fit on train only)
        x_train_ohe = pd.get_dummies(x_train, drop_first=True)
        x_val_ohe = pd.get_dummies(x_val, drop_first=True)
        x_test_ohe = pd.get_dummies(x_test, drop_first=True)

        # Align columns so train/val/test match (add missing columns with 0s)
        x_train_ohe, x_val_ohe = x_train_ohe.align(x_val_ohe, join="left", axis=1, fill_value=0)
        x_train_ohe, x_test_ohe = x_train_ohe.align(x_test_ohe, join="left", axis=1, fill_value=0)

        covariates_ohe = x_train_ohe.columns  # all columns are covariates after OHE

        # Impute missing values in covariates (fit on train only)
        imputer = SimpleImputer().fit(x_train_ohe.loc[:, covariates_ohe.tolist()])

        x_train_imputed = imputer.transform(x_train_ohe.loc[:, covariates_ohe.tolist()])
        x_train_imputed = pd.DataFrame(x_train_imputed, columns=covariates_ohe, index=x_train.index)

        x_val_imputed = imputer.transform(x_val_ohe.loc[:, covariates_ohe.tolist()])
        x_val_imputed = pd.DataFrame(x_val_imputed, columns=covariates_ohe, index=x_val.index)

        x_test_imputed = imputer.transform(x_test_ohe.loc[:, covariates_ohe.tolist()])
        x_test_imputed = pd.DataFrame(x_test_imputed, columns=covariates_ohe, index=x_test.index)

        # Determine evaluation time points (eval_times)
        percentiles, n_bins, _K = k_to_percentile_grid(args.k)
        if args.binning == 'quantile':
            base_eval_times = np.percentile(y_trainval["time"], percentiles)
        elif args.binning == 'equal_width':
            base_eval_times = np.linspace(y_trainval["time"].min(), y_trainval["time"].max(), n_bins + 2)[1:-1]
        else:  # km_quantile
            km_time, km_surv = kaplan_meier_estimator(y_trainval["event"], y_trainval["time"])
            if len(km_time) == 0:
                base_eval_times = np.percentile(y_trainval["time"], percentiles)
            else:
                surv_levels = np.linspace(km_surv[0], km_surv[-1], n_bins + 2)[1:-1]
                base_eval_times = []
                for s in surv_levels:
                    idx = np.searchsorted(-km_surv, -s)
                    if idx < len(km_time):
                        base_eval_times.append(float(km_time[idx]))
                base_eval_times = np.array(base_eval_times)
                if len(base_eval_times) == 0:
                    base_eval_times = np.percentile(y_trainval["time"], percentiles)
        base_eval_times = np.unique(base_eval_times)

        # Filter test set to only include times within the range of train+val
        max_trainval_time = y_trainval["time"].max()
        # Don't remove samples, just truncate times that exceed max_trainval_time
        y_test_filtered = y_test.copy()
        test_time_mask = y_test_filtered["time"] >= max_trainval_time
        y_test_filtered["time"][test_time_mask] = max_trainval_time - 0.01  # Small buffer
        y_test_filtered["event"][test_time_mask] = False  # Convert to censored
        x_test_filtered = x_test_imputed.copy()

        print(f"Test samples after time truncation: {len(y_test_filtered)} / {len(y_test)} (truncated {test_time_mask.sum()} samples)")

        # Get valid evaluation times for validation
        val_eval_times = get_valid_eval_times(base_eval_times, y_val["time"])
        print(f"Validation eval times: {len(val_eval_times)} points")

        if len(val_eval_times) == 0:
            print("Warning: No valid evaluation times for validation - skipping dataset")
            continue

        # Train and validate using MITRA model
        # Construct MITRA training set
        X_mitra_train, y_mitra_train = construct_mitra_binary_trainset(
            x_train_imputed, y_train, val_eval_times
        )
        
        if len(X_mitra_train) == 0:
            print("Warning: No training data created - skipping dataset")
            continue
        
        # Sample 10000 rows if dataset is larger
        if len(X_mitra_train) > 10000:
            print(f"\nSampling 10000 rows from {len(X_mitra_train)} total rows...")
            sample_idx = np.random.choice(len(X_mitra_train), size=10000, replace=False)
            X_mitra_train = X_mitra_train.iloc[sample_idx].reset_index(drop=True)
            y_mitra_train = y_mitra_train.iloc[sample_idx].reset_index(drop=True)

        # Convert to TabularDataset
        train_data = pd.DataFrame(X_mitra_train)
        train_data['target'] = y_mitra_train
        train_data = TabularDataset(train_data)
        
        # Initialize and train Mitra model
        # Create a temporary directory for AutoGluon
        tmp_dir = tempfile.mkdtemp()
        mitra_predictor = TabularPredictor(label='target', path=tmp_dir)
        mitra_predictor.fit(
            train_data,
            hyperparameters={
                'MITRA': {'fine_tune': False}
            }
        )
        
        # Construct MITRA validation set
        X_mitra_val, val_patient_ids = construct_mitra_binary_testset(
            x_val_imputed, val_eval_times
        )

        if len(X_mitra_val) == 0:
            print("Warning: No validation data created - skipping dataset")
            shutil.rmtree(tmp_dir)
            continue

        # Convert val data to TabularDataset 
        val_data = TabularDataset(pd.DataFrame(X_mitra_val))

        # Predict survival probabilities
        probs = mitra_predictor.predict_proba(val_data)
        probs = probs.to_numpy()
        surv_probs = probs[:, 0]

        # Clean up temporary directory
        shutil.rmtree(tmp_dir)

        n_patients = len(x_val_imputed)
        S_val = np.zeros((n_patients, len(val_eval_times)))
        
        for i, t in enumerate(val_eval_times):
            mask = np.isclose(X_mitra_val["eval_time"], t)
            S_val[val_patient_ids[mask], i] = surv_probs[mask]

        # Ensure survival probabilities are non-increasing over time
        S_val = np.minimum.accumulate(S_val, axis=1)

        # Calculate risk scores as average of (1 - survival_prob) across multiple eval times
        val_risk_scores = np.mean(1 - S_val, axis=1)  # Average risk across all evaluation times

        # Validation metrics
        val_c_index, *_ = concordance_index_censored(
            y_val["event"], y_val["time"], val_risk_scores  # Use scalar risk score
        )
        
        val_ipcw_c_index, *_ = concordance_index_ipcw(
            y_train, y_val, val_risk_scores, tau=y_train["time"].max()
        )

        val_ibs = integrated_brier_score(y_train, y_val, S_val, val_eval_times) if len(val_eval_times) > 0 else np.nan
        
        _, val_mean_auc = cumulative_dynamic_auc(
            y_train, y_val, val_risk_scores, val_eval_times  # Use scalar risk score
        ) if len(val_eval_times) > 0 else (None, np.nan)

        val_results = {
            'c_index': float(val_c_index),
            'ipcw_c_index': float(val_ipcw_c_index),
            'ibs': float(val_ibs),
            'mean_auc': float(val_mean_auc)
        }

        # Retrain on train+val, evaluate on test 
        # Combine train and val sets
        x_trainval_ohe = pd.get_dummies(x_trainval, drop_first=True)
        x_test_ohe = pd.get_dummies(x_test, drop_first=True)
        x_trainval_ohe, x_test_ohe = x_trainval_ohe.align(x_test_ohe, join="left", axis=1, fill_value=0)
        covariates_ohe = x_trainval_ohe.columns

        # Impute missing values in covariates (fit on train+val only)
        imputer = SimpleImputer().fit(x_trainval_ohe.loc[:, covariates_ohe.tolist()])

        x_trainval_imputed = imputer.transform(x_trainval_ohe.loc[:, covariates_ohe.tolist()])
        x_trainval_imputed = pd.DataFrame(x_trainval_imputed, columns=covariates_ohe, index=x_trainval.index)

        x_test_imputed = imputer.transform(x_test_ohe.loc[:, covariates_ohe.tolist()])
        x_test_imputed = pd.DataFrame(x_test_imputed, columns=covariates_ohe, index=x_test.index)

        # Apply same time truncation to test set
        max_trainval_time = y_trainval["time"].max()
        y_test_filtered = y_test.copy()
        test_time_mask = y_test_filtered["time"] >= max_trainval_time
        y_test_filtered["time"][test_time_mask] = max_trainval_time - 0.01  # Small buffer
        y_test_filtered["event"][test_time_mask] = False  # Convert to censored
        x_test_filtered = x_test_imputed.copy()

        # Get valid evaluation times for test
        test_eval_times = get_valid_eval_times(base_eval_times, y_test_filtered["time"])
        print(f"Test eval times: {len(test_eval_times)} points")
        
        if len(test_eval_times) == 0:
            print("Warning: No valid evaluation times for test - using validation results only")
            test_results = {
                'c_index': np.nan,
                'ipcw_c_index': np.nan,
                'ibs': np.nan,
                'mean_auc': np.nan
            }
        else:
            # Construct MITRA training set 
            X_mitra_train, y_mitra_train = construct_mitra_binary_trainset(
                x_trainval_imputed, y_trainval, test_eval_times
            )

            if len(X_mitra_train) == 0:
                print("Warning: No training data for test phase - using validation results only")
                test_results = {
                    'c_index': np.nan,
                    'ipcw_c_index': np.nan,
                    'ibs': np.nan,
                    'mean_auc': np.nan
                }
            else:
                # Sample 10000 rows if dataset is larger
                if len(X_mitra_train) > 10000:
                    print(f"\nSampling 10000 rows from {len(X_mitra_train)} total rows...")
                    sample_idx = np.random.choice(len(X_mitra_train), size=10000, replace=False)
                    X_mitra_train = X_mitra_train.iloc[sample_idx].reset_index(drop=True)
                    y_mitra_train = y_mitra_train.iloc[sample_idx].reset_index(drop=True)

                # Convert to TabularDataset 
                train_data = pd.DataFrame(X_mitra_train)
                train_data['target'] = y_mitra_train
                train_data = TabularDataset(train_data)

                # Initialize and train Mitra model
                # Create a temporary directory for AutoGluon
                tmp_dir = tempfile.mkdtemp()
                mitra_predictor = TabularPredictor(label='target', path=tmp_dir)
                mitra_predictor.fit(
                    train_data,
                    hyperparameters={
                        'MITRA': {'fine_tune': False}
                    }
                )

                # Construct MITRA test set
                X_mitra_test, test_patient_ids = construct_mitra_binary_testset(
                    x_test_filtered, test_eval_times
                )

                if len(X_mitra_test) == 0:
                    print("Warning: No test data created - using validation results only")
                    test_results = {
                        'c_index': np.nan,
                        'ipcw_c_index': np.nan,
                        'ibs': np.nan,
                        'mean_auc': np.nan
                    }
                else:
                    # Convert test data to TabularDataset 
                    test_data = TabularDataset(pd.DataFrame(X_mitra_test))

                    # Predict survival probabilities
                    probs = mitra_predictor.predict_proba(test_data)
                    probs = probs.to_numpy()
                    surv_probs = probs[:, 0]

                    n_patients = len(x_test_filtered)
                    S_test = np.zeros((n_patients, len(test_eval_times)))

                    for i, t in enumerate(test_eval_times):
                        mask = np.isclose(X_mitra_test["eval_time"], t)
                        S_test[test_patient_ids[mask], i] = surv_probs[mask]

                    # Ensure survival probabilities are non-increasing over time    
                    S_test = np.minimum.accumulate(S_test, axis=1)

                    # Calculate risk scores as average of (1 - survival_prob) across multiple eval times
                    test_risk_scores = np.mean(1 - S_test, axis=1)  # Average risk across all evaluation times

                    # Final metrics on test set
                    test_c_index, *_ = concordance_index_censored(
                        y_test_filtered["event"], 
                        y_test_filtered["time"], 
                        test_risk_scores  # Use scalar risk score
                    )

                    test_ipcw_c_index, *_ = concordance_index_ipcw(
                        y_trainval, y_test_filtered, test_risk_scores, tau=y_trainval["time"].max()
                    )

                    test_ibs = integrated_brier_score(y_trainval, y_test_filtered, S_test, test_eval_times) if len(test_eval_times) > 0 else np.nan
                    
                    _, test_mean_auc = cumulative_dynamic_auc(
                        y_trainval, y_test_filtered, test_risk_scores, test_eval_times  # Use scalar risk score
                    ) if len(test_eval_times) > 0 else (None, np.nan)

                    test_results = {
                        'c_index': float(test_c_index),
                        'ipcw_c_index': float(test_ipcw_c_index),
                        'ibs': float(test_ibs),
                        'mean_auc': float(test_mean_auc)
                    }

                # Clean up temporary directory
                shutil.rmtree(tmp_dir)
        # Create results row
        best_row = {
            "dataset": dataset_name,
            "k": args.k,
            "n_eval_times": len(test_eval_times) if len(test_eval_times) > 0 else len(val_eval_times),
            "val_c_index": round(val_results['c_index'] if not np.isnan(val_results['c_index']) else -1, 4),
            "val_ipcw_c_index": round(val_results['ipcw_c_index'] if not np.isnan(val_results['ipcw_c_index']) else -1, 4),
            "val_ibs": round(val_results['ibs'] if not np.isnan(val_results['ibs']) else -1, 4),
            "val_mean_auc": round(val_results['mean_auc'] if not np.isnan(val_results['mean_auc']) else -1, 4),
            "c_index": round(test_results['c_index'] if not np.isnan(test_results['c_index']) else -1, 4),
            "ipcw_c_index": round(test_results['ipcw_c_index'] if not np.isnan(test_results['ipcw_c_index']) else -1, 4),
            "ibs": round(test_results['ibs'] if not np.isnan(test_results['ibs']) else -1, 4),
            "mean_auc": round(test_results['mean_auc'] if not np.isnan(test_results['mean_auc']) else -1, 4),
        }
        
        print("="*50)
        print(f"Final results for dataset {dataset_name} (k={args.k}):")
        print(f"Number of eval time points: {best_row['n_eval_times']}")
        print(f"Validation C-index: {best_row['val_c_index']:.4f}")
        print(f"Validation IPCW C-index: {best_row['val_ipcw_c_index']:.4f}")
        print(f"Validation Integrated Brier Score (IBS): {best_row['val_ibs']:.4f}")
        print(f"Validation mean AUC: {best_row['val_mean_auc']:.4f}")
        print(f"Test C-index: {best_row['c_index']:.4f}")
        print(f"Test IPCW C-index: {best_row['ipcw_c_index']:.4f}")
        print(f"Test Integrated Brier Score (IBS): {best_row['ibs']:.4f}")
        print(f"Test mean AUC: {best_row['mean_auc']:.4f}")

        # ========================= write results =========================
        if best_row is not None:
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["dataset", "k", "n_eval_times", "val_c_index", "val_ipcw_c_index", "val_ibs", "val_mean_auc", "c_index", "ipcw_c_index", "ibs", "mean_auc"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow(best_row)

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        continue