# =============================================================================
# TabPFN Multiclass Survival Implementation
# =============================================================================
# This script performs survival analysis using TabPFN.
# The model is trained as a delta-conditioned multiclass classifier over
# discretized time bins: covariates + delta are used as inputs, and the target is
# the observed time bin. At inference, delta is fixed to 1 to request the event-
# time distribution.
# =============================================================================

import pandas as pd
import numpy as np
if not hasattr(np, "infty"):
    np.infty = np.inf
import os
import csv
import argparse
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from tabpfn import TabPFNClassifier
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored, integrated_brier_score, concordance_index_ipcw
import torch
# xgboost is no longer used (replaced by AutoGluon/MITRA)
import warnings
import random
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
parser = argparse.ArgumentParser(description='Run TabPFN multiclass survival analysis (delta-conditioned)')
parser.add_argument('--k', type=int, default=10, help='Number of discrete time bins to use (default: 10)')
args = parser.parse_args()

# =============================================================================
# Data Configuration
# =============================================================================

# Directory containing the datasets
data_dir = os.path.join("static_data")

# List all CSV files in the directory
dataset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# Number of time bins requested by the user
K = int(args.k)
if K < 2:
    raise ValueError("--k must be at least 2 so the time axis can be discretized")

# Output CSV path based on K to match the multiclass benchmark format
csv_path = f"static_tabpfn_evaluation_multiclass_{K}.csv"

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
    
    # Apply basic time constraints with buffer to avoid boundary issues
    valid_times = base_eval_times[
        (base_eval_times > min_test_time + buffer) & 
        (base_eval_times < max_test_time - buffer)
    ]
    
    return valid_times


def make_time_bin_edges(times, num_bins):
    """Build monotone bin edges from observed times using quantiles."""
    times = np.asarray(times, dtype=float)
    times = times[np.isfinite(times)]
    if len(times) == 0:
        return np.array([])

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(times, quantiles)
    edges = np.unique(edges)

    if len(edges) < 2:
        return np.array([])

    # Expand the outer boundaries slightly so every observed time falls in-range.
    edges = edges.astype(float)
    edges[0] -= 1e-8
    edges[-1] += 1e-8
    return edges


def assign_time_bins(times, bin_edges):
    """Assign each time to a discrete bin index."""
    times = np.asarray(times, dtype=float)
    if len(bin_edges) < 2:
        return np.array([], dtype=int)

    # Bin j corresponds to (edge[j], edge[j+1]] under this convention.
    return np.searchsorted(bin_edges[1:-1], times, side="right")


def construct_multiclass_trainset(x_train, y_train, bin_edges):
    """Construct a delta-conditioned multiclass training set."""
    X_rows = x_train.copy().reset_index(drop=True)
    X_rows["delta_input"] = np.asarray(y_train["event"], dtype=float)
    raw_bins = assign_time_bins(y_train["time"], bin_edges)
    active_bins = np.unique(raw_bins)
    if len(active_bins) == 0:
        return X_rows, pd.Series(dtype=int), active_bins

    bin_to_class = {bin_id: class_id for class_id, bin_id in enumerate(active_bins)}
    y_target = pd.Series([bin_to_class[bin_id] for bin_id in raw_bins], dtype=int)
    return X_rows, y_target, active_bins


def construct_multiclass_testset(x_test, delta_value=1.0):
    """Construct a delta-conditioned test set with a fixed query delta."""
    X_rows = x_test.copy().reset_index(drop=True)
    X_rows["delta_input"] = float(delta_value)
    return X_rows


def survival_from_class_probs(class_probs):
    """Convert class probabilities over time bins into a survival matrix."""
    class_probs = np.asarray(class_probs, dtype=float)
    if class_probs.ndim != 2:
        raise ValueError("class_probs must be a 2D array")

    # Treat the class probabilities as mass over increasing time bins and use
    # the tail mass as survival, matching the discrete baseline's cumulative
    # survival construction.
    tail = np.flip(np.cumsum(np.flip(class_probs, axis=1), axis=1), axis=1)
    if tail.shape[1] == 1:
        return np.zeros((tail.shape[0], 1), dtype=float)

    return np.hstack([tail[:, 1:], np.zeros((tail.shape[0], 1), dtype=float)])


# TabPFN will provide predict_proba as a numpy array; no conversion helper needed.

def clean_column_names(df):
    """
    Clean DataFrame column names to be compatible with XGBoost requirements.
    
    XGBoost has restrictions on feature names and doesn't handle certain
    special characters well. This function sanitizes column names by
    replacing problematic characters with underscores.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with potentially problematic column names
    
    Returns:
    --------
    df : DataFrame
        DataFrame with cleaned column names (modified in-place)
    """
    # Replace problematic characters that can cause XGBoost issues
    clean_names = df.columns.str.replace('[', '_').str.replace(']', '_').str.replace('<', '_')
    df.columns = clean_names

    return df

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

        # Defaults for row-level classification metrics (not used in this setup)
        test_auc_roc = np.nan
        test_bce = np.nan
        # Defaults for validation survival metrics
        val_c_index = np.nan
        val_ipcw_c_index = np.nan
        val_ibs = np.nan
        val_mean_auc = np.nan

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

        # Clean column names
        x_train_ohe = clean_column_names(x_train_ohe)
        x_val_ohe = clean_column_names(x_val_ohe) 
        x_test_ohe = clean_column_names(x_test_ohe)

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

        # Build quantile-based time bins from the training split only.
        val_bin_edges = make_time_bin_edges(y_train["time"], K)
        if len(val_bin_edges) < 2:
            print(f"No valid time bins for validation set, skipping dataset {dataset_name}")
            continue

        val_eval_times_all = val_bin_edges[1:]
        val_eval_times = get_valid_eval_times(val_eval_times_all, y_val["time"])
        print(f"Validation eval times: {len(val_eval_times)} points")
        if len(val_eval_times) == 0:
            print(f"No valid evaluation times for validation set, skipping dataset {dataset_name}")
            continue

        # Build the delta-conditioned multiclass training set from the training split.
        X_train_clf, y_train_clf, val_active_bins = construct_multiclass_trainset(
            x_train_imputed, y_train, val_bin_edges
        )
        if len(val_active_bins) < 2:
            print(f"Not enough populated time bins for validation set, skipping dataset {dataset_name}")
            continue

        val_active_eval_times = val_eval_times_all[val_active_bins]
        val_eval_times_used = get_valid_eval_times(val_active_eval_times, y_val["time"])
        if len(val_eval_times_used) == 0:
            print(f"No valid populated evaluation times for validation set, skipping dataset {dataset_name}")
            continue

        val_eval_mask = np.isin(val_active_eval_times, val_eval_times_used)

        n_val_bins = len(val_active_bins)
        # Train TabPFN on the delta-conditioned multiclass training set
        tabpfn_model = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu', ignore_pretraining_limits=True)
        tabpfn_model.fit(X_train_clf, y_train_clf)

        # Predict event-time bin probabilities on validation with delta fixed to 1.
        X_val_pred = construct_multiclass_testset(x_val_imputed, delta_value=1.0)
        val_probs = tabpfn_model.predict_proba(X_val_pred)

        # Ensure class order (0..n_val_bins-1) — TabPFN returns columns in label order
        val_class_probs = val_probs[:, :n_val_bins]
        val_S_full = survival_from_class_probs(val_class_probs)
        val_S = val_S_full[:, val_eval_mask]
        val_risk_scores = np.mean(1 - val_S, axis=1)

        val_c_index, *_ = concordance_index_censored(y_val["event"], y_val["time"], val_risk_scores)
        val_ipcw_c_index, *_ = concordance_index_ipcw(y_train, y_val, val_risk_scores, tau=y_train["time"].max())
        val_ibs = integrated_brier_score(y_train, y_val, val_S, val_eval_times_used)
        _, val_mean_auc = cumulative_dynamic_auc(y_train, y_val, val_risk_scores, val_eval_times_used)

        # Retrain on train+val and evaluate on test with delta forced to 1.
        x_trainval_ohe = pd.get_dummies(x_trainval, drop_first=True)
        x_test_ohe = pd.get_dummies(x_test, drop_first=True)
        x_trainval_ohe, x_test_ohe = x_trainval_ohe.align(x_test_ohe, join="left", axis=1, fill_value=0)

        x_trainval_ohe = clean_column_names(x_trainval_ohe)
        x_test_ohe = clean_column_names(x_test_ohe)

        covariates_ohe = x_trainval_ohe.columns
        imputer = SimpleImputer().fit(x_trainval_ohe.loc[:, covariates_ohe.tolist()])

        x_trainval_imputed = imputer.transform(x_trainval_ohe.loc[:, covariates_ohe.tolist()])
        x_trainval_imputed = pd.DataFrame(x_trainval_imputed, columns=covariates_ohe, index=x_trainval.index)

        x_test_imputed = imputer.transform(x_test_ohe.loc[:, covariates_ohe.tolist()])
        x_test_imputed = pd.DataFrame(x_test_imputed, columns=covariates_ohe, index=x_test.index)

        y_test_filtered = y_test.copy()

        test_bin_edges = make_time_bin_edges(y_trainval["time"], K)
        if len(test_bin_edges) < 2:
            print(f"No valid time bins for test set, skipping dataset {dataset_name}")
            continue

        test_eval_times_all = test_bin_edges[1:]
        test_eval_times = get_valid_eval_times(test_eval_times_all, y_test_filtered["time"])
        print(f"Test eval times: {len(test_eval_times)} points")
        if len(test_eval_times) == 0:
            print(f"No valid evaluation times for test set, skipping dataset {dataset_name}")
            continue

        X_trainval_clf, y_trainval_clf, test_active_bins = construct_multiclass_trainset(
            x_trainval_imputed, y_trainval, test_bin_edges
        )
        if len(test_active_bins) < 2:
            print(f"Not enough populated time bins for test set, skipping dataset {dataset_name}")
            continue

        test_active_eval_times = test_eval_times_all[test_active_bins]
        test_eval_times_used = get_valid_eval_times(test_active_eval_times, y_test_filtered["time"])
        if len(test_eval_times_used) == 0:
            print(f"No valid populated evaluation times for test set, skipping dataset {dataset_name}")
            continue

        test_eval_mask = np.isin(test_active_eval_times, test_eval_times_used)

        n_test_bins = len(test_active_bins)
        # Train TabPFN on the delta-conditioned multiclass training set
        tabpfn_model = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu', ignore_pretraining_limits=True)
        tabpfn_model.fit(X_trainval_clf, y_trainval_clf)

        X_test_pred = construct_multiclass_testset(x_test_imputed, delta_value=1.0)
        test_probs = tabpfn_model.predict_proba(X_test_pred)
        test_class_probs = test_probs[:, :n_test_bins]

        # Row-level classification metrics are not meaningful for this survival classifier.
        # Keep them as NaN/-1 for schema compatibility.
        test_auc_roc = np.nan
        test_bce = np.nan

        # Convert class probabilities into a survival matrix and a median-time risk score.
        S_test_full = survival_from_class_probs(test_class_probs)
        S = S_test_full[:, test_eval_mask]
        risk_scores = np.mean(1 - S, axis=1)

        c_index, *_ = concordance_index_censored(y_test_filtered["event"], y_test_filtered["time"], risk_scores)
        ipcw_c_index, *_ = concordance_index_ipcw(y_trainval, y_test_filtered, risk_scores, tau=y_trainval["time"].max())
        ibs = integrated_brier_score(y_trainval, y_test_filtered, S, test_eval_times_used)
        _, mean_auc = cumulative_dynamic_auc(y_trainval, y_test_filtered, risk_scores, test_eval_times_used)

        print("="*50)
        print(f"Final results for dataset {dataset_name} (K={K}):")
        print(f"Validation C-index: {float(val_c_index):.4f}")
        print(f"Validation IPCW C-index: {float(val_ipcw_c_index):.4f}")
        print(f"Validation Integrated Brier Score (IBS): {float(val_ibs):.4f}")
        print(f"Validation mean AUC: {float(val_mean_auc):.4f}")
        print(f"Test C-index: {float(c_index):.4f}")
        print(f"Test IPCW C-index: {float(ipcw_c_index):.4f}")
        print(f"Test Integrated Brier Score (IBS): {float(ibs):.4f}")
        print(f"Test mean AUC: {float(mean_auc):.4f}")
        print(f"Row-level AUC-ROC: {test_auc_roc if not np.isnan(test_auc_roc) else -1:.4f}")
        print(f"Row-level BCE: {test_bce if not np.isnan(test_bce) else -1:.4f}")

        best_row = {
            "dataset": dataset_name,
            "K": K,
            "val_c_index": round(float(val_c_index), 4) if not np.isnan(val_c_index) else -1,
            "val_ipcw_c_index": round(float(val_ipcw_c_index), 4) if not np.isnan(val_ipcw_c_index) else -1,
            "val_ibs": round(float(val_ibs) if not np.isnan(val_ibs) else -1, 4),
            "val_mean_auc": round(float(val_mean_auc) if not np.isnan(val_mean_auc) else -1, 4),
            "c_index": round(float(c_index), 4),
            "ipcw_c_index": round(float(ipcw_c_index), 4),
            "ibs": round(float(ibs) if not np.isnan(ibs) else -1, 4),
            "mean_auc": round(float(mean_auc) if not np.isnan(mean_auc) else -1, 4),
            "auc_roc": -1,
            "bce": -1,
        }

        # ========================= write results =========================
        if best_row is not None:
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "dataset",
                        "K",
                        "val_c_index",
                        "val_ipcw_c_index",
                        "val_ibs",
                        "val_mean_auc",
                        "c_index",
                        "ipcw_c_index",
                        "ibs",
                        "mean_auc",
                        "auc_roc",
                        "bce",
                    ],
                )
                if not file_exists:
                    writer.writeheader()
                writer.writerow(best_row)

    except Exception as e:
        print(f"Error: {e}")
        continue
