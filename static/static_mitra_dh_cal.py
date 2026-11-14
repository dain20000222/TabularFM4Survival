# =============================================================================
# MITRA-DH with post-hoc calibration
# =============================================================================
# The MITRA-DH script of Table 7 / Table 1 (../tmlr_mitra_icl_rerun_10_13/static_km/static_mitra_discrete.py, MITRA in
# context; same split, grid, row construction and S = prod (1 - h)), extended to store what calibrate_static.py needs:
# raw validation and test survival predictions and 5-fold out-of-fold predictions of the training subjects (validation
# phase: train subjects on the validation grid; test phase: train+val subjects on the test grid), in the npz format of
# static_cf_w.py. Output: preds/K{K}/static_mitra_dh{_kmq}/{dataset}.npz, results/static_mitra_dh{_kmq}_evaluation_final_{K}.csv.
# Then: python ../tmlr_ipcw_loss_10_25/calibrate_static.py --models static_mitra_dh_kmq --fit_on oof
#   python static_mitra_dh_cal.py --k 10 --binning km_quantile --oof_folds 5
# =============================================================================
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
import os as _os, sys as _sys  # ICL
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "tmlr_mitra_icl_rerun_10_13"))  # ICL
from icl_predictor import TabularDataset, TabularPredictor  # ICL: MITRA in context, no AutoGluon
MAX_TRAIN_ROWS = float("inf")  # ICL: no subsample; MITRA draws its own 8,192-row context
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
parser.add_argument('--k', type=float, default=10, help='Step size for evaluation time percentiles (default: 10)')
parser.add_argument('--binning', type=str, default='quantile', choices=['quantile', 'equal_width', 'km_quantile'],
                    help='Binning strategy: quantile (default), equal_width, or km_quantile (KM-weighted)')
parser.add_argument('--oof_folds', type=int, default=5, help='DHCAL: folds of the out-of-fold predictions (0: none)')
parser.add_argument('--data_dir', default='static_data')
parser.add_argument('--datasets', nargs='*', default=None)
args = parser.parse_args()

# =============================================================================
# Data Configuration
# =============================================================================

# Directory containing the datasets
data_dir = args.data_dir  # DHCAL

# List all CSV files in the directory
dataset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# Output CSV path based on k parameter
binning_suffix = f"_{args.binning}" if args.binning != "quantile" else ""
k_token = f"{float(args.k):.8g}"
K = int(np.ceil(100 / float(args.k) - 1e-6))  # DHCAL: k = 7 -> K = 15 (as the other runs)
tag = "static_mitra_dh" + ("_kmq" if args.binning == "km_quantile" else "")
os.makedirs("results", exist_ok=True)
pred_dir = os.path.join("preds", f"K{K}", tag); os.makedirs(pred_dir, exist_ok=True)
csv_path = os.path.join("results", f"{tag}_evaluation_final_{K}.csv")
if os.path.isfile(csv_path):
    os.remove(csv_path)

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

def construct_mitra_discrete_hazard_trainset(x_train, y_train, times):
    """Construct person-bin training data for discrete hazard modeling."""
    if len(times) == 0:
        print("Warning: No valid evaluation times for training set construction")
        return pd.DataFrame(), pd.Series(dtype=int)

    times = np.sort(np.unique(times))

    T = y_train["time"]
    delta = y_train["event"]
    n_train = len(T)

    dataset_rows = []
    binary_labels = []

    for i in range(n_train):
        T_i = T[i]
        delta_i = delta[i]
        x_i = x_train.iloc[i].values

        for j, t_end in enumerate(times):
            t_start = 0.0 if j == 0 else times[j - 1]

            if T_i <= t_start:
                break

            row = np.concatenate([x_i, [t_end]])

            if delta_i and (T_i <= t_end):
                dataset_rows.append(row)
                binary_labels.append(1)
                break

            if T_i > t_end:
                dataset_rows.append(row)
                binary_labels.append(0)
                continue

            break

    if len(dataset_rows) == 0:
        print("Warning: No valid training samples created")
        return pd.DataFrame(), pd.Series(dtype=int)

    feature_cols = list(x_train.columns) + ["eval_time"]
    X_mitra = pd.DataFrame(dataset_rows, columns=feature_cols)
    y_mitra = pd.Series(binary_labels)

    n_ones = sum(binary_labels)
    n_zeros = len(binary_labels) - n_ones
    print(f"\nTraining set class distribution:")
    print(f"Class 0 (no event): {n_zeros} ({n_zeros/len(binary_labels)*100:.1f}%)")
    print(f"Class 1 (event):    {n_ones} ({n_ones/len(binary_labels)*100:.1f}%)")
    print(f"Total samples:      {len(binary_labels)}")

    return X_mitra, y_mitra

def construct_mitra_discrete_hazard_testset(x_test, times):
    """Construct person-bin inference dataset for discrete hazard modeling."""
    if len(times) == 0:
        print("Warning: No valid evaluation times for test set construction")
        feature_cols = list(x_test.columns) + ["eval_time"]
        return pd.DataFrame(columns=feature_cols), np.array([])

    times = np.sort(np.unique(times))
    n_test = x_test.shape[0]

    test_rows = []
    patient_ids = []

    for i in range(n_test):
        x_i = x_test.iloc[i].values
        for t in times:
            row = np.concatenate([x_i, [t]])
            test_rows.append(row)
            patient_ids.append(i)

    feature_cols = list(x_test.columns) + ["eval_time"]
    X_mitra_test = pd.DataFrame(test_rows, columns=feature_cols)

    return X_mitra_test, np.array(patient_ids)


def get_positive_class_probs(probs_df):
    """Extract positive-class (hazard=1) probabilities from AutoGluon output."""
    if 1 in probs_df.columns:
        return probs_df[1].to_numpy()
    if "1" in probs_df.columns:
        return probs_df["1"].to_numpy()
    if probs_df.shape[1] >= 2:
        return probs_df.iloc[:, 1].to_numpy()
    return probs_df.iloc[:, 0].to_numpy()


def row_labels_and_mask_discrete(y_structured, patient_ids, eval_times):
    """Row-wise discrete-hazard labels + validity mask for row metrics."""
    times = y_structured["time"][patient_ids]
    events = y_structured["event"][patient_ids]

    eval_times = np.asarray(eval_times, dtype=float)
    valid_mask = np.zeros_like(eval_times, dtype=bool)

    uniq = np.sort(np.unique(eval_times))
    start_map = {float(uniq[0]): 0.0}
    for prev, cur in zip(uniq[:-1], uniq[1:]):
        start_map[float(cur)] = float(prev)

    t_start = np.array([start_map.get(float(t), 0.0) for t in eval_times], dtype=float)
    y_true = ((events == True) & (times > t_start) & (times <= eval_times)).astype(int)

    cens_mask = (events == False) & (eval_times <= times)
    valid_mask[cens_mask] = True

    event_mask = (events == True)
    if np.any(event_mask):
        event_bin_end = np.zeros_like(times, dtype=float)
        for idx in np.where(event_mask)[0]:
            t = float(times[idx])
            j = int(np.searchsorted(uniq, t, side="left"))
            if j >= len(uniq):
                j = len(uniq) - 1
            event_bin_end[idx] = float(uniq[j])

        event_bin_start = np.zeros_like(times, dtype=float)
        for idx in np.where(event_mask)[0]:
            event_bin_start[idx] = float(start_map.get(float(event_bin_end[idx]), 0.0))

        valid_mask[event_mask & (eval_times <= event_bin_start)] = True
        valid_mask[event_mask & (eval_times == event_bin_end)] = True

    return y_true, valid_mask


# =============================================================================
# DHCAL: MITRA-DH with the predictions needed for post-hoc calibration
# =============================================================================
from sklearn.model_selection import StratifiedKFold  # noqa: E402


def dh_fit_predict(x_fit, y_fit, x_pred, times):
    """The DH pipeline of the original script: one-hot (columns of the fit set) and mean imputation fitted on the fit
    subjects, person-bin hazard rows on `times`, MITRA in context, S(t_k | x) = prod_{j<=k} (1 - h_j)."""
    a = pd.get_dummies(x_fit, drop_first=True)
    b = pd.get_dummies(x_pred, drop_first=True)
    a, b = a.align(b, join="left", axis=1, fill_value=0)
    cols = a.columns.tolist()
    imp = SimpleImputer().fit(a.loc[:, cols])
    a = pd.DataFrame(imp.transform(a.loc[:, cols]), columns=cols, index=x_fit.index)
    b = pd.DataFrame(imp.transform(b.loc[:, cols]), columns=cols, index=x_pred.index)
    Xtr, ytr = construct_mitra_discrete_hazard_trainset(a, y_fit, times)
    train_data = pd.DataFrame(Xtr)
    train_data["target"] = ytr
    tmp_dir = tempfile.mkdtemp()
    pred = TabularPredictor(label="target", path=tmp_dir)
    pred.fit(TabularDataset(train_data), hyperparameters={"MITRA": {"fine_tune": False}})
    Xte, pids = construct_mitra_discrete_hazard_testset(b, times)
    h = get_positive_class_probs(pred.predict_proba(TabularDataset(pd.DataFrame(Xte))))
    shutil.rmtree(tmp_dir)
    H = np.zeros((len(b), len(times)))
    for i, t in enumerate(times):
        m = np.isclose(Xte["eval_time"], t)
        H[pids[m], i] = h[m]
    return np.cumprod(1 - np.clip(H, 1e-8, 1 - 1e-8), axis=1)


def oof(x, y, times):
    out = np.full((len(y), len(times)), np.nan)
    for i_fit, i_pred in StratifiedKFold(args.oof_folds, shuffle=True, random_state=SEED).split(x, y["event"]):
        out[i_pred] = dh_fit_predict(x.iloc[i_fit], y[i_fit], x.iloc[i_pred], times)
    return out


def metrics(y_ref, y_ev, S, times):
    risk = np.mean(1 - S, axis=1)
    return {"c_index": concordance_index_censored(y_ev["event"], y_ev["time"], risk)[0],
            "ipcw_c_index": concordance_index_ipcw(y_ref, y_ev, risk, tau=y_ref["time"].max())[0],
            "ibs": integrated_brier_score(y_ref, y_ev, S, times), "mean_auc": cumulative_dynamic_auc(y_ref, y_ev, risk, times)[1]}


r4 = lambda v: round(float(v), 4) if v is not None and np.isfinite(v) else -1
files = sorted(dataset_files)
if args.datasets:
    files = [f for f in files if f[:-4] in set(args.datasets)]
for file_name in files:
    dataset_name = file_name[:-4]
    try:
        print("=" * 50 + f"\nProcessing dataset: {dataset_name}")
        df = pd.read_csv(os.path.join(data_dir, file_name))
        if "pid" in df.columns:
            df = df.drop(columns=["pid"])
        x = df[df.columns.difference(["time", "event"])].copy()
        y = Surv.from_arrays(event=df["event"].astype(bool), time=df["time"])
        x_trainval, x_test, y_trainval, y_test = train_test_split(x, y, test_size=0.15, stratify=y["event"], random_state=SEED)
        x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.1765, stratify=y_trainval["event"], random_state=SEED)
        percentiles, n_bins, _ = k_to_percentile_grid(args.k)
        if args.binning == "quantile":
            base = np.percentile(y_trainval["time"], percentiles)
        else:
            km_time, km_surv = kaplan_meier_estimator(y_trainval["event"], y_trainval["time"])
            idx = np.searchsorted(-km_surv, -np.linspace(km_surv[0], km_surv[-1], n_bins + 2)[1:-1])
            base = km_time[idx[idx < len(km_time)]].astype(float)
        base = np.unique(base)
        tmax = y_trainval["time"].max()
        y_test_f = y_test.copy()
        m = y_test_f["time"] >= tmax
        y_test_f["time"][m] = tmax - 0.01
        y_test_f["event"][m] = False
        val_times, test_times = get_valid_eval_times(base, y_val["time"]), get_valid_eval_times(base, y_test_f["time"])
        if len(val_times) == 0 or len(test_times) == 0:
            print("No valid evaluation times - skipping dataset")
            continue
        S_val = dh_fit_predict(x_train, y_train, x_val, val_times)
        val = metrics(y_train, y_val, S_val, val_times)
        tr_oof = oof(x_train, y_train, val_times) if args.oof_folds else np.zeros((0, 0))
        S_test = dh_fit_predict(x_trainval, y_trainval, x_test, test_times)
        test = metrics(y_trainval, y_test_f, S_test, test_times)
        tv_oof = oof(x_trainval, y_trainval, test_times) if args.oof_folds else np.zeros((0, 0))
        risk = np.mean(1 - S_test, axis=1)
        np.savez_compressed(os.path.join(pred_dir, f"{dataset_name}.npz"),
                            y_train_time=y_trainval["time"], y_train_event=y_trainval["event"],
                            y_test_time=y_test_f["time"], y_test_event=y_test_f["event"],
                            cidx_risk=risk, cidx_tau=tmax, ibs_S=S_test, ibs_times=test_times, auc_risk=risk, auc_times=test_times,
                            test_S_raw=S_test, val_S_raw=S_val, val_times=val_times,
                            y_val_time=y_val["time"], y_val_event=y_val["event"], y_tr_time=y_train["time"], y_tr_event=y_train["event"],
                            tr_oof_S_raw=tr_oof, tv_oof_S_raw=tv_oof)
        row = {"dataset": dataset_name, "K": K, "val_c_index": r4(val["c_index"]), "val_ipcw_c_index": r4(val["ipcw_c_index"]),
               "val_ibs": r4(val["ibs"]), "val_mean_auc": r4(val["mean_auc"]), "c_index": r4(test["c_index"]),
               "ipcw_c_index": r4(test["ipcw_c_index"]), "ibs": r4(test["ibs"]), "mean_auc": r4(test["mean_auc"])}
        new = not os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{dataset_name}: IPCW-C {row['ipcw_c_index']} IBS {row['ibs']} AUC {row['mean_auc']}")
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
