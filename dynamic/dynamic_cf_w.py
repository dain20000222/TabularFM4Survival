# =============================================================================
# Dynamic CF models (XGBoost / TabPFN / MITRA) with the IPCW-weighted loss (2026-10-25)
# =============================================================================
# Copy of ../tmlr_ipcw_10_15/dynamic/dynamic_xgboost_cf.py (the paper's dynamic pipeline on data_mean:
# patient split, landmarks from the training event times, LOCF rows, one model pooled over landmarks
# with tau as a feature, cummin, risk score, risk-file format). Changes (IPCWLOSS):
#   * --model xgboost | tabpfn | mitra. XGBoost keeps the paper's hyperparameter grid; TabPFN and MITRA
#     (in context, classifiers.py) tune only the two feature flags, as in the paper.
#   * training rows of landmark tau: patients at risk at tau (U_i > tau) x horizons t > tau; label
#     Y~ = 1(U_i <= t); the row's weight (cf_weights.py, all times measured from tau):
#       --weighting ipcw : W = G(t - tau | x(tau)) / G(U_i - tau | x(tau)) if delta_i = 1, U_i <= t;
#                          1 if U_i > t; 0 otherwise (censored at or before t: row dropped)
#       --weighting abl  : 1, 1, 0  (the paper's unweighted masked loss; = the paper's rows)
#     G (--censoring km | kmf | cox | rsf) is fitted at every landmark on the TRAINING patients at
#     risk at tau, with their LOCF covariates at tau.
#   * XGBoost: W as sample weights, no rescaling.
#   * TabPFN / MITRA (in context, no training loss; they are not modified unless fine-tuned):
#     default = every row with W > 0 in the context (tag cf_{model}_full); these rows are the same
#     for W and W^abl (= the paper's rows). With --thin (comparison only) each row is kept with
#     probability W (Bernoulli, seed 42; tag cf_{model}_thin_{G}). TabPFN then subsamples 50,000 rows
#     when there are more (as the paper).
# Output: cf_xgboost_{ipcw_{G}|abl}_risk_{k}.csv, cf_{mitra,tabpfn}_{full|thin_{G}}_risk_{k}.csv
# (the paper's risk-file format); evaluate with eval_dynamic_w.py.
#   python dynamic_cf_w.py --model xgboost --k 20 --weighting ipcw --censoring km
#   python dynamic_cf_w.py --model mitra --k 20                       # full context
#   python dynamic_cf_w.py --model mitra --k 20 --thin --censoring km
# =============================================================================
# =============================================================================
# Dynamic XGBoost Survival Analysis Implementation
# =============================================================================
# This script performs dynamic survival analysis using XGBoost (Extreme Gradient
# Boosting) with longitudinal data. It implements landmark analysis where survival
# predictions are made at multiple time points using Last Observation Carried
# Forward (LOCF) for missing covariate values. XGBoost transforms survival
# analysis into binary classification problems using gradient boosting.
# =============================================================================

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
import random
import csv
import sys
import torch
from sksurv.nonparametric import kaplan_meier_estimator

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration and Setup
# =============================================================================

# ============================================================
# Parse command line arguments
# ============================================================
parser = argparse.ArgumentParser(description='Run dynamic XGBoost with different k values for landmark quantiles')
parser.add_argument('--k', type=float, default=20, help='Step size for landmark time percentiles (default: 20)')
parser.add_argument('--binning', type=str, default='quantile', choices=['quantile', 'km_quantile', 'pooled'],  # POOLED (11-05)
                    help='Landmark binning strategy: quantile or km_quantile (default: km_quantile)')
parser.add_argument('--model', default='xgboost', choices=['xgboost', 'tabpfn', 'mitra', 'mitra_lora'])  # IPCWLOSS; DLORA (11-01)
parser.add_argument('--weighting', default='ipcw', choices=['ipcw', 'abl'])  # IPCWLOSS
parser.add_argument('--censoring', default='km', choices=['km', 'kmf', 'cox', 'rsf'])  # IPCWLOSS: G model of the loss
parser.add_argument('--floor', type=float, default=None, help='floor of G in the weights (default: 0 for km, 0.05 for kmf / cox / rsf); floor sensitivity')  # IPCWLOSS
parser.add_argument('--thin', action='store_true', help='in-context models: keep each row with probability W (comparison only)')  # IPCWLOSS
parser.add_argument('--datasets', nargs='*', default=None)  # IPCWLOSS
# DLORA (11-01): MITRA fine-tuned with LoRA on the weighted loss (mitra_lora_w.py), one adapter per dataset and
# feature setting, trained on the pooled rows of all landmarks of the training patients (patient-grouped episodes).
parser.add_argument('--lora_lrs', type=float, nargs='*', default=[1e-4, 3e-4])
parser.add_argument('--lora_steps', type=int, default=200)
parser.add_argument('--lora_rank', type=int, default=8)
parser.add_argument('--lora_select', default='cindex', choices=['bce', 'cindex'],
                    help='learning rate and stopping step: weighted validation BCE, or validation IPCW C-index (mean over landmarks)')
# DOOF (11-01): out-of-fold predictions of the training patients, for calibration (calibrate_dynamic.py)
parser.add_argument('--oof_folds', type=int, default=0)
# MC (11-07): multi-class formulation (SurvivalPFN): at each landmark, (X(tau), delta) -> interval of the remaining time
# among the later landmarks; S(t) = P(interval index > index of t | delta = 1). No loss weights.
parser.add_argument('--formulation', default='cf', choices=['cf', 'mc'])
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
data_dir = os.path.join("data_mean")
dataset_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
if args.datasets:  # IPCWLOSS
    dataset_files = [f for f in dataset_files if f[:-4] in set(args.datasets)]

binning_suffix = f"_{args.binning}" if args.binning != "quantile" else ""
k_token = str(float(args.k))  # 20.0, 16.66666, 12.5, 10.0: the names of the paper's risk files
# IPCWLOSS: tag. In-context models without --thin use every row with W > 0 (identical for W and W^abl).
IN_CONTEXT = args.model in ("tabpfn", "mitra")
if IN_CONTEXT and not args.thin:
    args.weighting = "abl"  # the row set {W > 0}; no G needed
    run_tag = f"cf_{args.model}_full"
elif IN_CONTEXT:
    assert args.weighting == "ipcw", "--thin needs --weighting ipcw (thinning with W^abl keeps every row)"
    run_tag = f"cf_{args.model}_thin_{args.censoring}"
else:
    assert not args.thin, "--thin applies to the in-context models only"
    run_tag = f"cf_{args.model}_{args.weighting}" + (f"_{args.censoring}" if args.weighting == "ipcw" else "")
    if args.model == "mitra_lora":
        run_tag += "_selc" if args.lora_select == "cindex" else ""
if args.floor is not None:
    assert args.weighting == "ipcw", "--floor applies to the weights of loss (B) only"
    run_tag += f"_f{args.floor:g}"
if args.formulation == "mc":  # MC (11-07)
    assert args.model in ("xgboost", "tabpfn", "mitra") and not args.thin and args.floor is None
    run_tag = f"mc_{args.model}"
risk_out_path = f"{run_tag}_risk_{k_token}{binning_suffix}.csv"
oof_out_path = f"{run_tag}_oof_{k_token}{binning_suffix}.csv"  # DOOF

print(f"Total datasets: {len(dataset_files)}")
print(f"Datasets to process: {dataset_files if dataset_files else 'None'}")
print(f"Using k={args.k} with binning={args.binning} for landmark generation")

# Hyperparameters to loop over
hyperparameters = []
if args.model == "xgboost":
    for quantile_info in [False, True]:
        for n_estimators in [100, 200]:
            for learning_rate in [0.1, 0.3]:
                for max_depth in [3, 6]:
                    for time_since in [False, True]:
                        hyperparameters.append({
                            'quantile_info': quantile_info,
                            'n_estimators': n_estimators,
                            'learning_rate': learning_rate,
                            'max_depth': max_depth,
                            'time_since': time_since
                        })
else:  # IPCWLOSS: in-context models tune the two feature flags only (as the paper's MITRA / TabPFN scripts)
    for quantile_info in [False, True]:
        for time_since in [False, True]:
            hyperparameters.append({'quantile_info': quantile_info, 'time_since': time_since})

# ============================================================
# Helper functions
# ============================================================
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
    if binning == "pooled":
        # POOLED: quantiles of the pooled covariate measurement times (start times of the rows, > 0) and
        # event times T (one per patient with an event; censoring times are not used), on np.arange(k, 100, k).
        per_pid = train_df.groupby("pid").agg(t_end=(time_col_for_tau, "max"), status=("event", "max"))
        ev = per_pid.loc[per_pid["status"] == 1, "t_end"].to_numpy(dtype=float)
        # measurement times after each patient's baseline row (11-06: was "> 0", which kept the entry ages of oldmort)
        if "time" in train_df.columns and time_col_for_tau != "time":
            t0 = train_df.groupby("pid")["time"].transform("min")
            meas = train_df.loc[train_df["time"] > t0, "time"].to_numpy(dtype=float)
        else:
            meas = np.array([])
        pooled = np.concatenate([meas, ev])
        if len(pooled) == 0:
            return np.array([], dtype=float)
        percentiles = [p for p in np.arange(k, 100, k) if round(p, 2) < 100]
        return np.unique(np.asarray(np.percentile(pooled, percentiles), dtype=float))
    if binning == "quantile":
        # the rule of the paper's dynamic runs (root dynamic_mitra.py): percentiles of the
        # event-row times (not one time per patient) on np.arange(k, 100, k), dropping 100.
        # The per-patient / exact-1/K version below differs for csl and for k = 16.66666.
        event_times = train_df.loc[train_df["event"] == 1, time_col_for_tau].to_numpy(dtype=float)
        if len(event_times) == 0:
            return np.array([], dtype=float)
        percentiles = [p for p in np.arange(k, 100, k) if round(p, 2) < 100]
        return np.asarray(np.percentile(event_times, percentiles), dtype=float)
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

def construct_binary_trainset_for_landmark(df, tau, future_times, feature_cols, default_values, taus, quantile_info, time_since, followup_col="time2"):
    """
    Construct binary training dataset for a specific landmark tau.
    
    For each patient with T_i > tau OR delta_i = 1:
    - Get LOCF covariates at tau
    - For each future time t in future_times (where t >= tau):
      - Create row: (X_tau, t, [t_idx], label) - t_idx optional
      - Label = 1 if event occurred by time t, 0 otherwise
    
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
    _ROW_META.clear()
    
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
                
        # For each future evaluation time
        for t in future_times:
            if t <= tau:
                continue  # Skip times before landmark
            
            # Label: 1 if event occurred by time t, 0 otherwise
            label = int(d_i == 1 and t_i <= t)

            if d_i == 0 and t_i < t:
                continue  # Censored before t → cannot label
            
            extras = [float(t)]  # eval_time

            if quantile_info:
                t_idx = next((i for i, tau_val in enumerate(taus) if np.isclose(tau_val, t)), -1)
                extras.append(float(t_idx))

            if time_since:
                extras.append(float(tsl))  # time_since_last

            row = np.concatenate([x_tau, extras])
            
            rows.append(row)
            labels.append(label)
            _ROW_META.append((pid, t_i, d_i, float(t)))
    
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
# MC (11-07): multi-class formulation
# ============================================================
def construct_mc_trainset_for_landmark(df, tau, taus, feature_cols, default_values, quantile_info, time_since, followup_col="time2"):
    """One row per patient at risk at tau: LOCF covariates at tau (+ landmark index, + time since the last
    measurement), the event indicator delta, and the class = index of the interval of T among the later landmarks
    (class c <=> taus_after[c-1] < T <= taus_after[c]; taus_after = landmarks after tau). The columns are those of the
    CF rows (eval_time is a placeholder = tau and is not used by the MC model; t_idx is the landmark index)."""
    po = df.sort_values(followup_col).groupby("pid").agg(t_event=(followup_col, "max"), status_event=("event", "max"))
    po = po[po["t_event"] > tau]
    after = np.asarray([t for t in taus if t > tau], dtype=float)
    li = next((i for i, tv in enumerate(taus) if np.isclose(tv, tau)), -1)
    rows, cls, dl, pids = [], [], [], []
    for pid in po.index:
        if time_since:
            x_tau, tsl = get_locf_covariates(df, pid, tau, feature_cols, default_values, return_time_since=True)
        else:
            x_tau = get_locf_covariates(df, pid, tau, feature_cols, default_values)
        extras = [float(tau)] + ([float(li)] if quantile_info else []) + ([float(tsl)] if time_since else [])
        rows.append(np.concatenate([x_tau, extras]))
        cls.append(int(np.searchsorted(after, float(po.loc[pid, "t_event"]), side="left")))
        dl.append(float(po.loc[pid, "status_event"]))
        pids.append(pid)
    names = list(feature_cols) + ["eval_time"] + (["t_idx"] if quantile_info else []) + (["time_since_last"] if time_since else [])
    return pd.DataFrame(rows, columns=names), pd.Series(cls, dtype=int), np.asarray(dl, dtype=float), np.asarray(pids)


class DynMC:
    """MC model pooled over landmarks (inputs: the CF columns without eval_time, t_idx = landmark index, + delta)."""

    def __init__(self, name, taus, hp=None):
        from mc_classifier import MCClassifier
        self.taus, self.hp = np.asarray(taus, dtype=float), hp or {}
        self.clf = MCClassifier(name, SEED)
        if name == "xgboost":  # the dynamic XGBoost grid (hyperparameters of the loop)
            import xgboost as xgb
            self._xgb = lambda C: xgb.XGBClassifier(n_estimators=self.hp["n_estimators"], learning_rate=self.hp["learning_rate"],  # noqa: E731
                                                    max_depth=self.hp["max_depth"], random_state=SEED,
                                                    eval_metric="mlogloss" if C > 2 else "logloss")

    def _inputs(self, X, delta):
        Z = X.drop(columns=["eval_time"]).reset_index(drop=True).copy()
        Z["delta_input"] = np.asarray(delta, dtype=float)
        return Z

    def fit(self, X, delta, y):
        self.cols = list(X.columns)
        Z, y = self._inputs(X, delta), np.asarray(y, dtype=int)
        if self.clf.name == "xgboost":
            self.clf.classes = np.unique(y)
            self.clf.model = None
            if len(self.clf.classes) > 1:
                self.clf.model = self._xgb(len(self.clf.classes))
                self.clf.model.fit(Z.to_numpy(dtype=float), np.searchsorted(self.clf.classes, y))
        else:
            self.clf.fit(Z, y)
        return self

    def predict_S(self, A):
        """A: rows (pid, tau, t) in the CF column order -> S(t | X(tau), delta = 1) per row."""
        X = pd.DataFrame(np.asarray(A, dtype=float), columns=self.cols)
        tau, t = X["tau"].to_numpy(), X["eval_time"].to_numpy()
        if "t_idx" in X:  # in the prediction rows t_idx is the index of the horizon; the model uses the landmark index
            X["t_idx"] = [float(np.argmin(np.abs(self.taus - v))) for v in tau]
        P = self.clf.predict_class_proba(self._inputs(X, np.ones(len(X))), len(self.taus) + 1)
        cum = np.cumsum(P, axis=1)
        S = np.ones(len(X))
        for i in range(len(X)):
            if t[i] > tau[i] + 1e-12:
                j = int(np.searchsorted(self.taus[self.taus > tau[i] + 1e-12], t[i] - 1e-9, side="left"))
                S[i] = 1.0 - cum[i, j]
        return np.clip(S, 0.0, 1.0)

    def close(self):
        self.clf.close()


# ============================================================
# Global accumulators for CSV output
# ============================================================
all_risk_rows = []
all_oof_rows = []  # DOOF
all_lora_info = []  # DLORA: selected learning rate / step per dataset and feature setting


# =============================================================================
# IPCW: censoring model per landmark and the weights of the CF rows
# =============================================================================
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from classifiers import Classifier  # noqa: E402  (IPCWLOSS)
from cf_weights import cf_weights, fit_censoring, thin  # noqa: E402  (IPCWLOSS)

_ROW_META = []   # (pid, t_i, d_i, t) of every training row of the current landmark
_CM_CACHE = {}   # (dataset, tau) -> fitted censoring model of the loss
_W_STATS = []    # per dataset / hyperparameter / landmark: weight statistics


def _at_risk(frame, tau, feature_cols):
    """Patients of `frame` at risk at tau: LOCF covariates at tau, time from tau, event."""
    po = (frame.sort_values(time_col_for_tau).groupby("pid")
          .agg(t_event=(time_col_for_tau, "max"), status_event=("event", "max")))
    po = po[po["t_event"] > tau]
    X = np.array([get_locf_covariates(frame, p, tau, feature_cols, default_values) for p in po.index]).reshape(len(po), -1)
    return X, po["t_event"].to_numpy(float) - tau, po["status_event"].to_numpy() == 1


def apply_cf_weights(X, y, tau, feature_cols, hyperparameter):
    """IPCWLOSS: W (or W^abl) of every training row of landmark tau; zero-weight rows dropped.
    XGBoost: returns the weights; TabPFN / MITRA: every kept row (default) or rows kept with probability W (--thin)."""
    meta = np.array(_ROW_META, dtype=float).reshape(-1, 4)
    _ROW_META.clear()
    if len(X) == 0:
        apply_cf_weights.last_pids = np.zeros(0)
        return X, y, np.ones(0)
    assert len(meta) == len(X), (len(meta), len(X))
    pid, t_i, d_i, t = meta.T
    cm, floor = None, 0.0
    if args.weighting == "ipcw":
        key = (dataset_name, float(tau))
        if key not in _CM_CACHE:
            Xc, tc, ec = _at_risk(train_processed, tau, feature_cols)
            _CM_CACHE[key] = fit_censoring(args.censoring, Xc, tc, ec, seed=SEED, floor=args.floor)
        cm, floor = _CM_CACHE[key]
    Xs = X[list(feature_cols)].to_numpy(dtype=float)
    w = cf_weights(cm, floor, Xs, t_i - tau, d_i == 1, t - tau, args.weighting)
    keep = w > 0
    X, y, w = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True), w[keep]
    apply_cf_weights.last_pids = pid[keep]  # DLORA / DOOF: patient of every kept row
    st = {"dataset": dataset_name, "k": args.k, "hyperparameter": str(hyperparameter), "tau": float(tau),
          "censoring_model": cm.kind_used if cm is not None else "none", "rows": int(len(w)),
          "rows_zero": int((~keep).sum()), "weight_mean": float(w.mean()) if len(w) else np.nan,
          "weight_min": float(w.min()) if len(w) else np.nan, "weight_max": float(w.max()) if len(w) else np.nan}
    if args.model != "xgboost" and args.thin:
        k2 = thin(w, SEED)
        X, y, w = X[k2].reset_index(drop=True), y[k2].reset_index(drop=True), np.ones(int(k2.sum()))
        apply_cf_weights.last_pids = apply_cf_weights.last_pids[k2]
        st["rows_thinned_out"] = int((~k2).sum())
    _W_STATS.append(st)
    print(f"weights ({st['censoring_model']}) at tau={tau:.6g}: {st['rows']} rows, zero {st['rows_zero']}, "
          f"mean W {st['weight_mean']:.3f}" + (f", thinned out {st['rows_thinned_out']}" if args.thin else ""))
    return X, y, w


# ============================================================
# DLORA / DOOF (11-01): fine-tuning and out-of-fold predictions
# ============================================================
def _outcomes(frame):
    return (frame.sort_values(time_col_for_tau).groupby("pid")
            .agg(t_event=(time_col_for_tau, "max"), status_event=("event", "max")))


def _landmark_rows(frame, pids, taus, quantile_info, time_since, feature_cols, default_values, cols):
    """Prediction rows (as for the validation / test patients) of the patients `pids` of `frame` at every landmark
    where they are at risk: X in the training column order, and (pid, tau, t) of every row."""
    po = _outcomes(frame)
    parts, meta = [], []
    for tau_idx in range(len(taus) - 1):
        tau, future = taus[tau_idx], taus[tau_idx:]
        at = po.index[(po["t_event"] > tau) & po.index.isin(pids)]
        sub = frame[frame["pid"].isin(at)]
        if len(sub) == 0:
            continue
        X, p, t = construct_binary_testset_for_landmark(sub, tau, future, feature_cols, default_values, taus,
                                                        quantile_info, time_since, time_col_for_tau)
        if len(X) == 0:
            continue
        X = X.copy()
        X["tau"] = float(tau)
        parts.append(X[cols])
        meta.append(pd.DataFrame({"pid": np.asarray(p), "tau": float(tau), "t": np.asarray(t, dtype=float)}))
    if not parts:
        return None, None
    return pd.concat(parts, ignore_index=True), pd.concat(meta, ignore_index=True)


def _monotone(meta, S):
    df = meta.assign(S=np.asarray(S, dtype=float)).sort_values(["tau", "pid", "t"], kind="mergesort")
    df["S"] = df.groupby(["tau", "pid"])["S"].cummin()
    return df


def _mean_cindex(meta, S, frame_eval, frame_train):
    """Mean over landmarks of the IPCW C-index of the risk score of the pipeline (mean of 1 - S over the horizons
    after tau), as metrics.py: training patients with follow-up beyond tau for G, evaluation times truncated at
    the last training time."""
    from sksurv.metrics import concordance_index_ipcw
    from sksurv.util import Surv
    df = _monotone(meta, S)
    po_e, po_t = _outcomes(frame_eval), _outcomes(frame_train)
    tmax = po_t["t_event"].max()
    cs = []
    for tau, g in df.groupby("tau"):
        fut = g[~np.isclose(g["t"].to_numpy(), float(tau))]
        if fut.empty:
            continue
        risk = fut.groupby("pid")["S"].apply(lambda v: float(np.mean(1.0 - v)))
        tr = po_t[po_t["t_event"] > tau]
        ev = po_e.loc[risk.index].copy()
        over = ev["t_event"] > tmax
        ev.loc[over, "status_event"] = 0
        ev.loc[over, "t_event"] = tmax
        try:
            cs.append(concordance_index_ipcw(Surv.from_arrays(tr["status_event"].to_numpy() == 1, tr["t_event"].to_numpy(float)),
                                             Surv.from_arrays(ev["status_event"].to_numpy() == 1, ev["t_event"].to_numpy(float)),
                                             risk.to_numpy())[0])
        except Exception:
            pass
    return float(np.mean(cs)) if cs else np.nan


def fit_dynamic_lora(X_train_all, y_train, w_train, pid_train, taus, quantile_info, time_since, feature_cols, default_values,
                     hyperparameter, lr=None, n_steps=None):
    """DLORA: MITRA + LoRA on the pooled weighted rows (episodes grouped by patient). Without n_steps: learning rate
    and stopping step chosen on the validation patients (weighted validation BCE with the per-landmark G of the
    training patients, or the mean validation IPCW C-index), step 0 (no fine-tuning) a candidate."""
    from mitra_lora_w import MitraLoRA
    cols = list(X_train_all.columns)
    make = lambda lr_: MitraLoRA(rank=args.lora_rank, alpha=2.0 * args.lora_rank, lr=lr_, max_steps=args.lora_steps, seed=SEED, verbose=False)  # noqa: E731
    Xa = X_train_all.to_numpy(dtype=np.float32)
    if n_steps is not None:  # refit on a subset (out-of-fold) with the selected setting
        return make(lr).fit(Xa, y_train, pid_train, n_steps=n_steps, sample_weight=w_train), None
    # validation rows with labels and weights (BCE criterion; also required by the early stopping loop)
    vx, vy, vw = [], [], []
    n_stats = len(_W_STATS)
    for tau_idx in range(len(taus) - 1):
        tau, future = taus[tau_idx], taus[tau_idx:]
        Xv, yv = construct_binary_trainset_for_landmark(val_processed, tau, future, feature_cols, default_values, taus,
                                                        quantile_info, time_since, time_col_for_tau)
        Xv, yv, wv = apply_cf_weights(Xv, yv, tau, feature_cols, hyperparameter)
        if len(Xv):
            Xv = Xv.copy()
            Xv["tau"] = float(tau)
            vx.append(Xv[cols]); vy.append(np.asarray(yv)); vw.append(wv)
    del _W_STATS[n_stats:]  # the validation rows are not training rows
    Xv, yv, wv = pd.concat(vx, ignore_index=True).to_numpy(dtype=np.float32), np.concatenate(vy), np.concatenate(vw)
    veval_X, veval_fn = None, None
    if args.lora_select == "cindex":
        Xe, meta = _landmark_rows(val_processed, val_processed["pid"].unique(), taus, quantile_info, time_since, feature_cols, default_values, cols)
        veval_X = Xe.to_numpy(dtype=np.float32)
        veval_fn = lambda p1: -_mean_cindex(meta, 1.0 - p1, val_processed, train_processed)  # noqa: E731
    best = None
    for lr_ in args.lora_lrs:
        m = make(lr_).fit(Xa, y_train, pid_train, X_val=Xv, y_val=yv, sample_weight=w_train, val_weight=wv,
                          val_eval_X=veval_X, val_eval_fn=veval_fn)
        print(f"  [DLORA] lr={lr_:g}: best step {m.best_step}, val {args.lora_select} {m.best_val_loss:.4f} (step 0: {m.step0_val_loss:.4f})")
        if best is None or m.best_val_loss < best.best_val_loss - 1e-9:
            best = m
    info = {"dataset": dataset_name, "k": args.k, "hyperparameter": str(hyperparameter), "lora_lr": best.lr, "lora_best_step": best.best_step,
            "lora_select": args.lora_select, "val_step0": best.step0_val_loss, "val_best": best.best_val_loss}
    all_lora_info.append(info)
    return best, info


def oof_dynamic(X_train_all, y_train, w_train, pid_train, taus, quantile_info, time_since, feature_cols, default_values,
                hyperparameter, lora_info=None):
    """DOOF: out-of-fold survival predictions of the training patients: --oof_folds folds by patient; each fold is
    predicted by the model fitted on the rows of the other folds (in context: the other folds as context; XGBoost:
    refit; LoRA: refit with the selected learning rate and number of steps), at every landmark, monotone in t."""
    from sklearn.model_selection import KFold
    assert args.model in ("mitra", "tabpfn", "xgboost", "mitra_lora"), "out-of-fold predictions: mitra, tabpfn, xgboost or mitra_lora"
    cols = list(X_train_all.columns)
    pids = np.unique(pid_train)
    rows = []
    for f, (i_fit, i_pred) in enumerate(KFold(args.oof_folds, shuffle=True, random_state=SEED).split(pids)):
        mask = np.isin(pid_train, pids[i_fit])
        Xf, yf, wf, gf = X_train_all[mask].reset_index(drop=True), y_train[mask], w_train[mask], pid_train[mask]
        if args.formulation == "mc":  # MC (11-07): refit / new context on the other folds
            mdl = DynMC(args.model, taus, hyperparameter).fit(Xf, oof_dynamic.d_all[mask], yf)
            p0 = lambda A, m=mdl: m.predict_S(A)  # noqa: E731
        elif args.model in ("mitra", "tabpfn"):  # DOOF (11-03): TabPFN as MITRA (in context, other folds as context)
            if args.model == "tabpfn" and len(Xf) > 50000:  # as the real fit
                keep = np.random.RandomState(SEED).choice(len(Xf), size=50000, replace=False)
                Xf, yf = Xf.iloc[keep].reset_index(drop=True), np.asarray(yf)[keep]
            mdl = Classifier(args.model, SEED).fit(Xf, yf)
            p0 = lambda A, m=mdl: m.predict_p0(pd.DataFrame(A, columns=cols))  # noqa: E731
        elif args.model == "xgboost":
            from xgboost import XGBClassifier
            mdl = XGBClassifier(n_estimators=hyperparameter["n_estimators"], learning_rate=hyperparameter["learning_rate"],
                                max_depth=hyperparameter["max_depth"], random_state=SEED, eval_metric="logloss")
            mdl.fit(Xf.to_numpy(dtype=float), yf, sample_weight=wf)
            p0 = lambda A, m=mdl: m.predict_proba(np.asarray(A, dtype=float))[:, 0]  # noqa: E731
        else:
            mdl, _ = fit_dynamic_lora(Xf, yf, wf, gf, taus, quantile_info, time_since, feature_cols, default_values, hyperparameter,
                                      lr=lora_info["lora_lr"], n_steps=lora_info["lora_best_step"])
            p0 = lambda A, m=mdl: m.predict_proba(np.asarray(A, dtype=np.float32))[:, 0]  # noqa: E731
        Xp, meta = _landmark_rows(train_processed, pids[i_pred], taus, quantile_info, time_since, feature_cols, default_values, cols)
        if Xp is None:
            continue
        df = _monotone(meta, p0(Xp.to_numpy(dtype=float)))
        df = df[~np.isclose(df["t"].to_numpy(), df["tau"].to_numpy())]
        rows += [{"pid": int(p), "tau": float(tau), "t": float(t), "risk": np.nan, "surv_prob": float(sv), "dataset": dataset_name,
                  "hyperparameter": str(hyperparameter), "eval_set": "oof", "k": args.k}
                 for p, tau, t, sv in zip(df["pid"], df["tau"], df["t"], df["S"])]
    print(f"  [DOOF] {len(rows)} out-of-fold survival predictions of {len(pids)} training patients")
    return rows


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
        # then train ONE XGBoost model per dataset per hyperparameter.
        # -----------------------------
        combined_X_parts = []
        combined_y_parts = []
        combined_w_parts = []  # IPCW
        combined_pid_parts = []  # DLORA / DOOF
        combined_d_parts = []  # MC: event indicators of the training rows

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
            if args.formulation == "mc":  # MC (11-07): one row per patient at risk, class = interval of T, delta as input
                X_train, y_train, d_mc, pid_mc = construct_mc_trainset_for_landmark(
                    train_processed, tau, taus, feature_cols_processed, default_values, quantile_info, time_since, time_col_for_tau)
                w_train = np.ones(len(X_train))
                apply_cf_weights.last_pids = pid_mc
                combined_d_parts.append(d_mc)
            else:
                X_train, y_train = construct_binary_trainset_for_landmark(
                    train_processed, tau, future_times, feature_cols_processed, default_values, taus,
                    quantile_info, time_since, time_col_for_tau
                )
                X_train, y_train, w_train = apply_cf_weights(X_train, y_train, tau, feature_cols_processed, hyperparameter)  # IPCWLOSS

            if len(X_train) == 0:
                print(f"No training data for tau{tau_idx}; skipping")
                continue

            # Add tau as a feature so a single model can learn across landmarks
            X_train = X_train.copy()
            X_train["tau"] = float(tau)

            combined_X_parts.append(X_train)
            combined_y_parts.append(y_train)
            combined_w_parts.append(w_train)  # IPCW
            combined_pid_parts.append(apply_cf_weights.last_pids)  # DLORA / DOOF

        if len(combined_X_parts) == 0:
            print(f"No training data across any tau; skipping dataset {dataset_name}")
            continue

        X_train_all = pd.concat(combined_X_parts, axis=0, ignore_index=True)
        y_train_all = pd.concat(combined_y_parts, axis=0, ignore_index=True)

        # Print class distribution
        n_ones = y_train_all.sum()
        n_zeros = len(y_train_all) - n_ones
        print(f"Training samples (train only, all taus): {len(y_train_all)} (event: {n_ones}, no event: {n_zeros})")

        # -----------------------------
        # Train XGBoost model (single model for all taus)
        # -----------------------------
        w_all = np.concatenate(combined_w_parts)  # IPCWLOSS
        if args.model == "tabpfn" and len(X_train_all) > 50000:  # as the paper's dynamic_tabpfn.py
            print(f"Sampling 50000 rows from {len(X_train_all)} total rows...")
            sample_idx = np.random.choice(len(X_train_all), size=50000, replace=False)
            X_train_all = X_train_all.iloc[sample_idx].reset_index(drop=True)
            y_train_all = y_train_all.iloc[sample_idx].reset_index(drop=True)
            w_all = w_all[sample_idx]  # DOOF (11-03): keep weights and patient ids aligned with the sampled rows
            if args.formulation == "mc":
                combined_d_parts = [np.concatenate(combined_d_parts)[sample_idx]]
            combined_pid_parts = [np.concatenate(combined_pid_parts)[sample_idx]]
        X_train_array = X_train_all.to_numpy(dtype=float)
        y_train_array = y_train_all.to_numpy(dtype=int)

        d_all = np.concatenate(combined_d_parts) if args.formulation == "mc" else None  # MC
        if args.formulation == "mc":  # MC (11-07)
            mc_model = DynMC(args.model, taus, hyperparameter).fit(X_train_all, d_all, y_train_array)
            predict_p0 = lambda A, m=mc_model: m.predict_S(A)  # noqa: E731  (survival at t of each row)
        elif args.model == "xgboost":
            from xgboost import XGBClassifier  # IPCWLOSS: lazy import (the mitra env has no xgboost)
            xgboost_model = XGBClassifier(
                n_estimators=hyperparameter['n_estimators'],
                learning_rate=hyperparameter['learning_rate'],
                max_depth=hyperparameter['max_depth'],
                random_state=SEED,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            xgboost_model.fit(X_train_array, y_train_array, sample_weight=w_all)  # IPCWLOSS: W as written, no rescaling
            predict_p0 = lambda A: xgboost_model.predict_proba(A)[:, 0]  # noqa: E731
        elif args.model == "mitra_lora":  # DLORA
            pid_all = np.concatenate(combined_pid_parts)
            lora_model, lora_info = fit_dynamic_lora(X_train_all, y_train_array, w_all, pid_all, taus, quantile_info, time_since,
                                                     feature_cols_processed, default_values, hyperparameter)
            predict_p0 = lambda A, m=lora_model: m.predict_proba(np.asarray(A, dtype=np.float32))[:, 0]  # noqa: E731
        else:  # IPCWLOSS: in context on the kept rows (all rows with W > 0, or the thinned rows with --thin)
            icl_model = Classifier(args.model, SEED).fit(X_train_all, y_train_array)
            predict_p0 = lambda A: icl_model.predict_p0(pd.DataFrame(A, columns=list(X_train_all.columns)))  # noqa: E731
        oof_dynamic.d_all = d_all  # MC
        if args.oof_folds:  # DOOF
            all_oof_rows += oof_dynamic(X_train_all, y_train_array, w_all, np.concatenate(combined_pid_parts), taus, quantile_info,
                                        time_since, feature_cols_processed, default_values, hyperparameter,
                                        lora_info if args.model == "mitra_lora" else None)

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

                X_val_array = X_val[list(X_train_all.columns)].to_numpy(dtype=float)
                surv_probs = predict_p0(X_val_array)  # IPCWLOSS: P(Y=0 | X, t) = survival at t

                # Enforce monotone survival: S(t) must be non-increasing in t for each pid at this tau
                tmp = pd.DataFrame({
                    "pid": val_pids,
                    "t": val_eval_times.astype(float),
                    "surv_prob": surv_probs.astype(float),
                })

                # Sort by pid then time, apply cummin (non-increasing survival)
                tmp = tmp.sort_values(["pid", "t"], kind="mergesort")
                tmp["surv_prob"] = tmp.groupby("pid")["surv_prob"].cummin()

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

                X_test_array = X_test[list(X_train_all.columns)].to_numpy(dtype=float)
                surv_probs = predict_p0(X_test_array)  # IPCWLOSS
                tmp = pd.DataFrame({
                    "pid": test_pids,
                    "t": test_eval_times.astype(float),
                    "surv_prob": surv_probs.astype(float),
                })
                tmp = tmp.sort_values(["pid", "t"], kind="mergesort")
                tmp["surv_prob"] = tmp.groupby("pid")["surv_prob"].cummin()
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
    if all_oof_rows:  # DOOF
        pd.DataFrame(all_oof_rows).to_csv(oof_out_path, index=False)
        print(f"Out-of-fold predictions saved to: {oof_out_path}")
    if all_lora_info:  # DLORA
        pd.DataFrame(all_lora_info).to_csv(f"{run_tag}_lorainfo_{k_token}{binning_suffix}.csv", index=False)
    if _W_STATS:  # IPCWLOSS
        pd.DataFrame(_W_STATS).to_csv(f"{run_tag}_weights_{k_token}.csv", index=False)
    print(f"Results saved to: {risk_out_path}")
    print(f"Total rows: {len(results_df)}")
    val_rows = (results_df['eval_set'] == 'val').sum()
    test_rows = (results_df['eval_set'] == 'test').sum()
    print(f"  Val rows: {val_rows} ({(~results_df[results_df['eval_set']=='val']['risk'].isna()).sum()} risk scores, {(~results_df[results_df['eval_set']=='val']['surv_prob'].isna()).sum()} surv probs)")
    print(f"  Test rows: {test_rows} ({(~results_df[results_df['eval_set']=='test']['risk'].isna()).sum()} risk scores, {(~results_df[results_df['eval_set']=='test']['surv_prob'].isna()).sum()} surv probs)")
else:
    print("No results to save")

print("\nProcessing complete!")
