# IPCWLOSS: copy of ../tmlr_dyn_metrics_fix_10_18/eval_dynamic_nanfix.py; runs the metrics.py / metrics_val.py of THIS folder (NANFIX + cf_* files).
# =============================================================================
# Dynamic metrics with covariate-dependent censoring weights
# NANFIX copy (2026-10-18) of ../tmlr_ipcw_10_15/dynamic/eval_dynamic.py: runs the metrics.py /
# metrics_val.py of THIS folder (out-of-follow-up metrics are NaN, not 0). Only the path of ../common changed.
# =============================================================================
# Runs metrics.py and metrics_val.py (copies of the paper's data_mean scripts,
# tmlr_datamean_rerun_10_11/tcsa) unchanged on the *_risk_*.csv files of a
# folder, after replacing the three sksurv IPCW functions they import by the
# versions of ../common/ipcw.py with a censoring model G(t | x(tau)):
#
#   At landmark tau the paper weights by the Kaplan-Meier of the censoring times
#   of the TRAINING patients still at risk at tau. Here G is fitted on the same
#   patients, with their LOCF covariates at tau (one-hot of the num_ / fac_
#   columns of data_mean, median-imputed on the training patients) and times
#   measured from tau, and evaluated at the test (or validation) patients'
#   covariates at tau.
#
#   --censoring marginal   sksurv, i.e. the paper's numbers (nothing replaced)
#               km         the same through ipcw.py (check: equals marginal)
#               cox / rsf  ridge Cox / random survival forest of the censoring times
#               selected   per dataset (and landmark set) the model with the lowest
#                          IBS of the censoring time on the validation patients at
#                          risk, averaged over the landmarks (fitted on train)
#   --floor                G is floored at this value
#
# The risk files are linked into OUT, where metrics.csv, metrics_val.csv,
# censoring_selection.csv and the logs are written.
#
#   python eval_dynamic.py --preds preds --censoring selected --floor 0.05 --out results/selected_f0.05
# =============================================================================

import argparse
import glob
import os
import runpy
import sys

import numpy as np

if not hasattr(np, "infty"):  # older scikit-survival uses np.infty, removed in NumPy 2
    np.infty = np.inf
import pandas as pd
import sksurv.metrics as skm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tmlr_ipcw_10_15", "common"))  # NANFIX copy
import ipcw  # noqa: E402
from censoring import KINDS, CensoringModel, select_censoring  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--preds", required=True, help="folder with the *_risk_*.csv files")
parser.add_argument("--censoring", default="selected", choices=["marginal", "km", "cox", "rsf", "selected"])
parser.add_argument("--floor", type=float, default=0.05)
parser.add_argument("--data_mean", default=os.path.join(HERE, "data_mean"))
parser.add_argument("--out", required=True)
parser.add_argument("--agg", default="custom", choices=["custom", "sksurv"],
                    help="PKGAGG: aggregation of the horizons within a landmark (metrics.py DYN_AGG)")
args = parser.parse_args()
os.environ["DYN_AGG"] = args.agg  # PKGAGG: read by metrics.py
SEED = 42

preds = os.path.abspath(args.preds)
data_mean = os.path.abspath(args.data_mean)
out = os.path.abspath(args.out)
os.makedirs(out, exist_ok=True)
for f in glob.glob(os.path.join(out, "*_risk_*.csv")):
    os.remove(f)
for f in sorted(glob.glob(os.path.join(preds, "*_risk_*.csv"))):
    os.symlink(f, os.path.join(out, os.path.basename(f)))
if not os.path.exists(os.path.join(out, "data_mean")):
    os.symlink(data_mean, os.path.join(out, "data_mean"))


# ------------------------------------------------------------ covariates at tau
_cov = {}


def covariate_frame(dataset):
    """One-hot covariate rows of data_mean (pid, time, features)."""
    if dataset not in _cov:
        df = pd.read_csv(os.path.join(data_mean, f"{dataset}.csv"))
        feats = [c for c in df.columns if c.startswith("num_") or c.startswith("fac_")]
        X = pd.get_dummies(df[feats], drop_first=True).astype(float)
        _cov[dataset] = pd.concat([df[["pid", "time"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    return _cov[dataset]


_cov_at = {}


def covariates_at(dataset, pids, tau, train_pids):
    key = (dataset, float(tau), np.asarray(pids).tobytes(), len(train_pids))
    if key not in _cov_at:
        _cov_at[key] = _covariates_at(dataset, pids, tau, train_pids)
    return _cov_at[key]


def _covariates_at(dataset, pids, tau, train_pids):
    """LOCF covariates at tau (last row starting at or before tau), imputed with the
    medians of the training patients' rows."""
    f = covariate_frame(dataset)
    feats = [c for c in f.columns if c not in ("pid", "time")]
    med = f.loc[f["pid"].isin(train_pids), feats].median()
    last = f[f["time"] <= tau].sort_values("time").groupby("pid").tail(1).set_index("pid")
    first = f.sort_values("time").groupby("pid").head(1).set_index("pid")  # (no row before tau: first row)
    rows = last.reindex(pids)
    miss = rows[feats].isna().all(axis=1)
    rows.loc[miss, feats] = first.reindex(np.asarray(pids)[miss.to_numpy()])[feats].to_numpy()
    return rows[feats].fillna(med).fillna(0.0).to_numpy(dtype=float)


# --------------------------------------------------------- censoring models
_cm, _sel, sel_rows = {}, {}, []
_orig = {n: getattr(skm, n) for n in ("concordance_index_ipcw", "cumulative_dynamic_auc", "brier_score")}
_selecting = [False]  # the selection's own IBS (sksurv) must use the original functions


def _ctx():
    """dataset, landmark, test / train patient frames and val / train pids of the caller."""
    ctx = {}
    fr = sys._getframe(3)  # skip _ctx <- test_G <- the replacement metric (its own `tau` argument)
    while fr is not None:
        loc = fr.f_locals
        for k in ("tau", "merged", "survival_train_tau"):
            if k not in ctx and k in loc:
                ctx[k] = loc[k]
        for k in ("dataset_name", "val_pids", "train_pids", "tau_times", "survival_all"):
            if k not in ctx and k in loc:
                ctx[k] = loc[k]
        fr = fr.f_back
    return ctx


def censoring_model(dataset, tau, kind, train_frame, train_pids):
    key = (dataset, float(tau), kind, len(train_frame))
    if key not in _cm:
        X = covariates_at(dataset, train_frame["pid"].to_numpy(), tau, train_pids)
        _cm[key] = CensoringModel(kind, seed=SEED).fit(X, train_frame["time"].to_numpy(float) - tau,
                                                        train_frame["event"].to_numpy().astype(bool))
    return _cm[key]


def selected_kind(c):
    dataset, taus = c["dataset_name"], tuple(sorted(float(t) for t in c["tau_times"]))
    key = (dataset, taus)
    if key not in _sel:
        surv, train_pids, val_pids = c["survival_all"], c["train_pids"], c["val_pids"]
        tr = surv[surv["pid"].isin(train_pids)]
        va = surv[surv["pid"].isin(val_pids)]
        pairs = []
        for tau in taus:
            a, b = tr[tr["time"] > tau], va[va["time"] > tau]
            if len(a) and len(b):
                pairs.append((covariates_at(dataset, a["pid"].to_numpy(), tau, train_pids),
                              a["time"].to_numpy(float) - tau, a["event"].to_numpy().astype(bool),
                              covariates_at(dataset, b["pid"].to_numpy(), tau, train_pids),
                              b["time"].to_numpy(float) - tau, b["event"].to_numpy().astype(bool)))
        _selecting[0] = True
        try:
            kind, scores = select_censoring(pairs, seed=SEED)
        finally:
            _selecting[0] = False
        _sel[key] = kind
        sel_rows.append({"dataset": dataset, "landmarks": len(taus), "selected": kind,
                         **{f"val_cens_ibs_{k}": v for k, v in scores.items()}})
        print(f"[eval_dynamic] {dataset} ({len(taus)} landmarks): censoring model {kind} {scores}")
    return _sel[key]


_G = {}


def test_G():
    c = _ctx()
    dataset, tau = c["dataset_name"], float(c["tau"])
    kind = args.censoring if args.censoring != "selected" else selected_kind(c)
    m = c["merged"]
    pids = m["pid"].to_numpy()
    # the same landmark / patients recur for every hyperparameter setting and metric
    key = (dataset, tau, kind, len(c["survival_train_tau"]), pids.tobytes())
    if key not in _G:
        cm = censoring_model(dataset, tau, kind, c["survival_train_tau"], c["train_pids"])
        X = covariates_at(dataset, pids, tau, c["train_pids"])
        _G[key] = ipcw.TestG(cm, X, len(m), floor=args.floor, shift=tau)
    return _G[key]


# ------------------------------------------------ replacements of the sksurv metrics
def concordance_index_ipcw(survival_train, survival_test, estimate, tau=None, tied_tol=1e-8):
    if _selecting[0]:
        return _orig["concordance_index_ipcw"](survival_train, survival_test, estimate, tau=tau, tied_tol=tied_tol)
    G = test_G()
    c = ipcw.cindex_ipcw(survival_test["event"], survival_test["time"], estimate, G, tau=tau, tied_tol=tied_tol)
    return (c, np.nan, np.nan, np.nan, np.nan)


def cumulative_dynamic_auc(survival_train, survival_test, estimate, times, tied_tol=1e-8):
    if _selecting[0]:
        return _orig["cumulative_dynamic_auc"](survival_train, survival_test, estimate, times, tied_tol=tied_tol)
    G = test_G()
    return ipcw.cumulative_dynamic_auc(survival_test["event"], survival_test["time"], estimate, times, G, tied_tol)


def brier_score(survival_train, survival_test, estimate, times):
    if _selecting[0]:
        return _orig["brier_score"](survival_train, survival_test, estimate, times)
    G = test_G()
    return ipcw.brier_scores(survival_test["event"], survival_test["time"], estimate, times, G)


if args.censoring != "marginal":
    skm.concordance_index_ipcw = concordance_index_ipcw
    skm.cumulative_dynamic_auc = cumulative_dynamic_auc
    skm.brier_score = brier_score

os.chdir(out)
print(f"[eval_dynamic] censoring={args.censoring} floor={args.floor}: "
      f"{len(glob.glob('*_risk_*.csv'))} risk files -> {out}")
sys.argv = ["metrics.py"]
runpy.run_path(os.path.join(HERE, "metrics.py"), run_name="__main__")
if sel_rows:
    pd.DataFrame(sel_rows).to_csv("censoring_selection.csv", index=False)
sys.argv = ["metrics_val.py"]
runpy.run_path(os.path.join(HERE, "metrics_val.py"), run_name="__main__")
print(f"[eval_dynamic] done: {os.path.join(out, 'metrics_val.csv')}")
