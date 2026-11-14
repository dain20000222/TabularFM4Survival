# =============================================================================
# Post-hoc calibration of a dynamic CF model, same procedure as calibrate_static.py
# =============================================================================
# Inputs: the risk file of dynamic_cf_w.py ({tag}_risk_{k}.csv: validation and test predictions of the model
# fitted on the training patients) and its out-of-fold file ({tag}_oof_{k}.csv, --oof_folds: survival predictions
# of the training patients, each from the model fitted on the other folds).
# Per dataset and feature setting (hyperparameter):
#   rows (patient i, landmark tau, horizon t > tau) of the patients at risk at tau: p = 1 - S(t | tau), label
#   1(U_i <= t), weight W of the loss with G = Kaplan-Meier of the censoring times of the training patients at risk
#   at tau, times from tau (cf_weights.py); rows with W = 0 dropped.
#   (b) calibrators fitted on the out-of-fold training rows: platt sigma(a logit p + b); platt_t one intercept per
#       horizon t; isotonic (weighted);
#   (c) the option with the lowest weighted validation BCE is selected per dataset and setting ("none" a candidate);
#   (d) the selected calibrator is applied to the validation and test survival probabilities, which are then made
#       monotone in t again. The risk-score rows (C-index, AUC) are left unchanged, so the calibration changes only
#       the survival probabilities, i.e. the Brier score / IBS ("calibs" in the static setting).
# Output: {tag}_cal_risk_{k}.csv in the same format (evaluate with eval_dynamic_w.py) and {tag}_cal_selection_{k}.csv.
#   python calibrate_dynamic.py --risk preds/cf_mitra_full_risk_20.0.csv --oof preds/cf_mitra_full_oof_20.0.csv --data_mean data_mean
# =============================================================================
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cf_weights import cf_weights, fit_censoring  # noqa: E402

warnings.filterwarnings("ignore")
p = argparse.ArgumentParser()
p.add_argument("--risk", required=True)
p.add_argument("--oof", required=True)
p.add_argument("--data_mean", default=os.path.join(HERE, "data_mean"))
p.add_argument("--calibrators", nargs="+", default=["platt", "platt_t", "isotonic"])
p.add_argument("--out_dir", default=None, help="default: the folder of --risk")
args = p.parse_args()
EPS = 1e-6


def logit(q):
    q = np.clip(q, EPS, 1 - EPS)
    return np.log(q / (1 - q))


def fit_calibrator(kind, pv, kv, yv, wv, n_k):
    if len(np.unique(yv)) < 2:
        return None
    if kind == "platt":
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pv)[:, None], yv, sample_weight=wv)
        return lambda q, k: lr.predict_proba(logit(q)[:, None])[:, 1]
    if kind == "platt_t":
        def feats(q, k):
            Z = np.zeros((len(q), 1 + n_k))
            Z[:, 0] = logit(q)
            Z[np.arange(len(q)), 1 + np.asarray(k)] = 1.0
            return Z
        lr = LogisticRegression(C=1e6, max_iter=2000, fit_intercept=False).fit(feats(pv, kv), yv, sample_weight=wv)
        return lambda q, k: lr.predict_proba(feats(q, k))[:, 1]
    if kind == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(pv, yv, sample_weight=wv)
        return lambda q, k: iso.predict(q)
    raise ValueError(kind)


def wbce(pr, y, w):
    pr = np.clip(pr, EPS, 1 - EPS)
    return float(np.sum(w * -(y * np.log(pr) + (1 - y) * np.log(1 - pr))) / np.sum(w))


def outcomes(dataset):
    """Patient-level outcome as metrics.py: last follow-up time, event indicator."""
    df = pd.read_csv(os.path.join(args.data_mean, f"{dataset}.csv"))
    tcol = "time2" if "time2" in df.columns else "time"
    return df.groupby("pid").agg(time=(tcol, "max"), event=("event", "max"))


risk = pd.read_csv(args.risk)
oof = pd.read_csv(args.oof)
out_rows, sel_rows = [], []
for (dataset, hp), g in risk.groupby(["dataset", "hyperparameter"]):
    o = oof[(oof.dataset == dataset) & (oof.hyperparameter == hp)]
    po = outcomes(dataset)
    eval_pids = set(g.pid)
    train = po[~po.index.isin(eval_pids)]
    grid = np.unique(np.concatenate([g.t.dropna().to_numpy(), o.t.to_numpy()]))
    G = {}

    def rows_of(frame):
        """p, grid level, label, weight of the survival-probability rows of `frame` (rows with W > 0)."""
        f = frame[frame.t.notna() & frame.surv_prob.notna()]
        U = po.loc[f.pid, "time"].to_numpy(float)
        D = po.loc[f.pid, "event"].to_numpy() == 1
        tau, t = f.tau.to_numpy(float), f.t.to_numpy(float)
        w = np.zeros(len(f))
        for tv in np.unique(tau):
            if tv not in G:
                at = train[train.time > tv]
                G[tv] = fit_censoring("km", np.zeros((len(at), 1)), at.time.to_numpy(float) - tv, at.event.to_numpy() == 1, seed=42)
            cm, floor = G[tv]
            m = tau == tv
            w[m] = cf_weights(cm, floor, np.zeros((m.sum(), 1)), U[m] - tv, D[m], t[m] - tv, "ipcw")
        keep = w > 0
        return (1.0 - f.surv_prob.to_numpy(float))[keep], np.searchsorted(grid, t)[keep], (U <= t).astype(int)[keep], w[keep]

    fit_rows = rows_of(o)
    val_rows = rows_of(g[g.eval_set == "val"])
    if len(fit_rows[0]) == 0 or len(val_rows[0]) == 0 or len(np.unique(fit_rows[2])) < 2:
        out_rows.append(g)
        sel_rows.append({"dataset": dataset, "hyperparameter": hp, "selected": "none (too few rows)"})
        continue
    cal = {c: fit_calibrator(c, *fit_rows, len(grid)) for c in args.calibrators}
    scores = {"none": wbce(val_rows[0], val_rows[2], val_rows[3])}
    for c, f in cal.items():
        scores[c] = wbce(f(val_rows[0], val_rows[1]), val_rows[2], val_rows[3]) if f is not None else np.inf
    chosen = min(scores, key=scores.get)
    sel_rows.append({"dataset": dataset, "hyperparameter": hp, "selected": chosen, "n_fit_rows": len(fit_rows[0]),
                     "n_val_rows": len(val_rows[0]), **{f"val_bce_{k}": round(v, 5) for k, v in scores.items()}})
    g = g.copy()
    if chosen != "none":
        m = g.t.notna() & g.surv_prob.notna()
        q = 1.0 - g.loc[m, "surv_prob"].to_numpy(float)
        g.loc[m, "surv_prob"] = 1.0 - np.clip(cal[chosen](q, np.searchsorted(grid, g.loc[m, "t"].to_numpy(float))), 0.0, 1.0)
        s = g[m].sort_values(["eval_set", "pid", "tau", "t"], kind="mergesort")
        g.loc[s.index, "surv_prob"] = s.groupby(["eval_set", "pid", "tau"])["surv_prob"].cummin()
    out_rows.append(g)
    print(f"{dataset} {hp}: val BCE " + ", ".join(f"{k} {v:.4f}" for k, v in scores.items()) + f" -> {chosen}")

out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.risk))
base = os.path.basename(args.risk)
tag, k = base.split("_risk_")
out_path = os.path.join(out_dir, f"{tag}_cal_risk_{k}")
pd.concat(out_rows, ignore_index=True).to_csv(out_path, index=False)
pd.DataFrame(sel_rows).to_csv(os.path.join(out_dir, f"{tag}_cal_selection_{k}"), index=False)
print("wrote", out_path)
print(pd.DataFrame(sel_rows).selected.value_counts().to_string())
