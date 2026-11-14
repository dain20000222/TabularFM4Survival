# =============================================================================
# Post-hoc calibration of the static CF models with the IPCW weights
# =============================================================================
# The unweighted in-context CF models under-estimate survival at late grid times. This is addressed by
# calibration (or fine-tuning), with every choice made per dataset on the validation split.
#
# Procedure (--fit_on oof, the revised one of 10-31):
#   (a) static_cf_w.py --oof_folds 5 stores out-of-fold predictions of the training subjects (each fold
#       predicted with the other folds as context / refit), in the validation phase (train subjects,
#       validation grid) and in the test phase (train+val subjects, test grid).
#   (b) Calibrators are fitted on the out-of-fold TRAINING rows (i, k): p = 1 - S_oof(t_k | x_i) (raw),
#       label Y~_ik = 1(U_i <= t_k), weight W_ik of the loss (G = --censoring fitted on the same subjects);
#       rows with W = 0 are dropped. platt: sigma(a logit p + b); platt_t: sigma(a logit p + b_k), one
#       intercept per grid level; isotonic: weighted isotonic regression. All monotone in p.
#   (c) Selection per dataset and K on the VALIDATION predictions of the train-only model: the weighted
#       validation BCE (the value of loss (B) on the validation rows, W with G from train) of "none" and of
#       every calibrator; the smallest wins ("none" is a candidate).
#   (d) Test: the calibrators are refitted on the out-of-fold train+val rows and applied to the test
#       predictions of the train+val model; "{tag}_cal" holds the selected option per dataset, and
#       "{tag}_{calibrator}" every option (for analysis).
# --fit_on val is the earlier procedure (calibrators fitted on the validation rows, applied to the test
# predictions; the selection column then is in-sample and only informative).
# Outputs: preds/K{K}/{tag}_{cal|platt|platt_t|isotonic}/{dataset}.npz, results/{...}_evaluation_final_{K}.csv,
# results/calibration_selection.csv (validation BCE of every option and the choice), results/calibration_summary.csv.
#   python calibrate_static.py --models static_mitra_cf_full static_tabpfn_cf_full static_xgboost_cf_ipcw_km --fit_on oof
# =============================================================================
import argparse
import csv
import os
import sys
import warnings

import numpy as np
import pandas as pd

if not hasattr(np, "infty"):
    np.infty = np.inf
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cf_weights import CENSORING, cf_weights, fit_censoring  # noqa: E402

warnings.filterwarnings("ignore")
p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", required=True, help="model folders in preds/K*/ (tags)")
p.add_argument("--calibrators", nargs="+", default=["platt", "platt_t", "isotonic"])
p.add_argument("--fit_on", default="oof", choices=["oof", "val"])
p.add_argument("--censoring", default="km", choices=list(CENSORING), help="G of the weights W")
p.add_argument("--floor", type=float, default=None)
p.add_argument("--K", type=int, nargs="*", default=[5, 10, 15, 20])
p.add_argument("--datasets", nargs="*", default=None)
p.add_argument("--data_dir", default=os.path.join(HERE, "static_data"), help="covariates (for G = cox / rsf only)")
args = p.parse_args()
EPS = 1e-6
KIND, _ = CENSORING[args.censoring]


def logit(q):
    q = np.clip(q, EPS, 1 - EPS)
    return np.log(q / (1 - q))


def fit_calibrator(kind, pv, kv, yv, wv, n_k):
    """Returns f(p, k) -> calibrated p; None when it cannot be fitted."""
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


def rows(S, times, U, D, X, cm, floor, levels):
    """Calibration rows of the subjects of S: p, grid level, label, weight (rows with W = 0 dropped)."""
    n, m = S.shape
    subj, j = np.repeat(np.arange(n), m), np.tile(np.arange(m), n)
    w = cf_weights(cm, floor, X[subj], U[subj], D[subj], times[j], "ipcw")
    keep = w > 0
    pr = 1.0 - S.ravel()
    y = (U[subj] <= times[j]).astype(int)
    return pr[keep], np.tile(levels, n)[keep], y[keep], w[keep]


def wbce(pr, y, w):
    pr = np.clip(pr, EPS, 1 - EPS)
    return float(np.sum(w * -(y * np.log(pr) + (1 - y) * np.log(1 - pr))) / np.sum(w))


def survival_metrics(y_tr, y_ev, S, times):
    risk = np.mean(1 - S, axis=1)
    return {"c_index": concordance_index_censored(y_ev["event"], y_ev["time"], risk)[0],
            "ipcw_c_index": concordance_index_ipcw(y_tr, y_ev, risk, tau=y_tr["time"].max())[0],
            "ibs": integrated_brier_score(y_tr, y_ev, S, times),
            "mean_auc": cumulative_dynamic_auc(y_tr, y_ev, risk, times)[1]}


def append_csv(path, row):
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def r4(v):
    return round(float(v), 4) if v is not None and np.isfinite(v) else np.nan


_cov = {}


def covariates(dataset):
    """Preprocessed covariates of the train, val and train+val subjects in the order of the npz (G = cox / rsf)."""
    if KIND == "km":
        return None
    if dataset not in _cov:
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(os.path.join(args.data_dir, f"{dataset}.csv")).drop(columns=["pid"])
        x = df[df.columns.difference(["time", "event"])].copy()
        y = Surv.from_arrays(event=df["event"].astype(bool), time=df["time"])
        x_tv, x_te, y_tv, y_te = train_test_split(x, y, test_size=0.15, stratify=y["event"], random_state=42)
        x_tr, x_va, y_tr, y_va = train_test_split(x_tv, y_tv, test_size=0.1765, stratify=y_tv["event"], random_state=42)

        def prep(a, b):
            A = pd.get_dummies(a, drop_first=True).astype(float)
            B = pd.get_dummies(b, drop_first=True).reindex(columns=A.columns, fill_value=0).astype(float)
            return A.fillna(A.mean()).to_numpy(), B.fillna(A.mean()).to_numpy()
        a, b = prep(x_tr, x_va)
        c, _ = prep(x_tv, x_te)
        _cov[dataset] = (a, b, c)
    return _cov[dataset]


def X_of(dataset, which, n):
    cov = covariates(dataset)
    return np.zeros((n, 1)) if cov is None else cov[{"train": 0, "val": 1, "trainval": 2}[which]]


os.makedirs("results", exist_ok=True)
# one selection / summary file per call, so that parallel calls for different models do not overwrite each other
sel_path = os.path.join("results", f"calibration_selection_{'+'.join(args.models)}.csv")
if os.path.isfile(sel_path):
    os.remove(sel_path)
for K in args.K:
    for tag in args.models:
        pdir = os.path.join("preds", f"K{K}", tag)
        if not os.path.isdir(pdir):
            print(f"no {pdir}")
            continue
        # "cal": the option selected on validation; "calibs": the same, but the C-index / AUC use the risk score of the
        # uncalibrated model (calibration changes only the survival probabilities, i.e. the IBS); then every option
        options = ["cal", "calibs"] + args.calibrators
        outs = {c: os.path.join("preds", f"K{K}", f"{tag}_{c}") for c in options}
        for o in outs.values():
            os.makedirs(o, exist_ok=True)
        csvs = {c: os.path.join("results", f"{tag}_{c}_evaluation_final_{K}.csv") for c in options}
        for f in csvs.values():
            if os.path.isfile(f):
                os.remove(f)
        for fn in sorted(os.listdir(pdir)):
            if not fn.endswith(".npz"):
                continue
            name = fn[:-4]
            if args.datasets and name not in set(args.datasets):
                continue
            rec = dict(np.load(os.path.join(pdir, fn)))
            need = ["val_S_raw", "test_S_raw"] + (["tr_oof_S_raw", "tv_oof_S_raw"] if args.fit_on == "oof" else [])
            if any(k not in rec or rec[k].size == 0 for k in need):
                print(f"K={K} {tag} {name}: missing {[k for k in need if k not in rec or rec[k].size == 0]} (rerun static_cf_w.py"
                      f"{' --oof_folds 5' if args.fit_on == 'oof' else ''})")
                continue
            tv, tt = rec["val_times"].astype(float), rec["ibs_times"].astype(float)
            grid = np.unique(np.concatenate([tv, tt]))
            lev_v, lev_t = np.searchsorted(grid, tv), np.searchsorted(grid, tt)
            Utr, Dtr = rec["y_tr_time"].astype(float), rec["y_tr_event"].astype(bool)          # train only
            Uv, Dv = rec["y_val_time"].astype(float), rec["y_val_event"].astype(bool)            # validation
            Utv, Dtv = rec["y_train_time"].astype(float), rec["y_train_event"].astype(bool)      # train+val
            cm_tr, floor = fit_censoring(args.censoring, X_of(name, "train", len(Utr)), Utr, Dtr, seed=42, floor=args.floor)
            cm_tv, _ = fit_censoring(args.censoring, X_of(name, "trainval", len(Utv)), Utv, Dtv, seed=42, floor=args.floor)
            # ---- (b) fit data of the validation phase and (c) the validation rows
            if args.fit_on == "oof":
                fit_v = rows(rec["tr_oof_S_raw"].astype(float), tv, Utr, Dtr, X_of(name, "train", len(Utr)), cm_tr, floor, lev_v)
            else:
                fit_v = rows(rec["val_S_raw"].astype(float), tv, Uv, Dv, X_of(name, "val", len(Uv)), cm_tr, floor, lev_v)
            val_rows = rows(rec["val_S_raw"].astype(float), tv, Uv, Dv, X_of(name, "val", len(Uv)), cm_tr, floor, lev_v)
            if len(np.unique(val_rows[2])) < 2 or len(np.unique(fit_v[2])) < 2:
                print(f"K={K} {tag} {name}: one class, skipped")
                continue
            val_bce = {"none": wbce(val_rows[0], val_rows[2], val_rows[3])}
            for c in args.calibrators:
                f = fit_calibrator(c, *fit_v, len(grid))
                val_bce[c] = wbce(f(val_rows[0], val_rows[1]), val_rows[2], val_rows[3]) if f is not None else np.inf
            selected = min(val_bce, key=val_bce.get)
            append_csv(sel_path, {"dataset": name, "K": K, "model": tag, "fit_on": args.fit_on, **{f"val_bce_{k}": r4(v) for k, v in val_bce.items()},
                                  "selected": selected, "n_fit_rows": len(fit_v[0]), "n_val_rows": len(val_rows[0])})
            # ---- (d) test
            St = rec["test_S_raw"].astype(float)
            nt, mt = St.shape
            y_tv = Surv.from_arrays(Dtv, Utv)
            y_te = Surv.from_arrays(rec["y_test_event"].astype(bool), rec["y_test_time"].astype(float))
            if args.fit_on == "oof":
                fit_t = rows(rec["tv_oof_S_raw"].astype(float), tt, Utv, Dtv, X_of(name, "trainval", len(Utv)), cm_tv, floor, lev_t)
            else:
                fit_t = fit_v
            k_test = np.tile(lev_t, nt)
            results = {}
            for c in args.calibrators + ["none"]:
                if c == "none":
                    pt = 1.0 - St
                else:
                    f = fit_calibrator(c, *fit_t, len(grid))
                    if f is None:
                        continue
                    pt = np.clip(f(1.0 - St.ravel(), k_test), 0.0, 1.0).reshape(nt, mt)
                S = np.minimum.accumulate(1.0 - pt, axis=1)
                met = survival_metrics(y_tv, y_te, S, tt)
                T_r, D_r = y_te["time"][np.repeat(np.arange(nt), mt)], y_te["event"][np.repeat(np.arange(nt), mt)]
                row_t = np.tile(tt, nt)
                valid = D_r | (row_t <= T_r)
                y_true = (D_r & (T_r <= row_t)).astype(int)[valid]
                p_ev = np.clip(pt.ravel()[valid], 1e-8, 1 - 1e-8)
                results[c] = (S, met, roc_auc_score(y_true, p_ev) if len(np.unique(y_true)) > 1 else np.nan, log_loss(y_true, p_ev, labels=[0, 1]))
            for c in options:
                src = selected if c in ("cal", "calibs") else c
                if src not in results:
                    continue
                S, met, auc_roc, bce = results[src]
                risk = np.mean(1 - S, axis=1)
                if c == "calibs":  # ranking metrics and risk score of the uncalibrated model, IBS of the selected calibration
                    S0, met0, auc_roc, _ = results["none"]
                    risk = np.mean(1 - S0, axis=1)
                    met = {**met0, "ibs": met["ibs"]}
                np.savez_compressed(os.path.join(outs[c], fn), y_train_time=rec["y_train_time"], y_train_event=rec["y_train_event"],
                                    y_test_time=rec["y_test_time"], y_test_event=rec["y_test_event"], cidx_risk=risk,
                                    cidx_tau=rec["cidx_tau"], ibs_S=S, ibs_times=tt, auc_risk=risk, auc_times=tt)
                append_csv(csvs[c], {"dataset": name, "K": K, "val_c_index": np.nan, "val_ipcw_c_index": np.nan, "val_ibs": np.nan,
                                     "val_mean_auc": np.nan, "c_index": r4(met["c_index"]), "ipcw_c_index": r4(met["ipcw_c_index"]),
                                     "ibs": r4(met["ibs"]), "mean_auc": r4(met["mean_auc"]), "auc_roc": r4(auc_roc), "bce": r4(bce),
                                     "calibrator": src, "fit_on": args.fit_on, "censoring_model": args.censoring})
            print(f"K={K} {tag} {name}: fit rows {len(fit_v[0])}, val rows {len(val_rows[0])}, val BCE "
                  + ", ".join(f"{k} {v:.4f}" for k, v in val_bce.items()) + f" -> {selected}")
# summary
summ = []
for K in args.K:
    for tag in args.models:
        for c in [None, "cal", "calibs"] + args.calibrators:
            f = os.path.join("results", f"{tag}{'_' + c if c else ''}_evaluation_final_{K}.csv")
            if os.path.isfile(f):
                d = pd.read_csv(f)
                summ.append({"K": K, "model": tag, "calibrator": c or "none", **d[["ipcw_c_index", "mean_auc", "ibs"]].mean().to_dict()})
if summ:
    s = pd.DataFrame(summ).groupby(["model", "calibrator"])[["ipcw_c_index", "mean_auc", "ibs"]].mean().round(4)
    print(s.to_string())
    s.to_csv(os.path.join("results", f"calibration_summary_{'+'.join(args.models)}.csv"))
    if os.path.isfile(sel_path):
        sel = pd.read_csv(sel_path)
        print("\nselected per dataset x K:")
        print(sel.groupby("model").selected.value_counts().to_string())
