# Monotonicity of the CF predictions, and clipping vs. isotonic regression over time (Appendix D.3, Table 7), for the CF
# models as used in the paper: MITRA-CF and TabPFN-CF with post-hoc calibration for the IBS (C-index / integrated AUC from
# the risk score of the uncalibrated model, as in Table 1), XGBoost-CF without calibration.
# The calibration is that of calibrate_static.py (calibrators fitted on the out-of-fold rows with the weights W, option
# chosen on the weighted validation BCE, refit on train+val); here the calibrated probabilities are kept BEFORE the
# monotone transformation, which is done by clipping (paper) or by isotonic regression over time.
#   python monotonicity.py --preds <dir with K5, K10, ...> --models static_mitra_cf_full_kmq ... --out out.csv
import argparse, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cf_weights import cf_weights, fit_censoring  # noqa: E402
warnings.filterwarnings("ignore")
p = argparse.ArgumentParser()
p.add_argument("--preds", required=True); p.add_argument("--models", nargs="+", required=True)
p.add_argument("--K", type=int, nargs="*", default=[5, 10, 15, 20]); p.add_argument("--out", required=True)
p.add_argument("--no_cal", nargs="*", default=["static_xgboost_cf_ipcw_km_kmq"], help="models used without calibration in the paper")
args = p.parse_args()
EPS = 1e-6
def logit(q):
    q = np.clip(q, EPS, 1 - EPS); return np.log(q / (1 - q))
def fit_calibrator(kind, pv, kv, yv, wv, n_k):
    if len(np.unique(yv)) < 2: return None
    if kind == "platt":
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pv)[:, None], yv, sample_weight=wv)
        return lambda q, k: lr.predict_proba(logit(q)[:, None])[:, 1]
    if kind == "platt_t":
        def feats(q, k):
            Z = np.zeros((len(q), 1 + n_k)); Z[:, 0] = logit(q); Z[np.arange(len(q)), 1 + np.asarray(k)] = 1.0; return Z
        lr = LogisticRegression(C=1e6, max_iter=2000, fit_intercept=False).fit(feats(pv, kv), yv, sample_weight=wv)
        return lambda q, k: lr.predict_proba(feats(q, k))[:, 1]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(pv, yv, sample_weight=wv)
    return lambda q, k: iso.predict(q)
def rows(S, times, U, D, cm, floor, levels):
    n, m = S.shape; subj, j = np.repeat(np.arange(n), m), np.tile(np.arange(m), n)
    w = cf_weights(cm, floor, np.zeros((n * m, 1)), U[subj], D[subj], times[j], "ipcw"); keep = w > 0
    return (1.0 - S.ravel())[keep], np.tile(levels, n)[keep], (U[subj] <= times[j]).astype(int)[keep], w[keep]
def wbce(pr, y, w):
    pr = np.clip(pr, EPS, 1 - EPS); return float(np.sum(w * -(y * np.log(pr) + (1 - y) * np.log(1 - pr))) / np.sum(w))
def iso_time(S):
    ir = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0); x = np.arange(S.shape[1])
    return np.vstack([ir.fit_transform(x, s) for s in S])
clip = lambda S: np.minimum.accumulate(S, axis=1)

out = []
for K in args.K:
    for tag in args.models:
        pdir = os.path.join(args.preds, f"K{K}", tag)
        if not os.path.isdir(pdir): print("no", pdir); continue
        for fn in sorted(os.listdir(pdir)):
            if not fn.endswith(".npz"): continue
            rec = dict(np.load(os.path.join(pdir, fn))); name = fn[:-4]
            St = rec["test_S_raw"].astype(float) if "test_S_raw" in rec else None
            if St is None or St.size == 0: print(f"K={K} {tag} {name}: no raw test predictions"); continue
            tt = rec["ibs_times"].astype(float); nt, mt = St.shape
            Utv, Dtv = rec["y_train_time"].astype(float), rec["y_train_event"].astype(bool)
            y_tv = Surv.from_arrays(Dtv, Utv); y_te = Surv.from_arrays(rec["y_test_event"].astype(bool), rec["y_test_time"].astype(float))
            sel = "none"
            Scal = St
            if tag not in args.no_cal:
                need = ["val_S_raw", "tr_oof_S_raw", "tv_oof_S_raw"]
                if any(k not in rec or rec[k].size == 0 for k in need): print(f"K={K} {tag} {name}: no out-of-fold predictions"); continue
                tv = rec["val_times"].astype(float); grid = np.unique(np.concatenate([tv, tt])); lev_v, lev_t = np.searchsorted(grid, tv), np.searchsorted(grid, tt)
                Utr, Dtr = rec["y_tr_time"].astype(float), rec["y_tr_event"].astype(bool); Uv, Dv = rec["y_val_time"].astype(float), rec["y_val_event"].astype(bool)
                cm_tr, floor = fit_censoring("km", np.zeros((len(Utr), 1)), Utr, Dtr, seed=42, floor=None)
                cm_tv, _ = fit_censoring("km", np.zeros((len(Utv), 1)), Utv, Dtv, seed=42, floor=None)
                fit_v = rows(rec["tr_oof_S_raw"].astype(float), tv, Utr, Dtr, cm_tr, floor, lev_v)
                val_rows = rows(rec["val_S_raw"].astype(float), tv, Uv, Dv, cm_tr, floor, lev_v)
                if len(np.unique(val_rows[2])) >= 2 and len(np.unique(fit_v[2])) >= 2:
                    vb = {"none": wbce(val_rows[0], val_rows[2], val_rows[3])}
                    for c in ("platt", "platt_t", "isotonic"):
                        f = fit_calibrator(c, *fit_v, len(grid)); vb[c] = wbce(f(val_rows[0], val_rows[1]), val_rows[2], val_rows[3]) if f is not None else np.inf
                    sel = min(vb, key=vb.get)
                    if sel != "none":
                        fit_t = rows(rec["tv_oof_S_raw"].astype(float), tt, Utv, Dtv, cm_tv, floor, lev_t)
                        f = fit_calibrator(sel, *fit_t, len(grid))
                        if f is not None: Scal = 1.0 - np.clip(f(1.0 - St.ravel(), np.tile(lev_t, nt)), 0.0, 1.0).reshape(nt, mt)
            row = {"dataset": name, "K": K, "model": tag, "calibrator": sel}
            for lab, rep in (("clip", clip), ("iso", iso_time)):
                Sr, Sc = rep(St), rep(Scal)                      # ranking metrics: raw model; IBS: the curve used in the paper
                risk = np.mean(1 - Sr, axis=1)
                try: row[f"cindex_{lab}"] = concordance_index_ipcw(y_tv, y_te, risk, tau=float(rec["cidx_tau"]))[0]
                except Exception: row[f"cindex_{lab}"] = np.nan
                try: row[f"auc_{lab}"] = cumulative_dynamic_auc(y_tv, y_te, risk, tt)[1]
                except Exception: row[f"auc_{lab}"] = np.nan
                try: row[f"ibs_{lab}"] = integrated_brier_score(y_tv, y_te, np.clip(Sc, 0.0, 1.0), tt) if mt > 1 else np.nan
                except Exception: row[f"ibs_{lab}"] = np.nan
            d = np.diff(Scal, axis=1)
            row["obs_with_violation"] = float((d > 1e-12).any(1).mean()); row["largest_increase"] = float(np.clip(d, 0, None).max(1).mean())
            out.append(row)
    pd.DataFrame(out).to_csv(args.out, index=False)
R = pd.DataFrame(out)
print(R.groupby("model")[[c for c in R.columns if c.startswith(("cindex", "auc", "ibs", "obs", "largest"))]].mean().round(4).T)
