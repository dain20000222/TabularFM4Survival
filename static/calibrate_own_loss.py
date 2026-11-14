# Post-hoc calibration of the DH and MC models with their OWN loss: the loss that the discrete-hazard formulation defines
# and the loss that SurvivalPFN defines for the multi-class formulation, not the weighted loss of the CF formulation.
# Same procedure as ../tmlr_ipcw_loss_10_25/calibrate_static.py (calibrators fitted on the 5-fold out-of-fold predictions,
# option chosen per dataset and K on the validation loss with "none" as a candidate, refit on train+val for the test
# predictions), same input files (preds/K{K}/{tag}/{dataset}.npz with tr_oof_S_raw, tv_oof_S_raw, val_S_raw, test_S_raw).
# Only the calibrated quantity and the loss differ:
#   --loss dh   quantity: the hazard h_k = 1 - S(t_k)/S(t_{k-1}); rows (i, k) with U_i > t_{k-1}, label 1(event in (t_{k-1}, t_k]),
#               no row if censored inside the interval; unweighted BCE (the discrete-hazard loss). Calibrators: platt
#               (sigma(a logit h + b)), platt_t (one intercept per interval), isotonic. S_cal(t_k) = prod_{j<=k} (1 - h_j_cal).
#   --loss mc   quantity: the class probabilities p_j = S(t_{j-1}) - S(t_j) (j = 1..m, and p_{m+1} = S(t_m)) of the delta = 1
#               query; rows: the observations with an observed event (for them the stored prediction is the model's prediction
#               at their own delta), class = interval of U_i; multi-class cross-entropy (the SurvivalPFN loss on these rows).
#               Calibrators: temp (p_j^a, renormalised), vector (a log p_j + b_j, softmax). S_cal(t_k) = 1 - sum_{j<=k} p_j_cal.
# The C-index and integrated AUC stay those of the uncalibrated model; only the IBS uses the calibrated curve.
#   python calibrate_own_loss.py --loss dh --preds <dir with K5, K10, ...> --models static_mitra_dh_kmq static_tabpfn_dh_kmq --out results_dh
import argparse, os, warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv

warnings.filterwarnings("ignore")
p = argparse.ArgumentParser()
p.add_argument("--loss", required=True, choices=["dh", "mc"])
p.add_argument("--preds", required=True)
p.add_argument("--models", nargs="+", required=True)
p.add_argument("--K", type=int, nargs="*", default=[5, 10, 15, 20])
p.add_argument("--out", default="results")
args = p.parse_args()
EPS = 1e-6
logit = lambda q: np.log(np.clip(q, EPS, 1 - EPS) / (1 - np.clip(q, EPS, 1 - EPS)))


# ------------------------------------------------------------------ DH
def hazards(S):
    S = np.clip(np.minimum.accumulate(S, axis=1), EPS, 1.0)
    prev = np.concatenate([np.ones((len(S), 1)), S[:, :-1]], axis=1)
    return np.clip(1.0 - S / prev, 0.0, 1.0)

def dh_rows(S, times, U, D):
    h = hazards(S); n, m = h.shape
    start = np.concatenate([[-np.inf], times[:-1]])
    at_risk = U[:, None] > start[None, :]
    inside = U[:, None] <= times[None, :]
    keep = at_risk & ~(inside & ~D[:, None])           # censored inside the interval: no label
    y = (inside & D[:, None]).astype(int)
    k = np.tile(np.arange(m), (n, 1))
    return h[keep], k[keep], y[keep]

def dh_fit(kind, hv, kv, yv, m):
    if len(np.unique(yv)) < 2: return None
    if kind == "platt":
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(hv)[:, None], yv)
        return lambda h: lr.predict_proba(logit(h).reshape(-1, 1))[:, 1].reshape(h.shape)
    if kind == "platt_t":
        def feats(h, k):
            Z = np.zeros((len(h), 1 + m)); Z[:, 0] = logit(h); Z[np.arange(len(h)), 1 + k] = 1.0; return Z
        lr = LogisticRegression(C=1e6, max_iter=2000, fit_intercept=False).fit(feats(hv, kv), yv)
        return lambda h: lr.predict_proba(feats(h.ravel(), np.tile(np.arange(h.shape[1]), h.shape[0])))[:, 1].reshape(h.shape)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(hv, yv)
    return lambda h: iso.predict(h.ravel()).reshape(h.shape)

def bce(pr, y):
    pr = np.clip(pr, EPS, 1 - EPS); return float(np.mean(-(y * np.log(pr) + (1 - y) * np.log(1 - pr))))

def dh_apply(f, S):
    h = hazards(S); return np.cumprod(1.0 - (h if f is None else np.clip(f(h), 0.0, 1.0)), axis=1)

def dh_loss(f, S, times, U, D):
    h = hazards(S); n, m = h.shape
    hc = h if f is None else np.clip(f(h), 0.0, 1.0)
    start = np.concatenate([[-np.inf], times[:-1]])
    at_risk = U[:, None] > start[None, :]; inside = U[:, None] <= times[None, :]
    keep = at_risk & ~(inside & ~D[:, None])
    return bce(hc[keep], (inside & D[:, None]).astype(int)[keep])


# ------------------------------------------------------------------ MC
def classes(S):
    S = np.clip(np.minimum.accumulate(S, axis=1), 0.0, 1.0)
    P = np.concatenate([1.0 - S[:, :1], S[:, :-1] - S[:, 1:], S[:, -1:]], axis=1)
    P = np.clip(P, EPS, None); return P / P.sum(1, keepdims=True)

def mc_rows(S, times, U, D):
    return classes(S)[D], np.searchsorted(times, U[D], side="left")      # class j: t_{j-1} < U <= t_j ; the last class: U > t_m

def mc_fit(kind, P, c):
    if len(P) < 5: return None
    lp = np.log(P); m1 = P.shape[1]
    def nll(theta):
        a = np.exp(theta[0]); b = theta[1:] if kind == "vector" else 0.0
        return -np.mean(log_softmax(a * lp + b, axis=1)[np.arange(len(c)), c]) + (1e-3 * np.sum(np.square(theta[1:])) if kind == "vector" else 0.0)
    th = minimize(nll, np.zeros(1 + (m1 if kind == "vector" else 0)), method="L-BFGS-B").x
    a = np.exp(th[0]); b = th[1:] if kind == "vector" else 0.0
    return lambda Q: np.exp(log_softmax(a * np.log(Q) + b, axis=1))

def mc_apply(f, S):
    P = classes(S); P = P if f is None else f(P); return 1.0 - np.cumsum(P[:, :-1], axis=1)

def mc_loss(f, S, times, U, D):
    P, c = mc_rows(S, times, U, D); P = P if f is None else f(P)
    return float(-np.mean(np.log(np.clip(P[np.arange(len(c)), c], EPS, None)))) if len(c) else np.inf


CAL = {"dh": ["platt", "platt_t", "isotonic"], "mc": ["temp", "vector"]}[args.loss]
os.makedirs(args.out, exist_ok=True)
out = []
for K in args.K:
    for tag in args.models:
        pdir = os.path.join(args.preds, f"K{K}", tag)
        if not os.path.isdir(pdir):
            print("no", pdir); continue
        for fn in sorted(os.listdir(pdir)):
            if not fn.endswith(".npz"): continue
            rec = dict(np.load(os.path.join(pdir, fn))); name = fn[:-4]
            if any(k not in rec or rec[k].size == 0 for k in ["val_S_raw", "test_S_raw", "tr_oof_S_raw", "tv_oof_S_raw"]):
                print(f"K={K} {tag} {name}: no out-of-fold predictions"); continue
            tv, tt = rec["val_times"].astype(float), rec["ibs_times"].astype(float)
            Utr, Dtr = rec["y_tr_time"].astype(float), rec["y_tr_event"].astype(bool)
            Uv, Dv = rec["y_val_time"].astype(float), rec["y_val_event"].astype(bool)
            Utv, Dtv = rec["y_train_time"].astype(float), rec["y_train_event"].astype(bool)
            Ute, Dte = rec["y_test_time"].astype(float), rec["y_test_event"].astype(bool)
            oof_v, oof_t, Sv, St = (rec[k].astype(float) for k in ("tr_oof_S_raw", "tv_oof_S_raw", "val_S_raw", "test_S_raw"))
            if args.loss == "dh":
                fit = lambda kind, S, t, U, D: dh_fit(kind, *dh_rows(S, t, U, D), S.shape[1]); loss, apply = dh_loss, dh_apply
            else:
                fit = lambda kind, S, t, U, D: mc_fit(kind, *mc_rows(S, t, U, D)); loss, apply = mc_loss, mc_apply
            val = {"none": loss(None, Sv, tv, Uv, Dv)}
            for c in CAL:
                try:
                    f = fit(c, oof_v, tv, Utr, Dtr); val[c] = loss(f, Sv, tv, Uv, Dv) if f is not None else np.inf
                except Exception as e:
                    val[c] = np.inf
            sel = min(val, key=val.get)
            y_tv, y_te = Surv.from_arrays(Dtv, Utv), Surv.from_arrays(Dte, Ute)
            f = None if sel == "none" else fit(sel, oof_t, tt, Utv, Dtv)
            row = {"dataset": name, "K": K, "model": tag, "selected": sel, **{f"val_loss_{k}": round(v, 5) for k, v in val.items()}}
            for lab, g in (("ibs_none", None), ("ibs_own_loss", f)):
                try:
                    row[lab] = integrated_brier_score(y_tv, y_te, np.clip(apply(g, St), 0.0, 1.0), tt)
                except Exception as e:
                    row[lab] = np.nan
            out.append(row)
    pd.DataFrame(out).to_csv(os.path.join(args.out, f"own_loss_{args.loss}.csv"), index=False)
R = pd.DataFrame(out)
print(R.groupby("model")[["ibs_none", "ibs_own_loss"]].mean().round(4)); print(R.groupby("model").selected.value_counts())
