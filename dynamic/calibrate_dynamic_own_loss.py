# Dynamic setting: post-hoc calibration of the DH and MC models with their own loss (not with the weights of the CF loss).
# Same procedure and files as ../tmlr_ipcw_loss_10_25/calibrate_dynamic.py
# (calibrators fitted on the out-of-fold predictions of the training patients, option chosen per dataset and feature setting
# on the validation loss with "none" as a candidate, applied to the validation and test survival probabilities); only the
# calibrated quantity and the loss differ. For a patient at risk at the prediction time tau with horizons t_1 < ... < t_m:
#   --loss dh  hazards h_j = 1 - S(t_j | tau) / S(t_{j-1} | tau), t_0 = tau; rows with U > t_{j-1}, label 1(event in (t_{j-1}, t_j]),
#              no row if censored inside the interval; unweighted BCE. platt / platt_t (one intercept per horizon t) / isotonic.
#   --loss mc  class probabilities p_j = S(t_{j-1} | tau) - S(t_j | tau), p_{m+1} = S(t_m | tau) of the delta = 1 query; rows of the
#              patients with an observed event, class = interval of U; multi-class cross-entropy. temp / vector (one bias per horizon).
# Output: {tag}_cal_risk_{k}.csv and {tag}_cal_selection_{k}.csv in --out_dir, in the format of calibrate_dynamic.py, so that
# eval_dynamic_w.py and build_dynamic_long.py read them as the calibrated model. Risk-score rows are unchanged.
#   python calibrate_dynamic_own_loss.py --loss dh --risk preds/dynamic_mitra_discrete_risk_20_km_quantile.csv --oof preds/..._oof_... --data_mean data_mean --out_dir own/preds
import argparse, os, warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
p = argparse.ArgumentParser()
p.add_argument("--loss", required=True, choices=["dh", "mc"])
p.add_argument("--risk", required=True)
p.add_argument("--oof", required=True)
p.add_argument("--data_mean", required=True)
p.add_argument("--out_dir", required=True)
args = p.parse_args()
EPS = 1e-6
logit = lambda q: np.log(np.clip(q, EPS, 1 - EPS) / (1 - np.clip(q, EPS, 1 - EPS)))


def outcomes(dataset):
    df = pd.read_csv(os.path.join(args.data_mean, f"{dataset}.csv"))
    tcol = "time2" if "time2" in df.columns else "time"
    return df.groupby("pid").agg(time=(tcol, "max"), event=(("event", "max")))


def curves(frame, po, grid):
    """One record per (eval_set, pid, tau): index of its rows (sorted by t), horizons, S, grid levels, U, D."""
    f = frame[frame.t.notna() & frame.surv_prob.notna()]
    out = []
    for (es, pid, tau), g in f.groupby(["eval_set", "pid", "tau"], sort=False):
        g = g.sort_values("t")
        S = np.clip(np.minimum.accumulate(g.surv_prob.to_numpy(float)), EPS, 1.0)
        out.append(dict(idx=g.index.to_numpy(), tau=float(tau), t=g.t.to_numpy(float), S=S, lev=np.searchsorted(grid, g.t.to_numpy(float)),
                        U=float(po.loc[pid, "time"]), D=bool(po.loc[pid, "event"] == 1)))
    return out


# ---- DH
def dh_parts(c):
    S = c["S"]; prev = np.concatenate([[1.0], S[:-1]]); h = np.clip(1.0 - S / prev, 0.0, 1.0)
    start = np.concatenate([[c["tau"]], c["t"][:-1]])
    inside = c["U"] <= c["t"]
    keep = (c["U"] > start) & ~(inside & (not c["D"]))
    return h, keep, (inside & c["D"]).astype(int)

def dh_rows(cs):
    H, L, Y = [], [], []
    for c in cs:
        h, keep, y = dh_parts(c); H.append(h[keep]); L.append(c["lev"][keep]); Y.append(y[keep])
    return (np.concatenate(H), np.concatenate(L), np.concatenate(Y)) if H else (np.array([]), np.array([], int), np.array([], int))

def dh_fit(kind, hv, lv, yv, n_lev):
    if len(yv) == 0 or len(np.unique(yv)) < 2: return None
    if kind == "platt":
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(hv)[:, None], yv)
        return lambda h, l: lr.predict_proba(logit(h)[:, None])[:, 1]
    if kind == "platt_t":
        def feats(h, l):
            Z = np.zeros((len(h), 1 + n_lev)); Z[:, 0] = logit(h); Z[np.arange(len(h)), 1 + np.asarray(l)] = 1.0; return Z
        lr = LogisticRegression(C=1e6, max_iter=2000, fit_intercept=False).fit(feats(hv, lv), yv)
        return lambda h, l: lr.predict_proba(feats(h, l))[:, 1]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(hv, yv)
    return lambda h, l: iso.predict(h)

def dh_loss(f, cs):
    h, l, y = dh_rows(cs)
    if len(y) == 0: return np.inf
    q = np.clip(h if f is None else f(h, l), EPS, 1 - EPS)
    return float(np.mean(-(y * np.log(q) + (1 - y) * np.log(1 - q))))

def dh_apply(f, c):
    h, _, _ = dh_parts(c); return np.cumprod(1.0 - np.clip(f(h, c["lev"]), 0.0, 1.0))


# ---- MC
def mc_probs(c):
    S = c["S"]; P = np.clip(np.concatenate([[1.0 - S[0]], S[:-1] - S[1:], [S[-1]]]), EPS, None); return P / P.sum()

def mc_rows(cs):
    return [(np.log(mc_probs(c)), np.concatenate([c["lev"], [-1]]), int(np.searchsorted(c["t"], c["U"], side="left"))) for c in cs if c["D"]]

def mc_fit(kind, rows, n_lev):
    if len(rows) < 5: return None
    def z(theta, lp, lev):
        a = np.exp(theta[0]); return a * lp + (theta[1:][lev] if kind == "vector" else 0.0)    # lev = -1: the bias of "after the last horizon"
    def nll(theta):
        return -np.mean([log_softmax(z(theta, lp, lev))[c] for lp, lev, c in rows]) + (1e-3 * np.sum(np.square(theta[1:])) if kind == "vector" else 0.0)
    th = minimize(nll, np.zeros(1 + (n_lev + 1 if kind == "vector" else 0)), method="L-BFGS-B").x
    return lambda lp, lev: np.exp(log_softmax(z(th, lp, lev)))

def mc_loss(f, cs):
    rows = mc_rows(cs)
    if not rows: return np.inf
    return float(-np.mean([np.log(max((np.exp(lp) if f is None else f(lp, lev))[c], EPS)) for lp, lev, c in rows]))

def mc_apply(f, c):
    P = f(np.log(mc_probs(c)), np.concatenate([c["lev"], [-1]])); return 1.0 - np.cumsum(P[:-1])


CAL = {"dh": ["platt", "platt_t", "isotonic"], "mc": ["temp", "vector"]}[args.loss]
risk, oof = pd.read_csv(args.risk), pd.read_csv(args.oof)
out_rows, sel_rows = [], []
for (dataset, hp), g in risk.groupby(["dataset", "hyperparameter"], sort=False):
    o = oof[(oof.dataset == dataset) & (oof.hyperparameter == hp)]
    po = outcomes(dataset)
    grid = np.unique(np.concatenate([g.t.dropna().to_numpy(float), o.t.dropna().to_numpy(float)]))
    c_fit, c_val = curves(o, po, grid), curves(g[g.eval_set == "val"], po, grid)
    if args.loss == "dh":
        fit = lambda kind: dh_fit(kind, *dh_rows(c_fit), len(grid)); loss, apply = dh_loss, dh_apply
    else:
        fit = lambda kind: mc_fit(kind, mc_rows(c_fit), len(grid)); loss, apply = mc_loss, mc_apply
    scores, cal = {"none": loss(None, c_val)}, {}
    for kind in CAL:
        try:
            cal[kind] = fit(kind); scores[kind] = loss(cal[kind], c_val) if cal[kind] is not None else np.inf
        except Exception as e:
            scores[kind] = np.inf
    chosen = min(scores, key=scores.get) if np.isfinite(scores["none"]) else "none"
    sel_rows.append({"dataset": dataset, "hyperparameter": hp, "selected": chosen, "n_fit": len(c_fit), "n_val": len(c_val), **{f"val_loss_{k}": round(v, 5) for k, v in scores.items()}})
    g = g.copy()
    if chosen != "none":
        for c in curves(g, po, grid):
            g.loc[c["idx"], "surv_prob"] = np.minimum.accumulate(np.clip(apply(cal[chosen], c), 0.0, 1.0))
    out_rows.append(g)
    print(f"{dataset} {hp}: val loss " + ", ".join(f"{k} {v:.4f}" for k, v in scores.items()) + f" -> {chosen}")
os.makedirs(args.out_dir, exist_ok=True)
tag, k = os.path.basename(args.risk).split("_risk_")
pd.concat(out_rows, ignore_index=True).to_csv(os.path.join(args.out_dir, f"{tag}_cal_risk_{k}"), index=False)
pd.DataFrame(sel_rows).to_csv(os.path.join(args.out_dir, f"{tag}_cal_selection_{k}"), index=False)
print(pd.DataFrame(sel_rows).selected.value_counts().to_string())
