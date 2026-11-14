# =============================================================================
# Static CF models with the IPCW-weighted loss (2026-10-25), and its unweighted ablation
# =============================================================================
# Copy of ../tmlr_ipcw_10_15/static/static_cf.py (the paper's pipeline: split, one-hot, imputation,
# grid, refit on train+val, test truncation, cummin, metrics, stored predictions). Changes (IPCWLOSS):
#   * training rows: grid times t_k ONLY (no row at the subject's own observed time);
#   * --weighting ipcw : W_ik = G(t_k|x)/G(U_i|x) (event before t_k), 1 (observed at t_k), 0 (censored
#                        at or before t_k); --weighting abl : W^abl = 1, 1, 0  (cf_weights.py);
#   * --censoring km | kmf | cox | rsf : the G model of the loss (fitted on the training subjects of
#                        the phase: train for validation, train+val for test);
#   * XGBoost: W as sample weights, no rescaling.
#   * MITRA / TabPFN (in context, no training loss; they are not modified unless fine-tuned):
#     default = every row with W > 0 in the context (tag ..._full). These rows are the same for W and
#     W^abl, so one run serves loss (B) and loss (A) without the extra term. With --thin (comparison
#     only), each row is kept with probability W instead (Bernoulli, seed 42; tag ..._thin_{G}).
# Output: results/{tag}_evaluation_final_{K}.csv, preds/K{K}/{tag}/{dataset}.npz with
#   tag = static_xgboost_cf_{ipcw_{G}|abl} | static_{mitra,tabpfn}_cf_{full|thin_{G}};
#   evaluate with evaluate_static_w.py.
#   python static_cf_w.py --model xgboost --k 10 --weighting ipcw --censoring km
#   python static_cf_w.py --model mitra --k 10                      # full context
#   python static_cf_w.py --model mitra --k 10 --thin --censoring km
# =============================================================================
import argparse
import csv
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch

if not hasattr(np, "infty"):  # older scikit-survival uses np.infty, removed in NumPy 2
    np.infty = np.inf
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc,
                            integrated_brier_score)
from sksurv.util import Surv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tmlr_ipcw_10_15", "common"))
sys.path.insert(0, HERE)
from classifiers import Classifier  # noqa: E402
from cf_weights import CENSORING, cf_weights, fit_censoring, thin  # noqa: E402  (IPCWLOSS)
from mc_classifier import MCClassifier, mc_labels, mc_rows  # noqa: E402  (MC, 11-07)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["mitra", "tabpfn", "xgboost", "mitra_lora"])
# LORA (10-30): MITRA fine-tuned with LoRA on the weighted loss (mitra_lora_w.py). Learning rate chosen on the
# weighted validation BCE (= the loss on the validation rows), early stopping on it, refit on train+val
# for the selected number of steps. --weighting abl fine-tunes on the unweighted rows (ablation).
parser.add_argument("--oof_folds", type=int, default=0,
                    help="OOF (10-31): also store out-of-fold predictions of the training subjects (this many folds by subject) "
                         "in both phases, for calibrators fitted on the training split (calibrate_static.py --fit_on oof)")
parser.add_argument("--lora_lrs", type=float, nargs="*", default=[1e-4, 3e-4, 1e-3])
parser.add_argument("--lora_steps", type=int, default=200)
parser.add_argument("--lora_rank", type=int, default=8)
# RANK / SELECT / POOLK (10-31)
parser.add_argument("--lora_rank_lambdas", type=float, nargs="*", default=[0.0],
                    help="weights lambda of the ranking term (notes/ranking_loss.png); a grid is searched like the learning rate")
parser.add_argument("--lora_select", default="bce", choices=["bce", "cindex"],
                    help="criterion for learning rate, lambda and stopping step: weighted validation BCE (loss B) or validation IPCW C-index")
parser.add_argument("--lora_crank", action="store_true",
                    help="CRANK: use the ranking loss (subject pairs as in the IPCW C-index, softplus with temperature kappa) "
                         "instead of the time-specific pairs; weight from --lora_rank_lambdas, kappa from --lora_kappas")
parser.add_argument("--lora_kappas", type=float, nargs="*", default=[1.0],
                    help="CRANK: kappa relative to the standard deviation of the risk score over the training subjects at step 0")
parser.add_argument("--binning", default="quantile", choices=["quantile", "km_quantile"],
                    help="KMBIN (11-01): grid t_k = percentiles of the train+val observed times (paper) or KM-quantile bins as "
                         "Appendix A5 (static_rebuttal_km_equal_3_27): K-1 equally spaced levels between the first and last value "
                         "of the Kaplan-Meier curve of train+val, t_k = first time the curve reaches the level")
parser.add_argument("--tag_suffix", default="", help="appended to the run tag (to keep variants of the same setting apart)")
parser.add_argument("--pool_k", action="store_true",
                    help="train on the union of the grids of K = 5, 10, 15, 20 (t_k enters as the time value), evaluate on the --k grid")
parser.add_argument("--k", type=float, default=10, help="percentile step of the evaluation times (K = 100 / k)")
parser.add_argument("--weighting", default="ipcw", choices=["ipcw", "abl"])  # IPCWLOSS
parser.add_argument("--censoring", default="km", choices=list(CENSORING))  # IPCWLOSS: G model of the loss
parser.add_argument("--floor", type=float, default=None, help="floor of G in the weights (default: 0 for km, 0.05 for kmf / cox / rsf); floor sensitivity")  # IPCWLOSS
parser.add_argument("--thin", action="store_true", help="in-context models: keep each row with probability W (comparison only)")  # IPCWLOSS
parser.add_argument("--datasets", nargs="*", default=None)
parser.add_argument("--data_dir", default="static_data")
parser.add_argument("--formulation", default="cf", choices=["cf", "mc"],  # MC (11-07)
                    help="cf: cumulative failure (the paper); mc: multi-class over the grid intervals of U given (X, delta), as SurvivalPFN")
args = parser.parse_args()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

K = int(np.ceil(100.0 / float(args.k)))
# IPCWLOSS: tag. In-context models without --thin use every row with W > 0 (identical for W and W^abl),
# so the weighting and G play no role and the run is tagged "full".
IN_CONTEXT = args.model in ("mitra", "tabpfn")
if IN_CONTEXT and not args.thin:
    args.weighting = "abl"  # the row set {W > 0}; no G needed
    tag = f"static_{args.model}_cf_full"
elif IN_CONTEXT:
    assert args.weighting == "ipcw", "--thin needs --weighting ipcw (thinning with W^abl keeps every row)"
    tag = f"static_{args.model}_cf_thin_{args.censoring}"
else:
    assert not args.thin, "--thin applies to the in-context models only"
    tag = f"static_{args.model}_cf_{args.weighting}" + (f"_{args.censoring}" if args.weighting == "ipcw" else "")
if args.floor is not None:
    assert args.weighting == "ipcw", "--floor applies to the weights of loss (B) only"
    tag += f"_f{args.floor:g}"
if args.model == "mitra_lora":  # RANK / CRANK / SELECT
    tag += (("_crank" if args.lora_crank else "_rank") if max(args.lora_rank_lambdas) > 0 else "") + ("_selc" if args.lora_select == "cindex" else "")
if args.pool_k:  # POOLK
    tag += "_poolk"
if args.model == "mitra_lora" and args.oof_folds:  # OOF: rerun with out-of-fold predictions, kept apart from the first run
    tag += "_oof"
if args.binning == "km_quantile":  # KMBIN
    assert not args.pool_k, "--pool_k is defined for the quantile grid only"
    tag += "_kmq"
if args.formulation == "mc":  # MC (11-07): no loss weights, so one tag per model (and grid)
    assert args.model in ("mitra", "tabpfn", "xgboost") and not args.thin and not args.pool_k and args.floor is None
    tag = f"static_{args.model}_mc" + ("_kmq" if args.binning == "km_quantile" else "")
tag += args.tag_suffix
os.makedirs("results", exist_ok=True)
pred_dir = os.path.join("preds", f"K{K}", tag)
os.makedirs(pred_dir, exist_ok=True)
csv_path = os.path.join("results", f"{tag}_evaluation_final_{K}.csv")
rows_path = os.path.join("results", f"{tag}_rows_{K}.csv")
print(f"run: {tag}, K={K}  (IPCWLOSS: weighting={args.weighting}, G={args.censoring}, thin={args.thin})")
for path in (csv_path, rows_path):  # a rerun replaces the earlier (possibly failed) run
    if os.path.isfile(path):
        os.remove(path)

dataset_files = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".csv"))
if args.datasets:
    dataset_files = [f for f in dataset_files if f[:-4] in set(args.datasets)]


# ----------------------------------------------------------------- helpers
def get_valid_eval_times(base_eval_times, test_times, buffer=0.01):
    return base_eval_times[(base_eval_times > test_times.min() + buffer) & (base_eval_times < test_times.max() - buffer)]


def construct_rows(x, y, times):
    """IPCWLOSS: one CF row per subject and GRID time t_k (no row at the subject's own time);
    label Y~_ik = 1(U_i <= t_k) with U_i = min(T_i, C_i). Rows with weight 0 are dropped later.
    Returns the row frame, labels, the subject index and horizon of every row."""
    T, D = y["time"], y["event"]
    n, m = len(T), len(times)
    subj, hor = np.repeat(np.arange(n), m), np.tile(np.asarray(times, dtype=float), n)
    lab = (T[subj] <= hor).astype(int)
    X = pd.DataFrame(np.column_stack([x.to_numpy(dtype=float)[subj], hor]), columns=list(x.columns) + ["eval_time"])
    return X, lab, subj, hor


def build_training_set(x, y, times, kind, phase, dataset_name, g_fit=None):
    """g_fit: (x, y) on which G is fitted (default: the subjects of the rows; LORA: the training subjects
    when the rows are the validation rows used for early stopping)."""
    if args.formulation == "mc":  # MC (11-07): one row per subject, (X, delta) -> grid interval of U
        X, lab = mc_rows(x, y["time"], y["event"]), mc_labels(y["time"], times)
        build_training_set.last_subj = np.arange(len(y))
        return X, lab, None, {"dataset": dataset_name, "K": K, "phase": phase, "n_subjects": len(y), "rows_full": len(y),
                              "rows_used": len(y), "n_classes": int(len(np.unique(lab))), "censoring_model": "none"}
    X, lab, subj, hor = construct_rows(x, y, times)
    stats = {"dataset": dataset_name, "K": K, "phase": phase, "n_subjects": len(y),
             "rows_full": len(lab), "label1_full": lab.mean()}
    # IPCWLOSS: W (or W^abl) per row; zero-weight rows dropped; XGBoost gets W as sample weights;
    # in-context models use every kept row (default) or keep each row with probability W (--thin)
    xa = x.to_numpy(dtype=float)
    xg, yg = (xa, y) if g_fit is None else (g_fit[0].to_numpy(dtype=float), g_fit[1])
    cm, floor = fit_censoring(kind, xg, yg["time"], yg["event"], seed=SEED, floor=args.floor) if args.weighting == "ipcw" else (None, 0.0)
    w = cf_weights(cm, floor, xa[subj], y["time"][subj], y["event"][subj], hor, args.weighting)
    keep = w > 0
    X, lab, subj, hor, w = X[keep].reset_index(drop=True), lab[keep], subj[keep], hor[keep], w[keep]
    stats.update({"censoring_model": cm.kind_used if cm is not None else "none", "rows_zero": int((~keep).sum()),
                  "weight_mean": float(w.mean()), "weight_min": float(w.min()), "weight_max": float(w.max()),
                  "share_lt1": float(np.mean(w < 1 - 1e-12))})
    if args.model in ("xgboost", "mitra_lora"):
        weight = w
    elif not args.thin:
        weight = None
    else:
        k2 = thin(w, SEED)
        X, lab, subj, hor = X[k2].reset_index(drop=True), lab[k2], subj[k2], hor[k2]
        weight = None
        stats["rows_thinned_out"] = int((~k2).sum())
    stats.update({"rows_used": len(lab), "label1_used": lab.mean()})
    build_training_set.last_subj = subj  # LORA: patient groups of the kept rows
    return X, lab, weight, stats


class LoRAClassifier:
    """LORA: MITRA + LoRA adapters trained on the weighted loss (mitra_lora_w.py), with the Classifier interface."""

    def __init__(self, lr, lam=0.0):
        from mitra_lora_w import MitraLoRA  # needs autogluon (mitra env)
        self.lr, self.lam = lr, lam
        self.m = MitraLoRA(rank=args.lora_rank, alpha=2.0 * args.lora_rank, lr=lr, max_steps=args.lora_steps, seed=SEED, verbose=False)

    def fit(self, X, y, groups, w, X_val=None, y_val=None, w_val=None, n_steps=None, pair_groups=None, val_eval_X=None, val_eval_fn=None,
            crank=None):
        self.m.fit(X.to_numpy(dtype=np.float32), np.asarray(y), groups, X_val=None if X_val is None else X_val.to_numpy(dtype=np.float32),
                   y_val=None if y_val is None else np.asarray(y_val), n_steps=n_steps, sample_weight=w, val_weight=w_val,
                   rank_lambda=self.lam, pair_groups=pair_groups, val_eval_X=val_eval_X, val_eval_fn=val_eval_fn, crank=crank)
        self.best_step, self.best_val_loss, self.step0_val_loss = self.m.best_step, self.m.best_val_loss, self.m.step0_val_loss
        return self

    def predict_p0(self, X):
        return self.m.predict_proba(X.to_numpy(dtype=np.float32))[:, 0]

    def close(self):
        import torch
        self.m = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_clf():
    """MC (11-07): the classifier of the formulation."""
    return MCClassifier(args.model, SEED) if args.formulation == "mc" else Classifier(args.model, SEED)


def predict_S(clf, x_eval, times):
    n, m = len(x_eval), len(times)
    if args.formulation == "mc":  # MC: S at the training grid (= times; no --pool_k), monotone by construction
        S = clf.predict_S(x_eval, times)
        return S, np.tile(np.asarray(times, dtype=float), n), np.repeat(np.arange(n), m), S.ravel()
    Xe = pd.DataFrame(np.column_stack([np.repeat(x_eval.to_numpy(dtype=float), m, axis=0), np.tile(times, n)]),
                      columns=list(x_eval.columns) + ["eval_time"])
    p0 = clf.predict_p0(Xe)
    return p0.reshape(n, m), Xe["eval_time"].to_numpy(), np.repeat(np.arange(n), m), p0


def make_crank(x, y, times, kappa_rel):
    """CRANK: inputs of the ranking loss for the subjects of (x, y): their full-grid rows (covariates, eval_time),
    subject ids (= row positions, the id space of build_training_set.last_subj), T, delta and the IPCW weight G(T)^-2.
    G exactly as in the C-index metric: scikit-survival's CensoringDistributionEstimator (marginal Kaplan-Meier of
    the censoring times of these subjects), no floor. Subjects whose event time has G(T) = 0 (only possible at the
    end of follow-up) get weight 0, i.e. T_max = the last time with G > 0 (the metric would raise an error there)."""
    if not args.lora_crank:
        return None
    from sksurv.nonparametric import CensoringDistributionEstimator
    n, m = len(x), len(times)
    Xr = np.column_stack([np.repeat(x.to_numpy(dtype=float), m, axis=0), np.tile(np.asarray(times, dtype=float), n)])
    T, D = np.asarray(y["time"], dtype=float), np.asarray(y["event"], dtype=bool)
    G = CensoringDistributionEstimator().fit(Surv.from_arrays(D, T)).predict_proba(T)
    gw = np.where(G > 0, 1.0 / np.where(G > 0, G, 1.0) ** 2, 0.0)
    return {"X": Xr, "group": np.repeat(np.arange(n), m), "subject": np.arange(n), "T": T,
            "D": (D & (G > 0)).astype(float), "gw": gw, "kappa_rel": kappa_rel, "n_subj": 128}


def oof_predict(x, y, times, kind, phase, dataset_name, train_times=None, lora_info=None):
    """OOF: out-of-fold raw predictions S(t | x) of the subjects of (x, y) at `times`: subjects are split into
    --oof_folds folds (stratified on the event indicator, seed 42); each fold is predicted by the classifier
    fitted on the rows of the other folds (in-context models: the other folds as context; XGBoost: refit),
    with the same rows and weights as the real fit (training rows on train_times, default `times`).
    mitra_lora (10-31): each fold is fine-tuned with the learning rate, lambda and number of steps selected
    on validation for this dataset (lora_info), without a new search."""
    from sklearn.model_selection import StratifiedKFold
    out = np.full((len(y), len(times)), np.nan)
    for f, (i_fit, i_pred) in enumerate(StratifiedKFold(args.oof_folds, shuffle=True, random_state=SEED).split(x, y["event"])):
        Xf, yf, wf, _ = build_training_set(x.iloc[i_fit], y[i_fit], times if train_times is None else train_times, kind,
                                           f"{phase}_oof{f}", dataset_name)
        if args.model == "mitra_lora":
            subj_f = build_training_set.last_subj
            clf = LoRAClassifier(lora_info["lora_lr"], lora_info["lora_lambda"]).fit(
                Xf, yf, subj_f, wf, n_steps=lora_info["lora_best_step"], pair_groups=Xf["eval_time"].to_numpy(),
                crank=make_crank(x.iloc[i_fit], y[i_fit], times if train_times is None else train_times, lora_info.get("lora_kappa", 1.0)))
        else:
            clf = make_clf().fit(Xf, yf, wf)
        out[i_pred] = predict_S(clf, x.iloc[i_pred], times)[0]
        clf.close()
    return out


def survival_metrics(y_tr, y_ev, S, times):
    risk = np.mean(1 - S, axis=1)
    return {"c_index": concordance_index_censored(y_ev["event"], y_ev["time"], risk)[0],
            "ipcw_c_index": concordance_index_ipcw(y_tr, y_ev, risk, tau=y_tr["time"].max())[0],
            "ibs": integrated_brier_score(y_tr, y_ev, S, times),
            "mean_auc": cumulative_dynamic_auc(y_tr, y_ev, risk, times)[1]}


def preprocess(x_fit, x_other):
    a = pd.get_dummies(x_fit, drop_first=True)
    b = pd.get_dummies(x_other, drop_first=True)
    a, b = a.align(b, join="left", axis=1, fill_value=0)
    cols = a.columns
    imp = SimpleImputer().fit(a.loc[:, cols.tolist()])
    return (pd.DataFrame(imp.transform(a.loc[:, cols.tolist()]), columns=cols, index=x_fit.index),
            pd.DataFrame(imp.transform(b.loc[:, cols.tolist()]), columns=cols, index=x_other.index))


def append_csv(path, row):
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def r4(v):
    return round(float(v), 4) if v is not None and not np.isnan(v) else -1


# --------------------------------------------------------------------- main
for file_name in dataset_files:
    dataset_name = file_name[:-4]
    try:
        print("=" * 50 + f"\nProcessing dataset: {dataset_name}")
        df = pd.read_csv(os.path.join(args.data_dir, file_name)).drop(columns=["pid"])
        x = df[df.columns.difference(["time", "event"])].copy()
        y = Surv.from_arrays(event=df["event"].astype(bool), time=df["time"])

        x_trainval, x_test, y_trainval, y_test = train_test_split(x, y, test_size=0.15, stratify=y["event"], random_state=SEED)
        x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.1765,
                                                          stratify=y_trainval["event"], random_state=SEED)
        if args.binning == "km_quantile":  # KMBIN: as static_rebuttal_km_equal_3_27 (Appendix A5), K - 1 levels
            from sksurv.nonparametric import kaplan_meier_estimator
            km_time, km_surv = kaplan_meier_estimator(y_trainval["event"], y_trainval["time"])
            levels = np.linspace(km_surv[0], km_surv[-1], (K - 1) + 2)[1:-1]
            idx = np.searchsorted(-km_surv, -levels)
            base_eval_times = np.unique(km_time[idx[idx < len(km_time)]].astype(float))
        else:
            base_eval_times = np.unique(np.percentile(y_trainval["time"], np.arange(args.k, 100, args.k)))

        max_trainval_time = y_trainval["time"].max()
        y_test_f = y_test.copy()
        m = y_test_f["time"] >= max_trainval_time
        y_test_f["time"][m] = max_trainval_time - 0.01
        y_test_f["event"][m] = False

        val_times = get_valid_eval_times(base_eval_times, y_val["time"])
        test_times = get_valid_eval_times(base_eval_times, y_test_f["time"])
        if len(val_times) == 0 or len(test_times) == 0:
            print("No valid evaluation times - skipping dataset")
            continue
        # POOLK: training rows on the union of the percentile grids of all K (evaluation stays on the --k grid)
        if args.pool_k:
            union_base = np.unique(np.concatenate([np.percentile(y_trainval["time"], np.arange(kk, 100, kk)) for kk in (20, 10, 7, 5)]))
            train_times_val = get_valid_eval_times(union_base, y_val["time"])
            train_times_test = get_valid_eval_times(union_base, y_test_f["time"])
        else:
            train_times_val, train_times_test = val_times, test_times

        x_train_i, x_val_i = preprocess(x_train, x_val)
        x_trainval_i, x_test_i = preprocess(x_trainval, x_test)

        kind = args.censoring  # IPCWLOSS

        # ---------------- validation phase (train -> val)
        Xtr, ytr, wtr, st = build_training_set(x_train_i, y_train, train_times_val, kind, "val", dataset_name)
        lora_info = {}
        if args.model == "mitra_lora":  # LORA
            groups = build_training_set.last_subj
            Xv, yv, wv, _ = build_training_set(x_val_i, y_val, val_times, kind, "val_rows", dataset_name, g_fit=(x_train_i, y_train))
            veval_X, veval_fn = None, None
            if args.lora_select == "cindex":  # SELECT: minus the validation IPCW C-index of the full validation grid
                nv, mv = len(x_val_i), len(val_times)
                veval_X = pd.DataFrame(np.column_stack([np.repeat(x_val_i.to_numpy(dtype=float), mv, axis=0), np.tile(val_times, nv)]),
                                       columns=list(x_val_i.columns) + ["eval_time"]).to_numpy(dtype=np.float32)

                def veval_fn(p1, nv=nv, mv=mv):
                    S_ = np.minimum.accumulate(1.0 - p1.reshape(nv, mv), axis=1)
                    try:
                        return -concordance_index_ipcw(y_train, y_val, np.mean(1 - S_, axis=1), tau=y_train["time"].max())[0]
                    except Exception:
                        return 0.0
            best = None
            configs = [(lr, lam, kap) for lr in args.lora_lrs for lam in args.lora_rank_lambdas
                       for kap in (args.lora_kappas if (args.lora_crank and lam > 0) else [1.0])]
            for lr, lam, kap in configs:
                if True:
                    m = LoRAClassifier(lr, lam).fit(Xtr, ytr, groups, wtr, Xv, yv, wv, pair_groups=Xtr["eval_time"].to_numpy(),
                                                    val_eval_X=veval_X, val_eval_fn=veval_fn,
                                                    crank=make_crank(x_train_i, y_train, train_times_val, kap) if lam > 0 else None)
                    m.kap = kap
                    print(f"  lr={lr:g} lambda={lam:g} kappa={kap:g}: best step {m.best_step}, val {args.lora_select} {m.best_val_loss:.4f} (step 0: {m.step0_val_loss:.4f})")
                    if best is None or m.best_val_loss < best.best_val_loss - 1e-9:
                        if best is not None:
                            best.close()
                        best, clf = m, m
                    else:
                        m.close()
            lora_info = {"lora_lr": best.lr, "lora_lambda": best.lam, "lora_kappa": best.kap, "lora_select": args.lora_select, "lora_best_step": best.best_step,
                         "lora_step0_val_bce": best.step0_val_loss, "lora_best_val_bce": best.best_val_loss}
            st.update(lora_info)
        else:
            clf = make_clf().fit(Xtr, ytr, wtr)
        S_val_raw = predict_S(clf, x_val_i, val_times)[0]  # CALIB: raw validation predictions are stored
        S_val = np.minimum.accumulate(S_val_raw, axis=1)
        clf.close()
        val = survival_metrics(y_train, y_val, S_val, val_times)
        append_csv(rows_path, st)
        tr_oof = oof_predict(x_train_i, y_train, val_times, kind, "val", dataset_name, train_times_val, lora_info) if args.oof_folds else None  # OOF

        # ---------------- test phase (train+val -> test)
        Xtr, ytr, wtr, st = build_training_set(x_trainval_i, y_trainval, train_times_test, kind, "test", dataset_name)
        if args.model == "mitra_lora":  # LORA: refit on train+val for the selected number of steps (and lambda)
            clf = LoRAClassifier(lora_info["lora_lr"], lora_info["lora_lambda"]).fit(
                Xtr, ytr, build_training_set.last_subj, wtr, n_steps=lora_info["lora_best_step"], pair_groups=Xtr["eval_time"].to_numpy(),
                crank=make_crank(x_trainval_i, y_trainval, train_times_test, lora_info["lora_kappa"]) if lora_info["lora_lambda"] > 0 else None)
            st.update(lora_info)
        else:
            clf = make_clf().fit(Xtr, ytr, wtr)
        S_raw, row_t, row_pid, p0 = predict_S(clf, x_test_i, test_times)
        clf.close()
        S = np.minimum.accumulate(S_raw, axis=1)
        test = survival_metrics(y_trainval, y_test_f, S, test_times)
        append_csv(rows_path, st)
        tv_oof = oof_predict(x_trainval_i, y_trainval, test_times, kind, "test", dataset_name, train_times_test, lora_info) if args.oof_folds else None  # OOF

        # row-level AUC / BCE on the observable rows (as in the paper's scripts)
        T_r, D_r = y_test_f["time"][row_pid], y_test_f["event"][row_pid]
        valid = D_r | (row_t <= T_r)
        y_true = (D_r & (T_r <= row_t)).astype(int)[valid]
        p_ev = np.clip(1.0 - p0[valid], 1e-8, 1 - 1e-8)
        auc_roc = roc_auc_score(y_true, p_ev) if len(np.unique(y_true)) > 1 else np.nan
        bce = log_loss(y_true, p_ev, labels=[0, 1])

        risk = np.mean(1 - S, axis=1)
        np.savez_compressed(
            os.path.join(pred_dir, f"{dataset_name}.npz"),
            y_train_time=y_trainval["time"], y_train_event=y_trainval["event"],
            y_test_time=y_test_f["time"], y_test_event=y_test_f["event"],
            cidx_risk=risk, cidx_tau=max_trainval_time,
            ibs_S=S, ibs_times=test_times, auc_risk=risk, auc_times=test_times,
            # CALIB (10-27): raw predictions of both phases and the train-only outcomes, for post-hoc
            # calibration fitted on the validation split (calibrate_static.py)
            test_S_raw=S_raw, val_S_raw=S_val_raw, val_times=val_times,
            y_val_time=y_val["time"], y_val_event=y_val["event"],
            y_tr_time=y_train["time"], y_tr_event=y_train["event"],
            # OOF (10-31): out-of-fold predictions of the training subjects (val phase: train subjects at val_times;
            # test phase: train+val subjects at test_times); empty arrays when --oof_folds is 0
            tr_oof_S_raw=tr_oof if tr_oof is not None else np.zeros((0, 0)),
            tv_oof_S_raw=tv_oof if tv_oof is not None else np.zeros((0, 0)))

        row = {"dataset": dataset_name, "K": K,
               "val_c_index": r4(val["c_index"]), "val_ipcw_c_index": r4(val["ipcw_c_index"]),
               "val_ibs": r4(val["ibs"]), "val_mean_auc": r4(val["mean_auc"]),
               "c_index": r4(test["c_index"]), "ipcw_c_index": r4(test["ipcw_c_index"]),
               "ibs": r4(test["ibs"]), "mean_auc": r4(test["mean_auc"]),
               "auc_roc": r4(auc_roc), "bce": r4(bce), "censoring_model": st.get("censoring_model", "")}
        append_csv(csv_path, row)
        print(f"{dataset_name}: C {row['c_index']} IPCW-C {row['ipcw_c_index']} IBS {row['ibs']} "
              f"AUC {row['mean_auc']} | rows {st['rows_used']}/{st['rows_full']}")
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        continue
