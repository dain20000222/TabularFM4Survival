# One long table of the static results (one row per model, dataset and K) from the runs of run_static.sh.
#   runs/full_km_K<K>_g<g>/      CF and DH models (results/), baselines and MTLR / BCESurv (csv files of the run folder)
#   runs/mc_static_<model>_K<K>/ multi-class models
#   runs/own_loss/results/       calibration of the DH and MC models with their own loss
#   runs/monotonicity/, runs/calw/, runs/mitra_ctx*_K<K>/   inputs of Table 7, Figure 5 and Table 8 (if they were run)
# Rows "<model> + calibration": IBS after post-hoc calibration; the C-index and integrated AUC are those of the
# uncalibrated model (calibration is used for the IBS only). The calibrated MITRA-DH, TabPFN-DH and MC rows use the
# calibration with the model's own loss; the calibrated CF rows use the weights of the CF loss. Negative values
# (-1: metric undefined) are NaN.
#   python build_static_long.py -> static_long.csv, static_calibration_val_bce.csv, static_monotonicity.csv,
#       static_calibration_bcesurv_weights.csv, static_context_sampling.csv (same format as the files of ../results/,
#       which hold the results behind the paper)
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
DS = {f[:-4] for f in os.listdir(os.path.join(HERE, "static_data")) if f.endswith(".csv")}
KS = (5, 10, 15, 20)
MET = ["ipcw_c_index", "mean_auc", "ibs"]
EXTRA = ["val_ipcw_c_index", "val_mean_auc", "val_ibs", "auc_roc", "bce"]
# model -> (tag of the result files, model group of run_static.sh)
CFDH = {"MITRA-CF": ("static_mitra_cf_full_kmq", 0), "TabPFN-CF": ("static_tabpfn_cf_full_kmq", 1),
        "XGBoost-CF": ("static_xgboost_cf_ipcw_km_kmq", 2), "MITRA-DH": ("static_mitra_dh_kmq", 3),
        "TabPFN-DH": ("static_tabpfn_dh_kmq", 4), "XGBoost-DH": ("static_xgboost_dh_kmq", 5),
        "XGBoost-CF (W abl)": ("static_xgboost_cf_abl_kmq", 8)}  # W abl: BCESurv weights (XGBoost BCESurv in the paper)
MC = {"MITRA-MC": "mitra", "TabPFN-MC": "tabpfn", "XGBoost-MC": "xgboost"}
OWN = {"static_mitra_dh_kmq": "MITRA-DH", "static_tabpfn_dh_kmq": "TabPFN-DH", "static_mitra_mc_kmq": "MITRA-MC",
       "static_tabpfn_mc_kmq": "TabPFN-MC", "static_xgboost_mc_kmq": "XGBoost-MC"}


def clean(x):
    for c in MET + EXTRA:
        if c in x:
            x[c] = pd.to_numeric(x[c], errors="coerce")
            x.loc[x[c] < 0, c] = np.nan
    return x


def read_eval(path):
    if not os.path.isfile(path):
        return None
    x = pd.read_csv(path).drop_duplicates("dataset", keep="last")
    return clean(x[["dataset"] + [c for c in MET + EXTRA if c in x]].copy())


def add_pair(rows, name, raw, cal, K):
    """raw and calibrated evaluation tables of one model -> rows; calibrated: IBS calibrated, rest of the raw model."""
    if raw is None:
        return
    rows.append(raw.assign(model=name, K=K))
    if cal is not None:
        c = raw.drop(columns=["ibs", "val_ibs", "bce"], errors="ignore").merge(cal[["dataset"] + [v for v in ("ibs", "val_ibs", "bce") if v in cal]], on="dataset")
        rows.append(c.assign(model=name + " + calibration", K=K))


rows = []
for K in KS:
    for name, (tag, g) in CFDH.items():
        d = os.path.join(RUNS, f"full_km_K{K}_g{g}", "results")
        add_pair(rows, name, read_eval(os.path.join(d, f"{tag}_evaluation_final_{K}.csv")),
                 read_eval(os.path.join(d, f"{tag}_calibs_evaluation_final_{K}.csv")), K)
    for name, m in MC.items():
        d = os.path.join(RUNS, f"mc_static_{m}_K{K}", "results")
        add_pair(rows, name, read_eval(os.path.join(d, f"static_{m}_mc_kmq_evaluation_final_{K}.csv")),
                 read_eval(os.path.join(d, f"static_{m}_mc_kmq_calibs_evaluation_final_{K}.csv")), K)
    for f in glob.glob(os.path.join(RUNS, f"full_km_K{K}_g6", "static_baseline_evaluation_final_*_km_quantile.csv")):
        b = pd.read_csv(f).drop_duplicates("dataset_name", keep="last")
        for code, name in (("cph", "CoxPH"), ("rsf", "RSF"), ("dh", "DeepHit"), ("ds", "DeepSurv")):
            x = pd.DataFrame({"dataset": b.dataset_name, **{m: b.get(f"test_{code}_{m}") for m in MET},
                              **{f"val_{m}": b.get(f"val_{code}_{m}") for m in MET}})
            rows.append(clean(x).assign(model=name, K=K))
    for code, name in (("linear_mtlr", "Linear MTLR"), ("neural_mtlr", "Neural MTLR"), ("bce_surv", "BCESurv")):  # paper: MTLR, Neural MTLR, MLP BCESurv
        for f in glob.glob(os.path.join(RUNS, f"full_km_K{K}_g7", f"static_{code}_evaluation_final_*_km_quantile.csv")):
            b = pd.read_csv(f).drop_duplicates("dataset_name", keep="last")
            x = pd.DataFrame({"dataset": b.dataset_name, **{m: b.get(f"test_{m}") for m in MET}, **{f"val_{m}": b.get(f"val_{m}") for m in MET}})
            rows.append(clean(x).assign(model=name, K=K))
if not rows:
    raise SystemExit("no results under runs/ (run run_static.sh first)")
L = pd.concat(rows, ignore_index=True)
L = L[L.dataset.isin(DS)]
dup = L.duplicated(["dataset", "K", "model"], keep=False)
assert not dup.any(), L[dup].head()

# calibration of the DH and MC models with their own loss: replaces the IBS of their "+ calibration" rows
own = [pd.read_csv(f) for f in glob.glob(os.path.join(RUNS, "own_loss", "results", "*", "own_loss_*.csv"))]
if own:
    O = pd.concat(own, ignore_index=True)
    O = O[O.dataset.isin(DS)].assign(model=lambda x: x.model.map(OWN) + " + calibration").drop_duplicates(["dataset", "K", "model"], keep="last")
    L = L.merge(O[["dataset", "K", "model", "ibs_own_loss"]], on=["dataset", "K", "model"], how="left")
    is_dhmc = L.model.str.match(r"(MITRA|TabPFN)-DH \+ calibration|(MITRA|TabPFN|XGBoost)-MC \+ calibration")
    L.loc[is_dhmc, "ibs"] = L.loc[is_dhmc, "ibs_own_loss"]
    L = L.drop(columns="ibs_own_loss")
else:
    print("no own-loss calibration found: the calibrated DH / MC rows keep the calibration with the CF weights")
L.to_csv(os.path.join(HERE, "static_long.csv"), index=False)

# validation weighted BCE of each classification model as it is used (the selected calibrator; no calibration for the static
# XGBoost-CF / XGBoost-DH models), from the calibration with the weights of the CF loss: used by ../analysis/common.py
TAG = {f"static_{b}_{t}_kmq": f"{n}-{f}" for b, n in (("mitra", "MITRA"), ("tabpfn", "TabPFN"), ("xgboost", "XGBoost")) for f, t in (("DH", "dh"), ("MC", "mc"))}
TAG.update({"static_mitra_cf_full_kmq": "MITRA-CF", "static_tabpfn_cf_full_kmq": "TabPFN-CF", "static_xgboost_cf_ipcw_km_kmq": "XGBoost-CF"})
RAW = ["XGBoost-CF", "XGBoost-DH"]
sel = []
for f in glob.glob(os.path.join(RUNS, "*", "results", "calibration_selection_*_kmq*.csv")):
    x = pd.read_csv(f)
    x = x[x.model.isin(TAG) & x.dataset.isin(DS)]
    if len(x):
        x["val_BCE"] = [r["val_bce_none"] if TAG[r.model] in RAW else r[f"val_bce_{r.selected}"] for _, r in x.iterrows()]
        sel.append(x.assign(model=x.model.map(TAG))[["dataset", "K", "model", "val_BCE"]])
if sel:
    V = pd.concat(sel).drop_duplicates(["dataset", "K", "model"], keep="last").dropna(subset=["val_BCE"]).sort_values(["model", "K", "dataset"])
    V.to_csv(os.path.join(HERE, "static_calibration_val_bce.csv"), index=False)
print(L.dropna(subset=MET).groupby(["model", "K"]).dataset.nunique().unstack().to_string())

# Appendix inputs (written only if the corresponding runs exist)
# Table 7: monotonicity.py outputs (runs/monotonicity/*.csv)
mono = [pd.read_csv(f) for f in glob.glob(os.path.join(RUNS, "monotonicity", "*.csv"))]
if mono:
    Mo = pd.concat(mono, ignore_index=True).drop_duplicates(["dataset", "K", "model"])
    Mo["model"] = Mo.model.map({"static_mitra_cf_full_kmq": "MITRA-CF", "static_tabpfn_cf_full_kmq": "TabPFN-CF", "static_xgboost_cf_ipcw_km_kmq": "XGBoost-CF"})
    Mo[Mo.dataset.isin(DS)].to_csv(os.path.join(HERE, "static_monotonicity.csv"), index=False)
# Figure 5: calibration of MITRA-CF / TabPFN-CF fitted with the BCESurv weights (cal_abl.py, runs/calw/K<K>/results)
cw = []
for name, tag in (("MITRA-CF", "static_mitra_cf_full_kmq"), ("TabPFN-CF", "static_tabpfn_cf_full_kmq")):
    f = glob.glob(os.path.join(RUNS, "calw", "K*", "results", f"{tag}_cal_evaluation_final_*.csv"))
    if f:
        x = pd.concat(pd.read_csv(p) for p in f).drop_duplicates(["dataset", "K"], keep="last")
        cw.append(x[["dataset", "K", "ibs"]].assign(model=name))
if cw:
    pd.concat(cw, ignore_index=True)[["model", "dataset", "K", "ibs"]].to_csv(os.path.join(HERE, "static_calibration_bcesurv_weights.csv"), index=False)
# Table 8: MITRA-CF with a sample-stratified ("subject") or horizon-stratified ("horizon") context (mitra_ctx.py)
P = L[L.model == "MITRA-CF"].set_index(["dataset", "K"])[["ipcw_c_index", "mean_auc"]].join(
    L[L.model == "MITRA-CF + calibration"].set_index(["dataset", "K"])[["ibs"]])
ctx = []
for mode in ["subject", "horizon"]:
    for K in KS:
        w = os.path.join(RUNS, f"mitra_ctx{mode}_K{K}", "results")
        f_raw = glob.glob(os.path.join(w, f"static_mitra_cf_full_kmq_ctx{mode}_evaluation_final_{K}.csv"))
        f_cal = glob.glob(os.path.join(w, f"static_mitra_cf_full_kmq_ctx{mode}_calibs_evaluation_final_{K}.csv"))
        f_rows = glob.glob(os.path.join(w, f"static_mitra_cf_full_kmq_ctx{mode}_rows_{K}.csv"))
        if not (f_raw and f_cal and f_rows):
            continue
        a = pd.read_csv(f_raw[0]).drop_duplicates("dataset", keep="last").set_index("dataset")
        c = pd.read_csv(f_cal[0]).drop_duplicates("dataset", keep="last").set_index("dataset")
        n = pd.read_csv(f_rows[0]); n = n[n.phase == "test"].drop_duplicates("dataset", keep="last").set_index("dataset").rows_used
        for d in a.index:
            v = {m: a.loc[d, m] for m in ["ipcw_c_index", "mean_auc"]}
            v["ibs"] = c.loc[d, "ibs"] if d in c.index else np.nan
            ctx.append(dict(mode=mode, dataset=d, K=K, rows=n.get(d, np.nan), **{m: (np.nan if v[m] is None or v[m] < 0 else v[m]) for m in MET}))
if ctx:
    pd.DataFrame(ctx).join(P, on=["dataset", "K"], rsuffix="_uniform").to_csv(os.path.join(HERE, "static_context_sampling.csv"), index=False)
