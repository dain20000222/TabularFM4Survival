# Shared loaders and style for the figures of this folder (reader questions on the draft, 11-07).
# Inputs: ../results/static_long.csv, dynamic_long.csv, dynamic_long_tau.csv; the calibration selection files (validation
# weighted BCE of the calibrated classification models); ../tmlr_static_full_11_06/dataset_rules.csv (dataset characteristics).
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300, "savefig.bbox": "tight", "axes.linewidth": 1.2})
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..")
F = os.path.join(R, "results")
CAL = " + calibration"
# XGBoost-CF / XGBoost-DH without post-hoc calibration (as Table 1 v2, ../tmlr_table1_bcesurv_11_09); MC models keep it.
# XGB_CAL=1 reproduces the earlier outputs (XGBoost calibrated) in v2/; the default writes to v3/.
XGB_CAL = os.environ.get("XGB_CAL", "0") == "1"
XGB_RAW = [] if XGB_CAL else ["XGBoost-CF", "XGBoost-DH", "XGBoost-CF (W abl)"]
OUTV = os.path.join(HERE, "out")
os.makedirs(OUTV, exist_ok=True)
MET = [("C-index", False), ("Integrated AUC", False), ("IBS", True)]
BACK = ["MITRA", "TabPFN", "XGBoost"]
FORMS = ["CF", "DH", "MC"]
CLS = [f"{b}-{f}" for b in BACK for f in FORMS]
TFM = ["MITRA-CF", "MITRA-DH", "TabPFN-CF", "TabPFN-DH"]
POOL = {"static": ["CoxPH", "RSF", "DeepHit", "DeepSurv"], "dynamic": ["Landmark CoxPH", "Landmark RSF", "Dynamic DeepHit", "Joint Model", "TCSA"]}
BCOL = {"MITRA": "#C62835", "TabPFN": "#E8963A", "XGBoost": "#2F6FE0"}
COL = {"MITRA-CF": "#C62835", "MITRA-DH": "#F08080", "MITRA-MC": "#7A1520", "TabPFN-CF": "#E8963A", "TabPFN-DH": "#F5C98B", "TabPFN-MC": "#9A5A10",
       "XGBoost-CF": "#2F6FE0", "XGBoost-DH": "#9DBCF2", "XGBoost-MC": "#173A80"}
DYN_DS = ["aids", "csl", "epileptic", "oldmort", "FRTCS"]
RENAME = {"ipcw_c_index": "C-index", "mean_auc": "Integrated AUC", "ibs": "IBS", "val_ipcw_c_index": "val_C-index",
          "val_mean_auc": "val_Integrated AUC", "val_ibs": "val_IBS"}


def calibrated(D):
    """Rows of the calibrated models under the plain name (C-index / integrated AUC of a '+ calibration' row are those of the raw model)."""
    have = {m[: -len(CAL)] for m in D.model if m.endswith(CAL)}
    X = D[~D.model.isin(have)].copy()
    X["model"] = X.model.str.replace(CAL, "", regex=False)
    return X


def static_long(raw=False):
    L = pd.read_csv(os.environ.get("STATIC_LONG", os.path.join(F, "static_long.csv"))).rename(columns=RENAME)  # STATIC_LONG: another long table (11-10: own-loss calibration)
    # MITRA-MC / TabPFN-MC: K = 5, 10 only (<= 10 classes); the few K = 15, 20 rows are datasets with fewer grid points
    L = L[~(L.model.str.match(r"(MITRA|TabPFN)-MC") & (L.K > 10))]
    if raw:
        return L
    S = calibrated(L[~L.model.isin([m + CAL for m in XGB_RAW])])
    # validation weighted BCE of the model as used (the selected calibrator, or no calibration for static XGBoost):
    # results/static_calibration_val_bce.csv, written by static/build_static_long.py
    sel = pd.read_csv(os.environ.get("STATIC_VAL_BCE", os.path.join(F, "static_calibration_val_bce.csv")))
    S = S.merge(sel.drop_duplicates(["dataset", "K", "model"], keep="last"), on=["dataset", "K", "model"], how="left")
    # IBS choice on validation: classification models by the calibrated validation weighted BCE, the others by the validation IBS
    S["val_IBS_sel"] = S.val_BCE.where(S.model.isin(CLS) & ~S.model.isin(XGB_RAW), S.val_IBS)
    return S


def dynamic_long():
    D = calibrated(pd.read_csv(os.path.join(F, "dynamic_long.csv")))
    D["val_IBS_sel"] = D.val_IBS
    return D


def chars():
    c = pd.read_csv(os.path.join(F, "static_dataset_rules.csv")).set_index("dataset")
    c = c[c.included]
    c["censoring"] = 1 - (c.events_train + c.events_val) / (c.n_train + c.n_val)
    c["events_per_covariate"] = c.events_train / c.covariates
    return c


def valcol(m):
    return "val_IBS_sel" if m == "IBS" else "val_" + m


def best(X, models, m, lower, unit):
    """Per unit, the test value of the model of `models` with the best validation value of metric m."""
    y = X[X.model.isin(models)].dropna(subset=[m, valcol(m)]).sort_values(valcol(m), ascending=lower)
    return y.groupby(unit).first()[[m, "model"]]


def rel_gain(a, r, lower):
    return ((r - a) if lower else (a - r)) / r * 100


def cse(x, ds):
    """Standard error of the mean of x, clustered by dataset ds."""
    x = pd.Series(np.asarray(x, float), index=np.asarray(ds))
    N, nd = len(x), x.index.nunique()
    if nd < 2:
        return np.nan
    u = (x - x.mean()).groupby(level=0).sum()
    return float(np.sqrt(nd / (nd - 1) * (u ** 2).sum()) / N)


def arrow(lower):
    return "↓" if lower else "↑"
