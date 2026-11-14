# Row-level classification metrics of the calibrated CF models (MITRA-CF, TabPFN-CF, XGBoost-CF; KM grid; the selected
# calibration "<tag>_cal"), with the weights W of the training loss (G = marginal KM of the train+val censoring times,
# floor 1e-6): weighted BCE = weighted mean of the row BCEs, weighted ROC-AUC = ROC-AUC with the same sample weights.
# Input of Figure 6 (../analysis/fig6_correlation.py). Reads the stored predictions of run_static.sh
# (runs/full_km_K<K>_g<g>/preds/K<K>/<tag>_cal/<dataset>.npz).
#   python classification_metrics.py -> static_classification_metrics.csv (same format as the file of ../results/)
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
FLOOR, EPS = 1e-6, 1e-8
TAGS = {"MITRA-CF": ("static_mitra_cf_full_kmq", 0), "TabPFN-CF": ("static_tabpfn_cf_full_kmq", 1), "XGBoost-CF": ("static_xgboost_cf_ipcw_km_kmq", 2)}


def km_censoring(time, event):
    order = np.argsort(time)
    t, c = time[order], ~event[order]
    uniq = np.unique(t[c])
    surv, s = [], 1.0
    for u in uniq:
        s *= 1.0 - np.sum((t == u) & c) / np.sum(t >= u)
        surv.append(s)
    uniq, surv = np.asarray(uniq), np.asarray(surv)

    def G(x):
        x = np.asarray(x, float)
        if len(uniq) == 0:
            return np.ones_like(x)
        idx = np.searchsorted(uniq, x, side="right") - 1
        return np.where(idx >= 0, surv[np.clip(idx, 0, None)], 1.0)
    return G


def row_metrics(rec):
    T, D = rec["y_test_time"].astype(float), rec["y_test_event"].astype(bool)
    S, times = rec["ibs_S"].astype(float), rec["ibs_times"].astype(float)
    G = km_censoring(rec["y_train_time"].astype(float), rec["y_train_event"].astype(bool))
    Tr, Dr, tr = np.repeat(T, len(times)), np.repeat(D, len(times)), np.tile(times, len(T))
    p = np.clip(1.0 - S.ravel(), EPS, 1 - EPS)
    y = (Dr & (Tr <= tr)).astype(int)
    bce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    w = np.zeros(len(y))
    ev, alive = Dr & (Tr <= tr), Tr > tr
    w[ev] = np.maximum(G(tr[ev]), FLOOR) / np.maximum(G(Tr[ev]), FLOOR)
    w[alive] = 1.0
    k = w > 0
    return {"bce_w": np.sum(w[k] * bce[k]) / np.sum(w[k]),
            "auc_roc_w": roc_auc_score(y[k], p[k], sample_weight=w[k]) if len(np.unique(y[k])) > 1 else np.nan}


rows = []
for name, (tag, g) in TAGS.items():
    for K in (5, 10, 15, 20):
        d = os.path.join(HERE, "runs", f"full_km_K{K}_g{g}", "preds", f"K{K}", tag + "_cal")
        if not os.path.isdir(d):
            print("missing", d)
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".npz"):
                rows.append({"model": name, "dataset": f[:-4], "K": K, **row_metrics(dict(np.load(os.path.join(d, f))))})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "static_classification_metrics.csv"), index=False)
print(out.groupby("model")[["bce_w", "auc_roc_w"]].mean().round(4), "\n", out.groupby("model").dataset.nunique())
