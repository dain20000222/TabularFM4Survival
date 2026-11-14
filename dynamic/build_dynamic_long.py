# One long table of the dynamic results (one row per model and dataset) from the runs of run_dynamic.sh.
#   runs/results/metrics.csv            every model, and the CF models after post-hoc calibration
#   runs/own_loss/results/metrics.csv   the DH models after calibration with their own loss
# Per model and dataset, the representation of the history is the one with the best mean validation IPCW C-index (ties:
# first in file order); the prediction times are the landmarks that every model has for the dataset. Rows
# "<model> + calibration" use the representation of the uncalibrated model, its C-index and integrated AUC, and the
# calibrated IBS (calibration is used for the IBS only).
#   python build_dynamic_long.py -> dynamic_long.csv (mean over the prediction times; same format as
#                                   ../results/dynamic_long.csv, the table behind the paper), dynamic_long_tau.csv
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# K of each dataset: the largest K <= 10 with at least 5 training events in every interval
PICK = {"aids": 6, "csl": 8, "epileptic": 7, "heart": 5, "oldmort": 10, "FRTCS": 8}
K_OF_k = {25.0: 4, 20.0: 5, 16.667: 6, 14.286: 7, 12.5: 8, 10.0: 10}
SRC = [os.path.join(HERE, "runs", "results", "metrics.csv")]
OWN_SRC = [os.path.join(HERE, "runs", "own_loss", "results", "metrics.csv")]
OWN_SRC = [p for p in OWN_SRC if os.path.isfile(p)]
OWN = bool(OWN_SRC)
NAMES = {"Cf Mitra Full": "MITRA-CF", "Cf Tabpfn Full": "TabPFN-CF", "Cf Xgboost Ipcw Km": "XGBoost-CF",
         "Dynamic Mitra Discrete": "MITRA-DH", "Dynamic Tabpfn Discrete": "TabPFN-DH", "Dynamic Xgboost Discrete": "XGBoost-DH",
         "Dynamic Deephit": "Dynamic DeepHit", "Joint Model": "Joint Model", "Landmark Cox": "Landmark CoxPH", "Landmark Rsf": "Landmark RSF",
         "Dynamic Tcsa": "TCSA"}
MET = {"cindex_ipcw": "C-index", "weighted_avg_auc": "Integrated AUC", "integrated_brier": "IBS"}

parts, have = [], set()
for p in OWN_SRC + SRC:
    m = pd.read_csv(p)
    is_dh_cal = m.model.str.endswith(" Cal") & m.model.str.contains(r"^Dynamic .* Discrete", regex=True)
    m = m[is_dh_cal] if p in OWN_SRC else (m[~is_dh_cal] if OWN else m)
    m = m[m.metric.isin(MET)].copy()
    m["dataset"] = m.dataset.str.replace(".csv", "", regex=False)
    m["K"] = m.k.round(3).map(K_OF_k)
    m = m[[PICK.get(d) == K for d, K in zip(m.dataset, m.K)]]
    m["base"] = m.model.str.replace(r" Cal$", "", regex=True)
    m = m[m.base.isin(NAMES)]
    key = pd.Series(list(zip(m.model, m.dataset)), index=m.index)
    m = m[~key.isin(have)]
    have |= set(zip(m.model, m.dataset))
    parts.append(m)
M = pd.concat(parts, ignore_index=True)
M["tau_r"] = M.tau.round(3)

# prediction times every model has for the dataset
sets = M.groupby(["dataset", "model"]).tau_r.apply(frozenset)
common = sets.groupby(level=0).apply(lambda s: frozenset.intersection(*s))
M = M[[t in common[d] for d, t in zip(M.dataset, M.tau_r)]]

# representation of the history: best mean validation IPCW C-index of the uncalibrated model
raw = M[M.model == M.base]
hp = raw[raw.metric == "cindex_ipcw"].groupby(["base", "dataset", "hyperparameter"], sort=False).val_score.mean().reset_index()
sel = hp.loc[hp.groupby(["base", "dataset"], sort=False).val_score.idxmax(), ["base", "dataset", "hyperparameter"]]
X = M.merge(sel, on=["base", "dataset", "hyperparameter"])

rows_tau = []
for (model, base, d), g in X.groupby(["model", "base", "dataset"]):
    name = NAMES[base] + (" + calibration" if model != base else "")
    for (mt, tau), h in g.groupby(["metric", "tau_r"]):
        rows_tau.append(dict(dataset=d, K=PICK[d], model=name, metric=MET[mt], tau=tau, test=h.test_score.mean(), val=h.val_score.mean()))
T = pd.DataFrame(rows_tau)
# calibrated rows: C-index and integrated AUC of the uncalibrated model
cal = T.model.str.endswith(" + calibration") & T.metric.isin(["C-index", "Integrated AUC"])
rawT = T[~T.model.str.endswith(" + calibration")].set_index(["dataset", "model", "metric", "tau"])
T.loc[cal, ["test", "val"]] = [rawT.loc[(d, m.replace(" + calibration", ""), mt, t), ["test", "val"]].to_numpy()
                              for d, m, mt, t in zip(T.dataset[cal], T.model[cal], T.metric[cal], T.tau[cal])]
T.to_csv(os.path.join(HERE, "dynamic_long_tau.csv"), index=False)
A = T.groupby(["dataset", "K", "model", "metric"])[["test", "val"]].mean().unstack("metric")
out = pd.DataFrame(index=A.index)
for v in MET.values():
    out[v] = A[("test", v)]
    out["val_" + v] = A[("val", v)]
out = out.reset_index()
out.to_csv(os.path.join(HERE, "dynamic_long.csv"), index=False)
print(out.groupby("model").dataset.nunique().to_string())
print("prediction times per dataset:", {d: len(c) for d, c in common.items()})
