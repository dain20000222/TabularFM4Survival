# Table 7 (Appendix D.3): monotonicity violations of the CF predictions before clipping, and clipping vs. isotonic regression.
# Input: ../results/static_monotonicity.csv (static/monotonicity.py), per model, dataset and K:
#   obs_with_violation  share of test observations whose predicted survival probability increases between two adjacent grid points
#   largest_increase    mean over ALL test observations of the largest increase (zero without a violation)
# "Size" in the table is the mean largest increase among the observations WITH a violation, largest_increase / obs_with_violation,
# averaged over the datasets with at least one violation.
# Rows per K: means over the datasets; row "All": means over the dataset-K combinations of Table 1 (= the values of Table 1).
#   python table7_monotonicity.py -> out/monotonicity_table.tex, out/monotonicity_violation_size.csv
import os

import numpy as np
import pandas as pd

H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, "out"); os.makedirs(OUT, exist_ok=True)
R = pd.read_csv(os.path.join(H, "..", "results", "static_monotonicity.csv"))
L = pd.read_csv(os.path.join(H, "..", "results", "static_long.csv"))
R["size_given_violation"] = np.where(R.obs_with_violation > 0, R.largest_increase / R.obs_with_violation, np.nan)
core = ["MITRA-CF + calibration", "TabPFN-CF + calibration", "XGBoost-CF", "MITRA-DH + calibration", "TabPFN-DH + calibration", "XGBoost-DH", "XGBoost-MC + calibration",
        "Linear MTLR", "Neural MTLR", "BCESurv", "XGBoost-CF (W abl)", "CoxPH", "RSF", "DeepHit", "DeepSurv"]
X = L[L.model.isin(core)].dropna(subset=["ipcw_c_index", "ibs"]); ok = X.groupby(["dataset", "K"]).model.nunique(); ok = set(ok[ok == len(core)].index)
T1 = R[[k in ok for k in zip(R.dataset, R.K)]]
cols = ["obs_with_violation", "cindex_clip", "cindex_iso", "auc_clip", "auc_iso", "ibs_clip", "ibs_iso"]


def line(lab, k, r, size):
    return f"{lab} & {k} & {100 * r.obs_with_violation:.1f}\\% & {size:.3f} & {r.cindex_clip:.3f} & {r.cindex_iso:.3f} & {r.auc_clip:.3f} & {r.auc_iso:.3f} & {r.ibs_clip:.3f} & {r.ibs_iso:.3f} \\\\"


out, rows = [], []
for i, m in enumerate(["MITRA-CF", "TabPFN-CF", "XGBoost-CF"]):
    if i:
        out.append("\\midrule")
    Rm = R[R.model == m]
    g, s = Rm.groupby("K")[cols].mean(), Rm.groupby("K").size_given_violation.agg(["mean", "count"])
    for j, (K, r) in enumerate(g.iterrows()):
        out.append(line(f"\\textbf{{{m}}}" if j == 0 else "", K, r, s.loc[K, "mean"]))
        rows.append(dict(model=m, K=K, violations_pct=100 * r.obs_with_violation, size_given_violation=s.loc[K, "mean"], datasets_with_violation=int(s.loc[K, "count"])))
    out.append(line("", "All", T1[T1.model == m][cols].mean(), Rm.size_given_violation.mean()))
    rows.append(dict(model=m, K="All", violations_pct=100 * T1[T1.model == m].obs_with_violation.mean(), size_given_violation=Rm.size_given_violation.mean(),
                     datasets_with_violation=int(Rm.size_given_violation.notna().sum())))
open(os.path.join(OUT, "monotonicity_table.tex"), "w").write("\n".join(out) + "\n")
pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "monotonicity_violation_size.csv"), index=False)
print("\n".join(out))
