# Paired comparison of each TFM with XGBoost under the same formulation.
# Input: ../tmlr_own_loss_tables_11_10/static_long_own.csv (54 datasets, KM grid). Models as in Table 1: TFM models with post-hoc
# calibration (IBS), XGBoost-CF / XGBoost-DH without. Per dataset, the metric is averaged over K in {5, 10, 15, 20}.
# Win rate: share of datasets on which the TFM has the better value; SE = sqrt(p (1 - p) / n).
# Relative gain: (TFM - XGBoost) / XGBoost in %, sign such that positive = TFM better; mean and standard error over the datasets.
# Output: tfm_vs_xgboost.csv, tfm_vs_xgboost.tex
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)  # outputs
L = pd.read_csv(os.path.join(HERE, "..", "results", "static_long.csv"))
CAL = " + calibration"
MET = [("ipcw_c_index", 1, "C-Index"), ("mean_auc", 1, "Integrated AUC"), ("ibs", -1, "IBS")]


def per_dataset(m, cal):
    x = L[L.model == m].groupby("dataset")[["ipcw_c_index", "mean_auc"]].mean()
    return x.join(L[L.model == m + (CAL if cal else "")].groupby("dataset").ibs.mean())


rows, tex = [], []
for f in ["CF", "DH"]:
    xg = per_dataset(f"XGBoost-{f}", False)
    for b in ["MITRA", "TabPFN"]:
        t = per_dataset(f"{b}-{f}", True)
        r = {"comparison": f"{b}-{f} vs. XGBoost-{f}"}
        win, gain = [], []
        for c, s, name in MET:
            g = (s * (t[c] - xg[c]) / xg[c] * 100).dropna()
            n = len(g); p = (g > 0).mean()
            r.update({f"{name} n": n, f"{name} wins": int((g > 0).sum()), f"{name} win %": round(100 * p, 1),
                      f"{name} win SE": round(100 * np.sqrt(p * (1 - p) / n), 1), f"{name} gain %": round(g.mean(), 2),
                      f"{name} gain SE": round(g.sem(), 2), f"{name} median gain %": round(g.median(), 2)})
            win.append(f"${100 * p:.0f} \\pm {100 * np.sqrt(p * (1 - p) / n):.0f}$")
            gain.append(f"${g.mean():+.1f} \\pm {g.sem():.1f}$")
        rows.append(r)
        tex.append(f"{b}-{f} vs.\\ XGBoost-{f} & " + " & ".join(win + gain) + " \\\\")
    tex.append("\\midrule")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "tfm_vs_xgboost.csv"), index=False)
open(os.path.join(OUT, "tfm_vs_xgboost.tex"), "w").write("\n".join(tex[:-1]) + "\n")
print(pd.DataFrame(rows).T.to_string())
