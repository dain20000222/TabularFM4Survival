# Fig. 5 v2: the correlation figure in the style of Figs 3, 4, 6 (print size, 8 pt, legend at the bottom), small
# enough to sit next to the text (wrapfigure, about half the text width). Same data as ../final_11_07/make_static.py ("Figure 3"):
# calibrated CF models (MITRA-CF, TabPFN-CF, XGBoost-CF; the weighted BCE / ROC-AUC are computed from the calibrated probabilities,
# so XGBoost-CF is calibrated here too), mean over K per dataset.
#   python static_corr_v2.py -> v3/fig5_correlation_v2.png
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from common import BCOL, HERE, OUTV

FS = 8
plt.rcParams.update({"font.size": FS, "axes.titlesize": 9, "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5, "axes.linewidth": 0.8})
F7 = os.path.join(HERE, "..", "results")
L = pd.read_csv(os.path.join(F7, "static_long.csv"))
CAL = " + calibration"
C = L[L.model.isin([m + CAL for m in ("MITRA-CF", "TabPFN-CF", "XGBoost-CF")])].copy()
C["model"] = C.model.str.replace(CAL, "", regex=False)
F = pd.read_csv(os.path.join(F7, "static_classification_metrics.csv"))  # static/classification_metrics.py
F = F.merge(C[["dataset", "K", "model", "ipcw_c_index", "ibs"]], on=["dataset", "K", "model"])
A = F.groupby(["model", "dataset"]).mean(numeric_only=True).reset_index()
COL = {"MITRA-CF": BCOL["MITRA"], "TabPFN-CF": BCOL["TabPFN"], "XGBoost-CF": BCOL["XGBoost"]}
LBL = {"ipcw_c_index": "C-Index ↑", "ibs": "IBS ↓", "auc_roc_w": "Weighted ROC-AUC ↑", "bce_w": "Weighted BCE ↓"}
fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.3))
rs = {}
for r, sm in enumerate(["ipcw_c_index", "ibs"]):
    for c, cm in enumerate(["auc_roc_w", "bce_w"]):
        ax = axes[r, c]
        for mo, col in COL.items():
            a = A[(A.model == mo) & A[cm].notna() & A[sm].notna()]
            ax.scatter(a[cm], a[sm], s=7, color=col, edgecolor="white", linewidth=0.3, alpha=0.85, label=mo, zorder=3)
        a = A[A[cm].notna() & A[sm].notna()]
        rr = pearsonr(a[cm], a[sm])[0]
        rs[(sm, cm)] = rr
        sl, ic = np.polyfit(a[cm], a[sm], 1)
        xx = np.linspace(a[cm].min(), a[cm].max(), 50)
        ax.plot(xx, sl * xx + ic, "--", color="0.2", lw=0.9, zorder=4)
        ax.text(0.04 if rr > 0 or r == 1 else 0.96, 0.95, f"$r = {rr:.2f}$", transform=ax.transAxes, ha="left" if (rr > 0 or r == 1) else "right",
                va="top", fontsize=FS - 0.5, zorder=5, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.7", lw=0.5))
        ax.grid(alpha=0.3)
        if r == 1:
            ax.set_xlabel(LBL[cm])
        if c == 0:
            ax.set_ylabel(LBL[sm])
        ax.tick_params(length=2)
h, lab = axes[0, 0].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=FS - 0.5, markerscale=1.8, handletextpad=0.2, columnspacing=0.8,
           bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.6)
fig.savefig(os.path.join(OUTV, "fig6_correlation.png"), dpi=300, bbox_inches="tight")
print({k: round(v, 2) for k, v in rs.items()}, A.dataset.nunique())
