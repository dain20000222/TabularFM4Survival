# IBS of the CF and DH models of MITRA and TabPFN as a function of K, before and after post-hoc calibration.
# Input: ../tmlr_own_loss_tables_11_10/static_long_own.csv (54 datasets, KM grid; CF calibrated with the weighted BCE, DH with the DH loss).
# Output: dh_k_calibration.png, dh_k_calibration.csv (mean and standard error over the 54 datasets).
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300, "savefig.bbox": "tight", "axes.linewidth": 1.2})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)  # outputs
L = pd.read_csv(os.path.join(HERE, "..", "results", "static_long.csv"))
COL = {"MITRA-CF": "#C62835", "MITRA-DH": "#F08080", "TabPFN-CF": "#E8963A", "TabPFN-DH": "#F5C98B"}
CAL = " + calibration"
rows = []
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
for ax, b in zip(axes, ["MITRA", "TabPFN"]):
    for f in ["CF", "DH"]:
        for cal, ls, mk in ((False, "--", "o"), (True, "-", "s")):
            m = f"{b}-{f}"
            g = L[L.model == m + (CAL if cal else "")].groupby("K").ibs.agg(["mean", "sem", "count"])
            ax.errorbar(g.index, g["mean"], yerr=g["sem"], color=COL[m], ls=ls, marker=mk, lw=2, ms=5, capsize=3,
                        mfc=COL[m] if cal else "white", label=f"{m}, {'after' if cal else 'before'} calibration")
            for k, r in g.iterrows():
                rows.append({"model": m, "calibrated": cal, "K": k, "ibs_mean": round(r["mean"], 4), "ibs_se": round(r["sem"], 4), "datasets": int(r["count"])})
    ax.set_title(b, fontsize=12, fontweight="bold")
    ax.set_xticks([5, 10, 15, 20])
    ax.set_xlabel("Number of intervals K", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
axes[0].set_ylabel("IBS (mean over 54 datasets) ↓", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "dh_k_calibration.png"))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "dh_k_calibration.csv"), index=False)
