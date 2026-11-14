# Fig. 6 v3: where the weights W enter each model, W^abl - W in the test IBS, per K, 54 datasets.
#   XGBoost-CF: W in training (sample weights); no post-hoc calibration.
#   MITRA-CF, TabPFN-CF: W only in the post-hoc calibration (weighted BCE); calibration fit with W (paper) vs W^abl
#   (../tmlr_calw_11_09). Paired by dataset and K; mean over datasets +- s.e. Positive = W is better.
# Legend / y label / shading: W^abl = BCESurv weighting, W = CF weighting.
#   python static_weights_v3.py -> v2/fig6_weights_v3.png (difference), v2/fig6_weights_v3_pct.png (% reduction vs W^abl), v2/fig6_weights_v3.csv
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
from common import COL, HERE

FS = 8
plt.rcParams.update({"font.size": FS, "axes.titlesize": 9, "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5, "axes.linewidth": 0.8})
P = pd.read_csv(os.path.join(HERE, "..", "results", "static_long.csv"))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
ibs = lambda name: P[P.model == name].drop_duplicates(["dataset", "K"]).set_index(["dataset", "K"]).ibs
CW = pd.read_csv(os.path.join(HERE, "..", "results", "static_calibration_bcesurv_weights.csv"))  # static/cal_abl.py (run_static.sh)


def calw(tag):
    return CW[CW.model == tag].drop_duplicates(["dataset", "K"], keep="last").set_index(["dataset", "K"]).ibs


LINES = {"XGBoost-CF": (ibs("XGBoost-CF (W abl)"), ibs("XGBoost-CF"), "XGBoost"),
         "MITRA-CF": (calw("MITRA-CF"), ibs("MITRA-CF + calibration"), "MITRA"),
         "TabPFN-CF": (calw("TabPFN-CF"), ibs("TabPFN-CF + calibration"), "TabPFN")}
COLS = {"MITRA": "#C62835", "TabPFN": "#E8963A", "XGBoost": "#2F6FE0"}
VERS = {"diff": ("Difference in IBS for\nBCESurv vs CF weighting", lambda d: d.abl - d.w, "fig5_weights_difference.png"),
        "pct": ("Reduction in IBS (%) vs\nBCESurv weighting", lambda d: 100 * (d.abl - d.w) / d.abl, "fig5_weights.png")}
rows = []
for v, (ylab, f, fname) in VERS.items():
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    hi = 0
    for i, (lab, (a, w, b)) in enumerate(LINES.items()):
        d = pd.concat([a.rename("abl"), w.rename("w")], axis=1).dropna()
        d["y"] = f(d)
        s = d.groupby(level="K")["y"].agg(["mean", "sem", "count"])
        s["W better on"] = d.groupby(level="K").apply(lambda g: (g.w < g.abl).sum())
        rows.append(s.assign(model=lab, version=v))
        hi = max(hi, (s["mean"] + s["sem"]).max())
        ax.errorbar(s.index + (i - 1) * 0.35, s["mean"], yerr=s["sem"], color=COLS[b], marker="o", ms=3.5, lw=1.3, elinewidth=0.8,
                    capsize=1.5, label=lab)
    top, bot = hi * 1.3, -hi * 0.3  # below 0 kept small
    ax.axhline(0, color="black", lw=0.8)
    ax.axhspan(0, top, color="#2F6FE0", alpha=0.08, lw=0)
    ax.axhspan(bot, 0, color="0.5", alpha=0.10, lw=0)
    ax.text(0.03, 0.96, "CF weighting better", transform=ax.transAxes, ha="left", va="top", fontsize=FS - 1, color="#1f4f9e")
    ax.text(0.97, 0.03, "BCESurv weighting better", transform=ax.transAxes, ha="right", va="bottom", fontsize=FS - 1.5, color="0.3")
    ax.set_ylim(bot, top)
    ax.set_xticks([5, 10, 15, 20])
    ax.set_xlabel("$K$")
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=3, frameon=False, fontsize=FS - 0.5, columnspacing=1.0, handlelength=1.6)
    fig.savefig(os.path.join(OUT, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
T = pd.concat(rows).reset_index()
T.round(5).to_csv(os.path.join(OUT, "fig5_weights.csv"), index=False)
print(T.round(4).to_string())
