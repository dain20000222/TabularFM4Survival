# Static figures: drawn at the printed size (6.75 in wide, fonts 8-9 pt as \footnotesize), less text.
#   fig2_gains_v2.png   Fig. 2: relative gain over the classical baselines (validation choice among CoxPH, RSF, DeepHit, DeepSurv),
#                       axis showing the upside, right: share of datasets with a gain and number with a gain >= 20%;
#                       bottom: best TFM (validation choice between MITRA and TabPFN) vs XGBoost, one row per formulation
#   fig2_strip_v2.png   alternative to Fig. 2: one strip of the 54 datasets per metric (best TFM vs classical baselines), sorted
#   fig3_size_v2.png    Fig. 3: median gain over the classical baselines by training observations (<200, 200-700, >700)
#   fig4_formulations_v2.png  Fig. 4: CF - DH / CF - MC medians, shaded by which formulation wins, no win counts on the plot
#   fig4_formulations_table.tex  the win counts as a table (alternative / appendix)
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from common import OUTV, BACK, BCOL, COL, HERE, MET, POOL, TFM, best, chars, rel_gain, static_long

FS, FT = 8, 9.5  # footnotesize-like body, titles
plt.rcParams.update({"font.size": FS, "axes.titlesize": FT, "axes.labelsize": FS, "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS,
                     "legend.fontsize": FS - 0.5, "axes.linewidth": 0.8})
OUT = OUTV
os.makedirs(OUT, exist_ok=True)
S = static_long()
U = ["dataset", "K"]
NAME = {"C-index": "C-Index", "Integrated AUC": "Integrated AUC", "IBS": "IBS"}
ARR = {False: "↑", True: "↓"}


def per_dataset_gain(a_vals, r_vals, lower):
    j = pd.concat([a_vals.rename("a"), r_vals.rename("r")], axis=1).dropna()
    return rel_gain(j.a, j.r, lower).groupby(level="dataset").mean()


V = {m: S.dropna(subset=[m]).pivot_table(index=U, columns="model", values=m) for m, _ in MET}
BASE = {m: best(S, POOL["static"], m, lower, U)[m] for m, lower in MET}
BEST_TFM_FORM = {(m, f): best(S, [f"MITRA-{f}", f"TabPFN-{f}"], m, lower, U)[m] for m, lower in MET for f in ("CF", "DH")}
BEST_TFM = {m: best(S, TFM, m, lower, U)[m] for m, lower in MET}

# ---------------- Fig. 2 (gains)
ROWS = [(a, a, "base") for a in ["MITRA-CF", "MITRA-DH", "TabPFN-CF", "TabPFN-DH", "XGBoost-CF", "XGBoost-DH"]]
ROWS += [("TFM-CF vs XGBoost-CF", "TFM-CF", "XGBoost-CF"), ("TFM-DH vs XGBoost-DH", "TFM-DH", "XGBoost-DH")]
LO, HI = -20, 50
rng = np.random.default_rng(0)
fig = plt.figure(figsize=(6.75, 3.9))
gs = fig.add_gridspec(1, 6, width_ratios=[1, 0.5] * 3, wspace=0.10)
ax_first = None
yrow = {lab: i + (1 if i >= 6 else 0) for i, (lab, _, _) in enumerate(ROWS)}
stats = []
for k, (m, lower) in enumerate(MET):
    ax = fig.add_subplot(gs[0, 2 * k], sharey=ax_first) if ax_first is not None else fig.add_subplot(gs[0, 2 * k])
    ax_first = ax_first or ax
    tx = fig.add_subplot(gs[0, 2 * k + 1], sharey=ax_first)
    tx.axis("off")
    for lab, a, ref in ROWS:
        if ref == "base":
            g = per_dataset_gain(V[m][a], BASE[m], lower)
            col, mk = COL[a.replace("-DH", "-CF")], ("o" if a.endswith("CF") else "^")
        else:
            g = per_dataset_gain(BEST_TFM_FORM[(m, a[-2:])], V[m][ref], lower)
            col, mk = "#7a1f2b", ("o" if a.endswith("CF") else "^")
        g = g.to_numpy()
        y = yrow[lab]
        q1, med, q3 = np.percentile(g, [25, 50, 75])
        ax.add_patch(plt.Rectangle((max(q1, LO), y - 0.3), min(q3, HI) - max(q1, LO), 0.6, fc=col, alpha=0.22, ec=col, lw=0.8))
        jit = y + rng.uniform(-0.2, 0.2, len(g))
        inn = (g >= LO) & (g <= HI)
        ax.scatter(g[inn], jit[inn], s=7, color=col, edgecolor="black", linewidth=0.2, zorder=3, marker=mk)
        ax.scatter(np.clip(g[~inn], LO + 0.8, HI - 0.8), jit[~inn], s=9, facecolor="white", edgecolor=col, linewidth=0.7, zorder=3, marker=mk)
        ax.plot([np.clip(med, LO, HI)] * 2, [y - 0.34, y + 0.34], color="black", lw=1.6, zorder=4)
        share, big = (g > 0).mean(), int((g >= 20).sum())
        tx.text(0.08, y, f"{share:.0%}", va="center", ha="left", fontsize=FS + 0.5, fontweight="bold")
        tx.text(0.80, y, f"{big}", va="center", ha="center", fontsize=FS)
        stats.append(dict(metric=m, row=lab, better=share, n_ge20=big, n=len(g), median=med))
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(6, color="gray", lw=0.6, ls=":")
    ax.set_xlim(LO, HI)
    ax.set_xticks([-20, 0, 20, 40])
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("Relative gain (%)")
    ax.set_title(f"{NAME[m]} {ARR[lower]}", fontweight="bold")
    tx.text(-0.02, -1.1, "better", fontsize=FS - 2, color="0.3", va="center", ha="left")
    tx.text(0.86, -1.1, "≥20%", fontsize=FS - 2, color="0.3", va="center", ha="center")
    if k == 0:
        ax.set_yticks(list(yrow.values()), list(yrow))
    else:
        ax.tick_params(axis="y", labelleft=False, left=False)
ax_first.set_ylim(len(ROWS) + 0.6, -1.7)
ax_first.text(-0.03, -1.1, "vs classical baselines", transform=ax_first.get_yaxis_transform(), ha="right", va="center", fontsize=FS, fontweight="bold", color="0.25")
fig.savefig(os.path.join(OUT, "fig2_gains_v2.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
pd.DataFrame(stats).round(3).to_csv(os.path.join(OUT, "fig2_stats.csv"), index=False)

# ---------------- Fig. 2 alternative: strip of the 54 datasets (best TFM vs classical baselines)
G = {m: per_dataset_gain(BEST_TFM[m], BASE[m], lower) for m, lower in MET}
order = G["C-index"].sort_values(ascending=False).index.tolist()
fig, axes = plt.subplots(3, 1, figsize=(6.75, 2.3), sharex=True)
for ax, (m, lower) in zip(axes, MET):
    g = G[m].reindex(order)
    ax.imshow(np.clip(g.to_numpy(), -20, 20)[None, :], cmap="RdBu", vmin=-20, vmax=20, aspect="auto")
    ax.set_yticks([0], [f"{NAME[m]} {ARR[lower]}"])
    ax.set_xticks([])
    ok = g.dropna()
    ax.text(len(order) + 0.8, 0, f"better on {(ok > 0).mean():.0%}\ngain ≥20%: {(ok >= 20).sum()}", va="center", fontsize=FS - 0.5)
    for s in ax.spines.values():
        s.set_visible(False)
axes[-1].set_xlabel("54 datasets, sorted by the C-Index gain of the best TFM over the classical baselines", fontsize=FS)
sm = plt.cm.ScalarMappable(cmap="RdBu", norm=plt.Normalize(-20, 20))
cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.08, pad=0.25, aspect=40, shrink=0.5)
cb.set_label("Relative gain (%)", fontsize=FS - 0.5)
fig.savefig(os.path.join(OUT, "fig2_strip_v2.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------- Fig. 3: median gain by training observations
C = chars()
BINS = [("<200", C.n_train < 200), ("200–700", (C.n_train >= 200) & (C.n_train <= 700)), (">700", C.n_train > 700)]
MODELS = ["MITRA-CF", "MITRA-DH", "TabPFN-CF", "TabPFN-DH", "XGBoost-CF", "XGBoost-DH"]
fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.3), sharex=True)
for ax, (m, lower) in zip(axes, MET):
    for a in MODELS:
        g = per_dataset_gain(V[m][a], BASE[m], lower)
        med = [g[g.index.isin(C.index[mask])].median() for _, mask in BINS]
        ax.plot(range(3), med, "-" if a.endswith("CF") else "--", color=COL[a.replace("-DH", "-CF")], marker="o" if a.endswith("CF") else "^", lw=1.4, ms=4, label=a)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(3), [b for b, _ in BINS])
    ax.set_xlim(-0.3, 2.3)
    ax.grid(alpha=0.3)
    ax.set_title(f"{NAME[m]} {ARR[lower]}", fontweight="bold")
    ax.set_xlabel("Training observations")
axes[0].set_ylabel("Median gain (%) over\nclassical baselines")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.12), handlelength=2.2, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(OUT, "fig3_size_v2.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------- Fig. 4: CF vs DH / MC, shaded, no counts
KS = [5, 10, 15, 20]
rows = []
rng = np.random.default_rng(0)
for m, lower in MET:
    v = V[m]
    for b in BACK:
        for other in ("DH", "MC"):
            for K in KS:
                if f"{b}-{other}" not in v:
                    continue
                q = v.xs(K, level="K")[[f"{b}-CF", f"{b}-{other}"]].dropna()
                if len(q) < 10:
                    continue
                d = q.iloc[:, 0] - q.iloc[:, 1]  # CF - other, no sign flip
                boot = np.median(rng.choice(d.to_numpy(), (2000, len(d))), axis=1)
                rows.append(dict(metric=m, classifier=b, versus=other, K=K, median=d.median(), lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5),
                                 cf_better=int(((d < 0) if lower else (d > 0)).sum()), n=len(d)))
T = pd.DataFrame(rows)
fig, axes = plt.subplots(2, 3, figsize=(6.75, 3.9), sharex=True)
for r, other in enumerate(("DH", "MC")):
    for c, (m, lower) in enumerate(MET):
        ax = axes[r, c]
        t_all = T[(T.metric == m) & (T.versus == other)]
        lim = max(abs(t_all.lo.min()), abs(t_all.hi.max())) * 1.15
        up, down = (other, "CF") if lower else ("CF", other)  # IBS: lower is better, so CF wins below 0
        oc, otxt = ("#E8963A", "#9a5a10") if other == "DH" else ("#8E5CC4", "#5b3a86")  # DH: orange, MC: purple
        cu, cd = (oc, "#2F6FE0") if lower else ("#2F6FE0", oc)
        ax.axhspan(0, lim, color=cu, alpha=0.08, lw=0)
        ax.axhspan(-lim, 0, color=cd, alpha=0.08, lw=0)
        ax.text(0.03, 0.96, f"{up} better", transform=ax.transAxes, ha="left", va="top", fontsize=FS - 1, color="#1f4f9e" if up == "CF" else otxt)
        ax.text(0.03, 0.04, f"{down} better", transform=ax.transAxes, ha="left", va="bottom", fontsize=FS - 1, color="#1f4f9e" if down == "CF" else otxt)
        for i, b in enumerate(BACK):
            t = T[(T.metric == m) & (T.classifier == b) & (T.versus == other)]
            x = np.array([KS.index(k) for k in t.K]) + (i - 1) * 0.1
            ax.errorbar(x, t["median"], yerr=[t["median"] - t.lo, t.hi - t["median"]], color=BCOL[b], marker="o", ms=3.5, lw=1.3,
                        elinewidth=0.7, capsize=1.5, alpha=0.95, label=b)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_ylim(-lim, lim)
        ax.set_xticks(range(len(KS)), [str(k) for k in KS])
        ax.set_xlim(-0.4, len(KS) - 0.6)
        ax.grid(alpha=0.25)
        if r == 0:
            ax.set_title(f"{NAME[m]} {ARR[lower]}", fontweight="bold")
        if r == 1:
            ax.set_xlabel("K")
        if c == 0:
            ax.set_ylabel(f"CF − {other}")
h, lab = axes[0, 0].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(OUT, "fig4_formulations_v2.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# table of win counts (CF better / datasets) and medians
lines = [r"\begin{table}[t]", r"\centering",
         r"\caption{\dk{\textbf{CF vs.\ DH and MC per classifier} (static setting, KM grid). Number of datasets on which CF is better than the other "
         r"formulation of the same classifier (of 54; Integrated AUC: of the datasets in which it is defined). MITRA-MC and TabPFN-MC: $K \leq 10$ only.}}",
         r"\label{tab:formulation_wins}", r"\resizebox{\textwidth}{!}{", r"\begin{tabular}{ll cccc cccc cccc}", r"\toprule",
         r" & & \multicolumn{4}{c}{\textbf{C-Index}} & \multicolumn{4}{c}{\textbf{Integrated AUC}} & \multicolumn{4}{c}{\textbf{IBS}} \\",
         r"\cmidrule(r){3-6} \cmidrule(r){7-10} \cmidrule(r){11-14}", r"\textbf{Comparison} & \textbf{Classifier} & " + " & ".join([f"$K={k}$" for k in KS] * 3) + r" \\", r"\midrule"]
for other in ("DH", "MC"):
    for b in BACK:
        cells = []
        for m, _ in MET:
            for K in KS:
                t = T[(T.metric == m) & (T.classifier == b) & (T.versus == other) & (T.K == K)]
                cells.append(f"{int(t.cf_better.iloc[0])}" if len(t) else "--")
        lines.append(f"CF vs.\\ {other} & {b} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule" if other == "DH" else r"\bottomrule")
lines += [r"\end{tabular}", "}", r"\end{table}"]
open(os.path.join(OUT, "fig4_formulations_table.tex"), "w").write("\n".join(lines) + "\n")
print("ok")
