# Static Fig. 2 split in two:
#   fig2a_heatmap.png   TFM models vs the classical baselines (validation choice among CoxPH, RSF, DeepHit, DeepSurv; no XGBoost),
#                       median / 70th / 90th percentile over the datasets of the per-dataset relative gain (%), per metric and K
#   fig2b_better_on.tex share of datasets (%) where the best TFM (validation choice between MITRA and TabPFN) is better, with the
#                       per-dataset gain averaged over K first: best TFM-CF / TFM-DH vs the classical baselines and vs XGBoost
#   fig2b_better_on.png preview of the same table
#   IBS sign flipped (positive = the model is better); CF / DH models with post-hoc calibration for the IBS (as Table 1).
#   python static_fig2_split.py
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from common import OUTV, HERE, MET, POOL, best, rel_gain, static_long

FS = 7.5
plt.rcParams.update({"font.size": FS, "axes.titlesize": 9.5, "xtick.labelsize": FS, "ytick.labelsize": FS})
OUT = OUTV
S = static_long()
U = ["dataset", "K"]
KS = [5, 10, 15, 20]
NAME = {"C-index": "C-Index", "Integrated AUC": "iAUC", "IBS": "IBS"}
V = {m: S.dropna(subset=[m]).pivot_table(index=U, columns="model", values=m) for m, _ in MET}
BASE = {m: best(S, POOL["static"], m, lower, U)[m] for m, lower in MET}
BEST = {(m, f): best(S, [f"MITRA-{f}", f"TabPFN-{f}"], m, lower, U)[m] for m, lower in MET for f in ("CF", "DH")}


def gain(a, r, lower):
    j = pd.concat([a.rename("a"), r.rename("r")], axis=1).dropna()
    return rel_gain(j.a, j.r, lower)


# ---------------- Fig. 2a: heatmap, TFMs vs classical baselines
STATS = [("Median", lambda g: g.median(), (-4, 4)), ("Top 30%", lambda g: g.quantile(0.7), (-10, 10)),
         ("Top 10%", lambda g: g.quantile(0.9), (-20, 20))]
ROWS = ["MITRA-CF", "MITRA-DH", "TabPFN-CF", "TabPFN-DH"]
fig, axes = plt.subplots(len(STATS), 3, figsize=(6.75, 1.0 + 0.32 * len(ROWS) * len(STATS)), sharex=True,
                         gridspec_kw=dict(hspace=0.08, wspace=0.04))
summ = []
for c, (m, lower) in enumerate(MET):
    G = {a: gain(V[m][a], BASE[m], lower) for a in ROWS}
    for r, (sname, f, (vmin, vmax)) in enumerate(STATS):
        piv = pd.DataFrame({K: {a: f(G[a].xs(K, level="K")) for a in ROWS} for K in KS}).loc[ROWS].round(1) + 0.0
        summ += [dict(metric=m, stat=sname, row=a, **{f"K={K}": piv.loc[a, K] for K in KS}) for a in ROWS]
        ax = axes[r, c]
        sns.heatmap(piv, annot=True, fmt=".1f", cmap="RdYlGn", center=0, vmin=vmin, vmax=vmax, ax=ax, cbar=(c == 2),
                    linewidths=0.6, linecolor="white", annot_kws=dict(fontsize=FS - 0.5, fontweight="bold"),
                    cbar_kws=dict(pad=0.02, aspect=8))
        if c == 2:
            ax.collections[0].colorbar.ax.tick_params(labelsize=FS - 1)
        ax.set_ylabel(sname if c == 0 else "", fontsize=FS + 0.5, fontweight="bold")
        if c:
            ax.set_yticks([])
        else:
            ax.tick_params(axis="y", rotation=0)
        ax.set_xlabel("Number of intervals $K$" if r == len(STATS) - 1 else "", fontsize=FS)
        if r == 0:
            ax.set_title(NAME[m], fontweight="bold")
fig.savefig(os.path.join(OUT, "fig2a_heatmap.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
pd.DataFrame(summ).to_csv(os.path.join(OUT, "fig2a_heatmap_summary.csv"), index=False)

# ---------------- Fig. 2b: better-on table, gain averaged over K per dataset
CMP = [("Best TFM-CF", "Classical baselines", lambda m: BEST[(m, "CF")], lambda m: BASE[m]),
       ("Best TFM-DH", "Classical baselines", lambda m: BEST[(m, "DH")], lambda m: BASE[m]),
       ("Best TFM-CF", "XGBoost-CF", lambda m: BEST[(m, "CF")], lambda m: V[m]["XGBoost-CF"]),
       ("Best TFM-DH", "XGBoost-DH", lambda m: BEST[(m, "DH")], lambda m: V[m]["XGBoost-DH"])]
tab = []
for lab, ref, fa, fr in CMP:
    row = dict(model=lab, reference=ref)
    for m, lower in MET:
        g = gain(fa(m), fr(m), lower).groupby(level="dataset").mean()
        row[NAME[m]] = 100 * (g > 0).mean()
        row[NAME[m] + "_n"] = f"{int((g > 0).sum())}/{len(g)}"
        row[NAME[m] + "_med"] = g.median()
    tab.append(row)
T = pd.DataFrame(tab)
T.round(2).to_csv(os.path.join(OUT, "fig2b_better_on.csv"), index=False)
print(T.to_string())

CAP = (r"\caption{\dk{Share of the static datasets (\%) on which the best TFM (validation choice between MITRA and TabPFN) is "
       r"better than the reference, with the standard error in parentheses. The relative gain is averaged over $K \in \{5, 10, 15, 20\}$ "
       r"for each dataset. Classical baselines: validation choice among CoxPH, RSF, DeepHit and DeepSurv. The Integrated AUC is "
       r"undefined on \texttt{wpbc} and is computed over the other datasets.}}")
LAB = {("Best TFM-CF", "Classical baselines"): "Best TFM-CF over classical baselines",
       ("Best TFM-DH", "Classical baselines"): "Best TFM-DH over classical baselines",
       ("Best TFM-CF", "XGBoost-CF"): "Best TFM-CF over XGBoost-CF", ("Best TFM-DH", "XGBoost-DH"): "Best TFM-DH over XGBoost-DH"}
L = [r"\begin{table}[t]", r"\centering", CAP, r"\label{tab:better_on}", r"\begin{tabular}{lccc}", r"\toprule",
     r"Comparison across the 54 static datasets & C-Index & Integrated AUC & IBS \\", r"\midrule"]
for i, r in T.iterrows():
    if i == 2:
        L.append(r"\midrule")
    cells = []
    for k in ("C-Index", "iAUC", "IBS"):
        n = int(r[k + "_n"].split("/")[1])
        se = 100 * (r[k] / 100 * (1 - r[k] / 100) / n) ** 0.5
        cells.append(f"{r[k]:.0f} ({se:.0f})")
    L.append(LAB[(r.model, r.reference)] + " & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "fig2b_better_on.tex"), "w").write("\n".join(L) + "\n")

fig, ax = plt.subplots(figsize=(4.6, 1.3))
ax.axis("off")
cells = [[r.model, r.reference] + [f"{r[k]:.0f}% ({r[k + '_n']})" for k in ("C-Index", "iAUC", "IBS")] for _, r in T.iterrows()]
t = ax.table(cellText=cells, colLabels=["Model", "Reference", "C-Index", "iAUC", "IBS"], loc="center", cellLoc="center")
t.auto_set_font_size(False)
t.set_fontsize(FS)
t.auto_set_column_width(list(range(5)))
t.scale(1, 1.4)
fig.savefig(os.path.join(OUT, "fig2b_better_on.png"), dpi=300, bbox_inches="tight")
print("ok")
