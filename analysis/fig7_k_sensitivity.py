# Figure 7 (Appendix D.2): static performance for each number of intervals K, by model group (KM grid, 54 datasets).
# Per dataset and K, the model of each group is the one with the best validation C-Index; mean over the datasets in
# which every plotted model has a result, +- one standard error. MITRA / TabPFN models with post-hoc calibration for the
# IBS, XGBoost-CF / XGBoost-DH without (as in Table 1).
#   python fig7_k_sensitivity.py -> out/k_sensitivity_group.png, out/k_sensitivity_tfm.png, out/k_sensitivity.csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
L = pd.read_csv(os.environ.get("STATIC_LONG", os.path.join(HERE, "..", "results", "static_long.csv")))
KK = [5, 10, 15, 20]
MET = [("ipcw_c_index", "C-index", False), ("mean_auc", "Integrated AUC", False), ("ibs", "Integrated Brier Score", True)]


def cells(D, models, Ks=(5, 10, 15, 20)):
    # a dataset x K combination is kept when every model has a result for it
    X = D[D.model.isin(models) & D.K.isin(Ks)].dropna(subset=[m for m, _, _ in MET], how="all")
    n = X.groupby(["dataset", "K"]).model.nunique()
    keep = n[n == len(models)].index
    return X.set_index(["dataset", "K"]).loc[keep].reset_index()


# names as in static_long.csv: "Linear MTLR" = MTLR and "BCESurv" = MLP BCESurv in the paper
GROUPS = {"CF (ours)": ["MITRA-CF + calibration", "TabPFN-CF + calibration", "XGBoost-CF"],
          "DH": ["MITRA-DH + calibration", "TabPFN-DH + calibration", "XGBoost-DH"],
          "Classical": ["CoxPH", "RSF"], "Deep learning": ["DeepSurv", "DeepHit"],
          "Discrete-time baselines": ["Linear MTLR", "Neural MTLR", "BCESurv"]}
GSTYLE = {"CF (ours)": ("#2F6FE0", "o", "-"), "DH": ("#2E8B57", "s", "--"), "Classical": ("#E08A2C", "o", "-"),
          "Deep learning": ("#C62835", "o", "-"), "Discrete-time baselines": ("#7B68EE", "D", "-")}
TFM = [("MITRA-CF + calibration", "MITRA-CF", "#C62835", "o", "-"), ("TabPFN-CF + calibration", "TabPFN-CF", "#EE8A2C", "o", "-"),
       ("XGBoost-CF", "XGBoost-CF", "#2F6FE0", "o", "-"), ("MITRA-DH + calibration", "MITRA-DH", "#8B1E24", "s", "--"),
       ("TabPFN-DH + calibration", "TabPFN-DH", "#8B5A2B", "s", "--"), ("XGBoost-DH", "XGBoost-DH", "#1F3F8F", "s", "--")]
METRICS = [("ipcw_c_index", r"C-Index $\uparrow$"), ("mean_auc", r"Integrated AUC $\uparrow$"), ("ibs", r"Integrated Brier Score $\downarrow$")]
se = lambda v: v.std(ddof=1) / np.sqrt(len(v))  # noqa: E731


def draw(series, out):  # series: list of (label, color, marker, ls, frame with K, metric, mean, se)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.6))
    handles = []
    for ax, (metric, title) in zip(axes, METRICS):
        for label, color, marker, ls, d in series:
            d = d[d.metric == metric].sort_values("K")
            main = ls == "-"
            (h,) = ax.plot(d.K, d["mean"], ls=ls, color=color, marker=marker, label=label, lw=3.2 if main else 2.4, ms=9,
                           markeredgecolor="white", markeredgewidth=1.2, zorder=3)
            ax.fill_between(d.K, d["mean"] - d["se"], d["mean"] + d["se"], color=color, alpha=0.13 if main else 0.08, lw=0, zorder=1)
            if ax is axes[0]:
                handles.append(h)
        ax.set_title(title, fontsize=17, fontweight="bold")
        ax.set_xlabel("Number of Discrete Intervals, $K$", fontsize=14, fontweight="bold")
        ax.set_xticks(KK)
        ax.grid(alpha=0.4)
        for s in ax.spines.values():
            s.set_linewidth(1.6)
    axes[0].set_ylabel("Mean Value", fontsize=15, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=True, fontsize=12.5, bbox_to_anchor=(0.5, 1.07), handlelength=3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


models = sorted({m for g in GROUPS.values() for m in g} | {m for m, *_ in TFM})
X = cells(L, models)
COLS = ["ipcw_c_index", "mean_auc", "ibs"]
out = []
for g, ms in GROUPS.items():
    x = X[X.model.isin(ms)].assign(o=lambda v: v.model.map(ms.index)).sort_values(["dataset", "K", "o"])
    x = x.loc[x.groupby(["dataset", "K"]).val_ipcw_c_index.idxmax()]
    for m in COLS:
        s = x.groupby("K")[m].agg(["mean", se]).reset_index()
        s.columns = ["K", "mean", "se"]
        out.append(s.assign(figure="group", series=g, metric=m))
for model, *_ in TFM:
    for m in COLS:
        s = X[X.model == model].groupby("K")[m].agg(["mean", se]).reset_index()
        s.columns = ["K", "mean", "se"]
        out.append(s.assign(figure="tfm", series=model, metric=m))
O = pd.concat(out, ignore_index=True)
O.to_csv(os.path.join(OUT, "k_sensitivity.csv"), index=False)
draw([(g, *GSTYLE[g], O[(O.figure == "group") & (O.series == g)]) for g in GROUPS], os.path.join(OUT, "k_sensitivity_group.png"))
draw([(lab, c, mk, ls, O[(O.figure == "tfm") & (O.series == m)]) for m, lab, c, mk, ls in TFM], os.path.join(OUT, "k_sensitivity_tfm.png"))
print(X.dataset.nunique(), X.groupby(["dataset", "K"]).ngroups)
print(O[O.metric == "ibs"].pivot_table(index=["figure", "series"], columns="K", values="mean").round(4).to_string())
