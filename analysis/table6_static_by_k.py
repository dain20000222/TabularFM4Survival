# Companion tables of Table 1 with MC (11-10), same data and conventions as make_table1_mc.py:
#   static_by_k.tex     every model of Table 1 at each K (mean test value); MITRA-MC / TabPFN-MC: K = 5, 10 only ("--" otherwise)
#   static_all_km.tex   every classification model with and without post-hoc calibration (+ the baselines), ranks as in Table 1
# Colouring as Table 1: best dark blue + bold, second best light blue, ties share the colour.
#   python make_by_k_and_all.py
import os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)  # outputs
L = pd.read_csv(os.environ.get("STATIC_LONG", os.path.join(HERE, "..", "results", "static_long.csv")))
CAL = " + calibration"
MET = [("ipcw_c_index", False), ("mean_auc", False), ("ibs", True)]
L = L[~(L.model.str.match(r"(MITRA|TabPFN)-MC") & (L.K > 10))]
REN = {"XGBoost-CF (W abl)": "XGBoost BCESurv", "XGBoost-CF (W abl)" + CAL: "XGBoost BCESurv" + CAL, "BCESurv": "MLP BCESurv"}
L = L.replace({"model": REN})
RAW = ["XGBoost-CF", "XGBoost-DH", "XGBoost BCESurv"]
D = L[~L.model.isin([m + CAL for m in RAW])]
have = {m[: -len(CAL)] for m in D.model if m.endswith(CAL)}
D = D[~D.model.isin(have)].copy(); D["model"] = D.model.str.replace(CAL, "", regex=False)
SECS = [("Cumulative-Failure Models", ["MITRA-CF", "TabPFN-CF", "XGBoost-CF"]), ("Discrete-time Hazard Models", ["MITRA-DH", "TabPFN-DH", "XGBoost-DH"]),
        ("Multi-class Models", ["MITRA-MC", "TabPFN-MC", "XGBoost-MC"]),
        ("Other Discrete-time Classification Models", ["Linear MTLR", "Neural MTLR", "MLP BCESurv", "XGBoost BCESurv"]),
        ("Classical \\& Neural Survival Baselines", ["CoxPH", "RSF", "DeepHit", "DeepSurv"])]
PART = ["MITRA-MC", "TabPFN-MC"]
LEG = "Best and second best (ties share the colour) in \\colorbox{blue!25}{dark} and \\colorbox{blue!10}{light} blue."

def colour(vals, lower, dec):
    v = {k: round(x, dec) for k, x in vals.items() if not pd.isna(x)}
    lv = sorted(set(v.values()), reverse=not lower)
    return {k: ("b" if x == lv[0] else ("s" if len(lv) > 1 and x == lv[1] else "")) for k, x in v.items()}
def fmt(txt, c):
    if c == "b":
        h, sep, t = txt.partition("$"); return "\\cellcolor{blue!25}\\textbf{" + h + "}" + sep + t
    return ("\\cellcolor{blue!10}" + txt) if c == "s" else txt

# ---------------- per K
ORDER = [m for _, ms in SECS for m in ms]; FULL = [m for m in ORDER if m not in PART]; KS = [5, 10, 15, 20]
M, N = {}, {}
for m, lower in MET:
    for K in KS:
        p = D[(D.K == K) & D.model.isin(ORDER)].pivot_table(index="dataset", columns="model", values=m).reindex(columns=ORDER)
        p = p[p[FULL].notna().all(axis=1)]
        M[(m, K)] = p.mean(); N[(m, K)] = len(p)
cols = [(m, K) for m, _ in MET for K in KS]
C = {c: colour(M[c].to_dict(), c[0] == "ibs", 3) for c in cols}
lines = ["\\begin{table}[htbp]", "\\centering",
         "\\caption{\\dk{\\textbf{Static setting for each number of intervals $K$} (KM grid). Mean test value over the datasets in which every model has the metric ("
         + ", ".join(f"$K={K}$: ${N[('ipcw_c_index', K)]}$" for K in KS) + " datasets; Integrated AUC: " + ", ".join(f"${N[('mean_auc', K)]}$" for K in KS)
         + "). Models and calibration as in Table~\\ref{tab:performance_comparison_km}, which pools the dataset--$K$ combinations. MITRA-MC and TabPFN-MC support at most $10$ classes. " + LEG + "}}",
         "\\label{tab:static_by_k}", "\\resizebox{\\textwidth}{!}{", "\\begin{tabular}{l cccc cccc cccc}", "\\toprule",
         " & \\multicolumn{4}{c}{\\textbf{C-Index} $\\uparrow$} & \\multicolumn{4}{c}{\\textbf{Integrated AUC} $\\uparrow$} & \\multicolumn{4}{c}{\\textbf{IBS} $\\downarrow$} \\\\",
         "\\cmidrule(r){2-5} \\cmidrule(r){6-9} \\cmidrule(r){10-13}", "\\textbf{Model} & " + " & ".join([f"$K={K}$" for K in KS] * 3) + " \\\\"]
for sec, ms in SECS:
    lines += ["\\midrule", f"\\multicolumn{{13}}{{l}}{{\\textit{{{sec}}}}} \\\\"]
    for mod in ms:
        lines.append(mod + " & " + " & ".join("--" if pd.isna(M[c][mod]) else fmt(f"{M[c][mod]:.3f}", C[c][mod]) for c in cols) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
open(os.path.join(OUT, "static_by_k.tex"), "w").write("\n".join(lines) + "\n")
print("by K:", {K: N[("ipcw_c_index", K)] for K in KS})

# ---------------- every model with and without calibration
A = L.copy()
pair = lambda ms: sum(([m, m + CAL] for m in ms), [])  # noqa: E731
SECA = [(SECS[0][0], pair(SECS[0][1])), (SECS[1][0], pair(SECS[1][1])), (SECS[2][0], pair(SECS[2][1])),
        (SECS[3][0], ["Linear MTLR", "Neural MTLR", "MLP BCESurv"] + pair(["XGBoost BCESurv"])), SECS[4]]
ORDA = [m for _, ms in SECA for m in ms]; PARTA = pair(PART); FULLA = [m for m in ORDA if m not in PARTA]
X = A[A.model.isin(ORDA)].dropna(subset=[m for m, _ in MET], how="all")
n = X[X.model.isin(FULLA)].groupby(["dataset", "K"]).model.nunique(); keep = n[n == len(FULLA)].index
X = X.set_index(["dataset", "K"]).loc[keep].reset_index()
def cse(s):
    s = s.dropna(); Nn, ds = len(s), s.index.get_level_values(0); Dn = ds.nunique(); u = (s - s.mean()).groupby(ds).sum()
    return np.sqrt(Dn / (Dn - 1) * (u ** 2).sum()) / Nn
T, ranks = {}, []
for m, lower in MET:
    p = X.pivot_table(index=["dataset", "K"], columns="model", values=m)[ORDA]; p = p[p[FULLA].notna().all(axis=1)]
    r = p.rank(axis=1, ascending=lower); ranks.append(r)
    T[m] = pd.DataFrame({"value": p.mean(), "value_se": p.apply(cse), "rank": r.mean(), "rank_se": r.apply(cse)})
avg = pd.concat(ranks).groupby(level=[0, 1]).mean(); T["avg"] = pd.DataFrame({"rank": avg.mean(), "rank_se": avg.apply(cse)})
cc = {(m, "value"): colour(T[m]["value"].to_dict(), lo, 3) for m, lo in MET}
cc.update({(m, "rank"): colour(T[m]["rank"].to_dict(), True, 1) for m, _ in MET}); cc[("avg", "rank")] = colour(T["avg"]["rank"].to_dict(), True, 1)
cap = ("\\caption{\\dk{\\bo{Static setting, every classification model with and without post-hoc calibration} (KM grid). Means over the $" + str(X.dataset.nunique())
       + "$ datasets and $K \\in \\{5,10,15,20\\}$; standard errors clustered by dataset. Calibration does not change the C-Index and Integrated AUC. "
       "Table~\\ref{tab:performance_comparison_km} uses the calibrated MITRA, TabPFN, and XGBoost-MC models and the uncalibrated XGBoost-CF, XGBoost-DH, and XGBoost BCESurv models. "
       "$^\\dagger$MITRA-MC and TabPFN-MC support at most $10$ classes, so their results are over $K \\in \\{5, 10\\}$; ranks are among the rows available at each $K$. " + LEG + "}}")
lines = ["\\begin{table}[htbp]", "\\centering", cap, "\\label{tab:performance_comparison_km_all}", "\\resizebox{\\textwidth}{!}{", "\\begin{tabular}{lccccccc}", "\\toprule",
         "\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{2}{c}{\\textbf{C-Index}} & \\multicolumn{2}{c}{\\textbf{Integrated AUC}} & \\multicolumn{2}{c}{\\textbf{Integrated Brier Score}} & \\multirow{2}{*}{\\textbf{Avg. Rank $\\downarrow$}} \\\\",
         "\\cmidrule(r){2-3} \\cmidrule(r){4-5} \\cmidrule(r){6-7}", " & Value $\\uparrow$ & Rank $\\downarrow$ & Value $\\uparrow$ & Rank $\\downarrow$ & Value $\\downarrow$ & Rank $\\downarrow$ & \\\\"]
for sec, ms in SECA:
    lines += ["\\midrule", f"\\multicolumn{{8}}{{l}}{{\\textit{{{sec}}}}} \\\\"]
    for mod in ms:
        cs = []
        for m, _ in MET:
            t = T[m].loc[mod]
            cs.append(fmt(f"{t['value']:.3f}$_{{\\pm {t['value_se']:.3f}}}$", cc[(m, "value")][mod])); cs.append(fmt(f"{t['rank']:.1f}$_{{\\pm {t['rank_se']:.1f}}}$", cc[(m, "rank")][mod]))
        a = T["avg"].loc[mod]; cs.append(fmt(f"{a['rank']:.1f}$_{{\\pm {a['rank_se']:.1f}}}$", cc[("avg", "rank")][mod]))
        base = mod.replace(CAL, "")
        lines.append(f"\\textbf{{{mod}}}" + ("$^\\dagger$" if base in PART else "") + " & " + " & ".join(cs) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
open(os.path.join(OUT, "static_all_km.tex"), "w").write("\n".join(lines) + "\n")
print("all:", X.dataset.nunique(), len(keep), len(ORDA)); print(pd.concat(T, axis=1).loc[ORDA].round(3)[[("ipcw_c_index","value"),("ibs", "value"), ("avg", "rank")]].to_string())
