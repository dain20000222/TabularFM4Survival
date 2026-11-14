# Dynamic table for the main text: CF and DH models and the dynamic survival baselines only
# (no MC, no landmark MTLR / BCESurv): 11 rows, ranks among these 11. Also a check: C-index below 0.5.
#   python dynamic_table_main.py [long csv, default ../final_11_07/dynamic_long_m5.csv] -> dynamic_table_main.tex/.csv, cindex_below_half.md
import os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)  # outputs
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "dynamic_long.csv")
L = pd.read_csv(SRC)
CAL = " + calibration"
have = {m[: -len(CAL)] for m in L.model if m.endswith(CAL)}
D = L[~L.model.isin(have)].copy(); D["model"] = D.model.str.replace(CAL, "", regex=False)   # calibrated rows: IBS calibrated, ranking metrics of the raw model
SEC = [("Cumulative-Failure Models", ["MITRA-CF", "TabPFN-CF", "XGBoost-CF"]), ("Discrete-time Hazard Models", ["MITRA-DH", "TabPFN-DH", "XGBoost-DH"]),
       ("Dynamic Survival Baselines", ["Landmark CoxPH", "Landmark RSF", "Dynamic DeepHit", "Joint Model", "TCSA"])]
ROWS = [m for _, g in SEC for m in g]
D = D[D.model.isin(ROWS)]
DS = sorted(D.dataset.unique()); n = len(DS)
assert n == 6 and D.groupby("model").dataset.nunique().eq(6).all(), D.groupby("model").dataset.nunique()
MET = [("C-index", False), ("Integrated AUC", False), ("IBS", True)]
se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
T, R = {}, {}
for m, lower in MET:
    V = D.pivot(index="model", columns="dataset", values=m).loc[ROWS]
    Rk = V.rank(ascending=lower)
    T[m] = (V.mean(1), V.apply(se, axis=1), Rk.mean(1), Rk.apply(se, axis=1)); R[m] = Rk
avg = pd.concat([R[m] for m, _ in MET]).groupby(level=0).mean()      # per dataset: mean of the three ranks
AR = (avg.mean(1), avg.apply(se, axis=1))

def marks(s, lower, dec):
    r = s.round(dec); u = np.sort(r.unique()); u = u if lower else u[::-1]
    return {k: (1 if v == u[0] else 2 if len(u) > 1 and v == u[1] else 0) for k, v in r.items()}
def cell(v, s, mk, dec):
    a, b = f"{v:.{dec}f}", f"$_{{\\pm {s:.{dec}f}}}$"
    return ("\\cellcolor{blue!25}\\textbf{" + a + "}" if mk == 1 else "\\cellcolor{blue!10}" + a if mk == 2 else a) + b

mk = {}
for m, lower in MET:
    mk[m, "v"] = marks(T[m][0], lower, 3); mk[m, "r"] = marks(T[m][2], True, 1)
mk["avg"] = marks(AR[0], True, 1)
tex = ["\\begin{table}[t]\n\\centering\n\\caption{\\textbf{Dynamic setting.} We show results on average across the $6$ longitudinal \\emph{SurvSet} datasets, each averaged over its prediction times. "
       "Ranks are among the $11$ models shown. Best and second best (ties share the colour) in \\colorbox{blue!25}{dark} and \\colorbox{blue!10}{light} blue. The error bars denote one standard error over the datasets.}\n"
       "\\label{tab:performance_comparison_dynamic}\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{lccccccc}\n\\toprule\n"
       "\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{2}{c}{\\textbf{C-Index}} & \\multicolumn{2}{c}{\\textbf{Integrated AUC}} & \\multicolumn{2}{c}{\\textbf{Integrated Brier Score}} & \\multirow{2}{*}{\\textbf{Avg. Rank $\\downarrow$}} \\\\\n"
       "\\cmidrule(r){2-3} \\cmidrule(r){4-5} \\cmidrule(r){6-7}\n & Value $\\uparrow$ & Rank $\\downarrow$ & Value $\\uparrow$ & Rank $\\downarrow$ & Value $\\downarrow$ & Rank $\\downarrow$ & \\\\\n"]
out = []
for i, (sec, mods) in enumerate(SEC):
    tex.append("\\midrule\n" + f"\\multicolumn{{8}}{{l}}{{\\textit{{{sec}}}}} \\\\\n")
    for mod in mods:
        cs = []
        for m, lower in MET:
            v, s, r, rs = (x[mod] for x in T[m]); cs += [cell(v, s, mk[m, "v"][mod], 3), cell(r, rs, mk[m, "r"][mod], 1)]
        cs.append(cell(AR[0][mod], AR[1][mod], mk["avg"][mod], 1))
        tex.append(f"\\textbf{{{mod}}} & " + " & ".join(cs) + " \\\\\n")
        out.append(dict(model=mod, **{f"{m} {k}": round(float(x[mod]), 4) for m, _ in MET for k, x in zip(("value", "se", "rank", "rank_se"), T[m])}, avg_rank=round(float(AR[0][mod]), 2), avg_rank_se=round(float(AR[1][mod]), 2)))
tex.append("\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n")
open(os.path.join(OUT, "dynamic_table_main.tex"), "w").write("".join(tex))
O = pd.DataFrame(out); O.to_csv(os.path.join(OUT, "dynamic_table_main.csv"), index=False)
print(O[["model", "C-index value", "Integrated AUC value", "IBS value", "avg_rank"]].to_string(index=False))

# ---- C-index below 0.5 on each dataset
V = D.pivot(index="model", columns="dataset", values="C-index").loc[ROWS]
Vv = D.pivot(index="model", columns="dataset", values="val_C-index").loc[ROWS]
md = ["# Test C-index below 0.5 in the dynamic setting (11 models, 6 datasets)\n", "| Model | Dataset | Test C-index | Validation C-index |", "|---|---|---|---|"]
for mod in ROWS:
    for d in DS:
        if V.loc[mod, d] < 0.5: md.append(f"| {mod} | {d} | {V.loc[mod, d]:.3f} | {Vv.loc[mod, d]:.3f} |")
md += ["", "Test C-index of every model on every dataset:\n", "```", V.round(3).to_string(), "```"]
open(os.path.join(OUT, "cindex_below_half.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md[:8]))
