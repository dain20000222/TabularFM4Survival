# Table 1 with the multi-class (MC) models, from ../results/static_long.csv.
#   Rows: CF, DH, MC (MITRA / TabPFN / XGBoost), MTLR / BCESurv group, classical and neural baselines (17 rows).
#   Calibration (IBS only): MITRA and TabPFN models (CF, DH, MC) and XGBoost-MC; XGBoost-CF, XGBoost-DH and XGBoost BCESurv: none.
#   MITRA-MC and TabPFN-MC exist only for K = 5, 10 (at most 10 classes). Their values and ranks are over K = 5, 10; all other rows
#   over K = 5, 10, 15, 20. Rank: per dataset, K and metric, among the models available at that K (17 at K <= 10, 15 at K > 10).
#   Cells: the dataset-K combinations in which every all-K model has results (as Table 1 v2: 214).
#   Group names and caption as in the paper.
#   python make_table1_mc.py -> table1_static_mc.tex (gitignored), table1_static_mc.csv, table1_static_mc.png
import os, subprocess, shutil
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)  # outputs
L = pd.read_csv(os.environ.get("STATIC_LONG", os.path.join(HERE, "..", "results", "static_long.csv")))
CAL = " + calibration"
MET = [("ipcw_c_index", False), ("mean_auc", False), ("ibs", True)]
RAW = ["XGBoost-CF", "XGBoost-DH", "XGBoost-CF (W abl)"]                      # static XGBoost CF / DH: not calibrated
D = L[~L.model.isin([m + CAL for m in RAW])]
have = {m[: -len(CAL)] for m in D.model if m.endswith(CAL)}
D = D[~D.model.isin(have)].copy()
D["model"] = D.model.str.replace(CAL, "", regex=False).replace({"XGBoost-CF (W abl)": "XGBoost BCESurv", "BCESurv": "MLP BCESurv"})
D = D[~(D.model.isin(["MITRA-MC", "TabPFN-MC"]) & (D.K > 10))]                # a few K > 10 rows exist where the grid has <= 10 points
SECS = [("Cumulative-Failure Models", ["MITRA-CF", "TabPFN-CF", "XGBoost-CF"]),
        ("Discrete-time Hazard Models", ["MITRA-DH", "TabPFN-DH", "XGBoost-DH"]),
        ("Multi-class Models", ["MITRA-MC", "TabPFN-MC", "XGBoost-MC"]),
        ("Other Discrete-time Classification Models", ["Linear MTLR", "Neural MTLR", "MLP BCESurv", "XGBoost BCESurv"]),
        ("Classical \\& Neural Survival Baselines", ["CoxPH", "RSF", "DeepHit", "DeepSurv"])]
ORDER = [m for _, ms in SECS for m in ms]
PART = ["MITRA-MC", "TabPFN-MC"]
FULL = [m for m in ORDER if m not in PART]
X = D[D.model.isin(ORDER)].dropna(subset=[m for m, _ in MET], how="all")
n = X[X.model.isin(FULL)].groupby(["dataset", "K"]).model.nunique()
keep = n[n == len(FULL)].index
X = X.set_index(["dataset", "K"]).loc[keep].reset_index()

def cse(s):
    s = s.dropna(); N, ds = len(s), s.index.get_level_values(0); Dn = ds.nunique()
    u = (s - s.mean()).groupby(ds).sum()
    return np.sqrt(Dn / (Dn - 1) * (u ** 2).sum()) / N

T, ranks = {}, []
for m, lower in MET:
    p = X.pivot_table(index=["dataset", "K"], columns="model", values=m)[ORDER]
    p = p[p[FULL].notna().all(axis=1)]                # metric defined for every all-K model
    r = p.rank(axis=1, ascending=lower)               # among the models available in the cell
    ranks.append(r)
    T[m] = pd.DataFrame({"value": p.mean(), "value_se": p.apply(cse), "rank": r.mean(), "rank_se": r.apply(cse), "n": p.notna().sum()})
avg = pd.concat(ranks).groupby(level=[0, 1]).mean()
T["avg"] = pd.DataFrame({"rank": avg.mean(), "rank_se": avg.apply(cse)})
out = pd.concat(T, axis=1).loc[ORDER]
out.round(4).to_csv(os.path.join(OUT, "table1_static_mc.csv"))
print(X.dataset.nunique(), len(keep)); print(out.round(3).to_string())

def top2(series, ascending, dec):
    v = series.reindex(ORDER).map(lambda x: float(f"{x:.{dec}f}"))
    return [v.index[v == lv].tolist() for lv in sorted(v.unique(), reverse=not ascending)[:2]]
best = {(m, "value"): top2(T[m]["value"], lo, 3) for m, lo in MET}
best.update({(m, "rank"): top2(T[m]["rank"], True, 1) for m, _ in MET}); best[("avg", "rank")] = top2(T["avg"]["rank"], True, 1)
def cell(key, model, txt):
    b = best[key]
    if model in b[0]:
        h, t = txt.split("$", 1); return "\\cellcolor{blue!25}\\textbf{" + h + "}$" + t
    return ("\\cellcolor{blue!10}" + txt) if model in b[1] else txt
CAP = ("\\caption{\\bo{Static Setting.} We show results on average across $54$ \\textit{SurvSet} datasets and across discretization granularities $K \\in \\{5,10,15,20\\}$ "
       "(we form discretization bins by taking the quantiles of a Kaplan-Meier survival curve fit on the training set). Best and second best (ties share the colour) in "
       "\\colorbox{blue!25}{dark} and \\colorbox{blue!10}{light} blue. The error bars denote one standard error. }")
lines = ["\\begin{table}[H]", "\\centering", CAP, "\\label{tab:performance_comparison_km}", "\\resizebox{\\textwidth}{!}{", "\\begin{tabular}{lccccccc}", "\\toprule",
         "\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{2}{c}{\\textbf{C-Index}} & \\multicolumn{2}{c}{\\textbf{Integrated AUC}} & \\multicolumn{2}{c}{\\textbf{Integrated Brier Score}} & \\multirow{2}{*}{\\textbf{Avg. Rank $\\downarrow$}} \\\\",
         "\\cmidrule(r){2-3} \\cmidrule(r){4-5} \\cmidrule(r){6-7}", " & Value $\\uparrow$ & Rank $\\downarrow$ & Value $\\uparrow$ & Rank $\\downarrow$ & Value $\\downarrow$ & Rank $\\downarrow$ & \\\\"]
for sec, ms in SECS:
    lines += ["\\midrule", f"\\multicolumn{{8}}{{l}}{{\\textit{{{sec}}}}} \\\\"]
    for mod in ms:
        cs = []
        for m, _ in MET:
            t = T[m].loc[mod]
            cs.append(cell((m, "value"), mod, f"{t['value']:.3f}$_{{\\pm {t['value_se']:.3f}}}$"))
            cs.append(cell((m, "rank"), mod, f"{t['rank']:.1f}$_{{\\pm {t['rank_se']:.1f}}}$"))
        a = T["avg"].loc[mod]; cs.append(cell(("avg", "rank"), mod, f"{a['rank']:.1f}$_{{\\pm {a['rank_se']:.1f}}}$"))
        name = f"\\textbf{{{mod}}}" + ("$^\\dagger$" if mod in PART else "")
        lines.append(name + " & " + " & ".join(cs) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "}",
          "\\\\[1mm] {\\footnotesize \\dk{$^\\dagger$MITRA-MC and TabPFN-MC support at most $10$ classes, so their results are over $K \\in \\{5, 10\\}$. Ranks are among the models available at each $K$.}}",
          "\\end{table}"]
tex = "\n".join(lines) + "\n"
open(os.path.join(OUT, "table1_static_mc.tex"), "w").write(tex)
if shutil.which("pdflatex"):
    tmp = os.path.join(OUT, "_render"); os.makedirs(tmp, exist_ok=True)
    doc = ("\\documentclass{article}\n\\usepackage[margin=0.5cm,paperwidth=26cm,paperheight=26cm]{geometry}\n\\usepackage{booktabs,multirow,graphicx,amsmath,float}\n\\usepackage[table]{xcolor}\n"
           "\\usepackage{caption}\n\\captionsetup{width=24cm}\n\\pagestyle{empty}\n\\providecommand{\\dk}[1]{#1}\\providecommand{\\bo}[1]{\\textbf{#1}}\n\\begin{document}\n" + tex.replace("\\caption{", "\\caption*{") + "\\end{document}\n")
    open(os.path.join(tmp, "t.tex"), "w").write(doc)
    if subprocess.run(["pdflatex", "-interaction=nonstopmode", "t.tex"], cwd=tmp, stdout=subprocess.DEVNULL).returncode == 0:
        subprocess.run(["pdftoppm", "-png", "-r", "170", "-singlefile", "t.pdf", "t"], cwd=tmp)
        from PIL import Image, ImageChops
        im = Image.open(os.path.join(tmp, "t.png")).convert("RGB"); bb = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
        im.crop((max(bb[0] - 20, 0), max(bb[1] - 20, 0), bb[2] + 20, bb[3] + 20)).save(os.path.join(OUT, "table1_static_mc.png"))
    else: print("pdflatex failed")
