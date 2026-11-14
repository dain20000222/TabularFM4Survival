# Table 8 (Appendix D.4): MITRA-CF with a sample-stratified or horizon-stratified context vs. the uniform context of the main
# experiments, on the dataset-K combinations whose training set exceeds MITRA's context of 8,192 rows (the others are
# identical by construction).
# Input: ../results/static_context_sampling.csv (runs of static/mitra_ctx.py, collected by static/build_static_long.py);
# mode "subject" = sample-stratified, mode "horizon" = horizon-stratified; "*_uniform": MITRA-CF of Table 1.
#   python table8_context_sampling.py -> out/context_sampling.csv
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
MET = ["ipcw_c_index", "mean_auc", "ibs"]
C = pd.read_csv(os.path.join(HERE, "..", "results", "static_context_sampling.csv"))
big = C[C.rows > 8192]
print(f"combinations above 8,192 training rows: {big[['dataset', 'K']].drop_duplicates().shape[0]} ({big.dataset.nunique()} datasets)")
u = big.drop_duplicates(["dataset", "K"])
rows = [dict(sampling="Uniform", **{m: u[m + "_uniform"].mean() for m in MET}, **{"better_" + m: "" for m in MET})]
for mode, name in (("subject", "Sample-stratified"), ("horizon", "Horizon-stratified")):
    g = big[big["mode"] == mode]
    better = [int(((g[m] < g[m + "_uniform"]) if m == "ibs" else (g[m] > g[m + "_uniform"])).sum()) for m in MET]
    nn = [int(g[[m, m + "_uniform"]].dropna().shape[0]) for m in MET]
    rows.append(dict(sampling=name, **{m: g[m].mean() for m in MET}, **{"better_" + m: f"{b}/{x}" for m, b, x in zip(MET, better, nn)}))
T = pd.DataFrame(rows)
T.round(4).to_csv(os.path.join(OUT, "context_sampling.csv"), index=False)
print(T.round(3).to_string(index=False))
small = C[C.rows <= 8192]
print(f"check, combinations below the limit (should equal uniform): max |C-index difference| = {(small.ipcw_c_index - small.ipcw_c_index_uniform).abs().max():.4f}")
