# Patient splits for the R joint model (split/{dataset}_{train,val,test}_pids.csv), identical to the split of every
# Python dynamic script: pids in order of first appearance, train_test_split(test_size=0.3, seed 42), then 50/50.
import os
import pandas as pd
from sklearn.model_selection import train_test_split

os.makedirs("split", exist_ok=True)
for f in sorted(os.listdir("data_mean")):
    if not f.endswith(".csv"):
        continue
    pids = pd.read_csv(os.path.join("data_mean", f))["pid"].unique()
    tr, tmp = train_test_split(pids, test_size=0.3, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, random_state=42)
    for name, p in (("train", tr), ("val", va), ("test", te)):
        pd.DataFrame({"pid": p}).to_csv(os.path.join("split", f"{f[:-4]}_{name}_pids.csv"), index=False)
    print(f[:-4], len(tr), len(va), len(te))
