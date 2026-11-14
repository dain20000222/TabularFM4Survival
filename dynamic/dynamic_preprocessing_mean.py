#!/usr/bin/env python3
"""
Add patient-wise running summaries to dynamic survival datasets.

- num_*  → running mean  → <col>_mean
- fac_*  → running mode  → <col>_mode

pid, event, time, time2 are NOT duplicated.

Input : ./data
Output: ./dynamic_data_mean
"""

from pathlib import Path
import sys
import pandas as pd
from collections import Counter

EXCLUDE = {"pid", "event", "time", "time2"}


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def sort_within_pid(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["pid", "time", "time2"], kind="mergesort")


def running_mode(series: pd.Series) -> pd.Series:
    """
    Compute running mode with deterministic tie-breaking
    (ties resolved by most recent value).
    """
    counts = Counter()
    modes = []

    for val in series:
        if pd.notna(val):
            counts[val] += 1

        max_count = max(counts.values()) if counts else None
        candidates = {k for k, v in counts.items() if v == max_count}

        # tie-breaking: choose the most recent value
        mode = val if val in candidates else next(iter(candidates))
        modes.append(mode)

    return pd.Series(modes, index=series.index)


def add_running_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"pid", "event", "time", "time2"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")

    df = sort_within_pid(df)
    out = df.copy()

    covariates = [c for c in df.columns if c not in EXCLUDE]

    # Numeric: running mean
    for c in covariates:
        if c.startswith("num"):
            out[f"{c}_mean"] = (
                out.groupby("pid", sort=False)[c]
                   .expanding(min_periods=1)
                   .mean()
                   .reset_index(level=0, drop=True)
            )

    # Categorical: running mode
    for c in covariates:
        if c.startswith("fac"):
            out[f"{c}_mode"] = (
                out.groupby("pid", sort=False)[c]
                   .apply(running_mode)
                   .reset_index(level=0, drop=True)
            )

    return out


def main() -> int:
    in_dir = Path("data")
    out_dir = Path("data_mean")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".txt"}]
    if not files:
        print("No datasets found in ./data", file=sys.stderr)
        return 1

    for path in files:
        try:
            df = read_table(path)
            df_out = add_running_features(df)
        except Exception as e:
            print(f"[SKIP] {path.name}: {e}", file=sys.stderr)
            continue

        out_path = out_dir / path.name
        df_out.to_csv(out_path, index=False)
        print(f"[OK] Saved {out_path} ({df.shape[1]} → {df_out.shape[1]} columns)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
