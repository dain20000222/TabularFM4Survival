import os
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from elote import EloCompetitor, LambdaArena
from scipy.stats import ttest_rel

warnings.filterwarnings("ignore")

MEDIAN = False


def get_k_eval_mapping():
    """Define the mapping between k values and number of evaluation times"""
    return {
        # New baseline k values (fractional) correspond to larger K via:
        #   n_eval_times = round(100 / k) - 1
        # so that K = n_eval_times + 1 ≈ round(100 / k).
        # 2.5: 39,  # K=40
        # 2.857142857142857: 34,  # K=35
        # 3.3333333333333335: 29,  # K=30
        # 4.0: 24,  # K=25
        5: 19,
        7: 14,
        10: 9,
        20: 4,
        25: 3,
    }


def get_baseline_k_to_K_mapping():
    """Map baseline k (percentile step) to K (# discrete intervals).

    Baseline files are still keyed by k and report n_eval_times. For the
    plotting x-axis ("Number of Discrete Intervals, K"), we convert via:
        K = n_eval_times + 1
    using the expected n_eval_times from get_k_eval_mapping().
    """
    m = get_k_eval_mapping()
    return {k: int(v) + 1 for k, v in m.items()}


def get_target_K_values():
    """Target interval counts for plots/tables."""
    return [5, 10, 15, 20]


def get_target_baseline_k_values():
    """Baseline (args.k) values to load if present.

    Baseline runs are still keyed by k (percentile step). Historically this code
    used k in {5, 7, 10, 20, 25}; now we also support additional k values that
    correspond to the expanded K grid.
    """
    # NOTE: baseline filenames use the literal k value formatting (e.g. "4.0").
    # Keep these as numbers (some are fractional) and stringify when building
    # filenames.
    return [
        2.5,
        2.857142857142857,
        3.3333333333333335,
        4.0,
        5,
        7,
        10,
        20,
        25,
    ]


def load_data():
    """Load evaluation CSV files.

    - Baseline files are keyed by k (legacy naming), so we load them by k.
    - Non-baseline models (MITRA/TabPFN/XGBoost and discrete/multiclass variants) are
      keyed by K (# intervals) and contain a K column, so we discover them via
      glob and load all available files.
    """
    data = {}
    k_values = get_target_baseline_k_values()
    k_eval_mapping = get_k_eval_mapping()

    print("Loading data files...")

    for k in k_values:
        expected_eval_times = k_eval_mapping.get(k, "unknown")
        print(f"\nLoading files for k={k} (expected {expected_eval_times} eval times):")

        # Load baseline evaluations (updated filename)
        baseline_file = f"static_baseline_evaluation_final_{k}_ipcw_c_index.csv"
        try:
            data[f"baseline_{k}"] = pd.read_csv(baseline_file)
            print(f"  ✓ {baseline_file}: {len(data[f'baseline_{k}'])} datasets")
        except FileNotFoundError:
            print(f"  ⚠️ {baseline_file} not found")

    # Load non-baseline model evaluations by discovery (K-keyed)
    print("\nLoading non-baseline model files (K-keyed)...")
    patterns = [
        "static_tabpfn_v3_evaluation_final_*.csv",
        "static_tabpfn_v3_evaluation_discrete_*.csv",
        "static_tabpfn_evaluation_final_*.csv",
        "static_mitra_evaluation_final_*.csv",
        "static_tabpfn_evaluation_multiclass_*.csv",
        "static_mitra_evaluation_multiclass_*.csv",
        "static_xgboost_evaluation_final_*.csv",
        "static_tabpfn_evaluation_discrete_*.csv",
        "static_mitra_evaluation_discrete_*.csv",
        "static_xgboost_evaluation_discrete_*.csv",
        "static_xgboost_evaluation_multiclass_*.csv",
    ]

    for pat in patterns:
        for p in sorted(Path(".").glob(pat)):
            key = p.stem
            if key in data:
                continue
            try:
                df = pd.read_csv(p)
                if "K" not in df.columns:
                    print(f"  ⚠️ {p.name} missing K column, skipping")
                    continue
                data[key] = df
                print(f"  ✓ {p.name}: {len(df)} datasets")
            except Exception as e:
                print(f"  ⚠️ Failed to load {p.name}: {e}")

    print(f"\nData loaded successfully. Total files: {len(data)}")

    return data


def prepare_comparison_data(data):
    """Prepare data for comparison by aligning datasets and metrics"""

    metrics = ["c_index", "ipcw_c_index", "ibs", "mean_auc"]
    k_values = get_target_baseline_k_values()
    k_eval_mapping = get_k_eval_mapping()
    baseline_k_to_K = get_baseline_k_to_K_mapping()

    results = []

    # Collect all K values present in non-baseline files
    non_baseline_keys = [
        k for k in data.keys() if k.startswith("static_") and ("baseline" not in k)
    ]
    all_K = set()
    for key in non_baseline_keys:
        df = data[key]
        if "K" in df.columns:
            all_K.update(df["K"].dropna().astype(int).tolist())

    all_K2 = list(set(baseline_k_to_K.values()) & set(all_K))
    target_K = set(get_target_K_values())
    K_values = [K for K in sorted(all_K2) if K in target_K]

    # # Add baseline-derived K values too (for baselines only)
    # all_K.update(baseline_k_to_K.values())
    # # Prefer the explicit grid for consistent x-axis across plots.
    # K_values = [K for K in get_target_K_values() if K in all_K]
    # # Still include any unexpected K values that appear in discovered files.
    # K_values += [K for K in sorted(all_K) if K not in K_values]

    for K in K_values:
        # Baseline key is still by k; map K back to any k that yields it.
        k_for_baseline = None
        for kk, KK in baseline_k_to_K.items():
            if KK == K:
                k_for_baseline = kk
                break

        baseline_key = (
            f"baseline_{k_for_baseline}" if k_for_baseline is not None else None
        )

        tabpfn_key = "static_tabpfn_evaluation_final"
        tabpfn_v3_key = "static_tabpfn_v3_evaluation_final"
        tabpfn_v3_discrete_key = "static_tabpfn_v3_evaluation_discrete"
        mitra_key = "static_mitra_evaluation_final"
        tabpfn_multiclass_key = "static_tabpfn_evaluation_multiclass"
        mitra_multiclass_key = "static_mitra_evaluation_multiclass"
        xgboost_key = "static_xgboost_evaluation_final"
        xgboost_multiclass_key = "static_xgboost_evaluation_multiclass"
        tabpfn_discrete_key = "static_tabpfn_evaluation_discrete"
        mitra_discrete_key = "static_mitra_evaluation_discrete"
        xgboost_discrete_key = "static_xgboost_evaluation_discrete"

        # Check if data exists for this K
        available_keys = []

        if baseline_key is not None and baseline_key in data:
            available_keys.append(baseline_key)

        # For non-baseline models, we may have multiple files; include if any match K
        def _keys_for_prefix(prefix: str):
            out = []
            for key in non_baseline_keys:
                if key.startswith(prefix + "_") or key == prefix:
                    df = data[key]
                    if "K" in df.columns and (df["K"].astype(int) == int(K)).any():
                        out.append(key)
            return out

        tabpfn_keys = _keys_for_prefix(tabpfn_key)
        tabpfn_v3_keys = _keys_for_prefix(tabpfn_v3_key)
        tabpfn_v3_discrete_keys = _keys_for_prefix(tabpfn_v3_discrete_key)
        mitra_keys = _keys_for_prefix(mitra_key)
        tabpfn_multiclass_keys = _keys_for_prefix(tabpfn_multiclass_key)
        mitra_multiclass_keys = _keys_for_prefix(mitra_multiclass_key)
        xgboost_keys = _keys_for_prefix(xgboost_key)
        xgboost_multiclass_keys = _keys_for_prefix(xgboost_multiclass_key)
        tabpfn_discrete_keys = _keys_for_prefix(tabpfn_discrete_key)
        mitra_discrete_keys = _keys_for_prefix(mitra_discrete_key)
        xgboost_discrete_keys = _keys_for_prefix(xgboost_discrete_key)

        available_keys.extend(
            tabpfn_keys
            + tabpfn_v3_keys
            + tabpfn_v3_discrete_keys
            + mitra_keys
            + tabpfn_multiclass_keys
            + mitra_multiclass_keys
            + xgboost_keys
            + xgboost_multiclass_keys
            + tabpfn_discrete_keys
            + mitra_discrete_keys
            + xgboost_discrete_keys
        )

        if len(available_keys) < 2:
            print(f"⚠️ Skipping K={K} - insufficient data ({len(available_keys)} files)")
            continue

        # Get datasets for each available model
        all_datasets = set()
        model_datasets = {}

        # Process baseline models (Cox, RSF, DeepHit, DeepSurv)
        if baseline_key is not None and baseline_key in data:
            baseline_df = data[baseline_key]
            # Use 'dataset_name' column for baseline files
            baseline_datasets = set(baseline_df["dataset_name"].str.lower())
            model_datasets["baseline"] = (baseline_df, baseline_datasets)
            all_datasets.update(baseline_datasets)

        # Process TabPFN
        if tabpfn_keys:
            tabpfn_df = pd.concat([data[k] for k in tabpfn_keys], ignore_index=True)
            tabpfn_df = tabpfn_df[tabpfn_df["K"].astype(int) == int(K)].copy()
            tabpfn_datasets = set(tabpfn_df["dataset"].str.lower())
            model_datasets["tabpfn"] = (tabpfn_df, tabpfn_datasets)
            all_datasets.update(tabpfn_datasets)

        # Process TabPFN-v3
        if tabpfn_v3_keys:
            tabpfn_v3_df = pd.concat([data[k] for k in tabpfn_v3_keys], ignore_index=True)
            tabpfn_v3_df = tabpfn_v3_df[tabpfn_v3_df["K"].astype(int) == int(K)].copy()
            tabpfn_v3_datasets = set(tabpfn_v3_df["dataset"].str.lower())
            model_datasets["tabpfn_v3"] = (tabpfn_v3_df, tabpfn_v3_datasets)
            all_datasets.update(tabpfn_v3_datasets)

        # Process TabPFN-v3-DH
        if tabpfn_v3_discrete_keys:
            tabpfn_v3_discrete_df = pd.concat(
                [data[k] for k in tabpfn_v3_discrete_keys], ignore_index=True
            )
            tabpfn_v3_discrete_df = tabpfn_v3_discrete_df[
                tabpfn_v3_discrete_df["K"].astype(int) == int(K)
            ].copy()
            tabpfn_v3_discrete_datasets = set(
                tabpfn_v3_discrete_df["dataset"].str.lower()
            )
            model_datasets["tabpfn_v3_discrete"] = (
                tabpfn_v3_discrete_df,
                tabpfn_v3_discrete_datasets,
            )
            all_datasets.update(tabpfn_v3_discrete_datasets)

        # Process MITRA
        if mitra_keys:
            mitra_df = pd.concat([data[k] for k in mitra_keys], ignore_index=True)
            mitra_df = mitra_df[mitra_df["K"].astype(int) == int(K)].copy()
            mitra_datasets = set(mitra_df["dataset"].str.lower())
            model_datasets["mitra"] = (mitra_df, mitra_datasets)
            all_datasets.update(mitra_datasets)

        # Process MITRA-MC
        if mitra_multiclass_keys:
            mitra_multiclass_df = pd.concat(
                [data[k] for k in mitra_multiclass_keys], ignore_index=True
            )
            mitra_multiclass_df = mitra_multiclass_df[
                mitra_multiclass_df["K"].astype(int) == int(K)
            ].copy()
            mitra_multiclass_datasets = set(
                mitra_multiclass_df["dataset"].str.lower()
            )
            model_datasets["mitra_multiclass"] = (
                mitra_multiclass_df,
                mitra_multiclass_datasets,
            )
            all_datasets.update(mitra_multiclass_datasets)

        # Process TabPFN-MC
        if tabpfn_multiclass_keys:
            tabpfn_multiclass_df = pd.concat(
                [data[k] for k in tabpfn_multiclass_keys], ignore_index=True
            )
            tabpfn_multiclass_df = tabpfn_multiclass_df[
                tabpfn_multiclass_df["K"].astype(int) == int(K)
            ].copy()
            tabpfn_multiclass_datasets = set(
                tabpfn_multiclass_df["dataset"].str.lower()
            )
            model_datasets["tabpfn_multiclass"] = (
                tabpfn_multiclass_df,
                tabpfn_multiclass_datasets,
            )
            all_datasets.update(tabpfn_multiclass_datasets)

        # Process XGBoost
        if xgboost_keys:
            xgboost_df = pd.concat([data[k] for k in xgboost_keys], ignore_index=True)
            xgboost_df = xgboost_df[xgboost_df["K"].astype(int) == int(K)].copy()
            xgboost_datasets = set(xgboost_df["dataset"].str.lower())
            model_datasets["xgboost"] = (xgboost_df, xgboost_datasets)
            all_datasets.update(xgboost_datasets)

        # Process XGBoost-MC (multiclass time-bin classifier)
        if xgboost_multiclass_keys:
            xgboost_multiclass_df = pd.concat(
                [data[k] for k in xgboost_multiclass_keys], ignore_index=True
            )
            xgboost_multiclass_df = xgboost_multiclass_df[
                xgboost_multiclass_df["K"].astype(int) == int(K)
            ].copy()
            xgboost_multiclass_datasets = set(
                xgboost_multiclass_df["dataset"].str.lower()
            )
            model_datasets["xgboost_multiclass"] = (
                xgboost_multiclass_df,
                xgboost_multiclass_datasets,
            )
            all_datasets.update(xgboost_multiclass_datasets)

        # Process TabPFN-DH
        if tabpfn_discrete_keys:
            tabpfn_discrete_df = pd.concat(
                [data[k] for k in tabpfn_discrete_keys], ignore_index=True
            )
            tabpfn_discrete_df = tabpfn_discrete_df[
                tabpfn_discrete_df["K"].astype(int) == int(K)
            ].copy()
            tabpfn_discrete_datasets = set(tabpfn_discrete_df["dataset"].str.lower())
            model_datasets["tabpfn_discrete"] = (
                tabpfn_discrete_df,
                tabpfn_discrete_datasets,
            )
            all_datasets.update(tabpfn_discrete_datasets)

        # Process MITRA-DH
        if mitra_discrete_keys:
            mitra_discrete_df = pd.concat(
                [data[k] for k in mitra_discrete_keys], ignore_index=True
            )
            mitra_discrete_df = mitra_discrete_df[
                mitra_discrete_df["K"].astype(int) == int(K)
            ].copy()
            mitra_discrete_datasets = set(mitra_discrete_df["dataset"].str.lower())
            model_datasets["mitra_discrete"] = (
                mitra_discrete_df,
                mitra_discrete_datasets,
            )
            all_datasets.update(mitra_discrete_datasets)

        # Process XGBoost-DH
        if xgboost_discrete_keys:
            xgboost_discrete_df = pd.concat(
                [data[k] for k in xgboost_discrete_keys], ignore_index=True
            )
            xgboost_discrete_df = xgboost_discrete_df[
                xgboost_discrete_df["K"].astype(int) == int(K)
            ].copy()
            xgboost_discrete_datasets = set(xgboost_discrete_df["dataset"].str.lower())
            model_datasets["xgboost_discrete"] = (
                xgboost_discrete_df,
                xgboost_discrete_datasets,
            )
            all_datasets.update(xgboost_discrete_datasets)

        # Find common datasets across all available models
        common_datasets = all_datasets.copy()
        for model_name, (df, datasets) in model_datasets.items():
            common_datasets = common_datasets.intersection(datasets)

        # If baseline is present, ensure dataset alignment across ALL models by
        # removing any dataset where baseline has NaN for any baseline metric.
        # (We drop datasets, not rows, to keep per-K dataset counts consistent.)
        if "baseline" in model_datasets and common_datasets:
            baseline_df, _ = model_datasets["baseline"]
            baseline_models = ["cph", "rsf", "dh", "ds"]
            baseline_metric_cols = [
                f"test_{bm}_{m}" for bm in baseline_models for m in metrics
            ]
            baseline_metric_cols = [
                c for c in baseline_metric_cols if c in baseline_df.columns
            ]

            if baseline_metric_cols:
                baseline_sub = baseline_df[
                    baseline_df["dataset_name"].str.lower().isin(common_datasets)
                ].copy()
                baseline_sub["_dataset_lower"] = baseline_sub[
                    "dataset_name"
                ].str.lower()

                # Any NaN in baseline metrics => drop that dataset for this K.
                nan_by_ds = (
                    baseline_sub.set_index("_dataset_lower")[baseline_metric_cols]
                    .isna()
                    .any(axis=1)
                )
                bad_datasets = set(nan_by_ds[nan_by_ds].index.tolist())
                if bad_datasets:
                    common_datasets = common_datasets.difference(bad_datasets)
                    print(
                        f"  🧹 K={K}: dropped {len(bad_datasets)} datasets due to baseline NaNs"
                    )

        expected_eval_times = (int(K) - 1) if int(K) >= 2 else "unknown"
        print(f"\nK={K} ({expected_eval_times} eval times):")
        print(f"  Available models: {list(model_datasets.keys())}")
        print(f"  Common datasets: {len(common_datasets)}")

        # Process each common dataset
        for dataset in common_datasets:
            # Add TabPFN results
            if "tabpfn" in model_datasets:
                tabpfn_df, _ = model_datasets["tabpfn"]
                tabpfn_row = tabpfn_df[
                    tabpfn_df["dataset"].str.lower() == dataset
                ].iloc[0]

                actual_eval_times = tabpfn_row.get("n_eval_times", "unknown")

                for metric in metrics:
                    tabpfn_value = (
                        tabpfn_row[metric] if metric in tabpfn_row else np.nan
                    )

                    result_row = {
                        "dataset": dataset,
                        "k": K,
                        "model": "TabPFN-CF",
                        "metric": metric,
                        "value": tabpfn_value,
                    }
                    results.append(result_row)

            # Add TabPFN-v3 results
            if "tabpfn_v3" in model_datasets:
                tabpfn_v3_df, _ = model_datasets["tabpfn_v3"]
                tabpfn_v3_row = tabpfn_v3_df[
                    tabpfn_v3_df["dataset"].str.lower() == dataset
                ].iloc[0]

                actual_eval_times = tabpfn_v3_row.get("n_eval_times", "unknown")

                for metric in metrics:
                    tabpfn_v3_value = (
                        tabpfn_v3_row[metric] if metric in tabpfn_v3_row else np.nan
                    )

                    # result_row = {
                    #     "dataset": dataset,
                    #     "k": K,
                    #     "model": "TabPFNv3-CF",
                    #     "metric": metric,
                    #     "value": tabpfn_v3_value,
                    # }
                    # results.append(result_row)

            # Add TabPFNv3-DH results
            if "tabpfn_v3_discrete" in model_datasets:
                tabpfn_v3_discrete_df, _ = model_datasets["tabpfn_v3_discrete"]
                tabpfn_v3_discrete_row = tabpfn_v3_discrete_df[
                    tabpfn_v3_discrete_df["dataset"].str.lower() == dataset
                ].iloc[0]

                actual_eval_times = tabpfn_v3_discrete_row.get(
                    "n_eval_times", "unknown"
                )

                for metric in metrics:
                    tabpfn_v3_discrete_value = (
                        tabpfn_v3_discrete_row[metric]
                        if metric in tabpfn_v3_discrete_row
                        else np.nan
                    )

                    # result_row = {
                    #     "dataset": dataset,
                    #     "k": K,
                    #     "model": "TabPFNv3-DH",
                    #     "metric": metric,
                    #     "value": tabpfn_v3_discrete_value,
                    # }
                    # results.append(result_row)

            # Add MITRA results
            if "mitra" in model_datasets:
                mitra_df, _ = model_datasets["mitra"]
                mitra_row = mitra_df[mitra_df["dataset"].str.lower() == dataset].iloc[0]

                actual_eval_times = mitra_row.get("n_eval_times", "unknown")

                for metric in metrics:
                    mitra_value = mitra_row[metric] if metric in mitra_row else np.nan

                    result_row = {
                        "dataset": dataset,
                        "k": K,
                        "model": "MITRA-CF",
                        "metric": metric,
                        "value": mitra_value,
                    }
                    results.append(result_row)

            # Add MITRA-MC results
            if "mitra_multiclass" in model_datasets:
                mitra_multiclass_df, _ = model_datasets["mitra_multiclass"]
                mitra_multiclass_row = mitra_multiclass_df[
                    mitra_multiclass_df["dataset"].str.lower() == dataset
                ].iloc[0]

                actual_eval_times = mitra_multiclass_row.get("n_eval_times", "unknown")

                for metric in metrics:
                    mitra_multiclass_value = (
                        mitra_multiclass_row[metric]
                        if metric in mitra_multiclass_row
                        else np.nan
                    )

                    result_row = {
                        "dataset": dataset,
                        "k": K,
                        "model": "MITRA-MC",
                        "metric": metric,
                        "value": mitra_multiclass_value,
                    }
                    results.append(result_row)

            # Add TabPFN-MC results
            if "tabpfn_multiclass" in model_datasets:
                tabpfn_multiclass_df, _ = model_datasets["tabpfn_multiclass"]
                tabpfn_multiclass_row = tabpfn_multiclass_df[
                    tabpfn_multiclass_df["dataset"].str.lower() == dataset
                ].iloc[0]

                actual_eval_times = tabpfn_multiclass_row.get("n_eval_times", "unknown")

                for metric in metrics:
                    tabpfn_multiclass_value = (
                        tabpfn_multiclass_row[metric]
                        if metric in tabpfn_multiclass_row
                        else np.nan
                    )

                    result_row = {
                        "dataset": dataset,
                        "k": K,
                        "model": "TabPFN-MC",
                        "metric": metric,
                        "value": tabpfn_multiclass_value,
                    }
                    results.append(result_row)

            # Add XGBoost results (new model)
            if "xgboost" in model_datasets:
                xgboost_df, _ = model_datasets["xgboost"]
                xgboost_row = xgboost_df[
                    xgboost_df["dataset"].str.lower() == dataset
                ].iloc[0]

                actual_eval_times = xgboost_row.get("n_eval_times", "unknown")

                for metric in metrics:
                    xgboost_value = (
                        xgboost_row[metric] if metric in xgboost_row else np.nan
                    )

                    result_row = {
                        "dataset": dataset,
                        "k": K,
                        "model": "XGBoost-CF",
                        "metric": metric,
                        "value": xgboost_value,
                    }
                    results.append(result_row)

            # Add XGBoost-MC results (multiclass time-bin classifier)
            if "xgboost_multiclass" in model_datasets:
                xgboost_multiclass_df, _ = model_datasets["xgboost_multiclass"]
                xgboost_multiclass_row = xgboost_multiclass_df[
                    xgboost_multiclass_df["dataset"].str.lower() == dataset
                ].iloc[0]

                actual_eval_times = xgboost_multiclass_row.get("n_eval_times", "unknown")

                for metric in metrics:
                    xgboost_multiclass_value = (
                        xgboost_multiclass_row[metric]
                        if metric in xgboost_multiclass_row
                        else np.nan
                    )

                    result_row = {
                        "dataset": dataset,
                        "k": K,
                        "model": "XGBoost-MC",
                        "metric": metric,
                        "value": xgboost_multiclass_value,
                    }
                    results.append(result_row)

            # Add baseline model results (Cox, RSF, DeepHit, DeepSurv)
            if "baseline" in model_datasets:
                baseline_df, _ = model_datasets["baseline"]
                baseline_row = baseline_df[
                    baseline_df["dataset_name"].str.lower() == dataset
                ].iloc[0]

                actual_eval_times_cph = baseline_row.get("n_eval_times_cph", "unknown")
                actual_eval_times_rsf = baseline_row.get("n_eval_times_rsf", "unknown")
                actual_eval_times_dh = baseline_row.get("n_eval_times_dh", "unknown")
                actual_eval_times_ds = baseline_row.get("n_eval_times_ds", "unknown")

                # Process Cox (CPH) results
                for metric in metrics:
                    cph_col = f"test_cph_{metric}"
                    if cph_col in baseline_row:
                        value = baseline_row[cph_col]
                        result_row = {
                            "dataset": dataset,
                            "k": K,
                            "model": "CPH",
                            "metric": metric,
                            "value": value,
                        }
                        results.append(result_row)

                # Process RSF results
                for metric in metrics:
                    rsf_col = f"test_rsf_{metric}"
                    if rsf_col in baseline_row:
                        value = baseline_row[rsf_col]
                        result_row = {
                            "dataset": dataset,
                            "k": K,
                            "model": "RSF",
                            "metric": metric,
                            "value": value,
                        }
                        results.append(result_row)

                # Process DeepHit (DH) results
                for metric in metrics:
                    dh_col = f"test_dh_{metric}"
                    if dh_col in baseline_row:
                        value = baseline_row[dh_col]
                        result_row = {
                            "dataset": dataset,
                            "k": K,
                            "model": "DeepHit",
                            "metric": metric,
                            "value": value,
                        }
                        results.append(result_row)

                # Process DeepSurv (DS) results
                for metric in metrics:
                    ds_col = f"test_ds_{metric}"
                    if ds_col in baseline_row:
                        value = baseline_row[ds_col]
                        result_row = {
                            "dataset": dataset,
                            "k": K,
                            "model": "DeepSurv",
                            "metric": metric,
                            "value": value,
                        }
                        results.append(result_row)

            # Add TabPFN-DH results
            if "tabpfn_discrete" in model_datasets:
                tabpfn_discrete_df, _ = model_datasets["tabpfn_discrete"]
                tabpfn_discrete_row = tabpfn_discrete_df[
                    tabpfn_discrete_df["dataset"].str.lower() == dataset
                ].iloc[0]

                actual_eval_times = tabpfn_discrete_row.get("n_eval_times", "unknown")

                for metric in metrics:
                    tabpfn_discrete_value = (
                        tabpfn_discrete_row[metric]
                        if metric in tabpfn_discrete_row
                        else np.nan
                    )

                    result_row = {
                        "dataset": dataset,
                        "k": K,
                        "model": "TabPFN-DH",
                        "metric": metric,
                        "value": tabpfn_discrete_value,
                    }
                    results.append(result_row)

            # Add MITRA-DH results
            if "mitra_discrete" in model_datasets:
                mitra_discrete_df, _ = model_datasets["mitra_discrete"]
                mitra_discrete_row = mitra_discrete_df[
                    mitra_discrete_df["dataset"].str.lower() == dataset
                ].iloc[0]

                actual_eval_times = mitra_discrete_row.get("n_eval_times", "unknown")

                for metric in metrics:
                    mitra_discrete_value = (
                        mitra_discrete_row[metric]
                        if metric in mitra_discrete_row
                        else np.nan
                    )

                    result_row = {
                        "dataset": dataset,
                        "k": K,
                        "model": "MITRA-DH",
                        "metric": metric,
                        "value": mitra_discrete_value,
                    }
                    results.append(result_row)

            # Add XGBoost-DH results
            if "xgboost_discrete" in model_datasets:
                xgboost_discrete_df, _ = model_datasets["xgboost_discrete"]
                xgboost_discrete_row = xgboost_discrete_df[
                    xgboost_discrete_df["dataset"].str.lower() == dataset
                ].iloc[0]

                actual_eval_times = xgboost_discrete_row.get("n_eval_times", "unknown")

                for metric in metrics:
                    xgboost_discrete_value = (
                        xgboost_discrete_row[metric]
                        if metric in xgboost_discrete_row
                        else np.nan
                    )

                    result_row = {
                        "dataset": dataset,
                        "k": K,
                        "model": "XGBoost-DH",
                        "metric": metric,
                        "value": xgboost_discrete_value,
                    }
                    results.append(result_row)

    out = pd.DataFrame(results)
    return out


def bootstrap_median_se(data, n_iterations=10000):
    dataset = np.array(data)
    n = len(dataset)

    # 1. Resample: Create a (iterations x n) matrix of random picks from the data
    # 'replace=True' is the key to bootstrapping
    resamples = np.random.choice(dataset, size=(n_iterations, n), replace=True)

    # 2. Calculate the median for every single row (iteration)
    resample_medians = np.median(resamples, axis=1)

    # 3. The Standard Error is simply the Standard Deviation of these medians
    se_median = np.std(resample_medians, ddof=1)

    return float(se_median)


def create_metric_vs_quantiles_plot(combined_df):
    """Create a figure with 3 subplots showing metric values vs number of quantiles"""

    # Set publication-quality style to match metrics_plot.py
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#EAEAF2",  # Light grey panel background
            "axes.edgecolor": "#B8B8C0",  # Grey panel border
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": "white",  # White grid lines
            "grid.linewidth": 1.0,
            "axes.axisbelow": True,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.frameon": True,
            "legend.facecolor": "#F2F2F7",  # Light grey legend background
            "legend.edgecolor": "#D0D0DA",  # Grey legend border
            "legend.framealpha": 1.0,
            "lines.linewidth": 2.6,  # Thicker lines like metrics_plot.py
            "lines.markersize": 6.5,  # Larger markers
        }
    )

    # Higher-contrast palette (classifier models only)
    # Use clearly separated hues across families, and a noticeably darker shade for "-DH".
    colorblind_colors = {
        "MITRA-CF": "#D64D51",
        "MITRA-MC": "#B73E44",
        "MITRA-DH": "#7A1E22",
        "TabPFN-CF": "#E9965E",
        # "TabPFNv3-CF": "#4F9D8A",
        # "TabPFNv3-DH": "#2F6F63",
        "TabPFN-MC": "#C9722A",
        "TabPFN-DH": "#7A4A10",
        "XGBoost-CF": "#437CE4",
        "XGBoost-MC": "#2E9F6E",
        "XGBoost-DH": "#0B2E8A",
    }

    # Stronger style separation: solid for base, dashed for discrete
    line_styles = {
        "MITRA-CF": "-",
        "MITRA-MC": "-.",
        "MITRA-DH": "--",
        "TabPFN-CF": "-",
        # "TabPFNv3-CF": "-",
        # "TabPFNv3-DH": "--",
        "TabPFN-MC": "-.",
        "TabPFN-DH": "--",
        "XGBoost-CF": "-",
        "XGBoost-MC": "-.",
        "XGBoost-DH": "--",
    }

    # Stronger marker separation: filled square for base, filled circle for discrete
    marker_styles = {
        "MITRA-CF": "s",
        "MITRA-MC": "D",
        "MITRA-DH": "o",
        "TabPFN-CF": "s",
        # "TabPFNv3-CF": "^",
        # "TabPFNv3-DH": "v",
        "TabPFN-MC": "D",
        "TabPFN-DH": "o",
        "XGBoost-CF": "s",
        "XGBoost-MC": "^",
        "XGBoost-DH": "o",
    }

    # Define model families (each will become its own 3-panel figure)
    model_families = {
        "mitra": ["MITRA-CF", "MITRA-MC", "MITRA-DH"],
        "tabpfn": [
            "TabPFN-CF",
            # "TabPFNv3-CF",
            "TabPFN-MC",
            "TabPFN-DH",
            # "TabPFNv3-DH",
        ],
        "xgboost": ["XGBoost-CF", "XGBoost-MC", "XGBoost-DH"],
    }

    # Define target interval counts K to plot
    target_quantiles = get_target_K_values()
    metrics = ["ipcw_c_index", "mean_auc", "ibs"]
    metric_labels = ["C-Index ↑", "Integrated AUC ↑", "Integrated Brier Score ↓"]

    # Fixed y-axis limits per metric panel (applied across all family figures)
    fixed_metric_ylim = {
        "ipcw_c_index": (0.60, 0.75),
        "mean_auc": (0.60, 0.80),
        "ibs": (0.15, 0.3),
    }

    # Create output directory
    output_dir = "experimental_results_7_7"
    os.makedirs(output_dir, exist_ok=True)

    # (previously: auto/robust y-limits) -- now using fixed_metric_ylim

    for family_name, family_models in model_families.items():
        # Create figure with 3 subplots - reduced height from 5 to 4
        fig, axes = plt.subplots(1, 3, figsize=(17, 4))

        last_plot_df = pd.DataFrame()

        for i, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i]

            # Prepare data for this metric
            metric_data = []

            for K in combined_df["k"].unique():
                k_df = combined_df[combined_df["k"] == K]

                n_quantiles = int(K)
                if n_quantiles not in target_quantiles:
                    continue

                # Get data for this metric and k value
                k_metric_data = k_df[k_df["metric"] == metric]
                if k_metric_data.empty:
                    continue

                # Calculate mean and standard error for each model (across datasets)
                for model in k_metric_data["model"].unique():
                    if model not in family_models:
                        continue

                    model_vals = k_metric_data[k_metric_data["model"] == model]["value"]
                    if len(model_vals) == 0:
                        continue

                    mean_val = float(model_vals.mean())
                    n_datasets = int(len(model_vals))
                    se_val = (
                        float(model_vals.std(ddof=1) / np.sqrt(n_datasets))
                        if n_datasets > 1
                        else 0.0
                    )

                    if MEDIAN:
                        mean_val = float(np.median(model_vals))
                        se_val = bootstrap_median_se(model_vals)

                    metric_data.append(
                        {
                            "n_quantiles": n_quantiles,
                            "model": model,
                            "mean": mean_val,
                            "se": se_val,
                            "n_datasets": n_datasets,
                        }
                    )

            plot_df = pd.DataFrame(metric_data)
            last_plot_df = plot_df

            if plot_df.empty:
                ax.set_title(f"{metric_label}\n(No data available)")
                ax.set_xlabel(
                    "Number of Discrete Intervals, $K$", fontsize=14, fontweight="bold"
                )
                if i == 0:
                    ax.set_ylabel("Avg Score", fontsize=14, fontweight="bold")
                ax.set_xticks(target_quantiles)
                ax.set_xticklabels([str(q) for q in target_quantiles])
                ax.tick_params(axis="both", which="major", width=1.0, length=4)
                continue

            available_models = [
                m for m in family_models if m in plot_df["model"].unique()
            ]

            for model in available_models:
                model_data = plot_df[plot_df["model"] == model].sort_values(
                    "n_quantiles"
                )
                if len(model_data) == 0:
                    continue

                color = colorblind_colors.get(model, "#888888")
                line_style = line_styles.get(model, "-")
                marker_style = marker_styles.get(model, "o")

                # Shaded standard error band (mean ± SE)
                y = model_data["mean"].to_numpy(dtype=float)
                se = model_data["se"].to_numpy(dtype=float)
                x = model_data["n_quantiles"].to_numpy(dtype=float)
                ax.fill_between(
                    x,
                    y - se,
                    y + se,
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                    zorder=1,
                )

                ax.plot(
                    model_data["n_quantiles"],
                    model_data["mean"],
                    label=model,
                    color=color,
                    marker=marker_style,
                    markersize=6.5,
                    linewidth=3.0,
                    linestyle=line_style,
                    alpha=1.0,
                    zorder=2,
                )

            ax.set_xlabel(
                "Number of Discrete Intervals, $K$", fontsize=14, fontweight="bold"
            )
            ax.set_title(f"{metric_label}", fontsize=16, fontweight="bold")
            if i == 0:
                ax.set_ylabel("Avg Score", fontsize=14, fontweight="bold")
            ax.set_xticks(target_quantiles)
            ax.set_xticklabels([str(q) for q in target_quantiles])
            ax.tick_params(axis="both", which="major", width=1.0, length=4)

        # Apply fixed per-metric y-limits across all family figures
        for i, metric in enumerate(metrics):
            if metric in fixed_metric_ylim:
                axes[i].set_ylim(*fixed_metric_ylim[metric])

        # Legend for this family
        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=len(labels) if len(labels) > 0 else 1,
            frameon=True,
            fancybox=False,
            shadow=False,
            framealpha=1.0,
            edgecolor="#D0D0DA",
            fontsize=12,
            handlelength=3.0,
            columnspacing=1.8,
            borderpad=0.7,
        )
        legend.get_frame().set_facecolor("#F2F2F7")

        plt.tight_layout()

        filename = os.path.join(output_dir, f"static_metrics_{family_name}.png")
        plt.savefig(
            filename, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"Saved: {filename}")

        # Print summary statistics for this family
        print(
            f"\n📊 Summary of data points per model/quantile combination ({family_name}):"
        )
        if not last_plot_df.empty:
            summary = last_plot_df.pivot_table(
                values="n_datasets", index="model", columns="n_quantiles", fill_value=0
            )
            ordered_models = [m for m in family_models if m in summary.index]
            ordered_quantiles = [q for q in target_quantiles if q in summary.columns]
            if ordered_models and ordered_quantiles:
                summary = summary.reindex(
                    index=ordered_models, columns=ordered_quantiles
                )
                print(summary)

        plt.show()
        plt.close()

    # Reset matplotlib parameters
    plt.rcParams.update(plt.rcParamsDefault)


def generate_latex_table(
    df,
    output_dir="experimental_results_7_7",
    forced_models=None,
    write_file=True,
    return_latex=False,
):
    """Generate a comprehensive LaTeX table with metrics and rankings with statistical significance"""

    print(
        "\n📊 Generating comprehensive results table with statistical significance..."
    )

    # Metrics to include (excluding c_index as suggested)
    # Order: C-Index, Integrated AUC, Integrated Brier Score
    metrics = ["ipcw_c_index", "mean_auc", "ibs"]

    # Define model order (same as in average_rankings_by_k.png)
    model_order = [
        "MITRA-CF",
        "TabPFN-CF",
        # "TabPFNv3-CF",
        "XGBoost-CF",
        "MITRA-MC",
        "TabPFN-MC",
        "XGBoost-MC",
        "MITRA-DH",
        "TabPFN-DH",
        # "TabPFNv3-DH",
        "XGBoost-DH",
        "CPH",
        "RSF",
        "DeepHit",
        "DeepSurv",
    ]

    # Get all models in the desired order
    if forced_models is None:
        available_models = [
            model for model in model_order if model in df["model"].unique()
        ]
    else:
        available_models = list(forced_models)

    # Initialize results storage
    table_data = []

    non_dh_models = [
        "MITRA-CF",
        "TabPFN-CF",
        # "TabPFNv3-CF",
        "XGBoost-CF",
        "MITRA-MC",
        "TabPFN-MC",
        "XGBoost-MC",
    ]
    dh_models = ["MITRA-DH", "TabPFN-DH", "XGBoost-DH"]
    baseline_models_for_layout = ["CPH", "RSF", "DeepHit", "DeepSurv"]

    elo_available = True
    print("📊 ELO rating calculation available (elote)")

    # Function to perform statistical significance test
    def is_significantly_better(metric, model1_values, model2_values, alpha=0.05):
        """
        Test if model1 is significantly better than model2
        Uses paired t-test for paired data
        """
        try:
            if len(model1_values) != len(model2_values) or len(model1_values) < 3:
                return False

            if metric == "ibs":  # Lower is better for IBS
                # Test if model1 < model2 (model1 is better)
                # H0: mean(model1) >= mean(model2), H1: mean(model1) < model2
                stat, p_value = ttest_rel(
                    model1_values, model2_values, alternative="less"
                )
            else:  # Higher is better for other metrics
                # Test if model1 > model2 (model1 is better)
                # H0: mean(model1) <= mean(model2), H1: mean(model1) > model2
                stat, p_value = ttest_rel(
                    model1_values, model2_values, alternative="greater"
                )

            return p_value < alpha
        except Exception as e:
            print(f"Warning: t-test failed for comparison: {e}")
            return False

    def calculate_elo_ratings(metric, dataset_means):
        """Calculate Elo ratings with elote, averaged over datasets."""
        try:
            direction = "lower" if metric == "ibs" else "higher"
            elo_rows = []

            for dataset in dataset_means["dataset"].unique():
                dataset_data = dataset_means[dataset_means["dataset"] == dataset]
                if len(dataset_data) < 2:
                    continue

                perf = dict(zip(dataset_data["model"], dataset_data["value"]))
                models_here = sorted(perf.keys())
                matchups = list(combinations(models_here, 2))

                if not matchups:
                    continue

                def compare(a, b):
                    va, vb = perf[a], perf[b]
                    if va == vb:
                        return None
                    if direction == "higher":
                        return va > vb
                    return va < vb

                arena = LambdaArena(
                    compare,
                    base_competitor=EloCompetitor,
                    base_competitor_kwargs={"initial_rating": 1000},
                )
                arena.tournament(matchups)
                lb = arena.leaderboard()
                ratings = {row["competitor"]: row["rating"] for row in lb}

                for m in models_here:
                    elo_rows.append(
                        {
                            "dataset": dataset,
                            "model": m,
                            "elo": int(round(ratings.get(m, 1000))),
                        }
                    )

            if not elo_rows:
                return {}

            elo_long = pd.DataFrame(elo_rows)
            elo_mean = (
                elo_long.groupby("model", as_index=False)["elo"]
                .mean()
                .assign(elo=lambda d: d["elo"].round().astype(int))
            )
            return dict(zip(elo_mean["model"], elo_mean["elo"]))
        except Exception as e:
            print(f"Warning: ELO calculation failed for metric {metric}: {e}")
            return {}

    # Calculate metric values for each model and identify best/second-best performers
    metric_rankings = {}

    for metric in metrics:
        metric_subset = df[df["metric"] == metric]
        if metric_subset.empty:
            continue

        # Group by dataset and calculate mean for each model across K
        dataset_means = (
            metric_subset.groupby(["dataset", "model"])["value"].mean().reset_index()
        )

        # Get values for each model across all datasets
        model_values = {}
        common_datasets = set(dataset_means["dataset"].unique())

        for model in available_models:
            model_data = dataset_means[dataset_means["model"] == model]
            model_datasets = set(model_data["dataset"].unique())
            common_datasets = common_datasets.intersection(model_datasets)

        # Filter to common datasets only
        for model in available_models:
            model_data = dataset_means[dataset_means["model"] == model]
            model_data = model_data[model_data["dataset"].isin(common_datasets)]
            if len(model_data) > 0:
                model_values[model] = model_data.sort_values("dataset")["value"].values

        if not model_values:
            continue

        # Calculate mean performance and rank models across datasetss
        model_means = {model: np.mean(values) for model, values in model_values.items()}
        if MEDIAN:
            model_means = {
                model: np.median(values) for model, values in model_values.items()
            }

        if metric == "ibs":  # Lower is better
            sorted_models = sorted(model_means.items(), key=lambda x: x[1])
        else:  # Higher is better
            sorted_models = sorted(
                model_means.items(), key=lambda x: x[1], reverse=True
            )

        # Identify best and second-best models with significance testing
        best_model = sorted_models[0][0]
        second_best_model = sorted_models[1][0] if len(sorted_models) > 1 else None

        # Test if best model is significantly better than others
        best_values = model_values[best_model]
        is_best_significant = True

        for other_model, other_values in model_values.items():
            if other_model == best_model:
                continue
            if not is_significantly_better(metric, best_values, other_values):
                is_best_significant = False
                break

        # Test if second-best is significantly better than third-best
        is_second_significant = False
        if second_best_model and len(sorted_models) > 2:
            second_values = model_values[second_best_model]
            third_best_model = sorted_models[2][0]
            third_values = model_values[third_best_model]
            is_second_significant = is_significantly_better(
                metric, second_values, third_values
            )

        elo_ratings = calculate_elo_ratings(metric, dataset_means)

        metric_rankings[metric] = {
            "best": best_model if is_best_significant else None,
            "second_best": second_best_model if is_second_significant else None,
            "elo_ratings": elo_ratings,
        }

    # Process each model in the specified order
    for model in available_models:
        model_data = df[df["model"] == model]
        row_data = {"model": model}

        # Calculate metrics and their standard errors
        for metric in metrics:
            # Averaging over K
            dataset_vals = (
                df[(df["model"] == model) & (df["metric"] == metric)]
                .groupby("dataset")["value"]
                .mean()
                .values
            )

            if len(dataset_vals) > 0:
                mean_val = float(np.mean(dataset_vals))
                n = int(len(dataset_vals))  # <- number of datasets
                se = float(np.std(dataset_vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

                if MEDIAN:
                    mean_val = float(np.median(dataset_vals))
                    se = bootstrap_median_se(dataset_vals)

                row_data[f"{metric}_mean"] = mean_val
                row_data[f"{metric}_se"] = se
                row_data[f"{metric}_n"] = n

                # Determine ranking status
                rankings = metric_rankings.get(metric, {})
                row_data[f"{metric}_is_best_significant"] = model == rankings.get(
                    "best"
                )
                row_data[f"{metric}_is_second_significant"] = model == rankings.get(
                    "second_best"
                )
                row_data[f"{metric}_elo"] = rankings.get("elo_ratings", {}).get(
                    model, 1000
                )
            else:
                row_data[f"{metric}_mean"] = np.nan
                row_data[f"{metric}_se"] = np.nan
                row_data[f"{metric}_n"] = 0
                row_data[f"{metric}_is_best_significant"] = False
                row_data[f"{metric}_is_second_significant"] = False
                row_data[f"{metric}_elo"] = 1000

        # Calculate rankings for each metric
        for metric in metrics:
            metric_subset = df[df["metric"] == metric]
            if not metric_subset.empty:
                # Group by dataset and calculate mean for each model over K
                dataset_means = (
                    metric_subset.groupby(["dataset", "model"])["value"]
                    .mean()
                    .reset_index()
                )

                # For each dataset, rank the models
                ranks = []

                for dataset in dataset_means["dataset"].unique():
                    dataset_data = dataset_means[dataset_means["dataset"] == dataset]

                    if metric == "ibs":  # Lower is better
                        dataset_ranks = dataset_data["value"].rank(ascending=True)
                    else:  # Higher is better
                        dataset_ranks = dataset_data["value"].rank(ascending=False)

                    # Find this model's rank for this dataset
                    model_idx = dataset_data[dataset_data["model"] == model].index
                    if len(model_idx) > 0:
                        model_rank = dataset_ranks.loc[model_idx[0]]
                        ranks.append(model_rank)

                if ranks:
                    avg_rank = np.mean(ranks)
                    rank_se = (
                        np.std(ranks, ddof=1) / np.sqrt(len(ranks))
                        if len(ranks) > 1
                        else 0.0
                    )
                else:
                    avg_rank = np.nan
                    rank_se = np.nan

                row_data[f"{metric}_rank"] = avg_rank
                row_data[f"{metric}_rank_se"] = rank_se

        # ---- Dataset-wise overall average rank + SE (proper) ----
        # Build per-metric per-dataset rank maps for THIS model
        per_metric_dataset_ranks = {}  # metric -> {dataset: rank}

        for metric in metrics:
            metric_subset = df[df["metric"] == metric]
            if metric_subset.empty:
                per_metric_dataset_ranks[metric] = {}
                continue

            # dataset_means: one value per (dataset, model); average over K
            dataset_means = (
                metric_subset.groupby(["dataset", "model"])["value"]
                .mean()
                .reset_index()
            )

            ds_rank_map = {}
            for dataset in dataset_means["dataset"].unique():
                dd = dataset_means[dataset_means["dataset"] == dataset]

                # Rank direction
                if metric == "ibs":  # lower is better
                    dd_ranks = dd["value"].rank(ascending=True, method="average")
                else:  # higher is better
                    dd_ranks = dd["value"].rank(ascending=False, method="average")

                # Extract this model's rank for this dataset
                idx = dd[dd["model"] == model].index
                if len(idx) > 0:
                    ds_rank_map[dataset] = float(dd_ranks.loc[idx[0]])

            per_metric_dataset_ranks[metric] = ds_rank_map

        # Use only datasets where this model has ranks for ALL metrics
        common_ds = None
        for metric in metrics:
            ds_set = set(per_metric_dataset_ranks.get(metric, {}).keys())
            common_ds = ds_set if common_ds is None else (common_ds & ds_set)

        dataset_avg_ranks = []
        if common_ds:
            for d in sorted(common_ds):
                dataset_avg_ranks.append(
                    float(np.mean([per_metric_dataset_ranks[m][d] for m in metrics]))
                )

        if dataset_avg_ranks:
            row_data["overall_rank"] = float(np.mean(dataset_avg_ranks))
            row_data["overall_rank_se"] = (
                float(
                    np.std(dataset_avg_ranks, ddof=1) / np.sqrt(len(dataset_avg_ranks))
                )
                if len(dataset_avg_ranks) > 1
                else 0.0
            )
        else:
            row_data["overall_rank"] = np.nan
            row_data["overall_rank_se"] = np.nan
        # --------------------------------------------------------

        all_elos = []
        for metric in metrics:
            if f"{metric}_elo" in row_data:
                all_elos.append(row_data[f"{metric}_elo"])

        row_data["overall_elo"] = np.mean(all_elos) if all_elos else 1000

        table_data.append(row_data)

    # Now determine best/second-best for each column (for coloring)
    # Ties are determined only from rounded display values (SE is not used).
    column_rankings = {}

    def _best_second_sets(model_value_pairs, higher_is_better):
        """Return joint-best and joint-second sets allowing ties by rounded values."""
        if not model_value_pairs:
            return set(), set()

        value_to_models = {}
        for model, value in model_value_pairs:
            value_to_models.setdefault(value, set()).add(model)

        unique_values = sorted(value_to_models.keys(), reverse=higher_is_better)
        best_value = unique_values[0]
        best_models = value_to_models[best_value]

        if len(unique_values) < 2:
            return best_models, set()

        second_value = unique_values[1]
        second_models = value_to_models[second_value]
        return best_models, second_models

    # For each metric's value column (displayed with 3 decimals)
    for metric in metrics:
        pairs = []
        for row in table_data:
            raw = row.get(f"{metric}_mean", np.nan)
            if np.isnan(raw):
                continue
            pairs.append((row["model"], float(np.round(raw, 3))))

        best_set, second_set = _best_second_sets(
            pairs, higher_is_better=(metric != "ibs")
        )
        column_rankings[f"{metric}_value_best"] = best_set
        column_rankings[f"{metric}_value_second"] = second_set

    # For each metric's rank column (displayed with 1 decimal, lower is better)
    for metric in metrics:
        pairs = []
        for row in table_data:
            raw = row.get(f"{metric}_rank", np.nan)
            if np.isnan(raw):
                continue
            pairs.append((row["model"], float(np.round(raw, 1))))

        best_set, second_set = _best_second_sets(pairs, higher_is_better=False)
        column_rankings[f"{metric}_rank_best"] = best_set
        column_rankings[f"{metric}_rank_second"] = second_set

    # For each metric's ELO column (displayed as integer, higher is better)
    for metric in metrics:
        pairs = []
        for row in table_data:
            raw = row.get(f"{metric}_elo", np.nan)
            if np.isnan(raw):
                continue
            pairs.append((row["model"], float(int(round(raw)))))

        best_set, second_set = _best_second_sets(pairs, higher_is_better=True)
        column_rankings[f"{metric}_elo_best"] = best_set
        column_rankings[f"{metric}_elo_second"] = second_set

    # Overall rank (displayed with 1 decimal, lower is better)
    overall_rank_pairs = []
    for row in table_data:
        raw = row.get("overall_rank", np.nan)
        if np.isnan(raw):
            continue
        overall_rank_pairs.append((row["model"], float(np.round(raw, 1))))

    best_set, second_set = _best_second_sets(overall_rank_pairs, higher_is_better=False)
    column_rankings["overall_rank_best"] = best_set
    column_rankings["overall_rank_second"] = second_set

    if elo_available:
        overall_elo_pairs = []
        for row in table_data:
            raw = row.get("overall_elo", np.nan)
            if np.isnan(raw):
                continue
            overall_elo_pairs.append((row["model"], float(int(round(raw)))))

        best_set, second_set = _best_second_sets(
            overall_elo_pairs, higher_is_better=True
        )
        column_rankings["overall_elo_best"] = best_set
        column_rankings["overall_elo_second"] = second_set

    # Generate LaTeX table
    latex_table = []

    # Table header
    latex_table.append("\\begin{table}[htbp]")
    latex_table.append("\\centering")
    latex_table.append(
        "\\caption{Quantile method, averaged over: $K=5,10,15,20$}"
    )
    latex_table.append("\\label{tab:performance_comparison}")
    latex_table.append("\\resizebox{\\textwidth}{!}{")
    if elo_available:
        latex_table.append("\\begin{tabular}{l|ccc|ccc|ccc|cc}")
    else:
        latex_table.append("\\begin{tabular}{l|ccc|ccc|ccc|c}")
    latex_table.append("\\toprule")

    if elo_available:
        header_line1 = "\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{3}{c|}{\\textbf{C-Index}} & \\multicolumn{3}{c|}{\\textbf{Integrated AUC}} & \\multicolumn{3}{c|}{\\textbf{Integrated Brier Score}} & \\multirow{2}{*}{\\textbf{Avg. Rank $\\downarrow$}} & \\multirow{2}{*}{\\textbf{Avg. ELO $\\uparrow$}} \\\\"
        header_line2 = " & \\textbf{Value $\\uparrow$} & \\textbf{Rank $\\downarrow$} & \\textbf{ELO $\\uparrow$} & \\textbf{Value $\\uparrow$} & \\textbf{Rank $\\downarrow$} & \\textbf{ELO $\\uparrow$} & \\textbf{Value $\\downarrow$} & \\textbf{Rank $\\downarrow$} & \\textbf{ELO $\\uparrow$} & & \\\\"
    else:
        header_line1 = "\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{3}{c|}{\\textbf{C-Index}} & \\multicolumn{3}{c|}{\\textbf{Integrated AUC}} & \\multicolumn{3}{c|}{\\textbf{Integrated Brier Score}} & \\multirow{2}{*}{\\textbf{Avg. Rank $\\downarrow$}} \\\\"
        header_line2 = " & \\textbf{Value $\\uparrow$} & \\textbf{Rank $\\downarrow$} & \\textbf{ELO $\\uparrow$} & \\textbf{Value $\\uparrow$} & \\textbf{Rank $\\downarrow$} & \\textbf{ELO $\\uparrow$} & \\textbf{Value $\\downarrow$} & \\textbf{Rank $\\downarrow$} & \\textbf{ELO $\\uparrow$} & \\\\"

    latex_table.append(header_line1)
    latex_table.append(header_line2)
    latex_table.append("\\midrule")

    group_break_after = [
        m
        for m in [
            "XGBoost-CF",
            "XGBoost-MC",
            dh_models[-1],
        ]
        if m in available_models
    ]

    for row in table_data:
        model_name = row["model"].replace("_", "\\_")
        line_parts = [f"\\textbf{{{model_name}}}"]

        for metric in metrics:
            if not np.isnan(row.get(f"{metric}_mean", np.nan)):
                mean_val = row[f"{metric}_mean"]
                se_val = row[f"{metric}_se"]
                value_str = f"{mean_val:.3f} ({se_val:.3f})"

                if row.get(f"{metric}_is_best_significant", False):
                    value_str = f"\\textbf{{{value_str}}}"

                if row["model"] in column_rankings.get(f"{metric}_value_best", set()):
                    value_str = f"\\colorbox{{blue!20}}{{{value_str}}}"
                elif row["model"] in column_rankings.get(
                    f"{metric}_value_second", set()
                ):
                    value_str = f"\\colorbox{{orange!20}}{{{value_str}}}"
            else:
                value_str = "--"

            if not np.isnan(row.get(f"{metric}_rank", np.nan)):
                rank_val = row[f"{metric}_rank"]
                rank_se_val = row[f"{metric}_rank_se"]
                rank_str = f"{rank_val:.1f} ({rank_se_val:.1f})"

                if row.get(f"{metric}_is_best_significant", False):
                    rank_str = f"\\textbf{{{rank_str}}}"

                if row["model"] in column_rankings.get(f"{metric}_rank_best", set()):
                    rank_str = f"\\colorbox{{blue!20}}{{{rank_str}}}"
                elif row["model"] in column_rankings.get(
                    f"{metric}_rank_second", set()
                ):
                    rank_str = f"\\colorbox{{orange!20}}{{{rank_str}}}"
            else:
                rank_str = "--"

            if not np.isnan(row.get(f"{metric}_elo", np.nan)):
                elo_val = row[f"{metric}_elo"]
                elo_str = f"{elo_val:.0f}"

                if row.get(f"{metric}_is_best_significant", False):
                    elo_str = f"\\textbf{{{elo_str}}}"

                if row["model"] in column_rankings.get(f"{metric}_elo_best", set()):
                    elo_str = f"\\colorbox{{blue!20}}{{{elo_str}}}"
                elif row["model"] in column_rankings.get(
                    f"{metric}_elo_second", set()
                ):
                    elo_str = f"\\colorbox{{orange!20}}{{{elo_str}}}"
            else:
                elo_str = "--"

            line_parts.extend([value_str, rank_str, elo_str])

        if not np.isnan(row.get("overall_rank", np.nan)):
            overall_rank_val = row["overall_rank"]
            overall_rank_se_val = row["overall_rank_se"]
            overall_str = f"{overall_rank_val:.1f} ({overall_rank_se_val:.1f})"
            if row["model"] in column_rankings.get("overall_rank_best", set()):
                overall_str = f"\\colorbox{{blue!20}}{{{overall_str}}}"
            elif row["model"] in column_rankings.get("overall_rank_second", set()):
                overall_str = f"\\colorbox{{orange!20}}{{{overall_str}}}"
        else:
            overall_str = "--"

        line_parts.append(overall_str)

        if elo_available:
            overall_elo_val = row.get("overall_elo", 1000)
            overall_elo_str = f"{overall_elo_val:.0f}"
            if row["model"] in column_rankings.get("overall_elo_best", set()):
                overall_elo_str = f"\\colorbox{{blue!20}}{{{overall_elo_str}}}"
            elif row["model"] in column_rankings.get("overall_elo_second", set()):
                overall_elo_str = f"\\colorbox{{orange!20}}{{{overall_elo_str}}}"
            line_parts.append(overall_elo_str)

        latex_table.append(" & ".join(line_parts) + " " + r"\\")

        if row["model"] in group_break_after:
            latex_table.append("\\hline")

    # Table footer
    latex_table.append("\\bottomrule")
    latex_table.append("\\end{tabular}")
    latex_table.append("}")
    latex_table.append("\\end{table}")

    # Save to file
    if write_file:
        os.makedirs(output_dir, exist_ok=True)
        latex_file = os.path.join(output_dir, "static_performance_table.tex")

        with open(latex_file, "w") as f:
            f.write("\n".join(latex_table))

        print(f"Comprehensive LaTeX table saved to: {latex_file}")

    # Print summary of results
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE AND RANKING SUMMARY")
    print("=" * 80)
    for metric in metrics:
        rankings = metric_rankings.get(metric, {})
        best = rankings.get("best", "No significantly best model")
        second = rankings.get("second_best", "No significantly second-best model")
        print(f"{metric.upper()}:")
        print(f"  Best: {best}")
        print(f"  Second: {second}")
        if elo_available and rankings.get("elo_ratings"):
            elo_sorted = sorted(
                rankings["elo_ratings"].items(), key=lambda x: x[1], reverse=True
            )
            print(
                "  ELO Rankings: "
                + ", ".join([f"{model}({rating:.0f})" for model, rating in elo_sorted])
            )

    print("\nCOLUMN RANKINGS (for coloring):")
    print("=" * 30)
    for key, model in column_rankings.items():
        print(f"  {key}: {model}")

    # Also print to console for easy copy-paste
    print("\n" + "=" * 80)
    print("COMPREHENSIVE LATEX TABLE (copy-paste ready)")
    print("=" * 80)
    print("\n".join(latex_table))

    if return_latex:
        return table_data, latex_table
    return table_data


def generate_latex_tables_all_and_per_k(df, output_dir="experimental_results_7_7"):
    """Generate one concatenated LaTeX table for all K values with K separator rows."""

    os.makedirs(output_dir, exist_ok=True)

    combined_tex_file = os.path.join(
        output_dir, "static_performance_tables_all_and_per_k.tex"
    )

    model_order = [
        "MITRA-CF",
        "TabPFN-CF",
        "TabPFNv3-CF",
        "XGBoost-CF",
        "MITRA-MC",
        "TabPFN-MC",
        "XGBoost-MC",
        "MITRA-DH",
        "TabPFN-DH",
        # "TabPFNv3-DH",
        "XGBoost-DH",
        "CPH",
        "RSF",
        "DeepHit",
        "DeepSurv",
    ]
    available_models = [m for m in model_order if m in df["model"].unique()]

    n_cols = 12
    full_table = []
    full_table.append("\\begin{table}[htbp]")
    full_table.append("\\centering")
    full_table.append(
        "\\caption{Performance comparison across survival analysis models (all K in one table)}"
    )
    full_table.append("\\label{tab:performance_comparison_all_k_single}")
    full_table.append("\\resizebox{\\textwidth}{!}{")
    full_table.append("\\begin{tabular}{l|ccc|ccc|ccc|cc}")
    full_table.append("\\toprule")

    full_table.append(
        "\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{3}{c|}{\\textbf{C-Index}} & \\multicolumn{3}{c|}{\\textbf{Integrated AUC}} & \\multicolumn{3}{c|}{\\textbf{Integrated Brier Score}} & \\multirow{2}{*}{\\textbf{Avg. Rank $\\downarrow$}} & \\multirow{2}{*}{\\textbf{Avg. ELO $\\uparrow$}} "
        + r"\\"
    )
    full_table.append(
        " & \\textbf{Value $\\uparrow$} & \\textbf{Rank $\\downarrow$} & \\textbf{ELO $\\uparrow$} & \\textbf{Value $\\uparrow$} & \\textbf{Rank $\\downarrow$} & \\textbf{ELO $\\uparrow$} & \\textbf{Value $\\downarrow$} & \\textbf{Rank $\\downarrow$} & \\textbf{ELO $\\uparrow$} & & "
        + r"\\"
    )
    full_table.append("\\midrule")

    # Add top block: aggregate over all K values
    _, all_k_latex = generate_latex_table(
        df,
        output_dir=output_dir,
        forced_models=available_models,
        write_file=False,
        return_latex=True,
    )
    mid_idx = next(
        (i for i, ln in enumerate(all_k_latex) if ln.strip() == "\\midrule"), None
    )
    bot_idx = next(
        (i for i, ln in enumerate(all_k_latex) if ln.strip() == "\\bottomrule"), None
    )
    all_body_rows = (
        all_k_latex[mid_idx + 1 : bot_idx]
        if mid_idx is not None and bot_idx is not None
        else []
    )

    first_block = True
    full_table.append(
        f"\\multicolumn{{{n_cols}}}{{|c|}}{{\\large\\textbf{{$K=\\mathrm{{All}}$}}}} "
        + r"\\"
    )
    full_table.append("\\hline")
    full_table.extend(all_body_rows)
    first_block = False

    for K in sorted(df["k"].unique()):
        k_df = df[df["k"] == K].copy()
        if k_df.empty:
            continue

        _, k_latex = generate_latex_table(
            k_df,
            output_dir=output_dir,
            forced_models=available_models,
            write_file=False,
            return_latex=True,
        )

        # Extract body rows between \midrule and \bottomrule
        mid_idx = next(
            (i for i, ln in enumerate(k_latex) if ln.strip() == "\\midrule"), None
        )
        bot_idx = next(
            (i for i, ln in enumerate(k_latex) if ln.strip() == "\\bottomrule"), None
        )
        body_rows = (
            k_latex[mid_idx + 1 : bot_idx]
            if mid_idx is not None and bot_idx is not None
            else []
        )

        if not first_block:
            full_table.append("\\midrule")
        first_block = False

        full_table.append(
            f"\\multicolumn{{{n_cols}}}{{|c|}}{{\\large\\textbf{{$K={int(K)}$}}}} "
            + r"\\"
        )
        full_table.append("\\hline")
        full_table.extend(body_rows)

    full_table.append("\\bottomrule")
    full_table.append("\\end{tabular}")
    full_table.append("}")
    full_table.append("\\end{table}")

    with open(combined_tex_file, "w") as f:
        f.write("\n".join(full_table).strip() + "\n")

    print(f"Saved single concatenated LaTeX table to: {combined_tex_file}")


def main():
    """Main function to run the comparison analysis"""

    print("Survival Analysis Model Comparison with Variable K Values")
    print("=" * 60)

    # Create output directory
    output_dir = "experimental_results_7_7"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = load_data()

    if not data:
        print("❌ No data files found. Please check your CSV files.")
        return

    # Prepare comparison data
    comparison_df = prepare_comparison_data(data)

    if comparison_df.empty:
        print("❌ No comparison data available. Please check your CSV files.")
        return

    # Print summary of loaded data
    print("\n📊 Data summary:")
    print(f"Total evaluations: {len(comparison_df)}")
    print(f"Models: {', '.join(sorted(comparison_df['model'].unique()))}")
    print(f"K values: {', '.join(map(str, sorted(comparison_df['k'].unique())))}")
    print(f"Datasets: {comparison_df['dataset'].nunique()}")

    # Generate LaTeX tables
    print("\n📊 Generating LaTeX tables...")
    table_data_full = generate_latex_table(comparison_df, output_dir)

    # Create metrics vs quantiles plot
    print("\nCreating metrics vs quantiles plot...")
    create_metric_vs_quantiles_plot(comparison_df)

    # Print model count summary
    model_counts = comparison_df["model"].value_counts()
    print("\nModel evaluation counts:")
    for model, count in model_counts.items():
        print(f"  - {model}: {count} evaluations")


if __name__ == "__main__":
    main()
