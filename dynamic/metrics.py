import pandas as pd
import numpy as np
import re
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc, brier_score
from sksurv.util import Surv


def load_survival_dataset(path):
    """
    Load survival dataset and get final outcome for each patient.
    Returns DataFrame with columns: pid, time, event
    """
    df = pd.read_csv(path)
    pat_df = (
        df.groupby("pid", as_index=False)
          .agg(time=("time2", "max"),
               event=("event", "max"))
    )
    return pat_df


def compute_harrel_cindex(survival_test, pred_df, landmark_times):
    """
    Compute Harrell's C-index at each landmark time.
    
    Parameters:
    -----------
    survival_test : DataFrame
        Ground truth survival data with columns: pid, time, event
    pred_df : DataFrame
        Prediction dataframe with columns: pid, tau, t, risk, surv_prob, dataset
        Rows where t is nan contain risk predictions at landmark time tau
    landmark_times : list
        List of landmark times (tau values) to evaluate
    
    Returns:
    --------
    dict : Dictionary mapping tau to C-index value
    """
    results = {}
    
    for tau in landmark_times:
        # Filter predictions at this landmark time (where t is nan)
        tau_preds = pred_df[(pred_df['tau'] == tau) & (pred_df['t'].isna())].copy()
        
        if len(tau_preds) == 0:
            error_msg = f"No predictions found for tau={tau}"
            print(f"Warning: {error_msg}")
            results[tau] = f"ERROR: {error_msg}"
            continue
        
        # Merge with ground truth
        merged = tau_preds.merge(survival_test, on='pid', how='inner')
        
        if len(merged) == 0:
            error_msg = f"No matching patients for tau={tau}"
            print(f"Warning: {error_msg}")
            results[tau] = f"ERROR: {error_msg}"
            continue
        
        # Prepare data for C-index calculation
        # event indicator (boolean array)
        event = merged['event'].astype(bool).values
        # observed time
        time = merged['time'].values
        # risk scores - sksurv expects higher risk = worse prognosis 
        estimate = merged['risk'].values
        
        try:
            # Calculate concordance index
            c_index = concordance_index_censored(event, time, estimate)
            # c_index returns (concordance_index, concordant, discordant, tied_risk, tied_time)
            results[tau] = c_index[0]
            print(f"Tau={tau}: C-index={c_index[0]:.4f} (n={len(merged)})")
        except Exception as e:
            print(f"Error calculating C-index for tau={tau}: {str(e)}")
            results[tau] = np.nan
    
    return results


def compute_ipcw_cindex(survival_train, survival_test, pred_df, landmark_times):
    """
    Compute IPCW C-index at each landmark time.
    
    Parameters:
    -----------
    survival_train : DataFrame
        Training survival data with columns: pid, time, event
    survival_test : DataFrame
        Test survival data with columns: pid, time, event
    pred_df : DataFrame
        Prediction dataframe with columns: pid, tau, t, risk, surv_prob, dataset
        Rows where t is nan contain risk predictions at landmark time tau
    landmark_times : list
        List of landmark times (tau values) to evaluate
    
    Returns:
    --------
    dict : Dictionary mapping tau to C-index value
    """
    results = {}
    
    # Check if we have training data
    if len(survival_train) == 0:
        print(f"Warning: No training data available for IPCW. All patients in test set.")
        for tau in landmark_times:
            results[tau] = np.nan
        return results
    
    for tau in landmark_times:
        # Filter predictions at this landmark time (where t is nan)
        tau_preds = pred_df[(pred_df['tau'] == tau) & (pred_df['t'].isna())].copy()
        
        if len(tau_preds) == 0:
            print(f"Warning: No predictions found for tau={tau}")
            results[tau] = np.nan
            continue
        
        # Merge with ground truth
        merged = tau_preds.merge(survival_test, on='pid', how='inner')
        
        if len(merged) == 0:
            print(f"Warning: No matching patients for tau={tau}")
            results[tau] = np.nan
            continue
        
        # Filter training data to only include times greater than tau
        survival_train_tau = survival_train[survival_train['time'] > tau].copy()
        
        if len(survival_train_tau) == 0:
            print(f"Warning: No training data with time > tau={tau}")
            results[tau] = np.nan
            continue
        
        # Prepare training data for IPCW (used for censoring weight estimation)
        survival_train_tau_copy = survival_train_tau.copy()
        survival_train_tau_copy['event'] = survival_train_tau_copy['event'].astype(bool)
        train_y = Surv.from_dataframe('event', 'time', survival_train_tau_copy)
        
        # Prepare test data for C-index calculation
        merged_copy = merged.copy()
        merged_copy['event'] = merged_copy['event'].astype(bool)
        test_y = Surv.from_dataframe('event', 'time', merged_copy)
        estimate = merged['risk'].values
        
        try:
            # Calculate IPCW concordance index
            c_index = concordance_index_ipcw(train_y, test_y, estimate, tau=None)
            # c_index returns (concordance_index, concordant, discordant, tied_risk, tied_time)
            results[tau] = c_index[0]
            print(f"Tau={tau}: IPCW C-index={c_index[0]:.4f} (n_test={len(merged)}, n_train={len(survival_train_tau)})")
        except Exception as e:
            print(f"Error calculating IPCW C-index for tau={tau}: {str(e)}")
            results[tau] = np.nan
    
    return results


def compute_ipcw_auc(survival_train, survival_test, pred_df, landmark_times):
    """
    Compute IPCW AUC for all pairs (tau, t) where t > tau.
    
    Formula: AUC(t | tau) where tau is the landmark time (conditioning time)
    and t is the evaluation time (future time).
    
    Parameters:
    -----------
    survival_train : DataFrame
        Training survival data with columns: pid, time, event
    survival_test : DataFrame
        Test survival data with columns: pid, time, event
    pred_df : DataFrame
        Prediction dataframe with columns: pid, tau, t, risk, surv_prob, dataset
        Rows where t is nan contain risk predictions at landmark time tau
    landmark_times : list
        List of landmark times (tau values) to evaluate
    
    Returns:
    --------
    dict : Dictionary mapping (tau, t) tuples to AUC value
    """
    results = {}
    
    # Check if we have training data
    if len(survival_train) == 0:
        print(f"Warning: No training data available for IPCW AUC. All patients in test set.")
        return results
    
    # For each tau (landmark time), compute AUC at future times
    for tau_idx, tau in enumerate(landmark_times):
        # Get future times (all landmark times after current tau)
        future_times = [t for t in landmark_times if t > tau]
        
        if len(future_times) == 0:
            print(f"  No future times for tau={tau}, skipping AUC calculation")
            continue
        
        print(f"\n  Computing AUC for tau={tau} at future times: {future_times}")
        
        # Filter predictions at this landmark time (where t is nan)
        tau_preds = pred_df[(pred_df['tau'] == tau) & (pred_df['t'].isna())].copy()
        
        if len(tau_preds) == 0:
            print(f"  Warning: No predictions found for tau={tau}")
            for t in future_times:
                results[(tau, t)] = np.nan
            continue
        
        # Merge with ground truth
        merged = tau_preds.merge(survival_test, on='pid', how='inner')
        
        if len(merged) == 0:
            print(f"  Warning: No matching patients for tau={tau}")
            for t in future_times:
                results[(tau, t)] = np.nan
            continue
        
        # Filter training data to only include times greater than tau
        survival_train_tau = survival_train[survival_train['time'] > tau].copy()
        
        if len(survival_train_tau) == 0:
            print(f"  Warning: No training data with time > tau={tau}")
            for t in future_times:
                results[(tau, t)] = np.nan
            continue
        
        # Prepare training data for IPCW (used for censoring weight estimation)
        survival_train_tau_copy = survival_train_tau.copy()
        survival_train_tau_copy['event'] = survival_train_tau_copy['event'].astype(bool)
        train_y = Surv.from_dataframe('event', 'time', survival_train_tau_copy)
        
        # Prepare test data
        merged_copy = merged.copy()
        merged_copy['event'] = merged_copy['event'].astype(bool)
        test_y = Surv.from_dataframe('event', 'time', merged_copy)
        estimate = merged['risk'].values
        
        try:
            # Calculate cumulative dynamic AUC at all future times
            auc_scores, mean_auc = cumulative_dynamic_auc(
                train_y, test_y, estimate, times=future_times
            )
            
            # Store results for each (tau, t) pair
            for t, auc in zip(future_times, auc_scores):
                results[(tau, t)] = auc
                if np.isnan(auc):
                    print(f"    AUC(t={t} | tau={tau}): NaN")
                else:
                    print(f"    AUC(t={t} | tau={tau}): {auc:.4f}")
                
        except Exception as e:
            msg = str(e)
            if "all times must be within follow-up time of test data" in msg:
                print(f"  IPCW AUC: times out of test follow-up for tau={tau}; setting AUC=0 for all future times")
                for t in future_times:
                    results[(tau, t)] = 0.0
            else:
                print(f"  Error calculating IPCW AUC for tau={tau}: {msg}")
                for t in future_times:
                    results[(tau, t)] = np.nan
    
    return results


def compute_brier_score(survival_train, survival_test, pred_df, landmark_times):
    """
    Compute Brier Score for all pairs (tau, t) where t > tau.
    
    Formula: BS(t | tau) where tau is the landmark time (conditioning time)
    and t is the evaluation time (future time).
    
    Parameters:
    -----------
    survival_train : DataFrame
        Training survival data with columns: pid, time, event
    survival_test : DataFrame
        Test survival data with columns: pid, time, event
    pred_df : DataFrame
        Prediction dataframe with columns: pid, tau, t, risk, surv_prob, dataset
        Rows where risk is nan contain survival probability predictions at time t given tau
    landmark_times : list
        List of landmark times (tau values) to evaluate
    
    Returns:
    --------
    dict : Dictionary mapping (tau, t) tuples to Brier Score value
    """
    results = {}
    
    # Check if we have training data
    if len(survival_train) == 0:
        print(f"Warning: No training data available for Brier Score. All patients in test set.")
        return results
    
    # For each tau (landmark time), compute Brier Score at future times
    for tau_idx, tau in enumerate(landmark_times):
        # Get future times (all landmark times after current tau)
        future_times = [t for t in landmark_times if t > tau]
        
        if len(future_times) == 0:
            print(f"  No future times for tau={tau}, skipping Brier Score calculation")
            continue
        
        print(f"\n  Computing Brier Score for tau={tau} at future times: {future_times}")
        
        # For each future time t, get survival probability predictions
        for t in future_times:
            # Filter predictions at landmark time tau for evaluation time t (where risk is nan)
            tau_t_preds = pred_df[
                (pred_df['tau'] == tau) & 
                (pred_df['t'] == t) & 
                (pred_df['risk'].isna())
            ].copy()
            
            if len(tau_t_preds) == 0:
                print(f"  Warning: No survival probability predictions for tau={tau}, t={t}")
                results[(tau, t)] = np.nan
                continue
            
            # Merge with ground truth
            merged = tau_t_preds.merge(survival_test, on='pid', how='inner')
            
            if len(merged) == 0:
                print(f"  Warning: No matching patients for tau={tau}, t={t}")
                results[(tau, t)] = np.nan
                continue
            
            # Filter training data to only include times greater than tau
            survival_train_tau = survival_train[survival_train['time'] > tau].copy()
            
            if len(survival_train_tau) == 0:
                print(f"  Warning: No training data with time > tau={tau}")
                results[(tau, t)] = np.nan
                continue
            
            # Prepare training data for censoring weight estimation
            survival_train_tau_copy = survival_train_tau.copy()
            survival_train_tau_copy['event'] = survival_train_tau_copy['event'].astype(bool)
            train_y = Surv.from_dataframe('event', 'time', survival_train_tau_copy)
            
            # Prepare test data
            merged_copy = merged.copy()
            merged_copy['event'] = merged_copy['event'].astype(bool)
            test_y = Surv.from_dataframe('event', 'time', merged_copy)
            
            # Get survival probability predictions (convert to numpy array)
            surv_prob = merged['surv_prob'].values
            
            # Reshape to match expected format: (n_samples, n_times)
            # We only have one time point, so reshape to (n_samples, 1)
            surv_prob_array = surv_prob.reshape(-1, 1)
            
            try:
                # Calculate Brier Score at time t
                # Note: brier_score expects survival probabilities, not risk scores
                _, bs = brier_score(train_y, test_y, surv_prob_array, times=[t])
                
                results[(tau, t)] = bs[0]
                print(f"    BS(t={t} | tau={tau}): {bs[0]:.4f}")
                
            except Exception as e:
                msg = str(e)
                if "all times must be within follow-up time of test data" in msg:
                    print(f"  Brier Score: t={t} out of test follow-up for tau={tau}; setting BS=0")
                    results[(tau, t)] = 0.0
                else:
                    print(f"  Error calculating Brier Score for tau={tau}, t={t}: {msg}")
                    results[(tau, t)] = np.nan
    
    return results


def compute_weighted_average_auc(pred_df, auc_ipcw_dict, landmark_times):
    """
    Compute weighted average of AUC over time for each landmark time tau_k.
    
    Formula: AUC_avg(tau_k) = sum(AUC(t | tau_k) * w_t) / sum(w_t)
    where w_t = S(tau_{t-1} | tau_k) - S(tau_t | tau_k)
    and S is the survival function.
    
    Parameters:
    -----------
    pred_df : DataFrame
        Prediction dataframe with columns: pid, tau, t, risk, surv_prob, dataset
    auc_ipcw_dict : dict
        Dictionary mapping (tau, t) tuples to AUC values
    landmark_times : list
        Sorted list of landmark times
    
    Returns:
    --------
    dict : Dictionary mapping tau to weighted average AUC
    """
    results = {}
    
    for k, tau_k in enumerate(landmark_times[:-1]):  # Exclude last tau as it has no future times
        # Get future times (tau_{k+1} to tau_m)
        future_times = [t for t in landmark_times if t > tau_k]
        
        if len(future_times) == 0:
            continue

        # Special case: only one future timepoint (e.g., tau_{k+1} is the last landmark)
        if len(future_times) == 1:
            t_only = future_times[0]
            auc = auc_ipcw_dict.get((tau_k, t_only), 0.0)
            results[tau_k] = 0.0 if auc is None or np.isnan(auc) else float(auc)
            continue

        # For this tau_k, get all patients' survival probabilities at different times
        # We need S(t | tau_k) for each t in future_times
        surv_probs_by_time = {}
        
        for t in future_times:
            # Get survival probabilities at time t given tau_k
            surv_data = pred_df[
                (pred_df['tau'] == tau_k) & 
                (pred_df['t'] == t) & 
                (pred_df['risk'].isna())
            ].copy()
            
            if len(surv_data) > 0:
                # Average survival probability across all patients
                surv_probs_by_time[t] = surv_data['surv_prob'].mean()
        
        # Compute weights w_t = S(tau_{t-1} | tau_k) - S(tau_t | tau_k)
        weights = {}
        weighted_auc_sum = 0
        weight_sum = 0
        
        for i, t in enumerate(future_times):
            # Get S(tau_t | tau_k)
            S_t = surv_probs_by_time.get(t, None)
            
            # Get S(tau_{t-1} | tau_k)
            if i == 0:
                # For the first future time, tau_{t-1} = tau_k, so S(tau_k | tau_k) = 1
                S_t_minus_1 = 1.0
            else:
                t_minus_1 = future_times[i-1]
                S_t_minus_1 = surv_probs_by_time.get(t_minus_1, None)
            
            # Check if we have both survival probabilities
            if S_t is None or S_t_minus_1 is None:
                continue
            
            # Compute weight
            w_t = S_t_minus_1 - S_t
            
            # Get AUC(t | tau_k)
            auc = auc_ipcw_dict.get((tau_k, t), None)
            
            if auc is not None and not np.isnan(auc) and w_t > 0:
                weights[t] = w_t
                weighted_auc_sum += auc * w_t
                weight_sum += w_t
        
        # Compute weighted average
        if weight_sum > 0:
            results[tau_k] = weighted_auc_sum / weight_sum
            print(f"  Weighted Average AUC(tau={tau_k}): {results[tau_k]:.4f}")
        else:
            # No valid weights for this tau_k — use simple mean of AUCs (unweighted)
            aucs = [auc_ipcw_dict.get((tau_k, t)) for t in future_times]
            aucs = [a for a in aucs if a is not None and not np.isnan(a)]
            if len(aucs) > 0:
                results[tau_k] = np.mean(aucs)
                print(f"  Mean AUC(tau={tau_k}): {results[tau_k]:.4f}")
            else:
                results[tau_k] = 0
                print(f"  Mean AUC(tau={tau_k}): 0 (no valid AUCs)")
    return results


def compute_integrated_brier_score(brier_score_dict, landmark_times):
    """
    Compute Integrated Brier Score (IBS) for each landmark time tau_k.
    
    Formula: IBS(tau_k) = integral_{tau_{k+1}}^{tau_m} BS(t | tau_k) dw(t)
    where w(t) = t / tau_max
    
    Using numpy trapezoidal rule for numerical integration.
    
    Parameters:
    -----------
    brier_score_dict : dict
        Dictionary mapping (tau, t) tuples to Brier Score values
    landmark_times : list
        Sorted list of landmark times
    
    Returns:
    --------
    dict : Dictionary mapping tau to integrated Brier Score
    """
    results = {}
    
    if len(landmark_times) == 0:
        return results
    
    tau_max = landmark_times[-1]
    
    for k, tau_k in enumerate(landmark_times[:-1]):  # Exclude last tau as it has no future times
        # Get future times (tau_{k+1} to tau_m)
        future_times = [t for t in landmark_times if t > tau_k]
        
        if len(future_times) == 0:
            continue
        
        # Collect Brier Scores for all future times
        brier_scores = []
        valid_times = []
        
        for t in future_times:
            bs = brier_score_dict.get((tau_k, t), None)
            # Skip if bs is None or NaN
            if bs is not None and not np.isnan(bs):
                brier_scores.append(bs)
                valid_times.append(t)
        
        # If only one future timepoint is available, define IBS(tau_k) = BS(t_K | tau_k)
        if len(valid_times) == 1:
            ibs = float(brier_scores[0])  # only point available
            results[tau_k] = ibs
            print(f"  IBS(tau={tau_k}): {ibs:.4f} (single future timepoint)")
            continue

        # Need at least 2 points for trapezoidal integration
        if len(valid_times) < 2:
            continue
        
        # Compute integral using trapezoidal rule (numpy)
        ibs = np.trapz(brier_scores, valid_times) / tau_max
        
        results[tau_k] = ibs
        print(f"  IBS(tau={tau_k}): {ibs:.4f}")
    
    return results


def main():
    import glob
    
    # Find all model prediction files with k values
    all_model_files = sorted(glob.glob('*risk*.csv'))

    # Hardcoded model allowlist (exactly 10 models)
    allowed_models = {
        'dynamic_mitra',
        'dynamic_mitra_discrete',
        'dynamic_tabpfn',
        'dynamic_tabpfn_discrete',
        'dynamic_xgboost',
        'dynamic_xgboost_discrete',
        'dynamic_deephit',
        'joint_model',
        'landmark_cox',
        'landmark_rsf',
    }

    # Keep only files whose model prefix (before "_risk_") is in the hardcoded list
    all_model_files = [
        f for f in all_model_files
        if f.replace('.csv', '').split('_risk_')[0] in allowed_models
    ]

    # List to store rows for CSV output
    output_rows = []
    
    for model_file in all_model_files:
        print("\n" + "="*70)
        print(f"Processing {model_file}")
        print("="*70)

        # Extract model name and k from patterns like:
        # model_risk_20.csv, model_risk_20.0.csv, model_risk_20_km_quantile.csv
        stem = model_file[:-4] if model_file.endswith('.csv') else model_file
        m = re.match(r'^(?P<model>.+?)_risk_(?P<k>-?\d+(?:\.\d+)?)', stem)
        if m:
            model_name = m.group('model').replace('_', ' ').title()
            k_value = float(m.group('k'))
        else:
            model_name = stem.replace('_', ' ').title()
            k_value = None

        # Skip k == 10
        # if k_value is not None and np.isclose(k_value, 10.0):
        #     print(f"Skipping model {model_name} with k={k_value} (k=10)")
        #     continue

        print(f"Model: {model_name}, k: {k_value}")

        # Load predictions
        pred_df = pd.read_csv(model_file)
        
        # Get unique combinations of dataset and hyperparameter (not eval_set)
        if 'hyperparameter' in pred_df.columns:
            groups = pred_df.groupby(['dataset', 'hyperparameter'])
        else:
            # Fallback if columns don't exist
            print("Warning: 'hyperparameter' column not found")
            groups = pred_df.groupby(['dataset'])
        
        for group_keys, group_pred_df in groups:
            # Unpack group keys
            if isinstance(group_keys, tuple):
                dataset_name, hyperparameter = group_keys
            else:
                dataset_name = group_keys
                hyperparameter = None
            
            print("\n" + "-"*70)
            print(f"Dataset: {dataset_name}, Hyperparameter: {hyperparameter}")
            print("-"*70)
            
            # Separate val and test data
            val_pred_df = group_pred_df[group_pred_df['eval_set'] == 'val'].copy()
            test_pred_df = group_pred_df[group_pred_df['eval_set'] == 'test'].copy()
            
            if len(val_pred_df) == 0 or len(test_pred_df) == 0:
                print(f"Warning: Missing val or test data for {dataset_name}")
                continue
            
            # Load ground truth survival data for this dataset (if exists in a data folder)
            survival_path = f'data/{dataset_name}.csv'
            print(f"Loading ground truth from {survival_path}")
            try:
                survival_all = load_survival_dataset(survival_path)
                print(f"Loaded {len(survival_all)} patients total")
            except FileNotFoundError:
                print(f"Warning: File not found, skipping dataset {dataset_name}")
                continue
            
            # Process val and test separately
            for eval_set, dataset_pred_df in [('val', val_pred_df), ('test', test_pred_df)]:
                print(f"\n  Processing {eval_set} set...")
                
                eval_pids = set(dataset_pred_df['pid'].unique())

                val_pids  = set(val_pred_df['pid'].unique())
                test_pids = set(test_pred_df['pid'].unique())

                all_pids = set(survival_all['pid'].unique())
                train_pids = all_pids - (val_pids | test_pids)

                survival_eval = survival_all[survival_all['pid'].isin(eval_pids)].copy()
                survival_train = survival_all[survival_all['pid'].isin(train_pids)].copy()
                
                # Truncate test/val patients whose observed time exceeds max training time
                # This is necessary for IPCW calculations which require all observation times
                # to be within the range of the training data
                max_train_time = survival_train['time'].max()
                survival_eval_filtered = survival_eval.copy()
                
                # Truncate times and censor events that exceed max_train_time
                n_truncated = (survival_eval_filtered['time'] > max_train_time).sum()
                survival_eval_filtered.loc[survival_eval_filtered['time'] > max_train_time, 'event'] = 0
                survival_eval_filtered.loc[survival_eval_filtered['time'] > max_train_time, 'time'] = max_train_time
                
                if n_truncated > 0:
                    print(f"  Truncated {n_truncated}/{len(survival_eval)} {eval_set} patients with time > max_train_time={max_train_time:.3f}")
                
                print(f"  Eval patients ({eval_set}): {len(survival_eval_filtered)}, Train patients: {len(survival_train)}")
                
                # Identify unique landmark times for this dataset
                # Get all unique tau values (where t is NaN) - these are the conditioning times with risk predictions
                tau_values = set(dataset_pred_df[dataset_pred_df['t'].isna()]['tau'].unique())
                # Get all unique t values (where t is not NaN) - these are additional evaluation times
                t_values = set(dataset_pred_df[dataset_pred_df['t'].notna()]['t'].unique())
                # Combine them to get all landmark times (for AUC and Brier Score)
                landmark_times = sorted(tau_values.union(t_values))
                # For C-index, only use tau values where we have risk predictions
                tau_times = sorted(tau_values)
                print(f"  Found {len(tau_times)} tau times for C-index: {tau_times}")
                print(f"  Found {len(landmark_times)} landmark times for AUC/Brier: {landmark_times}")
                
                # Calculate Harrell's C-index (only at tau times with risk predictions)
                # print(f"\n  Calculating Harrell's C-index for {eval_set}...")
                # c_indices_harrell = compute_harrel_cindex(survival_eval, dataset_pred_df, tau_times)
                
                # Calculate IPCW C-index (only at tau times with risk predictions)
                print(f"\n  Calculating IPCW C-index for {eval_set}...")
                c_indices_ipcw = compute_ipcw_cindex(survival_train, survival_eval_filtered, dataset_pred_df, tau_times)
                
                # Calculate IPCW AUC
                print(f"\n  Calculating IPCW AUC for {eval_set}...")
                auc_ipcw = compute_ipcw_auc(survival_train, survival_eval_filtered, dataset_pred_df, landmark_times)
                
                # Calculate Brier Score
                print(f"\n  Calculating Brier Score for {eval_set}...")
                brier_scores = compute_brier_score(survival_train, survival_eval_filtered, dataset_pred_df, landmark_times)
                
                # Calculate Weighted Average AUC
                print(f"\n  Calculating Weighted Average AUC for {eval_set}...")
                weighted_avg_auc = compute_weighted_average_auc(dataset_pred_df, auc_ipcw, landmark_times)
                
                # Calculate Integrated Brier Score
                print(f"\n  Calculating Integrated Brier Score for {eval_set}...")
                integrated_brier = compute_integrated_brier_score(brier_scores, landmark_times)
                
                # Store results with eval_set suffix
                results_dict = {
                    # f'{eval_set}_cindex_harrell': c_indices_harrell,
                    f'{eval_set}_cindex_ipcw': c_indices_ipcw,
                    # f'{eval_set}_auc_ipcw': auc_ipcw,
                    # f'{eval_set}_brier_score': brier_scores,
                    f'{eval_set}_weighted_avg_auc': weighted_avg_auc,
                    f'{eval_set}_integrated_brier': integrated_brier
                }
                
                # Store in a temporary structure
                if eval_set == 'val':
                    val_results = results_dict
                else:
                    test_results = results_dict
            
            # Now combine val and test results into single rows
            print("\n  Combining val and test results...")
            
            # Add Harrell's C-index results
            # for tau in set(list(val_results.get('val_cindex_harrell', {}).keys()) + list(test_results.get('test_cindex_harrell', {}).keys())):
            #     val_score = val_results.get('val_cindex_harrell', {}).get(tau, np.nan)
            #     test_score = test_results.get('test_cindex_harrell', {}).get(tau, np.nan)
            #     output_rows.append({
            #         'model': model_name,
            #         'dataset': dataset_name,
            #         'hyperparameter': str(hyperparameter) if hyperparameter is not None else '',
            #         'k': k_value if k_value is not None else '',
            #         'metric': 'cindex_harrell',
            #         'tau': round(tau, 6),
            #         't': np.nan,
            #         'val_score': val_score,
            #         'test_score': test_score
            #     })
            
            # Add IPCW C-index results
            for tau in set(list(val_results['val_cindex_ipcw'].keys()) + list(test_results['test_cindex_ipcw'].keys())):
                val_score = val_results['val_cindex_ipcw'].get(tau, np.nan)
                test_score = test_results['test_cindex_ipcw'].get(tau, np.nan)
                output_rows.append({
                    'model': model_name,
                    'dataset': dataset_name,
                    'hyperparameter': str(hyperparameter) if hyperparameter is not None else '',
                    'k': k_value if k_value is not None else '',
                    'metric': 'cindex_ipcw',
                    'tau': round(tau, 6),
                    't': np.nan,
                    'val_score': val_score,
                    'test_score': test_score
                })
            
            # Add IPCW AUC results
            # all_auc_keys = set(list(val_results.get('val_auc_ipcw', {}).keys()) + list(test_results.get('test_auc_ipcw', {}).keys()))
            # for (tau, t) in all_auc_keys:
            #     val_score = val_results.get('val_auc_ipcw', {}).get((tau, t), np.nan)
            #     test_score = test_results.get('test_auc_ipcw', {}).get((tau, t), np.nan)
            #     output_rows.append({
            #         'model': model_name,
            #         'dataset': dataset_name,
            #         'hyperparameter': str(hyperparameter) if hyperparameter is not None else '',
            #         'k': k_value if k_value is not None else '',
            #         'metric': 'auc_ipcw',
            #         'tau': round(tau, 6),
            #         't': round(t, 6),
            #         'val_score': val_score,
            #         'test_score': test_score
            #     })
            
            # Add Brier Score results
            # all_bs_keys = set(list(val_results.get('val_brier_score', {}).keys()) + list(test_results.get('test_brier_score', {}).keys()))
            # for (tau, t) in all_bs_keys:
            #     val_score = val_results.get('val_brier_score', {}).get((tau, t), np.nan)
            #     test_score = test_results.get('test_brier_score', {}).get((tau, t), np.nan)
            #     output_rows.append({
            #         'model': model_name,
            #         'dataset': dataset_name,
            #         'hyperparameter': str(hyperparameter) if hyperparameter is not None else '',
            #         'k': k_value if k_value is not None else '',
            #         'metric': 'brier_score',
            #         'tau': round(tau, 6),
            #         't': round(t, 6),
            #         'val_score': val_score,
            #         'test_score': test_score
            #     })
            
            # Add Weighted Average AUC results
            for tau in set(list(val_results['val_weighted_avg_auc'].keys()) + list(test_results['test_weighted_avg_auc'].keys())):
                val_score = val_results['val_weighted_avg_auc'].get(tau, np.nan)
                test_score = test_results['test_weighted_avg_auc'].get(tau, np.nan)
                output_rows.append({
                    'model': model_name,
                    'dataset': dataset_name,
                    'hyperparameter': str(hyperparameter) if hyperparameter is not None else '',
                    'k': k_value if k_value is not None else '',
                    'metric': 'weighted_avg_auc',
                    'tau': round(tau, 6),
                    't': np.nan,
                    'val_score': val_score,
                    'test_score': test_score
                })
            
            # Add Integrated Brier Score results
            for tau in set(list(val_results['val_integrated_brier'].keys()) + list(test_results['test_integrated_brier'].keys())):
                val_score = val_results['val_integrated_brier'].get(tau, np.nan)
                test_score = test_results['test_integrated_brier'].get(tau, np.nan)
                output_rows.append({
                    'model': model_name,
                    'dataset': dataset_name,
                    'hyperparameter': str(hyperparameter) if hyperparameter is not None else '',
                    'k': k_value if k_value is not None else '',
                    'metric': 'integrated_brier',
                    'tau': round(tau, 6),
                    't': np.nan,
                    'val_score': val_score,
                    'test_score': test_score
                })
    
    # Save to CSV
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv('metrics.csv', index=False)
    print("\n" + "="*70)
    print(f"Results saved to metrics.csv ({len(output_df)} rows)")
    print(f"Unique models: {sorted(output_df['model'].unique())}")
    print(f"Unique datasets: {sorted(output_df['dataset'].unique())}")
    print(f"Unique k values: {sorted(output_df['k'].unique())}")
    print("="*70)


if __name__ == "__main__":
    main()