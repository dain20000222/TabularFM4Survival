# REPORT copy (2026-10-18) of ../tmlr_dyn_metrics_fix_10_18/metrics_val.py (NANFIX), unchanged.
# (copy of tmlr_datamean_rerun_10_11/tcsa/metrics_val.py; output metrics_val.csv)
# NANFIX (2026-10-18): the *_all lists keep NaN entries (landmarks whose metric could not be computed);
# the averages and the hyperparameter selection drop them, as before.
import pandas as pd
import numpy as np


def select_best_hyperparameters(metrics_csv='metrics.csv', val_metric='cindex_ipcw', output_csv='best_hyperparameters.csv'):
    """
    Select the best hyperparameter for each {model, dataset, k} combination based on validation metrics.
    
    The selection is done by:
    1. For each {model, dataset, k} combination, get all val_scores for the specified metric
    2. Calculate the average val_score across all (tau, t) combinations for each hyperparameter
    3. Select the hyperparameter with the best average val_score (highest for c-index/AUC, lowest for Brier)
    
    Parameters:
    -----------
    metrics_csv : str
        Path to the metrics CSV file
    val_metric : str
        The metric to use for hyperparameter selection (e.g., 'cindex_harrell', 'cindex_ipcw', 'auc_ipcw', 'brier_score')
    output_csv : str
        Path to save the output CSV file with best hyperparameters
    """
    
    # Load metrics
    print(f"Loading metrics from {metrics_csv}...")
    df = pd.read_csv(metrics_csv)
    
    print(f"Total rows: {len(df)}")
    print(f"Available metrics: {sorted(df['metric'].unique())}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    
    # Filter for the specified metric
    metric_df = df[df['metric'] == val_metric].copy()
    
    if len(metric_df) == 0:
        print(f"Error: No data found for metric '{val_metric}'")
        return
    
    print(f"\nFiltered to {len(metric_df)} rows for metric '{val_metric}'")
    
    # Determine if higher is better or lower is better
    # For c-index and AUC metrics, higher is better
    # For Brier score and integrated Brier, lower is better
    higher_is_better = val_metric in ['cindex_harrell', 'cindex_ipcw', 'auc_ipcw', 'weighted_avg_auc']
    
    print(f"Metric optimization: {'maximize' if higher_is_better else 'minimize'}")
    
    # Results list
    results = []
    
    # Group by model, dataset, and k
    for (model, dataset, k), group in metric_df.groupby(['model', 'dataset', 'k']):
        print(f"\n{'='*70}")
        print(f"Processing: Model={model}, Dataset={dataset}, k={k}")
        print(f"{'='*70}")
        
        # Get unique hyperparameters
        hyperparameters = group['hyperparameter'].unique()
        print(f"Hyperparameters: {len(hyperparameters)}")
        
        # For each hyperparameter, calculate average val_score across all (tau, t) combinations
        hp_avg_scores = {}
        for hp in hyperparameters:
            hp_data = group[group['hyperparameter'] == hp]
            # Get all val_scores for this hyperparameter (across different tau, t combinations)
            scores = hp_data['val_score'].dropna()
            if len(scores) > 0:
                hp_avg_scores[hp] = scores.mean()
        
        if len(hp_avg_scores) == 0:
            print(f"  Warning: Could not calculate average scores for any hyperparameter")
            continue
        
        # Select hyperparameter with best average score
        if higher_is_better:
            # Higher is better: select max
            best_hp = max(hp_avg_scores.items(), key=lambda x: x[1])
        else:
            # Lower is better: select min
            best_hp = min(hp_avg_scores.items(), key=lambda x: x[1])
        
        best_hp_name = best_hp[0]
        best_hp_avg_score = best_hp[1]
        
        print(f"  Best hyperparameter: {best_hp_name}")
        print(f"  Average val_score: {best_hp_avg_score:.4f}")
        
        # Print all hyperparameters for comparison
        print("\n  All hyperparameters (sorted by avg val_score):")
        sorted_hps = sorted(hp_avg_scores.items(), key=lambda x: x[1], reverse=higher_is_better)
        for hp, avg_score in sorted_hps:
            print(f"    {hp}: avg_val_score={avg_score:.4f}")
        
        # Get statistics for best hyperparameter
        best_hp_data = group[group['hyperparameter'] == best_hp_name]
        val_scores = best_hp_data['val_score'].dropna()
        
        # The best_val_score is already calculated as the average
        best_val_score = best_hp_avg_score
        
        # Get test scores for all 3 metrics using the best hyperparameter
        all_metrics = [
            # 'cindex_harrell', 
            'cindex_ipcw', 
            # 'auc_ipcw', 
            # 'brier_score', 
            'weighted_avg_auc', 
            'integrated_brier'
        ]
        test_scores = {}
        
        for metric in all_metrics:
            # Filter the full dataframe for this model, dataset, k, hyperparameter, and metric
            test_data = df[
                (df['model'] == model) & 
                (df['dataset'] == dataset) & 
                (df['k'] == k) & 
                (df['hyperparameter'] == best_hp_name) & 
                (df['metric'] == metric)
            ]
            
            if len(test_data) > 0:
                test_vals = test_data['test_score'].dropna()
                # Compute average test score across all (tau, t) combinations
                test_scores[f'avg_{metric}'] = test_vals.mean() if len(test_vals) > 0 else np.nan
                # Store all test scores as a list
                test_scores[f'{metric}_all'] = test_data['test_score'].tolist()  # NANFIX: keep NaN so that the list stays aligned with the landmarks
            else:
                test_scores[f'avg_{metric}'] = np.nan
                test_scores[f'{metric}_all'] = []
        
        results.append({
            'model': model,
            'dataset': dataset,
            'k': k,
            #'best_hyperparameter': best_hp_name,
            #'avg_val_score': best_val_score,
            **test_scores  # Add all test scores
        })
    
    # Create output DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV only if output_csv is provided
    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"\n{'='*70}")
        print(f"Results saved to {output_csv}")
        print(f"Total combinations: {len(results_df)}")
        print(f"{'='*70}")
    
    # Print summary statistics
    if len(results_df) > 0:
        print("\nSummary by model:")
        for model in sorted(results_df['model'].unique()):
            model_df = results_df[results_df['model'] == model]
            print(f"  {model}: {len(model_df)} dataset-k combinations")
    
    return results_df


def main():
    """
    Main function to run hyperparameter selection for all validation metrics.
    """
    # Define all metrics to process
    all_metrics = [
        # 'cindex_harrell', 
        'cindex_ipcw', 
        # 'auc_ipcw', 
        # 'brier_score', 
        #'weighted_avg_auc', 
        #'integrated_brier'
    ]
    
    # Process all metrics
    print("\n" + "="*70)
    print("Processing all 3 metrics")
    print("="*70 + "\n")
    
    all_results = []
    
    for metric in all_metrics:
        print(f"\n{'#'*70}")
        print(f"# Processing metric: {metric}")
        print(f"{'#'*70}\n")
        
        results_df = select_best_hyperparameters(
            metrics_csv='metrics.csv',
            val_metric=metric,
            output_csv=None  # Don't save individual files
        )
        
        # Add val_metric column
        if results_df is not None and len(results_df) > 0:
            results_df['val_metric'] = metric
            all_results.append(results_df)
    
    # Combine all results
    if len(all_results) > 0:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Reorder columns to put val_metric near the front
        cols = combined_df.columns.tolist()
        # Move val_metric after model, dataset, k
        if 'val_metric' in cols:
            cols.remove('val_metric')
            cols.insert(3, 'val_metric')  # Insert after model, dataset, k
            combined_df = combined_df[cols]
        
        # Save to single file
        output_file = 'metrics_val.csv'  # IPCW (was metrics_val_tcsa.csv)
        combined_df.to_csv(output_file, index=False)
        
        print("\n" + "="*70)
        print("All metrics processed successfully!")
        print("="*70)
        print(f"\nResults saved to: {output_file}")
        print(f"Total rows: {len(combined_df)}")
        print(f"\nRows per metric:")
        for metric in all_metrics:
            count = len(combined_df[combined_df['val_metric'] == metric])
            print(f"  {metric}: {count}")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
