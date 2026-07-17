# =============================================================================
# Static Baseline Survival Analysis
# =============================================================================
# This script performs comprehensive survival analysis using baseline models:
# - Cox Proportional Hazards (CoxPH)
# - Random Survival Forest (RSF)
# - DeepHit (neural network for competing risks)
# - DeepSurv (neural Cox model)
# =============================================================================

import pandas as pd
import numpy as np
import os
import csv
import argparse 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sksurv.util import Surv
from sklearn.pipeline import make_pipeline
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored, integrated_brier_score, cumulative_dynamic_auc, concordance_index_ipcw
from sksurv.nonparametric import kaplan_meier_estimator
import torch
from pycox.models import DeepHitSingle, CoxPH
import torchtuples as tt
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from itertools import product
import random
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Configuration and Setup
# =============================================================================

# Pandas 2.x compatibility shim for scikit-survival
if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items

# Set random seeds for reproducibility across all libraries
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run baseline survival analysis with different evaluation time configurations')
parser.add_argument('--k', type=float, default=10.0, help='Step size for evaluation time percentiles (default: 10.0)')
parser.add_argument('--val_metric', type=str, default='ipcw_c_index',
                    choices=['c_index', 'ipcw_c_index', 'ibs', 'mean_auc'],
                    help='Validation metric for hyperparameter selection (default: c_index)')
parser.add_argument('--binning', type=str, default='quantile', choices=['quantile', 'equal_width', 'km_quantile'],
                    help='Binning strategy: quantile (default), equal_width, or km_quantile (KM-weighted)')
args = parser.parse_args()

# =============================================================================
# Data Configuration
# =============================================================================

# Directory containing the datasets
data_dir = os.path.join("static_data")

# List all CSV files in the directory (sorted for consistent processing order)
dataset_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

# Output CSV path based on k parameter and validation metric
binning_suffix = f"_{args.binning}" if args.binning != "quantile" else ""
k_token = f"{float(args.k):.8g}"
csv_path = f"static_baseline_evaluation_final_{k_token}_{args.val_metric}{binning_suffix}.csv"


def k_to_percentile_grid(k: float, tol: float = 1e-6):
    kf = float(k)
    if not (0.0 < kf < 100.0):
        raise ValueError(f"k must be in (0, 100), got {k}")
    K = 100.0 / kf
    K_int = int(round(K))
    if np.isclose(K, K_int, rtol=0.0, atol=tol) and K_int >= 2:
        percentiles = (np.arange(1, K_int) * (100.0 / K_int))
        return percentiles, (K_int - 1), K_int
    percentiles = np.arange(kf, 100.0, kf)
    return percentiles, int(len(percentiles)), None

# =============================================================================
# Hyperparameter Grid Configurations
# =============================================================================

# CoxPH hyperparameter grid - regularization parameter alpha
cph_grid = {
    'alpha': [1e-6, 1e-3, 1e-1]
}

# Random Survival Forest hyperparameter grid
rsf_grid = {
    'n_estimators': [50, 100, 200],          # Number of trees in the forest
    'min_samples_split': [2, 5, 10, 20]      # Minimum samples required to split a node
}

# DeepHit hyperparameter grid - neural network for competing risks
deephit_grid = {
    'dropout': [0, 0.1],                     # Dropout rate for regularization
    'first_layer_size': [128, 256, 512],     # Size of first hidden layer
    'second_layer_size': [32],               # Size of second hidden layer
    'learning_rate': [1e-3, 1e-4, 1e-5]      # Learning rate for optimizer
}

# DeepSurv hyperparameter grid - neural Cox model
deepsurv_grid = {
    'dropout': [0, 0.1],                     # Dropout rate for regularization
    'first_layer_size': [128, 256, 512],     # Size of first hidden layer
    'second_layer_size': [32],               # Size of second hidden layer
    'learning_rate': [1e-3, 1e-4, 1e-5]     # Learning rate for optimizer
}

# =============================================================================
# Evaluation Metrics Functions
# =============================================================================

def compute_c_index(model, x_test, y_test, is_deephit=False, is_deepsurv=False):
    """
    Compute concordance index (C-index) for survival models.
    
    The C-index measures the proportion of all pairs of subjects whose
    survival times can be ordered, where the predicted risk scores 
    agree with the observed survival order.
    
    Parameters:
    -----------
    model : object
        The fitted survival model to evaluate
    x_test : array-like, shape (n_samples, n_features)
        Test features
    y_test : structured array, shape (n_samples,)
        Test survival data with 'time' and 'event' fields
    is_deephit : bool, optional (default=False)
        Whether the model is a DeepHit model
    is_deepsurv : bool, optional (default=False)
        Whether the model is a DeepSurv model
    
    Returns:
    --------
    c_index : float
        The concordance index value
    """
    # Handle different model types for risk score computation
    if is_deephit:
        # DeepHit: Convert tensor input if needed and compute expected survival time
        if isinstance(x_test, torch.Tensor):
            x_test_np = x_test.cpu().numpy()
        else:
            x_test_np = x_test
            
        deephit_pmf = model.predict_pmf(x_test_np)
        if isinstance(deephit_pmf, torch.Tensor):
            deephit_pmf = deephit_pmf.detach().cpu().numpy()
        time_bins = np.arange(1, deephit_pmf.shape[1] + 1)
        expected_time = np.sum(deephit_pmf * time_bins, axis=1)
        risk_scores = -expected_time  # Negative because higher risk = lower survival time

    elif is_deepsurv:
        # DeepSurv: Handle tensor input and extract risk scores
        if isinstance(x_test, torch.Tensor):
            risk_scores = model.predict(x_test)
            if isinstance(risk_scores, torch.Tensor):
                risk_scores = risk_scores.detach().cpu().numpy()
            risk_scores = risk_scores.flatten()
        else:
            risk_scores = model.predict(x_test).flatten()

    else:
        # Standard survival models (CoxPH, RSF)
        risk_scores = model.predict(x_test)

    # Compute concordance index using scikit-survival
    c_index, *_ = concordance_index_censored(
        y_test["event"], y_test["time"], risk_scores
    )
    return float(c_index)

def compute_ipcw_c_index(model, x_test, y_test, y_train, times, is_deephit=False, is_deepsurv=False):
    """
    Compute Inverse Probability of Censoring Weighted (IPCW) concordance index.
    
    The IPCW C-index adjusts for informative censoring by weighting observations
    based on the inverse probability of being censored.
    
    Parameters:
    -----------
    model : object
        The fitted survival model to evaluate
    x_test : array-like, shape (n_samples, n_features)
        Test features
    y_test : structured array, shape (n_samples,)
        Test survival data with 'time' and 'event' fields
    y_train : structured array, shape (n_samples,)
        Training survival data for censoring estimation
    times : array-like
        Time points for evaluation (not used in current implementation)
    is_deephit : bool, optional (default=False)
        Whether the model is a DeepHit model
    is_deepsurv : bool, optional (default=False)
        Whether the model is a DeepSurv model
    
    Returns:
    --------
    ipcw_c_index : float
        The IPCW concordance index value
    """
    # Use the same risk score calculation as standard C-index for consistency
    if is_deephit:
        # DeepHit: Handle tensor input and compute expected survival time
        if isinstance(x_test, torch.Tensor):
            x_test_np = x_test.cpu().numpy()
        else:
            x_test_np = x_test
            
        deephit_pmf = model.predict_pmf(x_test_np)
        if isinstance(deephit_pmf, torch.Tensor):
            deephit_pmf = deephit_pmf.detach().cpu().numpy()
        time_bins = np.arange(1, deephit_pmf.shape[1] + 1)
        expected_time = np.sum(deephit_pmf * time_bins, axis=1)
        risk_scores = -expected_time

    elif is_deepsurv:
        # DeepSurv: Handle tensor input and extract risk scores
        if isinstance(x_test, torch.Tensor):
            risk_scores = model.predict(x_test)
            if isinstance(risk_scores, torch.Tensor):
                risk_scores = risk_scores.detach().cpu().numpy()
            risk_scores = risk_scores.flatten()
        else:
            risk_scores = model.predict(x_test).flatten()

    else:
        # Standard survival models
        risk_scores = model.predict(x_test)

    # Use maximum training time as tau for proper truncation
    tau = y_train["time"].max()
    
    # Compute IPCW C-index
    ipcw_c_index, *_ = concordance_index_ipcw(y_train, y_test, risk_scores, tau=tau)
    return float(ipcw_c_index)

def compute_ibs(model, x_test, y_test, y_train, times, is_deephit=False, is_deepsurv=False, labtrans=None):
    """
    Compute Integrated Brier Score (IBS) for survival models.
    
    The IBS measures the mean squared difference between predicted survival
    probabilities and observed survival outcomes over time.
    
    Parameters:
    -----------
    model : object
        The fitted survival model to evaluate
    x_test : array-like, shape (n_samples, n_features)
        Test features
    y_test : structured array, shape (n_samples,)
        Test survival data with 'time' and 'event' fields
    y_train : structured array, shape (n_samples,)
        Training survival data for baseline estimation
    times : array-like, shape (n_times,)
        Time points at which to evaluate the Brier score
    is_deephit : bool, optional (default=False)
        Whether the model is a DeepHit model
    is_deepsurv : bool, optional (default=False)
        Whether the model is a DeepSurv model
    labtrans : LabTransDiscreteTime, optional
        Label transformer for DeepHit (required if is_deephit=True)
    
    Returns:
    --------
    ibs : float
        The integrated Brier score value
    """
    # Generate survival probability predictions based on model type
    if is_deephit:
        # DeepHit: Get survival probabilities and interpolate to evaluation times
        deephit_surv_probs = model.predict_surv_df(x_test).T.values
        surv_probs = np.array([np.interp(times, labtrans.cuts, sp) for sp in deephit_surv_probs])

    elif is_deepsurv:
        # DeepSurv: Compute baseline hazards and get survival function
        model.compute_baseline_hazards()
        deepsurv_surv = model.predict_surv_df(x_test)
        deepsurv_surv_probs = deepsurv_surv.T.values
        surv_probs = np.array([np.interp(times, deepsurv_surv.index, sp) for sp in deepsurv_surv_probs])

    else:
        # Standard survival models: Extract survival functions
        surv_funcs = model.predict_survival_function(x_test)
        surv_probs = np.row_stack([sf(times) for sf in surv_funcs])

    # Compute integrated Brier score
    ibs = integrated_brier_score(y_train, y_test, surv_probs, times)
    return float(ibs)


def compute_mean_auc(model, x_test, y_test, y_train, times, is_deephit=False, is_deepsurv=False):
    """
    Compute mean time-dependent Area Under the Curve (AUC) using scalar risk scores.
    
    The time-dependent AUC measures the discriminatory ability of the model
    at specific time points. This function uses the same risk score calculation
    as the C-index for consistency.
    
    Parameters:
    -----------
    model : object
        The fitted survival model to evaluate
    x_test : array-like, shape (n_samples, n_features)
        Test features
    y_test : structured array, shape (n_samples,)
        Test survival data with 'time' and 'event' fields
    y_train : structured array, shape (n_samples,)
        Training survival data for baseline estimation
    times : array-like, shape (n_times,)
        Time points at which to evaluate the AUC
    is_deephit : bool, optional (default=False)
        Whether the model is a DeepHit model
    is_deepsurv : bool, optional (default=False)
        Whether the model is a DeepSurv model
    
    Returns:
    --------
    mean_auc : float
        The mean AUC value across all time points
    """
    # Use the same risk score calculation as C-index for consistency
    if is_deephit:
        # DeepHit: Handle tensor input and compute expected survival time
        if isinstance(x_test, torch.Tensor):
            x_test_np = x_test.cpu().numpy()
        else:
            x_test_np = x_test
            
        deephit_pmf = model.predict_pmf(x_test_np)
        if isinstance(deephit_pmf, torch.Tensor):
            deephit_pmf = deephit_pmf.detach().cpu().numpy()
        time_bins = np.arange(1, deephit_pmf.shape[1] + 1)
        expected_time = np.sum(deephit_pmf * time_bins, axis=1)
        risk_scores = -expected_time

    elif is_deepsurv:
        # DeepSurv: Handle tensor input and extract risk scores
        if isinstance(x_test, torch.Tensor):
            risk_scores = model.predict(x_test)
            if isinstance(risk_scores, torch.Tensor):
                risk_scores = risk_scores.detach().cpu().numpy()
            risk_scores = risk_scores.flatten()
        else:
            risk_scores = model.predict(x_test).flatten()

    else:
        # Standard survival models
        risk_scores = model.predict(x_test)

    # Compute cumulative dynamic AUC and return the mean
    _, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)

    return mean_auc

def evaluate_model_fold(model, x_test, y_test, y_train, eval_times, model_type=None, labtrans=None):
    """
    Evaluate a survival model and return all performance metrics.
    
    This is a convenience function that computes all four evaluation metrics
    (C-index, IPCW C-index, IBS, and mean AUC) for a given model and test set.
    
    Parameters:
    -----------
    model : object
        The fitted survival model to evaluate
    x_test : array-like, shape (n_samples, n_features)
        Test features
    y_test : structured array, shape (n_samples,)
        Test survival data with 'time' and 'event' fields
    y_train : structured array, shape (n_samples,)
        Training survival data for baseline estimation
    eval_times : array-like
        Time points for evaluation
    model_type : str, optional
        Type of model ('deephit', 'deepsurv', or None for standard models)
    labtrans : LabTransDiscreteTime, optional
        Label transformer for DeepHit models
    
    Returns:
    --------
    results : dict
        Dictionary containing all evaluation metrics:
        - 'c_index': Concordance index
        - 'ipcw_c_index': IPCW concordance index
        - 'ibs': Integrated Brier score
        - 'mean_auc': Mean time-dependent AUC
    """
    # Determine model type flags
    is_deephit = model_type == 'deephit'
    is_deepsurv = model_type == 'deepsurv'
    
    # Compute all evaluation metrics
    c_index = compute_c_index(model, x_test, y_test, is_deephit, is_deepsurv)
    ipcw_c_index = compute_ipcw_c_index(model, x_test, y_test, y_train, eval_times, is_deephit, is_deepsurv)
    ibs = compute_ibs(model, x_test, y_test, y_train, eval_times, is_deephit, is_deepsurv, labtrans)
    mean_auc = compute_mean_auc(model, x_test, y_test, y_train, eval_times, is_deephit, is_deepsurv)
    
    return {
        'c_index': float(c_index),
        'ipcw_c_index': float(ipcw_c_index),
        'ibs': float(ibs),
        'mean_auc': float(mean_auc)
    }

# =============================================================================
# Hyperparameter Optimization Functions
# =============================================================================

def grid_search_cph(x_train, y_train, x_val, y_val, param_grid, eval_times, val_metric='c_index'):
    """
    Perform grid search hyperparameter optimization for Cox Proportional Hazards model.
    
    This function searches over different values of the regularization parameter
    (alpha) to find the best performing CoxPH model based on validation metrics.
    
    Parameters:
    -----------
    x_train : array-like, shape (n_samples, n_features)
        Training features
    y_train : structured array, shape (n_samples,)
        Training survival data with 'time' and 'event' fields
    x_val : array-like, shape (n_samples, n_features)
        Validation features
    y_val : structured array, shape (n_samples,)
        Validation survival data with 'time' and 'event' fields
    param_grid : dict
        Dictionary containing hyperparameters to search over
    eval_times : array-like
        Evaluation time points to use for validation
    val_metric : str, optional (default='c_index')
        Validation metric to optimize ('c_index', 'ipcw_c_index', 'ibs', 'mean_auc')
    
    Returns:
    --------
    best_params : dict
        Best hyperparameters found during grid search
    best_score : float
        Best score achieved with the optimal hyperparameters
    """
    # Initialize best score based on metric type (IBS should be minimized, others maximized)
    best_score = np.inf if val_metric == 'ibs' else -np.inf
    best_params = None
    
    # Iterate through all parameter combinations
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        try:
            # Create and fit CoxPH model with current parameters
            model = make_pipeline(
                StandardScaler(),
                CoxPHSurvivalAnalysis(alpha=param_dict['alpha'])
            )
            model.fit(x_train, y_train)
    
            # Skip if no valid evaluation times
            if len(eval_times) == 0:
                continue
                
            # Evaluate model on validation set
            val_results = evaluate_model_fold(model, x_val, y_val, y_train, eval_times)
            score = val_results[val_metric]
            
            # Update best parameters based on metric type
            if val_metric == 'ibs':
                # For IBS, lower is better
                if score < best_score:
                    best_score = score
                    best_params = param_dict
            else:
                # For other metrics, higher is better
                if score > best_score:
                    best_score = score
                    best_params = param_dict
                    
        except Exception as e:
            print(f"CoxPH trial failed with params {param_dict}: {e}")
            continue
    
    return best_params, best_score

def grid_search_rsf(x_train, y_train, x_val, y_val, param_grid, eval_times, val_metric='c_index'):
    """
    Grid search for Random Survival Forest hyperparameters.

    Parameters
    - x_train : array-like, shape (n_samples, n_features)
    - y_train : structured array, shape (n_samples,)
    - x_val : array-like, shape (n_samples, n_features)
    - y_val : structured array, shape (n_samples,)
    - param_grid : dict
        Dictionary containing hyperparameters to search over.
    - eval_times : array-like
        Evaluation time points to use for validation
    - val_metric : str
        Validation metric to optimize ('c_index', 'ipcw_c_index', 'ibs', 'mean_auc')

    Returns
    - best_params: dict
        Best hyperparameters found.
    - best_score: float
        Best score achieved with the best hyperparameters.
    """
    # For IBS, we want to minimize (lower is better), so we use negative values for comparison
    best_score = np.inf if val_metric == 'ibs' else -np.inf
    best_params = None
    
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        try:
            model = make_pipeline(
                StandardScaler(),
                RandomSurvivalForest(
                    n_estimators=param_dict['n_estimators'],
                    min_samples_split=param_dict['min_samples_split'],
                    random_state=SEED
                )
            )
            model.fit(x_train, y_train)
            
            # Get RSF-specific valid times based on the fitted model
            rsf_eval_times = get_valid_eval_times(eval_times, y_val["time"], model=model)
            if len(rsf_eval_times) == 0:
                continue
                
            val_results = evaluate_model_fold(model, x_val, y_val, y_train, rsf_eval_times)
            score = val_results[val_metric]
            
            # For IBS, lower is better; for others, higher is better
            if val_metric == 'ibs':
                if score < best_score:
                    best_score = score
                    best_params = param_dict
            else:
                if score > best_score:
                    best_score = score
                    best_params = param_dict
                    
        except Exception as e:
            print(f"RSF trial failed with params {param_dict}: {e}")
            continue
    
    return best_params, best_score

def grid_search_deephit(x_train, y_train, x_val, y_val, param_grid, eval_times, val_metric='c_index'):
    """
    Grid search for DeepHit hyperparameters.

    Parameters
    - x_train : array-like, shape (n_samples, n_features)
    - y_train : structured array, shape (n_samples,)
    - x_val : array-like, shape (n_samples, n_features)
    - y_val : structured array, shape (n_samples,)
    - param_grid : dict
        Dictionary containing hyperparameters to search over.
    - eval_times : array-like
        Evaluation time points to use for validation
    - val_metric : str
        Validation metric to optimize ('c_index', 'ipcw_c_index', 'ibs', 'mean_auc')

    Returns 
    - best_params: dict
        Best hyperparameters found.
    - best_score: float
        Best score achieved with the best hyperparameters.
    """
    best_score = np.inf if val_metric == 'ibs' else -np.inf
    best_params = None
    
    num_bins = len(eval_times)
    
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        try:
            # Preprocess
            scaler = StandardScaler()
            X_train_arr = scaler.fit_transform(x_train)
            X_val_arr = scaler.transform(x_val)
            
            # Label transform
            labtrans = LabTransDiscreteTime(num_bins)
            y_train_dh = labtrans.fit_transform(y_train["time"], y_train["event"])
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_train_dh = torch.tensor(np.asarray(X_train_arr), dtype=torch.float32, device=device)
            X_val_dh = torch.tensor(np.asarray(X_val_arr), dtype=torch.float32, device=device)
            y_train_time = torch.tensor(y_train_dh[0], dtype=torch.long, device=device)
            y_train_event = torch.tensor(y_train_dh[1], dtype=torch.float32, device=device)
            
            # Model
            in_features = X_train_dh.shape[1]
            out_features = labtrans.out_features
            num_nodes = [param_dict['first_layer_size'], param_dict['second_layer_size']]
            net = tt.practical.MLPVanilla(in_features, num_nodes, out_features,
                                          batch_norm=True, dropout=param_dict['dropout'])
            
            model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
            model.optimizer.set_lr(param_dict['learning_rate'])
            model.fit(X_train_dh, (y_train_time, y_train_event), batch_size=256, epochs=100, verbose=False)
            
            dh_eval_times = get_valid_eval_times(eval_times, y_val["time"])
            if len(dh_eval_times) == 0:
                continue
                
            val_results = evaluate_model_fold(model, X_val_dh, y_val, y_train, dh_eval_times,
                                            model_type='deephit', labtrans=labtrans)
            score = val_results[val_metric]
            
            # For IBS, lower is better; for others, higher is better
            if val_metric == 'ibs':
                if score < best_score:
                    best_score = score
                    best_params = param_dict
            else:
                if score > best_score:
                    best_score = score
                    best_params = param_dict
                    
        except Exception as e:
            print(f"DeepHit trial failed with params {param_dict}: {e}")
            continue
    
    return best_params, best_score

def grid_search_deepsurv(x_train, y_train, x_val, y_val, param_grid, eval_times, val_metric='c_index'):
    """
    Grid search for DeepSurv hyperparameters.

    Parameters
    - x_train : array-like, shape (n_samples, n_features)
    - y_train : structured array, shape (n_samples,)
    - x_val : array-like, shape (n_samples, n_features)
    - y_val : structured array, shape (n_samples,)
    - param_grid : dict
        Dictionary containing hyperparameters to search over.
    - eval_times : array-like
        Evaluation time points to use for validation
    - val_metric : str
        Validation metric to optimize ('c_index', 'ipcw_c_index', 'ibs', 'mean_auc')

    Returns
    - best_params: dict
        Best hyperparameters found.
    - best_score: float
        Best score achieved with the best hyperparameters.
    """
    best_score = np.inf if val_metric == 'ibs' else -np.inf
    best_params = None
    
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        try:
            # Preprocess
            scaler = StandardScaler()
            X_train_arr = scaler.fit_transform(x_train)
            X_val_arr = scaler.transform(x_val)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_train_ds = torch.tensor(np.asarray(X_train_arr), dtype=torch.float32, device=device)
            X_val_ds = torch.tensor(np.asarray(X_val_arr), dtype=torch.float32, device=device)
            
            y_train_time = torch.tensor(y_train["time"].copy(), dtype=torch.float32, device=device)
            y_train_event = torch.tensor(y_train["event"].astype(int).copy(), dtype=torch.float32, device=device)
            
            # For numerical stability
            epsilon = 1e-6
            y_train_event[y_train_event == 0] = epsilon
            
            # Model
            in_features = X_train_ds.shape[1]
            num_nodes = [param_dict['first_layer_size'], param_dict['second_layer_size']]
            net = tt.practical.MLPVanilla(in_features, num_nodes, 1,
                                          batch_norm=True, dropout=param_dict['dropout'])
            
            model = CoxPH(net, tt.optim.Adam)
            model.optimizer.set_lr(param_dict['learning_rate'])
            model.fit(X_train_ds, (y_train_time, y_train_event), batch_size=256, epochs=100, verbose=False)
            model.compute_baseline_hazards()
            
            ds_eval_times = get_valid_eval_times(eval_times, y_val["time"])
            if len(ds_eval_times) == 0:
                continue
                
            val_results = evaluate_model_fold(model, X_val_ds, y_val, y_train, ds_eval_times,
                                            model_type='deepsurv')
            score = val_results[val_metric]
            
            # For IBS, lower is better; for others, higher is better
            if val_metric == 'ibs':
                if score < best_score:
                    best_score = score
                    best_params = param_dict
            else:
                if score > best_score:
                    best_score = score
                    best_params = param_dict
                    
        except Exception as e:
            print(f"DeepSurv trial failed with params {param_dict}: {e}")
            continue
    
    return best_params, best_score

def get_valid_eval_times(base_eval_times, test_times, model=None, buffer=0.01):
    """
    Filter evaluation times to be within valid data follow-up period.
    
    This function ensures that evaluation time points are within the range
    of observed data to avoid extrapolation issues. For Random Survival Forest
    models, additional constraints are applied based on unique event times.
    
    Parameters:
    -----------
    base_eval_times : array-like
        Initial set of evaluation time points
    test_times : array-like
        Observed time values from test/validation data
    model : object, optional
        Fitted model (used for RSF-specific time constraints)
    buffer : float, optional (default=0.01)
        Small buffer to ensure times are strictly within bounds
    
    Returns:
    --------
    valid_times : array-like
        Filtered evaluation times within valid range
    """
    # Get min and max time values from test data
    min_test_time = test_times.min()
    max_test_time = test_times.max()
    
    # Apply basic time constraints with buffer to avoid boundary issues
    valid_times = base_eval_times[
        (base_eval_times > min_test_time + buffer) & 
        (base_eval_times < max_test_time - buffer)
    ]
    
    # Apply additional constraints for Random Survival Forest models
    if model is not None:
        # Extract the RSF estimator from pipeline if needed
        if hasattr(model, "named_steps"):
            estimator = None
            for step_name in reversed(model.named_steps):
                step = model.named_steps[step_name]
                if hasattr(step, "unique_times_"):
                    estimator = step
                    break
        else:
            estimator = model if hasattr(model, "unique_times_") else None
            
        # Further constrain times based on RSF's unique event times
        if estimator is not None and hasattr(estimator, "unique_times_"):
            event_times = estimator.unique_times_
            if len(event_times) > 1:
                valid_times = valid_times[
                    (valid_times > event_times[0] + buffer) & 
                    (valid_times < event_times[-1] - buffer)
                ]
    
    return valid_times

# =============================================================================
# Main Dataset Processing Loop
# =============================================================================

for file_name in dataset_files:
    try:
        # =================================================================
        # Dataset Loading and Initial Setup
        # =================================================================
        
        dataset_name = file_name.replace(".csv", "")
        file_path = os.path.join(data_dir, file_name)
        print("="*50)
        print(f"\nProcessing dataset: {dataset_name}")

        # Load dataset and drop patient ID column
        df = pd.read_csv(file_path)
        df.drop(columns=['pid'], inplace=True)
        shape = df.shape
        print(f"Dataset shape: {shape}")

        # =================================================================
        # Data Exploration and Column Identification
        # =================================================================
        
        # Define standard survival analysis columns
        time_col = "time"
        event_col = "event"
        
        # Calculate and display censoring statistics
        censored = (df[event_col] == 0).sum()
        censored_percent = (censored)/len(df)*100
        print(f"Percentage of censored data: {censored_percent}%")
        
        # Identify covariate columns (all except time and event)
        covariates = df.columns.difference([time_col, event_col])

        # =================================================================
        # Data Preparation and Train/Validation/Test Split
        # =================================================================
        
        # Prepare features and survival target
        x = df[covariates].copy()
        y = Surv.from_arrays(event=df[event_col].astype(bool), time=df[time_col])

        # Stratified split: 70% train+val, 15% test (stratified by event status)
        x_trainval, x_test, y_trainval, y_test = train_test_split(
            x, y, test_size=0.15, stratify=y["event"], random_state=SEED
        )

        # Further split train+val: ~70% train, ~15% val (from original 85%)
        x_train, x_val, y_train, y_val = train_test_split(
            x_trainval, y_trainval, test_size=0.1765, stratify=y_trainval["event"], random_state=SEED
        )
        
        # =================================================================
        # Feature Engineering and Preprocessing
        # =================================================================
        
        # One-hot encode categorical variables (fit only on training data)
        x_train_ohe = pd.get_dummies(x_train, drop_first=True)
        x_val_ohe   = pd.get_dummies(x_val, drop_first=True)
        x_test_ohe  = pd.get_dummies(x_test, drop_first=True)

        # Align feature columns across splits (add missing columns with 0s)
        x_train_ohe, x_val_ohe = x_train_ohe.align(x_val_ohe, join="left", axis=1, fill_value=0)
        x_train_ohe, x_test_ohe = x_train_ohe.align(x_test_ohe, join="left", axis=1, fill_value=0)

        covariates_ohe = x_train_ohe.columns  # Final set of covariate columns

        # Impute missing values using SimpleImputer (fit only on training data)
        imputer = SimpleImputer().fit(x_train_ohe.loc[:, covariates_ohe.tolist()])

        x_train_imputed = imputer.transform(x_train_ohe.loc[:, covariates_ohe.tolist()])
        x_train_imputed = pd.DataFrame(x_train_imputed, columns=covariates_ohe, index=x_train.index)

        x_val_imputed = imputer.transform(x_val_ohe.loc[:, covariates_ohe.tolist()])
        x_val_imputed = pd.DataFrame(x_val_imputed, columns=covariates_ohe, index=x_val.index)

        x_test_imputed = imputer.transform(x_test_ohe.loc[:, covariates_ohe.tolist()])
        x_test_imputed = pd.DataFrame(x_test_imputed, columns=covariates_ohe, index=x_test.index)

        # =================================================================
        # Evaluation Time Points Configuration
        # =================================================================
        
        # Generate evaluation time points from k-th to (100-k)-th percentiles
        percentiles, n_bins, _ = k_to_percentile_grid(args.k)
        if args.binning == 'quantile':
            base_eval_times = np.percentile(y_trainval["time"], percentiles)
        elif args.binning == 'equal_width':
            base_eval_times = np.linspace(y_trainval["time"].min(), y_trainval["time"].max(), n_bins + 2)[1:-1]
        else:  # km_quantile
            km_time, km_surv = kaplan_meier_estimator(y_trainval["event"], y_trainval["time"])
            if len(km_time) == 0:
                base_eval_times = np.percentile(y_trainval["time"], percentiles)
            else:
                surv_levels = np.linspace(km_surv[0], km_surv[-1], n_bins + 2)[1:-1]
                base_eval_times = []
                for s in surv_levels:
                    idx = np.searchsorted(-km_surv, -s)
                    if idx < len(km_time):
                        base_eval_times.append(float(km_time[idx]))
                base_eval_times = np.array(base_eval_times)
                if len(base_eval_times) == 0:
                    base_eval_times = np.percentile(y_trainval["time"], percentiles)
        base_eval_times = np.unique(base_eval_times)

        # Apply time truncation to test set to prevent extrapolation
        max_trainval_time = y_trainval["time"].max()
        # Don't remove samples, just truncate times that exceed training range
        y_test_filtered = y_test.copy()
        test_time_mask = y_test_filtered["time"] >= max_trainval_time
        y_test_filtered["time"][test_time_mask] = max_trainval_time - 0.01  # Small buffer
        y_test_filtered["event"][test_time_mask] = False  # Convert to censored
        x_test_filtered = x_test_imputed.copy()

        print(f"Test samples after time truncation: {len(y_test_filtered)} / {len(y_test)} (truncated {test_time_mask.sum()} samples)")

        # Filter evaluation times to be within the valid range
        eval_times = get_valid_eval_times(base_eval_times, y_test_filtered["time"])
        print(f"Evaluation time points: {len(eval_times)} points")

        # =================================================================
        # Hyperparameter Optimization Phase
        # ================================================================= 
        # Grid search for CoxPH
        best_params_cph, best_score_cph = grid_search_cph(x_train_imputed, y_train, x_val_imputed, y_val, cph_grid, eval_times, args.val_metric)
        
        # Fallback for CoxPH if grid search fails
        if best_params_cph is None:
            print("CoxPH grid search failed, using default parameters")
            best_params_cph = {'alpha': 1e-4}
            best_score_cph = 0.0

        print(f"Best CoxPH params: {best_params_cph}, Score: {best_score_cph:.4f} ({args.val_metric})")

        # Evaluate CoxPH on validation set to get all validation metrics
        val_eval_times = get_valid_eval_times(base_eval_times, y_val["time"])
        cph_val_model = make_pipeline(StandardScaler(), CoxPHSurvivalAnalysis(alpha=best_params_cph["alpha"]))
        cph_val_model.fit(x_train_imputed, y_train)
        cph_val_results = evaluate_model_fold(cph_val_model, x_val_imputed, y_val, y_train, val_eval_times)

        # Grid search for RSF
        best_params_rsf, best_score_rsf = grid_search_rsf(x_train_imputed, y_train, x_val_imputed, y_val, rsf_grid, eval_times, args.val_metric)
        
        # Fallback for RSF if grid search fails
        if best_params_rsf is None:
            print("RSF grid search failed, using default parameters")
            best_params_rsf = {'n_estimators': 100, 'min_samples_split': 5}
            best_score_rsf = 0.0
            
        print(f"Best RSF params: {best_params_rsf}, Score: {best_score_rsf:.4f} ({args.val_metric})")

        # Evaluate RSF on validation set to get all validation metrics
        rsf_val_model = make_pipeline(
            StandardScaler(),
            RandomSurvivalForest(
                n_estimators=best_params_rsf["n_estimators"],
                min_samples_split=best_params_rsf["min_samples_split"],
                random_state=SEED
            )
        )
        rsf_val_model.fit(x_train_imputed, y_train)
        val_eval_times_rsf = get_valid_eval_times(base_eval_times, y_val["time"], model=rsf_val_model)
        rsf_val_results = evaluate_model_fold(rsf_val_model, x_val_imputed, y_val, y_train, val_eval_times_rsf)

        # Grid search for DeepHit
        best_params_dh, best_score_dh = grid_search_deephit(x_train_imputed, y_train, x_val_imputed, y_val, deephit_grid, eval_times, args.val_metric)
        
        # Fallback for DeepHit if grid search fails
        if best_params_dh is None:
            print("DeepHit grid search failed, using default parameters")
            best_params_dh = {
                'dropout': 0.1,
                'first_layer_size': 128,
                'second_layer_size': 32,
                'learning_rate': 1e-3
            }
            best_score_dh = 0.0
            
        print(f"Best DeepHit params: {best_params_dh}, Score: {best_score_dh:.4f} ({args.val_metric})")

        # Evaluate DeepHit on validation set to get all validation metrics
        try:
            # Preprocess validation data for DeepHit
            dh_val_prep = make_pipeline(StandardScaler())
            X_train_dh_val_arr = dh_val_prep.fit_transform(x_train_imputed)
            X_val_dh_val_arr = dh_val_prep.transform(x_val_imputed)

            num_bins_val = len(eval_times)
            
            # Label transform
            labtrans_val = LabTransDiscreteTime(num_bins_val)
            y_train_dh_val = labtrans_val.fit_transform(y_train["time"], y_train["event"])

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_train_dh_val = torch.tensor(np.asarray(X_train_dh_val_arr), dtype=torch.float32, device=device)
            X_val_dh_val = torch.tensor(np.asarray(X_val_dh_val_arr), dtype=torch.float32, device=device)

            y_train_time_val = torch.tensor(y_train_dh_val[0], dtype=torch.long, device=device)
            y_train_event_val = torch.tensor(y_train_dh_val[1], dtype=torch.float32, device=device)

            # Model architecture using best grid search params
            in_features = X_train_dh_val.shape[1]
            out_features = labtrans_val.out_features
            num_nodes = [best_params_dh["first_layer_size"], best_params_dh["second_layer_size"]]
            net_val = tt.practical.MLPVanilla(in_features, num_nodes, out_features,
                                          batch_norm=True, dropout=best_params_dh["dropout"])

            deephit_val = DeepHitSingle(net_val, tt.optim.Adam,
                                    alpha=0.2,
                                    sigma=0.1,
                                    duration_index=labtrans_val.cuts)
            deephit_val.optimizer.set_lr(best_params_dh["learning_rate"])
            deephit_val.fit(X_train_dh_val, (y_train_time_val, y_train_event_val),
                        batch_size=256, epochs=300, verbose=False)

            val_eval_times_dh = get_valid_eval_times(base_eval_times, y_val["time"])
            dh_val_results = evaluate_model_fold(deephit_val, X_val_dh_val, y_val, y_train, val_eval_times_dh,
                                             model_type='deephit', labtrans=labtrans_val)
        except Exception as e:
            print(f"DeepHit validation evaluation failed: {e}")
            dh_val_results = {'c_index': 0.0, 'ipcw_c_index': 0.0, 'ibs': 0.0, 'mean_auc': 0.0}

         # Grid search for DeepSurv
        best_params_ds, best_score_ds = grid_search_deepsurv(x_train_imputed, y_train, x_val_imputed, y_val, deepsurv_grid, eval_times, args.val_metric)
        
        # Fallback for DeepSurv if grid search fails
        if best_params_ds is None:
            print("DeepSurv grid search failed, using default parameters")
            best_params_ds = {
                'dropout': 0.1,
                'first_layer_size': 128,
                'second_layer_size': 32,
                'learning_rate': 1e-3
            }
            best_score_ds = 0.0

        print(f"Best DeepSurv params: {best_params_ds}, Score: {best_score_ds:.4f} ({args.val_metric})")

        # Evaluate DeepSurv on validation set to get all validation metrics
        try:
            # Preprocess validation data for DeepSurv
            ds_val_prep = make_pipeline(StandardScaler())
            X_train_ds_val_arr = ds_val_prep.fit_transform(x_train_imputed)
            X_val_ds_val_arr = ds_val_prep.transform(x_val_imputed)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_train_ds_val = torch.tensor(np.asarray(X_train_ds_val_arr), dtype=torch.float32, device=device)
            X_val_ds_val = torch.tensor(np.asarray(X_val_ds_val_arr), dtype=torch.float32, device=device)

            y_train_time_ds_val = torch.tensor(y_train["time"].copy(), dtype=torch.float32, device=device)
            y_train_event_ds_val = torch.tensor(y_train["event"].astype(int).copy(), dtype=torch.float32, device=device)

            # For numerical stability
            epsilon = 1e-6
            y_train_event_ds_val[y_train_event_ds_val == 0] = epsilon

            # Model architecture
            num_nodes = [best_params_ds["first_layer_size"], best_params_ds["second_layer_size"]]
            net_ds_val = tt.practical.MLPVanilla(in_features, num_nodes, 1,
                                          batch_norm=True, dropout=best_params_ds["dropout"])

            deepsurv_val = CoxPH(net_ds_val, tt.optim.Adam)
            deepsurv_val.optimizer.set_lr(best_params_ds["learning_rate"])
            deepsurv_val.fit(X_train_ds_val, (y_train_time_ds_val, y_train_event_ds_val),
                         batch_size=256, epochs=300, verbose=False)
            deepsurv_val.compute_baseline_hazards()

            val_eval_times_ds = get_valid_eval_times(base_eval_times, y_val["time"])
            ds_val_results = evaluate_model_fold(deepsurv_val, X_val_ds_val, y_val, y_train, val_eval_times_ds,
                                             model_type='deepsurv')
        except Exception as e:
            print(f"DeepSurv validation evaluation failed: {e}")
            ds_val_results = {'c_index': 0.0, 'ipcw_c_index': 0.0, 'ibs': 0.0, 'mean_auc': 0.0}

        # Save best hyperparameters for this dataset
        hp_out = f"static_baseline_hyperparameter_{args.k}_{args.val_metric}.csv"
        write_hp_header = not os.path.exists(hp_out) or os.stat(hp_out).st_size == 0

        with open(hp_out, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_hp_header:
                writer.writerow([
                    "dataset_name", "k", "val_metric",
                    # CoxPH
                    "cph_alpha", "cph_score",
                    # RSF
                    "rsf_n_estimators", "rsf_min_samples_split", "rsf_score",
                    # DeepHit
                    "dh_dropout", "dh_first_layer_size", "dh_second_layer_size", "dh_learning_rate", "dh_score",
                    # DeepSurv
                    "ds_dropout", "ds_first_layer_size", "ds_second_layer_size", "ds_learning_rate", "ds_score"
                ])
            writer.writerow([
                dataset_name, args.k, args.val_metric,
                # CoxPH
                best_params_cph["alpha"], best_score_cph,
                # RSF
                best_params_rsf["n_estimators"], best_params_rsf["min_samples_split"], best_score_rsf,
                # DeepHit 
                best_params_dh["dropout"], best_params_dh["first_layer_size"], best_params_dh["second_layer_size"],
                best_params_dh["learning_rate"], best_score_dh,
                # DeepSurv
                best_params_ds["dropout"], best_params_ds["first_layer_size"], best_params_ds["second_layer_size"],
                best_params_ds["learning_rate"], best_score_ds
            ])

        # Retrain on train+val, evaluate on test 
        # Combine train and val sets
        x_trainval_ohe = pd.get_dummies(x_trainval, drop_first=True)
        x_test_ohe = pd.get_dummies(x_test, drop_first=True)
        x_trainval_ohe, x_test_ohe = x_trainval_ohe.align(x_test_ohe, join="left", axis=1, fill_value=0)
        covariates_ohe = x_trainval_ohe.columns

        # Impute missing values in covariates (fit on train+val only)
        imputer = SimpleImputer().fit(x_trainval_ohe.loc[:, covariates_ohe.tolist()])

        x_trainval_imputed = imputer.transform(x_trainval_ohe.loc[:, covariates_ohe.tolist()])
        x_trainval_imputed = pd.DataFrame(x_trainval_imputed, columns=covariates_ohe, index=x_trainval.index)

        x_test_imputed = imputer.transform(x_test_ohe.loc[:, covariates_ohe.tolist()])
        x_test_imputed = pd.DataFrame(x_test_imputed, columns=covariates_ohe, index=x_test.index)

        # Apply same time truncation to test set
        max_trainval_time = y_trainval["time"].max()
        y_test_filtered = y_test.copy()
        test_time_mask = y_test_filtered["time"] >= max_trainval_time
        y_test_filtered["time"][test_time_mask] = max_trainval_time - 0.01  # Small buffer
        y_test_filtered["event"][test_time_mask] = False  # Convert to censored
        x_test_filtered = x_test_imputed.copy()

        # Further filter eval_times to be within the range of the filtered test set
        eval_times = eval_times[
            (eval_times > y_test_filtered["time"].min()) & 
            (eval_times < y_test_filtered["time"].max())
        ]

         # Train RSF
        rsf = make_pipeline(
            StandardScaler(),
            RandomSurvivalForest(
                n_estimators=best_params_rsf["n_estimators"],
                min_samples_split=best_params_rsf["min_samples_split"],
                random_state=SEED
            )
        )
        rsf.fit(x_trainval_imputed, y_trainval)

        # Get valid evaluation times for RSF (includes RSF-specific constraints)
        eval_times_rsf = get_valid_eval_times(base_eval_times, y_test_filtered["time"], model=rsf)
        print(f"RSF eval times: {len(eval_times_rsf)} points")

        rsf_results = evaluate_model_fold(rsf, x_test_filtered, y_test_filtered, y_trainval, eval_times_rsf)

        # Train CoxPH
        cph = make_pipeline(StandardScaler(), CoxPHSurvivalAnalysis(alpha=best_params_cph["alpha"]))
        cph.fit(x_trainval_imputed, y_trainval)
        
        # Get valid evaluation times for CoxPH
        eval_times_cph = get_valid_eval_times(base_eval_times, y_test_filtered["time"])
        print(f"CoxPH eval times: {len(eval_times_cph)} points")
        
        cph_results = evaluate_model_fold(cph, x_test_filtered, y_test_filtered, y_trainval, eval_times_cph)

        # Train DeepHit
        # Preprocess via pipeline: StandardScaler (fit on trainval only)
        dh_prep = make_pipeline(StandardScaler())
        X_train_dh_arr = dh_prep.fit_transform(x_trainval_imputed)
        X_test_dh_arr = dh_prep.transform(x_test_filtered)

        # Get valid evaluation times for DeepHit
        eval_times_dh = get_valid_eval_times(base_eval_times, y_test_filtered["time"])
        print(f"DeepHit eval times: {len(eval_times_dh)} points")

        num_bins_final = len(eval_times_dh)
        
        # Label transform
        labtrans = LabTransDiscreteTime(num_bins_final)
        y_trainval_dh = labtrans.fit_transform(y_trainval["time"], y_trainval["event"])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train_dh = torch.tensor(np.asarray(X_train_dh_arr), dtype=torch.float32, device=device)
        X_test_dh = torch.tensor(np.asarray(X_test_dh_arr), dtype=torch.float32, device=device)

        y_trainval_time = torch.tensor(y_trainval_dh[0], dtype=torch.long, device=device)
        y_trainval_event = torch.tensor(y_trainval_dh[1], dtype=torch.float32, device=device)

        # Model architecture using best grid search params
        in_features = X_train_dh.shape[1]
        out_features = labtrans.out_features
        num_nodes = [best_params_dh["first_layer_size"], best_params_dh["second_layer_size"]]
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features,
                                      batch_norm=True, dropout=best_params_dh["dropout"])

        deephit = DeepHitSingle(net, tt.optim.Adam,
                                alpha=0.2,
                                sigma=0.1,
                                duration_index=labtrans.cuts)
        deephit.optimizer.set_lr(best_params_dh["learning_rate"])
        deephit.fit(X_train_dh, (y_trainval_time, y_trainval_event),
                    batch_size=256, epochs=300, verbose=False)

        dh_results = evaluate_model_fold(deephit, X_test_dh, y_test_filtered, y_trainval, eval_times_dh,
                                         model_type='deephit', labtrans=labtrans)

        # Train DeepSurv
        # Preprocess via pipeline: StandardScaler (fit on trainval only)
        ds_prep = make_pipeline(StandardScaler())
        X_train_ds_arr = ds_prep.fit_transform(x_trainval_imputed)
        X_test_ds_arr = ds_prep.transform(x_test_filtered)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train_ds = torch.tensor(np.asarray(X_train_ds_arr), dtype=torch.float32, device=device)
        X_test_ds = torch.tensor(np.asarray(X_test_ds_arr), dtype=torch.float32, device=device)

        y_trainval_time = torch.tensor(y_trainval["time"].copy(), dtype=torch.float32, device=device)
        y_trainval_event = torch.tensor(y_trainval["event"].astype(int).copy(), dtype=torch.float32, device=device)

        # For numerical stability
        epsilon = 1e-6
        y_trainval_event[y_trainval_event == 0] = epsilon

        # Model architecture
        num_nodes = [best_params_ds["first_layer_size"], best_params_ds["second_layer_size"]]
        net = tt.practical.MLPVanilla(in_features, num_nodes, 1,
                                      batch_norm=True, dropout=best_params_ds["dropout"])

        deepsurv = CoxPH(net, tt.optim.Adam)
        deepsurv.optimizer.set_lr(best_params_ds["learning_rate"])
        deepsurv.fit(X_train_ds, (y_trainval_time, y_trainval_event),
                     batch_size=256, epochs=300, verbose=False)
        deepsurv.compute_baseline_hazards()

        eval_times_ds = get_valid_eval_times(base_eval_times, y_test_filtered["time"])
        print(f"DeepSurv eval times: {len(eval_times_ds)} points")

        ds_results = evaluate_model_fold(deepsurv, X_test_ds, y_test_filtered, y_trainval, eval_times_ds,
                                         model_type='deepsurv')

        # Print results 
        print("="*50)
        print(f"Final test results for dataset {dataset_name}:")
        
        print(f"\nRandom Survival Forest:")
        print(f"C-index: {rsf_results['c_index']:.4f}")
        print(f"IPCW C-index: {rsf_results['ipcw_c_index']:.4f}")
        print(f"IBS: {rsf_results['ibs']:.4f}")
        print(f"Mean AUC: {rsf_results['mean_auc']:.4f}")

        print(f"\nCoxPH:")
        print(f"C-index: {cph_results['c_index']:.4f}")
        print(f"IPCW C-index: {cph_results['ipcw_c_index']:.4f}")
        print(f"IBS: {cph_results['ibs']:.4f}")
        print(f"Mean AUC: {cph_results['mean_auc']:.4f}")

        print(f"\nDeepHit:")
        print(f"C-index: {dh_results['c_index']:.4f}")
        print(f"IPCW C-index: {dh_results['ipcw_c_index']:.4f}")
        print(f"IBS: {dh_results['ibs']:.4f}")
        print(f"Mean AUC: {dh_results['mean_auc']:.4f}")

        print(f"\nDeepSurv:")
        print(f"C-index: {ds_results['c_index']:.4f}")
        print(f"IPCW C-index: {ds_results['ipcw_c_index']:.4f}")
        print(f"IBS: {ds_results['ibs']:.4f}")
        print(f"Mean AUC: {ds_results['mean_auc']:.4f}")

        # Prepare results for CSV
        results = [dataset_name]
        
        # Add RSF results
        results.extend([
            rsf_results['c_index'],
            rsf_results['ipcw_c_index'],
            rsf_results['ibs'],
            rsf_results['mean_auc']
        ])
        
        # Add CoxPH results
        results.extend([
            cph_results['c_index'],
            cph_results['ipcw_c_index'],
            cph_results['ibs'],
            cph_results['mean_auc']
        ])
        
        # Add DeepHit results
        results.extend([
            dh_results['c_index'],
            dh_results['ipcw_c_index'],
            dh_results['ibs'],
            dh_results['mean_auc']
        ])
        
        # Add DeepSurv results
        results.extend([
            ds_results['c_index'],
            ds_results['ipcw_c_index'],
            ds_results['ibs'],
            ds_results['mean_auc']
        ])

        # Write the results into a file with k parameter in filename
        output_file = csv_path

        # Check if file is empty or doesn't exist
        write_header = not os.path.exists(output_file) or os.stat(output_file).st_size == 0

        with open(output_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "dataset_name", "k", "val_metric", "n_eval_times_rsf", "n_eval_times_cph", "n_eval_times_dh", "n_eval_times_ds",
                    # Validation metrics
                    "val_rsf_c_index", "val_rsf_ipcw_c_index", "val_rsf_ibs", "val_rsf_mean_auc",
                    "val_cph_c_index", "val_cph_ipcw_c_index", "val_cph_ibs", "val_cph_mean_auc",
                    "val_dh_c_index", "val_dh_ipcw_c_index", "val_dh_ibs", "val_dh_mean_auc",
                    "val_ds_c_index", "val_ds_ipcw_c_index", "val_ds_ibs", "val_ds_mean_auc",
                    # Test metrics
                    "test_rsf_c_index", "test_rsf_ipcw_c_index", "test_rsf_ibs", "test_rsf_mean_auc",
                    "test_cph_c_index", "test_cph_ipcw_c_index", "test_cph_ibs", "test_cph_mean_auc",
                    "test_dh_c_index", "test_dh_ipcw_c_index", "test_dh_ibs", "test_dh_mean_auc",
                    "test_ds_c_index", "test_ds_ipcw_c_index", "test_ds_ibs", "test_ds_mean_auc"
                ])
            writer.writerow([
                dataset_name, args.k, args.val_metric, len(eval_times_rsf), len(eval_times_cph), len(eval_times_dh), len(eval_times_ds),
                # Validation results
                rsf_val_results['c_index'], rsf_val_results['ipcw_c_index'], rsf_val_results['ibs'], rsf_val_results['mean_auc'],
                cph_val_results['c_index'], cph_val_results['ipcw_c_index'], cph_val_results['ibs'], cph_val_results['mean_auc'],
                dh_val_results['c_index'], dh_val_results['ipcw_c_index'], dh_val_results['ibs'], dh_val_results['mean_auc'],
                ds_val_results['c_index'], ds_val_results['ipcw_c_index'], ds_val_results['ibs'], ds_val_results['mean_auc'],
                # Test results
                rsf_results['c_index'], rsf_results['ipcw_c_index'], rsf_results['ibs'], rsf_results['mean_auc'],
                cph_results['c_index'], cph_results['ipcw_c_index'], cph_results['ibs'], cph_results['mean_auc'],
                dh_results['c_index'], dh_results['ipcw_c_index'], dh_results['ibs'], dh_results['mean_auc'],
                ds_results['c_index'], ds_results['ipcw_c_index'], ds_results['ibs'], ds_results['mean_auc']
            ])

    except Exception as e:
        print(f" Skipping {file_name}: {e}")
        continue