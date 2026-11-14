# Copy of ../tmlr_baselines_rebuttal_6_9/mtlr_evaluation.py (static discrete-time baselines) + --binning km_quantile (KM-quantile grid of Table 7, 11-03);
# everything else unchanged. Output: static_<model>_evaluation_final_<k>_ipcw_c_index[_km_quantile].csv
#!/usr/bin/env python3
"""Linear and neural MTLR static survival analysis.

The discrete-time model uses training-fitted LabTransDiscreteTime cuts.
Evaluation is performed at percentile-based horizons.
"""

import argparse
import csv
import os
import random
import traceback
import warnings
from itertools import product

import numpy as np

# Compatibility with older pycox/torchtuples code.
if not hasattr(np, "infty"):
    np.infty = np.inf

import pandas as pd
import torch
import torchtuples as tt

from pycox.models import MTLR
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv


# =============================================================================
# Configuration and Setup
# =============================================================================

SEED = 42

def reset_seed(seed=SEED):
    """Reset all RNGs before each independent fit."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


reset_seed()

if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items


# =============================================================================
# Command-Line Arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description=(
        "Run linear or neural MTLR static survival analysis with an "
        "explicit np.percentile discrete-time grid shared by MTLR and sksurv."
    )
)

parser.add_argument(
    "--model",
    type=str,
    default="linear_mtlr",
    choices=["linear_mtlr", "neural_mtlr"],
    help="MTLR model to run.",
)

parser.add_argument(
    "--k",
    type=float,
    default=10,
    help=(
        "Percentile step used for the common grid. k=10 gives "
        "10th, 20th, ..., and 90th percentiles plus pycox boundary cuts."
    ),
)

parser.add_argument(
    "--val_metric",
    type=str,
    default="c_index",
    choices=["c_index", "ipcw_c_index", "ibs", "mean_auc"],
    help="Validation metric used for hyperparameter selection.",
)

parser.add_argument('--binning', default='quantile', choices=['quantile', 'km_quantile'])  # KMBIN (11-03)
args = parser.parse_args()

if not (0 < args.k < 100):
    raise ValueError("--k must be in the interval (0, 100).")


# =============================================================================
# Training Configuration
# =============================================================================

MAX_EPOCHS = 200
BATCH_SIZE = 256


class BestValidationWeights(tt.callbacks.Callback):
    """Keep the best validation-loss weights in memory during training."""

    def __init__(self):
        self.best_loss = np.inf
        self.best_epoch = 0
        self.best_state = None

    def on_fit_start(self):
        self.best_state = {
            name: parameter.detach().clone()
            for name, parameter in self.model.net.state_dict().items()
        }

    def on_epoch_end(self):
        current_loss = self.model.val_metrics.scores["loss"]["score"][-1]

        if current_loss < self.best_loss:
            self.best_loss = float(current_loss)
            self.best_epoch = len(
                self.model.val_metrics.scores["loss"]["score"]
            )
            self.best_state = {
                name: parameter.detach().clone()
                for name, parameter in self.model.net.state_dict().items()
            }

    def on_fit_end(self):
        self.model.net.load_state_dict(self.best_state)


# =============================================================================
# Data Configuration
# =============================================================================

data_dir = "static_data"

dataset_files = sorted(
    f for f in os.listdir(data_dir) if f.endswith(".csv")
)

csv_path = (
    f"static_{args.model}_evaluation_final_"
    f"{args.k:.1f}_ipcw_c_index" + ("_km_quantile" if args.binning == "km_quantile" else "") + ".csv"
)


# =============================================================================
# Hyperparameter Configuration
# =============================================================================

linear_mtlr_grid = {
    "learning_rate": [1e-2, 1e-3, 1e-4],
    "weight_decay": [1e-5, 1e-4, 1e-3],
}

neural_mtlr_grid = {
    "dropout": [0.1, 0.3, 0.5],
    "first_layer_size": [32, 64, 128],
    "second_layer_size": [16, 32, 64],
    "learning_rate": [1e-2, 1e-3, 1e-4],
    "weight_decay": [1e-3],
}

mtlr_grid = (
    linear_mtlr_grid
    if args.model == "linear_mtlr"
    else neural_mtlr_grid
)


# =============================================================================
# Evaluation-Time Configuration
# =============================================================================

def km_grid(y, k):
    """KMBIN (11-03): unsmoothed KM-quantile grid of Table 7 (static_cf_w.py --binning km_quantile): K - 1 equally spaced
    levels strictly between the first and last KM value of the given outcomes; grid point = first KM time at or below."""
    from sksurv.nonparametric import kaplan_meier_estimator
    K = int(np.ceil(100.0 / float(k)))
    km_t, km_s = kaplan_meier_estimator(np.asarray(y["event"], dtype=bool), np.asarray(y["time"], dtype=float))
    idx = np.searchsorted(-km_s, -np.linspace(km_s[0], km_s[-1], (K - 1) + 2)[1:-1])
    return np.unique(km_t[idx[idx < len(km_t)]].astype(np.float64))


def make_percentile_grid(times, k):
    """
    Construct the percentile evaluation horizons used by the baseline.

    LabTransDiscreteTime learns the model's internal cuts separately, just as
    it does in the DeepHit implementation.
    """
    times = np.asarray(times, dtype=np.float64)

    if times.ndim != 1 or len(times) == 0:
        raise ValueError("times must be a non-empty 1D array.")

    if not np.all(np.isfinite(times)):
        raise ValueError("Survival times contain NaN or infinite values.")

    if np.any(times < 0):
        raise ValueError("Survival times must be non-negative.")

    percentiles = np.arange(k, 100.0, k, dtype=np.float64)
    percentiles = np.unique(percentiles)
    percentiles = percentiles[percentiles <= 100.0]

    evaluation_grid = np.percentile(times, percentiles).astype(
        np.float64, copy=False
    )
    grid = np.unique(evaluation_grid)
    grid.sort()

    if len(grid) == 0:
        raise ValueError("The percentile evaluation grid is empty.")

    return grid


def get_valid_eval_times(base_grid, evaluation_times, train_times):
    """
    Return evaluation times that are literal values from base_grid.

    sksurv requires evaluation times to lie within the usable follow-up
    range. No new discretization or interpolation is performed.
    """
    base_grid = np.asarray(base_grid, dtype=np.float64)
    evaluation_times = np.asarray(evaluation_times, dtype=np.float64)
    train_times = np.asarray(train_times, dtype=np.float64)

    if len(evaluation_times) == 0:
        return np.array([], dtype=np.float64)

    # sksurv requires evaluation times strictly inside the observed test
    # follow-up interval. Use adjacent floating-point values rather than a
    # fixed offset, because the latter is arbitrary across time units.
    lower = np.nextafter(np.min(evaluation_times), np.inf)
    upper = np.nextafter(
        min(np.max(evaluation_times), np.max(train_times)),
        -np.inf,
    )
    mask = (base_grid > lower) & (base_grid < upper)

    return base_grid[mask]


def administratively_censor_at(y, boundary):
    """Censor observations at a boundary that must not be evaluated.

    IPCW metrics estimate their censoring distribution from the training
    split, so scikit-survival requires every evaluation observation to lie
    strictly below the largest training follow-up time. This keeps the
    observations in the evaluation cohort while removing information beyond
    that supported horizon.
    """
    boundary = float(boundary)
    censored_time = np.nextafter(boundary, -np.inf)

    y_censored = y.copy()
    outside_support = y_censored["time"] >= boundary
    y_censored["time"][outside_support] = censored_time
    y_censored["event"][outside_support] = False

    return y_censored


# =============================================================================
# Model Construction
# =============================================================================

def build_network(input_features, out_features, params):
    """Build the network following the pycox notebook structure."""
    if args.model == "linear_mtlr":
        return torch.nn.Linear(input_features, out_features)

    return tt.practical.MLPVanilla(
        input_features,
        [
            params["first_layer_size"],
            params["second_layer_size"],
        ],
        out_features,
        batch_norm=True,
        dropout=params["dropout"],
    )


def build_mtlr_model(input_features, labtrans, params):
    """
    Build MTLR in the same pattern as the pycox example:

        net = ...
        model = MTLR(net, tt.optim.Adam, duration_index=labtrans.cuts)
        model.optimizer.set_lr(lr)
    """
    if not isinstance(labtrans.cuts, np.ndarray):
        raise TypeError("labtrans.cuts must be a NumPy array.")

    network = build_network(
        input_features,
        labtrans.out_features,
        params,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MTLR(
        network,
        tt.optim.Adam(
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        ),
        device=device,
        duration_index=labtrans.cuts,
    )

    return model


# =============================================================================
# Prediction Helpers
# =============================================================================

def predict_survival(model, x_data):
    """
    Return survival predictions on model.duration_index.

    pycox MTLR inherits PMFBase. Its predict_surv() is the standard
    1 - cumulative PMF construction.
    """
    x_data = np.asarray(x_data, dtype=np.float32)

    surv_df = model.predict_surv_df(x_data)
    model_times = np.asarray(surv_df.index, dtype=np.float64)

    expected_times = np.asarray(model.duration_index, dtype=np.float64)
    if not np.array_equal(model_times, expected_times):
        raise ValueError(
            "Prediction duration index does not exactly match the MTLR grid."
        )

    # DataFrame is [time, patient]. sksurv wants [patient, time].
    survival = surv_df.to_numpy(dtype=np.float64).T
    survival = np.clip(survival, 0.0, 1.0)

    return survival


def select_survival_columns(model, survival, eval_times):
    """Interpolate MTLR survival predictions at evaluation horizons."""
    model_times = np.asarray(model.duration_index, dtype=np.float64)
    eval_times = np.asarray(eval_times, dtype=np.float64)

    return np.asarray(
        [np.interp(eval_times, model_times, curve) for curve in survival],
        dtype=np.float64,
    )


def mean_cumulative_risk(model, x_data, eval_times):
    """
    Scalar risk = mean predicted cumulative event probability
                  across the evaluation time points.

    Larger values = higher risk.
    """
    x_data = np.asarray(x_data, dtype=np.float32)

    survival = select_survival_columns(
        model,
        predict_survival(model, x_data),
        eval_times,
    )

    cumulative_risk = 1.0 - survival

    risk_scores = np.mean(
        cumulative_risk,
        axis=1,
    )

    return risk_scores


# =============================================================================
# Evaluation Metrics
# =============================================================================

def evaluate_mtlr(model, x_data, y_test, y_train, eval_times):
    """
    Compute:
        C-index
        IPCW C-index
        Integrated Brier Score
        Mean cumulative dynamic AUC

    Survival predictions are interpolated from the MTLR duration grid to
    the requested evaluation times.
    """
    x_data = np.asarray(x_data, dtype=np.float32)
    eval_times = np.asarray(eval_times, dtype=np.float64)

    if len(eval_times) < 1:
        raise ValueError("No valid evaluation times were supplied.")

    # Full MTLR survival curve, then interpolate to requested horizons.
    full_survival = predict_survival(model, x_data)
    survival = select_survival_columns(
        model,
        full_survival,
        eval_times,
    )

    # Scalar risk for C-index / AUC. Derived from the survival matrix above so
    # the curves are only predicted once per evaluation.
    risk_scores = np.mean(1.0 - survival, axis=1)

    # -------------------------------------------------------------
    # C-index
    # -------------------------------------------------------------
    c_index, *_ = concordance_index_censored(
        y_test["event"],
        y_test["time"],
        risk_scores,
    )

    # -------------------------------------------------------------
    # IPCW C-index
    # -------------------------------------------------------------
    # Use the same tau convention as every other model merged into the shared
    # results table (static_baseline_final.py, static_tabpfn_final.py): the
    # largest follow-up time of the training cohort. The evaluation labels are
    # administratively censored below this boundary by the caller, so the
    # fitted censoring survival stays positive.
    tau = float(np.max(y_train["time"]))

    ipcw_c_index, *_ = concordance_index_ipcw(
        y_train,
        y_test,
        risk_scores,
        tau=tau,
    )

    # -------------------------------------------------------------
    # Integrated Brier Score
    # -------------------------------------------------------------
    ibs = integrated_brier_score(
        y_train,
        y_test,
        survival,
        eval_times,
    )

    # -------------------------------------------------------------
    # Cumulative dynamic AUC
    # -------------------------------------------------------------
    # The scalar risk (not the time-varying matrix) is passed here so that
    # mean_auc is the same estimand as for the models this table compares
    # against; sksurv treats 1-D and 2-D estimates differently.
    _, mean_auc = cumulative_dynamic_auc(
        y_train,
        y_test,
        risk_scores,
        eval_times,
    )

    return {
        "c_index": float(c_index),
        "ipcw_c_index": float(ipcw_c_index),
        "ibs": float(ibs),
        "mean_auc": float(mean_auc),
    }


# =============================================================================
# Model Fitting
# =============================================================================

def fit_mtlr(
    x_train,
    y_train,
    num_bins,
    params,
    epochs=200,
    x_val=None,
    y_val=None,
):
    """
    Fit MTLR using the DeepHit-style learned discretization.

    The integer number of bins is supplied, and the label transformer learns
    its cuts from the training outcomes.
    """
    # -------------------------------------------------------------
    # Standardization
    # -------------------------------------------------------------
    preprocessor = make_pipeline(StandardScaler())

    x_train_array = preprocessor.fit_transform(x_train).astype(
        np.float32,
        copy=False,
    )

    # -------------------------------------------------------------
    # DeepHit-style discrete-time grid learned from training outcomes.
    # -------------------------------------------------------------
    labtrans = LabTransDiscreteTime(int(num_bins))

    train_times = np.asarray(y_train["time"], dtype=np.float64)
    train_events = np.asarray(y_train["event"], dtype=bool)

    y_train_time_idx, y_train_event = labtrans.fit_transform(
        train_times,
        train_events,
    )

    y_train_transformed = (
        np.asarray(y_train_time_idx, dtype=np.int64),
        np.asarray(y_train_event, dtype=np.float32),
    )

    # -------------------------------------------------------------
    # Validation data
    # -------------------------------------------------------------
    val_data = None

    if x_val is not None and y_val is not None:
        x_val_array = preprocessor.transform(x_val).astype(
            np.float32,
            copy=False,
        )

        val_times = np.asarray(y_val["time"], dtype=np.float64)
        val_events = np.asarray(y_val["event"], dtype=bool)

        y_val_time_idx, y_val_event = labtrans.transform(
            val_times,
            val_events,
        )

        y_val_transformed = (
            np.asarray(y_val_time_idx, dtype=np.int64),
            np.asarray(y_val_event, dtype=np.float32),
        )

        val_data = (x_val_array, y_val_transformed)

    # -------------------------------------------------------------
    # Build model exactly once for this trial
    # -------------------------------------------------------------
    model = build_mtlr_model(
        x_train_array.shape[1],
        labtrans,
        params,
    )

    # -------------------------------------------------------------
    # Fit for the requested fixed number of epochs.
    # -------------------------------------------------------------
    callbacks = []
    best_weights = None
    if val_data is not None:
        best_weights = BestValidationWeights()
        callbacks.append(best_weights)

    model.fit(
        x_train_array,
        y_train_transformed,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        callbacks=callbacks,
        verbose=False,
        val_data=val_data,
    )

    best_epoch = best_weights.best_epoch if best_weights is not None else None

    return model, preprocessor, labtrans, best_epoch


# =============================================================================
# Hyperparameter Optimization
# =============================================================================

def grid_search_mtlr(
    x_train,
    y_train,
    x_val,
    y_val_fit,
    y_val_eval,
    num_bins,
    val_eval_times,
):
    """Select hyperparameters using the requested validation metric."""
    best_params = None
    best_score = np.inf if args.val_metric == "ibs" else -np.inf
    best_epoch = None

    for values in product(*mtlr_grid.values()):
        params = dict(zip(mtlr_grid.keys(), values))

        try:
            reset_seed()
            (
                model,
                preprocessor,
                labtrans,
                trial_best_epoch,
            ) = fit_mtlr(
                x_train,
                y_train,
                num_bins,
                params,
                epochs=MAX_EPOCHS,
                x_val=x_val,
                y_val=y_val_fit,
            )

            x_val_scaled = preprocessor.transform(x_val).astype(
                np.float32,
                copy=False,
            )

            # Score on the administratively censored labels, i.e. exactly
            # the labels the reported val_* metrics are computed on.
            results = evaluate_mtlr(
                model,
                x_val_scaled,
                y_val_eval,
                y_train,
                val_eval_times,
            )

            score = results[args.val_metric]

            if args.val_metric == "ibs":
                is_better = score < best_score
            else:
                is_better = score > best_score

            if is_better:
                best_params = params.copy()
                best_epoch = trial_best_epoch
                best_score = score

            print(
                f"  {args.model} params={params} "
                f"{args.val_metric}={score:.4f} "
                f"best_epoch={trial_best_epoch}"
            )

        except Exception as error:
            print(
                f"  {args.model} trial failed with params {params}: "
                f"{error}"
            )
            traceback.print_exc()

    return (
        best_params,
        best_score,
        best_epoch,
    )


# =============================================================================
# Main Dataset Processing Loop
# =============================================================================

for file_name in dataset_files:
    try:
        dataset_name = file_name.removesuffix(".csv")

        # -------------------------------------------------------------
        # Load data
        # -------------------------------------------------------------
        df = pd.read_csv(os.path.join(data_dir, file_name))

        # -------------------------------------------------------------
        # Remove patient ID
        # -------------------------------------------------------------
        if "pid" in df.columns:
            df = df.drop(columns=["pid"])

        # -------------------------------------------------------------
        # Covariates -- the original column order is kept so that this script
        # and bce_evaluation.py see identically ordered feature matrices.
        # -------------------------------------------------------------
        feature_columns = [
            c for c in df.columns
            if c not in ["time", "event"]
        ]
        x = df[feature_columns].copy()

        # -------------------------------------------------------------
        # Survival outcome
        # -------------------------------------------------------------
        y = Surv.from_arrays(
            df["event"].astype(bool).to_numpy(),
            df["time"].astype(np.float64).to_numpy(),
        )

        # -------------------------------------------------------------
        # Train / test split
        # -------------------------------------------------------------
        (
            x_trainval,
            x_test,
            y_trainval,
            y_test,
        ) = train_test_split(
            x,
            y,
            test_size=0.15,
            stratify=y["event"],
            random_state=SEED,
        )

        # -------------------------------------------------------------
        # Train / validation split
        # -------------------------------------------------------------
        (
            x_train,
            x_val,
            y_train,
            y_val,
        ) = train_test_split(
            x_trainval,
            y_trainval,
            test_size=0.1765,
            stratify=y_trainval["event"],
            random_state=SEED,
        )

        # -------------------------------------------------------------
        # One-hot encoding -- kept from user's preprocessing
        # -------------------------------------------------------------
        x_train_ohe = pd.get_dummies(
            x_train,
            drop_first=True,
        )

        x_val_ohe = pd.get_dummies(
            x_val,
            drop_first=True,
        )

        x_test_ohe = pd.get_dummies(
            x_test,
            drop_first=True,
        )

        x_train_ohe, x_val_ohe = x_train_ohe.align(
            x_val_ohe,
            join="left",
            axis=1,
            fill_value=0,
        )

        x_train_ohe, x_test_ohe = x_train_ohe.align(
            x_test_ohe,
            join="left",
            axis=1,
            fill_value=0,
        )

        # -------------------------------------------------------------
        # Imputation -- kept from user's preprocessing
        # -------------------------------------------------------------
        imputer = SimpleImputer().fit(x_train_ohe)

        x_train_imputed = imputer.transform(x_train_ohe).astype(
            np.float32,
            copy=False,
        )

        x_val_imputed = imputer.transform(x_val_ohe).astype(
            np.float32,
            copy=False,
        )

        x_test_imputed = imputer.transform(x_test_ohe).astype(
            np.float32,
            copy=False,
        )

        # -------------------------------------------------------------
        # TUNING TIME GRID
        # -------------------------------------------------------------
        # Hyperparameter tuning must not use validation outcomes to construct
        # its discretization grid. The final refit below gets its own grid
        # built from train+validation after selection is complete.
        tuning_grid = (km_grid(y_train, args.k) if args.binning == "km_quantile" else make_percentile_grid(
            y_train["time"],
            args.k,
        ))

        # The validation labels are administratively censored at the training
        # support boundary for IPCW-based evaluation. LabTransDiscreteTime
        # applies the same right-censoring rule to validation training labels.
        y_val_filtered = administratively_censor_at(
            y_val,
            np.max(y_train["time"]),
        )

        # -------------------------------------------------------------
        # Validation evaluation times from the baseline percentile horizons
        # -------------------------------------------------------------
        val_eval_times = get_valid_eval_times(
            tuning_grid,
            y_val_filtered["time"],
            y_train["time"],
        )

        if len(val_eval_times) == 0:
            print(
                f"No valid validation evaluation times for "
                f"{dataset_name}; skipping"
            )
            continue

        print(f"\nDataset: {dataset_name}")
        print(f"Model: {args.model}")
        print(f"Validation metric: {args.val_metric}")
        print(f"Maximum epochs: {MAX_EPOCHS}")
        print("Training-only percentile grid:")
        print(tuning_grid)
        print(f"Grid dtype: {tuning_grid.dtype}")
        print(f"Number of grid points: {len(tuning_grid)}")
        print(f"Grid minimum: {tuning_grid.min()}")
        print(f"Grid maximum: {tuning_grid.max()}")
        print(f"Training maximum time: {y_train['time'].max()}")
        print(f"Validation evaluation times:")
        print(val_eval_times)

        # -------------------------------------------------------------
        # Hyperparameter selection
        # -------------------------------------------------------------
        (
            best_params,
            best_score,
            best_epoch,
        ) = grid_search_mtlr(
            x_train_imputed,
            y_train,
            x_val_imputed,
            y_val,
            y_val_filtered,
            len(val_eval_times),
            val_eval_times,
        )

        if best_params is None:
            print(
                f"Grid search failed for {dataset_name}; skipping"
            )
            continue

        print(f"\nBest {args.model} params for {dataset_name}: {best_params}")
        print(f"Best validation {args.val_metric}: {best_score:.4f}")
        print(f"Best validation epoch: {best_epoch}")

        # -------------------------------------------------------------
        # Refit on train only for validation reporting
        # -------------------------------------------------------------
        reset_seed()
        (
            val_model,
            val_preprocessor,
            val_labtrans,
            _,
        ) = fit_mtlr(
            x_train_imputed,
            y_train,
            len(val_eval_times),
            best_params,
            epochs=best_epoch,
        )

        x_val_scaled = val_preprocessor.transform(x_val_imputed).astype(
            np.float32,
            copy=False,
        )

        val_results = evaluate_mtlr(
            val_model,
            x_val_scaled,
            y_val_filtered,
            y_train,
            val_eval_times,
        )

        # -------------------------------------------------------------
        # Test-time filtering for sksurv IPCW constraints
        # -------------------------------------------------------------
        # The design matrix for the final fit stacks the train block on top
        # of the validation block, so the labels have to be stacked in that
        # same order. y_trainval is still in the pre-split ordering produced by
        # train_test_split: pairing it with the stacked matrix assigns almost
        # every subject another subject's (time, event).
        trainval_index = x_train.index.append(x_val.index)
        y_trainval_aligned = np.concatenate([y_train, y_val])

        assert np.array_equal(
            y_trainval_aligned["time"],
            df.loc[trainval_index, "time"].astype(np.float64).to_numpy(),
        ), "train+val labels are misaligned with the stacked design matrix"
        assert np.array_equal(
            y_trainval_aligned["event"],
            df.loc[trainval_index, "event"].astype(bool).to_numpy(),
        ), "train+val labels are misaligned with the stacked design matrix"

        max_trainval_time = float(np.max(y_trainval_aligned["time"]))

        y_test_filtered = administratively_censor_at(
            y_test,
            max_trainval_time,
        )

        # The final model may use train+validation outcomes because model
        # selection has finished. Its learned cuts are fit independently.
        common_grid = (km_grid(y_trainval_aligned, args.k) if args.binning == "km_quantile" else make_percentile_grid(
            y_trainval_aligned["time"],
            args.k,
        ))

        # -------------------------------------------------------------
        # Test evaluation times from the baseline percentile horizons.
        # -------------------------------------------------------------
        test_eval_times = get_valid_eval_times(
            common_grid,
            y_test_filtered["time"],
            y_trainval_aligned["time"],
        )

        if len(test_eval_times) == 0:
            print(
                f"No valid test evaluation times for "
                f"{dataset_name}; skipping"
            )
            continue

        # -------------------------------------------------------------
        # Final model: train + validation
        # -------------------------------------------------------------
        reset_seed()
        x_trainval_imputed = np.vstack(
            [x_train_imputed, x_val_imputed]
        ).astype(np.float32, copy=False)

        assert x_trainval_imputed.shape[0] == len(y_trainval_aligned)

        (
            test_model,
            test_preprocessor,
            test_labtrans,
            _,
        ) = fit_mtlr(
            x_trainval_imputed,
            y_trainval_aligned,
            len(test_eval_times),
            best_params,
            epochs=best_epoch,
        )

        x_test_scaled = test_preprocessor.transform(x_test_imputed).astype(
            np.float32,
            copy=False,
        )

        # -------------------------------------------------------------
        # Diagnostics for the 0.5 C-index problem
        # -------------------------------------------------------------
        test_risk = mean_cumulative_risk(test_model, x_test_scaled, test_eval_times)

        print("Risk diagnostics:")
        print(f"  risk min: {np.min(test_risk):.8f}")
        print(f"  risk max: {np.max(test_risk):.8f}")
        print(f"  risk mean: {np.mean(test_risk):.8f}")
        print(f"  risk std: {np.std(test_risk):.8f}")
        print(f"  unique risk values: {np.unique(test_risk).size}")

        full_survival = predict_survival(test_model, x_test_scaled)
        print(
            f"  survival at first grid point: "
            f"mean={np.mean(full_survival[:, 0]):.8f}"
        )
        print(
            f"  survival at final grid point: "
            f"mean={np.mean(full_survival[:, -1]):.8f}"
        )

        test_results = evaluate_mtlr(
            test_model,
            x_test_scaled,
            y_test_filtered,
            y_trainval_aligned,
            test_eval_times,
        )

        print(f"Test results: {test_results}")

        # -------------------------------------------------------------
        # Save results
        # -------------------------------------------------------------
        row = {
            "dataset_name": dataset_name,
            "k": args.k,
            "val_metric": args.val_metric,
            "n_discrete_time_cuts": len(test_labtrans.cuts),
            "n_eval_times": len(test_eval_times),
            "best_epoch": best_epoch,
            "risk_std": float(np.std(test_risk)),
            "unique_risk": int(np.unique(test_risk).size),
            **{
                f"val_{key}": value
                for key, value in val_results.items()
            },
            **{
                f"test_{key}": value
                for key, value in test_results.items()
            },
        }

        write_header = (
            not os.path.exists(csv_path)
            or os.stat(csv_path).st_size == 0
        )

        with open(csv_path, "a", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=row.keys(),
            )

            if write_header:
                writer.writeheader()

            writer.writerow(row)

        print(
            f"Completed {dataset_name}: {test_results}"
        )

    except Exception as error:
        print(f"Skipping {file_name}: {error}")
        traceback.print_exc()


print(f"\nResults saved to: {csv_path}")
