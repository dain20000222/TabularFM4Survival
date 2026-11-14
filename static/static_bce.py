# Copy of ../tmlr_baselines_rebuttal_6_9/bce_evaluation.py (static discrete-time baselines) + --binning km_quantile (KM-quantile grid of Table 7, 11-03);
# everything else unchanged. Output: static_<model>_evaluation_final_<k>_ipcw_c_index[_km_quantile].csv
#!/usr/bin/env python3
"""BCESurv static survival analysis with pycox.

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

# Compatibility with older pycox versions that still refer to np.infty.
if not hasattr(np, "infty"):
    np.infty = np.inf

import pandas as pd
import torch
import torchtuples as tt

from pycox.models import BCESurv
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
    """Reset all relevant RNGs before each independent model fit."""
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
        "Run BCESurv static survival analysis with an explicit "
        "np.percentile-based common discrete-time grid."
    )
)

parser.add_argument(
    "--k",
    type=float,
    default=10,
    help=(
        "Percentile spacing for the common grid. For k=10 the grid is "
        "the 10th, 20th, ..., and 90th percentiles, plus pycox boundary cuts."
    ),
)

parser.add_argument(
    "--val_metric",
    type=str,
    default="c_index",
    choices=[
        "c_index",
        "ipcw_c_index",
        "ibs",
        "mean_auc",
    ],
    help="Validation metric used for hyperparameter selection.",
)

parser.add_argument('--binning', default='quantile', choices=['quantile', 'km_quantile'])  # KMBIN (11-03)
args = parser.parse_args()


# =============================================================================
# Training Configuration
# =============================================================================

MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
BATCH_SIZE = 256


class InMemoryEarlyStopping(tt.callbacks.EarlyStopping):
    """Early stopping that keeps best weights isolated per model trial."""

    def __init__(self, patience):
        super().__init__(
            patience=patience,
            checkpoint_model=False,
            load_best=False,
        )
        self.best_state = None

    def on_fit_start(self):
        super().on_fit_start()
        self.best_state = {
            name: parameter.detach().clone()
            for name, parameter in self.model.net.state_dict().items()
        }

    def on_epoch_end(self):
        previous_best = self.cur_best
        stop_signal = super().on_epoch_end()

        if self.cur_best != previous_best:
            self.best_state = {
                name: parameter.detach().clone()
                for name, parameter in self.model.net.state_dict().items()
            }

        return stop_signal

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
    f"static_bce_surv_evaluation_final_"
    f"{args.k:.1f}_ipcw_c_index" + ("_km_quantile" if args.binning == "km_quantile" else "") + ".csv"
)


# =============================================================================
# Hyperparameter Configuration
# =============================================================================
# BCESurv itself is the single model. We only tune its neural-network
# architecture and learning rate.

bce_grid = {
    "dropout": [0.1, 0.3, 0.5],
    "first_layer_size": [32, 64, 128],
    "second_layer_size": [16, 32, 64],
    "learning_rate": [1e-2, 1e-3, 1e-4],
}


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


def make_common_grid(times, k):
    """
    Construct the percentile evaluation horizons used by the baseline.

    LabTransDiscreteTime learns the model's internal cuts separately, just as
    it does in the DeepHit implementation.
    """
    times = np.asarray(times, dtype=np.float64)

    if times.size == 0:
        raise ValueError("Cannot construct a time grid from an empty array.")

    if not (0 < k < 100):
        raise ValueError("k must satisfy 0 < k < 100.")

    percentiles = np.arange(k, 100.0, k, dtype=np.float64)
    percentiles = np.unique(percentiles)
    percentiles.sort()

    evaluation_grid = np.percentile(times, percentiles).astype(np.float64)
    grid = np.unique(evaluation_grid)

    if grid.size == 0:
        raise ValueError("The percentile evaluation grid is empty.")

    return grid


# =============================================================================
# Evaluation Helpers
# =============================================================================

def get_valid_eval_times(common_grid, evaluation_times, train_times=None):
    """
    Select evaluation times strictly from the common grid.

    The returned horizons are evaluated by interpolating BCESurv predictions.
    """
    common_grid = np.asarray(common_grid, dtype=np.float64)
    evaluation_times = np.asarray(evaluation_times, dtype=np.float64)

    if evaluation_times.size == 0:
        return np.array([], dtype=np.float64)

    if train_times is None:
        upper_train_time = np.max(evaluation_times)
    else:
        upper_train_time = np.max(
            np.asarray(train_times, dtype=np.float64)
        )

    # sksurv requires evaluation times strictly inside the observed test
    # follow-up interval. Adjacent floating-point values avoid unit-dependent
    # offsets such as 0.01.
    lower = np.nextafter(np.min(evaluation_times), np.inf)
    upper = np.nextafter(
        min(np.max(evaluation_times), upper_train_time),
        -np.inf,
    )
    mask = (common_grid > lower) & (common_grid < upper)

    eval_times = common_grid[mask]

    return np.asarray(eval_times, dtype=np.float64)


def administratively_censor_at(y, boundary):
    """Censor outcomes at a boundary unsupported by IPCW estimation."""
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
    """Build the neural network used by BCESurv."""
    return tt.practical.MLPVanilla(
        input_features,
        [
            int(params["first_layer_size"]),
            int(params["second_layer_size"]),
        ],
        out_features,
        batch_norm=True,
        dropout=float(params["dropout"]),
    )


def build_bce_surv_model(input_features, labtrans, params):
    """Construct BCESurv with the learned labtrans cuts as duration_index."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    net = build_network(
        input_features=input_features,
        out_features=int(labtrans.out_features),
        params=params,
    )

    # Follow the pycox construction pattern: pass the optimizer class,
    # then set the learning rate on the instantiated optimizer.
    model = BCESurv(
        net,
        tt.optim.Adam,
        device=device,
        duration_index=labtrans.cuts,
    )

    model.optimizer.set_lr(float(params["learning_rate"]))

    return model


# =============================================================================
# Prediction Helpers
# =============================================================================

def predict_survival(model, x_data):
    """Return S(t|x) with rows=samples and columns=common-grid times."""
    x_data = np.asarray(x_data, dtype=np.float32)

    survival_df = model.predict_surv_df(x_data)

    return survival_df.T.to_numpy(dtype=np.float64)


def survival_at_eval_times(model, x_data, eval_times):
    """
    Interpolate BCESurv survival probabilities at evaluation horizons.
    """
    survival_df = model.predict_surv_df(
        np.asarray(x_data, dtype=np.float32)
    )

    eval_times = np.asarray(eval_times, dtype=np.float64)
    model_times = np.asarray(survival_df.index, dtype=np.float64)
    survival_curves = survival_df.T.to_numpy(dtype=np.float64)
    survival = np.asarray(
        [np.interp(eval_times, model_times, curve) for curve in survival_curves],
        dtype=np.float64,
    )
    survival = np.clip(survival, 0.0, 1.0)

    return survival


def mean_cumulative_risk(model, x_data, eval_times):
    """
    Scalar risk used for C-index and cumulative AUC:

        R_i = mean_j [1 - S_i(t_j)]

    Larger values mean higher predicted event risk.
    """
    survival = survival_at_eval_times(model, x_data, eval_times)

    risk_scores = np.mean(1.0 - survival, axis=1)

    return risk_scores


def bce_prediction_diagnostics(model, x_data, eval_times):
    """Summarize risk collapse and BCESurv's non-monotone curves.

    BCESurv estimates each survival-time output independently. These
    diagnostics intentionally report raw predictions; no post-processing is
    applied to make them monotone.
    """
    full_survival = np.clip(predict_survival(model, x_data), 0.0, 1.0)
    risk_scores = mean_cumulative_risk(model, x_data, eval_times)

    increases = np.diff(full_survival, axis=1) > 1e-8
    n_increases = int(np.count_nonzero(increases))
    n_transitions = int(increases.size)

    return {
        "risk_min": float(np.min(risk_scores)),
        "risk_max": float(np.max(risk_scores)),
        "risk_mean": float(np.mean(risk_scores)),
        "risk_std": float(np.std(risk_scores)),
        "unique_risk": int(np.unique(risk_scores).size),
        "n_survival_increases": n_increases,
        "survival_increase_rate": (
            float(n_increases / n_transitions) if n_transitions else 0.0
        ),
        "mean_survival_first_grid": float(np.mean(full_survival[:, 0])),
        "mean_survival_final_grid": float(np.mean(full_survival[:, -1])),
    }


# =============================================================================
# Training Helpers
# =============================================================================

def get_best_epoch_from_log(log):
    """Return best epoch according to minimum validation loss."""
    history = log.to_pandas()

    if "val_loss" not in history.columns:
        raise RuntimeError(
            "Validation loss ('val_loss') was not found in the training log."
        )

    val_loss = pd.to_numeric(
        history["val_loss"], errors="coerce"
    ).to_numpy(dtype=np.float64)

    valid = np.isfinite(val_loss)
    if not np.any(valid):
        raise RuntimeError("No finite validation losses were recorded.")

    valid_indices = np.where(valid)[0]
    best_index = valid_indices[np.argmin(val_loss[valid])]

    return (
        int(best_index + 1),
        float(val_loss[best_index]),
        int(len(val_loss)),
    )


# =============================================================================
# Evaluation Metrics
# =============================================================================

def evaluate_bce_surv(
    model,
    x_data,
    y_test,
    y_train,
    eval_times,
):
    """Evaluate BCESurv using the requested sksurv metrics."""
    x_data = np.asarray(x_data, dtype=np.float32)
    eval_times = np.asarray(eval_times, dtype=np.float64)

    if eval_times.size < 1:
        raise ValueError("No valid evaluation times were supplied.")

    survival = survival_at_eval_times(
        model,
        x_data,
        eval_times,
    )

    risk_scores = np.mean(
        1.0 - survival,
        axis=1,
    )

    risk_scores = np.asarray(risk_scores, dtype=np.float64)

    if not np.all(np.isfinite(risk_scores)):
        raise ValueError("Non-finite risk scores encountered.")

    # -------------------------------------------------------------
    # C-index
    # -------------------------------------------------------------
    c_index = concordance_index_censored(
        y_test["event"],
        y_test["time"],
        risk_scores,
    )[0]

    # -------------------------------------------------------------
    # IPCW C-index
    # -------------------------------------------------------------
    # Use the same tau convention as every other model merged into the shared
    # results table (static_baseline_final.py, static_tabpfn_final.py): the
    # largest follow-up time of the training cohort. The evaluation labels are
    # administratively censored below this boundary by the caller, so the
    # fitted censoring survival stays positive.
    tau = float(np.max(y_train["time"]))

    ipcw_c_index = concordance_index_ipcw(
        y_train,
        y_test,
        risk_scores,
        tau=tau,
    )[0]

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
        "mean_auc": float(mean_auc)
    }


# =============================================================================
# Model Fitting
# =============================================================================

def fit_bce_surv(
    x_train,
    y_train,
    num_bins,
    params,
    epochs=MAX_EPOCHS,
    x_val=None,
    y_val=None,
    patience=EARLY_STOPPING_PATIENCE,
):
    """
    Fit BCESurv using the DeepHit-style learned discretization.

    The integer number of bins is supplied, and the label transformer learns
    its cuts from the training outcomes.
    """
    reset_seed()

    # -------------------------------------------------------------
    # Standardization
    # -------------------------------------------------------------
    preprocessor = make_pipeline(StandardScaler())

    x_train_scaled = preprocessor.fit_transform(x_train).astype(
        np.float32, copy=False
    )

    # -------------------------------------------------------------
    # DeepHit-style discrete-time grid learned from training outcomes.
    # -------------------------------------------------------------
    labtrans = LabTransDiscreteTime(int(num_bins))

    train_times = np.asarray(y_train["time"], dtype=np.float64)
    train_events = np.asarray(y_train["event"], dtype=bool)

    y_train_idx, y_train_event = labtrans.fit_transform(
        train_times,
        train_events,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    x_train_tensor = torch.as_tensor(
        x_train_scaled,
        dtype=torch.float32,
        device=device,
    )

    y_train_idx_tensor = torch.as_tensor(
        y_train_idx,
        dtype=torch.long,
        device=device,
    )

    y_train_event_tensor = torch.as_tensor(
        y_train_event,
        dtype=torch.float32,
        device=device,
    )

    model = build_bce_surv_model(
        x_train_tensor.shape[1],
        labtrans,
        params,
    )

    # -------------------------------------------------------------
    # Validation data
    # -------------------------------------------------------------
    val_data = None

    if x_val is not None and y_val is not None:
        x_val_scaled = preprocessor.transform(x_val).astype(
            np.float32, copy=False
        )

        val_times = np.asarray(y_val["time"], dtype=np.float64)
        val_events = np.asarray(y_val["event"], dtype=bool)

        y_val_idx, y_val_event = labtrans.transform(
            val_times,
            val_events,
        )

        x_val_tensor = torch.as_tensor(
            x_val_scaled,
            dtype=torch.float32,
            device=device,
        )

        y_val_idx_tensor = torch.as_tensor(
            y_val_idx,
            dtype=torch.long,
            device=device,
        )

        y_val_event_tensor = torch.as_tensor(
            y_val_event,
            dtype=torch.float32,
            device=device,
        )

        val_data = (
            x_val_tensor,
            (
                y_val_idx_tensor,
                y_val_event_tensor,
            ),
        )

    callbacks = []

    if val_data is not None:
        callbacks.append(InMemoryEarlyStopping(patience=patience))

    log = model.fit(
        x_train_tensor,
        (
            y_train_idx_tensor,
            y_train_event_tensor,
        ),
        batch_size=BATCH_SIZE,
        epochs=epochs,
        callbacks=callbacks,
        val_data=val_data,
        verbose=False,
    )

    return model, preprocessor, labtrans, log


# =============================================================================
# Hyperparameter Optimization
# =============================================================================

def grid_search_bce_surv(
    x_train,
    y_train,
    x_val,
    y_val_fit,
    y_val_eval,
    num_bins,
    val_eval_times,
):
    """Grid-search the single BCESurv model over neural-network settings."""
    best_params = None
    best_epochs = None
    best_score = np.inf if args.val_metric == "ibs" else -np.inf
    best_val_loss = None
    best_epochs_run = None

    for values in product(*bce_grid.values()):
        params = dict(zip(bce_grid.keys(), values))

        try:
            # Use an identical seed for every trial so comparisons are not
            # confounded by a different random initialization.
            reset_seed()

            model, preprocessor, labtrans, log = fit_bce_surv(
                x_train,
                y_train,
                num_bins,
                params,
                epochs=MAX_EPOCHS,
                x_val=x_val,
                y_val=y_val_fit,
                patience=EARLY_STOPPING_PATIENCE,
            )

            trial_best_epoch, trial_best_val_loss, trial_epochs_run = (
                get_best_epoch_from_log(log)
            )

            x_val_scaled = preprocessor.transform(x_val).astype(
                np.float32, copy=False
            )

            results = evaluate_bce_surv(
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
                best_epochs = trial_best_epoch
                best_score = float(score)
                best_val_loss = float(trial_best_val_loss)
                best_epochs_run = int(trial_epochs_run)

            print(
                f"  params={params} "
                f"{args.val_metric}={score:.4f} "
                f"best_val_loss={trial_best_val_loss:.6f} "
                f"best_epoch={trial_best_epoch} "
                f"epochs_run={trial_epochs_run}"
            )

        except Exception:
            print(f"  trial failed with params {params}")
            traceback.print_exc()

    return (
        best_params,
        best_epochs,
        best_score,
        best_val_loss,
        best_epochs_run,
    )


# =============================================================================
# Main Dataset Processing Loop
# =============================================================================

for file_name in dataset_files:
    try:
        dataset_name = file_name.removesuffix(".csv")

        df = pd.read_csv(
            os.path.join(data_dir, file_name)
        )

        # -------------------------------------------------------------
        # Remove patient ID
        # -------------------------------------------------------------
        if "pid" in df.columns:
            df = df.drop(columns=["pid"])

        # -------------------------------------------------------------
        # Covariates
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
        # Train / test split: 85 / 15
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
        # Train / validation split: approximately 70 / 15
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
        # One-hot encoding
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

        # Align validation to training columns.
        x_train_ohe, x_val_ohe = x_train_ohe.align(
            x_val_ohe,
            join="left",
            axis=1,
            fill_value=0,
        )

        # Align test to training columns.
        x_train_ohe, x_test_ohe = x_train_ohe.align(
            x_test_ohe,
            join="left",
            axis=1,
            fill_value=0,
        )

        # -------------------------------------------------------------
        # Imputation
        # -------------------------------------------------------------
        imputer = SimpleImputer().fit(x_train_ohe)

        x_train_imputed = imputer.transform(x_train_ohe).astype(
            np.float32, copy=False
        )
        x_val_imputed = imputer.transform(x_val_ohe).astype(
            np.float32, copy=False
        )
        x_test_imputed = imputer.transform(x_test_ohe).astype(
            np.float32, copy=False
        )

        # -------------------------------------------------------------
        # TUNING GRID: training outcomes only
        # -------------------------------------------------------------
        tuning_grid = (km_grid(y_train, args.k) if args.binning == "km_quantile" else make_common_grid(
            y_train["time"],
            args.k,
        ))

        # Validation follow-up outside training support cannot be used by
        # IPCW metrics. LabTransDiscreteTime applies the same right-censoring
        # rule when constructing validation training labels.
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
            train_times=y_train["time"],
        )

        if len(val_eval_times) == 0:
            print(
                f"No validation evaluation times for {dataset_name}; skipping"
            )
            continue

        print(f"\nDataset: {dataset_name}")
        print("Model: BCESurv")
        print(f"Validation metric: {args.val_metric}")
        print(f"Maximum epochs: {MAX_EPOCHS}")
        print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
        print("Training-only percentile grid:")
        print(tuning_grid)
        print(f"Number of grid points: {len(tuning_grid)}")
        print(f"Grid dtype: {tuning_grid.dtype}")
        print(f"Grid maximum: {tuning_grid[-1]}")
        print(f"Training maximum: {np.max(y_train['time'])}")
        print(f"Validation evaluation times: {val_eval_times}")

        # -------------------------------------------------------------
        # Hyperparameter selection
        # -------------------------------------------------------------
        (
            best_params,
            best_epochs,
            best_score,
            best_val_loss,
            best_epochs_run,
        ) = grid_search_bce_surv(
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

        print(f"\nBest BCESurv params for {dataset_name}: {best_params}")
        print(f"Best validation {args.val_metric}: {best_score:.4f}")
        print(f"Best validation BCESurv loss: {best_val_loss:.6f}")
        print(f"Best epoch: {best_epochs}")
        print(f"Epochs actually run: {best_epochs_run}")

        # -------------------------------------------------------------
        # Refit selected hyperparameters on TRAIN ONLY for validation
        # reporting. This keeps validation evaluation comparable to the
        # selected hyperparameter search.
        # -------------------------------------------------------------
        reset_seed()

        (
            val_model,
            val_preprocessor,
            val_labtrans,
            _val_log,
        ) = fit_bce_surv(
            x_train_imputed,
            y_train,
            len(val_eval_times),
            best_params,
            epochs=best_epochs,
        )

        x_val_scaled = val_preprocessor.transform(
            x_val_imputed
        ).astype(np.float32, copy=False)

        val_results = evaluate_bce_surv(
            val_model,
            x_val_scaled,
            y_val_filtered,
            y_train,
            val_eval_times,
        )

        # -------------------------------------------------------------
        # Test-time handling
        # -------------------------------------------------------------
        # Test observations beyond the train+validation support are censored
        # for IPCW evaluation at the adjacent representable time point.
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

        # The final model may use train+validation outcomes because tuning is
        # complete. Its learned cuts are fit independently.
        common_grid = (km_grid(y_trainval_aligned, args.k) if args.binning == "km_quantile" else make_common_grid(
            y_trainval_aligned["time"],
            args.k,
        ))

        # Test evaluation times come from the baseline percentile horizons.
        test_eval_times = get_valid_eval_times(
            common_grid,
            y_test_filtered["time"],
            train_times=y_trainval_aligned["time"],
        )

        if len(test_eval_times) == 0:
            print(
                f"No test evaluation times for {dataset_name}; skipping"
            )
            continue

        # -------------------------------------------------------------
        # Final model on TRAIN + VALIDATION
        # -------------------------------------------------------------
        x_trainval_imputed = np.vstack(
            [
                x_train_imputed,
                x_val_imputed,
            ]
        ).astype(np.float32, copy=False)

        assert x_trainval_imputed.shape[0] == len(y_trainval_aligned)

        reset_seed()

        (
            test_model,
            test_preprocessor,
            test_labtrans,
            _test_log,
        ) = fit_bce_surv(
            x_trainval_imputed,
            y_trainval_aligned,
            len(test_eval_times),
            best_params,
            epochs=best_epochs,
        )

        x_test_scaled = test_preprocessor.transform(
            x_test_imputed
        ).astype(np.float32, copy=False)

        # -------------------------------------------------------------
        # Test evaluation
        # -------------------------------------------------------------
        test_results = evaluate_bce_surv(
            test_model,
            x_test_scaled,
            y_test_filtered,
            y_trainval_aligned,
            test_eval_times,
        )
        test_diagnostics = bce_prediction_diagnostics(
            test_model,
            x_test_scaled,
            test_eval_times,
        )

        print(f"Validation results: {val_results}")
        print(f"Test results: {test_results}")
        print("Raw BCESurv diagnostics:")
        for key, value in test_diagnostics.items():
            print(f"  {key}: {value}")

        # -------------------------------------------------------------
        # Save results
        # -------------------------------------------------------------
        row = {
            "dataset_name": dataset_name,
            "k": args.k,
            "val_metric": args.val_metric,
            "n_discrete_time_points": len(test_labtrans.cuts),
            "n_eval_times": len(test_eval_times),
            "best_epoch": best_epochs,
            "best_val_loss": best_val_loss,
            "early_stopping_epochs_run": best_epochs_run,
            **test_diagnostics,
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

        with open(
            csv_path,
            "a",
            newline="",
        ) as output_file:
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

    except Exception:
        print(f"Skipping {file_name} due to error:")
        traceback.print_exc()


print(f"\nResults saved to: {csv_path}")
