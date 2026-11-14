# =============================================================================
# Temporally-Consistent Survival Analysis (TCSA) dynamic baseline
# =============================================================================
# Reference implementation: Maystre & Russo, "Temporally-Consistent Survival
# Analysis", NeurIPS 2022 (https://github.com/spotify-research/tdsurv).
# The `tdsurv/` package in this folder is vendored verbatim from that
# repository (Apache-2.0); only this driver script is ours.
#
# TCSA models survival on a *discrete* grid of steps: given the state at step
# j, it predicts the hazard at each of the next K steps, and is fitted with
# TD(lambda) backups. We use our landmark grid as that step grid, so step j
# corresponds to landmark tau_j and "surviving Delta steps from step j" is
# exactly S(tau_{j+Delta} | T > tau_j, H_{tau_j}) -- the quantity every other
# dynamic model in this project predicts.
#
# Note that lambda_=1.0 reduces TD(lambda) to ordinary landmarking (this is
# what the reference notebooks call the "landmarking" baseline), so that
# setting doubles as a sanity check against landmark_cox.py.
#
# Output is the standard dynamic risk CSV consumed by metrics.py:
#   pid, tau, t, risk, surv_prob, dataset, hyperparameter, eval_set, k
# =============================================================================

import argparse
import inspect
import os
import random
import warnings

import numpy as np
if not hasattr(np, "infty"):
    np.infty = np.inf
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from tdsurv import CoxPH, Linear, unroll

warnings.filterwarnings("ignore")


# =============================================================================
# Command-Line Arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Run TCSA (temporally-consistent survival analysis) on the "
                "landmark grid used by the other dynamic models."
)
parser.add_argument("--binning", default="km_quantile", choices=["quantile", "km_quantile", "pooled"])  # KMBIN (11-01)
parser.add_argument(
    "--k",
    type=float,
    default=20,
    help="Step size for landmark time percentiles (default: 20).",
)
parser.add_argument(
    "--model",
    type=str,
    default="coxph",
    choices=["coxph", "linear"],
    help="tdsurv model class. 'coxph' has a nonparametric baseline hazard "
         "shared across states; 'linear' fits separate coefficients per step.",
)
args = parser.parse_args()


# =============================================================================
# Reproducibility Settings
# =============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# Data Configuration and I/O Setup
# =============================================================================

# data_mean = raw covariates plus their running means and modes, the data used
# for the dynamic results in the paper (Appendix C)
data_dir = os.path.join("data_mean")
dataset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

risk_out_path = f"dynamic_tcsa_risk_{args.k}" + ("" if args.binning == "quantile" else "_" + args.binning) + ".csv"  # KMBIN

print(f"Total datasets: {len(dataset_files)}")
print(f"Datasets to process: {dataset_files if dataset_files else 'None'}")
print(f"Using k={args.k} for landmark quantiles, model={args.model}")


# =============================================================================
# Hyperparameter Grid
# =============================================================================
# lambda_ is the TD trace-decay parameter. lambda_=1.0 is ordinary
# landmarking / MLE; smaller values use the temporal-difference backups that
# are the contribution of the method.

lambda_values = [0.0, 0.2, 0.5, 1.0]

# The tdsurv objective sums the weighted cross-entropy over sequences rather
# than averaging, so the penalty has to be fairly large before it bites. This
# spans the range used in the reference notebooks.
l2_values = [0.01, 0.1, 1.0, 10.0]

# Outer-loop iterations for the TD backups. Ignored when lambda_ == 1.0, where
# tdsurv forces n_iters=1 internally.
N_ITERS = 30


# =============================================================================
# Helper Functions
# =============================================================================

def build_patient_outcome(df):
    """Aggregate longitudinal rows to one (follow-up time, status) per patient."""
    followup_col = "time2" if "time2" in df.columns else "time"
    outcome = (
        df.sort_values(followup_col)
          .groupby("pid")
          .agg(
              t_event=(followup_col, "max"),
              status_event=("event", "max"),
          )
    )
    return outcome


def locf_state(patient_rows, landmark_time, covariates, default_values):
    """Last observation carried forward at `landmark_time`.

    Mirrors `apply_locf_landmark` in landmark_cox.py: the most recent row with
    observation time <= landmark_time, falling back to the training-set means
    when the patient has no observation yet.
    """
    observed = patient_rows[patient_rows["time"] <= landmark_time]

    if observed.empty:
        return np.array([default_values[c] for c in covariates], dtype=np.float64)

    latest = observed.iloc[-1]
    return np.array([latest[c] for c in covariates], dtype=np.float64)


def grid_position(t_event, taus):
    """Number of landmarks strictly below `t_event`.

    The patient is at risk at tau_0, ..., tau_{position-1}.
    """
    return int(np.searchsorted(np.asarray(taus, dtype=np.float64),
                               float(t_event),
                               side="left"))


def discretize_outcome(t_event, status_event, taus):
    """Map a continuous outcome onto the landmark grid, tdsurv-style.

    Returns `(ts, cs)` where `ts` is the number of steps for the sequence and
    `cs` the censoring indicator, following the convention in tdsurv:

    - uncensored: the patient survives steps 1..ts-1 and fails at step ts, and
      states are required at indices 0..ts-1;
    - censored: the patient is known to survive steps 1..ts, and states are
      required at indices 0..ts.

    Returns `(None, None)` when the patient carries no usable information
    (not at risk at any landmark, or censored before the second landmark).
    """
    horizon = len(taus)
    position = grid_position(t_event, taus)

    if position == 0:
        # Not at risk at any landmark.
        return (None, None)

    if position >= horizon:
        # Survives past the last landmark. We never predict beyond tau_{K-1},
        # so administratively censor there. This also covers events that occur
        # after the final landmark.
        return (horizon - 1, True)

    if int(status_event) == 1:
        # Event falls in (tau_{position-1}, tau_{position}].
        return (position, False)

    ts = position - 1
    if ts < 1:
        # Censored before the second landmark: no observed transition.
        return (None, None)

    return (ts, True)


def collect_states(df_processed, pids, taus, patient_outcome, covariates,
                   default_values, for_training=True):
    """LOCF state at every landmark where each patient is still at risk.

    Returns a dict `{pid: {j: state_vector}}` plus the per-patient `(ts, cs)`
    on the landmark grid (empty when `for_training` is False).

    `for_training=True` keeps only patients that yield a usable tdsurv
    sequence. For evaluation we must not apply that filter: every patient at
    risk at a landmark has to receive a prediction there, because that is the
    cohort the other dynamic models are scored on (see `apply_locf_landmark`
    in landmark_cox.py, which only requires `t_event > tau`). Filtering the
    evaluation cohort would make the metrics incomparable.
    """
    states = {}
    outcomes = {}

    for pid, patient_rows in df_processed.groupby("pid"):
        if pid not in patient_outcome.index:
            continue
        if pid not in pids:
            continue

        t_event = patient_outcome.loc[pid, "t_event"]
        status_event = patient_outcome.loc[pid, "status_event"]

        position = grid_position(t_event, taus)

        if position == 0:
            # Not at risk at any landmark; no model predicts for this patient.
            continue

        if for_training:
            ts, cs = discretize_outcome(t_event, status_event, taus)
            if ts is None:
                continue
            outcomes[pid] = (ts, cs)

        patient_rows = patient_rows.sort_values("time")

        # States are needed at every landmark the patient is at risk for.
        per_patient = {}
        for j in range(position):
            per_patient[j] = locf_state(
                patient_rows, taus[j], covariates, default_values
            )

        states[pid] = per_patient

    return states, outcomes


def positional_features(j, horizon):
    """One-hot encoding of the landmark index."""
    out = np.zeros(horizon, dtype=np.float64)
    out[j] = 1.0
    return out


def assemble_sequences(states, outcomes, horizon, scaler_mean, scaler_scale):
    """Build the (seqs, ts, cs) arrays tdsurv expects.

    `seqs` has shape (N, horizon, n_covariates + horizon); slots beyond a
    patient's at-risk range stay zero, and tdsurv never reads them (a state at
    index m is only accessed while the sequence is still active at step m).
    """
    pids = sorted(states.keys())
    n_cov = len(scaler_mean)
    n_feats = n_cov + horizon

    seqs = np.zeros((len(pids), horizon, n_feats), dtype=np.float64)
    ts = np.zeros(len(pids), dtype=int)
    cs = np.zeros(len(pids), dtype=bool)

    for i, pid in enumerate(pids):
        ts[i], cs[i] = outcomes[pid]
        for j, state in states[pid].items():
            if j >= horizon:
                continue
            scaled = (state - scaler_mean) / scaler_scale
            seqs[i, j] = np.concatenate([scaled, positional_features(j, horizon)])

    return seqs, ts, cs, pids


def build_model(model_name, horizon, n_feats):
    if model_name == "coxph":
        return CoxPH(horizon=horizon, n_feats=n_feats)
    return Linear(horizon=horizon, n_feats=n_feats)


# =============================================================================
# Global accumulator
# =============================================================================

all_risk_rows = []


# =============================================================================
# Main Dataset Processing Loop
# =============================================================================

for i, file_name in enumerate(sorted(dataset_files), start=1):
    dataset_name = file_name.replace(".csv", "")
    file_path = os.path.join(data_dir, file_name)

    print("=" * 70)
    print(f"\nProcessing dataset: {dataset_name} ({i}/{len(dataset_files)})")

    original_df = pd.read_csv(file_path)
    print(f"Original dataset shape: {original_df.shape}")

    patient_outcome = build_patient_outcome(original_df)

    # -------------------------------------------------------------
    # Split patients into train/val/test (70:15:15 by pid)
    # -------------------------------------------------------------
    original_pids = original_df["pid"].unique()
    pid_train, pid_temp = train_test_split(
        original_pids, test_size=0.3, random_state=SEED
    )
    pid_val, pid_test = train_test_split(
        pid_temp, test_size=0.5, random_state=SEED
    )
    print(
        f"Train patients: {len(pid_train)}, "
        f"Val patients: {len(pid_val)}, "
        f"Test patients: {len(pid_test)}"
    )

    train_df = original_df[original_df["pid"].isin(pid_train)].copy()
    val_df = original_df[original_df["pid"].isin(pid_val)].copy()
    test_df = original_df[original_df["pid"].isin(pid_test)].copy()

    # -------------------------------------------------------------
    # Landmarks (taus) computed from TRAIN ONLY
    # -------------------------------------------------------------
    has_time2 = "time2" in original_df.columns
    time_col_for_tau = "time2" if has_time2 else "time"

    event_times_train = train_df.loc[
        train_df["event"] == 1, time_col_for_tau
    ].values
    if len(event_times_train) == 0:
        print(f"Skipping {dataset_name}: No events observed in TRAIN split")
        continue

    percentiles = np.arange(args.k, 100, args.k)
    percentiles = [p for p in percentiles if round(p, 2) < 100]
    taus = list(np.percentile(event_times_train, percentiles))
    if args.binning == "km_quantile":  # KMBIN: unsmoothed KM-quantile landmarks (km_landmarks.py)
        import sys as _s, os as _o
        _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
        from km_landmarks import km_taus
        taus = list(km_taus(train_df, time_col_for_tau, args.k))
    elif args.binning == "pooled":  # POOLED (11-06): pooled measurement and event times (km_landmarks.pooled_taus)
        import sys as _s, os as _o
        _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
        from km_landmarks import pooled_taus
        taus = list(pooled_taus(train_df, time_col_for_tau, args.k))
    horizon = len(taus)

    if horizon < 2:
        print(f"Skipping {dataset_name}: need at least 2 landmarks")
        continue

    print(f"Landmarks (taus) from TRAIN: {[f'{t:.6g}' for t in taus]}")
    print(f"Number of evaluation times: {horizon} (k={args.k})")

    # -------------------------------------------------------------
    # Feature preprocessing (identical to landmark_cox.py)
    # -------------------------------------------------------------
    num_cols = [c for c in original_df.columns if c.startswith("num_")]
    fac_cols = [c for c in original_df.columns if c.startswith("fac_")]
    raw_feature_cols = num_cols + fac_cols

    x_train_ohe = pd.get_dummies(train_df[raw_feature_cols], drop_first=True)
    x_val_ohe = pd.get_dummies(val_df[raw_feature_cols], drop_first=True)
    x_test_ohe = pd.get_dummies(test_df[raw_feature_cols], drop_first=True)

    all_columns = (
        x_train_ohe.columns.union(x_val_ohe.columns).union(x_test_ohe.columns)
    )
    x_train_ohe = x_train_ohe.reindex(columns=all_columns, fill_value=0)
    x_val_ohe = x_val_ohe.reindex(columns=all_columns, fill_value=0)
    x_test_ohe = x_test_ohe.reindex(columns=all_columns, fill_value=0)

    imputer = SimpleImputer(strategy="median")
    x_train_proc = pd.DataFrame(
        imputer.fit_transform(x_train_ohe),
        columns=all_columns,
        index=x_train_ohe.index,
    )
    x_val_proc = pd.DataFrame(
        imputer.transform(x_val_ohe),
        columns=all_columns,
        index=x_val_ohe.index,
    )
    x_test_proc = pd.DataFrame(
        imputer.transform(x_test_ohe),
        columns=all_columns,
        index=x_test_ohe.index,
    )

    non_feature_cols = [
        c for c in original_df.columns if c not in raw_feature_cols
    ]

    train_processed = pd.concat(
        [train_df[non_feature_cols].reset_index(drop=True),
         x_train_proc.reset_index(drop=True)],
        axis=1,
    )
    val_processed = pd.concat(
        [val_df[non_feature_cols].reset_index(drop=True),
         x_val_proc.reset_index(drop=True)],
        axis=1,
    )
    test_processed = pd.concat(
        [test_df[non_feature_cols].reset_index(drop=True),
         x_test_proc.reset_index(drop=True)],
        axis=1,
    )

    feature_cols_processed = all_columns.tolist()
    default_values = {
        col: float(train_processed[col].mean())
        for col in feature_cols_processed
    }

    # -------------------------------------------------------------
    # LOCF states on the landmark grid
    # -------------------------------------------------------------
    train_states, train_outcomes = collect_states(
        train_processed, set(pid_train), taus, patient_outcome,
        feature_cols_processed, default_values, for_training=True,
    )

    if len(train_states) < 10:
        print(f"Skipping {dataset_name}: only {len(train_states)} usable "
              f"training sequences")
        continue

    # Standardization fitted on the training states only. The TD objective is
    # l2-penalised, so the covariate scale matters.
    train_state_matrix = np.vstack(
        [s for per_patient in train_states.values()
         for s in per_patient.values()]
    )
    scaler_mean = train_state_matrix.mean(axis=0)
    scaler_scale = train_state_matrix.std(axis=0)
    scaler_scale[scaler_scale == 0] = 1.0

    seqs_train, ts_train, cs_train, _ = assemble_sequences(
        train_states, train_outcomes, horizon, scaler_mean, scaler_scale
    )
    n_feats = seqs_train.shape[2]

    print(f"Training sequences: {len(seqs_train)} "
          f"(events: {int(np.sum(~cs_train))}), "
          f"horizon: {horizon}, features: {n_feats}")

    # Landmarking-style expansion: every suffix of a sequence becomes its own
    # training sequence, so the model also sees states at later landmarks.
    seqs_unrolled, ts_unrolled, cs_unrolled = unroll(
        seqs_train, ts_train, cs_train
    )
    print(f"After unrolling: {len(seqs_unrolled)} sequences")

    # -------------------------------------------------------------
    # Evaluation states (val and test)
    # -------------------------------------------------------------
    eval_states = {}
    for eval_set_name, eval_processed, eval_pids in [
        ("val", val_processed, set(pid_val)),
        ("test", test_processed, set(pid_test)),
    ]:
        states, _ = collect_states(
            eval_processed, eval_pids, taus, patient_outcome,
            feature_cols_processed, default_values, for_training=False,
        )
        eval_states[eval_set_name] = states
        print(f"{eval_set_name} patients with usable states: {len(states)}")

    # -------------------------------------------------------------
    # Hyperparameter loop
    # -------------------------------------------------------------
    for lambda_ in lambda_values:
        for l2 in l2_values:
            hp_config = {"lambda": lambda_, "l2": l2, "model": args.model}
            print(f"\n{'='*70}")
            print(f"Fitting {hp_config}")

            model = build_model(args.model, horizon, n_feats)

            fit_kwargs = {"lambda_": lambda_, "l2": l2}

            # Only the non-separable models (CoxPH, BetaGeom) run an outer loop
            # over the TD backups; SeparableModel.fit solves one step at a time
            # and takes no `n_iters`.
            accepts_n_iters = (
                "n_iters" in inspect.signature(model.fit).parameters
            )
            if lambda_ < 1.0 and accepts_n_iters:
                fit_kwargs["n_iters"] = N_ITERS

            try:
                model.fit(
                    seqs_unrolled, ts_unrolled, cs_unrolled, **fit_kwargs
                )
            except Exception as error:
                print(f"  Fit failed for {hp_config}: {error}")
                continue

            for eval_set_name in ("val", "test"):
                states = eval_states[eval_set_name]

                for pid, per_patient in states.items():
                    # The last landmark has no future evaluation time, so it
                    # produces neither a risk score nor survival probabilities.
                    for j in sorted(per_patient.keys()):
                        if j >= horizon - 1:
                            continue

                        scaled = (per_patient[j] - scaler_mean) / scaler_scale
                        x = np.concatenate(
                            [scaled, positional_features(j, horizon)]
                        )[None, :]

                        # survival_curve returns [1, S(1|x), ..., S(K|x)];
                        # S(delta) is survival delta landmarks ahead of tau_j.
                        curve = np.asarray(model.survival_curve(x))[0]

                        future_indices = list(range(j + 1, horizon))
                        surv_probs = [
                            float(curve[future_idx - j])
                            for future_idx in future_indices
                        ]

                        risk = float(np.mean([1.0 - s for s in surv_probs]))

                        all_risk_rows.append({
                            "pid": pid,
                            "tau": float(taus[j]),
                            "t": np.nan,
                            "risk": risk,
                            "surv_prob": np.nan,
                            "dataset": dataset_name,
                            "hyperparameter": str(hp_config),
                            "eval_set": eval_set_name,
                        })

                        for future_idx, surv_prob in zip(
                            future_indices, surv_probs
                        ):
                            all_risk_rows.append({
                                "pid": pid,
                                "tau": float(taus[j]),
                                "t": float(taus[future_idx]),
                                "risk": np.nan,
                                "surv_prob": surv_prob,
                                "dataset": dataset_name,
                                "hyperparameter": str(hp_config),
                                "eval_set": eval_set_name,
                            })

            print(f"  Done. Accumulated rows: {len(all_risk_rows)}")


# =============================================================================
# Write output CSV
# =============================================================================

if len(all_risk_rows) > 0:
    results_df = pd.DataFrame(
        all_risk_rows,
        columns=["pid", "tau", "t", "risk", "surv_prob", "dataset",
                 "hyperparameter", "eval_set"],
    )
    results_df["k"] = args.k

    results_df = (
        results_df
        .sort_values(
            ["k", "hyperparameter", "eval_set", "dataset", "pid", "tau", "t"],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    results_df.to_csv(risk_out_path, index=False)
    print(f"\nSaved TCSA predictions to: {risk_out_path}")
    print(f"Total rows: {len(results_df)}")
else:
    print("\nNo rows generated; nothing saved.")

print("\n" + "=" * 70)
print("Done.")
