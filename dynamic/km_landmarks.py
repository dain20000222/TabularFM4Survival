# Unsmoothed KM-quantile landmarks, the rule of dynamic_rebuttal_km_3_28/*.py and
# ../tmlr_ipcw_loss_10_25/dynamic_cf_w.py: patient-level outcome (last follow-up time, last event status) of the
# TRAINING patients, Kaplan-Meier curve of the event time, n_bins equally spaced levels strictly between its first
# and last value, landmark = the first KM time at which the curve is at or below the level.
import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator


def k_to_percentile_grid(k, tol=1e-6):
    kf = float(k)
    K = 100.0 / kf
    K_int = int(round(K))
    if np.isclose(K, K_int, rtol=0.0, atol=tol) and K_int >= 2:
        return np.arange(1, K_int) * (100.0 / K_int), K_int - 1, K_int
    percentiles = np.arange(kf, 100.0, kf)
    return percentiles, int(len(percentiles)), None


def km_taus(train_df, time_col, k):
    po = train_df.sort_values(time_col).groupby("pid").agg(t_event=(time_col, "max"), status_event=("event", "max"))
    event_times = po.loc[po["status_event"] == 1, "t_event"].to_numpy(dtype=float)
    percentiles, n_bins, _ = k_to_percentile_grid(k)
    if len(event_times) == 0:
        return np.array([], dtype=float)
    km_time, km_surv = kaplan_meier_estimator(po["status_event"].astype(bool).to_numpy(), po["t_event"].to_numpy(dtype=float))
    if len(km_time) == 0:
        return np.unique(np.percentile(event_times, percentiles))
    levels = np.linspace(float(km_surv[0]), float(km_surv[-1]), n_bins + 2)[1:-1]
    taus = [float(km_time[i]) for i in (np.searchsorted(-km_surv, -s) for s in levels) if i < len(km_time)]
    return np.unique(np.asarray(taus if taus else np.percentile(event_times, percentiles), dtype=float))


def pooled_taus(train_df, time_col_for_tau, k):
    """POOLED landmarks, the rule of --binning pooled in ../tmlr_ipcw_loss_10_25/dynamic_cf_w.py: percentiles
    np.arange(k, 100, k) of the pooled covariate measurement times after each training patient's baseline row and the
    event times T (one per training patient with an event; censoring times are not used). Ties (scheduled visits) can
    give fewer than K - 1 distinct landmarks."""
    per_pid = train_df.groupby("pid").agg(t_end=(time_col_for_tau, "max"), status=("event", "max"))
    ev = per_pid.loc[per_pid["status"] == 1, "t_end"].to_numpy(dtype=float)
    if "time" in train_df.columns and time_col_for_tau != "time":
        t0 = train_df.groupby("pid")["time"].transform("min")
        meas = train_df.loc[train_df["time"] > t0, "time"].to_numpy(dtype=float)
    else:
        meas = np.array([])
    pooled = np.concatenate([meas, ev])
    if len(pooled) == 0:
        return np.array([], dtype=float)
    percentiles = [q for q in np.arange(k, 100, k) if round(q, 2) < 100]
    return np.unique(np.asarray(np.percentile(pooled, percentiles), dtype=float))
