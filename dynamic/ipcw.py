# =============================================================================
# IPCW metrics with subject-specific censoring weights
# =============================================================================
# (copied from tmlr_nsjc_10_12/ipcw.py; TestG got the landmark shift)
#
# Same estimators as sksurv.metrics (concordance_index_ipcw, cumulative_dynamic_auc,
# brier_score / integrated_brier_score), but the censoring survival G may depend
# on the covariates of each test subject:
#
#   C-index (Uno / Gerds):  pair (i, j) with T_i < T_j, delta_i = 1, T_i < tau,
#                           weight 1 / (G(T_i | x_i) G(T_i | x_j))
#   cumulative/dynamic AUC: cases T_i <= t, delta_i = 1, weight 1 / G(T_i | x_i);
#                           controls T_j > t, weight 1 / G(t | x_j);
#                           mean AUC weighted by the KM of the test set (as sksurv)
#   Brier score at t:       cases est^2 / G(T_i | x_i), controls (1 - est)^2 / G(t | x_j)
#   IBS:                    trapezoidal rule over the evaluation times / (t_max - t_min)
#
# With a marginal G (same for every subject) every formula reduces exactly to
# the sksurv one (test_ipcw.py checks this), so marginal and conditional numbers
# differ only through the censoring model.
#
# `G` is any object with
#   G.own(t, idx)  -> G(t_i | x_i) for the test subjects idx (one time each)
#   G.matrix(t)    -> G(t_j | x_i), shape (n_test, len(t))
# TestG binds a censoring.CensoringModel to the test covariates.
# =============================================================================

import numpy as np
from sksurv.nonparametric import SurvivalFunctionEstimator
from sksurv.util import Surv


_trapz = getattr(np, "trapezoid", None) or np.trapz


class TestG:
    """Censoring model bound to the test covariates (X_test=None for KM).
    shift: the model was fitted on times measured from a landmark (dynamic
    setting); metric times t are absolute, so G is evaluated at t - shift."""

    def __init__(self, model, X_test, n_test, floor=0.0, shift=0.0):
        self.model, self.X, self.n, self.floor, self.shift = model, X_test, n_test, floor, shift

    def own(self, t, idx=None):
        """G(t_i | x_i) for the test subjects idx (default: all), one time each."""
        X = self.X if (self.X is None or idx is None) else self.X[idx]
        return np.maximum(self.model.own(X, np.asarray(t, dtype=float) - self.shift), self.floor)

    def matrix(self, t):
        g = self.model.matrix(self.X, np.atleast_1d(np.asarray(t, dtype=float)) - self.shift)
        return np.maximum(np.broadcast_to(g, (self.n, g.shape[1])), self.floor)


def _check_times(test_time, times):
    times = np.unique(np.atleast_1d(np.asarray(times, dtype=float)))
    if times.max() >= test_time.max() or times.min() < test_time.min():
        raise ValueError(f"all times must be within follow-up time of test data: "
                         f"[{test_time.min()}; {test_time.max()}[")
    return times


def cindex_ipcw(event, time, risk, G, tau=None, tied_tol=1e-8):
    event = np.asarray(event, dtype=bool)
    time = np.asarray(time, dtype=float)
    risk = np.asarray(risk, dtype=float)
    n = len(time)
    active = event & (time < tau) if tau is not None else event.copy()
    idx_ev = np.flatnonzero(active)
    if len(idx_ev) == 0:
        return np.nan
    G_own = G.own(time[idx_ev], idx_ev)
    if np.any(G_own == 0):
        raise ValueError("censoring survival function is zero at one or more time points")
    G_all = G.matrix(time[idx_ev])  # (n, n_events): G(T_i | x_j)

    order = np.argsort(time, kind="mergesort")
    pos = np.empty(n, dtype=int)
    pos[order] = np.arange(n)
    st = time[order]
    num = den = 0.0
    any_pair = False
    for col, i in enumerate(idx_ev):
        # comparable: T_j > T_i, or T_j == T_i and j censored
        start = np.searchsorted(st, time[i], side="right")
        mask = np.zeros(n, dtype=bool)
        mask[order[start:]] = True
        same = order[np.searchsorted(st, time[i], side="left"):start]
        mask[same[~event[same]]] = True
        if not mask.any():
            continue
        any_pair = True
        w = 1.0 / (G_own[col] * G_all[mask, col])
        est = risk[mask]
        ties = np.abs(est - risk[i]) <= tied_tol
        con = (est < risk[i]) & ~ties
        num += (w * con).sum() + 0.5 * (w * ties).sum()
        den += w.sum()
    if not any_pair or den == 0:
        return np.nan
    return num / den


def cumulative_dynamic_auc(event, time, risk, times, G, tied_tol=1e-8):
    event = np.asarray(event, dtype=bool)
    time = np.asarray(time, dtype=float)
    times = _check_times(time, times)
    risk = np.asarray(risk, dtype=float)
    n, m = len(time), len(times)
    if risk.ndim == 1:
        risk = np.broadcast_to(risk[:, None], (n, m))

    w_case = np.zeros(n)
    if event.any():
        g = G.own(time[event], np.flatnonzero(event))
        if np.any(g == 0):
            raise ValueError("censoring survival function is zero at one or more time points")
        w_case[event] = 1.0 / g
    w_ctrl = 1.0 / G.matrix(times)  # (n, m)

    scores = np.empty(m)
    for k in range(m):
        o = np.argsort(-risk[:, k], kind="quicksort")
        est = risk[o, k]
        tt, ee = time[o], event[o]
        is_case = (tt <= times[k]) & ee
        is_ctrl = tt > times[k]
        tp = np.cumsum(is_case * w_case[o])
        fp = np.cumsum(is_ctrl * w_ctrl[o, k])
        tp, fp = tp / tp[-1], fp / fp[-1]
        is_tied = np.abs(np.diff(np.r_[np.inf, est])) <= tied_tol
        drop = np.flatnonzero(is_tied) - 1
        tp, fp = np.delete(tp, drop), np.delete(fp, drop)
        scores[k] = _trapz(np.r_[0, tp], np.r_[0, fp])

    if m == 1:
        return scores, scores[0]
    surv = SurvivalFunctionEstimator().fit(Surv.from_arrays(event=event, time=time))
    s_times = surv.predict_proba(times)
    d = -np.diff(np.r_[1.0, s_times])
    return scores, (scores * d).sum() / (1.0 - s_times[-1])


def brier_scores(event, time, S, times, G):
    event = np.asarray(event, dtype=bool)
    time = np.asarray(time, dtype=float)
    times = _check_times(time, times)
    S = np.asarray(S, dtype=float)
    if S.ndim == 1 and len(times) == 1:
        S = S.reshape(-1, 1)
    g_y = G.own(time)
    g_y = np.where(g_y == 0, np.inf, g_y)
    g_t = G.matrix(times)
    g_t = np.where(g_t == 0, np.inf, g_t)
    out = np.empty(len(times))
    for k, t in enumerate(times):
        is_case = (time <= t) & event
        is_ctrl = time > t
        out[k] = np.mean(S[:, k] ** 2 * is_case / g_y + (1.0 - S[:, k]) ** 2 * is_ctrl / g_t[:, k])
    return times, out


def integrated_brier_score(event, time, S, times, G):
    times, bs = brier_scores(event, time, S, times, G)
    if len(times) < 2:
        raise ValueError("At least two time points must be given")
    return _trapz(bs, times) / (times[-1] - times[0])
