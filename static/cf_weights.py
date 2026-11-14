# =============================================================================
# Weights of the CF loss and their use for in-context models
# =============================================================================
# Static, for the row of subject i at grid time t_k, with U_i = min(T_i, C_i):
#     W_ik = G(t_k | x_i) / G(U_i | x_i)   if delta_i = 1 and U_i <= t_k   (event before t_k)
#          = 1                             if U_i > t_k                     (still observed at t_k)
#          = 0                             otherwise                        (censored at or before t_k)
# Dynamic, for the row of patient i at landmark tau and horizon t > tau: the same with
# G(. | x_i(tau)) the censoring survival among the training patients at risk at tau and
# all times measured from tau (U_i - tau, t - tau).
# Ablation weights W^abl: 1 for the first two cases, 0 otherwise (the unweighted masked loss).
#
# G models (--censoring): km (Kaplan-Meier, no floor), kmf (Kaplan-Meier floored at 0.05),
# cox (ridge Cox, floored at 0.05), rsf (random survival forest, floored at 0.05); the
# floor is applied to G before the ratio. CensoringModel is ../tmlr_ipcw_10_15/common/censoring.py.
#
# XGBoost: W enters as sample weights, exactly as written (no rescaling).
# MITRA / TabPFN (in context, no training loss): W is realised by thinning the context,
# i.e. row (i, k) is kept with probability W_ik (Bernoulli, fixed seed). Since W_ik <= 1
# this needs no duplication; the expected loss over the kept rows is the weighted loss.
# =============================================================================
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmlr_ipcw_10_15", "common"))
from censoring import CensoringModel  # noqa: E402

CENSORING = {"km": ("km", 0.0), "kmf": ("km", 0.05), "cox": ("cox", 0.05), "rsf": ("rsf", 0.05)}


def fit_censoring(name, X, time, event, seed=42, floor=None):
    """G model `name` fitted on (X, time, 1 - delta); floor = the default of the model unless given."""
    kind, default_floor = CENSORING[name]
    return CensoringModel(kind, seed=seed).fit(X, time, event), (default_floor if floor is None else floor)


def cf_weights(cm, floor, X_rows, U_rows, D_rows, t_rows, weighting):
    """W (weighting='ipcw') or W^abl (weighting='abl'), one value per (subject, grid time) row.
    X_rows: covariates of the row's subject; U_rows = min(T, C); D_rows = delta; t_rows = t_k."""
    U = np.asarray(U_rows, dtype=float)
    D = np.asarray(D_rows, dtype=bool)
    t = np.asarray(t_rows, dtype=float)
    w = np.zeros(len(U))
    event = D & (U <= t)
    observed = U > t
    w[observed] = 1.0
    if weighting == "abl":
        w[event] = 1.0
        return w
    if event.any():
        num = np.maximum(cm.own(X_rows[event], t[event]), floor)
        den = np.maximum(cm.own(X_rows[event], U[event]), floor)
        r = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)  # G(t) = G(U) = 0: no information, weight 1
        w[event] = r  # no clipping; t_k >= U and G non-increasing give r <= 1, reported as weight_max
    return w


def thin(w, seed):
    """Keep mask for an in-context model: row kept with probability w (w in [0, 1])."""
    rng = np.random.default_rng(seed)
    return rng.random(len(w)) < np.asarray(w, dtype=float)
