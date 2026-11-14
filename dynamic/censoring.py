# =============================================================================
# Censoring-distribution models G(t | x) = P(C > t | x)
# =============================================================================
# (copied from tmlr_nsjc_10_12/censoring.py; additions at the end: selection of
# the censoring model by validation IBS, and the IPCW weights of the CF loss)
#
# Used for
#   - the IPCW-weighted CF loss (XGBoost-CF):  ipcw_cf_weights()
#   - the IPCW metrics with a covariate-dependent censoring model (ipcw.py)
#
#   kind="km"   marginal Kaplan-Meier of the censoring times; identical to the
#               estimator sksurv uses inside its IPCW metrics
#               (CensoringDistributionEstimator).
#   kind="cox"  Cox model of the censoring hazard on standardized covariates,
#               Breslow baseline. The ridge penalty is chosen by 3-fold CV
#               (concordance) over ALPHAS; an unpenalized fit on small,
#               high-dimensional data (nki70, dataOvarian1) gives censoring
#               survivals of ~1e-100 and unusable weights.
#   kind="rsf"  random survival forest of the censoring times.
# A model that cannot be imported raises; only fitting failures fall back to KM.
#
# Every G is a right-continuous step function of t; left=True returns G(t-).
# Before the first censoring time G = 1, after the last it stays at its last
# value. Models fall back to KM when there are fewer than `min_censored`
# censored subjects or the fit fails (`kind_used` records what was used).
# =============================================================================

import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

if not hasattr(np, "infty"):  # older scikit-survival uses np.infty, removed in NumPy 2
    np.infty = np.inf


def _step(grid, values, t, left=False, before=1.0):
    """Evaluate a right-continuous step function given by (grid, values) at t."""
    t = np.asarray(t, dtype=float)
    idx = np.searchsorted(grid, t, side="left" if left else "right") - 1
    out = np.where(idx >= 0, values[np.clip(idx, 0, len(values) - 1)], before)
    return out


ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


class CensoringModel:
    def __init__(self, kind="km", alpha="auto", min_censored=10, seed=42):
        if kind not in ("km", "cox", "rsf"):
            raise ValueError(f"unknown censoring model {kind}")
        self.kind = kind
        self.alpha = alpha
        self.min_censored = min_censored
        self.seed = seed

    # ------------------------------------------------------------------ fit
    def fit(self, X, time, event):
        time = np.asarray(time, dtype=float)
        event = np.asarray(event, dtype=bool)
        self.kind_used = self.kind
        if self.kind != "km" and (X is None or (~event).sum() < self.min_censored):
            self.kind_used = "km"
        # Import outside the try: a broken installation (e.g. scikit-survival built for
        # NumPy 1 under NumPy 2: "numpy.dtype size changed") must stop the run instead
        # of silently giving KM results labelled cox / rsf.
        if self.kind_used == "cox":
            from sksurv.linear_model import CoxPHSurvivalAnalysis  # noqa: F401
        elif self.kind_used == "rsf":
            from sksurv.ensemble import RandomSurvivalForest  # noqa: F401
        if self.kind_used == "cox":
            try:
                self._fit_cox(np.asarray(X, dtype=float), time, event)
            except Exception as e:  # singular design, no convergence, ...
                print(f"  censoring Cox failed ({e}); using KM")
                self.kind_used = "km"
        elif self.kind_used == "rsf":
            try:
                self._fit_rsf(np.asarray(X, dtype=float), time, event)
            except Exception as e:
                print(f"  censoring RSF failed ({e}); using KM")
                self.kind_used = "km"
        if self.kind_used == "km":
            self._fit_km(time, event)
        return self

    def _fit_km(self, time, event):
        cens = CensoringDistributionEstimator().fit(Surv.from_arrays(event=event, time=time))
        self.km_grid_ = np.asarray(cens.unique_time_, dtype=float)
        self.km_prob_ = np.asarray(cens.prob_, dtype=float)

    def _fit_cox(self, X, time, event):
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        self.scaler_ = StandardScaler().fit(X)
        Xs = np.nan_to_num(self.scaler_.transform(X))
        keep = Xs.std(axis=0) > 0
        self.keep_ = keep
        y = Surv.from_arrays(event=~event, time=time)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alpha = self.alpha
            if alpha == "auto":
                from sklearn.model_selection import GridSearchCV, KFold
                cv = GridSearchCV(CoxPHSurvivalAnalysis(ties="breslow", n_iter=200), {"alpha": ALPHAS},
                                  cv=KFold(3, shuffle=True, random_state=self.seed), error_score=np.nan)
                cv.fit(Xs[:, keep], y)
                alpha = cv.best_params_["alpha"]
            self.alpha_ = alpha
            m = CoxPHSurvivalAnalysis(alpha=alpha, ties="breslow", n_iter=200)
            m.fit(Xs[:, keep], y)
        if not np.all(np.isfinite(m.coef_)):
            raise ValueError("non-finite coefficients")
        self.coef_ = m.coef_
        self.H0_grid_ = np.asarray(m.cum_baseline_hazard_.x, dtype=float)
        self.H0_vals_ = np.asarray(m.cum_baseline_hazard_.y, dtype=float)

    def _fit_rsf(self, X, time, event):
        from sksurv.ensemble import RandomSurvivalForest

        self.rsf_ = RandomSurvivalForest(n_estimators=200, min_samples_leaf=15, max_features="sqrt",
                                         n_jobs=-1, random_state=self.seed)
        self.rsf_.fit(X, Surv.from_arrays(event=~event, time=time))
        self.rsf_grid_ = np.asarray(self.rsf_.unique_times_, dtype=float)

    # -------------------------------------------------------------- predict
    def _rsf_S(self, X):
        """RSF survival curves of the rows of X, cached (the metrics ask for the same
        subjects many times)."""
        X = np.ascontiguousarray(np.asarray(X, dtype=float))
        key = (X.shape, hash(X.tobytes()))
        cache = self.__dict__.setdefault("_S_cache", {})
        if key not in cache:
            if len(cache) > 256:
                cache.clear()
            cache[key] = self.rsf_.predict_survival_function(X, return_array=True)
        return cache[key]

    def _risk(self, X):
        Xs = np.nan_to_num(self.scaler_.transform(np.asarray(X, dtype=float)))
        return np.exp(np.clip(Xs[:, self.keep_] @ self.coef_, -50, 50))

    def matrix(self, X, times, left=False):
        """G(times[j] | x_i) for every row i of X: shape (n, len(times))."""
        times = np.atleast_1d(np.asarray(times, dtype=float))
        n = len(X) if X is not None else 1
        if self.kind_used == "km":
            g = _step(self.km_grid_, self.km_prob_, times, left)
            return np.broadcast_to(g, (n, len(times))).copy()
        if self.kind_used == "cox":
            H = _step(self.H0_grid_, self.H0_vals_, times, left, before=0.0)
            return np.exp(-np.outer(self._risk(X), H))
        S = self._rsf_S(X)
        idx = np.searchsorted(self.rsf_grid_, times, side="left" if left else "right") - 1
        out = np.ones((S.shape[0], len(times)))
        ok = idx >= 0
        out[:, ok] = S[:, idx[ok]]
        return out

    def own(self, X, t, left=False):
        """G(t_i | x_i) for row i, one time per row: shape (n,)."""
        t = np.asarray(t, dtype=float)
        if self.kind_used == "km":
            return _step(self.km_grid_, self.km_prob_, t, left)
        if self.kind_used == "cox":
            H = _step(self.H0_grid_, self.H0_vals_, t, left, before=0.0)
            return np.exp(-self._risk(X) * H)
        S = self._rsf_S(X)
        idx = np.searchsorted(self.rsf_grid_, t, side="left" if left else "right") - 1
        return np.where(idx >= 0, S[np.arange(len(t)), np.clip(idx, 0, None)], 1.0)

    def keep_prob(self, X, T_event, t):
        """P(subject still under observation at t | event at T_event < t, x)
        = G(t | x) / G(T_event- | x), one row per (x, T_event, t)."""
        num = self.own(X, t, left=False)
        den = self.own(X, T_event, left=True)
        p = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 1.0)
        return np.clip(p, 0.0, 1.0)


# =============================================================================
# Selection of the censoring model by validation IBS
# =============================================================================
KINDS = ("km", "cox", "rsf")


def censoring_ibs(model, X_eval, time_eval, event_eval, time_fit, event_fit, n_times=9):
    """IBS of the censoring model as a survival model of the censoring time C,
    on held-out subjects (the roles of event and censoring are swapped: C is the
    'event', a death censors C). Evaluation times: deciles 10..90 of the observed
    censoring times of the held-out subjects, inside their follow-up range."""
    from sksurv.metrics import integrated_brier_score

    time_eval = np.asarray(time_eval, dtype=float).copy()
    cens_eval = ~np.asarray(event_eval, dtype=bool)
    # as the paper's test outcomes: held-out times beyond the fitted range are
    # truncated just below the largest fitted time and counted as not observed
    t_max = float(np.max(time_fit))
    beyond = time_eval >= t_max
    time_eval[beyond] = t_max - 1e-6
    cens_eval = cens_eval & ~beyond
    if cens_eval.sum() < 2:
        return np.nan
    t = np.unique(np.percentile(time_eval[cens_eval], np.linspace(10, 90, n_times)))
    t = t[(t > time_eval.min()) & (t < time_eval.max())]
    if len(t) < 2:
        return np.nan
    y_fit = Surv.from_arrays(event=~np.asarray(event_fit, dtype=bool), time=np.asarray(time_fit, dtype=float))
    y_ev = Surv.from_arrays(event=cens_eval, time=time_eval)
    if len(t) < 2:
        return np.nan
    try:
        G = np.clip(np.nan_to_num(model.matrix(X_eval, t), nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        return float(integrated_brier_score(y_fit, y_ev, G, t))
    except Exception as e:
        print(f"  censoring IBS failed ({e})")
        return np.nan


def select_censoring(pairs, kinds=KINDS, seed=42):
    """Pick the censoring model with the lowest mean validation IBS.

    pairs: list of (X_fit, time_fit, event_fit, X_eval, time_eval, event_eval);
    one pair for a static dataset (train -> val), one per landmark for a dynamic
    one (training patients at risk -> validation patients at risk, times measured
    from the landmark). Returns (kind, {kind: mean IBS}). Ties and all-NaN go to
    the simpler model (order of `kinds`)."""
    scores = {}
    for kind in kinds:
        vals = []
        for X_fit, t_fit, e_fit, X_ev, t_ev, e_ev in pairs:
            cm = CensoringModel(kind, seed=seed).fit(X_fit, t_fit, e_fit)
            if cm.kind_used != kind:  # fell back to KM: not a separate candidate here
                vals.append(np.nan)
                continue
            vals.append(censoring_ibs(cm, X_ev, t_ev, e_ev, t_fit, e_fit))
        vals = np.asarray(vals, dtype=float)
        scores[kind] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    best = min(finite, key=lambda k: (finite[k], kinds.index(k))) if finite else "km"
    return best, scores


# =============================================================================
# IPCW weights of the CF loss
# =============================================================================
def ipcw_cf_weights(model, X_rows, T_rows, D_rows, t_rows, floor=0.05):
    """Weights of the IPCW cumulative-failure loss, one per (subject, horizon) row:

        delta_i 1(U_i <= t_k) / G(U_i- | x_i)   label 1 (event by t_k)
        1(U_i > t_k)          / G(t_k  | x_i)   label 0 (still at risk at t_k)
        0                                        otherwise (censored at or before t_k)

    (U_i = min(T_i, C_i), delta_i = 1(T_i <= C_i)); G is floored at `floor`.
    Rows with weight 0 can be dropped. G(U_i-) is the left limit: the subject was
    still uncensored just before its event time."""
    T_rows = np.asarray(T_rows, dtype=float)
    D_rows = np.asarray(D_rows, dtype=bool)
    t_rows = np.asarray(t_rows, dtype=float)
    w = np.zeros(len(T_rows))
    ev = D_rows & (T_rows <= t_rows)
    risk = T_rows > t_rows
    if ev.any():
        w[ev] = 1.0 / np.maximum(model.own(X_rows[ev], T_rows[ev], left=True), floor)
    if risk.any():
        w[risk] = 1.0 / np.maximum(model.own(X_rows[risk], t_rows[risk], left=False), floor)
    return w
