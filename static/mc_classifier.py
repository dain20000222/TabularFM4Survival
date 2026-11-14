# =============================================================================
# Multi-class (MC) formulation, the classifier side; used by static_cf_w.py / dynamic_cf_w.py
# with --formulation mc. As SurvivalPFN: the input is (X, delta) and the class is the grid interval of the
# observed time U = min(T, C): class j <=> t_{j-1} < U <= t_j (t_0 = -inf, t_m = +inf for the m grid times),
# so there are m + 1 classes. At prediction delta = 1 and S(t_k | x) = P(class > k | x, delta = 1)
# = 1 - sum_{j <= k} p_j, monotone by construction. No loss weights (the event indicator is an input).
# Models as classifiers.py (XGBoost: same hyperparameters; TabPFN-2.5 and MITRA in context).
# MITRA's head has 10 outputs, so MITRA-MC needs at most 10 classes present in the training set; TabPFN's limit
# is the one of the installed tabpfn (an error is raised by the library). Classes that do not occur in the
# training set get probability 0.
# =============================================================================

import os
import sys

import numpy as np
import pandas as pd

ICL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmlr_mitra_icl_rerun_10_13")


def mc_rows(x, T, D):
    """Training inputs of the MC formulation: covariates + the event indicator."""
    X = pd.DataFrame(x).reset_index(drop=True).copy()
    X["delta_input"] = np.asarray(D, dtype=float)
    return X


def mc_labels(T, times):
    """Class = index of the grid interval of U (0 .. len(times))."""
    return np.searchsorted(np.asarray(times, dtype=float), np.asarray(T, dtype=float), side="left")


class MCClassifier:
    def __init__(self, name, seed=42):
        if name not in ("mitra", "tabpfn", "xgboost"):
            raise ValueError(name)
        self.name, self.seed = name, seed

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("the MC formulation has no loss weights")
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y, dtype=int)
        self.classes = np.unique(y)  # the classes present; mapped to 0 .. C-1 for the libraries
        yc = np.searchsorted(self.classes, y)
        C = len(self.classes)
        self.model = None
        if C == 1:  # a single class in the training set: predicted with probability 1
            return self
        if self.name == "xgboost":
            import xgboost as xgb

            self.model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=self.seed,
                                           eval_metric="mlogloss" if C > 2 else "logloss")
            self.model.fit(X.to_numpy(dtype=float), yc)
        elif self.name == "tabpfn":
            import torch
            from tabpfn import TabPFNClassifier

            self.model = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu", ignore_pretraining_limits=True)
            self.model.fit(X, yc)
        else:
            if ICL_DIR not in sys.path:
                sys.path.insert(0, ICL_DIR)
            from mitra_icl import MitraICL

            self.model = MitraICL(n_classes=C, seed=self.seed)  # raises for C > 10 (MITRA's head)
            self.model.fit(X.to_numpy(dtype=np.float32), yc)
        return self

    def predict_class_proba(self, X, n_total):
        """(n, n_total) class probabilities; columns of classes absent from the training set are 0."""
        X = pd.DataFrame(X).reset_index(drop=True)
        out = np.zeros((len(X), n_total))
        if self.model is None:
            out[:, self.classes[0]] = 1.0
            return out
        if self.name == "xgboost":
            p = self.model.predict_proba(X.to_numpy(dtype=float))
        elif self.name == "tabpfn":
            p = self.model.predict_proba(X)
            p = p[:, np.argsort(self.model.classes_)] if hasattr(self.model, "classes_") else p
        else:
            p = self.model.predict_proba(X.to_numpy(dtype=np.float32))
        out[:, self.classes] = p
        return out / out.sum(axis=1, keepdims=True)

    def predict_S(self, X, times):
        """S(t_k | x, delta = 1) at the m grid times of training: (n, m)."""
        X = pd.DataFrame(X).reset_index(drop=True).copy()
        X["delta_input"] = 1.0
        P = self.predict_class_proba(X, len(times) + 1)
        return np.clip(1.0 - np.cumsum(P, axis=1)[:, :len(times)], 0.0, 1.0)

    def close(self):
        if self.name in ("tabpfn", "mitra"):
            import gc

            import torch
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
