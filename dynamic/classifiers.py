# =============================================================================
# Binary classifiers of the CF formulation, configured as in the paper
# =============================================================================
#   mitra    MITRA used directly in context (the paper's MITRA-CF since 2026-10-13):
#            mitra_icl.MitraICL from ../../tmlr_mitra_icl_rerun_10_13, all training
#            rows passed, context of at most 8,192 rows drawn once at random
#   tabpfn   TabPFNClassifier(ignore_pretraining_limits=True)   (static_tabpfn_final.py)
#   xgboost  XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
#                                                               (static_xgboost_final.py)
# fit(X, y, sample_weight=None) / predict_p0(X) -> P(label = 0) (= survival for CF).
# Sample weights (the IPCW loss) are only used by XGBoost: MITRA and TabPFN are
# used in context, without a training loss.
# Imports are lazy so that a run only needs the package of its own model.
# =============================================================================

import os
import sys

import numpy as np
import pandas as pd

ICL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmlr_mitra_icl_rerun_10_13")  # IPCWLOSS: copy of ../tmlr_ipcw_10_15/common/classifiers.py, path adjusted


class Classifier:
    def __init__(self, name, seed=42):
        if name not in ("mitra", "tabpfn", "xgboost"):
            raise ValueError(name)
        self.name, self.seed = name, seed

    def fit(self, X, y, sample_weight=None):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y, dtype=int)
        if sample_weight is not None and self.name != "xgboost":
            raise ValueError("sample weights are only used for xgboost (MITRA / TabPFN are used in context)")
        if self.name == "xgboost":
            import xgboost as xgb

            self.model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                           random_state=self.seed, eval_metric="logloss")
            self.model.fit(X.to_numpy(dtype=float), y, sample_weight=sample_weight)
        elif self.name == "tabpfn":
            import torch
            from tabpfn import TabPFNClassifier

            self.model = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu",
                                          ignore_pretraining_limits=True)
            self.model.fit(X, y)
        else:
            if ICL_DIR not in sys.path:
                sys.path.insert(0, ICL_DIR)
            from mitra_icl import MitraICL

            self.model = MitraICL(n_classes=2, seed=self.seed)
            self.model.fit(X.to_numpy(dtype=np.float32), y)
        return self

    def predict_p0(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        if self.name == "xgboost":
            return self.model.predict_proba(X.to_numpy(dtype=float))[:, 0]
        if self.name == "tabpfn":
            return self.model.predict_proba(X)[:, 0]
        return self.model.predict_proba(X.to_numpy(dtype=np.float32))[:, 0]

    def close(self):
        if self.name in ("tabpfn", "mitra"):
            import gc

            import torch
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
