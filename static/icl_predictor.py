# =============================================================================
# Drop-in replacement for autogluon.tabular.{TabularDataset, TabularPredictor}
# =============================================================================
# The paper's MITRA scripts are run unchanged except for their import line, so
# that every formulation (CF, DH, MC; static and dynamic) uses MITRA directly in
# context (mitra_icl.MitraICL) instead of through AutoGluon. fit() takes the
# training frame with the label column; AutoGluon arguments such as
# hyperparameters / holdout_frac are accepted and ignored. predict_proba()
# returns a DataFrame with one column per class label, sorted, like AutoGluon.
# =============================================================================

import numpy as np
import pandas as pd

from mitra_icl import MitraICL


def TabularDataset(data):
    return pd.DataFrame(data)


class TabularPredictor:
    def __init__(self, label, path=None, problem_type=None, **kwargs):
        self.label = label

    def fit(self, train_data, **kwargs):
        X = train_data.drop(columns=[self.label])
        y = train_data[self.label].to_numpy()
        self.features_ = list(X.columns)
        self.classes_ = np.unique(y)
        codes = np.searchsorted(self.classes_, y)
        self.model_ = MitraICL(n_classes=max(2, len(self.classes_)))
        self.model_.fit(X.to_numpy(dtype=np.float32), codes)
        return self

    def predict_proba(self, data):
        X = pd.DataFrame(data)[self.features_].to_numpy(dtype=np.float32)
        p = self.model_.predict_proba(X)
        classes = list(self.classes_) + [f"_unused{i}" for i in range(p.shape[1] - len(self.classes_))]
        return pd.DataFrame(p, columns=classes)[list(self.classes_)]
