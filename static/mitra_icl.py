# =============================================================================
# MITRA used directly in context (the new MITRA-CF / -DH / -MC)
# =============================================================================
# The pretrained MITRA classifier is used without any adaptation and without
# AutoGluon's TabularPredictor: no 10,000-row subsample, no internal holdout, no
# AutoGluon feature generator. The context is all training rows, or 8,192 rows
# drawn at random once and kept fixed for every query when there are more
# (MITRA's own context size). This is MitraLoRA (mitra_lora.py, unchanged copy
# of the code behind the MITRA-ICL results) with zero adaptation steps; the only
# extension is the number of classes (2-10, MITRA's classification head has 10
# outputs), needed for the multiclass formulation. With n_classes=2 it is
# exactly the earlier MITRA-ICL.
# =============================================================================

import numpy as np
import torch

from mitra_lora import MITRA_DIM_OUTPUT, MitraLoRA


class MitraICL(MitraLoRA):
    def __init__(self, n_classes=2, seed=42, support_size=8192, query_size=1024, device=None, verbose=False):
        if not 2 <= n_classes <= MITRA_DIM_OUTPUT:
            raise ValueError(f"MITRA supports 2-{MITRA_DIM_OUTPUT} classes, got {n_classes}")
        super().__init__(max_steps=0, support_size=support_size, query_size=query_size,
                         seed=seed, device=device, verbose=verbose)
        self.n_classes = n_classes

    def fit(self, X, y):
        y = np.asarray(y).astype(np.int64)
        return super().fit(X, y, groups=np.arange(len(y)))

    def _logits(self, x_support, y_support, x_query):
        batch = self.collator([{
            "x_support": torch.as_tensor(x_support, dtype=torch.float32),
            "y_support": torch.as_tensor(y_support, dtype=torch.int64),
            "x_query": torch.as_tensor(x_query, dtype=torch.float32),
            "y_query": torch.full((len(x_query),), -100, dtype=torch.int64),
        }])
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        with self._autocast():
            out = self.model(
                batch["x_support"], batch["y_support"], batch["x_query"],
                batch["padding_features"], batch["padding_obs_support"], batch["padding_obs_query"],
            )
        # only the first n_classes outputs are real classes (as MitraClassifier.predict_proba)
        return out[0, :, :self.n_classes].float()

    @torch.no_grad()
    def predict_proba(self, X):
        if len(X) == 0:
            return np.zeros((0, self.n_classes))
        return super().predict_proba(X)
