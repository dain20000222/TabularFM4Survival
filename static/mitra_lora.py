# =============================================================================
# LoRA fine-tuning of the MITRA backbone for the classification formulation
# =============================================================================
# The static and dynamic scripts in this folder query MITRA exactly as before:
# each training row is (covariates, eval_time[, tau]) with the label
# 1{event by eval_time}, and censored patients only contribute rows up to their
# censoring time. The only difference is that, before predicting, low-rank
# adapters on the attention projections of the pretrained Tab2D network are
# trained on the training rows with the same masked binary cross-entropy.
#
# Why we do not use AutoGluon's own `fine_tune=True`:
#   AutoGluon splits the training rows into context and query sets at random
#   (row level). Our training set has many rows per patient (one per
#   eval_time / landmark) whose labels are deterministic functions of each
#   other, so a row-level split puts rows of the same patient on both sides and
#   the network can learn to copy labels from the context. Its early stopping
#   uses the same kind of split. Here every split is by patient (`groups`):
#   a patient's rows are either all in the context or all in the query.
#
# Everything below uses AutoGluon's Mitra building blocks (Tab2D, Preprocessor,
# CollatorWithPadding) with the settings AutoGluon uses for in-context
# prediction, so with zero training steps the model is plain MITRA
# in-context learning.
# =============================================================================

from __future__ import annotations

import contextlib
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from autogluon.tabular.models.mitra._internal.config.enums import Task
from autogluon.tabular.models.mitra._internal.data.collator import CollatorWithPadding
from autogluon.tabular.models.mitra._internal.data.preprocessor import Preprocessor
from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D

MITRA_CHECKPOINT = "autogluon/mitra-classifier"
MITRA_DIM_OUTPUT = 10  # size of the Tab2D classification head (max. number of classes)
N_CLASSES = 2          # our task is binary: 0 = no event by eval_time, 1 = event

LORA_TARGETS = {
    # q/k/v/o of both attention blocks (across rows and across features) in all 12 layers
    "attn": ("q", "k", "v", "o"),
    # additionally the two feed-forward blocks of every layer
    "attn_ffn": ("q", "k", "v", "o", "linear1", "linear2", "linear3", "linear4"),
}

# The pretrained network is loaded once per process and copied for every fit
_BASE_MODEL_CACHE = {}


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_base_model(device, checkpoint=MITRA_CHECKPOINT):
    # Loaded on the target device, as AutoGluon does: Tab2D decides at
    # construction whether to use flash attention (CUDA + flash_attn installed)
    key = (checkpoint, device)
    if key not in _BASE_MODEL_CACHE:
        # Tab2D.from_pretrained loads the checkpoint tensors onto `device`,
        # but constructs the module itself on CPU. Move the module as well so
        # its parameters match the batches created by `_logits`.
        _BASE_MODEL_CACHE[key] = Tab2D.from_pretrained(checkpoint, device=device).to(device)
    return copy.deepcopy(_BASE_MODEL_CACHE[key])


class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B A x, with W, b frozen and B initialised to zero."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / rank
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        update = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return self.base(x) + update * self.scaling


def inject_lora(model: nn.Module, rank: int, alpha: float, targets) -> list:
    """Freeze `model` and wrap every nn.Linear whose attribute name is in `targets`."""
    for p in model.parameters():
        p.requires_grad_(False)

    to_wrap = []
    for module in model.modules():
        for name, child in module.named_children():
            if name in targets and isinstance(child, nn.Linear):
                to_wrap.append((module, name, child))

    lora_params = []
    for module, name, child in to_wrap:
        wrapped = LoRALinear(child, rank, alpha)
        setattr(module, name, wrapped)
        lora_params += [wrapped.lora_A, wrapped.lora_B]
    return lora_params


class MitraLoRA:
    """MITRA binary classifier with optional LoRA adaptation.

    fit(X, y, groups, X_val, y_val): trains the adapters with patient-grouped
        episodes and keeps the step with the lowest validation BCE (step 0, i.e.
        no adaptation, is a candidate). Without validation data, trains for
        exactly `n_steps` steps.
    predict_proba(X): P(y=0), P(y=1) for each row, using the training rows as
        context, like MitraClassifier.predict_proba.

    With max_steps=0 (or n_steps=0) no adapter is created and this is plain
    in-context learning.
    """

    def __init__(
        self,
        rank=8,
        alpha=16.0,
        lr=3e-4,
        max_steps=200,
        eval_every=10,
        patience=5,
        warmup_steps=10,
        weight_decay=0.0,
        max_grad_norm=1.0,
        support_size=8192,
        query_size=1024,
        query_frac=0.2,
        max_val_rows=8192,
        targets="attn",
        device=None,
        seed=42,
        checkpoint=MITRA_CHECKPOINT,
        verbose=True,
    ):
        self.rank = rank
        self.alpha = alpha
        self.lr = lr
        self.max_steps = max_steps
        self.eval_every = eval_every
        self.patience = patience
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.support_size = support_size  # max. context rows (MITRA's own default is 8192)
        self.query_size = query_size      # max. query rows per forward pass
        self.query_frac = query_frac      # fraction of training patients used as query in an episode
        self.max_val_rows = max_val_rows  # validation rows used for early stopping
        self.targets = LORA_TARGETS[targets] if isinstance(targets, str) else tuple(targets)
        self.device = device or default_device()
        self.seed = seed
        self.checkpoint = checkpoint
        self.verbose = verbose
        self.collator = CollatorWithPadding(max_features=None, pad_to_max_features=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _autocast(self):
        if self.device.startswith("cuda"):
            # Same precision as AutoGluon's Mitra trainer (required by flash attention)
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _logits(self, x_support, y_support, x_query):
        """Logits of the two classes for x_query given the context (numpy arrays)."""
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
        # Only the first two outputs are real classes; MitraClassifier.predict_proba
        # also drops the others before the softmax, so the loss below is the BCE of
        # exactly the probabilities we predict with.
        return out[0, :, :N_CLASSES].float()

    def _support_subset(self, rng):
        n = len(self.Xt)
        if n <= self.support_size:
            return np.arange(n)
        return rng.choice(n, size=self.support_size, replace=False)

    @torch.no_grad()
    def _val_loss(self, Xv, yv, support_idx):
        self.model.eval()
        losses = []
        for start in range(0, len(Xv), self.query_size):
            logits = self._logits(self.Xt[support_idx], self.yt[support_idx], Xv[start:start + self.query_size])
            target = torch.as_tensor(yv[start:start + self.query_size], dtype=torch.int64, device=logits.device)
            losses.append(F.cross_entropy(logits, target, reduction="sum").item())
        return float(np.sum(losses) / len(Xv))

    def _lora_state(self):
        return [p.detach().cpu().clone() for p in self.lora_params]

    def _load_lora_state(self, state):
        with torch.no_grad():
            for p, s in zip(self.lora_params, state):
                p.copy_(s.to(p.device))

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X, y, groups, X_val=None, y_val=None, n_steps=None):
        t0 = time.time()
        rng = np.random.RandomState(self.seed)
        torch.manual_seed(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64)
        groups = np.asarray(groups)
        assert len(X) == len(y) == len(groups)

        # Preprocessing exactly as AutoGluon's Mitra trainer: mean imputation,
        # drop constant columns, random sign flip per feature (fixed per fit).
        # The preprocessor draws the flips from numpy's global RNG.
        np.random.seed(self.seed)
        self.preprocessor = Preprocessor(
            dim_embedding=None,
            n_classes=MITRA_DIM_OUTPUT,
            dim_output=MITRA_DIM_OUTPUT,
            use_quantile_transformer=False,
            use_feature_count_scaling=False,
            use_random_transforms=False,
            shuffle_classes=False,
            shuffle_features=False,
            random_mirror_regression=True,
            random_mirror_x=True,
            task=Task.CLASSIFICATION,
        )
        self.preprocessor.fit(X.copy(), y.copy())
        self.Xt = self.preprocessor.transform_X(X.copy())
        self.yt = self.preprocessor.transform_y(y.copy())

        self.model = load_base_model(self.device, self.checkpoint)
        self.lora_params = []
        self.best_step = 0
        self.steps_run = 0
        self.step0_val_loss = np.nan
        self.best_val_loss = np.nan
        self.history = []

        total_steps = self.max_steps if n_steps is None else int(n_steps)
        if total_steps <= 0:
            self.model.eval()
            self.train_seconds = time.time() - t0
            return self

        self.lora_params = inject_lora(self.model, self.rank, self.alpha, self.targets)
        optimizer = torch.optim.AdamW(self.lora_params, lr=self.lr, weight_decay=self.weight_decay)
        warmup = max(1, self.warmup_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup))

        # Patient-level index: rows of each group
        _, group_codes = np.unique(groups, return_inverse=True)
        order = np.argsort(group_codes, kind="mergesort")
        boundaries = np.flatnonzero(np.diff(group_codes[order])) + 1
        group_rows = np.split(order, boundaries)
        n_groups = len(group_rows)
        if n_groups < 2:
            raise ValueError("Need at least two patients to build context/query episodes")
        n_query_groups = min(n_groups - 1, max(1, int(round(self.query_frac * n_groups))))

        # Validation set for early stopping (fixed rows and fixed context)
        use_val = X_val is not None and y_val is not None and len(X_val) > 0 and n_steps is None
        if use_val:
            Xv = np.asarray(X_val, dtype=np.float32)
            yv = np.asarray(y_val).astype(np.int64)
            if len(Xv) > self.max_val_rows:
                keep = np.random.RandomState(self.seed + 1).choice(len(Xv), self.max_val_rows, replace=False)
                Xv, yv = Xv[keep], yv[keep]
            Xv = self.preprocessor.transform_X(Xv.copy())
            yv = self.preprocessor.transform_y(yv.copy())
            val_support = self._support_subset(np.random.RandomState(self.seed + 2))

            self.step0_val_loss = self._val_loss(Xv, yv, val_support)
            self.best_val_loss = self.step0_val_loss
            best_state = self._lora_state()
            self.history.append((0, np.nan, self.step0_val_loss))
            self._log(f"    [LoRA] step 0: val BCE {self.step0_val_loss:.4f}")
            bad_evals = 0

        step = 0
        while step < total_steps:
            step += 1
            self.model.train()

            perm = rng.permutation(n_groups)
            q_idx = np.concatenate([group_rows[g] for g in perm[:n_query_groups]])
            s_idx = np.concatenate([group_rows[g] for g in perm[n_query_groups:]])
            if len(q_idx) > self.query_size:
                q_idx = rng.choice(q_idx, size=self.query_size, replace=False)

            while True:
                s_used = s_idx if len(s_idx) <= self.support_size else rng.choice(s_idx, size=self.support_size, replace=False)
                try:
                    logits = self._logits(self.Xt[s_used], self.yt[s_used], self.Xt[q_idx])
                    target = torch.as_tensor(self.yt[q_idx], dtype=torch.int64, device=logits.device)
                    loss = F.cross_entropy(logits, target)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    break
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if self.support_size <= 512:
                        raise
                    self.support_size = self.support_size // 2
                    self._log(f"    [LoRA] CUDA OOM, reducing support_size to {self.support_size}")

            torch.nn.utils.clip_grad_norm_(self.lora_params, self.max_grad_norm)
            optimizer.step()
            scheduler.step()
            self.steps_run = step

            if use_val and (step % self.eval_every == 0 or step == total_steps):
                val_loss = self._val_loss(Xv, yv, val_support)
                self.history.append((step, float(loss.item()), val_loss))
                self._log(f"    [LoRA] step {step}: train BCE {loss.item():.4f}, val BCE {val_loss:.4f}")
                if val_loss < self.best_val_loss - 1e-6:
                    self.best_val_loss = val_loss
                    self.best_step = step
                    best_state = self._lora_state()
                    bad_evals = 0
                else:
                    bad_evals += 1
                    if bad_evals >= self.patience:
                        self._log(f"    [LoRA] early stopping at step {step} (best step {self.best_step})")
                        break

        if use_val:
            self._load_lora_state(best_state)
        else:
            self.best_step = step

        self.model.eval()
        self.train_seconds = time.time() - t0
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, X):
        """Returns an (n, 2) array [P(y=0), P(y=1)], as TabularPredictor.predict_proba."""
        self.model.eval()
        Xq = self.preprocessor.transform_X(np.asarray(X, dtype=np.float32).copy())
        # One fixed context for all queries, so all eval_times of a patient
        # are predicted from the same context rows
        support_idx = self._support_subset(np.random.RandomState(self.seed + 3))
        probs = []
        for start in range(0, len(Xq), self.query_size):
            logits = self._logits(self.Xt[support_idx], self.yt[support_idx], Xq[start:start + self.query_size])
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        if len(probs) == 0:
            return np.zeros((0, N_CLASSES))
        return np.concatenate(probs, axis=0)

    def summary(self):
        """Fit statistics written to the fit log."""
        return {
            "rank": self.rank if self.lora_params else 0,
            "lr": self.lr if self.lora_params else 0.0,
            "targets": "+".join(self.targets) if self.lora_params else "none",
            "steps_run": self.steps_run,
            "best_step": self.best_step,
            "step0_val_loss": self.step0_val_loss,
            "best_val_loss": self.best_val_loss,
            "n_train_rows": len(self.Xt),
            "support_size": self.support_size,
            "n_lora_params": int(sum(p.numel() for p in self.lora_params)),
            "train_seconds": round(self.train_seconds, 1),
        }
