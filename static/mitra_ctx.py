"""MITRA-CF with a sample-stratified ("subject") or horizon-stratified ("horizon") context (Appendix D.4, Table 8).

MITRA conditions on at most 8,192 training rows. In the main experiments, these rows are drawn uniformly at random from the
tuples with W > 0 when there are more. Here the context is drawn instead
  subject: round-robin over the subjects: one random row of every subject (in random order), then a second row of every subject
           with at least two rows, ... until 8,192 rows. Every subject is in the context whenever there are at most 8,192 subjects.
  horizon: round-robin over the grid points t_k: the same number of rows per grid point (as far as each grid point has rows).
A subject is identified by its covariate vector (all columns but the last, eval_time), as the expanded training rows of
static_cf_w.py carry no subject id; subjects with identical covariates are pooled.
Everything else is unchanged: static_cf_w.py is run as is, with MitraICL replaced by the class below.

  MITRA_CONTEXT=subject python mitra_ctx.py --model mitra --k 10 --binning km_quantile --oof_folds 5 --data_dir <54 datasets> \
      --tag_suffix _ctxsubject
"""
import os
import runpy
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mitra_icl  # noqa: E402

MODE = os.environ.get("MITRA_CONTEXT", "subject")
assert MODE in ("subject", "horizon"), MODE


def round_robin(groups, size, rng):
    """Indices of `size` rows: one random row per group (groups in random order), then a second one, ..."""
    order = rng.permutation(np.unique(groups))
    pools = {g: rng.permutation(np.flatnonzero(groups == g)) for g in order}
    take, level = [], 0
    while len(take) < size:
        added = False
        for g in order:
            p = pools[g]
            if level < len(p):
                take.append(p[level])
                added = True
                if len(take) == size:
                    break
        if not added:
            break
        level += 1
    return np.sort(np.asarray(take))


class BalancedMitraICL(mitra_icl.MitraICL):
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        if MODE == "subject":
            _, self._ctx_groups = np.unique(X[:, :-1], axis=0, return_inverse=True)
        else:
            _, self._ctx_groups = np.unique(X[:, -1], return_inverse=True)
        self._ctx_groups = np.asarray(self._ctx_groups).ravel()
        return super().fit(X, y)

    def _support_subset(self, rng):
        n = len(self.Xt)
        if n <= self.support_size:
            return np.arange(n)
        idx = round_robin(self._ctx_groups, self.support_size, rng)
        n_g = len(np.unique(self._ctx_groups))
        print(f"    [CTX {MODE}] rows {n} -> {len(idx)}; groups {n_g}, in context {len(np.unique(self._ctx_groups[idx]))}", flush=True)
        return idx


mitra_icl.MitraICL = BalancedMitraICL
sys.argv = [os.path.join(HERE, "static_cf_w.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
