# Tabular Foundation Models Can Do Survival Analysis

Official code for the paper **"Tabular Foundation Models Can Do Survival Analysis"**
by **Da In Kim, Wei Siang Lai, and Kelly W. Zhang** (Imperial College London).

We cast survival analysis as a sequence of binary classification problems, which
lets off-the-shelf **tabular foundation models (TFMs)** — such as MITRA and
TabPFN — perform both **static** and **dynamic** survival analysis through
in-context learning, with no survival-specific pretraining. This repository
reproduces the static and dynamic experiments in the paper.

📄 Paper: [arXiv:2601.22259](https://arxiv.org/abs/2601.22259) *(under submission)*

---

## Repository structure

```
.
├── requirements.txt
├── static/
│   ├── static_data/                 # 43 static datasets (CSV)
│   ├── static_baseline_final.py     # CoxPH, RSF, DeepSurv, DeepHit
│   ├── static_mitra_final.py         # MITRA-CF   (our classification formulation)
│   ├── static_tabpfn_final.py        # TabPFN-CF  (our classification formulation)
│   ├── static_xgboost_final.py       # XGBoost-CF (our classification formulation)
│   ├── static_mitra_discrete.py      # MITRA-DH   (discrete-time hazard variant)
│   ├── static_tabpfn_discrete.py     # TabPFN-DH  (discrete-time hazard variant)
│   ├── static_xgboost_discrete.py    # XGBoost-DH (discrete-time hazard variant)
│   ├── static_mitra_multiclass.py    # MITRA-MC   (multi-class PMF variant)
│   ├── static_tabpfn_multiclass.py   # TabPFN-MC  (multi-class PMF variant)
│   ├── static_xgboost_multiclass.py  # XGBoost-MC (multi-class PMF variant)
│   └── performance_static.py         # aggregation / metric tables / plots
└── dynamic/
    ├── data/                        # 5 dynamic (longitudinal) datasets (CSV)
    ├── ddh/                         # local Dynamic-DeepHit implementation
    ├── dynamic_{mitra,tabpfn,xgboost}.py          # -CF variants
    ├── dynamic_{mitra,tabpfn,xgboost}_discrete.py # -DH variants
    ├── dynamic_deephit.py           # Dynamic DeepHit
    ├── landmark_cox.py              # Landmark CoxPH
    ├── landmark_rsf.py              # Landmark RSF
    ├── dynamic_jm.R                 # Joint Model (R)
    ├── metrics.py                   # test metrics from risk CSVs
    └── metrics_val.py               # validation metrics from risk CSVs
```

**Model naming.** A `-CF` (classification-formulation) suffix denotes a model
adapted via our direct cumulative-failure reformulation (the `*_final.py`
scripts). A `-DH` suffix denotes the classical discrete-time hazard variant of
the same classifier (the `*_discrete.py` scripts). A `-MC` suffix denotes the
multi-class variant (the `*_multiclass.py` scripts), which predicts a single
categorical distribution over the discretized time bins (a censoring-conditioned
PMF whose tail sum recovers the survival function), as used by concurrent work.

---

## Installation

```bash
git clone https://github.com/dain20000222/TabularFM4Survival.git
cd TabularFM4Survival
pip install -r requirements.txt
```

Python 3.9 is recommended. The Joint Model baseline (`dynamic/dynamic_jm.R`)
additionally requires R with the `JM` package (`install.packages("JM")`); the
Dynamic-DeepHit dependency (`dynamic/ddh/`) is bundled and needs no separate
installation. A GPU is optional — the foundation-model scripts also run on CPU.

## Data

The 43 static and 5 dynamic pre-processed datasets are included under
`static/static_data/` and `dynamic/data/`. They are derived from
[**SurvSet**](https://github.com/ErikinBC/SurvSet), an open-source time-to-event
dataset repository (Drysdale, 2022); please cite SurvSet if you use them.

---

## Discretization controls

- `--k` **(`-CF`, `-DH`, and baseline scripts)** : step size (in percent)
  between evaluation-time percentiles. The number of discrete intervals is
  `K ≈ round(100 / k)`. The paper reports `K ∈ {5, 10, 15, 20}`:

  | `--k`      | 20 | 10 | 6.6666667 | 5  |
  |------------|----|----|-----------|----|
  | `K` (bins) | 5  | 10 | 15        | 20 |

- `--k` **(`-MC` scripts)** : the number of discrete time bins `K` *directly*
  (an integer, `K ≥ 2`). Here `--k 5` means `K = 5` and `--k 20` means `K = 20`
  — note this differs from the step-size convention above.

- `--binning` *(`-CF` / `-DH` / baseline scripts)* : bin-boundary placement —
  `quantile` (default; observed-event-time quantiles), `equal_width`, or
  `km_quantile` (censoring-aware, from Kaplan–Meier-quantile boundaries).
  Non-default choices add a `_<binning>` suffix to the output filename. The `-MC`
  scripts use quantile bin edges only and take no `--binning` flag.

---

## Usage

### Static experiments

Run from inside `static/` (scripts read `static_data/`):

```bash
cd static

# Our classification formulation (-CF), K=20  (i.e. --k 5)
python static_mitra_final.py   --k 5
python static_tabpfn_final.py  --k 5
python static_xgboost_final.py --k 5

# Discrete-time hazard variants (-DH)
python static_mitra_discrete.py --k 5

# Multi-class PMF variants (-MC); here --k is K directly, so --k 20 means K=20
python static_mitra_multiclass.py --k 20

# Classical / neural baselines
python static_baseline_final.py --k 5 --val_metric ipcw_c_index

# Censoring-aware (Kaplan–Meier-quantile) binning
python static_mitra_final.py --k 5 --binning km_quantile
```

Each script writes an evaluation CSV with validation and test metrics per
dataset (e.g. `static_mitra_evaluation_final_5.csv`). Sweep
`--k ∈ {20, 10, 6.6666667, 5}` for `K ∈ {5, 10, 15, 20}`, then run
`python performance_static.py` to produce the aggregated tables and
K-sensitivity plots.

### Dynamic experiments

Run from inside `dynamic/` (scripts read `data/` and the local `ddh/` module):

```bash
cd dynamic

# Step 1 — generate patient-level risk / survival predictions
python dynamic_mitra.py   --k 20
python dynamic_tabpfn.py  --k 20
python dynamic_xgboost.py --k 20
python dynamic_deephit.py --k 20
python landmark_cox.py    --k 20
python landmark_rsf.py    --k 20
Rscript dynamic_jm.R

# Step 2 — compute metrics from the risk CSVs
python metrics.py       # test-set metrics
python metrics_val.py   # validation-set metrics (for model selection)
```

Each model script evaluates **all** hyperparameter configurations and produces
`*_risk_<k>.csv`; model selection is deferred to the metric-computation stage.

---

## Reproducibility notes

- Random seeds are fixed inside every script (`SEED = 42`).
- Validation data is used only for hyperparameter selection; test data only for
  final reporting.

---

## Citation

```bibtex
@article{kim2026tabular,
  title   = {Tabular Foundation Models Can Do Survival Analysis},
  author  = {Kim, Da In and Lai, Wei Siang and Zhang, Kelly W.},
  journal = {arXiv preprint arXiv:2601.22259},
  year    = {2026}
}
```

---

## License

This code is released under the MIT License; see [`LICENSE`](LICENSE) for details.
The bundled datasets are derived from [SurvSet](https://github.com/ErikinBC/SurvSet)
and remain subject to their original terms.
