# Tabular Foundation Models Can Do Survival Analysis

Official code for the paper **"Tabular Foundation Models Can Do Survival Analysis"**
by **Da In Kim, Wei Siang Lai, and Kelly W. Zhang** (Imperial College London).

We cast survival analysis as a sequence of binary classification problems, which
lets off-the-shelf **tabular foundation models (TFMs)**, such as MITRA and
TabPFN, perform both **static** and **dynamic** survival analysis through
in-context learning, with no survival-specific pretraining. This repository
contains the code and the results behind the tables and figures of the paper.

📄 Paper: [arXiv:2601.22259](https://arxiv.org/abs/2601.22259)

> **Versions.** `main` holds the code of the revised paper (54 static and 6 dynamic
> datasets, weighted cumulative-failure loss, post-hoc calibration). The code of
> the first arXiv version (43 static and 5 dynamic datasets) is the commit
> [`b27d69c`](../../tree/b27d69c27a5c39b87822b1aebf838e3dd900000e).

---

## Repository structure

```
.
├── requirements.txt
├── static/                      # static setting (Table 1, Appendix D.2-D.5)
│   ├── static_data/             # 54 static datasets (CSV, from SurvSet)
│   ├── run_static.sh            # runs every model, the calibration and the collection of the results
│   ├── static_cf_w.py           # cumulative-failure (CF) and multi-class (MC) models: MITRA, TabPFN-2.5, XGBoost
│   ├── static_{mitra,tabpfn,xgboost}_dh_cal.py   # discrete-hazard (DH) models
│   ├── static_baseline_final.py # CoxPH, RSF, DeepHit, DeepSurv
│   ├── static_mtlr.py           # MTLR, Neural MTLR
│   ├── static_bce.py            # MLP BCESurv
│   ├── calibrate_static.py      # post-hoc calibration with the weights of the CF loss
│   ├── calibrate_own_loss.py    # post-hoc calibration of the DH / MC models with their own loss
│   ├── cal_abl.py               # calibration with the BCESurv weights (Figure 5)
│   ├── monotonicity.py          # monotonicity violations, clipping vs. isotonic regression (Table 7)
│   ├── mitra_ctx.py             # MITRA-CF with a stratified context (Table 8)
│   ├── classification_metrics.py# weighted BCE / ROC-AUC of the CF models (Figure 6)
│   ├── build_static_long.py     # collects the runs into one long table
│   └── cf_weights.py, classifiers.py, mc_classifier.py, censoring.py,
│       mitra_icl.py, icl_predictor.py, mitra_lora.py     # shared modules
├── dynamic/                     # dynamic setting (Table 2)
│   ├── data/                    # 6 longitudinal datasets (CSV, from SurvSet)
│   ├── data_mean/               # the same with running means / modes of the covariates (model input)
│   ├── split/                   # patient ids of the training / validation / test split
│   ├── run_dynamic.sh           # runs every model, the calibration, the evaluation and the collection
│   ├── dynamic_cf_w.py          # CF models: MITRA, TabPFN-2.5, XGBoost
│   ├── dynamic_{mitra,tabpfn,xgboost}_discrete.py   # DH models
│   ├── landmark_cox.py, landmark_rsf.py             # Landmark CoxPH, Landmark RSF
│   ├── dynamic_deephit.py (+ ddh/)                  # Dynamic DeepHit
│   ├── dynamic_tcsa.py (+ tdsurv/)                  # TCSA
│   ├── dynamic_jm.R                                 # Joint Model (R, JMbayes2)
│   ├── calibrate_dynamic.py, calibrate_dynamic_own_loss.py
│   ├── eval_dynamic_w.py, metrics.py, metrics_val.py   # evaluation
│   ├── build_dynamic_long.py    # collects the evaluation into one long table
│   └── dynamic_preprocessing_mean.py, make_split.py, km_landmarks.py, ... # preprocessing and shared modules
├── results/                     # the results behind the paper (one row per model, dataset and K)
└── analysis/                    # tables and figures of the paper from results/
```

**Model names.** `-CF` is our cumulative-failure formulation, `-DH` the
discrete-hazard formulation and `-MC` the multi-class formulation of the same
classifier. In the result files, `Linear MTLR` is the paper's *MTLR*, `BCESurv`
is *MLP BCESurv*, and `XGBoost-CF (W abl)` is *XGBoost BCESurv* (XGBoost-CF
trained with the BCESurv weights). A row `<model> + calibration` holds the IBS
after post-hoc calibration; the C-Index and Integrated AUC are those of the
uncalibrated model (calibration is used for the IBS only).

---

## Installation

```bash
pip install -r requirements.txt
```

The paper's runs used three Python environments, because the two TFMs need
different package versions:

| Environment | Used for | Key packages |
|---|---|---|
| `PY_MITRA` | MITRA models | `autogluon.tabular==1.4.0` |
| `PY_TABPFN` | TabPFN-2.5 models | `tabpfn==6.0.6` |
| `PY_SURV` | all other models, calibration, evaluation, analysis | `scikit-survival==0.21.0`, `numpy==1.26.4`, `xgboost==3.0.5`, `pycox==0.3.0` |

The run scripts take the python executable of each environment from the
variables `PY_MITRA`, `PY_TABPFN` and `PY_SURV` (default: `python` for all).
TCSA additionally needs `jax` (`PY_TCSA`); its reference implementation
[`tdsurv`](https://github.com/spotify-research/tdsurv) (Apache-2.0) is bundled
unmodified in `dynamic/tdsurv/`. The Joint Model needs R with `JMbayes2`.
The MITRA and TabPFN models need a GPU; all other steps run on CPU.

---

## Running the experiments

### Static setting

```bash
cd static
bash run_static.sh                       # every model, K = 5, 10, 15, 20
MODEL_GROUPS="2 5 8" bash run_static.sh  # only some model groups
KS="5 10" bash run_static.sh             # only some K
```

`run_static.sh` lists the model groups and the command of each model. Every
group and K runs in its own folder `runs/<group>_K<K>/`. The grid points are
the quantiles of the Kaplan-Meier estimate of the survival function
(`--binning km_quantile`); the scripts take the step of the grid in percent,
`--k = 100 / K`. After the model runs, the script fits the post-hoc
calibration and collects everything with `build_static_long.py`, which writes
`static_long.csv` in the format of `results/static_long.csv`.

### Dynamic setting

```bash
cd dynamic
bash run_dynamic.sh                      # every model on the 6 datasets
MODELS="4 5 6" DATASETS="heart csl" bash run_dynamic.sh
```

Each dataset uses its own number of intervals K (the largest K ≤ 10 with at
least 5 training events in every interval): aids 6, csl 8, epileptic 7,
heart 5, oldmort 10, FRTCS 8. `run_dynamic.sh` runs the models, the post-hoc
calibration and the evaluation, and `build_dynamic_long.py` writes
`dynamic_long.csv` in the format of `results/dynamic_long.csv`.

### Tables and figures

The scripts of `analysis/` read `results/` and write to `analysis/out/`.
To use your own runs instead, copy the files written by the two `build_*`
scripts into `results/` (or set `STATIC_LONG` to another long table).

| Paper | Script | Input |
|---|---|---|
| Table 1 | `table1_static.py` | `static_long.csv` |
| Table 2 | `table2_dynamic.py` | `dynamic_long.csv` |
| Figure 2 | `fig2_gains_heatmap.py` | `static_long.csv` |
| Figures 3, 4 | `fig3_fig4_gains_formulations.py` | `static_long.csv`, `static_dataset_rules.csv` |
| Figure 5 | `fig5_weights.py` | `static_long.csv`, `static_calibration_bcesurv_weights.csv` |
| Figure 6 | `fig6_correlation.py` | `static_long.csv`, `static_classification_metrics.csv` |
| Table 6 | `table6_static_by_k.py` | `static_long.csv` |
| Figure 7 | `fig7_k_sensitivity.py` | `static_long.csv` |
| Figure 8 | `fig8_calibration_by_k.py` | `static_long.csv` |
| Table 7 | `table7_monotonicity.py` | `static_monotonicity.csv` |
| Table 8 | `table8_context_sampling.py` | `static_context_sampling.csv` |
| Table 9 | `table9_tfm_vs_xgboost.py` | `static_long.csv` |

`results/static_dataset_rules.csv` lists the 69 static SurvSet datasets with
the three exclusion criteria of the paper (Table 4) and the statistics of the
54 datasets we use (Table 3).

---

## Reproducibility notes

- Random seeds are fixed inside every script (seed 42), and every model uses
  the same split of each dataset (70% training, 15% validation, 15% test).
- Validation data is used only for model selection and calibration; test data
  only for the reported metrics.
- The model scripts are the ones that produced the results of the paper; only
  import paths, comments and the wrapper scripts were adapted to this
  repository. The files of `results/` are the outputs of those runs.
- The scripts of `analysis/` reproduce the paper's tables and figures from
  `results/`.
- Exact numbers depend on package versions (e.g. the XGBoost models differ
  slightly between `xgboost` 2.1, 3.0 and later versions, and the random
  survival forests between `scikit-survival` versions); use the versions of
  `requirements.txt` to reproduce the paper.
- The datasets come from [SurvSet](https://github.com/ErikinBC/SurvSet) 0.2.11.

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
and remain subject to their original terms. `dynamic/tdsurv/` is the reference
implementation of TCSA by Spotify, Inc., licensed under Apache-2.0
(see `dynamic/tdsurv/LICENSE`).
