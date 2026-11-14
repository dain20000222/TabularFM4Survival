#!/bin/bash
# Dynamic experiments of the paper (Table 2): every model on the 6 longitudinal datasets of data_mean/, landmarks at the
# quantiles of the Kaplan-Meier estimate, with the K of each dataset (the largest K <= 10 with at least 5 training events
# in every interval): aids 6, csl 8, epileptic 7, heart 5, oldmort 10, FRTCS 8.
#
#   bash run_dynamic.sh                      # everything (sequential)
#   MODELS="4 5 6" bash run_dynamic.sh       # only some models (see below)
#   DATASETS="heart csl" bash run_dynamic.sh # only some datasets
#
# Models:
#   0 MITRA-CF   1 TabPFN-CF   2 MITRA-DH   3 TabPFN-DH   4 XGBoost-CF   5 XGBoost-DH
#   6 Landmark CoxPH and Landmark RSF   7 Dynamic DeepHit   8 TCSA   9 Joint Model (R)
# The CF and DH models are fitted with 5-fold out-of-fold predictions, which the post-hoc calibration uses.
# Each dataset runs in runs/work/<dataset>/; the risk files are collected in runs/preds/. After all models: calibration
# of the DH models with their own loss, evaluation (runs/results/, runs/own_loss/results/), then build_dynamic_long.py.
#
# Python environments (see ../README.md): PY_MITRA (autogluon), PY_TABPFN (tabpfn 6.0.6), PY_SURV (scikit-survival, pycox,
# xgboost), PY_TCSA (tdsurv, JAX); default "python" for all. The joint model needs R with JMbayes2 (RSCRIPT, default "Rscript").
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY_MITRA=${PY_MITRA:-python}; PY_TABPFN=${PY_TABPFN:-python}; PY_SURV=${PY_SURV:-python}; PY_TCSA=${PY_TCSA:-python}
RSCRIPT=${RSCRIPT:-Rscript}
MODELS=${MODELS:-"0 1 2 3 4 5 6 7 8 9"}
DATASETS=${DATASETS:-"aids csl epileptic heart oldmort FRTCS"}
# k = 100 / K is the step of the landmark grid in percent, as the scripts expect it
declare -A KSTEP=([aids]=16.6666667 [csl]=12.5 [epileptic]=14.2857143 [heart]=20 [oldmort]=10 [FRTCS]=12.5)
B=km_quantile
P=$HERE/runs/preds; mkdir -p $P $HERE/runs/results

for ds in $DATASETS; do
  k=${KSTEP[$ds]}
  W=$HERE/runs/work/$ds
  mkdir -p $W/data_mean && ln -sfn $HERE/data_mean/$ds.csv $W/data_mean/$ds.csv && ln -sfn $HERE/split $W/split && cd $W
  for t in $MODELS; do
    echo "== $ds (k = $k), model $t  ($(date))"
    {
    case $t in
      0) $PY_MITRA  $HERE/dynamic_cf_w.py --model mitra   --k $k --binning $B --oof_folds 5 ;;
      1) $PY_TABPFN $HERE/dynamic_cf_w.py --model tabpfn  --k $k --binning $B --oof_folds 5 ;;
      2) $PY_MITRA  $HERE/dynamic_mitra_discrete.py  --k $k --binning $B --oof_folds 5 ;;
      3) $PY_TABPFN $HERE/dynamic_tabpfn_discrete.py --k $k --binning $B --oof_folds 5 ;;
      4) $PY_SURV   $HERE/dynamic_cf_w.py --model xgboost --k $k --weighting ipcw --censoring km --binning $B --oof_folds 5 ;;
      5) $PY_SURV   $HERE/dynamic_xgboost_discrete.py --k $k --binning $B --oof_folds 5 ;;
      6) $PY_SURV   $HERE/landmark_cox.py --k $k --binning $B; $PY_SURV $HERE/landmark_rsf.py --k $k --binning $B ;;
      7) $PY_SURV   $HERE/dynamic_deephit.py --k $k --binning $B ;;
      8) JAX_PLATFORMS=cpu $PY_TCSA $HERE/dynamic_tcsa.py --k $k --model coxph --binning $B ;;
      9) $RSCRIPT   $HERE/dynamic_jm.R --k=$k --binning=$B ;;
    esac
    } > log_model$t.txt 2>&1
  done
  # post-hoc calibration on the out-of-fold predictions (CPU), with the weights of the CF loss
  for r in *_risk_*.csv; do
    o=${r/_risk_/_oof_}
    [ -f "$o" ] && $PY_SURV $HERE/calibrate_dynamic.py --risk $r --oof $o --data_mean data_mean >> log_calibrate.txt 2>&1
  done
  mv *_risk_*.csv *_oof_*.csv *_selection_*.csv $P/ 2>/dev/null
  mv cf_*_weights_*.csv $HERE/runs/results/ 2>/dev/null
done

# Calibration of the DH models with their own loss (CPU; reads the stored risk and out-of-fold files)
O=$HERE/runs/own_loss; mkdir -p $O/preds $O/results
for r in $P/dynamic_*_discrete_risk_*.csv; do
  [ -f "$r" ] || continue
  case $r in *_cal_risk_*) continue ;; esac
  o=${r/_risk_/_oof_}
  [ -f "$o" ] && $PY_SURV $HERE/calibrate_dynamic_own_loss.py --loss dh --risk $r --oof $o --data_mean $HERE/data_mean --out_dir $O/preds
done > $O/calibrate.log 2>&1

# Evaluation (IPCW metrics with the marginal Kaplan-Meier estimate of the censoring distribution)
cd $HERE
$PY_SURV eval_dynamic_w.py --preds $P        --censoring marginal --agg sksurv --data_mean data_mean --out runs/results          > runs/results/eval.log 2>&1
$PY_SURV eval_dynamic_w.py --preds $O/preds  --censoring marginal --agg sksurv --data_mean data_mean --out runs/own_loss/results > runs/own_loss/eval.log 2>&1
$PY_SURV build_dynamic_long.py
echo "End: $(date)"
