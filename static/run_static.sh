#!/bin/bash
# Static experiments of the paper (Table 1 and the static appendix tables): every model on the 54 datasets of static_data/,
# Kaplan-Meier-quantile grid, K = 5, 10, 15, 20.
#
#   bash run_static.sh                 # everything (sequential)
#   MODEL_GROUPS="2 5 8" bash run_static.sh  # only some model groups (see below)
#   KS="5 10" bash run_static.sh       # only some K
#
# Model groups (each group and K runs in runs/<group>_K<K>/, with its own log.txt, results/ and preds/):
#   0 MITRA-CF      1 TabPFN-CF      2 XGBoost-CF (weights W)
#   3 MITRA-DH      4 TabPFN-DH      5 XGBoost-DH
#   6 CoxPH / RSF / DeepHit / DeepSurv
#   7 MTLR / Neural MTLR / MLP BCESurv
#   8 XGBoost BCESurv (XGBoost-CF with the BCESurv weights)
#   9 MITRA-MC     10 TabPFN-MC     11 XGBoost-MC
#  12 MITRA-CF with a sample-stratified context   13 MITRA-CF with a horizon-stratified context   (Appendix D.4, Table 8)
# The CF, DH and MC models are fitted with 5-fold out-of-fold predictions, which the post-hoc calibration uses.
# After all groups: calibration of the DH and MC models with their own loss, then build_static_long.py.
#
# Python environments (see ../README.md): MITRA needs autogluon, TabPFN-2.5 needs tabpfn 6.0.6, the remaining models need
# scikit-survival, pycox and xgboost. Set PY_MITRA / PY_TABPFN / PY_SURV to the python executables of these environments
# (default: "python" for all three).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
D=$HERE/static_data
PY_MITRA=${PY_MITRA:-python}; PY_TABPFN=${PY_TABPFN:-python}; PY_SURV=${PY_SURV:-python}
MODEL_GROUPS=${MODEL_GROUPS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13"}
KS=${KS:-"5 10 15 20"}
# k = 100 / K is the step of the grid in percent, as the scripts expect it
declare -A KSTEP=([5]=20 [10]=10 [15]=6.6666667 [20]=5)
B=km_quantile
CF=$HERE/static_cf_w.py

for g in $MODEL_GROUPS; do
  for K in $KS; do
    k=${KSTEP[$K]}
    case $g in
      9) W=$HERE/runs/mc_static_mitra_K$K ;; 10) W=$HERE/runs/mc_static_tabpfn_K$K ;; 11) W=$HERE/runs/mc_static_xgboost_K$K ;;
      12) W=$HERE/runs/mitra_ctxsubject_K$K ;; 13) W=$HERE/runs/mitra_ctxhorizon_K$K ;;
      *) W=$HERE/runs/full_km_K${K}_g$g ;;
    esac
    mkdir -p $W && cd $W && ln -sfn $D static_data
    echo "== group $g, K = $K  ($(date))"
    {
    case $g in
      0)  $PY_MITRA  $CF --model mitra   --k $k --binning $B --oof_folds 5 --data_dir $D ;;
      1)  $PY_TABPFN $CF --model tabpfn  --k $k --binning $B --oof_folds 5 --data_dir $D ;;
      2)  $PY_SURV   $CF --model xgboost --k $k --weighting ipcw --censoring km --binning $B --oof_folds 5 --data_dir $D ;;
      3)  $PY_MITRA  $HERE/static_mitra_dh_cal.py   --k $k --binning $B --oof_folds 5 --data_dir $D ;;
      4)  $PY_TABPFN $HERE/static_tabpfn_dh_cal.py  --k $k --binning $B --oof_folds 5 --data_dir $D ;;
      5)  $PY_SURV   $HERE/static_xgboost_dh_cal.py --k $k --binning $B --oof_folds 5 --data_dir $D ;;
      6)  $PY_SURV   $HERE/static_baseline_final.py --k $k --binning $B --val_metric ipcw_c_index ;;
      7)  $PY_SURV   $HERE/static_mtlr.py --model linear_mtlr --k $k --val_metric ipcw_c_index --binning $B
          $PY_SURV   $HERE/static_mtlr.py --model neural_mtlr --k $k --val_metric ipcw_c_index --binning $B
          $PY_SURV   $HERE/static_bce.py --k $k --val_metric ipcw_c_index --binning $B ;;
      8)  $PY_SURV   $CF --model xgboost --k $k --weighting abl --binning $B --oof_folds 5 --data_dir $D ;;
      9)  $PY_MITRA  $CF --model mitra   --formulation mc --k $k --binning $B --oof_folds 5 --data_dir $D ;;
      10) $PY_TABPFN $CF --model tabpfn  --formulation mc --k $k --binning $B --oof_folds 5 --data_dir $D ;;
      11) $PY_SURV   $CF --model xgboost --formulation mc --k $k --binning $B --oof_folds 5 --data_dir $D ;;
      12) MITRA_CONTEXT=subject $PY_MITRA $HERE/mitra_ctx.py --model mitra --k $k --binning $B --oof_folds 5 --data_dir $D --tag_suffix _ctxsubject ;;
      13) MITRA_CONTEXT=horizon $PY_MITRA $HERE/mitra_ctx.py --model mitra --k $k --binning $B --oof_folds 5 --data_dir $D --tag_suffix _ctxhorizon ;;
    esac
    } > log.txt 2>&1
    # post-hoc calibration on the out-of-fold predictions (CPU), with the weights of the CF loss
    case $g in 0|1|2|3|4|5|8|9|10|11|12|13)
      for m in $(ls preds/K$K 2>/dev/null | grep -v -E "_(cal|calibs|platt|platt_t|isotonic)$"); do
        $PY_SURV $HERE/calibrate_static.py --models $m --fit_on oof --K $K --data_dir $D > log_calibrate_$m.txt 2>&1
      done ;;
    esac
  done
done

# Calibration of the DH and MC models with their own loss (CPU; reads the stored out-of-fold predictions)
O=$HERE/runs/own_loss; mkdir -p $O && cd $O
for K in $KS; do
  [ -d $HERE/runs/full_km_K${K}_g3/preds ] && $PY_SURV $HERE/calibrate_own_loss.py --loss dh --preds $HERE/runs/full_km_K${K}_g3/preds --models static_mitra_dh_kmq  --K $K --out results/dh_mitra_K$K
  [ -d $HERE/runs/full_km_K${K}_g4/preds ] && $PY_SURV $HERE/calibrate_own_loss.py --loss dh --preds $HERE/runs/full_km_K${K}_g4/preds --models static_tabpfn_dh_kmq --K $K --out results/dh_tabpfn_K$K
  for M in mitra tabpfn xgboost; do
    [ -d $HERE/runs/mc_static_${M}_K$K/preds ] && $PY_SURV $HERE/calibrate_own_loss.py --loss mc --preds $HERE/runs/mc_static_${M}_K$K/preds --models static_${M}_mc_kmq --K $K --out results/mc_${M}_K$K
  done
done

# Appendix D.3 (Table 7): monotonicity of the CF predictions, clipping vs. isotonic regression (CPU; reads the stored predictions)
TAGS=(static_mitra_cf_full_kmq static_tabpfn_cf_full_kmq static_xgboost_cf_ipcw_km_kmq)
mkdir -p $HERE/runs/monotonicity
for K in $KS; do for g in 0 1 2; do
  [ -d $HERE/runs/full_km_K${K}_g$g/preds ] && $PY_SURV $HERE/monotonicity.py --preds $HERE/runs/full_km_K${K}_g$g/preds --models ${TAGS[$g]} --K $K --out $HERE/runs/monotonicity/g${g}_K$K.csv > /dev/null
done; done

# Figure 5: post-hoc calibration of MITRA-CF and TabPFN-CF fitted with the BCESurv weights instead of the weights W (CPU).
# Runs in runs/calw/K<K>/ on links to the uncalibrated predictions, so that the calibrated outputs above are not touched.
for K in $KS; do
  for g in 0 1; do
    src=$HERE/runs/full_km_K${K}_g$g/preds/K$K/${TAGS[$g]}
    [ -d $src ] || continue
    out=$HERE/runs/calw/K$K/preds/K$K/${TAGS[$g]}; rm -rf $out; mkdir -p $out; ln -sfn $src/*.npz $out/
    (cd $HERE/runs/calw/K$K && $PY_SURV $HERE/cal_abl.py --models ${TAGS[$g]} --fit_on oof --K $K --data_dir $D > log_${TAGS[$g]}.txt 2>&1)
  done
done

# Figure 6: weighted BCE and ROC-AUC of the calibrated CF models
cd $HERE && $PY_SURV classification_metrics.py > /dev/null
cd $HERE && $PY_SURV build_static_long.py
echo "End: $(date)"
