"""Post-hoc calibration of the CF models fit WITHOUT the loss weights (Figure 5). The paper fits each calibrator with the
weighted binary cross-entropy (weights W of eq. (10)); here the weights are replaced by W^abl (the ratio set to 1, as in the
unweighted loss, i.e. the BCESurv weighting), everything else unchanged: calibrate_static.py is run as is, with cf_weights forced to
weighting="abl".
  python cal_abl.py --models <tags> --fit_on oof --K <K> --data_dir <static_data>    (run in a folder with preds/K<K>/<tag>/)"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = HERE
sys.path.insert(0, SRC)
import cf_weights as cw  # noqa: E402

_orig = cw.cf_weights


def cf_weights_abl(cm, floor, X_rows, U_rows, D_rows, t_rows, weighting):
    return _orig(cm, floor, X_rows, U_rows, D_rows, t_rows, "abl")


cw.cf_weights = cf_weights_abl
sys.argv = [os.path.join(SRC, "calibrate_static.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
