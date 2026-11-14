# =============================================================================
# Dynamic Joint Model Survival Analysis Implementation
# =============================================================================
# This R script performs dynamic survival analysis using Joint Models (JM) with 
# longitudinal biomarker data. Joint models simultaneously fit longitudinal
# sub-models for time-varying biomarkers and survival sub-models, accounting
# for the association between the longitudinal process and time-to-event.
# The analysis uses landmark time points for dynamic risk prediction.
# =============================================================================

suppressPackageStartupMessages({
  library(JMbayes2)     # Joint modeling framework
  library(dplyr)        # Data manipulation
  library(readr)        # File reading
  library(survival)     # Survival analysis
  library(nlme)         # Linear mixed-effects models
})

# =============================================================================
# Configuration and Setup  
# =============================================================================

# Set random seed for reproducibility
SEED <- 42
set.seed(SEED)

# Data directory containing longitudinal datasets
data_dir <- "data_mean"
dataset_files <- list.files(data_dir, pattern = "\\.csv$", full.names = TRUE)

# Landmark time step parameter (from command line or default)
k <- 20.0
binning <- "km_quantile"

# Parse command-line arguments: --k=<float> --binning=<quantile|km_quantile>
cli_args <- commandArgs(trailingOnly = TRUE)
if (length(cli_args) > 0) {
  i <- 1
  while (i <= length(cli_args)) {
    arg <- cli_args[[i]]

    if (startsWith(arg, "--k=")) {
      k_val <- suppressWarnings(as.numeric(sub("^--k=", "", arg)))
      if (!is.na(k_val)) k <- k_val
    } else if (arg == "--k" && i < length(cli_args)) {
      k_val <- suppressWarnings(as.numeric(cli_args[[i + 1]]))
      if (!is.na(k_val)) {
        k <- k_val
        i <- i + 1
      }
    } else if (startsWith(arg, "--binning=")) {
      binning <- sub("^--binning=", "", arg)
    } else if (arg == "--binning" && i < length(cli_args)) {
      binning <- cli_args[[i + 1]]
      i <- i + 1
    }

    i <- i + 1
  }
}

if (!(binning %in% c("quantile", "km_quantile", "pooled"))) {  # POOLED (11-06)
  stop("Invalid --binning value. Use one of: quantile, km_quantile, pooled")
}

if (!is.finite(k) || k <= 0 || k >= 100) {
  stop("k must be in (0, 100)")
}

# Format k parameter for consistent file naming
if (k %% 1 == 0) {
  k_str <- sprintf("%.1f", k)
} else {
  k_str <- format(k, scientific = FALSE, trim = TRUE)
}
binning_suffix <- if (binning == "quantile") "" else paste0("_", binning)
out_path <- paste0("joint_model_risk_", k_str, binning_suffix, ".csv")

cat("Using k=", k, " with binning=", binning, " for landmark generation\n", sep = "")

# =============================================================================
# Hyperparameter Configuration
# =============================================================================

# Grid of regularization parameters for Cox model component
alpha_grid <- c(1e-6, 1e-3, 1e-1)

# Global results accumulator
all_rows <- list()

# Debug flag for detailed logging
DEBUG <- TRUE
dbg <- function(...) { if (isTRUE(DEBUG)) cat("[DEBUG]", ..., "\n") }

# =============================================================================
# Biomarker Configuration
# =============================================================================
# 
# Hardcoded biomarker mapping for different datasets.
# This mapping specifies which longitudinal variables should be modeled
# as biomarkers in the joint model for each dataset.
# Multiple biomarkers per dataset are supported.
# =============================================================================

biomarker_map <- list(
  "aids.csv"      = c("num_CD4", "num_CD4_mean"),                                 # CD4 count markers
  "csl.csv"       = c("num_prot", "num_prot_prev", "num_prot_mean", "num_prot_prev_mean"),  # Protein markers  
  "epileptic.csv" = c("num_dose", "num_dose_mean"),                               # Medication dose
  "heart.csv"     = c("fac_transplant", "fac_transplant_mode"),                   # Transplant status
  "oldmort.csv"   = c("fac_civ"),                                                 # Civilian status
  # 11-06: datasets added in ../tmlr_dyn_new_datasets_11_06 (one longitudinal marker + its running mean, as aids / epileptic)
  "FRTCS.csv"      = c("num_sbp", "num_sbp_mean"),                                # Systolic blood pressure
  "heartvalve.csv" = c("num_lvmi", "num_lvmi_mean")                               # Left ventricular mass index (num_grad: 36% missing)
)

#' Get Hardcoded Biomarkers for Dataset
#' 
#' Retrieves the list of biomarker variable names that should be used
#' for joint modeling with the specified dataset. Only returns biomarkers
#' that actually exist in the dataset columns.
#' 
#' @param dataset_name String name of the dataset file (e.g., "aids.csv")
#' @param df Data frame containing the dataset
#' @return Character vector of biomarker column names present in the data
pick_hardcoded_biomarker <- function(dataset_name, df) {
  m <- biomarker_map[[dataset_name]]
  if (is.null(m) || length(m) == 0) return(character(0))
  m[m %in% names(df)]
}

# =============================================================================
# Data Loading and Splitting Functions
# =============================================================================

#' Load Patient ID Splits for Dataset
#' 
#' Loads pre-defined train/validation/test patient ID splits for a dataset.
#' Expects split files to be in the 'split' directory with standardized naming.
#' 
#' @param dataset_name String name of the dataset file (e.g., "aids.csv")  
#' @param split_dir Directory containing the split files (default: "split")
#' @return List with train, val, and test vectors of patient IDs
#' @throws Error if any split files are missing
load_pid_split <- function(dataset_name, split_dir = "split") {
  base <- sub("\\.csv$", "", dataset_name)

  train_file <- file.path(split_dir, paste0(base, "_train_pids.csv"))
  val_file   <- file.path(split_dir, paste0(base, "_val_pids.csv"))
  test_file  <- file.path(split_dir, paste0(base, "_test_pids.csv"))

  if (!file.exists(train_file) ||
      !file.exists(val_file) ||
      !file.exists(test_file)) {
    stop("Missing split files for dataset: ", base)
  }

  list(
    train = readr::read_csv(train_file, show_col_types = FALSE)$pid,
    val   = readr::read_csv(val_file,   show_col_types = FALSE)$pid,
    test  = readr::read_csv(test_file,  show_col_types = FALSE)$pid
  )
}

build_patient_outcome <- function(df) {
  followup_col <- if ("time2" %in% names(df)) "time2" else "time"
  df %>%
    arrange(.data[[followup_col]]) %>%
    group_by(pid) %>%
    summarize(
      t_event = max(.data[[followup_col]], na.rm = TRUE),
      status_event = dplyr::last(event),
      .groups = "drop"
    )
}

# Preprocess categorical biomarker into numeric longitudinal outcomes:
# - numeric -> keep as-is
# - binary factor/character -> 0/1 column <bm>__bin (level2 vs level1)
# - multi-class -> K-1 dummies <bm>__<Level> (ref=first level)
preprocess_categorical_biomarker <- function(df, bm_col) {
  if (!(bm_col %in% names(df))) return(list(df = df, marker_cols = character(0)))

  x <- df[[bm_col]]

  if (is.numeric(x) || is.integer(x)) {
    return(list(df = df, marker_cols = bm_col))
  }

  f <- as.factor(x)
  lv <- levels(f)
  nlv <- length(lv)

  if (nlv < 2) return(list(df = df, marker_cols = character(0)))

  if (nlv == 2) {
    new_name <- paste0(bm_col, "__bin")
    df[[new_name]] <- as.numeric(f == lv[2])
    return(list(df = df, marker_cols = new_name))
  }

  other_levels <- lv[-1]
  marker_cols <- character(0)
  for (L in other_levels) {
    new_name <- paste0(bm_col, "__", make.names(L))
    df[[new_name]] <- as.numeric(f == L)
    marker_cols <- c(marker_cols, new_name)
  }

  list(df = df, marker_cols = marker_cols)
}

# Cox covariates:
# - exclude pid, time, time2, event
# - exclude any *_mean or *_mode
# - exclude biomarker raw + derived marker cols
# - CSL special case: DROP num_prot_base (culprit)
pick_cox_covariates <- function(df, marker_cols, bm_raw = NULL, dataset_name = NULL) {
  drop_core <- c("pid", "time", "time2", "event")
  drop_mean_mode <- names(df)[grepl("_(mean|mode)$", names(df))]
  drop_all <- unique(c(drop_core, drop_mean_mode, marker_cols, bm_raw))

  covs <- setdiff(names(df), drop_all)

  if (!is.null(dataset_name) && dataset_name == "csl.csv") {
    if ("num_prot_base" %in% covs) {
      dbg("CSL special-case: dropping num_prot_base from Cox covariates (jm stability)")
      covs <- setdiff(covs, "num_prot_base")
    }
  }

  covs
}

# Robust percentile grid for any k.
k_to_percentile_grid <- function(k_step, tol = 1e-6) {
  if (!is.finite(k_step) || k_step <= 0 || k_step >= 100) {
    stop("k_step must be in (0, 100)")
  }

  K <- 100 / k_step
  K_int <- round(K)

  if (abs(K - K_int) <= tol && K_int >= 2) {
    percentiles <- (seq_len(K_int - 1) * (100 / K_int))
    return(list(percentiles = percentiles, n_bins = K_int - 1, K_int = K_int))
  }

  # same as np.arange(k, 100, k) in the Python scripts (km_landmarks.py): for k = 16.66666 the last percentile is
  # 99.99996, so n_bins = 6 as in every other model (the earlier "to = 100 - k_step" gave 5 and one landmark fewer)
  percentiles <- seq(from = k_step, to = 100, by = k_step)
  percentiles <- percentiles[percentiles > 0 & percentiles < 100]
  list(percentiles = percentiles, n_bins = length(percentiles), K_int = NA_integer_)
}

# Robust tau construction using quantile or KM-quantile binning.
safe_unique_taus <- function(times_all, events_all, k_step, binning = "km_quantile") {
  if (length(times_all) == 0 || length(events_all) == 0) return(numeric(0))
  if (length(times_all) != length(events_all)) return(numeric(0))

  times_all <- as.numeric(times_all)
  events_all <- as.integer(events_all)

  valid <- is.finite(times_all) & !is.na(events_all)
  times_all <- times_all[valid]
  events_all <- events_all[valid]

  event_times <- times_all[events_all == 1]
  if (length(event_times) == 0) return(numeric(0))

  grid <- k_to_percentile_grid(k_step)
  percentiles <- grid$percentiles
  n_bins <- grid$n_bins
  if (length(percentiles) == 0 || n_bins <= 0) return(numeric(0))

  if (binning == "quantile") {
    taus <- as.numeric(stats::quantile(
      event_times,
      probs = percentiles / 100,
      names = FALSE,
      na.rm = TRUE,
      type = 7
    ))
    return(sort(unique(taus)))
  }

  sf <- tryCatch(
    survival::survfit(survival::Surv(times_all, events_all) ~ 1),
    error = function(e) NULL
  )

  if (is.null(sf) || length(sf$time) == 0 || length(sf$surv) == 0) {
    taus <- as.numeric(stats::quantile(
      event_times,
      probs = percentiles / 100,
      names = FALSE,
      na.rm = TRUE,
      type = 7
    ))
    return(sort(unique(taus)))
  }

  km_time <- as.numeric(sf$time)
  km_surv <- as.numeric(sf$surv)

  if (length(km_time) == 0 || length(km_surv) == 0) {
    taus <- as.numeric(stats::quantile(
      event_times,
      probs = percentiles / 100,
      names = FALSE,
      na.rm = TRUE,
      type = 7
    ))
    return(sort(unique(taus)))
  }

  surv_levels <- seq(from = km_surv[1], to = km_surv[length(km_surv)], length.out = n_bins + 2)[2:(n_bins + 1)]

  taus <- numeric(0)
  for (s in surv_levels) {
    # KMBIN (11-01): the first KM time at which the curve is at or below the level, as the Python scripts
    # (np.searchsorted(-km_surv, -s)); the original findInterval(..., all.inside = TRUE) took one jump earlier
    idx <- which(km_surv <= s)[1]
    if (!is.na(idx) && idx <= length(km_time)) {
      taus <- c(taus, km_time[idx])
    }
  }

  if (length(taus) == 0) {
    taus <- as.numeric(stats::quantile(
      event_times,
      probs = percentiles / 100,
      names = FALSE,
      na.rm = TRUE,
      type = 7
    ))
  }

  sort(unique(taus))
}

make_newdata_at_tau <- function(df_long, pid_vec, tau) {
  df_long %>%
    filter(pid %in% pid_vec, time <= tau, time <= time2) %>%
    arrange(pid, time) %>%
    mutate(surv_time = tau, event = 0)
}

extract_risk_for_future <- function(pr, future_taus, tol = 1e-8) {
  if (!is.list(pr) || is.null(pr$pred) || is.null(pr$times)) return(NULL)

  times <- as.numeric(pr$times)
  preds <- as.numeric(pr$pred)

  out <- rep(NA_real_, length(future_taus))
  for (j in seq_along(future_taus)) {
    ft <- future_taus[j]
    idx <- which.min(abs(times - ft))
    if (length(idx) == 1 && abs(times[idx] - ft) <= tol) {
      out[j] <- preds[idx]
    } else if (length(idx) == 1 && abs(times[idx] - ft) <= 1e-4) {
      out[j] <- preds[idx]
    } else {
      out[j] <- NA_real_
    }
  }

  if (all(is.na(out))) return(NULL)
  out
}

fit_lme_list <- function(df_train_long, markers) {
  lme_list <- list()
  for (m in markers) {
    non_na <- sum(!is.na(df_train_long[[m]]))
    if (non_na < 10) {
      dbg("Skipping marker", m, "- too few non-NA values:", non_na)
      next
    }
    v <- stats::var(df_train_long[[m]], na.rm = TRUE)
    if (is.na(v) || v < 1e-12) {
      dbg("Skipping marker", m, "- near-zero variance:", v)
      next
    }

    fm <- tryCatch(
      lme(
        fixed = as.formula(paste0(m, " ~ time")),
        random = ~ time | pid,
        data = df_train_long,
        control = lmeControl(opt = "optim")
      ),
      error = function(e) {
        dbg("lme failed for marker", m, ":", e$message)
        return(NULL)
      }
    )
    if (!is.null(fm)) lme_list[[m]] <- fm
  }
  lme_list
}

# Remove unusable Cox covariates in TRAIN (all NA / single-level).
# Also align factor levels of val/test to train (unseen -> NA).
prepare_cox_covariates <- function(train_long, val_long, test_long, covars) {
  keep <- character(0)

  for (cc in covars) {
    x <- train_long[[cc]]
    x <- x[!is.na(x)]
    if (length(x) == 0) {
      dbg("Dropping Cox covariate", cc, "- all NA in train")
      next
    }
    if (length(unique(x)) < 2) {
      dbg("Dropping Cox covariate", cc, "- only one level/value in train:", unique(x)[1])
      next
    }
    keep <- c(keep, cc)
  }

  align_one <- function(train_df, other_df, cols) {
    for (cc in cols) {
      tr <- train_df[[cc]]
      if (is.numeric(tr) || is.integer(tr)) {
        train_df[[cc]] <- as.numeric(tr)
        other_df[[cc]] <- as.numeric(other_df[[cc]])
      } else {
        tr_chr <- as.character(tr)
        levels_tr <- sort(unique(tr_chr[!is.na(tr_chr)]))
        train_df[[cc]] <- factor(tr_chr, levels = levels_tr)

        oth_chr <- as.character(other_df[[cc]])
        oth_chr[!is.na(oth_chr) & !(oth_chr %in% levels_tr)] <- NA
        other_df[[cc]] <- factor(oth_chr, levels = levels_tr)
      }
    }
    list(train = train_df, other = other_df)
  }

  if (length(keep) > 0) {
    tmp <- align_one(train_long, val_long, keep)
    train_long <- tmp$train
    val_long   <- tmp$other

    tmp <- align_one(train_long, test_long, keep)
    train_long <- tmp$train
    test_long  <- tmp$other
  }

  list(keep = keep, train_long = train_long, val_long = val_long, test_long = test_long)
}

# One-hot encode baseline Cox covariates (train-driven levels), drop originals for non-numeric.
one_hot_encode_cox_covariates <- function(train_long, val_long, test_long, covars) {
  encoded_keep <- character(0)

  for (cc in covars) {
    tr <- train_long[[cc]]

    if (is.numeric(tr) || is.integer(tr)) {
      train_long[[cc]] <- as.numeric(tr)
      val_long[[cc]]   <- as.numeric(val_long[[cc]])
      test_long[[cc]]  <- as.numeric(test_long[[cc]])
      encoded_keep <- c(encoded_keep, cc)
      next
    }

    tr_f <- as.factor(tr)
    lv <- levels(tr_f)
    if (length(lv) < 2) {
      dbg("Dropping Cox covariate", cc, "- <2 levels after alignment")
      next
    }

    ref <- lv[1]
    other_lv <- lv[-1]
    new_cols <- character(0)

    for (L in other_lv) {
      new_name <- paste0(cc, "__", make.names(L))

      train_long[[new_name]] <- as.numeric(train_long[[cc]] == L)
      val_long[[new_name]]   <- as.numeric(val_long[[cc]] == L)
      test_long[[new_name]]  <- as.numeric(test_long[[cc]] == L)

      train_long[[new_name]][is.na(train_long[[cc]])] <- NA_real_
      val_long[[new_name]][is.na(val_long[[cc]])]     <- NA_real_
      test_long[[new_name]][is.na(test_long[[cc]])]   <- NA_real_

      new_cols <- c(new_cols, new_name)
    }

    train_long[[cc]] <- NULL
    val_long[[cc]]   <- NULL
    test_long[[cc]]  <- NULL

    if (length(new_cols) > 0) {
      encoded_keep <- c(encoded_keep, new_cols)
      dbg("One-hot encoded", cc, "(ref=", ref, ") ->", paste(new_cols, collapse = ", "))
    }
  }

  list(
    keep = encoded_keep,
    train_long = train_long,
    val_long = val_long,
    test_long = test_long
  )
}

assert_newdata_has_vars <- function(newdata, required_vars, pid, tau) {
  missing <- setdiff(required_vars, names(newdata))
  if (length(missing) > 0) {
    dbg("newdata missing vars for pid", pid, "tau", sprintf("%.4g", tau), ":", paste(missing, collapse = ", "))
    return(FALSE)
  }
  TRUE
}

# ----------------------------
# Start
# ----------------------------
dbg("Found", length(dataset_files), "dataset files.")
if (DEBUG && length(dataset_files) > 0) dbg("Example files:", paste(head(basename(dataset_files), 5), collapse = ", "))

# ----------------------------
# Main loop
# ----------------------------
for (file_path in dataset_files) {
  dataset_name <- basename(file_path)
  cat("\n============================================================\n")
  cat("Processing:", dataset_name, "\n")

  df <- tryCatch(
    read_csv(file_path, show_col_types = FALSE),
    error = function(e) {
      cat("Read failed:", e$message, "\n")
      return(NULL)
    }
  )
  if (is.null(df)) next

  dbg("Raw df shape:", nrow(df), "x", ncol(df))

  req <- c("pid", "event", "time")
  if (!all(req %in% names(df))) {
    cat("Missing required columns:", paste(setdiff(req, names(df)), collapse = ", "), "- skipping.\n")
    next
  }
  if (!("time2" %in% names(df))) {
    df$time2 <- df$time
    dbg("No time2 column; created time2 = time")
  }

  df <- df %>%
    filter(!is.na(pid), !is.na(time), !is.na(time2), !is.na(event)) %>%
    filter(time <= time2)

  dbg("After cleanup df shape:", nrow(df), "x", ncol(df))
  dbg("Unique pids:", length(unique(df$pid)))

  # ----------------------------
  # Choose + preprocess hardcoded biomarker(s)
  # ----------------------------
  bm_raw <- pick_hardcoded_biomarker(dataset_name, df)
  if (length(bm_raw) == 0) {
    cat("No hardcoded biomarker configured for this dataset (or column missing) - skipping.\n")
    next
  }

  markers <- character(0)
  for (bm in bm_raw) {
    tmp_bm <- preprocess_categorical_biomarker(df, bm)
    df <- tmp_bm$df
    markers <- c(markers, tmp_bm$marker_cols)
  }
  markers <- unique(markers)

  dbg("Hardcoded biomarker raw:", paste(bm_raw, collapse = ", "))
  dbg("Model biomarker columns:", paste(markers, collapse = ", "))

  if (length(markers) == 0) {
    cat("Biomarker unusable after preprocessing (all NA or single level) - skipping.\n")
    next
  }

  # ----------------------------
  # Cox covariates (baseline)
  # ----------------------------
  cox_cols <- pick_cox_covariates(df, marker_cols = markers, bm_raw = bm_raw, dataset_name = dataset_name)
  dbg("Candidate Cox covariates:", length(cox_cols))
  if (DEBUG && length(cox_cols) > 0) dbg("Example Cox covariates:", paste(head(cox_cols, 30), collapse = ", "))

  outcome <- build_patient_outcome(df)
  dbg("Outcome rows:", nrow(outcome))
  dbg("Outcome event count:", sum(outcome$status_event == 1, na.rm = TRUE))

  # ----------------------------
  # Load predefined PID splits
  # ----------------------------
  splits <- tryCatch(
    load_pid_split(dataset_name, split_dir = "split"),
    error = function(e) {
      cat("Split loading failed:", e$message, "\n")
      return(NULL)
    }
  )
  if (is.null(splits)) next

  train_pids <- splits$train
  val_pids   <- splits$val
  test_pids  <- splits$test

  dbg("Split sizes (from files): train", length(train_pids),
      "val", length(val_pids),
      "test", length(test_pids))

  df_train_long <- df %>% filter(pid %in% train_pids)
  df_val_long   <- df %>% filter(pid %in% val_pids)
  df_test_long  <- df %>% filter(pid %in% test_pids)

  # Prepare Cox covariates (filter unusable + align factor levels)
  prep <- prepare_cox_covariates(df_train_long, df_val_long, df_test_long, cox_cols)
  cox_keep      <- prep$keep
  df_train_long <- prep$train_long
  df_val_long   <- prep$val_long
  df_test_long  <- prep$test_long

  dbg("Keeping Cox covariates (pre one-hot):", length(cox_keep))
  if (DEBUG && length(cox_keep) > 0) dbg("Kept Cox covariates:", paste(head(cox_keep, 30), collapse = ", "))

  # One-hot encode baseline covariates for Cox/jm, train-driven
  if (length(cox_keep) > 0) {
    oh <- one_hot_encode_cox_covariates(df_train_long, df_val_long, df_test_long, cox_keep)
    cox_keep      <- oh$keep
    df_train_long <- oh$train_long
    df_val_long   <- oh$val_long
    df_test_long  <- oh$test_long
  }

  dbg("Keeping Cox covariates (post one-hot):", length(cox_keep))
  if (DEBUG && length(cox_keep) > 0) dbg("One-hot Cox covariates:", paste(head(cox_keep, 50), collapse = ", "))

  # Survival frames: extract covariates per pid from longitudinal frames
  extract_covars_per_pid <- function(df_long, keep_cols) {
    if (length(keep_cols) == 0) {
      return(df_long %>% distinct(pid))
    }
    df_long %>%
      arrange(pid, time) %>%
      group_by(pid) %>%
      summarize(across(all_of(keep_cols), ~{
        v <- .x
        v <- v[!is.na(v)]
        if (length(v) == 0) NA else v[1]
      }), .groups = "drop")
  }

  surv_train <- outcome %>%
    filter(pid %in% train_pids) %>%
    transmute(pid, surv_time = t_event, event = status_event) %>%
    left_join(extract_covars_per_pid(df_train_long, cox_keep), by = "pid")

  surv_val <- outcome %>%
    filter(pid %in% val_pids) %>%
    transmute(pid, surv_time = t_event, event = status_event) %>%
    left_join(extract_covars_per_pid(df_val_long, cox_keep), by = "pid")

  surv_test <- outcome %>%
    filter(pid %in% test_pids) %>%
    transmute(pid, surv_time = t_event, event = status_event) %>%
    left_join(extract_covars_per_pid(df_test_long, cox_keep), by = "pid")

  dbg("Surv train events:", sum(surv_train$event == 1, na.rm = TRUE))
  dbg("Surv val events:", sum(surv_val$event == 1, na.rm = TRUE))
  dbg("Surv test events:", sum(surv_test$event == 1, na.rm = TRUE))

  # Taus from TRAIN event times
  event_times_train <- surv_train$surv_time[surv_train$event == 1]
  if (length(event_times_train) < 5) {
    cat("Too few events in TRAIN (", length(event_times_train), ") - skipping.\n", sep = "")
    next
  }

  if (binning == "pooled") {
    # POOLED, as km_landmarks.pooled_taus: percentiles seq(k, <100, k) of the pooled covariate update
    # times (row start times after each training patient's first row) and the training event times (no censoring times)
    t0 <- ave(df_train_long$time, df_train_long$pid, FUN = min)
    meas <- df_train_long$time[df_train_long$time > t0]
    pooled <- c(meas, event_times_train)
    probs <- seq(k, 100, by = k); probs <- probs[round(probs, 2) < 100]
    taus <- sort(unique(as.numeric(stats::quantile(pooled, probs = probs / 100, names = FALSE, type = 7))))
  } else {
  taus <- safe_unique_taus(
    times_all = surv_train$surv_time,
    events_all = surv_train$event,
    k_step = k,
    binning = binning
  )
  }
  dbg("Computed taus length:", length(taus))
  dbg("Taus:", paste(sprintf("%.4g", taus), collapse = ", "))

  if (length(taus) < 2) {
    cat("Need at least 2 taus for requested output, but got ", length(taus), " - skipping.\n", sep = "")
    next
  }

  taus_eval <- taus
  n_tau <- length(taus_eval)
  dbg("Using taus_eval (n=", n_tau, "): ", paste(sprintf("%.4g", taus_eval), collapse = ", "), sep = "")

  # ----------------------------
  # Hyperparameter loop
  # ----------------------------
  for (alpha in alpha_grid) {
    hp_config <- paste0("{alpha=", alpha, "}")
    cat("\n--- HP:", hp_config, " ---\n")

    fm_surv <- NULL
    fit_jm <- NULL

    # 1) Fit Cox model (required for jm)
    if (length(cox_keep) > 0) {
      surv_fml <- as.formula(paste("Surv(surv_time, event) ~", paste(cox_keep, collapse = " + ")))
    } else {
      surv_fml <- Surv(surv_time, event) ~ 1
    }

    dbg("Fitting coxph with covariates:",
        ifelse(length(cox_keep) > 0, paste(cox_keep, collapse = ", "), "(none)"))
    fm_surv <- tryCatch(
      coxph(surv_fml, data = surv_train, x = TRUE),
      error = function(e) {
        cat("CoxPH fit failed:", e$message, "\n")
        return(NULL)
      }
    )
    if (is.null(fm_surv)) next

    # 2) Fit LME list (must have at least 1 to build jm)
    dbg("Fitting lme list (hardcoded biomarker columns)...")
    lme_list <- fit_lme_list(df_train_long, markers)
    dbg("Successfully fit", length(lme_list), "LME models.")
    if (length(lme_list) == 0) {
      cat("No LME models fit; cannot fit joint model for this dataset/HP. Skipping.\n")
      next
    }

    # 3) Fit joint model (jm) - NO FALLBACK
    dbg("Fitting joint model (jm)...")
    fit_jm <- tryCatch(
      jm(
        fm_surv,
        lme_list,
        time_var = "time",
        priors = list(alpha = list(dist = "normal", par = c(0, alpha)))
      ),
      error = function(e) {
        cat("JM fit failed:", e$message, "\n")
        return(NULL)
      }
    )
    if (is.null(fit_jm)) next
    dbg("JM fit OK.")

    needed_newdata_vars <- c("pid", "time", "surv_time", "event", cox_keep)

    # ----------------------------
    # Evaluate on VAL and TEST (JOINT ONLY)
    # ----------------------------
    for (eval_set_name in c("val", "test")) {
      eval_long <- if (eval_set_name == "val") df_val_long else df_test_long
      eval_surv <- if (eval_set_name == "val") surv_val else surv_test

      dbg("Eval set:", eval_set_name, "n subjects:", nrow(eval_surv), "model_type: joint")

      for (i_tau in 1:(n_tau - 1)) {
        tau <- taus_eval[i_tau]
        future_taus <- taus_eval[(i_tau + 1):n_tau]

        dbg(eval_set_name, "tau idx", i_tau, "tau", sprintf("%.4g", tau),
            "future_taus:", paste(sprintf("%.4g", future_taus), collapse = ", "))

        at_risk_pids <- eval_surv %>%
          filter(surv_time > tau) %>%
          pull(pid)

        dbg(eval_set_name, "tau", sprintf("%.4g", tau), "at-risk:", length(at_risk_pids))
        if (length(at_risk_pids) == 0) next

        nd <- make_newdata_at_tau(eval_long, at_risk_pids, tau)
        dbg("nd rows at tau:", nrow(nd), "unique pids in nd:", length(unique(nd$pid)))
        if (nrow(nd) == 0) next

        for (this_pid in unique(nd$pid)) {
          nd_one <- nd %>% filter(pid == this_pid)
          if (nrow(nd_one) == 0) next

          if (!assert_newdata_has_vars(nd_one, needed_newdata_vars, this_pid, tau)) next

          pr <- tryCatch(
            predict(fit_jm, newdata = nd_one, times = future_taus, process = "event"),
            error = function(e) {
              dbg("JM predict failed pid", this_pid, "tau", sprintf("%.4g", tau), ":", e$message)
              return(NULL)
            }
          )
          if (is.null(pr)) next

          risk_vec <- extract_risk_for_future(pr, future_taus)
          if (is.null(risk_vec) || all(is.na(risk_vec))) next

          # Risk(tau_i) = P(T <= tau_{i+1} | H(tau_i)) -> use risk_vec[1]
          if (is.na(risk_vec[1])) next
          risk_scalar <- risk_vec[1]

          all_rows[[length(all_rows) + 1]] <- data.frame(
            pid = this_pid,
            tau = tau,
            t = NA_real_,
            risk = risk_scalar,
            surv_prob = NA_real_,
            dataset = dataset_name,
            hyperparameter = hp_config,
            eval_set = eval_set_name,
            k = k
          )

          # Survival rows: S(future_tau | tau) = 1 - P(event by future_tau | tau)
          surv_vec <- 1 - risk_vec
          for (j in seq_along(future_taus)) {
            all_rows[[length(all_rows) + 1]] <- data.frame(
              pid = this_pid,
              tau = tau,
              t = future_taus[j],
              risk = NA_real_,
              surv_prob = surv_vec[j],
              dataset = dataset_name,
              hyperparameter = hp_config,
              eval_set = eval_set_name,
              k = k
            )
          }
        }
      }
    }
  }
}

dbg("Total rows collected:", length(all_rows))

# ----------------------------
# Write output (APPEND if file exists)
# ----------------------------
if (length(all_rows) > 0) {
  out_new <- bind_rows(all_rows) %>%
    arrange(k, hyperparameter, eval_set, dataset, pid, tau, t) %>%
    mutate(
      tau = round(tau, 6),
      t = ifelse(is.na(t), NA_real_, round(t, 6)),
      risk = ifelse(is.na(risk), NA_real_, round(risk, 10)),
      surv_prob = ifelse(is.na(surv_prob), NA_real_, round(surv_prob, 10))
    )

  if (file.exists(out_path)) {
    out_old <- readr::read_csv(out_path, show_col_types = FALSE)
    out <- bind_rows(out_old, out_new)
  } else {
    out <- out_new
  }

  write_csv(out, out_path, na = "")
  cat("\nSaved (appended if existed):", out_path, "\n")
} else {
  cat("\nNo rows generated; nothing saved.\n")
}
