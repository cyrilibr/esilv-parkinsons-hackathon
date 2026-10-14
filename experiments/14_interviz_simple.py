"""Experiment 14 — Inter-visit delta features on top of Exp-12 pipeline.

Hypothesis: the change in ON/OFF between consecutive visits (delta_on, delta_off)
and the time since the previous visit carry progression-velocity signal not
captured by global patient aggregates.

New features (no leakage, derived from inputs only):
  iv_delta_on, iv_delta_off        — score change vs previous visit
  iv_delta_on_corr                 — corrected-ON change vs previous visit
  iv_time_since_prev               — age gap to previous visit (years)
  iv_on_velocity, iv_off_velocity  — delta / time
  iv_visit_rank, iv_visits_remaining

Baseline (Exp-12): mean 5-fold RMSE = 4.1310, std = 0.0652
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold

import skore
from skore import EstimatorReport, Project, login

PROJECT_ROOT = Path(__file__).resolve().parents[1]
t_start = time.time()


# ---------------------------------------------------------------------------
# Bateman curve
# ---------------------------------------------------------------------------

def bateman(t, A, ke, ka):
    """Drug effect fraction: A*(exp(-ke*t) - exp(-ka*t)), clipped to [0,1)."""
    with np.errstate(over="ignore", invalid="ignore"):
        val = A * (np.exp(-ke * t) - np.exp(-ka * t))
    return np.clip(val, 0.0, 0.95)


def fit_bateman_on(df_train):
    """Fit Bateman curve on ON scores from train fold.

    Uses visits where on, time_since_intake_on, and target are all present.
    Returns (A, ke, ka) or None if fitting fails.
    """
    sub = df_train.dropna(subset=["on", "time_since_intake_on", "target"]).copy()
    sub = sub[sub["target"].abs() > 1e-6]
    if len(sub) < 50:
        return None
    t = sub["time_since_intake_on"].values.astype(float)
    # observed effect fraction: 1 - on/target  (positive when drug helps)
    frac = (1.0 - sub["on"].values.astype(float) / sub["target"].values.astype(float))
    frac = np.clip(frac, 0.0, 0.95)
    try:
        popt, _ = curve_fit(
            bateman, t, frac,
            p0=[0.6, 0.5, 2.0],
            bounds=([0.0, 0.01, 0.1], [1.0, 5.0, 20.0]),
            maxfev=5000,
        )
        return popt  # (A, ke, ka)
    except Exception:
        return None


def fit_bateman_off(df_train):
    """Fit decay curve on OFF scores from train fold.

    For OFF measurements, the patient took the drug long ago; effect is mostly
    in elimination phase. We model the residual effect as a simple exponential:
        off_obs ≈ target * (1 - B*exp(-ke_off * t_off))
    Returns (B, ke_off) or None.
    """
    sub = df_train.dropna(subset=["off", "time_since_intake_off", "target"]).copy()
    sub = sub[sub["target"].abs() > 1e-6]
    if len(sub) < 50:
        return None
    t = sub["time_since_intake_off"].values.astype(float)
    frac = (1.0 - sub["off"].values.astype(float) / sub["target"].values.astype(float))
    frac = np.clip(frac, 0.0, 0.95)

    def decay(t, B, ke):
        return np.clip(B * np.exp(-ke * t), 0.0, 0.95)

    try:
        popt, _ = curve_fit(
            decay, t, frac,
            p0=[0.3, 0.1],
            bounds=([0.0, 0.001], [1.0, 2.0]),
            maxfev=5000,
        )
        return popt  # (B, ke_off)
    except Exception:
        return None


def apply_on_correction(on_vals, t_on_vals, params_on):
    """Return on_corr = on / (1 - effect(t_on)). NaN where inputs are NaN."""
    if params_on is None:
        return np.full(len(on_vals), np.nan)
    A, ke, ka = params_on
    corr = np.full(len(on_vals), np.nan)
    valid = ~(np.isnan(on_vals) | np.isnan(t_on_vals))
    if valid.sum() == 0:
        return corr
    eff = bateman(t_on_vals[valid], A, ke, ka)
    denom = 1.0 - eff
    denom = np.where(denom < 0.05, 0.05, denom)  # avoid div/0
    corr[valid] = on_vals[valid] / denom
    return corr


def apply_off_correction(off_vals, t_off_vals, params_off):
    """Return off_corr = off / (1 - B*exp(-ke*t_off))."""
    if params_off is None:
        return np.full(len(off_vals), np.nan)
    B, ke = params_off

    def decay(t):
        return np.clip(B * np.exp(-ke * t), 0.0, 0.95)

    corr = np.full(len(off_vals), np.nan)
    valid = ~(np.isnan(off_vals) | np.isnan(t_off_vals))
    if valid.sum() == 0:
        return corr
    eff = decay(t_off_vals[valid])
    denom = 1.0 - eff
    denom = np.where(denom < 0.05, 0.05, denom)
    corr[valid] = off_vals[valid] / denom
    return corr


# ---------------------------------------------------------------------------
# Helpers (same as Exp 11)
# ---------------------------------------------------------------------------

def _patient_slope(ages, vals):
    mask = ~np.isnan(vals)
    a, v = ages[mask], vals[mask]
    if len(a) < 2 or np.std(a) < 1e-9:
        return np.nan, np.nan
    return tuple(float(x) for x in np.polyfit(a, v, 1))


def _patient_quad(ages, vals):
    mask = ~np.isnan(vals)
    a, v = ages[mask], vals[mask]
    if len(a) < 3 or np.std(a) < 1e-9:
        return np.nan, np.nan, np.nan
    return tuple(float(x) for x in np.polyfit(a, v, 2))


def _first_last(ages, vals):
    mask = ~np.isnan(vals)
    if mask.sum() == 0:
        return np.nan, np.nan
    order = np.argsort(ages[mask])
    v = vals[mask][order]
    return float(v[0]), float(v[-1])


# ---------------------------------------------------------------------------
# Feature engineering (Exp-11 pipeline, condensed)
# ---------------------------------------------------------------------------

def add_static_features(df):
    df = df.copy()
    df["years_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    df["off_minus_on"]  = df["off"] - df["on"]
    df["on_missing"]    = df["on"].isna().astype(int)
    df["off_missing"]   = df["off"].isna().astype(int)
    df["ledd_missing"]  = df["ledd"].isna().astype(int)
    return df


def build_all_features(df_raw, params_on=None, params_off=None):
    """Apply full feature pipeline to a raw dataframe (with patient_id).

    params_on  : Bateman (A, ke, ka) fitted on train fold, or None
    params_off : decay  (B, ke)      fitted on train fold, or None
    Returns feature DataFrame (patient_id dropped).
    """
    df = df_raw.copy().reset_index(drop=True)

    # --- 1. Bateman corrections ---
    on_vals  = df["on"].values.astype(float)
    off_vals = df["off"].values.astype(float)
    t_on     = df["time_since_intake_on"].values.astype(float)
    t_off    = df["time_since_intake_off"].values.astype(float)

    on_corr  = apply_on_correction(on_vals,  t_on,  params_on)
    off_corr = apply_off_correction(off_vals, t_off, params_off)

    df["on_corr"]  = on_corr
    df["off_corr"] = off_corr
    # diff between corrected and raw
    df["on_corr_delta"]  = on_corr  - on_vals
    df["off_corr_delta"] = off_corr - off_vals
    # corrected on-off gap
    df["on_off_corr_gap"] = on_corr - off_corr

    # --- 2. Quadratic aggregates ---
    quad_records = {}
    for pid, grp in df.groupby("patient_id"):
        ages = grp["age"].values.astype(float)
        on   = grp["on"].values.astype(float)
        off  = grp["off"].values.astype(float)
        on_a, on_b, on_c   = _patient_quad(ages, on)
        off_a, off_b, off_c = _patient_quad(ages, off)
        quad_records[pid] = {
            "pt_on_quad": on_a,  "pt_on_qb": on_b,  "pt_on_qc": on_c,
            "pt_off_quad": off_a, "pt_off_qb": off_b, "pt_off_qc": off_c,
        }
    quad_stats = pd.DataFrame(quad_records).T
    quad_fb = quad_stats.mean(numeric_only=True)

    # --- 3. Longitudinal aggregates ---
    long_records = {}
    for pid, grp in df.groupby("patient_id"):
        grp = grp.sort_values("age")
        ages = grp["age"].values.astype(float)
        on   = grp["on"].values.astype(float)
        off  = grp["off"].values.astype(float)
        ledd = grp["ledd"].values.astype(float)
        on_corr_p  = grp["on_corr"].values.astype(float)
        off_corr_p = grp["off_corr"].values.astype(float)
        n = len(grp)
        age_min, age_max = float(ages.min()), float(ages.max())
        on_v = on[~np.isnan(on)]; off_v = off[~np.isnan(off)]; ledd_v = ledd[~np.isnan(ledd)]
        on_s, on_i   = _patient_slope(ages, on)
        off_s, off_i = _patient_slope(ages, off)
        on_f, on_l   = _first_last(ages, on)
        off_f, off_l = _first_last(ages, off)
        diff_v = (on - off); diff_v = diff_v[~np.isnan(diff_v)]
        # corrected score aggregates
        on_corr_v  = on_corr_p[~np.isnan(on_corr_p)]
        off_corr_v = off_corr_p[~np.isnan(off_corr_p)]
        long_records[pid] = {
            "pt_on_mean": np.nanmean(on) if len(on_v) else np.nan,
            "pt_on_median": np.nanmedian(on) if len(on_v) else np.nan,
            "pt_on_std": np.nanstd(on) if len(on_v) >= 2 else np.nan,
            "pt_on_min": on_v.min() if len(on_v) else np.nan,
            "pt_on_max": on_v.max() if len(on_v) else np.nan,
            "pt_on_first": on_f, "pt_on_last": on_l,
            "pt_on_count": float(len(on_v)),
            "pt_on_miss_frac": float(np.isnan(on).mean()),
            "pt_on_slope": on_s, "pt_on_intercept": on_i,
            "pt_off_mean": np.nanmean(off) if len(off_v) else np.nan,
            "pt_off_median": np.nanmedian(off) if len(off_v) else np.nan,
            "pt_off_std": np.nanstd(off) if len(off_v) >= 2 else np.nan,
            "pt_off_min": off_v.min() if len(off_v) else np.nan,
            "pt_off_max": off_v.max() if len(off_v) else np.nan,
            "pt_off_first": off_f, "pt_off_last": off_l,
            "pt_off_count": float(len(off_v)),
            "pt_off_miss_frac": float(np.isnan(off).mean()),
            "pt_off_slope": off_s, "pt_off_intercept": off_i,
            "pt_ledd_mean": np.nanmean(ledd) if len(ledd_v) else np.nan,
            "pt_ledd_std": np.nanstd(ledd) if len(ledd_v) >= 2 else np.nan,
            "pt_ledd_min": ledd_v.min() if len(ledd_v) else np.nan,
            "pt_ledd_max": ledd_v.max() if len(ledd_v) else np.nan,
            "pt_age_min": age_min, "pt_age_max": age_max,
            "pt_age_span": age_max - age_min, "pt_visit_total": float(n),
            "pt_mean_on_off_gap": float(diff_v.mean()) if len(diff_v) else np.nan,
            # corrected score patient stats
            "pt_on_corr_mean":  np.nanmean(on_corr_p)  if len(on_corr_v)  else np.nan,
            "pt_off_corr_mean": np.nanmean(off_corr_p) if len(off_corr_v) else np.nan,
            "pt_on_corr_std":   np.nanstd(on_corr_p)   if len(on_corr_v) >= 2 else np.nan,
            "pt_off_corr_std":  np.nanstd(off_corr_p)  if len(off_corr_v) >= 2 else np.nan,
        }
    long_stats = pd.DataFrame(long_records).T
    long_fb = long_stats.mean(numeric_only=True)

    # --- 4. Assemble ---
    # Quad features
    qj = quad_stats.reindex(df["patient_id"]); qj.index = df.index
    for col in qj.columns:
        qj[col] = qj[col].fillna(quad_fb.get(col, np.nan))
    age = df["age"].values.astype(float)
    on_a = qj["pt_on_quad"].values; on_b = qj["pt_on_qb"].values; on_c = qj["pt_on_qc"].values
    off_a = qj["pt_off_quad"].values; off_b = qj["pt_off_qb"].values; off_c = qj["pt_off_qc"].values
    on_pred  = on_a  * age**2 + on_b  * age + on_c
    off_pred = off_a * age**2 + off_b * age + off_c
    df["pt_on_quad"]       = on_a
    df["pt_off_quad"]      = off_a
    df["pt_on_quad_pred"]  = on_pred
    df["pt_off_quad_pred"] = off_pred
    df["pt_on_quad_res"]   = on_vals - on_pred
    df["pt_off_quad_res"]  = off_vals - off_pred
    df = df.drop(columns=["pt_on_qb","pt_on_qc","pt_off_qb","pt_off_qc"], errors="ignore")

    # Long features
    lj = long_stats.reindex(df["patient_id"]); lj.index = df.index
    for col in lj.columns:
        lj[col] = lj[col].fillna(long_fb.get(col, np.nan))
    age_min  = lj["pt_age_min"].values; age_max = lj["pt_age_max"].values
    age_span = age_max - age_min
    age_pct  = np.where(age_span > 0, (age - age_min) / age_span, 0.5)
    lj["pt_age_percentile"] = age_pct

    # Medication + linear trend residuals
    on_s = lj["pt_on_slope"].values; on_i = lj["pt_on_intercept"].values
    off_s = lj["pt_off_slope"].values; off_i = lj["pt_off_intercept"].values
    pred_on  = on_s  * age + on_i
    pred_off = off_s * age + off_i
    df["on_dev_from_trend"]  = on_vals  - pred_on
    df["off_dev_from_trend"] = off_vals - pred_off
    df["pred_on_at_age"]  = pred_on
    df["pred_off_at_age"] = pred_off
    df["on_off_gap"]      = on_vals - off_vals
    ledd = df.get("ledd", pd.Series(np.nan, index=df.index)).values.astype(float)
    df["ledd_x_on_off_gap"] = ledd * (on_vals - off_vals)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["on_off_ratio"] = np.where(np.abs(off_vals) > 1e-9, on_vals / off_vals, np.nan)
    df["intake_on_x_on"]   = t_on  * on_vals
    df["intake_off_x_off"] = t_off * off_vals

    # --- 5. Inter-visit delta features ---
    # Sort by patient + age, compute row-to-row deltas within each patient.
    df_iv = df[["patient_id", "age", "on", "off", "on_corr"]].copy()
    df_iv = df_iv.sort_values(["patient_id", "age"])
    df_iv["iv_delta_on"]       = df_iv.groupby("patient_id")["on"].diff()
    df_iv["iv_delta_off"]      = df_iv.groupby("patient_id")["off"].diff()
    df_iv["iv_delta_on_corr"]  = df_iv.groupby("patient_id")["on_corr"].diff()
    df_iv["iv_time_since_prev"] = df_iv.groupby("patient_id")["age"].diff()
    with np.errstate(divide="ignore", invalid="ignore"):
        df_iv["iv_on_velocity"]  = df_iv["iv_delta_on"]  / df_iv["iv_time_since_prev"].replace(0, np.nan)
        df_iv["iv_off_velocity"] = df_iv["iv_delta_off"] / df_iv["iv_time_since_prev"].replace(0, np.nan)
    # Visit rank (0-based) and remaining visits
    df_iv["iv_visit_rank"]       = df_iv.groupby("patient_id").cumcount()
    df_iv["iv_visits_total"]     = df_iv.groupby("patient_id")["age"].transform("count")
    df_iv["iv_visits_remaining"] = df_iv["iv_visits_total"] - df_iv["iv_visit_rank"] - 1
    iv_cols = ["iv_delta_on", "iv_delta_off", "iv_delta_on_corr",
               "iv_time_since_prev", "iv_on_velocity", "iv_off_velocity",
               "iv_visit_rank", "iv_visits_remaining"]
    # Reindex back to original df order
    df_iv_sorted = df_iv[iv_cols].reindex(df.index)
    for col in iv_cols:
        df[col] = df_iv_sorted[col].values

    result = pd.concat([df.drop(columns=["patient_id"]), lj], axis=1)
    return result


# ---------------------------------------------------------------------------
# LGBM wrapper (same config as Exp 11)
# ---------------------------------------------------------------------------

class LGBMWrapper(BaseEstimator):
    CAT_COLS = ["cohort", "gene", "rater_id"]

    def __init__(self):
        pass

    @staticmethod
    def _cast_cats(X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col in LGBMWrapper.CAT_COLS:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    def fit(self, X, y):
        self.model_ = LGBMRegressor(
            n_estimators=2000, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=3.0,
            random_state=42, verbose=-1)
        self.model_.fit(self._cast_cats(X), y)
        return self

    def predict(self, X):
        return self.model_.predict(self._cast_cats(X))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
print("Loading data...")
X_train_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train_raw = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test_raw  = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

visits = X_train_raw.merge(y_train_raw, on="Index", validate="one_to_one")
visits = add_static_features(visits)
X_test_e = add_static_features(X_test_raw)

NUMERIC_COLS = ["sexM","age_at_diagnosis","age","years_since_diagnosis","ledd",
                "time_since_intake_on","time_since_intake_off","on","off",
                "off_minus_on","on_missing","off_missing","ledd_missing"]
CAT_COLS = ["cohort","gene","rater_id"]
ALL_FEATURE_COLS = ["patient_id"] + NUMERIC_COLS + CAT_COLS

X = visits[ALL_FEATURE_COLS]
y = visits["target"]
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

# ---------------------------------------------------------------------------
# 5-fold CV with OOF Bateman fitting
# ---------------------------------------------------------------------------
print("\nRunning 5-fold CV with OOF Bateman correction...")
fold_rmses = []
for i, (tr_idx, va_idx) in enumerate(cv_splits):
    t0 = time.time()
    X_tr = visits.iloc[tr_idx][ALL_FEATURE_COLS + ["target"]]
    X_va = visits.iloc[va_idx][ALL_FEATURE_COLS]
    y_tr = y.iloc[tr_idx]
    y_va = y.iloc[va_idx]

    # Fit Bateman on train fold
    params_on  = fit_bateman_on(X_tr)
    params_off = fit_bateman_off(X_tr)

    # Build features
    Xf_tr = build_all_features(X_tr[ALL_FEATURE_COLS], params_on, params_off)
    Xf_va = build_all_features(X_va,                   params_on, params_off)

    m = LGBMWrapper().fit(Xf_tr, y_tr)
    preds = m.predict(Xf_va)
    rmse = float(np.sqrt(np.mean((y_va.values - preds)**2)))
    fold_rmses.append(rmse)
    print(f"  Fold {i}: RMSE={rmse:.4f}  params_on={[round(float(x),3) for x in params_on] if params_on is not None else None}  ({time.time()-t0:.0f}s)")

mean_rmse = float(np.mean(fold_rmses))
std_rmse  = float(np.std(fold_rmses, ddof=1))
delta     = mean_rmse - 4.1310
print(f"\nMean RMSE : {mean_rmse:.6f}")
print(f"Std  RMSE : {std_rmse:.6f}")
print(f"vs Exp-11 : {delta:+.4f}  ({'IMPROVEMENT' if delta < 0 else 'no improvement'})")

# ---------------------------------------------------------------------------
# EstimatorReport (fold 0) + Hub
# ---------------------------------------------------------------------------
tr0, va0 = cv_splits[0]
X_tr0 = visits.iloc[tr0][ALL_FEATURE_COLS + ["target"]]
X_va0 = visits.iloc[va0][ALL_FEATURE_COLS]
y_tr0 = y.iloc[tr0]; y_va0 = y.iloc[va0]

params_on0  = fit_bateman_on(X_tr0)
params_off0 = fit_bateman_off(X_tr0)
Xf_tr0 = build_all_features(X_tr0[ALL_FEATURE_COLS], params_on0, params_off0)
Xf_va0 = build_all_features(X_va0,                   params_on0, params_off0)

m0 = LGBMWrapper().fit(Xf_tr0, y_tr0)

print("\nBuilding EstimatorReport (fold 0 pre-transformed)...")
est_report = EstimatorReport(
    LGBMWrapper(),
    X_train=Xf_tr0, y_train=y_tr0,
    X_test=Xf_va0,  y_test=y_va0,
)
print(f"Fold-0 RMSE: {est_report.metrics.rmse()}")

cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("14_interviz_simple", est_report)
print("Published: key=12_bateman")

# ---------------------------------------------------------------------------
# Final fit + submission
# ---------------------------------------------------------------------------
print("\nFitting Bateman on full train + final model...")
params_on_final  = fit_bateman_on(visits[ALL_FEATURE_COLS + ["target"]])
params_off_final = fit_bateman_off(visits[ALL_FEATURE_COLS + ["target"]])
print(f"Final Bateman ON  params: {[round(float(x),4) for x in params_on_final] if params_on_final is not None else None}")
print(f"Final Bateman OFF params: {[round(float(x),4) for x in params_off_final] if params_off_final is not None else None}")

Xf_all   = build_all_features(visits[ALL_FEATURE_COLS], params_on_final, params_off_final)
final    = LGBMWrapper().fit(Xf_all, y)

sample = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")
test_ordered = X_test_e.set_index("Index").loc[sample["Index"]]
Xf_test  = build_all_features(test_ordered[ALL_FEATURE_COLS], params_on_final, params_off_final)
predictions = final.predict(Xf_test)

submission = sample[["Index"]].copy()
submission["target"] = predictions
assert list(submission.columns) == ["Index", "target"]
assert len(submission) == len(sample)
assert submission["Index"].equals(sample["Index"])
assert submission["Index"].is_unique
assert np.isfinite(submission["target"].to_numpy()).all()
print("Submission validation passed.")

out = PROJECT_ROOT / "submissions" / "14_interviz_simple.csv"
out.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(out, index=False)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
elapsed = time.time() - t_start
print("\n===================================")
print("EXPERIMENT 14 — INTER-VISIT FEATURES")
print("===================================")
print(f"Per-fold RMSE : {[round(v, 4) for v in fold_rmses]}")
print(f"Mean RMSE     : {mean_rmse:.6f}")
print(f"Std  RMSE     : {std_rmse:.6f}")
print(f"vs Exp-12     : {delta:+.4f}  ({'IMPROVEMENT' if delta < 0 else 'no improvement'})")
print(f"Total runtime : {elapsed:.0f}s")
print(f"\nSubmission : {out}")
print("Hub key    : 12_bateman")
print("===================================")
