"""Experiment 12 finalizer — final fit + submission only.

CV results already obtained:
  Fold 0: 4.1698  Fold 1: 4.2148  Fold 2: 4.0774  Fold 3: 4.0565  Fold 4: 4.1366
  Mean: 4.131021  Std: 0.065159  vs Exp-11: -0.0334  (IMPROVEMENT)
  Hub: https://skore.probabl.ai/philippeechebest/bobathon-esilv/estimators/70011
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
t0 = time.time()


# --- Bateman helpers (identical to 12_bateman.py) ---

def bateman(t, A, ke, ka):
    with np.errstate(over="ignore", invalid="ignore"):
        val = A * (np.exp(-ke * t) - np.exp(-ka * t))
    return np.clip(val, 0.0, 0.95)


def fit_bateman_on(df_train):
    sub = df_train.dropna(subset=["on", "time_since_intake_on", "target"]).copy()
    sub = sub[sub["target"].abs() > 1e-6]
    if len(sub) < 50:
        return None
    t    = sub["time_since_intake_on"].values.astype(float)
    frac = np.clip(1.0 - sub["on"].values.astype(float) / sub["target"].values.astype(float), 0.0, 0.95)
    try:
        popt, _ = curve_fit(bateman, t, frac, p0=[0.6, 0.5, 2.0],
                            bounds=([0, 0.01, 0.1], [1, 5, 20]), maxfev=5000)
        return popt
    except Exception:
        return None


def fit_bateman_off(df_train):
    sub = df_train.dropna(subset=["off", "time_since_intake_off", "target"]).copy()
    sub = sub[sub["target"].abs() > 1e-6]
    if len(sub) < 50:
        return None
    t    = sub["time_since_intake_off"].values.astype(float)
    frac = np.clip(1.0 - sub["off"].values.astype(float) / sub["target"].values.astype(float), 0.0, 0.95)
    def decay(t, B, ke):
        return np.clip(B * np.exp(-ke * t), 0.0, 0.95)
    try:
        popt, _ = curve_fit(decay, t, frac, p0=[0.3, 0.1],
                            bounds=([0, 0.001], [1, 2]), maxfev=5000)
        return popt
    except Exception:
        return None


def apply_on_correction(on_vals, t_on_vals, params_on):
    if params_on is None:
        return np.full(len(on_vals), np.nan)
    A, ke, ka = params_on
    corr  = np.full(len(on_vals), np.nan)
    valid = ~(np.isnan(on_vals) | np.isnan(t_on_vals))
    if valid.sum() == 0:
        return corr
    eff   = bateman(t_on_vals[valid], A, ke, ka)
    denom = np.where(1.0 - eff < 0.05, 0.05, 1.0 - eff)
    corr[valid] = on_vals[valid] / denom
    return corr


def apply_off_correction(off_vals, t_off_vals, params_off):
    if params_off is None:
        return np.full(len(off_vals), np.nan)
    B, ke = params_off
    corr  = np.full(len(off_vals), np.nan)
    valid = ~(np.isnan(off_vals) | np.isnan(t_off_vals))
    if valid.sum() == 0:
        return corr
    eff   = np.clip(B * np.exp(-ke * t_off_vals[valid]), 0.0, 0.95)
    denom = np.where(1.0 - eff < 0.05, 0.05, 1.0 - eff)
    corr[valid] = off_vals[valid] / denom
    return corr


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


def add_static_features(df):
    df = df.copy()
    df["years_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    df["off_minus_on"]  = df["off"] - df["on"]
    df["on_missing"]    = df["on"].isna().astype(int)
    df["off_missing"]   = df["off"].isna().astype(int)
    df["ledd_missing"]  = df["ledd"].isna().astype(int)
    return df


def build_all_features(df_raw, params_on=None, params_off=None):
    df = df_raw.copy().reset_index(drop=True)
    on_vals  = df["on"].values.astype(float)
    off_vals = df["off"].values.astype(float)
    t_on     = df["time_since_intake_on"].values.astype(float)
    t_off    = df["time_since_intake_off"].values.astype(float)
    on_corr  = apply_on_correction(on_vals, t_on, params_on)
    off_corr = apply_off_correction(off_vals, t_off, params_off)
    df["on_corr"]         = on_corr
    df["off_corr"]        = off_corr
    df["on_corr_delta"]   = on_corr  - on_vals
    df["off_corr_delta"]  = off_corr - off_vals
    df["on_off_corr_gap"] = on_corr  - off_corr

    quad_records = {}
    for pid, grp in df.groupby("patient_id"):
        ages = grp["age"].values.astype(float)
        on   = grp["on"].values.astype(float)
        off  = grp["off"].values.astype(float)
        on_a, on_b, on_c    = _patient_quad(ages, on)
        off_a, off_b, off_c = _patient_quad(ages, off)
        quad_records[pid] = {"pt_on_quad": on_a, "pt_on_qb": on_b, "pt_on_qc": on_c,
                             "pt_off_quad": off_a, "pt_off_qb": off_b, "pt_off_qc": off_c}
    quad_stats = pd.DataFrame(quad_records).T
    quad_fb    = quad_stats.mean(numeric_only=True)

    long_records = {}
    for pid, grp in df.groupby("patient_id"):
        grp = grp.sort_values("age")
        ages = grp["age"].values.astype(float)
        on   = grp["on"].values.astype(float)
        off  = grp["off"].values.astype(float)
        ledd = grp["ledd"].values.astype(float)
        onc  = grp["on_corr"].values.astype(float)
        ofc  = grp["off_corr"].values.astype(float)
        n = len(grp)
        age_min, age_max = float(ages.min()), float(ages.max())
        on_v = on[~np.isnan(on)]; off_v = off[~np.isnan(off)]; ledd_v = ledd[~np.isnan(ledd)]
        on_s, on_i   = _patient_slope(ages, on)
        off_s, off_i = _patient_slope(ages, off)
        on_f, on_l   = _first_last(ages, on)
        off_f, off_l = _first_last(ages, off)
        diff_v = (on - off); diff_v = diff_v[~np.isnan(diff_v)]
        onc_v  = onc[~np.isnan(onc)]; ofc_v = ofc[~np.isnan(ofc)]
        long_records[pid] = {
            "pt_on_mean": np.nanmean(on) if len(on_v) else np.nan,
            "pt_on_median": np.nanmedian(on) if len(on_v) else np.nan,
            "pt_on_std": np.nanstd(on) if len(on_v) >= 2 else np.nan,
            "pt_on_min": on_v.min() if len(on_v) else np.nan,
            "pt_on_max": on_v.max() if len(on_v) else np.nan,
            "pt_on_first": on_f, "pt_on_last": on_l,
            "pt_on_count": float(len(on_v)), "pt_on_miss_frac": float(np.isnan(on).mean()),
            "pt_on_slope": on_s, "pt_on_intercept": on_i,
            "pt_off_mean": np.nanmean(off) if len(off_v) else np.nan,
            "pt_off_median": np.nanmedian(off) if len(off_v) else np.nan,
            "pt_off_std": np.nanstd(off) if len(off_v) >= 2 else np.nan,
            "pt_off_min": off_v.min() if len(off_v) else np.nan,
            "pt_off_max": off_v.max() if len(off_v) else np.nan,
            "pt_off_first": off_f, "pt_off_last": off_l,
            "pt_off_count": float(len(off_v)), "pt_off_miss_frac": float(np.isnan(off).mean()),
            "pt_off_slope": off_s, "pt_off_intercept": off_i,
            "pt_ledd_mean": np.nanmean(ledd) if len(ledd_v) else np.nan,
            "pt_ledd_std": np.nanstd(ledd) if len(ledd_v) >= 2 else np.nan,
            "pt_ledd_min": ledd_v.min() if len(ledd_v) else np.nan,
            "pt_ledd_max": ledd_v.max() if len(ledd_v) else np.nan,
            "pt_age_min": age_min, "pt_age_max": age_max,
            "pt_age_span": age_max - age_min, "pt_visit_total": float(n),
            "pt_mean_on_off_gap": float(diff_v.mean()) if len(diff_v) else np.nan,
            "pt_on_corr_mean":  np.nanmean(onc) if len(onc_v) else np.nan,
            "pt_off_corr_mean": np.nanmean(ofc) if len(ofc_v) else np.nan,
            "pt_on_corr_std":   np.nanstd(onc)  if len(onc_v) >= 2 else np.nan,
            "pt_off_corr_std":  np.nanstd(ofc)  if len(ofc_v) >= 2 else np.nan,
        }
    long_stats = pd.DataFrame(long_records).T
    long_fb    = long_stats.mean(numeric_only=True)

    qj = quad_stats.reindex(df["patient_id"]); qj.index = df.index
    for col in qj.columns:
        qj[col] = qj[col].fillna(quad_fb.get(col, np.nan))
    age    = df["age"].values.astype(float)
    on_a   = qj["pt_on_quad"].values;  on_b  = qj["pt_on_qb"].values;  on_c  = qj["pt_on_qc"].values
    off_a  = qj["pt_off_quad"].values; off_b = qj["pt_off_qb"].values; off_c = qj["pt_off_qc"].values
    on_pred  = on_a  * age**2 + on_b  * age + on_c
    off_pred = off_a * age**2 + off_b * age + off_c
    df["pt_on_quad"]       = on_a;  df["pt_off_quad"]      = off_a
    df["pt_on_quad_pred"]  = on_pred; df["pt_off_quad_pred"] = off_pred
    df["pt_on_quad_res"]   = on_vals - on_pred
    df["pt_off_quad_res"]  = off_vals - off_pred
    df = df.drop(columns=["pt_on_qb","pt_on_qc","pt_off_qb","pt_off_qc"], errors="ignore")

    lj = long_stats.reindex(df["patient_id"]); lj.index = df.index
    for col in lj.columns:
        lj[col] = lj[col].fillna(long_fb.get(col, np.nan))
    age_min  = lj["pt_age_min"].values; age_max = lj["pt_age_max"].values
    age_span = age_max - age_min
    lj["pt_age_percentile"] = np.where(age_span > 0, (age - age_min) / age_span, 0.5)

    on_s = lj["pt_on_slope"].values; on_i = lj["pt_on_intercept"].values
    off_s = lj["pt_off_slope"].values; off_i = lj["pt_off_intercept"].values
    pred_on  = on_s  * age + on_i; pred_off = off_s * age + off_i
    ledd = df.get("ledd", pd.Series(np.nan, index=df.index)).values.astype(float)
    df["on_dev_from_trend"]  = on_vals  - pred_on;  df["off_dev_from_trend"] = off_vals - pred_off
    df["pred_on_at_age"]     = pred_on;              df["pred_off_at_age"]    = pred_off
    df["on_off_gap"]         = on_vals - off_vals;   df["ledd_x_on_off_gap"]  = ledd * (on_vals - off_vals)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["on_off_ratio"] = np.where(np.abs(off_vals) > 1e-9, on_vals / off_vals, np.nan)
    df["intake_on_x_on"]   = t_on  * on_vals
    df["intake_off_x_off"] = t_off * off_vals

    return pd.concat([df.drop(columns=["patient_id"]), lj], axis=1)


class LGBMWrapper(BaseEstimator):
    CAT_COLS = ["cohort", "gene", "rater_id"]

    def __init__(self): pass

    @staticmethod
    def _cast_cats(X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col in LGBMWrapper.CAT_COLS:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    def fit(self, X, y):
        self.model_ = LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                                    min_child_samples=20, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=3.0,
                                    random_state=42, verbose=-1)
        self.model_.fit(self._cast_cats(X), y)
        return self

    def predict(self, X):
        return self.model_.predict(self._cast_cats(X))


# --- Data ---
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
X = visits[ALL_FEATURE_COLS]; y = visits["target"]

print("Fitting Bateman on full train...")
params_on_final  = fit_bateman_on(visits[ALL_FEATURE_COLS + ["target"]])
params_off_final = fit_bateman_off(visits[ALL_FEATURE_COLS + ["target"]])
print(f"ON  params: {[round(float(x),4) for x in params_on_final] if params_on_final is not None else None}")
print(f"OFF params: {[round(float(x),4) for x in params_off_final] if params_off_final is not None else None}")

Xf_all = build_all_features(visits[ALL_FEATURE_COLS], params_on_final, params_off_final)
final  = LGBMWrapper().fit(Xf_all, y)
print(f"Final fit done ({time.time()-t0:.0f}s)")

sample = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")
test_ordered = X_test_e.set_index("Index").loc[sample["Index"]]
Xf_test = build_all_features(test_ordered[ALL_FEATURE_COLS], params_on_final, params_off_final)
predictions = final.predict(Xf_test)

submission = sample[["Index"]].copy()
submission["target"] = predictions
assert list(submission.columns) == ["Index", "target"]
assert np.isfinite(submission["target"].to_numpy()).all()

out = PROJECT_ROOT / "submissions" / "12_bateman.csv"
out.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(out, index=False)
print(f"\nSubmission saved: {out}")
print(f"Total time: {time.time()-t0:.0f}s")
