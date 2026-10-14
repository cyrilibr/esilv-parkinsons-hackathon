"""Experiment 17 — Stacking on the per-patient smoothed curve.

Hypothesis: smoothing (Exp-15) removes visit-level noise but cannot fix a
patient-level offset (the whole curve too high/low). A stage-2 LightGBM fed
with the stage-1 curve of each patient (level, slope, curvature, smoothed
value, residual) plus the original features can correct that offset.

Leakage control:
  - stage-1 predictions on the training part are OUT-OF-FOLD (inner
    GroupKFold on patients), with Bateman refit inside each inner fold
  - stage-1 predictions on validation/test come from a model fit on the
    whole training part
  - target is never used as a feature

Reference: Exp-12 = 4.1310 (std 0.0652), Exp-15 = 4.0256 (std 0.0648)

Set SKIP_CV=1 to skip the CV and reuse the recorded fold scores.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold

from skore import EstimatorReport, Project, login

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEY = "17_stack_smooth"
BASELINE, BASELINE_STD = 4.1310, 0.0652
EXP15 = 4.0256
INNER_SPLITS = 5
SKIP_CV = os.environ.get("SKIP_CV", "0") == "1"
t_start = time.time()


# ---------------------------------------------------------------------------
# Bateman curve (unchanged from Exp-12)
# ---------------------------------------------------------------------------

def bateman(t, A, ke, ka):
    with np.errstate(over="ignore", invalid="ignore"):
        val = A * (np.exp(-ke * t) - np.exp(-ka * t))
    return np.clip(val, 0.0, 0.95)


def fit_bateman_on(df_train):
    sub = df_train.dropna(subset=["on", "time_since_intake_on", "target"]).copy()
    sub = sub[sub["target"].abs() > 1e-6]
    if len(sub) < 50:
        return None
    t = sub["time_since_intake_on"].values.astype(float)
    frac = (1.0 - sub["on"].values.astype(float) / sub["target"].values.astype(float))
    frac = np.clip(frac, 0.0, 0.95)
    try:
        popt, _ = curve_fit(bateman, t, frac, p0=[0.6, 0.5, 2.0],
                            bounds=([0.0, 0.01, 0.1], [1.0, 5.0, 20.0]),
                            maxfev=5000)
        return popt
    except Exception:
        return None


def fit_bateman_off(df_train):
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
        popt, _ = curve_fit(decay, t, frac, p0=[0.3, 0.1],
                            bounds=([0.0, 0.001], [1.0, 2.0]), maxfev=5000)
        return popt
    except Exception:
        return None


def apply_on_correction(on_vals, t_on_vals, params_on):
    if params_on is None:
        return np.full(len(on_vals), np.nan)
    A, ke, ka = params_on
    corr = np.full(len(on_vals), np.nan)
    valid = ~(np.isnan(on_vals) | np.isnan(t_on_vals))
    if valid.sum() == 0:
        return corr
    denom = 1.0 - bateman(t_on_vals[valid], A, ke, ka)
    denom = np.where(denom < 0.05, 0.05, denom)
    corr[valid] = on_vals[valid] / denom
    return corr


def apply_off_correction(off_vals, t_off_vals, params_off):
    if params_off is None:
        return np.full(len(off_vals), np.nan)
    B, ke = params_off
    corr = np.full(len(off_vals), np.nan)
    valid = ~(np.isnan(off_vals) | np.isnan(t_off_vals))
    if valid.sum() == 0:
        return corr
    eff = np.clip(B * np.exp(-ke * t_off_vals[valid]), 0.0, 0.95)
    denom = 1.0 - eff
    denom = np.where(denom < 0.05, 0.05, denom)
    corr[valid] = off_vals[valid] / denom
    return corr


# ---------------------------------------------------------------------------
# Helpers (unchanged)
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
# Feature engineering (Exp-12 pipeline, unchanged)
# ---------------------------------------------------------------------------

def add_static_features(df):
    df = df.copy()
    df["years_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    df["off_minus_on"] = df["off"] - df["on"]
    df["on_missing"] = df["on"].isna().astype(int)
    df["off_missing"] = df["off"].isna().astype(int)
    df["ledd_missing"] = df["ledd"].isna().astype(int)
    return df


def build_all_features(df_raw, params_on=None, params_off=None):
    df = df_raw.copy().reset_index(drop=True)

    on_vals = df["on"].values.astype(float)
    off_vals = df["off"].values.astype(float)
    t_on = df["time_since_intake_on"].values.astype(float)
    t_off = df["time_since_intake_off"].values.astype(float)

    on_corr = apply_on_correction(on_vals, t_on, params_on)
    off_corr = apply_off_correction(off_vals, t_off, params_off)
    df["on_corr"] = on_corr
    df["off_corr"] = off_corr
    df["on_corr_delta"] = on_corr - on_vals
    df["off_corr_delta"] = off_corr - off_vals
    df["on_off_corr_gap"] = on_corr - off_corr

    quad_records = {}
    for pid, grp in df.groupby("patient_id"):
        ages = grp["age"].values.astype(float)
        on = grp["on"].values.astype(float)
        off = grp["off"].values.astype(float)
        on_a, on_b, on_c = _patient_quad(ages, on)
        off_a, off_b, off_c = _patient_quad(ages, off)
        quad_records[pid] = {
            "pt_on_quad": on_a, "pt_on_qb": on_b, "pt_on_qc": on_c,
            "pt_off_quad": off_a, "pt_off_qb": off_b, "pt_off_qc": off_c,
        }
    quad_stats = pd.DataFrame(quad_records).T
    quad_fb = quad_stats.mean(numeric_only=True)

    long_records = {}
    for pid, grp in df.groupby("patient_id"):
        grp = grp.sort_values("age")
        ages = grp["age"].values.astype(float)
        on = grp["on"].values.astype(float)
        off = grp["off"].values.astype(float)
        ledd = grp["ledd"].values.astype(float)
        on_corr_p = grp["on_corr"].values.astype(float)
        off_corr_p = grp["off_corr"].values.astype(float)
        n = len(grp)
        age_min, age_max = float(ages.min()), float(ages.max())
        on_v = on[~np.isnan(on)]; off_v = off[~np.isnan(off)]; ledd_v = ledd[~np.isnan(ledd)]
        on_s, on_i = _patient_slope(ages, on)
        off_s, off_i = _patient_slope(ages, off)
        on_f, on_l = _first_last(ages, on)
        off_f, off_l = _first_last(ages, off)
        diff_v = (on - off); diff_v = diff_v[~np.isnan(diff_v)]
        on_corr_v = on_corr_p[~np.isnan(on_corr_p)]
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
            "pt_on_corr_mean": np.nanmean(on_corr_p) if len(on_corr_v) else np.nan,
            "pt_off_corr_mean": np.nanmean(off_corr_p) if len(off_corr_v) else np.nan,
            "pt_on_corr_std": np.nanstd(on_corr_p) if len(on_corr_v) >= 2 else np.nan,
            "pt_off_corr_std": np.nanstd(off_corr_p) if len(off_corr_v) >= 2 else np.nan,
        }
    long_stats = pd.DataFrame(long_records).T
    long_fb = long_stats.mean(numeric_only=True)

    qj = quad_stats.reindex(df["patient_id"]); qj.index = df.index
    for col in qj.columns:
        qj[col] = qj[col].fillna(quad_fb.get(col, np.nan))
    age = df["age"].values.astype(float)
    on_a = qj["pt_on_quad"].values; on_b = qj["pt_on_qb"].values; on_c = qj["pt_on_qc"].values
    off_a = qj["pt_off_quad"].values; off_b = qj["pt_off_qb"].values; off_c = qj["pt_off_qc"].values
    on_pred = on_a * age**2 + on_b * age + on_c
    off_pred = off_a * age**2 + off_b * age + off_c
    df["pt_on_quad"] = on_a
    df["pt_off_quad"] = off_a
    df["pt_on_quad_pred"] = on_pred
    df["pt_off_quad_pred"] = off_pred
    df["pt_on_quad_res"] = on_vals - on_pred
    df["pt_off_quad_res"] = off_vals - off_pred

    lj = long_stats.reindex(df["patient_id"]); lj.index = df.index
    for col in lj.columns:
        lj[col] = lj[col].fillna(long_fb.get(col, np.nan))
    age_min = lj["pt_age_min"].values; age_max = lj["pt_age_max"].values
    age_span = age_max - age_min
    lj["pt_age_percentile"] = np.where(age_span > 0, (age - age_min) / age_span, 0.5)

    on_s = lj["pt_on_slope"].values; on_i = lj["pt_on_intercept"].values
    off_s = lj["pt_off_slope"].values; off_i = lj["pt_off_intercept"].values
    pred_on = on_s * age + on_i
    pred_off = off_s * age + off_i
    df["on_dev_from_trend"] = on_vals - pred_on
    df["off_dev_from_trend"] = off_vals - pred_off
    df["pred_on_at_age"] = pred_on
    df["pred_off_at_age"] = pred_off
    df["on_off_gap"] = on_vals - off_vals
    ledd = df["ledd"].values.astype(float)
    df["ledd_x_on_off_gap"] = ledd * (on_vals - off_vals)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["on_off_ratio"] = np.where(np.abs(off_vals) > 1e-9, on_vals / off_vals, np.nan)
    df["intake_on_x_on"] = t_on * on_vals
    df["intake_off_x_off"] = t_off * off_vals

    pid = df["patient_id"].values
    result = pd.concat([df.drop(columns=["patient_id"]), lj], axis=1)
    result["_pid"] = pid
    return result


# ---------------------------------------------------------------------------
# Smoothing (Exp-15) + curve features for stacking
# ---------------------------------------------------------------------------

def _patient_groups(pid):
    return pd.Series(np.arange(len(pid))).groupby(np.asarray(pid)).indices.values()


def smooth_by_patient(pred, pid, age):
    """Exp-15 smoother: per-patient quadratic in age fitted on predictions."""
    pred = np.asarray(pred, dtype=float)
    age = np.asarray(age, dtype=float)
    out = pred.copy()
    for idx in _patient_groups(pid):
        a, p = age[idx], pred[idx]
        if len(idx) < 2 or np.ptp(a) < 1e-9:
            continue
        x = a - a.mean()
        out[idx] = np.polyval(np.polyfit(x, p, min(2, len(idx) - 1)), x)
    return out


def curve_features(pred, pid, age):
    """Per-visit stacking features from the stage-1 predictions of a patient."""
    pred = np.asarray(pred, dtype=float)
    age = np.asarray(age, dtype=float)
    n = len(pred)
    f = {k: np.full(n, np.nan) for k in
         ["s1_pred", "s1_smooth", "s1_resid", "s1_level", "s1_slope",
          "s1_curv", "s1_pt_mean", "s1_pt_std", "s1_resid_rms"]}
    f["s1_pred"] = pred.copy()
    for idx in _patient_groups(pid):
        a, p = age[idx], pred[idx]
        f["s1_pt_mean"][idx] = p.mean()
        f["s1_pt_std"][idx] = p.std()
        if len(idx) < 2 or np.ptp(a) < 1e-9:
            f["s1_smooth"][idx] = p
            f["s1_level"][idx] = p.mean()
            continue
        x = a - a.mean()
        d = min(2, len(idx) - 1)
        c = np.polyfit(x, p, d)
        sm = np.polyval(c, x)
        f["s1_smooth"][idx] = sm
        f["s1_level"][idx] = c[-1]
        f["s1_slope"][idx] = c[-2]
        if d == 2:
            f["s1_curv"][idx] = c[0]
        f["s1_resid_rms"][idx] = np.sqrt(np.mean((p - sm) ** 2))
    f["s1_resid"] = f["s1_pred"] - f["s1_smooth"]
    return pd.DataFrame(f)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
CAT = ["cohort", "gene", "rater_id"]


def _prep(X):
    df = X.drop(columns=["_pid"]).copy()
    for col in CAT:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def make_lgbm_stage1():
    return LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=63,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=3.0,
        random_state=42, verbose=-1)


def make_lgbm_stage2():
    return LGBMRegressor(
        n_estimators=1000, learning_rate=0.03, num_leaves=31,
        min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
        random_state=43, verbose=-1)


def _with_target(X_raw, y):
    df = X_raw.copy()
    df["target"] = np.asarray(y, dtype=float)
    return df


class Stage1:
    """Bateman fit + Exp-12 features + LGBM, all on the data it is fit on."""

    def fit(self, X_raw, y):
        d = _with_target(X_raw, y)
        self.p_on_ = fit_bateman_on(d)
        self.p_off_ = fit_bateman_off(d)
        self.model_ = make_lgbm_stage1().fit(_prep(self.features(X_raw)), y)
        return self

    def features(self, X_raw):
        return build_all_features(X_raw, self.p_on_, self.p_off_)

    def predict(self, X_raw):
        return self.model_.predict(_prep(self.features(X_raw)))


class StackSmooth(BaseEstimator, RegressorMixin):
    """Stage-1 (OOF on train) -> curve features -> stage-2 LGBM -> smoothing."""

    def __init__(self, inner_splits=INNER_SPLITS):
        self.inner_splits = inner_splits

    def fit(self, X, y):
        X_raw = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        pid = X_raw["patient_id"].values
        age = X_raw["age"].values

        # stage-1 out-of-fold predictions on the training data
        oof = np.full(len(y), np.nan)
        inner = GroupKFold(n_splits=self.inner_splits)
        for itr, iva in inner.split(X_raw, y, groups=pid):
            s1 = Stage1().fit(X_raw.iloc[itr], y[itr])
            oof[iva] = s1.predict(X_raw.iloc[iva])

        # stage-1 model on all training data (used for val/test)
        self.stage1_ = Stage1().fit(X_raw, y)

        F = self.stage1_.features(X_raw)
        S = curve_features(oof, pid, age)
        X2 = pd.concat([F.reset_index(drop=True), S], axis=1)
        self.cols2_ = [c for c in X2.columns if c != "_pid"]
        self.stage2_ = make_lgbm_stage2().fit(_prep(X2), y)
        return self

    def predict_all(self, X):
        X_raw = X.reset_index(drop=True)
        pid = X_raw["patient_id"].values
        age = X_raw["age"].values
        F = self.stage1_.features(X_raw)
        s1_raw = self.stage1_.model_.predict(_prep(F))
        S = curve_features(s1_raw, pid, age)
        X2 = pd.concat([F.reset_index(drop=True), S], axis=1)
        s2_raw = self.stage2_.predict(_prep(X2)[self.cols2_])
        return {
            "stage1 raw (=Exp-12)": s1_raw,
            "stage1 smoothed (=Exp-15)": smooth_by_patient(s1_raw, pid, age),
            "stage2 raw": s2_raw,
            "stage2 smoothed": smooth_by_patient(s2_raw, pid, age),
        }

    def predict(self, X):
        return self.predict_all(X)["stage2 smoothed"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
print("Loading data...", flush=True)
X_train_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train_raw = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

visits = X_train_raw.merge(y_train_raw, on="Index", validate="one_to_one")
visits = add_static_features(visits)
X_test_e = add_static_features(X_test_raw)

NUMERIC_COLS = ["sexM", "age_at_diagnosis", "age", "years_since_diagnosis", "ledd",
                "time_since_intake_on", "time_since_intake_off", "on", "off",
                "off_minus_on", "on_missing", "off_missing", "ledd_missing"]
ALL_FEATURE_COLS = ["patient_id"] + NUMERIC_COLS + CAT

X_all = visits[ALL_FEATURE_COLS].reset_index(drop=True)
y = visits["target"].values.astype(float)
groups = visits["patient_id"].values
cv_splits = list(GroupKFold(n_splits=5).split(X_all, y, groups=groups))

# ---------------------------------------------------------------------------
# 5-fold CV
# ---------------------------------------------------------------------------
VARIANTS = ["stage1 raw (=Exp-12)", "stage1 smoothed (=Exp-15)",
            "stage2 raw", "stage2 smoothed"]

if SKIP_CV:
    print("\nSKIP_CV=1: reusing recorded fold scores from the previous run.",
          flush=True)
    scores = {
        "stage1 raw (=Exp-12)": [4.1698, 4.2148, 4.0774, 4.0565, 4.1366],
        "stage1 smoothed (=Exp-15)": [4.0686, 4.1044, 3.9672, 3.9531, 4.0347],
        "stage2 raw": [np.nan] * 5,
        "stage2 smoothed": [3.7713, 3.8145, 3.7281, 3.7183, 3.7937],
    }
else:
    print(f"\nRunning 5-fold CV (inner {INNER_SPLITS}-fold stacking)...", flush=True)
    scores = {v: [] for v in VARIANTS}
    for i, (tr_idx, va_idx) in enumerate(cv_splits):
        t0 = time.time()
        m = StackSmooth().fit(X_all.iloc[tr_idx], y[tr_idx])
        preds = m.predict_all(X_all.iloc[va_idx])
        for v in VARIANTS:
            scores[v].append(float(np.sqrt(np.mean((y[va_idx] - preds[v]) ** 2))))
        print(f"  Fold {i}: exp15={scores['stage1 smoothed (=Exp-15)'][-1]:.4f}  "
              f"stack+smooth={scores['stage2 smoothed'][-1]:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

print("\nVariant                      mean     std     vs 4.131   vs exp15")
for v in VARIANTS:
    s = np.array(scores[v], dtype=float)
    if np.isnan(s).all():
        continue
    print(f"{v:27s}  {s.mean():.4f}  {s.std(ddof=1):.4f}   "
          f"{s.mean() - BASELINE:+.4f}    {s.mean() - EXP15:+.4f}")

final_s = np.array(scores["stage2 smoothed"], dtype=float)
ref_s = np.array(scores["stage1 smoothed (=Exp-15)"], dtype=float)
gain = BASELINE - final_s.mean()
paired = ref_s - final_s
print(f"\nStack+smooth : {final_s.mean():.4f}   std: {final_s.std(ddof=1):.4f}")
print(f"delta vs {BASELINE}: {-gain:+.4f}   gain > std: "
      f"{'YES' if gain > final_s.std(ddof=1) else 'NO'}")
print(f"Paired gain vs Exp-15 (same folds): {paired.mean():+.4f} "
      f"(improves {int((paired > 0).sum())}/5 folds)", flush=True)

# ---------------------------------------------------------------------------
# EstimatorReport (fold 0, skore refits it) + Hub
# ---------------------------------------------------------------------------
tr0, va0 = cv_splits[0]
print("\nBuilding EstimatorReport (fold 0, ~3 min)...", flush=True)
est_report = EstimatorReport(
    StackSmooth(),
    X_train=X_all.iloc[tr0], y_train=y[tr0],
    X_test=X_all.iloc[va0], y_test=y[va0],
)

print("Logging in to skore hub (check for a browser/auth prompt)...", flush=True)
cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put(KEY, est_report)
print(f"Published: key={KEY}", flush=True)

# ---------------------------------------------------------------------------
# Final fit + submission
# ---------------------------------------------------------------------------
print("\nFinal fit on full train (stage 1 OOF + stage 2, ~4 min)...", flush=True)
final = StackSmooth().fit(X_all, y)

sample = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")
test_ordered = X_test_e.set_index("Index").loc[sample["Index"]].reset_index()
predictions = final.predict(test_ordered[ALL_FEATURE_COLS])

submission = sample[["Index"]].copy()
submission["target"] = predictions
assert list(submission.columns) == ["Index", "target"]
assert len(submission) == len(sample)
assert submission["Index"].is_unique
assert np.isfinite(submission["target"].to_numpy()).all()

out = PROJECT_ROOT / "submissions" / f"{KEY}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(out, index=False)

print("\n===================================")
print("EXPERIMENT 17 — STACKING ON SMOOTHED CURVE")
print("===================================")
print(f"Per-fold RMSE : {[round(float(v), 4) for v in final_s]}")
print(f"Mean / std    : {final_s.mean():.4f} / {final_s.std(ddof=1):.4f}")
print(f"vs Exp-12     : {-gain:+.4f}   gain > std: "
      f"{'YES' if gain > final_s.std(ddof=1) else 'NO'}")
print(f"vs Exp-15     : {final_s.mean() - EXP15:+.4f}   "
      f"(paired, improves {int((paired > 0).sum())}/5 folds)")
print(f"Runtime       : {time.time() - t_start:.0f}s")
print(f"Submission    : {out}")
print(f"Hub key       : {KEY}")
print("===================================")