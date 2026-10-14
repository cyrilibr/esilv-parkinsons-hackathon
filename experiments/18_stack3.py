"""Experiment 18 — Third stacking level on the smoothed curves.

Builds on Exp-17 (stage 2 on stage-1 curves + smoothing = 3.7652).
Hypothesis: stage 2 is much better than stage 1, so feeding stage-2
out-of-fold curves (plus stage-1 curves) to a stage-3 model corrects more of
the patient-level error.

Leakage control:
  - inside each outer training part, stage-1 and stage-2 predictions are
    out-of-fold over the same inner GroupKFold on patients
  - validation/test patients are only predicted by models that never saw them
  - target is never used as a feature

The "stage2 smoothed" variant reproduces Exp-17 on the same folds.
Reference: Exp-12 = 4.1310, Exp-15 = 4.0256, Exp-17 = 3.7652 (std 0.0414)
"""

import json
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
KEY = "18_stack3"
BASELINE, BASELINE_STD = 4.1310, 0.0652
EXP17 = 3.7652
INNER_SPLITS = 5
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


def curve_features(pred, pid, age, prefix):
    """Per-visit stacking features from a stage's predictions of a patient."""
    pred = np.asarray(pred, dtype=float)
    age = np.asarray(age, dtype=float)
    n = len(pred)
    names = ["pred", "smooth", "resid", "level", "slope",
             "curv", "pt_mean", "pt_std", "resid_rms"]
    f = {k: np.full(n, np.nan) for k in names}
    f["pred"] = pred.copy()
    for idx in _patient_groups(pid):
        a, p = age[idx], pred[idx]
        f["pt_mean"][idx] = p.mean()
        f["pt_std"][idx] = p.std()
        if len(idx) < 2 or np.ptp(a) < 1e-9:
            f["smooth"][idx] = p
            f["level"][idx] = p.mean()
            continue
        x = a - a.mean()
        d = min(2, len(idx) - 1)
        c = np.polyfit(x, p, d)
        sm = np.polyval(c, x)
        f["smooth"][idx] = sm
        f["level"][idx] = c[-1]
        f["slope"][idx] = c[-2]
        if d == 2:
            f["curv"][idx] = c[0]
        f["resid_rms"][idx] = np.sqrt(np.mean((p - sm) ** 2))
    f["resid"] = f["pred"] - f["smooth"]
    return pd.DataFrame({f"{prefix}_{k}": f[k] for k in names})


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
CAT = ["cohort", "gene", "rater_id"]


def _prep(X):
    df = X.drop(columns=["_pid"], errors="ignore").copy()
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


def make_lgbm_stage3():
    return LGBMRegressor(
        n_estimators=800, learning_rate=0.03, num_leaves=31,
        min_child_samples=60, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=8.0,
        random_state=44, verbose=-1)


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


class Stack3(BaseEstimator, RegressorMixin):
    """Stage 1 -> curves -> stage 2 -> curves -> stage 3 -> smoothing.
    Stage-1 and stage-2 training predictions are out-of-fold (inner folds)."""

    def __init__(self, inner_splits=INNER_SPLITS):
        self.inner_splits = inner_splits

    def fit(self, X, y):
        X_raw = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        pid = X_raw["patient_id"].values
        age = X_raw["age"].values
        folds = list(GroupKFold(n_splits=self.inner_splits)
                     .split(X_raw, y, groups=pid))

        # --- stage 1: OOF on training data + full model
        oof1 = np.full(len(y), np.nan)
        for itr, iva in folds:
            s1 = Stage1().fit(X_raw.iloc[itr], y[itr])
            oof1[iva] = s1.predict(X_raw.iloc[iva])
        self.stage1_ = Stage1().fit(X_raw, y)
        F = self.stage1_.features(X_raw).reset_index(drop=True)

        # --- stage 2: OOF on training data (same inner folds) + full model
        X2 = pd.concat([F, curve_features(oof1, pid, age, "s1")], axis=1)
        X2p = _prep(X2)
        self.cols2_ = list(X2p.columns)
        oof2 = np.full(len(y), np.nan)
        for itr, iva in folds:
            m2 = make_lgbm_stage2().fit(X2p.iloc[itr], y[itr])
            oof2[iva] = m2.predict(X2p.iloc[iva])
        self.stage2_ = make_lgbm_stage2().fit(X2p, y)

        # --- stage 3 on original features + stage-1 and stage-2 curves
        X3 = pd.concat([X2, curve_features(oof2, pid, age, "s2")], axis=1)
        X3p = _prep(X3)
        self.cols3_ = list(X3p.columns)
        self.stage3_ = make_lgbm_stage3().fit(X3p, y)
        return self

    def predict_all(self, X):
        X_raw = X.reset_index(drop=True)
        pid = X_raw["patient_id"].values
        age = X_raw["age"].values
        F = self.stage1_.features(X_raw).reset_index(drop=True)
        s1 = self.stage1_.model_.predict(_prep(F))
        X2 = pd.concat([F, curve_features(s1, pid, age, "s1")], axis=1)
        s2 = self.stage2_.predict(_prep(X2)[self.cols2_])
        X3 = pd.concat([X2, curve_features(s2, pid, age, "s2")], axis=1)
        s3 = self.stage3_.predict(_prep(X3)[self.cols3_])
        return {
            "stage1 smoothed (=Exp-15)": smooth_by_patient(s1, pid, age),
            "stage2 smoothed (=Exp-17)": smooth_by_patient(s2, pid, age),
            "stage3 raw": s3,
            "stage3 smoothed": smooth_by_patient(s3, pid, age),
        }

    def predict(self, X):
        return self.predict_all(X)["stage3 smoothed"]


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
print(f"\nRunning 5-fold CV (3-level stacking, inner {INNER_SPLITS} folds)...",
      flush=True)
VARIANTS = ["stage1 smoothed (=Exp-15)", "stage2 smoothed (=Exp-17)",
            "stage3 raw", "stage3 smoothed"]
scores = {v: [] for v in VARIANTS}
for i, (tr_idx, va_idx) in enumerate(cv_splits):
    t0 = time.time()
    m = Stack3().fit(X_all.iloc[tr_idx], y[tr_idx])
    preds = m.predict_all(X_all.iloc[va_idx])
    for v in VARIANTS:
        scores[v].append(float(np.sqrt(np.mean((y[va_idx] - preds[v]) ** 2))))
    print(f"  Fold {i}: exp17={scores['stage2 smoothed (=Exp-17)'][-1]:.4f}  "
          f"stage3+smooth={scores['stage3 smoothed'][-1]:.4f}  "
          f"({time.time() - t0:.0f}s)", flush=True)

print("\nVariant                      mean     std     vs 4.131   vs exp17")
for v in VARIANTS:
    s = np.array(scores[v])
    print(f"{v:27s}  {s.mean():.4f}  {s.std(ddof=1):.4f}   "
          f"{s.mean() - BASELINE:+.4f}    {s.mean() - EXP17:+.4f}")

final_s = np.array(scores["stage3 smoothed"])
ref_s = np.array(scores["stage2 smoothed (=Exp-17)"])
gain = BASELINE - final_s.mean()
paired = ref_s - final_s
print(f"\nStage3+smooth: {final_s.mean():.4f}   std: {final_s.std(ddof=1):.4f}")
print(f"delta vs {BASELINE}: {-gain:+.4f}   gain > std: "
      f"{'YES' if gain > final_s.std(ddof=1) else 'NO'}")
print(f"Paired gain vs Exp-17 (same folds): {paired.mean():+.4f} "
      f"(improves {int((paired > 0).sum())}/5 folds)", flush=True)

# ---------------------------------------------------------------------------
# EstimatorReport (fold 0, skore refits it) + Hub
# ---------------------------------------------------------------------------
tr0, va0 = cv_splits[0]
print("\nBuilding EstimatorReport (fold 0, ~5 min)...", flush=True)
est_report = EstimatorReport(
    Stack3(),
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
print("\nFinal fit on full train (~6 min)...", flush=True)
final = Stack3().fit(X_all, y)

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
print("EXPERIMENT 18 — THREE-LEVEL STACKING")
print("===================================")
print(f"Per-fold RMSE : {[round(float(v), 4) for v in final_s]}")
print(f"Mean / std    : {final_s.mean():.4f} / {final_s.std(ddof=1):.4f}")
print(f"vs Exp-12     : {-gain:+.4f}   gain > std: "
      f"{'YES' if gain > final_s.std(ddof=1) else 'NO'}")
print(f"vs Exp-17     : {final_s.mean() - EXP17:+.4f}   "
      f"(paired, improves {int((paired > 0).sum())}/5 folds)")
print(f"Runtime       : {time.time() - t_start:.0f}s")
print(f"Submission    : {out}")
print(f"Hub key       : {KEY}")
print("===================================")