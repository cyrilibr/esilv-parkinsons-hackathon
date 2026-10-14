"""Experiment 15 — Per-patient smoothing of predictions (post-processing).

Hypothesis: the true OFF score is a smooth function of age within a patient
(per-patient quadratic fits the target with ~0.33 RMSE). The per-visit model
inherits visit-level noise from on/off. Projecting each patient's predictions
onto a low-degree polynomial in age removes that noise.

Pipeline = Exp-12 unchanged (no inter-visit features). Smoothing uses only
predictions + age (never target), so it is applied identically to test.

Baseline (Exp-12): mean 5-fold RMSE = 4.1310, std = 0.0652
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
KEY = "15_patient_smoothing"
BASELINE, BASELINE_STD = 4.1310, 0.0652
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
# Feature engineering (Exp-12 pipeline, unchanged, no inter-visit block)
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

    # keep patient_id (as _pid) for the smoothing step; the model ignores it
    pid = df["patient_id"].values
    result = pd.concat([df.drop(columns=["patient_id"]), lj], axis=1)
    result["_pid"] = pid
    return result


# ---------------------------------------------------------------------------
# The ONE new idea: per-patient smoothing of predictions
# ---------------------------------------------------------------------------

def smooth_by_patient(pred, pid, age, deg=2, alpha=1.0):
    """Fit a degree-`deg` polynomial in age to each patient's predictions and
    blend: alpha * smoothed + (1 - alpha) * raw. Target is never used."""
    pred = np.asarray(pred, dtype=float)
    age = np.asarray(age, dtype=float)
    out = pred.copy()
    for idx in pd.Series(np.arange(len(pid))).groupby(np.asarray(pid)).indices.values():
        a, p = age[idx], pred[idx]
        if np.ptp(a) < 1e-9 or len(idx) < 2:
            continue
        d = min(deg, len(idx) - 1)
        c = np.polyfit(a - a.mean(), p, d)
        out[idx] = alpha * np.polyval(c, a - a.mean()) + (1 - alpha) * p
    return out


class SmoothedLGBM(BaseEstimator, RegressorMixin):
    CAT_COLS = ["cohort", "gene", "rater_id"]

    def __init__(self, deg=2, alpha=1.0):
        self.deg = deg
        self.alpha = alpha

    def _prep(self, X):
        df = X.drop(columns=["_pid"]).copy()
        for col in self.CAT_COLS:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    def fit(self, X, y):
        self.model_ = LGBMRegressor(
            n_estimators=2000, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=3.0,
            random_state=42, verbose=-1)
        self.model_.fit(self._prep(X), y)
        return self

    def predict_raw(self, X):
        return self.model_.predict(self._prep(X))

    def predict(self, X):
        return smooth_by_patient(self.predict_raw(X), X["_pid"].values,
                                 X["age"].values, self.deg, self.alpha)


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
CAT_COLS = ["cohort", "gene", "rater_id"]
ALL_FEATURE_COLS = ["patient_id"] + NUMERIC_COLS + CAT_COLS

y = visits["target"]
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(visits, y, groups=groups))

# Variants evaluated on the SAME out-of-fold raw predictions
VARIANTS = [(None, 0.0)] + [(d, a) for d in (1, 2) for a in (0.5, 0.75, 1.0)]


def vname(d, a):
    return "raw (no smoothing)" if d is None else f"deg={d} alpha={a}"


# ---------------------------------------------------------------------------
# 5-fold CV
# ---------------------------------------------------------------------------
print("\nRunning 5-fold CV...", flush=True)
scores = {v: [] for v in VARIANTS}
for i, (tr_idx, va_idx) in enumerate(cv_splits):
    t0 = time.time()
    X_tr = visits.iloc[tr_idx][ALL_FEATURE_COLS + ["target"]]
    X_va = visits.iloc[va_idx][ALL_FEATURE_COLS]
    y_tr, y_va = y.iloc[tr_idx].values, y.iloc[va_idx].values

    p_on, p_off = fit_bateman_on(X_tr), fit_bateman_off(X_tr)   # OOF
    Xf_tr = build_all_features(X_tr[ALL_FEATURE_COLS], p_on, p_off)
    Xf_va = build_all_features(X_va, p_on, p_off)

    m = SmoothedLGBM().fit(Xf_tr, y_tr)
    raw = m.predict_raw(Xf_va)
    for d, a in VARIANTS:
        pred = raw if d is None else smooth_by_patient(
            raw, Xf_va["_pid"].values, Xf_va["age"].values, d, a)
        scores[(d, a)].append(float(np.sqrt(np.mean((y_va - pred) ** 2))))
    print(f"  Fold {i}: raw={scores[(None, 0.0)][-1]:.4f}  "
          f"deg2/a1={scores[(2, 1.0)][-1]:.4f}  ({time.time() - t0:.0f}s)",
          flush=True)

print("\nVariant                 mean     std     delta vs 4.131  gain>std")
for v in VARIANTS:
    s = np.array(scores[v])
    g = BASELINE - s.mean()
    print(f"{vname(*v):22s}  {s.mean():.4f}  {s.std(ddof=1):.4f}   "
          f"{-g:+.4f}          {'YES' if g > s.std(ddof=1) else 'NO'}")

best = min((v for v in VARIANTS if v[0] is not None),
           key=lambda v: np.mean(scores[v]))
best_s = np.array(scores[best])
raw_s = np.array(scores[(None, 0.0)])
gain = BASELINE - best_s.mean()
paired = raw_s - best_s   # same folds, same model -> paired comparison
print(f"\nBest variant : {vname(*best)}")
print(f"Mean RMSE    : {best_s.mean():.4f}   std: {best_s.std(ddof=1):.4f}")
print(f"delta vs {BASELINE}: {-gain:+.4f}   gain > std: "
      f"{'YES' if gain > best_s.std(ddof=1) else 'NO'}")
print(f"Paired gain vs raw (same folds): {paired.mean():+.4f} "
      f"(improves {int((paired > 0).sum())}/5 folds)")

# ---------------------------------------------------------------------------
# EstimatorReport (fold 0, best variant) + Hub
# ---------------------------------------------------------------------------
tr0, va0 = cv_splits[0]
X_tr0 = visits.iloc[tr0][ALL_FEATURE_COLS + ["target"]]
X_va0 = visits.iloc[va0][ALL_FEATURE_COLS]
p_on0, p_off0 = fit_bateman_on(X_tr0), fit_bateman_off(X_tr0)
Xf_tr0 = build_all_features(X_tr0[ALL_FEATURE_COLS], p_on0, p_off0)
Xf_va0 = build_all_features(X_va0, p_on0, p_off0)

print("\nBuilding EstimatorReport (fold 0)...", flush=True)
est_report = EstimatorReport(
    SmoothedLGBM(deg=best[0], alpha=best[1]),
    X_train=Xf_tr0, y_train=y.iloc[tr0].values,
    X_test=Xf_va0, y_test=y.iloc[va0].values,
)
cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put(KEY, est_report)
print(f"Published: key={KEY}")

# ---------------------------------------------------------------------------
# Final fit + submission (best variant)
# ---------------------------------------------------------------------------
print("\nFinal fit on full train...", flush=True)
full = visits[ALL_FEATURE_COLS + ["target"]]
p_on_f, p_off_f = fit_bateman_on(full), fit_bateman_off(full)
Xf_all = build_all_features(visits[ALL_FEATURE_COLS], p_on_f, p_off_f)
final = SmoothedLGBM(deg=best[0], alpha=best[1]).fit(Xf_all, y.values)

sample = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")
test_ordered = X_test_e.set_index("Index").loc[sample["Index"]].reset_index()
Xf_test = build_all_features(test_ordered[ALL_FEATURE_COLS], p_on_f, p_off_f)
predictions = final.predict(Xf_test)

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
print("EXPERIMENT 15 — PER-PATIENT SMOOTHING")
print("===================================")
print(f"Best variant  : {vname(*best)}")
print(f"Per-fold RMSE : {[round(v, 4) for v in best_s]}")
print(f"Mean / std    : {best_s.mean():.4f} / {best_s.std(ddof=1):.4f}")
print(f"vs Exp-12     : {-gain:+.4f}   gain > std: "
      f"{'YES' if gain > best_s.std(ddof=1) else 'NO'}")
print(f"Runtime       : {time.time() - t_start:.0f}s")
print(f"Submission    : {out}")
print(f"Hub key       : {KEY}")
print("===================================")