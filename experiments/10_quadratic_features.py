"""Experiment 10 — Quadratic patient trajectory features.

Hypothesis: per-patient target trajectories are quasi-perfectly fit by a
quadratic (R²=0.999 median vs 0.959 linear, diagnostic script 00).
Adding the quadratic coefficient (curvature) and residual features to the
Exp-09 pipeline should push RMSE below 4.190.

New features added by QuadraticAggregator on top of Exp-09:
  pt_on_quad      — quadratic coefficient of on ~ age (curvature)
  pt_off_quad     — quadratic coefficient of off ~ age
  pt_on_quad_res  — residual of on vs quadratic fit at current age
  pt_off_quad_res — residual of off vs quadratic fit at current age
  pt_on_quad_pred — quadratic-predicted on at current age
  pt_off_quad_pred— quadratic-predicted off at current age

Baseline (Exp-09): mean 5-fold RMSE = 4.1896, std = 0.057
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

import skore
from skore import EstimatorReport, Project, login

PROJECT_ROOT = Path(__file__).resolve().parents[1]
t_start = time.time()

# ---------------------------------------------------------------------------
# Helpers (identical to Exp 09)
# ---------------------------------------------------------------------------

def _patient_slope(ages, vals):
    mask = ~np.isnan(vals)
    a, v = ages[mask], vals[mask]
    if len(a) < 2 or np.std(a) < 1e-9:
        return np.nan, np.nan
    p = np.polyfit(a, v, 1)
    return float(p[0]), float(p[1])


def _patient_quad(ages, vals):
    """Fit quadratic; return (a, b, c) with a=curvature. Needs >= 3 points."""
    mask = ~np.isnan(vals)
    a, v = ages[mask], vals[mask]
    if len(a) < 3 or np.std(a) < 1e-9:
        return np.nan, np.nan, np.nan
    p = np.polyfit(a, v, 2)
    return float(p[0]), float(p[1]), float(p[2])


def _first_last(ages, vals):
    mask = ~np.isnan(vals)
    if mask.sum() == 0:
        return np.nan, np.nan
    order = np.argsort(ages[mask])
    v = vals[mask][order]
    return float(v[0]), float(v[-1])


# ---------------------------------------------------------------------------
# LongitudinalAggregator — identical to Exp 09
# ---------------------------------------------------------------------------

class LongitudinalAggregator(BaseEstimator, TransformerMixin):
    def _compute_patient_stats(self, df):
        records = {}
        for pid, grp in df.groupby("patient_id"):
            grp = grp.sort_values("age")
            ages = grp["age"].values.astype(float)
            on   = grp["on"].values.astype(float)
            off  = grp["off"].values.astype(float)
            ledd = grp["ledd"].values.astype(float)
            n = len(grp)
            age_min, age_max = float(ages.min()), float(ages.max())
            on_valid   = on[~np.isnan(on)]
            off_valid  = off[~np.isnan(off)]
            ledd_valid = ledd[~np.isnan(ledd)]
            on_slope,  on_int  = _patient_slope(ages, on)
            off_slope, off_int = _patient_slope(ages, off)
            on_first,  on_last   = _first_last(ages, on)
            off_first, off_last  = _first_last(ages, off)
            on_off_diffs = on - off
            valid_diff   = on_off_diffs[~np.isnan(on_off_diffs)]
            records[pid] = {
                "pt_on_mean": np.nanmean(on) if len(on_valid) else np.nan,
                "pt_on_median": np.nanmedian(on) if len(on_valid) else np.nan,
                "pt_on_std": np.nanstd(on) if len(on_valid) >= 2 else np.nan,
                "pt_on_min": on_valid.min() if len(on_valid) else np.nan,
                "pt_on_max": on_valid.max() if len(on_valid) else np.nan,
                "pt_on_first": on_first, "pt_on_last": on_last,
                "pt_on_count": float(len(on_valid)),
                "pt_on_miss_frac": float(np.isnan(on).mean()),
                "pt_on_slope": on_slope, "pt_on_intercept": on_int,
                "pt_off_mean": np.nanmean(off) if len(off_valid) else np.nan,
                "pt_off_median": np.nanmedian(off) if len(off_valid) else np.nan,
                "pt_off_std": np.nanstd(off) if len(off_valid) >= 2 else np.nan,
                "pt_off_min": off_valid.min() if len(off_valid) else np.nan,
                "pt_off_max": off_valid.max() if len(off_valid) else np.nan,
                "pt_off_first": off_first, "pt_off_last": off_last,
                "pt_off_count": float(len(off_valid)),
                "pt_off_miss_frac": float(np.isnan(off).mean()),
                "pt_off_slope": off_slope, "pt_off_intercept": off_int,
                "pt_ledd_mean": np.nanmean(ledd) if len(ledd_valid) else np.nan,
                "pt_ledd_std": np.nanstd(ledd) if len(ledd_valid) >= 2 else np.nan,
                "pt_ledd_min": ledd_valid.min() if len(ledd_valid) else np.nan,
                "pt_ledd_max": ledd_valid.max() if len(ledd_valid) else np.nan,
                "pt_age_min": age_min, "pt_age_max": age_max,
                "pt_age_span": age_max - age_min, "pt_visit_total": float(n),
                "pt_mean_on_off_gap": float(valid_diff.mean()) if len(valid_diff) else np.nan,
            }
        return pd.DataFrame(records).T

    def fit(self, X, y=None):
        stats = self._compute_patient_stats(X)
        self.global_fallback_ = stats.mean(numeric_only=True)
        return self

    def transform(self, X):
        df = X.copy().reset_index(drop=True)
        stats = self._compute_patient_stats(df)
        age_min  = stats["pt_age_min"].reindex(df["patient_id"]).values
        age_max  = stats["pt_age_max"].reindex(df["patient_id"]).values
        age_span = age_max - age_min
        age_pct  = np.where(age_span > 0, (df["age"].values - age_min) / age_span, 0.5)
        joined = stats.reindex(df["patient_id"])
        joined.index = df.index
        for col in joined.columns:
            joined[col] = joined[col].fillna(self.global_fallback_.get(col, np.nan))
        joined["pt_age_percentile"] = age_pct
        return pd.concat([df.drop(columns=["patient_id"]), joined], axis=1)


# ---------------------------------------------------------------------------
# QuadraticAggregator — NEW for Exp 10
# ---------------------------------------------------------------------------

class QuadraticAggregator(BaseEstimator, TransformerMixin):
    """Add quadratic trajectory features per patient.

    For each patient, fit a degree-2 polynomial on on~age and off~age.
    Adds:
      pt_on_quad, pt_off_quad        — curvature coefficient (a in a*t²+b*t+c)
      pt_on_quad_pred, pt_off_quad_pred — quadratic prediction at current age
      pt_on_quad_res, pt_off_quad_res   — residual (raw - quad_pred)
    """

    def _compute_quad_stats(self, df):
        records = {}
        for pid, grp in df.groupby("patient_id"):
            grp = grp.sort_values("age")
            ages = grp["age"].values.astype(float)
            on   = grp["on"].values.astype(float)
            off  = grp["off"].values.astype(float)
            on_a, on_b, on_c   = _patient_quad(ages, on)
            off_a, off_b, off_c = _patient_quad(ages, off)
            records[pid] = {
                "pt_on_quad":  on_a,  "pt_on_qb": on_b,  "pt_on_qc": on_c,
                "pt_off_quad": off_a, "pt_off_qb": off_b, "pt_off_qc": off_c,
            }
        return pd.DataFrame(records).T

    def fit(self, X, y=None):
        stats = self._compute_quad_stats(X)
        self.global_fallback_ = stats.mean(numeric_only=True)
        self.fitted_ = True
        return self

    def transform(self, X):
        # patient_id must still be present (called before LongitudinalAggregator)
        df = X.copy().reset_index(drop=True)

        stats = self._compute_quad_stats(df)
        joined = stats.reindex(df["patient_id"])
        joined.index = df.index
        for col in joined.columns:
            joined[col] = joined[col].fillna(self.global_fallback_.get(col, np.nan))

        age = df["age"].values.astype(float)
        on  = df["on"].values.astype(float)
        off = df["off"].values.astype(float)

        on_a  = joined["pt_on_quad"].values
        on_b  = joined["pt_on_qb"].values
        on_c  = joined["pt_on_qc"].values
        off_a = joined["pt_off_quad"].values
        off_b = joined["pt_off_qb"].values
        off_c = joined["pt_off_qc"].values

        on_pred  = on_a  * age**2 + on_b  * age + on_c
        off_pred = off_a * age**2 + off_b * age + off_c

        df["pt_on_quad"]       = on_a
        df["pt_off_quad"]      = off_a
        df["pt_on_quad_pred"]  = on_pred
        df["pt_off_quad_pred"] = off_pred
        df["pt_on_quad_res"]   = on  - on_pred
        df["pt_off_quad_res"]  = off - off_pred
        # drop helper intermediate cols
        df = df.drop(columns=["pt_on_qb", "pt_on_qc", "pt_off_qb", "pt_off_qc"],
                     errors="ignore")
        return df


# ---------------------------------------------------------------------------
# MedicationFeatureAdder — identical to Exp 09 (patient_id already dropped)
# ---------------------------------------------------------------------------

class MedicationFeatureAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        df = X.copy()
        age  = df["age"].values.astype(float)
        on   = df["on"].values.astype(float)
        off  = df["off"].values.astype(float)
        ledd = df.get("ledd", pd.Series(np.nan, index=df.index)).values.astype(float)
        on_slope  = df.get("pt_on_slope",      pd.Series(np.nan, index=df.index)).values.astype(float)
        on_int    = df.get("pt_on_intercept",  pd.Series(np.nan, index=df.index)).values.astype(float)
        off_slope = df.get("pt_off_slope",     pd.Series(np.nan, index=df.index)).values.astype(float)
        off_int   = df.get("pt_off_intercept", pd.Series(np.nan, index=df.index)).values.astype(float)
        pred_on  = on_slope  * age + on_int
        pred_off = off_slope * age + off_int
        df["on_dev_from_trend"]  = on  - pred_on
        df["off_dev_from_trend"] = off - pred_off
        df["pred_on_at_age"]     = pred_on
        df["pred_off_at_age"]    = pred_off
        df["on_off_gap"]         = on - off
        df["ledd_x_on_off_gap"]  = ledd * (on - off)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["on_off_ratio"]   = np.where(np.abs(off) > 1e-9, on / off, np.nan)
        tsi_on  = df.get("time_since_intake_on",  pd.Series(np.nan, index=df.index)).values.astype(float)
        tsi_off = df.get("time_since_intake_off", pd.Series(np.nan, index=df.index)).values.astype(float)
        df["intake_on_x_on"]   = tsi_on  * on
        df["intake_off_x_off"] = tsi_off * off
        return df


# ---------------------------------------------------------------------------
# LGBMWrapper — best config from Exp 09 (strong_reg)
# ---------------------------------------------------------------------------

class LGBMWrapper(BaseEstimator):
    CAT_COLS = ["cohort", "gene", "rater_id"]

    def __init__(self, n_estimators=2000, learning_rate=0.03, num_leaves=63,
                 min_child_samples=20, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=3.0,
                 random_state=42, verbose=-1):
        self.n_estimators = n_estimators; self.learning_rate = learning_rate
        self.num_leaves = num_leaves; self.min_child_samples = min_child_samples
        self.subsample = subsample; self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree; self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda; self.random_state = random_state
        self.verbose = verbose

    @staticmethod
    def _cast_cats(X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col in LGBMWrapper.CAT_COLS:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    def fit(self, X, y, **kw):
        self.model_ = LGBMRegressor(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            num_leaves=self.num_leaves, min_child_samples=self.min_child_samples,
            subsample=self.subsample, subsample_freq=self.subsample_freq,
            colsample_bytree=self.colsample_bytree, reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda, random_state=self.random_state,
            verbose=self.verbose)
        self.model_.fit(self._cast_cats(X), y, **kw)
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

def add_static_features(df):
    df = df.copy()
    df["years_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    df["off_minus_on"]  = df["off"] - df["on"]
    df["on_missing"]    = df["on"].isna().astype(int)
    df["off_missing"]   = df["off"].isna().astype(int)
    df["ledd_missing"]  = df["ledd"].isna().astype(int)
    return df

visits   = add_static_features(visits)
X_test_e = add_static_features(X_test_raw)

NUMERIC_COLS = [
    "sexM", "age_at_diagnosis", "age", "years_since_diagnosis",
    "ledd", "time_since_intake_on", "time_since_intake_off",
    "on", "off", "off_minus_on", "on_missing", "off_missing", "ledd_missing",
]
CAT_COLS = ["cohort", "gene", "rater_id"]
ALL_FEATURE_COLS = ["patient_id"] + NUMERIC_COLS + CAT_COLS

X = visits[ALL_FEATURE_COLS]
y = visits["target"]
groups = visits["patient_id"]

# Same splits as Exp 09
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

# ---------------------------------------------------------------------------
# Pipeline: QuadraticAggregator → LongitudinalAggregator → MedicationFeatureAdder → LGBM
# QuadraticAggregator must run first while patient_id is still in the DataFrame.
# LongitudinalAggregator drops patient_id in its transform output.
# ---------------------------------------------------------------------------
pipeline = make_pipeline(
    QuadraticAggregator(),
    LongitudinalAggregator(),
    MedicationFeatureAdder(),
    LGBMWrapper(),
)

# ---------------------------------------------------------------------------
# Smoke test — fold 0 only
# ---------------------------------------------------------------------------
print("\nSmoke test (fold 0)...")
tr0, va0 = cv_splits[0]
t0 = time.time()
smoke = clone(pipeline).fit(X.iloc[tr0], y.iloc[tr0])
smoke_preds = smoke.predict(X.iloc[va0])
smoke_rmse = float(np.sqrt(np.mean((y.iloc[va0].values - smoke_preds) ** 2)))
print(f"Fold-0 smoke RMSE: {smoke_rmse:.4f}  ({time.time()-t0:.0f}s)")
if smoke_rmse > 5.0:
    raise RuntimeError(f"Smoke RMSE {smoke_rmse:.4f} exceeds 5.0 — aborting")

# ---------------------------------------------------------------------------
# Full 5-fold CV
# ---------------------------------------------------------------------------
print("\nRunning full 5-fold GroupKFold CV...")
fold_rmses = []
for i, (tr_idx, va_idx) in enumerate(cv_splits):
    t0 = time.time()
    m = clone(pipeline).fit(X.iloc[tr_idx], y.iloc[tr_idx])
    preds = m.predict(X.iloc[va_idx])
    rmse = float(np.sqrt(np.mean((y.iloc[va_idx].values - preds) ** 2)))
    fold_rmses.append(rmse)
    print(f"  Fold {i}: RMSE={rmse:.4f}  ({time.time()-t0:.0f}s)")

mean_rmse = float(np.mean(fold_rmses))
std_rmse  = float(np.std(fold_rmses, ddof=1))
delta     = mean_rmse - 4.1896
print(f"\nMean RMSE : {mean_rmse:.6f}")
print(f"Std  RMSE : {std_rmse:.6f}")
print(f"vs Exp-09 : {delta:+.4f}")

# ---------------------------------------------------------------------------
# EstimatorReport + Hub publish
# ---------------------------------------------------------------------------
print("\nBuilding EstimatorReport (fold 0)...")
est_report = EstimatorReport(
    clone(pipeline),
    X_train=X.iloc[tr0], y_train=y.iloc[tr0],
    X_test=X.iloc[va0],  y_test=y.iloc[va0],
)
print(f"Fold-0 RMSE: {est_report.metrics.rmse()}")

cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("10_quadratic_features", est_report)
print("Published: key=10_quadratic_features")

# ---------------------------------------------------------------------------
# Final fit + submission
# ---------------------------------------------------------------------------
print("\nFitting final model on all training data...")
final = clone(pipeline).fit(X, y)

sample = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")
test_ordered = X_test_e.set_index("Index").loc[sample["Index"]]
X_test_final = test_ordered[ALL_FEATURE_COLS]
predictions  = final.predict(X_test_final)

submission = sample[["Index"]].copy()
submission["target"] = predictions
assert list(submission.columns) == ["Index", "target"]
assert len(submission) == len(sample)
assert submission["Index"].equals(sample["Index"])
assert submission["Index"].is_unique
assert np.isfinite(submission["target"].to_numpy()).all()
print("Submission validation passed.")

out = PROJECT_ROOT / "submissions" / "10_quadratic_features.csv"
out.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(out, index=False)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
elapsed = time.time() - t_start
print("\n===================================")
print("EXPERIMENT 10 — QUADRATIC FEATURES")
print("===================================")
print(f"Per-fold RMSE : {[round(v, 4) for v in fold_rmses]}")
print(f"Mean RMSE     : {mean_rmse:.6f}")
print(f"Std  RMSE     : {std_rmse:.6f}")
print(f"vs Exp-09     : {delta:+.4f}  ({'IMPROVEMENT' if delta < 0 else 'no improvement'})")
print(f"Total runtime : {elapsed:.0f}s")
print(f"\nSubmission : {out}")
print("Hub key    : 10_quadratic_features")
print("===================================")
