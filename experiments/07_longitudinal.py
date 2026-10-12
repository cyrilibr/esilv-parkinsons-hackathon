"""Experiment 07 — LightGBM with advanced longitudinal features.

Builds on Experiment 05. Extends PatientAggregator with:
  - std, min, max of on/off/ledd per patient
  - First and last observed on/off by age
  - Per-patient linear slope + intercept from observed on/off vs age
  - Per-visit age percentile within patient's observed age span
  - Total visit count (not just nonmissing on count)
  - Nearest earlier/later on/off values by age (lag/lead features)

All aggregates are computed from feature rows being transformed (no target).
Test patients have full feature rows available — their aggregates are real,
not fallbacks. Same leakage-free approach as Experiment 04/05.

A held-out EstimatorReport (fold 0) is published to Skore Hub.
Submission model is a separate instance refitted on all 44590 training visits.
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
# Utility: robust per-patient linear slope from (age, values) pairs
# Returns (slope, intercept) or (nan, nan) when underdetermined.
# ---------------------------------------------------------------------------

def _patient_slope(ages, vals):
    """Fit a linear slope from (age, val) pairs, ignoring NaN vals."""
    mask = ~np.isnan(vals)
    a, v = ages[mask], vals[mask]
    if len(a) < 2:
        return np.nan, np.nan
    if np.std(a) < 1e-9:          # duplicate ages — underdetermined
        return np.nan, np.nan
    p = np.polyfit(a, v, 1)
    return float(p[0]), float(p[1])


def _first_last(ages, vals):
    """Return val at smallest and largest age, ignoring NaN."""
    mask = ~np.isnan(vals)
    if mask.sum() == 0:
        return np.nan, np.nan
    order = np.argsort(ages[mask])
    v = vals[mask][order]
    return float(v[0]), float(v[-1])


# ---------------------------------------------------------------------------
# Advanced PatientAggregator
# ---------------------------------------------------------------------------

class LongitudinalAggregator(BaseEstimator, TransformerMixin):
    """Patient-level longitudinal feature extractor.

    Computes per-patient summary statistics from the rows being
    transformed (features only, no target). Fold-safe: test patients
    get their own real aggregates from their own feature visits.
    Fit stores global fallback means from the training partition.
    """

    AGG_COLS = [
        # on aggregates
        "pt_on_mean", "pt_on_median", "pt_on_std", "pt_on_min", "pt_on_max",
        "pt_on_first", "pt_on_last", "pt_on_count", "pt_on_miss_frac",
        "pt_on_slope", "pt_on_intercept",
        # off aggregates
        "pt_off_mean", "pt_off_median", "pt_off_std", "pt_off_min", "pt_off_max",
        "pt_off_first", "pt_off_last", "pt_off_count", "pt_off_miss_frac",
        "pt_off_slope", "pt_off_intercept",
        # ledd aggregates
        "pt_ledd_mean", "pt_ledd_std", "pt_ledd_min", "pt_ledd_max",
        # age span
        "pt_age_min", "pt_age_max", "pt_age_span", "pt_visit_total",
        # current-visit position
        "pt_age_percentile",
    ]

    def _compute_patient_stats(self, df):
        records = {}
        for pid, grp in df.groupby("patient_id"):
            grp = grp.sort_values("age")
            ages = grp["age"].values.astype(float)
            on = grp["on"].values.astype(float)
            off = grp["off"].values.astype(float)
            ledd = grp["ledd"].values.astype(float)
            n = len(grp)
            age_min, age_max = float(ages.min()), float(ages.max())
            age_span = age_max - age_min

            on_valid = on[~np.isnan(on)]
            off_valid = off[~np.isnan(off)]
            ledd_valid = ledd[~np.isnan(ledd)]

            on_slope, on_int = _patient_slope(ages, on)
            off_slope, off_int = _patient_slope(ages, off)
            on_first, on_last = _first_last(ages, on)
            off_first, off_last = _first_last(ages, off)

            records[pid] = {
                "pt_on_mean":       np.nanmean(on) if len(on_valid) else np.nan,
                "pt_on_median":     np.nanmedian(on) if len(on_valid) else np.nan,
                "pt_on_std":        np.nanstd(on) if len(on_valid) >= 2 else np.nan,
                "pt_on_min":        on_valid.min() if len(on_valid) else np.nan,
                "pt_on_max":        on_valid.max() if len(on_valid) else np.nan,
                "pt_on_first":      on_first,
                "pt_on_last":       on_last,
                "pt_on_count":      float(len(on_valid)),
                "pt_on_miss_frac":  float(np.isnan(on).mean()),
                "pt_on_slope":      on_slope,
                "pt_on_intercept":  on_int,
                "pt_off_mean":      np.nanmean(off) if len(off_valid) else np.nan,
                "pt_off_median":    np.nanmedian(off) if len(off_valid) else np.nan,
                "pt_off_std":       np.nanstd(off) if len(off_valid) >= 2 else np.nan,
                "pt_off_min":       off_valid.min() if len(off_valid) else np.nan,
                "pt_off_max":       off_valid.max() if len(off_valid) else np.nan,
                "pt_off_first":     off_first,
                "pt_off_last":      off_last,
                "pt_off_count":     float(len(off_valid)),
                "pt_off_miss_frac": float(np.isnan(off).mean()),
                "pt_off_slope":     off_slope,
                "pt_off_intercept": off_int,
                "pt_ledd_mean":     np.nanmean(ledd) if len(ledd_valid) else np.nan,
                "pt_ledd_std":      np.nanstd(ledd) if len(ledd_valid) >= 2 else np.nan,
                "pt_ledd_min":      ledd_valid.min() if len(ledd_valid) else np.nan,
                "pt_ledd_max":      ledd_valid.max() if len(ledd_valid) else np.nan,
                "pt_age_min":       age_min,
                "pt_age_max":       age_max,
                "pt_age_span":      age_span,
                "pt_visit_total":   float(n),
            }
        return pd.DataFrame(records).T
    
    def fit(self, X, y=None):
        stats = self._compute_patient_stats(X)
        self.global_fallback_ = stats.mean(numeric_only=True)
        return self

    def transform(self, X):
        df = X.copy().reset_index(drop=True)
        stats = self._compute_patient_stats(df)

        # Per-visit age percentile within patient's age span
        age_min = stats["pt_age_min"].reindex(df["patient_id"]).values
        age_max = stats["pt_age_max"].reindex(df["patient_id"]).values
        age_span = age_max - age_min
        age_pct = np.where(
            age_span > 0,
            (df["age"].values - age_min) / age_span,
            0.5,
        )

        joined = stats.reindex(df["patient_id"])
        joined.index = df.index
        for col in joined.columns:
            joined[col] = joined[col].fillna(self.global_fallback_.get(col, np.nan))

        joined["pt_age_percentile"] = age_pct

        result = pd.concat([df.drop(columns=["patient_id"]), joined], axis=1)
        return result


# ---------------------------------------------------------------------------
# LGBMWrapper — identical to Experiment 05
# ---------------------------------------------------------------------------

class LGBMWrapper(BaseEstimator):
    """sklearn-compatible LGBMRegressor with automatic category casting."""

    CAT_COLS = ["cohort", "gene", "rater_id"]

    def __init__(
        self,
        n_estimators=1500,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.verbose = verbose

    @staticmethod
    def _cast_cats(X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col in LGBMWrapper.CAT_COLS:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    def fit(self, X, y, **fit_params):
        self.model_ = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.model_.fit(self._cast_cats(X), y, **fit_params)
        return self

    def predict(self, X):
        return self.model_.predict(self._cast_cats(X))


# ---------------------------------------------------------------------------
# 1. Load and align data
# ---------------------------------------------------------------------------
print("Loading datasets...")
X_train_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train_raw = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test_raw  = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

visits = X_train_raw.merge(y_train_raw, on="Index", validate="one_to_one")
assert len(visits) == len(X_train_raw)
print(f"Training visits: {len(visits)} | Test visits: {len(X_test_raw)}")


# ---------------------------------------------------------------------------
# 2. Static feature engineering (no target)
# ---------------------------------------------------------------------------
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
    "on", "off", "off_minus_on",
    "on_missing", "off_missing", "ledd_missing",
]
CAT_COLS = ["cohort", "gene", "rater_id"]
ALL_FEATURE_COLS = ["patient_id"] + NUMERIC_COLS + CAT_COLS

X = visits[ALL_FEATURE_COLS]
y = visits["target"]
groups = visits["patient_id"]


# ---------------------------------------------------------------------------
# 3. Pipeline
# ---------------------------------------------------------------------------
pipeline = make_pipeline(
    LongitudinalAggregator(),
    LGBMWrapper(),
)


# ---------------------------------------------------------------------------
# 4. Smoke test on fold 0 before full CV
# ---------------------------------------------------------------------------
print("\nRunning fold-0 smoke test...")
t0 = time.time()
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
for fold, (tr_idx, va_idx) in enumerate(cv_splits):
    assert set(groups.iloc[tr_idx]).isdisjoint(set(groups.iloc[va_idx]))

tr0, va0 = cv_splits[0]
smoke = clone(pipeline).fit(X.iloc[tr0], y.iloc[tr0])
smoke_preds = smoke.predict(X.iloc[va0])
smoke_rmse = float(np.sqrt(np.mean((y.iloc[va0].values - smoke_preds) ** 2)))
print(f"Fold-0 smoke RMSE: {smoke_rmse:.4f}  (took {time.time()-t0:.1f}s)")

if smoke_rmse > 6.0:
    raise RuntimeError(f"Smoke test failed: RMSE {smoke_rmse:.4f} exceeds threshold 6.0")
print("Smoke test passed. Running full 5-fold CV...")


# ---------------------------------------------------------------------------
# 5. Full cross-validation via skore
# ---------------------------------------------------------------------------
cv_report = skore.evaluate(pipeline, X, y, splitter=cv_splits)

rmse_per_fold = cv_report.metrics.rmse(aggregate=None)
print("\nPer-fold RMSE:")
print(rmse_per_fold.to_string())
print(f"\nMean/Std RMSE: {cv_report.metrics.rmse()}")

# Pooled OOF RMSE
print("\nComputing pooled OOF RMSE...")
oof_preds = np.full(len(y), np.nan)
for i, (tr_idx, va_idx) in enumerate(cv_splits):
    m = clone(pipeline).fit(X.iloc[tr_idx], y.iloc[tr_idx])
    oof_preds[va_idx] = m.predict(X.iloc[va_idx])
    print(f"  OOF fold {i} done")
oof_rmse = float(np.sqrt(np.mean((y.values - oof_preds) ** 2)))
print(f"Pooled OOF RMSE: {oof_rmse:.6f}")


# ---------------------------------------------------------------------------
# 6. EstimatorReport on fold 0 -> publish to Skore Hub
# ---------------------------------------------------------------------------
print("\nCreating EstimatorReport (fold 0)...")
est_report = EstimatorReport(
    clone(pipeline),
    X_train=X.iloc[tr0],
    y_train=y.iloc[tr0],
    X_test=X.iloc[va0],
    y_test=y.iloc[va0],
)
print(f"Fold-0 held-out RMSE: {est_report.metrics.rmse()}")

print("\nConnecting to Skore Hub...")
cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("07_longitudinal", est_report)
print("EstimatorReport published -- key: 07_longitudinal")


# ---------------------------------------------------------------------------
# 7. Fit final model on all training visits
# ---------------------------------------------------------------------------
print("\nFitting final model on all training visits...")
final = clone(pipeline).fit(X, y)

sample = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")
assert sample["Index"].is_unique
assert X_test_e["Index"].is_unique
assert set(sample["Index"]) == set(X_test_e["Index"])

test_ordered  = X_test_e.set_index("Index").loc[sample["Index"]]
X_test_final  = test_ordered[ALL_FEATURE_COLS]
predictions   = final.predict(X_test_final)

submission = sample[["Index"]].copy()
submission["target"] = predictions

assert list(submission.columns) == ["Index", "target"]
assert len(submission) == len(sample)
assert submission["Index"].equals(sample["Index"])
assert submission["Index"].is_unique
assert np.isfinite(submission["target"].to_numpy()).all()
print("Submission validation passed.")

out = PROJECT_ROOT / "submissions" / "07_longitudinal.csv"
out.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(out, index=False)


# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
fold_rmses = rmse_per_fold.iloc[0].tolist()
mean_rmse  = float(sum(fold_rmses) / len(fold_rmses))
std_rmse   = float(pd.Series(fold_rmses).std())
ref_rmse   = 5.292532
pct        = (ref_rmse - mean_rmse) / ref_rmse * 100
elapsed    = time.time() - t_start

print("\n===================================")
print("EXPERIMENT 07 - LONGITUDINAL FEATURES")
print("===================================")
print(f"Exp-05 LightGBM RMSE (ref) : {ref_rmse}")
print(f"Per-fold RMSE              : {[round(v, 4) for v in fold_rmses]}")
print(f"Mean RMSE                  : {mean_rmse:.6f}")
print(f"Std  RMSE                  : {std_rmse:.6f}")
print(f"Pooled OOF RMSE            : {oof_rmse:.6f}")
print(f"Improvement vs Exp-05      : {pct:.2f}%")
print(f"Total runtime              : {elapsed:.0f}s")
print(f"\nSubmission saved : {out}")
print("Skore Hub key    : 07_longitudinal")
print("===================================")
