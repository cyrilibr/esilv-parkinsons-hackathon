"""Experiment 11 — finalize: hub publish + submission only.

CV results already obtained:
  Fold 0: 4.1774  Fold 1: 4.2506  Fold 2: 4.1349  Fold 3: 4.0793  Fold 4: 4.1798
  Mean: 4.164392  Std: 0.063165  vs Exp-10: -0.0070  (IMPROVEMENT)

This script just builds the EstimatorReport (fold 0), pushes to hub, fits
the final model and writes the submission CSV.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import GroupKFold

import skore
from skore import EstimatorReport, Project, login

PROJECT_ROOT = Path(__file__).resolve().parents[1]
t_start = time.time()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patient_slope(ages, vals):
    mask = ~np.isnan(vals)
    a, v = ages[mask], vals[mask]
    if len(a) < 2 or np.std(a) < 1e-9:
        return np.nan, np.nan
    p = np.polyfit(a, v, 1)
    return float(p[0]), float(p[1])


def _patient_quad(ages, vals):
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
# Transformers
# ---------------------------------------------------------------------------

class QuadraticAggregator(BaseEstimator, TransformerMixin):
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
        df = X.copy().reset_index(drop=True)
        stats = self._compute_quad_stats(df)
        joined = stats.reindex(df["patient_id"])
        joined.index = df.index
        for col in joined.columns:
            joined[col] = joined[col].fillna(self.global_fallback_.get(col, np.nan))
        age = df["age"].values.astype(float)
        on  = df["on"].values.astype(float)
        off = df["off"].values.astype(float)
        on_a  = joined["pt_on_quad"].values;  on_b  = joined["pt_on_qb"].values;  on_c  = joined["pt_on_qc"].values
        off_a = joined["pt_off_quad"].values; off_b = joined["pt_off_qb"].values; off_c = joined["pt_off_qc"].values
        on_pred  = on_a  * age**2 + on_b  * age + on_c
        off_pred = off_a * age**2 + off_b * age + off_c
        df["pt_on_quad"]       = on_a
        df["pt_off_quad"]      = off_a
        df["pt_on_quad_pred"]  = on_pred
        df["pt_off_quad_pred"] = off_pred
        df["pt_on_quad_res"]   = on  - on_pred
        df["pt_off_quad_res"]  = off - off_pred
        df = df.drop(columns=["pt_on_qb","pt_on_qc","pt_off_qb","pt_off_qc"], errors="ignore")
        return df


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
            on_valid   = on[~np.isnan(on)];  off_valid  = off[~np.isnan(off)];  ledd_valid = ledd[~np.isnan(ledd)]
            on_slope, on_int   = _patient_slope(ages, on)
            off_slope, off_int = _patient_slope(ages, off)
            on_first, on_last   = _first_last(ages, on)
            off_first, off_last = _first_last(ages, off)
            on_off_diffs = on - off
            valid_diff = on_off_diffs[~np.isnan(on_off_diffs)]
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
            df["on_off_ratio"] = np.where(np.abs(off) > 1e-9, on / off, np.nan)
        tsi_on  = df.get("time_since_intake_on",  pd.Series(np.nan, index=df.index)).values.astype(float)
        tsi_off = df.get("time_since_intake_off", pd.Series(np.nan, index=df.index)).values.astype(float)
        df["intake_on_x_on"]   = tsi_on  * on
        df["intake_off_x_off"] = tsi_off * off
        return df


class RaterBiasEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing

    def fit(self, X, y):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        df = df.reset_index(drop=True)
        target = pd.Series(y, name="target").reset_index(drop=True)
        off_bias, on_bias, counts = {}, {}, {}
        for rater, grp_idx in df.groupby("rater_id").groups.items():
            grp = df.iloc[grp_idx]; tgt = target.iloc[grp_idx]; n = len(grp); counts[rater] = n
            off_vals = grp["off"].values.astype(float); tgt_vals = tgt.values.astype(float)
            valid_off = ~np.isnan(off_vals)
            off_bias[rater] = (float(np.mean(off_vals[valid_off] - tgt_vals[valid_off])) * n / (n + self.smoothing)) if valid_off.sum() > 0 else 0.0
            on_vals = grp["on"].values.astype(float); valid_on = ~np.isnan(on_vals)
            on_bias[rater] = (float(np.mean(on_vals[valid_on] - tgt_vals[valid_on])) * n / (n + self.smoothing)) if valid_on.sum() > 0 else 0.0
        self.off_bias_ = off_bias; self.on_bias_ = on_bias; self.counts_ = counts; self.fitted_ = True
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        rater = df["rater_id"].astype(str)
        off_b = rater.map(self.off_bias_).fillna(0.0).values
        on_b  = rater.map(self.on_bias_).fillna(0.0).values
        cnt   = rater.map(self.counts_).fillna(0.0).values
        df["rater_off_bias"]       = off_b
        df["rater_on_bias"]        = on_b
        df["rater_visit_count"]    = cnt
        off_vals = df["off"].values.astype(float); on_vals = df["on"].values.astype(float)
        df["rater_off_bias_x_off"] = off_b * np.where(np.isnan(off_vals), 0.0, off_vals)
        df["rater_on_bias_x_on"]   = on_b  * np.where(np.isnan(on_vals),  0.0, on_vals)
        return df


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
        self.reg_lambda = reg_lambda; self.random_state = random_state; self.verbose = verbose

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
            reg_lambda=self.reg_lambda, random_state=self.random_state, verbose=self.verbose)
        self.model_.fit(self._cast_cats(X), y, **kw)
        return self

    def predict(self, X):
        return self.model_.predict(self._cast_cats(X))


class Pipeline11(BaseEstimator):
    """QuadAgg → LongAgg → MedAgg → RaterBias(y) → LGBM."""

    def __init__(self):
        self.quad  = QuadraticAggregator()
        self.long  = LongitudinalAggregator()
        self.med   = MedicationFeatureAdder()
        self.rater = RaterBiasEncoder(smoothing=20.0)
        self.lgbm  = LGBMWrapper()

    def _forward(self, X, y=None, fitting=False):
        X1 = self.quad.fit_transform(X)  if fitting else self.quad.transform(X)
        X2 = self.long.fit_transform(X1) if fitting else self.long.transform(X1)
        X3 = self.med.fit_transform(X2)   if fitting else self.med.transform(X2)
        if fitting:
            X4 = self.rater.fit(X3, y).transform(X3)
        else:
            X4 = self.rater.transform(X3)
        return X4

    def fit(self, X, y):
        Xf = self._forward(X, y=y, fitting=True)
        self.lgbm.fit(Xf, y)
        return self

    def predict(self, X):
        return self.lgbm.predict(self._forward(X, fitting=False))


# ---------------------------------------------------------------------------
# Data
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
NUMERIC_COLS = ["sexM","age_at_diagnosis","age","years_since_diagnosis","ledd",
                "time_since_intake_on","time_since_intake_off","on","off",
                "off_minus_on","on_missing","off_missing","ledd_missing"]
CAT_COLS = ["cohort","gene","rater_id"]
ALL_FEATURE_COLS = ["patient_id"] + NUMERIC_COLS + CAT_COLS
X = visits[ALL_FEATURE_COLS]
y = visits["target"]
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
tr0, va0 = cv_splits[0]

# ---------------------------------------------------------------------------
# EstimatorReport (fold 0) + hub
# ---------------------------------------------------------------------------
print("Fitting fold-0 model for EstimatorReport...")
t0 = time.time()
m0 = Pipeline11().fit(X.iloc[tr0], y.iloc[tr0])
preds0 = m0.predict(X.iloc[va0])
fold0_rmse = float(np.sqrt(np.mean((y.iloc[va0].values - preds0)**2)))
print(f"Fold-0 RMSE: {fold0_rmse:.4f}  ({time.time()-t0:.0f}s)")

est_report = EstimatorReport(
    Pipeline11(),
    X_train=X.iloc[tr0], y_train=y.iloc[tr0],
    X_test=X.iloc[va0],  y_test=y.iloc[va0],
)
print(f"EstimatorReport RMSE: {est_report.metrics.rmse()}")

cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("11_rater_bias", est_report)
print("Published: key=11_rater_bias")

# ---------------------------------------------------------------------------
# Final fit + submission
# ---------------------------------------------------------------------------
print("\nFitting final model on all training data...")
t0 = time.time()
final = Pipeline11().fit(X, y)
print(f"Final fit done ({time.time()-t0:.0f}s)")

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

out = PROJECT_ROOT / "submissions" / "11_rater_bias.csv"
out.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(out, index=False)

elapsed = time.time() - t_start
print("\n===================================")
print("EXPERIMENT 11 — RATER BIAS (OOF)")
print("===================================")
print("CV results (from earlier run):")
print("Per-fold RMSE : [4.1774, 4.2506, 4.1349, 4.0793, 4.1798]")
print("Mean RMSE     : 4.164392")
print("Std  RMSE     : 0.063165")
print("vs Exp-10     : -0.0070  (IMPROVEMENT: 0.11 × std)")
print(f"Total runtime : {elapsed:.0f}s")
print(f"\nSubmission : {out}")
print("Hub key    : 11_rater_bias")
print("===================================")
