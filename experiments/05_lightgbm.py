"""Experiment 05 — LightGBM with patient-level longitudinal features.

Reuses the full feature engineering from Experiment 04 (PatientAggregator,
static derived features, missing-value indicators). Replaces HGBR with
LGBMRegressor. Categoricals are passed as pandas category dtype; LightGBM
handles them natively without OrdinalEncoder. NaN in numeric columns is
preserved — LightGBM handles missing values natively.

A held-out EstimatorReport (fold 0) is published to Skore Hub.
The submission model is a separate instance refitted on all 44 590 visits.
"""

import json
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


# ---------------------------------------------------------------------------
# PatientAggregator — identical to Experiment 04
# Computes per-patient mean/median on, off, ledd and visit count from the
# rows being transformed (no target involved → leakage-free).
# Fit stores a global fallback from the training partition.
# ---------------------------------------------------------------------------

class PatientAggregator(BaseEstimator, TransformerMixin):
    """Fold-safe patient-level feature aggregator (feature columns only)."""

    def fit(self, X, y=None):
        df = X.copy()
        agg = df.groupby("patient_id").agg(
            pt_on_mean=("on", "mean"),
            pt_on_median=("on", "median"),
            pt_off_mean=("off", "mean"),
            pt_off_median=("off", "median"),
            pt_ledd_mean=("ledd", "mean"),
            pt_visit_count=("on", "count"),
        )
        self.global_fallback_ = agg.mean(numeric_only=True)
        return self

    def transform(self, X):
        df = X.copy().reset_index(drop=True)
        agg = df.groupby("patient_id").agg(
            pt_on_mean=("on", "mean"),
            pt_on_median=("on", "median"),
            pt_off_mean=("off", "mean"),
            pt_off_median=("off", "median"),
            pt_ledd_mean=("ledd", "mean"),
            pt_visit_count=("on", "count"),
        )
        stats = agg.reindex(df["patient_id"])
        stats.index = df.index
        for col in stats.columns:
            stats[col] = stats[col].fillna(self.global_fallback_[col])
        result = pd.concat([df.drop(columns=["patient_id"]), stats], axis=1)
        return result


# ---------------------------------------------------------------------------
# LightGBM wrapper that accepts a DataFrame with category dtype columns
# (LightGBM reads them directly; no OrdinalEncoder needed).
# ---------------------------------------------------------------------------

class LGBMWrapper(BaseEstimator):
    """Thin sklearn-compatible wrapper around LGBMRegressor.

    Converts CAT_COLS to pandas category dtype before fit/predict so
    LightGBM picks them up as categoricals natively.
    Uses explicit constructor params so sklearn clone() works correctly.
    """

    CAT_COLS = ["cohort", "gene", "rater_id"]

    def __init__(
        self,
        n_estimators=1000,
        learning_rate=0.05,
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
# patient_id is carried for PatientAggregator, dropped inside it
ALL_FEATURE_COLS = ["patient_id"] + NUMERIC_COLS + CAT_COLS

X = visits[ALL_FEATURE_COLS]
y = visits["target"]
groups = visits["patient_id"]


# ---------------------------------------------------------------------------
# 3. Pipeline: PatientAggregator -> LGBMWrapper
#    No ColumnTransformer needed: LightGBM handles NaN and categoricals
#    natively via the wrapper's dtype conversion.
# ---------------------------------------------------------------------------
pipeline = make_pipeline(
    PatientAggregator(),
    LGBMWrapper(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    ),
)


# ---------------------------------------------------------------------------
# 4. Same 5 GroupKFold splits as Experiments 03 and 04
# ---------------------------------------------------------------------------
print("\nBuilding GroupKFold splits...")
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
for fold, (tr_idx, va_idx) in enumerate(cv_splits):
    assert set(groups.iloc[tr_idx]).isdisjoint(set(groups.iloc[va_idx])), (
        f"Patient leakage in fold {fold}"
    )
print("5 folds -- no patient leakage confirmed.")


# ---------------------------------------------------------------------------
# 5. Cross-validation via skore
# ---------------------------------------------------------------------------
print("\nRunning skore.evaluate (5-fold GroupKFold)...")
cv_report = skore.evaluate(pipeline, X, y, splitter=cv_splits)

rmse_per_fold = cv_report.metrics.rmse(aggregate=None)
print("\nPer-fold RMSE:")
print(rmse_per_fold.to_string())
print(f"\nMean/Std RMSE: {cv_report.metrics.rmse()}")


# ---------------------------------------------------------------------------
# 6. EstimatorReport on fold 0 -> publish to Skore Hub
# ---------------------------------------------------------------------------
print("\nCreating EstimatorReport (fold 0)...")
tr0, va0 = cv_splits[0]
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
project.put("05_lightgbm", est_report)
print("EstimatorReport published -- key: 05_lightgbm")


# ---------------------------------------------------------------------------
# 7. Fit final model on all training visits (separate instance from report)
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

out = PROJECT_ROOT / "submissions" / "05_lightgbm.csv"
out.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(out, index=False)


# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
fold_rmses = rmse_per_fold.iloc[0].tolist()
mean_rmse  = float(sum(fold_rmses) / len(fold_rmses))
std_rmse   = float(pd.Series(fold_rmses).std())
ref_rmse   = 5.345763
pct        = (ref_rmse - mean_rmse) / ref_rmse * 100

print("\n===================================")
print("EXPERIMENT 05 - LIGHTGBM")
print("===================================")
print(f"Exp-04 RMSE (ref)  : {ref_rmse}")
print(f"Per-fold RMSE      : {[round(v, 4) for v in fold_rmses]}")
print(f"Mean RMSE          : {mean_rmse:.6f}")
print(f"Std  RMSE          : {std_rmse:.6f}")
print(f"Improvement vs E04 : {pct:.2f}%")
print(f"\nSubmission saved : {out}")
print("Skore Hub key    : 05_lightgbm")
print("===================================")
