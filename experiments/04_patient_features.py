"""Experiment 04 — HistGradientBoosting + patient-level longitudinal features.

Builds on Experiment 03. Adds fold-safe patient aggregates:
  - years_since_diagnosis (static derived feature)
  - off_minus_on (when both available)
  - Missing-value indicators for on, off, ledd
  - Patient-level: mean/median on, mean/median off, mean ledd, visit count
    computed *per fold* from the training partition only.

The PatientAggregator is a stateful sklearn transformer: it learns the
per-patient stats from training rows and applies them (with global fallback)
to validation / test rows. patient_id is used only inside this transformer;
it is never passed as a raw predictor.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

import skore
from skore import EstimatorReport, Project, login

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Custom transformer: fold-safe patient-level aggregates
# ---------------------------------------------------------------------------

class PatientAggregator(BaseEstimator, TransformerMixin):
    """Compute per-patient statistics from feature rows — no target used.

    At fit time, learns the global mean fallback from the training partition.
    At transform time, computes per-patient aggregates from the rows being
    transformed (each patient's own visits), then joins them on.
    This is leakage-free: only input features (on, off, ledd) are aggregated,
    never the target. Patients with no usable rows fall back to the training
    global mean.
    """

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
        # Global fallbacks from training patients only
        self.global_fallback_ = agg.mean(numeric_only=True)
        return self

    def transform(self, X):
        df = X.copy().reset_index(drop=True)
        # Compute aggregates from the rows being transformed (features only)
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

        # Fill patients with all-NaN agg cols with global fallback
        for col in stats.columns:
            stats[col] = stats[col].fillna(self.global_fallback_[col])

        result = pd.concat([df.drop(columns=["patient_id"]), stats], axis=1)
        return result


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
# 2. Feature engineering (static, no target involved)
# ---------------------------------------------------------------------------
def add_static_features(df):
    df = df.copy()
    df["years_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    df["off_minus_on"] = df["off"] - df["on"]   # NaN when either missing
    df["on_missing"]   = df["on"].isna().astype(int)
    df["off_missing"]  = df["off"].isna().astype(int)
    df["ledd_missing"] = df["ledd"].isna().astype(int)
    return df

visits   = add_static_features(visits)
X_test_e = add_static_features(X_test_raw)

# Feature columns seen by the pipeline (patient_id included for aggregator,
# then dropped inside PatientAggregator.transform)
NUMERIC_COLS = [
    "sexM", "age_at_diagnosis", "age", "years_since_diagnosis",
    "ledd", "time_since_intake_on", "time_since_intake_off",
    "on", "off", "off_minus_on",
    "on_missing", "off_missing", "ledd_missing",
]
CAT_COLS = ["cohort", "gene", "rater_id"]
# patient_id is carried through X so PatientAggregator can use it,
# but it is never exposed to HGBR (dropped inside the transformer).
ALL_FEATURE_COLS = ["patient_id"] + NUMERIC_COLS + CAT_COLS

# After PatientAggregator.transform, patient_id is gone; new agg cols appended
PATIENT_AGG_COLS = [
    "pt_on_mean", "pt_on_median",
    "pt_off_mean", "pt_off_median",
    "pt_ledd_mean", "pt_visit_count",
]
NUMERIC_AFTER_AGG = NUMERIC_COLS + PATIENT_AGG_COLS  # for ColumnTransformer

X = visits[ALL_FEATURE_COLS]
y = visits["target"]
groups = visits["patient_id"]


# ---------------------------------------------------------------------------
# 3. Pipeline
#    PatientAggregator -> ColumnTransformer -> HGBR
#    Numeric cols pass through (NaN preserved for HGBR).
#    Cat cols: impute + OrdinalEncode.
# ---------------------------------------------------------------------------
cat_pipe = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
)

# After PatientAggregator the column order is: NUMERIC_AFTER_AGG + CAT_COLS.
# List every column explicitly: numeric first (passthrough), cats last.
# HGBR receives cats at the end; we tell it their column indices.
cat_start = len(NUMERIC_AFTER_AGG)
cat_indices = list(range(cat_start, cat_start + len(CAT_COLS)))

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUMERIC_AFTER_AGG),
        ("cat", cat_pipe, CAT_COLS),
    ]
)

hgbr_params = dict(
    categorical_features=cat_indices,
    random_state=0,
)

pipeline = make_pipeline(
    PatientAggregator(),
    preprocessor,
    HistGradientBoostingRegressor(**hgbr_params),
)


# ---------------------------------------------------------------------------
# 4. Same 5 GroupKFold splits as Experiment 03
# ---------------------------------------------------------------------------
print("\nBuilding GroupKFold splits...")
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

for fold, (tr_idx, va_idx) in enumerate(cv_splits):
    assert set(groups.iloc[tr_idx]).isdisjoint(set(groups.iloc[va_idx])), (
        f"Patient leakage in fold {fold}"
    )
print("5 folds -- no patient leakage confirmed.")


# ---------------------------------------------------------------------------
# 5. Evaluate with skore
# ---------------------------------------------------------------------------
print("\nRunning skore.evaluate (5-fold GroupKFold)...")
cv_report = skore.evaluate(pipeline, X, y, splitter=cv_splits)

rmse_per_fold = cv_report.metrics.rmse(aggregate=None)
print("\nPer-fold RMSE:")
print(rmse_per_fold.to_string())
mean_row = cv_report.metrics.rmse()
print(f"\nMean RMSE: {mean_row}")


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
project.put("04_patient_features", est_report)
print("EstimatorReport published -- key: 04_patient_features")

# ---------------------------------------------------------------------------
# 7. Fit on all training visits and generate submission
# ---------------------------------------------------------------------------
print("\nFitting final model on all training visits...")
final = clone(pipeline).fit(X, y)

sample = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")
assert sample["Index"].is_unique
assert X_test_e["Index"].is_unique
assert set(sample["Index"]) == set(X_test_e["Index"])

test_ordered = X_test_e.set_index("Index").loc[sample["Index"]]
X_test_final = test_ordered[ALL_FEATURE_COLS]

predictions = final.predict(X_test_final)

submission = sample[["Index"]].copy()
submission["target"] = predictions

assert list(submission.columns) == ["Index", "target"]
assert len(submission) == len(sample)
assert submission["Index"].equals(sample["Index"])
assert submission["Index"].is_unique
assert np.isfinite(submission["target"].to_numpy()).all()
print("Submission validation passed.")

out = PROJECT_ROOT / "submissions" / "04_patient_features.csv"
out.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(out, index=False)

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
fold_rmses = rmse_per_fold.iloc[0].tolist()
mean_rmse  = float(sum(fold_rmses) / len(fold_rmses))
std_rmse   = float(pd.Series(fold_rmses).std())
ref_rmse   = 7.484144
pct        = (ref_rmse - mean_rmse) / ref_rmse * 100

print("\n===================================")
print("EXPERIMENT 04 - PATIENT FEATURES")
print("===================================")
print(f"Exp-03 HGBR RMSE (ref) : {ref_rmse}")
print(f"Per-fold RMSE          : {[round(v, 4) for v in fold_rmses]}")
print(f"Mean RMSE              : {mean_rmse:.6f}")
print(f"Std  RMSE              : {std_rmse:.6f}")
print(f"Improvement vs Exp-03  : {pct:.2f}%")
print(f"\nSubmission saved : {out}")
print("Skore Hub key    : 04_patient_features")
print("===================================")
