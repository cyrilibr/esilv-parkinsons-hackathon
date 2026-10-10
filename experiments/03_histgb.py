"""Experiment 03 — HistGradientBoostingRegressor with GroupKFold.

HistGradientBoosting handles NaN natively (missing = signal, not noise).
Categorical features are encoded per-fold inside ColumnTransformer.
A held-out EstimatorReport (fold 0) is published to Skore Hub.
The submission model is refitted on all 44 590 training visits.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
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
# 2. Feature engineering: years_since_diagnosis
# ---------------------------------------------------------------------------
visits = visits.copy()
visits["years_since_diagnosis"] = visits["age"] - visits["age_at_diagnosis"]

X_test_raw = X_test_raw.copy()
X_test_raw["years_since_diagnosis"] = X_test_raw["age"] - X_test_raw["age_at_diagnosis"]

# ---------------------------------------------------------------------------
# 3. Feature sets (Index and patient_id excluded)
# ---------------------------------------------------------------------------
# NaN values in numeric cols are kept — HGBR learns their pattern
NUMERIC_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "years_since_diagnosis",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
CAT_COLS = ["cohort", "gene", "rater_id"]
FEATURE_COLS = NUMERIC_COLS + CAT_COLS

X = visits[FEATURE_COLS]
y = visits["target"]
groups = visits["patient_id"]

# ---------------------------------------------------------------------------
# 4. Pipeline — OrdinalEncoder for categoricals, passthrough for numerics
#    HGBR handles NaN natively; no imputer on numeric columns.
#    OrdinalEncoder uses encoded_missing_value so NaN in cat cols is safe.
# ---------------------------------------------------------------------------
cat_pipe = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_pipe, CAT_COLS),
    ],
    remainder="passthrough",  # numeric cols pass through unchanged (NaN preserved)
)
hgbr = make_pipeline(
    preprocessor,
    HistGradientBoostingRegressor(
        categorical_features=list(range(len(CAT_COLS))),
        random_state=0,
    ),
)

# ---------------------------------------------------------------------------
# 5. Grouped CV — same 5 folds as experiment 02
# ---------------------------------------------------------------------------
print("\nBuilding GroupKFold splits...")
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

for fold, (tr_idx, va_idx) in enumerate(cv_splits):
    assert set(groups.iloc[tr_idx]).isdisjoint(set(groups.iloc[va_idx])), (
        f"Patient leakage in fold {fold}"
    )
print("5 folds — no patient leakage confirmed.")

# ---------------------------------------------------------------------------
# 6. Cross-validation via skore
# ---------------------------------------------------------------------------
print("\nRunning skore.evaluate (5-fold GroupKFold)...")
cv_report = skore.evaluate(hgbr, X, y, splitter=cv_splits)

rmse_df = cv_report.metrics.rmse(aggregate=None)
print("\nPer-fold RMSE:")
print(rmse_df.to_string())
print(f"\nMean RMSE : {cv_report.metrics.rmse()}")

# ---------------------------------------------------------------------------
# 7. EstimatorReport on fold 0 → publish to Skore Hub
# ---------------------------------------------------------------------------
print("\nCreating EstimatorReport (fold 0)...")
tr0, va0 = cv_splits[0]
est_report = EstimatorReport(
    clone(hgbr),
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
project.put("03_histgb", est_report)
print("EstimatorReport published -- key: 03_histgb")

# ---------------------------------------------------------------------------
# 8. Fit on all training visits and generate submission
# ---------------------------------------------------------------------------
print("\nFitting final model on all training visits...")
final = clone(hgbr).fit(X, y)

sample = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")
assert sample["Index"].is_unique
assert X_test_raw["Index"].is_unique
assert set(sample["Index"]) == set(X_test_raw["Index"])

test_ordered = X_test_raw.set_index("Index").loc[sample["Index"]]
X_test_final = test_ordered[FEATURE_COLS]

predictions = final.predict(X_test_final)

submission = sample[["Index"]].copy()
submission["target"] = predictions

# Validate
assert list(submission.columns) == ["Index", "target"]
assert len(submission) == len(sample)
assert submission["Index"].equals(sample["Index"])
assert submission["Index"].is_unique
assert np.isfinite(submission["target"].to_numpy()).all()
print("Submission validation passed.")

out = PROJECT_ROOT / "submissions" / "03_histgb.csv"
out.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(out, index=False)

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
ridge_rmse = 10.445668
rmse_series = cv_report.metrics.rmse(aggregate=None)
fold_rmses = rmse_series.iloc[0].tolist()
mean_rmse = float(sum(fold_rmses) / len(fold_rmses))
std_rmse = float(pd.Series(fold_rmses).std())
pct_improvement = (ridge_rmse - mean_rmse) / ridge_rmse * 100

print("\n===================================")
print("EXPERIMENT 03 - HISTGRADIENTBOOSTING")
print("===================================")
print(f"Ridge grouped CV RMSE (ref): {ridge_rmse}")
print(f"HGBR per-fold RMSE         : {[round(v, 4) for v in fold_rmses]}")
print(f"HGBR mean RMSE             : {mean_rmse:.6f}")
print(f"HGBR std  RMSE             : {std_rmse:.6f}")
print(f"Improvement over Ridge     : {pct_improvement:.2f}%")
print(f"\nSubmission saved : {out}")
print("Skore Hub key    : 03_histgb")
print("===================================")
