
# %% [markdown]
# # 02 - Ridge Regression with GroupKFold
#
# Parkinson's Disease - True OFF Score Prediction
#
# Compare Ridge regression against a dummy baseline.
# Use patient-grouped cross-validation to avoid data leakage.
# Publish an EstimatorReport to Skore Hub.
# Generate a Kaggle submission CSV.

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import skore

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from skore import Project, login


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# %% 1. Load datasets

print("Loading datasets...")

X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

# Align training features and labels by Index
visits = X_train.merge(
    y_train,
    on="Index",
    validate="one_to_one"
)

assert len(visits) == len(X_train), "Training rows do not match"

print("Training visits:", len(visits))
print("Test visits:", len(X_test))
print("Training patients:", visits["patient_id"].nunique())


# %% 2. Select features

NUMERIC_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

CAT_COLS = [
    "cohort",
    "gene",
    "rater_id",
]

FEATURE_COLS = NUMERIC_COLS + CAT_COLS

# Index and patient_id are excluded from predictors
X = visits[FEATURE_COLS]
y = visits["target"]

# patient_id is used only for cross-validation
groups = visits["patient_id"]


# %% 3. Build preprocessing pipeline

numeric_pipe = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

cat_pipe = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    ),
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, NUMERIC_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ]
)

ridge = make_pipeline(
    preprocessor,
    Ridge(alpha=1.0),
)

dummy = DummyRegressor(strategy="mean")


# %% 4. GroupKFold validation

print("\nPreparing patient-grouped cross-validation...")

cv_splits = list(
    GroupKFold(n_splits=5).split(
        X,
        y,
        groups=groups,
    )
)

# Verify no patient is shared between train and validation
for fold, (train_idx, valid_idx) in enumerate(cv_splits):
    train_patients = set(groups.iloc[train_idx])
    valid_patients = set(groups.iloc[valid_idx])

    assert train_patients.isdisjoint(valid_patients), (
        f"Patient leakage detected in fold {fold}"
    )

print("5 grouped folds created without patient leakage.")


# %% 5. Evaluate dummy baseline

print("\nEvaluating DummyRegressor...")

dummy_report = skore.evaluate(
    dummy,
    X,
    y,
    splitter=cv_splits,
)

print("\nDummy grouped CV RMSE:")
print(dummy_report.metrics.rmse())


# %% 6. Evaluate Ridge regression

print("\nEvaluating Ridge regression...")

ridge_cv_report = skore.evaluate(
    ridge,
    X,
    y,
    splitter=cv_splits,
)

print("\nRidge grouped CV RMSE:")
print(ridge_cv_report.metrics.rmse())

print("\nRidge RMSE per fold:")
print(
    ridge_cv_report.metrics.rmse(
        aggregate=None
    )
)


# %% 7. Create an EstimatorReport
#
# Skore Hub requires an EstimatorReport for Kaggle.
#
# Use fold 0 as a patient-grouped train/validation split.
# This report evaluates Ridge on patients not seen
# during that fold's training.
#
# The final submission model will be trained separately
# using ALL training visits.

print("\nCreating EstimatorReport...")

train_idx, valid_idx = cv_splits[0]

X_fold_train = X.iloc[train_idx]
y_fold_train = y.iloc[train_idx]

X_fold_valid = X.iloc[valid_idx]
y_fold_valid = y.iloc[valid_idx]

final_report = skore.EstimatorReport(
    ridge,
    X_train=X_fold_train,
    y_train=y_fold_train,
    X_test=X_fold_valid,
    y_test=y_fold_valid,
)

print("\nEstimatorReport created successfully.")

print("\nHeld-out fold RMSE:")
print(final_report.metrics.rmse())


# %% 8. Publish report to Skore Hub

print("\nConnecting to Skore Hub...")

cfg = json.loads(
    (PROJECT_ROOT / ".skore").read_text()
)

login(mode="hub")

project = Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

print("\nPublishing EstimatorReport: 02_ridge")

project.put(
    "02_ridge",
    final_report,
)

print("\nEstimatorReport published successfully.")


# %% 9. Train final model on all training visits

print("\nTraining final Ridge model on all visits...")

final_model = clone(ridge)

final_model.fit(X, y)

print("Final model fitted successfully.")


# %% 10. Prepare test data

sample = pd.read_csv(
    PROJECT_ROOT / "data" / "sample_submission.csv"
)

# Make sure Index values are unique
assert sample["Index"].is_unique
assert X_test["Index"].is_unique

# Ensure the template and test set contain the same visits
assert set(sample["Index"]) == set(X_test["Index"])

# Match test rows to the submission template
test_ordered = X_test.set_index("Index").loc[
    sample["Index"]
]

X_test_final = test_ordered[FEATURE_COLS]


# %% 11. Generate predictions

print("\nGenerating predictions...")

predictions = final_model.predict(X_test_final)

submission = sample[["Index"]].copy()
submission["target"] = predictions


# %% 12. Validate submission

print("\nValidating submission...")

assert list(submission.columns) == [
    "Index",
    "target",
], "Incorrect submission columns"

assert len(submission) == len(X_test), (
    "Incorrect number of predictions"
)

assert submission["Index"].equals(
    sample["Index"]
), "Index alignment error"

assert submission["Index"].is_unique, (
    "Duplicate Index values"
)

assert np.isfinite(
    submission["target"].to_numpy()
).all(), "Invalid predictions detected"

print("Submission validation passed.")


# %% 13. Save submission

submissions_dir = PROJECT_ROOT / "submissions"

submissions_dir.mkdir(
    parents=True,
    exist_ok=True,
)

output_path = submissions_dir / "02_ridge.csv"

submission.to_csv(
    output_path,
    index=False,
)


# %% 14. Final summary

print("\n===================================")
print("EXPERIMENT 02 - RIDGE REGRESSION")
print("===================================")

print("\nModel: Ridge(alpha=1.0)")
print("Validation: 5-fold GroupKFold")

print("\nTraining visits:", len(X))
print("Test predictions:", len(submission))

print("\nDummy grouped CV RMSE:")
print(dummy_report.metrics.rmse())

print("\nRidge grouped CV RMSE:")
print(ridge_cv_report.metrics.rmse())

print("\nPrediction statistics:")
print("Mean:", round(float(submission["target"].mean()), 4))
print("Std:", round(float(submission["target"].std()), 4))
print("Min:", round(float(submission["target"].min()), 4))
print("Max:", round(float(submission["target"].max()), 4))

print("\nSubmission saved:")
print(output_path)

print("\nSkore Hub report key: 02_ridge")
print("===================================")
