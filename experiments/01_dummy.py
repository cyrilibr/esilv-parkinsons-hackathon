# %% [markdown]
# # 01 — Dummy mean baseline
#
# DummyRegressor(strategy="mean") predicts the training mean for every visit.
# This sets the floor: any real model must beat this RMSE.

# %%
import json
from pathlib import Path

import pandas as pd
import skore
from sklearn.dummy import DummyRegressor
from skore import Project, login

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# %% Load data and align by Index
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test  = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

visits = X_train.merge(y_train, on="Index")

feature_cols = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]
X = visits[feature_cols]
y = visits["target"]

# %% Evaluate with skore (random 80/20 holdout)
dummy = DummyRegressor(strategy="mean")
report = skore.evaluate(dummy, X, y)
report.metrics.rmse()

# %% Push to Hub
cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy", report)

# %% Save submission template (constant prediction = training mean)
sample = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")
dummy.fit(X, y)
sample["target"] = dummy.predict(X_test[feature_cols])
out = PROJECT_ROOT / "submissions" / "01_dummy.csv"
sample.to_csv(out, index=False)
{"submission_rows": len(sample), "constant_prediction": round(float(dummy.constant_[0]), 4)}
