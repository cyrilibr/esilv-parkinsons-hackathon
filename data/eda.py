# %% [markdown]
# # EDA: Parkinson's Disease OFF-Score Prediction
#
# Exploratory data analysis of the Parkinson's competition dataset,
# run before designing a model.
#
# - **Raw data** is read-only — `X_train.csv`, `y_train.csv`, and
#   `X_test.csv` in `data/`. This file never cleans or modifies them.
# - **Outputs** go under `EDA_DIR` (the repo's `data/`): an
#   `eda_<table>.html` report per table, summarised in `eda.md`.

# %%
import json
from pathlib import Path

import pandas as pd
import skrub

# This file lives in data/, so parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# EDA outputs always land here (created if missing); the raw data may
# live elsewhere.
EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# Three tables: features for training, targets, and test features.
# Merge X_train with y_train on Index to get the full labelled visits.

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

TRAIN = X_train.merge(y_train, on="Index")

{  # noqa: B018
    "X_train": X_train.shape,
    "y_train": y_train.shape,
    "X_test": X_test.shape,
    "TRAIN (merged)": TRAIN.shape,
}

# %% [markdown]
# ## Patient / visit counts
#
# Each row is a visit. Patients appear multiple times.

# %%
train_patient_stats = {
    "train_visits": len(TRAIN),
    "train_patients": TRAIN["patient_id"].nunique(),
    "test_visits": len(X_test),
    "test_patients": X_test["patient_id"].nunique(),
    "train_visits_per_patient_mean": round(
        len(TRAIN) / TRAIN["patient_id"].nunique(), 2
    ),
    "train_visits_per_patient_max": int(TRAIN["patient_id"].value_counts().max()),
    "train_visits_per_patient_median": float(
        TRAIN["patient_id"].value_counts().median()
    ),
    "patient_overlap_train_test": int(
        TRAIN["patient_id"].isin(X_test["patient_id"]).sum()
    ),
}
train_patient_stats  # noqa: B018

# %% [markdown]
# ## Table overview: training set (X_train + y_train merged)
#
# Full column profile: dtype, missingness, cardinality.

# %%
report_train = skrub.TableReport(TRAIN, title="train", verbose=0)
report_train.write_html(EDA_DIR / "eda_train.html")

summary_train = json.loads(report_train.json())
n_rows_train = summary_train.get("n_rows")
overview_train = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary_train.get("columns", [])
]
{"n_rows": n_rows_train, "n_columns": len(overview_train), "columns": overview_train}  # noqa: B018

# %% [markdown]
# ## Table overview: test set (X_test)

# %%
report_test = skrub.TableReport(X_test, title="test", verbose=0)
report_test.write_html(EDA_DIR / "eda_test.html")

summary_test = json.loads(report_test.json())
n_rows_test = summary_test.get("n_rows")
overview_test = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary_test.get("columns", [])
]
{"n_rows": n_rows_test, "n_columns": len(overview_test), "columns": overview_test}  # noqa: B018

# %% [markdown]
# ## Target distribution
#
# Distribution of the true OFF score (`target`) in the training set.

# %%
TARGET = "target"
target_col = next(
    (col for col in summary_train.get("columns", []) if col.get("name") == TARGET),
    None,
)
target_col  # noqa: B018

# %% [markdown]
# ## Missingness deep-dive
#
# Which columns are missing and how often? Are `on`/`off` sometimes
# both missing in the same visit?

# %%
missing_summary = sorted(
    [
        {"column": col["column"], "null_pct": col["null_pct"]}
        for col in overview_train
        if col["null_pct"] is not None and col["null_pct"] > 0
    ],
    key=lambda r: r["null_pct"],
    reverse=True,
)

# Count visits where on is missing, off is missing, or both are missing
on_null = TRAIN["on"].isna()
off_null = TRAIN["off"].isna()
missingness_counts = {
    "on_missing": int(on_null.sum()),
    "on_missing_pct": round(float(on_null.mean()) * 100, 2),
    "off_missing": int(off_null.sum()),
    "off_missing_pct": round(float(off_null.mean()) * 100, 2),
    "both_missing": int((on_null & off_null).sum()),
    "both_missing_pct": round(float((on_null & off_null).mean()) * 100, 2),
}
{"top_missing": missing_summary[:10], "on_off_missingness": missingness_counts}  # noqa: B018

# %% [markdown]
# ## LEDD missingness by cohort
#
# Is `ledd` missing at random, or concentrated in specific cohorts?

# %%
ledd_by_cohort = (
    TRAIN.groupby("cohort")["ledd"]
    .apply(lambda s: round(float(s.isna().mean()) * 100, 2))
    .reset_index()
    .rename(columns={"ledd": "ledd_null_pct"})
    .sort_values("ledd_null_pct", ascending=False)
    .to_dict(orient="records")
)
ledd_by_cohort  # noqa: B018

# %% [markdown]
# ## Structure signals
#
# Datetime columns (time-based splits) and high-cardinality ID/group
# columns (grouped splits).

# %%
datetime_cols = [
    col.get("name")
    for col in summary_train.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (col.get("n_unique") or 0) / n_rows_train
            if n_rows_train
            else None,
        }
        for col in summary_train.get("columns", [])
    ),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
{"datetime_cols": datetime_cols, "top_unique_ratio": unique_ratio[:10]}  # noqa: B018

# %% [markdown]
# ## Associations (training set with target)
#
# Strongest pairwise column associations. `off` and `on` vs `target`
# are the key relationships; near-perfect association flags potential
# leakage.

# %%
assoc = skrub.column_associations(TRAIN)
rows = (
    assoc.to_dicts()
    if hasattr(assoc, "to_dicts")
    else assoc.to_dict(orient="records")
)
target_links = [
    row
    for row in rows
    if row["left_column_name"] == TARGET or row["right_column_name"] == TARGET
]
{"with_target": target_links[:15], "strongest": rows[:10]}  # noqa: B018

# %% [markdown]
# ## Summary
#
# Findings and modelling implications are written up in `data/eda.md`.
