# EDA: Parkinson's Disease OFF-Score Prediction

Exploratory data analysis of `X_train.csv`, `y_train.csv`, and `X_test.csv`
from the competition dataset. Run before designing a model.

Interactive reports (open in a browser):

- [data/eda_train.html](eda_train.html) — full training set profile
- [data/eda_test.html](eda_test.html) — test set profile

---

## Dataset at a glance

| Split | Rows (visits) | Columns | Patients |
|---|---|---|---|
| Training (X_train + y_train) | 44 590 | 14 (13 features + target) | 5 576 |
| Test (X_test) | 11 013 | 13 (features only) | 1 395 |

**Target column:** `target` — the debiased true OFF MDS-UPDRS motor score.

The dataset is multi-visit longitudinal: each patient appears multiple times.
On average 8.0 visits per patient in train (median 7, max 12). Train and test
patients do **not** overlap (`patient_overlap_train_test = 0`), so Kaggle uses
a strict patient-level holdout.

---

## Per-column findings

| Column | Dtype | Null % | N unique | Notes |
|---|---|---|---|---|
| `Index` | Int64 | 0 % | 44 590 | Unique row ID — join key, never a feature |
| `patient_id` | String | 0 % | 5 576 | Group key for grouped CV |
| `cohort` | String | 0 % | 2 | Values A / B |
| `sexM` | Int64 | 0 % | 2 | Binary (0 / 1) |
| `gene` | String | **32.4 %** | 4 | Genetic variant; substantial missingness |
| `age_at_diagnosis` | Float64 | **5.2 %** | 563 | |
| `age` | Float64 | 0 % | 1 082 | Age at visit — continuous |
| `ledd` | Float64 | **36.6 %** | 1 320 | Levodopa equivalent daily dose |
| `time_since_intake_on` | Float64 | **46.4 %** | 64 | Hours since ON-state levodopa intake |
| `time_since_intake_off` | Float64 | **78.8 %** | 178 | Hours since OFF-state test — most missing |
| `rater_id` | String | 0 % | 50 | Clinic rater — potential confounder |
| `on` | Float64 | **29.6 %** | 83 | Measured ON motor score |
| `off` | Float64 | **42.4 %** | 101 | Measured OFF motor score |
| `target` | Float64 | 0 % | 867 | Debiased true OFF score — **the label** |

Seven of the thirteen feature columns have missing values.
`time_since_intake_off` is missing for 78.8 % of visits — the most absent
column in the dataset. Test missingness rates mirror train closely.

---

## Target distribution

The `target` (true OFF score) ranges from **0.0 to 109.5** on the training
set — well within the 0–132 theoretical MDS-UPDRS range, with no visits
reaching the ceiling.

| Statistic | Value |
|---|---|
| Mean | 37.5 |
| Median (p50) | 37.3 |
| Std dev | 16.5 |
| IQR (p25–p75) | 25.6 – 49.3 |
| Min / Max | 0.0 / 109.5 |

The distribution is **broadly bell-shaped and roughly symmetric** around the
mean (mean ≈ median). It is slightly right-skewed at the tail, with counts
falling off sharply above 65 (≈ 4.8 % of visits above 65, < 0.05 % above 87).
This is a regression task with a continuous, approximately normal target.

---

## Missingness deep-dive

Ranked by missing fraction (training set):

| Column | Null % | Clinical meaning |
|---|---|---|
| `time_since_intake_off` | 78.8 % | OFF tests are uncomfortable; patients frequently skip them |
| `time_since_intake_on` | 46.4 % | ON-state timing not always recorded |
| `off` | 42.4 % | OFF motor score not taken at every visit |
| `ledd` | 36.6 % | Dose not recorded for ~37 % of visits |
| `gene` | 32.4 % | Genetic test not performed for all patients |
| `on` | 29.6 % | ON motor score also frequently absent |
| `age_at_diagnosis` | 5.2 % | Generally known; minor gaps |

**Critical finding:** `on` and `off` are **never both missing in the same visit**
(0 co-missing visits). Every visit has at least one motor score. This means
missingness is not random dropout but reflects which medication-state test was
actually performed — a strong structural signal that the model can exploit.

### LEDD missingness by cohort

| Cohort | LEDD null % |
|---|---|
| B | 42.6 % |
| A | 35.9 % |

Both cohorts have substantial LEDD missingness, but cohort B is ~7 pp worse.
This suggests the two cohorts were collected under different protocols (one
cohort less systematically recording medication dose). **LEDD missingness is
non-random and encodes cohort-specific data-collection practices**, which is
itself a signal.

---

## Structure: groups and time

**No datetime columns** were found by skrub's type inference — there is no
timestamp column in the data.

**Group structure** is explicit via `patient_id` (5 576 distinct values in
train, unique ratio 12.5 %). Multiple visits per patient means rows are
**not independent**. The Kaggle holdout is strictly by patient; train and test
patients do not overlap.

→ Cross-validation **must** use `GroupKFold` (or `GroupShuffleSplit`) with
`groups = patient_id` to avoid patient data leaking across folds. A standard
random `KFold` would be over-optimistic.

`rater_id` (50 raters) is another grouping variable worth noting: the rater
introduces systematic scoring bias that the task explicitly aims to debias.

---

## Associations with target

From `skrub.column_associations` (Cramér's V + Pearson correlation):

| Feature | Cramér's V | Pearson r |
|---|---|---|
| `off` | 0.41 | **0.871** |
| `on` | 0.37 | **0.669** |
| `ledd` | 0.24 | 0.298 |
| `age` | 0.11 | 0.310 |
| `time_since_intake_on` | 0.18 | ~0 (0.0002) |
| `time_since_intake_off` | 0.09 | ~0 (0.008) |
| `cohort` | 0.09 | — |
| `gene` | 0.05 | — |
| `age_at_diagnosis` | 0.05 | 0.133 |
| `patient_id` | 0.03 | — |
| `Index` | 0.03 | ~0 (-0.004) |
| `sexM` | 0.02 | ~0 (-0.001) |
| `rater_id` | 0.02 | — |

**Key finding:** `off` (Pearson r = 0.87) and `on` (r = 0.67) are strongly
correlated with `target`, but the correlation is **not perfect**. The gap
between `off` (measured, biased) and `target` (debiased) is the core signal
the competition asks you to learn. `off` is *not* a copy of `target`.

`on` and `off` are also strongly correlated with each other (r = 0.87), which
is expected — both reflect underlying disease severity.

**Strongest overall association:** `age_at_diagnosis` ↔ `age` (Pearson r = 0.94),
which makes sense — a patient diagnosed at 60 will be 70 at a visit 10 years
later. These are highly colinear; the derived `time_since_diagnosis = age −
age_at_diagnosis` is likely a better, cleaner feature.

**Timing columns:** `time_since_intake_on` and `time_since_intake_off` show
near-zero linear correlation with `target` individually (Pearson ≈ 0), yet
moderate Cramér's V (0.18 / 0.09). This suggests they interact non-linearly
with `off`/`on` (a patient measured far from dose timing will have a different
`off` score — timing modifies the `off` → `target` relationship).

---

## Potential data leakage

**`off` is the highest single-column predictor of `target` (r = 0.87) but it
is NOT a leakage risk** — the problem explicitly gives you the measured `off`
score as an input; the goal is to debias it. Using `off` is correct.

**`Index`** — unique row identifier, carries no predictive signal (Pearson ≈ 0).
**Must be excluded from model features.** It is only used as a join key.

**`patient_id`** — a string ID; near-zero association with `target` (Cramér's V
0.03). Exclude as a feature but keep as a grouping key for CV.

**`rater_id`** — 50 raters; very low Cramér's V (0.02). The rater introduces
a scoring bias the task aims to remove. Including it as a feature risks the
model memorising rater quirks from train raters; whether raters appear in test
should be verified. Treat cautiously.

**No column has an implausibly perfect association with `target`** — there is
no obvious direct leakage (e.g. a column that is a transformation of `target`).

---

## Modelling implications

| Finding | Implication |
|---|---|
| Patient-level holdout, multiple visits per patient | Use `GroupKFold(groups=patient_id)` — never standard KFold |
| No datetime column | `TimeSeriesSplit` is not needed; grouped CV is the right default |
| 7 columns with missing values up to 79 % | Pipeline must impute; `skrub.TableVectorizer` or `HistGradientBoosting` handles this natively |
| `on` and `off` never both missing | Missing indicators for each are informative features |
| LEDD missingness differs by cohort | `ledd` missingness is non-random; a missingness indicator column is warranted |
| `off` r = 0.87, `on` r = 0.67 with target | Strong predictors; include both despite high collinearity |
| `age` and `age_at_diagnosis` r = 0.94 | Use `age − age_at_diagnosis` as `time_since_diagnosis`; drop or keep one of the raw columns |
| Target approximately normal, continuous | RMSE is an appropriate metric; no stratification needed |
| `Index`, `patient_id` | Exclude from feature matrix; use as join key and group key respectively |
| `cohort`, `gene`, `rater_id` | Low association with target; include as categoricals — tree models will down-weight them automatically |

---

## Open questions

1. **Rater bleed-through:** Are the 50 raters in the test set a subset of the
   50 train raters, or new ones? If new, `rater_id` must not be used as a
   feature. Check `X_test["rater_id"].isin(X_train["rater_id"]).all()`.
2. **Cohort semantics:** Cohorts A and B differ in LEDD missingness and may
   differ in clinical protocol. Are there other systematic differences (e.g.
   different MDS-UPDRS subsets scored)?
3. **`time_since_diagnosis`:** DAY1.md notes this can be derived as
   `age − age_at_diagnosis`. Confirm that `X_test` also lacks this column
   (vs. DAY1.md hint that it may already be present in test).
4. **Target floor at 0:** A score of 0 means no motor symptoms — clinically
   unusual for a PD patient. Are these a data artefact or real?
