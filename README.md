<p align="center">
  <img src="docs/figures/probabl-logo.svg" alt="Probabl" height="75">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/figures/ibm-logo.svg" alt="IBM" height="60">
</p>

<h1 align="center">Parkinson's True OFF Score Prediction</h1>
<p align="center"><strong>ESILV × IBM × Probabl — Bobathon Hackathon 2026</strong></p>

<p align="center">
  <a href="https://www.kaggle.com/competitions/bobathon-esilv/">Kaggle competition</a> ·
  <a href="https://skore.probabl.ai/philippeEtchebest/bobathon-esilv/estimators/63086">Best Skore EstimatorReport</a> ·
  <a href="journal/JOURNAL.md">Experiment journal</a> ·
  <a href="docs/CONTEXT.md">Scientific context</a>
</p>

> **Best recorded result: 4.1896 RMSE** — five-fold **patient-grouped cross-validation** with tuned LightGBM (Experiment 09). This is a local validation score, **not** a Kaggle public/private leaderboard score.

## Overview

The objective is to predict a patient's **debiased true OFF motor score** for every Parkinson's disease visit. The challenge uses a **synthetic, longitudinal, multi-cohort** dataset designed to resemble clinical observations. The target is related to the MDS-UPDRS motor examination score, but observed ON/OFF scores can be distorted by medication state, time since levodopa intake, missing examinations, and differences between raters.

Instead of treating visits as unrelated rows, our solution builds a longitudinal representation of each patient from the **available input observations**, then uses medication-aware features and a regularized **LightGBM regressor** to estimate the underlying OFF score.

This is an **academic machine-learning prototype**, not a validated clinical decision-support system.

### At a glance

| Item | Details |
| --- | --- |
| Task | Supervised regression: predict `target` (true/debiased OFF motor score) |
| Evaluation metric | Root mean squared error (RMSE; lower is better) |
| Training data | **44,590 visits**, **5,576 patients** |
| Hidden-label test inputs | **11,013 visits**, **1,395 different patients** |
| Validation protocol | **5-fold `GroupKFold` by `patient_id`** |
| Best model | Experiment **09 — tuned LightGBM** |
| Best grouped CV | **4.1896 RMSE** (fold mean) |
| Hub workspace / project | `philippeEtchebest` / `bobathon-esilv` |
| Submission artifact | `submissions/09_tuned_lgbm.csv` (generated locally, not committed) |

## Contents

- [Challenge and dataset](#challenge-and-dataset)
- [Approach and feature engineering](#approach-and-feature-engineering)
- [Experiments and results](#experiments-and-results)
- [Validation and leakage prevention](#validation-and-leakage-prevention)
- [Repository structure](#repository-structure)
- [Setup and reproducibility](#setup-and-reproducibility)
- [Kaggle submission](#kaggle-submission)
- [Interpretation and limitations](#interpretation-and-limitations)
- [Academic deliverables and references](#academic-deliverables-and-references)

## Challenge and dataset

Parkinson's motor symptoms are measured under changing medication conditions. An **ON** observation is taken while medication has an effect; an **OFF** observation is intended to better reflect untreated severity, but can be affected by incomplete medication washout, rater subjectivity, and missingness. The challenge asks us to estimate a *debiased true OFF* score from these imperfect measurements.

The challenge data are **synthetic**. They do not contain a real-world ground-truth clinical estimate of neurodegeneration. See [`docs/CONTEXT.md`](docs/CONTEXT.md) for the scientific assumptions and synthetic data-generation description.

### Input columns

| Category | Columns | Purpose |
| --- | --- | --- |
| Row / patient identifiers | `Index`, `patient_id` | Join/output key and patient grouping; not passed to the regressor as raw identifiers |
| Demographics / history | `sexM`, `age`, `age_at_diagnosis` | Patient and disease-stage characteristics |
| Cohort / genetics | `cohort`, `gene` | Contextual and categorical information |
| Medication | `ledd`, `time_since_intake_on`, `time_since_intake_off` | Treatment exposure and measurement timing |
| Observed measurements | `on`, `off` | Biased/noisy motor-score observations |
| Assessment context | `rater_id` | Clinician/rater category |
| Label (training only) | `target` | Debiased true OFF score to predict |

A single patient can have several visits. The training and Kaggle test populations have **no patient overlap**, so a random visit-level split would produce misleadingly optimistic evaluation.

Missingness is important: in the recorded EDA, approximately **42.4%** of visits lack an OFF measurement, **29.6%** lack ON, and **78.8%** lack `time_since_intake_off`. ON and OFF are never both missing in the same visit. The pipeline retains missing-value information instead of blindly replacing all missing scores with a global average.

Detailed exploration: [`data/eda.md`](data/eda.md) · [training profile](data/eda_train.html) · [test profile](data/eda_test.html).

## Approach and feature engineering

Our final recorded approach consists of three ordered stages:

```text
Input visits
     |
     v
LongitudinalAggregator
  - within-patient ON/OFF/LEDD statistics
  - age-based progression slopes and intercepts
  - first/last observation, visit count and age span
     |
     v
MedicationFeatureAdder
  - predicted ON/OFF from each patient's observed trend
  - current-visit deviations from those trends
  - ON/OFF gap, dose and intake-time interactions
     |
     v
Tuned LightGBMRegressor
     |
     v
Predicted debiased true OFF score
```

### 1. Basic visit-level features

- **Disease duration:** `age - age_at_diagnosis`.
- **Observed ON/OFF difference:** `off - on`, when both are available.
- **Explicit missingness indicators** for ON, OFF and LEDD.
- Raw numeric columns with native missing-value handling in LightGBM.
- `cohort`, `gene` and `rater_id` as pandas categorical columns.

### 2. Longitudinal patient representation (Experiment 07)

We estimate context from each patient's **input measurements**, including:

- ON/OFF mean, median, standard deviation, minimum and maximum.
- First and last observed ON/OFF measurement in age order.
- ON/OFF linear progression slope and intercept versus age, when estimable.
- Counts of valid ON/OFF readings, missing fractions, and LEDD summary statistics.
- Minimum and maximum observed age, patient age span, total visit count, and current visit's relative position in that span.

The patient-level transformer computes statistics from the **feature rows being transformed**. For a held-out patient, it uses that patient's available unlabelled visits; it never looks up that patient's `target`. When a patient has too little data to estimate a statistic, the implementation uses missing values and/or fallbacks derived from the training partition.

### 3. Medication and observation-bias features (Experiment 08)

These features capture the difference between a noisy current measurement and the patient's broader observed trajectory:

- Trend-predicted ON and OFF score at the current age.
- `on_dev_from_trend` and `off_dev_from_trend`.
- Observed ON/OFF gap and patient-mean gap.
- LEDD × ON/OFF gap interaction and ON/OFF ratio.
- Time-since-intake × observed-score interactions.

They are **data-driven features**, not a complete pharmacokinetic or causal medication model. Their individual contributions require controlled ablation to establish.

### 4. Tuned LightGBM (Experiment 09)

Experiment 09 evaluated several candidate configurations in a two-fold pilot and compared shortlisted configurations using all five patient-grouped folds. The selected `strong_reg` configuration uses:

| Parameter | Value |
| --- | ---: |
| `n_estimators` | 2,000 |
| `learning_rate` | 0.03 |
| `num_leaves` | 63 |
| `min_child_samples` | 20 |
| `subsample` / `subsample_freq` | 0.8 / 1 |
| `colsample_bytree` | 0.7 |
| `reg_alpha` / `reg_lambda` | 0.5 / 3.0 |
| `random_state` | 42 |

For the full candidate definitions, evaluation, final refit and CSV generation, see [`experiments/09_tuned_lgbm.py`](experiments/09_tuned_lgbm.py).

## Experiments and results

All values below are **recorded local validation results**, not Kaggle leaderboard scores.

| Experiment | Main change | Grouped CV RMSE | Outcome |
| --- | --- | ---: | --- |
| [01 — Dummy](experiments/01_dummy.py) | Mean prediction baseline | 16.499 | Reference |
| [02 — Ridge](experiments/02_ridge.py) | Numerical/categorical preprocessing and linear regression | 10.446 | Improvement |
| [03 — HistGradientBoosting](experiments/03_histgb.py) | Nonlinear model with native numeric missingness | 7.484 | Improvement |
| [04 — Patient features](experiments/04_patient_features.py) | Basic patient-level aggregates | 5.346 | Improvement |
| [05 — LightGBM](experiments/05_lightgbm.py) | LightGBM with patient features | 5.293 | Improvement |
| 06 — CatBoost | Incomplete experiment; computationally expensive | — | Abandoned |
| [07 — Longitudinal](experiments/07_longitudinal.py) | Patient trajectories and richer aggregates | **4.3474** | Improvement |
| [08 — Medication](experiments/08_medication.py) | Trend deviations and medication features | **4.2101** | Improvement |
| **[09 — Tuned LightGBM](experiments/09_tuned_lgbm.py)** | Stronger regularization and tuned boosting | **4.1896** | **Best recorded** |

The reported reduction from Experiment 05 (5.2925) to Experiment 09 (4.1896) is approximately **20.8% in RMSE**.

### Experiment 09: per-fold results

| Fold | RMSE |
| --- | ---: |
| 1 | 4.1865 |
| 2 | 4.2756 |
| 3 | 4.1527 |
| 4 | 4.1274 |
| 5 | 4.2055 |
| **Mean** | **4.1896** |
| **Reported standard deviation** | **0.057** |

### Published Skore reports

| Experiment | EstimatorReport |
| --- | --- |
| 07 — Longitudinal | [Hub report 62869](https://skore.probabl.ai/philippeEtchebest/bobathon-esilv/estimators/62869) |
| 08 — Medication | [Hub report 62868](https://skore.probabl.ai/philippeEtchebest/bobathon-esilv/estimators/62868) |
| **09 — Tuned LightGBM** | **[Hub report 63086](https://skore.probabl.ai/philippeEtchebest/bobathon-esilv/estimators/63086)** |

These published estimator reports describe **held-out fold models**; each submission script separately refits a model on all training visits before predicting the Kaggle test set. The correspondence between a held-out report and the full-data refit should be accepted by the competition organizers before submission.

Further modelling decisions, investigations and the invalid in-sample trend diagnostic are documented in [`journal/JOURNAL.md`](journal/JOURNAL.md).

## Validation and leakage prevention

The Kaggle test set contains entirely **unseen patients**, not just unseen rows. We therefore use five-fold `GroupKFold` with `patient_id` as the grouping key:

1. Each patient belongs to exactly one validation fold.
2. No patient's visits appear in both training and validation partitions of that fold.
3. The same split construction is used across Experiments 05–09 for comparable scores.
4. Patient-level features are generated from **input columns only**, including other available unlabelled visits for a held-out patient; no target-based group aggregation is performed.
5. `Index` is used only for merges and output alignment, and raw `patient_id` is used for grouping/feature creation rather than direct memorization by LightGBM.
6. The final prediction CSV is aligned with the provided submission template and validated for duplicate indices, missing or nonfinite predictions.

**Important distinction:** the exploratory `1.61 RMSE` patient trend diagnostic described in the journal was *in-sample* (the patient's own target values were used to fit and score the same trend). It is **not an eligible generalization estimate** and is excluded from comparisons.

The reported model-selection CV is useful evidence, but repeated experimental choices on the same folds can introduce some optimism. An independent Kaggle score has not been established in this README.

## Repository structure

```text
.
├── .bob/rules/AGENTS.md           # Mandatory Bob setup gate
├── data/
│   ├── X_train.csv                # Training input visits
│   ├── y_train.csv                # Training targets
│   ├── X_test.csv                 # Unlabelled test visits
│   ├── sample_submission.csv      # Required Index,target template
│   ├── eda.py / eda.md            # Reproducible exploratory analysis
│   └── eda_train.html / eda_test.html
├── docs/
│   ├── SETUP.md                   # Official setup steps
│   ├── DAY1.md / DAY2.md          # Organizer lab guides
│   ├── CONTEXT.md                 # Scientific / synthetic-data context
│   └── figures/
├── experiments/
│   ├── 01_dummy.py ... 05_lightgbm.py
│   ├── 07_longitudinal.py
│   ├── 08_medication.py
│   └── 09_tuned_lgbm.py           # Best recorded candidate
├── journal/JOURNAL.md             # Experiment history and decisions
├── scripts/skore-agent            # Configure Skore Hub credentials locally
├── tests/test_setup.py            # Seven setup checks
├── requirements.txt
└── README.md
```

The `submissions/` and `scratch/` directories are excluded from Git by design. Submission CSVs are **generated artifacts**, so the expected Kaggle CSV may exist on a teammate's machine without appearing on GitHub. Keep a separate local backup of the best CSV.

## Setup and reproducibility

> **Eligibility:** Official hackathon code must be written and run through **IBM Bob (IDE or CLI)**, following the organizer's lab-skill requirements, and model evaluations/reports must use **skore**. Consult [`docs/SETUP.md`](docs/SETUP.md) and the [competition rules](https://www.kaggle.com/competitions/bobathon-esilv/rules) for the authoritative conditions.

### Prerequisites

- Access to this private repository and the competition data.
- Python **3.12 or newer**.
- IBM Bob, with the lab skills installed and **Skill** permission enabled.
- Access to the correct team workspace at [Skore Hub](https://skore.probabl.ai).

### Windows / PowerShell

From a terminal in IBM Bob, using a Python 3.12+ interpreter:

```powershell
git clone https://github.com/cyrilibr/esilv-parkinsons-hackathon.git
cd esilv-parkinsons-hackathon

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install numpy pandas scikit-learn lightgbm
```

**Dependency reproducibility note:** `requirements.txt` currently pins the organizer setup tool (`skore-cli==0.4.1`) and includes `pytest`, but it does **not yet pin** the data-science package versions used for the published LightGBM result. The extra installation command provides the missing direct dependencies; for bit-for-bit reproducibility, record and pin the versions from the tested environment before archiving the final submission.

Follow [`docs/SETUP.md`](docs/SETUP.md) to install the Bob lab skills and log in. To create the local `.skore` configuration:

```powershell
.\.venv\Scripts\python.exe scripts/skore-agent
```

If an existing `.skore` points to a different workspace, the repository's helper supports:

```powershell
.\.venv\Scripts\python.exe scripts/skore-agent --switch-workspace
```

The local configuration contains **API credentials** and must never be committed, copied into the README or included in the final ZIP.

Run the mandatory setup checks inside the configured IBM Bob workspace:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_setup.py
```

These tests require the actual Bob environment, its installed lab skills, the latest chat's permission settings, and a working Skore Hub connection; they are not ordinary offline model unit tests.

### Run the best experiment

```powershell
.\.venv\Scripts\python.exe -u .\experiments\09_tuned_lgbm.py
```

This script:
1. Loads and aligns the training/test tables.
2. Runs a **two-fold hyperparameter pilot**.
3. Evaluates the two shortlisted configurations with **five-fold patient-grouped CV** through skore.
4. Publishes a held-out fold-0 EstimatorReport to the workspace selected in local `.skore`.
5. Refits the selected model on all labelled visits.
6. Writes `submissions/09_tuned_lgbm.csv` in sample-submission order.

**Expected runtime:** potentially several minutes or longer; it retrains multiple models, not just a single LightGBM estimator. If you already have a validated CSV, a rerun is unnecessary merely to locate that file. Check the unique-report-key requirements before republishing or making another Kaggle submission.

### Verify a generated CSV

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd, numpy as np; s=pd.read_csv('submissions/09_tuned_lgbm.csv'); t=pd.read_csv('data/sample_submission.csv'); assert s.columns.tolist()==['Index','target']; assert len(s)==len(t)==11013; assert s['Index'].equals(t['Index']); assert s['Index'].is_unique; assert np.isfinite(s['target'].to_numpy()).all(); print('CSV VALID')"
```

## Kaggle submission

Competition: **[bobathon-esilv](https://www.kaggle.com/competitions/bobathon-esilv/)**.

Before submitting:

1. **Confirm team identity:** the Kaggle team and the Skore Hub workspace must match according to the organizer's naming requirements. The Hub workspace used for these reports is `philippeEtchebest`; verify the actual Kaggle team name independently.
2. **Confirm the authorized upload window** on Kaggle and any class instructions. Do not upload outside the competition's permitted period.
3. Verify the final `Index,target` CSV has all **11,013** test visits in the sample's order.
4. Open the [Experiment 09 EstimatorReport](https://skore.probabl.ai/philippeEtchebest/bobathon-esilv/estimators/63086) and check that it meets the organizer's report-to-submission correspondence requirement.
5. Upload `submissions/09_tuned_lgbm.csv` using **Submit Predictions**.
6. In the **Submission Description**, include the **full associated Skore Hub EstimatorReport URL**. For example:

   ```text
   Experiment 09 — tuned LightGBM; 5-fold patient-grouped CV RMSE 4.1896.
   Skore EstimatorReport: https://skore.probabl.ai/philippeEtchebest/bobathon-esilv/estimators/63086
   ```

7. Confirm the submission has been accepted and scored on Kaggle, and select the intended team submission if required by the competition.

A **Skore EstimatorReport URL is mandatory** even if Kaggle produces a leaderboard score. Do not confuse a local CV result with a Kaggle public or final private result.

## Interpretation and limitations

- **Synthetic dataset:** results are specific to this competition and must not be interpreted as real-world clinical performance.
- **Out-of-patient generalization:** `GroupKFold` mimics the Kaggle patient holdout; folds are not a substitute for an independent clinical validation study.
- **Missingness:** ON/OFF/LEDD and timing records are incomplete; longitudinal features have limited reliability for sparse patient histories.
- **Model selection:** choosing feature sets and hyperparameters using the same CV folds may bias the reported estimate slightly downward.
- **Trend fitting:** patient ON/OFF trends use observed **feature measurements**, not true OFF labels; the separately documented target-fitted trend diagnostic is invalid for generalization.
- **Medication modelling:** engineered features capture correlations and interactions, not proven causal drug effects.
- **Remaining work:** a clean dependency lock, a final Kaggle acceptance check, and any video/archive hand-in required by DVL should be completed before final delivery.

## Academic deliverables and references

### Final hand-in checklist

- [x] EDA and dataset understanding documented.
- [x] Multiple baselines and successive modelling improvements recorded.
- [x] Best model implementation and five-fold patient-grouped validation saved.
- [x] Experiment 09 Skore EstimatorReport URL documented.
- [ ] Confirm Kaggle team and Skore workspace names correspond.
- [ ] Validate and upload the best rule-compliant prediction CSV on Kaggle.
- [ ] Confirm accepted submission and final leaderboard/result information.
- [ ] Pin exact tested package versions for the final reproducible archive.
- [ ] Record team membership/contributions and prepare the presentation video.
- [ ] Confirm and complete any required code archive, video and DVL uploads by the official deadlines.

The final deliverable deadlines and video instructions must be checked against the **actual DVL course assignment**; they cannot be verified from this repository alone.

### Source documents

- [Original Probabl Bobathon starter repository](https://github.com/probabl-ai/bobathon-esilv)
- [Organizer setup guide](docs/SETUP.md)
- [Day 1: EDA and dummy baseline](docs/DAY1.md)
- [Day 2: grouped validation, pipelines and submissions](docs/DAY2.md)
- [Scientific challenge context](docs/CONTEXT.md)
- [Exploratory analysis](data/eda.md)
- [Experimental record](journal/JOURNAL.md)
- [Kaggle competition and rules](https://www.kaggle.com/competitions/bobathon-esilv/rules)

---

*Built as an ESILV × IBM × Probabl Bobathon academic challenge project. Clinical background, evaluation guidelines and starter materials originate from the competition organizers; the experimental results in this README are those documented in this repository.*
