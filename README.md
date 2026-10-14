<p align="center">
  <img src="docs/figures/probabl-logo.svg" alt="Probabl" height="72">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/figures/ibm-logo.svg" alt="IBM" height="55">
</p>

<h1 align="center">Parkinson's True OFF Score Prediction</h1>
<p align="center"><strong>ESILV × IBM × Probabl · Bobathon 2026</strong></p>

<p align="center">
  <a href="https://www.kaggle.com/competitions/bobathon-esilv/">Kaggle challenge</a> ·
  <a href="journal/JOURNAL.md">Experiment journal</a> ·
  <a href="data/eda.md">Exploratory analysis</a> ·
  <a href="docs/CONTEXT.md">Scientific background</a>
</p>

> **Best result recorded in this repository: 3.6341 RMSE** — Experiment 18, three-stage stacking with patient-level smoothing, evaluated using five patient-grouped folds. This is a **teammate-reported local CV result**, not a verified Kaggle score; the nested stacking evaluation still merits an independent leakage audit.
>
> **Latest model with a report URL explicitly documented in the original team's Skore workspace:** Experiment 09, **4.1896 grouped CV RMSE**, [EstimatorReport 63086](https://skore.probabl.ai/philippeEtchebest/bobathon-esilv/estimators/63086).

## Project overview

The objective is to predict the **debiased (“true”) OFF MDS-UPDRS motor score** at each visit of a person with Parkinson's disease. Clinical ON/OFF measurements are imperfect proxies: medication effect, assessment timing, rater variability and missing examinations can obscure underlying motor severity.

We model visits **longitudinally** rather than as independent records. The project progresses from simple regression through patient-level feature aggregation and pharmacological correction to smoothing and cross-fitted stacked LightGBM models.

**Scope:** an academic experiment on a **synthetic dataset**, not a clinical device, diagnostic tool or validated real-patient outcome predictor.

| Project fact | Value |
| --- | --- |
| Prediction target | `target` — synthetic debiased true OFF score |
| Metric | Root mean squared error (RMSE; lower is better) |
| Training set | 44,590 visits from 5,576 patients |
| Test set | 11,013 visits from 1,395 **different** patients |
| Main validation | Five-fold `GroupKFold(groups=patient_id)` |
| Strongest reported experiment | `18_stack3` — 3.6341 RMSE |
| Better-documented fallback | `09_tuned_lgbm` — 4.1896 RMSE, associated Skore URL recorded |
| Kaggle leaderboard result | **Not documented in this repository** |

### Contents

- [Data and evaluation](#data-and-evaluation)
- [Modelling approach](#modelling-approach)
- [Experimental results](#experimental-results)
- [Skore reports and reproducibility status](#skore-reports-and-reproducibility-status)
- [Repository layout](#repository-layout)
- [Set up the project](#set-up-the-project)
- [Run the experiments](#run-the-experiments)
- [Prepare a Kaggle submission](#prepare-a-kaggle-submission)
- [Known limitations and next steps](#known-limitations-and-next-steps)
- [Team, credits and references](#team-credits-and-references)

## Data and evaluation

The organizers provide synthetic multi-cohort Parkinson's visits, with irregular intervals and multiple observations per patient. The intended prediction is a *debiased OFF* motor score rather than a directly observed clinic measurement. See [the organizer's context document](docs/CONTEXT.md) for the synthetic-generative assumptions.

| Input category | Columns |
| --- | --- |
| Identity and grouping | `Index`, `patient_id` |
| Demographics and disease stage | `sexM`, `age`, `age_at_diagnosis` |
| Cohort and genetics | `cohort`, `gene` |
| Clinical measurements | `on`, `off` |
| Medication and timing | `ledd`, `time_since_intake_on`, `time_since_intake_off` |
| Examiner | `rater_id` |
| Training label (absent from test inputs) | `target` |

The [EDA report](data/eda.md) documents missing values, patient histories and feature distributions. ON and OFF are not simultaneously absent from a visit, but either measurement can be missing. Missingness indicators and available patient history are therefore useful signals.

### Why the validation split matters

Kaggle tests on **entirely unseen patients**. A random row-level train/test split would leak patient identity through correlated visits, making the evaluation overly optimistic. Experiments therefore use a five-fold **patient-grouped** split; each patient's visits are assigned to exactly one held-out fold.

```python
from sklearn.model_selection import GroupKFold

splits = list(
    GroupKFold(n_splits=5).split(X, y, groups=patient_ids)
)
```

For each held-out patient, input-only aggregates may use that patient's **available unlabeled observations**, just as the competition test set provides multiple feature rows per unseen patient. **No held-out target values may be used** when constructing features, fitting medication corrections, or training stacking stages.

Metric:

\[
\operatorname{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

All model scores in this README are transcribed from the committed experiment journal or scripts. They are **not a claim of independently reproduced results**.

## Modelling approach

```text
Raw visits
   |
   +--> Visit-level features: diagnosis duration, medication, missingness
   |
   +--> Patient trajectories: ON/OFF statistics, slopes, age position
   |
   +--> Quadratic trajectories and rater-bias features
   |
   +--> Medication-aware Bateman corrections (trained within the appropriate folds)
   |
   v
LightGBM severity predictions
   |
   +--> Patient-level quadratic smoothing of predictions
   |
   +--> Stage-2 stacking using cross-fitted stage-1 predictions
   |
   +--> Stage-3 stacking using prediction-curve features (Exp 18)
   |
   v
Predicted debiased true OFF score
```

### Baselines and longitudinal features (Experiments 01–09)

We began with mean regression, Ridge, HistGradientBoosting and LightGBM. The largest early improvement came from representing a patient's history:

- Longitudinal ON/OFF means, medians, dispersion, extrema and valid-observation counts.
- First/last readings, age span, visit count and relative position within the observed trajectory.
- Linear ON/OFF slopes and fitted levels at the visit's age.
- Deviations of current ON/OFF readings from the fitted patient trends.
- Medication dose and intake-time interactions, categorical cohort/gene/rater variables, and missingness indicators.

Experiment 09 selected a more strongly regularized LightGBM configuration after a two-fold pilot and full five-fold comparison: 2,000 estimators, learning rate 0.03, 63 leaves, column subsampling 0.7, `reg_alpha=0.5` and `reg_lambda=3.0`. It reached **4.1896 grouped CV RMSE**.

### Quadratic features, rater effects and medication corrections (Experiments 10–14)

- **10 — Quadratic features:** estimates patient-specific curvature of ON/OFF measurements versus age, as well as quadratic predictions and visit-level residuals.
- **11 — Rater bias:** introduces rater-related bias features estimated using labelled training observations, with shrinkage. Fitting the correction on the correct training fold is essential.
- **12 — Bateman:** models an approximate medication response using a Bateman absorption/elimination curve and an OFF-decay correction; these learned parameters inform corrected ON/OFF features.
- **13 — Intervisit/LEDD experiment:** explored multiple modifications and performed worse in the journal. **The committed script currently has an incorrect duplicate report key**; do not publish it unchanged.
- **14 — Intervisit features:** adds within-patient consecutive-visit changes and velocities.

These corrections are modelling assumptions and engineered predictors, **not validated pharmacodynamic estimates**.

### Patient smoothing and stacking (Experiments 15–18)

**15 — Quadratic smoothing** fits an age-based curve to each patient's *predicted scores*, not to their target labels, reducing visit-level fluctuation. The recorded five-fold RMSE is **4.0256**.

**16 — Weighted/ridge smoothing** adjusts the smoothing weights and penalizes curvature; **4.0216** was the best recorded number, but the journal notes that the marginal improvement may not justify additional complexity.

**17 — Two-stage stacking** feeds patient-level curve features derived from cross-fitted stage-1 predictions into another LightGBM model, then smooths the output. It reports **3.7652** grouped CV RMSE.

**18 — Three-stage stacking** adds a third predictor that uses first- and second-stage prediction-curve features, followed by smoothing. It reports **3.6341** grouped CV RMSE (five folds). The script is computationally expensive and the construction of *nested* out-of-fold features needs additional review before this result should be treated as leakage-free.

## Experimental results

The following figures are **recorded mean five-fold RMSE values**, except where the journal explicitly notes an abandoned or pending run. Lower is better.

| # | Experiment / source | Main idea | Recorded RMSE |
| :-- | --- | --- | --: |
| 01 | [Dummy](experiments/01_dummy.py) | Mean baseline | 16.499 |
| 02 | [Ridge](experiments/02_ridge.py) | Linear regression with preprocessing | 10.446 |
| 03 | [HistGB](experiments/03_histgb.py) | Nonlinear regression | 7.484 |
| 04 | [Patient features](experiments/04_patient_features.py) | Patient aggregates | 5.346 |
| 05 | [LightGBM](experiments/05_lightgbm.py) | LightGBM with patient features | 5.293 |
| 06 | CatBoost | Computational cost | Abandoned |
| 07 | [Longitudinal](experiments/07_longitudinal.py) | Trajectory features | 4.3474 |
| 08 | [Medication](experiments/08_medication.py) | Trend deviations / timing interactions | 4.2101 |
| 09 | [Tuned LightGBM](experiments/09_tuned_lgbm.py) | Regularization tuning | 4.1896 |
| 10 | [Quadratic features](experiments/10_quadratic_features.py) | Curvature and fitted trajectories | 4.1714 |
| 11 | [Rater bias](experiments/11_rater_bias.py) | Rater correction features | 4.1644 |
| 12 | [Bateman](experiments/12_bateman.py) | Medication correction | 4.1310 |
| 13 | [Intervisit + LEDD](experiments/13_interviz_ledd.py) | Combined feature changes; worse | 4.221 |
| 14 | [Intervisit simple](experiments/14_interviz_simple.py) | Visit-to-visit differences | 4.095 |
| 15 | [Patient smoothing](experiments/15_patient_smoothing.py) | Quadratic smoothing of predictions | 4.0256 |
| 16 | [Smooth ridge](experiments/16_smooth_ridge.py) | Weighted, regularized smoothing | 4.0216 |
| 17 | [Two-stage stacking](experiments/17_stack_smooth.py) | Cross-fitted prediction curves | **3.7652** |
| **18** | **[Three-stage stacking](experiments/18_stack3.py)** | **Third stacked prediction stage** | **3.6341 (reported)** |

Experiment 19 (seed bagging) is described as *planned* in the journal, but **`experiments/19_seed_bagging.py` is not committed**; it is not a reproducible experiment in this repository.

### Fold-level evidence for the most recent models

| Fold | Exp 09 | Exp 17 | Exp 18 |
| :-- | --: | --: | --: |
| 1 | 4.1865 | 3.7713 | 3.6656 |
| 2 | 4.2756 | 3.8145 | 3.6458 |
| 3 | 4.1527 | 3.7281 | 3.6238 |
| 4 | 4.1274 | 3.7183 | 3.5969 |
| 5 | 4.2055 | 3.7937 | 3.6386 |
| **Mean** | **4.1896** | **3.7652** | **3.6341** |

The journal reports an **approximately 78% reduction in RMSE relative to the dummy baseline** from Experiment 01 to Experiment 18. The relative improvement is descriptive of the recorded validation scores, not proof of a similar improvement on Kaggle.

## Skore reports and reproducibility status

The project publishes reports using:

```python
cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
project = Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)
project.put(REPORT_KEY, estimator_report)
```

**A report URL is not interchangeable between experiments.** Every competition CSV requires an associated **Skore Hub EstimatorReport** and a new, matching report key.

### Report URLs recorded in the journal

| Experiment | Recorded report URL | Verification note |
| --- | --- | --- |
| 07 | [62869](https://skore.probabl.ai/philippeEtchebest/bobathon-esilv/estimators/62869) | Original workspace link |
| 08 | [62868](https://skore.probabl.ai/philippeEtchebest/bobathon-esilv/estimators/62868) | Original workspace link |
| 09 | [63086](https://skore.probabl.ai/philippeEtchebest/bobathon-esilv/estimators/63086) | Original workspace link |
| 10 | [69664](https://skore.probabl.ai/philippeechebest/bobathon-esilv/estimators/69664) | **Different workspace spelling in the journal; verify** |
| 11 | [69801](https://skore.probabl.ai/philippeechebest/bobathon-esilv/estimators/69801) | **Different workspace spelling in the journal; verify** |
| 12 | [70011](https://skore.probabl.ai/philippeechebest/bobathon-esilv/estimators/70011) | **Different workspace spelling in the journal; verify** |
| 13–18 | No verified URL in the committed journal | **Do not invent a URL; obtain it from the actual Skore project** |

The journal alternates between `philippeEtchebest` and `philippeechebest`. They are **not the same literal workspace name**. Verify your Kaggle team name, actual Skore workspace and permissions before using any report link for competition eligibility.

Publishing code exists for the later experiments; **the presence of `project.put(...)` in a script does not establish that publication succeeded**. A final submission CSV is also not available through GitHub because the `submissions/` directory is ignored.

## Repository layout

```text
.
├── .bob/rules/AGENTS.md          # Mandatory Bob setup checks
├── data/
│   ├── X_train.csv, y_train.csv   # Labelled training data
│   ├── X_test.csv                 # Unlabelled challenge inputs
│   ├── sample_submission.csv      # Index,target template
│   ├── eda.py, eda.md             # Exploratory analysis
│   └── eda_train.html, eda_test.html
├── docs/
│   ├── CONTEXT.md                # Challenge scientific context
│   ├── SETUP.md                  # Official Bob and Skore setup
│   └── DAY1.md, DAY2.md          # Organizer exercises
├── experiments/
│   ├── 00_diagnostic.py
│   ├── 01_dummy.py ... 05_lightgbm.py
│   ├── 07_longitudinal.py ... 11_rater_bias.py
│   ├── 12_bateman.py
│   ├── 12_bateman_finalize.py
│   └── 13_interviz_ledd.py ... 18_stack3.py
├── journal/JOURNAL.md            # Results, findings and limitations
├── scripts/skore-agent           # Local workspace authentication
├── tests/test_setup.py           # Environment / Hub setup checks
├── requirements.txt              # Organizer setup dependencies
└── README.md
```

**Not committed:** `.skore` (secret API credentials), `.venv/`, Bob skill installations, `scratch/` and `submissions/`. These paths are intentionally gitignored. The repository is currently **private**; do not share the challenge data or credentials without organizer approval.

## Set up the project

> **Competition workflow:** IBM Bob is the required coding agent, and model evaluation/reporting uses **skore**. Follow the organizer's [setup guide](docs/SETUP.md) and [Kaggle rules](https://www.kaggle.com/competitions/bobathon-esilv/rules). Documenting an experiment here is **not** a substitute for satisfying those rules.

### Prerequisites

- Git and Python **3.12+**.
- Access to this private repository and the hackathon data.
- **IBM Bob IDE or CLI**, with the organizer's skills installed and **Skill** permission enabled.
- A single Skore workspace shared by the Kaggle team, with the correct workspace name.

### Windows PowerShell (inside your IBM Bob project)

```powershell
git clone https://github.com/cyrilibr/esilv-parkinsons-hackathon.git
cd esilv-parkinsons-hackathon

python --version
python -m venv .venv

# If activation is blocked, consult docs/SETUP.md for PowerShell policy instructions.
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python -m pip install numpy pandas scipy scikit-learn lightgbm

skore skills install all --repo probabl-ai/skills-hackathon --agent bob
python scripts/skore-agent
python -m pytest tests/test_setup.py
```

The tracked `requirements.txt` lists `skore-cli==0.4.1` and `pytest` only. The experiment scripts also directly import NumPy, pandas, SciPy, scikit-learn and LightGBM; **their exact tested versions are not pinned**. The additional install command is a convenience, not a version-locked replication of the original environment. For a scientific archive, capture compatible versions from the environment actually used.

Already cloned the repository? From the project directory, use `git pull --ff-only origin main` after checking that local modifications are safely stored.

### Verify your own Skore workspace safely

```powershell
python -c "import json; print('Skore workspace:', json.load(open('.skore'))['workspace'])"
python -c "import numpy,pandas,scipy,sklearn,lightgbm,skore; print('Imports OK')"
python -m pytest tests/test_setup.py
```

To select the correct Hub workspace again, the local helper supports `python scripts/skore-agent --switch-workspace`. **Never print, send or commit the contents of `.skore`**; it includes credentials.

## Run the experiments

**Important:** these scripts are executable experiments, not a unified, lightweight test suite. Some run expensive model searches and five-fold cross-validation before reporting and final fit. Skore may also refit an estimator for its EstimatorReport.

### Known reproducible command: Experiment 09

```powershell
python -u .\experiments\09_tuned_lgbm.py
```

This runs the pilot, grouped CV, publishes `09_tuned_lgbm` and creates `submissions/09_tuned_lgbm.csv`.

### Run later experiments

```powershell
# Example of a complete (potentially long) three-stage stacking run:
python -u .\experiments\18_stack3.py
```

**Do not use Experiment 18 as a quick “generate CSV” command.** Its committed script always reruns all five outer folds, creates an EstimatorReport, then trains a final full-data model. The journal records roughly **57 minutes for CV alone**, before report and final-model training. The nested evaluation also requires review for potential cross-fold information sharing.

Experiment 17 has an explicit shortcut:

```powershell
$env:SKIP_CV = "1"
python -u .\experiments\17_stack_smooth.py
Remove-Item Env:SKIP_CV
```

That mode **reuses previously recorded outer-fold scores** rather than recomputing them. It still trains an EstimatorReport estimator and the full prediction model, so a valid report and CSV remain conditional on successful execution and publication.

**Experiment 13 warning:** `experiments/13_interviz_ledd.py` currently sets `KEY = "14_interviz_simple"`, which conflicts with Experiment 14. Do **not** publish the existing Experiment 13 script until IBM Bob verifies its implementation and corrects the key.

### Validate a generated submission CSV

Replace the filename with the **actual** CSV that was generated on your machine:

```powershell
python -c "import pandas as pd,numpy as np; f='submissions/18_stack3.csv'; s=pd.read_csv(f); t=pd.read_csv('data/sample_submission.csv'); assert list(s.columns)==['Index','target']; assert len(s)==len(t)==11013; assert s['Index'].equals(t['Index']); assert s['Index'].is_unique; assert np.isfinite(s['target'].to_numpy()).all(); print('CSV VALID:', f)"
```

This checks the file's structure, alignment and numerical values. It does **not** confirm model quality, publishing success or competition eligibility.

## Prepare a Kaggle submission

The organizer's [Day 2 guide](docs/DAY2.md) allows iterative submissions (e.g. baseline followed by improved models). It requires every upload to correspond to code written and run through Bob, evaluated with skore, and associated with a **new report key** and a Skore Hub **EstimatorReport URL in the Submission Description**.

1. **Choose a validated model**, not necessarily the lowest unverified number in the journal.
2. Verify the **Kaggle team and Skore workspace naming match**, that your account is in the team workspace, and that the model is eligible.
3. Use Bob to fit the chosen model on all training visits and produce `submissions/<report-key>.csv` containing **11,013 rows** and the exact `Index,target` header.
4. Obtain the **actual associated EstimatorReport URL** from the correct Skore Hub project. The report must correspond to that submission's experiment, not a different model.
5. On [Kaggle](https://www.kaggle.com/competitions/bobathon-esilv/), upload the CSV through **Submit Predictions** during the organizer's authorized window and paste its full report URL into the **Submission Description**.
6. Confirm Kaggle accepted the submission and follow any competition-specific instructions for choosing the final scored entry.

**Example — Experiment 09 only** (because its associated original-workspace URL is documented):

```text
CSV: submissions/09_tuned_lgbm.csv
Skore report key: 09_tuned_lgbm
Submission Description:
Experiment 09 — tuned LightGBM; grouped CV RMSE 4.1896.
Skore EstimatorReport: https://skore.probabl.ai/philippeEtchebest/bobathon-esilv/estimators/63086
```

**Do not pair `18_stack3.csv` with the `09_tuned_lgbm` report.** The notebook/journal reports a score for Experiment 18, but its final CSV, successful publication and exact Hub URL cannot be established from GitHub alone.

No public or private Kaggle score, official ranking or accepted final submission was verifiable from this repository at the time of this README update. Competition-specific submission limits and final-selection rules must be confirmed on Kaggle.

## Known limitations and next steps

| Topic | Current status / action |
| --- | --- |
| Synthetic data | Generalization to real Parkinson's clinical cohorts has not been established. |
| Patient holdout | GroupKFold is appropriate, but model selection on the same folds may introduce optimism. |
| Nested stacking (17/18) | Review the cross-fitting chain end to end. In Experiment 18, features feeding inner stage-2 folds can be built using stage-1 models trained on other inner folds, creating a possible indirect validation-target dependency. Treat 3.6341 as **provisional** pending an independent leakage audit. |
| Model interface | Some later classes inherit `BaseEstimator` before `RegressorMixin`; verify compatibility with the installed scikit-learn/Skore versions before assuming a complete execution will succeed. |
| Experiment 13 | Duplicate report key; do not run unchanged. |
| Experiment reports | Journal has unfilled report-link placeholders for 13–18 and inconsistent earlier workspace spellings. Verify report availability in the proper Hub workspace. |
| Dependencies | Pin the actual working versions of NumPy, pandas, SciPy, scikit-learn and LightGBM for a reproducible archive. |
| Predictions | CSVs are excluded from Git and must be regenerated or obtained from the teammate who produced them. |
| Kaggle and course hand-in | Submission acceptance, final ranking, required video/archive and deadlines must be verified from Kaggle and DVL. |

One diagnostic fitted a line directly to the **same patients' true `target` values** it evaluated and found approximately 1.61 RMSE. That is **in-sample target leakage**, not a generalization result; it must never be reported as a model performance floor. Details are in the [journal](journal/JOURNAL.md).

## Team, credits and references

The repository history records work by **Cyril Ibrahim (`@cyrilibr`)** on the initial experiments and project setup, and a Day 2 commit by **Ismaël Wane** adding Experiments 10–18. Exact personal contributions, authorship and group membership should be verified with the team before final academic submission.

This repository builds on the organizers' original teaching materials, included under `docs/`:

- [Official Bobathon starter repository](https://github.com/probabl-ai/bobathon-esilv)
- [Competition page](https://www.kaggle.com/competitions/bobathon-esilv/)
- [Competition-specific rules](https://www.kaggle.com/competitions/bobathon-esilv/rules)
- [Organizer setup instructions](docs/SETUP.md)
- [Day 1: EDA and baseline](docs/DAY1.md)
- [Day 2: modelling and grouped CV](docs/DAY2.md)
- [Scientific and data-generation context](docs/CONTEXT.md)
- [EDA narrative](data/eda.md)
- [Full experiment journal](journal/JOURNAL.md)

---

*Academic machine-learning project for the ESILV × IBM × Probabl Bobathon. This README documents repository evidence and openly distinguishes recorded results from results requiring independent verification.*
