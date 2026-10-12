# JOURNAL

## Status

- **Project:** Parkinson's disease OFF-score prediction (Bobathon ESILV)
- **Goal:** Predict the debiased true OFF MDS-UPDRS motor score (`target`) for
  every patient visit.
- **Metric:** RMSE (lower is better; baseline = dummy mean RMSE ≈ target std dev)

### Workspace decisions

| Decision | Value | Date |
|---|---|---|
| Tabular library | pandas | 2025-10-10 |
| CV splitter | `GroupKFold(groups=patient_id)` (candidate — EDA finding) | — |
| Scoring metric | RMSE | — |

---

## Data understanding (EDA)

- **Status:** done - 2025-10-10
- **Summary:** 44 590 training visits from 5 576 patients (8 visits/patient on
  average); 11 013 test visits from 1 395 patients with zero overlap. Target
  (`target`, debiased OFF score) is approximately normal (mean 37.5, std 16.5,
  range 0–109.5). Seven columns are missing up to 79 %; `on` and `off` are
  never simultaneously absent. Strong predictors: `off` (r = 0.87) and `on`
  (r = 0.67). Patient-level holdout → GroupKFold is mandatory.
- **Report:** [data/eda.md](../data/eda.md)

---

## Experiments

| # | Key | Grouped CV RMSE | Δ vs prev | Notes |
|---|---|---:|---:|---|
| 01 | `01_dummy` | 16.499 | — | DummyRegressor(mean) |
| 02 | `02_ridge` | 10.446 | −36.7% | Ridge + median impute + OHE |
| 03 | `03_histgb` | 7.484 | −28.4% | HistGBR, NaN native |
| 04 | `04_patient_features` | 5.346 | −28.6% | + patient aggregates (HGBR) |
| 05 | `05_lightgbm` | 5.293 | −1.0% | LightGBM, same features |
| 06 | `06_catboost` | abandoned | — | Computational cost too high |
| 07 | `07_longitudinal` | **4.347** | **−17.9%** | + slope/intercept/std/min/max/first/last/age-pct |
| 08 | `08_medication` | **4.210** | **−3.2%** | + trend-residuals, on-off gap, ledd×gap, timing interactions |

### Experiment 07 — key findings
- Per-patient ON/OFF linear slope + intercept (from feature rows only) were the highest-value new features
- Per-visit age percentile within patient span captures disease stage effectively
- Pooled OOF RMSE = 4.348 (consistent with fold average — no fold-specific outlier)
- std = 0.082 (tight across folds)
- **Best model so far: 07_longitudinal**
- Hub report: https://skore.probabl.ai/philippeEtchebest/bobathon-esilv/estimators/62869
