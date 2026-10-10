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

*(none yet)*
