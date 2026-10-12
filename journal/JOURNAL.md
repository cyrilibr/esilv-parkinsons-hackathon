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
| 09 | `09_tuned_lgbm` | **4.190** | **−0.5%** | strong_reg: reg_alpha=0.5, reg_lambda=3.0, colsample=0.7, 2000 trees |

### Experiment 07 — key findings
- Per-patient ON/OFF linear slope + intercept (from feature rows only) were the highest-value new features
- Per-visit age percentile within patient span captures disease stage effectively
- Pooled OOF RMSE = 4.348 (consistent with fold average — no fold-specific outlier)
- std = 0.082 (tight across folds)
- Hub report: https://skore.probabl.ai/philippeEtchebest/bobathon-esilv/estimators/62869

### Experiment 08 — key findings
- Trend-residual features (on/off deviation from patient's linear trend) added signal
- on_off_gap and ledd×gap: the ON-OFF spread correlates −0.42 with measurement bias
- Pooled OOF RMSE = 4.2106 | std = 0.065 (tighter than Exp-07)
- Hub report: https://skore.probabl.ai/philippeEtchebest/bobathon-esilv/estimators/62868
- **Previous best: 08_medication**

### Experiment 09 — key findings
- `strong_reg` config (reg_alpha=0.5, reg_lambda=3.0, colsample=0.7, 2000 trees, LR=0.03) won 2-fold pilot
- Full 5-fold mean RMSE = 4.1896 | std = 0.057 | folds: 4.1865/4.2756/4.1527/4.1274/4.2055
- Per-fold OOF improvements over Exp-08 on all 5 folds
- Hub report: https://skore.probabl.ai/philippeEtchebest/bobathon-esilv/estimators/63086
- **Best model so far: 09_tuned_lgbm**
- Fix: MedicationFeatureAdder must set self.fitted_ = True for Hub serialization

### Audit note — 1.61 in-sample RMSE
The diagnostic RMSE of 1.61 (scratch/07_diagnostic.py) used `np.polyfit(age, target, 1)`
fit and evaluated on the SAME patient's own target rows. This is fully in-sample leakage —
it cannot be interpreted as a generalization bound. The honest unseen-patient floor is
Exp-08 OOF RMSE = 4.21.
