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
| 10 | `10_quadratic_features` | **4.171** | **−0.5%** | + coeff quadratique par patient (on/off~age²), résidus quadratiques |
| 11 | `11_rater_bias` | **4.164** | **−0.2%** | + biais rater OOF (empirical Bayes, smoothing=20) |
| 12 | `12_bateman` | **4.131** | **−0.8%** | + correction pharmacologique Bateman OOF (A=0.61, ke=0.025, ka=1.43) |
| 13 | `13_interviz_ledd` | 4.221 | +2.2% | ❌ inter-visites + LEDD Bateman + pruning corrélation — pire |
| 14 | `14_interviz_simple` | 4.095 | −0.9% | + deltas inter-visites (delta_on, delta_off, rank, velocity) — gain < std |
| 15 | `15_patient_smoothing` | **4.026** | **−2.5%** | + smoothing quadratique OOF par patient (deg=2, alpha=1.0) — gain > std ✅ |
| 16 | `16_smooth_ridge` | 4.022 | −0.1% | lissage pondéré + pénalité courbure (lam=0.03, w_nooff=0.4) — gain < biais de sélection |
| 17 | `17_stack_smooth` | **3.765** | **−6.5%** | + stacking 2 étages sur courbes OOF par patient, puis lissage — 5/5 folds ✅ |
| 18 | `18_stack3` | **3.634** | **−3.5%** | + 3ᵉ étage de stacking sur courbes OOF de l'étage 2 — 5/5 folds ✅ |
| 19 | `19_seed_bagging` | en attente | — | étage 2 moyenné sur 8 graines (base 17) |


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


### Experiment 09 — key findings
- `strong_reg` config (reg_alpha=0.5, reg_lambda=3.0, colsample=0.7, 2000 trees, LR=0.03) won 2-fold pilot
- Full 5-fold mean RMSE = 4.1896 | std = 0.057 | folds: 4.1865/4.2756/4.1527/4.1274/4.2055
- Per-fold OOF improvements over Exp-08 on all 5 folds
- Hub report: https://skore.probabl.ai/philippeEtchebest/bobathon-esilv/estimators/63086
- **Best model so far: 09_tuned_lgbm**
- Fix: MedicationFeatureAdder must set self.fitted_ = True for Hub serialization

### Experiment 10 — key findings
- Quadratic fit (a*age²+b*age+c) per patient: R²=0.999 median vs 0.959 linear (diagnostic 00)
- New features: pt_on_quad/pt_off_quad (curvature), quadratic predictions, quadratic residuals
- Pipeline order: QuadraticAggregator → LongitudinalAggregator (must run before patient_id is dropped)
- Mean 5-fold RMSE = 4.1714 | std = 0.0645 | folds: 4.1642/4.2691/4.1296/4.1017/4.1925
- Hub report: https://skore.probabl.ai/philippeechebest/bobathon-esilv/estimators/69664


### Experiment 11 — key findings
- 100% rater overlap train↔test (50 raters, ~880 visits/rater) → supervised OOF bias viable
- Empirical Bayes shrinkage: bias_hat = n * raw_mean / (n + 20)
- Features: rater_off_bias, rater_on_bias, rater_visit_count + bias×score interactions
- Custom Pipeline11 (BaseEstimator) needed to pass y to RaterBiasEncoder.fit inside fold
- Note: gain (−0.007) is 0.11× std — marginal but positive on all folds
- Mean 5-fold RMSE = 4.1644 | std = 0.0632 | folds: 4.1774/4.2506/4.1349/4.0793/4.1798
- Hub report: https://skore.probabl.ai/philippeechebest/bobathon-esilv/estimators/69801


### Experiment 12 — key findings
- Courbe Bateman `A*(exp(-ke*t) - exp(-ka*t))` ajustée OOF sur les visites ON (23k visites)
- Paramètres stables entre folds : A≈0.61, ke≈0.025, ka≈1.43 (pic d'effet à ~t=2.5h)
- Courbe de décroissance OFF : `B*exp(-ke*t)` ajustée sur 9.5k visites (B≈0.24, ke≈0.012)
- Nouvelles features : on_corr, off_corr, on_corr_delta, off_corr_delta, on_off_corr_gap, + agrégats par patient des scores corrigés
- Gain : −0.033 RMSE (0.5× std) — meilleur gain depuis Exp-07
- Mean 5-fold RMSE = 4.1310 | std = 0.0652 | folds: 4.1698/4.2148/4.0774/4.0565/4.1366
- Hub report: https://skore.probabl.ai/philippeechebest/bobathon-esilv/estimators/70011


### Experiment 13 — key findings
- Hypothèse : deltas inter-visites + amplitude Bateman modulée par LEDD + agrégats sur scores corrigés + suppression des features corrélées (>0.995)
- Résultat : 4.221 → **pire** que 12 (+0.09)
- Cause probable : trop d'idées à la fois ; la suppression de 35 features corrélées retire du signal utile
- Leçon : **une seule idée par expérience**, ne jamais retirer de features de la base 12 sans test isolé
- Hub report: <lien>

### Experiment 14 — key findings
- Hypothèse : deltas inter-visites (delta_on, delta_off, temps depuis la visite précédente) ajoutés seuls à la base 12
- Résultat : 4.095 (−0.036), gain < std (0.065) → pas d'amélioration réelle
- Le signal temporel local est déjà capté par les agrégats longitudinaux (pentes, résidus de tendance)
- Hub report: <lien>

### Experiment 15 — key findings
- Hypothèse : la cible est une courbe lisse en âge par patient (quadratique R²=0.999) ; le modèle prédit visite par visite et hérite du bruit de on/off
- Méthode : post-traitement — pour chaque patient, ajustement d'une parabole en âge sur les **prédictions** (jamais la cible), remplacée par la courbe ajustée
- Variantes testées : deg 1 / deg 2, α ∈ {0.5, 0.75, 1.0} ; meilleure : deg=2, α=1
- deg=1 à α=1 dégrade (+0.03) → la trajectoire est bien courbe
- Mean 5-fold RMSE = 4.0256 | std = 0.0648 | folds: 4.0686/4.1044/3.9672/3.9531/4.0347
- Gain apparié vs 12 : −0.105, **5/5 folds**
- Hub report: <lien>

### Experiment 16 — key findings
- Hypothèse : lissage amélioré — poids réduit pour les visites sans `off` (w_nooff), pénalité ridge sur la courbure (lam)
- Grille 5 × 3 ; meilleure : lam=0.03, w_nooff=0.4
- Mean 5-fold RMSE = 4.0216 | std = 0.0634 — seulement −0.004 vs 15
- Gain inférieur au biais de sélection (~0.01 pour 15 variantes) → **non retenu**, la parabole simple de 15 suffit
- Hub report: <lien>

### Experiment 17 — key findings
- Hypothèse : le lissage corrige le bruit entre visites mais pas le **décalage global** de la courbe d'un patient ; un 2ᵉ modèle peut le corriger
- Méthode :
  - étage 1 = pipeline 12, prédictions **OOF** sur le train via GroupKFold interne (5 folds, Bateman réajusté dans chaque fold interne)
  - features de courbe par patient sur ces prédictions : niveau, pente, courbure, valeur lissée, résidu, moyenne, écart-type
  - étage 2 = LightGBM (1000 arbres, 31 feuilles, reg_lambda=5) sur features d'origine + features de courbe
  - lissage quadratique final des prédictions de l'étage 2
- Résultats (mêmes folds) :
  - étage 1 brut = 4.1310 (= exp 12), étage 1 lissé = 4.0256 (= exp 15)
  - étage 2 brut = 3.8185, **étage 2 lissé = 3.7652**
- Mean 5-fold RMSE = **3.7652** | std = **0.0414** | folds: 3.7713/3.8145/3.7281/3.7183/3.7937
- Gain apparié vs 15 : −0.260, **5/5 folds** ; std réduit (0.065 → 0.041)
- Coût : ~160 s par fold (7 LightGBM par fold)
- Fix skore : `EstimatorReport` ne supporte pas `fit=False` dans cette version, et rappelle `fit(X=..., y=...)` → les méthodes doivent s'appeler avec `X`
- Hub report: <lien>
- **Best model so far: 17_stack_smooth**

### Experiment 18 — key findings
- Hypothèse : l'étage 2 étant bien meilleur que l'étage 1, ses courbes OOF donnent un meilleur signal à un étage 3
- Méthode : étages 1 et 2 OOF sur les mêmes folds internes ; étage 3 (800 arbres, reg_lambda=8) sur features d'origine + courbes étage 1 + courbes étage 2 ; lissage final
- Mean 5-fold RMSE = **3.6341** | std = **0.0257** | folds: 3.6656/3.6458/3.6238/3.5969/3.6386
- Gain apparié vs 17 : −0.131, **5/5 folds**
- Coût : 3410 s au total (~57 min)
- Hub report: <lien>
- **Best model so far: 18_stack3**


### Audit note — 1.61 in-sample RMSE
The diagnostic RMSE of 1.61 (scratch/07_diagnostic.py) used `np.polyfit(age, target, 1)`
fit and evaluated on the SAME patient's own target rows. This is fully in-sample leakage —
it cannot be interpreted as a generalization bound. The honest unseen-patient floor is
now Exp-18 OOF RMSE = 3.634.