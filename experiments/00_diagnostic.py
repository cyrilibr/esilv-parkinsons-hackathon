"""Diagnostic script — no model training, just numbers.

Answers:
  a) Per-patient linear vs quadratic R² for target ~ time_since_diagnosis
  b) rater_id overlap between train and test
  c) (target - on)/target vs time_since_intake_on  &
     (target - off)/target vs time_since_intake_off
  d) Consistency of time_since_diagnosis in test vs (age - age_at_diagnosis)
  e) Why no experiment 06 (answered in text above)
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test  = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

visits = X_train.merge(y_train, on="Index", validate="one_to_one")
visits["time_since_diagnosis"] = visits["age"] - visits["age_at_diagnosis"]

print("=" * 60)
print("DIAGNOSTIC — no model training")
print("=" * 60)

# -----------------------------------------------------------------------
# (a) Per-patient R² for linear and quadratic fit of target ~ time
# -----------------------------------------------------------------------
print("\n--- (a) Per-patient linear vs quadratic R² ---")

lin_r2_list, quad_r2_list = [], []
for pid, grp in visits.groupby("patient_id"):
    if len(grp) < 3:
        continue
    t = grp["time_since_diagnosis"].values.astype(float)
    y = grp["target"].values.astype(float)
    mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 3:
        continue
    t, y = t[mask], y[mask]
    # linear
    p1 = np.polyfit(t, y, 1)
    res1 = y - np.polyval(p1, t)
    ss_res1 = np.sum(res1 ** 2)
    ss_tot  = np.sum((y - y.mean()) ** 2)
    r2_lin  = 1 - ss_res1 / ss_tot if ss_tot > 0 else np.nan
    # quadratic
    p2 = np.polyfit(t, y, 2)
    res2 = y - np.polyval(p2, t)
    ss_res2 = np.sum(res2 ** 2)
    r2_quad = 1 - ss_res2 / ss_tot if ss_tot > 0 else np.nan
    lin_r2_list.append(r2_lin)
    quad_r2_list.append(r2_quad)

lin_r2  = np.array(lin_r2_list)
quad_r2 = np.array(quad_r2_list)

print(f"Patients with >= 3 visits: {len(lin_r2)}")
print(f"Linear  R² — median: {np.nanmedian(lin_r2):.3f}  mean: {np.nanmean(lin_r2):.3f}  "
      f"p10: {np.nanpercentile(lin_r2, 10):.3f}  p90: {np.nanpercentile(lin_r2, 90):.3f}")
print(f"Quadrat R² — median: {np.nanmedian(quad_r2):.3f}  mean: {np.nanmean(quad_r2):.3f}  "
      f"p10: {np.nanpercentile(quad_r2, 10):.3f}  p90: {np.nanpercentile(quad_r2, 90):.3f}")
print(f"Quad vs Lin Δ median: {np.nanmedian(quad_r2 - lin_r2):.3f}")

# residuals distribution
all_residuals = []
for pid, grp in visits.groupby("patient_id"):
    if len(grp) < 2:
        continue
    t = grp["time_since_diagnosis"].values.astype(float)
    y = grp["target"].values.astype(float)
    mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 2:
        continue
    t, y = t[mask], y[mask]
    p1 = np.polyfit(t, y, 1)
    all_residuals.extend((y - np.polyval(p1, t)).tolist())
res = np.array(all_residuals)
print(f"Linear residuals — std: {res.std():.3f}  p5: {np.percentile(res,5):.3f}  "
      f"p95: {np.percentile(res,95):.3f}  abs-mean: {np.abs(res).mean():.3f}")

# -----------------------------------------------------------------------
# (b) rater_id overlap
# -----------------------------------------------------------------------
print("\n--- (b) rater_id overlap train ↔ test ---")

train_raters = set(visits["rater_id"].dropna().unique())
test_raters  = set(X_test["rater_id"].dropna().unique())

overlap = train_raters & test_raters
pct_test_covered = 100 * len(overlap) / len(test_raters) if test_raters else 0
print(f"Unique raters in train : {len(train_raters)}")
print(f"Unique raters in test  : {len(test_raters)}")
print(f"Overlap                : {len(overlap)}  ({pct_test_covered:.1f}% of test raters)")
print(f"Test raters NOT in train: {len(test_raters - train_raters)}")

# visits per rater in train
rater_counts = visits["rater_id"].value_counts()
print(f"Visits/rater in train — median: {rater_counts.median():.0f}  "
      f"min: {rater_counts.min()}  max: {rater_counts.max()}")

# -----------------------------------------------------------------------
# (c) Drug timing curves
# -----------------------------------------------------------------------
print("\n--- (c) Timing curves ---")

# (target - on) / target vs time_since_intake_on
sub_on = visits.dropna(subset=["on", "time_since_intake_on", "target"]).copy()
sub_on = sub_on[sub_on["target"].abs() > 1e-6]
sub_on["frac_on"] = (sub_on["target"] - sub_on["on"]) / sub_on["target"]

sub_off = visits.dropna(subset=["off", "time_since_intake_off", "target"]).copy()
sub_off = sub_off[sub_off["target"].abs() > 1e-6]
sub_off["frac_off"] = (sub_off["target"] - sub_off["off"]) / sub_off["target"]

bins_on = pd.cut(sub_on["time_since_intake_on"], bins=10)
bins_off = pd.cut(sub_off["time_since_intake_off"], bins=10)

print("(target-on)/target by time_since_intake_on decile (mean):")
print(sub_on.groupby(bins_on, observed=True)["frac_on"].mean().round(3).to_string())

print("\n(target-off)/target by time_since_intake_off decile (mean):")
print(sub_off.groupby(bins_off, observed=True)["frac_off"].mean().round(3).to_string())

# -----------------------------------------------------------------------
# (d) time_since_diagnosis consistency in test
# -----------------------------------------------------------------------
print("\n--- (d) time_since_diagnosis coherence in test ---")

if "time_since_diagnosis" in X_test.columns:
    test_tsd = X_test["time_since_diagnosis"]
    calc_tsd  = X_test["age"] - X_test["age_at_diagnosis"]
    diff = (test_tsd - calc_tsd).abs()
    print(f"Abs diff (time_since_diagnosis vs age-age_at_diagnosis):")
    print(f"  max={diff.max():.6f}  mean={diff.mean():.6f}  n_nonzero={( diff > 1e-6).sum()}")
else:
    calc_tsd = X_test["age"] - X_test["age_at_diagnosis"]
    print("'time_since_diagnosis' NOT in X_test columns — must be computed as age - age_at_diagnosis")
    print(f"  range: [{calc_tsd.min():.2f}, {calc_tsd.max():.2f}]  mean: {calc_tsd.mean():.2f}")

print("\n--- Done ---")
