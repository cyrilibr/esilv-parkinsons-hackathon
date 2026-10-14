"""
14_interviz_simple.py
Hypothesis: delta_on / delta_off / time_since_last_visit add signal on top of
the 12_bateman pipeline. Nothing else changes: no correlation pruning, no
LEDD-modulated Bateman, no extra inter-visit features.

Runs a control CV (feature OFF) then the treatment CV (feature ON), so the
delta is measured against both the 4.131 reference and a same-code control.
Target-dependent params (Bateman) are fitted inside fit() on the train fold.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold
from skore import EstimatorReport, Project, login

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEY = "14_interviz_simple"
BASELINE = 4.131
SEED = 42


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load():
    d = PROJECT_ROOT / "data"
    X = pd.read_csv(d / "X_train.csv")
    Y = pd.read_csv(d / "y_train.csv")
    Xt = pd.read_csv(d / "X_test.csv")
    X = X.merge(Y, on="Index", how="left", validate="1:1")
    y = X.pop("target").values.astype(float)
    test_index = Xt["Index"].values
    return X.drop(columns=["Index"]), y, Xt.drop(columns=["Index"]), test_index


# ----------------------------------------------------------------------------
# Bateman (original: ON curve + OFF decay, no LEDD modulation)
# ----------------------------------------------------------------------------
def g_on(t, ke, ka):
    return np.exp(-ke * t) - np.exp(-ka * t)


def fit_bateman(df, y, rng):
    p = {"A": 0.0, "ke": 0.025, "ka": 1.43, "B": 0.0, "kd": 0.012}
    m = df["on"].notna() & df["time_since_intake_on"].notna()
    if m.sum() > 100:
        idx = np.where(m)[0]
        idx = rng.choice(idx, min(len(idx), 20000), replace=False)
        t = df["time_since_intake_on"].values[idx].astype(float)
        r = y[idx] - df["on"].values[idx].astype(float)
        res = least_squares(
            lambda q: q[0] * g_on(t, q[1], q[2]) - r,
            x0=[10.0, 0.025, 1.43],
            bounds=([-100, 1e-3, 0.1], [100, 0.5, 10]))
        p["A"], p["ke"], p["ka"] = res.x
    m = df["off"].notna() & df["time_since_intake_off"].notna()
    if m.sum() > 100:
        idx = np.where(m)[0]
        idx = rng.choice(idx, min(len(idx), 20000), replace=False)
        t = df["time_since_intake_off"].values[idx].astype(float)
        r = y[idx] - df["off"].values[idx].astype(float)
        res = least_squares(
            lambda q: q[0] * np.exp(-q[1] * t) - r,
            x0=[2.0, 0.012], bounds=([-50, 1e-3], [50, 0.5]))
        p["B"], p["kd"] = res.x
    return p


def apply_bateman(df, p):
    out = pd.DataFrame(index=df.index)
    t_on = df["time_since_intake_on"].values.astype(float)
    t_off = df["time_since_intake_off"].values.astype(float)
    gon = np.nan_to_num(g_on(t_on, p["ke"], p["ka"]))
    doff = np.nan_to_num(np.exp(-p["kd"] * t_off))
    on = df["on"].values.astype(float)
    off = df["off"].values.astype(float)
    out["on_corr"] = on + p["A"] * gon
    out["off_corr"] = off + p["B"] * doff
    out["bat_delta_on"] = out["on_corr"] - on
    out["bat_delta_off"] = out["off_corr"] - off
    return out


# ----------------------------------------------------------------------------
# Patient aggregates (stats, slope/intercept, quadratic) - no target
# ----------------------------------------------------------------------------
def patient_agg(df, col, pre):
    """Vectorised patient aggregates using pandas groupby (no Python loops)."""
    tmp = df[["patient_id", "age", col]].copy().reset_index(drop=True)
    tmp["age"] = tmp["age"].astype(float)
    tmp[col] = tmp[col].astype(float)
    tmp_sorted = tmp.sort_values(["patient_id", "age"])

    g = tmp_sorted.groupby("patient_id")[col]
    agg_stats = pd.DataFrame({
        f"{pre}_mean":      g.mean(),
        f"{pre}_std":       g.std(ddof=0),
        f"{pre}_min":       g.min(),
        f"{pre}_max":       g.max(),
        f"{pre}_count":     g.count(),
        f"{pre}_miss_frac": tmp_sorted.groupby("patient_id")[col].apply(lambda x: x.isna().mean()),
        f"{pre}_first":     g.first(),
        f"{pre}_last":      g.last(),
        f"{pre}_age_span":  tmp_sorted.groupby("patient_id")["age"].apply(lambda x: x.max() - x.min()),
    })

    # slope / intercept via groupby apply (still needed for linear/quad)
    def _slope_icpt_curv(grp):
        a = grp["age"].values; v = grp[col].values
        m = ~np.isnan(v) & ~np.isnan(a)
        res = {}
        if m.sum() >= 2 and np.ptp(a[m]) > 1e-6:
            a0 = a[m].mean()
            c1 = np.polyfit(a[m] - a0, v[m], 1)
            res["slope"] = c1[0]; res["icpt"] = c1[1]; res["a0"] = a0
        if m.sum() >= 3 and np.ptp(a[m]) > 1e-6:
            a0 = a[m].mean()
            c2 = np.polyfit(a[m] - a0, v[m], 2)
            res["curv"] = c2[0]; res["q_slope"] = c2[1]; res["q_icpt"] = c2[2]
        return pd.Series(res)

    poly = tmp_sorted.groupby("patient_id").apply(_slope_icpt_curv)
    if isinstance(poly, pd.Series):
        poly = poly.to_frame().T if not poly.empty else pd.DataFrame()
    if not poly.empty:
        for c in poly.columns:
            agg_stats[f"{pre}_{c}"] = poly[c]

    # join back
    A = tmp[["patient_id"]].join(agg_stats, on="patient_id").drop(columns="patient_id")
    A.index = df.index

    # per-visit residuals from linear and quadratic predictions
    vals = tmp[col].values
    pred_lin  = np.full(len(tmp), np.nan)
    pred_quad = np.full(len(tmp), np.nan)
    if f"{pre}_slope" in A.columns:
        a0 = A.get(f"{pre}_a0", pd.Series(0.0, index=A.index)).fillna(0).values
        pred_lin = A[f"{pre}_slope"].fillna(0).values * (tmp["age"].values - a0) + A[f"{pre}_icpt"].fillna(0).values
    if f"{pre}_curv" in A.columns:
        a0 = A.get(f"{pre}_a0", pd.Series(0.0, index=A.index)).fillna(0).values
        age_c = tmp["age"].values - a0
        pred_quad = (A[f"{pre}_curv"].fillna(0).values * age_c**2
                     + A[f"{pre}_q_slope"].fillna(0).values * age_c
                     + A[f"{pre}_q_icpt"].fillna(0).values)
    A[f"{pre}_res_lin"]   = vals - pred_lin
    A[f"{pre}_pred_quad"] = pred_quad
    A[f"{pre}_res_quad"]  = vals - pred_quad
    # drop helper column
    A = A.drop(columns=[f"{pre}_a0"], errors="ignore")
    return A


# ----------------------------------------------------------------------------
# The ONE new idea: inter-visit features (no target)
# ----------------------------------------------------------------------------
def inter_visit(df):
    d = pd.DataFrame({
        "patient_id": df["patient_id"].values,
        "age": df["age"].values.astype(float),
        "on": df["on"].values.astype(float),
        "off": df["off"].values.astype(float),
    }, index=df.index).sort_values(["patient_id", "age"])
    grp = d.groupby("patient_id")
    out = pd.DataFrame(index=d.index)
    out["time_since_last_visit"] = d["age"] - grp["age"].shift(1)
    # change vs the last previously-known score for the same patient
    out["delta_on"] = d["on"] - grp["on"].ffill().groupby(
        d["patient_id"]).shift(1)
    out["delta_off"] = d["off"] - grp["off"].ffill().groupby(
        d["patient_id"]).shift(1)
    return out.loc[df.index]


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class FullModel(BaseEstimator, RegressorMixin):
    def __init__(self, use_intervisit=True, n_estimators=2000,
                 learning_rate=0.03, num_leaves=63, reg_alpha=0.5,
                 reg_lambda=3.0, colsample_bytree=0.7, random_state=SEED):
        self.use_intervisit = use_intervisit
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

    def _features(self, X):
        X = X.reset_index(drop=True)
        base = X.drop(columns=["patient_id"]).copy()
        for c, mp in self.cat_maps_.items():
            if c in X.columns:
                base[c] = X[c].astype(str).where(X[c].notna()).map(mp) \
                    .astype(float)
        on = X["on"].astype(float)
        off = X["off"].astype(float)
        ledd = X["ledd"].astype(float)
        # medication features (as in stage 3)
        gap = on - off
        base["on_off_gap"] = gap
        base["ledd_x_gap"] = ledd * gap
        base["on_off_ratio"] = on / off.replace(0, np.nan)
        base["t_on_x_on"] = X["time_since_intake_on"] * on
        base["t_off_x_off"] = X["time_since_intake_off"] * off
        parts = [base]
        # patient aggregates on raw scores (stages 1-2)
        for col, pre in [("on", "p_on"), ("off", "p_off")]:
            parts.append(patient_agg(X, col, pre))
        # Bateman corrected scores + their patient aggregates (stage 4)
        F = apply_bateman(X, self.bat_)
        parts.append(F)
        for col, pre in [("on_corr", "p_onc"), ("off_corr", "p_offc")]:
            tmp = X[["patient_id", "age"]].copy()
            tmp[col] = F[col].values
            parts.append(patient_agg(tmp, col, pre))
        if self.use_intervisit:
            parts.append(inter_visit(X))
        out = pd.concat(parts, axis=1)
        out = out.loc[:, ~out.columns.duplicated()]
        return out.astype(float)

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.cat_maps_ = {}
        for c in ["cohort", "gene", "rater_id"]:
            if c in X.columns:
                cats = sorted(X[c].dropna().astype(str).unique())
                self.cat_maps_[c] = {k: i for i, k in enumerate(cats)}
        self.bat_ = fit_bateman(X, y, rng)      # OOF under CV
        F = self._features(X)
        self.cols_ = list(F.columns)            # no pruning
        self.model_ = LGBMRegressor(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            num_leaves=self.num_leaves, reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda, colsample_bytree=self.colsample_bytree,
            subsample=0.8, subsample_freq=1, random_state=self.random_state,
            n_jobs=-1, verbose=-1)
        self.model_.fit(F, y)
        return self

    def predict(self, X):
        return self.model_.predict(self._features(X)[self.cols_])


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def run_cv(X, y, use_intervisit, label):
    gkf = GroupKFold(n_splits=5)
    scores, last = [], None
    for k, (tr, va) in enumerate(gkf.split(X, y, X["patient_id"].values), 1):
        m = FullModel(use_intervisit=use_intervisit).fit(X.iloc[tr], y[tr])
        s = rmse(y[va], m.predict(X.iloc[va]))
        scores.append(s)
        last = (tr, va)
        print(f"[{label}] fold {k}: RMSE = {s:.4f}  "
              f"(n_features={len(m.cols_)})")
    return np.array(scores), last


def main():
    X, y, Xt, test_index = load()

    ctrl, _ = run_cv(X, y, False, "control")
    trt, last_split = run_cv(X, y, True, "inter-visit")

    mean, std = trt.mean(), trt.std()
    print(f"\nControl (same code, no inter-visit): {ctrl.mean():.4f} "
          f"+/- {ctrl.std():.4f}")
    print(f"Treatment                          : {mean:.4f} +/- {std:.4f}")
    print(f"delta vs {BASELINE} (reference)      : {BASELINE - mean:+.4f}")
    d_ctrl = ctrl.mean() - mean
    print(f"delta vs control (true effect)      : {d_ctrl:+.4f}")
    print("gain > std (vs reference):", "YES" if BASELINE - mean > std else "NO")
    print("gain > std (vs control)  :", "YES" if d_ctrl > std else "NO")

    # --- skore hub
    cfg = json.loads((PROJECT_ROOT / ".skore").read_text())
    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub",
                      workspace=cfg["workspace"])
    tr, va = last_split
    est_report = EstimatorReport(
        FullModel(use_intervisit=True),
        X_train=X.iloc[tr], y_train=y[tr],
        X_test=X.iloc[va], y_test=y[va])
    project.put(KEY, est_report)

    # --- submission
    pred = FullModel(use_intervisit=True).fit(X, y).predict(Xt)
    out_dir = PROJECT_ROOT / "submissions"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame({"Index": test_index, "target": pred}).to_csv(
        out_dir / f"{KEY}.csv", index=False)
    print(f"submissions/{KEY}.csv written ({len(pred)} rows)")


if __name__ == "__main__":
    main()