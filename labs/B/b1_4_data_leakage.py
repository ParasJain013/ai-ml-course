# %% [markdown]
# # B1.4 Data leakage — lab
# Lesson: lessons/B/b1_4_data_leakage.md
# Run: uv run python labs/B/b1_4_data_leakage.py   (CPU, defaults: ~10 s)
# GPU: no

# %%
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:  # plain `uv run`: draw to files, never open windows
    matplotlib.use("Agg")
    # matplotlib 3.11 warns when plt.show() runs under Agg; under `uv run` it is a harmless no-op
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()


def repo_root() -> Path:
    """Find the course folder (works under `uv run` and in VS Code cells)."""
    here = Path(globals().get("__file__", Path.cwd())).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("Run labs from inside the ai-ml-course folder.")


ROOT = repo_root()
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "b1_4"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE only makes the tables and repetitions bigger.
GPU_SCALE = False
N_CUSTOMERS = 50_000 if GPU_SCALE else 4_000
N_MONTHS = 24  # signup cohorts, Jan 2024 .. Dec 2025
N_SMALL, N_NOISE, K_SELECT = 200, 5_000, 20  # the feature-selection leak: 200 rows, 5,000 noise columns
N_REPEATS = 100 if GPU_SCALE else 20  # repetitions of the feature-selection experiment
FLAG_AUC = 0.90  # leakage_check flags single features at least this predictive

# %%
# Data: synthetic subscription customers. The question: "will a customer who signs up now cancel
# within 90 days?". Each signup month (cohort) has its own churn level (marketing campaigns bring in
# better or worse customers), which is unknown for a cohort until its 90 days have passed.
section("data")


def make_churn(n: int, gen: np.random.Generator) -> pd.DataFrame:
    """Generate n customers with honest features, one leaky feature and a 90-day churn label."""
    month = gen.integers(0, N_MONTHS, n)  # 0 = 2024-01
    cohort_effect = gen.normal(0, 1.0, N_MONTHS)[month]  # (n,) per-cohort shift of the churn log-odds
    plan = gen.choice(["basic", "pro", "team"], n, p=[0.6, 0.3, 0.1])
    first_week_logins = gen.poisson(5, n)
    onboarding_done = (gen.random(n) < 0.6).astype(int)
    discount = (gen.random(n) < 0.3).astype(int)
    logit = (-0.6 - 0.25 * (first_week_logins - 5) - 0.8 * onboarding_done + 0.6 * discount
             + pd.Series(plan).map({"basic": 0.4, "pro": -0.2, "team": -0.8}).to_numpy() + cohort_effect)
    churned = (gen.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    # The leak: a "retention call" is logged when a customer asks to cancel, i.e. AFTER the outcome.
    retention_call = np.where(churned == 1, gen.random(n) < 0.85, gen.random(n) < 0.04).astype(int)
    return pd.DataFrame({
        "signup_month": [f"{2024 + m // 12}-{m % 12 + 1:02d}" for m in month],
        "plan": plan,
        "first_week_logins": first_week_logins,
        "onboarding_done": onboarding_done,
        "discount": discount,
        "retention_call": retention_call,
        "churned": churned,
    })


df = make_churn(N_CUSTOMERS, rng)
print(f"customers: {df.shape}   churn rate {df['churned'].mean():.4f}")
print(df.head(5).to_string())
by_month = df.groupby("signup_month")["churned"].mean()
print(f"churn rate by signup month: min {by_month.min():.4f}  max {by_month.max():.4f}  "
      f"(sd across months {by_month.std():.4f})")

# %%
# Worked example by hand: AUC on four scores, and how one test value bends a scaler fitted on everything.
section("worked example by hand")
s4, y4 = np.array([0.9, 0.8, 0.4, 0.3]), np.array([1, 0, 1, 0])
pairs = [(float(p), float(q)) for p in s4[y4 == 1] for q in s4[y4 == 0]]
print(f"scores {s4.tolist()}   labels {y4.tolist()}")
print(f"(positive, negative) pairs: {pairs}")
print(f"positive ranked higher in {sum(p > q for p, q in pairs)} of {len(pairs)} pairs -> AUC = "
      f"{sum(p > q for p, q in pairs) / len(pairs):.2f}")
tr3, te1 = np.array([1.0, 2.0, 3.0]), 10.0
mu_tr, sd_tr = tr3.mean(), tr3.std()
both = np.append(tr3, te1)
mu_all, sd_all = both.mean(), both.std()
print(f"scaler fitted on train [1, 2, 3]: mean {mu_tr:.4f} sd {sd_tr:.4f} -> test value 10 scales to "
      f"{(te1 - mu_tr) / sd_tr:.4f}")
print(f"scaler fitted on train + test [1, 2, 3, 10]: mean {mu_all:.4f} sd {sd_all:.4f} -> test value 10 scales to "
      f"{(te1 - mu_all) / sd_all:.4f}")
months8 = np.array(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"])
r_test = np.sort(np.random.default_rng(3).choice(8, 2, replace=False))
print(f"random split, test months: {months8[r_test].tolist()}   time split, test months: {months8[6:].tolist()}")

# %%
# From scratch: AUC as "the share of (positive, negative) pairs ranked the right way round".


def auc_score(y: IntArray, score: FloatArray) -> float:
    """ROC-AUC by ranks (Mann-Whitney U), ties counted half. y (n,) in {0, 1}, score (n,) -> scalar."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s_sorted = score[order]
    i = 0
    while i < len(score):  # average ranks over runs of tied scores
        j = i
        while j + 1 < len(score) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1  # ranks start at 1
        i = j + 1
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2  # pairs where a positive outranks a negative
    return float(u / (n_pos * n_neg))


section("from scratch: AUC")
print(f"hand example: auc_score = {auc_score(y4, s4):.4f}   with a tie [0.9, 0.4, 0.4, 0.3]: "
      f"{auc_score(y4, np.array([0.9, 0.4, 0.4, 0.3])):.4f}")
y_rand = rng.integers(0, 2, 1_000)
s_rand = rng.random(1_000)
print(f"random scores on 1000 rows: {auc_score(y_rand, s_rand):.4f}   perfect scores: "
      f"{auc_score(y_rand, y_rand + 0.1 * s_rand):.4f}   reversed: {auc_score(y_rand, -y_rand + 0.1 * s_rand):.4f}")

# %%
# From scratch: leakage_check. Score each feature ALONE; a single feature that nearly separates the
# classes is either a breakthrough or (much more often) a leak. Then ask when its value is known.
section("from scratch: leakage_check")


def leakage_check(frame: pd.DataFrame, target: str, threshold: float = FLAG_AUC) -> pd.DataFrame:
    """Single-feature AUC of every numeric column (categories: their training churn rate). -> table."""
    y = frame[target].to_numpy()
    rows = []
    for col in frame.columns.drop(target):
        x = frame[col]
        if x.dtype.kind in "biuf":
            s = x.to_numpy(np.float64)
        else:  # a category is scored by the target rate of its group (in-sample: optimistic, see lesson)
            s = x.map(frame.groupby(col)[target].mean()).to_numpy(np.float64)
        a = auc_score(y, s)
        rows.append({"feature": col, "auc_alone": round(max(a, 1 - a), 4), "flag": max(a, 1 - a) >= threshold})
    return pd.DataFrame(rows).sort_values("auc_alone", ascending=False).reset_index(drop=True)


report = leakage_check(df, "churned")
print(report.to_string())

# %%
# The model used for every leak: one-hot + scaling + logistic regression (B2.1 explains the model).
# A random 75/25 split unless the leak is about the split itself.


def make_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    """Preprocessing and a logistic regression in one Pipeline, fitted only by .fit()."""
    prep = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1_000))])


NUM = ["first_week_logins", "onboarding_done", "discount"]
CAT = ["plan"]  # signup_month joins in leak 3, where it is the subject
y_all = df["churned"].to_numpy()
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=y_all, random_state=SEED)
tr, te = df.iloc[idx_tr], df.iloc[idx_te]
results: dict[str, tuple[float, float]] = {}

# %%
# Leak 1: target leakage. retention_call is recorded after the customer asks to cancel.
section("leak 1: target leakage")
m_leak = make_model(NUM + ["retention_call"], CAT).fit(tr, tr["churned"])
m_ok = make_model(NUM, CAT).fit(tr, tr["churned"])
auc_leak = auc_score(te["churned"].to_numpy(), m_leak.predict_proba(te)[:, 1])
auc_ok = auc_score(te["churned"].to_numpy(), m_ok.predict_proba(te)[:, 1])
results["target leak\n(retention_call)"] = (auc_leak, auc_ok)
print(f"test AUC with retention_call {auc_leak:.4f}   without {auc_ok:.4f}")
print("at prediction time (the signup day) no retention call has happened: the 0.9+ was never available")

# %%
# Leak 2: train/test contamination. (a) A scaler fitted on all rows: a small effect here.
section("leak 2: train/test contamination")
X_num_all = df[NUM].to_numpy(np.float64)
scaler_all = StandardScaler().fit(X_num_all)  # WRONG: sees the test rows
scaler_tr = StandardScaler().fit(X_num_all[idx_tr])
lr_a = LogisticRegression(max_iter=1_000).fit(scaler_all.transform(X_num_all[idx_tr]), y_all[idx_tr])
lr_b = LogisticRegression(max_iter=1_000).fit(scaler_tr.transform(X_num_all[idx_tr]), y_all[idx_tr])
auc_a = auc_score(y_all[idx_te], lr_a.predict_proba(scaler_all.transform(X_num_all[idx_te]))[:, 1])
auc_b = auc_score(y_all[idx_te], lr_b.predict_proba(scaler_tr.transform(X_num_all[idx_te]))[:, 1])
results["scaler fitted\non all rows"] = (auc_a, auc_b)
print(f"(a) scaler fitted on all rows: test AUC {auc_a:.4f}   fitted on train only: {auc_b:.4f}   "
      f"difference {auc_a - auc_b:+.4f}")
print(f"    means used: all rows {show(scaler_all.mean_)}   train rows {show(scaler_tr.mean_)}")
p_a = lr_a.predict_proba(scaler_all.transform(X_num_all[idx_te]))[:, 1]
p_b = lr_b.predict_proba(scaler_tr.transform(X_num_all[idx_te]))[:, 1]
print(f"    largest change in any test probability: {np.max(np.abs(p_a - p_b)):.6f}")

# (b) The same bug with a step that learns more: choosing features by their correlation with y.
gen_sel = np.random.default_rng(SEED + 1)
auc_sel_leak, auc_sel_ok = [], []
for _ in range(N_REPEATS):
    Xn = gen_sel.normal(size=(N_SMALL, N_NOISE))  # (200, 5000) pure noise: no feature predicts y
    yn = gen_sel.integers(0, 2, N_SMALL)
    a_tr, a_te = train_test_split(np.arange(N_SMALL), test_size=0.25, stratify=yn,
                                  random_state=int(gen_sel.integers(1e9)))
    sel_all = SelectKBest(f_classif, k=K_SELECT).fit(Xn, yn)  # WRONG: picks with the test rows' labels
    clf = LogisticRegression(max_iter=1_000).fit(sel_all.transform(Xn[a_tr]), yn[a_tr])
    auc_sel_leak.append(auc_score(yn[a_te], clf.predict_proba(sel_all.transform(Xn[a_te]))[:, 1]))
    pipe = Pipeline([("sel", SelectKBest(f_classif, k=K_SELECT)), ("clf", LogisticRegression(max_iter=1_000))])
    pipe.fit(Xn[a_tr], yn[a_tr])  # right: selection learns from training rows only
    auc_sel_ok.append(auc_score(yn[a_te], pipe.predict_proba(Xn[a_te])[:, 1]))
results["feature selection\non all rows"] = (float(np.mean(auc_sel_leak)), float(np.mean(auc_sel_ok)))
print(f"(b) {N_NOISE} pure-noise features, {N_SMALL} rows, keep the best {K_SELECT}; mean over {N_REPEATS} repeats:")
print(f"    selected on all rows: test AUC {np.mean(auc_sel_leak):.4f} (min {np.min(auc_sel_leak):.4f})   "
      f"selected inside the pipeline: {np.mean(auc_sel_ok):.4f}   truth: 0.5")

# %%
# Leak 3: temporal leakage. A random split mixes every cohort into train and test; the deployed model
# meets cohorts whose churn level nobody knows yet.
section("leak 3: temporal leakage")
month_idx = (df["signup_month"].str[:4].astype(int) - 2024) * 12 + df["signup_month"].str[5:].astype(int) - 1
past, future = df[month_idx < 18], df[month_idx >= 18]  # train on Jan 2024 - Jun 2025, test on Jul - Dec 2025
CAT_T = CAT + ["signup_month"]  # the cohort as a one-hot feature
m_time = make_model(NUM, CAT_T).fit(past, past["churned"])
auc_time = auc_score(future["churned"].to_numpy(), m_time.predict_proba(future)[:, 1])
# For a fair comparison, a random split of the same size as the time split:
r_tr, r_te = train_test_split(np.arange(len(df)), test_size=len(future), random_state=SEED)
m_rand = make_model(NUM, CAT_T).fit(df.iloc[r_tr], df.iloc[r_tr]["churned"])
auc_rand = auc_score(df.iloc[r_te]["churned"].to_numpy(), m_rand.predict_proba(df.iloc[r_te])[:, 1])
m_nomonth = make_model(NUM, CAT).fit(past, past["churned"])
auc_nomonth = auc_score(future["churned"].to_numpy(), m_nomonth.predict_proba(future)[:, 1])
results["random split of\ntime-ordered data"] = (auc_rand, auc_time)
print(f"train {len(past)} rows (18 months) / test {len(future)} rows (6 later months)")
print(f"random split: test AUC {auc_rand:.4f}   time split: {auc_time:.4f}   time split without signup_month: "
      f"{auc_nomonth:.4f}")
print(f"test months seen in training: random split {df.iloc[r_te]['signup_month'].isin(df.iloc[r_tr]['signup_month']).mean():.4f}   "
      f"time split {future['signup_month'].isin(past['signup_month']).mean():.4f}")

# %%
# With scikit-learn: roc_auc_score agrees with auc_score on every number above.
section("with scikit-learn")
checks = [
    (y4, s4), (y4, np.array([0.9, 0.4, 0.4, 0.3])), (y_rand, s_rand),
    (te["churned"].to_numpy(), m_leak.predict_proba(te)[:, 1]),
    (future["churned"].to_numpy(), m_time.predict_proba(future)[:, 1]),
]
for y_c, s_c in checks:
    np.testing.assert_allclose(auc_score(y_c, s_c), roc_auc_score(y_c, s_c), rtol=1e-12)
print(f"roc_auc_score matches auc_score on {len(checks)} cases (ties included): OK")
print(f"roc_auc_score(hand example) = {roc_auc_score(y4, s4):.4f}")

# %%
# Key plot: each leak's inflated score next to the honest one.
section("key plot")
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=120)
names = list(results)
leaky = [results[k][0] for k in names]
clean = [results[k][1] for k in names]
xpos = np.arange(len(names))
b1 = ax.bar(xpos - 0.2, leaky, 0.4, label="leaky evaluation", color="tab:red", hatch="//")
b2 = ax.bar(xpos + 0.2, clean, 0.4, label="honest evaluation", color="tab:blue")
for bars in (b1, b2):
    for b_ in bars:
        ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 0.01, f"{b_.get_height():.2f}", ha="center", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
ax.axhline(0.5, color="k", ls="--", lw=1, label="chance (AUC 0.5)")
ax.set_xticks(xpos, names, fontsize=8)
ax.set_ylim(0, 1.08)
ax.set(ylabel="test ROC-AUC", title="Leaks inflate scores; the scaler leak is the small exception")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
path_png = IMG / "b1_4_leak_vs_clean_auc.png"
fig.savefig(path_png, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path_png.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Set N_NOISE = 50. How much does the feature-selection leak shrink, and why?
# 2. Make the cohort effect zero (gen.normal(0, 0.0, ...)). Does the time split still score lower?
# 3. Add a feature "logins_first_90_days" (counted over the whole 90-day window). Is it a leak? Run leakage_check.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
