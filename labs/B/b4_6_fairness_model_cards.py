# %% [markdown]
# # B4.6 Fairness: group metrics, an audit and a first model card — lab
# Lesson: lessons/B/b4_6_fairness_model_cards.md
# Run: uv run python labs/B/b4_6_fairness_model_cards.py   (CPU, defaults: ~5 s; downloads 5.3 MB once)
# GPU: no

# %%
from __future__ import annotations

import hashlib
import io
import sys
import time
import urllib.request
import warnings
from datetime import date
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:  # plain `uv run`: draw to files, never open windows
    matplotlib.use("Agg")
    # matplotlib 3.11 warns when plt.show() runs under Agg; under `uv run` it is a harmless no-op
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import fairlearn
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
OUT = ROOT / "outputs" / "b4_6"  # scratch figures and the generated model card, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE only raises the number of bootstrap resamples.
GPU_SCALE = False
N_BOOT = 2000 if GPU_SCALE else 500
THRESHOLD = 0.5
VAL_FRAC, TEST_FRAC = 0.2, 0.2

# UCI Adult, the same checksummed copy as B4.1 (VERIFY licence and link, as of 2026-10).
ADULT_FILE = DATA / "adult" / "adult-all.csv"
ADULT_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
ADULT_SHA256 = "21e0ea2f925a00338929a8c86c27354c72ac1c79819bcca81c7d91c3d64218c2"
ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
                 "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country",
                 "income"]
NUMERIC = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = ["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]

# %%
section("data")


def load_adult() -> pd.DataFrame:
    """Download (once), verify and parse UCI Adult. -> dataframe (48842, 15)."""
    # from labs/B/b4_1_classification_metrics.py
    if not ADULT_FILE.exists():
        ADULT_FILE.parent.mkdir(parents=True, exist_ok=True)
        raw = urllib.request.urlopen(ADULT_URL, timeout=60).read()
        ADULT_FILE.write_bytes(raw)
        print(f"~ downloaded {len(raw):,} bytes from {ADULT_URL}")
    raw = ADULT_FILE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ADULT_SHA256:
        raise ValueError(f"{ADULT_FILE} has the wrong sha256; delete it and rerun")
    return pd.read_csv(io.BytesIO(raw), header=None, names=ADULT_COLUMNS, skipinitialspace=True)


def train_val_test_split(
    n: int, val_frac: float, test_frac: float, gen: np.random.Generator, stratify: IntArray | None = None
) -> tuple[IntArray, IntArray, IntArray]:
    """Shuffle row indices and cut them three ways. n rows -> (train, val, test) index arrays."""
    # from labs/B/b1_1_ml_workflow.py
    groups = [np.arange(n)] if stratify is None else [np.flatnonzero(stratify == c) for c in np.unique(stratify)]
    parts: list[list[IntArray]] = [[], [], []]
    for rows in groups:
        rows = gen.permutation(rows)
        n_test, n_val = round(test_frac * len(rows)), round(val_frac * len(rows))
        parts[2].append(rows[:n_test])
        parts[1].append(rows[n_test:n_test + n_val])
        parts[0].append(rows[n_test + n_val:])
    train, val, test = (gen.permutation(np.concatenate(p)) for p in parts)
    return train, val, test


adult = load_adult()
y_all = (adult["income"] == ">50K").to_numpy(dtype=np.int64)
i_tr, i_va, i_te = train_val_test_split(len(y_all), VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED), y_all)
X_tr, X_va = adult.iloc[i_tr][NUMERIC + CATEGORICAL], adult.iloc[i_va][NUMERIC + CATEGORICAL]
y_tr, y_va = y_all[i_tr], y_all[i_va]
sex_va = adult.iloc[i_va]["sex"].to_numpy()
race_va = adult.iloc[i_va]["race"].to_numpy()
print(f"train {X_tr.shape}  val {X_va.shape}  (test {len(i_te)} rows kept aside)")
for g in ["Male", "Female"]:
    m = sex_va == g
    print(f"{g:<6}: {int(m.sum())} validation rows ({m.mean():.3f})   base rate (>50K) {y_va[m].mean():.4f}")

# %%
# Worked example by hand: two groups of 10 people.
section("worked example by hand")
hand = {"A": {"y": np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]), "pred": np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])},
        "B": {"y": np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]), "pred": np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])}}
for g, d in hand.items():
    y, p = d["y"], d["pred"]
    print(f"group {g}: base rate {y.mean():.2f}  selection rate {p.mean():.2f}  TPR {p[y == 1].mean():.2f}  "
          f"FPR {p[y == 0].mean():.4f}")
sel = [d["pred"].mean() for d in hand.values()]
tpr = [d["pred"][d["y"] == 1].mean() for d in hand.values()]
fpr = [d["pred"][d["y"] == 0].mean() for d in hand.values()]
print(f"demographic parity difference {max(sel) - min(sel):.2f}   equal opportunity difference {max(tpr) - min(tpr):.2f}   "
      f"equalized odds difference max({max(tpr) - min(tpr):.2f}, {max(fpr) - min(fpr):.4f}) = "
      f"{max(max(tpr) - min(tpr), max(fpr) - min(fpr)):.2f}")

# %%
# The model: the B4.1 logistic-regression pipeline. It sees `sex` as a feature, as many real models do.
section("the model")
pre = ColumnTransformer([("num", StandardScaler(), NUMERIC), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
model = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000))]).fit(X_tr, y_tr)
s_va = model.predict_proba(X_va)[:, 1]
pred_va = (s_va >= THRESHOLD).astype(np.int64)
print(f"validation ROC-AUC {roc_auc_score(y_va, s_va):.4f}   accuracy at {THRESHOLD} {np.mean(pred_va == y_va):.4f}")


# %%
# From scratch: rates per group and the three gap metrics.
section("from scratch: group metrics")


def group_rates(y: IntArray, pred: IntArray, groups: npt.NDArray[np.object_]) -> pd.DataFrame:
    """Size, base rate, selection rate, TPR and FPR for each group."""
    rows = []
    for g in np.unique(groups):
        m = groups == g
        yy, pp = y[m], pred[m]
        rows.append({"group": g, "n": int(m.sum()), "base_rate": yy.mean(), "selection_rate": pp.mean(),
                     "TPR": pp[yy == 1].mean() if yy.sum() else np.nan,
                     "FPR": pp[yy == 0].mean() if (yy == 0).sum() else np.nan})
    return pd.DataFrame(rows).set_index("group")


def gaps(table: pd.DataFrame) -> dict[str, float]:
    """Demographic parity, equal opportunity and equalized odds differences (max minus min over groups)."""
    dp = table["selection_rate"].max() - table["selection_rate"].min()
    eop = table["TPR"].max() - table["TPR"].min()
    fpr_gap = table["FPR"].max() - table["FPR"].min()
    return {"demographic parity": float(dp), "equal opportunity": float(eop), "equalized odds": float(max(eop, fpr_gap))}


rates_sex = group_rates(y_va, pred_va, sex_va)
print(rates_sex.round(4).to_string())
gaps_sex = gaps(rates_sex)
print("differences (max - min over groups): " + "  ".join(f"{k} {v:.4f}" for k, v in gaps_sex.items()))
print(f"selection-rate ratio (female / male) {rates_sex.loc['Female', 'selection_rate'] / rates_sex.loc['Male', 'selection_rate']:.4f}")

# %%
section("with fairlearn")
np.testing.assert_allclose(demographic_parity_difference(y_va, pred_va, sensitive_features=sex_va), gaps_sex["demographic parity"])
np.testing.assert_allclose(equal_opportunity_difference(y_va, pred_va, sensitive_features=sex_va), gaps_sex["equal opportunity"])
np.testing.assert_allclose(equalized_odds_difference(y_va, pred_va, sensitive_features=sex_va), gaps_sex["equalized odds"])
mf = MetricFrame(metrics={"selection_rate": selection_rate, "TPR": true_positive_rate, "FPR": false_positive_rate},
                 y_true=y_va, y_pred=pred_va, sensitive_features=sex_va)
np.testing.assert_allclose(mf.by_group[["selection_rate", "TPR", "FPR"]].to_numpy(dtype=float),
                           rates_sex[["selection_rate", "TPR", "FPR"]].to_numpy(dtype=float))
print("demographic_parity_difference, equal_opportunity_difference, equalized_odds_difference and MetricFrame match: OK")

# %%
# Small groups: bootstrap confidence intervals for the TPR of each race group.
section("small groups need confidence intervals")
boot_gen = np.random.default_rng(SEED)
ci_rows = []
for g in np.unique(race_va):
    m = np.flatnonzero(race_va == g)
    pos = m[y_va[m] == 1]
    tprs = [pred_va[boot_gen.choice(pos, len(pos), replace=True)].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(tprs, [2.5, 97.5])
    ci_rows.append((g, len(m), len(pos), pred_va[pos].mean(), lo, hi))
for g, n_g, n_pos, t, lo, hi in sorted(ci_rows, key=lambda r: -r[1]):
    print(f"{g:<19} n {n_g:>5}  positives {n_pos:>5}  TPR {t:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  width {hi - lo:.4f}")

# %%
# Thresholds: one for everyone, versus per-group thresholds chosen to equalize one metric.
section("one threshold versus per-group thresholds")


def threshold_for_rate(s: FloatArray, target: float) -> float:
    """The threshold at which a share `target` of these scores is at or above it."""
    return float(np.quantile(s, 1 - target, method="higher"))


policies: dict[str, dict[str, float]] = {"single threshold 0.5": {"Male": THRESHOLD, "Female": THRESHOLD}}
overall_rate = pred_va.mean()  # equal selection rates, keeping the overall number of positives the same
policies["equal selection rates"] = {g: threshold_for_rate(s_va[sex_va == g], overall_rate) for g in ["Male", "Female"]}
pos_scores = {g: s_va[(sex_va == g) & (y_va == 1)] for g in ["Male", "Female"]}
overall_tpr = pred_va[y_va == 1].mean()  # equal TPRs at the overall TPR of the single threshold
policies["equal true positive rates"] = {g: threshold_for_rate(pos_scores[g], overall_tpr) for g in ["Male", "Female"]}
policy_tables = {}
for name, th in policies.items():
    pred = (s_va >= np.where(sex_va == "Male", th["Male"], th["Female"])).astype(np.int64)
    tbl = group_rates(y_va, pred, sex_va)
    policy_tables[name] = tbl
    g = gaps(tbl)
    print(f"{name:<26} thresholds M {th['Male']:.3f} F {th['Female']:.3f}   accuracy {np.mean(pred == y_va):.4f}   "
          f"selection M {tbl.loc['Male', 'selection_rate']:.3f} F {tbl.loc['Female', 'selection_rate']:.3f}   "
          f"TPR M {tbl.loc['Male', 'TPR']:.3f} F {tbl.loc['Female', 'TPR']:.3f}   "
          f"FPR M {tbl.loc['Male', 'FPR']:.3f} F {tbl.loc['Female', 'FPR']:.3f}")
    print(f"{'':<26} DP diff {g['demographic parity']:.4f}   EO diff {g['equal opportunity']:.4f}   "
          f"equalized odds diff {g['equalized odds']:.4f}")

# %%
section("fairness through unawareness: drop `sex`")
cat_nosex = [c for c in CATEGORICAL if c != "sex"]
pre2 = ColumnTransformer([("num", StandardScaler(), NUMERIC), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_nosex)])
model2 = Pipeline([("pre", pre2), ("model", LogisticRegression(max_iter=2000))]).fit(X_tr[NUMERIC + cat_nosex], y_tr)
pred2 = (model2.predict_proba(X_va[NUMERIC + cat_nosex])[:, 1] >= THRESHOLD).astype(np.int64)
g2 = gaps(group_rates(y_va, pred2, sex_va))
print(f"without `sex`: DP diff {g2['demographic parity']:.4f} (was {gaps_sex['demographic parity']:.4f})   "
      f"equalized odds diff {g2['equalized odds']:.4f} (was {gaps_sex['equalized odds']:.4f})")
rel = adult.iloc[i_va]["relationship"].to_numpy()
for r in ["Husband", "Wife"]:
    m = rel == r
    print(f"relationship = {r:<7}: {int(m.sum())} rows, share female {np.mean(sex_va[m] == 'Female'):.4f}")
proxy = LogisticRegression(max_iter=2000).fit(
    pd.get_dummies(X_tr[cat_nosex]).astype(float), (adult.iloc[i_tr]["sex"] == "Female").astype(int))
X_va_dummies = pd.get_dummies(X_va[cat_nosex]).reindex(columns=pd.get_dummies(X_tr[cat_nosex]).columns, fill_value=0).astype(float)
print(f"predicting sex from the other categorical columns: ROC-AUC {roc_auc_score(sex_va == 'Female', proxy.predict_proba(X_va_dummies)[:, 1]):.4f}")

# %%
# Key plot: TPR and FPR per group under the three threshold policies.
section("key plot: threshold policies")
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), dpi=120, sharey=True)
for ax, metric in zip(axes, ["selection_rate", "TPR", "FPR"]):
    x = np.arange(len(policy_tables))
    for k, (g, hatch) in enumerate([("Male", ""), ("Female", "//")]):
        vals = [tbl.loc[g, metric] for tbl in policy_tables.values()]
        ax.bar(x + (k - 0.5) * 0.38, vals, width=0.38, label=g, hatch=hatch, color=["tab:blue", "tab:orange"][k])
    ax.set_xticks(x, ["single\n0.5", "equal\nselection", "equal\nTPR"])
    ax.set_title({"selection_rate": "selection rate", "TPR": "true positive rate", "FPR": "false positive rate"}[metric])
    ax.set_ylim(0, 1)
axes[0].set_ylabel("rate on validation rows")
axes[0].legend(loc="upper right")
fig.suptitle("Equalizing one rate across groups moves the others")
fig.tight_layout()
fig.savefig(IMG / "b4_6_threshold_policies.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved lessons/B/img/b4_6_threshold_policies.png")

# %%
# A first model card (template: course/STYLE_GUIDE.md section 6.10), filled from the numbers above.
section("model card")
rs = rates_sex
card = f"""# Model card: Adult income classifier (B4.6 teaching model)

## Model details
Logistic regression (scikit-learn {sklearn.__version__}) on 5 standardized numeric and 7 one-hot categorical
features, threshold {THRESHOLD}. Trained {date(2026, 10, 17).isoformat()} by the course lab; not versioned for deployment.

## Intended use
Teaching: auditing a classifier's error rates across groups. Predicts whether a person in the 1994 US census
extract reported income above $50,000.

## Out-of-scope uses
Any decision about a real person: credit, hiring, pay, benefits. The data is from one country and year (1994),
and the label is a proxy for nothing a decision-maker should act on today.

## Data
UCI Adult (Becker & Kohavi, 1996), 48,842 rows, stratified 60/20/20 split; metrics below on the {len(y_va):,}
validation rows. `fnlwgt` (a sampling weight) and `education` (duplicates `education_num`) excluded; `?` kept
as a category. Men are {np.mean(sex_va == 'Male'):.0%} of rows; the base rate of >$50K is {rs.loc['Male', 'base_rate']:.1%} for men and
{rs.loc['Female', 'base_rate']:.1%} for women.

## Metrics
Validation ROC-AUC {roc_auc_score(y_va, s_va):.3f}, accuracy {np.mean(pred_va == y_va):.3f} (majority-class baseline {1 - y_va.mean():.3f}).
No test-set result reported: this is a teaching model, and the test rows are kept for P2-style final evaluation.

## Decision threshold
{THRESHOLD}, the default: no costs were defined for this teaching model. A real use would set it from costs (B4.2).

## Metrics by group
| sex | n | base rate | selection rate | TPR | FPR |
|---|---|---|---|---|---|
| Male | {rs.loc['Male', 'n']} | {rs.loc['Male', 'base_rate']:.3f} | {rs.loc['Male', 'selection_rate']:.3f} | {rs.loc['Male', 'TPR']:.3f} | {rs.loc['Male', 'FPR']:.3f} |
| Female | {rs.loc['Female', 'n']} | {rs.loc['Female', 'base_rate']:.3f} | {rs.loc['Female', 'selection_rate']:.3f} | {rs.loc['Female', 'TPR']:.3f} | {rs.loc['Female', 'FPR']:.3f} |

Demographic parity difference {gaps_sex['demographic parity']:.3f}; equal opportunity difference {gaps_sex['equal opportunity']:.3f};
equalized odds difference {gaps_sex['equalized odds']:.3f}. Women with incomes above $50K are recognized less often
(lower TPR), and men without them are flagged more often (higher FPR).

## Known limitations
Small race groups have wide intervals (TPR CI widths up to {max(r[5] - r[4] for r in ci_rows):.2f}). Removing `sex` does not remove
the gaps: `relationship` (Husband/Wife) and other columns predict sex (ROC-AUC {roc_auc_score(sex_va == 'Female', proxy.predict_proba(X_va_dummies)[:, 1]):.2f}).

## Ethical considerations
The model uses `sex` directly. Per-group thresholds can equalize one rate but move the others, and may be
unlawful in some settings (VERIFY for the jurisdiction); the choice depends on which error harms whom.
"""
(OUT / "MODEL_CARD.md").write_text(card, encoding="utf-8")
print(card.rstrip())
print(f"\nsaved outputs/b4_6/MODEL_CARD.md (fairlearn {fairlearn.__version__})")

# %%
# Try this (not graded):
# 1. Audit by race with MetricFrame. Which gap would you report, and with what interval?
# 2. Use per-group thresholds that equalize FPR instead of TPR. What happens to the selection rates?
# 3. Retrain with class_weight="balanced" (B4.4) and recompute the gaps. Do they grow or shrink?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
