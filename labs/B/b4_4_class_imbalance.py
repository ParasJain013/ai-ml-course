# %% [markdown]
# # B4.4 Class imbalance — lab
# Lesson: lessons/B/b4_4_class_imbalance.md
# Run: uv run python labs/B/b4_4_class_imbalance.py   (CPU, defaults: ~25 s)
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
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

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
OUT = ROOT / "outputs" / "b4_4"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE only makes the dataset larger.
GPU_SCALE = False
N_ROWS = 600_000 if GPU_SCALE else 60_000
POSITIVE_RATE = 0.01
N_BINS = 10
VAL_FRAC, TEST_FRAC = 0.2, 0.2

# %%
# Data: a synthetic problem with 1% positives (think fraud or a rare failure), 10 features, 5 informative.
section("data")
X_all, y_all = make_classification(n_samples=N_ROWS, n_features=10, n_informative=5, n_redundant=2,
                                   weights=[1 - POSITIVE_RATE], flip_y=0.0, class_sep=1.0, random_state=SEED)
y_all = y_all.astype(np.int64)


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


i_tr, i_va, i_te = train_val_test_split(len(y_all), VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED), y_all)
X_tr, y_tr, X_va, y_va = X_all[i_tr], y_all[i_tr], X_all[i_va], y_all[i_va]
print(f"train {X_tr.shape}: {int(y_tr.sum())} positives ({y_tr.mean():.4f})   val {X_va.shape}: {int(y_va.sum())} positives "
      f"({y_va.mean():.4f})   (test {len(i_te)} rows kept aside)")

# %%
# Worked example by hand: the prior correction for undersampling.
section("worked example by hand")
pi, beta = 0.01, 1 / 99  # true positive rate; keep each negative with probability beta (1:1 after undersampling)
p_s = 0.5  # a model trained on the balanced sample says 0.5
odds_s = p_s / (1 - p_s)
odds_true = beta * odds_s
print(f"balanced-sample prediction {p_s}: odds {odds_s:.4f} -> true odds beta x odds = {odds_true:.4f} -> "
      f"p = {odds_true / (1 + odds_true):.4f}")
for p_s in [0.9, 0.99]:
    o = beta * p_s / (1 - p_s)
    print(f"balanced-sample prediction {p_s}: corrected p = {o / (1 + o):.4f}")
print(f"always predict negative: accuracy {1 - pi:.2f}, recall 0, F1 0")


# %%
# Strategies. Each returns validation scores; the model is always a logistic regression.
def oversample(X: FloatArray, y: IntArray, gen: np.random.Generator) -> tuple[FloatArray, IntArray]:
    """Draw positives with replacement until the classes are equal in size. -> resampled X, y."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    extra = gen.choice(pos, size=len(neg) - len(pos), replace=True)
    rows = gen.permutation(np.r_[neg, pos, extra])
    return X[rows], y[rows]


def undersample(X: FloatArray, y: IntArray, gen: np.random.Generator) -> tuple[FloatArray, IntArray, float]:
    """Keep all positives and an equal number of random negatives. -> X, y, kept share of negatives (beta)."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    kept = gen.choice(neg, size=len(pos), replace=False)
    rows = gen.permutation(np.r_[pos, kept])
    return X[rows], y[rows], len(pos) / len(neg)


def correct_prior(p: FloatArray, beta: float) -> FloatArray:
    """Undo a change in class balance: multiply the odds by beta (the kept share of negatives, or 1 / positive weight)."""
    odds = beta * p / (1 - p)
    return odds / (1 + odds)


def logreg() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


section("from scratch: resampling")
gen = np.random.default_rng(SEED)
X_over, y_over = oversample(X_tr, y_tr, gen)
X_under, y_under, beta_under = undersample(X_tr, y_tr, gen)
print(f"oversampled: {len(y_over)} rows, positive share {y_over.mean():.4f} ({len(np.unique(X_over[y_over == 1], axis=0))} distinct positives)")
print(f"undersampled: {len(y_under)} rows, positive share {y_under.mean():.4f}, kept share of negatives beta = {beta_under:.5f}")
neg_tr, pos_tr = int(np.sum(y_tr == 0)), int(np.sum(y_tr == 1))
w_balanced = neg_tr / pos_tr  # class_weight="balanced" gives positives n/(2 n_pos) and negatives n/(2 n_neg): ratio n_neg/n_pos
print(f"class_weight='balanced': a positive counts {w_balanced:.2f} times as much as a negative")

models = {
    "do nothing": logreg().fit(X_tr, y_tr),
    "class weights": LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr, y_tr),
    "oversample": logreg().fit(X_over, y_over),
    "undersample": logreg().fit(X_under, y_under),
}
scores = {name: m.predict_proba(X_va)[:, 1] for name, m in models.items()}

# Threshold tuning: keep the plain model, choose the threshold that maximizes F1 on out-of-fold training predictions.
oof = cross_val_predict(logreg(), X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
cands = np.unique(oof)
f1s = [f1_score(y_tr, (oof >= t).astype(int)) for t in cands[::10]]
t_best = float(cands[::10][int(np.argmax(f1s))])
print(f"threshold tuning: F1-maximizing threshold on out-of-fold training predictions = {t_best:.4f}")

# %%
section("comparison on the validation set")


def f1_at(y: IntArray, p: FloatArray, t: float) -> float:
    """F1 at threshold t, from the confusion-matrix counts (B4.1)."""
    pred = p >= t
    tp, fp, fn = np.sum(pred & (y == 1)), np.sum(pred & (y == 0)), np.sum(~pred & (y == 1))
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


rows_out = []
for name, p in list(scores.items()) + [("threshold tuning", scores["do nothing"])]:
    t = t_best if name == "threshold tuning" else 0.5
    pred = p >= t
    rows_out.append((name, t, float(np.mean(pred == y_va)), f1_at(y_va, p, t), float(np.mean(pred[y_va == 1])),
                     average_precision_score(y_va, p), roc_auc_score(y_va, p), brier_score_loss(y_va, p), float(p.mean())))
print(f"{'strategy':<17} {'thresh':>6} {'accuracy':>8} {'F1':>6} {'recall':>6} {'AP':>6} {'ROC-AUC':>7} {'Brier':>7} {'mean p':>7}")
for name, t, acc, f1, rec, ap, auc, br, mp in rows_out:
    print(f"{name:<17} {t:>6.3f} {acc:>8.4f} {f1:>6.4f} {rec:>6.4f} {ap:>6.4f} {auc:>7.4f} {br:>7.4f} {mp:>7.4f}")
print(f"validation prevalence {y_va.mean():.4f}")

# %%
section("prior correction repairs the probabilities")
fixes = {"class weights": 1 / w_balanced, "oversample": 1 / w_balanced, "undersample": beta_under}
for name, b in fixes.items():
    p_fix = correct_prior(scores[name], b)
    print(f"{name:<14} Brier {brier_score_loss(y_va, scores[name]):.4f} -> {brier_score_loss(y_va, p_fix):.4f}   "
          f"mean p {scores[name].mean():.4f} -> {p_fix.mean():.4f}   ROC-AUC unchanged {roc_auc_score(y_va, p_fix):.4f}")

# %%
# With scikit-learn: duplicating each positive k times is the same as giving it sample_weight k.
section("with scikit-learn: oversampling equals weighting")
k = 5
dup_rows = np.r_[np.arange(len(y_tr)), np.repeat(np.flatnonzero(y_tr == 1), k - 1)]
m_dup = logreg().fit(X_tr[dup_rows], y_tr[dup_rows])
m_w = logreg().fit(X_tr, y_tr, sample_weight=np.where(y_tr == 1, float(k), 1.0))
np.testing.assert_allclose(m_dup.coef_, m_w.coef_, rtol=1e-3, atol=1e-4)  # lbfgs stops at a tolerance, not the exact optimum
print(f"each positive duplicated {k}x versus sample_weight {k}: same coefficients (to solver tolerance): OK")
np.testing.assert_allclose(f1_at(y_va, scores["do nothing"], 0.5), f1_score(y_va, scores["do nothing"] >= 0.5))
print("from-scratch F1 matches sklearn.metrics.f1_score: OK")

# %%
# Key plot: reliability diagrams for the plain model and the three rebalanced ones. With 1% positives,
# equal-width bins above 0.2 hold a handful of rows, so we use equal-count bins (each holds a tenth of the
# validation rows) and log scales.
section("key plot: calibration after rebalancing")


def reliability_quantile(y: IntArray, p: FloatArray, n_bins: int) -> tuple[FloatArray, FloatArray]:
    """Mean predicted probability and observed positive rate in n_bins bins of equal size (sorted by p)."""
    order = np.argsort(p, kind="stable")
    chunks = np.array_split(order, n_bins)
    return np.array([p[c].mean() for c in chunks]), np.array([y[c].mean() for c in chunks])


fig, ax = plt.subplots(figsize=(5.5, 5), dpi=120)
ax.plot([1e-4, 1], [1e-4, 1], ":", color="gray", label="perfectly calibrated")
for (name, p), marker in zip(scores.items(), ["o", "s", "^", "v"]):
    mp, fp_ = reliability_quantile(y_va, p, N_BINS)
    ax.plot(mp, np.maximum(fp_, 1e-4), marker=marker, linestyle="-", label=f"{name} (Brier {brier_score_loss(y_va, p):.4f})")
    if name in ("do nothing", "class weights"):
        print(f"{name:<14} top bin: mean predicted {mp[-1]:.4f}, observed {fp_[-1]:.4f};  "
              f"bottom bin: mean predicted {mp[0]:.5f}, observed {fp_[0]:.5f}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-4, 1)
ax.set_ylim(1e-4, 1)
ax.set_xlabel("mean predicted probability in bin (log scale)")
ax.set_ylabel("observed share of positives in bin (log scale)")
ax.set_title("Rebalancing inflates every probability")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(IMG / "b4_4_rebalancing_calibration.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved lessons/B/img/b4_4_rebalancing_calibration.png (zero observed rates drawn at 1e-4)")

# %%
# Try this (not graded):
# 1. Undersample to 1:5 instead of 1:1 (beta = 5 n_pos / n_neg). How do AP and Brier change?
# 2. Replace logistic regression by LightGBM with scale_pos_weight=99. Does rebalancing change its ranking (AP)?
# 3. Choose the threshold by expected cost (B4.2) with c_FN = 50 c_FP instead of by F1.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
