# %% [markdown]
# # B4.1 Classification metrics — lab
# Lesson: lessons/B/b4_1_classification_metrics.md
# Run: uv run python labs/B/b4_1_classification_metrics.py   (CPU, defaults: ~10 s; downloads 5.3 MB once)
# GPU: no

# %%
from __future__ import annotations

import hashlib
import io
import sys
import time
import urllib.request
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB
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
OUT = ROOT / "outputs" / "b4_1"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(a: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(a, 4) + 0.0).tolist())


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; the data is already full size.
GPU_SCALE = False
N_BINS = 10  # bins of the reliability diagram
RARE_PREVALENCE = 0.02  # positive share in the "rare positives" version of the validation set
VAL_FRAC, TEST_FRAC = 0.2, 0.2

# UCI Adult (Census Income; Becker & Kohavi, 1996), CC BY 4.0 (VERIFY licence, as of 2026-10). The copy in
# J. Brownlee's Datasets repository joins adult.data and adult.test (48,842 rows) with the labels
# written the same way in both; the checksum guarantees everyone runs on the same file.
ADULT_DIR = DATA / "adult"
ADULT_FILE = ADULT_DIR / "adult-all.csv"
ADULT_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
ADULT_SHA256 = "21e0ea2f925a00338929a8c86c27354c72ac1c79819bcca81c7d91c3d64218c2"
ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
                 "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country",
                 "income"]
NUMERIC = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = ["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]

# %%
# Data: download once (idempotent), check the checksum, parse. "?" marks a missing value; we keep it
# as its own category. fnlwgt is a census sampling weight, not a property of the person, and
# education duplicates education_num, so both are left out.
section("data")


def load_adult() -> pd.DataFrame:
    """Download (once), verify and parse UCI Adult. -> dataframe (48842, 15)."""
    if not ADULT_FILE.exists():
        ADULT_DIR.mkdir(parents=True, exist_ok=True)
        raw = urllib.request.urlopen(ADULT_URL, timeout=60).read()
        ADULT_FILE.write_bytes(raw)
        print(f"~ downloaded {len(raw):,} bytes from {ADULT_URL}")  # first run only, hence ~
    digest = hashlib.sha256(ADULT_FILE.read_bytes()).hexdigest()
    if digest != ADULT_SHA256:
        raise ValueError(f"{ADULT_FILE} has sha256 {digest}, expected {ADULT_SHA256}; delete it and rerun")
    df = pd.read_csv(io.BytesIO(ADULT_FILE.read_bytes()), header=None, names=ADULT_COLUMNS,
                     skipinitialspace=True)
    return df


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
X_all = adult[NUMERIC + CATEGORICAL]
y_all = (adult["income"] == ">50K").to_numpy(dtype=np.int64)
i_tr, i_va, i_te = train_val_test_split(len(y_all), VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED), y_all)
X_tr, X_va, y_tr, y_va = X_all.iloc[i_tr], X_all.iloc[i_va], y_all[i_tr], y_all[i_va]
print(f"{ADULT_FILE.relative_to(ROOT)}: checksum OK, {len(adult)} rows")
print(f"positive (>50K) share: all {y_all.mean():.4f}  train {y_tr.mean():.4f}  val {y_va.mean():.4f}")
print(f"train {X_tr.shape}  val {X_va.shape}  (test {len(i_te)} rows kept aside)")
print(f"rows with a '?' somewhere: {int((adult[CATEGORICAL] == '?').any(axis=1).sum())}")

# %%
# Worked example by hand: eight scored rows.
section("worked example by hand")
y8 = np.array([1, 1, 0, 1, 0, 0, 1, 0])
s8 = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.1])


def confusion(y: IntArray, pred: IntArray) -> tuple[int, int, int, int]:
    """(TN, FP, FN, TP) for 0/1 labels and 0/1 predictions."""
    return (int(np.sum((y == 0) & (pred == 0))), int(np.sum((y == 0) & (pred == 1))),
            int(np.sum((y == 1) & (pred == 0))), int(np.sum((y == 1) & (pred == 1))))


def prf(y: IntArray, pred: IntArray) -> tuple[float, float, float]:
    """Precision, recall and F1 of the positive class."""
    _, fp, fn, tp = confusion(y, pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


tn, fp, fn, tp = confusion(y8, (s8 >= 0.5).astype(int))
p8, r8, f8 = prf(y8, (s8 >= 0.5).astype(int))
print(f"threshold 0.5: TN {tn}  FP {fp}  FN {fn}  TP {tp}   accuracy {(tp + tn) / 8:.4f}")
print(f"precision {p8:.4f}  recall {r8:.4f}  F1 {f8:.4f}  FPR {fp / (fp + tn):.4f}")
pos, neg = s8[y8 == 1], s8[y8 == 0]
wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
print(f"pairs (positive, negative) ranked correctly: {wins:g} of {len(pos) * len(neg)} -> AUC {wins / (len(pos) * len(neg)):.4f}")
order = np.argsort(-s8)
hits = np.cumsum(y8[order])
precision_at_hits = hits[y8[order] == 1] / (np.flatnonzero(y8[order] == 1) + 1)
print(f"precision at each positive, in score order: {show(precision_at_hits)} -> AP {precision_at_hits.mean():.4f}")
print(f"Brier score mean((s - y)^2) = {np.mean((s8 - y8) ** 2):.4f}")

# %%
# From scratch 1: the threshold metrics on the validation set, for a logistic-regression pipeline.
section("from scratch: confusion matrix, precision, recall, F1")
pre = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
logreg = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000))]).fit(X_tr, y_tr)
s_va = logreg.predict_proba(X_va)[:, 1]  # (n_val,) scores = predicted P(>50K)
for t in [0.3, 0.5, 0.7]:
    pred = (s_va >= t).astype(int)
    tn, fp, fn, tp = confusion(y_va, pred)
    pr, rc, f1 = prf(y_va, pred)
    print(f"threshold {t}: TN {tn:>5} FP {fp:>4} FN {fn:>4} TP {tp:>4}   accuracy {(tp + tn) / len(y_va):.4f}  "
          f"precision {pr:.4f}  recall {rc:.4f}  F1 {f1:.4f}")
print(f"always predict <=50K: accuracy {1 - y_va.mean():.4f}, recall 0")

# %%
# From scratch 2: the ROC curve and its area, and the PR curve with average precision.
section("from scratch: ROC-AUC and average precision")


def roc_points(y: IntArray, s: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """FPR, TPR and threshold at every distinct score, highest first. -> three (k + 1,) arrays."""
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s) - 1]  # last row of each run of equal scores
    tps = np.cumsum(y_sorted)[last]  # positives at or above each threshold
    fps = (last + 1) - tps  # negatives at or above each threshold
    tpr = np.r_[0.0, tps / y.sum()]
    fpr = np.r_[0.0, fps / (len(y) - y.sum())]
    return fpr, tpr, np.r_[np.inf, s_sorted[last]]


def auc_trapezoid(x: FloatArray, y: FloatArray) -> float:
    """Area under a curve through the points (x, y), x non-decreasing."""
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def average_precision(y: IntArray, s: FloatArray) -> float:
    """Sum over thresholds of (recall gained) x (precision at that threshold)."""
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s) - 1]
    tps = np.cumsum(y_sorted)[last]
    precision = tps / (last + 1)
    recall = tps / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


fpr_m, tpr_m, thr_m = roc_points(y_va, s_va)
auc_m, ap_m = auc_trapezoid(fpr_m, tpr_m), average_precision(y_va, s_va)
print(f"ROC curve: {len(fpr_m)} points   ROC-AUC {auc_m:.4f}   average precision (PR-AUC) {ap_m:.4f}   "
      f"prevalence {y_va.mean():.4f}")
for target in [0.5, 0.8, 0.95]:
    k = int(np.searchsorted(tpr_m, target))
    print(f"to catch {target:.0%} of the >50K rows: threshold {thr_m[k]:.4f}, FPR {fpr_m[k]:.4f}")
print(f"8-row example with the same functions: AUC {auc_trapezoid(*roc_points(y8, s8)[:2]):.4f}  "
      f"AP {average_precision(y8, s8):.4f}")

# %%
# Rare positives: keep every validation negative but only enough positives for a 2% prevalence.
section("ROC-AUC versus PR-AUC when positives are rare")
pos_idx, neg_idx = np.flatnonzero(y_va == 1), np.flatnonzero(y_va == 0)
n_pos_rare = round(RARE_PREVALENCE / (1 - RARE_PREVALENCE) * len(neg_idx))
rare = np.r_[neg_idx, np.random.default_rng(SEED).choice(pos_idx, n_pos_rare, replace=False)]
y_r, s_r = y_va[rare], s_va[rare]
auc_r = auc_trapezoid(*roc_points(y_r, s_r)[:2])
ap_r = average_precision(y_r, s_r)
print(f"original: {len(pos_idx)} positives / {len(y_va)} rows   ROC-AUC {auc_m:.4f}   AP {ap_m:.4f}")
print(f"rare:     {n_pos_rare} positives / {len(rare)} rows   ROC-AUC {auc_r:.4f}   AP {ap_r:.4f}")
for name, yy, ss in [("original", y_va, s_va), ("rare", y_r, s_r)]:
    pr, rc, _ = prf(yy, (ss >= 0.5).astype(int))
    print(f"{name:<8} at threshold 0.5: precision {pr:.4f}  recall {rc:.4f}")

# %%
# From scratch 3: reliability diagram and Brier score, for three models.
section("from scratch: calibration")


def reliability(y: IntArray, p: FloatArray, n_bins: int) -> tuple[FloatArray, FloatArray, IntArray]:
    """Mean predicted probability, observed positive rate and count per equal-width bin (empty bins dropped)."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)  # bin k holds [k/n, (k+1)/n); p = 1 goes in the last
    counts = np.bincount(bins, minlength=n_bins)
    keep = counts > 0
    mean_p = np.bincount(bins, weights=p, minlength=n_bins)[keep] / counts[keep]
    frac_pos = np.bincount(bins, weights=y, minlength=n_bins)[keep] / counts[keep]
    return mean_p, frac_pos, counts[keep]


def brier(y: IntArray, p: FloatArray) -> float:
    """Mean squared error of the probabilities."""
    return float(np.mean((p - y) ** 2))


gnb = Pipeline([("pre", ColumnTransformer([("num", StandardScaler(), NUMERIC),
                                           ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                            CATEGORICAL)])),
                ("model", GaussianNB())]).fit(X_tr, y_tr)
cal = CalibratedClassifierCV(gnb, method="isotonic", cv=5).fit(X_tr, y_tr)  # refits GNB on 4/5, calibrates on 1/5
models = {"logistic regression": s_va, "Gaussian naive Bayes": gnb.predict_proba(X_va)[:, 1],
          "GNB + isotonic calibration": cal.predict_proba(X_va)[:, 1]}
curves = {}
for name, p in models.items():
    mean_p, frac_pos, counts = reliability(y_va, p, N_BINS)
    curves[name] = (mean_p, frac_pos)
    gap = np.abs(mean_p - frac_pos)
    print(f"{name:<27} Brier {brier(y_va, p):.4f}   ROC-AUC {roc_auc_score(y_va, p):.4f}   "
          f"share of p < 0.01 or > 0.99: {np.mean((p < 0.01) | (p > 0.99)):.4f}   largest bin gap {gap.max():.4f}")
mean_p, frac_pos, counts = reliability(y_va, models["Gaussian naive Bayes"], N_BINS)
print("Gaussian NB, bin by bin (mean predicted -> observed, count):")
print("   " + "  ".join(f"{a:.2f}->{b:.2f} ({c})" for a, b, c in zip(mean_p, frac_pos, counts)))

# %%
# With scikit-learn: the same numbers from sklearn.metrics and sklearn.calibration.
section("with scikit-learn")
pred05 = (s_va >= 0.5).astype(int)
np.testing.assert_array_equal(confusion_matrix(y_va, pred05).ravel(), confusion(y_va, pred05))
sk_p, sk_r, sk_f, _ = precision_recall_fscore_support(y_va, pred05, average="binary")
np.testing.assert_allclose([sk_p, sk_r, sk_f], prf(y_va, pred05))
print("confusion_matrix and precision_recall_fscore_support match: OK")
sk_fpr, sk_tpr, _ = roc_curve(y_va, s_va, drop_intermediate=False)
np.testing.assert_allclose(sk_fpr, fpr_m)
np.testing.assert_allclose(sk_tpr, tpr_m)
np.testing.assert_allclose(roc_auc_score(y_va, s_va), auc_m)
np.testing.assert_allclose(average_precision_score(y_va, s_va), ap_m)
np.testing.assert_allclose([roc_auc_score(y8, s8), average_precision_score(y8, s8)],
                           [auc_trapezoid(*roc_points(y8, s8)[:2]), average_precision(y8, s8)])
print(f"roc_curve, roc_auc_score ({roc_auc_score(y_va, s_va):.4f}) and average_precision_score "
      f"({average_precision_score(y_va, s_va):.4f}) match: OK")
for name, p in models.items():
    frac_sk, mean_sk = calibration_curve(y_va, p, n_bins=N_BINS, strategy="uniform")
    np.testing.assert_allclose(mean_sk, curves[name][0])
    np.testing.assert_allclose(frac_sk, curves[name][1])
    np.testing.assert_allclose(brier_score_loss(y_va, p), brier(y_va, p))
print("calibration_curve and brier_score_loss match for all three models: OK")

# %%
# Key plot: reliability diagrams.
section("key plot: reliability diagram")
fig, ax = plt.subplots(figsize=(5.5, 5), dpi=120)
ax.plot([0, 1], [0, 1], ":", color="gray", label="perfectly calibrated")
for (name, (mp, fp_)), marker in zip(curves.items(), ["o", "s", "^"]):
    ax.plot(mp, fp_, marker=marker, linestyle="-", label=f"{name} (Brier {brier(y_va, models[name]):.3f})")
ax.set_xlabel("mean predicted probability in bin")
ax.set_ylabel("observed share of >50K in bin")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Naive Bayes says 0 or 1; calibration brings it to the diagonal")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(IMG / "b4_1_reliability_diagram.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), dpi=120)
ax1.plot(fpr_m, tpr_m, label=f"all rows (AUC {auc_m:.3f})")
fpr_r, tpr_r, _ = roc_points(y_r, s_r)
ax1.plot(fpr_r, tpr_r, "--", label=f"2% positives (AUC {auc_r:.3f})")
ax1.plot([0, 1], [0, 1], ":", color="gray")
ax1.set_xlabel("false positive rate")
ax1.set_ylabel("true positive rate (recall)")
ax1.set_title("ROC barely moves")
ax1.legend()
for (yy, ss, label, style) in [(y_va, s_va, f"all rows (AP {ap_m:.3f})", "-"), (y_r, s_r, f"2% positives (AP {ap_r:.3f})", "--")]:
    order = np.argsort(-ss)
    tp_cum = np.cumsum(yy[order])
    ax2.plot(tp_cum / yy.sum(), tp_cum / np.arange(1, len(yy) + 1), style, label=label)
ax2.set_xlabel("recall")
ax2.set_ylabel("precision")
ax2.set_title("PR collapses")
ax2.legend()
fig.tight_layout()
fig.savefig(OUT / "roc_pr_rare.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved lessons/B/img/b4_1_reliability_diagram.png and outputs/b4_1/roc_pr_rare.png")

# %%
# Try this (not graded):
# 1. Use method="sigmoid" (Platt scaling) in CalibratedClassifierCV. Is the Brier score better or worse than isotonic?
# 2. Compute the F1-maximizing threshold on validation. How far is it from 0.5, and why?
# 3. Replace logistic regression by a fully grown DecisionTreeClassifier and draw its reliability diagram.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
