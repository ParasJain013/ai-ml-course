# %% [markdown]
# # B4.1 Classification metrics — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b4_1_classification_metrics_ex2.py   (CPU, ~15 s; needs the B4.1 lab's download)

# %%
from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()
N_BOOT = 1000
ADULT_FILE = Path(__file__).resolve().parents[3] / "data" / "adult" / "adult-all.csv"
ADULT_SHA256 = "21e0ea2f925a00338929a8c86c27354c72ac1c79819bcca81c7d91c3d64218c2"
ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
                 "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country",
                 "income"]
NUMERIC = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = ["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
y6 = np.array([1, 0, 1, 1, 0, 0])
s6 = np.array([0.95, 0.85, 0.6, 0.45, 0.3, 0.2])
pred = (s6 >= 0.5).astype(int)
tp, fp = int(np.sum((pred == 1) & (y6 == 1))), int(np.sum((pred == 1) & (y6 == 0)))
fn, tn = int(np.sum((pred == 0) & (y6 == 1))), int(np.sum((pred == 0) & (y6 == 0)))
prec, rec = tp / (tp + fp), tp / (tp + fn)
print(f"(a) TP {tp} FP {fp} FN {fn} TN {tn}   accuracy {(tp + tn) / 6:.4f}  precision {prec:.4f}  recall {rec:.4f}  "
      f"F1 {2 * prec * rec / (prec + rec):.4f}  FPR {fp / (fp + tn):.4f}")
print(f"(b) ROC-AUC {roc_auc_score(y6, s6):.4f}   AP {average_precision_score(y6, s6):.4f}")
print(f"(c) Brier {np.mean((s6 - y6) ** 2):.4f}")
print(f"(d) with 10x negatives (expected counts): TP {tp} FP {10 * fp} -> precision {tp / (tp + 10 * fp):.4f}; "
      f"recall {rec:.4f} and FPR {10 * fp / (10 * (fp + tn)):.4f} unchanged")

# %%
section("data and model (as in the lab)")
if not ADULT_FILE.exists():
    raise FileNotFoundError("Run labs/B/b4_1_classification_metrics.py once first: it downloads UCI Adult.")
if hashlib.sha256(ADULT_FILE.read_bytes()).hexdigest() != ADULT_SHA256:
    raise ValueError(f"{ADULT_FILE} has the wrong checksum; delete it and rerun the lab")
adult = pd.read_csv(io.BytesIO(ADULT_FILE.read_bytes()), header=None, names=ADULT_COLUMNS, skipinitialspace=True)
X_all, y_all = adult[NUMERIC + CATEGORICAL], (adult["income"] == ">50K").to_numpy(dtype=np.int64)
gen = np.random.default_rng(SEED)  # the lab's stratified 60/20/20 split
parts: list[list[IntArray]] = [[], [], []]
for c in np.unique(y_all):
    rows = gen.permutation(np.flatnonzero(y_all == c))
    n_te, n_va = round(0.2 * len(rows)), round(0.2 * len(rows))
    parts[2].append(rows[:n_te])
    parts[1].append(rows[n_te:n_te + n_va])
    parts[0].append(rows[n_te + n_va:])
i_tr, i_va, _ = (gen.permutation(np.concatenate(p)) for p in parts)
pre = ColumnTransformer([("num", StandardScaler(), NUMERIC), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
model = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000))]).fit(X_all.iloc[i_tr], y_all[i_tr])
s, y = model.predict_proba(X_all.iloc[i_va])[:, 1], y_all[i_va]
print(f"val rows {len(y)}   positives {int(y.sum())}")

# %%
section("(a) rank formula for ROC-AUC")


def auc_rank(y: IntArray, s: FloatArray) -> float:
    """ROC-AUC = (sum of positive ranks - P(P+1)/2) / (P N), average ranks for ties."""
    ranks = rankdata(s)  # 1 = lowest score; ties get the average rank
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


np.testing.assert_allclose(auc_rank(y, s), roc_auc_score(y, s))
np.testing.assert_allclose(auc_rank(y6, s6), roc_auc_score(y6, s6))
print(f"rank formula {auc_rank(y, s):.6f} = roc_auc_score {roc_auc_score(y, s):.6f}: OK")

# %%
section("(b) bootstrap confidence intervals")
boot_rng = np.random.default_rng(SEED)
aucs, aps = np.empty(N_BOOT), np.empty(N_BOOT)
for b in range(N_BOOT):
    idx = boot_rng.integers(0, len(y), len(y))  # resample validation rows with replacement
    aucs[b], aps[b] = roc_auc_score(y[idx], s[idx]), average_precision_score(y[idx], s[idx])
for name, point, dist in [("ROC-AUC", roc_auc_score(y, s), aucs), ("AP", average_precision_score(y, s), aps)]:
    lo, hi = np.percentile(dist, [2.5, 97.5])
    print(f"{name:<8} {point:.4f}   95% percentile bootstrap CI [{lo:.4f}, {hi:.4f}]   width {hi - lo:.4f}")

# %%
section("(c) thresholds")
order = np.argsort(-s, kind="stable")
s_sorted, y_sorted = s[order], y[order]
last = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s) - 1]
tps = np.cumsum(y_sorted)[last]
precision, recall, thresholds = tps / (last + 1), tps / y.sum(), s_sorted[last]
f1 = 2 * precision * recall / (precision + recall)
k = int(np.argmax(f1))
print(f"max F1 {f1[k]:.4f} at threshold {thresholds[k]:.4f}: precision {precision[k]:.4f}  recall {recall[k]:.4f}")
k80 = int(np.flatnonzero(recall >= 0.8)[0])  # thresholds fall as k grows, so the first index is the highest one
print(f"highest threshold with recall >= 0.8: {thresholds[k80]:.4f}: precision {precision[k80]:.4f}  recall {recall[k80]:.4f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
