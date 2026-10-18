# %% [markdown]
# # B5.6 Anomaly detection — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b5_6_anomaly_detection_ex2.py   (CPU, ~10 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()
N_REF, N_ANOM, K_NN, BUDGET, N_BOOT = 200, 15, 5, 15, 1_000


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
x = np.array([3.0, 4, 4, 5, 6, 40])
z = (x - x.mean()) / x.std()
q1, q3 = np.quantile(x, [0.25, 0.75])
iqr = q3 - q1
print(f"(a) mean {x.mean():.4f}  std {x.std():.4f}  z(40) {z[-1]:.4f}  ceiling sqrt(5) {np.sqrt(5):.4f}")
print(f"    Q1 {q1:.4f}  Q3 {q3:.4f}  IQR {iqr:.4f}  fences [{q1 - 1.5 * iqr:.4f}, {q3 + 1.5 * iqr:.4f}]")
H7 = float(np.sum(1 / np.arange(1, 8)))
c8 = 2 * H7 - 2 * 7 / 8
print(f"(b) H(7) {H7:.4f}  c(8) {c8:.4f}  s(h=1.5) {2 ** (-1.5 / c8):.4f}  s(h=c) {2 ** (-c8 / c8):.4f}")
P = np.array([[0.0, 0], [2, 0], [0, 2], [2, 2], [1, 1], [6, 1]])
D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
np.fill_diagonal(D, np.inf)
print(f"(c) k = 2 distance scores: {np.round(np.sort(D, axis=1)[:, 1], 4).tolist()}")
print(f"(d) contamination 0.02 on 5000 rows flags {int(0.02 * 5000)}; a budget of 50/day is contamination {50 / 5000:.3f}")

# %%
# Novelty detection: fit on a clean reference set of benign rows only, score a new batch.
section("data: reference set and new batch")
bc = load_breast_cancer()
gen = np.random.default_rng(SEED)
benign = gen.permutation(np.flatnonzero(bc.target == 1))
malignant = gen.permutation(np.flatnonzero(bc.target == 0))
ref_rows = benign[:N_REF]
new_rows = np.r_[benign[N_REF:], malignant[:N_ANOM]]
dirty_ref_rows = np.r_[ref_rows, malignant[N_ANOM:2 * N_ANOM]]  # 15 *other* malignant rows slipped into the reference
y_new = (bc.target[new_rows] == 0).astype(np.int64)
print(f"reference {len(ref_rows)} benign rows; new batch {len(new_rows)} rows, {int(y_new.sum())} anomalies "
      f"(prevalence {y_new.mean():.4f})")


def robust_scaler(X_ref: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Median and IQR per feature, fitted on the reference rows only. X_ref (m, d) -> (d,), (d,)."""
    q1, q3 = np.quantile(X_ref, [0.25, 0.75], axis=0)
    return np.median(X_ref, axis=0), q3 - q1


def knn_novelty(X_ref: FloatArray, X_new: FloatArray, k: int) -> FloatArray:
    """Distance from each new row to its k-th nearest reference row. (m, d), (n, d) -> (n,)."""
    D2 = (X_new**2).sum(1)[:, None] + (X_ref**2).sum(1)[None, :] - 2 * X_new @ X_ref.T  # (n, m)
    return np.sqrt(np.sort(np.maximum(D2, 0.0), axis=1)[:, k - 1])


def bootstrap_ci(y: IntArray, s: FloatArray, gen: np.random.Generator) -> tuple[float, float]:
    """95% percentile bootstrap CI of average precision."""
    aps = []
    for _ in range(N_BOOT):
        i = gen.integers(0, len(y), len(y))
        if y[i].sum():
            aps.append(average_precision_score(y[i], s[i]))
    return float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))


def evaluate(ref: IntArray, label: str) -> None:
    med, iqr = robust_scaler(bc.data[ref])
    X_ref, X_new = (bc.data[ref] - med) / iqr, (bc.data[new_rows] - med) / iqr
    s_knn = knn_novelty(X_ref, X_new, K_NN)
    nn = NearestNeighbors(n_neighbors=K_NN).fit(X_ref)
    np.testing.assert_allclose(s_knn, nn.kneighbors(X_new)[0][:, -1], rtol=1e-6, atol=1e-8)
    iso = IsolationForest(n_estimators=200, random_state=SEED).fit(X_ref)
    s_iso = -iso.score_samples(X_new)
    for name, s in [(f"k-NN to reference (k={K_NN})", s_knn), ("IsolationForest", s_iso)]:
        lo, hi = bootstrap_ci(y_new, s, np.random.default_rng(SEED))
        top = np.argsort(-s, kind="stable")[:BUDGET]
        print(f"{label:<16} {name:<26} AP {average_precision_score(y_new, s):.4f} [{lo:.4f}, {hi:.4f}]  "
              f"P@{BUDGET} {y_new[top].mean():.4f}")


section("novelty detection: clean versus contaminated reference")
evaluate(ref_rows, "clean ref")
evaluate(dirty_ref_rows, "contaminated ref")
print("k-NN novelty score matches sklearn NearestNeighbors: OK")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
