# %% [markdown]
# # B1.4 Data leakage — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b1_4_data_leakage_ex2.py   (CPU, ~2 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()
N_ROWS, N_POSTCODES, N_FOLDS = 3_000, 500, 5


def section(name: str) -> None:
    print(f"\n== {name} ==")


def auc_score(y: IntArray, score: FloatArray) -> float:
    """ROC-AUC by average ranks (Mann-Whitney U). y (n,) in {0, 1}, score (n,) -> scalar."""
    # same idea as labs/B/b1_4_data_leakage.py, with ties ranked through np.unique
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    first = np.cumsum(counts) - counts + 1  # rank of the first element of each run of ties
    ranks = (first + (counts - 1) / 2)[inv]  # average rank of each run
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# %%
# Exercise 1 check.
section("exercise 1 check")
s1 = np.array([0.8, 0.7, 0.7, 0.5, 0.3, 0.2])
y1 = np.array([1, 1, 0, 0, 1, 0])
pairs = [(float(p), float(q)) for p in s1[y1 == 1] for q in s1[y1 == 0]]
u = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p, q in pairs)
print(f"(a) pairs {pairs}")
print(f"    U = {u} of {len(pairs)} -> AUC {u / len(pairs):.4f}   rank formula {auc_score(y1, s1):.4f}   "
      f"roc_auc_score {roc_auc_score(y1, s1):.4f}")
tr_b, te_b = np.array([10.0, 12.0, 14.0, 16.0]), np.array([30.0, 32.0])
all_b = np.concatenate([tr_b, te_b])
print(f"(b) train-fitted: mean {tr_b.mean()} sd {tr_b.std():.4f} -> test {np.round((te_b - tr_b.mean()) / tr_b.std(), 4).tolist()}")
print(f"    all-fitted:   mean {all_b.mean()} sd {all_b.std():.4f} -> test {np.round((te_b - all_b.mean()) / all_b.std(), 4).tolist()}")

# %%
# Exercise 2: target encoding a category with no real effect.
section("exercise 2: data")
gen = np.random.default_rng(SEED)
postcode = gen.integers(0, N_POSTCODES, N_ROWS)  # (n,) no effect on y
x = gen.normal(0, 1, N_ROWS)  # (n,) weakly useful
y = (gen.random(N_ROWS) < 1 / (1 + np.exp(-(-0.5 + 0.5 * x)))).astype(int)
perm = gen.permutation(N_ROWS)
tr, te = perm[: int(0.75 * N_ROWS)], perm[int(0.75 * N_ROWS):]
print(f"rows {N_ROWS}  postcodes {N_POSTCODES}  positive rate {y.mean():.4f}  "
      f"train {len(tr)}  test {len(te)}  training rows per postcode ~{len(tr) / N_POSTCODES:.1f}")


def fit_target_means(cat: IntArray, y: IntArray) -> tuple[dict[int, float], float]:
    """Label mean of each category, and the global mean for unseen ones."""
    means = {int(c): float(y[cat == c].mean()) for c in np.unique(cat)}
    return means, float(y.mean())


def encode(cat: IntArray, means: dict[int, float], default: float) -> FloatArray:
    """Replace each category by its learned label mean. (n,) -> (n,)."""
    return np.array([means.get(int(c), default) for c in cat])


def out_of_fold_encode(cat: IntArray, y: IntArray, n_folds: int, gen: np.random.Generator) -> FloatArray:
    """Encode each fold with means learned on the other folds. (n,) -> (n,)."""
    folds = gen.permutation(len(cat)) % n_folds  # (n,) fold id of each row
    out = np.empty(len(cat))
    for f in range(n_folds):
        means, default = fit_target_means(cat[folds != f], y[folds != f])
        out[folds == f] = encode(cat[folds == f], means, default)
    return out


def fit_and_score(f_tr: FloatArray, f_te: FloatArray) -> tuple[float, float]:
    """Logistic regression on [x, encoded postcode]. -> (train AUC, test AUC)."""
    A_tr = np.column_stack([x[tr], f_tr])
    A_te = np.column_stack([x[te], f_te])
    m = LogisticRegression(max_iter=1_000).fit(A_tr, y[tr])
    s_tr, s_te = m.predict_proba(A_tr)[:, 1], m.predict_proba(A_te)[:, 1]
    assert np.isclose(auc_score(y[te], s_te), roc_auc_score(y[te], s_te))
    return auc_score(y[tr], s_tr), auc_score(y[te], s_te)


section("exercise 2: three encodings")
m_x = LogisticRegression(max_iter=1_000).fit(x[tr, None], y[tr])
print(f"{'x only (no postcode)':<40} train AUC {auc_score(y[tr], m_x.predict_proba(x[tr, None])[:, 1]):.4f}   "
      f"test AUC {auc_score(y[te], m_x.predict_proba(x[te, None])[:, 1]):.4f}")
means_all, d_all = fit_target_means(postcode, y)  # (a) WRONG: uses the test labels
results = {"(a) means over all rows": fit_and_score(encode(postcode[tr], means_all, d_all), encode(postcode[te], means_all, d_all))}
means_tr, d_tr = fit_target_means(postcode[tr], y[tr])  # (b) in-sample on the training rows
results["(b) means over training rows, in-sample"] = fit_and_score(encode(postcode[tr], means_tr, d_tr),
                                                                   encode(postcode[te], means_tr, d_tr))
oof = out_of_fold_encode(postcode[tr], y[tr], N_FOLDS, np.random.default_rng(SEED))  # (c) right
results["(c) out-of-fold on training rows"] = fit_and_score(oof, encode(postcode[te], means_tr, d_tr))
for name, (a_tr, a_te) in results.items():
    print(f"{name:<40} train AUC {a_tr:.4f}   test AUC {a_te:.4f}")
print("from-scratch AUC matches roc_auc_score on every test score: OK")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
