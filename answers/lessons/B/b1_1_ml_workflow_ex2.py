# %% [markdown]
# # B1.1 The ML workflow — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b1_1_ml_workflow_ex2.py   (CPU, ~2 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()
N_CUSTOMERS = 300
CUSTOMER_SD, NOISE_SD = 4.0, 1.0  # hidden customer effect and row-level noise


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# Exercise 1 check: the stratified 50/25/25 split and the baselines, from the given shuffles.
section("exercise 1 check")
labels = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0])
targets = np.array([10, 4, 6, 12, 5, 3, 7, 11, 4, 6, 9, 5], dtype=np.float64)
orders = {1: [3, 10, 0, 7], 0: [5, 2, 9, 1, 11, 8, 4, 6]}
sets: dict[str, list[int]] = {"train": [], "val": [], "test": []}
for c, rows in orders.items():
    n_test, n_val = round(0.25 * len(rows)), round(0.25 * len(rows))
    sets["test"] += rows[:n_test]
    sets["val"] += rows[n_test:n_test + n_val]
    sets["train"] += rows[n_test + n_val:]
    print(f"class {c}: {len(rows)} rows -> test {rows[:n_test]}  val {rows[n_test:n_test + n_val]}  "
          f"train {rows[n_test + n_val:]}")
tr, va = np.sort(sets["train"]), np.sort(sets["val"])
print(f"train {tr.tolist()}   val {va.tolist()}   test {sorted(sets['test'])}")
maj = int(np.bincount(labels[tr]).argmax())
print(f"train labels {labels[tr].tolist()} -> majority {maj}   val labels {labels[va].tolist()} -> "
      f"accuracy {np.mean(labels[va] == maj):.4f}")
for name, c in (("mean", targets[tr].mean()), ("median", np.median(targets[tr]))):
    e = targets[va] - c
    print(f"{name} baseline {c:.4f}: val errors {e.tolist()}   MSE {np.mean(e**2):.4f}   MAE {np.mean(np.abs(e)):.4f}")
se = np.sqrt(0.8 * 0.2 / 400)
print(f"(c) SE = sqrt(0.8 * 0.2 / 400) = {se:.4f}   expected best of 10 = 0.80 + 1.54 * SE = {0.8 + 1.54 * se:.4f}")
gen_c = np.random.default_rng(SEED)
best = (gen_c.binomial(400, 0.8, size=(20_000, 10)) / 400).max(axis=1)
print(f"(c) simulation, 20000 repetitions: mean best of 10 = {best.mean():.4f}")

# %%
# Exercise 2: a group-aware split, and what a row-level split does to a memorizing model.
section("exercise 2: data")
gen = np.random.default_rng(SEED)
rows_per = gen.integers(2, 7, N_CUSTOMERS)  # 2..6 rows per customer
groups = np.repeat(np.arange(N_CUSTOMERS), rows_per)  # (n,) customer id of each row
n = len(groups)
age = gen.uniform(18, 80, N_CUSTOMERS)[groups]  # customer-level: same on all of a customer's rows
tenure = gen.uniform(0, 10, N_CUSTOMERS)[groups]
basket = gen.normal(5, 2, n)  # row-level
effect = gen.normal(0, CUSTOMER_SD, N_CUSTOMERS)[groups]  # hidden, not in the features
y = 1.0 * basket + effect + gen.normal(0, NOISE_SD, n)
X = np.column_stack([age, tenure, basket])  # (n, 3)
print(f"{N_CUSTOMERS} customers, {n} rows, X {X.shape}, rows per customer {rows_per.min()}..{rows_per.max()}")


def group_train_val_test_split(
    groups: IntArray, val_frac: float, test_frac: float, gen: np.random.Generator
) -> tuple[IntArray, IntArray, IntArray]:
    """Shuffle the distinct groups, cut them three ways, return the ROW indices of each part."""
    ids = gen.permutation(np.unique(groups))
    n_test, n_val = round(test_frac * len(ids)), round(val_frac * len(ids))
    test_ids, val_ids, train_ids = ids[:n_test], ids[n_test:n_test + n_val], ids[n_test + n_val:]
    return (np.flatnonzero(np.isin(groups, train_ids)), np.flatnonzero(np.isin(groups, val_ids)),
            np.flatnonzero(np.isin(groups, test_ids)))


def row_train_val_test_split(
    n: int, val_frac: float, test_frac: float, gen: np.random.Generator
) -> tuple[IntArray, IntArray, IntArray]:
    """The lesson's plain split, for contrast. -> (train, val, test) row indices."""
    rows = gen.permutation(n)
    n_test, n_val = round(test_frac * n), round(val_frac * n)
    return rows[n_test + n_val:], rows[n_test:n_test + n_val], rows[:n_test]


def test_group_split() -> None:
    """No customer appears in two parts; every row is used exactly once."""
    tr, va, te = group_train_val_test_split(groups, 0.2, 0.2, np.random.default_rng(1))
    assert len(np.concatenate([tr, va, te])) == n == len(np.unique(np.concatenate([tr, va, te])))
    g_tr, g_va, g_te = set(groups[tr]), set(groups[va]), set(groups[te])
    assert not (g_tr & g_va) and not (g_tr & g_te) and not (g_va & g_te)


test_group_split()
print("test_group_split: OK")


def one_nn_predict(X_train: FloatArray, y_train: FloatArray, X_query: FloatArray) -> FloatArray:
    """Copy the target of the closest training row. X_train (n, d), X_query (m, d) -> (m,)."""
    # from labs/B/b1_1_ml_workflow.py
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    A, Q = (X_train - mu) / sd, (X_query - mu) / sd
    d2 = (Q**2).sum(1)[:, None] - 2 * Q @ A.T + (A**2).sum(1)[None, :]
    return y_train[np.argmin(d2, axis=1)]


def r2(y: FloatArray, y_hat: FloatArray) -> float:
    """1 - SSE/SST. y, y_hat (n,) -> scalar."""
    return 1.0 - float(np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))


section("exercise 2: row split vs group split")
splits = {"row split": row_train_val_test_split(n, 0.2, 0.2, np.random.default_rng(SEED)),
          "group split": group_train_val_test_split(groups, 0.2, 0.2, np.random.default_rng(SEED))}
for name, (tr, va, te) in splits.items():
    shared = np.isin(groups[va], groups[tr]).mean()
    p = one_nn_predict(X[tr][:, :2], y[tr], X[va][:, :2])  # customer-level features: age, tenure
    print(f"{name:<12} train {len(tr)}  val {len(va)}  val rows whose customer is also in train {shared:.4f}   "
          f"1-NN val R^2 {r2(y[va], p):.4f}")
    # Linear baseline with the honest signal only (basket), for scale:
    A = np.column_stack([np.ones(len(tr)), X[tr, 2]])
    w = np.linalg.lstsq(A, y[tr], rcond=None)[0]
    print(f"{'':<12} linear on basket only: val R^2 {r2(y[va], w[0] + w[1] * X[va, 2]):.4f}")
var_share = CUSTOMER_SD**2 / (CUSTOMER_SD**2 + NOISE_SD**2 + 2.0**2)
print(f"share of the target variance that is the hidden customer effect: {var_share:.4f}")
print(f"copying a random row of the same customer: expected R^2 = 2 * share - 1 = {2 * var_share - 1:.4f}; "
      "of another customer: -1")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
