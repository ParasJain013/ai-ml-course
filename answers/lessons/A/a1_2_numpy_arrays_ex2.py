# %% [markdown]
# # A1.2 NumPy arrays — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a1_2_numpy_arrays_ex2.py   (CPU, ~5 s)

# %%
from __future__ import annotations

import math
import time

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import normalize

FloatArray = npt.NDArray[np.float64]
SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
for s1, s2 in [((4, 1), (3,)), ((2, 3), (2,)), ((5, 1, 4), (3, 1)), ((6, 3), (6,)), ((6, 3), (6, 1))]:
    try:
        print(f"{s1} op {s2} -> {(np.zeros(s1) + np.zeros(s2)).shape}")
    except ValueError:
        print(f"{s1} op {s2} -> error")
A = np.array([[1.0, 1.0], [2.0, 3.0]])  # (2, 2)
B = np.array([[1.0, 1.0], [4.0, 5.0], [1.0, 4.0]])  # (3, 2)
print("|a|^2:", (A**2).sum(axis=1).tolist(), " |b|^2:", (B**2).sum(axis=1).tolist())
print("A @ B.T:", (A @ B.T).tolist())
d_sq = (A**2).sum(axis=1)[:, None] + (B**2).sum(axis=1)[None, :] - 2 * A @ B.T
print("D^2:", d_sq.tolist())
print("D:  ", np.round(np.sqrt(d_sq), 4).tolist())

# %%
section("exercise 2: row_normalize")


def row_normalize(X: FloatArray) -> FloatArray:
    """Scale each row of X to unit L2 length; all-zero rows stay zero. X (n, d) -> (n, d)."""
    lengths = np.sqrt((X**2).sum(axis=1, keepdims=True))  # (n, 1)
    safe = np.where(lengths == 0.0, 1.0, lengths)  # (n, 1): avoid 0 / 0 -> nan
    return X / safe  # (n, d) / (n, 1) broadcasts over columns


def row_normalize_loop(X: FloatArray) -> FloatArray:
    """Loop reference for row_normalize. X (n, d) -> (n, d)."""
    out = []
    for row in X.tolist():
        length = math.sqrt(sum(v * v for v in row))
        out.append([v / length for v in row] if length > 0 else row)
    return np.array(out)


X_small = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, -1.0]])  # (3, 2), one zero row
print(np.round(row_normalize(X_small), 4).tolist())
X = rng.normal(size=(10_000, 64))  # (10000, 64)
X[5] = 0.0  # make sure a zero row is in the big test too
mine = row_normalize(X)
assert mine.shape == X.shape, mine.shape
assert not np.isnan(mine).any()
np.testing.assert_allclose(mine, row_normalize_loop(X), rtol=1e-6, atol=1e-8)
print("matches loop version: OK")
np.testing.assert_allclose(mine, normalize(X, norm="l2"), rtol=1e-6, atol=1e-8)
print("matches sklearn normalize: OK")
nonzero = np.abs(X).sum(axis=1) > 0
np.testing.assert_allclose(np.linalg.norm(mine[nonzero], axis=1), 1.0, rtol=1e-12)
print("non-zero rows have length 1, zero row stays zero:", bool((mine[5] == 0).all()))

t = time.perf_counter()
row_normalize(X)
t_vec = time.perf_counter() - t
t = time.perf_counter()
row_normalize_loop(X)
t_loop = time.perf_counter() - t
print(f"~ vectorized {t_vec * 1e3:.2f} ms, loop {t_loop * 1e3:.1f} ms, speed-up {t_loop / t_vec:.0f}x")

# %%
section("exercise 3 demo")
n = 4
y = np.array([1.0, 2.0, 3.0, 4.0])  # (4,)
y_hat = y[:, None]  # (4, 1): a perfect model, wrong shape
print("y_hat - y shape:", (y_hat - y).shape)
print("MSE with the bug:", np.mean((y_hat - y) ** 2), " correct MSE:", np.mean((y_hat[:, 0] - y) ** 2))

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
