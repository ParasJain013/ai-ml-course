# %% [markdown]
# # A1.1 The lab workflow — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a1_1_lab_workflow_ex2.py   (CPU, ~1 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")


def lcg(seed: int, a: int, c: int, m: int, n: int) -> IntArray:
    """Return n states of x_{k+1} = (a x_k + c) mod m, starting after the seed -> (n,)."""
    out = np.empty(n, dtype=np.int64)
    x = seed
    for k in range(n):
        x = (a * x + c) % m
        out[k] = x
    return out


def period(seed: int, a: int, c: int, m: int) -> int:
    """Count steps until a state repeats (the cycle the seed falls into)."""
    seen: dict[int, int] = {}
    x, k = seed, 0
    while x not in seen:
        seen[x] = k
        x = (a * x + c) % m
        k += 1
    return k - seen[x]


print("c=1 states:", lcg(2, 5, 1, 8, 8).tolist(), "period:", period(2, 5, 1, 8))
print("c=2 states:", lcg(2, 5, 2, 8, 8).tolist(), "period:", period(2, 5, 2, 8))


# %%
section("exercise 2: shuffle_split")


def shuffle_split(
    n: int, test_frac: float, rng: np.random.Generator
) -> tuple[IntArray, IntArray]:
    """Return (train_idx, test_idx) for a random split of rows 0..n-1. -> (n - n_test,), (n_test,)."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    n_test = int(round(n * test_frac))
    perm = rng.permutation(n)  # (n,) every index exactly once, in random order
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


N = 100  # n = 10 is too small for part (c): see the answer key's "likely mistakes"
train_a, test_a = shuffle_split(N, 0.3, np.random.default_rng(SEED))
train_b, test_b = shuffle_split(N, 0.3, np.random.default_rng(SEED))
print("(a) first 5 test idx, run 1:", test_a[:5].tolist())
print("(a) first 5 test idx, run 2:", test_b[:5].tolist())
assert np.array_equal(test_a, test_b) and np.array_equal(train_a, train_b)
print("(a) same seed, same split: OK")

assert np.intersect1d(train_a, test_a).size == 0
assert np.array_equal(np.sort(np.concatenate([train_a, test_a])), np.arange(N))
print("(b) disjoint and covers 0..n-1: OK")

rng = np.random.default_rng(SEED)
_ = rng.random()  # an unrelated extra draw before the split
_, test_c = shuffle_split(N, 0.3, rng)
print("(c) with an extra draw first:", test_c[:5].tolist(), "changed:", not np.array_equal(test_c, test_a))

split_rng, other_rng = np.random.default_rng(SEED).spawn(2)
_, test_d = shuffle_split(N, 0.3, split_rng)
split_rng, other_rng = np.random.default_rng(SEED).spawn(2)
_ = other_rng.random()  # the extra draw now goes to a different stream
_, test_e = shuffle_split(N, 0.3, split_rng)
print("(c) spawned stream, no extra draw:       ", test_d[:5].tolist())
print("(c) spawned stream, extra draw elsewhere:", test_e[:5].tolist())
assert np.array_equal(test_d, test_e)
print("(c) spawned split unaffected: OK")

# %%
section("why n = 10 fooled part (c)")
small = np.random.default_rng(SEED).permutation(10)
gen = np.random.default_rng(SEED)
_ = gen.random()
small_shifted = gen.permutation(10)
print("permutation(10):              ", small.tolist())
print("permutation(10) after a draw: ", small_shifted.tolist())

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
