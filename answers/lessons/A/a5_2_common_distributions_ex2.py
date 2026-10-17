# %% [markdown]
# # A5.2 Common distributions — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a5_2_common_distributions_ex2.py   (CPU, ~2 s)

# %%
from __future__ import annotations

import time
from math import comb, exp

import numpy as np
import numpy.typing as npt
from scipy import stats

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()
N_REPEATS = 100_000


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
section("exercise 1 check")
print(f"(a) P(X = 2) = {comb(5, 2) * 0.3**2 * 0.7**3:.4f}   mean = {5 * 0.3:.2f}   variance = {5 * 0.3 * 0.7:.2f}")
p0 = exp(-1.5)
print(f"(b) P(0) = {p0:.4f}   P(>= 1) = {1 - p0:.4f}   P(<= 2) = {p0 * (1 + 1.5 + 1.5**2 / 2):.4f}")
cdf = np.cumsum([0.1, 0.6, 0.3])
print("(c) cumulative =", show(cdf), " classes for u = 0.05, 0.65, 0.71, 0.1:",
      np.searchsorted(cdf, [0.05, 0.65, 0.71, 0.1], side="right").tolist())
L = np.linalg.cholesky(np.array([[1.0, 0.6], [0.6, 1.0]]))
print("(d) L =", show(L), "  L z for z = (1, 1):", show(L @ np.ones(2)), "  L L^T =", show(L @ L.T))


# %%
def sample_poisson(rate: float, size: int, gen: np.random.Generator) -> IntArray:
    """Poisson(rate) by inverse CDF on a truncated pmf table, using only gen.random(). -> (size,)."""
    pmf = [exp(-rate)]  # P(0)
    tail = 1.0 - pmf[0]
    k = 0
    while tail > 1e-12:  # extend the table until the missing probability is negligible
        k += 1
        pmf.append(pmf[-1] * rate / k)  # P(k) = P(k - 1) * rate / k
        tail -= pmf[-1]
    cdf = np.cumsum(pmf)  # (k + 1,)
    cdf[-1] = 1.0  # the < 1e-12 tail is folded into the last value
    return np.searchsorted(cdf, gen.random(size), side="right")


def sample_exponential(rate: float, size: int, gen: np.random.Generator) -> FloatArray:
    """Exponential(rate) by inverse CDF: -log(u) / rate with u in (0, 1]. -> (size,)."""
    return -np.log(1.0 - gen.random(size)) / rate


def counts_from_gaps(rate: float, repeats: int, gen: np.random.Generator) -> IntArray:
    """Count events in [0, 1) when gaps between events are Exponential(rate). -> (repeats,)."""
    max_events = int(rate + 10 * np.sqrt(rate) + 20)  # far more gaps than [0, 1) can hold
    gaps = sample_exponential(rate, repeats * max_events, gen).reshape(repeats, max_events)
    times = np.cumsum(gaps, axis=1)  # (repeats, max_events) event times
    assert np.all(times[:, -1] >= 1.0), "increase max_events"
    return (times < 1.0).sum(axis=1)


def max_pmf_gap(x: IntArray, pmf: FloatArray) -> float:
    """Largest |empirical frequency - pmf| over 0..len(pmf)-1. x (n,) -> scalar."""
    freq = np.bincount(x, minlength=len(pmf))[: len(pmf)] / len(x)
    return float(np.max(np.abs(freq - pmf)))


section("exercise 2 (a, b): samplers")
gen = np.random.default_rng(SEED)
pois = sample_poisson(4.0, N_REPEATS, gen)
expo = sample_exponential(4.0, N_REPEATS, gen)
print(f"Poisson(4): mean = {pois.mean():.4f}   variance = {pois.var(ddof=1):.4f}   (exact 4, 4)")
print(f"Exponential(4): mean = {expo.mean():.4f}   variance = {expo.var(ddof=1):.4f}   (exact 0.25, 0.0625)")

section("exercise 2 (c): counts from exponential gaps")
counts = counts_from_gaps(4.0, N_REPEATS, gen)
k = np.arange(16)
exact = stats.poisson(4.0).pmf(k)
print(f"{'k':>3}{'gaps':>9}{'sampler':>9}{'scipy':>9}")
for kk in range(9):
    print(f"{kk:>3}{np.mean(counts == kk):>9.4f}{np.mean(pois == kk):>9.4f}{exact[kk]:>9.4f}")
tol = 3 / np.sqrt(N_REPEATS)
gaps = {"gap counts vs scipy": max_pmf_gap(counts, exact), "sampler vs scipy": max_pmf_gap(pois, exact)}
for name, g in gaps.items():
    print(f"{name:<22} largest gap = {g:.4f}")
assert all(g < tol for g in gaps.values()), gaps
print(f"both below 3/sqrt(n) = {tol:.4f}: OK")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
