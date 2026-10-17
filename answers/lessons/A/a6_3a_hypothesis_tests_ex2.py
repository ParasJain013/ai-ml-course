# %% [markdown]
# # A6.3a Hypothesis tests — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a6_3a_hypothesis_tests_ex2.py   (CPU, ~15 s)

# %%
from __future__ import annotations

import itertools
import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy import stats

FloatArray = npt.NDArray[np.float64]
SEED = 0
T0 = time.perf_counter()
N_GROUP = 10
N_EXPERIMENTS = 1_000
N_PERM = 2_000
SHIFT = 1.8  # group B multiplied by 1.8 under the alternative
ALPHA = 0.05


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
a, b = np.array([2.0, 4, 6]), np.array([5.0, 7, 9])
pooled = np.concatenate([a, b])
obs = b.mean() - a.mean()
diffs = []
for chosen in itertools.combinations(range(6), 3):
    mask = np.zeros(6, dtype=bool)
    mask[list(chosen)] = True
    diffs.append((pooled[mask].tolist(), pooled[mask].mean() - pooled[~mask].mean()))
extreme = [(g, float(round(d, 4))) for g, d in diffs if abs(d) >= obs - 1e-12]
print(f"(a) observed {obs:.1f}; splits as extreme as observed: {extreme}")
print(f"(a) one-sided p = {sum(d >= obs - 1e-12 for _, d in diffs)}/20   two-sided p = {len(extreme)}/20")
va, vb = 4 / 4, 9 / 9
t = 3 / np.sqrt(va + vb)
nu = (va + vb) ** 2 / (va**2 / 3 + vb**2 / 8)
print(f"(b) SE = {np.sqrt(va + vb):.4f}   t = {t:.4f}   nu = {nu:.4f}   p = {2 * stats.t.sf(t, nu):.4f}")
crash = np.array([[45, 255], [25, 275]])
res = stats.chi2_contingency(crash, correction=False)
print(f"(c) expected {res.expected_freq.tolist()}   chi2 = {res.statistic:.4f}   df = {res.dof}   p = {res.pvalue:.4f}   "
      f"rates {45 / 300:.4f} vs {25 / 300:.4f}")


# %%
def perm_test(
    a: FloatArray,
    b: FloatArray,
    stat: Callable[[FloatArray, FloatArray], FloatArray],
    n_perm: int,
    gen: np.random.Generator,
) -> float:
    """Two-sided permutation p-value for stat(a, b); stat works on (k, n_a), (k, n_b) rows -> (k,)."""
    pooled = np.concatenate([a, b])
    perm = pooled[np.argsort(gen.random((n_perm, len(pooled))), axis=1)]  # (n_perm, n)
    null = stat(perm[:, : len(a)], perm[:, len(a):])  # (n_perm,)
    observed = stat(a[None, :], b[None, :])[0]
    return float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1))


def diff_means(a: FloatArray, b: FloatArray) -> FloatArray:
    """Row-wise mean(b) - mean(a). a (k, n_a), b (k, n_b) -> (k,)."""
    return b.mean(axis=1) - a.mean(axis=1)


section("exercise 2: type I error and power, n = 10 per group, log-normal sigma = 1")
rng = np.random.default_rng(SEED)
rates: dict[str, dict[str, float]] = {}
for scenario, factor in (("H0 true", 1.0), (f"B x {SHIFT}", SHIFT)):
    A = rng.lognormal(0.0, 1.0, size=(N_EXPERIMENTS, N_GROUP))
    Bm = factor * rng.lognormal(0.0, 1.0, size=(N_EXPERIMENTS, N_GROUP))
    p_welch = stats.ttest_ind(Bm, A, axis=1, equal_var=False).pvalue
    p_mw = stats.mannwhitneyu(Bm, A, axis=1).pvalue
    p_perm = np.array([perm_test(A[i], Bm[i], diff_means, N_PERM, rng) for i in range(N_EXPERIMENTS)])
    rates[scenario] = {name: float(np.mean(p < ALPHA)) for name, p in
                       (("Welch", p_welch), ("permutation", p_perm), ("Mann-Whitney", p_mw))}
print(f"{'scenario':>10}{'Welch':>10}{'permutation':>13}{'Mann-Whitney':>14}")
for scenario, r in rates.items():
    print(f"{scenario:>10}{r['Welch']:>10.3f}{r['permutation']:>13.3f}{r['Mann-Whitney']:>14.3f}")
print(f"Monte Carlo SE: at 0.05 {np.sqrt(0.05 * 0.95 / N_EXPERIMENTS):.4f}   at 0.5 {np.sqrt(0.25 / N_EXPERIMENTS):.4f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
