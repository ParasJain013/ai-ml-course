# %% [markdown]
# # A5.4 LLN, CLT and Monte Carlo — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a5_4_lln_clt_monte_carlo_ex2.py   (CPU, ~5 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from scipy import integrate, stats

FloatArray = npt.NDArray[np.float64]
SEED = 0
T0 = time.perf_counter()
K = 10  # M is the largest of K standard normals
Z95 = 1.96
N_COVERAGE = 2_000


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
for n in (100, 10_000):
    print(f"(a) n = {n:>6,}: standard error = sqrt(0.3 * 0.7 / n) = {np.sqrt(0.21 / n):.4f}")
print(f"(a) impressions for +-0.01: (1.96 * sqrt(0.21) / 0.01)^2 = {(1.96 * np.sqrt(0.21) / 0.01) ** 2:.1f}")
mean_t, sd_t = 400 * 20, np.sqrt(400) * 15
print(f"(b) total: mean = {mean_t:.0f} ms   sd = {sd_t:.0f} ms   z = {(8600 - mean_t) / sd_t:.2f}   "
      f"P(total > 8.6 s) = {stats.norm.sf((8600 - mean_t) / sd_t):.4f}")
x = 2 * np.array([0.1, 0.3, 0.5, 0.7, 0.9])
print(f"(c) x = {x.tolist()}   x^2 = {np.round(x**2, 2).tolist()}   mean = {np.mean(x**2):.2f}   "
      f"estimate = 2 * mean = {2 * np.mean(x**2):.2f}   exact 8/3 = {8 / 3:.4f}")


# %%
def mc_expected_max(n: int, gen: np.random.Generator) -> tuple[float, float]:
    """Monte Carlo estimate of E[max of K standard normals]. -> (estimate, standard error)."""
    m = gen.standard_normal((n, K)).max(axis=1)  # (n,) one maximum per repetition
    return float(m.mean()), float(m.std(ddof=1) / np.sqrt(n))


def max_density(x: float) -> float:
    """Density of the max of K i.i.d. standard normals: K phi(x) Phi(x)^(K - 1)."""
    return K * stats.norm.pdf(x) * stats.norm.cdf(x) ** (K - 1)


exact, quad_err = integrate.quad(lambda x: x * max_density(x), -np.inf, np.inf)
area, _ = integrate.quad(max_density, -np.inf, np.inf)

section("exercise 2 (a, b): Monte Carlo against quadrature")
print(f"quad: E[M] = {exact:.6f}   (density integrates to {area:.6f}; quad error estimate {quad_err:.1e})")
gen = np.random.default_rng(SEED)
print(f"{'n':>10}{'estimate':>11}{'+-1.96 SE':>11}{'|error|':>10}{'covers':>8}")
sd_m = 0.0
for n in (1_000, 10_000, 100_000, 1_000_000):
    est, se = mc_expected_max(n, gen)
    sd_m = se * np.sqrt(n)
    print(f"{n:>10,}{est:>11.5f}{Z95 * se:>11.5f}{abs(est - exact):>10.5f}{str(abs(est - exact) <= Z95 * se):>8}")

section("exercise 2 (c): coverage at n = 1,000")
hits = sum(abs(e - exact) <= Z95 * s for e, s in (mc_expected_max(1_000, gen) for _ in range(N_COVERAGE)))
print(f"{N_COVERAGE} runs: interval contained E[M] in {hits / N_COVERAGE:.3f} of runs")

section("exercise 2 (d): repetitions for 3 decimals")
n_needed = (Z95 * sd_m / 0.0005) ** 2  # half-width 0.0005 so that rounding to 3 decimals is right
print(f"sd of M = {sd_m:.4f}   n for a 95% half-width of 0.0005: {n_needed:,.0f}")
t = time.perf_counter()
mc_expected_max(1_000_000, gen)
print(f"~ 10^6 repetitions took {time.perf_counter() - t:.2f} s, so about {n_needed / 1e6 * (time.perf_counter() - t):.1f} s for that n")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
