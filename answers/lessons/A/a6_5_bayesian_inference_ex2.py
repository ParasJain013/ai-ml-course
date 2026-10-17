# %% [markdown]
# # A6.5 Bayesian inference — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a6_5_bayesian_inference_ex2.py   (CPU, ~3 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

FloatArray = npt.NDArray[np.float64]
SEED = 0
T0 = time.perf_counter()
N_DRAWS = 200_000  # posterior draws per comparison
N_GRID = 20_001  # grid points for the exact P(p_B > p_A)
N_AA = 1_000  # A/A experiments
AA_RATE, AA_USERS = 0.12, 1_000
AA_DRAWS = 20_000  # fewer draws per A/A experiment: the decision only needs P to about +-0.005


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
a, b, k, n = 3, 7, 9, 20
post = stats.beta(a + k, b + n - k)
w = (a + b) / (a + b + n)
print(f"(a) posterior Beta({a + k}, {b + n - k})   mean {post.mean():.4f}   mode {(a + k - 1) / (a + b + n - 2):.4f}   "
      f"MLE {k / n:.4f}")
print(f"(a) weight on prior {w:.4f}:  {w:.4f} * {a / (a + b):.2f} + {1 - w:.4f} * {k / n:.2f} = "
      f"{w * a / (a + b) + (1 - w) * k / n:.4f}")
print(f"(b) P(next email opened) = posterior mean = {post.mean():.4f}")
for name, (s, m) in {"X": (1, 1), "Y": (360, 400)}.items():
    print(f"(c) {name}: Beta(4, 1) + {s}/{m} -> Beta({4 + s}, {1 + m - s})   mean {(4 + s) / (5 + m):.4f}")


# %%
def ab_bayes(k_a: int, n_a: int, k_b: int, n_b: int, a: float = 1.0, b: float = 1.0,
             draws: int = N_DRAWS, gen: np.random.Generator | None = None) -> tuple[float, float, tuple[float, float]]:
    """Beta-Binomial A/B comparison. Returns P(p_B > p_A), mean lift p_B - p_A, 95% credible interval of the lift."""
    gen = gen if gen is not None else np.random.default_rng(SEED)
    p_a = gen.beta(a + k_a, b + n_a - k_a, size=draws)  # (draws,) posterior draws of p_A
    p_b = gen.beta(a + k_b, b + n_b - k_b, size=draws)  # (draws,) posterior draws of p_B
    lift = p_b - p_a  # (draws,)
    lo, hi = np.quantile(lift, [0.025, 0.975])
    return float(np.mean(lift > 0)), float(lift.mean()), (float(lo), float(hi))


section("exercise 2(a): Bayesian A/B versus the z-test")
rng = np.random.default_rng(SEED)
k_a, n_a, k_b, n_b = 120, 1_000, 145, 1_000
p_win, lift, (lo, hi) = ab_bayes(k_a, n_a, k_b, n_b, gen=rng)
z, p_one = proportions_ztest([k_b, k_a], [n_b, n_a], alternative="larger")
print(f"P(p_B > p_A) = {p_win:.4f}   mean lift {lift:.4f}   95% credible [{lo:.4f}, {hi:.4f}]")
print(f"one-sided z-test: z = {z:.3f}   p = {p_one:.4f}   1 - p = {1 - p_one:.4f}")

section("exercise 2(b): exact P(p_B > p_A) on a grid")
grid = np.linspace(0, 1, N_GRID)  # (N_GRID,) values of p_B
dens_b = stats.beta.pdf(grid, 1 + k_b, 1 + n_b - k_b)  # (N_GRID,) posterior density of p_B
cdf_a = stats.beta.cdf(grid, 1 + k_a, 1 + n_a - k_a)  # (N_GRID,) P(p_A < p_B = grid value)
exact = float(np.trapezoid(dens_b * cdf_a, grid))  # integral of f_B(x) F_A(x) dx
mc_se = np.sqrt(exact * (1 - exact) / N_DRAWS)
print(f"exact {exact:.4f}   Monte Carlo {p_win:.4f}   difference {p_win - exact:+.4f}   MC standard error {mc_se:.4f}")
assert abs(p_win - exact) < 4 * mc_se
print("OK: within 4 standard errors")

section("exercise 2(c): A/A experiments")
rng = np.random.default_rng(SEED)
wins = 0
for _ in range(N_AA):
    ka, kb = rng.binomial(AA_USERS, AA_RATE, size=2)
    wins += ab_bayes(int(ka), AA_USERS, int(kb), AA_USERS, draws=AA_DRAWS, gen=rng)[0] > 0.95
rate = wins / N_AA
se = np.sqrt(0.05 * 0.95 / N_AA)
print(f"P(p_B > p_A) > 0.95 in {wins} of {N_AA} A/A experiments = {rate:.3f}   "
      f"(one-sided alpha 0.05; +-2 se = [{0.05 - 2 * se:.3f}, {0.05 + 2 * se:.3f}])")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
