# %% [markdown]
# # A6.5 Bayesian inference: the Beta–Binomial model — lab
# Lesson: lessons/A/a6_5_bayesian_inference.md
# Run: uv run python labs/A/a6_5_bayesian_inference.py   (CPU, defaults: ~2 s)
# GPU: no

# %%
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:  # plain `uv run`: draw to files, never open windows
    matplotlib.use("Agg")
    # matplotlib 3.11 warns when plt.show() runs under Agg; under `uv run` it is a harmless no-op
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import special, stats

FloatArray = npt.NDArray[np.float64]
SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()


def repo_root() -> Path:
    """Find the course folder (works under `uv run` and in VS Code cells)."""
    here = Path(globals().get("__file__", Path.cwd())).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("Run labs from inside the ai-ml-course folder.")


ROOT = repo_root()
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "a6_5"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "A" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE only makes the grid and the Monte Carlo bigger.
GPU_SCALE = False
N_GRID = 100_001 if GPU_SCALE else 10_001  # grid points on [0, 1] for the grid posterior
N_DRAWS = 1_000_000 if GPU_SCALE else 200_000  # posterior draws for the A/B comparison
TRUE_P = 0.3  # the conversion rate the simulated data comes from
N_OBS = 200  # observations in the data stream
CHECKPOINTS = [0, 1, 2, 5, 10, 20, 50, 100, 200]

# %%
# Data: a stream of 200 conversions (1) and non-conversions (0) with true rate 0.3.
section("data")
stream = (rng.random(N_OBS) < TRUE_P).astype(np.int64)
print(f"{N_OBS} observations, true rate {TRUE_P}: first 20 = {stream[:20].tolist()}   total successes = {stream.sum()}")

# %%
# Worked example by hand: Beta(2, 2) prior, 3 successes in 4 trials.
section("worked example by hand")
a0, b0, k, n = 2, 2, 3, 4
a1, b1 = a0 + k, b0 + n - k
print(f"prior Beta({a0}, {b0}): mean {a0 / (a0 + b0):.4f}   data {k}/{n}: MLE {k / n:.4f}")
print(f"posterior Beta({a0} + {k}, {b0} + {n - k}) = Beta({a1}, {b1}): mean {a1}/{a1 + b1} = {a1 / (a1 + b1):.4f}   "
      f"mode (MAP) {a1 - 1}/{a1 + b1 - 2} = {(a1 - 1) / (a1 + b1 - 2):.4f}")
w = (a0 + b0) / (a0 + b0 + n)
print(f"weighted average: {w:.2f} * prior mean 0.5 + {1 - w:.2f} * MLE 0.75 = {w * 0.5 + (1 - w) * 0.75:.4f}")
print(f"P(next trial succeeds) = posterior mean = {a1 / (a1 + b1):.4f}")
lo, hi = stats.beta(a1, b1).ppf([0.025, 0.975])
print(f"95% credible interval: [{lo:.4f}, {hi:.4f}]   P(p > 0.5 | data) = {stats.beta(a1, b1).sf(0.5):.4f}")
seq_a, seq_b = a0, b0
for outcome in (1, 0, 1, 1):  # the same 3 successes and 1 failure, one at a time
    seq_a, seq_b = seq_a + outcome, seq_b + 1 - outcome
    print(f"  after observing {outcome}: Beta({seq_a}, {seq_b})")

# %%
# From scratch: the posterior on a grid (prior times likelihood, normalized), against the formula.
section("from scratch: grid posterior")
theta = np.linspace(0, 1, N_GRID)  # (N_GRID,) candidate values of p
d_theta = theta[1] - theta[0]


def grid_posterior(prior: FloatArray, k: int, n: int) -> FloatArray:
    """Normalized posterior density on theta. prior (N_GRID,) -> (N_GRID,)."""
    log_lik = special.xlogy(k, theta) + special.xlogy(n - k, 1 - theta)  # log of p^k (1-p)^(n-k), 0 log 0 = 0
    unnorm = prior * np.exp(log_lik - log_lik.max())  # subtract the max: no underflow for large n
    return unnorm / (unnorm.sum() * d_theta)  # (N_GRID,) integrates to 1


def grid_summary(post: FloatArray) -> tuple[float, float, float]:
    """Mean and equal-tailed 95% interval of a grid density. post (N_GRID,) -> (mean, low, high)."""
    cdf = np.cumsum(post) * d_theta
    mean = float(np.sum(theta * post) * d_theta)
    return mean, float(np.interp(0.025, cdf, theta)), float(np.interp(0.975, cdf, theta))


prior_22 = stats.beta(2, 2).pdf(theta)
post = grid_posterior(prior_22, k, n)
exact = stats.beta(a1, b1).pdf(theta)
mean_g, lo_g, hi_g = grid_summary(post)
print(f"3/4 with Beta(2, 2): largest |grid - Beta(5, 3) density| = {np.max(np.abs(post - exact)):.2e}")
print(f"  grid mean {mean_g:.4f}   grid 95% interval [{lo_g:.4f}, {hi_g:.4f}]")
k_all = int(stream.sum())
post_all = grid_posterior(np.ones(N_GRID), k_all, N_OBS)  # flat prior: Beta(1, 1)
print(f"all {N_OBS} observations, flat prior: largest |grid - Beta({1 + k_all}, {1 + N_OBS - k_all})| = "
      f"{np.max(np.abs(post_all - stats.beta(1 + k_all, 1 + N_OBS - k_all).pdf(theta))):.2e}")
odd_prior = np.where(theta < 0.5, 1.0, 3.0)  # a prior no formula handles: p > 0.5 three times as plausible
k10 = int(stream[:10].sum())
for name, pr in (("flat prior", np.ones(N_GRID)), ("step prior, p > 0.5 3x as plausible", odd_prior)):
    m_o, l_o, h_o = grid_summary(grid_posterior(pr, k10, 10))
    print(f"  first 10 observations ({k10}/10), {name}: mean {m_o:.4f}   95% interval [{l_o:.4f}, {h_o:.4f}]")

# %%
# From scratch: sequential updating, and posterior mean versus MLE as data arrives.
section("from scratch: posterior versus MLE as data arrives")
priors = {"Beta(1, 1)": (1, 1), "Beta(2, 2)": (2, 2), "Beta(20, 20)": (20, 20)}
cum_k = np.concatenate([[0], np.cumsum(stream)])  # successes after 0, 1, ..., N_OBS observations
print(f"{'n':>4}{'k':>4}{'MLE':>8}" + "".join(f"{name:>14}" for name in priors) + f"{'95% width (flat)':>18}")
for m in CHECKPOINTS:
    km = int(cum_k[m])
    mle = f"{km / m:>8.4f}" if m > 0 else f"{'-':>8}"
    means = "".join(f"{(a + km) / (a + b + m):>14.4f}" for a, b in priors.values())
    lo_m, hi_m = stats.beta(1 + km, 1 + m - km).ppf([0.025, 0.975])
    print(f"{m:>4}{km:>4}{mle}{means}{hi_m - lo_m:>18.4f}")
seq = (1, 1)
for x in stream:
    seq = (seq[0] + int(x), seq[1] + 1 - int(x))
print(f"one-at-a-time updates over all {N_OBS}: Beta{seq}   batch update: Beta({1 + k_all}, {1 + N_OBS - k_all})")

# %%
# From scratch: a Bayesian reading of an A/B test (the numbers of the A6.3a worked example).
section("from scratch: Bayesian A/B comparison")
k_a, n_a, k_b, n_b = 30, 200, 50, 200
post_a = rng.beta(1 + k_a, 1 + n_a - k_a, size=N_DRAWS)  # (N_DRAWS,) draws of p_A given its data
post_b = rng.beta(1 + k_b, 1 + n_b - k_b, size=N_DRAWS)
lift = post_b - post_a
print(f"A: {k_a}/{n_a} -> Beta({1 + k_a}, {1 + n_a - k_a})   B: {k_b}/{n_b} -> Beta({1 + k_b}, {1 + n_b - k_b})")
print(f"P(p_B > p_A | data) = {np.mean(lift > 0):.4f}   expected lift = {lift.mean():.4f}   "
      f"95% credible interval for the lift = [{np.quantile(lift, 0.025):.4f}, {np.quantile(lift, 0.975):.4f}]")
print(f"P(lift > 0.05 | data) = {np.mean(lift > 0.05):.4f}   (is the lift at least 5 points?)")
print(f"for comparison, the chi-square test of A6.3a gave p = {stats.chi2_contingency(np.array([[30, 170], [50, 150]]), correction=False).pvalue:.4f}")

# %%
# With SciPy: the conjugate posterior as a frozen distribution, and the posterior predictive.
section("with scipy")
post_dist = stats.beta(a1, b1)
np.testing.assert_allclose((post_dist.mean(), *post_dist.interval(0.95)), (mean_g, lo_g, hi_g), atol=1e-4)
print(f"stats.beta(5, 3): mean {post_dist.mean():.4f}   interval(0.95) {show(np.array(post_dist.interval(0.95)))}   "
      "matches the grid (to 1e-4, the grid spacing): OK")
pred = stats.betabinom(10, a1, b1)  # number of successes in the next 10 trials, averaged over the posterior
plug = stats.binom(10, k / n)  # plugging in the MLE instead
print(f"next 10 trials: posterior predictive mean {pred.mean():.4f}   sd {pred.std():.4f}   "
      f"plug-in Binomial(10, 0.75): mean {plug.mean():.4f}   sd {plug.std():.4f}")
print(f"P(0 or 1 successes in the next 10): posterior predictive {pred.cdf(1):.4f}   plug-in {plug.cdf(1):.1e}")
mc = rng.binomial(10, rng.beta(a1, b1, size=N_DRAWS))  # predictive by simulation: draw p, then draw data
np.testing.assert_allclose(mc.mean(), pred.mean(), atol=0.02)
print(f"simulated predictive (draw p from the posterior, then 10 trials): mean {mc.mean():.4f}   OK")

# %%
# Key plot: the posterior sharpening as data arrives, and posterior means versus the MLE.
section("key plot")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=120)
ax = axes[0]
styles = ["-", "--", "-.", ":", "-", "--"]
for m, style in zip([0, 1, 5, 20, 50, 200], styles):
    km = int(cum_k[m])
    ax.plot(theta, stats.beta(2 + km, 2 + m - km).pdf(theta), style, lw=1.8, label=f"n = {m:>3}, k = {km}")
ax.axvline(TRUE_P, color="k", lw=1, label=f"true p = {TRUE_P}")
ax.set(xlabel="conversion rate p", ylabel="posterior density", title="Beta(2, 2) prior sharpens as data arrives")
ax.legend(loc="upper right", fontsize=8)
ax = axes[1]
ms = np.arange(1, N_OBS + 1)
ax.plot(ms, cum_k[1:] / ms, "k-", lw=1, alpha=0.6, label="MLE k/n")
for (name, (a, b)), style in zip(priors.items(), ["-", "--", ":"]):
    ax.plot(ms, (a + cum_k[1:]) / (a + b + ms), style, lw=2, label=f"posterior mean, {name}")
ax.axhline(TRUE_P, color="tab:red", lw=1, label=f"true p = {TRUE_P}")
ax.set_xscale("log")
ax.set(xlabel="observations n (log scale)", ylabel="estimate of p", ylim=(0, 1),
       title="Posterior means shrink early guesses toward the prior")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
path_png = IMG / "a6_5_beta_binomial_updates.png"
fig.savefig(path_png, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path_png.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Use a Beta(30, 70) prior (a confident belief that p is about 0.3). How many observations until the
#    posterior mean is within 0.01 of the MLE?
# 2. Make the true rate drift from 0.3 to 0.5 halfway through the stream. What does the posterior do?
# 3. In the A/B cell, give B only 20 users. How do P(p_B > p_A) and the lift interval change?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
