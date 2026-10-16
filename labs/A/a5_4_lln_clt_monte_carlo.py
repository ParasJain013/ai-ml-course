# %% [markdown]
# # A5.4 Law of large numbers, central limit theorem and Monte Carlo estimation — lab
# Lesson: lessons/A/a5_4_lln_clt_monte_carlo.md
# Run: uv run python labs/A/a5_4_lln_clt_monte_carlo.py   (CPU, defaults: ~4 s)
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
from scipy import integrate, stats

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
OUT = ROOT / "outputs" / "a5_4"  # scratch figures, gitignored
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
# Nothing here needs a GPU; GPU_SCALE only makes the simulations bigger.
GPU_SCALE = False
N_RUNNING = 1_000_000 if GPU_SCALE else 100_000  # length of each running-mean sequence
N_PATHS = 5  # independent running-mean sequences
CLT_SIZES = [1, 2, 5, 30, 100]  # how many skewed draws are averaged
N_CLT = 200_000 if GPU_SCALE else 50_000  # sample means per CLT size
MC_SIZES = [100, 1_000, 10_000, 100_000, 1_000_000]  # Monte Carlo sample sizes
N_COVERAGE = 10_000 if GPU_SCALE else 2_000  # repeated Monte Carlo runs for the coverage check
Z95 = 1.96  # 97.5th percentile of N(0, 1): +-1.96 standard errors covers 95%
SIGMA_DIE = np.sqrt(35 / 12)  # standard deviation of one fair die (A5.1)

# %%
# Data: everything is simulated. The skewed variable for the CLT is the exponential with mean 1.
section("data")
expo = rng.exponential(1.0, size=200_000)
print(f"exponential(1): mean = {expo.mean():.4f}   median = {np.median(expo):.4f}   "
      f"skewness = {stats.skew(expo):.4f}   largest of 200,000 = {expo.max():.2f}")
print("a skewed variable: most values are small, a long tail to the right (mean > median)")

# %%
# Worked example by hand.
section("worked example by hand")
for n in (1, 4, 100):
    print(f"die: standard error of the mean of {n:>3} rolls = sigma / sqrt(n) = {SIGMA_DIE / np.sqrt(n):.4f}")
two_dice = np.convolve(np.ones(6) / 6, np.ones(6) / 6)  # pmf of the sum 2..12
print("pmf of the sum of two dice, totals 2..12, in 36ths:", np.round(two_dice * 36).astype(int).tolist())
mean100, sd100 = 100 * 3.5, np.sqrt(100 * 35 / 12)
print(f"sum of 100 dice: mean = {mean100:.0f}   sd = sqrt(100 * 35/12) = {sd100:.4f}   "
      f"P(sum > 380) ~ 1 - Phi({(380 - mean100) / sd100:.4f}) = {stats.norm.sf((380 - mean100) / sd100):.4f}")
inside, n_pts = 8, 10
p_hat = inside / n_pts
print(f"pi from {n_pts} points, {inside} inside: estimate = 4 * {p_hat} = {4 * p_hat:.1f}   "
      f"standard error = 4 * sqrt(p(1 - p)/n) = {4 * np.sqrt(p_hat * (1 - p_hat) / n_pts):.4f}")
u = np.array([0.1, 0.4, 0.7, 0.9])
fu = np.exp(u)
print(f"integral of e^x on [0, 1] from u = {u.tolist()}: e^u = {show(fu)}   estimate = {fu.mean():.4f}   "
      f"exact e - 1 = {np.e - 1:.4f}")
print(f"  sample sd of e^u = {fu.std(ddof=1):.4f}   standard error = {fu.std(ddof=1) / 2:.4f}")

# %%
# From scratch: the law of large numbers as running means, and a variable that breaks it.
section("from scratch: law of large numbers")


def running_mean(x: FloatArray) -> FloatArray:
    """Mean of the first k values for every k. x (n,) -> (n,)."""
    return np.cumsum(x) / np.arange(1, len(x) + 1)


paths = rng.integers(1, 7, size=(N_PATHS, N_RUNNING)).astype(np.float64)  # (N_PATHS, N_RUNNING)
run = np.apply_along_axis(running_mean, 1, paths)  # (N_PATHS, N_RUNNING)
checkpoints = [10, 100, 1_000, 10_000, 100_000]
for k in checkpoints:
    devs = np.abs(run[:, k - 1] - 3.5)
    print(f"after {k:>7,} rolls: largest |running mean - 3.5| over {N_PATHS} paths = {devs.max():.4f}   "
          f"2 sigma/sqrt(n) = {2 * SIGMA_DIE / np.sqrt(k):.4f}")
cauchy = rng.standard_cauchy(N_RUNNING)  # no mean exists: the LLN does not apply
run_c = running_mean(cauchy)
print("Cauchy running mean at the same checkpoints:", show(run_c[np.array(checkpoints) - 1]))

# %%
# From scratch: the central limit theorem with means of a skewed variable.
section("from scratch: central limit theorem")


def ks_to_normal(z: FloatArray) -> float:
    """Largest gap between the empirical CDF of z and the N(0, 1) CDF. z (n,) -> scalar."""
    zs = np.sort(z)
    n = len(zs)
    F = stats.norm.cdf(zs)
    return float(max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n)))


clt_z: dict[int, FloatArray] = {}
print(f"{'n':>4}{'mean of means':>15}{'sd of means':>13}{'1/sqrt(n)':>11}{'skewness':>10}{'2/sqrt(n)':>11}{'gap to N(0,1)':>15}")
for n in CLT_SIZES:
    means = rng.exponential(1.0, size=(N_CLT, n)).mean(axis=1)  # (N_CLT,) one mean per row
    z = (means - 1.0) / (1.0 / np.sqrt(n))  # standardize with the exact mean 1 and sd 1/sqrt(n)
    clt_z[n] = z
    print(f"{n:>4}{means.mean():>15.4f}{means.std():>13.4f}{1 / np.sqrt(n):>11.4f}"
          f"{stats.skew(means):>10.4f}{2 / np.sqrt(n):>11.4f}{ks_to_normal(z):>15.4f}")
tail = {n: float(np.mean(clt_z[n] > 1.96)) for n in CLT_SIZES}
print("P(standardized mean > 1.96), normal says 0.0250:", {n: round(t, 4) for n, t in tail.items()})

# %%
# From scratch: Monte Carlo estimates of pi and of an integral, with error bars.
section("from scratch: Monte Carlo with error bars")


def mc_pi(n: int, gen: np.random.Generator) -> tuple[float, float]:
    """Estimate pi from n points in the unit square. -> (estimate, standard error)."""
    xy = gen.random((n, 2))  # (n, 2)
    hit = (xy**2).sum(axis=1) < 1.0  # (n,) inside the quarter circle of radius 1
    p = hit.mean()
    return 4 * p, 4 * np.sqrt(p * (1 - p) / n)


def mc_integral(f, n: int, gen: np.random.Generator) -> tuple[float, float]:
    """Estimate the integral of f on [0, 1] as the mean of f(U). -> (estimate, standard error)."""
    fx = f(gen.random(n))  # (n,)
    return float(fx.mean()), float(fx.std(ddof=1) / np.sqrt(n))


mc_rows = []
print(f"{'n':>9}{'pi estimate':>13}{'+-1.96 SE':>11}{'|error|':>10}{'integral':>11}{'+-1.96 SE':>11}{'|error|':>10}")
for n in MC_SIZES:
    pi_hat, pi_se = mc_pi(n, rng)
    i_hat, i_se = mc_integral(np.exp, n, rng)
    mc_rows.append((n, abs(pi_hat - np.pi), pi_se, abs(i_hat - (np.e - 1)), i_se))
    print(f"{n:>9,}{pi_hat:>13.5f}{Z95 * pi_se:>11.5f}{abs(pi_hat - np.pi):>10.5f}"
          f"{i_hat:>11.5f}{Z95 * i_se:>11.5f}{abs(i_hat - (np.e - 1)):>10.5f}")
print(f"standard error at n = 100 divided by standard error at n = 1,000,000: pi {mc_rows[0][2] / mc_rows[-1][2]:.1f}, "
      f"integral {mc_rows[0][4] / mc_rows[-1][4]:.1f}   (1/sqrt(n) predicts 100)")

section("from scratch: do 95% error bars cover the truth 95% of the time?")
n_cov = 1_000
hits = 0
for _ in range(N_COVERAGE):
    est, se = mc_integral(np.exp, n_cov, rng)
    hits += abs(est - (np.e - 1)) <= Z95 * se
print(f"{N_COVERAGE} runs of n = {n_cov}: the interval estimate +- 1.96 SE contained e - 1 in "
      f"{hits / N_COVERAGE:.3f} of runs")

# %%
# With SciPy: deterministic quadrature for the same integral, and scipy's normal for the CLT.
section("with scipy")
val, err = integrate.quad(np.exp, 0.0, 1.0)
print(f"scipy.integrate.quad: {val:.10f}   (its own error estimate {err:.1e})   exact e - 1 = {np.e - 1:.10f}")
np.testing.assert_allclose(val, np.e - 1, rtol=1e-12)
print("quad matches the exact value: OK")
big_hat, big_se = mc_integral(np.exp, 1_000_000, np.random.default_rng(SEED))
assert abs(big_hat - val) < 4 * big_se
print(f"Monte Carlo with 10^6 draws: {big_hat:.5f} +- {Z95 * big_se:.5f}, agrees with quad within its error bar: OK")
print(f"stats.norm.ppf(0.975) = {stats.norm.ppf(0.975):.4f}   (the 1.96 above)")
for n in (5, 30):
    exact = stats.gamma(a=n, scale=1 / n)  # the mean of n Exponential(1) draws is exactly Gamma(n, 1/n)
    print(f"n = {n:>2}: exact P(mean > 1 + 1.96/sqrt(n)) = {exact.sf(1 + 1.96 / np.sqrt(n)):.4f}   "
          f"CLT says 0.0250   simulated {tail[n]:.4f}")

# %%
# Key plot: LLN running means, CLT histograms, Monte Carlo error versus n.
section("key plot")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), dpi=120)
ax = axes[0]
ks = np.unique(np.logspace(0, np.log10(N_RUNNING), 400).astype(int))  # log-spaced points to plot
for i in range(N_PATHS):
    ax.plot(ks, run[i, ks - 1], lw=0.8, alpha=0.8)
ax.plot(ks, 3.5 + 2 * SIGMA_DIE / np.sqrt(ks), "k--", lw=1, label="3.5 ± 2σ/√n")
ax.plot(ks, 3.5 - 2 * SIGMA_DIE / np.sqrt(ks), "k--", lw=1)
ax.set_xscale("log")
ax.set_ylim(2, 5)
ax.set(xlabel="rolls so far (log scale)", ylabel="running mean of a die",
       title="LLN: every running mean settles on 3.5")
ax.legend(loc="upper right")
ax = axes[1]
grid = np.linspace(-4, 5, 300)
for n, style in zip([1, 5, 30], ["-", "--", ":"]):
    ax.hist(clt_z[n], bins=np.linspace(-4, 5, 61), density=True, histtype="step", lw=1.4, ls=style,
            label=f"mean of {n} exponential draws")
ax.plot(grid, stats.norm.pdf(grid), "k-", lw=2, alpha=0.5, label="N(0, 1)")
ax.set(xlabel="standardized mean (mean − 1)·√n", ylabel="density",
       title="CLT: averages of skewed draws turn Gaussian")
ax.legend(loc="upper right", fontsize=7)
ax = axes[2]
ns = np.array([r[0] for r in mc_rows], dtype=float)
ax.loglog(ns, [r[1] for r in mc_rows], "o", label="|error|, π")
ax.loglog(ns, [r[3] for r in mc_rows], "s", label="|error|, ∫eˣ dx")
ax.loglog(ns, [r[2] for r in mc_rows], "-", color="tab:blue", label="standard error, π")
ax.loglog(ns, [r[4] for r in mc_rows], "--", color="tab:orange", label="standard error, ∫eˣ dx")
ax.set(xlabel="Monte Carlo samples n (log scale)", ylabel="error (log scale)",
       title="Monte Carlo error shrinks like 1/√n")
ax.legend(loc="lower left", fontsize=7)
fig.tight_layout()
path_png = IMG / "a5_4_lln_clt_mc.png"
fig.savefig(path_png, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path_png.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Replace the exponential in the CLT cell with rng.lognormal(0, 1.5). How large must n be now?
# 2. Plot run_c (the Cauchy running mean). Does it ever settle? Compare its jumps with cauchy.max().
# 3. Estimate the integral of e^x with u and 1 - u paired ("antithetic" draws). Is the error bar smaller?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
