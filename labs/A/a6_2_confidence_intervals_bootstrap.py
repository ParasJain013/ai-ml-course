# %% [markdown]
# # A6.2 Confidence intervals and the bootstrap — lab
# Lesson: lessons/A/a6_2_confidence_intervals_bootstrap.md
# Run: uv run python labs/A/a6_2_confidence_intervals_bootstrap.py   (CPU, defaults: ~15 s)
# GPU: no

# %%
from __future__ import annotations

import itertools
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:  # plain `uv run`: draw to files, never open windows
    matplotlib.use("Agg")
    # matplotlib 3.11 warns when plt.show() runs under Agg; under `uv run` it is a harmless no-op
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import stats

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
OUT = ROOT / "outputs" / "a6_2"  # scratch figures, gitignored
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
B = 10_000 if GPU_SCALE else 2_000  # bootstrap resamples per interval
N_COVERAGE = 5_000 if GPU_SCALE else 1_000  # repeated samples in the coverage simulation
COVERAGE_SIZES = [5, 10, 20, 50, 100]  # sample sizes in the coverage simulation
N_BCA = 1_000 if GPU_SCALE else 300  # runs of scipy's BCa interval (slower, so fewer)
CONF = 0.95
TRUE_MEAN, TRUE_MEDIAN = 1.0, float(np.log(2.0))  # of the Exponential(1) used in the simulations

# %%
# Data: the 4-value hand sample, a skewed sample of 40 session lengths, and two groups to compare.
section("data")
x_hand = np.array([3.0, 5.0, 4.0, 8.0])
sessions = rng.exponential(scale=6.0, size=40)  # minutes; skewed like real session lengths
group_a = rng.lognormal(np.log(20.0), 0.8, size=200)  # e.g. order values in variant A
group_b = rng.lognormal(np.log(24.0), 0.8, size=200)  # and in variant B
print("hand sample:", x_hand.tolist())
print(f"sessions: n = {len(sessions)}   mean = {sessions.mean():.4f}   median = {np.median(sessions):.4f}   "
      f"max = {sessions.max():.4f}   (true mean 6, true median {6 * np.log(2):.4f})")
print(f"group A: n = {len(group_a)}   median = {np.median(group_a):.4f}   "
      f"group B: n = {len(group_b)}   median = {np.median(group_b):.4f}")

# %%
# Worked example by hand: t-interval, three bootstrap resamples, the exact bootstrap, Wilson.
section("worked example by hand")
n_h = len(x_hand)
mean_h, s_h = x_hand.mean(), x_hand.std(ddof=1)
se_h = s_h / np.sqrt(n_h)
t_crit = stats.t.ppf(0.975, df=n_h - 1)
print(f"mean = {mean_h:.4f}   deviations = {(x_hand - mean_h).tolist()}   s = sqrt(14/3) = {s_h:.4f}   "
      f"SE = s/2 = {se_h:.4f}")
print(f"t(0.975, df=3) = {t_crit:.4f}   t-interval = {mean_h:.0f} +- {t_crit * se_h:.4f} = "
      f"[{mean_h - t_crit * se_h:.4f}, {mean_h + t_crit * se_h:.4f}]")
print(f"z-interval (wrong for n = 4) = {mean_h:.0f} +- {1.96 * se_h:.4f} = [{mean_h - 1.96 * se_h:.4f}, "
      f"{mean_h + 1.96 * se_h:.4f}]")
for idx in ([0, 0, 3, 2], [1, 3, 3, 2], [2, 2, 0, 1]):  # three resamples, written as row indices
    r = x_hand[idx]
    print(f"resample indices {idx} -> {r.tolist()} -> mean {r.mean():.2f}")
all_idx = np.array(list(itertools.product(range(n_h), repeat=n_h)))  # (256, 4): every resample
exact_means = x_hand[all_idx].mean(axis=1)  # (256,) each equally likely
print(f"all {len(all_idx)} resamples: mean of bootstrap means = {exact_means.mean():.4f}   "
      f"sd of bootstrap means = {exact_means.std():.4f}   sqrt(sum (x - mean)^2 / n / n) = "
      f"{np.sqrt(np.mean((x_hand - mean_h) ** 2) / n_h):.4f}")
lo_e, hi_e = np.quantile(exact_means, [0.025, 0.975])
print(f"exact percentile bootstrap interval = [{lo_e:.4f}, {hi_e:.4f}]")
k, m = 7, 20
p_hat = k / m
wald_half = 1.96 * np.sqrt(p_hat * (1 - p_hat) / m)
print(f"proportion {k}/{m} = {p_hat}: Wald 0.35 +- {wald_half:.4f} = [{p_hat - wald_half:.4f}, {p_hat + wald_half:.4f}]")


def wilson(k: int, m: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/m. -> (low, high)."""
    p = k / m
    centre = (p + z**2 / (2 * m)) / (1 + z**2 / m)
    half = z / (1 + z**2 / m) * np.sqrt(p * (1 - p) / m + z**2 / (4 * m**2))
    return centre - half, centre + half


lo_w, hi_w = wilson(k, m)
print(f"Wilson: centre = {(p_hat + 1.96**2 / 40) / (1 + 1.96**2 / 20):.4f}   interval = [{lo_w:.4f}, {hi_w:.4f}]")
lo_0, hi_0 = wilson(0, 50)
print(f"0/50 failures: Wald = [0, 0]   Wilson = [{lo_0:.4f}, {hi_0:.4f}]   rule of three 3/50 = {3 / 50:.4f}")

# %%
# From scratch: the percentile bootstrap for the mean and the median, against the t-interval.
section("from scratch: bootstrap intervals")


def bootstrap_stats(
    x: FloatArray, stat: Callable[..., FloatArray], n_boot: int, gen: np.random.Generator
) -> FloatArray:
    """Statistic of n_boot resamples drawn with replacement. x (n,) -> (n_boot,)."""
    idx = gen.integers(0, len(x), size=(n_boot, len(x)))  # (n_boot, n) row indices
    return stat(x[idx], axis=1)  # (n_boot,)


def percentile_ci(boot: FloatArray, conf: float = CONF) -> tuple[float, float]:
    """Middle conf share of the bootstrap distribution. boot (n_boot,) -> (low, high)."""
    a = (1 - conf) / 2
    lo, hi = np.quantile(boot, [a, 1 - a])
    return float(lo), float(hi)


def t_interval(x: FloatArray, conf: float = CONF) -> tuple[float, float]:
    """Student t interval for the mean. x (n,) -> (low, high)."""
    se = x.std(ddof=1) / np.sqrt(len(x))
    half = stats.t.ppf(1 - (1 - conf) / 2, df=len(x) - 1) * se
    return float(x.mean() - half), float(x.mean() + half)


boot_mean = bootstrap_stats(sessions, np.mean, B, rng)
boot_median = bootstrap_stats(sessions, np.median, B, rng)
print(f"mean   {sessions.mean():.4f}: t-interval {show(np.array(t_interval(sessions)))}   "
      f"bootstrap {show(np.array(percentile_ci(boot_mean)))}   bootstrap SE {boot_mean.std(ddof=1):.4f}   "
      f"s/sqrt(n) {sessions.std(ddof=1) / np.sqrt(len(sessions)):.4f}")
print(f"median {np.median(sessions):.4f}: no simple formula            "
      f"bootstrap {show(np.array(percentile_ci(boot_median)))}   bootstrap SE {boot_median.std(ddof=1):.4f}")
print(f"distinct values among {B} bootstrap medians: {len(np.unique(boot_median))}   (the median can only land on "
      "a data value or a midpoint)")

med_a = bootstrap_stats(group_a, np.median, B, rng)  # resample each group separately
med_b = bootstrap_stats(group_b, np.median, B, rng)
diff = med_b - med_a  # (B,) bootstrap distribution of the difference in medians
d_obs = np.median(group_b) - np.median(group_a)
lo_d, hi_d = percentile_ci(diff)
print(f"difference in medians B - A = {d_obs:.4f}   95% bootstrap CI [{lo_d:.4f}, {hi_d:.4f}]   "
      f"share of resamples with B <= A: {np.mean(diff <= 0):.4f}")

# %%
# From scratch: coverage. Does each 95% interval contain the true value 95% of the time?
section("from scratch: coverage simulation")


def coverage_run(n: int, runs: int, gen: np.random.Generator) -> dict[str, FloatArray]:
    """Intervals from `runs` Exponential(1) samples of size n. -> name: (runs, 2) array."""
    out: dict[str, list[tuple[float, float]]] = {"z": [], "t": [], "boot mean": [], "boot median": []}
    for _ in range(runs):
        x = gen.exponential(1.0, size=n)
        se = x.std(ddof=1) / np.sqrt(n)
        out["z"].append((x.mean() - 1.96 * se, x.mean() + 1.96 * se))
        out["t"].append(t_interval(x))
        out["boot mean"].append(percentile_ci(bootstrap_stats(x, np.mean, B, gen)))
        out["boot median"].append(percentile_ci(bootstrap_stats(x, np.median, B, gen)))
    return {name: np.array(v) for name, v in out.items()}


def covered(ci: FloatArray, truth: float) -> FloatArray:
    """True where an interval contains truth. ci (runs, 2) -> (runs,)."""
    return (ci[:, 0] <= truth) & (truth <= ci[:, 1])


coverage: dict[int, dict[str, float]] = {}
intervals_n20: dict[str, FloatArray] = {}
print(f"Exponential(1) samples, {N_COVERAGE} runs per n, {B} resamples per bootstrap; target 0.95")
print(f"{'n':>4}{'z mean':>9}{'t mean':>9}{'boot mean':>11}{'boot median':>13}{'t width':>9}{'boot width':>12}")
for n in COVERAGE_SIZES:
    cis = coverage_run(n, N_COVERAGE, rng)
    if n == 20:
        intervals_n20 = cis
    cov = {name: float(covered(ci, TRUE_MEDIAN if name == "boot median" else TRUE_MEAN).mean())
           for name, ci in cis.items()}
    coverage[n] = cov
    width_t = np.mean(cis["t"][:, 1] - cis["t"][:, 0])
    width_b = np.mean(cis["boot mean"][:, 1] - cis["boot mean"][:, 0])
    print(f"{n:>4}{cov['z']:>9.3f}{cov['t']:>9.3f}{cov['boot mean']:>11.3f}{cov['boot median']:>13.3f}"
          f"{width_t:>9.3f}{width_b:>12.3f}")
mc_se = np.sqrt(0.95 * 0.05 / N_COVERAGE)
print(f"Monte Carlo standard error of each coverage number: sqrt(0.95 * 0.05 / {N_COVERAGE}) = {mc_se:.4f}")
gauss_cov = np.mean([covered(np.array([t_interval(rng.normal(0, 1, 5))]), 0.0)[0] for _ in range(N_COVERAGE)])
print(f"control: t-interval on Gaussian data, n = 5: coverage {gauss_cov:.3f}")

# %%
# With SciPy: t.interval, scipy.stats.bootstrap (percentile and BCa), and the Wilson interval.
section("with scipy")
lo_s, hi_s = stats.t.interval(CONF, df=len(sessions) - 1, loc=sessions.mean(), scale=stats.sem(sessions))
np.testing.assert_allclose((lo_s, hi_s), t_interval(sessions), rtol=1e-12)
print(f"stats.t.interval: [{lo_s:.4f}, {hi_s:.4f}]   matches t_interval: OK")
res_p = stats.bootstrap((sessions,), np.mean, n_resamples=B, method="percentile", rng=np.random.default_rng(SEED))
res_b = stats.bootstrap((sessions,), np.mean, n_resamples=B, method="BCa", rng=np.random.default_rng(SEED))
mine = np.array(percentile_ci(boot_mean))
print(f"scipy percentile: [{res_p.confidence_interval.low:.4f}, {res_p.confidence_interval.high:.4f}]   "
      f"mine: {show(mine)}   scipy BCa: [{res_b.confidence_interval.low:.4f}, {res_b.confidence_interval.high:.4f}]")
# Different random resamples: agreement is up to Monte Carlo error. With B = 2,000 the 2.5% quantile of
# the bootstrap means moves by roughly 0.1 between seeds here, so we allow 0.25 (about 4% of the mean).
np.testing.assert_allclose(mine, [res_p.confidence_interval.low, res_p.confidence_interval.high], atol=0.25)
print("percentile intervals agree within Monte Carlo error: OK")
np.testing.assert_allclose(res_p.standard_error, boot_mean.std(ddof=1), rtol=0.1)
print(f"bootstrap standard error: scipy {res_p.standard_error:.4f}   mine {boot_mean.std(ddof=1):.4f}   OK")
ci_w = stats.binomtest(k, m).proportion_ci(confidence_level=CONF, method="wilson")
# scipy uses the exact 97.5% quantile 1.959964..., not the rounded 1.96 of the hand calculation
np.testing.assert_allclose((ci_w.low, ci_w.high), wilson(k, m, z=stats.norm.ppf(0.975)), rtol=1e-10)
print(f"binomtest(7, 20).proportion_ci(method='wilson'): [{ci_w.low:.4f}, {ci_w.high:.4f}]   matches wilson: OK")
ci_cp = stats.binomtest(k, m).proportion_ci(confidence_level=CONF, method="exact")
print(f"exact (Clopper-Pearson) for comparison: [{ci_cp.low:.4f}, {ci_cp.high:.4f}]")

bca_hits = 0
pct_hits = 0
for _ in range(N_BCA):
    x = rng.exponential(1.0, size=20)
    r_b = stats.bootstrap((x,), np.mean, n_resamples=B, method="BCa", rng=rng).confidence_interval
    r_p = stats.bootstrap((x,), np.mean, n_resamples=B, method="percentile", rng=rng).confidence_interval
    bca_hits += r_b.low <= TRUE_MEAN <= r_b.high
    pct_hits += r_p.low <= TRUE_MEAN <= r_p.high
print(f"n = 20, {N_BCA} runs: coverage of scipy percentile {pct_hits / N_BCA:.3f}   BCa {bca_hits / N_BCA:.3f}")

# %%
# Key plot: 50 intervals from repeated samples, and coverage against n.
section("key plot")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=120)
ax = axes[0]
ci50 = intervals_n20["t"][:50]
hit = covered(ci50, TRUE_MEAN)
for i, (lo, hi) in enumerate(ci50):
    if hit[i]:
        ax.plot([lo, hi], [i, i], "-", color="tab:blue", lw=1.5)
    else:
        ax.plot([lo, hi], [i, i], "-", color="tab:red", lw=2.5)
        ax.plot([lo, hi], [i, i], "x", color="tab:red", ms=6)
ax.axvline(TRUE_MEAN, color="k", ls="--", lw=1)
ax.text(TRUE_MEAN, 51, "true mean 1", ha="center", va="bottom")
ax.set_ylim(-1, 54)
ax.set(xlabel="95% t-interval for the mean (unitless)", ylabel="repeated sample #",
       title=f"50 samples of n = 20: {int((~hit).sum())} of 50 intervals (red, x) miss the truth")
ax = axes[1]
ns = np.array(COVERAGE_SIZES)
for name, style in [("z", "s:"), ("t", "o-"), ("boot mean", "^--"), ("boot median", "d-.")]:
    ax.plot(ns, [coverage[n][name] for n in COVERAGE_SIZES], style, label=name)
ax.axhline(0.95, color="k", lw=1, label="target 0.95")
ax.set_xscale("log")
ax.set_xticks(ns, [str(v) for v in ns])
ax.set(xlabel="sample size n (log scale)", ylabel="share of intervals containing the truth",
       title="Skewed data: 95% intervals undercover when n is small")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
path_png = IMG / "a6_2_coverage.png"
fig.savefig(path_png, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path_png.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Replace rng.exponential with rng.normal(1, 1, n) in coverage_run. Which methods reach 0.95 now?
# 2. Bootstrap np.max of the sessions. Plot the bootstrap distribution: why is it useless?
# 3. Make the sessions autocorrelated (x[t] = 0.8 x[t-1] + noise) and compare the i.i.d. bootstrap SE
#    with a block bootstrap that resamples blocks of 10 consecutive values.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
