# %% [markdown]
# # A5.2 Common distributions — lab
# Lesson: lessons/A/a5_2_common_distributions.md
# Run: uv run python labs/A/a5_2_common_distributions.py   (CPU, defaults: ~3 s)
# GPU: no

# %%
from __future__ import annotations

import sys
import time
import warnings
from math import comb, exp, factorial
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
IntArray = npt.NDArray[np.int64]
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
OUT = ROOT / "outputs" / "a5_2"  # scratch figures, gitignored
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
# Nothing here needs a GPU; GPU_SCALE only draws more samples.
GPU_SCALE = False
N_SAMPLES = 2_000_000 if GPU_SCALE else 200_000  # draws per sampler
P_COIN = 0.3  # Bernoulli success probability
N_TRIALS = 10  # binomial: number of Bernoulli trials
CAT_PROBS = np.array([0.2, 0.5, 0.3])  # categorical class probabilities, K = 3
RATE = 3.0  # Poisson mean rate (events per interval)
MU = np.array([1.0, -1.0])  # (2,) multivariate Gaussian mean
SIGMA = np.array([[4.0, 2.4], [2.4, 2.25]])  # (2, 2) covariance: sd 2 and 1.5, correlation 0.8

# %%
# Data: the only randomness the from-scratch samplers use is rng.random(), uniform on [0, 1).
section("data")
u = rng.random(5)
print("five draws of rng.random():", show(u))
print(f"each sampler below draws {N_SAMPLES:,} values")

# %%
# Worked example by hand: the numbers from the lesson.
section("worked example by hand")
print("Binomial(4, 0.5) pmf k = 0..4:", [round(comb(4, k) * 0.5**4, 4) for k in range(5)])
print("Poisson(rate 2) pmf k = 0..2:", [round(exp(-2) * 2**k / factorial(k), 4) for k in range(3)],
      f"  P(k >= 3) = {1 - sum(exp(-2) * 2**k / factorial(k) for k in range(3)):.4f}")
cdf = np.cumsum(CAT_PROBS)
print(f"categorical p = {CAT_PROBS.tolist()}: cumulative = {show(cdf)}   u = 0.65 -> class "
      f"{int(np.searchsorted(cdf, 0.65, side='right'))}   u = 0.15 -> class "
      f"{int(np.searchsorted(cdf, 0.15, side='right'))}")
u1, u2 = 0.5, 0.125
r, theta = np.sqrt(-2 * np.log(u1)), 2 * np.pi * u2
print(f"Box-Muller u1 = {u1}, u2 = {u2}: r = {r:.4f}  angle = {theta:.4f} rad  "
      f"z = ({r * np.cos(theta):.4f}, {r * np.sin(theta):.4f})")
L = np.linalg.cholesky(SIGMA)
print("Cholesky factor L of Sigma =", show(L), "  L L^T =", show(L @ L.T))
z = np.array([1.0, -1.0])
print(f"z = {show(z)} -> x = mu + L z = {show(MU + L @ z)}")
print(f"Gaussian: P(|Z| < 1) = {stats.norm.cdf(1) - stats.norm.cdf(-1):.4f}   "
      f"P(|Z| < 2) = {stats.norm.cdf(2) - stats.norm.cdf(-2):.4f}   "
      f"P(|Z| < 3) = {stats.norm.cdf(3) - stats.norm.cdf(-3):.4f}")

# %%
# From scratch: samplers built only on rng.random().
section("from scratch: samplers from uniform numbers")


def sample_bernoulli(p: float, size: int, gen: np.random.Generator) -> IntArray:
    """1 with probability p, else 0. -> (size,)."""
    return (gen.random(size) < p).astype(np.int64)


def sample_binomial(n_trials: int, p: float, size: int, gen: np.random.Generator) -> IntArray:
    """Number of successes in n_trials Bernoulli(p) trials. -> (size,)."""
    return (gen.random((size, n_trials)) < p).sum(axis=1)  # (size, n_trials) -> (size,)


def sample_categorical(probs: FloatArray, size: int, gen: np.random.Generator) -> IntArray:
    """Class index 0..K-1 by inverse CDF. probs (K,) summing to 1 -> (size,)."""
    cdf = np.cumsum(probs)  # (K,)
    cdf[-1] = 1.0  # guard against rounding: the last class must catch every u < 1
    return np.searchsorted(cdf, gen.random(size), side="right")  # first class with cdf > u


def sample_uniform(a: float, b: float, size: int, gen: np.random.Generator) -> FloatArray:
    """Uniform on [a, b): stretch and shift [0, 1). -> (size,)."""
    return a + (b - a) * gen.random(size)


def sample_standard_normal(size: int, gen: np.random.Generator) -> FloatArray:
    """Standard normal N(0, 1) by Box-Muller. -> (size,)."""
    half = (size + 1) // 2
    u1 = 1.0 - gen.random(half)  # in (0, 1], so log(u1) is finite
    u2 = gen.random(half)
    r = np.sqrt(-2.0 * np.log(u1))  # (half,) radius
    theta = 2.0 * np.pi * u2  # (half,) angle
    return np.concatenate([r * np.cos(theta), r * np.sin(theta)])[:size]  # two normals per pair


def sample_mvn(mu: FloatArray, Sigma: FloatArray, size: int, gen: np.random.Generator) -> FloatArray:
    """Multivariate Gaussian N(mu, Sigma) by x = mu + L z. mu (d,), Sigma (d, d) -> (size, d)."""
    d = len(mu)
    L = np.linalg.cholesky(Sigma)  # (d, d) lower triangular, L L^T = Sigma
    Z = sample_standard_normal(size * d, gen).reshape(size, d)  # (size, d) independent N(0, 1)
    return mu + Z @ L.T  # row i is mu + L z_i; (size, d)


gen = np.random.default_rng(SEED)
bern = sample_bernoulli(P_COIN, N_SAMPLES, gen)
binom = sample_binomial(N_TRIALS, P_COIN, N_SAMPLES, gen)
cat = sample_categorical(CAT_PROBS, N_SAMPLES, gen)
unif = sample_uniform(-1.0, 3.0, N_SAMPLES, gen)
gauss = 2.0 + 0.5 * sample_standard_normal(N_SAMPLES, gen)  # N(2, 0.5^2) from N(0, 1)
mvn = sample_mvn(MU, SIGMA, N_SAMPLES, gen)
assert mvn.shape == (N_SAMPLES, 2), mvn.shape

rows = [
    ("Bernoulli(0.3)", bern, P_COIN, P_COIN * (1 - P_COIN)),
    ("Binomial(10, 0.3)", binom, N_TRIALS * P_COIN, N_TRIALS * P_COIN * (1 - P_COIN)),
    ("Categorical(.2,.5,.3)", cat, CAT_PROBS @ np.arange(3), CAT_PROBS @ np.arange(3) ** 2 - (CAT_PROBS @ np.arange(3)) ** 2),
    ("Uniform(-1, 3)", unif, 1.0, 16 / 12),
    ("Normal(2, 0.5^2)", gauss, 2.0, 0.25),
]
print(f"{'distribution':<23}{'mean':>8}{'formula':>9}{'variance':>10}{'formula':>9}")
for name, x, m, v in rows:
    print(f"{name:<23}{x.mean():>8.4f}{m:>9.4f}{x.var(ddof=1):>10.4f}{v:>9.4f}")
print("multivariate Gaussian: sample mean =", show(mvn.mean(axis=0)), "  target =", show(MU))
print("  sample covariance =", show(np.cov(mvn, rowvar=False)), "  target =", show(SIGMA))

# %%
# With SciPy: overlay on scipy.stats and measure the largest gap.
section("with scipy.stats: the samplers match the exact distributions")


def max_pmf_gap(x: IntArray, pmf: FloatArray) -> float:
    """Largest |empirical frequency - pmf| over the support 0..len(pmf)-1. x (n,) -> scalar."""
    freq = np.bincount(x, minlength=len(pmf))[: len(pmf)] / len(x)
    return float(np.max(np.abs(freq - pmf)))


def ks_gap(x: FloatArray, cdf_fn) -> float:
    """Largest gap between the empirical CDF of x and a CDF (the KS statistic, A6.3). x (n,) -> scalar."""
    xs = np.sort(x)
    n = len(xs)
    F = cdf_fn(xs)  # (n,)
    return float(max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n)))


k10 = np.arange(N_TRIALS + 1)
checks = {
    "Bernoulli vs stats.bernoulli": max_pmf_gap(bern, stats.bernoulli(P_COIN).pmf([0, 1])),
    "Binomial vs stats.binom": max_pmf_gap(binom, stats.binom(N_TRIALS, P_COIN).pmf(k10)),
    "Categorical vs probs": max_pmf_gap(cat, CAT_PROBS),
    "Uniform vs stats.uniform": ks_gap(unif, stats.uniform(loc=-1, scale=4).cdf),
    "Box-Muller vs stats.norm": ks_gap(gauss, stats.norm(loc=2, scale=0.5).cdf),
}
for name, gap in checks.items():
    print(f"{name:<30} largest gap = {gap:.4f}")
tol = 3 / np.sqrt(N_SAMPLES)  # sampling noise in a frequency is at most ~0.5/sqrt(n); 3/sqrt(n) is generous
assert all(g < tol for g in checks.values()), checks
print(f"all gaps below 3/sqrt(n) = {tol:.4f}: OK")
mvn_lib = stats.multivariate_normal(mean=MU, cov=SIGMA).rvs(size=N_SAMPLES, random_state=SEED)  # (n, 2)
print("stats.multivariate_normal sample covariance =", show(np.cov(mvn_lib, rowvar=False)))
np.testing.assert_allclose(np.cov(mvn, rowvar=False), SIGMA, atol=0.05)  # sampling noise ~ 0.01-0.02
print("own multivariate sampler covariance within 0.05 of Sigma: OK")
print("exact moments from scipy:")
for name, rv in [("binom(10, 0.3)", stats.binom(N_TRIALS, P_COIN)), ("poisson(3)", stats.poisson(RATE)),
                 ("uniform(-1, 3)", stats.uniform(loc=-1, scale=4)), ("norm(2, 0.5)", stats.norm(2, 0.5))]:
    m, v = rv.stats(moments="mv")
    print(f"  {name:<16} mean = {float(m):.4f}   variance = {float(v):.4f}")
pois = rng.poisson(RATE, size=N_SAMPLES)  # NumPy's own sampler; the exercise builds one from scratch
print(f"rng.poisson(3): mean = {pois.mean():.4f}   variance = {pois.var(ddof=1):.4f}   "
      f"P(0) = {np.mean(pois == 0):.4f} vs exact {stats.poisson(RATE).pmf(0):.4f}")

# %%
# Key plot: each from-scratch sampler against scipy's exact pmf or density.
section("key plot")
fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), dpi=120)
ax = axes[0, 0]
freq = np.bincount(binom, minlength=N_TRIALS + 1) / N_SAMPLES
ax.bar(k10, freq, color="tab:blue", alpha=0.6, label="from scratch")
ax.plot(k10, stats.binom(N_TRIALS, P_COIN).pmf(k10), "kD", ms=4, label="scipy pmf")
ax.set(title="Binomial(10, 0.3): successes in 10 trials", xlabel="k successes", ylabel="probability")
ax.legend()
ax = axes[0, 1]
k_p = np.arange(13)
ax.bar(k_p, np.bincount(pois, minlength=13)[:13] / N_SAMPLES, color="tab:green", alpha=0.6, label="rng.poisson")
ax.plot(k_p, stats.poisson(RATE).pmf(k_p), "kD", ms=4, label="scipy pmf")
ax.set(title="Poisson(rate 3): events per interval", xlabel="k events", ylabel="probability",
       xticks=np.arange(0, 13, 2))
ax.legend()
ax = axes[0, 2]
ax.bar(np.arange(3), np.bincount(cat, minlength=3) / N_SAMPLES, color="tab:orange", alpha=0.6,
       label="from scratch (inverse CDF)")
ax.plot(np.arange(3), CAT_PROBS, "kD", ms=5, label="target probabilities")
ax.set(title="Categorical(0.2, 0.5, 0.3)", xlabel="class index", ylabel="probability", xticks=[0, 1, 2],
       ylim=(0, 0.72))
ax.legend(loc="upper right", fontsize=8)
ax = axes[1, 0]
ax.hist(unif, bins=40, density=True, histtype="stepfilled", color="tab:purple", alpha=0.6, label="from scratch")
ax.plot([-1, 3], [0.25, 0.25], "k-", label="scipy density 1/4")
ax.set(title="Uniform(-1, 3): flat density", xlabel="x", ylabel="density", ylim=(0, 0.35))
ax.legend(loc="lower center")
ax = axes[1, 1]
xs = np.linspace(0, 4, 200)
ax.hist(gauss, bins=60, density=True, histtype="stepfilled", color="tab:blue", alpha=0.6, label="Box-Muller")
ax.plot(xs, stats.norm(2, 0.5).pdf(xs), "k-", label="scipy density")
ax.set(title="Normal(2, 0.5²) from Box-Muller", xlabel="x", ylabel="density")
ax.legend(loc="upper left", fontsize=8)
ax = axes[1, 2]
ax.plot(mvn[:1500, 0], mvn[:1500, 1], ".", ms=2, color="tab:red", label="x = μ + L z (1,500 draws)")
vals, vecs = np.linalg.eigh(SIGMA)
t = np.linspace(0, 2 * np.pi, 200)
ell = MU + 2 * (vecs @ np.diag(np.sqrt(vals)) @ np.stack([np.cos(t), np.sin(t)])).T  # 2-sd ellipse
ax.plot(ell[:, 0], ell[:, 1], "k-", label="2-sd ellipse of Σ")
ax.set(title="2-D Gaussian, correlation 0.8", xlabel="x₁", ylabel="x₂")
ax.set_aspect("equal")
ax.legend(loc="upper left", fontsize=7)
fig.suptitle("Samplers built only on rng.random() match scipy.stats (200,000 draws each)")
fig.tight_layout()
path_png = IMG / "a5_2_samplers_vs_scipy.png"
fig.savefig(path_png, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path_png.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Remove the `cdf[-1] = 1.0` guard and use probs that do not sum to exactly 1 in float. What happens?
# 2. Sample the 2-D Gaussian with Z @ L instead of Z @ L.T. Which covariance do you get, and why?
# 3. Binomial(1000, 0.003) versus Poisson(3): overlay the two pmfs. Why are they so close?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
