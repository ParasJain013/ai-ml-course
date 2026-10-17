# %% [markdown]
# # A5.1 Random variables, expectation, variance and covariance — lab
# Lesson: lessons/A/a5_1_random_variables_expectation.md
# Run: uv run python labs/A/a5_1_random_variables_expectation.py   (CPU, defaults: ~3 s)
# GPU: no

# %%
from __future__ import annotations

import sys
import time
import warnings
from fractions import Fraction
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
OUT = ROOT / "outputs" / "a5_1"  # scratch figures, gitignored
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
# Nothing here needs a GPU; GPU_SCALE only makes the simulations bigger (and slower, ~1 min).
GPU_SCALE = False
N_BIG = 10_000_000 if GPU_SCALE else 1_000_000  # rolls for the "close to exact" estimates
N_REPEATS = 2_000 if GPU_SCALE else 500  # repeated experiments per sample size
SIZES = [10, 100, 1_000, 10_000, 100_000]  # sample sizes for "how far do estimates wander"

# %%
# Data: a million rolls of two fair dice and a million fair coin flips, from one generator.
section("data")
die1 = rng.integers(1, 7, size=N_BIG).astype(np.float64)  # (N_BIG,) values 1..6
die2 = rng.integers(1, 7, size=N_BIG).astype(np.float64)  # (N_BIG,) an independent second die
coin = (rng.random(N_BIG) < 0.5).astype(np.float64)  # (N_BIG,) 1 = heads, 0 = tails
total = die1 + die2  # (N_BIG,) the two-variable process: first die and the sum
print("die1", die1.shape, " first 10 rolls:", die1[:10].astype(int).tolist())
print("coin", coin.shape, " first 10 flips:", coin[:10].astype(int).tolist())
print("total = die1 + die2, first 10:", total[:10].astype(int).tolist())

# %%
# Worked example by hand: exact moments from the probability table, as fractions.
section("worked example: one fair die")
faces = [Fraction(k) for k in range(1, 7)]
p_face = Fraction(1, 6)
e_x = sum(x * p_face for x in faces)
e_x2 = sum(x * x * p_face for x in faces)
var_x = e_x2 - e_x**2
print(f"E[X] = {e_x} = {float(e_x):.4f}   E[X^2] = {e_x2} = {float(e_x2):.4f}")
print(f"Var[X] = E[X^2] - E[X]^2 = {var_x} = {float(var_x):.4f}   sd = {float(var_x) ** 0.5:.4f}")
print(f"Var[X] the long way, E[(X - 3.5)^2] = {sum((x - e_x) ** 2 * p_face for x in faces)}")

section("worked example: a joint table")
# P(X = x, Y = y) for x, y in {0, 1}: rows are x, columns are y.
joint = np.array([[0.4, 0.1], [0.1, 0.4]])  # (2, 2)
vals = np.array([0.0, 1.0])
p_x, p_y = joint.sum(axis=1), joint.sum(axis=0)  # marginals (2,), (2,)
ex, ey = p_x @ vals, p_y @ vals
exy = sum(joint[i, j] * vals[i] * vals[j] for i in range(2) for j in range(2))
cov_xy = exy - ex * ey
var_x01, var_y01 = p_x @ vals**2 - ex**2, p_y @ vals**2 - ey**2
print(f"marginals P(X) = {show(p_x)}  P(Y) = {show(p_y)}")
print(f"E[X] = {ex:.2f}  E[Y] = {ey:.2f}  E[XY] = {exy:.2f}  Cov = {cov_xy:.2f}")
print(f"Var[X] = {var_x01:.2f}  Var[Y] = {var_y01:.2f}  correlation = {cov_xy / np.sqrt(var_x01 * var_y01):.2f}")
print(f"independent? P(X=1, Y=1) = {joint[1, 1]:.2f}  vs  P(X=1) P(Y=1) = {p_x[1] * p_y[1]:.2f}")

section("worked example: uncorrelated but dependent")
x3 = np.array([-1.0, 0.0, 1.0])  # X uniform on {-1, 0, 1}, Y = X^2
p3 = np.full(3, 1 / 3)
cov_trap = p3 @ (x3 * x3**2) - (p3 @ x3) * (p3 @ x3**2)
print(f"Cov[X, X^2] = E[X^3] - E[X] E[X^2] = {cov_trap + 0.0:.4f}")
print(f"P(Y = 1) = {p3[[0, 2]].sum():.4f}   but P(Y = 1 | X = 1) = 1.0: Y is a function of X")

# %%
# From scratch: the estimators, then the simulation against the exact values.
section("from scratch: estimates from a million draws")


def sample_mean(x: FloatArray) -> float:
    """Sample mean (1/n) sum x_i. x (n,) -> scalar."""
    return float(x.sum() / len(x))


def sample_var(x: FloatArray, ddof: int = 1) -> float:
    """Sample variance, dividing by n - ddof (ddof=1: Bessel's correction). x (n,) -> scalar."""
    r = x - sample_mean(x)  # (n,) deviations from the sample mean
    return float(r @ r / (len(x) - ddof))


def sample_cov(x: FloatArray, y: FloatArray, ddof: int = 1) -> float:
    """Sample covariance of two variables, dividing by n - ddof. x (n,), y (n,) -> scalar."""
    assert x.shape == y.shape, (x.shape, y.shape)
    return float((x - sample_mean(x)) @ (y - sample_mean(y)) / (len(x) - ddof))


exact = {
    "die: mean": 3.5,
    "die: variance": 35 / 12,
    "coin: mean": 0.5,
    "coin: variance": 0.25,
    "total: mean": 7.0,
    "total: variance": 35 / 6,
    "Cov[die1, total]": 35 / 12,
    "Cov[die1, die2]": 0.0,
}
estimate = {
    "die: mean": sample_mean(die1),
    "die: variance": sample_var(die1),
    "coin: mean": sample_mean(coin),
    "coin: variance": sample_var(coin),
    "total: mean": sample_mean(total),
    "total: variance": sample_var(total),
    "Cov[die1, total]": sample_cov(die1, total),
    "Cov[die1, die2]": sample_cov(die1, die2),
}
print(f"{'quantity':<18}{'exact':>9}{'estimate':>11}{'error':>10}")
for key in exact:
    print(f"{key:<18}{exact[key]:>9.4f}{estimate[key]:>11.4f}{estimate[key] - exact[key]:>+10.4f}")
corr = estimate["Cov[die1, total]"] / np.sqrt(estimate["die: variance"] * estimate["total: variance"])
print(f"correlation of die1 and total: estimate {corr:.4f}   exact 1/sqrt(2) = {1 / np.sqrt(2):.4f}")

# %%
# From scratch: the rules for sums and scaling, checked on the same draws.
section("from scratch: rules for scaling and sums")
y_lin = 10 - 2 * die1  # Y = a X + b with a = -2, b = 10
print(f"E[10 - 2X]: estimate {sample_mean(y_lin):.4f}   rule 10 - 2 E[X] = {10 - 2 * 3.5:.4f}")
print(f"Var[10 - 2X]: estimate {sample_var(y_lin):.4f}   rule (-2)^2 Var[X] = {4 * 35 / 12:.4f}")
diff = die1 - total  # = -die2
print(f"Var[die1 + die2] = {sample_var(total):.4f}   Var[die1] + Var[die2] = "
      f"{sample_var(die1) + sample_var(die2):.4f}   (+ 2 Cov = {2 * sample_cov(die1, die2):+.4f})")
print(f"Var[die1 - total] = {sample_var(diff):.4f}   Var + Var - 2 Cov = "
      f"{sample_var(die1) + sample_var(total) - 2 * sample_cov(die1, total):.4f}")

# %%
# From scratch: how far estimates wander. Repeat the whole experiment N_REPEATS times per size.
section("from scratch: how far estimates wander")
sigma = np.sqrt(35 / 12)
spread: dict[int, FloatArray] = {}
print(f"{'n':>7}{'sd of mean estimates':>23}{'sigma/sqrt(n)':>15}{'sd of variance estimates':>27}")
for n in SIZES:
    rolls = rng.integers(1, 7, size=(N_REPEATS, n)).astype(np.float64)  # (N_REPEATS, n)
    means = rolls.mean(axis=1)  # (N_REPEATS,) one estimate per experiment
    variances = rolls.var(axis=1, ddof=1)  # (N_REPEATS,)
    spread[n] = means
    print(f"{n:>7}{means.std():>23.4f}{sigma / np.sqrt(n):>15.4f}{variances.std():>27.4f}")

# %%
# With NumPy and SciPy: the library estimators and exact moments.
section("with numpy and scipy")
print(f"np.mean = {np.mean(die1):.4f}   np.var(ddof=1) = {np.var(die1, ddof=1):.4f}"
      f"   np.var (default ddof=0) = {np.var(die1):.4f}")
C = np.cov(np.stack([die1, die2, total]))  # (3, 3): np.cov wants variables in rows by default
print("np.cov of (die1, die2, total) =", show(C))
np.testing.assert_allclose(sample_mean(die1), np.mean(die1), rtol=1e-12)
np.testing.assert_allclose(sample_var(die1), np.var(die1, ddof=1), rtol=1e-10)
np.testing.assert_allclose(sample_cov(die1, total), C[0, 2], rtol=1e-10)
print("matches numpy: OK")
die_rv = stats.randint(1, 7)  # discrete uniform on 1..6 (upper bound excluded, like range)
m, v = die_rv.stats(moments="mv")
print(f"scipy.stats.randint(1, 7): mean = {float(m):.4f}   variance = {float(v):.4f}")
spinner = stats.rv_discrete(values=([0, 2, 10], [0.6, 0.3, 0.1]))  # any finite table
print(f"rv_discrete over {{0, 2, 10}} with p = (0.6, 0.3, 0.1): mean = {spinner.mean():.4f}"
      f"   variance = {spinner.var():.4f}")
np.testing.assert_allclose([float(m), float(v)], [3.5, 35 / 12], rtol=1e-12)
print("scipy's exact moments match the hand calculation: OK")

# %%
# Key plot: 200 estimates of the die's mean at each sample size, with the +-2 sigma/sqrt(n) band.
section("key plot")
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
plot_rng = np.random.default_rng(SEED + 1)  # jitter only, so the estimates above are unchanged
for n in SIZES:
    pts = spread[n][:200]
    jitter = np.exp(plot_rng.uniform(-0.12, 0.12, size=len(pts)))
    ax.plot(n * jitter, pts, "o", ms=2.5, alpha=0.35, color="tab:blue")
ns = np.logspace(np.log10(SIZES[0]) - 0.2, np.log10(SIZES[-1]) + 0.2, 100)
ax.plot(ns, 3.5 + 2 * sigma / np.sqrt(ns), "--", color="tab:orange", label="3.5 ± 2σ/√n")
ax.plot(ns, 3.5 - 2 * sigma / np.sqrt(ns), "--", color="tab:orange")
ax.axhline(3.5, color="k", lw=1, label="exact E[X] = 3.5")
ax.set_xscale("log")
ax.set_xlabel("rolls per experiment, n (log scale)")
ax.set_ylabel("estimated mean of a die")
ax.set_title("Each estimate is random; 100× more rolls makes the spread 10× smaller")
ax.legend(loc="upper right")
fig.tight_layout()
path_png = IMG / "a5_1_estimates_wander.png"
fig.savefig(path_png, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path_png.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Replace the die with a "loaded" die: rng.choice(6, p=[...]) + 1. Recompute the exact moments.
# 2. Use ddof=0 in the wandering cell with n = 10. Is the average variance estimate below 35/12?
# 3. Make die2 = die1 for half of the rows. What happens to Cov[die1, die2] and Var[total]?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
