# %% [markdown]
# # A6.2 Confidence intervals and the bootstrap — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a6_2_confidence_intervals_bootstrap_ex2.py   (CPU, ~10 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from scipy import stats

FloatArray = npt.NDArray[np.float64]
SEED = 0
T0 = time.perf_counter()
N_TEST = 500
ACC_1, ACC_2, AGREE = 0.86, 0.88, 0.90
B = 2_000
N_COVERAGE = 500
Z95 = 1.96


def section(name: str) -> None:
    print(f"\n== {name} ==")


def wilson(k: int, m: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion k/m. -> (low, high)."""
    p = k / m
    centre = (p + z**2 / (2 * m)) / (1 + z**2 / m)
    half = z / (1 + z**2 / m) * np.sqrt(p * (1 - p) / m + z**2 / (4 * m**2))
    return centre - half, centre + half


# %%
section("exercise 1 check")
x = np.array([1.2, 0.8, 1.5, 1.1, 2.4])
se = x.std(ddof=1) / np.sqrt(len(x))
t_star = stats.t.ppf(0.975, df=4)
print(f"(a) mean {x.mean():.4f}   s {x.std(ddof=1):.4f}   SE {se:.4f}   t* {t_star:.3f}   "
      f"t interval [{x.mean() - t_star * se:.4f}, {x.mean() + t_star * se:.4f}]   "
      f"z half-width {Z95 * se:.4f} vs t half-width {t_star * se:.4f}")
for idx in ([4, 0, 0, 2, 1], [3, 3, 1, 4, 4]):
    r = x[idx]
    print(f"(b) indices {idx} -> {r.tolist()}   mean {r.mean():.2f}   median {np.median(r):.2f}")
print(f"(c) exact bootstrap SE = sqrt(mean((x - mean)^2)) / sqrt(n) = {x.std(ddof=0) / np.sqrt(len(x)):.4f}")
lo, hi = wilson(0, 120)
lo = max(lo, 0.0)  # exactly 0 for k = 0; floating point gives -1e-18
print(f"(d) 0/120: Wald [0, 0]   rule of three {3 / 120:.4f}   Wilson [{lo:.4f}, {hi:.4f}]")


# %%
def make_pair(gen: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Correctness of two classifiers on the same N_TEST examples. -> (c1, c2), each (N_TEST,)."""
    c1 = (gen.random(N_TEST) < ACC_1).astype(np.float64)
    # Disagree on 10% of examples, with 6% "1 wrong, 2 right" and 4% "1 right, 2 wrong", so that
    # accuracy rises by 0.02. Conditional flip probabilities: 0.06 / 0.14 and 0.04 / 0.86.
    u = gen.random(N_TEST)
    flip = np.where(c1 == 0, u < 0.06 / (1 - ACC_1), u < 0.04 / ACC_1)
    c2 = np.where(flip, 1 - c1, c1)
    return c1, c2


def paired_boot_diff(c1: FloatArray, c2: FloatArray, n_boot: int, gen: np.random.Generator) -> FloatArray:
    """Accuracy difference on resampled examples, same rows for both. c1, c2 (n,) -> (n_boot,)."""
    idx = gen.integers(0, len(c1), size=(n_boot, len(c1)))  # (n_boot, n)
    return c2[idx].mean(axis=1) - c1[idx].mean(axis=1)


def unpaired_boot_diff(c1: FloatArray, c2: FloatArray, n_boot: int, gen: np.random.Generator) -> FloatArray:
    """The wrong way: resample each classifier's results independently. -> (n_boot,)."""
    i1 = gen.integers(0, len(c1), size=(n_boot, len(c1)))
    i2 = gen.integers(0, len(c2), size=(n_boot, len(c2)))
    return c2[i2].mean(axis=1) - c1[i1].mean(axis=1)


section("exercise 2 (a): intervals for one accuracy")
rng = np.random.default_rng(SEED)
c1, c2 = make_pair(rng)
k = int(c1.sum())
acc = k / N_TEST
wald = Z95 * np.sqrt(acc * (1 - acc) / N_TEST)
boot = c1[rng.integers(0, N_TEST, size=(B, N_TEST))].mean(axis=1)
print(f"classifier 1: {k}/{N_TEST} correct = {acc:.4f}")
print(f"Wald      [{acc - wald:.4f}, {acc + wald:.4f}]")
print(f"Wilson    [{wilson(k, N_TEST)[0]:.4f}, {wilson(k, N_TEST)[1]:.4f}]")
print(f"bootstrap [{np.quantile(boot, 0.025):.4f}, {np.quantile(boot, 0.975):.4f}]")

section("exercise 2 (b): difference between two classifiers on the same examples")
print(f"accuracy 1 = {c1.mean():.4f}   accuracy 2 = {c2.mean():.4f}   agreement = {np.mean(c1 == c2):.4f}   "
      f"observed difference = {c2.mean() - c1.mean():.4f}")
d_p = paired_boot_diff(c1, c2, B, rng)
d_u = unpaired_boot_diff(c1, c2, B, rng)
print(f"paired   (resample examples): [{np.quantile(d_p, 0.025):.4f}, {np.quantile(d_p, 0.975):.4f}]   "
      f"SE {d_p.std(ddof=1):.4f}")
print(f"unpaired (wrong):             [{np.quantile(d_u, 0.025):.4f}, {np.quantile(d_u, 0.975):.4f}]   "
      f"SE {d_u.std(ddof=1):.4f}")
print(f"corr of the two correctness vectors = {np.corrcoef(c1, c2)[0, 1]:.4f}")

section("exercise 2 (c): coverage of the paired interval")
hits = 0
for _ in range(N_COVERAGE):
    a1, a2 = make_pair(rng)
    d = paired_boot_diff(a1, a2, B, rng)
    hits += np.quantile(d, 0.025) <= ACC_2 - ACC_1 <= np.quantile(d, 0.975)
print(f"{N_COVERAGE} repetitions: the paired 95% interval contained the true difference 0.02 in "
      f"{hits / N_COVERAGE:.3f} of them   (Monte Carlo SE {np.sqrt(0.95 * 0.05 / N_COVERAGE):.3f})")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
