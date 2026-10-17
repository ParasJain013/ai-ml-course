# %% [markdown]
# # A6.3b Multiple comparisons and A/B tests — lab
# Lesson: lessons/A/a6_3b_multiple_comparisons_ab_tests.md
# Run: uv run python labs/A/a6_3b_multiple_comparisons_ab_tests.py   (CPU, defaults: ~10 s)
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
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

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
OUT = ROOT / "outputs" / "a6_3b"  # scratch figures, gitignored
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
ALPHA = 0.05
M_TESTS = 20  # tests per family in the multiple-comparisons simulation
N_PER_TEST = 50  # observations per group in each of those tests
N_FAMILIES = 10_000 if GPU_SCALE else 2_000  # simulated families of tests
N_REAL = 5  # tests with a real effect in the "some true effects" scenario
EFFECT_D = 0.8  # their standardized effect (Cohen's d)
DAYS = 30  # length of the A/B test
USERS_PER_DAY = 1_000  # per arm
BASE_RATE = 0.10  # conversion rate of both arms in the A/A test
N_AB = 10_000 if GPU_SCALE else 2_000  # simulated A/A tests for peeking

# %%
# Data: everything is simulated. One family of 20 two-group comparisons where nothing is going on.
section("data")
family_a = rng.normal(0, 1, size=(M_TESTS, N_PER_TEST))
family_b = rng.normal(0, 1, size=(M_TESTS, N_PER_TEST))  # same distribution: every H0 is true
print(f"one family: {M_TESTS} comparisons of two groups of {N_PER_TEST}, no real differences anywhere")
print(f"A/A test: {DAYS} days x {USERS_PER_DAY:,} users per arm per day, conversion {BASE_RATE:.0%} in both arms")

# %%
# Worked example by hand: family-wise error, three corrections, a sample size, a ratio-mismatch check.
section("worked example by hand")
print(f"P(at least one false positive in 20 independent tests) = 1 - 0.95^20 = {1 - 0.95**20:.4f}   "
      f"expected false positives = 20 * 0.05 = {20 * 0.05:.1f}")
print(f"Bonferroni threshold = 0.05 / 20 = {0.05 / 20:.4f}")
p_hand = np.array([0.001, 0.008, 0.012, 0.030, 0.200])
m_h = len(p_hand)
holm_thr = ALPHA / (m_h - np.arange(m_h))  # alpha/5, alpha/4, ..., alpha/1
bh_thr = ALPHA * np.arange(1, m_h + 1) / m_h  # alpha*1/5, ..., alpha*5/5
print(f"sorted p = {p_hand.tolist()}")
print(f"Bonferroni: p <= {ALPHA / m_h:.4f}? {(p_hand <= ALPHA / m_h).tolist()}")
print(f"Holm thresholds {show(holm_thr)}   p <= threshold? {(p_hand <= holm_thr).tolist()}  (stop at the first False)")
print(f"BH thresholds   {show(bh_thr)}   p <= threshold? {(p_hand <= bh_thr).tolist()}  (largest True is k = 4)")
z_a, z_b = stats.norm.ppf(1 - ALPHA / 2), stats.norm.ppf(0.80)
p1, p2 = 0.10, 0.11
n_arm = (z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2
print(f"sample size: z_0.975 = {z_a:.4f}   z_0.80 = {z_b:.4f}   variance sum = {p1 * (1 - p1) + p2 * (1 - p2):.4f}   "
      f"n per arm = {n_arm:,.0f}")
print(f"  halve the detectable lift to 0.5 points: n per arm = "
      f"{(z_a + z_b) ** 2 * (p1 * (1 - p1) + 0.105 * 0.895) / 0.005**2:,.0f}")
srm = np.array([50_600, 49_400])
chi_srm = np.sum((srm - 50_000) ** 2 / 50_000)
print(f"sample ratio check: observed {srm.tolist()} vs expected [50000, 50000]   chi-square = {chi_srm:.2f}   "
      f"p = {stats.chi2.sf(chi_srm, 1):.5f}")

# %%
# From scratch: three corrections, and what they do to 20 tests on pure noise.
section("from scratch: multiple comparisons")


def bonferroni(p: FloatArray, alpha: float = ALPHA) -> npt.NDArray[np.bool_]:
    """Reject where p <= alpha / m. p (m,) -> (m,) bool."""
    return p <= alpha / len(p)


def holm(p: FloatArray, alpha: float = ALPHA) -> npt.NDArray[np.bool_]:
    """Holm step-down: compare sorted p to alpha/(m - k), stop at the first failure. p (m,) -> (m,) bool."""
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= alpha / (m - np.arange(m))  # (m,) in sorted order
    n_reject = m if passed.all() else int(np.argmin(passed))  # index of the first failure
    reject = np.zeros(m, dtype=bool)
    reject[order[:n_reject]] = True
    return reject


def benjamini_hochberg(p: FloatArray, alpha: float = ALPHA) -> npt.NDArray[np.bool_]:
    """BH step-up: reject the k smallest, k = largest with p_(k) <= k alpha / m. p (m,) -> (m,) bool."""
    m = len(p)
    order = np.argsort(p)
    below = np.nonzero(p[order] <= alpha * np.arange(1, m + 1) / m)[0]
    reject = np.zeros(m, dtype=bool)
    if len(below):
        reject[order[: below[-1] + 1]] = True
    return reject


p_one = stats.ttest_ind(family_b, family_a, axis=1, equal_var=False).pvalue  # (M_TESTS,)
print(f"one family of 20 null tests, smallest p-values: {show(np.sort(p_one)[:4])}")
print(f"significant at 0.05 uncorrected: {int(np.sum(p_one < ALPHA))}   Bonferroni: {int(bonferroni(p_one).sum())}   "
      f"Holm: {int(holm(p_one).sum())}   BH: {int(benjamini_hochberg(p_one).sum())}")

methods = {"none": lambda p: p < ALPHA, "Bonferroni": bonferroni, "Holm": holm, "BH": benjamini_hochberg}
sim_fwer: dict[str, float] = {}  # all-null scenario, for the key plot
for scenario, n_real in (("all 20 null", 0), (f"{N_REAL} real effects (d = {EFFECT_D})", N_REAL)):
    a = rng.normal(0, 1, size=(N_FAMILIES, M_TESTS, N_PER_TEST))
    b = rng.normal(0, 1, size=(N_FAMILIES, M_TESTS, N_PER_TEST))
    b[:, :n_real, :] += EFFECT_D  # the first n_real tests of every family have a real effect
    p_all = stats.ttest_ind(b, a, axis=2, equal_var=False).pvalue  # (N_FAMILIES, M_TESTS)
    is_real = np.arange(M_TESTS) < n_real
    print(f"{scenario}, {N_FAMILIES} families:")
    print(f"  {'method':<11}{'P(any false +)':>15}{'false + / family':>18}{'true + / family':>17}{'FDR':>7}")
    for name, fn in methods.items():
        rej = np.array([fn(row) for row in p_all])  # (N_FAMILIES, M_TESTS)
        false_pos = rej[:, ~is_real].sum(axis=1)
        true_pos = rej[:, is_real].sum(axis=1)
        fdr = np.mean(false_pos / np.maximum(false_pos + true_pos, 1))  # 0 when nothing is rejected
        if n_real == 0:
            sim_fwer[name] = float(np.mean(false_pos > 0))
        print(f"  {name:<11}{np.mean(false_pos > 0):>15.4f}{false_pos.mean():>18.4f}{true_pos.mean():>17.4f}{fdr:>7.4f}")
fwer_curve = {m: 1 - (1 - ALPHA) ** m for m in (1, 5, 10, 20, 50, 100)}
print("formula 1 - 0.95^m:", {m: round(v, 4) for m, v in fwer_curve.items()})

# %%
# From scratch: peeking at an A/A test every day and stopping at the first p < 0.05.
section("from scratch: peeking")


def two_prop_z_p(k_a: FloatArray, n_a: FloatArray, k_b: FloatArray, n_b: FloatArray) -> FloatArray:
    """Two-sided p-value of the pooled two-proportion z-test; works elementwise on arrays."""
    pool = (k_a + k_b) / (n_a + n_b)
    se = np.sqrt(pool * (1 - pool) * (1 / n_a + 1 / n_b))
    z = (k_b / n_b - k_a / n_a) / se
    return 2 * stats.norm.sf(np.abs(z))


daily_a = rng.binomial(USERS_PER_DAY, BASE_RATE, size=(N_AB, DAYS))  # conversions per day, arm A
daily_b = rng.binomial(USERS_PER_DAY, BASE_RATE, size=(N_AB, DAYS))  # same rate: an A/A test
cum_a, cum_b = daily_a.cumsum(axis=1), daily_b.cumsum(axis=1)  # (N_AB, DAYS) running totals
users = USERS_PER_DAY * np.arange(1, DAYS + 1)  # (DAYS,) users per arm so far
p_daily = two_prop_z_p(cum_a, users, cum_b, users)  # (N_AB, DAYS) the p-value you would see each day
sig_by_day = np.maximum.accumulate(p_daily < ALPHA, axis=1)  # "has it ever been significant so far?"
fixed = np.mean(p_daily[:, -1] < ALPHA)
print(f"{N_AB} A/A tests (no real difference), {DAYS} days:")
print(f"  look once, at day {DAYS}: false positive rate = {fixed:.4f}")
for d in (1, 5, 10, 20, 30):
    print(f"  peek daily, stop at first p < 0.05, by day {d:>2}: false positive rate = {sig_by_day[:, d - 1].mean():.4f}")
first_day = np.argmax(p_daily < ALPHA, axis=1) + 1
ever = sig_by_day[:, -1]
print(f"  of the tests that ever crossed 0.05, still below 0.05 at day {DAYS}: {np.mean(p_daily[ever, -1] < ALPHA):.4f}")
print(f"  median day of the first 'significant' peek: {np.median(first_day[ever]):.0f}")
calib_a = rng.binomial(USERS_PER_DAY, BASE_RATE, size=(N_AB, DAYS)).cumsum(axis=1)  # fresh A/A tests
calib_b = rng.binomial(USERS_PER_DAY, BASE_RATE, size=(N_AB, DAYS)).cumsum(axis=1)
min_p = two_prop_z_p(calib_a, users, calib_b, users).min(axis=1)  # (N_AB,) smallest p over 30 peeks
thr = float(np.quantile(min_p, ALPHA))  # a constant threshold that 5% of A/A tests ever cross
print(f"  calibrated per-peek threshold (from separate A/A runs) = {thr:.4f}; with it the false positive "
      f"rate here = {np.mean(p_daily.min(axis=1) < thr):.4f}")

# %%
# From scratch: power and sample size for a conversion test, checked by simulation.
section("from scratch: power and sample size")


def n_per_arm(p1: float, p2: float, alpha: float = ALPHA, power: float = 0.80) -> float:
    """Users per arm for a two-sided two-proportion z-test to detect p1 -> p2."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return float((za + zb) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2)


n_req = int(np.ceil(n_per_arm(0.10, 0.11)))
for n_test, label in ((n_req // 4, "a quarter of it"), (n_req, "the computed n")):
    ka = rng.binomial(n_test, 0.10, size=N_AB)
    kb = rng.binomial(n_test, 0.11, size=N_AB)
    pw = np.mean(two_prop_z_p(ka, np.full(N_AB, n_test), kb, np.full(N_AB, n_test)) < ALPHA)
    print(f"10% -> 11% with {n_test:>6,} users per arm ({label}): simulated power = {pw:.4f}")
print(f"minimum detectable effect with 2,000 users per arm (power 0.8): "
      f"{(stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * np.sqrt(2 * 0.1 * 0.9 / 2_000):.4f} "
      "(absolute, around a 10% base rate)")

# %%
# With statsmodels: the same corrections, the same z-test, and its power calculator.
section("with statsmodels")
for name, method in (("Bonferroni", "bonferroni"), ("Holm", "holm"), ("BH", "fdr_bh")):
    mine = {"Bonferroni": bonferroni, "Holm": holm, "BH": benjamini_hochberg}[name](p_hand)
    theirs = multipletests(p_hand, alpha=ALPHA, method=method)[0]
    assert np.array_equal(mine, theirs), name
    print(f"{name:<11} rejects {mine.tolist()}   multipletests agrees: OK")
for _ in range(200):  # random p-value vectors, as a stronger check
    p_rand = rng.random(12) ** 3
    for fn, method in ((holm, "holm"), (benjamini_hochberg, "fdr_bh")):
        assert np.array_equal(fn(p_rand), multipletests(p_rand, alpha=ALPHA, method=method)[0])
print("Holm and BH agree with multipletests on 200 random p-value vectors: OK")
adj = multipletests(p_hand, method="holm")[1]
print(f"Holm-adjusted p-values (compare each with 0.05): {show(adj)}")
z_sm, p_sm = proportions_ztest([int(cum_b[0, -1]), int(cum_a[0, -1])], [int(users[-1])] * 2)
np.testing.assert_allclose(p_sm, p_daily[0, -1], rtol=1e-10)
print(f"proportions_ztest matches the from-scratch two-proportion z-test (p = {p_sm:.4f}): OK")
h = proportion_effectsize(0.11, 0.10)
n_sm = NormalIndPower().solve_power(effect_size=h, alpha=ALPHA, power=0.80, alternative="two-sided")
print(f"statsmodels NormalIndPower with Cohen's h = {h:.4f}: n per arm = {n_sm:,.0f}   formula above: {n_req:,}")

# %%
# Key plot: false positives grow with the number of tests and with the number of peeks.
section("key plot")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=120)
ax = axes[0]
ms = np.arange(1, 101)
ax.plot(ms, 1 - (1 - ALPHA) ** ms, "-", label="no correction: 1 − 0.95^m")
ax.plot(ms, 1 - (1 - ALPHA / ms) ** ms, "--", lw=2.5, label="Bonferroni: α/m per test (just under α)")
ax.plot([M_TESTS, M_TESTS], [sim_fwer["none"], sim_fwer["Bonferroni"]], "kD", ms=6,
        label=f"simulated, m = {M_TESTS}: {sim_fwer['none']:.2f} and {sim_fwer['Bonferroni']:.2f}")
ax.set(xlabel="number of independent tests m (all nulls true)", ylabel="P(at least one false positive)",
       title="Uncorrected, 20 tests give a 64% chance of a false alarm", ylim=(0, 1.02))
ax.legend(loc="center right", fontsize=8)
ax = axes[1]
days = np.arange(1, DAYS + 1)
ax.plot(days, sig_by_day.mean(axis=0), "o-", ms=3, label="peek daily, stop at first p < 0.05")
ax.plot(days, (p_daily < ALPHA).mean(axis=0), "s--", ms=3, label="look once, on that day")
ax.axhline(ALPHA, color="k", lw=1, ls=":", label="α = 0.05")
ax.set(xlabel="day of the A/A test", ylabel="share of A/A tests declared 'significant'",
       title=f"Peeking turns 5% into {sig_by_day[:, -1].mean():.0%} false positives", ylim=(0, 0.35))
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
path_png = IMG / "a6_3b_multiple_tests_peeking.png"
fig.savefig(path_png, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path_png.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Make the 20 tests correlated (add the same noise vector to every test's group B). Does Bonferroni
#    become too strict?
# 2. Peek weekly instead of daily (4 looks). How far does the false positive rate fall?
# 3. Give arm B a real lift (11%) and compare the day peeking first declares a win with the final estimate.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
