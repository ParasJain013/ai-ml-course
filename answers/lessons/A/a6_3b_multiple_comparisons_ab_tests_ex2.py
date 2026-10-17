# %% [markdown]
# # A6.3b Multiple comparisons and A/B tests — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a6_3b_multiple_comparisons_ab_tests_ex2.py   (CPU, ~3 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from scipy import stats
from statsmodels.stats.multitest import multipletests

FloatArray = npt.NDArray[np.float64]
SEED = 0
T0 = time.perf_counter()
ALPHA = 0.05
DAYS = 28
USERS_PER_DAY = 1_000  # per arm
BASE, LIFTED = 0.10, 0.11
N_RUNS = 2_000


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
p = np.array([0.004, 0.011, 0.019, 0.04, 0.07, 0.2, 0.5, 0.8])
print(f"(a) FWER for 8 tests = {1 - 0.95**8:.4f}")
for method in ("bonferroni", "holm", "fdr_bh"):
    print(f"(a) {method:<10} rejects {multipletests(p, alpha=ALPHA, method=method)[0].astype(int).tolist()}")
za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
n = (za + zb) ** 2 * (0.05 * 0.95 + 0.055 * 0.945) / 0.005**2
print(f"(b) n per arm = {n:,.0f}   days at 2,000 per arm per day = {n / 2_000:.1f}")
chi = 2 * 200**2 / 10_000
print(f"(c) SRM chi-square = {chi:.2f}   p = {stats.chi2.sf(chi, 1):.4f}")


# %%
def daily_p_values(rate_b: float, gen: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Cumulative two-proportion z-test each day. -> (p (N_RUNS, DAYS), lift estimate (N_RUNS, DAYS))."""
    ka = gen.binomial(USERS_PER_DAY, BASE, size=(N_RUNS, DAYS)).cumsum(axis=1)
    kb = gen.binomial(USERS_PER_DAY, rate_b, size=(N_RUNS, DAYS)).cumsum(axis=1)
    n_so_far = USERS_PER_DAY * np.arange(1, DAYS + 1)  # (DAYS,)
    pa, pb = ka / n_so_far, kb / n_so_far
    pool = (ka + kb) / (2 * n_so_far)
    z = (pb - pa) / np.sqrt(pool * (1 - pool) * 2 / n_so_far)
    return 2 * stats.norm.sf(np.abs(z)), pb - pa


def apply_rule(p: FloatArray, lift: FloatArray, threshold: float, peek: bool) -> tuple[FloatArray, FloatArray]:
    """Win decision and reported lift per run. p, lift (N_RUNS, DAYS) -> (win (N_RUNS,), lift (N_RUNS,))."""
    if not peek:  # one look on the last day
        win = (p[:, -1] < threshold) & (lift[:, -1] > 0)
        return win, lift[:, -1]
    hit = (p < threshold) & (lift > 0)  # a "B wins" signal on that day
    stop_day = np.where(hit.any(axis=1), np.argmax(hit, axis=1), DAYS - 1)
    rows = np.arange(len(p))
    return hit.any(axis=1), lift[rows, stop_day]


rng = np.random.default_rng(SEED)
section("exercise 2: calibrate a per-look threshold on separate A/A runs")
p_cal, _ = daily_p_values(BASE, rng)
thr = float(np.quantile(p_cal.min(axis=1), ALPHA))
print(f"threshold so that 5% of A/A runs ever cross it in {DAYS} looks: {thr:.4f}")

section("exercise 2: three stopping rules")
p_aa, lift_aa = daily_p_values(BASE, rng)
p_ab, lift_ab = daily_p_values(LIFTED, rng)
rules = {"fixed horizon": (ALPHA, False), "naive peeking": (ALPHA, True), "calibrated peeking": (thr, True)}
print(f"{'rule':<20}{'false + (A/A)':>14}{'power (11%)':>13}{'mean lift among wins':>22}")
for name, (t, peek) in rules.items():
    win0, _ = apply_rule(p_aa, lift_aa, t, peek)
    # the A/A false-positive rate counts wins in either direction, as a two-sided test would
    fp = np.mean(p_aa.min(axis=1) < t) if peek else np.mean(p_aa[:, -1] < t)
    win1, rep1 = apply_rule(p_ab, lift_ab, t, peek)
    print(f"{name:<20}{fp:>14.4f}{win1.mean():>13.4f}{rep1[win1].mean():>22.4f}")
print(f"true lift = {LIFTED - BASE:.4f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
