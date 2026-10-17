# %% [markdown]
# # A6.3a Hypothesis tests: p-values, permutation, t and chi-square — lab
# Lesson: lessons/A/a6_3a_hypothesis_tests.md
# Run: uv run python labs/A/a6_3a_hypothesis_tests.py   (CPU, defaults: ~5 s)
# GPU: no

# %%
from __future__ import annotations

import itertools
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
OUT = ROOT / "outputs" / "a6_3a"  # scratch figures, gitignored
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
N_PERM = 50_000 if GPU_SCALE else 10_000  # permutations per permutation test
N_PER_GROUP = 80  # checkout times per group
N_SIM = 10_000 if GPU_SCALE else 2_000  # simulated experiments for the p-value histograms
ALPHA = 0.05

# %%
# Data: checkout times (seconds) for the old and the new flow, and a conversion table.
section("data")
old_flow = rng.lognormal(np.log(60.0), 0.5, size=N_PER_GROUP)
new_flow = rng.lognormal(np.log(52.0), 0.5, size=N_PER_GROUP)
print(f"old flow: n = {len(old_flow)}   mean = {old_flow.mean():.4f} s   sd = {old_flow.std(ddof=1):.4f}")
print(f"new flow: n = {len(new_flow)}   mean = {new_flow.mean():.4f} s   sd = {new_flow.std(ddof=1):.4f}")
plans = np.array([[120, 60, 20], [90, 75, 35]])  # rows: desktop, mobile; columns: free, pro, team
print("plan chosen by device (rows desktop, mobile; columns free, pro, team):")
print(plans)

# %%
# Worked example by hand: an exact permutation test, Welch's t, a 2x2 chi-square, Cohen's d.
section("worked example by hand")
a_small = np.array([3.0, 4.0, 5.0])
b_small = np.array([6.0, 7.0, 8.0])
pooled = np.concatenate([a_small, b_small])
obs = b_small.mean() - a_small.mean()
diffs = []
for chosen in itertools.combinations(range(6), 3):  # every way to call 3 of the 6 values "B"
    mask = np.zeros(6, dtype=bool)
    mask[list(chosen)] = True
    diffs.append(pooled[mask].mean() - pooled[~mask].mean())
diffs_arr = np.array(diffs)
print(f"permutation: observed mean(B) - mean(A) = {obs:.1f}   splits = {len(diffs_arr)}")
print(f"all 20 differences, sorted: {sorted(np.round(diffs_arr, 4).tolist())}")
print(f"one-sided p = #(diff >= 3) / 20 = {np.mean(diffs_arr >= obs - 1e-12):.2f}   "
      f"two-sided p = #(|diff| >= 3) / 20 = {np.mean(np.abs(diffs_arr) >= obs - 1e-12):.2f}")

a_w = np.array([3.0, 5.0, 4.0, 8.0])
b_w = np.array([7.0, 9.0, 8.0, 10.0, 11.0])
va, vb = a_w.var(ddof=1) / len(a_w), b_w.var(ddof=1) / len(b_w)
se_w = np.sqrt(va + vb)
t_w = (b_w.mean() - a_w.mean()) / se_w
df_w = (va + vb) ** 2 / (va**2 / (len(a_w) - 1) + vb**2 / (len(b_w) - 1))
print(f"Welch: means {a_w.mean():.0f} and {b_w.mean():.0f}   variances {a_w.var(ddof=1):.4f} and {b_w.var(ddof=1):.4f}")
print(f"  s_A^2/n_A = {va:.4f}   s_B^2/n_B = {vb:.4f}   SE = {se_w:.4f}   t = {t_w:.4f}   df = {df_w:.4f}   "
      f"two-sided p = {2 * stats.t.sf(abs(t_w), df_w):.4f}")
sp = np.sqrt(((len(a_w) - 1) * a_w.var(ddof=1) + (len(b_w) - 1) * b_w.var(ddof=1)) / (len(a_w) + len(b_w) - 2))
print(f"  Cohen's d = difference / pooled sd = 4 / {sp:.4f} = {4 / sp:.4f}")

conv = np.array([[30, 170], [50, 150]])  # rows: variant A, B; columns: converted, not converted
row, col, total = conv.sum(axis=1), conv.sum(axis=0), conv.sum()
expected = np.outer(row, col) / total
chi2_hand = np.sum((conv - expected) ** 2 / expected)
print(f"2x2 table {conv.tolist()}   expected {expected.tolist()}")
print(f"  chi-square = {chi2_hand:.4f}   df = 1   p = {stats.chi2.sf(chi2_hand, 1):.4f}   "
      f"Cramer's V = sqrt(chi2 / n) = {np.sqrt(chi2_hand / total):.4f}")
p_pool = col[0] / total
z_prop = (50 / 200 - 30 / 200) / np.sqrt(p_pool * (1 - p_pool) * (1 / 200 + 1 / 200))
print(f"  two-proportion z = 0.10 / sqrt(0.2 * 0.8 * (1/200 + 1/200)) = {z_prop:.4f}   z^2 = {z_prop**2:.4f}")

# %%
# From scratch: a permutation test, Welch's t-test and a chi-square test of independence.
section("from scratch: three tests")


def perm_test_diff_means(a: FloatArray, b: FloatArray, n_perm: int, gen: np.random.Generator) -> tuple[float, FloatArray]:
    """Two-sided permutation test of mean(b) - mean(a). a (n_a,), b (n_b,) -> (p, null (n_perm,))."""
    pooled = np.concatenate([a, b])  # (n_a + n_b,)
    keys = gen.random((n_perm, len(pooled)))  # (n_perm, n) random sort keys
    perm = pooled[np.argsort(keys, axis=1)]  # (n_perm, n) each row a shuffled copy
    null = perm[:, len(a):].mean(axis=1) - perm[:, : len(a)].mean(axis=1)  # (n_perm,)
    observed = b.mean() - a.mean()
    # +1 in numerator and denominator: the observed split is one of the permutations, so p is never 0
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    return float(p), null


def welch_t(a: FloatArray, b: FloatArray) -> tuple[float, float, float]:
    """Welch's two-sample t-test of mean(b) - mean(a). a (n_a,), b (n_b,) -> (t, df, two-sided p)."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (b.mean() - a.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    return float(t), float(df), float(2 * stats.t.sf(abs(t), df))


def chi2_independence(table: npt.NDArray[np.int64]) -> tuple[float, int, float, FloatArray]:
    """Chi-square test of independence. table (r, c) -> (statistic, df, p, expected (r, c))."""
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()  # (r, c)
    stat = float(np.sum((table - expected) ** 2 / expected))
    df = (table.shape[0] - 1) * (table.shape[1] - 1)
    return stat, df, float(stats.chi2.sf(stat, df)), expected


p_perm, null_dist = perm_test_diff_means(old_flow, new_flow, N_PERM, rng)
t_val, df_val, p_welch = welch_t(old_flow, new_flow)
d_obs = new_flow.mean() - old_flow.mean()
print(f"checkout: mean(new) - mean(old) = {d_obs:.4f} s")
print(f"permutation test: p = {p_perm:.4f}   ({N_PERM} shuffles; null sd = {null_dist.std():.4f})")
print(f"Welch t-test:     t = {t_val:.4f}   df = {df_val:.2f}   p = {p_welch:.4f}")
sp = np.sqrt((old_flow.var(ddof=1) + new_flow.var(ddof=1)) / 2)  # equal group sizes
print(f"effect size: {d_obs:.2f} s ({d_obs / old_flow.mean():.1%} of the old mean)   Cohen's d = {d_obs / sp:.4f}")
stat, dof, p_chi, exp_plans = chi2_independence(plans)
print(f"plans x device: chi-square = {stat:.4f}   df = {dof}   p = {p_chi:.6f}")
print("expected counts under independence:", show(exp_plans))
print("row shares (free, pro, team):", show(plans / plans.sum(axis=1, keepdims=True)))
print(f"Cramer's V = sqrt(chi2 / (n * (min(r, c) - 1))) = {np.sqrt(stat / (plans.sum() * (min(plans.shape) - 1))):.4f}")

# %%
# From scratch: what p-values look like when the null is true and when it is false.
section("from scratch: p-values under the null and under an effect")
null_a = rng.lognormal(np.log(60.0), 0.5, size=(N_SIM, N_PER_GROUP))
null_b = rng.lognormal(np.log(60.0), 0.5, size=(N_SIM, N_PER_GROUP))  # same distribution: H0 true
alt_b = rng.lognormal(np.log(52.0), 0.5, size=(N_SIM, N_PER_GROUP))  # a real 13% speed-up
p_null = stats.ttest_ind(null_b, null_a, axis=1, equal_var=False).pvalue  # (N_SIM,) vectorized Welch
p_alt = stats.ttest_ind(alt_b, null_a, axis=1, equal_var=False).pvalue
print(f"H0 true, {N_SIM} experiments: share with p < 0.05 = {np.mean(p_null < ALPHA):.4f}   "
      f"(the type I error rate; should be ~0.05)")
print(f"H0 true: share of p-values in each tenth of [0, 1]: {show(np.histogram(p_null, bins=10, range=(0, 1))[0] / N_SIM)}")
print(f"real effect: share with p < 0.05 = {np.mean(p_alt < ALPHA):.4f}   (the power at n = {N_PER_GROUP} per group)")
print(f"real effect: share with p < 0.001 = {np.mean(p_alt < 0.001):.4f}   median p = {np.median(p_alt):.4f}")
ci_cover = []
for i in range(200):  # the 95% CI for the difference excludes 0 exactly when p < 0.05
    a_i, b_i = null_a[i], alt_b[i]
    _, df_i, p_i = welch_t(a_i, b_i)
    half = stats.t.ppf(0.975, df_i) * np.sqrt(a_i.var(ddof=1) / len(a_i) + b_i.var(ddof=1) / len(b_i))
    d_i = b_i.mean() - a_i.mean()
    ci_cover.append((p_i < ALPHA) == (not (d_i - half <= 0 <= d_i + half)))
print(f"in 200 experiments, 'p < 0.05' and '95% CI excludes 0' agree every time: {all(ci_cover)}")

# %%
# With SciPy: the same tests, plus Kolmogorov-Smirnov and Mann-Whitney for comparison.
section("with scipy")
res_t = stats.ttest_ind(new_flow, old_flow, equal_var=False)
np.testing.assert_allclose((res_t.statistic, res_t.pvalue), (t_val, p_welch), rtol=1e-10)
print(f"ttest_ind(equal_var=False): t = {res_t.statistic:.4f}   p = {res_t.pvalue:.4f}   matches welch_t: OK")
res_student = stats.ttest_ind(new_flow, old_flow)  # default equal_var=True: Student's t
print(f"ttest_ind default (Student, equal variances): p = {res_student.pvalue:.4f}")
res_p = stats.permutation_test(
    (old_flow, new_flow), lambda a, b, axis: b.mean(axis=axis) - a.mean(axis=axis),
    n_resamples=N_PERM, vectorized=True, rng=np.random.default_rng(SEED),
)
# Different random shuffles: the two p-values agree up to Monte Carlo error, sqrt(p(1-p)/N_PERM) ~ 0.003
np.testing.assert_allclose(res_p.pvalue, p_perm, atol=0.01)
print(f"permutation_test: p = {res_p.pvalue:.4f}   mine {p_perm:.4f}   agree within Monte Carlo error: OK")
res_c = stats.chi2_contingency(plans)
np.testing.assert_allclose((res_c.statistic, res_c.pvalue), (stat, p_chi), rtol=1e-10)
np.testing.assert_allclose(res_c.expected_freq, exp_plans, rtol=1e-12)
print(f"chi2_contingency(plans): chi2 = {res_c.statistic:.4f}   p = {res_c.pvalue:.6f}   dof = {res_c.dof}   matches: OK")
res_c2 = stats.chi2_contingency(conv, correction=False)
res_y = stats.chi2_contingency(conv)  # 2x2 default applies Yates' continuity correction
print(f"2x2 conversion table: correction=False chi2 = {res_c2.statistic:.4f} (p = {res_c2.pvalue:.4f})   "
      f"default (Yates) chi2 = {res_y.statistic:.4f} (p = {res_y.pvalue:.4f})")
res_ks = stats.ks_2samp(old_flow, new_flow)
res_mw = stats.mannwhitneyu(new_flow, old_flow)
print(f"ks_2samp: D = {res_ks.statistic:.4f}   p = {res_ks.pvalue:.4f}   "
      f"mannwhitneyu: U = {res_mw.statistic:.1f}   p = {res_mw.pvalue:.4f}")

# %%
# Key plot: the permutation null distribution, and p-value histograms under H0 and under an effect.
section("key plot")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=120)
ax = axes[0]
ax.hist(null_dist, bins=60, color="tab:blue", alpha=0.7, label="differences after shuffling labels")
ax.axvline(d_obs, color="tab:red", lw=2, ls="--", label=f"observed {d_obs:.1f} s")
ax.axvline(-d_obs, color="tab:red", lw=1, ls=":", label="as extreme, other side")
ax.set(xlabel="mean(new) − mean(old), seconds", ylabel="count of shuffles",
       title=f"Permutation null: observed difference is rare (p = {p_perm:.4f})")
ax.legend(loc="upper left", fontsize=8)
ax = axes[1]
bins = np.linspace(0, 1, 21)
ax.hist(p_null, bins=bins, histtype="step", lw=2, ls="-", label="no effect (H0 true): flat")
ax.hist(p_alt, bins=bins, histtype="step", lw=2, ls="--", label="13% speed-up: piles up near 0")
ax.axvline(ALPHA, color="k", lw=1, label="α = 0.05")
ax.set(xlabel="p-value", ylabel=f"count out of {N_SIM} experiments",
       title="Under H0 every p-value is equally likely")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
path_png = IMG / "a6_3a_null_and_pvalues.png"
fig.savefig(path_png, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path_png.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Change N_PER_GROUP to 20 and to 300. How does the share of p < 0.05 under the effect (the power) move?
# 2. Give the new flow a 3x larger spread (sigma 1.5) but the same median. What do Welch, Student and
#    Mann-Whitney say, and which question is each answering?
# 3. Replace the mean by the median in perm_test_diff_means. Is it more or less powerful here?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
