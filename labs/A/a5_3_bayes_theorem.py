# %% [markdown]
# # A5.3 Conditional probability and Bayes' theorem — lab
# Lesson: lessons/A/a5_3_bayes_theorem.md
# Run: uv run python labs/A/a5_3_bayes_theorem.py   (CPU, defaults: ~3 s)
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
from sklearn.metrics import confusion_matrix, precision_score, recall_score

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
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
OUT = ROOT / "outputs" / "a5_3"  # scratch figures, gitignored
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
# Nothing here needs a GPU; GPU_SCALE only simulates a larger population.
GPU_SCALE = False
N_PEOPLE = 10_000_000 if GPU_SCALE else 1_000_000  # simulated population
BASE_RATE = 0.01  # P(disease): 1 in 100 people have it
SENSITIVITY = 0.90  # P(positive | disease)
SPECIFICITY = 0.91  # P(negative | healthy), so P(positive | healthy) = 0.09
SIM_BASE_RATES = [0.001, 0.01, 0.1, 0.5]  # base rates simulated for the key plot

# %%
# Data: a simulated population. Each person is a Bernoulli draw for disease, then for the test result.
section("data")


def simulate_population(
    n: int, base_rate: float, sens: float, spec: float, gen: np.random.Generator
) -> tuple[BoolArray, BoolArray]:
    """Draw disease status and one test result per person. -> (disease (n,), positive (n,))."""
    disease = gen.random(n) < base_rate  # (n,) Bernoulli(base_rate)
    u = gen.random(n)  # one uniform per person for the test
    positive = np.where(disease, u < sens, u < 1.0 - spec)  # (n,) the test's behaviour depends on status
    return disease, positive


disease, positive = simulate_population(N_PEOPLE, BASE_RATE, SENSITIVITY, SPECIFICITY, rng)
print(f"population: {N_PEOPLE:,} people   with disease: {disease.sum():,}   tested positive: {positive.sum():,}")

# %%
# Worked example by hand: 10,000 people as natural frequencies, then the formula, then odds.
section("worked example: natural frequencies (10,000 people)")
n_hand = 10_000
sick = n_hand * BASE_RATE
healthy = n_hand - sick
true_pos = sick * SENSITIVITY
false_pos = healthy * (1 - SPECIFICITY)
print(f"sick = {sick:.0f}, of whom positive = {true_pos:.0f}   healthy = {healthy:.0f}, of whom positive = {false_pos:.0f}")
print(f"P(disease | positive) = {true_pos:.0f} / ({true_pos:.0f} + {false_pos:.0f}) = {true_pos / (true_pos + false_pos):.4f}")


def posterior(prior: float | FloatArray, sens: float, spec: float) -> float | FloatArray:
    """Bayes: P(D | +) = P(+ | D) P(D) / P(+), with P(+) from the law of total probability."""
    p_pos = sens * prior + (1.0 - spec) * (1.0 - prior)  # P(+) = P(+|D)P(D) + P(+|H)P(H)
    return sens * prior / p_pos


section("worked example: Bayes' formula and the odds form")
p_pos = SENSITIVITY * BASE_RATE + (1 - SPECIFICITY) * (1 - BASE_RATE)
post1 = posterior(BASE_RATE, SENSITIVITY, SPECIFICITY)
print(f"P(+) = 0.9 * 0.01 + 0.09 * 0.99 = {p_pos:.4f}   P(D | +) = 0.009 / {p_pos:.4f} = {post1:.4f}")
lr_pos = SENSITIVITY / (1 - SPECIFICITY)
prior_odds = BASE_RATE / (1 - BASE_RATE)
post_odds = prior_odds * lr_pos
print(f"likelihood ratio LR+ = 0.9 / 0.09 = {lr_pos:.1f}   prior odds = 1/99 = {prior_odds:.4f}   "
      f"posterior odds = {post_odds:.4f}   -> probability {post_odds / (1 + post_odds):.4f}")
post2 = posterior(post1, SENSITIVITY, SPECIFICITY)
print(f"second independent positive: prior {post1:.4f} -> posterior {post2:.4f}   "
      f"(odds {post_odds:.4f} x 10 = {post_odds * lr_pos:.4f})")
p_d_neg = (1 - SENSITIVITY) * BASE_RATE / ((1 - SENSITIVITY) * BASE_RATE + SPECIFICITY * (1 - BASE_RATE))
print(f"P(D | negative) = 0.1 * 0.01 / (0.1 * 0.01 + 0.91 * 0.99) = {p_d_neg:.5f}")

section("worked example: which machine made the defective part?")
share = np.array([0.5, 0.3, 0.2])  # P(machine): machines A, B, C
defect = np.array([0.01, 0.02, 0.05])  # P(defect | machine)
joint = share * defect  # (3,) P(machine and defect)
print(f"P(machine and defect) = {show(joint)}   P(defect) = {joint.sum():.4f}")
print(f"P(machine | defect) = {show(joint / joint.sum())}   (prior was {show(share)})")

# %%
# From scratch: count simulated people and compare with the formula.
section("from scratch: counting the simulated population")
n_pos = positive.sum()
p_sim = (disease & positive).sum() / n_pos
print(f"P(disease | positive): counted {p_sim:.4f}   formula {post1:.4f}   "
      f"(standard error of the count ~ {np.sqrt(post1 * (1 - post1) / n_pos):.4f})")
print(f"P(positive | disease): counted {positive[disease].mean():.4f}   (sensitivity {SENSITIVITY})")
print(f"P(positive | healthy): counted {positive[~disease].mean():.4f}   (1 - specificity {1 - SPECIFICITY:.2f})")
p_d_neg_sim = (disease & ~positive).sum() / (~positive).sum()
print(f"P(disease | negative): counted {p_d_neg_sim:.5f}   formula {p_d_neg:.5f}")
# A second, independent test on everyone who tested positive the first time.
pos_idx = np.flatnonzero(positive)
u2 = rng.random(len(pos_idx))
second = np.where(disease[pos_idx], u2 < SENSITIVITY, u2 < 1.0 - SPECIFICITY)  # (n_pos,)
p_sim2 = disease[pos_idx][second].mean()
print(f"P(disease | two positives): counted {p_sim2:.4f}   formula {post2:.4f}")
assert abs(p_sim - post1) < 4 * np.sqrt(post1 * (1 - post1) / n_pos)
print("counted and formula agree within sampling noise: OK")

# %%
# With scikit-learn: precision is P(disease | positive), recall is the sensitivity.
section("with scikit-learn: the confusion matrix says the same thing")
y_true, y_pred = disease.astype(int), positive.astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f"confusion matrix: TN = {tn:,}  FP = {fp:,}  FN = {fn:,}  TP = {tp:,}")
prec, rec = precision_score(y_true, y_pred), recall_score(y_true, y_pred)
print(f"precision_score = TP / (TP + FP) = {prec:.4f}   (our counted P(D | +) = {p_sim:.4f})")
print(f"recall_score = TP / (TP + FN) = {rec:.4f}   specificity = TN / (TN + FP) = {tn / (tn + fp):.4f}")
np.testing.assert_allclose(prec, p_sim, rtol=1e-12)
print("precision_score equals our count: OK")

# %%
# Key plot: P(disease | positive) against the base rate, for three tests, with simulated points.
section("key plot")
tests = [(0.90, 0.91, "o"), (0.99, 0.99, "s"), (0.999, 0.999, "^")]
grid = np.logspace(-4, np.log10(0.5), 200)
fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
for sens, spec, marker in tests:
    ax.plot(grid, posterior(grid, sens, spec), "-", lw=1.5,
            label=f"sensitivity {sens}, specificity {spec}  (LR+ = {sens / (1 - spec):.0f})")
    sims = []
    for i, br in enumerate(SIM_BASE_RATES):
        d, pos = simulate_population(N_PEOPLE, br, sens, spec, np.random.default_rng(SEED + 10 + i))
        sims.append((d & pos).sum() / pos.sum())
    ax.plot(SIM_BASE_RATES, sims, marker, color=ax.lines[-1].get_color(), ms=6, mfc="none")
    print(f"sens {sens}, spec {spec}: simulated P(D|+) at base rates {SIM_BASE_RATES} = {show(np.array(sims))}")
ax.plot(grid, grid, "k:", lw=1, label="no test: P(D) itself")
ax.axvline(BASE_RATE, color="0.6", lw=0.8)
ax.annotate(f"base rate 1%: P(D|+) = {post1:.2f}", (BASE_RATE, post1), xytext=(0.0013, 0.55),
            arrowprops={"arrowstyle": "->", "color": "0.3"}, fontsize=8)
ax.set_xscale("log")
ax.set_ylim(0, 1.02)
ax.set_xlabel("base rate P(disease) (log scale)")
ax.set_ylabel("P(disease | positive test)")
ax.set_title("A positive result means little when the condition is rare\n(lines: Bayes' formula; markers: 1,000,000 simulated people)", fontsize=10)
ax.legend(loc="upper left", fontsize=7.5)
fig.tight_layout()
path_png = IMG / "a5_3_posterior_vs_base_rate.png"
fig.savefig(path_png, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path_png.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Set SPECIFICITY = 0.999 and keep SENSITIVITY = 0.90. Which matters more for P(D | +) here, and why?
# 2. Give the second test a correlated error: healthy people who were false positives the first time
#    are false positives again with probability 0.5. How does P(D | two positives) change?
# 3. Simulate 1,000 people instead of 1,000,000. How much does the counted P(D | +) wander? (A5.1)

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
