# %% [markdown]
# # A5.3 Conditional probability and Bayes' theorem — answer to exercise 2
# (and checks of exercises 1 and 3)
# Run: uv run python answers/lessons/A/a5_3_bayes_theorem_ex2.py   (CPU, ~1 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]
SEED = 0
T0 = time.perf_counter()
N_PEOPLE = 1_000_000
BASE_RATE = 0.01
SENS_A, SPEC_A = 0.90, 0.91
SENS_B, SPEC_B = 0.80, 0.97
P_C = 0.05  # share of healthy people with the cross-reacting condition C
FP_C = 0.5  # P(positive | healthy, C), for each test
FP_A_NOT_C, FP_B_NOT_C = 0.0684, 0.00526  # P(positive | healthy, not C)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def posterior(prior: float, sens: float, spec: float) -> float:
    """P(D | +) by Bayes' theorem."""
    return sens * prior / (sens * prior + (1 - spec) * (1 - prior))


# %%
section("exercise 1 check")
print(f"(a) 1,000 emails: spam 200 -> flagged {200 * 0.95:.0f}, not {200 * 0.05:.0f};"
      f" legit 800 -> flagged {800 * 0.02:.0f}, not {800 * 0.98:.0f}")
p_flag = posterior(0.2, 0.95, 0.98)
p_not = 0.05 * 0.2 / (0.05 * 0.2 + 0.98 * 0.8)
print(f"(b) P(spam | flagged) = {p_flag:.4f}   P(spam | not flagged) = {p_not:.4f}")
lr = 0.95 / 0.02
odds = lr * 0.2 / 0.8
print(f"(c) LR+ = {lr:.1f}   posterior odds = {lr:.1f} x 0.25 = {odds:.4f} -> {odds / (1 + odds):.4f}")
print(f"(d) at 1% spam: P(spam | flagged) = {posterior(0.01, 0.95, 0.98):.4f}")
section("exercise 3 check")
print(f"P(fraud | flagged) = {posterior(0.001, 0.99, 0.99):.4f}")
print(f"per 1,000,000 transactions: fraud flagged = {1e6 * 0.001 * 0.99:,.0f}   "
      f"honest flagged = {1e6 * 0.999 * 0.01:,.0f}")


# %%
def simulate_two_tests(correlated: bool, gen: np.random.Generator) -> tuple[BoolArray, BoolArray, BoolArray, BoolArray]:
    """Disease, test A, test B and condition C for N_PEOPLE people. -> four (n,) boolean arrays."""
    n = N_PEOPLE
    disease = gen.random(n) < BASE_RATE
    cond_c = ~disease & (gen.random(n) < P_C) if correlated else np.zeros(n, dtype=bool)
    ua, ub = gen.random(n), gen.random(n)  # separate uniforms: independent given the status
    if correlated:
        fp_a = np.where(cond_c, FP_C, FP_A_NOT_C)  # (n,) per-person false positive probability
        fp_b = np.where(cond_c, FP_C, FP_B_NOT_C)
    else:
        fp_a, fp_b = np.full(n, 1 - SPEC_A), np.full(n, 1 - SPEC_B)
    test_a = np.where(disease, ua < SENS_A, ua < fp_a)
    test_b = np.where(disease, ub < SENS_B, ub < fp_b)
    return disease, test_a, test_b, cond_c


lr_a, lr_b = SENS_A / (1 - SPEC_A), SENS_B / (1 - SPEC_B)
odds = BASE_RATE / (1 - BASE_RATE) * lr_a * lr_b
p_formula = odds / (1 + odds)

section("exercise 2 (a): conditionally independent tests")
gen = np.random.default_rng(SEED)
d, a, b, _ = simulate_two_tests(False, gen)
both = a & b
print(f"people positive on both: {both.sum():,}")
print(f"P(D | A+, B+): counted {d[both].mean():.4f}   odds form {p_formula:.4f}"
      f"   (LR_A = {lr_a:.2f}, LR_B = {lr_b:.2f})")

section("exercise 2 (b): correlated false positives")
d, a, b, c = simulate_two_tests(True, gen)
healthy = ~d
print(f"false positive rate A = {a[healthy].mean():.4f} (target 0.09)   B = {b[healthy].mean():.4f} (target 0.03)")
both = a & b
print(f"P(A+, B+ | healthy): counted {both[healthy].mean():.5f}   independence predicts "
      f"{(1 - SPEC_A) * (1 - SPEC_B):.5f}   exact {P_C * FP_C**2 + (1 - P_C) * FP_A_NOT_C * FP_B_NOT_C:.5f}")
print(f"P(D | A+, B+): counted {d[both].mean():.4f}   odds form (assumes independence) {p_formula:.4f}")
p_both_h = P_C * FP_C**2 + (1 - P_C) * FP_A_NOT_C * FP_B_NOT_C
p_true = SENS_A * SENS_B * BASE_RATE / (SENS_A * SENS_B * BASE_RATE + p_both_h * (1 - BASE_RATE))
print(f"exact with the correlated model: {p_true:.4f}   share of healthy double-positives with C: "
      f"{c[both & healthy].mean():.4f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
