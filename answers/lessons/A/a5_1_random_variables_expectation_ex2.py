# %% [markdown]
# # A5.1 Random variables, expectation, variance and covariance — answer to exercise 2
# (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a5_1_random_variables_expectation_ex2.py   (CPU, ~1 s)

# %%
from __future__ import annotations

import time
from fractions import Fraction
from itertools import product

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
vals, probs = [0, 1, 4], [Fraction(5, 10), Fraction(3, 10), Fraction(2, 10)]
e_x = sum(v * p for v, p in zip(vals, probs))
e_x2 = sum(v * v * p for v, p in zip(vals, probs))
var_x = e_x2 - e_x**2
print(f"(a) E[X] = {float(e_x)}   E[X^2] = {float(e_x2)}   Var[X] = {float(var_x)}   sd = {float(var_x) ** 0.5:.4f}")
e_y_direct = sum((3 - 2 * v) * p for v, p in zip(vals, probs))
print(f"(b) E[3 - 2X] = {float(3 - 2 * e_x)} (rule) = {float(e_y_direct)} (table)   Var = {float(4 * var_x)}")
print(f"(c) Var[X1 + X2] = {float(2 * var_x)}   Cov[X1, X1 + X2] = {float(var_x)}")
joint = {(0, 0): Fraction(3, 10), (0, 1): Fraction(2, 10), (1, 0): Fraction(1, 10), (1, 1): Fraction(4, 10)}
ex = sum(x * p for (x, _), p in joint.items())
ey = sum(y * p for (_, y), p in joint.items())
exy = sum(x * y * p for (x, y), p in joint.items())
cov = exy - ex * ey
vx, vy = ex - ex**2, ey - ey**2  # X^2 = X for 0/1 values
print(f"(d) E[X] = {float(ex)}  E[Y] = {float(ey)}  E[XY] = {float(exy)}  Cov = {float(cov)}"
      f"  corr = {float(cov) / (float(vx) * float(vy)) ** 0.5:.4f}")
print(f"(d) P(X=1, Y=1) = {float(joint[1, 1])} vs P(X=1) P(Y=1) = {float(ex * ey)}: not independent")

# %%
section("exercise 2 (a): exact, by enumerating 36 outcomes")
pairs = list(product(range(1, 7), repeat=2))  # 36 equally likely (die1, die2) outcomes
p = Fraction(1, 36)
e_m = sum(max(a, b) * p for a, b in pairs)
e_l = sum(min(a, b) * p for a, b in pairs)
e_m2 = sum(max(a, b) ** 2 * p for a, b in pairs)
e_ml = sum(max(a, b) * min(a, b) * p for a, b in pairs)
var_m = e_m2 - e_m**2
cov_ml = e_ml - e_m * e_l
exact = {"E[M]": e_m, "E[L]": e_l, "Var[M]": var_m, "Cov[M, L]": cov_ml}
for k, v in exact.items():
    print(f"{k:<10}= {str(v):<10}= {float(v):.4f}")
print(f"E[ML] = {e_ml} = E[die1] E[die2] = 3.5 * 3.5 = {3.5 * 3.5}")


# %%
def sample_mean(x: FloatArray) -> float:
    """Sample mean. x (n,) -> scalar."""
    return float(x.sum() / len(x))


def sample_var(x: FloatArray, ddof: int = 1) -> float:
    """Sample variance with n - ddof in the denominator. x (n,) -> scalar."""
    r = x - sample_mean(x)
    return float(r @ r / (len(x) - ddof))


def sample_cov(x: FloatArray, y: FloatArray, ddof: int = 1) -> float:
    """Sample covariance with n - ddof in the denominator. x (n,), y (n,) -> scalar."""
    assert x.shape == y.shape, (x.shape, y.shape)
    return float((x - sample_mean(x)) @ (y - sample_mean(y)) / (len(x) - ddof))


section("exercise 2 (b): simulation")
print(f"{'n':>9}{'E[M]':>9}{'E[L]':>9}{'Var[M]':>9}{'Cov[M,L]':>10}")
for n in (100, 10_000, 1_000_000):
    dice = rng.integers(1, 7, size=(n, 2)).astype(np.float64)  # (n, 2)
    m, low = dice.max(axis=1), dice.min(axis=1)  # (n,), (n,)
    est = [sample_mean(m), sample_mean(low), sample_var(m), sample_cov(m, low)]
    C = np.cov(np.stack([m, low]))  # (2, 2), rows are variables
    np.testing.assert_allclose([est[2], est[3]], [C[0, 0], C[0, 1]], rtol=1e-10)
    print(f"{n:>9}" + "".join(f"{e:>9.4f}" for e in est[:3]) + f"{est[3]:>10.4f}")
print(f"{'exact':>9}" + "".join(f"{float(v):>9.4f}" for v in list(exact.values())[:3])
      + f"{float(cov_ml):>10.4f}")
print("own estimators match np.cov: OK")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
