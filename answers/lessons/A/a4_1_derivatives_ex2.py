# %% [markdown]
# # A4.1 Derivatives — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a4_1_derivatives_ex2.py   (CPU, ~1 s)

# %%
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

SEED = 0
T0 = time.perf_counter()
EPS64 = float(np.finfo(np.float64).eps)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
inv = lambda x: 1.0 / x  # noqa: E731
x, h = 2.0, 0.1
fwd = (inv(x + h) - inv(x)) / h
cen = (inv(x + h) - inv(x - h)) / (2 * h)
print(f"1/x at 2: exact -0.25   forward = {fwd:.6f} (error {fwd + 0.25:+.6f})   central = {cen:.6f} (error {cen + 0.25:+.6f})")
print(f"predicted: forward h/2 f'' = {h / 2 * 2 / x**3:+.6f}   central h^2/6 f''' = {h**2 / 6 * (-6 / x**4):+.6f}")


def f2(x1: float, x2: float) -> float:
    """f(x1, x2) = 3 x1^2 - 2 x1 x2 + x2^3."""
    return 3 * x1**2 - 2 * x1 * x2 + x2**3


print("exact partials at (1, -1): df/dx1 = 6 x1 - 2 x2 =", 6 * 1 - 2 * (-1), "  df/dx2 = -2 x1 + 3 x2^2 =", -2 * 1 + 3 * (-1) ** 2)
fd = (f2(1, -0.9) - f2(1, -1)) / 0.1
print(f"f(1, -0.9) = {f2(1, -0.9):.4f}  f(1, -1) = {f2(1, -1):.4f}  forward df/dx2 = {fd:.4f}  error = {fd - 1:+.4f}")
for hh in (1e-12, 1e-5):
    est = (np.exp(1 + hh) - np.exp(1 - hh)) / (2 * hh)
    print(f"central difference of exp at 1 with h = {hh:.0e}: error = {abs(est - np.e):.1e}")

# %%
section("exercise 2: scaled steps")


def central_diff(f: Callable[[float], float], x: float, h: float, recompute: bool) -> float:
    """Central difference with step h; if recompute, divide by the run actually taken."""
    if not recompute:
        return (f(x + h) - f(x - h)) / (2.0 * h)
    x_plus, x_minus = x + h, x - h  # both rounded to the nearest float64
    return (f(x_plus) - f(x_minus)) / (x_plus - x_minus)  # the run that was really taken


h0 = (3 * EPS64) ** (1 / 3)
print(f"h0 = (3 eps)^(1/3) = {h0:.2e}   relative errors |estimate - exact| / |exact|")
functions = (("sin", np.sin, np.cos), ("log", np.log, lambda t: 1.0 / t))
header = f"{'f':>4} {'x':>7} {'spacing':>8} | {'fixed':>8} {'fixed+re':>9} {'scaled':>8} {'scaled+re':>10}"
print(header)
for name, f, df in functions:
    for x in (1.0, 1e3, 1e6):
        exact = df(x)
        errs = []
        for h in (1e-5, h0 * max(1.0, abs(x))):  # fixed step, then step scaled with |x|
            for recompute in (False, True):
                errs.append(abs(central_diff(f, x, h, recompute) - exact) / abs(exact))
        print(f"{name:>4} {x:>7.0e} {np.spacing(x):>8.1e} | {errs[0]:>8.1e} {errs[1]:>9.1e} {errs[2]:>8.1e} {errs[3]:>10.1e}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
