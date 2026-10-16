# Answer key: A4.1 Derivatives and partial derivatives

Lesson: [`lessons/A/a4_1_derivatives.md`](../../../lessons/A/a4_1_derivatives.md)

## Exercise 1 (by hand)

**Solution (a).** With the power rule term by term: $f(x) = x^4 - 3x$, $f'(x) = 4x^3 - 3$,
$f''(x) = 12x^2$, $f'''(x) = 24x$. At $x = 1$: $f'(1) = 1$, $f''(1) = 12$, $f'''(1) = 24$.

Function values: $f(1.1) = 1.4641 - 3.3 = -1.8359$, $f(1) = 1 - 3 = -2$, $f(0.9) = 0.6561 - 2.7 = -2.0439$.

- Forward: $(-1.8359 - (-2))/0.1 = 0.1641/0.1 = 1.641$. Error $0.641$. Predicted $\frac{f''}{2}h = 6 \times 0.1 = 0.6$.
- Central: $(-1.8359 - (-2.0439))/0.2 = 0.208/0.2 = 1.04$. Error $0.04$. Predicted $\frac{f'''}{6}h^2 = 4 \times 0.01 = 0.04$.

The **central** prediction is exact and the **forward** one is not. For a degree-4 polynomial Taylor's
formula has exactly five terms: $f(x + h) = f + f'h + \frac{f''}{2}h^2 + \frac{f'''}{6}h^3 + \frac{f''''}{24}h^4$
(with $f'''' = 24$). The forward error is *all* the leftover terms divided by $h$:
$\frac{f''}{2}h + \frac{f'''}{6}h^2 + \frac{f''''}{24}h^3 = 0.6 + 0.04 + 0.001 = 0.641$ ✓; the prediction kept only the
first. In the central difference the even-power terms ($h^2$, $h^4$) cancel, so its error is only the
$h^3$ term over $h$: $\frac{f'''}{6}h^2$, nothing left out.

**Solution (b).** $g(x_1, x_2) = x_1^2x_2 + 3x_2 - x_1$. Holding $x_2$ fixed, $x_1^2x_2$ is a constant times
$x_1^2$: $\frac{\partial g}{\partial x_1} = 2x_1x_2 - 1$. Holding $x_1$ fixed, $x_1^2x_2 + 3x_2$ is
$(x_1^2 + 3)\,x_2$: $\frac{\partial g}{\partial x_2} = x_1^2 + 3$. At $(2, -1)$: $-5$ and $7$.

Check: $g(2.1, -1) = -4.41 - 3 - 2.1 = -9.51$, $g(1.9, -1) = -3.61 - 3 - 1.9 = -8.51$, central
$(-9.51 + 8.51)/0.2 = -5$ ✓ (exact: $g$ is quadratic in $x_1$).

```text
== exercise 1 check ==
f(1.1) = -1.8359  f(1) = -2.0000  f(0.9) = -2.0439
forward = 1.6410 (error 0.6410, f''h/2 = 0.6000)
central = 1.0400 (error 0.0400, f'''h^2/6 = 0.0400)
dg/dx1 at (2, -1) = 2 x1 x2 - 1 = -5   dg/dx2 = x1^2 + 3 = 7
g(2.1, -1) = -9.5100  g(1.9, -1) = -8.5100  central = -5.0000
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) $f'$, $f''$, $f'''$ correct | 2 |
| (a) Forward and central values and errors correct | 2 |
| (a) Predictions 0.6 and 0.04, and the correct "central exact, forward not" with the Taylor reason | 3 |
| (b) Both partial derivatives as formulas | 2 |
| (b) Values $-5$ and $7$ at $(2, -1)$ | 1 |
| (b) Central-difference check done holding $x_2 = -1$ | 2 |

**Likely mistakes:**

- Treating $x_2$ as a variable while differentiating in $x_1$ (writing $\partial g/\partial x_1 = 2x_1x_2 + x_1^2 - 1$,
  a product rule applied where none is needed). Partial means the other inputs are constants.
- Dividing the central difference by $h$ instead of $2h$, getting a slope twice as large.
- Arithmetic slips in $1.1^4$: compute $1.1^2 = 1.21$, then $1.21^2 = 1.4641$.
- Saying "the central prediction is approximate because Taylor is an approximation". For a polynomial
  Taylor's formula is exact once you include every term; the question is which terms the prediction kept.

## Exercise 2 (code)

**Solution (a).** Add the Taylor expansions of $f(x + h)$ and $f(x - h)$: the odd powers cancel and the
even ones double, $f(x + h) + f(x - h) = 2f(x) + f''(x)h^2 + \frac{f''''(x)}{12}h^4 + \dots$ Subtract $2f(x)$ and
divide by $h^2$: the estimate is $f''(x) + \frac{f''''(x)}{12}h^2 + \dots$, so the truncation error is
proportional to $h^2$. Rounding: each of the three values is off by up to about $\varepsilon\lvert f\rvert$,
and the combination $1, -2, 1$ can add them up to about $4\varepsilon\lvert f\rvert$; dividing by $h^2$ gives
$\approx 4\varepsilon\lvert f\rvert/h^2$ (any constant of order 1 is accepted).

**Solution (b, c).** [`a4_1_derivatives_ex2.py`](a4_1_derivatives_ex2.py):

```python
def second_diff(fn: Callable[[float], float], x: float, h: float) -> float:
    """Central second difference (f(x+h) - 2 f(x) + f(x-h)) / h^2. Truncation error ~ f'''' h^2 / 12."""
    return (fn(x + h) - 2.0 * fn(x) + fn(x - h)) / (h * h)
```

```text
== exercise 2: second derivative ==
h = 1e-01   error = 2.00e-02
h = 1e-02   error = 2.00e-04
h = 1e-03   error = 2.00e-06
h = 1e-04   error = 1.59e-08
h = 1e-05   error = 3.21e-06
h = 1e-06   error = 3.76e-04
h = 1e-07   error = 3.48e-02
h = 1e-08   error = 5.34e+00
predicted best h = 1.5e-04   error there = 6.3e-08
```

Truncation at $h = 0.1$ is $\frac{24}{12} \times 0.01 = 0.02$ ✓ (exact again, because $x^4$ has no sixth
derivative). Minimizing $\frac{\lvert f''''\rvert}{12}h^2 + \frac{4\varepsilon\lvert f\rvert}{h^2}$: set the
derivative with respect to $h$ to zero, $\frac{\lvert f''''\rvert}{6}h = \frac{8\varepsilon\lvert f\rvert}{h^3}$,
so $h^4 = 48\varepsilon\lvert f\rvert/\lvert f''''\rvert$, $h^\star \approx 1.5 \times 10^{-4}$, about $\varepsilon^{1/4}$. The sweep's best
is near $10^{-4}$ ✓.

Why a larger step: the rounding error is divided by $h^2$ instead of $h$, so it grows much faster as $h$
shrinks and the balance point moves to larger $h$. At $h = 10^{-8}$, a perfectly good step for a
forward first difference, the second difference is off by 5, i.e. 40%.

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Taylor derivation of the $h^2$ truncation error, odd terms cancelling | 3 |
| (a) Rounding error $\propto \varepsilon\lvert f\rvert/h^2$ with a reason | 2 |
| (b) `second_diff` correct, typed, with a docstring; $x^4$ written with multiplications | 2 |
| (b) Sweep over the stated range and a log-log plot with labelled axes and a title | 3 |
| (c) Predicted best $h \approx 10^{-4}$ (any order-1 constant) and compared with the sweep | 2 |
| (c) Explains the larger step by the $1/h^2$ rounding term | 2 |

**Likely mistakes:**

- Writing `x ** 4` and getting slightly different rounding on another machine: not wrong, but the
  exercise asked for multiplications so results are reproducible.
- Plotting on linear axes, where the U-shape is invisible.
- Expecting the best $h$ to be $\sqrt[3]{\varepsilon}$ (copied from the first-derivative case) without redoing
  the balance with the new error terms.

## Exercise 3 (think)

**Strong answer:** The test cannot tell a bug from rounding. In float32 $\varepsilon \approx 1.2 \times 10^{-7}$,
so with $h = 10^{-7}$ the rounding term $\varepsilon\lvert f\rvert/h$ is about $1.2\lvert f\rvert$: the numerical
gradient has *no* correct digits when the loss is of order 1. (Worse, adding $h = 10^{-7}$ to a parameter
of size 2 or more does not change it at all in float32, whose numbers there are $2.4 \times 10^{-7}$ apart,
giving a numerical gradient of exactly 0.) Even the best
float32 central difference, at $h \approx \sqrt[3]{\varepsilon} \approx 5 \times 10^{-3}$, has only 4–5 correct
digits, so `rtol = 1e-7` would fail on a perfect gradient. Rewrite the test: cast the parameters and
inputs to float64 for the check (the gradient formula does not depend on the dtype), use central
differences with $h \approx 10^{-6}$, compare with a relative error
$\frac{\lvert a - b\rvert}{\max(\lvert a\rvert + \lvert b\rvert, \text{tiny})}$ against about $10^{-6}$, use small random
inputs away from any kinks, and add a negative control: deliberately break the gradient (drop a factor)
and confirm the test fails. Then a failure means a bug.

**Weak answer:** "Loosen the tolerance to `rtol = 1e-2` so the test passes." Weak because it hides the
cause: at $h = 10^{-7}$ in float32 the numerical gradient is noise, so a tolerance loose enough to pass
correct code will also pass many wrong gradients (a missing factor of 2 is a 50% error, but a
transposed gradient or a wrong term on a few parameters can slip through).

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Identifies rounding as the cause, with $\varepsilon_{32} \approx 10^{-7}$ and the $\varepsilon\lvert f\rvert/h$ term | 3 |
| Notes that `rtol = 1e-7` is impossible in float32 even at the best $h$ | 2 |
| Rewrites the check in float64, central differences, $h \approx 10^{-6}$, relative error metric | 3 |
| Adds a negative control or another way to make sure the test can fail | 2 |

## Scoring

Total 36 pts. Pass ≥ 70% (26 pts). Topic ids to revisit for each lost criterion: exercise 1 → A4.1
"The math" (rules table, partial derivatives, Taylor's formula); exercise 2 → A4.1 "The math"
(rounding error) and "Key plot"; exercise 3 → A4.1 "In practice" and A3.4 (machine epsilon).
