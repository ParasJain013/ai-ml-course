# Answer key: A4.1 Derivatives and partial derivatives

Lesson: [`lessons/A/a4_1_derivatives.md`](../../../lessons/A/a4_1_derivatives.md)

## Exercise 1 (by hand)

**Solution (a).** $f(x) = 1/x = x^{-1}$. Power rule: $f'(x) = -x^{-2}$, so $f'(2) = -0.25$. For the
error predictions: $f''(x) = 2x^{-3}$, $f''(2) = 0.25$; $f'''(x) = -6x^{-4}$, $f'''(2) = -0.375$.

- Forward, $h = 0.1$: $f(2.1) = 0.476190$, so $\frac{0.476190 - 0.5}{0.1} = -0.2381$. Error
  $-0.2381 - (-0.25) = +0.0119$. Predicted $\frac{h}{2}f'' = 0.05 \cdot 0.25 = +0.0125$ (the next term,
  $\frac{h^2}{6}f''' = -0.000625$, brings it to $0.0119$).
- Central: $f(1.9) = 0.526316$, so $\frac{0.476190 - 0.526316}{0.2} = -0.2506$. Error $-0.000627$.
  Predicted $\frac{h^2}{6}f''' = \frac{0.01}{6} \cdot (-0.375) = -0.000625$ ✓.

**Solution (b).** $\frac{\partial f}{\partial x_1} = 6x_1 - 2x_2 = 6 + 2 = 8$ and
$\frac{\partial f}{\partial x_2} = -2x_1 + 3x_2^2 = -2 + 3 = 1$. Forward difference in $x_2$:
$f(1, -0.9) = 3 + 1.8 - 0.729 = 4.071$, $f(1, -1) = 3 + 2 - 1 = 4$, so $\frac{4.071 - 4}{0.1} = 0.71$.
Error $-0.29$. Holding $x_1 = 1$, the second derivative in $x_2$ is $6x_2 = -6$ and the third is 6, so
the prediction is $\frac{h}{2}(-6) + \frac{h^2}{6}(6) = -0.3 + 0.01 = -0.29$, exact because $f$ is cubic in
$x_2$. The sign is negative because the slope is *decreasing* as $x_2$ increases (negative second
derivative), so the average slope over $[-1, -0.9]$ is below the slope at $-1$.

**Solution (c).** With $f$, $f'''$ of order 1 and $h = 10^{-12}$, truncation is
$\frac{h^2}{6} \approx 10^{-25}$ (irrelevant) but rounding is $\varepsilon/h \approx 2 \times 10^{-4}$: only
about 4 correct digits. Use $h \approx (3\varepsilon)^{1/3} \approx 10^{-5}$, which balances the two at
about $10^{-10}$.

```text
== exercise 1 check ==
1/x at 2: exact -0.25   forward = -0.238095 (error +0.011905)   central = -0.250627 (error -0.000627)
predicted: forward h/2 f'' = +0.012500   central h^2/6 f''' = -0.000625
exact partials at (1, -1): df/dx1 = 6 x1 - 2 x2 = 8   df/dx2 = -2 x1 + 3 x2^2 = 1
f(1, -0.9) = 4.0710  f(1, -1) = 4.0000  forward df/dx2 = 0.7100  error = -0.2900
central difference of exp at 1 with h = 1e-12: error = 2.1e-04
central difference of exp at 1 with h = 1e-05: error = 5.9e-11
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) $f'(2) = -0.25$ via the power rule with $k = -1$ | 1 |
| (a) Forward and central values to 4 decimals with errors | 2 |
| (a) Predicted errors from $f''$ and $f'''$, and the comment that they match | 2 |
| (b) Both partial derivatives (8 and 1) | 2 |
| (b) Forward estimate 0.71 and an explanation of the size and sign of the error | 2 |
| (c) Rounding error $\approx \varepsilon/h \approx 10^{-4}$ dominates; recommends $h \approx 10^{-5}$ with the reason | 3 |

**Likely mistakes:**

- Power rule with a negative power: writing $f'(x) = -x^{-1}$ or $+x^{-2}$. Check numerically: the
  forward difference has the right sign.
- Forgetting that $f(1, -1)$ has $-2x_1x_2 = +2$ (two negatives). Write every sign before adding.
- Treating $x_2$ as variable when differentiating with respect to $x_1$: $\frac{\partial}{\partial x_1}(-2x_1x_2)$
  is $-2x_2$, not $-2x_2 - 2x_1$.
- (c): agreeing that smaller is better because the definition has $h \to 0$. The limit is exact
  arithmetic; floats stop at about 16 digits.

## Exercise 2 (code)

**Solution:** [`a4_1_derivatives_ex2.py`](a4_1_derivatives_ex2.py). The core:

```python
def central_diff(f: Callable[[float], float], x: float, h: float, recompute: bool) -> float:
    """Central difference with step h; if recompute, divide by the run actually taken."""
    if not recompute:
        return (f(x + h) - f(x - h)) / (2.0 * h)
    x_plus, x_minus = x + h, x - h  # both rounded to the nearest float64
    return (f(x_plus) - f(x_minus)) / (x_plus - x_minus)  # the run that was really taken
```

```text
== exercise 2: scaled steps ==
h0 = (3 eps)^(1/3) = 8.73e-06   relative errors |estimate - exact| / |exact|
   f       x  spacing |    fixed  fixed+re   scaled  scaled+re
 sin   1e+00  2.2e-16 |  2.1e-11   2.2e-11  1.2e-11    1.1e-11
 sin   1e+03  1.1e-13 |  2.5e-09   1.2e-11  1.3e-05    1.3e-05
 sin   1e+06  1.2e-10 |  4.0e-06   1.5e-11  9.3e-01    9.3e-01
 log   1e+00  2.2e-16 |  3.4e-11   3.3e-11  2.5e-11    2.5e-11
 log   1e+03  1.1e-13 |  6.1e-09   3.6e-09  6.5e-12    1.6e-12
 log   1e+06  1.2e-10 |  8.3e-08   4.1e-06  6.5e-12    6.8e-12
```

How to read it:

- **At $x = 1$ every variant is fine** (about $10^{-11}$): the step and the function's scale agree.
- **$\sin$ with a fixed step needs the recompute.** Near $10^6$ floats are $1.2 \times 10^{-10}$ apart, so
  $x + 10^{-5}$ is off by up to about $6 \times 10^{-11}$, a relative error of about $10^{-5}$ in the run.
  The function was evaluated at the rounded points, so dividing by $2h$ pretends a run that was not
  taken: error $4 \times 10^{-6}$. Dividing by the real run removes it: $1.5 \times 10^{-11}$.
- **$\sin$ with a scaled step is wrong, badly.** $\sin$ curves on a scale of about 1 whatever $x$ is
  (its period is $2\pi$), so a step of $8.7$ at $x = 10^6$ spans more than a full period: truncation
  error $\frac{h^2}{6}\lvert f'''\rvert$ is of order 1. Scaling by $\lvert x\rvert$ is not a law.
- **$\log$ wants the scaled step.** $\log$ curves on a scale of about $x$ itself
  ($f'''/f' = 2/x^2$), so the relative truncation error is about $h^2/(3x^2)$, which stays tiny for
  $h \propto x$. A fixed $10^{-5}$ at $x = 10^6$ is far too small for the function's scale: the
  numerator $f(x + h) - f(x - h) \approx 2 \times 10^{-11}$ is measured on values near 13.8, stored
  about $2 \times 10^{-15}$ apart, so only about 4 digits survive; the two fixed-step numbers
  ($8 \times 10^{-8}$ and $4 \times 10^{-6}$) are both rounding noise, and the recompute cannot help.
- **The rule:** choose $h$ relative to the length scale on which $f$ changes (often $\lvert x\rvert$ for
  powers, logs and exponentials of the input; a fixed scale for periodic functions or standardized
  inputs), and always divide by the run actually taken; it costs nothing.

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| `central_diff` with a correct recompute (divides by `x_plus - x_minus`) | 3 |
| All 12 numbers computed as *relative* errors, in a readable table | 2 |
| Explains the $\sin$ fixed-step error at $10^6$ with float spacing and the recompute fix | 2 |
| Explains why the scaled step fails for $\sin$ (period $2\pi$, truncation) | 2 |
| Explains why the scaled step wins for $\log$ (scale $\propto x$; fixed step is rounding noise) | 2 |
| States the general rule (step on the function's length scale; recompute always) | 1 |

**Likely mistakes:**

- Reporting absolute errors. At $x = 10^6$, $\log' = 10^{-6}$, so an absolute error of $10^{-10}$
  looks excellent while being a relative error of $10^{-4}$.
- Concluding "the scaled step is always better" from the $\log$ rows, or "always worse" from the
  $\sin$ rows. The point of having both functions is that neither is.
- Over-reading the noisy cells: $8 \times 10^{-8}$ against $4 \times 10^{-6}$ in the $\log$ fixed-step
  row is luck in rounding, not an effect of the recompute.
- Using `x + h - x` to recompute and then dividing by `2 * that`, which assumes $x - h$ was rounded by
  the same amount. It usually is not.

## Exercise 3 (think)

**Strong answer:** One central-difference gradient needs 2 loss evaluations per parameter: $2 \times 10^7$
forward passes over the batch, against backpropagation's cost of roughly one forward plus one backward
pass (a small constant, about 2 to 3 forward passes). So it is millions of times slower; a step that
takes 1 s with backprop would take weeks. Accuracy is also poor: in float32 the best central difference
keeps about 5 digits and the best step is about $10^{-3}$, large enough to straddle ReLU kinks; dropout
and other randomness break it entirely unless seeded. Finite differences are worth it as a *test*: when
you write a custom layer, loss or `backward` by hand, or implement an autograd engine. In a test suite:
use tiny inputs (a few examples, a few units), cast everything to float64, fix all seeds and switch off
dropout, avoid points near kinks (or perturb inputs away from them), check only a random subset of
coordinates if the parameter count is large, compare with a relative error and a threshold such as
$10^{-6}$ to $10^{-7}$, and run it in CI on every change to the layer. PyTorch's
`torch.autograd.gradcheck` does exactly this and is the default tool.

**Weak answer:** "Finite differences are inaccurate, so never use them; trust autograd." Weak because
it misses the cost argument (the real reason they are not used for training) and throws away the main
tool for catching wrong gradients in custom code, where autograd cannot help (for example a hand-written
`backward`).

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Cost: about $2P = 2 \times 10^7$ forward passes versus a small constant for backprop | 3 |
| Accuracy in float32 (about 5 digits, large step, kinks, randomness) | 2 |
| When it is worth it: checking hand-written or custom gradients | 2 |
| A robust test design (float64, tiny sizes, seeds, away from kinks, relative error threshold, CI) | 3 |

## Scoring

Total 34 pts. Pass ≥ 70% (24 pts). Topic ids to revisit for each lost criterion: exercise 1 → A4.1 "The
math" (rules, truncation error, rounding error); exercise 2 → A4.1 "In practice" and A3.4 (float
spacing, cancellation); exercise 3 → A4.1 "In practice" and A4.3 (why backpropagation is cheap).
