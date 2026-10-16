# A4.1 Derivatives and partial derivatives

**Core** · Part A, week 4 (course week 04) · ~1.5 h · Lab:
[`labs/A/a4_1_derivatives.py`](../../labs/A/a4_1_derivatives.py)

## Why it matters

Training a model means changing its parameters so the loss goes down, and the **derivative** is what
says which way is down and how steeply. Every training loop you will write, from linear regression
in [B1.5](../B/b1_5_linear_regression.md) to a GPT in Part F, is driven by derivatives. You will
derive many of them by hand, and a hand-derived derivative is wrong more often than you expect:
a dropped factor of 2, a sign, a transpose. A wrong gradient does not crash; the model still trains,
just badly, and you can lose days. The cheap defence is a **finite-difference** check: nudge the
input a little, measure how much the output moves, and compare. This lesson builds that check and
shows how to choose the size of the nudge.

## Prerequisite check

- Can you evaluate $x^3$, $e^x$ and $\log x$ at a given point with NumPy? If not, revisit
  [A1.2](a1_2_numpy_arrays.md).
- Can you compute the slope of the straight line through two points $(x_1, y_1)$ and $(x_2, y_2)$?
  If not, revisit [A2.4](a2_4_projections_least_squares.md) (fitting a line), or simply recall
  slope = rise / run.
- Can you expand $(a + b)^2$ and $(a + b)^3$ by hand? (Used in every derivation below; no topic to
  revisit, just practise $(a + b)^2 = a^2 + 2ab + b^2$.)
- Do you know why `0.1 + 0.2 == 0.3` is `False` in Python? If not, the short explanation is in
  "The math" below; [A3.4](a3_4_numerical_stability.md) (Depth) has the long one.

## Intuition

### Zoom in until the curve looks straight

Plot $f(x) = x^3$ and zoom in on the point $x = 2$. At first you see a curve. Zoom in ten times and it
bends less. Zoom in a thousand times and, across the window, it is indistinguishable from a straight
line. The slope of that line is the **derivative** of $f$ at 2, written $f'(2)$. For $x^3$ at 2 it is
12: near $x = 2$, nudging $x$ by a small amount $h$ moves $f$ by about $12h$.

That sentence is the whole idea, and the one to remember:

$$f(x + h) \approx f(x) + f'(x)\,h \qquad \text{for small } h.$$

A derivative is a local exchange rate: output change per unit of input change, valid near one point.
Positive means "increase $x$ and $f$ goes up", negative means it goes down, zero means you are at a
flat spot (perhaps a minimum, which is what training looks for). Large in size means a small nudge
changes $f$ a lot.

### Measure it with two evaluations

You do not need calculus to *estimate* a derivative. Evaluate $f$ at two nearby points and compute
rise over run, the slope of the **secant** line through them. With $x = 2$, $h = 0.1$:
$(2.1^3 - 2^3)/0.1 = (9.261 - 8)/0.1 = 12.61$. That is close to 12, not equal, because the curve bends
between 2 and 2.1. A smaller $h$ gets closer. This is a **finite difference**, and it works for any
function you can evaluate, including a 50-line loss function whose derivative you derived by hand
and want to check.

It cannot be pushed to $h = 10^{-20}$, though. Computers store numbers with about 16 significant
digits, so $f(x + h)$ and $f(x)$ become equal in every stored digit and the difference is noise. The
key plot shows the result: the error falls as $h$ shrinks, then rises again.

### Many inputs: one knob at a time

A loss depends on many parameters at once. A **partial derivative** asks the same question about one
of them while the others are frozen: "if I turn only knob $w$, how fast does the loss move?" It is
written $\frac{\partial f}{\partial w}$ (the curly $\partial$ says "the others are held fixed"). To
estimate it, nudge $w$ alone and leave every other input untouched. Collect all the partial
derivatives into one vector and you have the gradient, the subject of [A4.2](a4_2_gradients_jacobians.md).

> **Bridge:** a Prometheus `rate(http_requests_total[5m])` turns an ever-growing counter into
> requests per second: the change in the counter divided by the time window. That is a finite
> difference, and requests per second is the derivative of the counter with respect to time.
> **Where it breaks:** `rate()` has to use a fixed, fairly wide window because the counter only moves
> in whole requests; a derivative is the limit as the window shrinks to zero, and for a smooth
> function you can shrink it a long way (but, as this lesson shows, not all the way).

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $f$, $g$ | functions; here $f:\mathbb{R} \to \mathbb{R}$ unless stated | — |
| $x$ | the input point | scalar |
| $h$ | step size (the nudge) | scalar, $h > 0$ |
| $f'(x)$, $\frac{df}{dx}$ | derivative of $f$ at $x$ (two notations for the same thing) | scalar |
| $f''(x)$, $f'''(x)$ | second and third derivatives (the derivative of $f'$, and of $f''$) | scalar |
| $\frac{\partial f}{\partial x_j}$ | partial derivative of $f$ with respect to input $j$, others fixed | scalar |
| $\mathbf{x}$ | a vector of inputs $(x_1, \dots, x_n)$ | `(n,)` |
| $\mathbf{e}_j$ | unit vector: 1 in position $j$, 0 elsewhere | `(n,)` |
| $\varepsilon$ | **machine epsilon**, $2.2 \times 10^{-16}$ for float64 | scalar |

**The definition.** The slope of the secant from $x$ to $x + h$ is $\frac{f(x + h) - f(x)}{h}$. The
derivative is the value it approaches as $h$ shrinks towards 0:

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}.$$

"$\lim_{h \to 0}$" means: the number the expression gets arbitrarily close to as $h$ gets closer to
0 (it is never evaluated *at* 0, where it would be $0/0$).

**Deriving one by hand.** For $f(x) = x^3$, expand $(x + h)^3 = x^3 + 3x^2h + 3xh^2 + h^3$, so

$$\frac{(x + h)^3 - x^3}{h} = \frac{3x^2h + 3xh^2 + h^3}{h} = 3x^2 + 3xh + h^2 \;\xrightarrow{\;h \to 0\;}\; 3x^2.$$

So $f'(x) = 3x^2$, and $f'(2) = 12$. The leftover $3xh + h^2$ is exactly the error of the secant
slope, a fact the worked example uses.

**The rules you will use.** The same expansion trick, done once for each, gives this table. You do
not need to re-derive it; you need to recognise the entries and check them numerically.

| $f(x)$ | $f'(x)$ | Note |
|---|---|---|
| $c$ (a constant) | $0$ | a flat line has no slope |
| $x^k$ | $k\,x^{k-1}$ | any real $k$: $\sqrt{x} = x^{1/2} \to \tfrac{1}{2}x^{-1/2}$ |
| $e^x$ | $e^x$ | the function that is its own slope |
| $\log x$ | $1/x$ | natural log, $x > 0$ |
| $\sin x$, $\cos x$ | $\cos x$, $-\sin x$ | $x$ in radians |
| $c\,f(x)$ | $c\,f'(x)$ | constants factor out |
| $f(x) + g(x)$ | $f'(x) + g'(x)$ | derivatives add |
| $f(x)\,g(x)$ | $f'(x)\,g(x) + f(x)\,g'(x)$ | product rule |

Composite functions such as $e^{-x^2}$ or $(wx - y)^2$ need the chain rule, the subject of
[A4.3](a4_3_chain_rule_graphs.md). Until then, expand: $(wx - y)^2 = w^2x^2 - 2wxy + y^2$, a sum of
terms the table handles.

**Higher derivatives.** $f''$ is the derivative of $f'$: the rate at which the slope itself changes,
which is how *curved* $f$ is. For $x^3$: $f' = 3x^2$, $f'' = 6x$, $f''' = 6$. A positive $f''$ means the
curve bends upwards like a bowl; you will use that in [A4.5](a4_5_convexity_lagrange.md).

**Partial derivatives.** For $f(x_1, \dots, x_n)$, the partial derivative with respect to $x_j$ treats
every other input as a constant:

$$\frac{\partial f}{\partial x_j}(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x} + h\,\mathbf{e}_j) - f(\mathbf{x})}{h}.$$

Adding $h\mathbf{e}_j$ changes coordinate $j$ only. Example: the squared error of a line $wx + b$ on a
single example $(x, y)$ is $f(w, b) = (wx + b - y)^2$. Here $x$ and $y$ are *data*, fixed numbers; the
variables are $w$ and $b$. Write the residual $r = wx + b - y$ and expand $f = r^2$ as
$w^2x^2 + 2wx(b - y) + (b - y)^2$. Differentiating in $w$ with $b$ fixed: $2wx^2 + 2x(b - y) = 2x(wx + b - y)$.
Doing the same in $b$ (expand $(b + (wx - y))^2$ instead):

$$\frac{\partial f}{\partial w} = 2\,r\,x, \qquad \frac{\partial f}{\partial b} = 2\,r.$$

Read them: the push on each parameter is proportional to the residual (no error, no push), and the
push on $w$ is scaled by the input $x$ that $w$ multiplies.

**Why finite differences have an error: Taylor's formula.** Near $x$, a smooth function is well
approximated by a polynomial in $h$ whose coefficients are its derivatives (**Taylor's formula**,
stated, not proved):

$$f(x + h) = f(x) + f'(x)\,h + \frac{f''(x)}{2}\,h^2 + \frac{f'''(x)}{6}\,h^3 + \dots$$

The first two terms are the "zoom in until it is straight" line; the rest describe the bending. For
$x^3$ this is exactly the expansion above, with $f''(2)/2 = 6$ and $f'''/6 = 1$. Now compute the two
estimates.

*Forward difference.* Subtract $f(x)$ and divide by $h$:

$$\frac{f(x + h) - f(x)}{h} = f'(x) + \underbrace{\frac{f''(x)}{2}h + \frac{f'''(x)}{6}h^2 + \dots}_{\text{error} \;\approx\; \frac{f''(x)}{2}h}$$

The error is proportional to $h$: halve the step, halve the error.

*Central difference.* Write the formula for $f(x - h)$ too: it is the same with $-h$ in place of $h$,
so the odd powers flip sign: $f(x - h) = f(x) - f'(x)h + \frac{f''}{2}h^2 - \frac{f'''}{6}h^3 + \dots$
Subtract it from $f(x + h)$. The $f(x)$ terms cancel, the $h^2$ terms cancel, and the odd terms double:
$f(x + h) - f(x - h) = 2f'(x)h + \frac{f'''}{3}h^3 + \dots$ Divide by $2h$:

$$\frac{f(x + h) - f(x - h)}{2h} = f'(x) + \underbrace{\frac{f'''(x)}{6}h^2 + \dots}_{\text{error} \;\approx\; \frac{f'''(x)}{6}h^2}$$

The error is proportional to $h^2$: halve the step, quarter the error. That is why gradient checks
use central differences: one extra function evaluation buys a much smaller error.

**Rounding error.** A float64 stores about 16 significant digits. The gap between 1.0 and the next
representable number is the machine epsilon $\varepsilon = 2.2 \times 10^{-16}$, and every stored
result is rounded to within about $\varepsilon/2$ *relative* error. So the computed $f(x + h)$ can be
off by roughly $\varepsilon\lvert f(x)\rvert$, and the computed difference $f(x + h) - f(x)$ carries that
absolute error whatever $h$ is. Dividing by $h$ blows it up:

$$\text{rounding error of the estimate} \approx \frac{\varepsilon\,\lvert f(x)\rvert}{h}.$$

This one *grows* as $h$ shrinks. Total error, for the forward difference, is about
$\frac{\lvert f''\rvert}{2}h + \frac{\varepsilon\lvert f\rvert}{h}$: one term falls with $h$, the other
rises, so there is a best $h$ in between. Its derivative with respect to $h$ is
$\frac{\lvert f''\rvert}{2} - \frac{\varepsilon\lvert f\rvert}{h^2}$ (power rule, with $1/h = h^{-1}$), which is zero when
$h^2 = 2\varepsilon\lvert f\rvert/\lvert f''\rvert$, so $h^\star = \sqrt{2\varepsilon\lvert f\rvert/\lvert f''\rvert}$. For
ordinary functions ($\lvert f\rvert$ and $\lvert f''\rvert$ of similar size) that is about $\sqrt{\varepsilon} \approx 10^{-8}$. For the central difference,
$\frac{\lvert f'''\rvert}{6}h^2 + \frac{\varepsilon\lvert f\rvert}{h}$ has derivative
$\frac{\lvert f'''\rvert}{3}h - \frac{\varepsilon\lvert f\rvert}{h^2}$, zero at
$h^\star = \sqrt[3]{3\varepsilon\lvert f\rvert/\lvert f'''\rvert}$, about $\sqrt[3]{\varepsilon} \approx 10^{-5}$
to $10^{-6}$. (These formulas are estimates of the order of magnitude, not exact.)

## Worked example by hand

**1. Slopes of $f(x) = x^3$ at $x = 2$, where $f'(2) = 12$.**

With $h = 0.1$: $f(2.1) = 2.1 \times 2.1 \times 2.1 = 4.41 \times 2.1 = 9.261$ and
$f(1.9) = 3.61 \times 1.9 = 6.859$.

- Forward: $(9.261 - 8)/0.1 = 12.61$. Error $0.61$. Prediction from the expansion: $3xh + h^2 = 0.6 + 0.01 = 0.61$ ✓.
- Central: $(9.261 - 6.859)/0.2 = 2.402/0.2 = 12.01$. Error $0.01$. Prediction: $\frac{f'''}{6}h^2 = h^2 = 0.01$ ✓
  (for a cubic the series stops, so the prediction is exact).

With $h = 0.05$: $f(2.05) = 8.615125$, $f(1.95) = 7.414875$.

- Forward: $(8.615125 - 8)/0.05 = 12.3025$. Error $0.3025$, about half of $0.61$.
- Central: $(8.615125 - 7.414875)/0.1 = 12.0025$. Error $0.0025$, exactly a quarter of $0.01$.

**2. Partial derivatives of $f(w, b) = (wx + b - y)^2$ for the example $x = 2$, $y = 3$, at $(w, b) = (1, 0)$.**

Residual $r = 1 \cdot 2 + 0 - 3 = -1$, so $f = 1$. From the formulas: $\frac{\partial f}{\partial w} = 2rx = 2 \cdot (-1) \cdot 2 = -4$
and $\frac{\partial f}{\partial b} = 2r = -2$. Both are negative: increasing $w$ or $b$ makes the line
pass closer to the target, so the error falls.

Check $\partial f/\partial w$ with a central difference, $h = 0.01$, holding $b = 0$:
$f(1.01, 0) = (2.02 - 3)^2 = 0.98^2 = 0.9604$ and $f(0.99, 0) = (1.98 - 3)^2 = 1.02^2 = 1.0404$, so
$(0.9604 - 1.0404)/0.02 = -0.08/0.02 = -4$ ✓. It is exact, not approximate, because $f$ is a quadratic
in $w$: $f''' = 0$, so the central difference has no truncation error at all.

At $(w, b) = (1.5, 0)$ the line hits the target ($1.5 \times 2 = 3$), $r = 0$, and both partial
derivatives are 0: a flat spot, here the minimum.

The lab prints the same numbers:

```text
== worked example: slopes of x^3 at x = 2 ==
h = 0.1   forward = 12.6100 (error 0.6100, predicted 0.6100)   central = 12.0100 (error 0.0100, predicted 0.0100)
h = 0.05  forward = 12.3025 (error 0.3025, predicted 0.3025)   central = 12.0025 (error 0.0025, predicted 0.0025)

== worked example: partial derivatives ==
at (w, b) = (1.0, 0.0): residual r = -1.0   f = 1.0
by hand: df/dw = 2 r x = -4.0   df/db = 2 r = -2.0
central differences with h = 0.01: df/dw = -4.0000   df/db = -2.0000
f(1.01, 0) = 0.9604   f(0.99, 0) = 1.0404
```

## Key plot

![Log-log plot of the error of forward and central differences of x cubed at x = 2 against the step size h: both errors fall along their truncation lines as h shrinks, then rise jaggedly along the rounding line; central reaches about 1e-11 near h = 1e-5, forward only about 1e-7 near h = 1e-8](img/a4_1_error_vs_step.png)

Read it from right to left, as $h$ shrinks. On the right, the error follows the dashed truncation
lines: slope 1 for forward (error $\propto h$), slope 2 for central (error $\propto h^2$), so central is
far more accurate at the same $h$. On the left, both follow the dotted rounding line $\varepsilon\lvert f\rvert/h$,
jagged because rounding is erratic, sometimes lucky. The vertical lines mark the predicted best steps.
The U-shape is the reason `h = 1e-12` is a bad choice for a gradient check, not a careful one.

## From scratch in NumPy

Both estimators are one line each:

```python
def forward_diff(f: Callable[[float], float], x: float, h: float) -> float:
    """Forward-difference slope (f(x + h) - f(x)) / h. Error shrinks like h."""
    return (f(x + h) - f(x)) / h


def central_diff(f: Callable[[float], float], x: float, h: float) -> float:
    """Central-difference slope (f(x + h) - f(x - h)) / 2h. Error shrinks like h^2."""
    return (f(x + h) - f(x - h)) / (2 * h)
```

The lab evaluates both for $x^3$ at 2 over a sweep of step sizes. The function is written `x * x * x`
rather than `x ** 3` so that every machine rounds it identically (basic arithmetic is exactly
specified by the IEEE 754 standard; library functions like `exp` can differ in the last bit between
platforms):

```text
== from scratch: error against step size ==
      h   forward error   central error
  1e-01        6.10e-01        1.00e-02
  1e-03        6.00e-03        1.00e-06
  1e-05        6.00e-05        2.12e-10
  1e-07        5.84e-07        6.32e-09
  1e-09        9.93e-07        9.93e-07
  1e-11        9.93e-07        9.93e-07
  1e-13        9.59e-03        9.59e-03
predicted best h: forward 1.7e-08 (error there 2.0e-07)   central 9.6e-06 (error there 1.6e-11)
```

Each factor-of-100 drop in $h$ cuts the forward error 100 times and the central error 10,000 times,
until about $h = 10^{-5}$. Below that, rounding takes over: at $h = 10^{-13}$ the central difference
is as inaccurate as it was at $h = 10^{-1}$. At $h = 10^{-9}$ and below the two estimates are identical,
because the difference $f(x + h) - f(x)$ is now mostly rounding noise either way.

For a function of a vector, a partial derivative nudges one coordinate. `e_j` is the unit vector
$\mathbf{e}_j$; the code is the definition with a central difference:

```python
def partial(f: Callable[[FloatArray], float], x: FloatArray, j: int, h: float = 1e-6) -> float:
    """Central-difference partial derivative of f along coordinate j. x (n,) -> scalar."""
    e_j = np.zeros_like(x)  # (n,), unit vector along coordinate j
    e_j[j] = 1.0
    return (f(x + h * e_j) - f(x - h * e_j)) / (2 * h)
```

The default `h = 1e-6` sits near the bottom of the central curve for functions whose values are of
order 1 to 100. Applied to the squared error with $\boldsymbol{\theta} = (w, b)$ packed into one array:

```text
== from scratch: partial derivatives of f(w, b) ==
df/dw: numerical = -4.000000   by hand = -4.0   match: True
df/db: numerical = -2.000000   by hand = -2.0   match: True
at (w, b) = (1.5, 0): [0.0, 0.0]
```

Loop `partial` over every $j$ and you have a numerical gradient. [A4.2](a4_2_gradients_jacobians.md)
turns that loop into a general gradient checker.

## With SciPy and NumPy

There is no single "derivative" call you will use in training (that is the job of autograd,
[C1.3](../C/c1_3_autograd_engine.md)), but three library tools do the same job as this lesson's code:

- `scipy.differentiate.derivative(f, x)` runs central differences at several step sizes and
  extrapolates, with an error estimate. It needs `f` to work elementwise on arrays. (New in SciPy 1.15;
  VERIFY (as of 2026-10) if you change the pinned SciPy 1.18.1.)
- `scipy.optimize.approx_fprime(x, f)` estimates every partial derivative of `f` at once. It uses
  **forward** differences with step $\sqrt{\varepsilon} \approx 1.5 \times 10^{-8}$ by default, so expect
  about 8 correct digits, not 16.
- `np.gradient(values, spacing)` differentiates *sampled data* (values on a grid, not a function you
  can call): central differences inside, one-sided differences at the two ends.

```text
== with scipy and numpy ==
scipy.differentiate.derivative(exp, 1): 2.7182818285   exact e = 2.7182818285   success = True
central_diff(exp, 1, h=1e-5) matches scipy.differentiate.derivative: OK
scipy.optimize.approx_fprime at (1, 0) = [-4.0, -2.0]   (default step 1.49e-08 = sqrt(eps))
approx_fprime matches the hand-derived partials (rtol 1e-6): OK
np.gradient on sampled sin (spacing 0.0100): max error inside = 1.5e-05   at the ends = 3.0e-03
```

The `np.gradient` line shows the theory once more: inside the grid the central error is about
$\frac{h^2}{6}\lvert\sin'''\rvert \le \frac{10^{-4}}{6} \approx 1.7 \times 10^{-5}$; at the ends, where only a one-sided
difference is possible, it is about $\frac{h}{2}\lvert\sin''\rvert$, 200 times worse. Derivatives of
measured data are noisiest at the edges of the series.

## In practice

- **When to use it:** to *check* derivatives, never to train with. A numerical gradient over $n$
  parameters costs $2n$ evaluations of the loss; backpropagation ([A4.3](a4_3_chain_rule_graphs.md))
  costs about two. Use finite differences in tests, on small inputs, whenever you write a gradient by
  hand or a custom autograd function.
- **Failure modes:**
  - *Step too small:* `h = 1e-12` "for accuracy" gives an error near $10^{-4}$ relative. You notice the
    check fails for a gradient you know is right; plot the error against $h$ or use $10^{-6}$.
  - *float32:* $\varepsilon \approx 1.2 \times 10^{-7}$, so even the best central difference has only about
    5 correct digits, and `h = 1e-6` is deep in the rounding region. You notice relative errors of
    $10^{-2}$ to $10^{-3}$ on correct code; run gradient checks in float64.
  - *Kinks:* `abs`, `max(0, x)` (ReLU) and `round` have no derivative at their corners; a central
    difference straddling a corner returns the average of the two slopes. You notice a failure at a
    single coordinate whose input is within $h$ of the corner; move the test point.
  - *Forward differences mistaken for exact:* `approx_fprime` agrees with the truth to about $10^{-8}$
    relative at best. Comparing with `rtol=1e-12` fails on correct code.
  - *Derivatives of noisy data:* `np.gradient` on measured data amplifies noise by $1/h$. You notice a
    jagged derivative; smooth first or fit a model and differentiate the model.
- **Defaults that work:** central differences, float64, $h = 10^{-6}$ for inputs and outputs of
  order 1 (scale $h$ with $\lvert x\rvert$ when inputs are large: $h = 10^{-6}\max(1, \lvert x\rvert)$), and
  compare with a relative tolerance around $10^{-6}$.

## Exercises

Save your work to `my-work/A/a4_1_derivatives.md` (or `.py`) and ask for a review.

1. **By hand:** (a) Let $f(x) = x^4 - 3x$. Find $f'(x)$, $f''(x)$ and $f'''(x)$ with the rules table.
   At $x = 1$ with $h = 0.1$, compute the forward and the central difference, their errors, and the
   errors predicted by $\frac{f''}{2}h$ and $\frac{f'''}{6}h^2$. One prediction is exact and one is
   not: which, and why? (Hint: write out Taylor's formula for a degree-4 polynomial in full.) (b) Let $g(x_1, x_2) = x_1^2x_2 + 3x_2 - x_1$. Find both partial
   derivatives as formulas and evaluate them at $(2, -1)$. Check $\partial g/\partial x_1$ with a central
   difference, $h = 0.1$.
2. **Code:** The second derivative can be estimated with $\frac{f(x + h) - 2f(x) + f(x - h)}{h^2}$. (a) Use
   Taylor's formula to show that its truncation error is proportional to $h^2$, and argue that its
   rounding error is about $\varepsilon\lvert f\rvert / h^2$. (b) Write `second_diff(f, x, h)`, sweep $h$
   from $10^{-1}$ to $10^{-8}$ for $f(x) = x^4$ at $x = 1$ (written with multiplications), and plot the
   error on log-log axes. (c) Predict the best $h$ from your two error terms and compare it with the
   sweep. Why does the second derivative need a *larger* step than the first?
3. **Think:** A teammate writes a custom loss function, derives its gradient by hand, and adds a
   test that compares it with a numerical gradient. The model trains in float32, so the test also
   runs in float32, with `h = 1e-7` and `rtol = 1e-7`. The test fails on about half the parameters,
   and the teammate concludes the hand-derived gradient is wrong. What do you tell them, using this
   lesson's error terms? How would you rewrite the test so that a failure really means a bug?

## Takeaways

1. A derivative is a local exchange rate: $f(x + h) \approx f(x) + f'(x)\,h$ for small $h$.
2. A partial derivative nudges one input and freezes the rest; for the squared error
   $(wx + b - y)^2$ they are $2rx$ and $2r$, proportional to the residual $r$.
3. Forward differences have error $\propto h$; central differences $\propto h^2$. Always check with
   central differences.
4. Rounding adds an error $\approx \varepsilon\lvert f\rvert/h$ that grows as $h$ shrinks, so the total
   error is U-shaped: in float64 use $h \approx 10^{-6}$ for central differences, never $10^{-12}$.
5. Finite differences are for checking derivatives, not for training: they cost two evaluations per
   parameter.

**What builds on this:** [A4.2](a4_2_gradients_jacobians.md) (all the partial derivatives at once:
the gradient, and a gradient checker); [A4.3](a4_3_chain_rule_graphs.md) (derivatives of composite
functions, and backpropagation); [A4.4](a4_4_gradient_descent.md) (using the derivative to walk
downhill); [C1.3](../C/c1_3_autograd_engine.md) (your autograd engine, tested with these checks).
