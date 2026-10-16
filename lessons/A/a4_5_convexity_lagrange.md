# A4.5 Convexity, constrained optimization and Lagrange multipliers

**Depth** · Part A, week 4 (course week 04) · ~1.5 h · Lab:
[`labs/A/a4_5_convexity_lagrange.py`](../../labs/A/a4_5_convexity_lagrange.py)

## Why it matters

Gradient descent ([A4.4](a4_4_gradient_descent.md)) finds *a* point where the gradient is zero. Whether
that point is *the* best answer depends on the shape of the loss. For a **convex** loss (linear and
logistic regression, ridge, SVMs) every valley is the same valley: any minimum is the global one, and
two runs from different starts agree. For a non-convex loss (every neural network) the starting point
and the random seed change the answer. Separately, many problems come with **constraints**: weights
with a bounded size, probabilities that sum to 1, a budget. **Lagrange multipliers** turn a constrained
problem into equations you can solve, and the multiplier itself has a meaning: the price of the
constraint. This is a Depth topic; you need it for SVMs in [B2.5](../B/b2_5_svm.md), and it explains
why L2 regularization and a weight-size limit are the same thing.

> **In this lesson only**, $\lambda$ is a Lagrange multiplier and $\mu$ is the multiplier of an
> inequality constraint. Neither is the regularization strength (until the last part of "The math",
> which shows that they turn out to be the same thing) or an eigenvalue.

## Prerequisite check

- Can you compute a gradient and say why it is perpendicular to the contour lines? If not, revisit
  [A4.2](a4_2_gradients_jacobians.md).
- Can you run gradient descent and explain the role of the Hessian $\frac{2}{n}\mathbf{X}^\top\mathbf{X}$? If not,
  revisit [A4.4](a4_4_gradient_descent.md).
- Can you say what positive semi-definite means for a symmetric matrix, in terms of its eigenvalues?
  If not, revisit [A3.3](a3_3_covariance_matrices.md).
- Can you compute a second derivative $f''$? If not, revisit [A4.1](a4_1_derivatives.md).

## Intuition

### Bowls versus egg cartons

A **convex** function is shaped like a bowl, possibly a stretched or flat-bottomed one, but never with
a second dip. The test is geometric: pick any two points on the graph and draw the straight segment
(the **chord**) between them; for a convex function the chord never goes below the graph. A function
shaped like an egg carton fails: a chord across a bump passes under it. The payoff of convexity is a
guarantee: walk downhill from *anywhere* and you reach the lowest point, because there is no other dip
to get stuck in. For a non-convex function, where you start decides which dip you end in (the key plot's
middle panel).

### Constraints: the lowest point on a trail

You are not allowed to roam the whole hillside; you must stay on a marked trail. Where is the lowest
point *on the trail*? Walk along it watching the contour lines (lines of equal height) it crosses.
While the trail crosses contour lines, you can still go lower by continuing. At the lowest point the
trail stops crossing and just *touches* a contour line, running along it. Two curves that touch run in
the same direction there, so their perpendiculars are parallel. The perpendicular to a contour line is
the gradient of the height $\nabla f$ (A4.2); the perpendicular to the trail is the gradient of the
function $h$ that defines the trail ($h(\mathbf{x}) = c$). So at the constrained optimum

$$\nabla f = \lambda\,\nabla h$$

for some number $\lambda$, the **Lagrange multiplier**. That is the whole method: these equations plus
"stay on the trail" pin down the answer.

### The multiplier is a price

$\lambda$ also answers a practical question: if the constraint were loosened by one unit (the trail moved
a little, $c \to c + 1$), by how much would the best value change? About $\lambda$. Economists call it a
**shadow price**.

> **Bridge:** you are asked to minimize p99 latency subject to a cloud budget of \$10,000 a month. The
> multiplier on the budget is the answer to "how many milliseconds would one more dollar buy?" If it
> is near zero, the budget is not what limits you (the constraint is inactive); if it is large, the
> budget is the bottleneck. **Where it breaks:** the multiplier is a local rate, valid for small changes:
> it does not tell you what a second \$10,000 buys, and real cost curves have steps (instance sizes)
> where no derivative exists.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $f$ | the function to minimize (the objective) | $\mathbb{R}^n \to \mathbb{R}$ |
| $\mathbf{x}$, $\mathbf{y}$ | points | `(n,)` |
| $t$ | mixing weight between two points | scalar in $[0, 1]$ |
| $\mathbf{H}$ | Hessian of $f$ (A4.4) | `(n, n)` |
| $h(\mathbf{x}) = c$ | an equality constraint | $h:\mathbb{R}^n \to \mathbb{R}$, $c$ scalar |
| $g(\mathbf{x}) \le 0$ | an inequality constraint | $g:\mathbb{R}^n \to \mathbb{R}$ |
| $\lambda$, $\mu$ | multipliers of an equality and an inequality constraint (this lesson only) | scalar |
| $\mathcal{F}(\mathbf{x}, \lambda)$ | the **Lagrangian** (written $\mathcal{F}$ so it is not confused with the loss $\mathcal{L}$) | scalar |
| $f^\star(c)$ | the best value of $f$ subject to $h(\mathbf{x}) = c$ | scalar |

**Convexity, defined.** A point between $\mathbf{x}$ and $\mathbf{y}$ is $t\mathbf{x} + (1 - t)\mathbf{y}$ for some
$t \in [0, 1]$ ($t = 1$ gives $\mathbf{x}$, $t = 0$ gives $\mathbf{y}$). The chord at that point has height
$t f(\mathbf{x}) + (1 - t) f(\mathbf{y})$. $f$ is **convex** when the graph is never above the chord:

$$f\big(t\mathbf{x} + (1 - t)\mathbf{y}\big) \le t\,f(\mathbf{x}) + (1 - t)\,f(\mathbf{y}) \quad\text{for all } \mathbf{x}, \mathbf{y} \text{ and } t \in [0, 1].$$

It is **strictly convex** if the inequality is strict whenever $\mathbf{x} \ne \mathbf{y}$ and $0 < t < 1$ (no flat
stretches), which guarantees at most one minimizer.

**Tests you can apply.** (Stated, not proved.)

- One variable, twice differentiable: convex exactly when $f''(x) \ge 0$ everywhere (the slope never
  decreases). $x^2$, $e^x$ and $-\log x$ (for $x > 0$) pass; $x^4 - 3x^2 + x$ fails where
  $12x^2 - 6 < 0$, i.e. for $\lvert x\rvert < 0.71$.
- Several variables: convex exactly when the Hessian is positive semi-definite everywhere (all its
  eigenvalues $\ge 0$): curvature is non-negative in every direction. The MSE's Hessian
  $\frac{2}{n}\mathbf{X}^\top\mathbf{X}$ is positive semi-definite (A4.4), so **the MSE is convex**. If some column of
  $\mathbf{X}$ duplicates another, an eigenvalue is 0: the MSE is convex but not strictly, with a flat valley
  of equally good weights (A2.3's rank deficiency again).
- Building blocks: a sum of convex functions is convex; a positive multiple of a convex function is
  convex; the maximum of convex functions is convex ($\lvert x\rvert = \max(x, -x)$, ReLU $= \max(0, x)$);
  a convex function of a linear map ($\mathbf{w} \mapsto f(\mathbf{X}\mathbf{w})$) is convex. These give convexity of
  ridge (MSE plus $\lambda\lVert\mathbf{w}\rVert^2$) and of logistic regression's log loss (stated here, checked in
  [B2.1](../B/b2_1_logistic_regression.md)).

**Why a local minimum of a convex function is global.** Suppose $\mathbf{x}$ is a local minimum (nothing
nearby is lower) but some $\mathbf{y}$ has $f(\mathbf{y}) < f(\mathbf{x})$. Points $t\mathbf{x} + (1 - t)\mathbf{y}$ with $t$ close
to 1 are close to $\mathbf{x}$, and by convexity
$f \le t f(\mathbf{x}) + (1 - t) f(\mathbf{y}) < t f(\mathbf{x}) + (1 - t) f(\mathbf{x}) = f(\mathbf{x})$: lower than $f(\mathbf{x})$, arbitrarily
close to it. That contradicts "nothing nearby is lower", so no such $\mathbf{y}$ exists. For a differentiable
convex function, any point with $\nabla f = \mathbf{0}$ is therefore a global minimum.

**Equality constraints: the Lagrange conditions.** Minimize $f(\mathbf{x})$ subject to $h(\mathbf{x}) = c$. The
trail argument in symbols: a small step $\boldsymbol{\delta}$ keeps you on the constraint (to first order) when
$\nabla h \cdot \boldsymbol{\delta} = 0$, i.e. $\boldsymbol{\delta}$ is perpendicular to $\nabla h$. At a constrained minimum no
such step may lower $f$, so $\nabla f \cdot \boldsymbol{\delta} = 0$ for every $\boldsymbol{\delta}$ perpendicular to $\nabla h$ (if
it were positive for some $\boldsymbol{\delta}$, it would be negative for $-\boldsymbol{\delta}$). A vector perpendicular to
everything that is perpendicular to $\nabla h$ points along $\nabla h$. Hence

$$\nabla f(\mathbf{x}) = \lambda\,\nabla h(\mathbf{x}), \qquad h(\mathbf{x}) = c.$$

That is $n + 1$ equations for $n + 1$ unknowns ($\mathbf{x}$ and $\lambda$). The same equations come from setting
every partial derivative of the **Lagrangian** $\mathcal{F}(\mathbf{x}, \lambda) = f(\mathbf{x}) - \lambda\,(h(\mathbf{x}) - c)$ to zero:
$\partial\mathcal{F}/\partial\mathbf{x} = 0$ gives the first, $\partial\mathcal{F}/\partial\lambda = 0$ gives the constraint. With this sign
convention $\lambda = \frac{df^\star}{dc}$, the shadow price (stated; the worked example checks it). Books and
libraries differ on the sign in front of $\lambda$; check the convention before reading a library's
multiplier.

The conditions are necessary, not sufficient: they also hold at constrained maxima and saddle points.
When $f$ is convex and the constraint is linear ($h(\mathbf{x}) = \mathbf{a}^\top\mathbf{x}$), any solution is the minimum.

**Inequality constraints: KKT.** Minimize $f(\mathbf{x})$ subject to $g(\mathbf{x}) \le 0$. Either the constraint
is **inactive** at the answer ($g < 0$: the unconstrained minimum is already allowed, and the constraint
changes nothing) or **active** ($g = 0$: you are pressed against the wall, and it behaves like an
equality). The **Karush–Kuhn–Tucker (KKT) conditions** (stated, not proved) cover both cases at once:

$$\nabla f + \mu\,\nabla g = \mathbf{0}, \qquad g(\mathbf{x}) \le 0, \qquad \mu \ge 0, \qquad \mu\,g(\mathbf{x}) = 0.$$

The last one, **complementary slackness**, says $\mu = 0$ unless the constraint is active. $\mu \ge 0$ says
the wall can only push you back in, never pull you out: at an active constraint $-\nabla f$ (downhill)
points *out* of the allowed region, along $\nabla g$. In an SVM ([B2.5](../B/b2_5_svm.md)) each training
point gives one inequality constraint, and the points whose constraints are active ($\mu > 0$) are the
**support vectors**; all the others have $\mu = 0$ and do not affect the answer.

**Constraint versus penalty: where ridge comes from.** Minimizing the MSE subject to
$\lVert\mathbf{w}\rVert^2 \le r^2$ has KKT condition $\nabla\mathcal{L} + \mu\cdot 2\mathbf{w} = \mathbf{0}$ (since $\nabla\lVert\mathbf{w}\rVert^2 = 2\mathbf{w}$). That is
exactly the condition for the minimum of the *unconstrained* function $\mathcal{L}(\mathbf{w}) + \mu\lVert\mathbf{w}\rVert^2$, which is
ridge regression ([B2.2](../B/b2_2_regularization.md)) with $\lambda = \mu$. A limit on the weights' size and
a penalty on it are the same problem; the multiplier converts one into the other. Exercise 2 checks it.

## Worked example by hand

**1. The closest point to the origin on the line $x + y = 1$.** Minimize $f = x^2 + y^2$ subject to
$h = x + y = 1$. Gradients: $\nabla f = (2x, 2y)$, $\nabla h = (1, 1)$. The condition $\nabla f = \lambda\nabla h$ gives
$2x = \lambda$ and $2y = \lambda$, so $x = y = \lambda/2$. Substituting into the constraint: $\lambda/2 + \lambda/2 = 1$, so
$\lambda = 1$, $(x, y) = (0.5, 0.5)$, $f^\star = 0.5$.

Shadow price: for a general $c$ the same steps give $x = y = c/2$ and $f^\star(c) = c^2/2$, so
$\frac{df^\star}{dc} = c = 1 = \lambda$ ✓. Numerically, moving the line to $c = 1.01$ gives $f^\star = 0.51005$: a rise
of $0.01005$ for $0.01$ of loosening, a rate of $1.005 \approx \lambda$.

**2. The shortest $\mathbf{w}$ with $\mathbf{a}^\top\mathbf{w} = c$**, for $\mathbf{a} = (1, 2)$, $c = 5$. Minimize
$\lVert\mathbf{w}\rVert^2$: $\nabla f = 2\mathbf{w} = \lambda\mathbf{a}$, so $\mathbf{w} = \frac{\lambda}{2}\mathbf{a}$, the solution points along $\mathbf{a}$.
The constraint: $\frac{\lambda}{2}\mathbf{a}^\top\mathbf{a} = c$, so $\lambda = \frac{2c}{\lVert\mathbf{a}\rVert^2} = \frac{10}{5} = 2$ and
$\mathbf{w} = \frac{c\,\mathbf{a}}{\lVert\mathbf{a}\rVert^2} = (1, 2)$, with $\lVert\mathbf{w}\rVert^2 = 5$. This is the minimum-norm solution the
pseudo-inverse returns ([A3.2](a3_2_svd.md)) for one equation in two unknowns.

**3. An inequality, active and inactive.** Minimize $(x - 2)^2$ subject to $x \le b$, written
$g(x) = x - b \le 0$, so $\nabla g = 1$. KKT stationarity: $2(x - 2) + \mu = 0$.

- $b = 1$: the unconstrained minimum $x = 2$ is not allowed, so the constraint is active, $x = 1$, and
  $\mu = -2(1 - 2) = 2 \ge 0$ ✓. Loosening the bound to $1.01$ lowers the best value from 1 to $0.9801$, a
  rate of about 2.
- $b = 3$: $x = 2$ is allowed, the constraint is inactive, $\mu = 0$, and stationarity is just
  $2(x - 2) = 0$ ✓.

```text
== worked example: closest point on a line ==
solution (x, y) = [0.5, 0.5]   lambda = 1.0   f* = 0.5
grad f = [1.0, 1.0]   grad h = [1.0, 1.0]   grad f = lambda * grad h: True
c = 1.01: f* = 0.51005   (f*(c) - f*(1)) / (c - 1) = 1.0050   lambda = 1.0
c = 1.1: f* = 0.60500   (f*(c) - f*(1)) / (c - 1) = 1.0500   lambda = 1.0

== worked example: minimum-norm w ==
w = c a / |a|^2 = [1.0, 2.0]   |w|^2 = 5.0   lambda = 2c/|a|^2 = 2.0
np.linalg.pinv gives the same w: [1.0, 2.0]

== worked example: inequality constraint ==
min (x - 2)^2 s.t. x <= 1.0: x* = 1.0   constraint active   multiplier mu = 2.0
min (x - 2)^2 s.t. x <= 3.0: x* = 2.0   constraint inactive   multiplier mu = 0.0
```

## Key plot

![Three panels. Left: a convex parabola with 41 starting points and every gradient-descent run ending at the single minimum. Middle: the non-convex function x to the fourth minus 3 x squared plus x, with starting points left of about 0.17 ending at the global minimum near minus 1.30 and the rest at a local minimum near 1.13. Right: circular contour lines of x squared plus y squared, the line x plus y equals 1 touching the smallest reachable contour at (0.5, 0.5), with the gradients of f and of the constraint drawn parallel there](img/a4_5_convex_nonconvex_lagrange.png)

Left and middle: gradient descent run from 41 evenly spaced starts. On the convex function all 41 runs
end at the same point. On the non-convex one, the starts split at the hump near $x = 0.17$ (the dotted
line): those on the left reach the global minimum ($f = -3.51$), those on the right get stuck in the
local one ($f = -1.07$), and nothing in the gradient tells them so. Right: the Lagrange picture. The
line is the constraint, the circles are contours of $f$, and at the optimum the line just touches a
contour, with $\nabla f$ and $\nabla h$ parallel.

## From scratch in NumPy

A chord test samples random pairs of points and random $t$, and counts chords that pass below the
curve. One violation proves non-convexity; finding none proves nothing (you might have missed the
bad region), so treat a 0% as "no counterexample found":

```python
def chord_violations(
    f: Callable[[FloatArray], FloatArray], lo: float, hi: float, n: int, rng: np.random.Generator
) -> float:
    """Share of random chords on [lo, hi] that dip below f. 0 means no counterexample found."""
    x1 = rng.uniform(lo, hi, size=n)  # (n,)
    x2 = rng.uniform(lo, hi, size=n)  # (n,)
    t = rng.uniform(0.0, 1.0, size=n)  # (n,)
    left = f(t * x1 + (1 - t) * x2)  # f at a point between x1 and x2
    right = t * f(x1) + (1 - t) * f(x2)  # the chord at the same point
    return float(np.mean(left > right + 1e-12))  # 1e-12: ignore rounding on straight pieces
```

```text
== from scratch: chord test ==
x^2              chords below the curve:   0.0%   no violation found
|x|              chords below the curve:   0.0%   no violation found
exp(x)           chords below the curve:   0.0%   no violation found
x^4 - 3x^2 + x   chords below the curve:  46.6%   NOT convex
sin(x)           chords below the curve:  49.1%   NOT convex

== from scratch: curvature tests ==
x^4 - 3x^2 + x: f'' = 12x^2 - 6 < 0 on (-0.70, 0.70) -> not convex there
MSE Hessian (2/n) X^T X eigenvalues = [0.4589, 74.0413]  -> all >= 0: convex
with a duplicated column: eigenvalues = [0.0, 0.4637, 146.5367]  -> convex but not strictly (a flat valley)
```

$\lvert x\rvert$ passes even though it has no derivative at 0: convexity does not need smoothness. (The grid
prints $\pm 0.70$ because its points are $0.01$ apart; the exact boundary is $\pm\sqrt{0.5} = \pm 0.707$.)

Gradient descent from 41 starts, run for every start at once as an array:

```text
== from scratch: where gradient descent ends up ==
convex (x - 0.5)^2: distinct endpoints = [0.5]
non-convex: endpoint x = -1.3008  f = -3.5139  reached from starts in [-2.0, 0.1]  (22 of 41)
non-convex: endpoint x = 1.1309  f = -1.0702  reached from starts in [0.2, 2.0]  (19 of 41)
critical points (roots of f') = [-1.3008, 0.1699, 1.1309]  f'' there = [14.3062, -5.6535, 9.3472]
```

The three points where $f' = 0$ are two minima ($f'' > 0$) and a maximum ($f'' < 0$) between them; the
maximum is the watershed that splits the starts.

## With SciPy

`scipy.optimize.minimize(..., method="SLSQP", constraints=[...])` handles equality (`"eq"`, meaning
`fun(x) == 0`) and inequality (`"ineq"`, meaning `fun(x) >= 0`, note the direction) constraints. Its
result carries the multipliers in `res.multipliers` (VERIFY (as of 2026-10) for SciPy 1.18.1: the
attribute is recent). With constraints written as $h(\mathbf{x}) - c = 0$ and $b - x \ge 0$, its signs match
this lesson's: a positive multiplier is the rate at which loosening the constraint improves $f$.

```text
== with scipy ==
line:    x = [0.5, 0.5]   f* = 0.5000   multiplier = 1.0000
min-norm: w = [1.0, 2.0]   f* = 5.0000   multiplier = 2.0000
x <= 1.0:  x = [1.0]   multiplier = [2.0]
x <= 3.0:  x = [2.0]   multiplier = [0.0]
SLSQP matches every hand solution (atol 1e-6): OK
BFGS on the non-convex function from 1.5 -> 1.1309, from -1.5 -> -1.3008: a library optimizer is still local
```

The last line matters: a better optimizer does not fix non-convexity. BFGS, like gradient descent,
finds the minimum of the valley it starts in.

## In practice

- **When to use it:** check convexity before trusting that a solver's answer is the best one, and before
  comparing two runs (two runs of a convex fit should agree; two runs of a neural network will not). Use
  Lagrange or KKT conditions to solve small constrained problems by hand, to read an SVM, and to
  understand regularization; in code, use a constrained solver (SLSQP, `trust-constr`) or reparametrize
  so the constraint holds automatically (softmax for probabilities that sum to 1,
  [C1.4](../C/c1_4_stable_softmax_cross_entropy.md); `exp` for a positive scale).
- **Failure modes:**
  - *Treating a stationary point as a minimum:* $\nabla f = \mathbf{0}$ also holds at maxima and saddles, and
    the Lagrange conditions also hold at constrained maxima. You notice it when the value is worse than
    at a nearby feasible point; check second-order information or compare several starts.
  - *Assuming convexity:* "the loss went down and stopped" says nothing about global optimality for a
    non-convex model. You notice different answers from different seeds; that is expected, so report
    the spread across seeds.
  - *Convexity lost by composition:* a convex function of a *non-linear* map need not be convex (MSE of a
    neural network's output is not convex in the weights).
  - *Sign conventions:* libraries define the Lagrangian and the direction of inequalities differently;
    a "negative multiplier" may be a convention, not an error. Check on a problem you solved by hand.
  - *Flat directions:* convex but not strictly convex (duplicated features) means many equally good
    answers; different solvers return different ones. You notice unstable coefficients with an unchanged
    loss; add a small ridge penalty or remove the redundancy.
- **Defaults that work:** for convex problems, trust a good solver and verify with the optimality
  conditions; for non-convex ones, run several seeds and report the spread; prefer a penalty (ridge)
  over an explicit norm constraint in practice, since it is unconstrained and the two are equivalent.

## Exercises

Save your work to `my-work/A/a4_5_convexity_lagrange.md` (or `.py`) and ask for a review.

1. **By hand:** (a) Minimize $x^2 + 2y^2$ subject to $x + y = 3$ with a Lagrange multiplier. Give the
   point, the value $f^\star$ and $\lambda$. Then find $f^\star(c)$ for a general $x + y = c$ and check that
   $\frac{df^\star}{dc} = \lambda$ at $c = 3$. (b) Which of these are convex on the stated domain? Justify each in
   one line with $f''$, a building-block rule, or a chord that dips below:
   $e^{-x}$; $x^3$; $\max(0, x)$; $-\log x$ for $x > 0$; $x^4 - x^2$; $\lvert x - 1\rvert + x^2$.
2. **Code:** Generate $\mathbf{X}$ `(60, 3)` with standard normal entries and
   $\mathbf{y} = \mathbf{X}(3, -2, 1) + $ noise with standard deviation 0.5 (`SEED = 0`, $\mathbf{X}$ first). Let
   $\mathcal{L}(\mathbf{w}) = \frac1n\lVert\mathbf{X}\mathbf{w} - \mathbf{y}\rVert^2$ and $r$ = half the length of the least-squares $\mathbf{w}$.
   (a) Write **projected gradient descent** for $\min\mathcal{L}(\mathbf{w})$ subject to $\lVert\mathbf{w}\rVert \le r$: after each
   gradient step, if $\mathbf{w}$ is outside the ball, scale it back onto the sphere. (b) Solve the same
   problem with SLSQP and read its multiplier $\mu$. (c) Verify the KKT stationarity condition at your
   answer, and show that ridge regression with $\lambda = \mu$ (by `np.linalg.solve` and by scikit-learn's
   `Ridge`) gives the same $\mathbf{w}$.
3. **Think:** A credit-risk team retrains a model every week and must explain each prediction to
   auditors. One engineer proposes replacing their logistic regression (a convex fit) with a small
   neural network that scores 1% better on validation. Using convexity, give two concrete operational
   problems the switch could cause beyond explainability, and say how you would measure whether they
   matter before deciding.

## Takeaways

1. Convex means every chord lies on or above the graph; equivalently $f'' \ge 0$, or a positive
   semi-definite Hessian. Then every local minimum is global and gradient descent finds it from any start.
2. MSE, ridge and logistic regression are convex; neural networks are not, so their answers depend on the
   start and the seed, and a better optimizer does not change that.
3. At a constrained optimum the objective's contour touches the constraint: $\nabla f = \lambda\nabla h$, solved
   together with $h(\mathbf{x}) = c$.
4. The multiplier is a shadow price, $\lambda = df^\star/dc$; for inequalities (KKT) it is $\ge 0$ and is 0 when
   the constraint is inactive.
5. A norm constraint on the weights and an L2 penalty are the same problem, with the multiplier as the
   penalty strength; support vectors are an SVM's active constraints.

**What builds on this:** [B2.2](../B/b2_2_regularization.md) (ridge and lasso as constraints);
[B2.5](../B/b2_5_svm.md) (the SVM's margin, KKT and support vectors);
[C1.4](../C/c1_4_stable_softmax_cross_entropy.md) (softmax as a way to satisfy "probabilities sum to 1");
[C3.1](../C/c3_1_optimizers.md) (optimizing non-convex losses in practice).
