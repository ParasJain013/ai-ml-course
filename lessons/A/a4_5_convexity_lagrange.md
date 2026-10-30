# A4.5 Convexity, constrained optimization and Lagrange multipliers

**Depth** · Part A, week 4 (course week 04) · ~1.5 h · Lab:
[`labs/A/a4_5_convexity_lagrange.py`](../../labs/A/a4_5_convexity_lagrange.py)

## Why it matters

Two questions decide how much you can trust an optimizer's answer. First: *is there only one valley?*
For a **convex** loss (linear regression, ridge, logistic regression, SVMs) every local minimum is the
global one, so any correct optimizer, from any start, gives the same model, and a different answer means
a bug. For a non-convex loss (every neural network) the answer depends on the start, the seed and the
learning rate, and comparing two runs needs care. Second: *what if the parameters must obey a rule*
(probabilities sum to 1, weights stay non-negative, a budget is fixed)? **Lagrange multipliers** turn
such a constrained problem into equations you can solve, and they are the machinery inside support
vector machines ([B2.5](../B/b2_5_svm.md)). This is a Depth topic: skip it now and you must come back
before B2.5.

## Prerequisite check

- Can you compute a gradient and a Hessian, and say what the Hessian of the MSE is? If not, revisit
  [A4.2](a4_2_gradients_jacobians.md).
- Can you find the eigenvalues of a symmetric 2×2 matrix from its trace and determinant? If not,
  revisit [A3.1](a3_1_eigenvalues_eigenvectors.md).
- Do you know what "positive semi-definite" means for a matrix, and why $\mathbf{w}^\top\mathbf{X}^\top\mathbf{X}\mathbf{w} \ge 0$? If
  not, revisit [A3.3](a3_3_covariance_matrices.md).
- Can you run gradient descent and say what the learning rate does? If not, revisit
  [A4.4](a4_4_gradient_descent.md).

## Intuition

### Convex: a bowl with no false bottoms

A function is **convex** if, whenever you pick two points on its graph and stretch a string between them,
the graph never rises above the string. A bowl passes; a landscape with two dips fails, because the string
between the dips passes *below* the hump in the middle. In a convex landscape, wherever you stand, going
downhill leads to the one lowest region, so gradient descent cannot get stuck in a worse valley. In a
non-convex one, which valley you reach depends on where you start (the key plot).

### Constraints: walking along a fence

Now you must stay on a path, say a straight fence across a hillside, and you want the lowest point on it.
Walk along the fence. While the fence crosses the contour lines, walking one way goes downhill, so keep
going. You stop where the fence just touches a contour line without crossing it: there, the fence is
*tangent* to the contour. At that point the hill's gradient (perpendicular to the contour, A4.2) is also
perpendicular to the fence. The fence's own "gradient" is perpendicular to it too, so the two gradients
are parallel: $\nabla f = \lambda\nabla g$ for some number $\lambda$. That number, the **Lagrange multiplier**,
turns out to be the exchange rate between the constraint and the objective.

> **Bridge:** a Lagrange multiplier is a shadow price. If you minimize p99 latency subject to spending
> exactly \$10,000 a month on servers, the multiplier at the optimum says how much latency one more dollar
> of budget would buy: the marginal value of relaxing the constraint. **Where it breaks:** the multiplier is
> a derivative, so it is valid only for small changes around the optimum, and only when everything is
> smooth. Real budgets buy whole instances, so the true trade-off comes in steps.

## The math

> **In this lesson only,** $\lambda$ is the Lagrange multiplier of an equality constraint and $\mu$ the
> multiplier of an inequality constraint; neither is the regularization strength or an eigenvalue. The
> Lagrangian is written $F$, to keep $\mathcal{L}$ for losses.

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $f$ | the objective to minimize | `(p,)` → scalar |
| $\mathbf{x}$, $\mathbf{y}$ | two points in the domain | `(p,)` |
| $t$ | a mixing weight in $[0, 1]$ | scalar |
| $\mathbf{H}$ | Hessian of $f$ | `(p, p)` |
| $g$, $c$ | constraint function and its level: $g(\mathbf{x}) = c$ or $g(\mathbf{x}) \le c$ | `(p,)` → scalar; scalar |
| $\lambda$ | Lagrange multiplier of an equality constraint | scalar |
| $\mu$ | multiplier of an inequality constraint, $\mu \ge 0$ | scalar |
| $F$ | the **Lagrangian** | scalar |
| $f^\star(c)$ | the optimal value of $f$ as a function of the constraint level | scalar |
| $\mathbf{p}$, $\mathbf{a}$ | a target point and a constraint normal in the projection example | `(p,)` |

### Convex functions

$f$ is **convex** if for every pair of points $\mathbf{x}, \mathbf{y}$ and every $t \in [0, 1]$,

$$f\big(t\mathbf{x} + (1 - t)\mathbf{y}\big) \le t\,f(\mathbf{x}) + (1 - t)\,f(\mathbf{y}).$$

The left side is $f$ at a point on the straight line between $\mathbf{x}$ and $\mathbf{y}$ ($t = 1$ is $\mathbf{x}$, $t = 0$ is
$\mathbf{y}$); the right side is the same fraction of the way along the chord. "The graph lies below every
chord." **Strictly convex** means $<$ whenever $\mathbf{x} \ne \mathbf{y}$ and $0 < t < 1$.

**The test you will actually use (stated, not proved).** For a twice-differentiable $f$ of one variable,
$f$ is convex if and only if $f''(x) \ge 0$ everywhere: the slope never decreases. For several variables,
if and only if the Hessian is positive semi-definite everywhere: all its eigenvalues are $\ge 0$ (A3.3).
Read it with A4.2's second-order picture $f(\mathbf{x} + \boldsymbol{\delta}) \approx f(\mathbf{x}) + \nabla f\cdot\boldsymbol{\delta} + \frac12\boldsymbol{\delta}^\top\mathbf{H}\boldsymbol{\delta}$:
a PSD Hessian means the curvature term is never negative in any direction.

**Why local means global.** Suppose $\mathbf{x}$ is a local minimum of a convex $f$, but some $\mathbf{y}$ has
$f(\mathbf{y}) < f(\mathbf{x})$. Points on the segment from $\mathbf{x}$ toward $\mathbf{y}$ satisfy
$f(t\mathbf{x} + (1 - t)\mathbf{y}) \le t f(\mathbf{x}) + (1 - t) f(\mathbf{y}) < f(\mathbf{x})$ for every $t < 1$, including $t$
very close to 1, i.e. points as close to $\mathbf{x}$ as you like. So $\mathbf{x}$ was not a local minimum after all.
Contradiction: no lower point exists. Also, where $\nabla f = \mathbf{0}$ on a convex function you are at a global
minimum, and a strictly convex function has at most one minimizer.

**Which ML losses are convex.**

- *MSE of linear regression:* the Hessian $\frac{2}{n}\mathbf{X}^\top\mathbf{X}$ is PSD because
  $\mathbf{v}^\top\mathbf{X}^\top\mathbf{X}\mathbf{v} = \lVert\mathbf{X}\mathbf{v}\rVert^2 \ge 0$ for every $\mathbf{v}$. Convex, so A4.4's gradient descent had
  to land on the normal-equation answer.
- *Ridge:* adds a penalty $\alpha\lVert\mathbf{w}\rVert^2$ (the regularization strength, written $\lambda$ elsewhere in the
  course), so the Hessian gains $2\alpha\mathbf{I}$ and all eigenvalues are $\ge 2\alpha > 0$: strictly
  convex, one answer even when features are collinear.
- *Sums* of convex functions are convex; so are logistic regression's loss and the SVM's hinge loss
  (stated here, shown in [B2.1](../B/b2_1_logistic_regression.md) and [B2.5](../B/b2_5_svm.md)).
- *Neural networks are not.* Swapping two hidden units (with their weights) gives the same function and
  the same loss at a different point, so there are many separate minima, with humps between them.

A point where $\nabla f = \mathbf{0}$ but the Hessian has eigenvalues of both signs is a **saddle point**: a minimum
along some directions and a maximum along others. Non-convex losses have many of them.

### Equality constraints: Lagrange multipliers

**Problem:** minimize $f(\mathbf{x})$ subject to $g(\mathbf{x}) = c$.

**Condition.** At a constrained optimum $\mathbf{x}^\star$, moving along the constraint surface must not change $f$
to first order. Directions $\mathbf{u}$ along the surface are those with $\nabla g\cdot\mathbf{u} = 0$ (they keep $g$
constant, A4.2), so we need $\nabla f\cdot\mathbf{u} = 0$ for every such $\mathbf{u}$: $\nabla f$ is perpendicular to the
surface. So is $\nabla g$. In $p$ dimensions the surface has only one perpendicular direction (stated), so the two
are parallel:

$$\nabla f(\mathbf{x}^\star) = \lambda\,\nabla g(\mathbf{x}^\star), \qquad g(\mathbf{x}^\star) = c.$$

That is $p + 1$ equations for $p + 1$ unknowns ($\mathbf{x}^\star$ and $\lambda$). A compact way to write them: define
the Lagrangian

$$F(\mathbf{x}, \lambda) = f(\mathbf{x}) - \lambda\,\big(g(\mathbf{x}) - c\big).$$

Setting $\nabla_{\mathbf{x}}F = \nabla f - \lambda\nabla g = \mathbf{0}$ gives the parallel-gradients condition, and setting
$\partial F/\partial\lambda = -(g(\mathbf{x}) - c) = 0$ gives back the constraint. A constrained problem in $p$ unknowns
becomes an unconstrained "find where all derivatives are zero" problem in $p + 1$ unknowns. (The solution is
a stationary point of $F$, typically a saddle of $F$, not a minimum; do not run gradient descent on $F$.)

**What $\lambda$ means (stated, checked in the worked example).** If $f^\star(c)$ is the optimal value when the
constraint level is $c$, then

$$\frac{df^\star}{dc} = \lambda.$$

Relax the constraint by a small $\Delta c$ and the best achievable $f$ changes by about $\lambda\,\Delta c$.

### Inequality constraints: the KKT conditions

**Problem:** minimize $f(\mathbf{x})$ subject to $g(\mathbf{x}) \le c$. There are only two possibilities.

- *Inactive:* the unconstrained minimum already satisfies $g < c$. The constraint does not matter:
  $\nabla f = \mathbf{0}$, multiplier $\mu = 0$.
- *Active:* the unconstrained minimum is outside the allowed region, so the answer sits on the boundary
  $g = c$ and behaves like an equality constraint. Because we may move *into* the region freely, $f$ must
  not decrease in that direction, which fixes the sign of the multiplier.

The **KKT conditions** (Karush–Kuhn–Tucker) cover both cases at once. With $F = f + \mu\,(g - c)$:

$$\nabla f + \mu\nabla g = \mathbf{0}, \qquad g(\mathbf{x}) \le c, \qquad \mu \ge 0, \qquad \mu\,\big(g(\mathbf{x}) - c\big) = 0.$$

The last one, **complementary slackness**, says: either the constraint is tight ($g = c$) or its multiplier is
zero. For a convex $f$ and a convex $g$, a point satisfying all four is the global constrained minimum
(stated). Here $\mu = -df^\star/dc \ge 0$: loosening an inequality can only help, so $\mu$ measures how much.

This is exactly the structure of the SVM in [B2.5](../B/b2_5_svm.md): one inequality constraint per
training point ("be on the correct side of the margin"), one multiplier per point, and complementary
slackness says only the points *on* the margin, the **support vectors**, get a non-zero multiplier and shape
the answer.

## Worked example by hand

**1. The chord test.** $f(x) = x^2$ between $-1$ and $1$, $t = \frac12$: $f(0) = 0$; chord midpoint
$\frac12(1) + \frac12(1) = 1$. Below ✓. Now $f(x) = x^4 - 3x^2 + x$: $f(-1) = 1 - 3 - 1 = -3$, $f(1) = 1 - 3 + 1 = -1$,
chord midpoint $-2$, but $f(0) = 0$: the graph is *above* the chord, so it is not convex. The second
derivative agrees: $f''(x) = 12x^2 - 6 < 0$ for $\lvert x\rvert < 1/\sqrt2 \approx 0.707$.

**2. The Hessian test on two quadratics.** $x^2 + xy + y^2$ has Hessian $\begin{pmatrix}2&1\\1&2\end{pmatrix}$ (differentiate
twice: $\partial^2/\partial x^2 = 2$, $\partial^2/\partial x\partial y = 1$). Trace 4, determinant 3, eigenvalues 3 and 1: convex.
$x^2 + 3xy + y^2$ has Hessian $\begin{pmatrix}2&3\\3&2\end{pmatrix}$: trace 4, determinant $4 - 9 = -5$, so
$\lambda = \frac{4 \pm \sqrt{16 + 20}}{2} = \frac{4 \pm 6}{2}$: 5 and $-1$. Not convex. The origin is a saddle: along $(t, -t)$ the
function is $t^2 - 3t^2 + t^2 = -t^2$, going *down*, while along $(t, t)$ it is $5t^2$, going up.

**3. A Lagrange multiplier.** Minimize $f = x^2 + y^2$ subject to $x + y = 1$. $\nabla f = (2x, 2y)$,
$\nabla g = (1, 1)$, so $2x = \lambda$ and $2y = \lambda$: $x = y = \lambda/2$. The constraint gives $\lambda/2 + \lambda/2 = 1$, so
$\lambda = 1$, $x = y = 0.5$, $f^\star = 0.5$. Check the meaning of $\lambda$: with $x + y = c$ the same steps give
$x = y = c/2$ and $f^\star(c) = c^2/2$, whose derivative at $c = 1$ is $c = 1 = \lambda$ ✓.

**4. An inequality constraint.** Minimize $f = (x - 2)^2 + (y - 2)^2$ subject to $x + y \le 1$. The
unconstrained minimum $(2, 2)$ has $x + y = 4 > 1$: infeasible, so the constraint is active and $x + y = 1$.
Stationarity: $(2(x - 2), 2(y - 2)) + \mu(1, 1) = \mathbf{0}$, so $x - 2 = y - 2$, $x = y = 0.5$, and
$\mu = -2(0.5 - 2) = 3 \ge 0$ ✓. With $x + y \le 5$ instead, $(2, 2)$ is feasible, the answer is $(2, 2)$ and
$\mu = 0$. In both cases $\mu \times \text{slack} = 0$.

```text
== worked example: chord test ==
x^2             between -1 and 1: f(midpoint) = +0.0000   chord midpoint = +1.0000   below the chord
x^4 - 3x^2 + x  between -1 and 1: f(midpoint) = +0.0000   chord midpoint = -2.0000   ABOVE the chord: not convex
x^2                      worst chord gap on [-2, 2] = +0.0000   -> convex
exp(x)                   worst chord gap on [-2, 2] = +0.0000   -> convex
x^3                      worst chord gap on [-2, 2] = +3.9294   -> NOT convex
(x - 0.5)^2 + 0.25 x^4   worst chord gap on [-2, 2] = +0.0000   -> convex
x^4 - 3x^2 + x           worst chord gap on [-2, 2] = +2.2034   -> NOT convex

== worked example: Hessian test ==
x^2 + xy + y^2   Hessian = [[2.0, 1.0], [1.0, 2.0]]  eigenvalues = [1.0, 3.0]  -> convex
x^2 + 3xy + y^2  Hessian = [[2.0, 3.0], [3.0, 2.0]]  eigenvalues = [-1.0, 5.0]  -> NOT convex (a saddle)
x^2 + 3xy + y^2 along (t, -t): f = 2t^2 - 3t^2 = -t^2, e.g. at t = 1: -1
MSE Hessian (2/n) X^T X for a random X (30, 3): smallest eigenvalue = 2.2799  (>= 0: the MSE is always convex)

== worked example: Lagrange multiplier ==
solution (x, y) = [0.5, 0.5]  f* = 0.5000  grad f = [1.0, 1.0]  grad g = [1.0, 1.0]  lambda = 1.0000
  constraint x + y = 1.1: f* = 0.6050   (f* change) / (c change) = 1.0500  -> lambda = 1.0
  constraint x + y = 1.01: f* = 0.5101   (f* change) / (c change) = 1.0050  -> lambda = 1.0

== worked example: inequality constraint ==
x + y <= 1.0: solution = [0.5, 0.5]  x + y = 1.0  mu = 3.0  (active; mu * slack = 0.0)
x + y <= 5.0: solution = [2.0, 2.0]  x + y = 4.0  mu = 0.0  (inactive; mu * slack = 0.0)
```

The chord test samples 2,000 random chords on $[-2, 2]$; a positive worst gap proves non-convexity, while a
zero gap is only evidence (a sample cannot prove "for every pair"). The ratio $\Delta f^\star/\Delta c$ approaches
$\lambda = 1$ as $\Delta c$ shrinks, like a forward difference (A4.1). For $x^3$ the chord test fails because
$f'' = 6x < 0$ for $x < 0$.

## Key plot

![Two panels. Left: a convex curve with 41 starting points along it; gradient descent from every one ends at the same red star. Right: the non-convex curve x^4 - 3x^2 + x; starts left of the local maximum at x = 0.17 end at the left minimum, starts right of it end at the right minimum](img/a4_5_convex_vs_nonconvex.png)

Each marker is a starting point for gradient descent; its colour and shape say where it ended. On the
convex function all 41 starts reach the same point. On the non-convex one the local maximum at
$x \approx 0.17$ divides them: 22 reach the global minimum ($f = -3.51$), 19 get stuck in the worse one
($f = -1.07$). Nothing in the optimizer's output tells you which kind of minimum you got.

## From scratch in NumPy

For a quadratic objective and a linear constraint, the Lagrange conditions are *linear*, so one call to
`np.linalg.solve` finds both $\mathbf{x}$ and $\lambda$. The problem: the point closest to $\mathbf{p}$ on the hyperplane
$\mathbf{a}\cdot\mathbf{x} = c$, i.e. minimize $\lVert\mathbf{x} - \mathbf{p}\rVert^2$ subject to $\mathbf{a}\cdot\mathbf{x} = c$, with $\mathbf{x}, \mathbf{p}, \mathbf{a}$ `(p,)`.
$F = \lVert\mathbf{x} - \mathbf{p}\rVert^2 - \lambda(\mathbf{a}\cdot\mathbf{x} - c)$; its gradient in $\mathbf{x}$ (A4.2 results (a) and (c) with
$\mathbf{A} = \mathbf{I}$) is $2(\mathbf{x} - \mathbf{p}) - \lambda\mathbf{a}$. Setting it to zero and adding the constraint:

$$\begin{pmatrix} 2\mathbf{I} & -\mathbf{a} \\ \mathbf{a}^\top & 0 \end{pmatrix}\begin{pmatrix}\mathbf{x}\\ \lambda\end{pmatrix} = \begin{pmatrix} 2\mathbf{p} \\ c\end{pmatrix}, \quad \text{shape } (p + 1, p + 1).$$

Solving by hand: the first block row gives $\mathbf{x} = \mathbf{p} + \frac{\lambda}{2}\mathbf{a}$; substituting into $\mathbf{a}\cdot\mathbf{x} = c$ gives
$\frac{\lambda}{2} = \frac{c - \mathbf{a}\cdot\mathbf{p}}{\mathbf{a}\cdot\mathbf{a}}$, so $\mathbf{x} = \mathbf{p} + \frac{c - \mathbf{a}\cdot\mathbf{p}}{\mathbf{a}\cdot\mathbf{a}}\mathbf{a}$. That is a projection, the
cousin of A2.4's: the correction $\mathbf{x} - \mathbf{p}$ is along the constraint's normal $\mathbf{a}$. Worked example 3 is
the case $\mathbf{p} = \mathbf{0}$, $\mathbf{a} = (1, 1)$, $c = 1$.

```text
== from scratch: KKT linear system ==
min x^2 + y^2 s.t. x + y = 1 via KKT: x = [0.5, 0.5]   lambda = 1.0000
random d = 5: a . x = 1.000000   matches the closed-form projection: OK   lambda = -1.2997
residual x - p is parallel to a (the constraint's normal): cos angle = 1.000000
```

($d$ in the lab's printout is the dimension $p$.) The random case has a negative $\lambda$: for an equality
constraint the multiplier can have either sign, since raising $c$ may help or hurt. Only inequality
multipliers are forced to be $\ge 0$.

Gradient descent from 41 starting points on each function of the key plot:

```text
== from scratch: where gradient descent ends up ==
convex (x - 0.5)^2 + 0.25 x^4: distinct end points = [0.4534]
x^4 - 3x^2 + x: 22 of 41 starts end at x = -1.3008  f = -3.5139
x^4 - 3x^2 + x: 19 of 41 starts end at x = +1.1309  f = -1.0702
the dividing line is the local maximum at x = 0.1699: starts left of it go left, right of it go right
```

## With SciPy

`scipy.optimize.minimize(method="SLSQP")` handles equality and inequality constraints. Two conventions to
know: an inequality constraint is written as `fun(x) >= 0`, so $x + y \le 1$ becomes `1 - x - y`; and the
result's `multipliers` (present in the pinned SciPy 1.18, VERIFY (as of 2026-10) if you upgrade) use the same
signs as this lesson for these two problems:

```text
== with scipy ==
equality:   x = [0.5, 0.5]  multiplier = 1.0000  (lambda by hand = 1)  matches: OK
inequality: x = [0.5, 0.5]  multiplier = 3.0000  (mu by hand = 3)  matches: OK
BFGS on x^4 - 3x^2 + x from x0 = -1.0: x = -1.3008  f = -3.5139  success = True
BFGS on x^4 - 3x^2 + x from x0 = +0.5: x = +1.1309  f = -1.0702  success = True
```

The comparisons use `atol=1e-6` because SLSQP stops at about $10^{-7}$ accuracy by default. The last two lines
are the key plot's lesson in library form: a quasi-Newton optimizer (BFGS, a smarter cousin of gradient
descent) reports `success = True` from both starts and returns two different answers. "Success" means "found a
point where the gradient is zero", not "found the best point".

## In practice

- **When to use it:** ask "is this loss convex?" before trusting or comparing optimizer results. Linear,
  ridge, lasso, logistic regression and SVMs: yes, so one run is enough and a mismatch between two correct
  implementations is a bug. Neural networks: no, so results vary with seed and settings. Use constrained
  optimization when the parameters must satisfy rules: SLSQP or `trust-constr` for small smooth problems,
  closed-form KKT systems for quadratic objectives with linear constraints.
- **Failure modes:**
  - *Assuming convexity:* comparing two neural-network configurations with one seed each. You notice that
    rerunning the "better" one with another seed reverses the ranking; use several seeds and report the
    spread.
  - *Trusting "success":* a local optimizer stops at any stationary point, including saddles of a
    non-convex function. You notice different answers from different starts; try several starts, or
    reformulate as a convex problem.
  - *Wrong constraint sign:* writing $x + y - 1$ for SciPy's `fun(x) >= 0` enforces $x + y \ge 1$. You notice
    a solution that violates the constraint you meant; always print $g(\mathbf{x}^\star)$ after solving.
  - *Penalties instead of constraints:* adding $\rho\,(g(\mathbf{x}) - c)^2$ to the loss only approximately enforces
    the constraint unless $\rho$ is huge, and huge $\rho$ makes the problem ill-conditioned (A4.4). You notice
    a small but non-zero violation; use a real constrained solver, or reparameterize (softmax for
    probabilities, $\exp$ for positive values).
  - *Badly scaled constraints:* a constraint in units of $10^6$ next to an objective of order 1 confuses
    SLSQP's tolerances. You notice failures or many iterations; rescale both to order 1.
- **Defaults that work:** check convexity with the Hessian's eigenvalues at a few random points (any negative
  eigenvalue proves non-convexity); for convex problems use any standard solver and expect identical results
  across starts; for non-convex ones run at least 3 seeds; for constrained problems print the multipliers and
  the constraint values at the solution, since they tell you which constraints are binding and what relaxing
  them is worth.

## Exercises

Save your work to `my-work/A/a4_5_convexity_lagrange.md` (or `.py`) and ask for a review.

1. **By hand:** (a) Maximize $f = xy$ subject to $x + y = 10$ with a Lagrange multiplier (the condition
   $\nabla f = \lambda\nabla g$ is the same for a maximum). Find $x$, $y$, $\lambda$ and $f^\star$, then check that
   $\lambda = df^\star/dc$ by solving for a general $c$. (b) Convex or not, with a one-line reason each: $e^x$;
   $x^3$; $\lvert x\rvert$ (no second derivative at 0: use the chord definition); $-\log x$ for $x > 0$;
   $x^2 + xy + 2y^2$; $x^2 - y^2$. (c) Minimize $(x - 3)^2$ subject to $x \le 1$, and then subject to $x \le 5$.
   Give $x^\star$ and $\mu$ for each, and check all four KKT conditions.
2. **Code:** With more features than examples ($n < d$), $\mathbf{X}\mathbf{w} = \mathbf{y}$ has infinitely many exact
   solutions. (a) Use Lagrange multipliers (one per equation, a vector $\boldsymbol{\nu}$ of shape `(n,)`) to derive
   the solution of smallest norm: minimize $\lVert\mathbf{w}\rVert^2$ subject to $\mathbf{X}\mathbf{w} = \mathbf{y}$. (b) Implement it with
   `np.linalg.solve` (no `inv`) for a random $\mathbf{X}$ `(3, 6)` and $\mathbf{y}$ `(3,)` with `SEED = 0`, and check it
   against `np.linalg.pinv(X) @ y` ([A3.2](a3_2_svd.md)), `np.linalg.lstsq` and SLSQP. (c) Construct a
   *different* exact solution (hint: add a vector from the null space of $\mathbf{X}$) and show it has a larger
   norm. Why is the minimum-norm solution orthogonal to the null space?
3. **Think:** Your team compares two model variants by training each once and keeping the one with the
   lower validation loss. For a logistic regression this worked reliably; after switching to a small neural
   network, the "winner" changes from week to week even though the data barely change. Explain the difference
   using convexity. What would you change in the comparison procedure, and what would you say to a
   stakeholder who asks why the neural network's metric "moves around"?

## Takeaways

1. Convex = the graph lies below every chord; test it with $f'' \ge 0$, or a PSD Hessian, everywhere.
2. For a convex function every local minimum is global, so gradient descent from any start (with a stable
   rate) reaches the same answer; the MSE, ridge and logistic losses are convex, neural-network losses are not.
3. At an equality-constrained optimum the gradients are parallel, $\nabla f = \lambda\nabla g$; the Lagrangian
   $F = f - \lambda(g - c)$ packs this and the constraint into "all derivatives zero".
4. The multiplier is a shadow price: $\lambda = df^\star/dc$, the change in the best value per unit of relaxation.
5. For inequalities (KKT), either the constraint is inactive and $\mu = 0$, or it is active and $\mu \ge 0$;
   complementary slackness is why only support vectors matter in an SVM.

**What builds on this:** [A5.5](a5_5_mle_map.md) (maximum likelihood, where a constraint such as
"probabilities sum to 1" can be handled with a multiplier); [B2.2](../B/b2_2_regularization.md) (ridge and
lasso as constrained problems); [B2.5](../B/b2_5_svm.md) (the SVM's dual problem and support vectors);
[C3.1](../C/c3_1_optimizers.md) (optimizers for non-convex losses).
