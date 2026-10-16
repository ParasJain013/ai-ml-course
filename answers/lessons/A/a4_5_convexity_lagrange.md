# Answer key: A4.5 Convexity, constrained optimization and Lagrange multipliers

Lesson: [`lessons/A/a4_5_convexity_lagrange.md`](../../../lessons/A/a4_5_convexity_lagrange.md)

## Exercise 1 (by hand)

**Solution (a).** $f = x^2 + 2y^2$, $h = x + y = 3$. $\nabla f = (2x, 4y)$, $\nabla h = (1, 1)$. From
$\nabla f = \lambda\nabla h$: $2x = \lambda$ and $4y = \lambda$, so $x = \lambda/2$, $y = \lambda/4$, and $x = 2y$. The constraint
gives $2y + y = 3$: $y = 1$, $x = 2$, $\lambda = 4$, $f^\star = 4 + 2 = 6$.

General $c$: $3y = c$, so $y = c/3$, $x = 2c/3$ and $f^\star(c) = \frac{4c^2}{9} + \frac{2c^2}{9} = \frac{2c^2}{3}$.
$\frac{df^\star}{dc} = \frac{4c}{3}$, which is 4 at $c = 3$ ✓ $= \lambda$. (Numerically: $f^\star(3.01) = 6.04007$, a rise of
$0.04007$ for $0.01$, rate $4.007$.)

**Solution (b).**

| Function | Convex? | One-line reason |
|---|---|---|
| $e^{-x}$ | yes | $f'' = e^{-x} > 0$ |
| $x^3$ | no | $f'' = 6x < 0$ for $x < 0$; the chord from $(-2, -8)$ to $(0, 0)$ is at $-4$ at $x = -1$, below $f(-1) = -1$ |
| $\max(0, x)$ | yes | the maximum of two linear (hence convex) functions |
| $-\log x$, $x > 0$ | yes | $f'' = 1/x^2 > 0$ |
| $x^4 - x^2$ | no | $f'' = 12x^2 - 2 < 0$ for $\lvert x\rvert < 0.41$; the chord from $-0.5$ to $0.5$ is at height $f(\pm 0.5) = -0.1875$, below $f(0) = 0$ |
| $\lvert x - 1\rvert + x^2$ | yes | a sum of two convex functions ($\lvert\cdot\rvert$ of a linear map, and $x^2$) |

```text
== exercise 1 check ==
(a) SLSQP: (x, y) = [2.0, 1.0]   f* = 6.0000   multiplier = 4.0000
    c = 3.0: f*(c) = 2c^2/3 = 6.00000
    c = 3.01: f*(c) = 2c^2/3 = 6.04007
    (f*(3.01) - f*(3)) / 0.01 = 4.0067
(b) exp(-x)          chords below the curve:   0.0%
(b) x^3              chords below the curve:  49.8%
(b) max(0, x)        chords below the curve:   0.0%
(b) -log(x), x > 0   chords below the curve:   0.0%
(b) x^4 - x^2        chords below the curve:   6.9%
(b) |x - 1| + x^2    chords below the curve:   0.0%
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Lagrange equations set up correctly | 2 |
| (a) $(2, 1)$, $f^\star = 6$, $\lambda = 4$ | 2 |
| (a) $f^\star(c) = 2c^2/3$ and $df^\star/dc = 4 = \lambda$ | 2 |
| (b) Six correct verdicts (1 pt each) | 6 |
| (b) Reasons valid (a correct rule, $f''$ sign, or a specific dipping chord) | 2 |

**Likely mistakes:**

- In (a), solving $\nabla f = \mathbf{0}$ (unconstrained, giving $(0, 0)$, which violates $x + y = 3$).
- Calling $x^3$ convex "because it is increasing": increasing and convex are different properties.
- Calling $\max(0, x)$ not convex because it has a kink: convexity does not require a derivative.
- Calling $x^4 - x^2$ convex because "even powers are convex": $-x^2$ is concave and wins near 0. The chord
  test found violations in only 6.9% of random chords: a small but real failure.

## Exercise 2 (code)

**Solution:** [`a4_5_convexity_lagrange_ex2.py`](a4_5_convexity_lagrange_ex2.py). The projection onto the ball:

```python
def project(w: FloatArray, r: float) -> FloatArray:
    """Closest point to w in the ball |w| <= r: shrink w onto the sphere if it is outside."""
    norm = np.linalg.norm(w)
    return w if norm <= r else w * (r / norm)
```

With step $1/\lambda_{\max}$ of the MSE Hessian (A4.4) and stopping when an update moves $\mathbf{w}$ by less than
$10^{-12}$:

```text
== exercise 2: data and the unconstrained fit ==
least squares w = [2.9572, -1.9295, 1.1719]   |w| = 3.7205   radius r = 1.8602

== exercise 2: projected gradient descent ==
projected GD: 19 steps   w = [1.6013, -0.7238, 0.6102]   |w| = 1.8602

== exercise 2: SLSQP, the multiplier and ridge ==
SLSQP: w = [1.6013, -0.7238, 0.6102]   multiplier mu = 1.0545
projected GD matches SLSQP (atol 1e-6): OK
KKT stationarity: grad L + 2 mu w = [0.0, 0.0, 0.0]
ridge with lambda = mu: w = [1.6013, -0.7238, 0.6102]   |w| = 1.8602
sklearn Ridge(alpha = n mu = 63.2719) gives the same w: OK
```

The constraint is active ($\lVert\mathbf{w}\rVert = r$ exactly and $\mu > 0$), as it must be when $r$ is smaller than the
unconstrained solution's length. SLSQP's inequality is written $r^2 - \lVert\mathbf{w}\rVert^2 \ge 0$; with its sign convention
the KKT condition reads $\nabla\mathcal{L} + 2\mu\mathbf{w} = \mathbf{0}$, the gradient of $\mathcal{L} + \mu\lVert\mathbf{w}\rVert^2$, i.e. ridge with
$\lambda = \mu$. scikit-learn's `Ridge` minimizes $\lVert\mathbf{y} - \mathbf{X}\mathbf{w}\rVert^2 + \alpha\lVert\mathbf{w}\rVert^2$, which is $n$ times ours,
so $\alpha = n\mu$ (A4.2 exercise 2).

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| (a) Correct projection (only when outside; scales onto the sphere) | 3 |
| (a) Step size justified (from the Hessian) and a convergence test with a cap | 2 |
| (b) SLSQP set up with the inequality in the `>= 0` direction; multiplier read | 3 |
| (b) Projected GD and SLSQP agree to $10^{-6}$ | 2 |
| (c) KKT stationarity checked numerically | 2 |
| (c) Ridge with $\lambda = \mu$ matches, by `solve` (system $\mathbf{X}^\top\mathbf{X} + n\mu\mathbf{I}$) and by `Ridge(alpha = n\mu)` | 4 |

**Likely mistakes:**

- Writing the SLSQP constraint as $\lVert\mathbf{w}\rVert^2 - r^2$ ("ineq" means $\ge 0$), which demands $\lVert\mathbf{w}\rVert \ge r$: the
  solver returns the least-squares $\mathbf{w}$ and a zero multiplier.
- Projecting onto the sphere every step, even when inside: that solves a different problem
  ($\lVert\mathbf{w}\rVert = r$), which happens to coincide here only because the constraint is active.
- Using ridge with $\lambda = 2\mu$ or $\alpha = \mu$: track the factor 2 from $\nabla\lVert\mathbf{w}\rVert^2 = 2\mathbf{w}$ and the $n$
  from scikit-learn's scaling.

## Exercise 3 (think)

**Strong answer:** Logistic regression's loss is convex (strictly, with a little regularization), so
each weekly fit reaches *the* unique optimum for that week's data: retraining is deterministic, small data
changes give small coefficient changes, and two engineers reproduce the same model. A neural network's
loss is non-convex: (1) **reproducibility and stability**: the answer depends on initialization, seed,
data order and even hardware; two retrains on the same data can give different predictions for the same
customer, so a customer's score can flip week to week for no business reason, and an audit cannot
reproduce a past decision unless every seed and version is pinned; (2) **silent training failures**: a
run can stop in a poor local minimum or plateau, and nothing in the loss says so, so the weekly job needs
extra monitoring and sometimes several restarts; (3) also acceptable: harder to guarantee monotone
behaviour (higher income never lowers the score), more hyperparameters to keep tuned. To measure: retrain
the network several times (different seeds) on the same week and on consecutive weeks; report the spread
of validation scores (is the 1% larger than the seed-to-seed spread?) and the share of customers whose
decision flips between retrains, and compare both with logistic regression's. Decide with those numbers.

**Weak answer:** "Use the neural network because it is 1% better; set a random seed so it is
reproducible." Weak because a fixed seed does not make results stable to small data changes (the next
week's data is different), it ignores that the 1% may be within seed-to-seed noise, and it never
measures prediction flips.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Convex fit has a unique optimum: deterministic, stable retrains | 2 |
| Two concrete operational problems of non-convexity (seed dependence and flips, silent poor minima, ...) | 4 |
| A measurement plan: multiple seeds, spread versus the 1% gain, decision-flip rate | 3 |
| A decision rule tied to those measurements | 1 |

## Scoring

Total 40 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → A4.5
"The math" (Lagrange conditions, convexity tests) and A4.1 (second derivatives); exercise 2 → A4.5
"Constraint versus penalty", A4.4 (step size) and A4.2 (ridge gradient); exercise 3 → A4.5 "Intuition"
and "In practice".
