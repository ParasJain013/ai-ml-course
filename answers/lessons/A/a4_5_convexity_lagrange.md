# Answer key: A4.5 Convexity, constrained optimization and Lagrange multipliers

Lesson: [`lessons/A/a4_5_convexity_lagrange.md`](../../../lessons/A/a4_5_convexity_lagrange.md)

## Exercise 1 (by hand)

**Solution (a).** $\nabla f = (y, x)$, $\nabla g = (1, 1)$, so $y = \lambda$ and $x = \lambda$. The constraint gives
$2\lambda = 10$: $\lambda = 5$, $x = y = 5$, $f^\star = 25$. For a general level $c$: $x = y = c/2$, $f^\star(c) = c^2/4$,
$df^\star/dc = c/2 = 5$ at $c = 10$ ✓. (Numerically: $f^\star(10.1) = 25.5025$, a change of $0.5025$ for $0.1$ of
budget, ratio 5.025.) It is a maximum, not a minimum, because along the line $f = x(10 - x)$ is a
downward parabola.

**Solution (b).**

| Function | Convex? | Reason |
|---|---|---|
| $e^x$ | yes | $f'' = e^x > 0$ |
| $x^3$ | no | $f'' = 6x < 0$ for $x < 0$ |
| $\lvert x\rvert$ | yes | $\lvert tx + (1 - t)y\rvert \le t\lvert x\rvert + (1 - t)\lvert y\rvert$ (triangle inequality); the kink is fine |
| $-\log x$, $x > 0$ | yes | $f'' = 1/x^2 > 0$ |
| $x^2 + xy + 2y^2$ | yes | Hessian $\begin{pmatrix}2&1\\1&4\end{pmatrix}$: trace 6, determinant 7, both eigenvalues positive (1.59, 4.41) |
| $x^2 - y^2$ | no | Hessian $\operatorname{diag}(2, -2)$ has a negative eigenvalue: a saddle |

**Solution (c).** $x \le 1$: the unconstrained minimum $x = 3$ is infeasible, so the constraint is active:
$x^\star = 1$. Stationarity: $2(x - 3) + \mu \cdot 1 = 0$ gives $\mu = 4$. KKT: stationarity ✓, feasible
($1 \le 1$) ✓, $\mu = 4 \ge 0$ ✓, $\mu(x - 1) = 0$ ✓. Check: $f^\star(c) = (c - 3)^2$ for $c < 3$, so
$-df^\star/dc = -2(c - 3) = 4$ at $c = 1$ ✓. $x \le 5$: $x^\star = 3$ is feasible, inactive, $\mu = 0$; stationarity
$2(3 - 3) + 0 = 0$ ✓, feasible ✓, $\mu \ge 0$ ✓, $\mu \cdot (3 - 5) = 0$ ✓.

```text
== exercise 1 check ==
max xy s.t. x + y = 10.0: x = y = 5.0  lambda = 5.0  f* = 25.0000
max xy s.t. x + y = 10.1: x = y = 5.05  lambda = 5.05  f* = 25.5025
(f*(10.1) - f*(10)) / 0.1 = 5.0250  -> lambda = 5 (exactly df*/dc = c / 2 = 5 at c = 10)
x^2 + xy + 2y^2  Hessian eigenvalues = [1.5858, 4.4142]
x^2 - y^2        Hessian eigenvalues = [-2.0, 2.0]
min (x - 3)^2 s.t. x <= 1.0: x* = 1.0000  mu = 4.0000
min (x - 3)^2 s.t. x <= 5.0: x* = 3.0000  mu = 0.0000
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) $x = y = 5$, $\lambda = 5$, $f^\star = 25$ | 2 |
| (a) $f^\star(c) = c^2/4$ and $df^\star/dc = 5 = \lambda$ | 2 |
| (b) Six correct verdicts | 3 |
| (b) A valid reason for each (second derivative, Hessian eigenvalues, or the chord definition for $\lvert x\rvert$) | 3 |
| (c) $x^\star = 1$, $\mu = 4$ and $x^\star = 3$, $\mu = 0$ | 2 |
| (c) All four KKT conditions checked in both cases | 2 |

**Likely mistakes:**

- Calling $\lvert x\rvert$ "not convex because it is not differentiable". Differentiability is not required;
  the second-derivative test is only a convenient sufficient tool where it applies.
- Calling $x^2 + xy + 2y^2$ non-convex because of the cross term $xy$. Mixed terms are fine as long as the
  Hessian's eigenvalues stay non-negative.
- In (c), writing the stationarity condition as $2(x - 3) - \mu = 0$ and getting $\mu = -4$: a sign
  convention slip. With $F = f + \mu(g - c)$ and a constraint $g \le c$, the multiplier of an active
  constraint must come out $\ge 0$; a negative value means the sign of $g$ is flipped.
- Forgetting to check whether the unconstrained minimum is feasible before assuming the constraint is active.

## Exercise 2 (code)

**Solution:** [`a4_5_convexity_lagrange_ex2.py`](a4_5_convexity_lagrange_ex2.py).

(a) With $\boldsymbol{\nu}$ `(n,)`, $F(\mathbf{w}, \boldsymbol{\nu}) = \lVert\mathbf{w}\rVert^2 - \boldsymbol{\nu}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$. The gradient in $\mathbf{w}$ (A4.2 results (a)
and (b)) is $2\mathbf{w} - \mathbf{X}^\top\boldsymbol{\nu}$; setting it to zero gives $\mathbf{w} = \mathbf{X}^\top\boldsymbol{\nu}/2$, shape `(d,)`. Substitute into
$\mathbf{X}\mathbf{w} = \mathbf{y}$: $(\mathbf{X}\mathbf{X}^\top)\,\boldsymbol{\nu}/2 = \mathbf{y}$, an `(n, n)` system that is invertible when the $n$ rows of $\mathbf{X}$ are
independent. So

$$\mathbf{w}^\star = \mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top)^{-1}\mathbf{y},$$

computed as `X.T @ np.linalg.solve(X @ X.T, y)`. This is the pseudo-inverse of A3.2 for a wide matrix.

```text
== exercise 2: minimum-norm solution ==
w = [0.184, 0.1325, 0.1106, -0.3202, -0.5472, 0.2092]   ||w|| = 0.7137   X w - y = [0.0, 0.0, 0.0]
matches pinv, lstsq and SLSQP: OK
another solution w + 0.5 z: X w2 - y = [0.0, 0.0, 0.0]   ||w2|| = 0.8714 > ||w|| = 0.7137
|w . z| < 1e-12 (orthogonal): True   ||w2||^2 = ||w||^2 + 0.25 ||z||^2: True
```

(c) Any $\mathbf{z}$ with $\mathbf{X}\mathbf{z} = \mathbf{0}$ (here a row of $\mathbf{V}^\top$ beyond the rank, A3.2) gives another exact solution
$\mathbf{w} + t\mathbf{z}$. Why $\mathbf{w}^\star$ is orthogonal to every such $\mathbf{z}$: $\mathbf{w}^\star = \mathbf{X}^\top(\boldsymbol{\nu}/2)$ is a combination of the
rows of $\mathbf{X}$, and $\mathbf{X}\mathbf{z} = \mathbf{0}$ says $\mathbf{z}$ is orthogonal to every row. So
$\lVert\mathbf{w}^\star + t\mathbf{z}\rVert^2 = \lVert\mathbf{w}^\star\rVert^2 + t^2\lVert\mathbf{z}\rVert^2$ (Pythagoras), which is smallest at $t = 0$. This is the
"implicit regularization" you will meet again: gradient descent from $\mathbf{w} = \mathbf{0}$ on an underdetermined least
squares problem only ever moves in the row space ($\nabla = \frac{2}{n}\mathbf{X}^\top\mathbf{r}$), so it converges to this
minimum-norm solution.

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Lagrangian with a vector multiplier, stationarity $2\mathbf{w} = \mathbf{X}^\top\boldsymbol{\nu}$, shapes stated | 3 |
| (a) Final formula $\mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top)^{-1}\mathbf{y}$ | 2 |
| (b) Implemented with `solve`, not `inv` | 2 |
| (b) Matches `pinv`, `lstsq` and SLSQP with stated tolerances | 3 |
| (c) A second exact solution from the null space, with a larger norm | 2 |
| (c) Orthogonality explained (row space versus null space) | 2 |

**Likely mistakes:**

- Writing $(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$, the normal-equation formula: for $n < d$, $\mathbf{X}^\top\mathbf{X}$ is `(d, d)` with rank
  at most $n$, so it is singular and `solve` raises (or returns garbage if rounding hides the singularity).
- Using a scalar multiplier for $n$ equations. Each equation needs its own.
- Building a null-space vector by guessing; use the SVD (or `scipy.linalg.null_space`) and verify $\mathbf{X}\mathbf{z} \approx \mathbf{0}$.
- Comparing with SLSQP at `rtol=1e-6` and concluding the derivation is wrong when it fails: SLSQP's default
  tolerance is about $10^{-7}$ to $10^{-6}$. Justify the looser tolerance in a comment.

## Exercise 3 (think)

**Strong answer:** Logistic regression's loss is convex, so (with enough iterations) training always reaches
the same optimum: the trained model is a deterministic function of the data, and a one-run comparison
measures only the variants' real difference plus the small effect of the data changing. A neural network's
loss is non-convex: the result depends on the random initialization, the order of mini-batches, dropout and
other randomness, so each training run lands in a different local minimum with a somewhat different
validation loss. If that run-to-run spread is as large as the difference between the variants, a single run
picks the winner by chance, and the "winner" flips. Changes: train each variant with several seeds (at least 3
to 5), compare mean validation loss with its spread (or a paired comparison per seed), declare a winner only
if the gap clearly exceeds the noise ([A6.3](../../../lessons/A/a6_3_hypothesis_testing.md)), fix and log all
seeds and data versions so a run can be reproduced exactly, and consider averaging (ensembling) seeds if
stability matters in production. To a stakeholder: "Training a neural network involves randomness, like
shuffling a deck before dealing; each training run ends up at a slightly different model. The metric moving
by about this much is normal variation; we now report the average of several runs with its normal range, and
only call a change real if it is bigger than that range."

**Weak answer:** "Neural networks overfit, so the validation loss is unstable; add regularization." Weak
because overfitting is not the cause of run-to-run variation (an underfit network also varies with the seed),
it ignores convexity, which is the actual difference between the two models, and it does not fix the flawed
single-run comparison.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Convex loss → same optimum every run; non-convex → depends on seed, order, initialization | 3 |
| Identifies run-to-run variance comparable to the gap as the reason the winner flips | 2 |
| Procedure changes: multiple seeds, compare means with spread or paired tests, fix and log seeds | 3 |
| A clear, non-technical explanation for the stakeholder | 2 |

## Scoring

Total 38 pts. Pass ≥ 70% (27 pts). Topic ids to revisit for each lost criterion: exercise 1 → A4.5 "The
math" (convexity tests, Lagrange, KKT); exercise 2 → A4.5 "From scratch" and A3.2 (pseudo-inverse, null
space); exercise 3 → A4.5 "Why local means global" and A6.3 (comparing noisy measurements).
