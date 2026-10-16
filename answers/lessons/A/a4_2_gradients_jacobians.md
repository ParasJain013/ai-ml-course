# Answer key: A4.2 Gradients, Jacobians and the shapes of matrix derivatives used in ML

Lesson: [`lessons/A/a4_2_gradients_jacobians.md`](../../../lessons/A/a4_2_gradients_jacobians.md)

## Exercise 1 (by hand)

**Solution (a).** $f(\mathbf{x}) = (x_1 - 2x_2)^2 + x_1x_2$. Expand the square to use only the rules
table: $x_1^2 - 4x_1x_2 + 4x_2^2 + x_1x_2 = x_1^2 - 3x_1x_2 + 4x_2^2$. Then
$\partial f/\partial x_1 = 2x_1 - 3x_2$ and $\partial f/\partial x_2 = -3x_1 + 8x_2$. At $(3, 1)$:
$\nabla f = (6 - 3,\; -9 + 8) = (3, -1)$. (Written without expanding:
$\partial f/\partial x_1 = 2(x_1 - 2x_2) + x_2$, $\partial f/\partial x_2 = -4(x_1 - 2x_2) + x_1$, same values.)

**Solution (b).** The MSE part is pattern 4, $\frac{2}{n}\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$. The penalty
$\lambda\lVert\mathbf{w}\rVert^2 = \lambda\,\mathbf{w}^\top\mathbf{I}\mathbf{w}$ is pattern 2 with the symmetric $\mathbf{M} = \lambda\mathbf{I}$,
gradient $2\lambda\mathbf{w}$ (or directly: $\lambda\sum_j w_j^2$ has partial $2\lambda w_j$). Gradients add:

$$\nabla_{\mathbf{w}}\mathcal{L} = \frac{2}{n}\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}) + 2\lambda\mathbf{w}, \quad \text{shape } (d,).$$

For the worked example the MSE part is $(0.6667, 2)$ and $2\lambda\mathbf{w} = 2(0.5)(0, 1) = (0, 1)$, so
$\nabla\mathcal{L} = (0.6667, 3)$.

**Solution (c).** Row 1 is $\nabla(x_1x_3) = (x_3, 0, x_1)$; row 2 is $\nabla(x_2^2 - x_1) = (-1, 2x_2, 0)$.
At $(1, 2, 3)$: $\mathbf{J} = \begin{pmatrix}3&0&1\\-1&4&0\end{pmatrix}$, shape `(2, 3)`: 2 outputs, 3 inputs.

```text
== exercise 1 check ==
(a) gradient at (3, 1) = [3.0, -1.0]
(b) ridge gradient = [0.6667, 3.0]
(c) Jacobian = [[3.0, 0.0, 1.0], [-1.0, 4.0, 0.0]]  shape (2, 3)
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) Correct partial derivatives and $(3, -1)$ | 3 |
| (b) Gradient formula with both terms, derived from the patterns (not quoted) | 3 |
| (b) Shape `(d,)` stated and value $(0.6667, 3)$ | 2 |
| (c) Correct entries, rows = outputs | 3 |
| (c) Shape `(2, 3)` | 1 |

**Likely mistakes:**

- In (a), forgetting the inner factor when differentiating $(x_1 - 2x_2)^2$ in $x_2$ (writing $2(x_1 - 2x_2)$
  instead of $-4(x_1 - 2x_2)$). Expanding first avoids this until the chain rule (A4.3) is solid.
- In (b), writing the penalty's gradient as $\lambda\mathbf{w}$ (dropping the 2) or putting the $\frac1n$ on
  the penalty too.
- In (c), laying the Jacobian out as `(3, 2)` (columns = outputs). The course convention, and NumPy's
  and SciPy's, is rows = outputs.

## Exercise 2 (code)

**Solution (a).** Write $\mathbf{Z} = \mathbf{X}\mathbf{W}$, `(n, k)`. The data term is
$\frac1n\sum_{i,c}(Z_{ic} - Y_{ic})^2$, whose partial derivative in $Z_{ic}$ is $\frac{2}{n}(Z_{ic} - Y_{ic})$.
So $\mathbf{G} = \frac{2}{n}(\mathbf{X}\mathbf{W} - \mathbf{Y})$, `(n, k)`, and pattern 5 gives $\mathbf{X}^\top\mathbf{G}$. The penalty
adds $2\lambda\mathbf{W}$ (entry by entry, as in exercise 1b):

$$\nabla_{\mathbf{W}}\mathcal{L} = \frac{2}{n}\mathbf{X}^\top(\mathbf{X}\mathbf{W} - \mathbf{Y}) + 2\lambda\mathbf{W}, \quad \text{shape } (d, k).$$

**Solution (b–d).** [`a4_2_gradients_jacobians_ex2.py`](a4_2_gradients_jacobians_ex2.py):

```python
def loss_grad(W: FloatArray, X: FloatArray, Y: FloatArray, lam: float) -> FloatArray:
    """Gradient X^T G + 2 lam W with G = (2/n)(XW - Y). -> (d, k), the shape of W."""
    G = 2.0 / X.shape[0] * (X @ W - Y)  # (n, k): dL/dZ for Z = XW
    return X.T @ G + 2.0 * lam * W  # (d, n) @ (n, k) -> (d, k)
```

Setting the gradient to zero: $\frac{2}{n}\mathbf{X}^\top\mathbf{X}\mathbf{W} - \frac{2}{n}\mathbf{X}^\top\mathbf{Y} + 2\lambda\mathbf{W} = \mathbf{0}$;
multiply by $\frac{n}{2}$: $(\mathbf{X}^\top\mathbf{X} + n\lambda\mathbf{I})\mathbf{W} = \mathbf{X}^\top\mathbf{Y}$, solved for all $k$ columns at once
with `np.linalg.solve`. scikit-learn's `Ridge` minimizes $\lVert\mathbf{Y} - \mathbf{X}\mathbf{W}\rVert_F^2 + \alpha\lVert\mathbf{W}\rVert_F^2$,
which is $n$ times our loss when $\alpha = n\lambda$; multiplying a loss by a positive constant does not move
its minimum. Its `coef_` is `(k, d)`, the transpose of our $\mathbf{W}$.

```text
== exercise 2: multi-output ridge gradient ==
n=8 d=3 k=2: X^T G + 2 lam W   shape (3, 2)  rel error < 1e-8: True
n=8 d=3 k=3: X^T G + 2 lam W   shape (3, 3)  rel error < 1e-8: True
n=8 d=3 k=3: transposed formula shape (3, 3)  rel error 0.82  (shape matches, gradient wrong)

== exercise 2: closed form versus scikit-learn ==
gradient at the solution is zero: True
matches sklearn Ridge(alpha = n * lam = 0.8, fit_intercept=False): OK
```

The square case is the point of (c): with $d = k$, $\mathbf{G}^\top\mathbf{X}$ is also `(3, 3)`, so an
`assert grad.shape == W.shape` passes while the gradient is wrong. Only the numerical check catches it.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| (a) Correct gradient, with $\mathbf{G} = \frac{2}{n}(\mathbf{X}\mathbf{W} - \mathbf{Y})$ identified | 3 |
| (b) Implementation vectorized (no Python loop over $n$), typed, shapes in comments | 2 |
| (b) Checked with a copied `numerical_gradient` and a relative error, passes | 3 |
| (c) Transposed formula shown to pass the shape check and fail the gradient check | 3 |
| (d) Closed form via `solve` (not `inv`), gradient zero there | 2 |
| (d) `alpha = n * lam` worked out and matched with `Ridge`, handling `coef_`'s `(k, d)` layout | 3 |

**Likely mistakes:**

- Using `alpha = lam` and getting a mismatch, then loosening the tolerance instead of finding the
  factor $n$.
- Comparing `ridge.coef_` with $\mathbf{W}$ directly: it is `(k, d)`.
- Forgetting `fit_intercept=False`, so scikit-learn centers the data and fits a different model.
- Testing the transposed formula only in the non-square case, where it crashes with a shape error,
  and concluding that shape errors would always catch it.

## Exercise 3 (think)

**Strong answer:** A central-difference check needs 2 forward passes per parameter:
$2 \times 10^7 \times 0.05\,\text{s} = 10^6\,\text{s}$, about 11.6 days. Not usable in CI. Cheap options: (1) shrink
the model in the test (same code, tiny widths such as 3 and 5, different from each other) so a full
check has a few hundred parameters and runs in seconds; (2) for the full-size model, check a random
subset of, say, 50 coordinates, or check a few random **directional derivatives**: pick a random unit
vector $\mathbf{u}$, compute $(f(\boldsymbol{\theta} + h\mathbf{u}) - f(\boldsymbol{\theta} - h\mathbf{u}))/2h$ with 2 forward passes and
compare it with $\nabla f \cdot \mathbf{u}$. A random direction touches every parameter, so a bug anywhere
shows up unless it happens to be orthogonal to $\mathbf{u}$, which is unlikely; repeat with 3–5
directions. Run in float64 and at random inputs. What it catches: wrong formulas, transposes, missing
constants, wrong signs, broken code paths. What it misses: bugs only triggered at sizes or values the
test does not use (shape-specific code paths, overflow at real scale, kinks such as ReLU at exactly
0, masking or padding that the tiny test never exercises), nondeterminism, and bugs in code
the test does not run (for example only in mixed precision or on GPU), and it cannot tell whether the
loss itself is the right loss. Keep one slow full-size check in a nightly job.

**Weak answer:** "Run the numerical check on the full model overnight." Weak because the arithmetic
gives about 12 days, not a night, and because it shows no idea how to make a test cheap while keeping
its power.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Cost computed correctly (about $10^6$ s, days, not hours) | 2 |
| A cheap check: tiny model and/or random-subset or directional-derivative checks | 3 |
| Mentions float64, random points, distinct dimensions | 2 |
| Realistic list of what it misses (scale-specific paths, kinks, untested code paths) | 3 |

## Scoring

Total 38 pts. Pass ≥ 70% (27 pts). Topic ids to revisit for each lost criterion: exercise 1 → A4.2
"The math" (patterns 1–4, Jacobian layout) and A4.1 (partial derivatives); exercise 2 → A4.2 pattern 5
and "From scratch", A2.4 (normal equations); exercise 3 → A4.2 "In practice" and A4.1 (cost of finite
differences).
