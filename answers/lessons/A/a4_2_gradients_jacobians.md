# Answer key: A4.2 Gradients, Jacobians and the shapes of matrix derivatives used in ML

Lesson: [`lessons/A/a4_2_gradients_jacobians.md`](../../../lessons/A/a4_2_gradients_jacobians.md)

## Exercise 1 (by hand)

**Solution (a).** The loss is a sum of two terms, so its gradient is the sum of their gradients. The
first is the MSE, result (d): $\frac{2}{n}\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$. The second is
$\lambda\,\mathbf{w}^\top\mathbf{I}\mathbf{w}$, result (b) with the symmetric $\mathbf{M} = \lambda\mathbf{I}$: $2\lambda\mathbf{w}$. So

$$\nabla_{\mathbf{w}}\mathcal{L} = \frac{2}{n}\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}) + 2\lambda\mathbf{w}.$$

Shapes: $\mathbf{X}^\top$ `(d, n)` times the residual `(n,)` is `(d,)`; $\mathbf{w}$ is `(d,)`; the sum is `(d,)` ✓.

**Solution (b).** $\mathbf{X}\mathbf{w} = (1, 2, 3, 4)$ for $\mathbf{w} = (1, 1)$, so $\mathbf{r} = (1, 2, 3, 4) - (1, 3, 4, 4) = (0, -1, -1, 0)$.
$\mathbf{X}^\top\mathbf{r} = (0 - 1 - 1 + 0,\; 0 - 1 - 2 + 0) = (-2, -3)$, times $\frac{2}{4}$: $(-1, -1.5)$. Penalty:
$2 \cdot 0.5 \cdot (1, 1) = (1, 1)$. Gradient: $(0, -0.5)$.

**Solution (c).** Set the gradient to 0 and multiply by $n/2$:
$\mathbf{X}^\top\mathbf{X}\mathbf{w} - \mathbf{X}^\top\mathbf{y} + n\lambda\mathbf{w} = \mathbf{0}$, so

$$(\mathbf{X}^\top\mathbf{X} + n\lambda\mathbf{I})\,\mathbf{w} = \mathbf{X}^\top\mathbf{y}.$$

Here $n\lambda = 2$: $\begin{pmatrix}6&6\\6&16\end{pmatrix}\mathbf{w} = (12, 23)$. Subtract the first equation from the second:
$10w_2 = 11$, so $w_2 = 1.1$, and $w_1 = (12 - 6.6)/6 = 0.9$. Ridge gives $\mathbf{w} = (0.9, 1.1)$ against
$(1.5, 1.0)$ unpenalized: the intercept dropped by 0.6 while the slope *rose* by 0.1.

Why: the penalty shrinks most where the data care least. The eigenvalues of $\mathbf{X}^\top\mathbf{X}$ are 1.19
and 16.8 (A3.1); the small one belongs to the direction $\approx (0.91, -0.42)$, "raise the intercept and
lower the slope together", along which the fit barely changes because the line still passes near the
middle of the points. Ridge buys most of its reduction in $\lVert\mathbf{w}\rVert^2$ along that cheap direction,
mostly by cutting the intercept, and the slope rises to keep the line through the data. This is also
the practical lesson: you usually do *not* penalize the intercept, since its size depends only on where
$y$ is centred ([B2.2](../../../lessons/B/b2_2_regularization.md)).

**Solution (d).** $\mathbf{J} = \begin{pmatrix} x_2^2 & 2x_1x_2 \\ e^{x_1} & -1\end{pmatrix}$, at $(0, 1)$:
$\begin{pmatrix}1&0\\1&-1\end{pmatrix}$, shape `(2, 2)` (2 outputs, 2 inputs).

```text
== exercise 1 check ==
residual at w = (1, 1): [0.0, -1.0, -1.0, 0.0]   gradient = [0.0, -0.5]   numerical = [0.0, -0.5]
(X^T X + n lam I) = [[6.0, 6.0], [6.0, 16.0]]   X^T y = [12.0, 23.0]   w_ridge = [0.9, 1.1]
unpenalized w = [1.5, 1.0]   eigenvalues of X^T X = [1.1898, 16.8102]
Jacobian of (x1 x2^2, e^x1 - x2) at (0, 1) = [[1.0, 0.0], [1.0, -1.0]]  shape (2, 2)
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Gradient $\frac{2}{n}\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}) + 2\lambda\mathbf{w}$ derived from the listed results | 3 |
| (a) Shape check stated | 1 |
| (b) Residual, $\mathbf{X}^\top\mathbf{r}$ and gradient $(0, -0.5)$ | 2 |
| (c) System $(\mathbf{X}^\top\mathbf{X} + n\lambda\mathbf{I})\mathbf{w} = \mathbf{X}^\top\mathbf{y}$ with the factor $n$ handled correctly | 2 |
| (c) Solution $(0.9, 1.1)$ | 2 |
| (c) A sensible explanation of why the intercept moved (weakly determined direction, or "penalty on the intercept is arbitrary") | 2 |
| (d) Jacobian entries and shape `(2, 2)` | 2 |

**Likely mistakes:**

- Writing $\lambda\mathbf{w}$ instead of $2\lambda\mathbf{w}$ (the same forgotten 2 as the lab's bug).
- Writing $(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})$ and losing the $n$: true only if the loss uses a *sum* of
  squares instead of the mean. scikit-learn's `Ridge` uses the sum, so its `alpha` corresponds to
  $n\lambda$ here (VERIFY (as of 2026-10) in the `Ridge` docs): a real source of "my numbers don't match".
- Putting the Jacobian's rows and columns the wrong way round, `(p, m)` instead of `(m, p)`. Rule: one
  *row per output*.
- $\partial(e^{x_1})/\partial x_1$ at 0 is $e^0 = 1$, not 0.

## Exercise 2 (code)

**Solution:** [`a4_2_gradients_jacobians_ex2.py`](a4_2_gradients_jacobians_ex2.py).

(a) $\mathbf{b}$ is added to every row: $R_{ic} = (\mathbf{X}\mathbf{W})_{ic} + b_c - Y_{ic}$. So $b_c$ appears in all $n$
entries of column $c$, each with slope 1, and the loss $\frac{1}{n}\sum_{i,c}R_{ic}^2$ gives
$\partial\mathcal{L}/\partial b_c = \frac{2}{n}\sum_i R_{ic}$. **Broadcasting forward means summing backward.** $\mathbf{W}$
is unchanged from the lesson: $\frac{2}{n}\mathbf{X}^\top\mathbf{R}$.

```python
R = X @ W + b - Y            # (n, k); b broadcasts over the n rows
dW = 2.0 / n * X.T @ R       # (d, k)
db = 2.0 / n * R.sum(axis=0) # (k,): b was copied to every row, so its gradient sums over rows
```

(b) With `y` of shape `(n, 1)` the residual becomes `(n, n)` and the "gradient" `(d, n)`. Nothing raises;
code that later does `w -= lr * g` would raise only because of the shape, and code that averages or
sums it "to make the shapes work" trains on garbage.

```text
== exercise 2a: bias by broadcasting ==
dW shape (3, 2)  rel error 6.1e-11
db shape (2,)  rel error 3.4e-11

== exercise 2b: the (n, 1) broadcasting bug ==
residual shape (5, 5)   gradient shape (3, 5)   expected (3,)
rel error of the mean over the extra axis vs the correct gradient: 1.3e+00
with y.ravel(): residual shape (5,)   rel error 7.2e-12
the assertion that catches it: AssertionError: residual has shape (5, 5), expected (5,)
```

The assertion: `assert r.shape == (n,), r.shape` right after computing the residual (or
`assert y.ndim == 1` when the data are loaded).

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| Derives $\partial\mathcal{L}/\partial\mathbf{b} = \frac{2}{n}\sum_i\mathbf{R}_{i,:}$ with a reason (b appears in every row) | 3 |
| Correct `dW` and `db` code with shapes asserted | 2 |
| Both checked against a numerical gradient with a relative error ≤ 1e-7 | 3 |
| Reproduces the `(n, 1)` bug and prints the residual and gradient shapes | 2 |
| Shows the wrong gradient disagrees with the numerical gradient of the correct loss | 2 |
| A shape assertion placed where it catches the bug | 2 |

**Likely mistakes:**

- `db = 2 / n * R.sum()`: summing over both axes gives a scalar, not `(k,)`. `R.sum(axis=0)` sums over
  rows (examples) and keeps one entry per output.
- `db = 2 / n * R.mean(axis=0)`: an extra division by $n$.
- Checking the buggy gradient against a numerical gradient of the *buggy* loss: both are computed
  from the same broadcast, the check passes, and the bug survives. Check against the loss as specified.
- "Fixing" the shape error with `g.mean(axis=1)` or `g[:, 0]` until it runs. If shapes do not match,
  the bug is upstream.

## Exercise 3 (think)

**Strong answer:** A constant factor $c > 0$ does keep the direction, so plain gradient descent with
learning rate $\eta$ on the wrong gradient is identical to the correct gradient with $c\eta$: the
teammate is right *only* in that narrow case, and the "right" learning rate silently changes. It
matters when the gradient is combined with anything else: a regularization term or a second loss
whose gradient is correct (the balance between the terms changes by $c$), gradient clipping at a fixed
norm, per-parameter factors that differ between layers (then the error is no longer one constant),
learning-rate values or schedules copied from a paper, and Adam-style optimizers, which mostly cancel
a global scale (so it hides, then reappears if the code is reused with SGD). A numerical gradient check
catches it immediately: the relative error is the same constant everywhere (0.5 for a missing 2). What
a check cannot catch is a correct gradient of the *wrong loss*: the checker compares the code's
gradient with the code's loss, so if the loss itself does not match the spec (a mean where the paper
sums, a missing term, a sign), both agree. That needs different tests: compute the loss by hand on a
tiny example and compare with the code, compare with a reference implementation (scikit-learn,
PyTorch's built-in loss) on the same inputs, and check known properties (loss is zero at a perfect
prediction, matches the closed-form optimum).

**Weak answer:** "It doesn't matter as long as the loss goes down." Weak because a decreasing loss
does not show the gradient is right (a wrong but descent-ish direction also lowers the loss), it
ignores how the factor interacts with regularization and other terms, and it has no plan to detect
either kind of bug.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Identifies the narrow case where it is equivalent (plain GD, learning rate absorbs $c$) | 2 |
| At least two cases where it matters (mixed terms or regularization, clipping, per-layer factors, copied hyperparameters, optimizer interactions) | 3 |
| States that a numerical check catches it, with the constant-relative-error signature | 2 |
| Explains that a check cannot catch a wrong loss, and proposes tests that can | 3 |

## Scoring

Total 38 pts. Pass ≥ 70% (27 pts). Topic ids to revisit for each lost criterion: exercise 1 → A4.2
"Four derivatives" and A2.4 (normal equations); exercise 2 → A4.2 "In practice" and A1.2
(broadcasting); exercise 3 → A4.2 "In practice" and A4.4 (how the learning rate scales the step).
