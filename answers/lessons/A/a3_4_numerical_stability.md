# Answer key: A3.4 Numerical stability

Lesson: [`lessons/A/a3_4_numerical_stability.md`](../../../lessons/A/a3_4_numerical_stability.md)

## Exercise 1 (by hand)

**Solution (a).** Columns $\mathbf{a}_1 = (1, 2, 2)$, $\mathbf{a}_2 = (4, 2, 5)$.

- $r_{11} = \lVert\mathbf{a}_1\rVert = \sqrt{1 + 4 + 4} = 3$, $\mathbf{q}_1 = (1, 2, 2)/3$.
- $r_{12} = \mathbf{q}_1\cdot\mathbf{a}_2 = (4 + 4 + 10)/3 = 6$. $\mathbf{v}_2 = \mathbf{a}_2 - 6\mathbf{q}_1 = (4 - 2, 2 - 4, 5 - 4) = (2, -2, 1)$.
- $r_{22} = \sqrt{4 + 4 + 1} = 3$, $\mathbf{q}_2 = (2, -2, 1)/3$. Check $\mathbf{q}_1\cdot\mathbf{q}_2 = (2 - 4 + 2)/9 = 0$ ✓.
- $\mathbf{Q} = \frac13\begin{pmatrix}1&2\\2&-2\\2&1\end{pmatrix}$ `(3, 2)`, $\mathbf{R} = \begin{pmatrix}3&6\\0&3\end{pmatrix}$.
- $\mathbf{Q}^\top\mathbf{y}$ for $\mathbf{y} = (1, 1, 1)$: $\big((1 + 2 + 2)/3,\ (2 - 2 + 1)/3\big) = (5/3, 1/3)$.
- Back substitution of $\mathbf{R}\mathbf{w} = (5/3, 1/3)$: $3w_2 = 1/3 \Rightarrow w_2 = 1/9$; $3w_1 + 6 \cdot \frac19 = \frac53 \Rightarrow 3w_1 = \frac53 - \frac23 = 1 \Rightarrow w_1 = \frac13$.
- Check with the normal equations: $\mathbf{A}^\top\mathbf{A} = \begin{pmatrix}9&18\\18&45\end{pmatrix}$, $\mathbf{A}^\top\mathbf{y} = (5, 11)$,
  $\det = 405 - 324 = 81$, $\mathbf{w} = \frac{1}{81}(45 \cdot 5 - 18 \cdot 11,\ -18 \cdot 5 + 9 \cdot 11) = \frac{1}{81}(27, 9) = (\frac13, \frac19)$ ✓.
  (Also $\mathbf{A}^\top\mathbf{A} = \mathbf{R}^\top\mathbf{R}$: $\begin{pmatrix}3&0\\6&3\end{pmatrix}\begin{pmatrix}3&6\\0&3\end{pmatrix} = \begin{pmatrix}9&18\\18&45\end{pmatrix}$ ✓.)

**Solution (b).** $\ell_{11} = \sqrt9 = 3$, $\ell_{21} = 3/3 = 1$, $\ell_{22} = \sqrt{5 - 1} = 2$:
$\mathbf{L} = \begin{pmatrix}3&0\\1&2\end{pmatrix}$, and $\mathbf{L}\mathbf{L}^\top = \begin{pmatrix}9&3\\3&1 + 4\end{pmatrix}$ ✓. For
$\begin{pmatrix}1&2\\2&1\end{pmatrix}$: $\ell_{11} = 1$, $\ell_{21} = 2$, then $\ell_{22} = \sqrt{1 - 4} = \sqrt{-3}$, which
is not real. The matrix is not positive definite (eigenvalues 3 and −1; for example
$\mathbf{w} = (1, -1)$ gives $\mathbf{w}^\top\mathbf{C}\mathbf{w} = 1 - 4 + 1 = -2 < 0$).

**Solution (c).** For a diagonal matrix the singular values are the absolute diagonal entries, so
$\kappa = 100/0.01 = 10^4$. Relative error in $\mathbf{x}$ up to $\kappa \times 10^{-6} = 10^{-2}$: about 1%,
roughly 2 correct digits instead of 6.

```text
== exercise 1 check ==
Q = [[0.3333, 0.6667], [0.6667, -0.6667], [0.6667, 0.3333]]  R = [[3.0, 6.0], [0.0, 3.0]]   (Q * 3 = [[1.0, 2.0], [2.0, -2.0], [2.0, 1.0]] )
Q^T y = [1.6667, 0.3333]  w = [0.3333, 0.1111]  normal equations give [0.3333, 0.1111]
cholesky([[9, 3], [3, 5]]) = [[3.0, 0.0], [1.0, 2.0]]
cholesky([[1, 2], [2, 1]]): not positive definite (pivot 1 is -3)
cond(diag(100, 0.01)) = 10000.0
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) $\mathbf{q}_1$, $r_{12}$, $\mathbf{v}_2$, $\mathbf{q}_2$ and $\mathbf{R}$ correct, orthogonality checked | 4 |
| (a) $\mathbf{Q}^\top\mathbf{y}$, back substitution, $\mathbf{w} = (1/3, 1/9)$ | 3 |
| (a) Normal-equations check | 1 |
| (b) $\mathbf{L} = \begin{pmatrix}3&0\\1&2\end{pmatrix}$ verified | 2 |
| (b) Breakdown located at $\ell_{22}$ and linked to "not positive definite" | 2 |
| (c) $\kappa = 10^4$ and about 1% (2 digits) | 2 |

**Likely mistakes:**

- Forgetting to normalize $\mathbf{v}_2$, so $\mathbf{Q}$ has a column of length 3.
- Computing $r_{12}$ as $\mathbf{a}_1\cdot\mathbf{a}_2$ (= 18) instead of $\mathbf{q}_1\cdot\mathbf{a}_2$ (= 6).
- Back substitution started from the top. The last equation of an upper-triangular system has one
  unknown; start there.
- In (c), taking the condition number as the largest entry (100) or as a ratio of determinants. For a
  diagonal matrix it is largest over smallest absolute entry.

## Exercise 2 (code)

**Solution:** [`a3_4_numerical_stability_ex2.py`](a3_4_numerical_stability_ex2.py).

```python
def ridge_cholesky(X: FloatArray, y: FloatArray, lam: float) -> FloatArray:
    """Minimize |y - X w|^2 + lam |w|^2 by Cholesky on X^T X + lam I. X (n, d), y (n,) -> w (d,)."""
    A = X.T @ X + lam * np.eye(X.shape[1])  # (d, d), symmetric positive definite for lam > 0
    L = cholesky(A)
    return back_substitution(L.T, forward_substitution(L, X.T @ y))
```

```text
== exercise 2: ridge regression by Cholesky ==
cond(X) = 2.14e+04   cond(X^T X) = 4.58e+08
  lambda  cond(X^T X + lam I)         w0        w1  w0 + w1  matches sklearn
   1e-08             4.47e+08   163.1552 -161.1535   2.0017  True
   1e-04             1.85e+06     1.6742    0.3284   2.0026  True
   1e-02             1.86e+04     1.0080    0.9945   2.0025  True
   1e+00             1.87e+02     0.9938    0.9936   1.9874  True
   1e+01             1.96e+01     0.9302    0.9302   1.8604  True
plain least squares (lstsq): w0, w1 = [167.1426, -165.1409]  w0 + w1 = 2.0017
other weights at lambda = 1e-2: [-2.0029, 0.5078, 3.0035]  true: [-2.0, 0.5, 3.0]
```

Columns 0 and 1 differ by about $10^{-4}$, so the data pins down $w_0 + w_1$ (the effect of "the shared
feature") but says almost nothing about $w_0 - w_1$ (the effect of the tiny difference between them). In
SVD terms, the direction $(1, -1, 0, 0, 0)/\sqrt2$ has a singular value of only about $10^{-4}\sqrt{n/2} \approx 7 \times 10^{-4}$, and the
least-squares solution divides the noise along that direction by it: hence $w_0 \approx 167$ and
$w_1 \approx -165$, numbers set by the noise, which would change completely with a new noise draw. Their sum
is stable (about 2.00, the true $1 + 1$) because it lies along a well-conditioned direction.

Adding $\lambda\mathbf{I}$ adds $\lambda$ to every eigenvalue of $\mathbf{X}^\top\mathbf{X}$ (A3.1, exercise 1c), which lifts the
tiny one and drops $\kappa$ from $4.6 \times 10^8$ to $1.9 \times 10^4$ at $\lambda = 10^{-2}$. The penalty
$\lambda\lVert\mathbf{w}\rVert^2$ prefers splitting the shared effect equally, so $w_0 \approx w_1 \approx 1$. The price is
bias: at $\lambda = 10$ every weight is pulled toward 0 and $w_0 + w_1$ falls to 1.86. Somewhere around
$10^{-2}$ to 1 the conditioning is fixed and the bias is still small; in practice $\lambda$ is chosen by
cross-validation, and features are standardized first so one $\lambda$ treats them all alike
([B2.2](../../../lessons/B/b2_2_regularization.md)).

**Rubric (15 pts):**

| Criterion | Points |
|---|---|
| Data generated as specified (`default_rng(0)`, order X, column noise, y noise); shapes stated | 1 |
| Own Cholesky plus forward and back substitution; no `inv`, no `np.linalg.solve` | 4 |
| Table of $\lambda$, $\kappa$, $w_0$, $w_1$, $w_0 + w_1$; matches `Ridge(..., solver="cholesky")` with `assert_allclose` | 3 |
| Plain `lstsq` fit shown | 1 |
| Explains huge opposite-sign $w_0, w_1$ as noise amplified along the near-null direction, and why the sum is stable | 3 |
| Explains how $\lambda$ lowers $\kappa$ (shifts eigenvalues) and introduces bias, with a sensible range | 3 |

**Likely mistakes:**

- Comparing with `Ridge()` without `fit_intercept=False`: scikit-learn then centers the data and fits an
  intercept, so the weights differ and the check fails for a reason unrelated to the code.
- Using $\lambda = 0$ in `ridge_cholesky`. At $\kappa \approx 4.6 \times 10^8$ Cholesky still succeeds here, but
  on exactly dependent columns it raises; that is the case ridge exists for.
- Concluding "large $\lambda$ is safest". Conditioning keeps improving, but the bias grows; the sum
  column shows the cost.
- Reading $w_0 = 163$ as "feature 0 matters a lot". With collinear features, individual coefficients are
  not interpretable at all.

## Exercise 3 (think)

**Strong answer:** Three conditioning problems stacked. First, scale: `annual_revenue` around $10^7$ and
`conversion_rate` around $10^{-2}$ make the columns of $\mathbf{X}$ differ in size by $10^9$, so $\kappa(\mathbf{X})$ is
huge even if the features are unrelated. Second, near-perfect collinearity: `marketing_spend_inr` is
`marketing_spend_usd` times an exchange rate that barely moves, so the two columns are nearly dependent
(exactly dependent if a fixed rate was used); $\mathbf{X}^\top\mathbf{X}$ is nearly singular, the data determines
only a combination of the two coefficients, and each refit lands somewhere different along the
near-null direction: sign flips week to week, as in exercise 2. Third, precision: if the fit uses the
normal equations, it works with $\kappa^2$; float32 has $\varepsilon \approx 10^{-7}$, so any $\kappa^2$ above about
$10^7$ leaves no correct digits, pivots round to zero, and a division by zero or overflow produces
`inf` and then `nan`.

Checks, in order: (1) the feature list and a correlation matrix or `np.linalg.matrix_rank(X)` and
`np.linalg.cond(X)` on the raw and on the standardized features; (2) how the model is fitted (hand-written
normal equations or `inv`? which library solver?); (3) the dtype on each platform, and where the first
`nan` appears; (4) whether the coefficient changes coincide with exchange-rate changes or data refreshes.

Changes: drop one of the two spend columns (or convert to one currency); standardize features
inside a pipeline fitted on training data only ([B1.4](../../../lessons/B/b1_4_data_leakage.md));
fit with `lstsq` / QR or a library estimator, never the normal equations; add ridge regularization if
other features remain correlated; fit closed-form models in float64 (the GPU buys nothing for a model
this size); and add monitoring: alert when $\kappa(\mathbf{X})$ or the week-to-week coefficient change exceeds a
threshold, and fail the job on any `nan`.

**Weak answer:** "The GPU is buggy; run it on the CPU. The coefficients change because the market
changes." Weak because it blames the hardware and the business for what conditioning explains, and the
CPU float64 run would still flip signs because the two spend columns are collinear.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Identifies feature scaling as a cause of high $\kappa$ | 2 |
| Identifies the USD/INR collinearity and links it to unstable, sign-flipping coefficients | 3 |
| Explains the float32 `nan` via $\varepsilon \approx 10^{-7}$ and (if normal equations) $\kappa^2$ | 2 |
| Ordered checks and concrete fixes (drop a column, standardize, stable solver, ridge, float64, monitoring) | 3 |

## Scoring

Total 39 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → A3.4
"The math" (QR, Cholesky, condition number) and A2.3 (back substitution); exercise 2 → A3.4 "In practice",
A3.2 (singular values and near-null directions), later B2.2; exercise 3 → A3.4 "Intuition" and
"In practice", A2.3 (dependent columns).
