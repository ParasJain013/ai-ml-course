# Answer key: B1.5 Linear regression from scratch versus scikit-learn

Lesson: [`lessons/B/b1_5_linear_regression.md`](../../../lessons/B/b1_5_linear_regression.md)

## Exercise 1 (by hand)

**Solution (a).** $\bar{x} = 10/5 = 2$, $\bar{y} = 15/5 = 3$. Deviations $dx = -2, -1, 0, 1, 2$ and $dy = -2, 0, -1, 2, 1$. $\sum dx\,dy = 4 + 0 + 0 + 2 + 2 = 8$,
$\sum dx^2 = 4 + 1 + 0 + 1 + 4 = 10$, so $w = 0.8$ and $b = 3 - 0.8 \times 2 = 1.4$. Predictions $1.4, 2.2, 3.0, 3.8, 4.6$; residuals $-0.4, 0.8, -1.0, 1.2, -0.6$
(sum 0). $\mathrm{SSE} = 0.16 + 0.64 + 1 + 1.44 + 0.36 = 3.6$, $\mathrm{SST} = 4 + 0 + 1 + 4 + 1 = 10$, $R^2 = 1 - 3.6/10 = 0.64$.

**(b)** $\mathbf{A}^\top\mathbf{A} = \begin{pmatrix} 5 & 10 \\ 10 & 30 \end{pmatrix}$ ($n$, $\sum x$; $\sum x$, $\sum x^2$) and $\mathbf{A}^\top\mathbf{y} = (15, 38)$ ($\sum y$, $\sum xy = 0 + 3 + 4 + 15 + 16$). Check:
$5(1.4) + 10(0.8) = 15$ and $10(1.4) + 30(0.8) = 38$.

**(c)** $\mathbf{H} = \frac25\mathbf{A}^\top\mathbf{A} = \begin{pmatrix} 2 & 4 \\ 4 & 12 \end{pmatrix}$. Trace 14, determinant $24 - 16 = 8$, so $\lambda^2 - 14\lambda + 8 = 0$ and $\lambda = 7 \pm \sqrt{41}$: 0.597 and 13.403.
$\kappa = 13.403/0.597 = 22.5$. Stable only for $\eta < 2/13.403 = 0.149$.

**(d)** $\sigma_x = \sqrt{10/5} = \sqrt2 = 1.414$; $z = dx/\sqrt2 = -1.414, -0.707, 0, 0.707, 1.414$; $\sum z = 0$ and $\sum z^2 = 5$, so $\mathbf{H} = 2\mathbf{I}$. $\sum z_iy_i = \sum dx_i\,y_i/\sqrt2 = 8/1.414 = 5.657$.
Gradient at $\mathbf{0}$: $-\frac25(15, 5.657) = (-6, -2.263)$. One step with $\eta = 0.5$: $\boldsymbol{\theta} = (3, 1.131)$, where the gradient is zero. Back to raw units:
$w = 1.131/1.414 = 0.8$, $b = 3 - 0.8 \times 2 = 1.4$.

```text
== exercise 1 check ==
(a) sum dx*dy = 8.0  sum dx^2 = 10.0  slope 0.8000  intercept 1.4000
    residuals [-0.4, 0.8, -1.0, 1.2, -0.6]  SSE 3.6000  SST 10.0000  R^2 0.6400
(b) A^T A = [[5.0, 10.0], [10.0, 30.0]]  A^T y = [15.0, 38.0]  solve -> [1.4, 0.8]
(c) Hessian [[2.0, 4.0], [4.0, 12.0]]  eigenvalues [0.5969, 13.4031] (7 -/+ sqrt(41))  condition number 22.46  max stable lr 0.1492
(d) sd 1.4142  z [-1.4142, -0.7071, 0.0, 0.7071, 1.4142]  gradient at 0 [-6.0, -2.2627]  after one step [3.0, 1.1314]  gradient there [0.0, 0.0]
    back to raw: slope 0.8000  intercept 1.4000
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Slope, intercept, residuals summing to zero, SSE, SST and $R^2$ | 4 |
| (b) Both matrices and the check | 2 |
| (c) Hessian with the $\frac2n$ factor, eigenvalues from trace and determinant, $\kappa$, largest stable rate | 4 |
| (d) Standardized values, the gradient, the step, and the conversion back | 4 |

**Likely mistakes:**

- Forgetting the $\frac{2}{n}$ in the Hessian and using $\mathbf{A}^\top\mathbf{A}$: the eigenvalues come out 2.5 times larger and the stable rate 2.5
  times smaller. $\kappa$ is unaffected, which is why it is the number to remember.
- Standardizing with $n - 1$ ($\sigma = 1.581$): then $\sum z^2 = 4$, $\mathbf{H}$ is $1.6\mathbf{I}$ and a step of 0.5 overshoots slightly. The lab and
  `StandardScaler` divide by $n$.
- Converting back with $w_z \times \sigma$ instead of $w_z/\sigma$: a weight per standard deviation is larger than a weight per unit when $\sigma > 1$.
- Writing $\mathbf{A}^\top\mathbf{y}$ as $(\sum y, \sum x)$: the second entry is $\sum x_iy_i$.

## Exercise 2 (code)

**Solution:** [`b1_5_linear_regression_ex2.py`](b1_5_linear_regression_ex2.py).

```text
== exercise 2: data ==
train (266, 10)  val (88, 10)  without a copy: bmi weight 5.5932  val R^2 0.4655
near copy: corr(bmi, bmi + noise) on train = 0.99999720

== exercise 2: four fits ==
(i) exact copy: condition number of A^T A 1.424e+18   rank 11 of 12
  solve              bmi weights     -55.8862      61.4795   sum   5.5932   val R^2 0.4655
  lstsq              bmi weights       2.7966       2.7966   sum   5.5932   val R^2 0.4655
  LinearRegression   bmi weights       2.7966       2.7966   sum   5.5932   val R^2 0.4655
  GD from zero       bmi weights       2.7966       2.7966   sum   5.5932   val R^2 0.4655   (7768 steps, smallest Hessian eigenvalue 1.35e-16)
  pinv (min-norm)    bmi weights       2.7966       2.7966
(ii) near copy: condition number of A^T A 1.506e+09   rank 12 of 12
  solve              bmi weights    -320.0582     325.6215   sum   5.5633   val R^2 0.4542
  lstsq              bmi weights    -320.0582     325.6215   sum   5.5633   val R^2 0.4542
  LinearRegression   bmi weights    -320.0582     325.6215   sum   5.5633   val R^2 0.4542
  GD from zero       bmi weights      -1.4548       7.0487   sum   5.5938   val R^2 0.4653   (20000 steps, smallest Hessian eigenvalue 5.45e-06)
```

**Why the predictions agree while the weights do not.** With two identical columns the prediction depends only on the
*sum* of their weights: $w_a x + w_b x = (w_a + w_b)x$. Every pair with $w_a + w_b = 5.5932$ gives exactly the same predictions, so there are
infinitely many least-squares solutions and the data cannot choose between them. All four methods find the right sum and
the same validation $R^2$; they differ only in how they split it.

- **`solve`** on a matrix that is singular in exact arithmetic did not raise here: rounding made $\mathbf{A}^\top\mathbf{A}$ look barely invertible
  (condition number $1.4 \times 10^{18}$, past the $10^{16}$ that double precision can resolve), and it returned an arbitrary split, −55.9 and +61.5.
  On another machine or data order it could raise `LinAlgError` or give a different split. A singular-looking system
  that does not raise is the dangerous case.
- **`lstsq`, `LinearRegression` and `pinv`** return the **minimum-norm** solution: of all the solutions, the one with the
  smallest $\lVert\boldsymbol{\theta}\rVert$, which splits the weight equally, 2.80 and 2.80.
- **Gradient descent from zero** converges to the same minimum-norm solution. Reason: every gradient,
  $\frac{2}{n}\mathbf{A}^\top(\cdot)$, is a combination of the rows of $\mathbf{A}$, so it has zero component along the direction $(0, \dots, 1, -1)$ that changes the
  split without changing predictions. Starting from 0 (equal split), gradient descent never moves along that direction.
  (Standardizing does not change this: the two columns standardize to the same $z$, and dividing equal weights by the
  same $\sigma$ keeps them equal.)

**Near copy.** Now the columns differ by noise of 0.01 kg/m², so the solution is unique, but it is decided by that noise:
the closed-form methods fit the noise with weights of −320 and +326, whose difference multiplies the tiny noise column. Validation $R^2$
drops from 0.4655 to 0.4542 because the noise on validation rows is different. Gradient descent from zero, stopped after
20,000 steps, has barely moved along the flat direction (its Hessian eigenvalue is $5 \times 10^{-6}$), so it sits near the equal split and
scores 0.4653. That is early stopping acting as regularization: it keeps weights small in the directions the data
barely supports. Ridge regression (B2.2) does the same on purpose.

One implementation detail matters: with $\lambda_{\min}$ near zero the lab's "best" rate $2/(\lambda_{\min} + \lambda_{\max})$ is essentially $2/\lambda_{\max}$, the edge
of stability, and the steepest direction then shrinks by only $1 - 2/\kappa$ per step. The solution uses $1/\lambda_{\max}$, which kills the steep
directions quickly. Accept either if the student notices and handles it.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Both cases built on the lab's split; noise with `SEED = 0` | 2 |
| Four methods fitted per case; bmi weights, condition number and validation $R^2$ reported | 5 |
| Explains that only the sum of the two weights is determined, so predictions agree | 3 |
| Identifies the minimum-norm solution (lstsq/pinv) and why GD from zero reaches it | 3 |
| Near copy: weights fit the noise; the early-stopped GD stays near the equal split and does slightly better | 3 |

**Likely mistakes:**

- Treating `solve` returning numbers as proof that the system is fine. Check the condition number or the rank.
- Concluding gradient descent "failed" in case (ii) because its weights differ from `LinearRegression`: it is not
  converged along a direction that carries almost no signal, and its predictions are as good or better.
- Standardizing the copied columns separately with different statistics, or with statistics from all rows (B1.4).
- Reading the −320/+326 weights as two strong, opposite effects of body mass.

## Exercise 3 (think)

**Strong answer:** A raw weight is the change in predicted progression per one unit of that feature *with every other
feature in the model held fixed*, in this sample of 442 patients. Neither statement follows from it.

- `s1` (total cholesterol) is strongly correlated with `s2` (LDL, 0.88 on the training rows) and is related to `s3` (HDL) and
  `s4` (total/HDL ratio). "Raise `s1` by one with `s2`, `s3` and `s4` fixed" barely describes a real patient: total cholesterol is
  largely made of LDL and HDL, so the weight describes a leftover combination, and its opposite sign to `s2` reflects the
  correlation (they share credit), not protection. Small changes to the data or to the feature set can flip it; the
  lab showed that the bmi weight halves when other features are added.
- The weights carry no uncertainty: with 266 training rows and correlated serum measurements, the `s1` weight may not be
  distinguishable from zero.
- `sex` is coded 1/2 and the dataset documentation does not say which group is which; the weight says that, among patients
  with the same values of everything else, those coded 2 had predictions 26.7 lower. That is an association in an observational
  sample, not an effect of sex, and its group label must be confirmed before it is read at all.
- Neither weight is causal: the model was fitted to predict, with no design that separates cause from confounding; model explanations (B3.4) share this limit.

What to show instead: the model's validation error next to the baselines (it explains about 47% of the variance); weights per
standard deviation with bootstrap intervals, so instability is visible; the correlation matrix of the serum measurements; predictions
for a few realistic patient profiles; and, if the question is "does lowering cholesterol slow progression", say that this
model cannot answer it and that it needs a study designed for it.

**Weak answer:** "The weight on s1 is negative, so cholesterol is protective, but the effect is small because −1.53 is
close to zero." Weak because it accepts the causal reading, compares a raw-unit weight by its size, and misses the
correlation that makes the sign unreliable.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| States what a weight means: per unit, others held fixed, association in this data | 2 |
| Explains the `s1`/`s2` correlation and why it makes the sign unreliable | 3 |
| Notes the undocumented sex coding and that the weight is not an effect of sex | 2 |
| A concrete alternative to show (uncertainty, standardized weights, profiles, validation error) | 3 |

## Scoring

Total 40 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → B1.5 "The math" and A4.4
(condition number and learning rate); exercise 2 → A2.4 (least squares and projections), A3.4 (numerical stability) and
B2.2 (regularization); exercise 3 → B1.5 "Reading the weights" and B3.4.
