# Answer key: A4.4 Gradient descent

Lesson: [`lessons/A/a4_4_gradient_descent.md`](../../../lessons/A/a4_4_gradient_descent.md)

## Exercise 1 (by hand)

**Solution (a).** $\mathcal{L}(w) = 2(w + 1)^2 = \frac{a}{2}(w - w^\star)^2$ with $\frac{a}{2} = 2$, so $a = 4$ and $w^\star = -1$.
(Equivalently: $\mathcal{L}' = 4(w + 1)$, $\mathcal{L}'' = 4$.) Converges for $0 < \eta < 2/a = 0.5$; one step at
$\eta = 1/a = 0.25$. The error $e = w - (-1)$ starts at $e_0 = 2$ and is multiplied by $1 - 4\eta$:

- $\eta = 0.1$, factor $0.6$: $e = 1.2, 0.72, 0.432$, so $w = 0.2, -0.28, -0.568$. Smooth approach from one side.
- $\eta = 0.4$, factor $-0.6$: $e = -1.2, 0.72, -0.432$, so $w = -2.2, -0.28, -1.432$. Same distances,
  alternating sides of $-1$: overshoot and oscillation, still converging.

**Solution (b).** $\nabla\mathcal{L} = \mathbf{H}\mathbf{w}$ and the eigenvalues of a diagonal matrix are its diagonal,
4 and 1. Largest stable $\eta = 2/4 = 0.5$. Best fixed $\eta = 2/(1 + 4) = 0.4$; factors $1 - 0.4 \times 4 = -0.6$ and
$1 - 0.4 \times 1 = 0.6$, both of size $0.6 = \frac{\kappa - 1}{\kappa + 1}$ with $\kappa = 4$. Steps: $0.6^k = 10^{-6}$
gives $k = \frac{\ln 10^{-6}}{\ln 0.6} = \frac{-13.82}{-0.511} = 27.05$, so **28 steps**.

**Solution (c).** With 400, $\kappa = 400$: the steep direction forces $\eta < 2/400$, which moves the
gentle direction (curvature 1) by less than 0.5% per step, so the factor becomes $\frac{399}{401} = 0.995$ and
the step count grows roughly in proportion to $\kappa$ (28 → about 2,764).

```text
== exercise 1 check ==
(a) lr = 0.1: factor 0.6   w = [0.2, -0.28, -0.568]
(a) lr = 0.4: factor -0.6   w = [-2.2, -0.28, -1.432]
(b) lr_max = 0.5   lr_best = 0.4   factor = 0.6   steps for 1e-6 = 28
(c) H = diag(40, 1): factor 0.9512   steps for 1e-6 = 277
(c) H = diag(400, 1): factor 0.9950   steps for 1e-6 = 2764
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) $a = 4$, range $(0, 0.5)$, one-step $\eta = 0.25$ | 3 |
| (a) Both sequences correct, with the monotone versus oscillating description | 3 |
| (b) $\eta_{\max} = 0.5$, $\eta^\star = 0.4$, factor $0.6$ | 3 |
| (b) 28 steps, with the logarithm shown | 2 |
| (c) The steep direction caps $\eta$, so the gentle one crawls; steps $\propto \kappa$ | 3 |

**Likely mistakes:**

- Reading $a = 2$ off $2(w + 1)^2$ (forgetting the $\frac{a}{2}$ convention), which gives the range
  $(0, 1)$: at $\eta = 0.6$ that diverges. Always take $a = \mathcal{L}''$.
- Taking the step count as $\ln(10^{-6})/\ln(0.6)$ rounded *down* to 27: after 27 steps the factor is
  $1.0 \times 10^{-6}$ only to two digits, still slightly above. Round up.
- In (b), choosing $\eta = 1/\lambda_{\max} = 0.25$ "to kill the steep direction": it does, but leaves the
  gentle factor at $0.75$, which needs 49 steps.

## Exercise 2 (code)

**Solution:** [`a4_4_gradient_descent_ex2.py`](a4_4_gradient_descent_ex2.py). The best fixed rate comes from
the Hessian's eigenvalues, and standardized weights are converted back with
$w_0 = a - \sum_j b_j\mu_j/s_j$, $w_j = b_j/s_j$:

```text
== exercise 2: raw features ==
condition number = 108.1   lr = 0.02129   steps = 1175   w = [1.1307, -1.9557, 0.5273]
matches lstsq to 4 decimals: OK

== exercise 2: standardized features ==
condition number = 2.3   lr = 0.50000   steps = 22   w (raw scale) = [1.1307, -1.9557, 0.5273]
matches lstsq to 4 decimals: OK
correlation of x and x^2 on this data = 0.3893
```

$\kappa$ is not 1 after standardizing because $x$ and $x^2$ are correlated (0.39 here: on $[-3, 4]$ larger
$x$ tends to have larger $x^2$). Standardizing makes the columns orthogonal to the ones column and gives
them equal length, but it cannot make them orthogonal to *each other*; the Hessian's off-diagonal
entry is $2 \times$ their correlation, and the eigenvalues of
$2\begin{pmatrix}1&\rho\\\rho&1\end{pmatrix}$ are $2(1 \pm \rho)$, so $\kappa = \frac{1 + 0.39}{1 - 0.39} \approx 2.3$ ✓.
(Decorrelating them is what PCA whitening does, [B5.4](../../../lessons/B/b5_4_pca.md).) Learners' exact
numbers depend on their draw order; the pattern (κ of order 100 versus about 2, steps in the
thousands versus tens, both matching `lstsq`) is what is graded.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Data generated as specified, design matrix `(40, 3)` | 1 |
| Vectorized GD with gradient-norm stopping and an iteration cap | 3 |
| Hessian and condition number computed; best fixed $\eta$ used for both fits | 3 |
| Step counts reported for both | 2 |
| Standardized weights converted back correctly (intercept term includes all features) | 3 |
| Both checked against `lstsq` to 4 decimals with `assert_allclose` | 2 |
| Explains $\kappa \ne 1$ by the correlation between $x$ and $x^2$ | 2 |

**Likely mistakes:**

- Standardizing the ones column too (its standard deviation is 0: division by zero, `nan`).
- Converting the intercept back as `w_s[0]` alone, forgetting $-\sum b_j\mu_j/s_j$.
- Using the same learning rate for raw and standardized features: either the raw fit diverges or the
  standardized one is needlessly slow.
- Stopping on a fixed number of steps and declaring a match that only holds to 2 decimals.

## Exercise 3 (think)

**Strong answer:** The Hessian of the MSE is $\frac{2}{n}\mathbf{X}^\top\mathbf{X}$. A column with values around $5 \times 10^4$
puts an entry of order $2 \times (5 \times 10^4)^2 = 5 \times 10^9$ on the diagonal, so $\lambda_{\max}$ jumps to about
$5 \times 10^9$ and the largest stable learning rate $2/\lambda_{\max}$ drops to about $4 \times 10^{-10}$. With
$\eta = 0.01$, the error along that eigen-direction is multiplied by about $1 - 0.01 \times 5 \times 10^9$, i.e.
$-5 \times 10^7$, every step: it overflows to `inf` within a few dozen steps and becomes `nan`. Even if
$\eta$ were lowered to be stable, $\kappa$ would be about $10^9$ or more and the other weights would barely
move. Changes: (1) standardize every numeric feature inside the training pipeline, with the scaler fit
on training data only and saved with the model, so a new feature cannot enter unscaled; (2) make
training fail loudly and early: check for non-finite loss or weights every step and abort with a
message naming the step and learning rate, alert when the final gradient norm or validation loss is
outside expected bounds, and do not deploy the model from a failed run. Also acceptable: compute the
condition number or feature scales as a data check before training; use a solver less sensitive to
scaling for this linear model (closed form or L-BFGS).

**Weak answer:** "Lower the learning rate until it stops producing `nan`." Weak because it treats the
symptom: the stable rate is now about $10^{-10}$, at which every other weight trains millions of times
slower, so the model silently underfits; and the next unscaled feature breaks it again.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Links the large feature to a huge $\lambda_{\max}$ of $\frac{2}{n}\mathbf{X}^\top\mathbf{X}$ | 3 |
| Explains divergence: $\eta > 2/\lambda_{\max}$, factor $\lvert 1 - \eta\lambda\rvert \gg 1$ | 2 |
| Scaling inside the pipeline (fit on training data, saved with the model) | 3 |
| A loud failure: non-finite checks, alerts, no deploy on failure | 2 |

## Scoring

Total 40 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → A4.4
"The math" (one-parameter bowl, eigen-directions) and A3.1 (eigenvalues); exercise 2 → A4.4 "From
scratch" (standardizing) and A3.3 (correlation); exercise 3 → A4.4 "In practice" and B1.4 (fit the
scaler on training data).
