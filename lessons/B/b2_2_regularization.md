# B2.2 Regularization: L1, L2 and elastic net, read as MAP priors

**Core** · Part B, week 2 (course week 08) · ~2 h · Lab:
[`labs/B/b2_2_regularization.py`](../../labs/B/b2_2_regularization.py)

## Why it matters

Unpenalized fits break in three ways you have already met. Correlated features get huge weights of opposite sign
(`s1` −52 and `s2` +36 per standard deviation in B1.5). Separable classes send logistic-regression weights to infinity
(B2.1). And with more features than rows, least squares has infinitely many perfect fits. **Regularization** adds a
penalty on the size of the weights. It trades a little training fit for weights that are stable, finite, and better on
new data. The L1 version (the **lasso**) also sets some weights to *exactly* zero, so it selects features. Every
production linear model is regularized (scikit-learn's `LogisticRegression` is by default). The same L2 penalty
reappears as weight decay in every neural network (C3.1, C3.5).

## Prerequisite check

- Can you explain MAP estimation and why a Gaussian prior on the weights gives the ridge penalty $\lambda\lVert\mathbf{w}\rVert^2$? If not, revisit
  [A5.5](../A/a5_5_mle_map.md).
- Can you fit linear regression with the normal equations and explain what collinear features do to the weights? If
  not, revisit [B1.5](b1_5_linear_regression.md).
- Can you write the log-loss gradient and say what perfect separation does to logistic regression? If not, revisit
  [B2.1](b2_1_logistic_regression.md).
- Can you say what the condition number of $\mathbf{X}^\top\mathbf{X}$ measures? If not, revisit [A3.4](../A/a3_4_numerical_stability.md).
- Do you know why features must be standardized with training statistics? If not, revisit [B1.3](b1_3_feature_engineering.md).

## Intuition

A least-squares fit chases every wiggle in the training data. When two features carry nearly the same information
(total cholesterol and LDL), the data cannot tell how to split the credit between them. A weight of −52 on one and +36
on the other fits barely better than −1.6 and −2.5. Those huge, opposite weights come from noise in 266 rows, and they
amplify noise in new rows.

Regularization adds a cost for large weights to the training loss:

$$\text{objective} = \text{data loss} + \lambda \times \text{size of } \mathbf{w}.$$

The strength $\lambda$ sets the exchange rate. At $\lambda = 0$ you get the ordinary fit. As $\lambda \to \infty$ every weight goes to zero and the model predicts
the mean. In between, the model spends weight only where the data pays for it. You choose $\lambda$ on validation data.

Two ways to measure "size" behave very differently:

- **L2 (ridge):** $\lVert\mathbf{w}\rVert_2^2 = \sum_j w_j^2$. The penalty's slope, $2\lambda w_j$, shrinks to nothing as $w_j$ approaches 0. Its push towards zero gets weaker
  the closer you get, so weights shrink smoothly but never reach exactly 0. Correlated features end up sharing the credit.
- **L1 (lasso):** $\lVert\mathbf{w}\rVert_1 = \sum_j \lvert w_j\rvert$. The penalty's slope is $\pm\lambda$ whatever the size of the weight: a constant push towards zero. If the data's
  pull on a weight is weaker than $\lambda$, the weight stays at exactly 0. So the lasso selects features, and from a correlated
  group it tends to keep one and drop the rest.
- **Elastic net** mixes the two: some exact zeros from L1, and the sharing among correlated features from L2.

**The Bayesian reading (A5.5).** A penalty is a prior belief about the weights, written as a cost. L2 is a Gaussian prior:
"weights are probably small". L1 is a Laplace prior, which is sharply peaked at zero: "most weights are probably *exactly*
near zero, and a few may be large". Regularized fitting is MAP estimation; $\lambda$ says how strongly you believe the prior relative to the data.

> **Bridge:** regularization is like a cost on cloud resources in a Terraform plan. Without it, the optimizer provisions
> whatever fits the load best, including oversized instances for noise. With a price on size, it provisions only what
> pays for itself. L1 is a fixed fee per resource, so small resources get switched off entirely. L2 is a price that
> grows with size, so everything gets smaller but little is switched off. **Where it breaks:** cloud costs are real money,
> while $\lambda$ is a knob with no natural units, chosen by validation error. And the "resources" are weights on *standardized*
> features: without standardizing, the price depends on each feature's units, which is meaningless.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{Z}$ | standardized training features (each column mean 0, $\frac1n\sum_i z_{ij}^2 = 1$) | `(n, d)` |
| $\mathbf{y}$, $\bar{y}$ | targets and their training mean | `(n,)`, scalar |
| $\mathbf{w}$, $b$ | weights (penalized) and intercept (never penalized) | `(d,)`, scalar |
| $\lambda$ | regularization strength (`alpha` in scikit-learn) | scalar $\ge 0$ |
| $\rho$ | elastic-net mixing weight (`l1_ratio`): 1 = lasso, 0 = ridge | scalar in $[0, 1]$ |
| $\mathbf{r}$ | residuals $\mathbf{y} - \mathbf{Z}\mathbf{w} - b$ | `(n,)` |
| $a_j$, $c_j$ | for coordinate descent: $\frac1n\mathbf{z}_j^\top\mathbf{r}^{(-j)}$ and $\frac1n\mathbf{z}_j^\top\mathbf{z}_j$ ($= 1$ when standardized) | scalars |
| $\mathbf{r}^{(-j)}$ | partial residual: residuals of the model with feature $j$ removed | `(n,)` |
| $S(a, t)$ | soft-threshold $\operatorname{sign}(a)\max(\lvert a\rvert - t, 0)$ | scalar |
| $\mathbf{z}_j$ | column $j$ of $\mathbf{Z}$ | `(n,)` |

**The three objectives, exactly as scikit-learn writes them.** The scalings differ, so the same `alpha` means different
things in different classes (VERIFY as of 2026-10, scikit-learn 1.9.1):

$$\text{Ridge:}\quad \lVert\mathbf{y} - \mathbf{Z}\mathbf{w} - b\rVert^2 + \lambda\lVert\mathbf{w}\rVert_2^2$$

$$\text{Lasso:}\quad \frac{1}{2n}\lVert\mathbf{y} - \mathbf{Z}\mathbf{w} - b\rVert^2 + \lambda\lVert\mathbf{w}\rVert_1$$

$$\text{Elastic net:}\quad \frac{1}{2n}\lVert\mathbf{y} - \mathbf{Z}\mathbf{w} - b\rVert^2 + \lambda\rho\lVert\mathbf{w}\rVert_1 + \frac{\lambda(1 - \rho)}{2}\lVert\mathbf{w}\rVert_2^2$$

Ridge uses the *sum* of squared errors; the lasso and elastic net use half the *mean*. So `Ridge(alpha=10)` and
`ElasticNet(alpha=10, l1_ratio=0)` are different models: the second equals `Ridge(alpha=10 * n)`.

**Where the penalties come from (MAP, A5.5).** With Gaussian noise, $-\log p(\mathbf{y} \mid \mathbf{w})$ is the squared error up to constants.
A Gaussian prior $w_j \sim \mathcal{N}(0, \tau^2)$ adds $-\log p(\mathbf{w}) = \sum_j w_j^2/(2\tau^2) + \text{const}$: the L2 penalty. A **Laplace prior** $p(w_j) = \frac{1}{2s}e^{-\lvert w_j\rvert/s}$ adds
$-\log p(\mathbf{w}) = \sum_j \lvert w_j\rvert/s + \text{const}$: the L1 penalty. Tighter priors (small $\tau$ or $s$) mean a larger $\lambda$. The Laplace density has a sharp
peak at 0 and heavier tails than the Gaussian, which matches "most weights are 0, a few are large".

**The intercept.** The penalty does not involve $b$, so the derivative of any of these objectives with respect to $b$ is a
positive constant times $-\sum_i r_i$ ($-2\sum_i r_i$ for ridge, $-\frac1n\sum_i r_i$ for the lasso). It is zero when the residuals sum to zero. Now
$\sum_i r_i = \sum_i y_i - nb - \sum_j w_j\sum_i z_{ij}$, and each column of $\mathbf{Z}$ sums to zero ($\sum_i z_{ij} = 0$), so $\sum_i r_i = n(\bar{y} - b)$. That vanishes exactly when $b = \bar{y}$, whatever $\mathbf{w}$ is. So we set $b = \bar{y}$ and fit $\mathbf{w}$ to the centred target
$\mathbf{y} - \bar{y}$. The intercept is not penalized: shifting every target by 100 should shift the intercept by 100, not fight a prior.

**Ridge in closed form.** The gradient of $\lVert\mathbf{y} - \bar{y} - \mathbf{Z}\mathbf{w}\rVert^2 + \lambda\lVert\mathbf{w}\rVert^2$ is $-2\mathbf{Z}^\top(\mathbf{y} - \bar{y} - \mathbf{Z}\mathbf{w}) + 2\lambda\mathbf{w}$. Setting it to zero:

$$(\mathbf{Z}^\top\mathbf{Z} + \lambda\mathbf{I}_d)\,\mathbf{w} = \mathbf{Z}^\top(\mathbf{y} - \bar{y}), \qquad \text{shapes } (d, d)(d,) = (d, n)(n,).$$

Adding $\lambda$ to the diagonal adds $\lambda$ to every eigenvalue of $\mathbf{Z}^\top\mathbf{Z}$ (if $\mathbf{Z}^\top\mathbf{Z}\mathbf{v} = \mu\mathbf{v}$, then $(\mathbf{Z}^\top\mathbf{Z} + \lambda\mathbf{I})\mathbf{v} = (\mu + \lambda)\mathbf{v}$). The smallest eigenvalue,
the direction where the data says almost nothing, gets lifted the most in relative terms. So the condition number
$\mu_{\max}/\mu_{\min}$ falls to $(\mu_{\max} + \lambda)/(\mu_{\min} + \lambda)$, and the system is solvable even when $d > n$ (where $\mu_{\min} = 0$).

**Lasso by coordinate descent.** The L1 penalty has a kink at zero, so there is no closed form and no gradient at
$w_j = 0$. But one coordinate at a time is easy. Hold every weight except $w_j$ fixed. The residual without feature $j$ is
$\mathbf{r}^{(-j)} = \mathbf{y} - \bar{y} - \sum_{k \ne j}\mathbf{z}_kw_k$. Expanding the squared error as a function of $w_j$ alone:

$$f(w_j) = \frac{1}{2n}\lVert\mathbf{r}^{(-j)} - \mathbf{z}_jw_j\rVert^2 + \lambda\lvert w_j\rvert = \frac{c_j}{2}w_j^2 - a_jw_j + \lambda\lvert w_j\rvert + \text{const},$$

with $a_j = \frac1n\mathbf{z}_j^\top\mathbf{r}^{(-j)}$ (how much feature $j$ alone would explain of what is left) and $c_j = \frac1n\mathbf{z}_j^\top\mathbf{z}_j = 1$. Minimize this one-dimensional function piece by piece:

- If $w_j > 0$, $\lvert w_j\rvert = w_j$, and the derivative $c_jw_j - a_j + \lambda = 0$ gives $w_j = (a_j - \lambda)/c_j$. That is positive only if $a_j > \lambda$.
- If $w_j < 0$, $\lvert w_j\rvert = -w_j$, and the same steps give $w_j = (a_j + \lambda)/c_j$, negative only if $a_j < -\lambda$.
- If $\lvert a_j\rvert \le \lambda$, neither case is consistent. The minimum is at the kink: $w_j = 0$. (Check: just right of zero the slope is $-a_j + \lambda \ge 0$,
  just left it is $-a_j - \lambda \le 0$, so $f$ goes up both ways.)

All three cases in one formula:

$$w_j \leftarrow \frac{S(a_j, \lambda)}{c_j}, \qquad S(a, t) = \operatorname{sign}(a)\max(\lvert a\rvert - t, 0).$$

The **soft-threshold** $S$ shrinks $a$ towards zero by $t$ and stops at zero. Cycle through $j = 1, \dots, d$ and repeat until no weight changes.
Each update can only lower the objective, which is convex, so this converges to the minimum. For the elastic net, the
same steps give $w_j \leftarrow S(a_j, \lambda\rho)/(c_j + \lambda(1 - \rho))$: the L1 part thresholds, the L2 part shrinks.

**The largest useful $\lambda$.** Start at $\mathbf{w} = \mathbf{0}$, so $\mathbf{r}^{(-j)} = \mathbf{y} - \bar{y}$. Every weight stays at zero if $\lvert a_j\rvert \le \lambda$ for all $j$, that is if
$\lambda \ge \lambda_{\max} = \max_j \lvert\mathbf{z}_j^\top(\mathbf{y} - \bar{y})\rvert/n$. Lasso paths start there.

**Why L1 makes zeros: the geometry.** Penalized fitting is equivalent to minimizing the loss subject to $\lVert\mathbf{w}\rVert \le t$ for some
budget $t$: read $\lambda$ as the Lagrange multiplier of that constraint (A4.5). Each $\lambda$ corresponds to a $t$, and a larger $\lambda$ to a smaller $t$. The loss contours are ellipses around the least-squares solution. The constrained solution
is where the smallest ellipse first touches the ball $\lVert\mathbf{w}\rVert \le t$. The L2 ball is round, so the touching point is almost never on an axis.
The L1 ball is a diamond with corners *on* the axes, and an ellipse coming from a generic direction usually hits a corner
first. A corner has some coordinates exactly zero. In $d$ dimensions the L1 ball has $2d$ corners and many edges with zeros, so
this gets more likely, not less.

**L2 logistic regression.** Add $\frac{\lambda}{2}\lVert\mathbf{w}\rVert^2$ to the mean log-loss of B2.1. The gradient gains $\lambda\mathbf{w}$ (and nothing for $b$):

$$\nabla_{\mathbf{w}} = \frac1n\mathbf{Z}^\top(\mathbf{p} - \mathbf{y}) + \lambda\mathbf{w}.$$

scikit-learn minimizes $C\sum_i\ell_i + \frac12\lVert\mathbf{w}\rVert^2$. Dividing by $Cn$ (which does not move the minimum) gives $\frac1n\sum_i\ell_i + \frac{1}{2Cn}\lVert\mathbf{w}\rVert^2$, so $\lambda = 1/(Cn)$. A larger $C$
means a weaker penalty. Even on separable data the penalized loss has a finite minimum, because the penalty grows
without bound as $\lVert\mathbf{w}\rVert$ grows.

## Worked example by hand

One feature, four points, both centred (so $b = 0$): $x = -2, -1, 1, 2$ and $y = -3, -1, 0, 4$.
$\sum xy = 6 + 1 + 0 + 8 = 15$ and $\sum x^2 = 4 + 1 + 1 + 4 = 10$. Least squares: $w = 15/10 = 1.5$.

**Ridge** (sum-of-squares form). With one feature the normal equation is a single number: $w = \sum xy/(\sum x^2 + \lambda)$. $\lambda = 5$: $w = 15/15 = 1.0$.
$\lambda = 10$: $w = 15/20 = 0.75$. Ridge divides by a larger number. The weight shrinks but never reaches 0 for any finite $\lambda$.

**Lasso** (half-mean form). $a = \sum xy/n = 15/4 = 3.75$ and $c = \sum x^2/n = 10/4 = 2.5$. $\lambda = 1$: $w = S(3.75, 1)/2.5 = 2.75/2.5 = 1.1$. $\lambda = 3.75$: $S(3.75, 3.75) = 0$, so $w = 0$. Any
$\lambda \ge 3.75$ gives exactly 0. The lasso *subtracts* a fixed amount, so once the data's pull ($a = 3.75$) is used up, the weight is gone.

```text
== worked example by hand ==
x = [-2.0, -1.0, 1.0, 2.0]   y = [-3.0, -1.0, 0.0, 4.0]   (both have mean 0, so b = 0)
sum xy = 15.0   sum x^2 = 10.0   least squares w = 1.5000
ridge, lambda =  5.0: w = sum xy / (sum x^2 + lambda) = 15.0/15.0 = 1.0000
ridge, lambda = 10.0: w = sum xy / (sum x^2 + lambda) = 15.0/20.0 = 0.7500
lasso: a = sum xy / n = 3.75   c = sum x^2 / n = 2.5
lasso, lambda =  1.0: w = S(a, lambda) / c = S(3.75, 1.0) / 2.5 = 1.1000
lasso, lambda = 3.75: w = S(a, lambda) / c = S(3.75, 3.75) / 2.5 = 0.0000
lasso, lambda =  4.0: w = S(a, lambda) / c = S(3.75, 4.0) / 2.5 = 0.0000
sklearn Ridge(alpha=5) w = 1.0000   Lasso(alpha=1) w = 1.1000
```

## Key plot

![Left: elliptical loss contours around the least-squares point (2, 0.6), an L1 diamond and an L2 circle of radius 1; the contours first touch the diamond at its corner (1, 0), where w2 is exactly zero, and touch the circle at (0.91, 0.42). Right: lasso weights for bmi, s5, s1 and s2 on the diabetes data against alpha on a log scale; s2 hits zero at alpha about 0.5, s1 at about 3, while bmi and s5 survive until alpha about 44](img/b2_2_l1_vs_l2.png)

Left: the same loss, two budgets of the same radius. The L2 solution keeps both weights; the L1 solution sits on a
corner with $w_2 = 0$ exactly. Right: on the diabetes data, as $\alpha$ grows the lasso drops `s2`, then `s1`, while `bmi` and `s5` survive
almost to the end. A ridge path (saved to `outputs/b2_2/ridge_path.png`) has no such kinks: every weight shrinks smoothly
and none reaches zero.

## From scratch in NumPy

**The data.** Diabetes again, split as in B1.1 and B1.5, standardized with training statistics. The target is centred, so
the intercept is the training mean:

```text
== data ==
train Z (266, 10)  val Z (88, 10)  (test 88 rows kept aside)
training target mean (the intercept) 155.1466
correlation of s1 and s2 on train: 0.8812
```

**Ridge.** The closed form, checked against `Ridge(alpha=lambda)` at three strengths:

```python
def fit_ridge(Z, y, lam):
    """Minimize |y - Z w - b|^2 + lam |w|^2 for standardized Z. Z (n, d), y (n,) -> (b, w (d,))."""
    b = float(y.mean())                                                    # intercept decouples
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - b))  # (d,)
    return b, w
```

```text
== from scratch: ridge ==
lambda    0.0: s1 -51.7839  s2  35.7439  bmi 23.8452  |w| 83.5362  val R^2 0.4655
lambda   10.0: s1 -11.5071  s2   4.0126  bmi 23.9076  |w| 43.0094  val R^2 0.4765
lambda  100.0: s1  -1.6118  s2  -2.5333  bmi 19.2470  |w| 32.0792  val R^2 0.4714
matches sklearn Ridge(alpha=lambda) at every lambda: OK
eigenvalues of Z^T Z: smallest 2.6358  largest 974.6076  condition number 369.8
with lambda = 10: condition number of Z^T Z + 10 I = 77.9
```

At $\lambda = 0$ this is B1.5's fit: `s1` and `s2` at −52 and +36. A penalty of 10 cuts them to −11.5 and +4.0. The other weights barely
move (bmi 23.85 to 23.91): ridge shrinks most where the data is least sure, along the low-eigenvalue direction ("s1 up, s2
down") that the correlated pair creates. Validation $R^2$ improves from 0.466 to 0.477. At $\lambda = 100$, `s1` and `s2` share the credit with the
same sign, as two measures of cholesterol should.

**Lasso and elastic net by coordinate descent.** One function covers both. It keeps the residual up to date, so each
coordinate update costs $O(n)$:

```python
for j in range(n_feat):
    r += Z[:, j] * w[j]                        # partial residual: the model without feature j
    a_j = Z[:, j] @ r / n_rows                 # how much feature j alone would explain
    w_new = soft_threshold(a_j, alpha * l1_ratio) / (col_sq[j] + alpha * (1 - l1_ratio))
    r -= Z[:, j] * w_new                       # put feature j back with its new weight
    w[j] = w_new
```

```text
== from scratch: lasso by coordinate descent ==
alpha  0.1: 1130 sweeps  non-zero 10/10  val R^2 0.4698  kept ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
alpha  1.0:   34 sweeps  non-zero  8/10  val R^2 0.4757  kept ['age', 'sex', 'bmi', 'bp', 's1', 's3', 's5', 's6']
alpha  5.0:   25 sweeps  non-zero  5/10  val R^2 0.4696  kept ['sex', 'bmi', 'bp', 's3', 's5']
matches sklearn Lasso(alpha) at every alpha: OK
lasso alpha 1 weights: {'age': 0.1367, 'sex': -10.9073, 'bmi': 24.6266, 'bp': 15.7308, 's1': -4.0702, 's2': 0.0, 's3': -9.9328, 's4': 0.0, 's5': 24.4881, 's6': 2.0861}
smallest alpha that zeroes every weight: max_j |z_j . (y - y_bar)| / n = 43.6294
```

- At $\alpha = 1$, two weights are exactly 0.0 (not $10^{-17}$): `s2` and `s4`. At $\alpha = 5$, a five-feature model is within 0.007 of the best lasso validation $R^2$ (0.476, below).
- Small $\alpha$ needs many sweeps (1,130): with little penalty the correlated features trade credit back and forth slowly.
  This is the same ill-conditioning that slowed gradient descent in B1.5.
- At $\alpha \ge 43.63$ every weight is zero, by the $\lambda_{\max}$ formula.

**What each penalty does with a correlated pair:**

```text
== from scratch: elastic net and correlated features ==
elastic net alpha 1, l1_ratio 0.5: non-zero 10/10   matches sklearn ElasticNet: OK
least squares                        s1  -51.7839   s2   35.7439
ridge, lambda 10                     s1  -11.5071   s2    4.0126
lasso, alpha 1                       s1   -4.0702   s2    0.0000
elastic net, alpha 1, l1_ratio 0.5   s1   -0.6807   s2   -1.9819
```

Least squares lets the pair cancel each other with huge weights. Ridge pulls them towards each other. The lasso keeps
one and drops the other; which one it keeps can flip with a small change in the data, so do not read the choice as
"s1 matters and s2 does not". The elastic net keeps both, small and with the same sign: the grouping behaviour that motivated it.

**Adding L2 to logistic regression.** On all 30 breast-cancer features, which B2.1 showed are separable, the penalized
loss has a finite minimum and gradient descent matches `LogisticRegression(C)` once $\lambda = 1/(Cn)$:

```text
== L2 logistic regression from scratch ==
C  0.1 -> lambda = 1/(C n) = 0.025000:  1903 steps  |w| 1.7618  max |p_mine - p_sklearn| 1.4e-07
C  1.0 -> lambda = 1/(C n) = 0.002500: 14431 steps  |w| 3.6049  max |p_mine - p_sklearn| 3.7e-07
L2 logistic regression matches LogisticRegression(C) on all 30 features: OK
L1 logistic regression (C 0.1, l1_ratio 1, saga): non-zero weights 9/30
```

The weaker penalty ($C = 1$) needs more steps, because a smaller $\lambda$ leaves the problem less well conditioned. With L1 instead
(`l1_ratio=1`, which needs the `saga` or `liblinear` solver), 21 of the 30 weights are exactly zero.

## With scikit-learn

The checks above already compare each from-scratch fit with `Ridge`, `Lasso`, `ElasticNet` and `LogisticRegression`. The
remaining question is how to choose $\lambda$. Here we scan it on the validation set (B4.3 does it properly, with cross-validation):

```text
== choosing lambda on the validation set ==
ridge: best val R^2 0.4788 at lambda 36.25   (lambda -> 0: 0.4656)
lasso: best val R^2 0.4764 at alpha 0.534 with 9 non-zero weights
order in which features enter the lasso as alpha falls: ['s5', 'bmi', 'bp', 's3', 'sex', 's6', 's1', 'age', 's4', 's2']
ridge weights at lambda = 1e4 (none exactly 0): [0.377, -0.038, 1.0873, 0.8158, 0.2626, 0.1923, -0.6989, 0.7484, 1.0384, 0.7373]
```

- Regularization helps a little here: validation $R^2$ from 0.466 to 0.479. With 266 rows and 10 features, plain least squares is
  not badly overfitted. The gain in *stability* of the weights is much larger than the gain in score. With more
  features than rows, or the 30 separable features of B2.1, the score gain would be large too.
- 0.4788 against 0.4764 is not evidence that ridge beats the lasso: 88 validation rows cannot separate them, and each
  number was chosen as the best of 60, so both are optimistic (B1.1).
- The entry order (`s5`, `bmi`, `bp` first) is a sensible ranking of importance for this model; `s2` enters last because
  `s1` already carries most of its information.
- Even at $\lambda = 10^4$, every ridge weight is non-zero.

`RidgeCV`, `LassoCV` and `ElasticNetCV` choose the strength by cross-validation inside `fit`. Use them inside a pipeline
after the scaler, so the scaling is refitted in every fold (B1.4).

## In practice

- **When to use it:** always, for linear and logistic models. **Ridge** (or `LogisticRegression`'s default L2) when you
  expect many small effects and want stable weights with correlated features. **Lasso** when you expect few features
  to matter and want a sparse, cheaper model. **Elastic net** when you want sparsity but have groups of correlated
  features (text n-grams, sensor channels, one-hot dummies of related categories).
- **Failure modes:**
  - *Penalizing unscaled features.* The penalty compares weight sizes, and a weight's size depends on its feature's
    units. A feature in grams gets a tiny weight and is barely penalized; the same feature in kilograms is penalized
    1,000 times harder. Standardize first, inside the pipeline.
  - *Penalizing the intercept.* Predictions are pulled towards 0 instead of the mean. scikit-learn does not penalize
    `intercept_` for these models; hand-written code (or a column of ones inside $\mathbf{X}$) must not either.
  - *Reading lasso selection as truth.* Among correlated features the lasso's choice is close to arbitrary and changes
    between resamples. Check stability by refitting on bootstrap samples before telling anyone "only these features matter".
  - *Choosing $\lambda$ on the test set, or tuning it outside cross-validation with a pre-fitted scaler.* Both are leakage (B1.4).
  - *Mixing up conventions.* `alpha` in `Ridge` vs `Lasso` (sum vs half-mean), `C` as the *inverse* strength in
    `LogisticRegression` and `SVC` (B2.5), `weight_decay` in PyTorch (C3.1). Read the objective in the docs.
- **Defaults that work:** `make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))` for regression;
  `LogisticRegressionCV` or a `C` grid of `np.logspace(-3, 3, 13)` for classification; `ElasticNetCV(l1_ratio=[.1, .5, .9, .95, 1])`
  when sparsity matters. In scikit-learn 1.9.1, `LogisticRegression` takes `l1_ratio` (0 = L2, 1 = L1) with `C`; `penalty` is
  deprecated (VERIFY as of 2026-10).

## Exercises

Save your work to `my-work/B/b2_2_regularization.md` (or `.py`) and ask for a review.

1. **By hand:** Two standardized features with $\frac1n\mathbf{Z}^\top\mathbf{Z} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ (uncorrelated) and $\frac1n\mathbf{Z}^\top(\mathbf{y} - \bar{y}) = (3, -0.5)$, $n = 10$. (a) Find the least-squares
   weights. (b) Find the lasso weights for $\lambda = 1$ and for $\lambda = 4$ (in the half-mean form; with uncorrelated features, one sweep of
   coordinate descent is exact: say why). (c) Find the ridge weights in scikit-learn's sum form for $\lambda = 10$. (d) Find the
   elastic-net weights for $\lambda = 1$, $\rho = 0.5$. (e) Compute $\lambda_{\max}$ for the lasso.
2. **Code:** Bootstrap stability of the lasso. On the lab's diabetes training split, draw 200 bootstrap samples
   (`rng = np.random.default_rng(0)`, rows sampled with replacement), standardize each sample with its own statistics, and fit
   your coordinate-descent lasso at $\alpha = 1$ and elastic net at $\alpha = 1$, $\rho = 0.5$. For every feature report the fraction of fits in which
   it is non-zero, and the mean and standard deviation of its weight. Which features are "selected" reliably, and what
   happens to `s1` and `s2`? Check one of your fits against `sklearn.linear_model.Lasso`.
3. **Think:** A product team has a churn model with 400 features (many one-hot dummies, several near-duplicate usage
   counters) and 3,000 customers. They ask you to "use the lasso to tell us the 10 features that drive churn, so we can
   act on them". Explain what you would deliver, what the lasso can and cannot tell them, and how you would choose
   between ridge, lasso and elastic net for the production model.

## Takeaways

1. Regularization adds $\lambda \times$ (size of $\mathbf{w}$) to the loss. It trades a little training fit for stable, finite weights that
   generalize; $\lambda$ is chosen on validation data, never on the test set.
2. L2 (ridge) is a Gaussian prior. It adds $\lambda$ to every eigenvalue of $\mathbf{Z}^\top\mathbf{Z}$, has a closed form, shrinks smoothly, never zeroes a weight,
   and makes correlated features share credit.
3. L1 (lasso) is a Laplace prior. Its constant push gives exact zeros through the soft-threshold $S(a, \lambda)$, which you
   compute by coordinate descent. Its choice among correlated features is unstable.
4. Elastic net combines both: sparsity with grouping. In `LogisticRegression`, $C$ is the inverse strength, $\lambda = 1/(Cn)$, and a penalty
   makes separable data solvable.
5. Standardize before penalizing, never penalize the intercept, and read each library's objective: `alpha` and `C` mean
   different things in different classes.

**What builds on this:** [B2.5](b2_5_svm.md) (the SVM's $C$ and its L2 penalty); [B3.3](b3_3_gradient_boosting.md) (L1/L2 on
leaf values in XGBoost); [B4.3](b4_3_cross_validation_tuning.md) (choosing $\lambda$ by cross-validation);
[B4.5](b4_5_bias_variance.md) (regularization as a bias–variance dial); [C3.1](../C/c3_1_optimizers.md) (AdamW's
decoupled weight decay); [C3.5](../C/c3_5_regularization.md) (dropout, early stopping and weight decay in networks);
P2 (the M1 pipeline ends in a regularized logistic regression).

**References:** R. Tibshirani, "Regression Shrinkage and Selection via the Lasso", *Journal of the Royal Statistical
Society B* 58(1), 1996. A. E. Hoerl, R. W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems",
*Technometrics* 12(1), 1970. H. Zou, T. Hastie, "Regularization and Variable Selection via the Elastic Net", *Journal of the
Royal Statistical Society B* 67(2), 2005. J. Friedman, T. Hastie, R. Tibshirani, "Regularization Paths for Generalized
Linear Models via Coordinate Descent", *Journal of Statistical Software* 33(1), 2010. scikit-learn User Guide, "Linear
Models" (VERIFY link: https://scikit-learn.org/stable/modules/linear_model.html).
