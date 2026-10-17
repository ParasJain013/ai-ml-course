# B1.5 Linear regression from scratch versus scikit-learn

**Core** · Part B, week 1 (course week 07) · ~2 h · Lab:
[`labs/B/b1_5_linear_regression.py`](../../labs/B/b1_5_linear_regression.py)

## Why it matters

**Linear regression** is the first model of Part B and the one every later model is built from or compared
with: logistic regression (B2.1) is linear regression passed through a sigmoid, regularization (B2.2) adds a
penalty to its loss, and the last layer of most neural networks is a linear map trained by gradient descent.
It is also a strong baseline in its own right, fast, and readable. Part A gave you both ways to fit it:
the normal equations (A2.4) and gradient descent (A4.4). This lesson puts them on a real dataset, shows that
they and scikit-learn agree to four decimals, and shows the one practical trap that decides whether gradient
descent works at all: feature scaling.

## Prerequisite check

- Can you write the normal equations $\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}$ and say why they give the least-squares fit? If not, revisit
  [A2.4](../A/a2_4_projections_least_squares.md).
- Can you compute the gradient of the mean squared error and take a gradient-descent step? If not, revisit
  [A4.2](../A/a4_2_gradients_jacobians.md) and [A4.4](../A/a4_4_gradient_descent.md).
- Can you explain what the condition number of the Hessian does to gradient descent? If not, revisit
  [A4.4](../A/a4_4_gradient_descent.md) and [A3.4](../A/a3_4_numerical_stability.md).
- Can you standardize a column with training statistics? If not, revisit [B1.3](b1_3_feature_engineering.md).
- Do you know the mean baseline's score and what $R^2$ measures? If not, revisit [B1.1](b1_1_ml_workflow.md).

## Intuition

A linear regression predicts a number as a weighted sum of the features plus a constant:
$\hat{y} = w_1x_1 + \dots + w_dx_d + b$. Each weight says how much the prediction moves when its feature goes up by one
unit *and every other feature stays the same*. Fitting means choosing the weights and the bias that make the
squared errors on the training rows as small as possible.

Two routes reach the same answer:

- **Closed form.** The minimum of the squared error is where its gradient is zero, which gives a small linear
  system, the normal equations, solved in one step. Geometrically (A2.4), the predictions are the projection of
  the target vector onto the space spanned by the feature columns. Cost: building $\mathbf{X}^\top\mathbf{X}$ takes about $nd^2$
  operations and solving it about $d^3$, so it is exact and fast while $d$ is in the hundreds or low thousands.
- **Gradient descent.** Start from zero, step downhill along the negative gradient, repeat. Each step costs
  about $nd$ operations and never forms a $d \times d$ matrix, so it scales to many features and to data that arrives in
  batches (stochastic gradient descent, A4.4). It is also how every neural network is trained, which is why
  doing it by hand here matters.

Gradient descent has a catch. The loss surface of a linear regression is a bowl; its shape is set by the
features. If one feature is measured in hundreds (cholesterol, around 190) and another in fractions (a log
triglyceride level, around 4.6, varying by 0.5), the bowl is a long narrow valley. A learning rate small enough
not to overshoot across the steep direction crawls along the flat one. Standardizing every feature (B1.3) makes
the bowl much rounder. On the diabetes data the condition number drops from 51 million to 370, and gradient descent
goes from hopeless (after 20,000 steps the loss is still ten times the minimum) to converged in 4,347 steps.
Standardizing changes the units of the weights, not the predictions, and you can convert the weights back.

> **Bridge:** closed form versus gradient descent is like computing a result with one SQL aggregate versus a
> batch job that walks the table in chunks and refines an estimate. The query is exact and quick while
> everything fits; the batch job scales to data that does not. **Where it breaks:** the aggregate does not care
> how your columns are scaled, but gradient descent does. Its speed, and whether it finishes at all, depend on
> the feature scales and the learning rate.

**Reading the weights.** A weight is "per unit of this feature, holding the others fixed". Three cautions. Units
differ (a weight of 81 per unit of `s5` and 1.2 per unit of `bp` are not comparable; weights on standardized features,
"per standard deviation", are). Correlated features share credit unpredictably: in the lab, `s1` (total cholesterol) and
`s2` (LDL) correlate at 0.88 and get weights of opposite sign, which says more about the correlation than about
biology. And a weight is an association in this data, not the effect of changing the feature (B3.4 makes the same point about model explanations).

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{X}$ | training features, one row per example | `(n, d)` |
| $\mathbf{y}$ | training targets | `(n,)` |
| $\mathbf{w}$, $b$ | weights and bias (intercept) | `(d,)`, scalar |
| $\mathbf{A}$ | $\mathbf{X}$ with a column of ones in front: $[\mathbf{1}, \mathbf{X}]$ | `(n, d + 1)` |
| $\boldsymbol{\theta}$ | all parameters, $(b, w_1, \dots, w_d)$ | `(d + 1,)` |
| $\hat{\mathbf{y}}$ | predictions $\mathbf{A}\boldsymbol{\theta} = \mathbf{X}\mathbf{w} + b$ | `(n,)` |
| $\mathbf{r}$ | residuals $\mathbf{y} - \hat{\mathbf{y}}$ | `(n,)` |
| $\mathcal{L}(\boldsymbol{\theta})$ | mean squared error $\frac{1}{n}\lVert\mathbf{A}\boldsymbol{\theta} - \mathbf{y}\rVert^2$ | scalar |
| $\mathbf{H}$ | Hessian of $\mathcal{L}$, $\frac{2}{n}\mathbf{A}^\top\mathbf{A}$ | `(d + 1, d + 1)` |
| $\lambda_{\min}, \lambda_{\max}$ | smallest and largest eigenvalues of $\mathbf{H}$; $\kappa = \lambda_{\max}/\lambda_{\min}$ is its condition number | scalars |
| $\eta$ | learning rate | scalar |
| $\mu_j$, $\sigma_j$ | training mean and standard deviation of feature $j$ | scalars |
| $\mathbf{Z}$ | standardized features, $z_{ij} = (x_{ij} - \mu_j)/\sigma_j$ | `(n, d)` |

**The loss and its gradient.** Putting the bias into $\boldsymbol{\theta}$ with a column of ones keeps every formula in one piece:

$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{n}\lVert\mathbf{A}\boldsymbol{\theta} - \mathbf{y}\rVert^2, \qquad \nabla_{\boldsymbol{\theta}}\mathcal{L} = \frac{2}{n}\,\mathbf{A}^\top(\mathbf{A}\boldsymbol{\theta} - \mathbf{y}) \quad\text{(shape } (d + 1,)\text{, A4.2)}.$$

**Closed form.** Setting the gradient to zero gives the normal equations

$$\mathbf{A}^\top\mathbf{A}\,\boldsymbol{\theta} = \mathbf{A}^\top\mathbf{y},$$

a $(d + 1) \times (d + 1)$ system, solved with `np.linalg.solve` (never by inverting, A3.4). At the solution, $\mathbf{A}^\top\mathbf{r} = \mathbf{0}$: the residuals are
orthogonal to every column, including the ones column, so **the training residuals sum to zero**.

**One feature, by hand.** With $d = 1$, the first normal equation (the row for the ones column) reads $\sum_i (b + wx_i - y_i) = 0$,
so $b = \bar{y} - w\bar{x}$: the fitted line passes through the point of means. Substituting into the second gives

$$w = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2}.$$

**Gradient descent** repeats $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\,\nabla_{\boldsymbol{\theta}}\mathcal{L}$. For this quadratic loss the Hessian $\mathbf{H} = \frac{2}{n}\mathbf{A}^\top\mathbf{A}$ is constant, and A4.4 showed:
the iteration is stable only if $\eta < 2/\lambda_{\max}$; the best fixed rate is $\eta = 2/(\lambda_{\min} + \lambda_{\max})$; and with it, the error along the
slowest direction shrinks by the factor

$$\rho = \frac{\kappa - 1}{\kappa + 1}$$

per step. With $\kappa = 370$, $\rho = 0.9946$: about 1,280 steps per factor of 1,000. With $\kappa = 5 \times 10^7$, $\rho = 0.99999996$: about $1.8 \times 10^8$ steps
per factor of 1,000.

**Why standardizing helps.** Standardize each column with its training mean and standard deviation. Each column of
$\mathbf{Z}$ then has mean 0 and $\frac{1}{n}\sum_i z_{ij}^2 = 1$. In $\mathbf{H} = \frac{2}{n}[\mathbf{1}, \mathbf{Z}]^\top[\mathbf{1}, \mathbf{Z}]$, the entry for the ones column is $\frac{2}{n} \cdot n = 2$, its entries
with each feature are $\frac{2}{n}\sum_i z_{ij} = 0$, and every diagonal entry of the feature block is $\frac{2}{n}\sum_i z_{ij}^2 = 2$. So the bias is decoupled from the
weights, every direction starts with curvature 2, and the only thing left to make $\kappa$ large is correlation between
features (the off-diagonal entries, $2 \times$ the correlation of two features). Raw features have neither property: the
column of `s1` values (around 190) gives a curvature about $2 \times 190^2$ while `s5` (around 4.6) gives about $2 \times 4.6^2$.

**Converting back.** A model fitted on standardized features predicts

$$\hat{y} = b_z + \sum_j w_{z,j}\frac{x_j - \mu_j}{\sigma_j} = \underbrace{\Big(b_z - \sum_j \frac{w_{z,j}}{\sigma_j}\mu_j\Big)}_{b} + \sum_j \underbrace{\frac{w_{z,j}}{\sigma_j}}_{w_j} x_j,$$

so $w_j = w_{z,j}/\sigma_j$ (per raw unit) and $b = b_z - \sum_j w_j\mu_j$. Same predictions, same model, different units.

**$R^2$.** On any set of rows, $R^2 = 1 - \sum(y_i - \hat{y}_i)^2 / \sum(y_i - \bar{y})^2$ (B1.1). On the training rows of a least-squares fit with an intercept it lies
between 0 and 1 and never decreases when a feature is added; on validation rows it can do anything, which is why it
is reported there.

## Worked example by hand

Four points: $x = 1, 2, 3, 4$ and $y = 2, 3, 5, 6$.

**Closed form.** $\bar{x} = 2.5$, $\bar{y} = 4$. Deviations: $dx = -1.5, -0.5, 0.5, 1.5$ and $dy = -2, -1, 1, 2$. $\sum dx\,dy = 3 + 0.5 + 0.5 + 3 = 7$ and $\sum dx^2 = 2.25 + 0.25 + 0.25 + 2.25 = 5$,
so $w = 7/5 = 1.4$ and $b = 4 - 1.4 \times 2.5 = 0.5$. Predictions $1.9, 3.3, 4.7, 6.1$; residuals $0.1, -0.3, 0.3, -0.1$ (they sum to zero). $\mathrm{SSE} = 0.01 + 0.09 + 0.09 + 0.01 = 0.2$,
$\mathrm{SST} = 4 + 1 + 1 + 4 = 10$, $R^2 = 1 - 0.2/10 = 0.98$. The normal equations give the same: $\mathbf{A}^\top\mathbf{A} = \begin{pmatrix} 4 & 10 \\ 10 & 30 \end{pmatrix}$, $\mathbf{A}^\top\mathbf{y} = (16, 47)$, and solving
$4b + 10w = 16$, $10b + 30w = 47$ gives $b = 0.5$, $w = 1.4$.

**Gradient descent on the standardized feature.** The standard deviation of $x$ (divide by $n$) is $\sqrt{5/4} = 1.118$, so $z = dx/1.118 = -1.342, -0.447,
0.447, 1.342$. With $\mathbf{A} = [\mathbf{1}, \mathbf{z}]$: $\mathbf{A}^\top\mathbf{A} = \begin{pmatrix} 4 & 0 \\ 0 & 4 \end{pmatrix}$ (the $z$ values sum to 0 and their squares to 4), so $\mathbf{H} = \frac{2}{4}\mathbf{A}^\top\mathbf{A} = 2\mathbf{I}$, both
eigenvalues are 2, and the best learning rate is $2/(2 + 2) = 0.5$. At $\boldsymbol{\theta} = \mathbf{0}$ the predictions are 0, so

$$\nabla\mathcal{L} = \frac{2}{4}\mathbf{A}^\top(\mathbf{0} - \mathbf{y}) = -\frac12\begin{pmatrix} \sum y_i \\ \sum z_iy_i \end{pmatrix} = -\frac12\begin{pmatrix} 16 \\ 6.261 \end{pmatrix} = \begin{pmatrix} -8 \\ -3.130 \end{pmatrix},$$

where $\sum z_iy_i = (-1.342)(2) + (-0.447)(3) + (0.447)(5) + (1.342)(6) = 6.261$. One step with $\eta = 0.5$: $\boldsymbol{\theta} = (4, 1.565)$, and the gradient there is zero.
With a perfectly round bowl ($\kappa = 1$), gradient descent lands on the minimum in one step. Back to raw units: $w = 1.565/1.118 = 1.4$
and $b = 4 - 1.4 \times 2.5 = 0.5$, the closed-form answer.

```text
== worked example by hand ==
x-bar = 2.5   y-bar = 4.0   dx = [-1.5, -0.5, 0.5, 1.5]   dy = [-2.0, -1.0, 1.0, 2.0]
sum dx*dy = 7.0   sum dx^2 = 5.0   slope = 1.4000   intercept = 0.5000
predictions = [1.9, 3.3, 4.7, 6.1]   residuals = [0.1, -0.3, 0.3, -0.1]
SSE = 0.2000   SST = 10.0000   R^2 = 1 - SSE/SST = 0.9800
normal equations: X^T X = [[4.0, 10.0], [10.0, 30.0]]   X^T y = [16.0, 47.0]   solution = [0.5, 1.4]
standardized x = [-1.3416, -0.4472, 0.4472, 1.3416]   (sd = sqrt(1.25) = 1.1180)
Z^T Z = [[4.0, 0.0], [0.0, 4.0]]   gradient at w = 0: (2/n) Z^T (Z w - y) = [-8.0, -3.1305]
one step with lr = 0.5: w = [4.0, 1.5652]   gradient there = [0.0, 0.0]
back to raw units: slope = 1.5652 / 1.1180 = 1.4000   intercept = 4.0000 - slope * 2.5 = 0.5000
```

## Key plot

![Log-log plot of training MSE minus its minimum against gradient-descent step: with standardized features the gap falls from about 3e4 to below 1e-6 by step 2,000; with raw features at their best learning rate the gap stays flat near 2.5e4 for all 20,000 steps](img/b1_5_gd_scaled_vs_raw.png)

Same data, same loss, same algorithm. With standardized features gradient descent reaches the minimum; with raw
features, even at the best possible learning rate, 20,000 steps barely move the loss, because one direction of
the bowl is 51 million times flatter than another.

## From scratch in NumPy

**The data.** The diabetes dataset bundled with scikit-learn, in raw units: 442 patients, ten baseline
measurements (age, sex coded 1/2, body mass index, average blood pressure, and six blood serum measurements
`s1`–`s6`), and a measure of disease progression one year later. Same split as B1.1, so the baselines carry over:

```text
== data ==
train X (266, 10)  val X (88, 10)  (test 88 rows kept aside)
feature means (train): {'age': 48.94, 'sex': 1.51, 'bmi': 26.32, 'bp': 95.12, 's1': 189.57, 's2': 115.5, 's3': 49.72, 's4': 4.08, 's5': 4.66, 's6': 92.02}
feature sds   (train): {'age': 13.22, 'sex': 0.5, 'bmi': 4.26, 'bp': 13.59, 's1': 33.89, 's2': 29.47, 's3': 13.6, 's4': 1.19, 's5': 0.52, 's6': 11.07}
```

**Normal equations.**

```python
def fit_normal_equations(X, y):
    """Least squares with an intercept. X (n, d), y (n,) -> (b, w (d,))."""
    A = add_intercept(X)                                  # (n, d + 1)
    theta = np.linalg.solve(A.T @ A, A.T @ y)            # (d + 1,)
    return float(theta[0]), theta[1:]
```

```text
== from scratch: normal equations ==
intercept b = -396.8159
weights w (raw units): {'age': 0.0784, 'sex': -26.6913, 'bmi': 5.5932, 'bp': 1.2254, 's1': -1.5279, 's2': 1.213, 's3': 0.9313, 's4': 7.171, 's5': 81.007, 's6': 0.228}
checks: mean training residual 5.79e-14   max |X^T r| 1.07e-08
condition number of A^T A (raw features + ones): 5.150e+07
```

The checks are the math made visible: the training residuals sum to zero (to $10^{-14}$) and are orthogonal to every
column (to $10^{-8}$, relative to sums of products in the hundreds of thousands). The condition number of $\mathbf{A}^\top\mathbf{A}$ is $5 \times 10^7$.
Double precision has about 16 significant digits, so `solve` still loses only about 8 of them here (A3.4): fine for the
closed form, fatal for gradient descent.

**Gradient descent on standardized features.** Standardize with training statistics, run the A4.4 loop with the best
fixed learning rate, convert back:

```python
mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)            # (d,) training statistics only (B1.4)
AZ = add_intercept((X_tr - mu) / sd)                     # (n, d + 1)
lam = np.linalg.eigvalsh(2.0 / n * AZ.T @ AZ)            # Hessian eigenvalues, ascending
theta_z, losses, steps = gradient_descent(AZ, y_tr, lr=2.0 / (lam[0] + lam[-1]))
w_back = theta_z[1:] / sd                                # per raw unit
b_back = theta_z[0] - w_back @ mu
```

```text
== from scratch: gradient descent ==
standardized: Hessian eigenvalues 0.0198 .. 7.3279   condition number 369.8   lr 0.2722   converged in 4347 steps
weights per 1 sd of each feature: {'age': 1.0369, 'sex': -13.3442, 'bmi': 23.8452, 'bp': 16.6567, 's1': -51.7839, 's2': 35.7439, 's3': 12.6667, 's4': 8.5396, 's5': 41.883, 's6': 2.5243}
converted back: b = -396.8159   max |w - w_normal_equations| = 1.30e-09
GD on standardized features matches the normal equations to 4 decimals: OK
```

Even standardized, the condition number is 370, not 1: the blood serum measurements are strongly correlated with
each other, which standardizing cannot remove. That is why it takes 4,347 steps rather than a handful. The converted
weights match the normal equations to $10^{-9}$. The weights per standard deviation are comparable with each other: `bmi`
(+23.8), `s5` (+41.9), `bp` (+16.7) carry the most, while `s1` and `s2` (−51.8 and +35.7) mostly cancel each other, as correlated
features do.

**Gradient descent on raw features.**

```text
== from scratch: gradient descent without scaling ==
raw: Hessian eigenvalues 2.872e-03 .. 1.479e+05   condition number 5.150e+07   best lr 1.353e-05
with the standardized run's lr 0.2722: diverged at step 33: lr = 0.2721942083081054 is too large
with lr 1.353e-05: 20000 steps   train MSE 27496.87   (minimum 2816.37)
weights after 20000 steps: bmi 5.6456 (target 5.5932)   s5 -0.0533 (target 81.0070)
slowest direction shrinks by (kappa - 1)/(kappa + 1) = 0.9999999612 per step -> about 1.779e+08 steps to cut its error 1,000-fold
```

The standardized run's learning rate diverges within 33 steps on raw features: the steepest direction (the large
cholesterol values) has curvature $1.5 \times 10^5$, so the rate must be below $2/(1.5 \times 10^5) \approx 1.4 \times 10^{-5}$. At that rate, 20,000 steps fit the
steep directions (the `bmi` weight is already close) and leave the flat ones untouched: `s5`, whose small values give
it little curvature, is still near its starting value of 0 instead of 81, and the training MSE is ten times the
minimum. The convergence-rate formula says reaching the answer would take hundreds of millions of steps.

## With scikit-learn

`LinearRegression` solves least squares directly (a least-squares solver on centred data, not gradient descent), so it
needs no scaling and no learning rate:

```text
== with scikit-learn ==
LinearRegression intercept_ -396.8159   matches normal equations: OK
LinearRegression coef_ [0.0784, -26.6913, 5.5932, 1.2254, -1.5279, 1.213, 0.9313, 7.171, 81.007, 0.228]
np.linalg.lstsq matches: OK
validation: MSE 3133.62  RMSE 55.98  R^2 0.4655
train:      MSE 2816.37  R^2 0.5218
B1.1 baselines on the same validation rows: mean R^2 -0.0378 (RMSE 78.00), bmi rule R^2 0.3461 (RMSE 61.92)
bmi alone, linear: val R^2 0.3598   slope 10.2339 per kg/m^2
correlation of s1 and s2 on train: 0.8812   their weights: s1 -1.5279  s2 1.2130
```

- `LinearRegression` and `np.linalg.lstsq` match the normal equations to six significant figures, and their validation
  predictions are identical.
- On validation rows the model explains 47% of the variance ($R^2$ 0.466, RMSE 56.0), against $-0.04$ for the mean baseline
  and 0.35 for the one-feature bmi rule from B1.1. Ten features beat one by 0.12 of $R^2$: a real but modest gain, and
  the honest way to report it.
- Training $R^2$ (0.52) is above validation $R^2$ (0.47): the usual small generalization gap of a model with 11 parameters on
  266 rows.
- A linear fit on `bmi` alone reaches 0.36 on validation: the bmi slope is 10.2 per kg/m² alone, but 5.6 when the other
  features are in the model, because `bmi` is correlated with them. A weight depends on which other features are present.
- The correlated pair `s1`/`s2` (correlation 0.88) gets weights of opposite sign. Do not read them separately.

## In practice

- **When to use it:** as the first real model on any regression problem, right after the baselines; whenever you need
  a fast, explainable model; as a sanity check on fancier models (if gradient boosting barely beats it, the extra
  complexity may not be worth it). The closed form for up to a few thousand features; gradient descent or SGD when
  features or rows are too many for memory, or when training online.
- **Failure modes:**
  - *Gradient descent on unscaled features.* You notice a loss that diverges at any reasonable learning rate, or one
    that decreases and then stalls far above what `LinearRegression` reaches. Standardize (inside a pipeline, B1.3).
  - *Collinear features.* Weights with surprising signs or huge magnitudes that change when a row is removed; `solve`
    may fail with a singular matrix if two columns are identical. You notice via correlations between features or a
    large condition number. Do not interpret those weights individually; drop or combine features, or regularize (B2.2).
  - *Non-linear effects.* Residuals that curve when plotted against a feature. Add features with the right shape
    (B1.3) or use a non-linear model (B3).
  - *Outliers.* Squared error lets one extreme point pull the line. You notice large residuals on a few rows. Inspect
    them (B1.2); consider a robust loss (Huber) or a log target.
  - *Extrapolation.* Predictions for feature values outside the training range follow the line forever (negative
    durations, impossible prices). Check the ranges at prediction time.
  - *Reading weights as causes.* A weight is an association conditional on the other features, in this data.
- **Defaults that work:** `LinearRegression()` for the closed form (`fit_intercept=True` is the default; VERIFY as of 2026-10,
  scikit-learn 1.9.1); `StandardScaler` before anything iterative (`SGDRegressor`, gradient descent, regularized models);
  report validation RMSE and $R^2$ next to the B1.1 baselines; plot residuals against predictions and against each feature.

## Exercises

Save your work to `my-work/B/b1_5_linear_regression.md` (or `.py`) and ask for a review.

1. **By hand:** Five points, $x = 0, 1, 2, 3, 4$ and $y = 1, 3, 2, 5, 4$. (a) Fit the line with the deviation formulas and compute the
   residuals, SSE, SST and $R^2$. (b) Write $\mathbf{A}^\top\mathbf{A}$ and $\mathbf{A}^\top\mathbf{y}$ for $\mathbf{A} = [\mathbf{1}, \mathbf{x}]$ and check your answer solves the normal equations. (c) For
   raw $x$, compute the Hessian $\frac{2}{n}\mathbf{A}^\top\mathbf{A}$, its two eigenvalues, the condition number and the largest stable learning rate. (d)
   Standardize $x$ and show that one gradient-descent step from $\boldsymbol{\theta} = \mathbf{0}$ with $\eta = 0.5$ lands on the solution; convert it back to
   raw units.
2. **Code:** Collinearity. Starting from the lab's diabetes split, add (i) an exact copy of the `bmi` column, and
   separately (ii) a near-copy, `bmi` plus Gaussian noise with standard deviation 0.01 (`SEED = 0`). For each, fit with
   `np.linalg.solve` on the normal equations, `np.linalg.lstsq`, `LinearRegression`, and your gradient descent from zero
   on standardized features. Report the two bmi weights, the condition number of $\mathbf{A}^\top\mathbf{A}$ and the validation $R^2$ of each
   method. Explain why the predictions agree while the weights do not, and which solution gradient descent from zero
   converges to in case (i).
3. **Think:** A clinician reads the lab's weights (raw units) and concludes: "total cholesterol (`s1`, weight −1.53)
   protects against progression, and patients with `sex = 2` progress more slowly (weight −26.7)." Explain what these
   weights do and do not say, what in the data makes the `s1` weight especially hard to read, and what you would show
   the clinician instead.

## Takeaways

1. Linear regression minimizes the mean squared error of $\mathbf{X}\mathbf{w} + b$; adding a column of ones folds the bias into one
   parameter vector and one gradient, $\frac{2}{n}\mathbf{A}^\top(\mathbf{A}\boldsymbol{\theta} - \mathbf{y})$.
2. The closed form solves $\mathbf{A}^\top\mathbf{A}\boldsymbol{\theta} = \mathbf{A}^\top\mathbf{y}$ in one step (use `solve` or `lstsq`, never an inverse); at the solution the
   residuals are orthogonal to every column and sum to zero.
3. Gradient descent reaches the same answer only if the problem is well conditioned: its speed is governed by
   $\kappa = \lambda_{\max}/\lambda_{\min}$, and unscaled features can make $\kappa$ astronomically large.
4. Standardize features for any iterative fit, then convert back: $w_j = w_{z,j}/\sigma_j$, $b = b_z - \sum_j w_j\mu_j$; the predictions do not
   change.
5. Report validation error next to the baselines, and read weights with care: they are per unit, conditional on the
   other features, unstable under correlation, and not causal.

**What builds on this:** [B2.1](b2_1_logistic_regression.md) (the same linear score through a sigmoid, fitted by
gradient descent); [B2.2](b2_2_regularization.md) (ridge and lasso fix collinear weights);
[B4.2](b4_2_regression_metrics.md) (choosing between MSE, MAE and others);
[B4.5](b4_5_bias_variance.md) (training versus validation error); [C1.2](../C/c1_2_backprop.md) (gradient descent
through layers of linear maps); P2 (the baseline model in M1 is a regularized linear model).

**References:** T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009,
chapter 3. B. Efron, T. Hastie, I. Johnstone, R. Tibshirani, "Least Angle Regression", *Annals of Statistics* 32(2), 2004
(the source of the diabetes dataset). scikit-learn User Guide, "Linear Models" (VERIFY link:
https://scikit-learn.org/stable/modules/linear_model.html).
