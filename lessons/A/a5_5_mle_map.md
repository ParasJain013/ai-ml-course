# A5.5 MLE and MAP

**Core** · Part A, week 5 (course week 05) · ~2.5 h · Lab:
[`labs/A/a5_5_mle_map.py`](../../labs/A/a5_5_mle_map.py)

## Why it matters

Why is linear regression trained with squared error, logistic regression with log-loss, and a
language model with cross-entropy? Why does adding $\lambda\lVert\mathbf{w}\rVert^2$ to the loss help on small
datasets? These look like separate recipes, and engineers often treat them as conventions to copy.
They are one idea: **maximum likelihood estimation (MLE)** picks the parameters under which the
observed data is most probable, and each loss is the negative log-likelihood of a particular noise
model. Add a belief about the parameters before seeing data, and Bayes' theorem
([A5.3](a5_3_bayes_theorem.md)) turns MLE into **MAP** estimation, which is where regularization comes
from. Knowing this tells you which loss to use when the textbook one does not fit your data (outliers,
counts, skewed targets), and what a regularization strength actually means.

## Prerequisite check

- Can you write the Gaussian density and the Bernoulli pmf, and say what $\mu$, $\sigma$ and $p$ are? If not,
  revisit [A5.2](a5_2_common_distributions.md).
- Can you state Bayes' theorem with prior, likelihood and posterior? If not, revisit
  [A5.3](a5_3_bayes_theorem.md).
- Can you differentiate $\log x$, $x^2$ and $(x - a)^2$, and set a derivative to zero to find a minimum? If
  not, revisit [A4.1](a4_1_derivatives.md).
- Can you run gradient descent with a stopping test and explain the role of the learning rate? If not,
  revisit [A4.4](a4_4_gradient_descent.md).
- Can you write the least-squares solution $\mathbf{w} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ and say what shape each part has?
  If not, revisit [A2.4](a2_4_projections_least_squares.md).

## Intuition

### Which coin made these flips?

You flip a coin 4 times and get heads, heads, tails, heads. Three suspects: a fair coin ($p = 0.5$), a
coin with $p = 0.75$, and one with $p = 0.9$. How likely is *this exact sequence* under each?
$0.5^3 \times 0.5 = 0.0625$; $0.75^3 \times 0.25 = 0.105$; $0.9^3 \times 0.1 = 0.073$. The $p = 0.75$ coin makes the data most
probable, so it is the maximum-likelihood answer. Try every $p$ between 0 and 1 and the winner is
exactly $3/4$: the observed fraction of heads. MLE formalizes the obvious answer, and then gives
answers where nothing is obvious.

The **likelihood** is the probability of the data you saw, viewed as a function of the parameters. It
is not the probability of the parameters: the data is fixed and you are scoring candidate settings.

### Why "negative log"

The probability of $n$ independent observations is a product of $n$ numbers below 1, which underflows
float64 to 0 after a few hundred terms ([A3.4](a3_4_numerical_stability.md)). Taking the log turns the
product into a sum, keeps the numbers in a sane range, and does not move the maximum (log is
increasing). Flipping the sign turns "maximize" into "minimize", which is what optimizers do. So in
practice you always minimize the **negative log-likelihood (NLL)**, and the NLL averaged over examples
is the loss function.

### Each loss is a noise model

Assume a model's prediction is right on average and the errors are Gaussian. Then the NLL is, apart
from constants, the sum of squared errors. Assume a 0/1 label is a Bernoulli draw with the predicted
probability; the NLL is log-loss. Assume errors follow a sharper, heavier-tailed distribution (Laplace);
the NLL is the absolute error. Choosing a loss is choosing a probability model of the data, whether
you say so or not.

### MAP: the data plus a prior belief

With 5 data points, MLE can produce extreme estimates (a slope of 40 when every sensible model has a
slope near 1). If before seeing data you believe weights are probably small, write that belief as a
**prior** distribution over the weights, and maximize the **posterior** (prior × likelihood) instead.
That is **MAP (maximum a posteriori)** estimation. A Gaussian prior centred at 0 adds exactly the term
$\lambda\lVert\mathbf{w}\rVert^2$ to the squared-error loss: L2 regularization is a Gaussian prior. With lots of data the
likelihood dominates and the prior hardly matters; with little data the prior keeps estimates sane.

> **Bridge:** MAP is a default value with a confidence weight. A config system that merges a sensible
> default with a user's override, trusting the override more the more evidence it has (say, a
> feature flag's rollout percentage moving from the default as metrics accumulate), behaves like a
> posterior. **Where it breaks:** a config override replaces the default outright; MAP always blends,
> and how much weight the prior keeps depends on its variance and the amount of data, never on a rule
> you write by hand.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\boldsymbol{\theta}$ | parameters of a model or distribution | — |
| $x_1, \dots, x_n$ | i.i.d. observations | scalars |
| $L(\boldsymbol{\theta})$ | likelihood $\prod_i p(x_i \mid \boldsymbol{\theta})$ | scalar |
| $\mathcal{L}_{\text{NLL}}(\boldsymbol{\theta})$ | negative log-likelihood $-\sum_i \log p(x_i \mid \boldsymbol{\theta})$ | scalar |
| $\hat{\boldsymbol{\theta}}$ | an estimate (MLE or MAP; the text says which) | — |
| $\mu, \sigma^2$ | Gaussian mean and variance | scalars |
| $p$, $k$ | Bernoulli probability and number of successes | scalar, int |
| $\mathbf{X}$, $\mathbf{y}$, $\mathbf{w}$ | design matrix, targets, weights | `(n, d)`, `(n,)`, `(d,)` |
| $\mathbf{r}$ | residuals $\mathbf{y} - \mathbf{X}\mathbf{w}$ | `(n,)` |
| $\tau^2$ | prior variance of each weight | scalar |
| $\lambda$ | regularization strength (here $\sigma^2/\tau^2$) | scalar |
| $\mathbf{D}$ | diagonal matrix with 1 for penalized weights, 0 for the intercept | `(d, d)` |

**Likelihood and NLL.** For i.i.d. data, probabilities multiply (A5.1 independence), so

$$L(\boldsymbol{\theta}) = \prod_{i=1}^n p(x_i \mid \boldsymbol{\theta}), \qquad \mathcal{L}_{\text{NLL}}(\boldsymbol{\theta}) = -\log L(\boldsymbol{\theta}) = -\sum_{i=1}^n \log p(x_i \mid \boldsymbol{\theta}),$$

and $\hat{\boldsymbol{\theta}}_{\text{MLE}} = \arg\min_{\boldsymbol{\theta}}\mathcal{L}_{\text{NLL}}(\boldsymbol{\theta})$ ("the $\boldsymbol{\theta}$ that minimizes"). For continuous data $p$ is a density;
that is fine, since only the location of the minimum matters. Dividing by $n$ (the mean NLL) does not
move the minimum either; losses are usually reported as means.

**Gaussian MLE.** $\log p(x \mid \mu, \sigma) = -\frac12\log(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2}$ (the log of the A5.2 density). Sum and negate:

$$\mathcal{L}_{\text{NLL}}(\mu, \sigma) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2.$$

*The mean.* Only the second term depends on $\mu$. Its derivative is $\frac{1}{2\sigma^2}\sum 2(x_i - \mu)(-1) = -\frac{1}{\sigma^2}\sum(x_i - \mu)$.
Setting it to zero: $\sum x_i = n\mu$, so $\hat{\mu} = \bar{x}$, **the sample mean**. Minimizing a sum of squared
distances to $\mu$ is the same problem, so "the Gaussian MLE of the mean" and "the least-squares
constant" are one thing.

*The variance.* Write $\frac{n}{2}\log(2\pi\sigma^2) = \frac{n}{2}\log 2\pi + n\log\sigma$. The derivative with respect to $\sigma$ is
$\frac{n}{\sigma} - \frac{1}{\sigma^3}\sum(x_i - \mu)^2$. Setting it to zero and putting in $\hat{\mu}$:

$$\hat{\sigma}^2 = \frac1n\sum_{i=1}^n (x_i - \bar{x})^2.$$

It divides by $n$, not $n - 1$: the MLE of the variance is the slightly biased one from A3.3 and A5.1
($\mathbb{E}[\hat{\sigma}^2] = \frac{n - 1}{n}\sigma^2$, stated). MLE optimizes fit to *this* sample, not unbiasedness; for large $n$ the
difference vanishes.

*By gradient descent.* To optimize over $\sigma > 0$ without a constraint, use $s = \log\sigma$ (any real $s$ gives a
positive $\sigma = e^s$). The mean NLL per example is $\frac12\log 2\pi + s + \frac{(x - \mu)^2}{2e^{2s}}$, with gradient

$$\frac{\partial}{\partial\mu} = -\frac{\overline{(x - \mu)}}{\sigma^2}, \qquad \frac{\partial}{\partial s} = 1 - \frac{\overline{(x - \mu)^2}}{\sigma^2},$$

where the bars are averages over the data. (For $s$: the derivative of $e^{-2s}$ is $-2e^{-2s}$, so the last
term differentiates to $-\frac{(x - \mu)^2}{e^{2s}}$.) Both are zero exactly at $\hat{\mu}$ and $\hat{\sigma}$.

**Bernoulli MLE and log-loss.** For one 0/1 outcome, $p(y \mid p) = p^y(1 - p)^{1 - y}$: it equals $p$ when $y = 1$ and
$1 - p$ when $y = 0$. So $-\log p(y \mid p) = -[y\log p + (1 - y)\log(1 - p)]$: the **log-loss** (binary
cross-entropy) of one example. Over $n$ flips with $k$ heads,
$\mathcal{L}_{\text{NLL}}(p) = -[k\log p + (n - k)\log(1 - p)]$; its derivative $-\frac{k}{p} + \frac{n - k}{1 - p}$ is zero when
$k(1 - p) = (n - k)p$, so $\hat{p} = k/n$. In logistic regression ([B2.1](../B/b2_1_logistic_regression.md)) each
example gets its own $p_i = \sigma(\mathbf{x}_i^\top\mathbf{w} + b)$, and minimizing the mean log-loss is exactly MLE for $\mathbf{w}$ and $b$.

**Squared error is Gaussian MLE.** Model a target as a line plus Gaussian noise: $y_i = \mathbf{x}_i^\top\mathbf{w} + \varepsilon_i$ with
$\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ independent, so $y_i \sim \mathcal{N}(\mathbf{x}_i^\top\mathbf{w}, \sigma^2)$. The Gaussian NLL above, with $\mu$ replaced
by $\mathbf{x}_i^\top\mathbf{w}$, is

$$\mathcal{L}_{\text{NLL}}(\mathbf{w}, \sigma) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\lVert\mathbf{y} - \mathbf{X}\mathbf{w}\rVert^2.$$

For any fixed $\sigma$, the first term does not involve $\mathbf{w}$ and the second is a positive constant times the
sum of squared residuals. So the $\mathbf{w}$ that minimizes the NLL is the $\mathbf{w}$ that minimizes the MSE: least
squares (A2.4). The same derivative step as before then gives $\hat{\sigma}^2 = \frac1n\lVert\mathbf{y} - \mathbf{X}\hat{\mathbf{w}}\rVert^2$, the MSE at the
fitted line. Squared error is the right loss *if* the noise is Gaussian with constant variance; when it is
not, MLE tells you what to use instead (exercise 1(c)).

**MAP.** Treat the parameters as uncertain, with a prior $p(\mathbf{w})$. Bayes' theorem (A5.3) with the data as
evidence:

$$p(\mathbf{w} \mid \text{data}) = \frac{p(\text{data} \mid \mathbf{w})\,p(\mathbf{w})}{p(\text{data})}.$$

The denominator does not depend on $\mathbf{w}$, so maximizing the posterior means minimizing

$$-\log p(\mathbf{w} \mid \text{data}) = \underbrace{-\log p(\text{data} \mid \mathbf{w})}_{\text{NLL}} \underbrace{- \log p(\mathbf{w})}_{\text{penalty}} + \text{const}.$$

**A Gaussian prior is L2 regularization.** Give each penalized weight an independent prior
$w_j \sim \mathcal{N}(0, \tau^2)$: "probably within about $2\tau$ of zero". Then $-\log p(\mathbf{w}) = \sum_j\frac{w_j^2}{2\tau^2} + \text{const}$. With the
Gaussian NLL for fixed $\sigma$, the MAP objective is

$$\frac{1}{2\sigma^2}\lVert\mathbf{y} - \mathbf{X}\mathbf{w}\rVert^2 + \frac{1}{2\tau^2}\lVert\mathbf{w}\rVert^2 \;\propto\; \lVert\mathbf{y} - \mathbf{X}\mathbf{w}\rVert^2 + \lambda\lVert\mathbf{w}\rVert^2, \qquad \lambda = \frac{\sigma^2}{\tau^2}$$

(multiply through by $2\sigma^2$, which does not move the minimum). That is **ridge regression**. Its
gradient (A4.2 patterns) is $-2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w}) + 2\lambda\mathbf{w}$; setting it to zero gives
$(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})\mathbf{w} = \mathbf{X}^\top\mathbf{y}$. The intercept is normally *not* given a prior (shifting all targets by a
constant should shift the intercept freely), so replace $\mathbf{I}$ by $\mathbf{D}$, the identity with a 0 in the
intercept's position: $\hat{\mathbf{w}}_{\text{MAP}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{D})^{-1}\mathbf{X}^\top\mathbf{y}$, shapes `(d, d)(d,)` → `(d,)`. Reading
$\lambda = \sigma^2/\tau^2$: noisier data (large $\sigma$) or a tighter prior (small $\tau$) means more regularization.
More data makes $\mathbf{X}^\top\mathbf{X}$ grow like $n$ while $\lambda$ stays fixed, so the prior's influence fades. (A
Laplace prior gives the L1 penalty $\lambda\lVert\mathbf{w}\rVert_1$, the lasso; [B2.2](../B/b2_2_regularization.md).)

## Worked example by hand

**1. Gaussian MLE for $x = (2, 4, 6)$.** $\hat{\mu} = 12/3 = 4$. Squared deviations $4, 0, 4$, so $\hat{\sigma}^2 = 8/3 = 2.667$
(the $n - 1$ version would be 4). With $\sigma = 2$ fixed, $\mathcal{L}_{\text{NLL}}(\mu) = \frac32\log(8\pi) + \frac18\sum(x_i - \mu)^2$.
$\frac32\log(8\pi) = \frac32 \times 3.2242 = 4.8363$. At $\mu = 4$: $\sum(x_i - \mu)^2 = 8$, NLL $= 4.8363 + 1 = 5.8363$. At $\mu = 3$ or
$5$: the squares are $1 + 1 + 9 = 11$, NLL $= 4.8363 + 1.375 = 6.2113$. The NLL is a parabola in $\mu$ with its bottom
at the mean.

**2. Bernoulli, 3 heads in 4 flips.** Likelihoods $0.0625$, $0.1055$, $0.0729$ at $p = 0.5, 0.75, 0.9$. Mean NLL at
$p = 0.75$: $-\frac14(3\ln 0.75 + \ln 0.25) = -\frac14(3 \times (-0.2877) - 1.3863) = \frac{0.8630 + 1.3863}{4} = 0.5623$. At $p = 0.5$
it is $\ln 2 = 0.6931$, the log-loss of a coin-flip guess.

**3. MAP for a one-weight line** $\hat{y} = wx$ through $(1, 1)$ and $(2, 3)$. Least squares minimizes
$(1 - w)^2 + (3 - 2w)^2$; the derivative $-2(1 - w) - 4(3 - 2w) = 0$ gives $10w = 14$, $w = 1.4$, i.e. $\frac{\sum xy}{\sum x^2} = \frac{7}{5}$.
Adding $\lambda w^2$ adds $2\lambda w$ to the derivative: $w = \frac{\sum xy}{\sum x^2 + \lambda}$. With $\lambda = 1$, $w = 7/6 = 1.1667$; with
$\lambda = 5$, $w = 0.7$. The prior pulls towards 0, and harder the larger $\lambda$ is relative to $\sum x^2$ (the
amount of data).

```text
== worked example: Gaussian MLE for x = (2, 4, 6) ==
mu_hat = mean = 4.0000   sigma2_hat = (4 + 0 + 4)/3 = 2.6667   (ddof=1 would give 4.0000)
NLL with sigma = 2 at mu = 3.0: 6.2113
NLL with sigma = 2 at mu = 4.0: 5.8363
NLL with sigma = 2 at mu = 5.0: 6.2113

== worked example: Bernoulli MLE for 3 heads in 4 flips ==
p = 0.5 : likelihood p^3 (1 - p) = 0.0625   mean NLL (log-loss) = 0.6931
p = 0.75: likelihood p^3 (1 - p) = 0.1055   mean NLL (log-loss) = 0.5623
p = 0.9 : likelihood p^3 (1 - p) = 0.0729   mean NLL (log-loss) = 0.6547

== worked example: MAP for a one-weight line ==
least squares (MLE) w = sum(xy)/sum(x^2) = 7/5 = 1.4000
MAP with lambda = sigma^2/tau^2 = 1.0: w = 7/(5 + 1) = 1.1667
MAP with lambda = sigma^2/tau^2 = 5.0: w = 7/(5 + 5) = 0.7000
```

## Key plot

![Three panels. Left: the Gaussian mean negative log-likelihood as a function of mu is a parabola with its minimum at the sample mean 5.08, and gradient-descent iterates slide down it from mu = 0. Middle: the Bernoulli log-loss of 100 flips as a function of p is lowest at 0.28, the observed fraction of heads. Right: the MAP slope against the prior standard deviation tau on a log scale rises from 0 for tau = 0.003 to the least-squares slope 0.552 for tau above 1, with scikit-learn Ridge results as black diamonds on the curve](img/a5_5_nll_and_map.png)

Left and middle: the NLL as a loss curve; its minimum is the MLE, which is the sample mean and the
observed fraction. Right: the prior's strength as a dial. A very tight prior (small $\tau$, large $\lambda$) pins
the slope at 0; a vague prior gives back the MLE; in between, the data and the prior compromise.
`Ridge(alpha=σ²/τ²)` lands exactly on the MAP curve.

## From scratch in NumPy

**Gaussian MLE, two ways.** The closed form is two lines (`mean`, and the mean squared deviation). Gradient
descent uses the A4.4 loop, generalized to take any gradient function, on $(\mu, s = \log\sigma)$. The gradient is
checked numerically first (A4.1), as every hand-written gradient should be:

```python
def gauss_mean_nll_grad(theta: FloatArray, x: FloatArray) -> FloatArray:
    """Gradient of gauss_mean_nll with respect to (mu, s). -> (2,)."""
    mu, s = theta
    var = np.exp(2 * s)
    d_mu = -(x - mu).mean() / var  # d/dmu of (x - mu)^2 / (2 var), averaged
    d_s = 1.0 - ((x - mu) ** 2).mean() / var  # d/ds of s + (x - mu)^2 exp(-2s) / 2
    return np.array([d_mu, d_s])
```

```text
== data ==
line: X (50, 2)  y (50,)   Gaussian sample: (40,), first 5 = [6.0054, 6.9794, 4.6714, 2.8513, 6.7461]
coin: 100 flips, 28 heads

== from scratch: Gaussian MLE, closed form and gradient descent ==
closed form: mu_hat = 5.0800   sigma_hat = 1.9724   (true 5.0 and 2.0, n = 40)
gradient check at (mu, s) = [3.0, 0.2]: analytic [-1.3943, -4.5077]  numerical [-1.3943, -4.5077]: OK
gradient descent from (mu, s) = (0, 0), lr = 0.1: 1019 steps   mu = 5.0800   sigma = exp(s) = 1.9724
gradient descent matches the closed form: OK
```

The estimates (5.08, 1.97) are close to the true (5, 2) but not equal: with 40 draws, the standard error
of the mean is $2/\sqrt{40} = 0.32$ (A5.4), so being off by 0.08 is normal. MLE recovers the parameters that
best explain *this sample*.

**Bernoulli MLE.** The NLL over a fine grid of $p$ has its minimum at the observed fraction:

```text
== from scratch: Bernoulli MLE ==
p_hat = heads / flips = 0.2800   grid search minimum of the NLL at p = 0.2800
mean NLL at p_hat = 0.5930   at p = 0.5: 0.6931
```

**Squared error is Gaussian MLE, numerically.** On the A2.4 line data, the lab minimizes the MSE over $\mathbf{w}$
and, separately, the Gaussian NLL over $(\mathbf{w}, \log\sigma)$ with `scipy.optimize.minimize` (BFGS, A4.4):

```text
== from scratch: squared error is Gaussian MLE ==
least squares w = [1.7575, 0.5523]
argmin MSE      w = [1.7575, 0.5523]
argmin Gaussian NLL over (w, sigma): w = [1.7575, 0.5523]   sigma = 0.9967   sqrt(MSE at w) = 0.9967
MSE minimizer = Gaussian NLL minimizer = least squares: OK
```

Same weights three ways, and the fitted noise level is the root of the MSE, as derived.

**MAP is ridge.** With noise $\sigma = 1$ (the value the data was generated with) and a prior $w_1 \sim \mathcal{N}(0, 0.1^2)$ on
the slope, $\lambda = 1/0.01 = 100$:

```python
def neg_log_posterior(w: FloatArray) -> float:
    """-log p(w | data) up to a constant: sum r^2 / (2 sigma^2) + w1^2 / (2 tau^2). w (2,) -> scalar."""
    r = y_line - X @ w
    return float(r @ r / (2 * NOISE_SD**2) + w[1] ** 2 / (2 * PRIOR_SD**2))


w_map_closed = np.linalg.solve(X.T @ X + lam * D, X.T @ y_line)  # (X^T X + lambda D) w = X^T y
```

```text
== from scratch: MAP with a Gaussian prior = ridge ==
sigma = 1.0, prior sd of slope tau = 0.1  ->  lambda = sigma^2 / tau^2 = 100
MAP by minimizing the negative log posterior: w = [2.312, 0.4471]
ridge closed form (X^T X + lambda D)^-1 X^T y: w = [2.312, 0.4471]
MLE (no prior): w = [1.7575, 0.5523]   the slope shrank from 0.5523 to 0.4471
MAP = ridge: OK
```

The prior pulled the slope towards 0 by 19%, and the unpenalized intercept moved *up* to compensate,
so the line still passes through the middle of the data. (Here the prior is deliberately too tight for
illustration: the true slope 0.5 is five prior standard deviations from 0.)

## With SciPy and scikit-learn

```text
== with scipy and scikit-learn ==
stats.norm.fit: loc = 5.0800   scale = 1.9724   (our closed form 5.0800, 1.9724)
stats.norm.fit is the MLE (divides by n): OK
sklearn log_loss at p_hat = 0.5930   our Bernoulli mean NLL = 0.5930
LinearRegression: w = [1.7575, 0.5523]   (Gaussian MLE)
Ridge(alpha = 100): w = [2.312, 0.4471]   (MAP with tau = 0.1)
matches scikit-learn: OK
```

- Every `scipy.stats` distribution has `.fit(data)`, which returns maximum-likelihood parameters
  (`stats.norm.fit` returns $\hat{\mu}$ and the divide-by-$n$ $\hat{\sigma}$).
- `sklearn.metrics.log_loss(y, p)` is the mean Bernoulli NLL. It clips probabilities away from 0 and 1;
  your own version must too, or one confident wrong prediction gives `inf` (`log(0)`).
- `Ridge(alpha=λ)` minimizes $\lVert\mathbf{y} - \mathbf{X}\mathbf{w} - b\rVert^2 + \alpha\lVert\mathbf{w}\rVert^2$ and does not penalize the intercept, so
  `alpha` is exactly $\sigma^2/\tau^2$ in the MAP reading. `LogisticRegression(C=...)` puts its constant on the
  other side: $C$ multiplies the log-loss sum, so a *larger* $C$ is *less* regularization (exercise 2 shows
  $C = \tau^2$). VERIFY (as of 2026-10), scikit-learn 1.9.1: `LogisticRegression`'s `penalty` argument is
  deprecated; `C=np.inf` means no penalty.

## In practice

- **When to use it:** whenever you choose a loss, write down the noise model it implies and check it
  against the residuals. Whenever you choose a regularization strength, read it as a prior: how far
  from zero do you believe the weights can plausibly be, in units where the features are standardized?
  (Standardize first: a Gaussian prior with one $\tau$ for every weight only makes sense when the features
  share a scale.)
- **Failure modes:**
  - *Outliers under squared error.* Gaussian noise makes a 10σ error essentially impossible, so the MLE
    bends the whole fit to reduce it. You notice a few huge residuals and a fit that changes a lot when
    they are removed. Use MAE (Laplace noise), Huber loss, or model the log of the target.
  - *Non-constant noise.* Errors that grow with the prediction (prices, durations) violate "one $\sigma$".
    You notice a funnel shape in the residual-versus-prediction plot. Model the log, or predict $\sigma$ too.
  - *Perfect separation in logistic regression.* If a feature splits the classes perfectly, the
    likelihood keeps increasing as the weight grows to infinity: the MLE does not exist. You notice weights
    growing without bound and a solver warning. A prior (L2, i.e. finite $C$) fixes it; this is one reason
    scikit-learn regularizes by default.
  - *`log(0)`.* A predicted probability of exactly 0 or 1 for the wrong class gives an infinite loss. You
    notice `inf` or `nan` losses. Clip probabilities, or compute log-loss from logits ([C1.3](../C/c1_3_autograd_engine.md)
    onward does this).
  - *Penalizing the intercept, or unscaled features.* The intercept shrinks towards 0 for no reason, and
    features in large units are barely regularized while small-unit ones are crushed. You notice
    coefficients changing when you change units. Exclude the intercept; standardize.
- **Defaults that work:** MSE for roughly symmetric noise of constant size; log-loss for probabilities;
  standardize features and tune $\lambda$ (or $C$) on a log grid with cross-validation ([B4.3](../B/b4_3_cross_validation_tuning.md));
  compute losses from logits; plot residuals before trusting a loss choice.

## Exercises

Save your work to `my-work/A/a5_5_mle_map.md` (or `.py`) and ask for a review.

1. **By hand:** (a) Counts of support tickets in four hours were 2, 0, 3 and 1. Assuming a Poisson model
   with rate $\lambda$ (A5.2's pmf; here $\lambda$ is the rate, not a regularization strength), write the NLL, differentiate,
   and find $\hat{\lambda}$. (b) For $x = (2, 4, 6)$ with known $\sigma^2 = 4$ and a prior $\mu \sim \mathcal{N}(0, \tau^2)$, derive the MAP estimate
   of $\mu$ as a formula, then evaluate it for $\tau^2 = 1$ and for $\tau^2 = 100$. Explain both numbers. (c) Suppose the
   noise in $y_i = \mathbf{x}_i^\top\mathbf{w} + \varepsilon_i$ has the Laplace density $p(\varepsilon) = \frac{1}{2b}e^{-\lvert\varepsilon\rvert/b}$ with a fixed $b > 0$. Show that the MLE of
   $\mathbf{w}$ minimizes the mean absolute error.
2. **Code:** Generate $n = 200$ examples: $x \sim \mathcal{N}(0, 1)$, then $y \sim \text{Bernoulli}(\sigma(2x - 0.5))$ (`SEED = 0`, draw $x$
   first, then one uniform per example). (a) Fit $w$ and $b$ by gradient descent on the mean Bernoulli NLL
   (the gradient is $\frac1n\mathbf{X}^\top(\mathbf{p} - \mathbf{y})$ for the design matrix with a ones column; A4.3) and compare with
   `LogisticRegression(C=np.inf)`. (b) Add a Gaussian prior $w \sim \mathcal{N}(0, \tau^2)$ on the weight only, with $\tau^2 = 0.05$;
   fit the MAP estimate by gradient descent and show that it matches `LogisticRegression(C=0.05)`. Derive
   why $C = \tau^2$. (c) Repeat (a) and (b) on only the first 20 examples. What happens to the gap between MLE
   and MAP, and why?
3. **Think:** A food-delivery company predicts delivery time with a linear model trained on MSE. Most
   deliveries take 20–40 minutes, but about 1% take over 2 hours (a courier's bike broke, a restaurant
   lost the order). What noise model does MSE assume, and how do these deliveries distort the model?
   Propose at least two alternatives, say what probability model each corresponds to, and how you would
   choose between them for this product.

## Takeaways

1. The likelihood scores parameters by how probable they make the observed data; in practice you
   minimize the negative log-likelihood, a sum over examples.
2. Gaussian MLE: $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac1n\sum(x_i - \bar{x})^2$ (divides by $n$). Bernoulli MLE: $\hat{p} = k/n$.
3. Squared error is the Gaussian NLL, log-loss is the Bernoulli NLL; choosing a loss is choosing a noise model,
   so check the residuals.
4. MAP minimizes NLL $- \log$ prior; a Gaussian prior $\mathcal{N}(0, \tau^2)$ on the weights is L2 regularization with
   $\lambda = \sigma^2/\tau^2$ (ridge); `LogisticRegression`'s $C$ is the prior variance.
5. The prior matters most with little data and fades as $n$ grows; do not penalize the intercept, and
   standardize features so one prior width fits all weights.

**What builds on this:** [A6.4](a6_4_entropy_cross_entropy_kl.md) (cross-entropy is the NLL of a
categorical model); [A6.5](a6_5_bayesian_inference.md) (the full posterior, not just its peak);
[B1.5](../B/b1_5_linear_regression.md) (linear regression in practice);
[B2.1](../B/b2_1_logistic_regression.md) (logistic regression as Bernoulli MLE);
[B2.2](../B/b2_2_regularization.md) (L1, L2 and elastic net as priors);
[B5.3](../B/b5_3_gmm_em.md) (MLE for mixtures with EM); [C2.1](../C/c2_1_tensors_autograd_modules.md)
onward (every PyTorch loss is an NLL).
