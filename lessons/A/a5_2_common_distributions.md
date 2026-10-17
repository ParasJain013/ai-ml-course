# A5.2 Common distributions

**Core** · Part A, week 5 (course week 05) · ~1.5 h · Lab:
[`labs/A/a5_2_common_distributions.py`](../../labs/A/a5_2_common_distributions.py)

## Why it matters

A handful of distributions describe most of the randomness an ML engineer meets: yes/no outcomes
(did the user click?), counts of successes (how many of 10 emails were opened?), counts of events
(how many requests this second?), categories (which class, which next token?), and measurements
(a height, a sensor reading, a noise term). Each comes with a known mean, variance and shape. Picking
the right one tells you what a model's output should look like ([B2.1](../B/b2_1_logistic_regression.md)
outputs a Bernoulli probability, a language model outputs a categorical distribution over tokens,
[F2.4](../F/f2_4_decoding.md)), which loss to train with ([A5.5](a5_5_mle_map.md)), and when the data
does not fit the assumption. This lesson builds every sampler from one uniform random number
generator, so none of them is a black box.

## Prerequisite check

- Can you compute $\mathbb{E}[X]$ and $\operatorname{Var}[X]$ from a table of values and probabilities, and use
  $\operatorname{Var}[aX + b] = a^2\operatorname{Var}[X]$? If not, revisit [A5.1](a5_1_random_variables_expectation.md).
- Can you say what `np.cumsum`, `np.bincount` and `np.searchsorted` return for a small array? If not,
  revisit [A1.2](a1_2_numpy_arrays.md).
- Can you read a covariance matrix (variances on the diagonal, covariances off it) and say what its
  eigenvectors mean for a scatter plot? If not, revisit [A3.3](a3_3_covariance_matrices.md).
- Do you know what a Cholesky factor is ($\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$, $\mathbf{L}$ lower triangular)? If not, revisit
  [A3.4](a3_4_numerical_stability.md).
- Can you overlay a histogram and a line on one matplotlib axis? If not, revisit [A1.4](a1_4_matplotlib.md).

## Intuition

### Discrete and continuous

A **discrete** random variable takes separate values (0, 1, 2, …), and its distribution is a
**probability mass function (pmf)**: the probability of each value, adding up to 1. A **continuous**
one takes any value in a range, and the probability of hitting any one exact value is zero. Its
distribution is a **probability density function (pdf)**: probability *per unit of x*. Probabilities
are areas under the density. A density can be larger than 1 (a uniform on $[0, 0.5]$ has density 2)
because it is a rate, not a probability; only its total area must be 1.

Both kinds have a **cumulative distribution function (CDF)**, $F(x) = P(X \le x)$: it climbs from 0 to 1,
in steps for a discrete variable and smoothly for a continuous one.

### Seven distributions, seven situations

| Situation | Distribution | Example |
|---|---|---|
| One yes/no outcome | **Bernoulli**($p$) | did this user click? |
| Number of yeses in $n$ independent tries | **Binomial**($n$, $p$) | clicks among 10 impressions |
| Number of events in an interval, at a steady rate | **Poisson**(rate) | requests per second, typos per page |
| One of $K$ categories | **Categorical**($p_1, \dots, p_K$) | the class of an image, the next token |
| Any value in a range equally likely | **Uniform**($a$, $b$) | a random hash bucket, `rng.random()` |
| A measurement that is a sum of many small effects | **Gaussian**($\mu$, $\sigma^2$) | heights, sensor noise, model residuals |
| Several correlated measurements together | **Multivariate Gaussian**($\boldsymbol{\mu}$, $\boldsymbol{\Sigma}$) | height and weight, an embedding cloud |

### Every sampler starts from a uniform number

A computer produces uniform numbers in $[0, 1)$ (`rng.random()`, A1.1). Everything else is a
transformation of them:

- **Bernoulli:** draw $u$; output 1 if $u < p$. The chance that $u$ lands below $p$ is exactly $p$.
- **Categorical (inverse CDF):** cut $[0, 1)$ into consecutive pieces of lengths $p_1, p_2, \dots, p_K$. Draw
  $u$; the piece it lands in is the class. The piece boundaries are the running totals (the CDF), so the
  search is `np.searchsorted(cumsum, u)`.
- **Binomial:** add up $n$ Bernoullis.
- **Uniform($a$, $b$):** stretch and shift, $a + (b - a)u$.
- **Gaussian (Box–Muller):** turn two uniforms into a random radius and a random angle, and read off
  the two coordinates. The math section shows why the result is exactly Gaussian.
- **Multivariate Gaussian:** take independent standard Gaussians and transform them with a matrix,
  the way [A2.1](a2_1_matrices_as_linear_maps.md) transformed a circle of points into an ellipse.

> **Bridge:** the inverse-CDF sampler is weighted load balancing. To route 20% of traffic to server
> 0, 50% to server 1 and 30% to server 2, a load balancer hashes each request to a number in $[0, 1)$
> and compares it with the cumulative weights 0.2 and 0.7. **Where it breaks:** a load balancer wants
> the *same* request to land on the same server (a deterministic hash); a sampler wants independent
> draws, so it needs a fresh $u$ every time.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $p$ | success probability of one Bernoulli trial | scalar in $[0, 1]$ |
| $n$ | number of trials (binomial) | int |
| $k$ | a count: successes or events | int $\ge 0$ |
| $\binom{n}{k}$ | "$n$ choose $k$" $= \frac{n!}{k!(n - k)!}$: the number of ways to pick which $k$ of $n$ trials succeed | int |
| $\lambda$ | Poisson rate: the mean number of events per interval (see the note below) | scalar $> 0$ |
| $p_1, \dots, p_K$ | categorical class probabilities, $\mathbf{p}$ | `(K,)` |
| $a, b$ | ends of a uniform range | scalars |
| $\mu, \sigma^2$ | Gaussian mean and variance | scalars |
| $F(x)$ | CDF, $P(X \le x)$ | scalar |
| $u, u_1, u_2$ | uniform draws on $[0, 1)$ | scalars |
| $\boldsymbol{\mu}$, $\boldsymbol{\Sigma}$ | mean vector and covariance matrix of a multivariate Gaussian | `(d,)`, `(d, d)` |
| $\mathbf{z}$ | vector of independent standard Gaussians | `(d,)` |
| $\mathbf{L}$ | Cholesky factor, $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$ | `(d, d)` |

> **In this lesson only,** $\lambda$ is the Poisson rate, not the regularization strength or an
> eigenvalue. Every library and textbook uses $\lambda$ for it.

**Bernoulli($p$).** $P(X = 1) = p$, $P(X = 0) = 1 - p$. Mean $p$. Since $X^2 = X$ for 0/1 values,
$\mathbb{E}[X^2] = p$ and $\operatorname{Var}[X] = p - p^2 = p(1 - p)$: largest at $p = 0.5$, zero at $p = 0$ or 1.

**Binomial($n$, $p$).** $X = $ the number of successes in $n$ independent Bernoulli($p$) trials.
One particular sequence with $k$ successes and $n - k$ failures has probability $p^k(1 - p)^{n - k}$ (independent
trials multiply), and there are $\binom{n}{k}$ such sequences, so

$$P(X = k) = \binom{n}{k}p^k(1 - p)^{n - k}, \quad k = 0, 1, \dots, n.$$

$X$ is a sum of $n$ independent Bernoullis, so by A5.1 the means add and the variances add:
mean $np$, variance $np(1 - p)$.

**Poisson($\lambda$).** Split an interval into $n$ tiny slices, each holding an event with a small
probability $\lambda/n$, independently. The count is Binomial($n$, $\lambda/n$) with mean $\lambda$. As $n \to \infty$ the
binomial pmf becomes (stated, not derived; the lab's "try this" compares the two)

$$P(X = k) = e^{-\lambda}\frac{\lambda^k}{k!}, \quad k = 0, 1, 2, \dots$$

Mean $\lambda$ and variance $\lambda$: the binomial variance $n\frac{\lambda}{n}(1 - \frac{\lambda}{n})$ tends to $\lambda$. **Mean equals
variance** is the Poisson's fingerprint; real counts often have variance larger than the mean
(**overdispersion**), which is a sign the Poisson model is wrong.

**Categorical($\mathbf{p}$).** $P(X = j) = p_j$ for classes $j = 1, \dots, K$ (indices $0, \dots, K - 1$ in code), with
$\sum_j p_j = 1$. A Bernoulli is the $K = 2$ case. The mean of a class *index* is rarely meaningful (class
2 is not "twice" class 1); the probabilities are the distribution. This is the output of every
softmax classifier.

**Inverse-CDF sampling.** For a categorical, let $F_j = p_1 + \dots + p_j$ (the running totals, $F_K = 1$).
Draw $u$ and return the first $j$ with $u < F_j$. Class $j$ wins exactly when $F_{j-1} \le u < F_j$, an
interval of length $p_j$, so it wins with probability $p_j$. The same idea works for any distribution
whose CDF you can invert: return $F^{-1}(u)$.

**Uniform($a$, $b$).** Density $\frac{1}{b - a}$ on $[a, b]$, zero elsewhere. Mean $\frac{a + b}{2}$ (the middle). For $U$
uniform on $[0, 1]$: $\mathbb{E}[U^2] = \int_0^1 u^2\,du = \frac13$, so $\operatorname{Var}[U] = \frac13 - \frac14 = \frac{1}{12}$. Then
$X = a + (b - a)U$ has variance $\frac{(b - a)^2}{12}$ by the $a^2$ rule.

**Gaussian $\mathcal{N}(\mu, \sigma^2)$.** Density

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\Big(-\frac{(x - \mu)^2}{2\sigma^2}\Big).$$

Read it from the inside out: $(x - \mu)^2/\sigma^2$ is the squared distance from the mean in units of $\sigma$; the
exponential turns that into a bell that falls off very fast; the constant in front makes the area 1.
Mean $\mu$, variance $\sigma^2$. Two facts used constantly:

- **Standardizing:** if $X \sim \mathcal{N}(\mu, \sigma^2)$ then $Z = (X - \mu)/\sigma \sim \mathcal{N}(0, 1)$, the **standard normal**, and
  conversely $X = \mu + \sigma Z$. So one standard-normal sampler gives every Gaussian.
- **68–95–99.7:** a Gaussian lands within 1, 2 and 3 standard deviations of its mean with probability
  0.6827, 0.9545 and 0.9973. A value more than 5 standard deviations above the mean happens about once
  in 3.5 million draws.

**Why Box–Muller works.** Take two independent standard normals $(Z_1, Z_2)$. Their joint density is the
product $\frac{1}{2\pi}e^{-(z_1^2 + z_2^2)/2}$, which depends only on the squared radius $r^2 = z_1^2 + z_2^2$: the point
cloud is perfectly round. So the angle is uniform on $[0, 2\pi)$, and it remains to get the radius right.
For this density, $P(r^2 > t) = e^{-t/2}$ (stated: integrate the density over the region outside the circle of
squared radius $t$). Now set $r^2 = -2\log u_1$ with $u_1$ uniform on $(0, 1]$. Then
$P(-2\log u_1 > t) = P(u_1 < e^{-t/2}) = e^{-t/2}$, exactly what is needed. So

$$r = \sqrt{-2\log u_1}, \quad \theta = 2\pi u_2, \quad z_1 = r\cos\theta, \quad z_2 = r\sin\theta$$

are two independent standard normals from two uniforms. ($u_1$ must avoid 0, where the log is
$-\infty$; the lab uses `1 - rng.random()`, which lies in $(0, 1]$.)

**Multivariate Gaussian $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$.** A vector $\mathbf{x}$, shape `(d,)`, with mean $\boldsymbol{\mu}$ `(d,)` and
covariance $\boldsymbol{\Sigma}$ `(d, d)` (symmetric, positive definite). Its density is

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}\lvert\boldsymbol{\Sigma}\rvert^{1/2}}\exp\!\Big(-\tfrac12(\mathbf{x} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\Big),$$

where $\lvert\boldsymbol{\Sigma}\rvert$ is the determinant. The quadratic form in the exponent, a scalar `(1, d)(d, d)(d, 1)`, is
the squared **Mahalanobis distance**: distance from the mean measured in standard deviations *along
each eigen-direction* of $\boldsymbol{\Sigma}$. Points with equal density lie on ellipses whose axes are the eigenvectors
of $\boldsymbol{\Sigma}$, with half-lengths proportional to the square roots of its eigenvalues (A3.3). With $d = 1$ the
formula is the one-dimensional density above.

**Sampling it.** Draw $\mathbf{z}$ `(d,)` of independent standard normals, and set $\mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\mathbf{z}$ with
$\mathbf{L}\mathbf{L}^\top = \boldsymbol{\Sigma}$. Its mean is $\boldsymbol{\mu}$ (since $\mathbb{E}[\mathbf{z}] = \mathbf{0}$). Its covariance matrix is
$\mathbb{E}[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^\top]$, a `(d, d)` matrix whose $(j, k)$ entry is $\operatorname{Cov}[x_j, x_k]$:

$$\mathbb{E}[(\mathbf{L}\mathbf{z})(\mathbf{L}\mathbf{z})^\top] = \mathbf{L}\,\mathbb{E}[\mathbf{z}\mathbf{z}^\top]\,\mathbf{L}^\top = \mathbf{L}\mathbf{I}\mathbf{L}^\top = \boldsymbol{\Sigma}.$$

$\mathbb{E}[\mathbf{z}\mathbf{z}^\top] = \mathbf{I}$ because its diagonal entries are $\mathbb{E}[z_j^2] = 1$ (variance 1, mean 0) and its
off-diagonal entries are $\mathbb{E}[z_jz_k] = \mathbb{E}[z_j]\mathbb{E}[z_k] = 0$ (independent). This is the matrix version of
$\operatorname{Var}[aX] = a^2\operatorname{Var}[X]$. In code, with rows as examples, $\mathbf{Z}$ is `(n, d)` and each row must become
$\mathbf{L}\mathbf{z}_i$, so the batch is `Z @ L.T`, shape `(n, d)`.

## Worked example by hand

**Binomial(4, 0.5).** $\binom{4}{k} = 1, 4, 6, 4, 1$ and $0.5^4 = \frac{1}{16}$, so the pmf is
$\frac{1}{16}, \frac{4}{16}, \frac{6}{16}, \frac{4}{16}, \frac{1}{16} = 0.0625, 0.25, 0.375, 0.25, 0.0625$. Mean $4 \times 0.5 = 2$, variance $4 \times 0.25 = 1$.

**Poisson with $\lambda = 2$.** $e^{-2} = 0.1353$. $P(0) = 0.1353$, $P(1) = 0.1353 \times 2 = 0.2707$,
$P(2) = 0.1353 \times \frac{4}{2} = 0.2707$. $P(k \ge 3) = 1 - (0.1353 + 0.2707 + 0.2707) = 0.3233$. (Each term is the previous
one times $\lambda/k$: a cheap way to compute the pmf.)

**Categorical, $\mathbf{p} = (0.2, 0.5, 0.3)$.** Running totals $(0.2, 0.7, 1.0)$. $u = 0.65$: the first total
above 0.65 is 0.7, class 1. $u = 0.15$: class 0.

**Box–Muller, $u_1 = 0.5$, $u_2 = 0.125$.** $r = \sqrt{-2\ln 0.5} = \sqrt{1.3863} = 1.1774$;
$\theta = 2\pi \times 0.125 = \pi/4 = 0.7854$; $\cos\theta = \sin\theta = 0.7071$, so $z_1 = z_2 = 0.8326$.

**A 2-D Gaussian.** $\boldsymbol{\mu} = (1, -1)$, $\boldsymbol{\Sigma} = \begin{pmatrix}4 & 2.4\\2.4 & 2.25\end{pmatrix}$ (standard deviations 2 and 1.5,
correlation $2.4/(2 \times 1.5) = 0.8$, the A3.3 example). Cholesky, entry by entry from
$\mathbf{L}\mathbf{L}^\top = \begin{pmatrix}L_{11}^2 & L_{11}L_{21}\\L_{11}L_{21} & L_{21}^2 + L_{22}^2\end{pmatrix}$:
$L_{11} = \sqrt4 = 2$; $L_{21} = 2.4/2 = 1.2$; $L_{22} = \sqrt{2.25 - 1.44} = \sqrt{0.81} = 0.9$. For
$\mathbf{z} = (1, -1)$: $\mathbf{L}\mathbf{z} = (2 \times 1,\ 1.2 \times 1 + 0.9 \times (-1)) = (2, 0.3)$, so $\mathbf{x} = (3, -0.7)$.

```text
== worked example by hand ==
Binomial(4, 0.5) pmf k = 0..4: [0.0625, 0.25, 0.375, 0.25, 0.0625]
Poisson(rate 2) pmf k = 0..2: [0.1353, 0.2707, 0.2707]   P(k >= 3) = 0.3233
categorical p = [0.2, 0.5, 0.3]: cumulative = [0.2, 0.7, 1.0]   u = 0.65 -> class 1   u = 0.15 -> class 0
Box-Muller u1 = 0.5, u2 = 0.125: r = 1.1774  angle = 0.7854 rad  z = (0.8326, 0.8326)
Cholesky factor L of Sigma = [[2.0, 0.0], [1.2, 0.9]]   L L^T = [[4.0, 2.4], [2.4, 2.25]]
z = [1.0, -1.0] -> x = mu + L z = [3.0, -0.7]
Gaussian: P(|Z| < 1) = 0.6827   P(|Z| < 2) = 0.9545   P(|Z| < 3) = 0.9973
```

## Key plot

![Six panels comparing samplers with scipy.stats: bars of the from-scratch binomial, NumPy Poisson and inverse-CDF categorical frequencies sit under black diamonds of the exact probabilities; the uniform histogram is flat at height one quarter; the Box-Muller histogram traces the normal density; and 1,500 draws of the correlated 2-D Gaussian fill a tilted ellipse](img/a5_2_samplers_vs_scipy.png)

Each bar or histogram comes from 200,000 draws; each black marker or line is scipy's exact pmf or
density. They agree to within sampling noise everywhere. The bottom-right panel is the matrix
transformation at work: round clouds of independent normals, stretched and tilted by $\mathbf{L}$ into the
ellipse of $\boldsymbol{\Sigma}$.

## From scratch in NumPy

Each sampler takes the generator explicitly and touches only `gen.random()`:

```python
def sample_bernoulli(p: float, size: int, gen: np.random.Generator) -> IntArray:
    """1 with probability p, else 0. -> (size,)."""
    return (gen.random(size) < p).astype(np.int64)


def sample_binomial(n_trials: int, p: float, size: int, gen: np.random.Generator) -> IntArray:
    """Number of successes in n_trials Bernoulli(p) trials. -> (size,)."""
    return (gen.random((size, n_trials)) < p).sum(axis=1)  # (size, n_trials) -> (size,)


def sample_categorical(probs: FloatArray, size: int, gen: np.random.Generator) -> IntArray:
    """Class index 0..K-1 by inverse CDF. probs (K,) summing to 1 -> (size,)."""
    cdf = np.cumsum(probs)  # (K,)
    cdf[-1] = 1.0  # guard against rounding: the last class must catch every u < 1
    return np.searchsorted(cdf, gen.random(size), side="right")  # first class with cdf > u
```

`side="right"` returns the first index whose running total is strictly greater than $u$, which is the
rule "$u < F_j$". The guard matters: `np.cumsum([0.1] * 10)[-1]` is `0.9999999999999999` in float64, and a
$u$ above it would fall off the end and return the invalid class $K$.

```python
def sample_standard_normal(size: int, gen: np.random.Generator) -> FloatArray:
    """Standard normal N(0, 1) by Box-Muller. -> (size,)."""
    half = (size + 1) // 2
    u1 = 1.0 - gen.random(half)  # in (0, 1], so log(u1) is finite
    u2 = gen.random(half)
    r = np.sqrt(-2.0 * np.log(u1))  # (half,) radius
    theta = 2.0 * np.pi * u2  # (half,) angle
    return np.concatenate([r * np.cos(theta), r * np.sin(theta)])[:size]  # two normals per pair


def sample_mvn(mu: FloatArray, Sigma: FloatArray, size: int, gen: np.random.Generator) -> FloatArray:
    """Multivariate Gaussian N(mu, Sigma) by x = mu + L z. mu (d,), Sigma (d, d) -> (size, d)."""
    d = len(mu)
    L = np.linalg.cholesky(Sigma)  # (d, d) lower triangular, L L^T = Sigma
    Z = sample_standard_normal(size * d, gen).reshape(size, d)  # (size, d) independent N(0, 1)
    return mu + Z @ L.T  # row i is mu + L z_i; (size, d)
```

A Gaussian with any mean and variance is `mu + sigma * sample_standard_normal(...)`, and a uniform on
$[a, b)$ is `a + (b - a) * gen.random(size)`. The sample moments match the formulas:

```text
== from scratch: samplers from uniform numbers ==
distribution               mean  formula  variance  formula
Bernoulli(0.3)           0.2998   0.3000    0.2099   0.2100
Binomial(10, 0.3)        2.9998   3.0000    2.0960   2.1000
Categorical(.2,.5,.3)    1.0977   1.1000    0.4905   0.4900
Uniform(-1, 3)           1.0002   1.0000    1.3299   1.3333
Normal(2, 0.5^2)         1.9990   2.0000    0.2494   0.2500
multivariate Gaussian: sample mean = [0.999, -0.9993]   target = [1.0, -1.0]
  sample covariance = [[3.9933, 2.3882], [2.3882, 2.2357]]   target = [[4.0, 2.4], [2.4, 2.25]]
```

(The categorical row uses the class index as a number only to have something to average.)

## With SciPy

Matching means and variances is not enough: two different distributions can share both. The lab
compares the whole shape. For discrete samplers it takes the largest gap between the observed
frequency of each value and scipy's pmf; for continuous ones, the largest gap between the
**empirical CDF** (the fraction of samples at or below each $x$) and scipy's CDF. That second number is
the Kolmogorov–Smirnov statistic, which [A6.3a](a6_3a_hypothesis_tests.md) turns into a test.

```text
== with scipy.stats: the samplers match the exact distributions ==
Bernoulli vs stats.bernoulli   largest gap = 0.0002
Binomial vs stats.binom        largest gap = 0.0017
Categorical vs probs           largest gap = 0.0012
Uniform vs stats.uniform       largest gap = 0.0017
Box-Muller vs stats.norm       largest gap = 0.0021
all gaps below 3/sqrt(n) = 0.0067: OK
stats.multivariate_normal sample covariance = [[3.9991, 2.392], [2.392, 2.237]]
own multivariate sampler covariance within 0.05 of Sigma: OK
exact moments from scipy:
  binom(10, 0.3)   mean = 3.0000   variance = 2.1000
  poisson(3)       mean = 3.0000   variance = 3.0000
  uniform(-1, 3)   mean = 1.0000   variance = 1.3333
  norm(2, 0.5)     mean = 2.0000   variance = 0.2500
rng.poisson(3): mean = 3.0023   variance = 2.9984   P(0) = 0.0492 vs exact 0.0498
```

The tolerance $3/\sqrt{n}$ comes from A5.1: a frequency estimated from $n$ draws has standard deviation
$\sqrt{p(1 - p)/n} \le 0.5/\sqrt{n}$, so a correct sampler stays far below $3/\sqrt{n}$, while a wrong one (a
mis-scaled Gaussian, an off-by-one class) misses by much more at $n = 200{,}000$.

What to know about the library calls (VERIFY (as of 2026-10), SciPy 1.18.1, NumPy 2.5.4):

- `scipy.stats` objects are "frozen" distributions: `stats.binom(10, 0.3)` has `.pmf`, `.cdf`, `.ppf`
  (the inverse CDF), `.rvs` (samples), `.mean()`, `.var()` and `.stats(moments="mv")`.
- `stats.norm(loc, scale)` takes the **standard deviation** as `scale`, not the variance.
  `stats.uniform(loc, scale)` is uniform on `[loc, loc + scale]`, not `[loc, scale]`.
- `stats.multivariate_normal(mean, cov)` takes the covariance matrix; `.rvs(size=n)` returns `(n, d)`.
- For speed, NumPy's generator has its own samplers: `rng.binomial`, `rng.poisson`, `rng.normal`,
  `rng.choice(K, p=probs)` (categorical), `rng.multivariate_normal`. Pass `random_state=rng` (or a seed)
  to scipy's `.rvs` so its draws are reproducible too.

## In practice

- **When to use it:** pick the distribution from the *data-generating process*, then check it
  against the data. Yes/no → Bernoulli; successes out of a known number of tries → binomial; counts
  with no fixed maximum → Poisson; one of several labels → categorical; continuous measurements that
  are sums of many small effects → Gaussian.
- **Failure modes:**
  - *Overdispersed counts.* Support tickets per hour are bursty (outages, Monday mornings), so the
    variance is many times the mean. A Poisson model then underestimates spikes. You notice it by
    comparing the sample variance with the sample mean; use a negative binomial or model the rate.
  - *Gaussian assumed for heavy tails.* Latencies, revenue and file sizes have a few huge values; a
    Gaussian puts essentially zero probability beyond 5σ, so "5σ events" happen weekly. You notice it
    in a histogram on a log x-axis (A6.1); model the log instead.
  - *Gaussian assumed for bounded values.* A probability or a percentage cannot go below 0, but a
    Gaussian fitted to values near 0 puts mass there. You notice negative samples from your simulator.
  - *Independence assumed.* The binomial needs independent trials; ten impressions shown to the
    same user are not. You notice the observed variance of the count exceeds $np(1 - p)$.
  - *`Z @ L` instead of `Z @ L.T`.* Rows get the wrong covariance ($\mathbf{L}^\top\mathbf{L}$ instead of $\mathbf{L}\mathbf{L}^\top$), with no
    error. You notice it only if you check the sample covariance, so always check it.
- **Defaults that work:** use NumPy's generator samplers in real code (faster and tested); write your
  own only to understand them; always compare a new simulator's output with an exact pmf or CDF; for
  a covariance, sample with Cholesky and check `np.cov(X, rowvar=False)` against the target.

## Exercises

Save your work to `my-work/A/a5_2_common_distributions.md` (or `.py`) and ask for a review.

1. **By hand:** (a) For Binomial(5, 0.3), compute $P(X = 2)$, the mean and the variance. (b) For a Poisson
   with rate 1.5, compute $P(X = 0)$, $P(X \ge 1)$ and $P(X \le 2)$. (c) With $\mathbf{p} = (0.1, 0.6, 0.3)$ and the
   inverse-CDF rule, which class does each of $u = 0.05$, $0.65$, $0.71$ and $0.1$ give? (d) Find the
   Cholesky factor of $\boldsymbol{\Sigma} = \begin{pmatrix}1 & 0.6\\0.6 & 1\end{pmatrix}$ and transform $\mathbf{z} = (1, 1)$ into a sample with mean
   $\boldsymbol{\mu} = (0, 0)$. Check that $\mathbf{L}\mathbf{L}^\top = \boldsymbol{\Sigma}$.
2. **Code:** (a) Write `sample_poisson(rate, size, gen)` using only `gen.random()`, by inverse CDF:
   compute the pmf $P(0), P(1), \dots$ with the "times $\lambda/k$" recurrence up to a $k$ where the remaining tail
   is below $10^{-12}$, and reuse the categorical idea. (b) Write `sample_exponential(rate, size, gen)` as
   $-\log(u)/\text{rate}$ with $u$ in $(0, 1]$, and check its mean ($1/\text{rate}$) and variance ($1/\text{rate}^2$).
   (c) Simulate a process where the gaps between events are exponential with rate 4 and count the events
   that land in $[0, 1)$; repeat 100,000 times. Show that your count distribution, your Poisson sampler
   and `stats.poisson(4)` agree (largest pmf gap below $3/\sqrt{n}$). `SEED = 0`.
3. **Think:** For each of these, name a distribution, one assumption that is likely to fail in
   production, and how you would detect the failure from data: (i) the number of 5xx errors per minute
   on an API; (ii) whether a shown ad is clicked; (iii) the daily revenue of an online shop; (iv) which of
   four subscription plans a new user picks; (v) the measurement noise of a temperature sensor.

## Takeaways

1. A pmf gives probabilities of values; a pdf gives probability per unit, and probabilities are areas
   under it; the CDF $P(X \le x)$ exists for both.
2. Bernoulli $p$ / $p(1 - p)$; binomial $np$ / $np(1 - p)$; Poisson $\lambda$ / $\lambda$; uniform $\frac{a + b}{2}$ /
   $\frac{(b - a)^2}{12}$; Gaussian $\mu$ / $\sigma^2$ (mean / variance).
3. Every sampler is a transformation of uniform numbers: threshold (Bernoulli), inverse CDF
   (categorical), sum (binomial), radius and angle (Box–Muller), matrix (multivariate Gaussian).
4. $\mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\mathbf{z}$ with $\mathbf{L}\mathbf{L}^\top = \boldsymbol{\Sigma}$ has covariance $\boldsymbol{\Sigma}$; in batch form `mu + Z @ L.T`.
5. Check a sampler against the exact distribution's shape, not just its mean and variance, and check
   a model's distributional assumption against the data (variance versus mean for counts, tails for
   measurements).

**What builds on this:** [A5.3](a5_3_bayes_theorem.md) (populations simulated with Bernoulli draws);
[A5.4](a5_4_lln_clt_monte_carlo.md) (why sums become Gaussian); [A5.5](a5_5_mle_map.md) (fitting
these distributions to data, and the losses they imply); [B2.4](../B/b2_4_naive_bayes.md) (naive Bayes
with Bernoulli, categorical and Gaussian features); [B5.3](../B/b5_3_gmm_em.md) (mixtures of
multivariate Gaussians); [F2.4](../F/f2_4_decoding.md) (sampling the next token from a categorical);
[G1.4](../G/g1_4_diffusion_ddpm.md) (diffusion models add and remove Gaussian noise).
