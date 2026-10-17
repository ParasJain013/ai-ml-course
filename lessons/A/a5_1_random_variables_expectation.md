# A5.1 Random variables, expectation, variance and covariance

**Core** · Part A, week 5 (course week 05) · ~2 h · Lab:
[`labs/A/a5_1_random_variables_expectation.py`](../../labs/A/a5_1_random_variables_expectation.py)

## Why it matters

Every number a model learns from is a sample: one week of traffic, one batch of images, one split of
the data. Run the same experiment again and you get a different accuracy, a different average latency,
a different gradient. Probability is the language for "how much would this number change if I
repeated the experiment?", and its three workhorses are the **expectation** (the long-run average),
the **variance** (how far values typically land from it) and the **covariance** (whether two
quantities move together). Without them you cannot tell a real improvement from noise, you cannot
reason about mini-batch gradients ([A4.4](a4_4_gradient_descent.md)), and every loss function in this
course ([A5.5](a5_5_mle_map.md)) stays a formula you memorized rather than one you understand. This
lesson builds all three by simulation first, then by formula, and shows the two agree.

## Prerequisite check

- Can you compute a mean and a standard deviation of an array with NumPy, and say what `axis=0`
  does? If not, revisit [A1.2](a1_2_numpy_arrays.md).
- Can you draw random integers and uniform numbers with `rng = np.random.default_rng(0)`, and say
  why the course never uses `np.random.seed`? If not, revisit [A1.1](a1_1_lab_workflow.md).
- Can you compute a sample covariance by hand for four points and say why it divides by $n - 1$? If
  not, revisit [A3.3](a3_3_covariance_matrices.md).
- Can you read a scatter plot and a log-scale axis? If not, revisit [A1.4](a1_4_matplotlib.md).

## Intuition

### A random variable is a number produced by a random process

Roll a die. The outcome is a face; the **random variable** $X$ is the number written on it. Flip a
coin and let $X = 1$ for heads, 0 for tails. Pick a random user and let $X$ be their session length in
seconds. In each case there is a process that could come out several ways, and a rule that turns each
way into a number. The rule is fixed; which number you get is not.

Before you roll, $X$ has no value, but it has a **distribution**: a list of the values it can take and
how likely each is. For a fair die: 1 to 6, each with probability $\frac16$. Everything else in this
lesson is a summary of that list.

> **Bridge:** a random variable is like a function `() => number` that reads from a random source each
> time you call it, such as `Math.floor(Math.random() * 6) + 1`. The distribution is its contract: the
> values it may return and how often. **Where it breaks:** in probability, $X$ is one draw that is
> *about to happen*; you reason about the contract, not about any one call. And two random variables
> can be linked (the sum of two dice depends on the first die), which the function picture hides.

### Expectation: the long-run average

Roll the die a million times and average the results. You will get very close to 3.5, a value the die
can never show. That long-run average is the **expectation** $\mathbb{E}[X]$. You can compute it without
rolling: weight each value by its probability and add,
$1 \cdot \frac16 + 2 \cdot \frac16 + \dots + 6 \cdot \frac16 = 3.5$.

Simulation and formula are two routes to the same number. Simulation is approximate but works for any
process you can code; the formula is exact but needs the distribution written down. This course uses
each to check the other.

### Variance: how far from the average, typically

Two games both pay 3.5 on average: one always pays 3.5, the other is the die. The die is riskier: its
outcomes land 2.5 away from the average at worst. The **variance** measures that spread: the average
*squared* distance from the mean. Its square root, the **standard deviation**, is back in the units of
$X$ and is the "typical distance from the mean". (Squares are used because they are always positive
and, as you will see, they make sums of variables easy.)

### Covariance: do two variables move together?

Roll two dice; let $X$ be the first and $S$ the total. When the first die is high, the total tends to
be high too: the two **covary**. The **covariance** is positive when above-average values of one come
with above-average values of the other, negative when they come with below-average values, and zero
when there is no *linear* tendency either way. Two separate dice have covariance 0: knowing one tells
you nothing about the other. This is the same covariance as in [A3.3](a3_3_covariance_matrices.md);
there it was computed from data, here it is a property of the process that generated the data.

### The estimate itself is random

The sample mean of 10 rolls is a random variable too: repeat the 10 rolls and you get another number.
Its spread shrinks as you take more rolls, like $1/\sqrt{n}$. That fact (made precise in
[A5.4](a5_4_lln_clt_monte_carlo.md)) is why a metric measured on 100 users can wander far more than one
measured on 100,000, and it is the key plot of this lesson.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $X, Y, S$ | random variables (uppercase: the variable; lowercase $x$: one value it takes) | scalar |
| $p(x)$ | probability mass function: $P(X = x)$ for a discrete $X$; density for a continuous one | scalar |
| $p(x, y)$ | joint probability $P(X = x \text{ and } Y = y)$ | scalar |
| $\mathbb{E}[X]$, $\mu$ | expectation (mean) of $X$ | scalar |
| $\operatorname{Var}[X]$, $\sigma^2$ | variance; $\sigma$ is the standard deviation | scalar |
| $\operatorname{Cov}[X, Y]$ | covariance | scalar |
| $\rho$ | correlation, $\operatorname{Cov}[X,Y] / (\sigma_X\sigma_Y)$ | scalar in $[-1, 1]$ |
| $a, b$ | ordinary (non-random) numbers | scalar |
| $x_1, \dots, x_n$ | $n$ observed draws of $X$ (index from 1 in math, 0 in code) | `(n,)` |
| $\bar{x}$, $s^2$ | sample mean and sample variance of the draws | scalar |

**Expectation.** For a discrete random variable, weight each value by its probability:

$$\mathbb{E}[X] = \sum_x x\,p(x).$$

For a continuous one the sum becomes an integral, $\mathbb{E}[X] = \int x\,p(x)\,dx$, where $p(x)$ is a
density (probability per unit of $x$, A5.2). For any function $g$, the same weighting gives
$\mathbb{E}[g(X)] = \sum_x g(x)\,p(x)$: apply $g$ to each value, keep the probabilities.

**Linearity**, the most used rule in probability: for any numbers $a, b$ and any two random variables
(independent or not),

$$\mathbb{E}[aX + b] = a\,\mathbb{E}[X] + b, \qquad \mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y].$$

Why the first: $\sum_x (ax + b)p(x) = a\sum_x x\,p(x) + b\sum_x p(x)$, and probabilities add up to 1. The
second follows the same way from the joint table: $\sum_{x,y}(x + y)p(x, y)$ splits into two sums, and
summing $p(x, y)$ over $y$ gives $p(x)$. Linearity does **not** extend to products or other non-linear
functions: in general $\mathbb{E}[XY] \ne \mathbb{E}[X]\,\mathbb{E}[Y]$ and $\mathbb{E}[X^2] \ne \mathbb{E}[X]^2$.

**Variance.** With $\mu = \mathbb{E}[X]$,

$$\operatorname{Var}[X] = \mathbb{E}\big[(X - \mu)^2\big] = \mathbb{E}[X^2] - \mu^2.$$

The shortcut on the right comes from expanding the square and using linearity:
$\mathbb{E}[X^2 - 2\mu X + \mu^2] = \mathbb{E}[X^2] - 2\mu\,\mathbb{E}[X] + \mu^2 = \mathbb{E}[X^2] - 2\mu^2 + \mu^2$. Variance is in
squared units (seconds²); the standard deviation $\sigma = \sqrt{\operatorname{Var}[X]}$ is in seconds.

**Scaling and shifting.** $\operatorname{Var}[aX + b] = a^2\operatorname{Var}[X]$. Shifting by $b$ moves the mean by $b$ too, so the
distances $X - \mu$ do not change; scaling by $a$ scales every distance by $a$ and every squared distance by
$a^2$. A negative $a$ still increases the variance: $\operatorname{Var}[-2X] = 4\operatorname{Var}[X]$.

**Covariance.** With $\mu_X = \mathbb{E}[X]$, $\mu_Y = \mathbb{E}[Y]$:

$$\operatorname{Cov}[X, Y] = \mathbb{E}\big[(X - \mu_X)(Y - \mu_Y)\big] = \mathbb{E}[XY] - \mu_X\mu_Y.$$

(Same expansion as the variance; indeed $\operatorname{Cov}[X, X] = \operatorname{Var}[X]$.) Like expectation, covariance is
linear in each argument: $\operatorname{Cov}[X, Y + Z] = \operatorname{Cov}[X, Y] + \operatorname{Cov}[X, Z]$ and $\operatorname{Cov}[aX + b, Y] = a\operatorname{Cov}[X, Y]$
(substitute into the definition and use linearity of $\mathbb{E}$; the shift $b$ cancels against the mean). The product
$(X - \mu_X)(Y - \mu_Y)$ is positive when both are on the same side of their means and negative when they
are on opposite sides; the covariance is its average. Dividing by both standard deviations gives the
**correlation** $\rho = \operatorname{Cov}[X, Y] / (\sigma_X\sigma_Y)$, a unit-free number between $-1$ and $1$.

**Variance of a sum.** Expand $\big((X - \mu_X) + (Y - \mu_Y)\big)^2$ and take the expectation of each term:

$$\operatorname{Var}[X + Y] = \operatorname{Var}[X] + \operatorname{Var}[Y] + 2\operatorname{Cov}[X, Y], \qquad \operatorname{Var}[X - Y] = \operatorname{Var}[X] + \operatorname{Var}[Y] - 2\operatorname{Cov}[X, Y].$$

**Independence.** $X$ and $Y$ are **independent** when knowing one changes nothing about the other:
$p(x, y) = p(x)\,p(y)$ for every pair of values. Then $\mathbb{E}[XY] = \sum_{x,y} xy\,p(x)p(y) = \mathbb{E}[X]\,\mathbb{E}[Y]$, so
$\operatorname{Cov}[X, Y] = 0$ and **variances of independent variables add**. The converse is false: covariance 0
only says there is no *linear* trend. If $X$ is uniform on $\{-1, 0, 1\}$ and $Y = X^2$, then $Y$ is
completely determined by $X$, yet $\operatorname{Cov}[X, Y] = \mathbb{E}[X^3] - \mathbb{E}[X]\,\mathbb{E}[X^2] = 0 - 0 = 0$.

**From data: the estimators.** Given draws $x_1, \dots, x_n$, replace "weight by probability" with "average
over the draws":

$$\bar{x} = \frac1n\sum_{i=1}^n x_i, \qquad s^2 = \frac{1}{n - 1}\sum_{i=1}^n (x_i - \bar{x})^2, \qquad s_{xy} = \frac{1}{n - 1}\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}).$$

The $n - 1$ is **Bessel's correction** from A3.3: deviations from $\bar{x}$ (fitted to the same data) are
slightly smaller on average than deviations from the true $\mu$, and dividing by $n - 1$ makes $s^2$ right
on average (**unbiased**). For $n$ in the thousands the difference is invisible; for $n = 10$ it is 10%.

**How far the sample mean wanders.** $\bar{x}$ is itself a random variable. If the draws are independent,
each with variance $\sigma^2$, the rules above give

$$\mathbb{E}[\bar{x}] = \frac1n\sum_i\mathbb{E}[X_i] = \mu, \qquad \operatorname{Var}[\bar{x}] = \frac{1}{n^2}\operatorname{Var}\Big[\sum_i X_i\Big] = \frac{1}{n^2}\cdot n\sigma^2 = \frac{\sigma^2}{n}.$$

(The $\frac{1}{n^2}$ is the $a^2$ rule with $a = \frac1n$; the $n\sigma^2$ is "independent variances add".) So the
standard deviation of the sample mean, its **standard error**, is $\sigma/\sqrt{n}$. A5.4 builds on this.

## Worked example by hand

**1. One fair die.** Values 1 to 6, each with $p = \frac16$.

- $\mathbb{E}[X] = \frac{1 + 2 + 3 + 4 + 5 + 6}{6} = \frac{21}{6} = 3.5$.
- $\mathbb{E}[X^2] = \frac{1 + 4 + 9 + 16 + 25 + 36}{6} = \frac{91}{6} = 15.1667$.
- $\operatorname{Var}[X] = \frac{91}{6} - 3.5^2 = \frac{91}{6} - \frac{49}{4} = \frac{182 - 147}{12} = \frac{35}{12} = 2.9167$, so $\sigma = 1.7078$.
- The long way agrees: squared distances from 3.5 are $6.25, 2.25, 0.25, 0.25, 2.25, 6.25$, which add to
  17.5, and $17.5/6 = 35/12$ ✓.
- Rules: $\mathbb{E}[10 - 2X] = 10 - 7 = 3$ and $\operatorname{Var}[10 - 2X] = 4 \times \frac{35}{12} = 11.6667$.
- Two independent dice, total $S = X_1 + X_2$: $\mathbb{E}[S] = 7$, $\operatorname{Var}[S] = \frac{35}{12} + \frac{35}{12} = \frac{35}{6} = 5.8333$.
  $\operatorname{Cov}[X_1, S] = \operatorname{Cov}[X_1, X_1] + \operatorname{Cov}[X_1, X_2] = \frac{35}{12} + 0$ (covariance is linear in each argument, like
  expectation). Correlation $\frac{35/12}{\sqrt{35/12}\sqrt{35/6}} = \frac{1}{\sqrt2} = 0.7071$.

**2. A joint table.** $X$ and $Y$ each take values 0 or 1, with

| | $Y = 0$ | $Y = 1$ |
|---|---|---|
| $X = 0$ | 0.4 | 0.1 |
| $X = 1$ | 0.1 | 0.4 |

- Marginals (row and column sums): $P(X = 1) = 0.1 + 0.4 = 0.5$, $P(Y = 1) = 0.5$. So $\mathbb{E}[X] = \mathbb{E}[Y] = 0.5$.
- $XY$ is 1 only in the bottom-right cell: $\mathbb{E}[XY] = 0.4$. $\operatorname{Cov}[X, Y] = 0.4 - 0.5 \times 0.5 = 0.15$.
- $X^2 = X$ for 0/1 values, so $\operatorname{Var}[X] = 0.5 - 0.25 = 0.25$, the same for $Y$. Correlation $\frac{0.15}{0.5 \times 0.5} = 0.6$.
- Not independent: $P(X = 1, Y = 1) = 0.4$, but $P(X = 1)P(Y = 1) = 0.25$.

**3. Uncorrelated but dependent.** $X$ uniform on $\{-1, 0, 1\}$, $Y = X^2$. $\mathbb{E}[X] = 0$, $\mathbb{E}[X^3] = \frac{-1 + 0 + 1}{3} = 0$,
so the covariance is 0. Yet $P(Y = 1) = \frac23$ while $P(Y = 1 \mid X = 1) = 1$.

```text
== worked example: one fair die ==
E[X] = 7/2 = 3.5000   E[X^2] = 91/6 = 15.1667
Var[X] = E[X^2] - E[X]^2 = 35/12 = 2.9167   sd = 1.7078
Var[X] the long way, E[(X - 3.5)^2] = 35/12

== worked example: a joint table ==
marginals P(X) = [0.5, 0.5]  P(Y) = [0.5, 0.5]
E[X] = 0.50  E[Y] = 0.50  E[XY] = 0.40  Cov = 0.15
Var[X] = 0.25  Var[Y] = 0.25  correlation = 0.60
independent? P(X=1, Y=1) = 0.40  vs  P(X=1) P(Y=1) = 0.25

== worked example: uncorrelated but dependent ==
Cov[X, X^2] = E[X^3] - E[X] E[X^2] = 0.0000
P(Y = 1) = 0.6667   but P(Y = 1 | X = 1) = 1.0: Y is a function of X
```

The lab uses Python's `Fraction` for the die so the answers come out exactly as on paper.

## Key plot

![Scatter of 200 estimated die means at each of five sample sizes from 10 to 100,000 rolls on a log x-axis; at 10 rolls the estimates range from about 2.1 to 4.7, at 100,000 they sit on 3.5; a dashed funnel at 3.5 plus or minus 2 sigma over root n contains almost all of them](img/a5_1_estimates_wander.png)

Every dot is one complete experiment: roll a die $n$ times and average. At $n = 10$ the estimates
spread from about 2 to 5; at $n = 100{,}000$ they are indistinguishable from 3.5. The dashed funnel is
$3.5 \pm 2\sigma/\sqrt{n}$; it narrows by $\sqrt{10} \approx 3.2$ per tick, so 100 times more data buys 10 times less
spread.

## From scratch in NumPy

The three estimators, written straight from the formulas:

```python
def sample_mean(x: FloatArray) -> float:
    """Sample mean (1/n) sum x_i. x (n,) -> scalar."""
    return float(x.sum() / len(x))


def sample_var(x: FloatArray, ddof: int = 1) -> float:
    """Sample variance, dividing by n - ddof (ddof=1: Bessel's correction). x (n,) -> scalar."""
    r = x - sample_mean(x)  # (n,) deviations from the sample mean
    return float(r @ r / (len(x) - ddof))


def sample_cov(x: FloatArray, y: FloatArray, ddof: int = 1) -> float:
    """Sample covariance of two variables, dividing by n - ddof. x (n,), y (n,) -> scalar."""
    assert x.shape == y.shape, (x.shape, y.shape)
    return float((x - sample_mean(x)) @ (y - sample_mean(y)) / (len(x) - ddof))
```

`r @ r` is the sum of squared deviations as a dot product (A1.5). The lab simulates a million rolls of
two dice (`rng.integers(1, 7, size=n)`: the upper bound is excluded, as in `range`) and a million coin
flips (`rng.random(n) < 0.5`), and compares every estimate with the exact value:

```text
== from scratch: estimates from a million draws ==
quantity              exact   estimate     error
die: mean            3.5000     3.5022   +0.0022
die: variance        2.9167     2.9175   +0.0008
coin: mean           0.5000     0.4994   -0.0006
coin: variance       0.2500     0.2500   -0.0000
total: mean          7.0000     7.0025   +0.0025
total: variance      5.8333     5.8326   -0.0008
Cov[die1, total]     2.9167     2.9155   -0.0011
Cov[die1, die2]      0.0000    -0.0019   -0.0019
correlation of die1 and total: estimate 0.7068   exact 1/sqrt(2) = 0.7071
```

Every error is in the third decimal, about the size $\sigma/\sqrt{n} = 1.7078/1000 \approx 0.0017$ predicts for
the mean. Note `Cov[die1, die2]`: two independent dice give an estimate of $-0.0019$, not 0. A sample
covariance is never exactly zero; whether $-0.0019$ is "zero" is a question about noise, answered
properly in [A6.3a](a6_3a_hypothesis_tests.md).

The rules hold on the same draws:

```text
== from scratch: rules for scaling and sums ==
E[10 - 2X]: estimate 2.9955   rule 10 - 2 E[X] = 3.0000
Var[10 - 2X]: estimate 11.6700   rule (-2)^2 Var[X] = 11.6667
Var[die1 + die2] = 5.8326   Var[die1] + Var[die2] = 5.8365   (+ 2 Cov = -0.0039)
Var[die1 - total] = 2.9190   Var + Var - 2 Cov = 2.9190
```

The third line shows the sum rule at work on samples: the two variances alone overshoot by 0.0039,
and the small sample covariance accounts for exactly that. The last line uses a strongly correlated
pair ($\operatorname{die1} - \operatorname{total} = -\operatorname{die2}$): without the $-2\operatorname{Cov}$ term the answer would be $8.75$ instead of $2.92$.

Finally, how far estimates wander. The lab repeats the whole experiment 500 times at each sample size
(one call, `rng.integers(1, 7, size=(500, n))`, then `.mean(axis=1)`) and measures the spread of the
500 estimates:

```text
== from scratch: how far estimates wander ==
      n   sd of mean estimates  sigma/sqrt(n)   sd of variance estimates
     10                 0.5092         0.5401                     0.9262
    100                 0.1662         0.1708                     0.2442
   1000                 0.0550         0.0540                     0.0764
  10000                 0.0171         0.0171                     0.0262
 100000                 0.0055         0.0054                     0.0079
```

The measured spread of the mean tracks $\sigma/\sqrt{n}$ (500 repeats are themselves a sample, so the match is
within a few percent, not exact). The variance estimate wanders too, and at $n = 10$ by a lot: its
standard deviation (0.93) is a third of the value it estimates (2.92). Small samples give unreliable
spreads as well as unreliable means.

## With NumPy and SciPy

```text
== with numpy and scipy ==
np.mean = 3.5022   np.var(ddof=1) = 2.9175   np.var (default ddof=0) = 2.9175
np.cov of (die1, die2, total) = [[2.9175, -0.0019, 2.9155], [-0.0019, 2.919, 2.917], [2.9155, 2.917, 5.8326]]
matches numpy: OK
scipy.stats.randint(1, 7): mean = 3.5000   variance = 2.9167
rv_discrete over {0, 2, 10} with p = (0.6, 0.3, 0.1): mean = 1.6000   variance = 8.6400
scipy's exact moments match the hand calculation: OK
```

- `np.var` divides by $n$ by default (`ddof=0`); pandas' `.var()` divides by $n - 1$ (`ddof=1`). With a
  million rows they agree to 4 decimals; with 10 rows they differ by 10%. Always pass `ddof`
  explicitly.
- `np.cov` treats **rows as variables** by default, the opposite of the course's rows-are-examples
  rule. The lab stacks three `(n,)` arrays into a `(3, n)` array on purpose. For an `(n, d)` data
  matrix, call `np.cov(X, rowvar=False)` (A3.3). The result is the `(3, 3)` covariance matrix: variances
  on the diagonal, covariances off it.
- `scipy.stats` gives **exact** moments of named distributions (`stats.randint(1, 7).stats(moments="mv")`),
  and `stats.rv_discrete(values=(xk, pk))` builds a distribution from any finite table. Use them to check
  hand calculations; A5.2 meets the named distributions properly.

## In practice

- **When to use it:** whenever you report a number computed from a sample (a metric, a latency, a
  conversion rate), also report how much it would vary: the standard deviation of the data and the
  standard error of the mean, $s/\sqrt{n}$. When you add up quantities (costs, latencies of sequential
  calls, errors), add variances only if the parts are independent; otherwise include the covariances.
- **Failure modes:**
  - *Treating a small-sample number as exact.* An A/B test with 100 users per arm can show a 5% "lift"
    that is pure noise. You notice by computing $s/\sqrt{n}$ and seeing it is as large as the difference.
  - *`ddof` mismatch.* NumPy and pandas disagree by default. You notice tiny, unexplained differences
    between two pipelines that should match; on small groups (`groupby` with 5 rows) the gap is large.
  - *Reading zero correlation as independence.* A relationship shaped like a U (latency high at both low
    and high load) has correlation near 0. You notice by plotting the scatter, not by trusting $\rho$.
  - *Adding variances of correlated parts.* The variance of the total cost of correlated services is
    larger than the sum of their variances, so capacity planned on the sum is too small. You notice when
    the total's observed spread exceeds the prediction.
  - *Expectations that do not exist.* For very heavy-tailed data (a few enormous values), the sample
    mean never settles; A5.4 shows the Cauchy distribution doing this. You notice a running mean that
    jumps every time a new extreme value arrives.
- **Defaults that work:** `rng = np.random.default_rng(SEED)` and pass `rng` around; `ddof=1` for
  sample variances and covariances unless you know why not; `np.cov(X, rowvar=False)` for an `(n, d)`
  matrix; check every simulated estimate against an exact value when one exists (VERIFY (as of 2026-10):
  `np.var` default `ddof=0` and `np.cov` default `rowvar=True`, NumPy 2.5.4).

## Exercises

Save your work to `my-work/A/a5_1_random_variables_expectation.md` (or `.py`) and ask for a review.

1. **By hand:** (a) A spinner shows $X = 0$ with probability 0.5, $X = 1$ with 0.3 and $X = 4$ with 0.2.
   Compute $\mathbb{E}[X]$, $\mathbb{E}[X^2]$, $\operatorname{Var}[X]$ and $\sigma$. (b) Let $Y = 3 - 2X$. Give $\mathbb{E}[Y]$ and $\operatorname{Var}[Y]$ using the
   rules, then check $\mathbb{E}[Y]$ directly from the table. (c) Two such spinners are spun independently,
   giving $X_1$ and $X_2$. Compute $\operatorname{Var}[X_1 + X_2]$ and $\operatorname{Cov}[X_1, X_1 + X_2]$. (d) For the joint table
   $P(X=0,Y=0) = 0.3$, $P(X=0,Y=1) = 0.2$, $P(X=1,Y=0) = 0.1$, $P(X=1,Y=1) = 0.4$, compute $\operatorname{Cov}[X, Y]$ and the
   correlation, and say whether $X$ and $Y$ are independent.
2. **Code:** Roll two fair dice; let $M$ be the larger and $L$ the smaller value. (a) Compute the exact
   $\mathbb{E}[M]$, $\mathbb{E}[L]$, $\operatorname{Var}[M]$ and $\operatorname{Cov}[M, L]$ by enumerating all 36 equally likely outcomes, as fractions.
   (b) Estimate the same four quantities by simulation with $n = 100$, $10{,}000$ and $1{,}000{,}000$ pairs
   (`SEED = 0`, draw both dice at once with `rng.integers(1, 7, size=(n, 2))`), with your own estimator
   functions, and check them against `np.cov`. (c) Explain why $\operatorname{Cov}[M, L]$ is positive, and find a shortcut
   for $\mathbb{E}[ML]$ that needs no enumeration.
3. **Think:** A teammate computes the correlation between per-minute p99 latency and per-minute error
   rate over one week of monitoring data, finds $\rho = 0.02$, and proposes dropping the error-rate alert
   because "it is independent of latency, which we already alert on". Give at least two ways this
   conclusion could be wrong, what you would look at to find out, and what you would decide.

## Takeaways

1. A random variable is a number produced by a random process; its distribution lists the values and
   their probabilities, and everything else summarizes that list.
2. $\mathbb{E}[X] = \sum x\,p(x)$ is linear: $\mathbb{E}[aX + b] = a\mathbb{E}[X] + b$ and $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$ always, but
   $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ only without covariance.
3. $\operatorname{Var}[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$; $\operatorname{Var}[aX + b] = a^2\operatorname{Var}[X]$; $\operatorname{Var}[X + Y] = \operatorname{Var}[X] + \operatorname{Var}[Y] + 2\operatorname{Cov}[X, Y]$.
4. Independence implies zero covariance, not the other way round: covariance only sees linear trends.
5. Estimates from data are random: the sample mean has standard error $\sigma/\sqrt{n}$, so 100 times more data
   gives 10 times less wander. Check simulations against exact values, and pass `ddof` explicitly.

**What builds on this:** [A5.2](a5_2_common_distributions.md) (the named distributions and their
means and variances); [A5.4](a5_4_lln_clt_monte_carlo.md) (why the sample mean converges and how it is
distributed); [A6.2](a6_2_confidence_intervals_bootstrap.md) (error bars on any estimate);
[A6.3a](a6_3a_hypothesis_tests.md) (is a difference real or noise?); [A4.4](a4_4_gradient_descent.md)
revisited (a mini-batch gradient is a sample mean, so its noise shrinks like $1/\sqrt{B}$).
