# A5.4 Law of large numbers, central limit theorem and Monte Carlo estimation

**Core** · Part A, week 5 (course week 05) · ~1.5 h · Lab:
[`labs/A/a5_4_lln_clt_monte_carlo.py`](../../labs/A/a5_4_lln_clt_monte_carlo.py)

## Why it matters

Almost every number you report in ML is an average over a sample: accuracy over a test set, loss
over a batch, conversion rate over an A/B arm, latency over a load test. Two theorems tell you how
far to trust such an average. The **law of large numbers** says it converges to the true value as the
sample grows. The **central limit theorem** says how it is distributed on the way: approximately
Gaussian, with a spread that shrinks like $1/\sqrt{n}$, which gives every average an error bar. Together
they also justify **Monte Carlo estimation**: when a quantity has no formula (the expected reward of a
policy in [H2.1](../H/h2_1_policy_gradients.md), a dropout model's predictive uncertainty, the value of
an integral in many dimensions), simulate and average, and report the error bar. Without this you
cannot say whether "accuracy rose from 91.2% to 91.6%" is real ([A6.3a](a6_3a_hypothesis_tests.md)).

## Prerequisite check

- Can you derive $\operatorname{Var}[\bar{x}] = \sigma^2/n$ for $n$ independent draws, from the rules for sums and scaling? If not,
  revisit [A5.1](a5_1_random_variables_expectation.md).
- Do you know the Gaussian's 68–95–99.7 rule and how to standardize, $z = (x - \mu)/\sigma$? If not, revisit
  [A5.2](a5_2_common_distributions.md).
- Can you compute a running total with `np.cumsum` and take a mean along an axis? If not, revisit
  [A1.2](a1_2_numpy_arrays.md).
- Can you read a log–log plot, where a straight line means a power law? If not, revisit
  [A1.4](a1_4_matplotlib.md).

## Intuition

### Averages settle down

Roll a die and keep a running average. After 10 rolls it might be 4.8; after 100, 3.3; after 100,000,
3.50 to two decimals. Each new roll moves the average by $\frac1n$ of its distance from the average, so
as $n$ grows each roll matters less, and the ups and downs cancel more and more completely. That is
the **law of large numbers (LLN)**: the sample mean converges to the expectation. It is why a casino
with millions of bets is a business and a gambler with ten is not.

### …at a predictable speed

A5.1 showed the sample mean has standard deviation $\sigma/\sqrt{n}$, its **standard error**. The square
root is the important part, and it is not a good deal: to halve the error you need four times the data;
for one more decimal digit, a hundred times the data.

### …and with a predictable shape

The **central limit theorem (CLT)** adds something surprising: whatever the shape of one draw's
distribution (lopsided, discrete, bimodal), the *average* of many independent draws has a
distribution close to a Gaussian. Two dice already give a triangle (totals 2 to 12 with counts
1, 2, …, 6, …, 2, 1); ten dice look like a bell. The intuition: an average is a sum of many small
independent pushes, extreme sums need many pushes in the same direction at once, and that is rare in
a very specific way that always produces the bell curve.

So the average is approximately $\mathcal{N}(\mu, \sigma^2/n)$, and the 68–95–99.7 rule gives error bars: the
average lands within $1.96\,\sigma/\sqrt{n}$ of $\mu$ about 95% of the time.

### Monte Carlo: compute by simulating

Want the area of an awkward shape? Throw random darts at a square that contains it and count the
fraction that land inside; area ≈ fraction × square's area. Want an expected value with no formula?
Simulate the process many times and average. Every such estimate is a sample mean, so the LLN says
it converges and the CLT says how big its error bar is. The error shrinks like $1/\sqrt{n}$ *regardless of
the number of dimensions*, which is why Monte Carlo beats grid-based methods for integrals over many
variables: a grid with 10 points per axis needs $10^{20}$ points in 20 dimensions.

> **Bridge:** a load test is Monte Carlo. You cannot compute the mean latency of your API from its
> code, so you send 10,000 requests and average, and the CLT gives the error bar on that average.
> **Where it breaks:** both theorems need independent draws from one fixed distribution. Requests that
> share a warming cache, a connection pool or a garbage-collection pause are correlated, so the true
> uncertainty is larger than $s/\sqrt{n}$; and a p99 is not a mean, so the CLT error bar does not apply
> to it directly ([A6.2](a6_2_confidence_intervals_bootstrap.md) handles percentiles with the bootstrap).

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $X_1, \dots, X_n$ | independent draws, each with the same distribution (**i.i.d.**: independent and identically distributed) | scalars |
| $\mu$, $\sigma^2$ | mean and variance of one draw | scalars |
| $\bar{X}_n$ | sample mean $\frac1n\sum_{i=1}^n X_i$ (a random variable before you draw; $\bar{x}$ once drawn) | scalar |
| $s$ | sample standard deviation (with $n - 1$) | scalar |
| $\mathrm{SE}$ | standard error of the mean, $\sigma/\sqrt{n}$, estimated by $s/\sqrt{n}$ | scalar |
| $Z$ | a standard normal random variable | scalar |
| $\Phi(z)$ | the standard normal CDF, $P(Z \le z)$ | scalar |
| $g$ | any function whose expectation we want | — |
| $f$, $[a, b]$ | a function to integrate and its interval | — |
| $U$ | uniform random variable on $[0, 1]$ (or $[a, b]$) | scalar |

**Mean and spread of the sample mean** (from A5.1): $\mathbb{E}[\bar{X}_n] = \mu$ and $\operatorname{Var}[\bar{X}_n] = \sigma^2/n$, so
$\mathrm{SE} = \sigma/\sqrt{n}$.

**Law of large numbers.** If the $X_i$ are i.i.d. with a finite mean $\mu$, then $\bar{X}_n \to \mu$ as $n \to \infty$.
Why, when the variance is also finite: the spread of $\bar{X}_n$ around $\mu$ is $\sigma/\sqrt{n}$, which goes to 0.
(Stated without proof: **Chebyshev's inequality**, $P(\lvert\bar{X}_n - \mu\rvert \ge \varepsilon) \le \frac{\sigma^2}{n\varepsilon^2}$, turns
this into "the probability of missing by more than any fixed $\varepsilon$ goes to 0".) The LLN says nothing
about *how fast* for a given $n$; that is the CLT's job.

**When it fails.** The **Cauchy distribution** (the ratio of two independent standard normals) has
tails so heavy that $\mathbb{E}[X]$ does not exist: the integral $\int x\,p(x)\,dx$ does not converge. Its running
mean never settles; an occasional enormous draw resets it. Real data with very heavy tails (file
sizes, wealth, some latencies) behaves in between: the mean exists, but convergence is slow and
jumpy.

**Central limit theorem.** If the $X_i$ are i.i.d. with mean $\mu$ and finite variance $\sigma^2$, then for
large $n$

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \approx \mathcal{N}(0, 1), \quad\text{equivalently}\quad \bar{X}_n \approx \mathcal{N}\Big(\mu, \frac{\sigma^2}{n}\Big), \quad \sum_{i=1}^n X_i \approx \mathcal{N}(n\mu, n\sigma^2).$$

The left-hand side is the sample mean standardized with its own mean and standard error (A5.2). The
precise statement is that the CDF of the left side converges to $\Phi$ (stated, not proved). How large
"large" must be depends on the shape of one draw. A measure of lopsidedness is the **skewness**,
$\gamma = \mathbb{E}\big[((X - \mu)/\sigma)^3\big]$: 0 for symmetric distributions, 2 for the exponential. The skewness of a
mean of $n$ i.i.d. draws is $\gamma/\sqrt{n}$ (stated; the lab checks it), so a skewed variable needs a larger $n$
before the bell shape is accurate, especially in the tails. A common rule of "$n \ge 30$" is fine for
mild skew and wrong for heavy skew.

**A 95% error bar.** $P(-1.96 \le Z \le 1.96) = 0.95$. If $\bar{X}_n$ is approximately $\mathcal{N}(\mu, \mathrm{SE}^2)$, then
$P(\lvert\bar{X}_n - \mu\rvert \le 1.96\,\mathrm{SE}) \approx 0.95$, so the interval

$$\bar{x} \pm 1.96\,\frac{s}{\sqrt{n}}$$

contains $\mu$ in about 95% of repeated experiments. (Replacing $\sigma$ by $s$ is fine for large $n$; small $n$
needs the $t$ distribution, A6.2.) The "95%" is about the procedure over many repetitions, not about
one particular interval; A6.2 returns to this.

**Planning a sample size.** To get a 95% error bar of half-width $E$, solve $1.96\,\sigma/\sqrt{n} = E$:
$n = (1.96\,\sigma/E)^2$. For a proportion $p$, $\sigma = \sqrt{p(1 - p)} \le 0.5$.

**Monte Carlo estimation.** To estimate $\mathbb{E}[g(X)]$, draw $x_1, \dots, x_n$ and average:

$$\widehat{\mathbb{E}[g(X)]} = \frac1n\sum_{i=1}^n g(x_i), \qquad \mathrm{SE} = \frac{s_g}{\sqrt{n}},$$

where $s_g$ is the sample standard deviation of the values $g(x_i)$. It is a sample mean, so both theorems
apply directly.

**Integrals are expectations.** For $U$ uniform on $[a, b]$ (density $\frac{1}{b - a}$, A5.2),
$\mathbb{E}[f(U)] = \int_a^b f(x)\frac{1}{b - a}\,dx$, so

$$\int_a^b f(x)\,dx = (b - a)\,\mathbb{E}[f(U)] \approx (b - a)\frac1n\sum_{i=1}^n f(u_i).$$

**$\pi$ by darts.** Draw points uniformly in the unit square $[0, 1]^2$. The quarter disc $x^2 + y^2 < 1$ has area
$\pi/4$, and the square has area 1, so a point lands inside with probability $p = \pi/4$. Each point is a
Bernoulli($p$) draw; the fraction inside, $\hat{p}$, estimates $p$ with standard error $\sqrt{p(1 - p)/n}$ (A5.2), so
$\hat{\pi} = 4\hat{p}$ with standard error $4\sqrt{\hat{p}(1 - \hat{p})/n}$.

## Worked example by hand

**1. Standard errors of a die's average.** $\sigma = \sqrt{35/12} = 1.7078$. Average of 4 rolls: $\mathrm{SE} = 1.7078/2 = 0.854$.
Of 100 rolls: $0.1708$. So "the average of 100 rolls is within about $2 \times 0.17 = 0.34$ of 3.5" 95% of the time.

**2. The CLT for a sum.** The sum of 100 dice has mean $100 \times 3.5 = 350$ and variance $100 \times \frac{35}{12} = 291.67$
(independent variances add), standard deviation $17.08$. $P(\text{sum} > 380) \approx P\big(Z > \frac{380 - 350}{17.08}\big) =
P(Z > 1.757) = 1 - \Phi(1.757) \approx 0.040$. Nobody could enumerate the $6^{100}$ outcomes; the CLT makes it
a one-line calculation.

**3. $\pi$ from 10 darts.** 8 of 10 land inside: $\hat{p} = 0.8$, $\hat{\pi} = 3.2$, $\mathrm{SE} = 4\sqrt{0.8 \times 0.2/10} = 4 \times 0.1265 = 0.506$.
So $3.2 \pm 1.0$: correct, and useless. To get $\pm 0.001$ you need about $(1.96 \times 4 \times 0.41/0.001)^2 \approx 10^7$ darts
($\sqrt{p(1 - p)} = \sqrt{0.785 \times 0.215} \approx 0.41$).

**4. An integral from four draws.** $\int_0^1 e^x\,dx = e - 1 = 1.7183$. With $u = 0.1, 0.4, 0.7, 0.9$:
$e^u = 1.1052, 1.4918, 2.0138, 2.4596$, average $1.7676$. Here $b - a = 1$. The sample standard deviation of
the four values is $0.5928$, so $\mathrm{SE} = 0.5928/\sqrt4 = 0.296$; the estimate is off by $0.049$, well inside one
standard error.

```text
== worked example by hand ==
die: standard error of the mean of   1 rolls = sigma / sqrt(n) = 1.7078
die: standard error of the mean of   4 rolls = sigma / sqrt(n) = 0.8539
die: standard error of the mean of 100 rolls = sigma / sqrt(n) = 0.1708
pmf of the sum of two dice, totals 2..12, in 36ths: [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
sum of 100 dice: mean = 350   sd = sqrt(100 * 35/12) = 17.0783   P(sum > 380) ~ 1 - Phi(1.7566) = 0.0395
pi from 10 points, 8 inside: estimate = 4 * 0.8 = 3.2   standard error = 4 * sqrt(p(1 - p)/n) = 0.5060
integral of e^x on [0, 1] from u = [0.1, 0.4, 0.7, 0.9]: e^u = [1.1052, 1.4918, 2.0138, 2.4596]   estimate = 1.7676   exact e - 1 = 1.7183
  sample sd of e^u = 0.5928   standard error = 0.2964
```

## Key plot

![Three panels. Left: five running means of die rolls on a log x-axis, wild for the first ten rolls and all settling on 3.5 inside a dashed funnel of plus or minus two sigma over root n. Middle: histograms of standardized means of 1, 5 and 30 exponential draws against the standard normal curve; the single draw is a lopsided spike, 5 is close, 30 nearly matches. Right: on log-log axes, Monte Carlo errors for pi and for the integral of e to the x, scattered around two straight lines of slope minus one half](img/a5_4_lln_clt_mc.png)

Left: the LLN, with the $\pm 2\sigma/\sqrt{n}$ funnel from A5.1. Middle: the CLT; one exponential draw is nothing
like a bell, an average of 30 nearly is. Right: Monte Carlo errors (markers) bounce around the standard
error (lines), and both fall along a line of slope $-\frac12$ on log–log axes: $1/\sqrt{n}$.

## From scratch in NumPy

**Law of large numbers.** A running mean is a cumulative sum divided by the count:

```python
def running_mean(x: FloatArray) -> FloatArray:
    """Mean of the first k values for every k. x (n,) -> (n,)."""
    return np.cumsum(x) / np.arange(1, len(x) + 1)
```

```text
== from scratch: law of large numbers ==
after      10 rolls: largest |running mean - 3.5| over 5 paths = 1.3000   2 sigma/sqrt(n) = 1.0801
after     100 rolls: largest |running mean - 3.5| over 5 paths = 0.2000   2 sigma/sqrt(n) = 0.3416
after   1,000 rolls: largest |running mean - 3.5| over 5 paths = 0.1220   2 sigma/sqrt(n) = 0.1080
after  10,000 rolls: largest |running mean - 3.5| over 5 paths = 0.0303   2 sigma/sqrt(n) = 0.0342
after 100,000 rolls: largest |running mean - 3.5| over 5 paths = 0.0099   2 sigma/sqrt(n) = 0.0108
Cauchy running mean at the same checkpoints: [-0.3344, -0.3506, 0.0157, 8.9728, 1.9988]
```

The worst of five paths sits near $2\sigma/\sqrt{n}$, occasionally a little outside (at $n = 10$ and $1{,}000$), as
expected: about 1 path in 20 is outside $\pm 2\,\mathrm{SE}$ at any moment. The Cauchy running mean is at 0.02
after 1,000 draws and at 9.0 after 10,000: one huge draw dragged it away, and nothing pulls it back.

**Central limit theorem.** Draw an `(N_CLT, n)` array of exponential values (mean 1, standard deviation 1,
skewness 2), average each row, and standardize with the exact mean 1 and standard error $1/\sqrt{n}$:

```python
means = rng.exponential(1.0, size=(N_CLT, n)).mean(axis=1)  # (N_CLT,) one mean per row
z = (means - 1.0) / (1.0 / np.sqrt(n))  # standardize with the exact mean 1 and sd 1/sqrt(n)
```

```text
== from scratch: central limit theorem ==
   n  mean of means  sd of means  1/sqrt(n)  skewness  2/sqrt(n)  gap to N(0,1)
   1         0.9981       0.9929     1.0000    1.9811     2.0000         0.1587
   2         0.9973       0.7052     0.7071    1.3957     1.4142         0.0960
   5         1.0014       0.4469     0.4472    0.8874     0.8944         0.0591
  30         0.9993       0.1813     0.1826    0.3682     0.3651         0.0253
 100         1.0000       0.1001     0.1000    0.2005     0.2000         0.0138
P(standardized mean > 1.96), normal says 0.0250: {1: 0.0511, 2: 0.048, 5: 0.0439, 30: 0.0322, 100: 0.029}
```

Every column matches the theory: the mean of the means stays at $\mu = 1$ (unbiased), their spread is
$\sigma/\sqrt{n}$, their skewness is $2/\sqrt{n}$, and the largest gap between their CDF and $\Phi$ (the KS statistic from A5.2)
shrinks. The last line is the practical warning: with $n = 30$ a "2.5% tail" actually holds 3.2% of the
means, because the skew is not gone. The CLT is accurate in the middle well before it is accurate in
the tails.

**Monte Carlo with error bars.** Both estimators return the estimate and its standard error:

```python
def mc_pi(n: int, gen: np.random.Generator) -> tuple[float, float]:
    """Estimate pi from n points in the unit square. -> (estimate, standard error)."""
    xy = gen.random((n, 2))  # (n, 2)
    hit = (xy**2).sum(axis=1) < 1.0  # (n,) inside the quarter circle of radius 1
    p = hit.mean()
    return 4 * p, 4 * np.sqrt(p * (1 - p) / n)


def mc_integral(f, n: int, gen: np.random.Generator) -> tuple[float, float]:
    """Estimate the integral of f on [0, 1] as the mean of f(U). -> (estimate, standard error)."""
    fx = f(gen.random(n))  # (n,)
    return float(fx.mean()), float(fx.std(ddof=1) / np.sqrt(n))
```

```text
== from scratch: Monte Carlo with error bars ==
        n  pi estimate  +-1.96 SE   |error|   integral  +-1.96 SE   |error|
      100      3.48000    0.26366   0.33841    1.70989    0.08885   0.00839
    1,000      3.15600    0.10116   0.01441    1.71879    0.03087   0.00051
   10,000      3.15920    0.03194   0.01761    1.72147    0.00967   0.00319
  100,000      3.13648    0.01020   0.00511    1.72195    0.00306   0.00367
1,000,000      3.14107    0.00322   0.00052    1.71806    0.00096   0.00022
standard error at n = 100 divided by standard error at n = 1,000,000: pi 81.9, integral 92.1   (1/sqrt(n) predicts 100)
```

Three things to notice. The error bars shrink by about $\sqrt{10} = 3.2$ per row. Most errors are inside their
bars but not all: $\pi$ at $n = 100$ (off by 0.34, bar 0.26) and the integral at $n = 100{,}000$ (off by 0.0037, bar
0.0031) are outside, about the 1 in 20 the "95%" allows. And the ratio of standard errors is 82 and 92,
not exactly 100, because each standard error is itself *estimated* from the sample: at $n = 100$ the $\pi$
estimate 3.48 gives $\hat{p} = 0.87$ and a too-small $\sqrt{\hat{p}(1 - \hat{p})}$. Error bars are noisy for small $n$ too.

Finally, the honest test of an error bar: repeat the experiment and count.

```text
== from scratch: do 95% error bars cover the truth 95% of the time? ==
2000 runs of n = 1000: the interval estimate +- 1.96 SE contained e - 1 in 0.957 of runs
```

## With SciPy

```text
== with scipy ==
scipy.integrate.quad: 1.7182818285   (its own error estimate 1.9e-14)   exact e - 1 = 1.7182818285
quad matches the exact value: OK
Monte Carlo with 10^6 draws: 1.71861 +- 0.00096, agrees with quad within its error bar: OK
stats.norm.ppf(0.975) = 1.9600   (the 1.96 above)
n =  5: exact P(mean > 1 + 1.96/sqrt(n)) = 0.0433   CLT says 0.0250   simulated 0.0439
n = 30: exact P(mean > 1 + 1.96/sqrt(n)) = 0.0340   CLT says 0.0250   simulated 0.0322
```

- `scipy.integrate.quad` integrates a smooth 1-D function to 14 digits with a few dozen evaluations,
  where Monte Carlo needs a million draws for 3 digits. For smooth low-dimensional integrals, use
  quadrature; Monte Carlo wins when the dimension is high, the function is only available as a
  simulation, or the "integral" is an expectation over a complicated process.
- `stats.norm.ppf(0.975)` is the inverse CDF (A5.2) and gives the 1.96. `stats.norm.cdf` and `.sf`
  (survival function, $1 - $ CDF, more accurate in the far tail) give $\Phi(z)$ and $1 - \Phi(z)$.
- The mean of $n$ exponential draws happens to have an exact named distribution (a **gamma**
  distribution, `stats.gamma(a=n, scale=1/n)`), so the lab can check the CLT's tail approximation against
  the truth: at $n = 30$ the true upper tail is 3.4%, not 2.5%. The simulation agrees with the exact value,
  not with the CLT. VERIFY (as of 2026-10), SciPy 1.18.1: `stats.skew` divides by $n$ (biased) by default.

## In practice

- **When to use it:** put an error bar of $\pm 1.96\,s/\sqrt{n}$ on every average you report (metric, loss,
  latency, conversion rate), and use $n = (1.96\,\sigma/E)^2$ to size an experiment before running it. Use Monte
  Carlo when the quantity is an expectation you can simulate but not compute.
- **Failure modes:**
  - *Correlated samples.* Per-request latencies within one minute, frames of one video, rows from one
    user: the effective $n$ is smaller than the row count, so $s/\sqrt{n}$ is too small. You notice that
    repeated experiments disagree by more than their error bars. Average per independent unit (per user,
    per day) first, or use a block bootstrap (A6.2).
  - *Heavy tails.* A few huge values dominate the mean and the CLT needs a very large $n$. You notice a
    running mean that jumps when an outlier arrives, and an error bar that changes a lot between runs.
    Report medians or percentiles with bootstrap intervals, or model the log.
  - *Small $n$ with skew.* With 20 values from a skewed distribution, the $\pm 1.96\,\mathrm{SE}$ interval covers less
    than 95% (the tail table above). You notice it only by simulating coverage, so simulate it.
  - *Proportions near 0 or 1.* With 3 errors in 10,000 requests, $\sqrt{\hat{p}(1 - \hat{p})/n}$ is a poor guide and the
    interval can dip below 0. Use an exact or Wilson interval (A6.2).
  - *Forgetting $(b - a)$* in a Monte Carlo integral over $[a, b]$, or sampling from the wrong range. You
    notice by testing the estimator on a function with a known integral first.
- **Defaults that work:** report "estimate ± 1.96 SE (n = …)"; check any new Monte Carlo estimator on a
  case with a known answer; check coverage by repeating the whole experiment a few hundred times;
  increase $n$ by 100× when you need one more digit, or find a smarter estimator (variance reduction,
  "try this" 3).

## Exercises

Save your work to `my-work/A/a5_4_lln_clt_monte_carlo.md` (or `.py`) and ask for a review.

1. **By hand:** (a) A button's click probability is 0.3. What is the standard error of the observed click
   rate after 100 and after 10,000 impressions? How many impressions do you need for a 95% error bar of
   $\pm 0.01$? (b) A server handles 400 independent requests; each costs on average 20 ms of CPU with a
   standard deviation of 15 ms. Using the CLT, estimate the probability that the total CPU time exceeds
   8.6 seconds ($\Phi(2) = 0.9772$). Which assumption would you doubt first in production? (c) Estimate
   $\int_0^2 x^2\,dx$ by Monte Carlo by hand using the five uniform draws $u = 0.1, 0.3, 0.5, 0.7, 0.9$ on $[0, 1]$
   (transform them to $[0, 2]$ first), and compare with the exact value.
2. **Code:** Let $M$ be the largest of 10 independent standard normal draws. (a) Estimate $\mathbb{E}[M]$ by Monte
   Carlo with $n = 10^3$, $10^4$, $10^5$ and $10^6$ repetitions (`SEED = 0`, one `(n, 10)` draw per size), each with a
   95% error bar. (b) Compute $\mathbb{E}[M]$ accurately with `scipy.integrate.quad`, using the fact that $M$ has
   density $10\,\varphi(x)\,\Phi(x)^9$ ($\varphi$ is the standard normal density), and check that your error bars contain it.
   (c) Repeat the $n = 10^3$ estimate 2,000 times and report the coverage of your 95% interval. (d) How many
   repetitions would you need for 3 correct decimals, and how long would that take on your laptop?
3. **Think:** A load test sends 200 requests to a new endpoint and reports "mean latency 180 ms ± 12 ms
   (95%)", computed as $\bar{x} \pm 1.96\,s/\sqrt{n}$. Latencies are heavy-tailed: a few requests time out at 30
   seconds. List what could make this error bar misleading, what you would report instead or in
   addition, and how you would change the test.

## Takeaways

1. LLN: the sample mean of i.i.d. draws converges to the expectation, provided the expectation exists
   (Cauchy breaks it; heavy tails slow it).
2. The standard error of a mean is $\sigma/\sqrt{n}$: 4× the data halves the error, 100× buys one digit.
3. CLT: the mean of many i.i.d. draws with finite variance is approximately $\mathcal{N}(\mu, \sigma^2/n)$, whatever one draw
   looks like; skewed data needs larger $n$, especially in the tails.
4. $\bar{x} \pm 1.96\,s/\sqrt{n}$ covers the true mean in about 95% of repeated experiments, if the draws are
   independent; check coverage by simulation.
5. Monte Carlo estimates any expectation or integral as a sample mean, with error $\propto 1/\sqrt{n}$ in any
   dimension; integrals are $(b - a)\,\mathbb{E}[f(U)]$.

**What builds on this:** [A6.2](a6_2_confidence_intervals_bootstrap.md) (confidence intervals and the
bootstrap); [A6.3a](a6_3a_hypothesis_tests.md) (is a difference bigger than its noise?);
[A5.5](a5_5_mle_map.md) (estimators and their spread); [C3.1](../C/c3_1_optimizers.md) (mini-batch
gradients are Monte Carlo estimates of the full gradient); [H2.1](../H/h2_1_policy_gradients.md) (policy
gradients estimate expected reward by sampling episodes).
