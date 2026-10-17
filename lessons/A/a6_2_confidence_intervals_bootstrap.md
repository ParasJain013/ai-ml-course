# A6.2 Confidence intervals and the bootstrap

**Core** · Part A, week 6 (course week 06) · ~2 h · Lab:
[`labs/A/a6_2_confidence_intervals_bootstrap.py`](../../labs/A/a6_2_confidence_intervals_bootstrap.py)

## Why it matters

"Model B's accuracy is 91.6%, model A's is 91.2%" is not a result until you know how much those
numbers would move with a different test set. A **confidence interval** is the standard way to say it:
"91.6% (95% CI 90.9% to 92.3%)". A5.4 gave you one interval, $\bar{x} \pm 1.96\,s/\sqrt{n}$, which needs a large
sample and only works for means. Real questions are about medians, p99 latencies, differences between
two groups, AUCs and correlations, often with small or skewed samples. The **bootstrap** gives an
interval for almost any statistic by resampling your own data, with ten lines of NumPy. In production,
missing intervals are how teams ship "improvements" that are noise and roll back real ones.

## Prerequisite check

- Can you explain why the standard error of a mean is $\sigma/\sqrt{n}$, and what the CLT says about the shape
  of $\bar{X}_n$? If not, revisit [A5.4](a5_4_lln_clt_monte_carlo.md).
- Can you use an inverse CDF (`stats.norm.ppf(0.975)` = 1.96) to turn a probability into a cut-off? If
  not, revisit [A5.2](a5_2_common_distributions.md).
- Can you index an array with an integer array of shape `(B, n)` and take a statistic along `axis=1`? If
  not, revisit [A1.2](a1_2_numpy_arrays.md).
- Can you compute a median and a quantile by hand? If not, revisit
  [A6.1](a6_1_descriptive_statistics_eda.md).

## Intuition

### What an interval is for

You measured one sample and got one number, the **point estimate**. If you could repeat the whole data
collection many times, you would get many different estimates, spread around the truth. A confidence
interval is a range built from *your one sample* that is wide enough to have caught the truth in a
stated share of those imaginary repetitions. "95% confidence" is a property of the recipe: if you used
it on many samples, about 95% of the intervals it produces would contain the true value.

What it does **not** mean: that this particular interval contains the truth with probability 95%. Once
you have computed $[4.66, 9.24]$, the true mean is either in it or not; the 95% describes how often the
method succeeds, like a test with a 95% pass rate on known cases. (A Bayesian **credible interval** does
make the "95% probability" statement, at the price of a prior; see [A6.5](a6_5_bayesian_inference.md).)
Also not: "95% of the data lies in the interval". An interval for the mean gets narrower as $n$ grows; the
spread of the data does not.

### The bootstrap: resample your sample

The problem: to see how an estimate varies across samples, you need many samples from the population,
and you have one. Efron's (1979) idea: your sample is your best picture of the population, so draw new
samples *from your sample*. Each **bootstrap resample** picks $n$ rows from your $n$ rows *with replacement*
(some rows twice, some not at all), and you recompute the statistic on it. Repeat $B$ = 2,000 times. The
spread of the 2,000 values estimates the spread of your statistic across real samples, and the middle
95% of them, from the 2.5th to the 97.5th percentile, is the **percentile bootstrap** interval.

Why with replacement? Without it, every resample would be your sample in a different order and every
statistic would be identical. With it, each resample is like a new draw from a population that looks
exactly like your data.

> **Bridge:** the bootstrap is a load test for a statistic. You cannot get more users, so you replay the
> ones you have in random combinations and watch how much the number wobbles. **Where it breaks:** a
> replay can never produce a value you never saw. The bootstrap of the maximum (or a p99.9 from 500
> requests) is useless, because the real tail is beyond your data, and resampling rows assumes the rows
> are independent; correlated rows (a user's many sessions, consecutive minutes) need resampling by
> user or by block of time.

### Small samples: the t interval

$\bar{x} \pm 1.96\,s/\sqrt{n}$ pretends the standard error is known exactly. With small $n$, $s$ itself is noisy and often
too small, so the interval must be wider. If the data is Gaussian, the right multiplier comes from the
**t distribution** with $n - 1$ **degrees of freedom**: 3.18 for $n = 4$, 2.09 for $n = 20$, 2.01 for $n = 50$, approaching
1.96. The t interval is exact for Gaussian data and approximately right for other shapes once $n$ is
moderate. For skewed data and small $n$, nothing is exact; the lab measures how wrong each method is.

### Proportions near 0 or 1

For a rate (conversion, error rate) the Wald interval $\hat{p} \pm 1.96\sqrt{\hat{p}(1 - \hat{p})/n}$ fails badly near 0 and 1:
with 0 errors in 50 requests it says $[0, 0]$, total certainty from 50 data points. The **Wilson
interval** fixes this by solving for the $p$ values that are consistent with the data instead of plugging
in $\hat{p}$. A handy check for zero events: the upper end of a 95% interval is roughly $3/n$ (the "rule of three"). Why 3: a rate $p$
is still plausible if seeing 0 events in $n$ tries has probability at least 5%; $(1 - p)^n \approx e^{-pn} = 0.05$ gives $pn = \ln 20 \approx 3$.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{x}$ | the observed sample $x_1, \dots, x_n$ | `(n,)` |
| $\theta$ | the true value of the quantity (population mean, median, ...) | scalar |
| $\hat{\theta}$ | the estimate computed from $\mathbf{x}$ | scalar |
| $1 - \alpha$ | the confidence level (0.95 means $\alpha = 0.05$) | scalar |
| $t_{n-1}$ | the t distribution with $n - 1$ degrees of freedom | — |
| $t^*$ | its $1 - \alpha/2$ quantile, `stats.t.ppf(0.975, n - 1)` for 95% | scalar |
| $B$ | number of bootstrap resamples | int |
| $\mathbf{x}^{*b}$ | the $b$-th bootstrap resample, $n$ rows drawn with replacement from $\mathbf{x}$ | `(n,)` |
| $\hat{\theta}^{*b}$ | the statistic computed on $\mathbf{x}^{*b}$ | scalar |
| $k$, $m$ | number of successes and trials for a proportion; $\hat{p} = k/m$ | ints |
| $z$ | the normal quantile, 1.96 for 95% | scalar |

**What "95%" means, as a formula.** A confidence interval is a pair of functions $L(\mathbf{x}), U(\mathbf{x})$ with
$P\big(L(\mathbf{X}) \le \theta \le U(\mathbf{X})\big) = 1 - \alpha$, where the probability is over the random sample $\mathbf{X}$; $\theta$ is fixed. That
probability is the **coverage** of the method. A method is checked by simulation exactly as the
definition says: draw many samples from a known distribution, build the interval each time, count.

**The t interval.** If $X_1, \dots, X_n$ are i.i.d. $\mathcal{N}(\mu, \sigma^2)$, then

$$T = \frac{\bar{X} - \mu}{s/\sqrt{n}} \sim t_{n-1}$$

(stated, not proved; W. S. Gosset published it in 1908 under the name "Student"). Compared with the
normal, $t_{n-1}$ has heavier tails, because the denominator $s$ is itself random and sometimes small. Since
$P(-t^* \le T \le t^*) = 1 - \alpha$, rearranging the inequality for $\mu$ gives the interval

$$\bar{x} \pm t^*\,\frac{s}{\sqrt{n}}.$$

The "$n - 1$ degrees of freedom" is the same $n - 1$ as in $s$: after the $n$ deviations $x_i - \bar{x}$ are forced to
sum to zero, only $n - 1$ of them are free.

**The percentile bootstrap.** For $b = 1, \dots, B$: draw $\mathbf{x}^{*b}$ by picking $n$ indices uniformly from $\{1, \dots, n\}$
with replacement, and compute $\hat{\theta}^{*b}$. The interval is

$$\big[Q^*(\alpha/2),\ Q^*(1 - \alpha/2)\big],$$

the $\alpha/2$ and $1 - \alpha/2$ quantiles of the $B$ values $\hat{\theta}^{*b}$. The **bootstrap standard error** is their sample standard
deviation. The idea (the **plug-in principle**): the empirical distribution, which puts probability $1/n$
on each observed value, stands in for the unknown population, and resampling from it stands in for
sampling from the population.

**A check you can do by hand: the bootstrap SE of a mean.** A resample is $n$ independent draws from the
empirical distribution, whose mean is $\bar{x}$ and whose variance is the *divide-by-$n$* variance
$\hat{\sigma}^2 = \frac1n\sum_i (x_i - \bar{x})^2$ (A5.1: it is the variance of a variable that takes each value with probability $1/n$). So by
$\operatorname{Var}[\bar{X}] = \sigma^2/n$ (A5.1), the bootstrap means have standard deviation $\hat{\sigma}/\sqrt{n} = \sqrt{\frac{n-1}{n}}\cdot\frac{s}{\sqrt{n}}$, slightly
smaller than the usual SE. For $n = 5$ the factor is $0.89$. On top of that, the percentile interval takes its
width from the resamples themselves, with nothing like the t distribution's extra width for a noisy $s$, so the
percentile bootstrap is too narrow in small samples. Two remedies exist: the **BCa** interval
("bias-corrected and accelerated", Efron 1987), which shifts and stretches the percentiles to correct for
bias and skew, and the "bootstrap-t". SciPy's `stats.bootstrap` uses BCa by default.

**Two groups.** To get an interval for a difference (B minus A in medians, means or rates), resample
group A within A and group B within B, keeping both sizes, compute the difference for each pair of
resamples, and take percentiles of the $B$ differences. Resampling within groups mirrors how the data
was collected: the group sizes were fixed by the experiment.

**Wilson interval for a proportion.** The interval is the set of $p$ for which $\lvert\hat{p} - p\rvert \le z\sqrt{p(1 - p)/m}$, using the
standard error at the candidate $p$, not at $\hat{p}$. Squaring and solving the quadratic in $p$ gives

$$\frac{\hat{p} + \frac{z^2}{2m}}{1 + \frac{z^2}{m}} \;\pm\; \frac{z}{1 + \frac{z^2}{m}}\sqrt{\frac{\hat{p}(1 - \hat{p})}{m} + \frac{z^2}{4m^2}}.$$

The centre is pulled from $\hat{p}$ towards $\frac12$, by about as much as adding $z^2/2 \approx 2$ successes and 2 failures, and
the interval never leaves $[0, 1]$.

## Worked example by hand

**1. A t interval.** Four response times, $3, 5, 4, 8$ s. Mean 5; deviations $-2, 0, -1, 3$; squares $4, 0, 1, 9$, sum 14;
$s = \sqrt{14/3} = 2.160$; $\mathrm{SE} = 2.160/\sqrt4 = 1.080$. With $n - 1 = 3$ degrees of freedom, $t^* = 3.182$ (from a table or
`stats.t.ppf(0.975, 3)`), so the interval is $5 \pm 3.182 \times 1.080 = 5 \pm 3.437 = [1.56, 8.44]$. The normal multiplier would give
$5 \pm 2.12 = [2.88, 7.12]$, too narrow by a third.

**2. Three bootstrap resamples.** Number the values 0 to 3 (0-based, as in the code). Draw four indices
with replacement, say $0, 0, 3, 2$: the resample is $3, 3, 8, 4$, mean $4.50$. Then $1, 3, 3, 2$: $5, 8, 8, 4$, mean $6.25$.
Then $2, 2, 0, 1$: $4, 4, 3, 5$, mean $4.00$. Each resample leaves some values out and repeats others; that is
where the variation comes from. With $n = 4$ there are only $4^4 = 256$ equally likely ordered resamples, so the
lab lists all of them: their means average exactly 5, their standard deviation is $0.935$, and the
formula above predicts $\hat{\sigma}/\sqrt{n} = \sqrt{14/4}/2 = \sqrt{3.5}/2 = 0.935$. The exact percentile interval is $[3.5, 7.0]$, much
narrower than the t interval: at $n = 4$, the bootstrap cannot know how much wider the population might
be than these four values.

**3. A proportion.** 7 conversions in 20 visits: $\hat{p} = 0.35$. Wald: $0.35 \pm 1.96\sqrt{0.35 \times 0.65/20} = 0.35 \pm 0.209 = [0.141, 0.559]$.
Wilson, with $z^2 = 3.8416$, $z^2/m = 0.1921$: centre $(0.35 + 0.0960)/1.1921 = 0.3742$; half-width
$\frac{1.96}{1.1921}\sqrt{0.011375 + 0.002401} = 1.6442 \times 0.11737 = 0.1930$ (the two terms under the root are
$\hat{p}(1 - \hat{p})/m = 0.35 \times 0.65/20$ and $z^2/4m^2 = 3.8416/1600$); interval $[0.181, 0.567]$, shifted towards $\frac12$. With 0 failures in 50:
Wald gives $[0, 0]$, Wilson $[0, 0.071]$, and the rule of three $3/50 = 0.06$.

```text
== worked example by hand ==
mean = 5.0000   deviations = [-2.0, 0.0, -1.0, 3.0]   s = sqrt(14/3) = 2.1602   SE = s/2 = 1.0801
t(0.975, df=3) = 3.1824   t-interval = 5 +- 3.4374 = [1.5626, 8.4374]
z-interval (wrong for n = 4) = 5 +- 2.1170 = [2.8830, 7.1170]
resample indices [0, 0, 3, 2] -> [3.0, 3.0, 8.0, 4.0] -> mean 4.50
resample indices [1, 3, 3, 2] -> [5.0, 8.0, 8.0, 4.0] -> mean 6.25
resample indices [2, 2, 0, 1] -> [4.0, 4.0, 3.0, 5.0] -> mean 4.00
all 256 resamples: mean of bootstrap means = 5.0000   sd of bootstrap means = 0.9354   sqrt(sum (x - mean)^2 / n / n) = 0.9354
exact percentile bootstrap interval = [3.5000, 7.0000]
proportion 7/20 = 0.35: Wald 0.35 +- 0.2090 = [0.1410, 0.5590]
Wilson: centre = 0.3742   interval = [0.1812, 0.5671]
0/50 failures: Wald = [0, 0]   Wilson = [0.0000, 0.0714]   rule of three 3/50 = 0.0600
```

## Key plot

![Left: 50 horizontal t-intervals from repeated samples of size 20 of a skewed variable around a dashed line at the true mean 1; one interval, drawn in red with x markers, lies entirely below it. Right: coverage against sample size on a log axis for four methods; all start below the 0.95 line at n = 5 (the bootstrap of the mean lowest at 0.78, the bootstrap of the median highest at 0.93) and approach 0.95 by n = 50](img/a6_2_coverage.png)

Left: what "95%" means. Each line is the interval from a different sample; most catch the truth, some
do not, and from one sample you cannot tell which kind you have. (This draw of 50 had 1 miss; at
92% coverage about 4 are expected, so this is a lucky 50.) Right: on skewed data, intervals for the mean
are too narrow when $n$ is small, and the percentile bootstrap is the worst of them there.

## From scratch in NumPy

**Bootstrap in one vectorized line.** Draw a `(B, n)` array of row indices, index the data with it, and
reduce along `axis=1`:

```python
def bootstrap_stats(
    x: FloatArray, stat: Callable[..., FloatArray], n_boot: int, gen: np.random.Generator
) -> FloatArray:
    """Statistic of n_boot resamples drawn with replacement. x (n,) -> (n_boot,)."""
    idx = gen.integers(0, len(x), size=(n_boot, len(x)))  # (n_boot, n) row indices
    return stat(x[idx], axis=1)  # (n_boot,)


def percentile_ci(boot: FloatArray, conf: float = CONF) -> tuple[float, float]:
    """Middle conf share of the bootstrap distribution. boot (n_boot,) -> (low, high)."""
    a = (1 - conf) / 2
    lo, hi = np.quantile(boot, [a, 1 - a])
    return float(lo), float(hi)
```

`x[idx]` has shape `(B, n)`: row $b$ is resample $b$. Any NumPy reduction that takes an `axis` works as `stat`
(`np.mean`, `np.median`, `np.percentile` with a `q` via `functools.partial`). The lab applies it to 40
simulated session lengths (exponential, true mean 6 minutes, true median $6\ln 2 = 4.16$) and to two groups
of 200 order values whose true medians are 20 and 24:

```text
== from scratch: bootstrap intervals ==
mean   6.9470: t-interval [4.6589, 9.2352]   bootstrap [4.8727, 9.2518]   bootstrap SE 1.1112   s/sqrt(n) 1.1312
median 5.1109: no simple formula            bootstrap [3.2651, 6.8781]   bootstrap SE 1.0057
distinct values among 2000 bootstrap medians: 92   (the median can only land on a data value or a midpoint)
difference in medians B - A = 2.5619   95% bootstrap CI [-2.5440, 7.6972]   share of resamples with B <= A: 0.1575
```

- For the mean, the bootstrap and t intervals roughly agree, and the bootstrap SE (1.11) is just under
  $s/\sqrt{n}$ (1.13), as the formula predicts ($\sqrt{39/40} = 0.987$). The bootstrap interval is not symmetric
  around 6.95: it follows the skew of the data.
- For the median there is no simple formula, and the bootstrap gives an interval with no new math. Its
  bootstrap distribution is lumpy (92 distinct values), because the median of a resample is always one
  of the data values or a midpoint of two.
- The true difference in medians is 4, but the interval $[-2.5, 7.7]$ includes 0. This is the honest
  answer for 200 noisy orders per group: "B may be up to 7.7 higher, or 2.5 lower; we cannot tell the
  sign yet". An interval that includes 0 does not show "no effect"; it shows that this sample cannot
  rule zero out. Its width tells you how much more data you need.

**Coverage.** The definition says: repeat the experiment, count how often the interval contains the
truth. The lab does that for samples from an Exponential(1) (true mean 1, true median $\ln 2$), 1,000
samples for each $n$, every bootstrap with 2,000 resamples:

```text
== from scratch: coverage simulation ==
Exponential(1) samples, 1000 runs per n, 2000 resamples per bootstrap; target 0.95
   n   z mean   t mean  boot mean  boot median  t width  boot width
   5    0.811    0.890      0.783        0.935    2.157       1.321
  10    0.872    0.896      0.861        0.935    1.326       1.069
  20    0.903    0.921      0.898        0.927    0.892       0.806
  50    0.942    0.949      0.939        0.947    0.561       0.539
 100    0.947    0.953      0.949        0.956    0.392       0.383
Monte Carlo standard error of each coverage number: sqrt(0.95 * 0.05 / 1000) = 0.0069
control: t-interval on Gaussian data, n = 5: coverage 0.959
```

How to read it. Each coverage number is itself an estimate from 1,000 runs, with a standard error of
0.007 (it is a proportion, A5.2), so differences under about 0.015 are noise. The control row checks the
code: on Gaussian data, where the t interval is exact, it covers 0.959 (within noise of 0.95). On the
skewed data, at $n = 5$ the "95%" intervals for the mean cover 78% to 89%: the t interval is best because it
is widest, and the percentile bootstrap is worst because it is narrowest (the $\sqrt{(n-1)/n}$ effect plus
normal-like percentiles). Why do skewed samples miss? A small sample usually lacks the rare large
values, so its mean *and* its $s$ are both too small, and the interval sits too low and is too narrow at
the same time; that is the red interval in the key plot. By $n = 50$ all methods are within noise of 0.95.
The bootstrap of the median does well even at $n = 5$, because the median does not depend on the tail.

## With SciPy

```text
== with scipy ==
stats.t.interval: [4.6589, 9.2352]   matches t_interval: OK
scipy percentile: [4.9342, 9.2552]   mine: [4.8727, 9.2518]   scipy BCa: [5.2310, 9.8828]
percentile intervals agree within Monte Carlo error: OK
bootstrap standard error: scipy 1.1179   mine 1.1112   OK
binomtest(7, 20).proportion_ci(method='wilson'): [0.1812, 0.5671]   matches wilson: OK
exact (Clopper-Pearson) for comparison: [0.1539, 0.5922]
n = 20, 300 runs: coverage of scipy percentile 0.917   BCa 0.913
```

- `stats.t.interval(0.95, df=n - 1, loc=x.mean(), scale=stats.sem(x))` is the t interval
  (`stats.sem` is $s/\sqrt{n}$ with `ddof=1`). It matches ours to machine precision.
- `stats.bootstrap((x,), np.mean, n_resamples=B, method="percentile", rng=...)` resamples with its own
  random numbers, so it agrees with ours only up to Monte Carlo error; both differ in the second
  decimal. The default `method="BCa"` shifts this interval to the right, following the right skew of the
  data. Note the tuple `(x,)`: `bootstrap` takes a sequence of samples so it can handle statistics of two
  groups. VERIFY (as of 2026-10), SciPy 1.18.1: `rng=` replaced the older `random_state=` argument.
- In the last line BCa does not beat the percentile interval at $n = 20$ (0.913 against 0.917, both within
  noise of each other with 300 runs, standard error about 0.013). BCa's corrections help most for biased
  or very skewed statistics; for the mean of 20 skewed values, no interval method rescues a sample that
  happens to lack the tail.
- `stats.binomtest(k, m).proportion_ci(method="wilson")` matches our Wilson formula (with the exact
  $z = 1.95996$). `method="exact"` is the Clopper–Pearson interval, which guarantees at least 95% coverage
  and is wider.

## In practice

- **When to use it:** attach an interval to every reported metric and every difference: test accuracy,
  AUC, a p95 latency, conversion rates, the gain of model B over model A. Use the t interval for means of
  roughly symmetric data, Wilson for proportions, and the bootstrap (BCa by default) for everything else:
  medians, percentiles, ratios, correlations, differences between groups, model metrics on a test set
  (resample the test rows and recompute the metric).
- **Failure modes:**
  - *Small skewed samples.* Every method undercovers (the table above). You notice by running a coverage
    simulation on data shaped like yours; report the interval with the sample size and prefer the median.
  - *Correlated rows.* Resampling rows of a time series, or many sessions per user, treats them as
    independent and gives intervals that are too narrow. You notice when repeated experiments disagree by
    more than their intervals. Resample the independent unit (users, days) or use a **block bootstrap**
    (resample blocks of consecutive rows).
  - *Extreme statistics.* The maximum, a p99.9 from a few hundred values, anything decided by one or two
    rows: the bootstrap cannot see beyond the data. You notice a bootstrap distribution with only a
    handful of distinct values. Collect more data or report a less extreme percentile.
  - *Reading the interval as a probability statement about this interval*, or as the range of the data.
  - *Comparing two intervals by overlap.* Two 95% intervals can overlap while the difference is clearly
    nonzero. Build the interval of the *difference* directly.
  - *Too few resamples.* With $B = 200$ the endpoints wobble between runs. You notice by rerunning with a
    different seed. Use $B \ge 2{,}000$ for 95% intervals (more for 99%).
- **Defaults that work:** `stats.bootstrap(..., method="BCa", n_resamples=9999)` (SciPy's defaults,
  VERIFY as of 2026-10); `SEED = 0` passed as `rng` so reports are reproducible; Wilson for rates; report
  "estimate (95% CI low to high, n = …)".

## Exercises

Save your work to `my-work/A/a6_2_confidence_intervals_bootstrap.md` (or `.py`) and ask for a review.

1. **By hand:** (a) Five page-load times are $1.2, 0.8, 1.5, 1.1, 2.4$ s. Compute the mean, $s$, the standard error
   and the 95% t interval ($t^* = 2.776$ for 4 degrees of freedom). How much narrower would the (wrong) $z$
   interval be? (b) Using the indices $4, 0, 0, 2, 1$ and then $3, 3, 1, 4, 4$ (0-based), write out two bootstrap resamples
   and their means and medians. (c) For the 5 values, what is the exact bootstrap standard error of the
   mean (no resampling needed)? (d) A new checkout flow had 0 failures in 120 attempts. Give the Wald
   interval, the rule-of-three upper bound and the Wilson interval for the failure rate. Which would you
   put in the release notes?
2. **Code:** Put an interval on the most reported metric of all, accuracy. Simulate a test set of $n = 500$
   examples where a classifier is right with probability 0.86 (`SEED = 0`). (a) Give the Wald, Wilson and
   percentile-bootstrap intervals for its accuracy. (b) Simulate a second classifier on the *same* 500
   examples that agrees with the first on 90% of them and has true accuracy 0.88 (hint: start from the first
   classifier's correctness vector and flip a random subset). Give a bootstrap interval for the difference
   in accuracy by resampling *examples* (the same indices for both classifiers), and compare it with the
   wrong approach of resampling each classifier's results independently. Which interval is narrower, and
   why? (c) Run a coverage check of your paired interval with 500 repetitions.
3. **Think:** Your team's dashboard shows the p99 latency of each service per hour, computed from about
   300 requests per hour for small services. A manager wants an alert whenever p99 rises 20% above the
   previous hour. Using what you learned about intervals, explain what will happen, and propose what to
   compute and alert on instead.

## Takeaways

1. A 95% confidence interval comes from a procedure that catches the true value in 95% of repeated
   samples; one computed interval either contains it or not.
2. The t interval $\bar{x} \pm t^*_{n-1}\,s/\sqrt{n}$ is exact for Gaussian data and widens for small $n$; Wilson is the default
   for proportions, and never gives $[0, 0]$.
3. The percentile bootstrap resamples rows with replacement, recomputes the statistic $B$ times and takes
   the middle 95%; it works for medians, percentiles, differences and model metrics with no new math.
4. For a difference, resample within each group (or resample shared examples when two models score the
   same test set) and build the interval of the difference, not two overlapping intervals.
5. Check coverage by simulation: small skewed samples, correlated rows and extreme statistics all make
   "95%" intervals cover much less than 95%.

**What builds on this:** [A6.3a](a6_3a_hypothesis_tests.md) (tests and intervals are two views of the same
question); [A6.3b](a6_3b_multiple_comparisons_ab_tests.md) (intervals for A/B test effects);
[A6.5](a6_5_bayesian_inference.md) (credible intervals); [B4.3](../B/b4_3_cross_validation_tuning.md)
(uncertainty of cross-validation scores); [J2.2](../J/j2_2_monitoring_drift.md) (alert thresholds on noisy
metrics); P1 (bootstrap CIs for every effect).

**References:** B. Efron, "Bootstrap Methods: Another Look at the Jackknife", *Annals of Statistics* 7(1),
1979. B. Efron, "Better Bootstrap Confidence Intervals", *Journal of the American Statistical
Association* 82(397), 1987. "Student" (W. S. Gosset), "The Probable Error of a Mean", *Biometrika* 6(1),
1908. E. B. Wilson, "Probable Inference, the Law of Succession, and Statistical Inference", *Journal of
the American Statistical Association* 22(158), 1927.
