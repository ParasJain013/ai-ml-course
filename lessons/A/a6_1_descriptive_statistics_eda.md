# A6.1 Descriptive statistics and EDA

**Core** · Part A, week 6 (course week 06) · ~1.5 h · Lab:
[`labs/A/a6_1_descriptive_statistics_eda.py`](../../labs/A/a6_1_descriptive_statistics_eda.py)

## Why it matters

Before you model anything, you have to know what is in the data: how big the values are, how spread
out, how lopsided, what is broken, and which columns move together. That is **exploratory data
analysis (EDA)**, and the numbers it produces are **descriptive statistics**. Two mistakes cost the
most in practice. The first is trusting a mean and a standard deviation on data with a long tail or a
few broken values, where one timeout or one mistyped price moves them more than everything else
combined. The second is reading a correlation as a cause and acting on it: shipping a "fix" that does
nothing, or a model feature that stops working the day the hidden cause changes. P1's report and every
dataset you meet in Part B start here.

## Prerequisite check

- Can you compute a mean, a sample standard deviation (with $n - 1$) and a covariance by hand? If not,
  revisit [A5.1](a5_1_random_variables_expectation.md).
- Can you sort, group and aggregate a DataFrame with `groupby`? If not, revisit [A1.3](a1_3_pandas.md).
- Do you know why a histogram of a skewed variable is easier to read on a log x-axis? If not, revisit
  [A1.4](a1_4_matplotlib.md).
- Can you fit a straight line by least squares and take its residuals? If not, revisit
  [A2.4](a2_4_projections_least_squares.md).

## Intuition

### Centre and spread, and who gets a vote

A summary statistic is a vote. The **mean** gives every value a vote proportional to its size, so one
huge value can outvote all the others: the average wealth in a room jumps when a billionaire walks in.
The **median**, the middle value after sorting, gives every value one vote of the same weight, so the
billionaire counts as "one more person above the middle" and nothing else. The median is a **robust
statistic**: it barely moves when a few values are wrong or extreme.

Spread works the same way. The standard deviation squares every distance from the mean, so a single
far-away value dominates it. Two robust alternatives:

- The **interquartile range (IQR)**: the distance between the 25th percentile (Q1) and the 75th (Q3),
  the width of the middle half of the data.
- The **median absolute deviation (MAD)**: the median of the distances from the median. "How far is a
  typical value from the typical value."

A third option sits between the mean and the median: the **trimmed mean** sorts the data, drops a fixed
share from each end (say 10%), and averages the rest.

How much bad data can each one survive? This is the **breakdown point**: the share of values you can
replace with garbage before the statistic can be pushed arbitrarily far. The mean's is $0$ (one value is
enough), a 10% trimmed mean's is 10%, and the median's and MAD's are 50%.

> **Bridge:** you already use robust statistics. Latency dashboards show p50, p95 and p99 rather than
> the mean, because the mean of latencies is dragged around by a handful of timeouts, and p50 is the
> median. **Where it breaks:** robust is not the same as right. If you are paying for total CPU time or
> total bytes, the mean (times $n$) *is* the quantity you pay for, and the tail is where the money goes.
> Choose the summary that answers the question; report the median *and* the mean when they differ.

### Shape

Two variables can share a mean and a standard deviation and look nothing alike. Data is
**skewed** when one tail is longer (latencies, prices, file sizes: a long right tail, mean above median);
**heavy-tailed** when extreme values are far more common than a Gaussian allows; **multimodal** when it
has several bumps (often a sign of two populations mixed together, such as mobile and desktop users).
None of this shows up in a mean and a standard deviation. It shows up in a histogram, which is why EDA
always plots. For a long right tail, plot the histogram on a log x-axis and summarize the log: the mean
of $\log x$, transformed back, is the **geometric mean**, which for data whose log is symmetric (such as the log-normal) sits near the median.

### Relationships: correlation

**Pearson correlation** $r$ measures how well a *straight line* describes two variables, from $-1$ (a
perfect falling line) through 0 (no linear trend) to $+1$ (a perfect rising line). **Spearman correlation**
$\rho$ replaces each value by its rank (1 for the smallest, 2 for the next, and so on) and then computes
Pearson's $r$ on the ranks. It measures how *monotonic* the relationship is (does $y$ always go up when $x$
goes up?), whether or not the trend is straight, and an outlier can only be "the largest rank", so it
cannot dominate.

Anscombe (1973) built four tiny datasets with the same means, variances, correlation and least-squares
line, and completely different pictures (the key plot below). The lesson of **Anscombe's quartet** is
the rule of EDA: summary statistics are a description of a plot you have already looked at, never a
replacement for it.

### Correlation is not causation

Servers with high CPU usage log more errors. Does CPU cause errors? Maybe, but there is an obvious
alternative: traffic drives both. A variable that affects both the "cause" and the "effect" is a
**confounder**, and it creates correlation with no causal link at all. Three standard explanations for a
correlation between $X$ and $Y$, besides "$X$ causes $Y$": $Y$ causes $X$ (reverse causation), a third
variable causes both (confounding), or the data was selected in a way that links them (selection bias:
if you only study incidents that were escalated, everything in the data is correlated with "escalated").

Confounding can even reverse a comparison. **Simpson's paradox**: a treatment can look better in every
subgroup and worse overall, because the subgroups are mixed in different proportions. The worked
example has the real numbers. The only reliable way to rule out confounders you have not measured is
to *assign* $X$ yourself, at random, which is what an experiment does ([A6.3b](a6_3b_multiple_comparisons_ab_tests.md)).

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{x}$ | a sample of one variable, $x_1, \dots, x_n$ | `(n,)` |
| $x_{(k)}$ | the $k$-th smallest value (an **order statistic**); $x_{(1)}$ is the minimum | scalar |
| $\bar{x}$, $s$ | sample mean and sample standard deviation (with $n - 1$, A5.1) | scalars |
| $Q(p)$ | the $p$-quantile: the value below which a share $p$ of the data lies; $Q(0.5)$ is the median | scalar |
| $\mathrm{IQR}$ | $Q(0.75) - Q(0.25)$ | scalar |
| $\mathrm{MAD}$ | $\operatorname{median}_i \lvert x_i - \operatorname{median}(\mathbf{x})\rvert$ | scalar |
| $\mathbf{y}$ | a second variable measured on the same $n$ examples | `(n,)` |
| $r$, $\rho$ | Pearson and Spearman correlation | scalars in $[-1, 1]$ |
| $R(x_i)$ | the rank of $x_i$ among $x_1, \dots, x_n$ (1 = smallest) | scalar |

Indices start at 1 in the math and 0 in the code: $x_{(1)}$ is `np.sort(x)[0]`.

**Median.** Sort the values. For odd $n$ the median is the middle one, $x_{((n+1)/2)}$; for even $n$ it is the
mean of the two middle ones.

**Quantiles.** There are about nine conventions for a quantile between two data points; they differ only
for small $n$. This course uses NumPy's default ("linear"): put the sorted values at 0-based positions
$0, 1, \dots, n - 1$, compute the position $h = (n - 1)p$, and interpolate linearly between the values at
$\lfloor h\rfloor$ and $\lfloor h\rfloor + 1$:

$$Q(p) = x_{(\lfloor h\rfloor + 1)} + (h - \lfloor h\rfloor)\big(x_{(\lfloor h\rfloor + 2)} - x_{(\lfloor h\rfloor + 1)}\big), \qquad h = (n - 1)p.$$

(The $+1$ and $+2$ convert the 0-based position to the 1-based order statistic.) pandas uses the same
convention; Excel's `QUARTILE.EXC` and many textbooks do not, so expect small differences for small $n$.

**IQR and Tukey's fences.** $\mathrm{IQR} = Q(0.75) - Q(0.25)$. Tukey's rule of thumb flags values outside
$[Q(0.25) - 1.5\,\mathrm{IQR},\ Q(0.75) + 1.5\,\mathrm{IQR}]$ as possible **outliers**; it is the rule behind the whiskers of a
box plot. For Gaussian data it flags about 0.7% of values, so in a large clean sample it *will* flag some;
"outside the fences" means "look at this", not "delete this".

**MAD, and why 1.4826.** For Gaussian data, half the values lie within $0.6745\sigma$ of the median
($\Phi^{-1}(0.75) = 0.6745$, A5.2), so the MAD is about $0.6745\sigma$. Dividing by 0.6745, that is multiplying by
$1/0.6745 = 1.4826$, turns the MAD into a robust estimate of $\sigma$. The **robust z-score**
$z_i = (x_i - \operatorname{median})/(1.4826\,\mathrm{MAD})$ is then the usual z-score with robust ingredients; a common rule
(Iglewicz and Hoaglin, 1993) flags $\lvert z_i\rvert > 3.5$.

**Trimmed mean.** Sort, drop $k = \lfloor \text{prop} \times n\rfloor$ values from each end, average the remaining $n - 2k$.

**Skewness.** Standardize each value, $u_i = (x_i - \bar{x})/\hat{\sigma}$ with $\hat{\sigma}$ dividing by $n$, and average the cubes:
$\text{skew} = \frac1n\sum_i u_i^3$ (this is `scipy.stats.skew`; pandas' `.skew()` adds a small-sample correction, so it differs slightly for small $n$). Cubing keeps the sign, so a long
right tail (a few large positive $u_i$) gives a positive skew, a long left tail a negative one, and a symmetric shape about 0.
A skew above about 1 is a hint to look at a log x-axis. Like the mean, it is not robust: one extreme value can dominate it.

**Pearson correlation.** The covariance divided by both standard deviations (A5.1). Written with
deviations $d^x_i = x_i - \bar{x}$ and $d^y_i = y_i - \bar{y}$, the $\frac{1}{n-1}$ factors cancel:

$$r = \frac{\sum_{i=1}^n d^x_i d^y_i}{\sqrt{\sum_{i=1}^n (d^x_i)^2}\,\sqrt{\sum_{i=1}^n (d^y_i)^2}}.$$

The numerator is large and positive when $x$ and $y$ are above their means together and below them
together. Dividing by the two root sums of squares makes $r$ unit-free and puts it in $[-1, 1]$ (by the
Cauchy–Schwarz inequality: it is the cosine of the angle between the centred vectors, A1.5). $r^2$ is the
share of $y$'s variance that the least-squares line on $x$ explains.

**Spearman correlation.** $\rho = r\big(R(\mathbf{x}), R(\mathbf{y})\big)$: Pearson on the ranks. Tied values get the average of the
ranks they occupy (two values tied for 2nd and 3rd both get 2.5). $\rho = 1$ exactly when $y$ increases
whenever $x$ does.

**Partial correlation.** To ask "are $x$ and $y$ related once $z$ is accounted for?", fit $x$ on $z$ and $y$ on $z$ by
least squares (A2.4), take the two sets of residuals (what $z$ cannot explain), and correlate those. If $z$
is the only link between $x$ and $y$, and the link is linear, the partial correlation is near 0. It can only
adjust for confounders you have measured.

## Worked example by hand

**1. A small latency sample.** Seven requests took $2, 3, 3, 4, 5, 7, 40$ ms (already sorted), $n = 7$.

- Mean: $64/7 = 9.143$ ms, larger than six of the seven values. The squared deviations from the mean sum to $1{,}127$, so the sample sd
  is $\sqrt{1{,}127/6} = 13.70$ ms. The single 40 contributes $(40 - 9.14)^2 = 952$ of that $1{,}127$: 84% of the variance is one request.
- Median: the 4th value, **4** ms.
- Quartiles: $h = 6 \times 0.25 = 1.5$, halfway between 0-based positions 1 and 2, both 3, so $Q1 = 3$.
  $h = 6 \times 0.75 = 4.5$, halfway between 5 and 7, so $Q3 = 6$. **IQR = 3** ms.
- MAD: distances from the median 4 are $2, 1, 1, 0, 1, 3, 36$; sorted $0, 1, 1, 1, 2, 3, 36$; their median is
  **1** ms. Robust estimate of $\sigma$: $1.4826$ ms, against $s = 13.70$.
- 20% trimmed mean: $\lfloor 0.2 \times 7\rfloor = 1$ value off each end leaves $3, 3, 4, 5, 7$, mean **4.4** ms.
- Tukey's fences: $[3 - 4.5,\ 6 + 4.5] = [-1.5, 10.5]$; only 40 is outside. Its robust z-score is
  $(40 - 4)/1.4826 = 24.3$.
- Drop the 40: the mean falls from 9.14 to 4.00, the median only from 4 to 3.5.

**2. Pearson versus Spearman.** $x = 1, 2, 3, 4$ and $y = 1, 2, 3, 20$. Means 2.5 and 6.5. Deviations:
$d^x = -1.5, -0.5, 0.5, 1.5$; $d^y = -5.5, -4.5, -3.5, 13.5$. Products: $8.25 + 2.25 - 1.75 + 20.25 = 29$. Sums of squares:
$\sum (d^x)^2 = 5$, $\sum (d^y)^2 = 30.25 + 20.25 + 12.25 + 182.25 = 245$. So $r = 29/\sqrt{5 \times 245} = 29/35 = 0.829$. The ranks of $y$ are
$1, 2, 3, 4$, the same as $x$'s, so $\rho = 1$: perfectly monotonic, not perfectly linear.

**3. Simpson's paradox, with real numbers.** A study of kidney-stone treatments (Charig et al., 1986)
reported:

| | Small stones | Large stones | All patients |
|---|---|---|---|
| Treatment A | 81/87 = **93.1%** | 192/263 = **73.0%** | 273/350 = 78.0% |
| Treatment B | 234/270 = 86.7% | 55/80 = 68.8% | 289/350 = **82.6%** |

A wins in both groups and loses overall. Why: large stones are harder to treat, and A was given mostly to
large stones (263 of its 350 patients) while B was given mostly to small ones (270 of 350). Stone size
confounds the comparison. The pooled rate of each treatment is a weighted average of its two group
rates, and the weights differ: $0.780 = \frac{87}{350}(0.931) + \frac{263}{350}(0.730)$, mostly the low large-stone rate. Which
table is "right" depends on the causal story: here size affected the choice of treatment, so compare
within size groups.

```text
== worked example by hand ==
n = 7   sum = 64   mean = 9.1429   sample sd = 13.7044
median = x[(n+1)/2] = 4th value = 4.0
Q1 at position (n-1)*0.25 = 1.5 -> 3.0   Q3 at position (n-1)*0.75 = 4.5 -> 6.0   IQR = 3.0
|x - median| = [2, 1, 1, 0, 1, 3, 36] -> sorted [0, 1, 1, 1, 2, 3, 36]   MAD = 1.0   scaled MAD = 1.4826 * MAD = 1.4826
20% trimmed mean: drop 1 from each end -> [3.0, 3.0, 4.0, 5.0, 7.0] -> 4.4
Tukey fences: [-1.5, 10.5]   outside: [40.0]
without the 40: mean = 4.0000   median = 3.5
x = [1.0, 2.0, 3.0, 4.0]   y = [1.0, 2.0, 3.0, 20.0]
deviations: dx = [-1.5, -0.5, 0.5, 1.5]   dy = [-5.5, -4.5, -3.5, 13.5]
sum dx*dy = 29.00   sum dx^2 = 5.00   sum dy^2 = 245.00   Pearson r = 0.8286
ranks of y = [1, 2, 3, 4] equal the ranks of x, so Spearman rho = 1.0
kidney-stone data (Charig et al., 1986):
treatment stone  success  patients  rate
        A small       81        87 0.931
        A large      192       263 0.730
        B small      234       270 0.867
        B large       55        80 0.688
pooled over stone size:
           success  patients  rate
treatment                         
A              273       350 0.780
B              289       350 0.826
```

## Key plot

![Anscombe's quartet: four scatter plots that share the same fitted line y = 3 + 0.5x and correlation 0.816; set I is a noisy line, II a smooth curve, III a perfect line with one outlier, IV a vertical stack of points plus one far-right point that creates the whole trend](img/a6_1_anscombe_quartet.png)

All four sets have mean $x = 9$, mean $y = 7.50$, variances 11 and 4.12, $r = 0.816$ and the line $y = 3 + 0.5x$. Only
set I is described well by those numbers. II is a curve (a line is the wrong model), III is a perfect
line spoiled by one point, and in IV a single point creates the entire correlation.

## From scratch in NumPy

**Robust summaries.** Each function is a few lines. The quantile follows the formula above, and the
median, IQR and MAD are built on it:

```python
def quantile_linear(x: FloatArray, p: float) -> float:
    """Quantile by linear interpolation between order statistics (NumPy's default). x (n,)."""
    xs = np.sort(x)
    h = (len(xs) - 1) * p  # 0-based position
    lo = int(np.floor(h))
    hi = min(lo + 1, len(xs) - 1)
    return float(xs[lo] + (h - lo) * (xs[hi] - xs[lo]))


def mad(x: FloatArray, scale: float = 1.0) -> float:
    """Median absolute deviation from the median, times scale. x (n,)."""
    return scale * median(np.abs(x - median(x)))
```

The lab then simulates 10,000 request latencies (log-normal, median about 120 ms) and replaces a
growing share with 30-second timeouts:

```text
== from scratch: robust summaries ==
hand sample: median 4.0   IQR 3.0   MAD 1.0   20% trimmed mean 4.4
hand sample robust z-scores: [-1.349, -0.6745, -0.6745, 0.0, 0.6745, 2.0235, 24.2817]   (|z| > 3.5 flags an outlier)
 timeouts      mean        sd   median     IQR  MAD*1.48  trim 10%
       0%     136.2      71.4    119.4    83.5      58.4     127.2
       1%     434.8    2972.4    120.3    85.2      59.4     128.4
       5%    1629.5    6509.3    124.6    92.7      63.9     134.4
      20%    6109.4   11946.0    142.6   164.9      88.5    3880.0
log-scale view: mean of log10(latency) = 2.0806 -> geometric mean 120.4 ms
```

Read the table by column. With 1% timeouts the mean triples and the standard deviation grows 40-fold;
the median, IQR and MAD barely move. The 10% trimmed mean survives 1% and 5% but not 20%: once more
than 10% of the values are garbage, trimming 10% cannot remove them all. That is its breakdown point
in action. Even the median drifts at 20% (from 119 to 143 ms), because every timeout pushes the middle
one position to the right; robust means "bounded damage", not "no damage". On the clean data, note
mean 136 > median 119: the log-normal's right tail. The geometric mean (120 ms) lands on the median.

**Correlation.** Pearson is the formula above. Spearman ranks first; ties share the average rank:

```python
def rank_average(x: FloatArray) -> FloatArray:
    """Ranks 1..n; tied values share the average of their ranks. x (n,) -> (n,)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):  # average the ranks within each group of ties
        tie = x == v
        ranks[tie] = ranks[tie].mean()
    return ranks
```

`ranks[order] = ...` is the inverse of `argsort`: `order[0]` is the index of the smallest value, so that
position gets rank 1, and so on.

```text
== from scratch: correlation ==
hand pair: Pearson 0.8286   Spearman 1.0000
ties: ranks of [10, 20, 20, 30] = [1.0, 2.5, 2.5, 4.0]
 set  mean x  mean y   var x   var y  Pearson  slope intercept  Spearman
   I    9.00    7.50   11.00    4.13    0.816  0.500      3.00     0.818
  II    9.00    7.50   11.00    4.13    0.816  0.500      3.00     0.691
 III    9.00    7.50   11.00    4.12    0.816  0.500      3.00     0.991
  IV    9.00    7.50   11.00    4.12    0.817  0.500      3.00     0.500
```

Every Pearson-based number agrees to two decimals, and Spearman tells the sets apart: III is almost
perfectly monotonic (0.991, the outlier is just "the largest rank"), II's curve goes down at the end (0.691),
and IV's vertical stack has no order at all apart from the one far point (0.500).

**A confounder.** The lab simulates 5,000 server-minutes. Traffic $z$ is uniform between 100 and 1,000
requests per second; CPU is $0.08z$ plus noise and errors are $0.01z$ plus noise. CPU never appears in the
formula for errors.

```text
== from scratch: a confounder ==
corr(cpu, errors) = 0.8065   (neither causes the other)
corr(traffic, cpu) = 0.9343   corr(traffic, errors) = 0.8627
partial correlation of cpu and errors given traffic = 0.0028
corr(cpu, errors) within traffic bands: [0.1103, 0.0963, 0.142, 0.1202, 0.0693, 0.1454]
if we set CPU at random (an intervention), corr(cpu, errors) = 0.0022
```

A correlation of 0.81 with no causal link. Three ways to see through it, in decreasing order of what
they need. Partial correlation (regress both on traffic, correlate the residuals) gives 0.003, but only
because traffic was measured and the relationships are linear. Stratifying into six traffic bands
shrinks the correlation to about 0.1, not 0: traffic still varies *inside* each band, so a little
confounding is left over (residual confounding; finer bands or the regression fix it). The last line is
what an experiment does: if *you* set the CPU at random, independently of everything else, the
correlation disappears, whether or not you know what the confounder was.

## With pandas and SciPy

```text
== with pandas and scipy ==
median               mine   124.6040   library   124.6040
quantile 0.9         mine   269.5338   library   269.5338
IQR                  mine    92.6726   library    92.6726
MAD (raw)            mine    43.1105   library    43.1105
MAD (normal scale)   mine    63.9156   library    63.9157
10% trimmed mean     mine   134.4136   library   134.4136
robust summaries match numpy / pandas / scipy: OK
pearson, spearman and ranks match scipy.stats on all four Anscombe sets: OK
DataFrame.corr(method='spearman') matches: OK
pandas describe() of the 5%-timeout latencies (mean and std are dragged by the timeouts):
count    10000.0
mean      1629.5
std       6509.3
min         21.8
25%         87.7
50%        124.6
75%        180.4
max      30000.0
```

- `Series.median()`, `Series.quantile(p)`, `np.quantile(x, p)`: same linear convention as above.
  `scipy.stats.iqr(x)`, `scipy.stats.median_abs_deviation(x, scale="normal")` (the scaled MAD; the default
  `scale=1.0` is the raw MAD), `scipy.stats.trim_mean(x, 0.1)` (cuts 10% from *each* end).
- `scipy.stats.pearsonr(x, y)` and `spearmanr(x, y)` return a result with `.statistic` and a `.pvalue`
  (A6.3a explains p-values). `df.corr(method="spearman")` gives the whole matrix. VERIFY (as of 2026-10),
  SciPy 1.18.1 and pandas 3.0.6.
- `df.describe()` is the first thing to run on any table, and the lab shows its trap: `mean` and `std`
  sit at the top, the robust numbers (`25%`, `50%`, `75%`) below them. With 5% timeouts the mean is
  1,630 ms and the median 125 ms; a reader who stops at the second line gets the wrong picture.

## In practice

- **When to use it:** on every new dataset, before any model. A practical EDA order: shape and dtypes;
  missing values per column (`df.isna().mean()`); `describe()` plus the median, IQR and share of distinct
  values per column; a histogram of every numeric column (log x-axis when the max is many times the
  median); bar charts of category counts; a Spearman correlation matrix and scatter plots of the pairs
  that matter; the same summaries split by the target or by time. Write down what surprises you; those
  notes become your cleaning log and your hypotheses (P1).
- **Failure modes:**
  - *Summaries without plots.* Anscombe's quartet. You notice only by plotting.
  - *Mean and sd on skewed or contaminated data.* You notice that the mean is far from the median, or
    that the max is hundreds of standard deviations away. Report the median and IQR (or model the log),
    and investigate the extremes before deciding whether they are errors or the most important rows.
  - *Deleting "outliers" automatically.* Tukey's fences flag about 0.7% of perfectly normal Gaussian data,
    and in a fraud or latency problem the extremes are the signal. Remove a value only with a reason
    you can write in the cleaning log (impossible value, unit error, test record).
  - *Pearson on non-linear or outlier-driven data.* You notice Pearson and Spearman disagreeing by
    more than about 0.1. Plot the pair.
  - *Correlation read as causation.* You notice when a "driver" found in observational data stops
    working after you change it. Ask "what else could cause both?" before you ask anything else, and
    prefer an experiment when the decision matters.
  - *Simpson's paradox in a dashboard.* A metric rises overall while it falls in every segment, because
    the mix of segments shifted (more mobile traffic, a new region). You notice by always splitting
    headline metrics by the largest segments.
- **Defaults that work:** report "median (IQR)" for skewed variables and "mean (sd)" only for roughly
  symmetric ones; use Spearman for a first correlation matrix; flag (do not delete) values with
  $\lvert\text{robust } z\rvert > 3.5$ or outside Tukey's fences; plot every numeric column on both linear and log axes once.

## Exercises

Save your work to `my-work/A/a6_1_descriptive_statistics_eda.md` (or `.py`) and ask for a review.

1. **By hand:** The daily sign-ups of a small product over eight days were $12, 15, 14, 10, 13, 95, 11, 14$.
   (a) Compute the mean, the median, Q1, Q3 and the IQR (use the $h = (n - 1)p$ rule), the MAD and the
   scaled MAD, and Tukey's fences. Which days are flagged? (b) Give the robust z-score of the 95. (c) Before
   deleting the 95, list two things it could be and what you would check for each. (d) For $x = 1, 2, 3, 4, 5$
   and $y = 5, 4, 3, 2, 20$, compute Pearson's $r$ and Spearman's $\rho$ by hand. Explain in one sentence why
   they disagree, and say which one you would put in a report about this data.
2. **Code:** Write `describe_robust(df: pd.DataFrame) -> pd.DataFrame` that returns, for every numeric
   column: count, share missing, mean, median, sd, IQR, scaled MAD, 10% trimmed mean, skewness, and the
   number of values with $\lvert\text{robust } z\rvert > 3.5$, using your own functions for the median, IQR, MAD and trimmed
   mean (missing values ignored) and checking each against pandas/SciPy. Test it on a synthetic table
   with `SEED = 0` and 2,000 rows: a Gaussian column, a log-normal column, a column with 3% of its values
   multiplied by 1,000 (a unit error), and a column with 10% missing values. Then compute the Spearman
   correlation matrix with your own `rank_average` and compare it with `df.corr(method="spearman")`.
   Which columns does your table say need a log or a closer look?
3. **Think:** An internal dashboard shows that teams that use feature flags have 40% fewer production
   incidents per deploy, and a director proposes making feature flags mandatory to cut incidents. Write
   the reply you would send: what could explain the correlation besides "flags prevent incidents", what
   you would check in the existing data, and what you would propose to find out whether flags help.

## Takeaways

1. The mean and standard deviation give extreme values the loudest vote; the median, IQR and MAD
   (breakdown point 50%) and the trimmed mean (its trim share) survive bad or extreme values.
2. The scaled MAD, $1.4826\,\mathrm{MAD}$, estimates $\sigma$ robustly; flag, do not automatically delete, values with
   robust $\lvert z\rvert > 3.5$ or outside Tukey's fences.
3. Pearson measures how well a line fits; Spearman (Pearson on ranks) measures monotonic trend and
   resists outliers. When they disagree, plot the pair.
4. Identical summaries can hide completely different data (Anscombe): plot first, then summarize.
5. Correlation can come from a confounder, reverse causation or selection; adjusting for measured
   confounders helps, and only randomized assignment rules out the unmeasured ones (Simpson's paradox
   shows how far pooled data can mislead).

**What builds on this:** [A6.2](a6_2_confidence_intervals_bootstrap.md) (how uncertain a median or a
correlation is); [A6.3a](a6_3a_hypothesis_tests.md) (is a difference real?);
[B1.2](../B/b1_2_data_cleaning.md) (cleaning decisions based on EDA);
[B1.3](../B/b1_3_feature_engineering.md) (log transforms and robust scaling);
[B4.6](../B/b4_6_fairness_model_cards.md) (per-group metrics and Simpson's paradox in fairness); P1 (the
EDA and report).

**References:** F. J. Anscombe, "Graphs in Statistical Analysis", *The American Statistician* 27(1),
1973. C. R. Charig, D. R. Webb, S. R. Payne and J. E. A. Wickham, "Comparison of treatment of renal
calculi by open surgery, percutaneous nephrolithotomy, and extracorporeal shockwave lithotripsy",
*BMJ* 292, 1986. B. Iglewicz and D. C. Hoaglin, *How to Detect and Handle Outliers*, ASQC, 1993.
