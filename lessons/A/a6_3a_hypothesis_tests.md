# A6.3a Hypothesis tests: p-values, permutation, t and chi-square

**Core** · Part A, week 6 (course week 06) · ~1.5 h · Lab:
[`labs/A/a6_3a_hypothesis_tests.py`](../../labs/A/a6_3a_hypothesis_tests.py)

> A6.3 in the outline (hypothesis testing) is taught in two lessons: this one covers single tests, and
> [A6.3b](a6_3b_multiple_comparisons_ab_tests.md) covers multiple comparisons and A/B tests.

## Why it matters

The new checkout flow looks 15 seconds faster in a sample of 160 users. Is that a real improvement, or
the kind of difference you would see between two random groups of users anyway? A **hypothesis test**
answers one precise version of that question: *if there were no real difference, how often would chance
alone produce a gap this large?* You will run tests to compare models, to check that a feature's
distribution has not shifted, to read A/B tests, and to back every claim in P1. Used carelessly, tests
are the most misread numbers in industry: a p-value is not the probability that you are right, and
"not significant" does not mean "no effect".

## Prerequisite check

- Can you explain what a sampling distribution and a standard error are, and why $\bar{x} \pm 1.96\,\mathrm{SE}$ covers
  the mean 95% of the time? If not, revisit [A5.4](a5_4_lln_clt_monte_carlo.md).
- Can you build a t interval and a bootstrap interval for a difference? If not, revisit
  [A6.2](a6_2_confidence_intervals_bootstrap.md).
- Can you compute $\operatorname{Var}[X - Y]$ for independent $X$ and $Y$? If not, revisit
  [A5.1](a5_1_random_variables_expectation.md).
- Can you explain why a positive result from an accurate test can still be likely wrong when the
  condition is rare? If not, revisit [A5.3](a5_3_bayes_theorem.md).

## Intuition

### Innocent until proven guilty

A test starts from a sceptical default, the **null hypothesis** $H_0$: "nothing is going on" (the two flows
have the same mean checkout time; device and plan choice are unrelated). The **alternative hypothesis**
$H_1$ is what you suspect ("they differ"). You then ask how surprising your data would be *if $H_0$ were
true*. The answer is the **p-value**: the probability, under $H_0$, of a result at least as extreme as the
one you observed. A small p-value means "if nothing were going on, data like this would be rare", which
is evidence against $H_0$. You decide in advance on a **significance level** $\alpha$ (usually 0.05) and reject $H_0$
when $p < \alpha$.

It is a court case. $H_0$ is "innocent". The p-value measures how hard the evidence is to explain if the
defendant is innocent. "Not guilty" is not "innocent": failing to reject $H_0$ means the evidence was
not strong enough, not that $H_0$ is true.

### The shuffle test

The cleanest way to see "what chance alone produces" is to make chance happen. If the flow a user got
made no difference, the labels "old" and "new" are arbitrary: any other assignment of the same 160
times to two groups of 80 was equally likely. So shuffle the labels, recompute the difference in means,
and repeat 10,000 times. That gives the **null distribution**, the differences that chance alone
produces. The p-value is the share of shuffles at least as extreme as the real difference. This is the
**permutation test**. It needs no formula and no assumption about the shape of the data, only that,
under $H_0$, the labels are exchangeable (any shuffle is as likely as the real one), which a random
assignment guarantees.

The t-test and the chi-square test are faster shortcuts: they approximate the null distribution with a
formula instead of shuffling.

> **Bridge:** a monitoring alert is a hypothesis test. $H_0$ is "the service is healthy"; the threshold
> (error rate above 2%, latency 3 standard deviations above baseline) is your $\alpha$; a page at 3 a.m. for a
> healthy service is a **type I error** (false positive), and a real outage that never fires is a **type II
> error** (false negative). Lowering the threshold catches more outages and wakes you more often.
> **Where it breaks:** an alert runs every minute forever, so even a tiny $\alpha$ eventually fires on noise;
> a statistical test is designed to be run *once*, on data collected for it. Running it repeatedly is
> exactly the peeking problem of [A6.3b](a6_3b_multiple_comparisons_ab_tests.md).

### Significant is not the same as important

A p-value mixes two things: how big the effect is and how much data you have. With a million users, a
0.01-second speed-up is "significant"; with 20 users, a 30% speed-up may not be. So always report the
**effect size**, the difference in the data's units (seconds, percentage points) with its confidence
interval (A6.2), and let the reader judge whether it matters. A standardized effect size such as
**Cohen's d** (difference over the standard deviation) is useful for comparing studies and for planning.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $H_0$, $H_1$ | null and alternative hypotheses | — |
| $\mathbf{a}$, $\mathbf{b}$ | the samples of groups A and B, sizes $n_A$, $n_B$ | `(n_A,)`, `(n_B,)` |
| $T$ | a test statistic; $t_{\text{obs}}$ its value on the real data | scalar |
| $p$ | p-value | scalar in $[0, 1]$ |
| $\alpha$ | significance level, the type I error rate you accept | scalar |
| $\beta$ | type II error rate; $1 - \beta$ is the **power** | scalar |
| $s_A^2$, $s_B^2$ | sample variances (with $n - 1$) | scalars |
| $\nu$ | degrees of freedom of a t or chi-square distribution | scalar |
| $O_{ij}$, $E_{ij}$ | observed and expected counts in row $i$, column $j$ of a table | `(r, c)` |
| $N$ | total count in a table | int |

**The p-value.** For a two-sided test of a difference,

$$p = P_{H_0}\big(\lvert T\rvert \ge \lvert t_{\text{obs}}\rvert\big),$$

the probability, computed as if $H_0$ were true, of a statistic at least as far from 0 as the observed one. A
**one-sided** test counts only one direction ($T \ge t_{\text{obs}}$); use it only if you decided before seeing the data
that a difference in the other direction would mean the same as no difference.

**Errors and power.**

| | $H_0$ true | $H_0$ false |
|---|---|---|
| reject ($p < \alpha$) | type I error, probability $\alpha$ | correct, probability $1 - \beta$ (power) |
| do not reject | correct | type II error, probability $\beta$ |

If $H_0$ is true and the test is valid, $p$ is uniformly distributed on $[0, 1]$: $P(p < 0.05) = 0.05$, $P(p < 0.5) = 0.5$. That is
what makes $\alpha$ a guaranteed false-positive rate. Power depends on the true effect, the noise and $n$; A6.3b
computes it.

**What a p-value is not.** It is not $P(H_0 \mid \text{data})$; that would need a prior, as Bayes' theorem showed (A5.3).
It is not the size of the effect. And $p \ge 0.05$ is not evidence that the effect is zero.

**Permutation test.** Pool the $n_A + n_B$ values. For $k = 1, \dots, K$: shuffle, call the first $n_A$ "A" and the rest
"B", and compute $T^{(k)} = \bar{b}^{(k)} - \bar{a}^{(k)}$. Then

$$p = \frac{1 + \#\{k : \lvert T^{(k)}\rvert \ge \lvert t_{\text{obs}}\rvert\}}{1 + K}.$$

The $+1$s count the real labelling as one of the permutations, so $p$ is never exactly 0. With few values you
can list every split instead of shuffling (an **exact** test). Any statistic works: means, medians, a
model's AUC.

**Welch's t-test.** The statistic is the difference in means divided by its standard error. Variances of
independent quantities add (A5.1), so

$$t = \frac{\bar{b} - \bar{a}}{\sqrt{\dfrac{s_A^2}{n_A} + \dfrac{s_B^2}{n_B}}}, \qquad \nu = \frac{\Big(\dfrac{s_A^2}{n_A} + \dfrac{s_B^2}{n_B}\Big)^2}{\dfrac{(s_A^2/n_A)^2}{n_A - 1} + \dfrac{(s_B^2/n_B)^2}{n_B - 1}}.$$

Under $H_0$ (equal means), $t$ approximately follows a t distribution with $\nu$ degrees of freedom (the
**Welch–Satterthwaite** approximation; $\nu$ is usually not an integer, and that is fine), and $p = 2P(t_\nu \ge \lvert t\rvert)$, which in code is `2 * stats.t.sf(abs(t), nu)` (`sf`, the "survival
function", is $1 - \text{CDF}$).
The older **Student's t-test** pools the two variances and assumes they are equal; Welch's does not, loses
almost nothing when they are equal, and is the better default. Both assume the *means* are roughly
normal, which the CLT provides for moderate $n$ unless the data is very skewed or heavy-tailed.

**Duality with confidence intervals.** For the same $\alpha$, the Welch test rejects ($p < 0.05$) exactly when the 95%
Welch interval $\bar{b} - \bar{a} \pm t^*_\nu\,\mathrm{SE}$ excludes 0. An interval carries more information: the test only says
whether 0 is plausible; the interval says which values are.

**Chi-square test of independence.** For a table of counts with $r$ rows and $c$ columns (rows: device; columns:
plan), $H_0$ says the row and column variables are independent. Independence means
$P(\text{row } i, \text{col } j) = P(\text{row } i)P(\text{col } j)$ (A5.1), so the expected count in each cell is

$$E_{ij} = \frac{(\text{row } i \text{ total}) \times (\text{column } j \text{ total})}{N},$$

and the statistic adds up the squared gaps, each scaled by what was expected:

$$\chi^2 = \sum_{i=1}^{r}\sum_{j=1}^{c}\frac{(O_{ij} - E_{ij})^2}{E_{ij}}.$$

Under $H_0$ it approximately follows a **chi-square distribution** with $\nu = (r - 1)(c - 1)$ degrees of freedom (once the
row and column totals are fixed, only that many cells are free). The approximation needs expected counts
of about 5 or more in every cell; otherwise use an exact or permutation test. For a 2×2 table, $\chi^2$ equals
the square of the two-proportion $z$ statistic, so they are the same test. SciPy applies **Yates' continuity
correction** to 2×2 tables by default, which makes $\chi^2$ a little smaller (more conservative).

**Effect sizes.**

- Raw: $\bar{b} - \bar{a}$ in the data's units, with its interval. Always report this one.
- **Cohen's d** $= (\bar{b} - \bar{a})/s_p$ with the pooled standard deviation $s_p = \sqrt{\frac{(n_A - 1)s_A^2 + (n_B - 1)s_B^2}{n_A + n_B - 2}}$. Cohen's
  rough labels: 0.2 small, 0.5 medium, 0.8 large (conventions, not laws).
- **Cramér's V** $= \sqrt{\chi^2/(N(\min(r, c) - 1))}$ for tables: 0 for independence, 1 for a perfect association.

**Other tests you will meet.** **Mann–Whitney U** compares two groups by ranks (is a random B value
likely to be larger than a random A value?), robust to outliers and skew. The two-sample
**Kolmogorov–Smirnov test** compares whole distributions using the largest gap between the two
empirical CDFs (A5.2), so it also detects changes in spread or shape; drift monitoring uses it (J2.2).

## Worked example by hand

**1. An exact permutation test.** A $= 3, 4, 5$ and B $= 6, 7, 8$: $\bar{b} - \bar{a} = 7 - 4 = 3$. Under $H_0$ any 3 of the 6 values could have
been "B": $\binom{6}{3} = 20$ equally likely splits. The pooled sum is 33, so a split whose B values sum to $S$ has
difference $S/3 - (33 - S)/3 = (2S - 33)/3$. A difference of at least 3 needs $S \ge 21$, and only $\{6, 7, 8\}$ (sum 21) qualifies:
one-sided $p = 1/20 = 0.05$. Equally extreme in the other direction needs $S \le 12$: only $\{3, 4, 5\}$. Two-sided
$p = 2/20 = 0.10$. Perfect separation of three against three is the most extreme result possible, and it still
cannot reach $p < 0.05$ two-sided: tiny samples cannot give strong evidence.

**2. Welch's t.** A $= 3, 5, 4, 8$ (mean 5, $s_A^2 = 14/3 = 4.667$) and B $= 7, 9, 8, 10, 11$ (mean 9, deviations $-2, 0, -1, 1, 2$,
$s_B^2 = 10/4 = 2.5$). Then $s_A^2/n_A = 1.1667$, $s_B^2/n_B = 0.5$, $\mathrm{SE} = \sqrt{1.6667} = 1.291$, $t = 4/1.291 = 3.098$.
$\nu = 1.6667^2/(1.1667^2/3 + 0.5^2/4) = 2.778/(0.4537 + 0.0625) = 5.38$. From the t distribution with 5.38 degrees of freedom,
$p = 0.024$ (`2 * stats.t.sf(3.098, 5.38)`; by hand, a t table at 5 degrees of freedom puts $t = 3.098$ between the
0.05 and 0.02 two-sided columns). Pooled sd $s_p = \sqrt{(3 \times 4.667 + 4 \times 2.5)/7} = \sqrt{3.429} = 1.852$, so Cohen's $d = 4/1.852 = 2.16$, a very large effect.

**3. A 2×2 chi-square.** Variant A: 30 of 200 users converted; B: 50 of 200. Totals: 80 converted of 400, so
the expected count is $200 \times 80/400 = 40$ converted and 160 not, in each row. $\chi^2 = \frac{(30 - 40)^2}{40} + \frac{(170 - 160)^2}{160} + \frac{(50 - 40)^2}{40} + \frac{(150 - 160)^2}{160}$
$= 2.5 + 0.625 + 2.5 + 0.625 = 6.25$, with $(2 - 1)(2 - 1) = 1$ degree of freedom: $p = 0.012$ (`stats.chi2.sf(6.25, 1)`). The same test as a two-proportion z:
pooled rate 0.2, $z = 0.10/\sqrt{0.2 \times 0.8 \times (1/200 + 1/200)} = 0.10/0.04 = 2.5$, and $2.5^2 = 6.25$, so the p-value is also $2(1 - \Phi(2.5)) = 2 \times 0.0062 = 0.012$. Effect: +10 percentage points
(15% to 25%); Cramér's $V = \sqrt{6.25/400} = 0.125$.

```text
== worked example by hand ==
permutation: observed mean(B) - mean(A) = 3.0   splits = 20
all 20 differences, sorted: [-3.0, -2.3333, -1.6667, -1.6667, -1.0, -1.0, -1.0, -0.3333, -0.3333, -0.3333, 0.3333, 0.3333, 0.3333, 1.0, 1.0, 1.0, 1.6667, 1.6667, 2.3333, 3.0]
one-sided p = #(diff >= 3) / 20 = 0.05   two-sided p = #(|diff| >= 3) / 20 = 0.10
Welch: means 5 and 9   variances 4.6667 and 2.5000
  s_A^2/n_A = 1.1667   s_B^2/n_B = 0.5000   SE = 1.2910   t = 3.0984   df = 5.3812   two-sided p = 0.0244
  Cohen's d = difference / pooled sd = 4 / 1.8516 = 2.1602
2x2 table [[30, 170], [50, 150]]   expected [[40.0, 160.0], [40.0, 160.0]]
  chi-square = 6.2500   df = 1   p = 0.0124   Cramer's V = sqrt(chi2 / n) = 0.1250
  two-proportion z = 0.10 / sqrt(0.2 * 0.8 * (1/200 + 1/200)) = 2.5000   z^2 = 6.2500
```

## Key plot

![Left: a bell-shaped histogram of 10,000 differences in mean checkout time after shuffling the labels, centred on 0 and mostly between minus 12 and plus 12 seconds, with the observed difference of minus 15.9 seconds marked far in the left tail. Right: histograms of p-values from 2,000 simulated experiments; with no effect the histogram is flat across 0 to 1, with a real 13% speed-up it has a tall spike below 0.05 and a long low tail](img/a6_3a_null_and_pvalues.png)

Left: the permutation test. Shuffling labels shows what chance produces; the real difference sits
beyond almost all of it ($p = 0.0009$). Right: when $H_0$ is true, p-values are spread evenly over $[0, 1]$, so 5% fall below
0.05 by construction. With a real effect they pile up near 0, but far from all of them: here only 40%
of experiments with a real effect reach $p < 0.05$.

## From scratch in NumPy

**Three tests.** The permutation test shuffles all $K$ copies at once: sorting a `(K, n)` array of random keys
along each row gives $K$ independent random orders.

```python
def perm_test_diff_means(a: FloatArray, b: FloatArray, n_perm: int, gen: np.random.Generator) -> tuple[float, FloatArray]:
    """Two-sided permutation test of mean(b) - mean(a). a (n_a,), b (n_b,) -> (p, null (n_perm,))."""
    pooled = np.concatenate([a, b])  # (n_a + n_b,)
    keys = gen.random((n_perm, len(pooled)))  # (n_perm, n) random sort keys
    perm = pooled[np.argsort(keys, axis=1)]  # (n_perm, n) each row a shuffled copy
    null = perm[:, len(a):].mean(axis=1) - perm[:, : len(a)].mean(axis=1)  # (n_perm,)
    observed = b.mean() - a.mean()
    # +1 in numerator and denominator: the observed split is one of the permutations, so p is never 0
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    return float(p), null
```

`welch_t` and `chi2_independence` are the formulas above, line for line. The checkout data is 80 simulated
times per flow (log-normal, true medians 60 s and 52 s, so the true speed-up is 13%):

```text
== from scratch: three tests ==
checkout: mean(new) - mean(old) = -15.8651 s
permutation test: p = 0.0009   (10000 shuffles; null sd = 4.9168)
Welch t-test:     t = -3.3176   df = 143.83   p = 0.0012
effect size: -15.87 s (-22.2% of the old mean)   Cohen's d = -0.5246
plans x device: chi-square = 10.0433   df = 2   p = 0.006594
expected counts under independence: [[105.0, 67.5, 27.5], [105.0, 67.5, 27.5]]
row shares (free, pro, team): [[0.6, 0.3, 0.1], [0.45, 0.375, 0.175]]
Cramer's V = sqrt(chi2 / (n * (min(r, c) - 1))) = 0.1585
```

The permutation test and Welch agree ($p \approx 0.001$), as they should with 80 per group. The observed speed-up
is 22%, larger than the true 13%: keep that in mind for the next block. In the plan table, mobile users
pick paid plans more often (55% against 40%); the test says this is unlikely to be chance, and the row
shares say what the difference actually is. Report both.

**What p-values look like.** The lab repeats the checkout experiment 2,000 times with no real effect and
2,000 times with the 13% speed-up, and runs Welch's test on each:

```text
== from scratch: p-values under the null and under an effect ==
H0 true, 2000 experiments: share with p < 0.05 = 0.0495   (the type I error rate; should be ~0.05)
H0 true: share of p-values in each tenth of [0, 1]: [0.098, 0.1075, 0.098, 0.108, 0.109, 0.0975, 0.095, 0.0955, 0.094, 0.0975]
real effect: share with p < 0.05 = 0.4005   (the power at n = 80 per group)
real effect: share with p < 0.001 = 0.0480   median p = 0.0869
in 200 experiments, 'p < 0.05' and '95% CI excludes 0' agree every time: True
```

Under $H_0$, each tenth of $[0, 1]$ gets about 10% of the p-values: the test delivers its promised 5% false-positive
rate even on skewed data, thanks to the CLT at $n = 80$. With a real 13% speed-up, only 40% of experiments
reach significance: this experiment is **underpowered**. Two consequences. First, a non-significant
result here is weak evidence of "no effect": 60% of the time a real effect is missed. Second, the
experiments that *do* reach significance are the ones where chance made the effect look bigger, which is
why our one sample showed 22% instead of 13%. Significant results from underpowered studies overestimate
effects (sometimes called the winner's curse). A6.3b shows how to choose $n$ to avoid this. The last line
checks the test–interval duality numerically.

## With SciPy

```text
== with scipy ==
ttest_ind(equal_var=False): t = -3.3176   p = 0.0012   matches welch_t: OK
ttest_ind default (Student, equal variances): p = 0.0011
permutation_test: p = 0.0012   mine 0.0009   agree within Monte Carlo error: OK
chi2_contingency(plans): chi2 = 10.0433   p = 0.006594   dof = 2   matches: OK
2x2 conversion table: correction=False chi2 = 6.2500 (p = 0.0124)   default (Yates) chi2 = 5.6406 (p = 0.0175)
ks_2samp: D = 0.2000   p = 0.0814   mannwhitneyu: U = 2368.0   p = 0.0045
```

- `stats.ttest_ind(b, a, equal_var=False)` is Welch's test. Its default, `equal_var=True`, is Student's
  test; pass `equal_var=False` explicitly. It takes `axis=`, so the lab runs 2,000 tests in one call.
- `stats.permutation_test((a, b), statistic, vectorized=True, n_resamples=K, rng=...)` shuffles with
  its own random numbers, so its p-value agrees with ours up to Monte Carlo error (about 0.0003 here).
- `stats.chi2_contingency(table)` returns the statistic, p-value, degrees of freedom and expected counts.
  For a 2×2 table it applies Yates' correction unless you pass `correction=False` (6.25 becomes 5.64).
  VERIFY (as of 2026-10), SciPy 1.18.1.
- `stats.mannwhitneyu` ($p = 0.005$) agrees with Welch; `stats.ks_2samp` ($p = 0.08$) does not reach 0.05. KS tests
  for *any* difference in distribution, which spreads its power thin; for a shift in the typical value, a
  t-test or Mann–Whitney is more powerful. Different tests answer different questions: pick the
  question first.

## In practice

- **When to use it:** to decide whether a difference you observed is distinguishable from noise: two
  models' error rates on the same test set, a metric before and after a change, a feature's distribution
  in two segments, an A/B test. Defaults: Welch for means, a permutation test whenever you doubt the
  assumptions or need a custom statistic (median, AUC, a ratio), chi-square (or Fisher's exact test for
  small counts) for categorical outcomes, Mann–Whitney for "tends to be larger" on skewed data.
- **Failure modes:**
  - *Reading $p$ as $P(H_0 \mid \text{data})$.* "$p = 0.03$, so 97% chance it works" is wrong; how likely an effect is
    depends on how plausible it was before (A5.3). You notice the error in the wording of a report.
  - *Reading "not significant" as "no effect".* With 40% power, most real effects are missed. Report the
    interval: "between −2.5 and +7.7 s" says what was ruled out and what was not.
  - *Significant but trivial.* With huge $n$, tiny effects get tiny p-values. You notice by reading the
    effect size before the p-value.
  - *Non-independent observations.* Ten sessions per user analysed as ten independent rows shrink the
    standard error and inflate significance. Test per user (aggregate first) or permute users, not rows.
  - *Choosing the test, the metric or one-sided versus two-sided after seeing the data.* This inflates false
    positives the way multiple testing does (A6.3b). Pre-register (write it down first), as P1 requires.
  - *Chi-square with tiny expected counts.* You notice cells with $E_{ij} < 5$; merge categories or use
    `stats.fisher_exact` or a permutation test.
- **Defaults that work:** two-sided tests at $\alpha = 0.05$ decided in advance; `equal_var=False`; report the effect
  in units with a 95% interval, the test, $n$ per group and the p-value (to 2 significant figures, "$p < 0.001$"
  below that); a permutation test as a check on any parametric result you plan to act on.

## Exercises

Save your work to `my-work/A/a6_3a_hypothesis_tests.md` (or `.py`) and ask for a review.

1. **By hand:** (a) Groups A $= 2, 4, 6$ and B $= 5, 7, 9$. List the splits needed for an exact permutation test of
   the difference in means, and give the one-sided and two-sided p-values. (b) Group A has $n_A = 4$, mean 10,
   variance 4; group B has $n_B = 9$, mean 13, variance 9. Compute Welch's $t$ and $\nu$. The two-sided p-value is
   0.064: what do you conclude at $\alpha = 0.05$, and what would you tell the team? (c) Crash reports: Android
   had 45 crashes in 300 sessions, iOS 25 in 300. Compute the expected counts, $\chi^2$ and the degrees of freedom
   ($p = 0.011$), and the difference in crash rates. (d) A result is "difference 2.1 s, $p = 0.03$". For each
   statement, say true or false and why: (i) there is a 3% chance the null hypothesis is true; (ii) if there
   were no difference, a gap of 2.1 s or more would occur in about 3% of experiments; (iii) there is a 97%
   chance the result will replicate; (iv) the effect is large.
2. **Code:** Write a general `perm_test(a, b, stat, n_perm, gen)` that takes any statistic of two samples.
   Then measure, by simulation with `SEED = 0`, the type I error rate and the power of three tests on small,
   skewed samples: Welch's t-test, your permutation test on the difference in means, and Mann–Whitney. Use
   10 values per group from a log-normal with $\sigma = 1$; for power, multiply group B by 1.8 (its median rises 80%).
   Use at least 1,000 simulated experiments (2,000 permutations each), report each rate with its Monte Carlo
   standard error, and say which test you would use for this kind of data and why.
3. **Think:** A colleague compares a new recommendation model with the current one offline on the same
   10,000 users and reports "NDCG improved from 0.412 to 0.415, $p = 0.0001$, ship it". Another colleague
   compared two prompts on 30 questions and reports "no significant difference, $p = 0.4$, so the prompts are
   equivalent". Write a short review of each claim: what the test can and cannot support, what is missing,
   and what you would ask for.

## Takeaways

1. A p-value is the probability, if $H_0$ were true, of data at least as extreme as yours; it is not the
   probability that $H_0$ is true, and it says nothing about the size of the effect.
2. A permutation test builds the null distribution by shuffling labels; it needs only exchangeability
   and works for any statistic. t and chi-square tests are formula shortcuts for the same idea.
3. Welch's t-test (`equal_var=False`) is the default for comparing means; the chi-square test compares
   observed and expected counts, $E_{ij} = \text{row}_i \times \text{col}_j / N$, with $(r - 1)(c - 1)$ degrees of freedom.
4. Under $H_0$ p-values are uniform, so $\alpha$ is the false-positive rate; power is the chance of detecting a real
   effect, and underpowered tests both miss effects and exaggerate the ones they find.
5. Always report the effect size in units with a confidence interval next to the p-value; $p < 0.05$ exactly
   when the 95% interval excludes 0.

**What builds on this:** [A6.3b](a6_3b_multiple_comparisons_ab_tests.md) (many tests, A/B tests, power and
sample size); [A6.5](a6_5_bayesian_inference.md) (the Bayesian alternative to p-values);
[B4.3](../B/b4_3_cross_validation_tuning.md) (comparing models fairly);
[J2.2](../J/j2_2_monitoring_drift.md) (KS and chi-square tests for drift); P1 (the pre-registered tests).

**References:** B. L. Welch, "The Generalization of 'Student's' Problem when Several Different Population
Variances are Involved", *Biometrika* 34(1/2), 1947. K. Pearson, "On the criterion that a given system of
deviations from the probable in the case of a correlated system of variables is such that it can be reasonably
supposed to have arisen from random sampling", *Philosophical Magazine* 50(302), 1900. R. L. Wasserstein and N. A. Lazar,
"The ASA Statement on p-Values: Context, Process, and Purpose", *The American Statistician* 70(2), 2016.
J. Cohen, *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed., 1988.
