# A6.3b Multiple comparisons and A/B tests

**Core** · Part A, week 6 (course week 06) · ~1 h · Lab:
[`labs/A/a6_3b_multiple_comparisons_ab_tests.py`](../../labs/A/a6_3b_multiple_comparisons_ab_tests.py)

> A6.3 in the outline (hypothesis testing) is taught in two lessons: [A6.3a](a6_3a_hypothesis_tests.md)
> covers single tests, and this one covers what goes wrong when you run many of them, and how to run an
> A/B test that gives a trustworthy answer.

## Why it matters

One test at $\alpha = 0.05$ has a 5% false-positive rate. Nobody runs one test. A dashboard compares 12 metrics
across 6 segments; an analyst tries three versions of a feature and two outlier rules; an A/B test is
checked every morning until it "wins". Each extra look is another chance for noise to cross the line,
and with enough chances it is almost certain to. This is how teams ship features that do nothing,
publish "insights" that do not replicate, and lose trust in their experiment platform. The fixes are
cheap: correct for multiple tests, decide the analysis before the data arrives, size the experiment for
the effect you care about, and stop it when you said you would.

## Prerequisite check

- Can you say what a p-value, $\alpha$, a type I error and power are? If not, revisit
  [A6.3a](a6_3a_hypothesis_tests.md).
- Can you run a chi-square test and a two-proportion z-test? If not, revisit
  [A6.3a](a6_3a_hypothesis_tests.md).
- Can you compute $P(\text{at least one}) = 1 - P(\text{none})$ for independent events? If not, revisit
  [A5.1](a5_1_random_variables_expectation.md).
- Can you use `stats.norm.ppf` to get the 1.96 for 95% and the 0.84 for 80%? If not, revisit
  [A5.2](a5_2_common_distributions.md).
- Do you know why randomizing who gets a treatment rules out confounders? If not, revisit
  [A6.1](a6_1_descriptive_statistics_eda.md).

## Intuition

### Many tests: buying lottery tickets

Each test on pure noise is a lottery ticket that wins (a false positive) with probability 0.05. Buy 20
tickets and the chance that at least one wins is $1 - 0.95^{20} = 64\%$. Nothing about any single test is wrong; the
error is in reporting the winner without saying how many tickets you bought. The same happens without
anyone running 20 formal tests: trying several metrics, subgroups, outlier rules or test variants and
reporting the one that "worked" is the same lottery, sometimes called the **garden of forking paths**
or **p-hacking**. Writing down the analysis before seeing the data (**pre-registration**, which P1
requires) is the defence.

Two kinds of correction, for two kinds of cost:

- **Family-wise error rate (FWER)** control: keep the chance of *even one* false positive at $\alpha$. **Bonferroni**
  tests each of $m$ hypotheses at $\alpha/m$; **Holm** does the same job and never rejects less. Use it when one false
  claim is costly (a release decision, a safety claim).
- **False discovery rate (FDR)** control: allow some false positives but keep their *share* among your
  discoveries at about $\alpha$. **Benjamini–Hochberg (BH)** does this and finds more real effects. Use it for screening
  many candidates (which of 500 features, genes or segments deserve a closer look), when a few false leads
  are acceptable.

### A/B tests: the experiment that makes causation possible

A6.1 said only randomized assignment rules out unmeasured confounders. An **A/B test** does exactly that:
each user is randomly assigned to the current version (control, A) or the new one (treatment, B), so the
two groups differ, on average, only in the version they saw. A difference in the outcome beyond what
chance explains is then caused by the change. The statistics are the tests from A6.3a; what makes an A/B
test trustworthy is the design around them:

1. **One primary metric** chosen in advance (the decision metric), plus a few **guardrail metrics** that
   must not get worse (latency, errors, unsubscribes).
2. A **sample size** computed in advance from the smallest effect worth detecting, the **minimum detectable
   effect (MDE)**, and the power you want (usually 80%).
3. A fixed **duration**, in whole weeks so every weekday is represented, and one analysis at the end.
4. Health checks before reading the result, above all the **sample ratio mismatch (SRM)** check: if you
   split 50/50 and got 50,600 against 49,400 users, the assignment or logging is broken, and the result
   cannot be trusted whatever its p-value.

### Peeking

The most common A/B mistake is to check the p-value every day and stop as soon as it dips below 0.05.
Under $H_0$, the daily p-value wanders up and down like a random walk; a single look at a fixed day falls
below 0.05 5% of the time, but the chance that the walk *ever* dips below 0.05 in 30 looks is much larger.
The lab measures it: 29%. And most of those "wins" are temporary: only 16% are still below 0.05 at day 30.

> **Bridge:** a canary deployment is an A/B test: route 5% of traffic to the new build and compare error
> rates with the rest. Deciding "the canary is fine" after five minutes because nothing has crossed a
> threshold yet, or rolling back the instant one metric blips, is peeking, in both directions. **Where it
> breaks:** a canary is a safety check, not a measurement; it is allowed to peek and stop early for
> *harm*, because rolling back is cheap and a bad build is expensive. What it cannot do is *prove* a
> small improvement: for that you need the sample size and the fixed horizon below.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $m$ | number of hypotheses tested | int |
| $p_1, \dots, p_m$ | their p-values; $p_{(1)} \le \dots \le p_{(m)}$ sorted | `(m,)` |
| $\alpha$ | target error rate (0.05) | scalar |
| $V$, $R$ | number of false rejections, number of all rejections | ints |
| $p_A$, $p_B$ | true conversion rates of control and treatment | scalars |
| $\delta$ | true difference $p_B - p_A$; the MDE when it is the smallest you care about | scalar |
| $n$ | users per arm | int |
| $z_q$ | the $q$-quantile of $\mathcal{N}(0, 1)$: $z_{0.975} = 1.96$, $z_{0.80} = 0.8416$ | scalar |
| $1 - \beta$ | power | scalar |

**Family-wise error rate.** $\mathrm{FWER} = P(V \ge 1)$. With $m$ independent tests of true nulls, each avoids a false
positive with probability $1 - \alpha$, so

$$\mathrm{FWER} = 1 - (1 - \alpha)^m, \qquad \mathbb{E}[V] = m\alpha.$$

For $m = 20$: 0.64, and one false positive on average.

**Bonferroni.** Reject $H_i$ when $p_i \le \alpha/m$. Why it works, for any dependence between the tests: the probability
that at least one of several events happens is at most the sum of their probabilities (the **union
bound**: overlaps are counted twice in the sum, never less). So $P(V \ge 1) \le \sum_{i \in \text{nulls}} P(p_i \le \alpha/m) \le m \cdot \alpha/m = \alpha$.

**Holm (step-down).** Sort the p-values. Compare $p_{(1)}$ with $\alpha/m$, $p_{(2)}$ with $\alpha/(m - 1)$, …, $p_{(k)}$ with $\alpha/(m - k + 1)$.
Reject in order and stop at the first p-value above its threshold; everything from there on is kept. The
first comparison is Bonferroni's; after that the thresholds relax, because once a hypothesis has been
rejected, fewer remain that could be false positives. Holm controls the FWER at $\alpha$ like Bonferroni and
rejects at least as much, so there is no reason to prefer plain Bonferroni except simplicity.

**Benjamini–Hochberg (step-up).** The **false discovery proportion** is $V/R$ (0 when $R = 0$), and the FDR is its
expectation. Sort the p-values and find the *largest* $k$ with $p_{(k)} \le \frac{k}{m}\alpha$; reject the $k$ smallest. For independent
(or positively dependent) tests this keeps $\mathrm{FDR} \le \alpha$ (stated, not proved). The thresholds grow linearly
($\alpha/m, 2\alpha/m, \dots, \alpha$), so BH rejects more than Holm when many effects are real.

**Sample size for a difference in proportions.** The estimated difference $\hat{\delta} = \hat{p}_B - \hat{p}_A$ has standard error
$\mathrm{SE} = \sqrt{(p_A(1 - p_A) + p_B(1 - p_B))/n}$ (variances of independent estimates add, A5.1; each $\hat{p}$ has variance $p(1 - p)/n$,
A5.2). The test rejects when $\hat{\delta}/\mathrm{SE} > z_{1-\alpha/2}$ (taking the direction of a real lift; the other tail adds almost
nothing). If the true difference is $\delta$, then $\hat{\delta} \approx \mathcal{N}(\delta, \mathrm{SE}^2)$ by the CLT, so $\hat{\delta}/\mathrm{SE}$ is approximately normal with
mean $\delta/\mathrm{SE}$ and standard deviation 1, and

$$\text{power} = P\Big(\frac{\hat{\delta}}{\mathrm{SE}} > z_{1-\alpha/2}\Big) \approx \Phi\Big(\frac{\delta}{\mathrm{SE}} - z_{1-\alpha/2}\Big).$$

Setting this to $1 - \beta$ means $\frac{\delta}{\mathrm{SE}} - z_{1-\alpha/2} = z_{1-\beta}$, that is $\delta = (z_{1-\alpha/2} + z_{1-\beta})\,\mathrm{SE}$. Substitute the SE and solve for $n$:

$$n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2\,\big(p_A(1 - p_A) + p_B(1 - p_B)\big)}{\delta^2}.$$

The $\delta^2$ in the denominator is the lesson: halving the effect you want to detect needs **four times** the
users. Read the other way, the **MDE** for a given $n$ is $(z_{1-\alpha/2} + z_{1-\beta})\sqrt{(p_A(1 - p_A) + p_B(1 - p_B))/n}$.

**Sample ratio mismatch.** With planned shares $\pi_A, \pi_B$ and $N$ users in total, the expected counts are $N\pi_A$ and $N\pi_B$, and
$\chi^2 = \sum (O - E)^2/E$ with 1 degree of freedom (a chi-square goodness-of-fit test). Experiment platforms flag
SRM at a strict threshold such as $p < 0.001$, because the check runs on every experiment.

**Peeking, and what to do instead.** If you look $K$ times and stop at the first $p < \alpha$, the false positive rate
is $P(\min_k p_k < \alpha)$, which grows with $K$ (and approaches 1 as $K \to \infty$). Options:

- One look, at the planned sample size (the **fixed-horizon** test). Simple and correct.
- A stricter threshold at every look, chosen so that the chance of *ever* crossing it under $H_0$ is $\alpha$. The lab
  finds it by simulation (0.0068 for 30 daily looks). **Group sequential designs** (Pocock 1977; O'Brien and
  Fleming 1979) give such boundaries in closed form, and "always-valid" sequential tests let you look at
  any time; experimentation platforms implement variants of both (VERIFY (as of 2026-10) which method your
  platform uses).

## Worked example by hand

**1. Twenty tests.** $1 - 0.95^{20} = 1 - 0.3585 = 0.6415$. Bonferroni threshold $0.05/20 = 0.0025$.

**2. Three corrections on five p-values.** Sorted: $0.001, 0.008, 0.012, 0.030, 0.200$, $m = 5$, $\alpha = 0.05$.

| $k$ | $p_{(k)}$ | Bonferroni $\alpha/m$ | Holm $\alpha/(m - k + 1)$ | BH $k\alpha/m$ |
|---|---|---|---|---|
| 1 | 0.001 | 0.01 ✓ | 0.0100 ✓ | 0.01 ✓ |
| 2 | 0.008 | 0.01 ✓ | 0.0125 ✓ | 0.02 ✓ |
| 3 | 0.012 | 0.01 ✗ | 0.0167 ✓ | 0.03 ✓ |
| 4 | 0.030 | 0.01 ✗ | 0.0250 ✗ (stop) | 0.04 ✓ |
| 5 | 0.200 | 0.01 ✗ | 0.05 (not reached) | 0.05 ✗ |

Bonferroni rejects 2, Holm 3 (it stops at $k = 4$), BH 4 (the largest $k$ that passes is 4, so the 4 smallest are
rejected; for BH a failure in the middle would not matter, only the largest passing $k$).

**3. Sample size.** Baseline conversion $p_A = 10\%$; the smallest lift worth shipping is 1 point ($p_B = 11\%$); $\alpha = 0.05$
two-sided, power 80%. $(1.96 + 0.8416)^2 = 7.849$; $0.10 \times 0.90 + 0.11 \times 0.89 = 0.09 + 0.0979 = 0.1879$; $n = 7.849 \times 0.1879/0.01^2 = 14{,}748.05$, rounded
up to 14,749 users per arm (the printout below shows the unrounded value). At 1,000 users per arm per day that is 15 days, so plan three full weeks (21 days). To detect half
the lift (0.5 points): 57,760 per arm, about four times as many.

**4. Sample ratio check.** 50,600 and 49,400 users against 50,000 each: $\chi^2 = \frac{600^2}{50{,}000} + \frac{600^2}{50{,}000} = 7.2 + 7.2 = 14.4$, 1 degree of
freedom, $p = 0.00015$ (`stats.chi2.sf(14.4, 1)`; equivalently $z = \sqrt{14.4} = 3.79$ standard errors from the planned split). A 1.2% imbalance looks harmless and is not: with 100,000 users it is far beyond chance.
Something in assignment or logging (a redirect that drops some B users, a bot filter that treats the
arms differently) is broken. Find it before reading any metric.

```text
== worked example by hand ==
P(at least one false positive in 20 independent tests) = 1 - 0.95^20 = 0.6415   expected false positives = 20 * 0.05 = 1.0
Bonferroni threshold = 0.05 / 20 = 0.0025
sorted p = [0.001, 0.008, 0.012, 0.03, 0.2]
Bonferroni: p <= 0.0100? [True, True, False, False, False]
Holm thresholds [0.01, 0.0125, 0.0167, 0.025, 0.05]   p <= threshold? [True, True, True, False, False]  (stop at the first False)
BH thresholds   [0.01, 0.02, 0.03, 0.04, 0.05]   p <= threshold? [True, True, True, True, False]  (largest True is k = 4)
sample size: z_0.975 = 1.9600   z_0.80 = 0.8416   variance sum = 0.1879   n per arm = 14,748
  halve the detectable lift to 0.5 points: n per arm = 57,760
sample ratio check: observed [50600, 49400] vs expected [50000, 50000]   chi-square = 14.40   p = 0.00015
```

## Key plot

![Left: the chance of at least one false positive rises steeply with the number of tests, passing 0.64 at 20 tests and approaching 1 by 100, while the Bonferroni curve stays flat just under 0.05; two diamonds mark the simulated values 0.66 and 0.05 at 20 tests. Right: over a 30-day A/A test, the share declared significant when peeking daily climbs from 0.05 on day 1 to 0.29 on day 30, while a single look on any given day stays at about 0.05](img/a6_3b_multiple_tests_peeking.png)

Two ways to buy extra lottery tickets: more tests (left) and more looks at one test (right). In both,
each individual check is a valid 5% test; the error is in stopping at the first success.

## From scratch in NumPy

**The corrections.** Holm and BH both sort, compare against a vector of thresholds and map the decision
back to the original order:

```python
def holm(p: FloatArray, alpha: float = ALPHA) -> npt.NDArray[np.bool_]:
    """Holm step-down: compare sorted p to alpha/(m - k), stop at the first failure. p (m,) -> (m,) bool."""
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= alpha / (m - np.arange(m))  # (m,) in sorted order
    n_reject = m if passed.all() else int(np.argmin(passed))  # index of the first failure
    reject = np.zeros(m, dtype=bool)
    reject[order[:n_reject]] = True
    return reject


def benjamini_hochberg(p: FloatArray, alpha: float = ALPHA) -> npt.NDArray[np.bool_]:
    """BH step-up: reject the k smallest, k = largest with p_(k) <= k alpha / m. p (m,) -> (m,) bool."""
    m = len(p)
    order = np.argsort(p)
    below = np.nonzero(p[order] <= alpha * np.arange(1, m + 1) / m)[0]
    reject = np.zeros(m, dtype=bool)
    if len(below):
        reject[order[: below[-1] + 1]] = True
    return reject
```

The code uses 0-based $k$, so Holm's threshold $\alpha/(m - k + 1)$ becomes `alpha / (m - np.arange(m))`. `np.argmin` on a boolean
array returns the first `False`. The lab runs 2,000 families of 20 Welch tests (50 values per group), first
with every null true, then with 5 real effects of $d = 0.8$:

```text
== from scratch: multiple comparisons ==
one family of 20 null tests, smallest p-values: [0.029, 0.1077, 0.1863, 0.1883]
significant at 0.05 uncorrected: 1   Bonferroni: 0   Holm: 0   BH: 0
all 20 null, 2000 families:
  method      P(any false +)  false + / family  true + / family    FDR
  none                0.6595            1.0060           0.0000 0.6595
  Bonferroni          0.0500            0.0530           0.0000 0.0500
  Holm                0.0500            0.0535           0.0000 0.0500
  BH                  0.0505            0.0565           0.0000 0.0505
5 real effects (d = 0.8), 2000 families:
  method      P(any false +)  false + / family  true + / family    FDR
  none                0.5080            0.7215           4.8855 0.1108
  Bonferroni          0.0340            0.0345           4.0360 0.0068
  Holm                0.0410            0.0425           4.1180 0.0081
  BH                  0.1740            0.2050           4.6235 0.0341
formula 1 - 0.95^m: {1: 0.05, 5: 0.2262, 10: 0.4013, 20: 0.6415, 50: 0.9231, 100: 0.9941}
```

The first family is the lottery in miniature: one test at $p = 0.029$, a "finding" that every correction
removes. Over 2,000 families with all nulls true, no correction gives a false positive in 66% of families
(the formula says 64%; the difference is simulation noise of about 0.01), and all three corrections hold it at
5%. With 5 real effects, the trade-off appears: Bonferroni and Holm keep false alarms rare and find about 4
of the 5 real effects; BH finds 4.6 of them, accepting a false positive in 17% of families, while the share of
false discoveries among its findings stays at 3.4%, under its 5% target. Holm finds slightly more than
Bonferroni with the same guarantee.

**Peeking.** Each of 2,000 A/A tests gets 1,000 users per arm per day at a 10% conversion rate in both arms; the
lab computes the p-value of the cumulative data every day:

```text
== from scratch: peeking ==
2000 A/A tests (no real difference), 30 days:
  look once, at day 30: false positive rate = 0.0475
  peek daily, stop at first p < 0.05, by day  1: false positive rate = 0.0455
  peek daily, stop at first p < 0.05, by day  5: false positive rate = 0.1490
  peek daily, stop at first p < 0.05, by day 10: false positive rate = 0.2010
  peek daily, stop at first p < 0.05, by day 20: false positive rate = 0.2580
  peek daily, stop at first p < 0.05, by day 30: false positive rate = 0.2925
  of the tests that ever crossed 0.05, still below 0.05 at day 30: 0.1624
  median day of the first 'significant' peek: 5
  calibrated per-peek threshold (from separate A/A runs) = 0.0068; with it the false positive rate here = 0.0475
```

A single look is a valid 5% test on any day. Peeking daily and stopping at the first $p < 0.05$ declares a winner in
29% of experiments where nothing changed, most of them early (median day 5), when the data is thinnest.
Requiring $p < 0.0068$ at every look brings the rate back to 5%. The price is power: a real effect must be
stronger to cross the stricter line.

**Power by simulation.** The formula's 14,749 users per arm, checked by running 2,000 simulated experiments:

```text
== from scratch: power and sample size ==
10% -> 11% with  3,687 users per arm (a quarter of it): simulated power = 0.2820
10% -> 11% with 14,749 users per arm (the computed n): simulated power = 0.8065
minimum detectable effect with 2,000 users per arm (power 0.8): 0.0266 (absolute, around a 10% base rate)
```

The formula delivers 81% power, as designed. A quarter of the users gives 28%: an experiment that will usually
report "no significant difference" for a lift that is really there. With only 2,000 users per arm, nothing
smaller than a 2.7-point lift (27% relative) can be detected reliably: run that test and "no effect" means very little.

## With statsmodels

```text
== with statsmodels ==
Bonferroni  rejects [True, True, False, False, False]   multipletests agrees: OK
Holm        rejects [True, True, True, False, False]   multipletests agrees: OK
BH          rejects [True, True, True, True, False]   multipletests agrees: OK
Holm and BH agree with multipletests on 200 random p-value vectors: OK
Holm-adjusted p-values (compare each with 0.05): [0.005, 0.032, 0.036, 0.06, 0.2]
proportions_ztest matches the from-scratch two-proportion z-test (p = 0.2486): OK
statsmodels NormalIndPower with Cohen's h = 0.0326: n per arm = 14,744   formula above: 14,749
```

- `statsmodels.stats.multitest.multipletests(p, alpha=0.05, method="holm")` (also `"bonferroni"`,
  `"fdr_bh"`) returns the decisions and **adjusted p-values**: each adjusted value can be compared directly
  with $\alpha$, which is easier to report than per-test thresholds. SciPy has `stats.false_discovery_control(p)`
  for BH-adjusted p-values. VERIFY (as of 2026-10), statsmodels 0.15.0.
- `proportions_ztest(counts, nobs)` is the two-proportion z-test; it matches ours exactly.
- `NormalIndPower().solve_power(effect_size=h, alpha=0.05, power=0.8)` uses Cohen's $h$, an effect size for
  proportions based on an arcsine transform. It gives 14,744 against our 14,749: two approximations that
  agree to 0.03%. Online calculators use one of these two formulas.

## In practice

- **When to use it:** every time you run more than one test (metrics, segments, variants, models), and for
  every A/B test. Write an analysis plan first: primary metric, guardrails, MDE, sample size, duration,
  test, correction, and what decision each outcome leads to.
- **Failure modes:**
  - *Many metrics, one winner.* "Revenue is flat but time on page is up, $p = 0.04$" from a dashboard of 20
    metrics. You notice by counting the metrics. Decide on one primary metric; correct the rest (Holm) or
    treat them as exploratory.
  - *Segment fishing.* "It works for Android users in Brazil." With 6 segments and 2 platforms you ran 12
    tests. Pre-register the segments you care about, correct, and re-test any surprise in a new experiment.
  - *Peeking.* You notice "winners" that fade over the following days, and a team that checks the result
    daily. Fix the horizon, or use a sequential method with proper boundaries.
  - *Underpowered tests.* You notice an experiment ended "not significant" with an interval that includes
    effects you would care about. Compute the sample size from the MDE before starting.
  - *Sample ratio mismatch.* You notice it only if you check: test the split on every experiment.
  - *Unit mismatch.* Randomizing users but analysing page views treats correlated views as independent and
    makes intervals too narrow (A6.3a). Analyse per randomization unit.
  - *Novelty and learning effects.* Users click on anything new in week 1. You notice a treatment effect
    that shrinks day by day; run for full weeks and plot the effect over time.
  - *Interference.* In marketplaces and social products, treated users affect control users (shared
    inventory, messages). You notice impossible results, such as both arms improving; randomize by region
    or cluster instead.
- **Defaults that work:** $\alpha = 0.05$ two-sided, power 0.8; Holm for a handful of confirmatory tests, BH for
  screening; SRM check at $p < 0.001$; durations in whole weeks; report the effect with its 95% interval and the
  sample size actually reached.

## Exercises

Save your work to `my-work/A/a6_3b_multiple_comparisons_ab_tests.md` (or `.py`) and ask for a review.

1. **By hand:** (a) An analyst tests 8 metrics at 0.05. What is the FWER if all nulls are true and the tests are
   independent? The p-values are $0.004, 0.011, 0.019, 0.04, 0.07, 0.2, 0.5, 0.8$. Which metrics are significant
   under Bonferroni, Holm and BH? (b) A sign-up page converts at 5%. How many users per arm do you need to detect
   a lift to 5.5% with 80% power at $\alpha = 0.05$ two-sided? The page gets 4,000 visitors a day, split equally. How
   many days would you plan, and why? (c) A 50/50 test reports 10,200 users in A and 9,800 in B. Run the SRM
   check. What do you do next? (d) Explain in two or three sentences why an experiment that stops as soon as
   $p < 0.05$ will, on average, overstate the size of the effect it reports, even when the effect is real.
2. **Code:** Simulate a 28-day A/B test with 1,000 users per arm per day and a 10% baseline conversion rate,
   2,000 runs per scenario, `SEED = 0`. Compare three stopping rules: (i) fixed horizon (one test on day 28);
   (ii) naive peeking (stop at the first daily $p < 0.05$); (iii) peeking with a per-look threshold that you
   calibrate on A/A simulations so that the false positive rate is 5%. For each rule, report (a) the false
   positive rate when both arms convert at 10%, (b) the power when B converts at 11%, and (c) among runs that
   declared B the winner, the average estimated lift, compared with the true 1 point. Which rule would you put
   in your team's experiment guidelines?
3. **Think:** A product manager plans an A/B test of a new onboarding flow: "We'll track 12 metrics across 6 user
   segments, check the dashboard every morning, and ship as soon as anything is significantly better." Write
   the experiment plan you would propose instead (half a page): primary metric and guardrails, how you choose
   the sample size and duration, the analysis and correction rules, and what happens if the primary metric is
   flat but a secondary metric wins.

## Takeaways

1. With $m$ tests at $\alpha$, the chance of at least one false positive is $1 - (1 - \alpha)^m$: 64% for 20 tests. Count the tests,
   including the informal ones (metrics, segments, variants, analysis choices).
2. Holm controls the family-wise error rate like Bonferroni and rejects at least as much; Benjamini–Hochberg
   controls the false discovery rate and finds more real effects. Choose by the cost of one false claim.
3. An A/B test makes causal claims possible by randomizing; trust comes from the design: one primary metric,
   guardrails, a pre-computed sample size, a fixed duration in whole weeks, and an SRM check.
4. Sample size per arm is $(z_{1-\alpha/2} + z_{1-\beta})^2(p_A(1 - p_A) + p_B(1 - p_B))/\delta^2$: half the detectable effect costs four times the users.
5. Peeking and stopping at the first $p < 0.05$ turned a 5% error rate into 29% over 30 daily looks; use a fixed horizon
   or a stricter per-look threshold from a sequential design.

**What builds on this:** [A6.5](a6_5_bayesian_inference.md) (a Bayesian reading of A/B tests);
[B4.3](../B/b4_3_cross_validation_tuning.md) (tuning many hyperparameters is multiple testing on the validation
set); [J2.2](../J/j2_2_monitoring_drift.md) (online experiments and drift alerts);
[I2.4](../I/i2_4_rag_evals.md) (comparing prompts and models on eval sets); P1 (the multiple-comparison
correction in your report).

**References:** C. E. Bonferroni, "Teoria statistica delle classi e calcolo delle probabilità", 1936. S. Holm, "A Simple
Sequentially Rejective Multiple Test Procedure", *Scandinavian Journal of Statistics* 6(2), 1979.
Y. Benjamini and Y. Hochberg, "Controlling the False Discovery Rate", *Journal of the Royal Statistical
Society B* 57(1), 1995. S. J. Pocock, "Group sequential methods in the design and analysis of clinical
trials", *Biometrika* 64(2), 1977. P. C. O'Brien and T. R. Fleming, "A multiple testing procedure for clinical
trials", *Biometrics* 35(3), 1979. R. Kohavi, D. Tang and Y. Xu, *Trustworthy Online Controlled Experiments*,
Cambridge University Press, 2020.
