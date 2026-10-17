# Answer key: A6.1 Descriptive statistics and EDA

Lesson: [`lessons/A/a6_1_descriptive_statistics_eda.md`](../../../lessons/A/a6_1_descriptive_statistics_eda.md)

## Exercise 1 (by hand)

**Solution (a).** Sorted: $10, 11, 12, 13, 14, 14, 15, 95$, $n = 8$.

- Mean $= 184/8 = 23.0$. Median $= (13 + 14)/2 = 13.5$ (even $n$: average the 4th and 5th values).
- Q1: $h = 7 \times 0.25 = 1.75$, between 0-based positions 1 and 2 (values 11 and 12): $11 + 0.75 \times 1 = 11.75$.
  Q3: $h = 7 \times 0.75 = 5.25$, between positions 5 and 6 (14 and 15): $14 + 0.25 \times 1 = 14.25$. IQR $= 2.5$.
- Fences: $11.75 - 1.5 \times 2.5 = 8.0$ and $14.25 + 3.75 = 18.0$. Only **95** is outside.
- $\lvert x - 13.5\rvert$ sorted: $0.5, 0.5, 0.5, 1.5, 1.5, 2.5, 3.5, 81.5$; MAD = mean of the 4th and 5th $= 1.5$; scaled MAD
  $= 1.4826 \times 1.5 = 2.224$.

**(b)** Robust z of 95: $(95 - 13.5)/2.224 = 36.6$. (The ordinary z-score, with mean 23 and sd 29.1, would be
only 2.5, because the 95 inflates the very mean and sd used to judge it. This is the "masking" effect.)

**(c)** Possible explanations and checks: a real spike (a launch, a press mention, a marketing email):
check the calendar, campaign logs and traffic sources for that day; a data error (a backfill, a
duplicated import, bot or test sign-ups): check for duplicate user ids, sign-ups with the same IP or
email domain, timestamps clustered in one minute; a definition change (the day a new sign-up flow
started counting something else). Only an error justifies deleting; a real spike stays and is reported
separately.

**(d)** $\bar{x} = 3$, $\bar{y} = 34/5 = 6.8$. $d^x = -2, -1, 0, 1, 2$; $d^y = -1.8, -2.8, -3.8, -4.8, 13.2$. Products:
$3.6 + 2.8 + 0 - 4.8 + 26.4 = 28$. $\sum (d^x)^2 = 10$, $\sum (d^y)^2 = 3.24 + 7.84 + 14.44 + 23.04 + 174.24 = 222.8$.
$r = 28/\sqrt{2{,}228} = 28/47.20 = 0.593$. Ranks of $y$: $4, 3, 2, 1, 5$; rank deviations $1, 0, -1, -2, 2$ against $-2, -1, 0, 1, 2$:
products $-2 + 0 + 0 - 2 + 4 = 0$, so $\rho = 0$. They disagree because the whole positive Pearson correlation
comes from the single point $(5, 20)$; the other four points *decrease*. Report Spearman (or, better, show
the scatter plot and say that the trend rests on one point).

```text
== exercise 1 check ==
sorted [10, 11, 12, 13, 14, 14, 15, 95]   mean 23.00   median 13.50
Q1 11.75   Q3 14.25   IQR 2.50   fences [8.00, 18.00]
|x - median| sorted [0.5, 0.5, 0.5, 1.5, 1.5, 2.5, 3.5, 81.5]   MAD 1.50   scaled 2.2239
robust z of 95 = 36.65
(d) Pearson 0.5932   Spearman 0.0000   ranks of y [4.0, 3.0, 2.0, 1.0, 5.0]
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Mean 23, median 13.5 (average of the middle two) | 2 |
| (a) Q1 11.75, Q3 14.25, IQR 2.5 by the $h = (n - 1)p$ rule; fences 8 and 18; 95 flagged | 3 |
| (a) MAD 1.5, scaled 2.22 | 2 |
| (b) Robust z ≈ 36.6 | 1 |
| (c) Two plausible explanations, each with a concrete check | 2 |
| (d) $r = 0.593$ and $\rho = 0$, with the one-point explanation and a choice | 4 |

**Likely mistakes:**

- Forgetting to sort before taking the median or quartiles.
- Using a different quartile convention (for example the median of each half: Q1 = 11.5, Q3 = 14.5). Accept
  it with full marks if the convention is stated; the point is to know there are several.
- Taking the MAD around the mean instead of the median, or forgetting that with even $n$ the median of the
  deviations is also an average of two values.
- In (d), ranking $y$ in the wrong direction or computing Spearman with the shortcut formula
  $1 - 6\sum d_i^2/(n(n^2 - 1))$ and making an arithmetic slip; the shortcut is fine without ties ($\sum d_i^2 = 9 + 1 + 1 + 9 + 0 = 20$,
  $1 - 120/120 = 0$).

## Exercise 2 (code)

**Solution:** [`a6_1_descriptive_statistics_eda_ex2.py`](a6_1_descriptive_statistics_eda_ex2.py).

```text
== exercise 2: describe_robust ==
               count  missing     mean  median        sd      IQR  MAD*1.48   trim10   skew  |z|>3.5
gaussian      2000.0      0.0   49.720  49.665    10.004   13.599    10.117   49.734 -0.062      2.0
lognormal     2000.0      0.0  165.351  96.947   207.061  143.492    87.189  122.503  3.737    177.0
unit_error    2000.0      0.0  616.836  20.120  3424.999    4.239     3.113   20.232  5.673     60.0
with_missing  1800.0      0.1   -0.001  -0.012     1.017    1.362     1.015    0.008 -0.108      0.0
median, IQR, MAD, trimmed mean and sd match pandas / scipy: OK

== exercise 2: Spearman matrix ==
              gaussian  lognormal  unit_error  with_missing
gaussian         1.000     -0.011       0.006        -0.009
lognormal       -0.011      1.000      -0.022         0.010
unit_error       0.006     -0.022       1.000         0.021
with_missing    -0.009      0.010       0.021         1.000
Spearman matrix matches DataFrame.corr(method='spearman'): OK
```

Reading the table: the Gaussian column has mean ≈ median and sd ≈ scaled MAD, and 2 flags in 2,000
(expected for a clean Gaussian at 3.5 robust sd). The log-normal column has mean 165 against median 97,
skew 3.7 and 177 flags: these are not errors, it is the shape, so it needs a log transform, not deletions.
The unit-error column has a mean 30 times its median, an sd 1,000 times its MAD, and exactly 60 flags
($3\% \times 2{,}000$): the robust summaries describe the 97% of good values and the flag count finds the bad
ones. The missing column is clean apart from 10% missing, which the count and `missing` columns report.
The Spearman matrix is near 0 everywhere because the columns were generated independently; a student
should say so rather than read meaning into $\pm 0.02$. Exact numbers depend on how the synthetic table is
generated; what is graded is that the checks pass and the reading is right.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| All ten statistics per numeric column, NaNs ignored correctly (count and share missing right) | 4 |
| Own median, IQR, MAD and trimmed mean, each checked with `assert_allclose` against pandas/SciPy | 4 |
| Synthetic table as specified (four column types, `SEED = 0`) | 2 |
| Own Spearman matrix checked against `df.corr(method="spearman")` on the same rows | 3 |
| Correct reading: log for the log-normal, investigate the unit error, nothing in the correlations | 2 |
| Readability: type hints, docstrings with shapes, no loops over rows | 1 |

**Likely mistakes:**

- Passing a column with NaNs to `np.sort`/`np.median`: NaNs sort to the end and corrupt the quantiles, and
  `np.median` returns `nan`. Drop them first (or use `np.nanmedian` and say so).
- Comparing a Spearman matrix built from complete rows with `df.corr`, which uses pairwise-complete
  rows: the numbers differ slightly when columns have different missing rows. Use the same rows.
- Using the raw MAD (no 1.4826) for the robust z-score: every flag count becomes much larger.
- Calling the 177 log-normal flags "outliers to delete".

## Exercise 3 (think)

**Strong answer:** The data is observational, so before mandating anything I would list what else could
produce the gap. **Confounders:** teams that adopted flags are likely more mature in other ways (better test
coverage, staged rollouts, more senior engineers, on-call discipline), or work on less risky services;
any of these lowers incidents by itself. **Reverse causation or selection:** teams that had many incidents
may be the ones still struggling to adopt flags, or teams adopt flags after their services stabilize; and
"incidents per deploy" depends on how often each team deploys (flag users may deploy many small changes,
which inflates the denominator). **Measurement:** do teams classify and report incidents the same way?
Checks in the existing data: compare teams within the same service tier, size and deploy frequency
(stratify or adjust), look at teams before and after they adopted flags (a team-level before/after
comparison, with a look at the overall trend over the same period), and check whether the difference
holds per deploy and per week. To find out whether flags help: roll out flags in a randomized or staggered
way (pick teams at random for the first wave, or randomize which risky changes ship behind a flag) and
compare incident rates, with the metric and the test chosen in advance (A6.3a, A6.3b). Reply: "The
correlation is promising but does not show flags cause the drop; here is a cheap way to find out before
we mandate it."

**Weak answer:** "A 40% reduction is large and the sample is big, so the effect is clearly real; we should
make flags mandatory." Weak because size and sample size say nothing about confounding: a large,
precisely measured correlation can be entirely due to which teams chose flags.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| At least two concrete confounders or selection effects specific to this setting | 3 |
| Mentions reverse causation, the denominator (per deploy) or measurement differences | 2 |
| A concrete check in existing data (stratify or adjust, before/after per team) | 2 |
| Proposes a randomized or staggered rollout with a pre-chosen metric | 2 |
| A clear, non-dismissive recommendation to the director | 1 |

## Scoring

Total 40 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → A6.1 "The
math" (quantiles, MAD, correlation) and A5.1 (covariance); exercise 2 → A6.1 "From scratch" and A1.3
(missing values in pandas); exercise 3 → A6.1 "Correlation is not causation" and A6.3b (experiments).
