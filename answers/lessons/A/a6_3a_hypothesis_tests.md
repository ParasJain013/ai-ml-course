# Answer key: A6.3a Hypothesis tests: p-values, permutation, t and chi-square

Lesson: [`lessons/A/a6_3a_hypothesis_tests.md`](../../../lessons/A/a6_3a_hypothesis_tests.md)

## Exercise 1 (by hand)

**Solution (a).** $\bar{b} - \bar{a} = 7 - 4 = 3$. The pooled values $2, 4, 5, 6, 7, 9$ sum to 33; a split whose B values sum to $S$ has
difference $(2S - 33)/3$. There are $\binom{6}{3} = 20$ splits. Difference $\ge 3$ needs $S \ge 21$: $\{5, 7, 9\}$ (21, difference 3) and
$\{6, 7, 9\}$ (22, difference 3.67). Difference $\le -3$ needs $S \le 12$: $\{2, 4, 6\}$ (12) and $\{2, 4, 5\}$ (11). One-sided $p = 2/20 = 0.10$;
two-sided $p = 4/20 = 0.20$. The learner does not need to list all 20 splits, only to show that no other triple
reaches the thresholds (the largest sums without 9 or without 7 are 20 or less).

**(b)** $s_A^2/n_A = 4/4 = 1$, $s_B^2/n_B = 9/9 = 1$, $\mathrm{SE} = \sqrt2 = 1.414$, $t = 3/1.414 = 2.12$.
$\nu = (1 + 1)^2/(1^2/3 + 1^2/8) = 4/0.4583 = 8.73$. With $p = 0.064$: do not reject $H_0$ at 0.05. What to tell the team: "a
3-unit difference is plausible but so is no difference; with 4 and 9 observations we cannot tell. The 95%
interval is roughly $3 \pm 2.27 \times 1.41$, about $-0.2$ to $6.2$." Not "there is no effect", and not "almost significant, so
it works".

**(c)** Totals: 70 crashes of 600 sessions, 300 per platform, so expected crashes $300 \times 70/600 = 35$ per platform and
265 non-crashes. $\chi^2 = 2 \times \frac{10^2}{35} + 2 \times \frac{10^2}{265} = 5.714 + 0.755 = 6.47$, 1 degree of freedom, $p = 0.011$. Crash rates: 15.0%
on Android against 8.3% on iOS, a difference of 6.7 percentage points (Android about 1.8 times as likely to crash
per session).

**(d)** (i) False: the p-value is computed assuming $H_0$ is true; it is $P(\text{data this extreme} \mid H_0)$, not $P(H_0 \mid \text{data})$.
(ii) True: that is the definition (two-sided: a gap of at least 2.1 s in either direction). (iii) False:
replication depends on the true effect and the power of the next study; for an effect that is barely
significant, the chance of $p < 0.05$ again is often near 50%. (iv) False, or rather unknown: $p$ mixes effect size
and sample size; whether 2.1 s matters depends on the context and on the interval.

```text
== exercise 1 check ==
(a) observed 3.0; splits as extreme as observed: [([2.0, 4.0, 6.0], -3.0), ([2.0, 4.0, 5.0], -3.6667), ([6.0, 7.0, 9.0], 3.6667), ([5.0, 7.0, 9.0], 3.0)]
(a) one-sided p = 2/20   two-sided p = 4/20
(b) SE = 1.4142   t = 2.1213   nu = 8.7273   p = 0.0638
(c) expected [[35.0, 265.0], [35.0, 265.0]]   chi2 = 6.4690   df = 1   p = 0.0110   rates 0.1500 vs 0.0833
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) 20 splits, the extreme ones identified, $p = 0.10$ one-sided and $0.20$ two-sided | 3 |
| (b) $t = 2.12$, $\nu = 8.73$, correct conclusion and a non-overclaiming sentence for the team | 4 |
| (c) Expected counts 35/265, $\chi^2 = 6.47$ with 1 df, rates 15% vs 8.3% | 3 |
| (d) All four judged correctly with a reason each | 4 |

**Likely mistakes:**

- In (a), counting only the observed split as extreme and forgetting the split with difference 3.67, which
  is *more* extreme.
- In (b), using $n$ instead of $n - 1$ in the denominator of $\nu$, or pooling variances (Student's test).
- In (c), dividing by the observed instead of the expected count, or using $(r)(c)$ degrees of freedom.
- In (d), accepting (i) or (iii): the two most common misreadings in industry.

## Exercise 2 (code)

**Solution:** [`a6_3a_hypothesis_tests_ex2.py`](a6_3a_hypothesis_tests_ex2.py).

```text
== exercise 2: type I error and power, n = 10 per group, log-normal sigma = 1 ==
  scenario     Welch  permutation  Mann-Whitney
   H0 true     0.033        0.058         0.052
   B x 1.8     0.132        0.222         0.236
Monte Carlo SE: at 0.05 0.0069   at 0.5 0.0158
```

Reading: under $H_0$, the permutation test and Mann–Whitney hold their 5% (0.058 and 0.052 are within about one
standard error of 0.05), while Welch is conservative (0.033): with 10 skewed values per group its t
approximation is off. Under the alternative, all three are badly underpowered at $n = 10$, but Welch is the
worst (13% against 22–24%): the means of heavy-tailed samples are noisy, and a test on ranks or a permutation
test uses the data more efficiently here. For small skewed samples, use Mann–Whitney (if "B tends to be
larger" is the question) or a permutation test on a statistic you care about (if it is the mean or the
median), and plan a larger sample. A learner's exact numbers depend on their simulation; the ordering and
the explanation are what is graded.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Generic `perm_test` that accepts any two-sample statistic, vectorized over permutations | 4 |
| Correct simulation design: same data fed to all three tests, $H_0$ and alternative scenarios | 4 |
| Type I error and power reported with Monte Carlo standard errors | 3 |
| Correct reading: Welch conservative and less powerful here; recommendation with reasons | 3 |
| Readability: `SEED = 0`, type hints, docstrings with shapes, the `+1` in the permutation p-value | 2 |

**Likely mistakes:**

- A permutation p-value without the $+1$: it can be exactly 0, which is impossible for a real test.
- Shuffling each group separately instead of the pooled values (that keeps the labels and tests nothing).
- Running the alternative with `B + 1.8` on the log scale or forgetting to apply the factor at all.
- Reporting 0.058 as "the permutation test is anticonservative" without comparing it to the Monte Carlo
  error of 0.007.

## Exercise 3 (think)

**Strong answer:** *NDCG claim.* With 10,000 users a 0.003 difference can be highly significant and still
not matter: what is the confidence interval, and what does +0.003 NDCG mean for clicks or revenue? Was the
test paired (both models on the same users, a paired bootstrap or a paired t-test on per-user NDCG), which
it should be? Were the users independent of the training data (leakage would inflate both), and was this
the only metric and the only model variant compared, or the best of many (A6.3b)? Offline gains often do
not transfer online; I would ask for the interval, the per-segment results, and an online A/B test before
shipping. *Prompt claim.* $p = 0.4$ with 30 questions is absence of evidence, not evidence of equivalence: the
test probably had little power. I would ask for the observed difference and its interval (for example
"accuracy 70% vs 63%, 95% CI −12 to +26 points"), which will likely show that large differences are still
plausible; to claim equivalence you must state a margin that does not matter and show the whole interval
lies inside it, which usually needs far more questions. Also: paired analysis (same questions for both
prompts), and non-determinism (sample each prompt several times per question).

**Weak answer:** "The first result is significant so the model is better; the second is not significant so
the prompts are the same." Weak because it reads significance as importance in the first case and
non-significance as proof of no effect in the second, the two classic errors.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| NDCG: statistical vs practical significance, asks for the interval and a business translation | 3 |
| NDCG: pairing, leakage or multiple variants, offline-to-online gap (any two) | 2 |
| Prompts: absence of evidence is not evidence of absence; low power at $n = 30$ | 3 |
| Prompts: asks for the interval, an equivalence margin or a paired design | 2 |

## Scoring

Total 40 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → A6.3a "The
math" (permutation, Welch, chi-square, what a p-value is not); exercise 2 → A6.3a "From scratch" and A5.4
(Monte Carlo error); exercise 3 → A6.3a "In practice", A6.2 (intervals) and A6.3b (multiple comparisons).
