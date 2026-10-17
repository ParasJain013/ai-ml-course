# Answer key: A6.3b Multiple comparisons and A/B tests

Lesson: [`lessons/A/a6_3b_multiple_comparisons_ab_tests.md`](../../../lessons/A/a6_3b_multiple_comparisons_ab_tests.md)

## Exercise 1 (by hand)

**Solution (a).** FWER $= 1 - 0.95^8 = 1 - 0.6634 = 0.337$. Sorted p-values (already sorted), $m = 8$:

| $k$ | $p_{(k)}$ | Bonferroni 0.00625 | Holm $0.05/(9 - k)$ | BH $0.05k/8$ |
|---|---|---|---|---|
| 1 | 0.004 | ✓ | 0.00625 ✓ | 0.00625 ✓ |
| 2 | 0.011 | ✗ | 0.00714 ✗ (stop) | 0.0125 ✓ |
| 3 | 0.019 | ✗ | — | 0.01875 ✗ |
| 4 | 0.04 | ✗ | — | 0.025 ✗ |
| 5–8 | 0.07 … 0.8 | ✗ | — | 0.03125 … 0.05 ✗ |

Bonferroni: 1 metric (0.004). Holm: 1 (stops at 0.011). BH: 2 (largest passing $k$ is 2). Three metrics had $p < 0.05$
uncorrected; only one survives FWER control.

**(b)** $n = (1.96 + 0.8416)^2 \times (0.05 \times 0.95 + 0.055 \times 0.945)/0.005^2 = 7.849 \times (0.0475 + 0.05198)/0.000025 = 7.849 \times 0.09948/0.000025 \approx 31{,}230$ per
arm. At 2,000 per arm per day: 15.6 days. Plan **21 days** (three full weeks), so that every weekday appears
equally often and weekly cycles do not bias the result; 14 days would be underpowered.

**(c)** Expected 10,000 each. $\chi^2 = 200^2/10{,}000 + 200^2/10{,}000 = 4 + 4 = 8.0$, 1 df, $p = 0.0047$. At the usual SRM threshold
of 0.001 this does not trigger an automatic alarm, but it is unlikely under a correct 50/50 split; investigate
before trusting the result: check the assignment code and the logging for each arm, look at the split by
day, platform and browser (an SRM confined to one segment points at the bug), and check for redirects,
crashes or bot filtering that affect one arm. If the cause cannot be found and fixed, do not use the result.
(Answers that call it a definite SRM at $\alpha = 0.05$ and stop to investigate also get full marks; the key is to
investigate before reading the metrics.)

**(d)** At any early look the estimated effect is the true effect plus noise. Stopping when $p < 0.05$ selects the
looks where the noise happened to push the estimate up (that is what made it cross the line), and once
stopped, the estimate never gets the chance to regress to the truth. So the reported effects of stopped
experiments are biased upward, most strongly for experiments that stop early, when the noise is largest.

```text
== exercise 1 check ==
(a) FWER for 8 tests = 0.3366
(a) bonferroni rejects [1, 0, 0, 0, 0, 0, 0, 0]
(a) holm       rejects [1, 0, 0, 0, 0, 0, 0, 0]
(a) fdr_bh     rejects [1, 1, 0, 0, 0, 0, 0, 0]
(b) n per arm = 31,231   days at 2,000 per arm per day = 15.6
(c) SRM chi-square = 8.00   p = 0.0047
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) FWER 0.337; Bonferroni 1, Holm 1, BH 2, with the thresholds shown | 4 |
| (b) $n \approx 31{,}000$ per arm and a whole-week duration with a reason | 4 |
| (c) $\chi^2 = 8$, $p \approx 0.005$, and a concrete investigation before trusting the result | 3 |
| (d) Selection on noise at the stopping time explained | 3 |

**Likely mistakes:**

- In (a), running Holm past the first failure, or stopping BH at the first failure (BH takes the largest
  passing $k$, even if a smaller $k$ failed).
- In (b), using the relative lift (10%) as $\delta$ instead of the absolute 0.005, or forgetting to square $\delta$; or
  dividing the daily traffic wrongly (4,000 visitors split equally is 2,000 per arm).
- In (c), concluding "1.0% imbalance, that's fine".
- In (d), saying the effect is overstated "because of false positives": the bias exists even when the
  effect is real.

## Exercise 2 (code)

**Solution:** [`a6_3b_multiple_comparisons_ab_tests_ex2.py`](a6_3b_multiple_comparisons_ab_tests_ex2.py).

```text
== exercise 2: calibrate a per-look threshold on separate A/A runs ==
threshold so that 5% of A/A runs ever cross it in 28 looks: 0.0075

== exercise 2: three stopping rules ==
rule                 false + (A/A)  power (11%)  mean lift among wins
fixed horizon               0.0410       0.9680                0.0101
naive peeking               0.2585       0.9830                0.0149
calibrated peeking          0.0620       0.9185                0.0141
true lift = 0.0100
```

Reading: (a) naive peeking has a 26% false positive rate; the fixed horizon (0.041) and the calibrated threshold
(0.062) are near 5%. Both differ from 0.05 by more than the Monte Carlo error of one rate (about 0.005), because
the calibrated threshold is itself estimated from 2,000 A/A runs; more calibration runs would tighten it. (b)
With 28,000 users per arm the test is well powered for a 1-point lift (about 1.9 times the 14,749 needed for 80%),
so all rules detect it most of the time; the calibrated rule pays a little power for its stricter threshold.
(c) Among declared winners, the fixed horizon reports the true lift (0.0101), while both peeking rules report
about 0.014 to 0.015, overstating the lift by 40% to 50%, because they stop when noise pushes the estimate up
(exercise 1(d)). Guideline: fixed horizon as the default; if the team must monitor early, use a calibrated or
group-sequential boundary for stopping, and still report the effect from a full-length run or with a
correction for early stopping. Learners' exact numbers vary with their simulation; the pattern is graded.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Cumulative daily p-values for A/A and A/B scenarios, vectorized | 3 |
| Calibrated threshold from separate A/A runs (not the ones used to evaluate it) | 3 |
| False positive rate, power and mean lift among wins for all three rules | 4 |
| Correct reading: peeking inflates false positives and overstates lifts; the calibrated rule fixes the first | 4 |
| A clear recommendation for guidelines | 1 |
| Readability: `SEED = 0`, type hints, docstrings with shapes | 1 |

**Likely mistakes:**

- Calibrating and evaluating on the same simulated runs, which makes the calibrated rule look exactly 5% by
  construction.
- Counting days from 0 so the "28-day" fixed test uses 27 days of data.
- Reporting the lift at day 28 for peeking runs that stopped earlier (that hides the overstatement).

## Exercise 3 (think)

**Strong answer:** Proposed plan. **Primary metric:** one metric that reflects the goal of onboarding, for
example the share of new users who complete a key action within 7 days (activation), chosen now. **Guardrails:**
2–4 metrics that must not get worse (sign-up completion, support tickets, page latency, 7-day churn), tested
one-sided for harm. **Sample size and duration:** pick the minimum detectable effect the business cares about
(for example +1 point on a 30% activation rate), compute $n$ per arm for 80% power at $\alpha = 0.05$, and translate it into
whole weeks of traffic; if that is longer than we can wait, either accept a larger MDE or do not run the test.
**Analysis:** one analysis at the planned end, with the SRM check first; the effect on the primary metric with
its 95% interval; a Welch or two-proportion test on the randomization unit (users). No daily decisions; a
dashboard can be watched for *harm* only (guardrail breaches with a strict threshold), which can stop the test.
**Other metrics and segments:** the remaining 10 metrics and the 6 segments are exploratory, reported with Holm-
or BH-adjusted p-values and labelled as hypotheses for future tests; only segments named in advance (say,
mobile versus desktop) get a confirmatory test. **Decision rules, written now:** ship if the primary metric's
interval excludes 0 in the good direction and no guardrail is harmed; if the primary metric is flat but a
secondary metric wins, do not ship on that basis; treat the secondary result as a new hypothesis and test it
with its own pre-registered experiment.

**Weak answer:** "Use Bonferroni on all 72 metric–segment combinations and check daily." Weak because a correction
for 72 tests leaves almost no power, daily checking is still peeking, and there is still no primary metric
or sample-size plan; it treats the symptom (many tests) without designing the experiment.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| One pre-chosen primary metric and named guardrails | 2 |
| Sample size from an MDE and power, duration in whole weeks | 2 |
| One planned analysis (no peeking for wins), SRM check, randomization unit | 2 |
| Secondary metrics and segments exploratory or corrected (Holm/BH) | 2 |
| Clear decision rule for "primary flat, secondary wins" | 2 |

## Scoring

Total 40 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → A6.3b "The math"
(corrections, sample size, SRM); exercise 2 → A6.3b "Peeking" and A5.4 (Monte Carlo error); exercise 3 →
A6.3b "Intuition" (A/B test design) and A6.1 (randomization and causation).
