# Answer key: B4.1 Classification metrics: confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC, calibration

Lesson: [`lessons/B/b4_1_classification_metrics.md`](../../../lessons/B/b4_1_classification_metrics.md)

## Exercise 1 (by hand)

Scores $0.95, 0.85, 0.6, 0.45, 0.3, 0.2$ with labels $1, 0, 1, 1, 0, 0$.

**Solution (a).** Flagged at 0.5: 0.95 (positive), 0.85 (negative), 0.6 (positive). TP = 2, FP = 1. Not flagged: 0.45 (positive) is FN = 1;
0.3 and 0.2 are TN = 2. Accuracy $= 4/6 = 0.6667$; precision $= 2/3 = 0.6667$; recall $= 2/3 = 0.6667$; $F_1 = 2 \cdot 2/(4 + 1 + 1) = 0.6667$; FPR $= 1/3 = 0.3333$.

**(b)** Positives $\{0.95, 0.6, 0.45\}$, negatives $\{0.85, 0.3, 0.2\}$: 9 pairs. 0.95 beats all 3; 0.6 beats 0.3 and 0.2; 0.45 beats 0.3 and 0.2. AUC $= 7/9 = 0.7778$.
AP: in score order the positives sit at positions 1, 3 and 4, with precision $1/1$, $2/3$ and $3/4$; AP $= (1 + 0.6667 + 0.75)/3 = 0.8056$.

**(c)** Squared errors $0.0025, 0.7225, 0.16, 0.3025, 0.09, 0.04$; sum $1.3175$; Brier $= 0.2196$.

**(d)** With ten times the negatives drawn from the same scores, the expected counts at 0.5 become FP = 10 and TN = 20 (each current
negative becomes ten). Recall depends only on positives: unchanged at 0.6667. FPR $= 10/30$: unchanged at 0.3333. ROC-AUC is
unchanged (the share of correctly ordered pairs is the same). Precision falls from $2/3$ to $2/12 = 0.1667$: the same model now raises five
false alarms per hit.

```text
== exercise 1 check ==
(a) TP 2 FP 1 FN 1 TN 2   accuracy 0.6667  precision 0.6667  recall 0.6667  F1 0.6667  FPR 0.3333
(b) ROC-AUC 0.7778   AP 0.8056
(c) Brier 0.2196
(d) with 10x negatives (expected counts): TP 2 FP 10 -> precision 0.1667; recall 0.6667 and FPR 0.3333 unchanged
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Correct confusion matrix and all five numbers | 4 |
| (b) AUC by pair counting (7/9) and AP (0.8056) with the method shown | 4 |
| (c) Brier 0.2196 | 2 |
| (d) Recall, FPR and AUC unchanged; precision falls to 0.1667, with the reason (FP scales with N) | 4 |

**Likely mistakes:**

- Swapping FP and FN, or the rows and columns of the matrix. Fix it by always asking "what did I predict, what was true?"
- Computing AP as the area under a straight-line PR curve (the trapezoid), which gives a different, optimistic number.
- Counting a pair as correctly ranked when the *negative* has the higher score, or forgetting that each positive pairs with every negative.
- In (d), saying FPR rises "because there are more false positives": FP and N grow by the same factor.

## Exercise 2 (code)

**Solution:** [`b4_1_classification_metrics_ex2.py`](b4_1_classification_metrics_ex2.py). The rank formula works because the sum of
the positives' ranks counts, for each positive, how many rows score at or below it; subtracting $P(P + 1)/2$ removes the
positive–positive comparisons (including each positive with itself), leaving the number of negatives below each positive, which is the number of
correctly ordered pairs:

```python
ranks = rankdata(s)                                  # 1 = lowest; ties get the average rank
auc = (ranks[y == 1].sum() - P * (P + 1) / 2) / (P * N)
```

```text
== data and model (as in the lab) ==
val rows 9768   positives 2337

== (a) rank formula for ROC-AUC ==
rank formula 0.903302 = roc_auc_score 0.903302: OK

== (b) bootstrap confidence intervals ==
ROC-AUC  0.9033   95% percentile bootstrap CI [0.8969, 0.9096]   width 0.0127
AP       0.7568   95% percentile bootstrap CI [0.7401, 0.7736]   width 0.0335

== (c) thresholds ==
max F1 0.6889 at threshold 0.3168: precision 0.6306  recall 0.7591
highest threshold with recall >= 0.8: 0.2754: precision 0.5974  recall 0.8002
```

A strong write-up notes that AP's interval is more than twice as wide as ROC-AUC's: AP depends heavily on the top of the
ranking, where few rows decide the precision. It also notes that the F1-optimal threshold (0.32) is far below 0.5, as expected when positives are
24% of rows. Both thresholds were chosen on validation, so the precision and recall reported there are slightly optimistic. The final
numbers come from the test set (B4.3).

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Rank-formula AUC with average ranks for ties; matches `roc_auc_score` (an assert) | 4 |
| Bootstrap resamples whole rows with replacement, recomputes both metrics, percentile CI | 4 |
| F1-optimal threshold found over all distinct scores, not a coarse grid only | 3 |
| Highest threshold reaching recall ≥ 0.8, with precision there | 2 |
| Interpretation: CI widths, threshold far from 0.5, optimism of validation-chosen thresholds | 3 |

**Likely mistakes:**

- Ranking from highest to lowest (`rankdata(-s)`), which turns the formula into $1 - \text{AUC}$.
- Bootstrapping scores and labels separately, which breaks the pairs.
- Searching thresholds on `np.linspace(0, 1, 11)` and missing the optimum.
- Reporting a bootstrap CI computed on training rows.

## Exercise 3 (think)

**Strong answer:** ROC-AUC 0.98 says little at 0.1% prevalence. With 2 million transactions a day there are about 2,000 frauds and
1,998,000 legitimate ones; an FPR of just 1% is about 20,000 false alarms, 40 times the review capacity. The team should not threshold at
0.5 by habit: they should choose the threshold from capacity and cost. Questions: what precision and recall does the model have at the
threshold that yields about 500 alerts a day (precision at k = 500)? What is the PR curve and average precision, against the baseline of 0.001?
How were these measured: on a time-based split, with labels that arrive weeks later, and with chargebacks counted as fraud? What does a missed fraud cost compared
with a review? Calibration matters if scores are used as probabilities: to estimate expected loss per transaction (amount × P(fraud)) and
rank by expected loss rather than score, to set a threshold from costs with $t = c_{FP}/(c_{FP} + c_{FN})$ (B4.2), or to report expected fraud volumes to
finance. If the queue only ever takes the top 500 by score, calibration does not change who is reviewed.
**Weak answer:** "0.98 is an excellent AUC, so the model is ready; maybe lower the threshold if the reviewers have spare time."
**Why weak:** it ignores prevalence, the capacity constraint and precision, and treats 0.5 as meaningful.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Explains why ROC-AUC is misleading at 0.1% (converts FPR into alert volume) | 3 |
| Asks for precision at the capacity (top 500) and/or the PR curve against prevalence | 3 |
| Asks about evaluation design (time split, label delay) or costs | 2 |
| Says when calibration matters (expected loss, cost-based threshold, volume forecasts) and when it does not | 2 |

## Scoring

Total 40 pts. Pass ≥ 70% (28). Topic ids to revisit for each lost criterion: confusion-matrix ratios (Ex 1a, 1d) → B4.1;
ROC-AUC and AP (Ex 1b, Ex 2a) → B4.1; bootstrap (Ex 2b) → A6.2; thresholds and costs (Ex 2c, Ex 3) → B4.1, B4.2.
