# B4.1 Classification metrics: confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC, calibration

**Core** · Part B, week 4 (course week 10) · ~2 h · Lab:
[`labs/B/b4_1_classification_metrics.py`](../../labs/B/b4_1_classification_metrics.py)

## Why it matters

"The model is 85% accurate" is the start of an argument, not the end of one. On the census data in this lesson,
predicting "earns at most $50K" for everyone is 76% accurate and useless. The metric decides which model you ship
and which threshold you use, and the wrong metric hides exactly the errors that cost money: missed fraud, flagged
good customers, untreated patients. This lesson builds every standard classification metric from scratch: the
confusion matrix and the numbers derived from it, the two curves that summarize a model across all thresholds (ROC
and precision–recall), and **calibration**, which asks whether a predicted 0.8 really means 80%. Interviews probe these
constantly. In production, the classic mistakes are reporting ROC-AUC on a problem with 0.1% positives, and feeding
uncalibrated scores into a decision that treats them as probabilities.

## Prerequisite check

- Can you explain why logistic regression outputs a probability, and what `predict_proba(X)[:, 1]` returns? If not, revisit
  [B2.1](b2_1_logistic_regression.md).
- Do you know why validation rows, never training rows, are used to measure a model? If not, revisit
  [B1.1](b1_1_ml_workflow.md).
- Can you compute log-loss and say why it punishes confident mistakes? If not, revisit
  [B2.1](b2_1_logistic_regression.md) and [A6.4](../A/a6_4_entropy_cross_entropy_kl.md).
- Can you compute a conditional probability $P(A \mid B)$ from a table of counts? If not, revisit
  [A5.3](../A/a5_3_bayes_theorem.md).

## Intuition

A classifier gives each row a *score*: here, the predicted probability that a person earns more than $50K. A
decision needs a yes or no, so you pick a *threshold* and call everything above it positive. Every threshold
gives four counts: true positives (flagged and right), false positives (flagged but wrong), false negatives
(missed), true negatives (correctly left alone). That 2 × 2 table is the **confusion matrix**, and almost every
metric is a ratio of its cells.

Two ratios matter most. **Precision**: of the rows I flagged, how many were right? **Recall** (also called sensitivity or
true positive rate): of the rows I should have flagged, how many did I catch? A spam filter tuned for precision rarely
hides a real email but lets spam through; tuned for recall it catches all spam but also some real mail. Moving
the threshold trades one for the other. Neither is "the" metric; the costs of the two errors decide.

To compare models without committing to a threshold, sweep all of them. Plotting true positive rate against false
positive rate gives the **ROC curve**; its area, **ROC-AUC**, is the chance that a random positive gets a higher score
than a random negative. Plotting precision against recall gives the **precision–recall curve**; its area is summarized by **average
precision** (AP, often called PR-AUC). When positives are rare, the two tell very different stories, and the PR curve is the honest one.

Finally, a ranking can be perfect while the numbers are nonsense. If a model says "0.9" for 1,000 people and only
400 of them are positive, its probabilities are *overconfident*. A **reliability diagram** checks this by grouping
predictions into bins and comparing the average prediction with the observed rate in each bin. When the scores feed
a cost calculation, a price or a triage queue, calibration matters as much as ranking.

> **Bridge:** precision and recall are the two error budgets of an alerting system. Precision is "what fraction of
> pages were real incidents" (low precision means alert fatigue); recall is "what fraction of incidents paged
> someone" (low recall means outages nobody saw). **Where it breaks:** with alerts you can usually find out about
> the incidents you missed afterwards; with many ML problems the false negatives are never observed (a loan you
> refused never defaults), so you only see half of the confusion matrix in production.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{y}$ | true labels, $y_i \in \{0, 1\}$ (1 = positive) | `(n,)` |
| $\mathbf{s}$ | scores (here predicted probabilities $\hat{p}_i$) | `(n,)` |
| $t$ | decision threshold: predict 1 when $s_i \ge t$ | scalar |
| TP, FP, FN, TN | counts of true positives, false positives, false negatives, true negatives at $t$ | ints |
| $P$, $N$ | number of positives and negatives, $P = \text{TP} + \text{FN}$, $N = \text{FP} + \text{TN}$ | ints |
| $\pi$ | prevalence (positive share) $P / n$ | scalar |
| $B_k$ | the rows whose score falls in bin $k$ of the reliability diagram | set |

**Step 1: ratios of the confusion matrix.**

$$\text{accuracy} = \frac{\text{TP} + \text{TN}}{n}, \quad \text{precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{recall} = \text{TPR} = \frac{\text{TP}}{P}, \quad \text{FPR} = \frac{\text{FP}}{N}.$$

**F1** is the harmonic mean of precision and recall:

$$F_1 = \frac{2}{1/\text{precision} + 1/\text{recall}} = \frac{2\,\text{precision}\cdot\text{recall}}{\text{precision} + \text{recall}} = \frac{2\,\text{TP}}{2\,\text{TP} + \text{FP} + \text{FN}}.$$

(For the last form, substitute the definitions and multiply top and bottom by $(\text{TP} + \text{FP})(\text{TP} + \text{FN})/\text{TP}$.) The harmonic mean is pulled towards the smaller of the two
numbers: precision 1.0 and recall 0.1 give $F_1 = 0.18$, not the arithmetic mean 0.55. Note that TN does not appear: F1 ignores how many negatives
you correctly left alone, which is what you want when negatives are plentiful and uninteresting.

**Step 2: the ROC curve and its area.** Sort the rows by score, highest first, and lower the threshold one distinct
score at a time. Each positive you pass raises TPR by $1/P$; each negative raises FPR by $1/N$. Starting at $(0, 0)$ and ending at $(1, 1)$, the curve
is a staircase. Its area by the trapezoid rule is $\text{AUC} = \sum_k (\text{FPR}_k - \text{FPR}_{k-1})\,\frac{\text{TPR}_k + \text{TPR}_{k-1}}{2}$.

Why is that the ranking probability? Each step right (one negative, width $1/N$) adds a strip whose height is the
TPR at that point: the share of positives already passed, which is the share of positives scored *above* this negative. Summing over
negatives: AUC $= \frac1N\sum_{\text{negatives } j}\frac{\#\{\text{positives above } j\}}{P} = \frac{\#\{(i, j): s_i > s_j\}}{P\,N}$, the fraction of (positive, negative) pairs ranked correctly. Tied scores make a diagonal step, which the trapezoid counts as half a pair.
AUC 0.5 is random ranking; 1.0 is perfect.

**Step 3: precision–recall and average precision.** At the same thresholds compute precision and recall. Average precision weights each threshold's precision by
the recall it adds:

$$\text{AP} = \sum_k (R_k - R_{k-1})\,P_k.$$

Recall rises only when a positive is passed, so AP is the mean of the precision measured right after each positive in score order. A random
ranking has precision $\pi$ everywhere, so the baseline AP is the prevalence, not 0.5. scikit-learn computes AP with this sum, not with a trapezoid,
because linear interpolation between PR points is optimistic (VERIFY as of 2026-10).

**Step 4: why the two areas react differently to rare positives.** FPR divides false positives by $N$, the number of
negatives. If you keep the same model and add negatives that look like the existing ones, FPR at each threshold stays the same, so the
ROC curve barely moves. Precision divides by TP + FP, and FP grows with $N$, so precision falls. With 1 positive per 1,000 rows, an FPR of 1% means
10 false alarms for every case found. The ROC curve calls that excellent; the PR curve shows precision below 0.1.

**Step 5: calibration.** A model is calibrated if, among the rows where it predicts $\hat{p} \approx q$, a share $q$ are positive. The reliability diagram
splits $[0, 1]$ into equal-width bins and, for each bin $B_k$, plots the mean prediction $\frac{1}{|B_k|}\sum_{i \in B_k}\hat{p}_i$ against the observed rate $\frac{1}{|B_k|}\sum_{i \in B_k}y_i$. Points on the diagonal mean calibrated; points
below it mean the model predicts too high in that bin, points above it too low. *Overconfident* means too extreme on either side: too high near 1 and too low near 0. The **Brier score** summarizes the probabilities in one number,

$$\text{Brier} = \frac1n\sum_i(\hat{p}_i - y_i)^2,$$

the mean squared error of the probabilities: 0 is perfect, and always predicting the prevalence scores $\pi(1 - \pi)$ (a share $\pi$ of rows cost $(1 - \pi)^2$ and the rest $\pi^2$: $\pi(1 - \pi)^2 + (1 - \pi)\pi^2 = \pi(1 - \pi)$). Like log-loss, it is a proper scoring rule: in expectation, it is minimized
by reporting the true probability. Ranking metrics are not: AUC is unchanged by any increasing transformation of the scores.

**Step 6: recalibrating.** Fit a monotone map from scores to probabilities on rows the model was not trained on.
**Platt scaling** fits $\sigma(a s + b)$, a one-feature logistic regression on the score. **Isotonic regression** fits the best non-decreasing step function. `CalibratedClassifierCV(cv=5)`
does this by cross-fitting: train the model on four folds, fit the map on the fifth, repeat, and average the five calibrated models.
Because the map is monotone, the ranking (and so ROC-AUC) barely changes; only the numbers move.

## Worked example by hand

Eight rows, sorted by score:

| Row | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| score $s$ | 0.9 | 0.8 | 0.7 | 0.6 | 0.55 | 0.4 | 0.3 | 0.1 |
| label $y$ | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 |

**Threshold 0.5.** Rows 1–5 are flagged: TP = 3 (rows 1, 2, 4), FP = 2 (rows 3, 5). Not flagged: FN = 1 (row 7), TN = 2 (rows 6, 8). Accuracy $= 5/8 = 0.625$. Precision $= 3/5 = 0.6$,
recall $= 3/4 = 0.75$, $F_1 = 2 \times 3/(6 + 2 + 1) = 6/9 = 0.6667$. FPR $= 2/4 = 0.5$.

**ROC-AUC by counting pairs.** Positive scores $\{0.9, 0.8, 0.6, 0.3\}$, negative scores $\{0.7, 0.55, 0.4, 0.1\}$, so 16 pairs. 0.9 and 0.8 beat all four negatives (8 pairs); 0.6 beats 0.55, 0.4 and 0.1 (3);
0.3 beats 0.1 (1). AUC $= 12/16 = 0.75$.

**Average precision.** Walk down the list and note the precision each time a positive appears: row 1: $1/1$; row 2: $2/2$; row 4: $3/4$; row 7: $4/7 = 0.5714$.
AP $= (1 + 1 + 0.75 + 0.5714)/4 = 0.8304$.

**Brier score.** Squared errors: $0.01, 0.04, 0.49, 0.16, 0.3025, 0.16, 0.49, 0.01$; sum $1.6625$; mean $0.2078$.

```text
== worked example by hand ==
threshold 0.5: TN 2  FP 2  FN 1  TP 3   accuracy 0.6250
precision 0.6000  recall 0.7500  F1 0.6667  FPR 0.5000
pairs (positive, negative) ranked correctly: 12 of 16 -> AUC 0.7500
precision at each positive, in score order: [1.0, 1.0, 0.75, 0.5714] -> AP 0.8304
Brier score mean((s - y)^2) = 0.2078
```

## Key plot

![Reliability diagram: logistic regression's points lie close to the diagonal; Gaussian naive Bayes's points lie mostly far below it, with most rows at predicted 0 or 1 and an observed rate of only 0.38 in the top bin; after isotonic calibration the naive Bayes points move near the diagonal but only span 0 to 0.55](img/b4_1_reliability_diagram.png)

Logistic regression's probabilities can be read as probabilities: its points track the diagonal (Brier 0.104). Gaussian
naive Bayes says 0 or 1 for almost everyone, and among the people it gives about 1.0, only 38% earn over $50K. Isotonic
calibration maps its scores back onto the diagonal (Brier 0.130). It cannot spread them further than the original scores
allow, so its predictions only reach about 0.55, and it cannot fix naive Bayes's weaker ranking.

## From scratch in NumPy

**The data.** UCI Adult: 48,842 people from the 1994 US census, 5 numeric and 7 categorical features, and the label "income
above $50K" (24% positive). A stratified 60/20/20 split. The model is a logistic-regression pipeline: standardized numeric
columns, one-hot categories (a `?` is kept as its own category), `LogisticRegression`. Its validation scores are what we measure.

```text
== data ==
data/adult/adult-all.csv: checksum OK, 48842 rows
positive (>50K) share: all 0.2393  train 0.2393  val 0.2393
train (29306, 12)  val (9768, 12)  (test 9768 rows kept aside)
rows with a '?' somewhere: 3620
```

**Threshold metrics.** `confusion` counts the four cells with boolean masks; `prf` applies Step 1.

```text
== from scratch: confusion matrix, precision, recall, F1 ==
threshold 0.3: TN  6304 FP 1127 FN  529 TP 1808   accuracy 0.8305  precision 0.6160  recall 0.7736  F1 0.6859
threshold 0.5: TN  6911 FP  520 FN  979 TP 1358   accuracy 0.8465  precision 0.7231  recall 0.5811  F1 0.6444
threshold 0.7: TN  7263 FP  168 FN 1479 TP  858   accuracy 0.8314  precision 0.8363  recall 0.3671  F1 0.5103
always predict <=50K: accuracy 0.7607, recall 0
```

The model is only 8.6 points more accurate than predicting the majority class, but its recall at 0.5 is 0.58 where the majority rule's is 0.
Raising the threshold from 0.3 to 0.7 raises precision from 0.62 to 0.84 and cuts recall from 0.77 to 0.37. Accuracy moves least of all, which is why
it is a poor guide here. F1 is highest at 0.3, not at the default 0.5.

**ROC-AUC and average precision.** Sort once, find the last row of each run of equal scores, and take cumulative counts there:

```python
order = np.argsort(-s, kind="stable")
s_sorted, y_sorted = s[order], y[order]
last = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s) - 1]   # last row of each run of equal scores
tps = np.cumsum(y_sorted)[last]                                # positives at or above each threshold
fps = (last + 1) - tps                                         # negatives at or above each threshold
tpr, fpr = np.r_[0.0, tps / y.sum()], np.r_[0.0, fps / (len(y) - y.sum())]
auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)          # trapezoid rule
precision, recall = tps / (last + 1), tps / y.sum()
ap = np.sum(np.diff(np.r_[0.0, recall]) * precision)           # Step 3
```

```text
== from scratch: ROC-AUC and average precision ==
ROC curve: 9210 points   ROC-AUC 0.9033   average precision (PR-AUC) 0.7568   prevalence 0.2393
to catch 50% of the >50K rows: threshold 0.5777, FPR 0.0502
to catch 80% of the >50K rows: threshold 0.2754, FPR 0.1696
to catch 95% of the >50K rows: threshold 0.0973, FPR 0.3582
8-row example with the same functions: AUC 0.7500  AP 0.8304
```

A random positive outscores a random negative 90% of the time. The ROC curve also answers operational questions: catching 95% of
high earners means a threshold of 0.097 and flagging 36% of everyone else.

**Rare positives.** Keep every validation negative, but only enough positives for a 2% prevalence (Step 4):

```text
== ROC-AUC versus PR-AUC when positives are rare ==
original: 2337 positives / 9768 rows   ROC-AUC 0.9033   AP 0.7568
rare:     152 positives / 7583 rows   ROC-AUC 0.8963   AP 0.2518
original at threshold 0.5: precision 0.7231  recall 0.5811
rare     at threshold 0.5: precision 0.1391  recall 0.5526
```

Same model, same kinds of rows. ROC-AUC barely moves (0.903 to 0.896); average precision falls from 0.76 to 0.25, and precision at 0.5 from
0.72 to 0.14: six false alarms for every true positive. If your positives are rare, report the PR curve or precision at the recall you need.
The lab saves both curves to `outputs/b4_1/roc_pr_rare.png`.

**Calibration.** Step 5 in code: bin index `min(int(p * 10), 9)`, then per-bin means with `np.bincount(..., weights=...)`:

```python
bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
counts = np.bincount(bins, minlength=n_bins)
keep = counts > 0
mean_p = np.bincount(bins, weights=p, minlength=n_bins)[keep] / counts[keep]
frac_pos = np.bincount(bins, weights=y, minlength=n_bins)[keep] / counts[keep]
```

```text
== from scratch: calibration ==
logistic regression         Brier 0.1041   ROC-AUC 0.9033   share of p < 0.01 or > 0.99: 0.2131   largest bin gap 0.0409
Gaussian naive Bayes        Brier 0.3850   ROC-AUC 0.8404   share of p < 0.01 or > 0.99: 0.9348   largest bin gap 0.7295
GNB + isotonic calibration  Brier 0.1303   ROC-AUC 0.8456   share of p < 0.01 or > 0.99: 0.0000   largest bin gap 0.1599
Gaussian NB, bin by bin (mean predicted -> observed, count):
   0.00->0.04 (3791)  0.15->0.16 (45)  0.24->0.09 (43)  0.35->0.16 (25)  0.45->0.18 (38)  0.54->0.12 (26)  0.65->0.15 (27)  0.75->0.15 (53)  0.85->0.12 (64)  1.00->0.38 (5656)
```

- Always predicting the prevalence would score a Brier of $0.2393 \times 0.7607 = 0.182$. Logistic regression (0.104) beats it; raw naive Bayes (0.385)
  is *worse than knowing nothing*, because 93% of its predictions are below 0.01 or above 0.99, and 5,656 people get about 1.0 when only 38% of them are positive.
  B2.4 explained why: multiplying many per-feature likelihoods as if they were independent piles up evidence that is really counted twice.
- Isotonic calibration (cross-fitted, Step 6) brings the Brier score to 0.130 and removes every extreme prediction. Its ROC-AUC
  (0.846) stays close to the raw model's (0.840), still far below logistic regression's: calibration fixes the probabilities, not the ranking.

## With scikit-learn

Every number above has a scikit-learn function; the lab checks each one against ours:

```text
== with scikit-learn ==
confusion_matrix and precision_recall_fscore_support match: OK
roc_curve, roc_auc_score (0.9033) and average_precision_score (0.7568) match: OK
calibration_curve and brier_score_loss match for all three models: OK
```

- `confusion_matrix(y, pred).ravel()` returns `(tn, fp, fn, tp)` in that order, rows being the true class.
- `roc_curve` drops points on straight segments by default (`drop_intermediate=True`); the lab turns that off to compare every point.
- `calibration_curve` returns `(observed rate, mean predicted)`: observed first, the reverse of the order you plot them.
- `CalibratedClassifierCV(model, method="isotonic", cv=5)` refits the model inside cross-validation. Use `method="sigmoid"` (Platt) when you have few
  rows, since isotonic regression overfits small calibration sets. Fit it on training data only, never on the validation rows you report.

## In practice

- **When to use it:**
  - *A threshold is fixed by the process:* precision and recall at that threshold, plus the confusion matrix.
  - *Comparing rankers:* ROC-AUC when classes are reasonably balanced, average precision (or precision at a target recall) when positives are rare.
  - *The scores feed a decision as probabilities* (expected cost, pricing, risk tiers): Brier score or log-loss, and a reliability diagram.
  - Report the baseline next to every number: the majority class for accuracy, the prevalence for AP, $\pi(1 - \pi)$ for Brier.
- **Failure modes:**
  - *Accuracy on imbalanced data.* 99.9% accuracy at 0.1% fraud is the "never fraud" model. Look at recall.
  - *ROC-AUC on rare positives.* It stays high while precision collapses. Check the PR curve.
  - *Picking the threshold on the test set.* The threshold is a tuned setting; choose it on validation or cross-validation folds (B4.2, B4.3).
  - *Trusting uncalibrated scores.* Naive Bayes, SVM decision values, boosted trees with class weights (B4.4) and resampled models are often
    miscalibrated. Draw a reliability diagram before using the numbers as probabilities.
  - *Calibrating on the evaluation rows.* The calibration map is a fitted model; fit it on separate rows or with cross-fitting.
  - *Ignoring uncertainty.* With 113 validation rows, one row is 0.9 points of accuracy. Bootstrap a confidence interval (A6.2) before
    declaring a winner.
- **Defaults that work:** report a confusion matrix at the chosen threshold, ROC-AUC and AP from `predict_proba`, and the Brier score;
  `CalibratedClassifierCV(cv=5)` with `"sigmoid"` for small data and `"isotonic"` for thousands of rows (scikit-learn 1.9.1; VERIFY as of 2026-10).

## Exercises

Save your work to `my-work/B/b4_1_classification_metrics.md` (or `.py`) and ask for a review.

1. **By hand:** Six rows with scores $0.95, 0.85, 0.6, 0.45, 0.3, 0.2$ and labels $1, 0, 1, 1, 0, 0$. (a) At threshold 0.5, write the confusion matrix and compute accuracy,
   precision, recall, F1 and FPR. (b) Compute ROC-AUC by counting pairs, and average precision. (c) Compute the Brier score. (d) Ten times as many negatives arrive,
   with the same score distribution as the current negatives. Which of precision, recall, FPR and ROC-AUC change, and how, at threshold 0.5?
2. **Code:** On the lab's validation scores: (a) implement ROC-AUC with the rank formula, $\text{AUC} = \frac{\sum_{i: y_i = 1}\text{rank}(s_i) - P(P + 1)/2}{P\,N}$ (average ranks for ties, `scipy.stats.rankdata`),
   and check it against `roc_auc_score`; (b) compute 95% bootstrap confidence intervals (1,000 resamples of the validation rows, A6.2) for ROC-AUC and average
   precision; (c) find the threshold that maximizes F1, and the highest threshold that still gives recall ≥ 0.8, and report precision at each.
3. **Think:** A fraud team presents a model with ROC-AUC 0.98 on transactions where 0.1% are fraud, and proposes to send every transaction with a
   score above 0.5 to a manual-review team that can handle 500 cases a day out of 2 million transactions. What questions do you ask, which numbers do you want to see
   instead, and when would calibration matter for this team?

## Takeaways

1. Every threshold gives a confusion matrix; precision $= \text{TP}/(\text{TP} + \text{FP})$, recall $= \text{TP}/P$, FPR $= \text{FP}/N$, and $F_1 = 2\text{TP}/(2\text{TP} + \text{FP} + \text{FN})$ ignores true negatives.
2. ROC-AUC is the probability that a random positive outscores a random negative; average precision averages precision over the positives in score order,
   with the prevalence as its baseline.
3. When positives are rare, ROC-AUC barely changes while precision collapses: report the PR curve or precision at the recall you need.
4. Calibration asks whether predicted probabilities match observed rates; check it with a reliability diagram and the Brier score (baseline $\pi(1 - \pi)$).
5. Platt scaling and isotonic regression, fitted on held-out rows, repair probabilities while barely changing the ranking; they cannot make a weak
   ranker strong.

**What builds on this:** [B4.2](b4_2_regression_metrics.md) (choosing the threshold from costs);
[B4.3](b4_3_cross_validation_tuning.md) (cross-validated metrics); [B4.4](b4_4_class_imbalance.md) (imbalance, and which fixes
break calibration); [B4.6](b4_6_fairness_model_cards.md) (the same rates per group); [I2.4](../I/i2_4_rag_evals.md) (precision and
recall for retrieval); [J2.2](../J/j2_2_monitoring_drift.md) (monitoring metrics in production); P2 (the metric and threshold).

**References:** T. Fawcett, "An introduction to ROC analysis", *Pattern Recognition Letters* 27, 2006. J. Davis, M. Goadrich,
"The relationship between precision-recall and ROC curves", *ICML*, 2006. A. Niculescu-Mizil, R. Caruana, "Predicting good
probabilities with supervised learning", *ICML*, 2005. G. W. Brier, "Verification of forecasts expressed in terms of probability",
*Monthly Weather Review* 78, 1950. R. Becker, R. Kohavi, "Adult", UCI Machine Learning Repository, 1996 (the dataset).
scikit-learn User Guide, "Metrics and scoring" and "Probability calibration" (VERIFY links:
https://scikit-learn.org/stable/modules/model_evaluation.html, https://scikit-learn.org/stable/modules/calibration.html).
