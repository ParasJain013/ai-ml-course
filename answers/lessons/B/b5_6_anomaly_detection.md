# Answer key: B5.6 Anomaly detection: statistical rules, isolation forest, density-based scores

Lesson: [`lessons/B/b5_6_anomaly_detection.md`](../../../lessons/B/b5_6_anomaly_detection.md)

## Exercise 1 (by hand)

**Solution (a).** Mean $= 62/6 = 10.3333$. Deviations: $-7.33, -6.33, -6.33, -5.33, -4.33, 29.67$; squares $53.78, 40.11, 40.11, 28.44, 18.78, 880.11$, sum $1061.33$;
$\hat\sigma = \sqrt{1061.33/6} = \sqrt{176.89} = 13.30$. $z(40) = 29.67/13.30 = 2.23$. It cannot be flagged by $|z| > 3$: with $n = 6$ no z-score can exceed
$\sqrt{5} = 2.236$, and 40 is almost at that ceiling. Quartiles: $Q_1$ at position $5 \times 0.25 = 1.25$ of the sorted values, $4 + 0.25 \times (4 - 4) = 4$; $Q_3$ at
position 3.75, $5 + 0.75 \times (6 - 5) = 5.75$; IQR $= 1.75$; fences $[4 - 2.625,\ 5.75 + 2.625] = [1.375, 8.375]$. The fences flag 40: quartiles ignore how
extreme the largest value is, while $\hat\sigma$ is inflated by it (B1.2's ceiling, the lesson's masking argument).

**(b)** $H(7) = 1 + 0.5 + 0.3333 + 0.25 + 0.2 + 0.1667 + 0.1429 = 2.5929$. $c(8) = 2 \times 2.5929 - 2 \times 7/8 = 5.1857 - 1.75 = 3.4357$ (equivalently $2H(8) - 2$).
$s = 2^{-1.5/3.4357} = 2^{-0.4366} = 0.739$: clearly above 0.5, so suspicious.

**(c)** The four corners each have $(1, 1)$ at $\sqrt{2} = 1.414$ and two other corners at 2, so their 2nd-nearest distance is 2.0. The centre $(1, 1)$ has all four corners at
1.414, so its score is 1.414. $(6, 1)$ has $(2, 0)$ and $(2, 2)$ at $\sqrt{16 + 1} = 4.123$, so its score is 4.123: the most anomalous. The centre is *less* anomalous
than the corners: it sits in the densest spot. "In the middle of the data" and "dense" coincide here; the score measures density, not position.

**(d)** `contamination=0.02` sets the threshold so that 2% of the rows the model was fitted on are flagged: 100 alerts on 5,000 rows, twice what the team can review.
It changes nothing in the scores. Instead: choose the threshold from the budget, for example the score above which 1% of historical daily rows fall (about 50
a day), or simply send the top 50 each day ranked by score. Then monitor the daily alert count and the share confirmed by reviewers; if traffic grows, a fixed
score threshold and a fixed top-50 behave differently, and the choice is a product decision.

```text
== exercise 1 check ==
(a) mean 10.3333  std 13.3000  z(40) 2.2306  ceiling sqrt(5) 2.2361
    Q1 4.0000  Q3 5.7500  IQR 1.7500  fences [1.3750, 8.3750]
(b) H(7) 2.5929  c(8) 3.4357  s(h=1.5) 0.7389  s(h=c) 0.5000
(c) k = 2 distance scores: [2.0, 2.0, 2.0, 2.0, 1.4142, 4.1231]
(d) contamination 0.02 on 5000 rows flags 100; a budget of 50/day is contamination 0.010
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) Mean, $\hat\sigma$, $z = 2.23$, the $\sqrt{n - 1}$ ceiling, fences $[1.375, 8.375]$ and the reason the rules disagree | 4 |
| (b) $H(7)$, $c(8) = 3.436$, $s = 0.739$ | 3 |
| (c) All six scores; $(6, 1)$ most anomalous; the centre less anomalous than the corners, with the density reason | 3 |
| (d) Contamination only sets the threshold (100 a day); threshold from the budget instead | 2 |

**Likely mistakes:**

- Using $n - 1$ in $\hat\sigma$ ($z = 2.04$). Not wrong as statistics, but the lesson's ceiling $\sqrt{n - 1}$ is for the $n$ version; say which you use.
- Computing quartiles by "median of each half" (giving $Q_1 = 4$, $Q_3 = 6$): another valid convention, but it will not match NumPy. State the convention.
- Forgetting that the 2nd-nearest neighbour excludes the point itself, and reporting the 1st.
- Using $\ln$ for $H(7)$ in (b) when the exercise asks for the exact harmonic number ($\ln 7 + \gamma = 2.5231$, giving $c(8) = 3.296$).
- Believing `contamination` changes what the model learns.

## Exercise 2 (code)

**Solution:** [`b5_6_anomaly_detection_ex2.py`](b5_6_anomaly_detection_ex2.py).

```text
== data: reference set and new batch ==
reference 200 benign rows; new batch 172 rows, 15 anomalies (prevalence 0.0872)

== novelty detection: clean versus contaminated reference ==
clean ref        k-NN to reference (k=5)    AP 0.5956 [0.3520, 0.8512]  P@15 0.5333
clean ref        IsolationForest            AP 0.6325 [0.3851, 0.8964]  P@15 0.7333
contaminated ref k-NN to reference (k=5)    AP 0.4897 [0.2935, 0.7714]  P@15 0.5333
contaminated ref IsolationForest            AP 0.5926 [0.3474, 0.8437]  P@15 0.6000
k-NN novelty score matches sklearn NearestNeighbors: OK
```

**What a strong answer says:**

- The baseline AP is the new batch's prevalence, 0.087 (not the lab's 0.040): APs from the two settings are not directly comparable.
- With a clean reference, both detectors are about seven times better than random; the isolation forest puts 11 true anomalies in the top 15, the k-NN score 8. The
  bootstrap intervals overlap almost completely, so 15 labels cannot say which detector is better.
- Contaminating the reference with 15 malignant rows (7% of it) lowers both APs. The k-NN score loses more (0.60 → 0.49): a malignant row in the new batch now finds
  malignant neighbours in the reference and looks normal. The isolation forest loses less, because a few contaminating rows change the random splits only a little,
  but its top-15 precision still drops from 0.73 to 0.60. The general point: novelty detection assumes a clean reference; check it, and prefer robust
  methods when you cannot.
- The scaler is fitted on the reference rows only, as a deployed model would be.

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| Reference/new split as specified; scaler fitted on reference only | 3 |
| k-NN distance to reference rows from scratch, checked against `NearestNeighbors` (no self-exclusion needed here) | 3 |
| IsolationForest fitted on reference, scores negated from `score_samples` | 2 |
| AP with bootstrap CI and precision at 15 for both, both settings | 3 |
| Interpretation: prevalence baseline, overlapping CIs, why contamination hurts the k-NN score more | 3 |

**Likely mistakes:**

- Fitting the scaler (or the forest) on the reference plus the new batch: that is the outlier setting, not novelty detection.
- Excluding the nearest neighbour as "self" when the new rows are not in the reference: the score is then the 6th-nearest distance.
- Using `IsolationForest.predict` (labels at the `contamination` threshold) to compute AP, which needs a continuous score.
- Declaring a winner from point estimates whose CIs overlap.

## Exercise 3 (think)

**Strong answer:** Both, in stages. 40 confirmed cases over a year are too few to train and validate a supervised model well, and they describe only the fraud
patterns already caught. Start with unsupervised scores (an isolation forest on robustly scaled merchant features, plus a few transparent rules analysts already
trust, such as a refund rate far above the merchant's own history), and use the 40 cases to *evaluate*, not to train: AP and, more usefully, recall of known
fraud within the top 30 per day, back-tested day by day on last year's data with bootstrap CIs. Features should be relative to each merchant's own history and to
its segment (a large merchant is not anomalous for being large). The threshold is the budget: the 30 highest scores a day, possibly split between detectors. After
launch, every analyst verdict is a label: log it, measure precision at 30 weekly, and once a few hundred labels exist, train a supervised classifier (B4.4's
methods) and keep the anomaly score as one of its features, so novel fraud still surfaces. Watch for feedback loops (only reviewed merchants get labels, so the
labels reflect the detector's own choices) and for drift (seasonality, new merchant types), and re-fit the reference periodically on data the analysts cleared.
**Weak answer:** "Train an XGBoost classifier on the 40 fraud cases with class weights and flag everything above 0.5."
**Why weak:** 40 positives cannot support training and honest evaluation; 0.5 ignores the review budget; weights inflate probabilities (B4.4); and a supervised
model only finds fraud that looks like past fraud.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Justified choice (unsupervised first, labels for evaluation, supervised later or combined) | 3 |
| Evaluation: back-test with the 40 cases, precision/recall at the budget, CIs | 3 |
| Threshold from the 30-a-day budget, not 0.5 or a default contamination | 2 |
| After launch: logging verdicts, feedback loops, drift and re-fitting | 2 |

## Scoring

Total 36 pts. Pass ≥ 70% (26). Topic ids to revisit for each lost criterion: z-scores and fences (Ex 1a) → B1.2, B5.6; isolation score and $c(n)$ (Ex 1b) → B5.6;
distances and scaling (Ex 1c, Ex 2) → B2.3, B1.3; thresholds and budgets (Ex 1d, Ex 3) → B4.2, B5.6; AP and bootstrap CIs (Ex 2, Ex 3) → B4.1, A6.2; few positives
and supervised alternatives (Ex 3) → B4.4.
