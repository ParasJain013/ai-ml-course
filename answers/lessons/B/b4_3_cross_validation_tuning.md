# Answer key: B4.3 Cross-validation (k-fold, stratified, group, time-series) and hyperparameter tuning (grid, random, Bayesian)

Lesson: [`lessons/B/b4_3_cross_validation_tuning.md`](../../../lessons/B/b4_3_cross_validation_tuning.md)

## Exercise 1 (by hand)

Labels $y = (1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0)$ (positives at rows 0, 3, 7), customers $\mathbf{g} = (1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 5)$.

**Solution (a).** Unshuffled 4-fold: $12/4 = 3$ rows each: $\{0, 1, 2\}, \{3, 4, 5\}, \{6, 7, 8\}, \{9, 10, 11\}$.
Stratified 3-fold: class 1 appears first, so it gets code 0. The sorted codes are three 0s then nine 1s; positions $f, f + 3, f + 6, f + 9$ hold one code-0 and three code-1 entries for every
$f$. So each fold gets one positive and three negatives. Positives in order (rows 0, 3, 7) go to folds 1, 2, 3. Negatives in order (rows 1, 2, 4 | 5, 6, 8 | 9, 10, 11) go three to each fold. Test folds
$\{0, 1, 2, 4\}$, $\{3, 5, 6, 8\}$, $\{7, 9, 10, 11\}$. Time series, $k = 3$: $m = \lfloor 12/4\rfloor = 3$, blocks start at $12 - 9 = 3$: $\{3, 4, 5\}$, $\{6, 7, 8\}$, $\{9, 10, 11\}$, each trained on all earlier rows.

**(b)** Sizes: customer 5 has 4 rows, customer 2 has 3, customers 1 and 4 have 2, customer 3 has 1. Customer 5 → fold 1 (4 rows); customer 2 → fold 2 (3); customer 1 → fold 3 (2);
customer 4 → fold 3, the emptiest (4); customer 3 → fold 2 (4). Folds: $\{8, 9, 10, 11\}$, $\{2, 3, 4, 5\}$, $\{0, 1, 6, 7\}$, four rows each.

**(c)** The expected best of 20 equally good estimates sits about $1.87\sigma$ below the truth: $1.87 \times 0.004 \approx 0.0075$ RMSE of pure luck.

**(d)** Readmissions: `GroupKFold` by patient (several admissions per patient). Next-week sales: `TimeSeriesSplit` (train on the past only). One loan per person at 3%
defaults: `StratifiedKFold(shuffle=True)`. Photographers: `GroupKFold` by photographer, since a photographer's style, camera and locations repeat across their photos.

```text
== exercise 1 check ==
(a) KFold(4):          [0, 1, 2]  [3, 4, 5]  [6, 7, 8]  [9, 10, 11]
(a) StratifiedKFold(3): [0, 1, 2, 4]  [3, 5, 6, 8]  [7, 9, 10, 11]
(a) TimeSeriesSplit(3): [3, 4, 5]  [6, 7, 8]  [9, 10, 11]
(b) Step 3 rule: customer -> fold {5: 0, 2: 1, 1: 2, 4: 2, 3: 1}   fold sizes [4, 4, 4]
    GroupKFold(3) test rows: [8, 9, 10, 11]  [2, 3, 4, 5]  [0, 1, 6, 7]
(c) expected max of 20 standard normals 1.868 -> optimism about 0.0075 RMSE
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) All three sets of folds correct | 5 |
| (b) Group folds by the Step 3 rule, no customer split | 3 |
| (c) About 0.0075, with the reasoning (expected maximum of 20 normals × σ) | 2 |
| (d) Four correct splitters, each with a one-line reason | 4 |

**Likely mistakes:**

- Stratifying by giving each fold "a third of the positives" without the allocation rule, then mis-ordering the negatives.
- In the time-series split, putting the first block at rows 0–2. It starts after enough training history: $n - km$.
- Answering (c) with "0.004", the noise of one estimate, rather than the bias of the minimum of 20.
- Using stratified k-fold for the readmission case: it balances labels but still splits patients.

## Exercise 2 (code)

**Solution:** [`b4_3_cross_validation_tuning_ex2.py`](b4_3_cross_validation_tuning_ex2.py). Lag features use `shift`, so each row only sees earlier
days; the trailing mean is shifted by one day so it excludes the target day. The first 786 usable days are for development and the last 200 are the "future".

```text
== a daily series with lag features ==
rows after dropping incomplete lags 986   development 786   holdout (last 200 days)

== cross-validated estimates versus the future ==
shuffled KFold(5)          CV RMSE 3.8226   folds [3.937, 3.898, 3.556, 3.874, 3.849]
TimeSeriesSplit(5)         CV RMSE 4.5727   folds [4.711, 4.326, 4.77, 4.324, 4.733]
TimeSeriesSplit(5, gap=7)  CV RMSE 4.7348   folds [4.815, 4.387, 5.172, 4.581, 4.719]
holdout: last 200 days     RMSE 5.5418
training targets range 40.4..104.2   holdout targets range 80.6..111.9
```

**What a strong answer explains:**

- Shuffled k-fold is the most optimistic (3.82). Each validation day has training days on both sides of it, so the model has seen the trend level around that date, and its neighbours'
  lag features overlap with its own. It interpolates instead of forecasting.
- `TimeSeriesSplit` (4.57) trains only on the past, like production, and `gap=7` (4.73) also stops the training window's last days from overlapping with the validation
  rows' 7-day features. Both are closer to the holdout.
- The holdout is worse still (5.54) because of the trend. Its targets go up to 111.9, beyond the training maximum of 104.2, and trees cannot predict above the targets they saw
  (B3.2, B3.3). Time-series CV is pessimistic in the right direction, but it cannot fully reflect that the future goes further than any past fold did. Fix: model
  the trend (difference the series or predict the change from yesterday, B6.4).

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Series with trend, weekly cycle and noise; lag and trailing-mean features without look-ahead | 4 |
| Last 200 days held out and never used for choices | 2 |
| Three CV estimates computed on the development part only | 3 |
| Holdout RMSE computed with a model refitted on all development rows | 2 |
| Explains the shuffled optimism (neighbouring days, overlapping lags) | 3 |
| Notes the extrapolation problem (holdout beyond the training range) | 2 |

**Likely mistakes:**

- Computing the rolling mean without `shift(1)`, which puts the target inside its own feature (leakage, B1.4).
- Building features after shuffling, or splitting before creating lags so the first rows of each fold have no history.
- Concluding that time-series CV is "wrong" because it is worse: it is the honest one.

## Exercise 3 (think)

**Strong answer:**
1. *No test set:* the 0.912 is the score that selected the model, so it is optimistically biased.
2. *500 configurations:* the selection bias is large (Step 5): the best of 500 noisy estimates sits well above the typical true value. Nested CV or an untouched test set is needed.
3. *Several visits per patient with plain k-fold:* a patient's visits sit on both sides of the split, and the model can recognize patients. This biases AUC upwards, often by a lot.
4. *Three years of data with random folds:* the model trains on visits from after the ones it is validated on, and practice, coding and population drift over time. It is upward-biased for future patients.
5. *AUC alone:* it says nothing about calibration or about the threshold at which clinicians would act (B4.1, B4.2).

Instead: hold out the most recent 6–12 months as a test set, with patients disjoint from training (patients who appear only in the test period, or remove overlapping patients).
On the earlier data, use `StratifiedGroupKFold` by patient (time-ordered folds if drift is strong), with a fixed search budget (for example Optuna with 50 trials) inside it, preprocessing inside
the pipeline, and nested CV if a procedure-level estimate is needed. Refit on all development data, evaluate once on the test set with a bootstrap CI (resampling patients, not
visits), and report calibration and the metrics at the clinical threshold.
**Weak answer:** "500 configurations is thorough, so 0.912 is reliable. Maybe add a test set to double-check."
**Why weak:** it misses the patient grouping, time drift and selection bias, all of which push the number up, and it gives no design.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Selection bias from 500 configurations, upward | 2 |
| Patient-level leakage from multiple visits, upward | 3 |
| Temporal drift and training on future visits, upward | 2 |
| A concrete design: time-based test set, grouped CV, fixed budget, one test look, patient-level bootstrap | 3 |

## Scoring

Total 40 pts. Pass ≥ 70% (28). Topic ids to revisit for each lost criterion: splitter mechanics (Ex 1a, 1b) → B4.3; selection bias (Ex 1c, Ex 3) →
B4.3, A6.3b; choosing splitters and time leakage (Ex 1d, Ex 2, Ex 3) → B4.3, B1.4; extrapolation with trees (Ex 2) → B3.3, B6.4.
