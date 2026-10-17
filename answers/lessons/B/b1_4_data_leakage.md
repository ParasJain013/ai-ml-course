# Answer key: B1.4 Data leakage

Lesson: [`lessons/B/b1_4_data_leakage.md`](../../../lessons/B/b1_4_data_leakage.md)

## Exercise 1 (by hand)

**Solution (a).** Positives: 0.8, 0.7, 0.3. Negatives: 0.7, 0.5, 0.2. The nine pairs:

- 0.8 against 0.7, 0.5, 0.2: all correct, 3.
- 0.7 against 0.7 (tie, $\frac12$), 0.5, 0.2: 2.5.
- 0.3 against 0.7 and 0.5 (wrong), 0.2 (correct): 1.

$U = 6.5$, AUC $= 6.5/9 = 0.722$. Ranks, ascending: 0.2 → 1, 0.3 → 2, 0.5 → 3, the two 0.7s share ranks 4 and 5 → 4.5 each, 0.8 → 6.
Positive ranks: 6, 4.5, 2; $R_+ = 12.5$; $U = 12.5 - 3 \times 4/2 = 6.5$. Same answer.

**(b)** Training rows: mean 13, deviations $-3, -1, 1, 3$, variance $20/4 = 5$, standard deviation 2.236; the test values become
$17/2.236 = 7.60$ and $19/2.236 = 8.50$: far outside the training range, which is the truth. Fitted on all six: mean $114/6 = 19$,
deviations $-9, -7, -5, -3, 11, 13$, squares sum to 454, variance 75.67, standard deviation 8.70; the test values become 1.26 and 1.49,
and look ordinary. A drift detector or an outlier rule downstream would now miss them.

**(c)**

- `applicant_income`: fine, known at application (if it is the declared income on the form, not a later
  verified figure).
- `number_of_late_payments` on this loan: target leakage; late payments happen after the loan is granted
  and are part of how default comes about.
- `credit_score_at_application`: fine *if* it really is the score pulled at application time; "depends" if the
  column is refreshed nightly, in which case old rows carry later scores (temporal leakage). Ask how and when it is
  written.
- `loan_officer_decision_notes`: depends; notes written during the decision are known, notes appended later
  ("customer in arrears, called twice") are leakage. Ask whether the field is ever edited after approval.
- `average_default_rate_of_applicants_from_same_postcode` computed over the whole dataset: leakage on two
  counts, contamination (it includes the test rows' labels, and the row's own label) and temporal (it includes
  defaults that happened after this application). Compute it from earlier applications only, out of fold.

```text
== exercise 1 check ==
(a) pairs [(0.8, 0.7), (0.8, 0.5), (0.8, 0.2), (0.7, 0.7), (0.7, 0.5), (0.7, 0.2), (0.3, 0.7), (0.3, 0.5), (0.3, 0.2)]
    U = 6.5 of 9 -> AUC 0.7222   rank formula 0.7222   roc_auc_score 0.7222
(b) train-fitted: mean 13.0 sd 2.2361 -> test [7.6026, 8.4971]
    all-fitted:   mean 19.0 sd 8.6987 -> test [1.2646, 1.4945]
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Pair count with the tie as $\frac12$: 0.722 | 2 |
| (a) Rank formula with average ranks, same result | 2 |
| (b) Both standardizations correct, and what the all-rows version hides | 3 |
| (c) Each of the five features classified with a sound reason (1 pt each) | 5 |
| (c) For the "depends" cases, the right question about when the value is written | 2 |

**Likely mistakes:**

- Counting a tie as correct or as wrong (gives 0.778 or 0.667).
- Giving tied scores ranks 4 and 5 in their original order instead of 4.5 each.
- Calling `credit_score_at_application` simply fine without asking how the column is maintained: the name is a
  promise the data may not keep.
- Missing that the postcode average includes the row's own label (the same in-sample problem as exercise 2 (b)).

## Exercise 2 (code)

**Solution:** [`b1_4_data_leakage_ex2.py`](b1_4_data_leakage_ex2.py).

```text
== exercise 2: data ==
rows 3000  postcodes 500  positive rate 0.3860  train 2250  test 750  training rows per postcode ~4.5

== exercise 2: three encodings ==
x only (no postcode)                     train AUC 0.6383   test AUC 0.6420
(a) means over all rows                  train AUC 0.7667   test AUC 0.7542
(b) means over training rows, in-sample  train AUC 0.8006   test AUC 0.5332
(c) out-of-fold on training rows         train AUC 0.6387   test AUC 0.6407
from-scratch AUC matches roc_auc_score on every test score: OK
```

The postcode has no effect, so the honest answer is "the postcode adds nothing": test AUC about 0.64 with or
without it.

- **(a) Contamination.** Each test row's encoding is the mean label of its postcode *including itself* (and other
  test rows). With about 6 rows per postcode, a row's own label moves its postcode mean by about 1/6, so the encoding
  partly copies the label: test AUC 0.75, a fake gain of 0.11.
- **(b) Target leakage inside training.** Each training row's encoding includes its own label, so on the training
  rows the feature looks strongly predictive (train AUC 0.80). The model trusts it; on test rows, whose labels are not
  in the means, it is pure noise, and the model's heavy weight on it *hurts*: test AUC 0.53, worse than ignoring the
  postcode. The giveaway is the large train/test gap.
- **(c) Out of fold.** No row's encoding uses its own label, so the training rows see the feature as it will
  behave on new data: useless. The model learns a small weight, and train and test AUC agree with the model
  without the postcode.

The general rule: any encoding computed from labels must be fitted the way a model is, on rows other than the ones it
is applied to. scikit-learn's `TargetEncoder` does the out-of-fold step inside `fit_transform` (VERIFY as of 2026-10,
scikit-learn 1.9.1).

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Data with a no-effect high-cardinality category and one weak feature; `SEED = 0` | 2 |
| Three encodings implemented correctly; out-of-fold encoding never uses a row's own label | 5 |
| Train and test AUC reported for all three plus the no-postcode model | 3 |
| Explains (a) as contamination and (b) as in-sample target leakage, and why (b) hurts the test score | 4 |
| From-scratch AUC checked against `roc_auc_score` | 2 |

**Likely mistakes:**

- Encoding the test rows out of fold too: harmless but unnecessary; test rows should use the means from all
  training rows, like any fitted transformer.
- Not handling postcodes unseen in a fold (the encoder must fall back to a global mean, not `NaN`).
- Seeing (b)'s train AUC of 0.80 and concluding the postcode is informative: the train/test gap is the signal.
- Adding smoothing and concluding the leak is fixed. Smoothing (blending with the global mean) reduces the
  damage for small categories but does not remove the row's own label from its encoding.

## Exercise 3 (think)

**Strong answer:** Two of the top features are recorded *because a clinician already suspected sepsis*:
antibiotics are given and a lactate test is ordered as part of the sepsis workup. They are target leakage relative to
the goal (warn before doctors suspect it). The model mostly learns "a doctor already thinks this is sepsis", which is
useless as an early warning and will look excellent offline because the label (a sepsis diagnosis) is downstream of
the same suspicion. Checks: (1) for every feature, the timestamp at which it was recorded relative to the prediction
time and to the clinician's first suspicion; (2) retrain without the treatment and order features (or with only data
recorded before the first suspicion) and compare AUC; (3) look at lead time: for true positives, how long before the
diagnosis did the model's score cross the alarm threshold, and was that before the antibiotics? (4) single-feature AUCs
for the suspect features. Changes to the data build: define a prediction time for each training example (for example
every hour of a stay, or 6 hours before sepsis onset for cases), and build features only from measurements charted
before that time; exclude actions taken by clinicians (orders, treatments) as features, or include only those taken
before the prediction time and accept that the model then partly copies clinicians; split by patient and by time
(train on earlier admissions) so the evaluation mirrors deployment.

**Weak answer:** "0.97 is great; antibiotics are a strong sign of infection, so the feature is valid. Deploy it and
monitor." Weak because it does not ask when the features are recorded or what the model is for; it would deploy a
model that only fires after the clinicians already acted.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Identifies antibiotics and lactate orders as leakage relative to the prediction goal | 3 |
| At least two concrete checks (timestamps vs prediction time, ablation, lead time, single-feature AUC) | 3 |
| A data-build fix: explicit prediction times and features only from before them | 2 |
| Splits by patient and/or time, or another sound evaluation change | 2 |

## Scoring

Total 40 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → B1.4 "The math" (AUC by
ranks) and "Intuition" (three families); exercise 2 → B1.4 "Train/test contamination", B1.3 (fitted transformers) and
B4.3 (cross-validation); exercise 3 → B1.4 "How to catch leaks" and B1.1 (framing the prediction time).
