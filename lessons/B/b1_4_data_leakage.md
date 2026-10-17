# B1.4 Data leakage

**Core** · Part B, week 1 (course week 07) · ~2 h · Lab:
[`labs/B/b1_4_data_leakage.py`](../../labs/B/b1_4_data_leakage.py)

## Why it matters

**Data leakage** is information reaching training or evaluation that will not be available when the model
makes real predictions. It is the most common reason a model that scored brilliantly offline fails in
production, and it fails silently: the code runs, the score is high, and nobody finds out until the
business numbers do not move. Every symptom is the same, a score that is too good, so the defence is
not a library but a habit: for every feature, ask when its value is known; for every step that learns,
ask which rows it learned from; for every split, ask whether it looks like the future the model will
face. This lesson plants three kinds of leak in one dataset, measures how much each inflates the score,
and builds the checks that catch them.

## Prerequisite check

- Can you explain what the validation and test sets are for, and why the test set is used once? If not,
  revisit [B1.1](b1_1_ml_workflow.md).
- Can you explain why a scaler or imputer must be fitted on training rows, and what a `Pipeline` does? If
  not, revisit [B1.3](b1_3_feature_engineering.md) and [B1.2](b1_2_data_cleaning.md).
- Can you say why the best of many noisy scores is biased upwards? If not, revisit
  [B1.1](b1_1_ml_workflow.md) ("Why reusing a set is optimistic") and
  [A6.3b](../A/a6_3b_multiple_comparisons_ab_tests.md).
- Can you compute a sum of ranks, with ties given their average rank? If not, revisit
  [A6.1](../A/a6_1_descriptive_statistics_eda.md) (Spearman correlation).

## Intuition

### The question behind every leak: what is known at prediction time?

Draw the timeline of one prediction. Our example: a subscription service wants to know, on the day a
customer signs up, whether they will cancel within 90 days, so that onboarding staff can help the risky
ones. Anything recorded during the first week is known at the prediction (a week after signup, say);
anything recorded later is not. Leakage is any path by which the *later* information, or information
from the rows you evaluate on, gets into the model or its score. There are three families.

**1. Target leakage: a feature that encodes the answer.** The customer table has a column
`retention_call`: 1 if the retention team phoned the customer. The team phones people *who asked to
cancel*. So the column is mostly a copy of the label, recorded after the outcome. A model trained with
it scores 0.94 AUC; the same model without it scores 0.69. At prediction time no retention call has
happened, so the 0.94 was never real. Real examples: "account closed date", "refund issued", "number of
complaints in the month" (when the month includes the cancellation), a diagnosis code entered after the
test that diagnosed the disease, "total charges to date" in a churn table built after customers left.

**2. Train/test contamination: a fitting step that saw the test rows.** Every step that learns from data
(scaler, imputer, feature selection, target encoding, resampling) must learn from training rows only. If
it learns from all rows, the test rows influence the model before they are used to score it. How much
this inflates the score depends on how much the step learns *about the target*:

- A scaler fitted on all rows learns two numbers per column, nothing about the labels. In the lab the
  score does not change at all (difference 0.0000). It is still a bug: the habit that allows it allows
  the next one.
- Feature selection fitted on all rows learns which columns correlate with the labels, *including the
  test labels*. On 5,000 columns of pure noise, keeping the 20 that correlate best with the label over all
  rows yields a test AUC of 0.88, on data where the true AUC of any model is 0.5. Done inside a pipeline
  (selection learned on training rows only), the test AUC is 0.49.

Other forms: duplicate or near-duplicate rows on both sides of the split, rows of one patient or customer
on both sides (B1.1, exercise 2), oversampling before splitting (B4.4), and choosing models on the test set
(B1.1).

**3. Temporal leakage: training on the future.** If the model will predict for future customers, the
evaluation must look like that: train on the past, test on later data. A random split mixes months, so
every test customer has training neighbours from the same month. In the lab, each signup month has its
own churn level (marketing campaigns bring better or worse customers), and the model can learn it from the
month column. Under a random split that knowledge is available for every test row: AUC 0.80. Under a time
split (train on 18 months, test on the next 6) the test months were never seen and the AUC is 0.68, the
same as a model without the month column at all (0.67). At deployment, a new month's churn level is unknown
until its customers have had 90 days to churn, so 0.68 is the honest number. Related forms: features computed
over a window that extends past the prediction time ("logins in the first 90 days" for a prediction made at
day 7), rolling averages computed over the whole history, and normalizing by a statistic of the full series.

> **Bridge:** a leaky evaluation is a unit test whose fixture contains the expected output, written there by
> the code under test: the test passes, and proves nothing. **Where it breaks:** in a codebase you can find
> such a test by reading the code. Leakage usually lives in what a column *means* and when it was recorded,
> which no code review of the training script reveals. You find it by asking the data owner how each column
> is produced.

### How to catch leaks

1. **Be suspicious of a great score.** Compare with the baselines (B1.1) and with what the domain makes
   plausible. If a churn model is at 0.98 AUC, look for the leak before celebrating.
2. **Score each feature alone.** A `leakage_check` computes every feature's AUC on its own; one feature that
   nearly separates the classes is either a breakthrough or, far more often, a leak. In the lab it flags
   `retention_call` (0.91) and nothing else.
3. **Write a feature timeline.** For every column: where does it come from, when is its value fixed, and is
   that before the prediction time? This is the leakage audit P2 asks for.
4. **Drop a feature and watch.** If removing one column makes the score collapse, find out why that column
   is so informative.
5. **Fit only inside a pipeline**, and evaluate with a split that mirrors deployment (by time, by group).
6. **Compare splits.** A large gap between a random split and a time or group split means the random
   split is leaking.

### Scoring: ROC-AUC in one paragraph

The lab scores models with **ROC-AUC** (B4.1 covers it fully). The model outputs a score per row (a
probability of churn). Take every pair of one positive row (churned) and one negative row (stayed): the AUC
is the share of those pairs where the positive row got the higher score, counting ties as half. 1.0 means
every churner is ranked above every non-churner, 0.5 means the ranking is random, and 0 means it is perfectly
reversed. It depends only on the *order* of the scores, not their values, and not on any threshold.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{s}$ | model scores, one per row | `(n,)` |
| $\mathbf{y}$ | labels, 1 = positive (churned), 0 = negative | `(n,)` |
| $n_+$, $n_-$ | number of positive and negative rows | ints |
| $r_i$ | rank of $s_i$ among all $n$ scores, 1 = smallest; tied scores share their average rank | scalar |
| $R_+$ | sum of the ranks of the positive rows | scalar |
| $U$ | number of (positive, negative) pairs ranked correctly, ties counting $\frac12$ | scalar |
| $\mu_{\text{tr}}, \mu_{\text{all}}$ | a column's mean over training rows, over all rows | scalars |
| $n_{\text{tr}}, n_{\text{te}}$ | number of training and test rows | ints |

**AUC as a pair count.**

$$\mathrm{AUC} = \frac{U}{n_+ n_-}, \qquad U = \#\{(i, j) : y_i = 1,\ y_j = 0,\ s_i > s_j\} + \tfrac12\,\#\{(i, j) : y_i = 1,\ y_j = 0,\ s_i = s_j\}.$$

Counting all $n_+ n_-$ pairs is slow for big $n$; ranks do it in one sort. Sort all scores ascending and give
them ranks $1, \dots, n$. Take the positive rows in increasing order of score; the $j$-th of them has rank $r_{(j)}$, so
$r_{(j)} - 1$ rows lie below it, of which $j - 1$ are positives. The other $r_{(j)} - j$ are negatives it beats. Summing over the
$n_+$ positives,

$$U = \sum_{j=1}^{n_+}\big(r_{(j)} - j\big) = R_+ - \frac{n_+(n_+ + 1)}{2},$$

because $1 + 2 + \dots + n_+ = n_+(n_+ + 1)/2$. With average ranks for ties, a tied (positive, negative) pair contributes exactly
$\frac12$. This is the Mann–Whitney $U$ statistic; the AUC is $U$ divided by the number of pairs.

**Why the scaler leak is small.** Fitting on all rows uses $\mu_{\text{all}} = \frac{n_{\text{tr}}\,\mu_{\text{tr}} + n_{\text{te}}\,\mu_{\text{te}}}{n_{\text{tr}} + n_{\text{te}}}$ instead of $\mu_{\text{tr}}$: the shift is
$\frac{n_{\text{te}}}{n}(\mu_{\text{te}} - \mu_{\text{tr}})$, a fraction of a random difference between two samples, and it carries no information about the
test *labels*. For a linear model a shifted or rescaled column is absorbed by the intercept and the weight, so
the ranking of the test rows barely moves. The contamination that matters is any step whose fitted values
depend on $\mathbf{y}$.

**Why feature selection on all rows is large.** With $n = 200$ rows, the sample correlation between a pure-noise
column and the labels is roughly normal with standard deviation $1/\sqrt{n} = 0.071$. Among 5,000 such columns, the 20
largest $\lvert r\rvert$ lie beyond the point where only $20/5000 = 0.4\%$ of a two-sided normal remains, about 2.88 standard deviations:
$\lvert r\rvert \approx 0.20$. Those correlations were computed with the test rows' labels included, so part of each one *is* the test
rows' noise lining up with the test labels. The model then "predicts" that noise. This is the best-of-$k$ effect
from B1.1, with $k = 5000$ and the test set inside the selection.

**Why a random split leaks across time.** Write the churn log-odds of a customer as their features' effect plus
a month effect $c_{\text{month}}$. Under a random split, about 25% of each month's customers land in test and 75% in
training, so the model estimates every $c_{\text{month}}$ from the training customers of that month and applies it to the test
customers of the same month. Under a time split, the test months have no training rows: their $c_{\text{month}}$ cannot be
learned, exactly as in deployment.

## Worked example by hand

**AUC.** Scores $0.9, 0.8, 0.4, 0.3$ with labels $1, 0, 1, 0$. Positives: 0.9 and 0.4; negatives: 0.8 and 0.3. The four pairs:
$(0.9, 0.8)$ correct, $(0.9, 0.3)$ correct, $(0.4, 0.8)$ wrong, $(0.4, 0.3)$ correct. $U = 3$, AUC $= 3/4 = 0.75$. With ranks: sorted ascending
$0.3, 0.4, 0.8, 0.9$ get ranks $1, 2, 3, 4$; the positives have ranks 4 and 2, $R_+ = 6$, $U = 6 - 2 \times 3/2 = 3$. Same answer.

If the scores were $0.9, 0.4, 0.4, 0.3$ (the negative's 0.8 became 0.4, tied with a positive): pairs $(0.9, 0.4)$ correct, $(0.9, 0.3)$
correct, $(0.4, 0.4)$ tie $= \frac12$, $(0.4, 0.3)$ correct: $U = 3.5$, AUC $= 0.875$.

**The scaler leak.** Training values $1, 2, 3$, one test value 10. Fitted on training rows: mean 2, standard
deviation $\sqrt{(1 + 0 + 1)/3} = 0.816$, so the test value scales to $(10 - 2)/0.816 = 9.80$, an obvious outlier. Fitted on all four
values: mean 4, deviations $-3, -2, -1, 6$, variance $(9 + 4 + 1 + 36)/4 = 12.5$, standard deviation 3.54, so the test value scales to
$6/3.54 = 1.70$, unremarkable. The test row has changed how the test row looks. With hundreds of rows the effect is
diluted to almost nothing (the lab measures it); with one extreme test batch, or a step that uses labels, it is not.

**A time split.** Eight months of data, January to August. A random 25% test split might pick January and June:
the model is trained on February to May and on July and August, around the test months on both sides. A time
split tests on July and August, trained only on January to June, which is the situation the deployed model
will be in.

```text
== worked example by hand ==
scores [0.9, 0.8, 0.4, 0.3]   labels [1, 0, 1, 0]
(positive, negative) pairs: [(0.9, 0.8), (0.9, 0.3), (0.4, 0.8), (0.4, 0.3)]
positive ranked higher in 3 of 4 pairs -> AUC = 0.75
scaler fitted on train [1, 2, 3]: mean 2.0000 sd 0.8165 -> test value 10 scales to 9.7980
scaler fitted on train + test [1, 2, 3, 10]: mean 4.0000 sd 3.5355 -> test value 10 scales to 1.6971
random split, test months: ['Jan', 'Jun']   time split, test months: ['Jul', 'Aug']
```

## Key plot

![Grouped bar chart of test ROC-AUC, leaky against honest evaluation, for four cases: target leak 0.94 against 0.69, scaler fitted on all rows 0.66 against 0.66, feature selection on all rows 0.88 against 0.49, random split of time-ordered data 0.80 against 0.68; a dashed line marks chance at 0.5](img/b1_4_leak_vs_clean_auc.png)

Three of the four leaks add 0.13 to 0.39 of AUC; the scaler leak adds nothing measurable here. The worst one,
feature selection, turns pure noise into an apparently strong model.

## From scratch in NumPy

**The data.** 4,000 synthetic customers who signed up between January 2024 and December 2025. The label is
"cancelled within 90 days" (34% did). Honest features: `first_week_logins`, `onboarding_done`, `discount`, `plan`, and
`signup_month`. The leaky one: `retention_call`, logged for 85% of the customers who cancelled and 4% of the others.
Each signup month has a random churn level, so monthly churn rates range from 5% to 71%:

```text
== data ==
customers: (4000, 7)   churn rate 0.3445
  signup_month   plan  first_week_logins  onboarding_done  discount  retention_call  churned
0      2025-09    pro                  5                1         0               0        0
1      2025-04    pro                  3                1         0               0        0
2      2025-01  basic                  1                1         0               0        0
3      2024-07    pro                  1                1         1               1        1
4      2024-08  basic                  1                0         0               0        0
churn rate by signup month: min 0.0452  max 0.7068  (sd across months 0.1898)
```

**AUC by ranks.** Sort, assign average ranks to ties, apply $U = R_+ - n_+(n_+ + 1)/2$:

```python
def auc_score(y, score):
    """ROC-AUC by ranks (Mann-Whitney U), ties counted half. y (n,) in {0, 1}, score (n,) -> scalar."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ...                                   # walk runs of equal scores, give each run its average rank
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))
```

```text
== from scratch: AUC ==
hand example: auc_score = 0.7500   with a tie [0.9, 0.4, 0.4, 0.3]: 0.8750
random scores on 1000 rows: 0.4860   perfect scores: 1.0000   reversed: 0.0000
```

Random scores give about 0.5, a perfect ranking 1.0, a reversed one 0.0.

**`leakage_check`.** Score each column alone: numeric columns by their own values, categories by the churn rate
of their group:

```text
== from scratch: leakage_check ==
             feature  auc_alone   flag
0     retention_call     0.9080   True
1       signup_month     0.7350  False
2  first_week_logins     0.6188  False
3    onboarding_done     0.5879  False
4               plan     0.5784  False
5           discount     0.5399  False
```

`retention_call` is flagged at 0.91. (The check reports $\max(\mathrm{AUC}, 1 - \mathrm{AUC})$ so that a feature that predicts the
label *backwards* is caught too.) `signup_month` scores 0.74 alone: real information about each month, but also
the column through which the temporal leak flows. The group rates are computed on the same rows they score, so
the check is optimistic for columns with many categories; it is a smoke detector, not a verdict. A flag means:
find out when that value is recorded.

**Leak 1, target leakage.** The same pipeline (one-hot `plan`, scale numeric columns, logistic regression, which
B2.1 explains) with and without `retention_call`, on a stratified 75/25 random split:

```text
== leak 1: target leakage ==
test AUC with retention_call 0.9386   without 0.6898
at prediction time (the signup day) no retention call has happened: the 0.9+ was never available
```

**Leak 2, contamination.** (a) A scaler fitted on all 4,000 rows against one fitted on the 3,000 training rows; (b)
feature selection on 5,000 noise columns, inside and outside the pipeline, averaged over 20 repetitions:

```text
== leak 2: train/test contamination ==
(a) scaler fitted on all rows: test AUC 0.6592   fitted on train only: 0.6592   difference +0.0000
    means used: all rows [5.0292, 0.5985, 0.2888]   train rows [5.0173, 0.5983, 0.2867]
    largest change in any test probability: 0.000020
(b) 5000 pure-noise features, 200 rows, keep the best 20; mean over 20 repeats:
    selected on all rows: test AUC 0.8782 (min 0.7792)   selected inside the pipeline: 0.4897   truth: 0.5
```

The scaler's means differ in the second decimal, and no test probability moves by more than 0.00002: the ranking,
and so the AUC, is unchanged. The selection leak is the opposite: 0.88 on average and never below 0.78 in 20
repetitions, from features that carry no information at all. The fix is the same line in both cases: put the
step inside the `Pipeline`, so that `fit` sees only training rows.

**Leak 3, temporal.** `signup_month` joins the features (one-hot). The time split trains on January 2024 to June
2025 and tests on July to December 2025; the random split is given the same test size for a fair comparison:

```text
== leak 3: temporal leakage ==
train 2962 rows (18 months) / test 1038 rows (6 later months)
random split: test AUC 0.8030   time split: 0.6756   time split without signup_month: 0.6728
test months seen in training: random split 1.0000   time split 0.0000
```

Under the random split, 100% of test rows come from months that also appear in training; under the time split,
0%. The random split scores 0.13 higher, and under the time split the month column adds nothing (0.6756 with it,
0.6728 without), because a month that has not happened yet has no estimated effect: the one-hot encoder maps it
to all zeros (B1.3).

## With scikit-learn

`sklearn.metrics.roc_auc_score(y, score)` is the library AUC; it also handles ties as half:

```text
== with scikit-learn ==
roc_auc_score matches auc_score on 5 cases (ties included): OK
roc_auc_score(hand example) = 0.7500
```

The leak fixes are all scikit-learn structure rather than a single function:

```python
pipe = Pipeline([("sel", SelectKBest(f_classif, k=20)), ("clf", LogisticRegression())])
pipe.fit(X[train_rows], y[train_rows])          # selection and model both learn from training rows only
# Cross-validation refits the WHOLE pipeline in every fold (B4.3):
# cross_val_score(pipe, X, y, cv=5)
# Time-ordered data: TimeSeriesSplit; several rows per entity: GroupKFold (B4.3).
```

## In practice

- **When to use it:** run the checks on every project before trusting any score: a feature timeline for each
  column, single-feature scores, a pipeline for every fitted step, and a split that mirrors deployment. Re-run
  them whenever a new data source or feature is added.
- **Failure modes:**
  - *A feature recorded after the outcome.* You notice a single feature with near-perfect AUC, or a feature
    importance (B3.4) that dominates everything. Ask the data owner when and why the column is written.
  - *Preprocessing fitted before the split.* You notice `fit` or `fit_transform` called on the full dataset in
    the code. Move the step into the pipeline; for target encoding and feature selection this changes the
    score a lot.
  - *Random splits of time-ordered or grouped data.* You notice a large gap between random-split and
    time-split (or group-split) scores, or a production score far below the offline one. Split by time or by
    group from the start.
  - *Features aggregated over the wrong window.* "Logins in the first 90 days" for a prediction on day 7. You
    notice by writing down the window of every aggregate next to the prediction time.
  - *Leaks through the label definition.* The label is defined using a field that the features also derive
    from (for example both from the same status table updated at cancellation). You notice by tracing both back
    to their source tables.
  - *Duplicates across the split.* The same row, or the same entity, in train and test. Deduplicate (B1.2)
    and split by entity.
- **Defaults that work:** every fitted step inside a `Pipeline`; `TimeSeriesSplit` (or a manual cut-off date)
  whenever the model predicts the future; `GroupKFold` when rows share an entity; a `leakage_check` that flags any
  single feature above 0.9 AUC (or above the best full model's score) for review; a written feature timeline as
  part of every project.

## Exercises

Save your work to `my-work/B/b1_4_data_leakage.md` (or `.py`) and ask for a review.

1. **By hand:** (a) Compute the AUC of the scores $0.8, 0.7, 0.7, 0.5, 0.3, 0.2$ with labels $1, 1, 0, 0, 1, 0$, both by listing the
   pairs and with the rank formula (give tied scores their average rank). (b) A column has training values $10, 12,
   14, 16$ and test values $30, 32$. Standardize the test values with a scaler fitted on the training rows, and with one
   fitted on all six rows. (c) A model predicts, at the moment a loan application is submitted, whether the loan
   will default within a year. Classify each feature as target leakage, temporal leakage, fine, or "depends", and
   say what you would ask: `applicant_income`, `number_of_late_payments` (on this loan), `credit_score_at_application`,
   `loan_officer_decision_notes`, `average_default_rate_of_applicants_from_same_postcode` (computed over the whole
   dataset).
2. **Code:** Target encoding (replacing a category by the average label of its rows) is a common leak.
   Generate 3,000 rows with a high-cardinality category (500 postcodes) that has **no** real effect on a binary
   label, plus one weakly useful numeric feature (`SEED = 0`), and split 75/25. Fit a logistic regression on the
   numeric feature plus the target-encoded postcode, where the encoding's label means are computed (a) over all
   rows, training and test; (b) over the training rows, and used for those same training rows; (c) out of fold:
   split the training rows into 5 folds and encode each fold with the label means of the other 4 (test rows use the
   means of all training rows). For each, report the training AUC and the test AUC next to a model without the
   postcode, explain every gap, and check your from-scratch AUC against `roc_auc_score`.
3. **Think:** A hospital team reports a sepsis-prediction model with AUC 0.97. The top features are
   `antibiotics_administered`, `lactate_test_ordered` and `heart_rate`. The model is meant to warn nurses before
   doctors suspect sepsis. What do you suspect, how would you check it, and what would you change about how the
   training data is built?

## Takeaways

1. Leakage is information unavailable at prediction time reaching training or evaluation; it inflates scores
   silently and is found by asking when each value is known.
2. Target leakage: a feature recorded after (or because of) the outcome. Catch it with single-feature scores
   and a feature timeline.
3. Contamination: a fitted step that saw evaluation rows. Harmless-looking for a scaler, severe for anything
   that learns from labels (feature selection, target encoding). Fit everything inside a `Pipeline`.
4. Temporal leakage: random splits of time-ordered data and features computed over windows that reach past
   the prediction time. Evaluate with a split that looks like deployment.
5. A score that is too good is a bug report: compare with baselines, with a time or group split, and with
   the model minus its best feature.

**What builds on this:** [B4.3](b4_3_cross_validation_tuning.md) (`TimeSeriesSplit`, `GroupKFold`, pipelines in
cross-validation); [B4.4](b4_4_class_imbalance.md) (resampling only inside training folds);
[B3.4](b3_4_interpretability_shap.md) (a dominant feature as a leak signal);
[B6.3](b6_3_time_series_forecasting.md) (backtesting forecasts); P2 (the leakage audit, from M0 to the final
report) and every later project's audit.

**References:** S. Kaufman, S. Rosset, C. Perlich, O. Stitelman, "Leakage in Data Mining: Formulation, Detection,
and Avoidance", *ACM Transactions on Knowledge Discovery from Data* 6(4), 2012. S. Kapoor and A. Narayanan,
"Leakage and the Reproducibility Crisis in Machine-Learning-Based Science", *Patterns* 4(9), 2023. scikit-learn
User Guide, "Common pitfalls and recommended practices" (VERIFY link:
https://scikit-learn.org/stable/common_pitfalls.html). H. B. Mann and D. R. Whitney, "On a Test of Whether One of
Two Random Variables Is Stochastically Larger than the Other", *Annals of Mathematical Statistics* 18(1), 1947.
