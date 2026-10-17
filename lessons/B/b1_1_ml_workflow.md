# B1.1 The ML workflow

**Core** · Part B, week 1 (course week 07) · ~2 h · Lab:
[`labs/B/b1_1_ml_workflow.py`](../../labs/B/b1_1_ml_workflow.py)

## Why it matters

Most machine-learning projects that fail in production fail before the first model is trained: the
target was not what the business needed, the metric did not match the cost of mistakes, nobody
knew what score a trivial rule gets, or the reported score came from data the model had already
seen. This lesson is the frame every later model in the course sits in: turn a request into a
problem statement, split the data so that one part stays honest, and measure the **baselines** that
every model must beat. A model that has never been compared with "predict the average" has not been
evaluated.

## Prerequisite check

- Can you select rows of a DataFrame or array with an integer index array (`X[idx]`, `df.iloc[idx]`)?
  If not, revisit [A1.3](../A/a1_3_pandas.md) and [A1.2](../A/a1_2_numpy_arrays.md).
- Can you fit a line by least squares and say what a residual is? If not, revisit
  [A2.4](../A/a2_4_projections_least_squares.md).
- Can you say why the mean minimizes squared error and the median minimizes absolute error? If not,
  revisit [A5.1](../A/a5_1_random_variables_expectation.md).
- Can you compute the standard error of a proportion, $\sqrt{p(1 - p)/n}$? If not, revisit
  [A6.2](../A/a6_2_confidence_intervals_bootstrap.md).
- Can you summarize a column with its mean, median and quantiles before modelling it? If not, revisit
  [A6.1](../A/a6_1_descriptive_statistics_eda.md).

## Intuition

### From a request to a problem statement

Requests arrive vague: "use ML to reduce churn". Before touching data, write down six things:

| Question | Churn example |
|---|---|
| **Unit of prediction**: one prediction is about what? | one customer, once a month |
| **Target**: the number or label you predict, defined exactly | 1 if the customer cancels within the next 30 days, else 0 |
| **Prediction time**: when is the prediction made, and what is known then? | the 1st of each month, using data up to the end of the previous month |
| **Action**: who uses the prediction, and how? | the retention team calls the 200 highest-risk customers |
| **Metric**: how is a prediction judged, and why that one? | of the 200 called, how many would really have left (precision at 200), because the team's time is the cost |
| **Baseline**: what does a trivial rule score on that metric? | "call the 200 with the fewest logins last month" |

Each line prevents a specific failure. Without a *prediction time* you will happily use features that
are only known after the fact (B1.4). Without an *action* you cannot choose a metric, and without a
*baseline* you cannot tell whether the model is worth its maintenance cost. Sometimes the honest
outcome of framing is "a SQL rule does this well enough"; that is a good result, reached cheaply.

### What "generalize" means

A model is useful only on rows it has not seen: next month's customers, tomorrow's houses. A model
**generalizes** when its error on new data from the same source is close to its error on the data it
learned from. The easiest way to see the difference is a model that cannot generalize at all, a
lookup table. A **1-nearest-neighbour** "model" answers every query by copying the target of the
closest training row. On its own training rows, the closest row is the row itself, so it is perfect:
zero error, 100% accuracy. On new rows it is only as good as "the most similar thing I have seen",
which in the lab is barely better than predicting the average. Training error tells you how well a
model memorized; only error on held-out data tells you how well it will do.

### Three sets, three jobs

So you hold data back. Usually twice:

- **Training set** (about 60%): the model's parameters are fitted on it.
- **Validation set** (about 20%): you compare models and settings on it (which features, which
  algorithm, how much regularization) and keep the best. Every comparison is a decision informed by
  this data, so after many comparisons the validation score of the winner is optimistic too.
- **Test set**, or **holdout set** (about 20%): used *once*, at the end, to report how the chosen
  model will do. It answers "how good is the thing we ship", not "which thing should we ship".

Why can't the validation set be the test set? Because choosing is a form of fitting. If you try 100
models and keep the one with the best score on a set of 100 rows, you have selected the model that
got luckiest on those 100 rows. The key plot below shows it with models that are pure coin flips:
the "best" of 100 scores 0.63 on the reused set, while every one of them is 0.5 on new data.

> **Bridge:** the validation set is your staging environment, where you iterate until things look
> right; the test set is the one acceptance run before release. **Where it breaks:** in software you
> can fix a bug and rerun the acceptance tests, and they are still a valid check. In ML, every time
> a test score influences a decision (keep this feature, try another model), the test set leaks a
> little into your choices, and after enough rounds it measures how well you fitted the test set.

**Stratification.** In a classification problem with 37% of one class, a random 20% test set of 114
rows could easily contain 30% or 44% of it, and the score would move with the mix. A **stratified**
split shuffles and cuts each class separately, so every set keeps the class shares of the whole. It
is the default for classification.

### Baselines

A **baseline** is the simplest reasonable predictor. Three are worth computing every time:

1. **Predict the training mean** (regression) or **the majority class** (classification). This is
   the score of knowing nothing about the row. A classifier with 63% accuracy on a dataset that is
   63% one class has learned nothing.
2. **The median** for regression when the metric is absolute error, because the median is the
   constant that minimizes it.
3. **A one-feature rule**: the single most obviously useful column, used in the simplest way (a
   threshold, or an average per bin). If your model barely beats "patients with a high BMI progress
   faster", the model is not adding much.

Report every later model next to these numbers. In Part B you will see the bmi rule below reappear
as the bar that linear regression (B1.5) must clear.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ | the dataset: $n$ rows, each a feature vector and a target | — |
| $\mathbf{x}_i$ | features of row $i$ | `(d,)` |
| $y_i$ | target of row $i$ (a number, or a class label) | scalar |
| $f$ | a model: a function from features to a prediction, $\hat{y} = f(\mathbf{x})$ | — |
| $\ell(\hat{y}, y)$ | loss of one prediction, e.g. $(\hat{y} - y)^2$ or $[\hat{y} \ne y]$ (1 if wrong, 0 if right) | scalar |
| $R(f)$ | **true risk**: expected loss on new data from the same source | scalar |
| $\hat{R}_S(f)$ | **empirical risk** on a set $S$ of rows: the average loss there | scalar |
| $n_{\text{tr}}, n_{\text{val}}, n_{\text{test}}$ | sizes of the three sets | ints |
| $p$ | true accuracy of a classifier | scalar |
| $k$ | number of models compared on one set | int |
| $c$ | a constant prediction (a baseline) | scalar |
| $\bar{y}$ | mean of the training targets | scalar |

**Risk.** What we care about is the average loss on future rows:

$$R(f) = \mathbb{E}\big[\ell(f(\mathbf{X}), Y)\big],$$

where $(\mathbf{X}, Y)$ is a new row drawn from the same source as the data. We cannot compute it, but
we can average the loss over a set of rows $S$ that we have:

$$\hat{R}_S(f) = \frac{1}{|S|}\sum_{i \in S} \ell(f(\mathbf{x}_i), y_i).$$

If the rows in $S$ were drawn independently of everything that produced $f$, then $\hat{R}_S(f)$ is an
average of $|S|$ independent draws of the loss, so by A5.1 its expectation is exactly $R(f)$ (it is
**unbiased**), and by A5.4 its spread shrinks like $1/\sqrt{|S|}$. That is the whole argument for a
holdout set.

**Why training error is optimistic.** On the training set the condition fails: $f$ was chosen to make
$\hat{R}_{\text{train}}(f)$ small. The 1-nearest-neighbour model drives it to exactly 0 whatever the
true risk is. The gap $R(f) - \hat{R}_{\text{train}}(f)$ is the **generalization gap**; B4.5 studies how
it depends on model complexity and data size.

**Why reusing a set is optimistic.** Suppose $k$ models all have true accuracy $p = 0.5$ and you measure
each on the same $n_{\text{test}} = 100$ rows. Each measured accuracy is $p$ plus noise with standard
error

$$\mathrm{SE} = \sqrt{\frac{p(1 - p)}{n_{\text{test}}}} = \sqrt{\frac{0.25}{100}} = 0.05$$

(A6.2). Reporting the best of them reports the *maximum* of $k$ noisy numbers. The maximum of $k$
independent standard normal draws has an expected value of about 1.54 for $k = 10$, 2.51 for $k = 100$
and 3.24 for $k = 1000$ (a standard result for order statistics; the lab checks it by simulation). So
the expected reported accuracy is about $0.5 + 2.51 \times 0.05 = 0.625$ for $k = 100$, although no model
is better than a coin. The bias grows slowly with $k$ (like $\sqrt{2\log k}$) and shrinks with the size of
the set (like $1/\sqrt{n_{\text{test}}}$); it never goes away.

**The constant baselines are minimizers.** For the mean squared error of a constant $c$,

$$\frac{d}{dc}\,\frac{1}{n}\sum_{i=1}^{n}(y_i - c)^2 = -\frac{2}{n}\sum_{i=1}^{n}(y_i - c) = 0 \quad\Longrightarrow\quad c = \frac{1}{n}\sum_{i=1}^{n} y_i = \bar{y}.$$

So "predict the training mean" is the best you can do with no features under squared error. Under
absolute error $\frac1n\sum\lvert y_i - c\rvert$ the best constant is the median (A5.1), and under 0/1 loss it is the
majority class.

**$R^2$ compares a model with the mean baseline.** On a set $S$,

$$R^2 = 1 - \frac{\mathrm{SSE}}{\mathrm{SST}} = 1 - \frac{\sum_{i \in S}(y_i - \hat{y}_i)^2}{\sum_{i \in S}(y_i - \bar{y}_S)^2},$$

where $\bar{y}_S$ is the mean of the targets *in $S$*. $R^2 = 0$ means "as good as predicting the mean of $S$",
1 means perfect. On a validation set, the training mean $\bar{y}$ is not $\bar{y}_S$, so the mean baseline
scores slightly *below* zero: it pays for not knowing the validation mean. Negative $R^2$ means worse
than that constant.

**Split sizes.** With fractions $f_{\text{val}}$ and $f_{\text{test}}$, cut $n_{\text{test}} = \operatorname{round}(f_{\text{test}}\,n)$ and
$n_{\text{val}} = \operatorname{round}(f_{\text{val}}\,n)$ from a shuffled order and give the rest to training. A stratified split does
this within each class, so the class shares match to within one row per class.

## Worked example by hand

Ten rows, numbered 0 to 9 (0-based, as in the code). Suppose the shuffle produced the order
$7, 2, 9, 4, 0, 5, 8, 1, 6, 3$. With 20% test and 20% validation, $n_{\text{test}} = n_{\text{val}} = \operatorname{round}(0.2 \times 10) = 2$:
the first two shuffled rows (7, 2) are the test set, the next two (9, 4) the validation set, and the
remaining six (0, 1, 3, 5, 6, 8) the training set.

**Mean baseline.** The training targets are $3, 5, 8, 6, 7, 7$, with mean $36/6 = 6$. On the validation targets
$5$ and $10$, the errors are $5 - 6 = -1$ and $10 - 6 = 4$. MSE $= (1 + 16)/2 = 8.5$; MAE $= (1 + 4)/2 = 2.5$.

**Majority baseline.** The training labels are $0, 1, 1, 0, 1, 1$: four 1s and two 0s, so predict 1 for
everyone. The validation labels are $0, 1$, so the accuracy is $1/2 = 0.5$.

**One-feature rule.** Sort the training rows by their feature value and write the label next to it:
$(1, 0), (2, 1), (3, 1), (4, 0), (6, 1), (8, 1)$. Candidate thresholds are the midpoints between neighbours:
1.5, 2.5, 3.5, 5, 7. For "predict 1 if the feature is above $t$":

- $t = 1.5$: predicts $0, 1, 1, 1, 1, 1$; only the row with feature 4 is wrong: $5/6 = 0.833$.
- $t = 2.5$: predicts $0, 0, 1, 1, 1, 1$; the rows with features 2 and 4 are wrong: $4/6 = 0.667$.
- $t = 3.5$: $0, 0, 0, 1, 1, 1$; features 2, 3 and 4 wrong: $3/6 = 0.5$.
- $t = 5$: $0, 0, 0, 0, 1, 1$; features 2 and 3 wrong: $4/6 = 0.667$.
- $t = 7$: $0, 0, 0, 0, 0, 1$; features 2, 3 and 6 wrong: $3/6 = 0.5$.

Keep $t = 1.5$, chosen on the training rows only. The validation rows have features $0.5$ and $5.0$, so it
predicts $0, 1$, matching the labels $0, 1$: validation accuracy 1.0, against 0.5 for the majority baseline.
(Two validation rows say nothing reliable; the point is the procedure.)

```text
== worked example by hand ==
test rows [7, 2]   val rows [9, 4]   train rows [0, 1, 3, 5, 6, 8]
train targets [3.0, 5.0, 8.0, 6.0, 7.0, 7.0] -> mean baseline predicts 6.0000
val targets [5.0, 10.0]   MSE 8.5000   MAE 2.5000
train labels [0, 1, 1, 0, 1, 1] -> majority class 1   val labels [0, 1] -> accuracy 0.50
train (feature, label) sorted by feature: [(1.0, 0), (2.0, 1), (3.0, 1), (4.0, 0), (6.0, 1), (8.0, 1)]
rule 'predict 1 if feature > 1.5': train accuracy 0.8333
rule 'predict 1 if feature > 2.5': train accuracy 0.6667
rule 'predict 1 if feature > 3.5': train accuracy 0.5000
rule 'predict 1 if feature > 5.0': train accuracy 0.6667
rule 'predict 1 if feature > 7.0': train accuracy 0.5000
chosen threshold 1.5: val features [0.5, 5.0] -> predictions [0, 1]   val accuracy 1.00
```

## Key plot

![Line chart: mean best test accuracy among k coin-flip models on one test set of 100 rows, rising from 0.50 at k = 1 to 0.58 at k = 10, 0.63 at k = 100 and 0.66 at k = 1000, with a shaded band for the middle 90% of runs and a dashed line at the true accuracy 0.5](img/b1_1_test_reuse.png)

Every model is a coin flip, so every one of them is 50% accurate on new data. Choosing the best on
one reused set of 100 rows reports 58% after 10 tries and 63% after 100. The same thing happens,
more slowly and less visibly, every time you pick features or settings by looking at a test score.

## From scratch in NumPy

**The split.** Shuffle row indices with a seeded generator, cut off the test part, then the validation
part. For stratification, do the same inside each class and concatenate. The function returns index
arrays, so the same split applies to `X`, `y` and any DataFrame:

```python
def train_val_test_split(n, val_frac, test_frac, gen, stratify=None):
    """Shuffle row indices and cut them three ways. n rows -> (train, val, test) index arrays."""
    groups = [np.arange(n)] if stratify is None else [
        np.flatnonzero(stratify == c) for c in np.unique(stratify)]   # one group per class
    parts = [[], [], []]
    for rows in groups:
        rows = gen.permutation(rows)                    # shuffle within the group
        n_test, n_val = round(test_frac * len(rows)), round(val_frac * len(rows))
        parts[2].append(rows[:n_test])
        parts[1].append(rows[n_test:n_test + n_val])
        parts[0].append(rows[n_test + n_val:])
    return tuple(gen.permutation(np.concatenate(p)) for p in parts)   # train, val, test
```

(The lab's version adds type hints and input checks.) On the breast-cancer data (569 tumours, 357
benign, 212 malignant):

```text
== from scratch: train/validation/test split ==
plain:      sizes 341 / 114 / 114   share benign 0.6305 / 0.5789 / 0.6667
stratified: sizes 343 / 113 / 113   share benign 0.6268 / 0.6283 / 0.6283
disjoint and complete: OK
same seed -> identical train rows: True   different seed -> identical: False
share benign in the test set over 1000 splits: plain 0.4912..0.7544 (sd 0.0416)   stratified 0.6283..0.6283 (sd 0.0000)
```

- The plain split put 57.9% benign in validation and 66.7% in test, purely by chance. Over 1,000
  random splits, the share of benign cases in the test set ranged from 49% to 75%. The stratified split
  keeps 62.8% in every set, every time.
- Same seed, same split: the split is reproducible, which matters when you rerun an experiment next
  week and want the same validation rows.

**Baselines.** The diabetes data (442 patients, 10 measurements in raw units, target a measure of
disease progression a year later, mean 152) gets the mean, median and a bmi rule; the breast-cancer data
gets the majority class and a one-feature threshold rule. Everything is fitted on training rows and
scored on validation rows:

```python
mean_pred = y_reg[r_tr].mean()                         # learned on train
base_mean = np.full(len(yv), mean_pred)                # the same number for every validation row
edges, means = fit_one_feature_bins(X_reg[r_tr, j_bmi], y_reg[r_tr], n_bins=4)  # bmi quartiles
thr, above = fit_threshold_rule(X_clf[c_tr, j_feat], yc_tr)   # best single cut on training accuracy
```

```text
== from scratch: baselines ==
diabetes split 266 / 88 / 88   breast cancer split 343 / 113 / 113
[diabetes] mean baseline predicts 155.1466: val MSE 6084.76  RMSE 78.00  MAE 67.17  R^2 -0.0378
[diabetes] median baseline predicts 142.5000: val MSE 5868.20  MAE 64.09
[diabetes] bmi rule: cut points [23.5, 25.7, 28.75]   bin means [111.1364, 123.3231, 173.6618, 210.5821]
[diabetes] bmi rule: val MSE 3834.18  RMSE 61.92  MAE 53.70  R^2 0.3461
[cancer] majority class in train = 1 (benign)   val accuracy 0.6283   malignant cases caught: 0 of 42
[cancer] rule: predict 0 if worst concave points > 0.1358 else 1   train accuracy 0.9213   val accuracy 0.8938
[cancer] rule: malignant cases caught on val: 36 of 42
```

How to read it:

- The mean baseline's validation $R^2$ is $-0.04$: slightly worse than predicting the *validation* mean, as
  the math section said. Its RMSE of 78 is the error scale of knowing nothing (the target's standard
  deviation is 77).
- The median baseline has a lower MAE (64.1 against 67.2) and a lower MSE here; on another split the
  MSE ordering could flip, but the MAE ordering is what the math promises on the training data.
- The bmi rule, four bins with their training means, explains 35% of the variance: a strong baseline
  from one column. Any model on this data that does not clearly beat $R^2 = 0.35$ is not worth deploying.
- For breast cancer, the majority baseline gets 62.8% accuracy while catching **none** of the 42
  malignant tumours. Accuracy hides that completely; B4.1 builds the metrics that do not. One threshold
  on "worst concave points" gets 89% and catches 36 of 42.

**Generalization.** The 1-nearest-neighbour memorizer (standardize the features, find the closest
training row by squared distance, copy its target):

```text
== from scratch: memorizing is not generalizing ==
[diabetes] 1-NN: train R^2 1.0000   val R^2 0.0864   (mean baseline val R^2 -0.0378)
[cancer] 1-NN: train accuracy 1.0000   val accuracy 0.9292   (majority baseline 0.6283)
```

Perfect on its training rows in both cases. On validation rows, it explains 9% of the diabetes
variance (worse than the one-column bmi rule) but gets 93% on breast cancer, where tumours of the
same kind really do look alike. Same training score, very different generalization: you only find
out on held-out data.

**Reusing a test set.** The simulation draws 100 random labels and $k$ = 1,000 models that guess at
random, and records the best accuracy seen after trying 1, 2, …, $k$ of them; 300 repetitions:

```text
== from scratch: reusing the test set ==
best of    1 coin-flip models on one test set of 100: mean reported accuracy 0.5005   (true accuracy of every model: 0.5)
best of   10 coin-flip models on one test set of 100: mean reported accuracy 0.5775   (true accuracy of every model: 0.5)
best of  100 coin-flip models on one test set of 100: mean reported accuracy 0.6253   (true accuracy of every model: 0.5)
best of 1000 coin-flip models on one test set of 100: mean reported accuracy 0.6623   (true accuracy of every model: 0.5)
standard error of one accuracy on 100 rows at p = 0.5: sqrt(0.25 / 100) = 0.0500
```

The numbers match the order-statistics estimate in the math section: $0.5 + 0.05 \times (1.54, 2.51, 3.24) =
0.577, 0.626, 0.662$.

## With scikit-learn

scikit-learn's `train_test_split` cuts in two, so a three-way split is two calls. The second call must
take `VAL_FRAC / (1 - TEST_FRAC)` = 0.25 of what is left, to end up with 20% of the whole:

```python
idx_trval, idx_test = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
idx_train, idx_val = train_test_split(idx_trval, test_size=0.2 / 0.8,
                                      stratify=y[idx_trval], random_state=SEED)
```

`DummyRegressor(strategy="mean")` and `DummyClassifier(strategy="most_frequent")` are the constant
baselines as estimators, with the usual `fit`/`predict`/`score`:

```text
== with scikit-learn ==
sklearn sizes 341 / 114 / 114   mine 343 / 113 / 113
sklearn share benign 0.6276 / 0.6228 / 0.6316
same sizes (within rounding) and same class shares (within 0.01): OK
(the rows differ: each uses its own random numbers; only the recipe is the same)
DummyRegressor(mean) val R^2 -0.0378   matches mine: OK
DummyClassifier(most_frequent) val accuracy 0.6283   matches mine: OK
```

The sizes differ by two rows because of rounding: the from-scratch version rounds each class's share
separately (42.4 → 42 and 71.4 → 71), while scikit-learn rounds the test size up once (113.8 → 114).
The rows themselves differ because the two use different random numbers; what must agree is the
recipe: sizes, disjointness and class shares. The baselines and metrics agree exactly.

## In practice

- **When to use it:** always. Write the problem statement (the six-row table) before looking at the
  data closely; split before exploring further, so that what you see in the test rows cannot shape
  your features; compute the baselines before the first real model.
- **Failure modes:**
  - *Test set used for choices.* You notice when the test score drops on fresh data, or when the
    validation and test scores are suspiciously close after weeks of tuning. Keep a log of every time
    the test score was computed; the honest number is the first one.
  - *Rows of one entity on both sides of the split.* Several rows per patient, customer or photo
    session: a random split puts some in training and some in test, and the model recognizes the entity
    instead of learning the pattern. You notice a large drop when splitting by entity. Split by group
    (exercise 2; `GroupKFold` in B4.3).
  - *Time order ignored.* When the model will predict the future, a random split tests it on the past
    it was trained around. Split by time (B1.4, B6.3).
  - *Unstratified splits on small or imbalanced data.* Scores move with the class mix. Stratify.
  - *No baseline, or a weak one.* "92% accuracy" on a 90/10 dataset. Always report the majority or mean
    baseline and a one-feature rule next to the model.
  - *A metric nobody acts on.* The model optimizes accuracy, the business pays for missed fraud. Choose
    the metric from the action (B4.1, B4.2).
- **Defaults that work:** 60/20/20 for a few thousand rows; with 100,000+ rows, 1–5% each for
  validation and test is plenty (the SE depends on the number of rows, not the fraction). Cross-validation
  (B4.3) replaces the single validation set when data is small. `stratify=y` for classification,
  `random_state=SEED` always. `train_test_split` defaults to `shuffle=True` and a 25% test size
  (VERIFY as of 2026-10, scikit-learn 1.9.1).

## Exercises

Save your work to `my-work/B/b1_1_ml_workflow.md` (or `.py`) and ask for a review.

1. **By hand:** Twelve rows, numbered 0 to 11, have labels $1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0$ and targets
   $10, 4, 6, 12, 5, 3, 7, 11, 4, 6, 9, 5$. (a) Do a *stratified* 50/25/25 train/validation/test split by hand,
   using the shuffled order $3, 10, 0, 7$ for the rows labelled 1 and $5, 2, 9, 1, 11, 8, 4, 6$ for the rows labelled 0
   (cut test first, then validation, as in the lesson). List the rows in each set. (b) Compute the majority
   baseline's validation accuracy and the mean and median baselines' validation MSE and MAE. (c) Ten
   colleagues each try a different model and report the best validation accuracy on a validation set of
   400 rows, where every model's true accuracy is 0.80. Estimate the expected reported best.
2. **Code:** Many datasets have several rows per entity (visits per patient, orders per customer).
   Generate 300 customers with 2–6 rows each (`SEED = 0`). Give each customer two attributes that are
   the same on all of their rows (age, tenure), a row-level feature (basket size), and a hidden
   "customer effect" that shifts all of their targets, plus row-level noise. Write
   `group_train_val_test_split(groups, val_frac, test_frac, gen)` that keeps every customer's rows in one
   set, with a test that no customer is split. Compare the validation $R^2$ of the lab's
   1-nearest-neighbour model, using only the two customer attributes as features, under a row-level split
   and under your group split. Explain the difference, and say which number you would report.
3. **Think:** A product manager asks: "Can we use ML to reduce late food deliveries?" Write the
   six-row problem statement (unit, target, prediction time, action, metric, baseline) for one concrete
   version of this. Then name two things that, if you found them during framing, would make you
   recommend *not* building a model.

## Takeaways

1. Frame before you fit: unit of prediction, exact target, prediction time, the action taken, the
   metric that matches its cost, and the baseline.
2. A model generalizes if its error on new data is close to its training error; training error alone
   says nothing (a lookup table scores perfectly).
3. Training set to fit, validation set to choose, test set once to report. Choosing on a set is a form
   of fitting to it, so a reused set reports the luckiest model.
4. Stratify classification splits; split by group when rows share an entity, and by time when you
   predict the future.
5. Report every model next to the mean or majority baseline and a one-feature rule; a model that does
   not clearly beat them is not finished.

**What builds on this:** [B1.4](b1_4_data_leakage.md) (what leaks across the split);
[B1.5](b1_5_linear_regression.md) (the first model scored against these baselines);
[B4.1](b4_1_classification_metrics.md) and [B4.2](b4_2_regression_metrics.md) (metrics that match the
action); [B4.3](b4_3_cross_validation_tuning.md) (cross-validation, group and time splits);
[B4.5](b4_5_bias_variance.md) (the generalization gap); P2 (the problem statement is milestone M0).

**References:** M. Zinkevich, "Rules of Machine Learning: Best Practices for ML Engineering", Google
(VERIFY link: https://developers.google.com/machine-learning/guides/rules-of-ml). scikit-learn User Guide,
"Common pitfalls and recommended practices" (VERIFY link:
https://scikit-learn.org/stable/common_pitfalls.html). T. Hastie, R. Tibshirani, J. Friedman, *The
Elements of Statistical Learning*, 2nd ed., Springer, 2009, chapter 7 (model assessment and selection).
