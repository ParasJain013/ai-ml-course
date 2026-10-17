# Answer key: B1.1 The ML workflow

Lesson: [`lessons/B/b1_1_ml_workflow.md`](../../../lessons/B/b1_1_ml_workflow.md)

## Exercise 1 (by hand)

**Solution (a).** Rows labelled 1: 0, 3, 7, 10 (four rows). Rows labelled 0: 1, 2, 4, 5, 6, 8, 9, 11 (eight rows).
For each class, $n_{\text{test}} = n_{\text{val}} = \operatorname{round}(0.25 \times \text{class size})$: 1 and 1 for class 1, 2 and 2 for class 0.

- Class 1, order $3, 10, 0, 7$: test {3}, validation {10}, train {0, 7}.
- Class 0, order $5, 2, 9, 1, 11, 8, 4, 6$: test {5, 2}, validation {9, 1}, train {11, 8, 4, 6}.

Train = {0, 4, 6, 7, 8, 11}, validation = {1, 9, 10}, test = {2, 3, 5}. Each set holds one third label 1, as the
whole does (4 of 12).

**(b)** Training labels (rows 0, 4, 6, 7, 8, 11) are $1, 0, 0, 1, 0, 0$: majority 0. Validation labels (rows 1, 9, 10) are
$0, 0, 1$: accuracy $2/3 = 0.667$. Training targets are $10, 5, 7, 11, 4, 5$: mean $42/6 = 7$; sorted $4, 5, 5, 7, 10, 11$, median
$(5 + 7)/2 = 6$. Validation targets are $4, 6, 9$.

- Mean baseline 7: errors $-3, -1, 2$; MSE $= (9 + 1 + 4)/3 = 4.667$; MAE $= (3 + 1 + 2)/3 = 2.0$.
- Median baseline 6: errors $-2, 0, 3$; MSE $= (4 + 0 + 9)/3 = 4.333$; MAE $= 5/3 = 1.667$.

**(c)** $\mathrm{SE} = \sqrt{0.8 \times 0.2/400} = 0.02$. The best of 10 independent estimates is about 1.54 SE above the truth:
$0.80 + 1.54 \times 0.02 \approx 0.831$. In reality the ten models make many of the same mistakes on the same rows,
so their scores are correlated and the inflation is smaller, but it is never zero.

```text
== exercise 1 check ==
class 1: 4 rows -> test [3]  val [10]  train [0, 7]
class 0: 8 rows -> test [5, 2]  val [9, 1]  train [11, 8, 4, 6]
train [0, 4, 6, 7, 8, 11]   val [1, 9, 10]   test [2, 3, 5]
train labels [1, 0, 0, 1, 0, 0] -> majority 0   val labels [0, 0, 1] -> accuracy 0.6667
mean baseline 7.0000: val errors [-3.0, -1.0, 2.0]   MSE 4.6667   MAE 2.0000
median baseline 6.0000: val errors [-2.0, 0.0, 3.0]   MSE 4.3333   MAE 1.6667
(c) SE = sqrt(0.8 * 0.2 / 400) = 0.0200   expected best of 10 = 0.80 + 1.54 * SE = 0.8308
(c) simulation, 20000 repetitions: mean best of 10 = 0.8304
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) Splits each class separately with the right counts, and lists the three sets correctly | 4 |
| (b) Majority class from the training labels, validation accuracy 0.667 | 2 |
| (b) Mean 7 and median 6 from the *training* targets; correct MSE and MAE for both | 4 |
| (c) SE 0.02 and an estimate near 0.83, with the reason (maximum of noisy estimates) | 2 |

**Likely mistakes:**

- Cutting the 12 rows in one go (3 test, 3 validation) instead of within each class: not stratified.
- Computing the baselines from all 12 targets, or from the validation targets: the baseline is
  *learned* on training rows like any model.
- Using 1-based row numbers in (a).
- In (c), answering 0.80 ("they are all 0.80") or computing the SE with $p = 0.5$.

## Exercise 2 (code)

**Solution:** [`b1_1_ml_workflow_ex2.py`](b1_1_ml_workflow_ex2.py).

```text
== exercise 2: data ==
300 customers, 1245 rows, X (1245, 3), rows per customer 2..6
test_group_split: OK

== exercise 2: row split vs group split ==
row split    train 747  val 249  val rows whose customer is also in train 0.9277   1-NN val R^2 0.4708
             linear on basket only: val R^2 0.1684
group split  train 755  val 249  val rows whose customer is also in train 0.0000   1-NN val R^2 -0.7800
             linear on basket only: val R^2 0.1011
share of the target variance that is the hidden customer effect: 0.7619
copying a random row of the same customer: expected R^2 = 2 * share - 1 = 0.5238; of another customer: -1
```

Under the row split, 93% of validation rows belong to a customer who also has rows in training. With
only customer attributes as features, the nearest training row is one of that customer's own rows
(distance 0), so 1-NN copies a target that shares the hidden customer effect, and scores $R^2 = 0.47$. That is
recognition, not prediction: for a *new* customer, the group split shows the truth, $R^2 = -0.78$, far worse than
predicting the mean. The theory matches: if a fraction $\rho$ of the target variance is shared within a
customer, copying another row of the same customer gives $R^2 \approx 2\rho - 1$ (0.52 here), and copying a different
customer's row gives about $-1$. Report the group-split number if the model will be used on new customers.
(If it will only ever score existing customers, a split by *time* within customers is the honest one;
B1.4.) The linear model on basket size also drops (0.17 to 0.10), mostly because the group split's validation
set is a different sample of customers; its honest $R^2$ is low because 76% of the variance is the customer
effect, which no feature here explains.

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| Data generated with a shared customer effect and customer-level features; `SEED = 0` | 2 |
| `group_train_val_test_split` returns row indices with no customer in two sets, every row used once | 4 |
| A test that fails if a group is split (and passes) | 2 |
| Row split vs group split compared with the same model; the row split scores clearly higher | 3 |
| Explains why (the model recognizes the customer) and which number to report | 3 |

**Likely mistakes:**

- Splitting rows and then dropping the rows of customers that ended up in two sets: works, but the set
  sizes and class shares become uncontrolled. Split the *group ids*.
- Using `np.unique(groups)` without shuffling it: the split then depends on how ids were assigned.
- Including the row-level feature in the 1-NN distance: the effect is still there but diluted, and it is
  easy to conclude "no difference".
- Reporting the row-split score because "the data is the same". The deployment question decides the
  split.

## Exercise 3 (think)

**Strong answer:** One concrete version: *Unit:* one order, at the moment the restaurant accepts it.
*Target:* 1 if the delivery arrives more than 10 minutes after the time promised to the customer, else 0
(or the delay in minutes as a regression target). *Prediction time:* order acceptance; features are only
what is known then (distance, restaurant's current queue, courier availability, weather, time of day),
not the courier's actual pickup time. *Action:* for orders predicted late, the dispatcher assigns the
nearest courier immediately, or the app shows a longer promised time. *Metric:* for dispatch, recall of
late orders at a precision the dispatch team can handle (they can act on N orders an hour); for the
promise, MAE of the delay or the share of promises broken. *Baseline:* "late if the restaurant's average
prep time this hour is above X" or the current promised-time formula; the model must beat what the
company already does. Reasons not to build a model: (1) most lateness comes from one cause a rule or an
operational change fixes (one restaurant with a broken oven, a courier shortage every Friday evening),
so a dashboard and a process change beat ML; (2) there is no lever: predictions arrive when nothing can
be changed, or nobody will act on them; (3) the target is not recorded reliably (delivered times entered
by couriers after the fact); (4) the baseline already captures most of the achievable gain and the model
adds maintenance cost.

**Weak answer:** "Train a model on past deliveries to predict lateness and use accuracy as the metric;
always build the model because ML will find patterns." Weak because it has no unit, no prediction time
(so features may leak the outcome), no action (so no basis for the metric), accuracy is misleading if
most orders are on time, and it never considers that a rule or process change may be enough.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| All six items stated concretely and consistently with each other | 4 |
| Prediction time stated, with at least one feature correctly excluded as unknown at that time | 2 |
| Metric justified by the action (not "accuracy" by default) | 2 |
| Two credible reasons not to build a model | 2 |

## Scoring

Total 36 pts. Pass ≥ 70% (26 pts). Topic ids to revisit for each lost criterion: exercise 1 → B1.1 "Worked
example by hand" and A6.2 (standard errors); exercise 2 → B1.1 "In practice" (rows of one entity) and B4.3
(group splits); exercise 3 → B1.1 "Intuition" (problem statement), B1.4 (prediction time) and B4.1 (metrics).
