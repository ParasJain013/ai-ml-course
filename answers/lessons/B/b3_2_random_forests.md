# Answer key: B3.2 Random forests: bagging, feature subsampling, out-of-bag error

Lesson: [`lessons/B/b3_2_random_forests.md`](../../../lessons/B/b3_2_random_forests.md)

## Exercise 1 (by hand)

**Solution (a).** One draw misses a given row with probability $3/4$; four independent draws miss it with probability $(3/4)^4 = 81/256 = 0.3164$.
Each row is in the sample with probability $1 - 0.3164 = 0.6836$, and the expected number of distinct rows is the sum over rows
of that probability (linearity of expectation, A5.1): $4 \times 0.6836 = 2.734$.

**(b)** The majority of five is right when 3, 4 or 5 classifiers are right:
$\binom{5}{3}0.6^3 0.4^2 + \binom{5}{4}0.6^4 0.4 + 0.6^5 = 10(0.216)(0.16) + 5(0.1296)(0.4) + 0.07776 = 0.3456 + 0.2592 + 0.0778 = 0.6826$.

**(c)** $\operatorname{Var} = \rho\sigma^2 + (1 - \rho)\sigma^2/B = 1.2 + 2.8/B$. $B = 1$: 4.0. $B = 10$: $1.2 + 0.28 = 1.48$. $B \to \infty$: 1.2. Within 5% of the floor means
$2.8/B \le 0.05 \times 1.2 = 0.06$, so $B \ge 46.7$: **47 trees**. Beyond that, extra trees change almost nothing.

**(d)** Averaging leaves the expected prediction unchanged (Step 3), so a forest of depth-2 trees has the same bias as one
depth-2 tree: a staircase with at most 4 steps, which underfits almost any real problem. Averaging only removes variance,
and shallow trees do not have much variance to remove.

```text
== exercise 1 check ==
(a) n = 4: P(out of bag) = (3/4)^4 = 0.3164   expected distinct rows = 4 (1 - 0.3164) = 2.7344
(b) 5 classifiers at 0.6: P(majority right) = 0.68256
(c) B =  1: 4.0000
(c) B = 10: 1.4800
(c) B -> inf: 1.2000   within 5% of the floor needs (1-rho) s2 / B <= 0.05 rho s2 -> B >= 46.67, so 47 trees
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) $(3/4)^4 = 0.3164$ and expected distinct rows 2.73, with the reasoning | 3 |
| (b) Sums the 3-, 4- and 5-correct cases with binomial coefficients: 0.6826 | 3 |
| (c) 4.0, 1.48, 1.2 and the 47-tree calculation | 3 |
| (d) Averaging keeps the bias; shallow trees are high-bias, low-variance | 3 |

**Likely mistakes:**

- In (a), answering "$4 \times 3/4 = 3$ distinct rows", which treats the draws as if they could not repeat.
- In (b), computing only $0.6^3$ (exactly three right) or forgetting the binomial coefficients.
- In (c), writing the variance as $\sigma^2/B$ (independent trees) and concluding that 1,000 trees drive it to zero.
- In (d), saying a forest of shallow trees "averages out the errors". Errors from bias are shared by every tree and do not
  average out.

## Exercise 2 (code)

**Solution:** [`b3_2_random_forests_ex2.py`](b3_2_random_forests_ex2.py). The OOB part:

```python
for tree, in_bag_rows in zip(model.estimators_, model.estimators_samples_):
    oob = np.ones(m, dtype=bool)
    oob[in_bag_rows] = False                 # drawn at least once -> in the bag
    summed[oob] += tree.predict_proba(X[oob])
    counts[oob] += 1
proba = summed / counts[:, None]             # average over the trees that did not see each row
acc = np.mean(model.classes_[np.argmax(proba, axis=1)] == y)
```

```text
== data ==
train (1079, 64)  val (359, 64)

== OOB from estimators_samples_ ==
my OOB accuracy 0.973123   sklearn oob_score_ 0.973123   exact match: OK
trees per row that are out of bag: mean 108.0 (first 5 rows; expect about 0.368 x 300 = 110)

== max_features sweep ==
max_features  1: OOB acc 0.9666   val acc 0.9777   mean single-tree acc 0.6237   error correlation 0.1435
max_features  2: OOB acc 0.9722   val acc 0.9694   mean single-tree acc 0.6648   error correlation 0.1643
max_features  4: OOB acc 0.9768   val acc 0.9749   mean single-tree acc 0.7107   error correlation 0.1839
max_features  8: OOB acc 0.9731   val acc 0.9777   mean single-tree acc 0.7537   error correlation 0.2046
max_features 16: OOB acc 0.9741   val acc 0.9749   mean single-tree acc 0.7821   error correlation 0.2319
max_features 32: OOB acc 0.9666   val acc 0.9666   mean single-tree acc 0.7982   error correlation 0.2622
max_features 64: OOB acc 0.9583   val acc 0.9582   mean single-tree acc 0.8151   error correlation 0.2949
chosen by OOB: max_features 4 (OOB 0.9768)   its validation accuracy 0.9749
saved outputs/b3_2/ex2_max_features.png
```

What a strong write-up says about the sweep:

- Both curves are monotone where they should be: as `max_features` grows, single trees get better (0.62 → 0.82) and
  their errors get more correlated (0.14 → 0.29). The ensemble's accuracy is a compromise between the two (Step 2).
- OOB picks 4 (0.977). Everything from 2 to 16 lies within about 0.005 of OOB accuracy, which is 5–6 rows of 1,079; with 359 validation
  rows the validation ranking among them is noise (validation even ties 1 and 8 at the top). The robust conclusion is
  "anything from 2 to 16 is fine, and 32–64 (towards bagging) is worse", not "4 is optimal".
- Choosing by OOB keeps the validation set untouched for the final comparison, which is the point of OOB.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| OOB mask from `estimators_samples_` (in-bag means drawn at least once) | 3 |
| Averages `predict_proba` over OOB trees only, divides by the per-row count | 2 |
| Exact match with `oob_score_` (an assert), and handles rows never out of bag | 3 |
| Sweep over all seven values with OOB, validation, single-tree accuracy and error correlation | 3 |
| Plot readable (log-2 axis, labelled, legend) | 2 |
| Choice made by OOB only; interpretation acknowledges noise and the trade-off | 3 |

**Likely mistakes:**

- Treating `estimators_samples_` as the *out-of-bag* rows. It holds the in-bag indices, with duplicates.
- Averaging hard votes instead of probabilities, then failing to match `oob_score_` in a few rows.
- Dividing by the number of trees ($B$) instead of the number of out-of-bag trees for each row.
- Choosing `max_features` by validation accuracy and then reporting that same validation accuracy as the expected
  performance (optimistic, B4.3).
- Declaring a winner on a 0.003 difference.

## Exercise 3 (think)

**Strong answer:** The OOB estimate treats rows as independent. With 40 rows per customer, almost every customer has
rows in each tree's bootstrap sample, so the "out-of-bag" prediction for one of their transactions comes from trees that
saw that customer's other transactions, with the same card, merchant habits, and often the same chargeback label.
The forest recognizes customers, not fraud patterns. Time adds a second leak: rows from after a transaction are in the bag,
so the model uses the future to "predict" the past. OOB therefore behaves like random k-fold on grouped, time-ordered data, and it will be
optimistic: the live AUC on new transactions (and especially new customers) will be noticeably lower than 0.97, by how much
only an honest split can say. They should validate with a time-based split (train on months 1–18, validate on 19–21,
test on 22–24) and, within that, keep customers disjoint between train and validation if the model will score new
customers (B4.3 group and time-series splits; B1.4 leakage). OOB is still useful *inside* the training window as a quick
relative signal, for example to compare `max_features` values or check that more trees stop helping, as long as nobody
reports it as expected performance.
**Weak answer:** "OOB is a valid estimate because each tree is evaluated on rows it did not see, so 0.97 is fine; maybe
also check on a test set."
**Why weak:** it misses both leaks (customer grouping and time), so it does not see why the number is inflated or what
validation design fixes it.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Customer-level leakage through other rows of the same customer | 3 |
| Temporal leakage (future rows in the bag) | 2 |
| Predicts the live AUC will be lower and explains why | 1 |
| A concrete validation design: time-based split, grouped by customer where relevant | 3 |
| A legitimate remaining use of OOB | 1 |

## Scoring

Total 38 pts. Pass ≥ 70% (27). Topic ids to revisit for each lost criterion: bootstrap probabilities (Ex 1a) → A6.2;
voting and correlated averaging (Ex 1b–d) → B3.2, A5.1; OOB mechanics and `max_features` (Ex 2) → B3.2; leakage in
validation (Ex 3) → B1.4, B4.3.
