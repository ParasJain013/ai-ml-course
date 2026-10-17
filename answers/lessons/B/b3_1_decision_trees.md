# Answer key: B3.1 Decision trees: Gini and entropy splits built from scratch

Lesson: [`lessons/B/b3_1_decision_trees.md`](../../../lessons/B/b3_1_decision_trees.md)

## Exercise 1 (by hand)

Data: $x = 1, \dots, 8$, labels $0, 0, 1, 0, 1, 1, 1, 1$ (3 negatives, 5 positives).

**Solution (a).** Shares $(3/8, 5/8)$. $G = 1 - (0.375^2 + 0.625^2) = 1 - (0.140625 + 0.390625) = 0.46875$.
$H = -(0.375\log_2 0.375 + 0.625\log_2 0.625) = 0.375 \times 1.4150 + 0.625 \times 0.6781 = 0.5306 + 0.4238 = 0.9544$ bits.

**(b)** For the cut after the first $k$ rows, weighted Gini $= \frac{k}{8}G_L + \frac{8-k}{8}G_R$:

| Cut | Left labels | Right labels | $G_L$ | $G_R$ | Weighted | Gain |
|---|---|---|---|---|---|---|
| 1.5 | 0 | 0,1,0,1,1,1,1 (2/5) | 0 | 0.4082 | 0.3571 | 0.1116 |
| 2.5 | 0,0 | 1,0,1,1,1,1 (1/5) | 0 | 0.2778 | 0.2083 | 0.2604 |
| 3.5 | 0,0,1 | 0,1,1,1,1 (1/4) | 0.4444 | 0.3200 | 0.3667 | 0.1021 |
| **4.5** | 0,0,1,0 | 1,1,1,1 | 0.3750 | 0 | 0.1875 | **0.2812** |
| 5.5 | 0,0,1,0,1 | 1,1,1 | 0.4800 | 0 | 0.3000 | 0.1688 |
| 6.5 | 0,0,1,0,1,1 | 1,1 | 0.5000 | 0 | 0.3750 | 0.0938 |
| 7.5 | 0,0,1,0,1,1,1 | 1 | 0.4898 | 0 | 0.4286 | 0.0402 |

Best threshold: $x \le 4.5$, gain 0.28125.

**(c)** Parent error rate $= 3/8 = 0.375$ (predict the majority, 1). Cut 2.5: left $\{0, 0\}$ error 0; right $\{1,0,1,1,1,1\}$ error $1/6$;
weighted $\frac68 \cdot \frac16 = 0.125$. Cut 4.5: left $\{0,0,1,0\}$ error $1/4$; right pure; weighted $\frac48 \cdot \frac14 = 0.125$. Both cuts leave exactly one row
misclassified, so the error rate rates them equal. Gini prefers 4.5 (gain 0.2812 against 0.2604): its right child is pure
and its left child (3 of one class, 1 of the other) is easier to finish than a 1-in-6 node of 6 rows.

**(d)** Cut 2.5: right child $(1/6, 5/6)$, $H = 0.6500$; weighted $\frac68 \times 0.6500 = 0.4875$; gain $0.9544 - 0.4875 = 0.4669$.
Cut 4.5: left child $(3/4, 1/4)$, $H = 0.8113$; weighted $\frac48 \times 0.8113 = 0.4056$; gain $0.5488$. Entropy agrees with Gini: 4.5 is better.

**(e)** Targets $2, 4, 3, 10, 9, 14, 13, 15$, mean 8.75, root variance $187.5/8 = 23.4375$. The cut at 3.5 gives left $\{2, 4, 3\}$ (mean 3, squared
deviations $1 + 1 + 0 = 2$) and right $\{10, 9, 14, 13, 15\}$ (mean 12.2, squared deviations $4.84 + 10.24 + 3.24 + 0.64 + 7.84 = 26.8$). Weighted variance $= (2 + 26.8)/8 = 3.6$,
the lowest of the seven cuts (next best: 5.5 with 6.9, 4.5 with 7.4375). Leaves predict **3.0** for $x \le 3.5$ and **12.2** above.

```text
== exercise 1 check ==
(a) root Gini 0.4688   entropy 0.9544 bits
(b) Gini gain by cut: 1.5: 0.1116  2.5: 0.2604  3.5: 0.1021  4.5: 0.2812  5.5: 0.1688  6.5: 0.0938  7.5: 0.0402
(c,d) cut 2.5: error 0.3750 -> 0.1250   Gini gain 0.2604   entropy gain 0.4669
(c,d) cut 4.5: error 0.3750 -> 0.1250   Gini gain 0.2812   entropy gain 0.5488
(e) root variance 23.4375; weighted child variance by cut: 1.5: 16.9286  2.5: 12.4167  3.5: 3.6000  4.5: 7.4375  5.5: 6.9000  6.5: 14.2500  7.5: 17.8571
    best cut 3.5: leaves predict 3.0000 and 12.2000
    DecisionTreeRegressor stump: threshold 3.5  leaves [3.0, 12.2]
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Root Gini 0.4688 and entropy 0.9544 bits, with the class shares shown | 2 |
| (b) All seven gains with child impurities weighted by child size; best cut 4.5 | 4 |
| (c) Both cuts have weighted error 0.125; conclusion that the error rate cannot separate them while Gini prefers 4.5 | 3 |
| (d) Entropy gains 0.4669 and 0.5488; entropy agrees | 2 |
| (e) Best cut 3.5 by weighted variance (3.6), leaves 3.0 and 12.2 | 3 |

**Likely mistakes:**

- Averaging the two children's impurities without weighting by size (giving each child weight $\frac12$). That makes a split
  that isolates one row look as good as a balanced one.
- Using natural logs for entropy and then comparing with the bits figure. Either base picks the same split; just be consistent.
- Computing the misclassification rate of each child against the parent's majority class instead of the child's own
  majority.
- In (e), adding the two children's variances without weighting them by size. It happens to pick the same cut here, but
  it undervalues big children and overvalues tiny ones (a one-row child always has variance 0).
- Placing thresholds *at* data values (4 instead of 4.5). It gives the same partition, but it is not what CART reports.

## Exercise 2 (code)

**Solution:** [`b3_1_decision_trees_ex2.py`](b3_1_decision_trees_ex2.py). The key change from the classifier is the split
search: with running sums of $y$ and $y^2$, the sum of squared deviations of the left child at every cut is
$\sum y^2 - (\sum y)^2 / m_L$, so all cuts on a feature again cost one sort and one pass:

```python
s1, s2 = np.cumsum(ys)[:-1], np.cumsum(ys**2)[:-1]       # running sums, left child
t1, t2 = ys.sum() - s1, (ys**2).sum() - s2              # right child
sse_left = s2 - s1**2 / n_left
sse_right = t2 - t1**2 / (m - n_left)
score = (sse_left + sse_right) / m                      # weighted variance
```

A leaf stores `y.mean()`; a node with zero variance is not split.

```text
== data ==
train (266, 10)  val (88, 10)

== check against DecisionTreeRegressor ==
depth 1: best split at all  1 nodes OK, training predictions identical OK   val RMSE mine 66.33  sklearn 66.33
depth 2: best split at all  3 nodes OK, training predictions identical OK   val RMSE mine 58.43  sklearn 58.43
depth 3: best split at all  7 nodes OK, training predictions identical OK   val RMSE mine 64.14  sklearn 64.14
depth 4: best split at all 14 nodes OK, training predictions identical OK   val RMSE mine 65.35  sklearn 65.35

== depth sweep ==
depth  1: train RMSE  64.06   val RMSE  66.33
depth  2: train RMSE  57.39   val RMSE  58.43
depth  3: train RMSE  52.20   val RMSE  64.14
depth  4: train RMSE  47.09   val RMSE  65.35
depth  5: train RMSE  39.94   val RMSE  64.77
depth  6: train RMSE  33.65   val RMSE  68.84
depth  7: train RMSE  27.41   val RMSE  72.22
depth  8: train RMSE  23.14   val RMSE  73.60
depth  9: train RMSE  17.14   val RMSE  74.84
depth 10: train RMSE  13.03   val RMSE  75.40
depth 11: train RMSE   9.33   val RMSE  77.48
depth 12: train RMSE   5.59   val RMSE  77.21
no depth limit, min_leaf  5: train RMSE  36.55   val RMSE  70.52
no depth limit, min_leaf 10: train RMSE  46.07   val RMSE  67.66
no depth limit, min_leaf 20: train RMSE  52.65   val RMSE  58.61
B1.5 linear regression on the same validation rows: RMSE 55.98
saved outputs/b3_1/ex2_depth_rmse.png
```

**(b)** Training RMSE falls steadily to 5.6 at depth 12; validation RMSE is lowest at depth 2 (58.4) and climbs to 77 as the
tree memorizes 266 noisy rows. A strong answer picks depth 2, or `min_samples_leaf=20` (58.6), and says that with 88
validation rows the gap between 58.4 and 58.6 is noise: confirm the choice with cross-validation (B4.3) before trusting it.

**(c)** The best tree (58.4) is still worse than B1.5's linear regression (56.0). The diabetes target depends on the
features in a roughly additive, smooth way; a tree approximates each smooth trend with a few steps, and with 266 rows
it cannot afford many. One tree is a weak learner. B3.3 shows boosting many such small trees.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Variance (or SSE) split criterion, leaf = mean, midpoint thresholds, no cut between equal values | 3 |
| `max_depth` and `min_samples_leaf` honoured | 2 |
| Per-node check against a `DecisionTreeRegressor` stump passes at depths 1–4 | 3 |
| Training predictions identical to `DecisionTreeRegressor` (an assert, not a printout) | 2 |
| Depth sweep with a labelled plot; chooses a depth from validation, not training, error | 3 |
| Compares with B1.5 honestly and explains why one tree loses | 2 |
| Readability: type hints, shapes in docstrings, no copy of the classifier's dead code | 1 |

**Likely mistakes:**

- Computing the variance of each child with a Python loop over cuts: correct but $O(m^2)$ per feature, so depth 12 crawls. The
  running-sum trick is the point.
- Catastrophic cancellation in $\sum y^2 - (\sum y)^2/m$ when targets are large and close together (A3.4). It is fine here; on
  targets like house prices in dollars, centre $y$ first.
- Comparing against `DecisionTreeRegressor()` without a fixed `random_state`, then chasing differences caused by ties.
- Using the test rows to choose the depth.
- Reading "train RMSE 5.6 at depth 12" as a good model.

## Exercise 3 (think)

**Strong answer:** A 380-leaf, depth-12 tree is not an explanation anyone can check: nobody can read 380 rules, and each
leaf's rule rests on a handful of rows. The monthly root change from `income` to `debt_to_income` shows the high variance
of trees: the two features are correlated, their gains are close, and a small change in the data flips the winner and
rewrites everything below. A regulator who reads this month's rules learns little about next month's model. A 1-point AUC gain
is within the noise of a single validation split unless shown otherwise (B4.3 cross-validation, A6.2 bootstrap interval).
Risks: unstable explanations; overconfident leaf probabilities (pure leaves predict 0 or 1, bad for credit pricing);
and leaves that single out small groups, which can encode a proxy for a protected attribute (B4.6). Proposals: (1) keep the logistic regression, with
regularization and monotone checks, if its AUC is close, since its coefficients are the explanation; (2) if the tree's
non-linearity matters, use a shallow tree (depth 3–4, `min_samples_leaf` in the hundreds) as the explanation, or a
gradient-boosted model with monotone constraints plus SHAP explanations per decision (B3.3, B3.4); (3) document the
choice in a model card. Measure stability by refitting on bootstrap resamples or consecutive months and reporting how
often the top splits stay the same, how much the predictions for the same applicants change, and the variance of the AUC.
**Weak answer:** "Trees are interpretable because you can follow the path for each applicant, so the tree is the better
choice; just retrain it monthly and document the new tree each time."
**Why weak:** following one path through 12 questions does not explain the model's behaviour; it ignores instability,
the size of the AUC gain, overconfident probabilities and fairness, and proposes no way to measure any of them.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Explains why a 380-leaf tree is not a useful explanation | 2 |
| Links the root change to tree variance (correlated features, near-tied gains) | 2 |
| Questions whether 1 point of AUC is significant | 1 |
| Names at least two further risks (probabilities, fairness/proxies, small leaves) | 2 |
| A concrete alternative (shallow tree, keep LR, GBM with monotone constraints + SHAP) | 2 |
| A concrete stability measurement | 1 |

## Scoring

Total 40 pts. Pass ≥ 70% (28). Topic ids to revisit for each lost criterion: impurity and gain (Ex 1a–d, Ex 2
criterion 1) → B3.1, A6.4; regression trees (Ex 1e, Ex 2) → B3.1; choosing depth on validation → B1.1, B4.3; comparing
with a baseline → B1.5; stability and explanations (Ex 3) → B3.1, B3.4, B4.6.
