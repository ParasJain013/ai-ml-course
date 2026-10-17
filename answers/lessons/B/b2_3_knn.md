# Answer key: B2.3 K-nearest neighbours: distance metrics and the curse of dimensionality

Lesson: [`lessons/B/b2_3_knn.md`](../../../lessons/B/b2_3_knn.md)

## Exercise 1 (by hand)

**Solution (a).** Differences from the query $(0, 5)$: row 0 $(2, -2)$, row 1 $(2, -5)$, row 2 $(3, 0)$, row 3 $(5, 1)$, row 4 $(4, -1)$.
Euclidean: $\sqrt8 = 2.828$, $\sqrt{29} = 5.385$, $3$, $\sqrt{26} = 5.099$, $\sqrt{17} = 4.123$. Manhattan: $4, 7, 3, 6, 5$.

**(b)** Euclidean order: rows 0 (A), 2 (B), 4 (B), 3 (B), 1 (A). $k = 1$: A. $k = 3$: A, B, B, so B. $k = 5$: two A, three B, so B.
Manhattan order: rows 2 (B), 0 (A), 4 (B), 3 (B), 1 (A). $k = 1$: B. $k = 3$: B, A, B, so B. $k = 5$: B.
The metric changes the single nearest neighbour. Row 0 differs by 2 in both features, row 2 by 3 in one. Squaring
favours spreading the difference ($8 < 9$); absolute values favour concentrating it ($4 > 3$).

**(c)** Halve feature 2 for every point and the query: the query becomes $(0, 2.5)$, and the differences become $(2, -1)$, $(2, -2.5)$,
$(3, 0)$, $(5, 0.5)$, $(4, -0.5)$. Distances: $\sqrt5 = 2.236$, $\sqrt{10.25} = 3.202$, $3$, $\sqrt{25.25} = 5.025$, $\sqrt{16.25} = 4.031$. Order: rows 0 (A), 2 (B), 1 (A). $k = 3$: A.
Row 1 was far only because of its feature-2 gap; once that feature counts half as much, it moves into the neighbourhood
and flips the vote.

**(d)** $d = 4$: mean $4/6 = 0.667$, sd $\sqrt{28/180} = 0.394$, sd/mean 0.59. $d = 400$: mean 66.7, sd $\sqrt{2800/180} = 3.944$, sd/mean 0.059. A hundred times the
dimension gives ten times less relative spread ($1/\sqrt{100}$). In 400 dimensions almost every squared distance is within about 12%
($2 \times 0.059$) of 66.7.

```text
== exercise 1 check ==
(a) Euclidean [2.8284, 5.3852, 3.0, 5.099, 4.1231]   Manhattan [4.0, 7.0, 3.0, 6.0, 5.0]
(b) Euclidean order [0, 2, 4, 3, 1] -> k=1,3,5: ['A', 'B', 'B']
    Manhattan order [2, 0, 4, 3, 1] -> k=1,3,5: ['B', 'B', 'B']
(c) feature 2 halved: distances [2.2361, 3.2016, 3.0, 5.0249, 4.0311]   order [0, 2, 1, 4, 3] -> k=3: A
(d) d = 4: mean 0.6667  sd 0.3944  sd/mean 0.5916
(d) d = 400: mean 66.6667  sd 3.9441  sd/mean 0.0592
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) All ten distances | 3 |
| (b) Six predictions, with the neighbour order for each metric | 3 |
| (c) Rescales the query as well as the points; new order and the flipped prediction | 3 |
| (d) Means, standard deviations, ratios, and the $1/\sqrt{d}$ reading | 3 |

**Likely mistakes:**

- In (c), rescaling the training points but not the query (or the reverse). Every point, including queries, must go
  through the same transformation. This is the train/serve skew of B1.3 in miniature.
- Reporting $k = 5$ as "depends on the metric". With $k = n$ the prediction is always the majority class of the whole training set.
- Computing $\sqrt{7 \cdot 400}/180$ instead of $\sqrt{7 \cdot 400/180}$ in (d).

## Exercise 2 (code)

**Solution:** [`b2_3_knn_ex2.py`](b2_3_knn_ex2.py). A `GridSearchCV` over odd $k$ from 1 to 31 and both weightings, with stratified
5-fold cross-validation on the 106 training rows only, in three settings:

```text
== cross-validated search in three settings ==
13 real features                                   best k 17 uniform   CV acc 0.9810  val acc 0.9444
13 real + 100 noise                                best k  9 distance  CV acc 0.9061  val acc 0.8333
13 real + 100 noise, SelectKBest(13) in pipeline   best k 19 uniform   CV acc 0.9905  val acc 0.9722
                                                   selector kept 12 of 13 real features and 1 noise columns
```

- 100 noise columns cost about 7 points of cross-validated accuracy and 11 of validation accuracy, as in the lab.
- A univariate F-test (does the feature's mean differ between classes?) separates real from noise features easily
  here: the final selector keeps 12 of the 13 real ones. With the selector in the pipeline, the accuracy returns to the
  no-noise level. (The difference between 0.98 and 0.99 is one or two rows: noise.)
- **Why inside the pipeline:** the selector looks at the labels. If it is fitted on all training rows before
  cross-validation, each validation fold's labels have helped choose the features, and the cross-validated score is
  optimistic. B1.4 showed this inflating AUC from 0.5 to 0.88 on pure noise. Inside the pipeline, `GridSearchCV` refits it on
  each fold's training part only.

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| Cross-validation on training rows only, stratified, seeded; validation rows used once per setting | 3 |
| Scaler (and selector) inside the pipeline that is cross-validated | 3 |
| Grid over $k$ and weighting; best setting and both scores reported for all three settings | 3 |
| Noise features generated once and shared by the train and validation rows | 2 |
| Explanation of why the selector must be inside the pipeline, naming label leakage | 3 |

**Likely mistakes:**

- Running `SelectKBest(...).fit_transform(X_train, y_train)` once and cross-validating on the result. The score looks
  just as good here (the real features are easy to find), so the bug is invisible. On harder data it is not.
- Choosing $k$ by validation accuracy over 16 values and reporting that validation accuracy as the model's performance.
- Generating different noise for the training and validation rows, or noise with a different scale. That changes
  the experiment from "useless features" to "features that differ between train and serve".
- Including even $k$ values with two-class problems, which makes ties possible. Not wrong with three classes, but a habit to watch.

## Exercise 3 (think)

**Strong answer:** Before trusting the neighbours:
(1) *Features.* Counts and spend are heavy-tailed: one big spender dominates the distance. Apply `log1p` to counts and
spend, then standardize. Decide deliberately how much each feature group should weigh, because 150 category-view counts
would otherwise outvote a single recency feature. Normalizing each customer's category vector to proportions (or using
cosine distance) makes "likes the same categories" independent of "shops a lot".
(2) *Dimension.* 200 features is enough for distances to concentrate, and many will be sparse or irrelevant.
Reduce with PCA or truncated SVD (B5.4), or better, learn a customer embedding from purchase sequences (matrix
factorization, B6.1). Check that the nearest/farthest distance ratio is well below 1 on real data.
(3) *Latency.* Brute force over 5 million × 200 floats is about a billion multiply-adds per request: too slow for a page
load. Precompute neighbours in a nightly batch job, or serve from an approximate index (HNSW in pgvector or a vector
database), and measure recall@20 of the approximate search against exact search on a sample.
(4) *Cold start.* New customers have no behaviour; fall back to popularity or context.
(5) *Evaluation.* Offline: hold out each customer's most recent purchases (a time-based split, B1.4) and measure whether
items bought by the neighbours appear in them (precision@k, recall@k, NDCG, B6.2), against a "most popular items" baseline.
Online: an A/B test on click-through and revenue (A6.3b), because offline gains often do not transfer. Also check that
neighbours do not encode sensitive attributes in a way that produces unfair or creepy recommendations.

**Weak answer:** "Standardize the 200 features, use `NearestNeighbors(n_neighbors=20)` and recommend the most common items
among the neighbours. Check the accuracy on a test set."

**Why it is weak:** scaling alone ignores heavy tails, feature-group weighting and the curse of dimensionality at 200
dimensions; exact search does not meet latency at 5 million rows; and "accuracy on a test set" is undefined for
recommendations. It needs ranking metrics on a time-based holdout, a popularity baseline and an online test.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Feature transformation and weighting (log of counts, scaling, cosine or proportions) | 2 |
| Dimension: reduction or embeddings, and a check of distance contrast | 2 |
| Latency: batch precomputation or an approximate index, with recall measured against exact search | 2 |
| Evaluation: time-based holdout, ranking metrics, popularity baseline, A/B test | 3 |
| Mentions cold start, or sensitive attributes and fairness | 1 |

## Scoring

Total 36 pts. Pass ≥ 70% (26 pts). Topic ids to revisit for each lost criterion: Exercise 1 → B2.3, A1.2;
(d) → A5.4. Exercise 2 → B1.4 (selection inside the pipeline), B4.3 (cross-validation), B2.3. Exercise 3 → B2.3, B5.4,
B6.1, B6.2, I2.2.
