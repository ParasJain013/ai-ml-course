# B2.3 K-nearest neighbours: distance metrics and the curse of dimensionality

**Core** · Part B, week 2 (course week 08) · ~1.5 h · Lab: [`labs/B/b2_3_knn.py`](../../labs/B/b2_3_knn.py)

## Why it matters

**K-nearest neighbours (kNN)** predicts by finding the training examples most similar to the new one and copying
their answer. It has no training step and no formula for the boundary, so it can follow any shape. It is a useful
baseline and a strong model when the features are few and meaningful. Above all, it is the idea behind every
"find similar items" system you will build: semantic search, retrieval for RAG and recommendation all retrieve
nearest neighbours of an embedding (E1.3, I2.1, B6.1). Its two failure modes carry over to those systems unchanged.
Distances are meaningless when features are on different scales. And in high dimensions everything is roughly equally
far from everything else, unless the data lives on a low-dimensional structure. Both are cheap to check and expensive
to miss.

## Prerequisite check

- Can you compute a pairwise distance matrix between two sets of points without a Python loop? If not, revisit
  [A1.2](../A/a1_2_numpy_arrays.md).
- Can you compute the length of a vector and the dot product of two vectors, and relate them to the angle between
  them? If not, revisit [A1.5](../A/a1_5_vectors.md).
- Do you know why the mean of $d$ independent terms concentrates as $d$ grows? If not, revisit [A5.4](../A/a5_4_lln_clt_monte_carlo.md).
- Can you standardize features with training statistics inside a pipeline? If not, revisit [B1.3](b1_3_feature_engineering.md)
  and [B1.4](b1_4_data_leakage.md).
- Do you know what a validation set is for? If not, revisit [B1.1](b1_1_ml_workflow.md).

## Intuition

To guess the price of a flat, look up the five most similar flats that sold recently and average their prices. To
decide whether a wine comes from cultivar 1, 2 or 3, find the five wines with the most similar chemistry and take a vote.
That is the whole algorithm:

1. Store the training set. (That is the "training".)
2. For a query, compute its distance to every training example.
3. Take the $k$ closest. For classification, predict the majority class among them; for regression, predict the mean
   of their targets.

**$k$ is the knob.** With $k = 1$ the model copies its single nearest neighbour. It fits the training data perfectly (each training
point is its own nearest neighbour) and follows every noisy point, so the boundary is jagged. With $k = n$ it ignores
the query and always predicts the majority class. Values in between smooth the boundary. You choose $k$ on validation
data. In B4.5 terms, small $k$ means low bias and high variance.

**Everything depends on "closest".** Distance adds up differences across features, so a feature measured in large
units dominates. In the wine data proline varies with a standard deviation of 305 and the smallest-scale feature with
0.13, a ratio of 2,410. On raw features, proline alone makes up about half of the distance to the nearest neighbour, and
proline plus magnesium about three-quarters. The other 11 measurements barely count. Standardizing gives every feature
an equal say. Whether equal is *right* is a modelling choice; it is at least a choice, rather than an accident of units.

**The curse of dimensionality.** In two dimensions, 1,000 random points in a square leave a query's nearest neighbour
much closer than its farthest (ratio 0.02). In 1,000 dimensions the nearest is 89% as far as the farthest: everyone is
roughly equally far away, and "nearest" carries little information. Each new feature adds its own random difference to
every distance. Irrelevant features add noise to the distances without adding signal, and the real structure drowns.
kNN works in high dimensions only when the data really lives near a low-dimensional surface (images, text
embeddings), so that the effective dimension is small, and when the features are relevant.

> **Bridge:** kNN is a `SELECT ... ORDER BY distance(features, :query) LIMIT k` over the training table, followed by a
> `GROUP BY label` vote. pgvector (I2.2) runs exactly this query on embeddings. **Where it breaks:** a database index
> makes the lookup fast, but no index fixes a meaningless distance. If the columns are on different scales, or most of
> them are irrelevant, the query returns the wrong rows quickly. Also, an exact `ORDER BY` touches every row; the indexes
> that make it fast at scale (HNSW, IVF) are *approximate* and can miss true neighbours.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{X}$, $\mathbf{y}$ | training features (standardized) and labels or targets | `(n, d)`, `(n,)` |
| $\mathbf{q}$ | a query point | `(d,)` |
| $\mathbf{Q}$ | a batch of $m$ query points | `(m, d)` |
| $\mathbf{D}$ | distances from each query to each training row | `(m, n)` |
| $k$ | number of neighbours | int |
| $N_k(\mathbf{q})$ | the set of indices of the $k$ training rows closest to $\mathbf{q}$ | set of size $k$ |
| $K$ | number of classes | int |
| $\mathbb{1}[\cdot]$ | indicator: 1 if the condition holds, else 0 | — |

**Distances.** For two points $\mathbf{u}, \mathbf{v}$ of shape `(d,)`:

- Euclidean (L2): $\lVert\mathbf{u} - \mathbf{v}\rVert_2 = \sqrt{\sum_j (u_j - v_j)^2}$. The straight-line distance; the default.
- Manhattan (L1): $\lVert\mathbf{u} - \mathbf{v}\rVert_1 = \sum_j \lvert u_j - v_j\rvert$. Distance along a city grid. A single large difference counts less than in L2,
  because it is not squared.
- Cosine distance: $1 - \frac{\mathbf{u}^\top\mathbf{v}}{\lVert\mathbf{u}\rVert\,\lVert\mathbf{v}\rVert}$. It ignores length and compares direction only. That is the right notion for text and
  embeddings, where a longer document is not a different topic. For vectors scaled to length 1,
  $\lVert\mathbf{u} - \mathbf{v}\rVert^2 = \lVert\mathbf{u}\rVert^2 + \lVert\mathbf{v}\rVert^2 - 2\mathbf{u}^\top\mathbf{v} = 2 - 2\cos(\mathbf{u}, \mathbf{v})$, so Euclidean and cosine rank neighbours identically.

**All distances at once (A1.2).** $\lVert\mathbf{q} - \mathbf{x}\rVert^2 = \lVert\mathbf{q}\rVert^2 + \lVert\mathbf{x}\rVert^2 - 2\mathbf{q}^\top\mathbf{x}$, so for a batch,

$$\mathbf{D}^{(2)} = \underbrace{\lVert\mathbf{q}_i\rVert^2}_{(m, 1)} + \underbrace{\lVert\mathbf{x}_j\rVert^2}_{(1, n)} - 2\,\underbrace{\mathbf{Q}\mathbf{X}^\top}_{(m, n)},$$

one matrix product plus broadcasting. Ranking by squared distance gives the same neighbours as ranking by distance, so
the square root is skipped.

**Prediction.** Classification:
$\hat{y}(\mathbf{q}) = \arg\max_c \sum_{i \in N_k(\mathbf{q})} \mathbb{1}[y_i = c]$: the class with the most votes (ties broken by a fixed rule; scikit-learn picks the smallest
label). The vote shares $\frac1k\sum_{i \in N_k}\mathbb{1}[y_i = c]$ are what `predict_proba` returns: crude probabilities in steps of $1/k$. Regression:
$\hat{y}(\mathbf{q}) = \frac1k\sum_{i \in N_k(\mathbf{q})} y_i$. A **distance-weighted** variant weights each neighbour by $1/\text{distance}$, so close neighbours count more.

**Cost.** No training cost. Each prediction computes $n$ distances in $d$ dimensions: $O(nd)$ per query, and the whole training
set must be kept in memory. That is the reverse of linear models (slow to train, $O(d)$ to predict). Tree indexes
(KD-tree, ball tree) speed up exact search in low dimensions. In high dimensions they degrade towards brute force, which
is why embedding search uses approximate indexes (I2.2).

**Why distances concentrate.** Take two independent points uniform in $[0, 1]^d$. The squared distance is a sum of $d$ independent
terms, $\sum_j (u_j - v_j)^2$. For one term: $\mathbb{E}[u^2] = \int_0^1 u^2\,du = \frac13$ and $\mathbb{E}[u] = \frac12$, so $\mathbb{E}[(u - v)^2] = \mathbb{E}[u^2] - 2\mathbb{E}[u]\mathbb{E}[v] + \mathbb{E}[v^2] = \frac13 - \frac12 + \frac13 = \frac16$.
Its variance works out to $\frac{7}{180}$ (the difference $u - v$ has a triangular density $1 - \lvert t\rvert$ on $[-1, 1]$, giving $\mathbb{E}[(u - v)^4] = \frac{1}{15}$, and
$\frac{1}{15} - \frac{1}{36} = \frac{7}{180}$). Summing $d$ independent terms multiplies the mean by $d$ and the variance by $d$, so

$$\mathbb{E}\big[\lVert\mathbf{u} - \mathbf{v}\rVert^2\big] = \frac{d}{6}, \qquad \operatorname{sd}\big[\lVert\mathbf{u} - \mathbf{v}\rVert^2\big] = \sqrt{\frac{7d}{180}}, \qquad \frac{\text{sd}}{\text{mean}} = \frac{1.18}{\sqrt{d}}.$$

The typical squared distance grows like $d$, but its spread relative to its size shrinks like $1/\sqrt{d}$ (A5.4). All distances crowd
around the same value, so the gap between the nearest and the farthest neighbour vanishes in relative terms.

**Why neighbourhoods stop being local.** To capture a fraction $f$ of uniform data in a sub-cube, its side must be $f^{1/d}$. For 1% of the data,
the side is $0.01^{1/2} = 0.1$ in 2-D but $0.01^{1/100} = 0.955$ in 100-D. The "neighbourhood" spans 95% of the range of every feature,
so it is no longer local at all.

## Worked example by hand

Five training points in 2-D, with classes and regression targets:

| Row | $\mathbf{x}$ | class | target |
|---|---|---|---|
| 0 | (0, 0) | 0 | 1 |
| 1 | (2, 3) | 0 | 2 |
| 2 | (4, 2) | 1 | 3 |
| 3 | (5, 5) | 1 | 6 |
| 4 | (6, 3) | 1 | 5 |

Query $\mathbf{q} = (3, 3)$. Euclidean distances: row 0: $\sqrt{3^2 + 3^2} = \sqrt{18} = 4.243$; row 1: $\sqrt{1^2 + 0^2} = 1$; row 2: $\sqrt{1 + 1} = 1.414$; row 3: $\sqrt{4 + 4} = 2.828$;
row 4: $\sqrt{9 + 0} = 3$. Nearest first: rows 1, 2, 3, 4, 0.

- $k = 1$: row 1 alone, class 0. Regression: 2.
- $k = 3$: rows 1, 2, 3 with classes 0, 1, 1. Class 1 wins 2 to 1. Regression: $(2 + 3 + 6)/3 = 3.667$.
- $k = 5$: all rows, classes 0, 1, 1, 1, 0. Class 1 wins 3 to 2. Regression: the mean of all targets, 3.4.

The prediction flips between $k = 1$ and $k = 3$: the single nearest point disagrees with its neighbourhood. Manhattan
distances ($\lvert\Delta x_1\rvert + \lvert\Delta x_2\rvert$) are 6, 1, 2, 4, 3: the same order here.

**Scaling changes the answer.** Four people with (age in years, income in dollars) and a query aged 58 earning 45,000:

| Row | age | income | class | raw distance | standardized distance |
|---|---|---|---|---|---|
| 0 | 25 | 40,000 | 0 | 5,000.1 | 2.411 |
| 1 | 60 | 41,000 | 1 | 4,000.0 | **0.850** |
| 2 | 30 | 46,000 | 0 | **1,000.4** | 1.853 |
| 3 | 55 | 52,000 | 1 | 7,000.0 | 1.483 |

On raw units the nearest is row 2, a 30-year-old: the 28-year age gap adds only 0.4 to a distance of 1,000. After
dividing each column by its standard deviation (15.2 years and 4,763 dollars), the nearest is row 1, aged 60, whose income
is 4,000 away, less than one standard deviation. The prediction flips from class 0 to class 1.

```text
== worked example by hand ==
query [3.0, 3.0]
Euclidean distances [4.2426, 1.0, 1.4142, 2.8284, 3.0]   Manhattan distances [6.0, 1.0, 2.0, 4.0, 3.0]
neighbours by Euclidean distance (nearest first): [1, 2, 3, 4, 0]   their classes [0, 1, 1, 1, 0]
k = 1: votes [1, 0] -> class 0   regression: mean of targets [2.0] = 2.0000
k = 3: votes [1, 2] -> class 1   regression: mean of targets [2.0, 3.0, 6.0] = 3.6667
k = 5: votes [2, 3] -> class 1   regression: mean of targets [2.0, 3.0, 6.0, 5.0, 1.0] = 3.4000
age/income example: raw distances [5000.1, 4000.0, 1000.4, 7000.0] -> nearest row 2 (class 0)
  sd = [15.21, 4763.14]   standardized distances [2.4106, 0.85, 1.8532, 1.4828] -> nearest row 1 (class 1)
```

## Key plot

![Left: mean ratio of nearest to farthest distance for uniform random points, rising from 0.02 at d = 2 to 0.89 at d = 1000 on a log axis. Right: kNN validation accuracy on wine falling from 0.94 with no noise features to 0.64 with 1,000 pure-noise features added, still above the 0.41 majority-class rate](img/b2_3_curse_of_dimensionality.png)

Left: as the dimension grows, the nearest neighbour becomes almost as far away as the farthest point, so "nearest"
means less and less. Right: adding irrelevant features to the 13 real wine measurements drags kNN's accuracy down from
0.94 to 0.64. The real dimensions are still there; they are outvoted in every distance.

## From scratch in NumPy

**The data.** The wine dataset: 178 wines from three cultivars, 13 chemical measurements, stratified 60/20/20:

```text
== data ==
train X (106, 13)  val X (36, 13)  (test 36 rows kept aside)
class counts: train [35, 43, 28]  val [12, 14, 10]
feature sd (train): smallest 0.1265 (nonflavanoid_phenols)   largest 304.76 (proline)   ratio 2410
```

**Distances, neighbours, vote.** `np.argpartition` finds the $k$ smallest entries of each row in $O(n)$ without sorting all $n$; then only
those $k$ are sorted. `np.add.at` counts the votes for every query at once:

```python
def knn_classify(X_train, y_train, X_query, k, metric="euclidean"):
    """Majority vote of the k nearest training rows; ties go to the smallest label. -> (m,)."""
    D = pairwise_sq_dists(X_query, X_train) if metric == "euclidean" else manhattan_dists(X_query, X_train)
    nn = knn_indices(D, k)                                        # (m, k)
    votes = np.zeros((len(X_query), int(y_train.max()) + 1), dtype=np.int64)   # (m, K)
    np.add.at(votes, (np.arange(len(X_query))[:, None], y_train[nn]), 1)
    return votes.argmax(axis=1)                                   # first maximum = smallest label
```

```text
== from scratch: kNN classifier ==
  k  raw val acc  scaled val acc  scaled, manhattan
  1       0.7778          0.9444             1.0000
  3       0.6111          0.9722             0.9722
  5       0.6944          0.9444             0.9722
  7       0.6944          0.9722             0.9722
  9       0.6667          0.9722             0.9722
 15       0.6944          0.9444             0.9444
 25       0.7222          0.9444             0.9444
share of the squared distance to the raw nearest neighbour, top 3 features: {'proline': 0.4855, 'magnesium': 0.2657, 'alcalinity_of_ash': 0.1321} (together 0.7512 for the top 2)
```

- Standardizing lifts validation accuracy from about 0.6–0.78 to about 0.94–0.97 at every $k$. No choice of $k$ or metric
  comes close to that effect.
- Among the scaled results, the differences are one wine out of 36 (0.028). Manhattan's 1.0 at $k = 1$ is one or two
  lucky wines, not evidence that Manhattan is better. Do not choose a winner on 36 rows; use cross-validation (B4.3).
- On raw features, two of the 13 measurements make up three-quarters of every distance.

## With scikit-learn

```text
== with scikit-learn ==
from-scratch predictions identical to KNeighborsClassifier (k = 1, 5, 15), also inside a Pipeline: OK
Manhattan metric identical too: OK
predict_proba for 3 val wines (k = 5): [[0.8, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]   (vote shares, multiples of 1/5)
```

The predictions are identical, not just close: the same distances, the same neighbours, the same tie rule.
`make_pipeline(StandardScaler(), KNeighborsClassifier(k))` refits the scaler on whatever data `fit` receives, which keeps
cross-validation honest (B1.4). `predict_proba` is the vote share, so with $k = 5$ the only possible values are 0, 0.2, …, 1.

**kNN regression** on the diabetes split from B1.5:

```text
== from scratch: kNN regression on diabetes ==
k =  1: val R^2 0.0864
k =  5: val R^2 0.4991
k = 10: val R^2 0.4621
k = 20: val R^2 0.4879
k = 40: val R^2 0.4296
matches KNeighborsRegressor at every k: OK
linear regression on the same split (B1.5): val R^2 0.4655
```

$k = 1$ copies one patient's progression and scores 0.09. Averaging 5 to 20 neighbours brings kNN level with linear regression.
The best $k$ is chosen on these same 88 rows, so the 0.50 is optimistic; with ten features of modest signal, the two
models are roughly equal here. kNN's advantage appears when the relationship is strongly non-linear and the features are few.

**The curse of dimensionality, measured.**

```text
== curse of dimensionality: distance contrast ==
d =    2: mean nearest/farthest 0.0166   relative contrast (far - near)/near  82.0781   cube holding 1% of points has side 0.1000
d =    5: mean nearest/farthest 0.1125   relative contrast (far - near)/near   8.9214   cube holding 1% of points has side 0.3981
d =   10: mean nearest/farthest 0.2580   relative contrast (far - near)/near   2.9700   cube holding 1% of points has side 0.6310
d =   20: mean nearest/farthest 0.4179   relative contrast (far - near)/near   1.4080   cube holding 1% of points has side 0.7943
d =   50: mean nearest/farthest 0.5948   relative contrast (far - near)/near   0.6837   cube holding 1% of points has side 0.9120
d =  100: mean nearest/farthest 0.6979   relative contrast (far - near)/near   0.4344   cube holding 1% of points has side 0.9550
d =  200: mean nearest/farthest 0.7758   relative contrast (far - near)/near   0.2896   cube holding 1% of points has side 0.9772
d =  500: mean nearest/farthest 0.8511   relative contrast (far - near)/near   0.1751   cube holding 1% of points has side 0.9908
d = 1000: mean nearest/farthest 0.8946   relative contrast (far - near)/near   0.1179   cube holding 1% of points has side 0.9954
```

With 1,000 points, the farthest point is 83 times farther than the nearest in 2-D, but only 12% farther in 1,000-D.
Between $d = 100$ and $d = 1000$ (ten times the dimension), the relative contrast falls from 0.43 to 0.12. That is roughly the
$1/\sqrt{10} = 0.32$ factor the concentration formula predicts for the spread.

```text
== curse of dimensionality: noise features on wine ==
13 real +    0 noise features: kNN (k = 5) val accuracy 0.9444
13 real +   10 noise features: kNN (k = 5) val accuracy 0.9722
13 real +   30 noise features: kNN (k = 5) val accuracy 0.8333
13 real +  100 noise features: kNN (k = 5) val accuracy 0.8611
13 real +  300 noise features: kNN (k = 5) val accuracy 0.6667
13 real + 1000 noise features: kNN (k = 5) val accuracy 0.6389
```

The trend is clear even though single steps are noisy (10 noise features gaining one wine is luck). Each noise feature,
once standardized, contributes as much to every distance as a real one. With 300 noise features against 13 real ones,
the real signal is a small share of every distance. A linear model (B2.1, B2.2) would learn near-zero weights for the
noise. kNN cannot learn weights at all: every feature counts fully in the distance. Feature selection and dimensionality
reduction (B5.4) are part of using kNN, not optional extras.

## In practice

- **When to use it:** as a quick non-linear baseline with few (say under 20), meaningful, scaled features; when the
  decision boundary is irregular and you have enough data to cover the space; for "find similar" features (similar
  customers, products, documents), usually on learned embeddings with cosine distance; when new labelled data must be
  usable immediately without retraining (just add rows).
- **Failure modes:**
  - *Unscaled features.* Accuracy far below a linear model, and neighbours that agree on one large-unit column and
    nothing else. Inspect which features dominate the distances. Standardize in a pipeline.
  - *Many or irrelevant features.* Accuracy that drops as you add columns, and nearest and farthest distances that are
    nearly equal. Select features, reduce dimension (B5.4), or use a model that learns feature weights.
  - *Slow or memory-hungry prediction.* Latency grows with the training set: every request scans $n$ rows. Use a tree index
    in low dimensions or an approximate index (HNSW) in high ones, and subsample redundant training rows.
  - *Imbalanced classes.* The majority class wins most votes near the boundary. Use distance weighting, class-balanced
    sampling, or tune the decision threshold on the vote share (B4.4).
  - *Categorical features.* Euclidean distance on one-hot columns or ordinal codes may not express real similarity.
    Choose a suitable distance (Hamming, Gower) or encode deliberately.
  - *Duplicates between training and validation.* The validation score for $k = 1$ is unrealistically high if the same entity
    appears in both. Split by group (B1.1).
- **Defaults that work:** `make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))` (5 is scikit-learn's default;
  VERIFY as of 2026-10, scikit-learn 1.9.1), then tune `n_neighbors` over odd values 1–31 and `weights` in `{"uniform", "distance"}` by
  cross-validation. Use `metric="cosine"` for text or embedding vectors. Leave `algorithm="auto"`; it picks brute force or a tree.

## Exercises

Save your work to `my-work/B/b2_3_knn.md` (or `.py`) and ask for a review.

1. **By hand:** Training points $(2, 3)$ class A, $(2, 0)$ class A, $(3, 5)$ class B, $(5, 6)$ class B, $(4, 4)$ class B, and the query $(0, 5)$. (a) Compute the
   Euclidean and Manhattan distances to every point. (b) Predict the class for $k = 1, 3, 5$ with each metric. (c) The two
   features are later found to be in different units: feature 2 should be divided by 2 to be comparable with feature 1.
   Redo the Euclidean $k = 3$ prediction. (d) Using $\mathbb{E}[\lVert\mathbf{u} - \mathbf{v}\rVert^2] = d/6$ and $\operatorname{sd} = \sqrt{7d/180}$, compute the mean and standard deviation of the squared distance
   for $d = 4$ and $d = 400$, and the ratio sd/mean for each.
2. **Code:** Learn which features matter, then use kNN. On the wine split from the lab: (i) compute 5-fold
   cross-validated accuracy on the *training* rows for `make_pipeline(StandardScaler(), KNeighborsClassifier(k))` for odd $k$ from 1 to 31, with
   uniform and with distance weights, and pick the best setting; (ii) add 100 noise features as in the lab and repeat;
   (iii) with the 100 noise features still present, rank features by a univariate score computed *inside* the pipeline
   (`SelectKBest(f_classif, k=13)`) and repeat. Report the best cross-validated accuracy in each case and the validation
   accuracy of the chosen model. Explain why the selector must sit inside the pipeline.
3. **Think:** Your team wants to add "customers like you also bought" to an e-commerce site: represent each customer
   by 200 behavioural features (counts of views and purchases per category, recency, spend) and recommend what their 20
   nearest neighbours bought. List what you would check or change before trusting the neighbours: scaling and
   transformation of the features, the distance, the dimension, the latency at 5 million customers, and how you would
   evaluate whether the neighbours are any good.

## Takeaways

1. kNN stores the training set and predicts with the majority class (or mean target) of the $k$ nearest rows. Small $k$
   overfits and large $k$ oversmooths; choose $k$ on validation data.
2. Distances add up feature differences, so features must be on comparable scales. Standardizing changed wine accuracy from about
   0.7 to about 0.95, far more than any choice of $k$.
3. Compute all distances with $\lVert\mathbf{q}\rVert^2 + \lVert\mathbf{x}\rVert^2 - 2\mathbf{q}^\top\mathbf{x}$ (one matrix product); rank on squared distances; use cosine for text and embeddings
   (for unit vectors it ranks the same as Euclidean).
4. In high dimensions distances concentrate: the squared distance has mean $\propto d$ and relative spread $\propto 1/\sqrt{d}$, so nearest and farthest
   become alike. Irrelevant features drown the real ones, because kNN cannot down-weight them.
5. Training is free and prediction costs $O(nd)$ per query, with the whole training set in memory. At scale, use tree or
   approximate indexes, and remember that approximate means some true neighbours are missed.

**What builds on this:** [B4.5](b4_5_bias_variance.md) ($k$ as a bias–variance dial); [B5.4](b5_4_pca.md) (reducing
dimension before distance-based methods); [B5.6](b5_6_anomaly_detection.md) (distance to neighbours as an anomaly
score); [B6.1](b6_1_recommender_systems.md) (neighbourhood-based recommenders);
[E1.3](../E/e1_3_word2vec.md) (nearest neighbours in embedding space); [I2.1](../I/i2_1_embeddings_semantic_search.md) and
[I2.2](../I/i2_2_postgres_pgvector.md) (semantic search and approximate indexes).

**References:** T. Cover, P. Hart, "Nearest Neighbor Pattern Classification", *IEEE Transactions on Information Theory*
13(1), 1967. K. Beyer, J. Goldstein, R. Ramakrishnan, U. Shaft, "When Is 'Nearest Neighbor' Meaningful?", *ICDT*, 1999.
G. James et al., *An Introduction to Statistical Learning with Applications in Python*, Springer, 2023, chapters 2 and 3.
scikit-learn User Guide, "Nearest Neighbors" (VERIFY link: https://scikit-learn.org/stable/modules/neighbors.html).
