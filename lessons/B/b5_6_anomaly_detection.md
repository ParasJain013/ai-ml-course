# B5.6 Anomaly detection: statistical rules, isolation forest, density-based scores

**Core** · Part B, week 5 (course week 11) · ~1.5 h · Lab:
[`labs/B/b5_6_anomaly_detection.py`](../../labs/B/b5_6_anomaly_detection.py)

## Why it matters

Fraud, intrusions, faulty sensors, broken data pipelines and odd customers share one property: the cases you care about are too rare, too varied or too new
to train a classifier on. B4.4 showed what to do when positives are rare but labelled; this lesson covers the case where you have almost no labels at all.
**Anomaly detection** scores every row by how unlike the bulk of the data it is, and a person reviews the top of the list. In production this is an alerting
system, so it lives or dies by two questions: how many alerts a day can someone review, and what share of them is worth their time. This lesson builds three
detectors from scratch (rule-based, isolation-based, distance-based), checks them against scikit-learn, and evaluates them honestly with 15 labelled anomalies.

## Prerequisite check

- Can you compute Tukey's fences and a z-score by hand, and say why one extreme value can hide itself from a z-score? If not, revisit
  [B1.2](b1_2_data_cleaning.md).
- Can you describe how a decision tree splits the data on one feature at a threshold? If not, revisit [B3.1](b3_1_decision_trees.md).
- Can you compute Euclidean distances between rows with NumPy broadcasting, and say why features must be scaled first? If not, revisit
  [B2.3](b2_3_knn.md) and [B1.3](b1_3_feature_engineering.md).
- Can you read a precision–recall curve and say what the baseline of average precision is? If not, revisit [B4.1](b4_1_classification_metrics.md).
- Can you compute a percentile bootstrap confidence interval? If not, revisit [A6.2](../A/a6_2_confidence_intervals_bootstrap.md).

## Intuition

An anomaly detector produces a **score** per row, higher meaning "stranger", and a threshold turns scores into alerts. Three families of score:

1. **Statistical rules.** Per feature, how far is the value from the typical range? z-scores measure distance in standard deviations; **Tukey's fences** in
   **interquartile ranges** (B1.2). Cheap and explainable ("amount is 9 IQRs above normal"), but they look at one feature at a time, and a row can be strange
   only in *combination* (a small transaction at 3 a.m. from a new country), with every single feature normal.
2. **Isolation.** Cut the data with random splits: pick a feature at random, pick a threshold uniformly between its min and max, repeat on each side. A point far
   from the others is cut off alone after a few splits; a point in a dense crowd needs many. Average the number of cuts over many random trees: few cuts means
   anomalous. This is the **isolation forest** (Liu, Ting & Zhou, 2008). It never computes a distance or a density, and it handles many features.
3. **Distance or density.** A normal point has close neighbours; an anomaly sits in a sparse region. The distance to the $k$-th nearest neighbour is a simple
   density proxy (large distance = low density). The **local outlier factor** (LOF) compares each point's density with its neighbours' densities, so it can
   flag a point that is sparse *for its neighbourhood* even when another cluster is sparser overall.

Two settings use the same scores differently. In **outlier detection** you score a batch that may already contain anomalies (yesterday's transactions).
In **novelty detection** you fit on a reference set believed clean (last month's healthy traffic) and score new rows against it. Novelty detection is
easier when the reference really is clean; contaminated reference data teaches the detector that the anomalies are normal.

> **Bridge:** this is the logic of a monitoring alert. A static threshold on one metric (p99 latency > 2 s) is a statistical rule; an anomaly model over
> many metrics catches combinations no single threshold sees. The alert budget is the on-call engineer's patience. **Where it breaks:** an alert rule
> is written by someone who knows what "bad" looks like; an anomaly detector only knows "unusual". Unusual is not the same as bad: a successful marketing campaign
> is unusual. Every anomaly score needs human review or a downstream check before it becomes an action.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{X}$ | data matrix, one row per example | `(n, d)` |
| $x_i$ | one value of one feature (rules work per feature) | scalar |
| $\bar{x}$, $\hat\sigma$ | mean and standard deviation (dividing by $n$) of a feature | scalars |
| $Q_1, Q_3$ | first and third quartiles; $\mathrm{IQR} = Q_3 - Q_1$ | scalars |
| $\psi$ | subsample size used to grow one isolation tree (256 by default) | int |
| $h(\mathbf{x})$ | path length: number of splits from the root to the leaf where $\mathbf{x}$ lands | scalar |
| $c(m)$ | average path length of a point in a random tree built on $m$ points | scalar |
| $H(i)$ | harmonic number $1 + \frac12 + \dots + \frac1i$ | scalar |
| $s(\mathbf{x})$ | isolation-forest anomaly score in $(0, 1]$ | scalar |
| $k$ | number of neighbours (not the number of classes $K$) | int |
| $\delta_k(\mathbf{x})$ | distance from $\mathbf{x}$ to its $k$-th nearest other row | scalar |

**Rules as scores.** For feature $j$, $z_{ij} = (x_{ij} - \bar{x}_j)/\hat\sigma_j$. A row's score is $\max_j |z_{ij}|$, its worst feature. For Tukey's fences, the
row's score is the largest distance outside the fences, in IQR units, $\max_j \max\!\big(\frac{Q_{1j} - 1.5\,\mathrm{IQR}_j - x_{ij}}{\mathrm{IQR}_j}, \frac{x_{ij} - Q_{3j} - 1.5\,\mathrm{IQR}_j}{\mathrm{IQR}_j}, 0\big)$:
0 inside the fences, 1 if one IQR beyond a fence.

**Masking and swamping.** B1.2 proved $|z| \le \sqrt{n - 1}$: an outlier inflates $\hat\sigma$ and so shrinks its own z-score. With several outliers it gets
worse: they inflate $\hat\sigma$ together and hide each other. That is **masking**. They also pull $\bar{x}$ towards themselves, so normal points on the
other side look further from the mean than they are and can be flagged: **swamping**. Quartiles barely move until more than a quarter of the data is
extreme, which is why the robust rule resists both.

**The isolation forest score.** An isolation tree is grown on $\psi$ rows drawn without replacement. At each node, pick a feature uniformly at random and a split
value uniformly between that feature's min and max in the node; stop when a node holds one row or the depth reaches $\lceil \log_2 \psi \rceil$ (8 for $\psi = 256$;
anomalies are isolated well above that depth, so deeper splits only cost time). To score $\mathbf{x}$, drop it down each tree. If it reaches a leaf that still holds
$m > 1$ training rows, the tree stopped early, so add $c(m)$, the expected number of splits those $m$ rows would still have needed. Average over the trees to
get $E[h(\mathbf{x})]$, and normalize:
$$s(\mathbf{x}) = 2^{-E[h(\mathbf{x})]/c(\psi)}.$$
If $E[h] = c(\psi)$ (an average point), $s = 2^{-1} = 0.5$. If $E[h] \to 0$ (isolated at once), $s \to 1$. If $E[h]$ is much larger than $c(\psi)$, $s$ falls towards 0.
Dividing by $c(\psi)$ makes scores from forests with different $\psi$ comparable.

**Where $c(m)$ comes from.** Liu et al. borrow it from binary search trees (BSTs): a tree grown on $m$ points has $m - 1$ internal split nodes, like a BST built from
$m - 1$ keys, and a new point's path length is the cost of an unsuccessful search in that BST. (Isolation trees split at random *values*, not at random data
points, so the match is approximate; it is a normalizer, not an exact expectation.) The derivation for a random BST with $m - 1$ keys:

1. Sort the keys $1, \dots, m - 1$. A new point falls into one of the $m$ gaps between them; call its gap $g$, with each gap equally likely.
2. Key $j$ is on the new point's path exactly when $j$ was inserted before every key lying strictly between $j$ and the gap. (If some key between them had been
   inserted first, it would have sent $j$ and the gap down different sides, and the search would never meet $j$.) The keys from $j$ up to the gap form a run of
   $r$ keys, $j$ included, and in a random insertion order each of them is equally likely to be first, so the probability is $1/r$.
3. For a gap with $a$ keys to its left and $b$ to its right, the keys on the left give runs of length $1, 2, \dots, a$ and those on the right $1, \dots, b$, so the expected
   number of keys on the path is $H(a) + H(b)$.
4. Average over the $m$ gaps ($a = 0, \dots, m - 1$ and $b = m - 1 - a$): both sums are the same, so the average is $\frac{2}{m}\sum_{a=1}^{m-1} H(a)$.
5. The identity $\sum_{a=1}^{N} H(a) = (N + 1)H(N) - N$ (each $\frac1t$ appears in $H(t), \dots, H(N)$, that is $N - t + 1$ times, so the sum is
   $\sum_{t=1}^{N} \frac{N + 1 - t}{t} = (N + 1)H(N) - N$) with $N = m - 1$ gives
   $\frac{2}{m}\big(m\,H(m - 1) - (m - 1)\big)$, so
$$c(m) = 2H(m - 1) - \frac{2(m - 1)}{m}.$$
Using $H(m) = H(m - 1) + \frac1m$, the same number is $2H(m) - 2$; check $c(2) = 2 \cdot 1 - 1 = 1$: two points need exactly one split. The lab simulates random BSTs
and matches the formula.

**The harmonic approximation.** $H(i) = \sum_{t=1}^{i} \frac1t$ is the area of $i$ bars of width 1 and heights $1, \frac12, \dots$; the curve $1/t$ runs just under their
tops, and the area under it from 1 to $i$ is $\ln i$. The gap between bars and curve adds up to a constant, the Euler–Mascheroni constant $\gamma \approx 0.5772$,
so $H(i) \approx \ln i + \gamma$ (for $i = 255$: exact 6.1204, approximation 6.1185). Liu et al. and scikit-learn use this approximation (VERIFY as of 2026-10).

**The k-NN distance score.** $\delta_k(\mathbf{x}_i) = $ the $k$-th smallest of $\lVert\mathbf{x}_i - \mathbf{x}_l\rVert_2$ over $l \ne i$. All pairwise squared distances at once:
$\lVert\mathbf{x}_i - \mathbf{x}_l\rVert^2 = \lVert\mathbf{x}_i\rVert^2 + \lVert\mathbf{x}_l\rVert^2 - 2\,\mathbf{x}_i^\top\mathbf{x}_l$, a matrix of shape `(n, n)` (B2.3). Distances depend on
the units of every feature, so scale first. z-scores, fences and the isolation forest do not need scaling: each looks at one feature at a time, and a linear
rescaling of a feature moves its values, mean, quartiles and uniform split points together.

**From a score to alerts.** Any score becomes a decision with a threshold. If reviewers can handle $B$ alerts per batch (the **alert budget**; in this lesson only, $B$ is not a batch size), flag the top $B$: that is the threshold.
scikit-learn's `contamination` parameter does exactly this, choosing the threshold so that a given share of the fitted rows is flagged; it does not change the
scores or the model. Evaluate with labels where you have any: **average precision** (AP, B4.1; its baseline is the anomaly share) for the whole ranking, and
**precision at k** (the share of true anomalies among the top $k$) at the alert budget.

## Worked example by hand

**Masking.** Eight values: $10, 11, 12, 12, 13, 14, 50, 52$. Mean $= 174/8 = 21.75$. Deviations: $-11.75, -10.75, -9.75, -9.75, -8.75, -7.75, 28.25, 30.25$; their squares sum to
$2293.5$, so $\hat\sigma = \sqrt{2293.5/8} = \sqrt{286.69} = 16.93$. The z-scores of 50 and 52 are $28.25/16.93 = 1.67$ and $30.25/16.93 = 1.79$. Neither reaches 2,
let alone 3 (the ceiling here is $\sqrt{7} = 2.65$). Without the pair, the other six have mean 12 and $\hat\sigma = 1.29$, and 50 would have $z = 29.4$: the two outliers
mask each other.

**Tukey's fences** with NumPy's linear rule: $Q_1$ at position $7 \times 0.25 = 1.75$ (0-based) of the sorted values, $11 + 0.75 \times (12 - 11) = 11.75$; $Q_3$ at position 5.25,
$14 + 0.25 \times (50 - 14) = 23$. IQR $= 11.25$, fences $[11.75 - 16.875,\ 23 + 16.875] = [-5.125, 39.875]$: 50 and 52 are flagged. $Q_3$ moved (two of eight values are extreme, close to
the quarter that quartiles tolerate) but not enough to hide them.

**Isolation in one dimension.** Points $1, 2, 3, 4, 10$. The first split value is uniform on $[1, 10]$. It isolates 10 if it falls in $(4, 10)$: probability
$6/9 = 0.67$. Isolating 2 needs one cut between 1 and 2 *and* one between 2 and 3, each in a window of width 1: many splits on average.

**The normalizer and the score** for $\psi = 256$: $H(255) \approx \ln 255 + 0.5772 = 5.5413 + 0.5772 = 6.1185$, so
$c(256) = 2 \times 6.1185 - 2 \times 255/256 = 12.2370 - 1.9922 = 10.2448$. A point isolated after 3 splits on average scores $s = 2^{-3/10.2448} = 2^{-0.2928} = 0.816$;
one at $10.2448$ scores exactly $0.5$; one at 12 scores $2^{-1.1713} = 0.444$.

**k-NN distance.** Points $(0, 0), (1, 0), (0, 1), (1, 1), (4, 4)$. Each corner of the unit square has two neighbours at distance 1, so $\delta_1 = \delta_2 = 1$. The point $(4, 4)$:
its nearest is $(1, 1)$ at $\sqrt{9 + 9} = 4.24$; next are $(1, 0)$ and $(0, 1)$ at $\sqrt{9 + 16} = 5$. So $\delta_1 = 4.24$, $\delta_2 = 5$.

```text
== worked example by hand: rules ==
x = [10.0, 11.0, 12.0, 12.0, 13.0, 14.0, 50.0, 52.0]   mean 21.7500   std (ddof=0) 16.9318
z = [-0.694, -0.6349, -0.5758, -0.5758, -0.5168, -0.4577, 1.6685, 1.7866]   largest possible |z| for n = 8: sqrt(7) = 2.6458
Q1 11.7500  Q3 23.0000  IQR 11.2500  fences [-5.1250, 39.8750]  flagged [50.0, 52.0]
without the two outliers: mean 12.0000  std 1.2910  -> z of 50 would be 29.43

== worked example by hand: isolation and k-NN ==
1-D points [1.0, 2.0, 3.0, 4.0, 10.0]: P(first uniform cut isolates 10) = (10 - 4) / (10 - 1) = 0.6667
H(255) exact 6.1204   ln(255) + gamma 6.1185
c(256) = 2 H(255) - 2 * 255/256 = 10.2448   (exact-H version 10.2487)
c(256) = 2 H(256) - 2 (same number) = 10.2487
mean path length 3.0000 -> score s = 2^(-h / c) = 0.8163
mean path length 10.2448 -> score s = 2^(-h / c) = 0.5000
mean path length 12.0000 -> score s = 2^(-h / c) = 0.4440
k-NN scores, k = 1: [1.0, 1.0, 1.0, 1.0, 4.2426]   k = 2: [1.0, 1.0, 1.0, 1.0, 5.0]

== check: c(n) is the mean depth of a new point in a random BST ==
n =  2 points: simulated 1.000   2 H(n) - 2 = 1.000
n =  5 points: simulated 2.572   2 H(n) - 2 = 2.567
n = 10 points: simulated 3.849   2 H(n) - 2 = 3.858
identity at N = 7: sum H(g) = 13.742857   (N+1) H(N) - N = 13.742857
```

The two `c(256)` lines differ only in using the approximation (10.2448) or the exact harmonic number (10.2487). The simulation drops a random point into 20,000 random BSTs
for each size and averages its depth; it agrees with $2H(n) - 2$ to within simulation noise.

## Key plot

![Precision-recall curves on 372 breast-cancer rows with 15 anomalies: the isolation forest (AP 0.57) keeps precision between 0.67 and 0.87 up to recall 0.47, while the z-score (0.37), IQR (0.33) and k-NN (0.41) curves hover around 0.5; all lie far above the random baseline of 0.04](img/b5_6_pr_curves.png)

Each curve walks down one detector's ranking. Every detector is roughly ten times better than random (AP 0.33–0.57 against 0.04), and the isolation forest keeps its
precision high for longest. But with 15 anomalies, each step of recall is one row: the curves are jagged, and the **bootstrap** intervals below overlap heavily.

## From scratch in NumPy

**The data.** scikit-learn's breast cancer dataset (Wisconsin diagnostic: 30 numeric measurements of cell nuclei from 569 biopsies). We keep all 357 benign rows and a
random 15 of the 212 malignant ones, and pretend the malignant rows are rare unexplained anomalies. The labels are used only to evaluate. Features are scaled with
the batch's median and IQR: no labels are involved, and unlike the mean and standard deviation, the median and IQR are hardly moved by the anomalies themselves.

```text
== data ==
X (372, 30): 15 anomalies, prevalence 0.0403 (the AP of a random score)
```

**Rules as scores.** Each row gets its worst feature's $|z|$, or its largest excess beyond a fence:

```python
def zscore_score(X):
    Z = (X - X.mean(axis=0)) / X.std(axis=0)          # (n, d)
    return np.abs(Z).max(axis=1)                      # (n,)

def iqr_score(X):
    q1, q3 = np.quantile(X, [0.25, 0.75], axis=0)     # (d,) each
    iqr = q3 - q1
    below = (q1 - 1.5 * iqr - X) / iqr
    above = (X - q3 - 1.5 * iqr) / iqr
    return np.maximum(np.maximum(below, above), 0).max(axis=1)
```

```text
== from scratch: z-score and IQR scores ==
rows with some |z| > 3: 45 (anomalies among them 12)
rows outside a fence in some feature: 102 (anomalies among them 13)
-> a per-feature rule over 30 features flags many normal rows: each feature gets its own chance to fire
```

With 30 features, a fixed per-feature cut-off fires often: 102 of 372 rows are outside some fence. That is the multiple-comparisons problem of [A6.3b](../A/a6_3b_multiple_comparisons_ab_tests.md) in another form.
Used as *scores* (rank by the worst feature) the rules still carry information; used as fixed thresholds they swamp the reviewer.

**Isolation forest.** A tree is a nested tuple; growing it is recursion on random splits, and scoring follows one path down:

```python
def grow_tree(X, depth, max_depth, gen):
    m = len(X)
    if m <= 1 or depth >= max_depth:
        return ("leaf", m)
    lo, hi = X.min(axis=0), X.max(axis=0)
    usable = np.flatnonzero(hi > lo)                  # a constant feature cannot split
    if len(usable) == 0:
        return ("leaf", m)
    j = int(gen.choice(usable))
    v = float(gen.uniform(lo[j], hi[j]))
    left = X[:, j] < v
    return ("split", j, v, grow_tree(X[left], depth + 1, max_depth, gen),
            grow_tree(X[~left], depth + 1, max_depth, gen))

def path_length(x, tree, depth=0):
    if tree[0] == "leaf":
        return depth + c_factor(tree[1])              # unfinished leaf: add the expected rest
    _, j, v, left, right = tree
    return path_length(x, left if x[j] < v else right, depth + 1)
```

The score averages `path_length` over 100 trees, each grown on 256 rows sampled without replacement, and applies $s = 2^{-E[h]/c(256)}$.

```text
== from scratch: isolation forest ==
100 trees of 256 rows, max depth 8
mean score: normal rows 0.4146   anomalies 0.5581   rows above 0.5: 32
```

The paper's rule of thumb, "$s$ clearly above 0.5 is anomalous", is a guide, not a calibrated threshold: 32 rows exceed 0.5 here, more than twice a budget of 15 alerts.

**k-NN distance.** One matrix of squared distances, the diagonal set to infinity so a row is not its own neighbour, then the $k$-th smallest per row:

```text
== from scratch: k-NN distance score ==
k = 5: median score normal rows 2.6283   anomalies 7.4204
```

## With scikit-learn

```text
== with scikit-learn ==
IsolationForest: score range [0.3498, 0.7219]   Spearman rank correlation with ours 0.9671
  (different random trees, so not equal; same algorithm, so the rankings agree closely)
k-NN distance matches sklearn.neighbors.NearestNeighbors: OK
LocalOutlierFactor (n_neighbors=20): 28 rows with LOF > 1.5
contamination=0.04: flags 15 rows (8 anomalies) - it only sets the threshold
contamination=0.1: flags 38 rows (11 anomalies) - it only sets the threshold
```

- `IsolationForest.score_samples` returns *minus* the paper's $s$ (scikit-learn's convention: higher means more normal), so we negate it. `decision_function` subtracts
  the `contamination`-based threshold, and `predict` returns $-1$ for anomalies and $1$ for normal rows (VERIFY as of 2026-10). The forests are grown with
  different random numbers, so the scores cannot match exactly. A **Spearman correlation** of ranks of 0.97 says the two rank the rows almost the same way, which is what an
  anomaly score is for.
- `NearestNeighbors(n_neighbors=k + 1).kneighbors(X)` on the fitted rows returns each row itself first, at distance 0, so the $k$-th *other* neighbour is in column $k$.
  Our matrix version matches it exactly.
- `LocalOutlierFactor(n_neighbors=20)` is the density-ratio score: about 1 inside a cluster, well above 1 for a point sparser than its neighbours. With
  `novelty=False` (the default) it only scores the rows it was fitted on; set `novelty=True` to call `score_samples` on new rows.
- Raising `contamination` from 0.04 to 0.10 flags 38 rows instead of 15. The ranking is identical; only the cut moved.

**Evaluation with few labels.**

```text
== evaluation: AP, precision@15, bootstrap CI ==
score                        AP           95% CI   P@15
z-score (max |z|)        0.3739   [0.1857, 0.6561] 0.5333
IQR (max excess)         0.3288   [0.1639, 0.6123] 0.4667
isolation forest         0.5725   [0.3080, 0.8303] 0.6000
sklearn IsolationForest  0.4976   [0.2604, 0.7577] 0.5333
k-NN distance (k=5)      0.4058   [0.2192, 0.6897] 0.5333
LOF (sklearn)            0.4802   [0.2625, 0.7675] 0.5333
random score: AP = prevalence = 0.0403
our isolation forest with seeds 1-5: AP [0.5929, 0.5545, 0.5633, 0.5477, 0.5432]  (the forest's own randomness)
```

- Every detector is about ten times better than a random ranking. With a budget of 15 alerts, between 7 and 9 of them are real anomalies.
- **The intervals are wide and overlap.** With 15 positives, AP's 95% interval spans about 0.5. The isolation forest looks best, but these labels cannot prove it
  beats the k-NN score; a paired bootstrap of the difference (A6.2) or more labels would be needed. Report the interval, not just the point.
- The forest's own randomness moves AP between 0.54 and 0.59 across seeds, and scikit-learn's forest landed at 0.50. More trees (`GPU_SCALE`) shrink that.
- Precision at the budget (P@15) is the number the reviewers feel: 8 of 15 alerts real is a usable queue; 2 of 15 would be ignored within a week.

## In practice

- **When to use it:** labels are absent, very few, or describe only past kinds of anomaly; you need a ranked queue for human review or a gate before data
  enters a pipeline (data-quality checks, drift alerts in [J2.2](../J/j2_2_monitoring_drift.md)). If you have hundreds of labelled positives of the kind you
  care about, a supervised classifier with B4.4's techniques usually beats an unsupervised score; the two combine well (anomaly scores as features).
- **Failure modes:**
  - *Unusual is not interesting.* The top scores are often data errors, new-but-legitimate customers or a holiday. You notice it when reviewers dismiss most
    alerts; fix it by excluding known causes, adding context features, or scoring residuals from a forecast (B6.3) instead of raw values.
  - *Unscaled features in a distance score:* one feature in large units decides every neighbour. You notice it when the top anomalies all differ in the same
    one column.
  - *Contaminated reference data in novelty detection:* the detector learns that the anomalies are normal (exercise 2). Clean the reference set, or use the
    outlier setting with a robust method.
  - *Per-feature rules on many features:* each feature gets a chance to fire, so false alarms grow with $d$. Rank by a combined score instead.
  - *Masking and swamping* with mean-and-standard-deviation rules when several anomalies are present: use medians and IQRs (or the MAD).
  - *Evaluating on a handful of labels without intervals:* two detectors whose APs differ by 0.1 may be indistinguishable. Bootstrap, and add labels from reviewer feedback.
- **Defaults that work:** scale with `RobustScaler` (median and IQR) for distance-based scores; `IsolationForest(n_estimators=200, max_samples=256)` as the
  first detector (scikit-learn 1.9.1, default `max_samples="auto"` = min(256, n); VERIFY as of 2026-10); `LocalOutlierFactor(n_neighbors=20)` as the
  density-based second opinion; set the threshold from the review budget, not from `contamination="auto"`; log every reviewer verdict, so the
  labels you need for evaluation accumulate.

## Exercises

Save your work to `my-work/B/b5_6_anomaly_detection.md` (or `.py`) and ask for a review.

1. **By hand:** (a) For the values $3, 4, 4, 5, 6, 40$, compute the mean, $\hat\sigma$ (dividing by $n$), the z-score of 40, and Tukey's fences (NumPy's linear quartiles).
   Does $|z| > 3$ flag 40? Do the fences? Why? (b) Compute $c(8)$ with the exact harmonic number $H(7)$. What score $s$ does a point with mean path length 1.5 get
   in a forest grown with $\psi = 8$? (c) For the points $(0, 0), (2, 0), (0, 2), (2, 2), (1, 1), (6, 1)$, compute each point's distance to its 2nd nearest neighbour. Which is the most
   anomalous, and is $(1, 1)$, the centre of the square, more or less anomalous than the corners? (d) An isolation forest is fitted with `contamination=0.02`
   on 5,000 rows a day, and the fraud team can review 50 alerts a day. What does the contamination setting do, and what would you set instead?
2. **Code:** Novelty detection. From the breast cancer data, take 200 random benign rows as a clean reference set, and score a new batch of the other 157 benign
   rows plus 15 malignant ones. Fit the robust scaler on the reference rows only. Implement the k-NN distance from each new row to its 5th nearest *reference* row
   (check it against `NearestNeighbors`), and compare it with `IsolationForest(n_estimators=200)` fitted on the reference: report AP with a bootstrap CI and
   precision at 15. Then repeat with 15 *other* malignant rows slipped into the reference set, and explain what changes.
3. **Think:** A payments company asks you for a detector that flags suspicious merchants from daily features (volume, refund rate, average ticket, share of
   foreign cards). They have 40 confirmed fraud cases from the last year, and two analysts can review 30 merchants a day between them. Would you build an
   unsupervised detector, a supervised classifier, or both? How would you evaluate it before launch, set the threshold, and keep it useful after launch?

## Takeaways

1. An anomaly detector is a score plus a threshold; set the threshold from the review budget, and `contamination` is nothing more than that threshold.
2. Mean-and-standard-deviation rules suffer from masking and swamping (and $|z| \le \sqrt{n - 1}$); median-and-IQR rules resist both, but per-feature rules miss
   combinations and fire more often as features are added.
3. An isolation forest scores how few random splits isolate a point: $s = 2^{-E[h]/c(\psi)}$ with $c(m) = 2H(m - 1) - 2(m - 1)/m$, the mean path length of a
   random binary search tree. It needs no scaling and handles many features.
4. Distance to the $k$-th neighbour (and LOF) measure density; they need scaled features and, for novelty detection, a clean reference set.
5. With few labels, evaluate the ranking with AP and precision at the budget, with bootstrap intervals; here every detector beat random tenfold, and 15 labels
   could not separate them.

**What builds on this:** P3 (anomalies inside the segmentation project; two detectors and a manual review of the top 20); [B6.3](b6_3_time_series_forecasting.md)
(anomalies as large forecast residuals); [G1.1](../G/g1_1_autoencoders.md) (reconstruction error as an anomaly score);
[J2.2](../J/j2_2_monitoring_drift.md) (drift detection and alerting in production).

**References:** F. T. Liu, K. M. Ting, Z.-H. Zhou, "Isolation forest", *ICDM*, 2008. M. M. Breunig, H.-P. Kriegel, R. T. Ng, J. Sander, "LOF: identifying
density-based local outliers", *SIGMOD*, 2000. S. Ramaswamy, R. Rastogi, K. Shim, "Efficient algorithms for mining outliers from large data sets", *SIGMOD*, 2000
(the $k$-NN distance score). scikit-learn User Guide, "Novelty and Outlier Detection" (VERIFY link: https://scikit-learn.org/stable/modules/outlier_detection.html).
