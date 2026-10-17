# B3.1 Decision trees: Gini and entropy splits built from scratch

**Core** · Part B, week 3 (course week 09) · ~2.5 h · Lab:
[`labs/B/b3_1_decision_trees.py`](../../labs/B/b3_1_decision_trees.py)

## Why it matters

On tabular data (rows from a database, columns of mixed meaning and units), the models that win in practice are
ensembles of **decision trees**: random forests (B3.2) and gradient boosting (B3.3), both built from the algorithm in
this lesson. A single tree needs no feature scaling, captures interactions without feature engineering, and a shallow
one reads as if/else rules. A deep one memorizes the training rows and gives confident wrong answers on new ones;
understanding why is understanding why forests and boosting exist. In production the usual mistakes are trusting a
deep tree's 0/1 probabilities, and treating its rules as stable when a retrain rewrites them.

## Prerequisite check

- Can you compute the entropy of a distribution such as $(0.25, 0.75)$, in bits? If not, revisit
  [A6.4](../A/a6_4_entropy_cross_entropy_kl.md).
- Can you explain why the validation set, not the training set, decides how complex a model should be? If not,
  revisit [B1.1](b1_1_ml_workflow.md).
- Do you know why we standardize features for logistic regression and kNN? (Trees are the exception.) If not, revisit
  [B1.3](b1_3_feature_engineering.md).
- Can you compute log-loss for a predicted probability, and say why a probability of exactly 0 on the true class is
  a disaster? If not, revisit [B2.1](b2_1_logistic_regression.md).

## Intuition

A decision tree plays twenty questions with the data. Each question looks at one feature and asks "is it at most some
threshold?": *is `worst concave points` ≤ 0.14?* The answer sends the example left or right, to the next question, until
it reaches a **leaf**, which gives the prediction: the majority class of the training rows that ended up there (or,
for a probability, their class shares).

Learning a tree means choosing the questions. The algorithm used by scikit-learn is **CART** (Classification And
Regression Trees, Breiman and colleagues, 1984). It is greedy:

1. Start with all training rows in one node.
2. Try every feature and every threshold between two neighbouring values. For each candidate, split the rows in two
   and measure how *mixed* the classes are in the two children.
3. Keep the split that leaves the children least mixed. Recurse into each child.
4. Stop when a node is pure (one class), when a depth limit is reached, or when a split would leave too few rows in a
   child.

"How mixed" is the node's **impurity**, measured by **Gini impurity** or by **entropy**. Both are 0 for a pure node and
largest for a 50/50 node. The best split is the one that lowers the impurity the most; that drop is the split's gain
(with entropy, the **information gain**).

Three consequences follow straight from this recipe:

- **No scaling needed.** A split only compares a feature with a threshold. Multiply `mean area` by 1,000 or take its log,
  and the rows are sorted in the same order, so the tree makes the same splits (with rescaled thresholds).
- **Non-linear for free.** Splitting on radius and then, inside one branch, on texture creates an interaction:
  texture matters only for large tumours. A linear model would need a product feature for that (B1.3).
- **Greedy, so not optimal, and unstable.** The root split is chosen without looking ahead, and everything below depends
  on it. Change a few training rows and a different feature may win at the root, and the whole tree changes. Finding
  the best tree overall is computationally intractable; greedy search is what everyone uses.

> **Bridge:** a fitted tree is a nested `if/else` routing function, the kind you would write by hand for
> feature-flag targeting or a pricing rule: `if (country === "DE") { if (plan === "pro") ... }`. **Where it breaks:** you
> write the rules once and they mean what you intended. A tree's rules are learned from one sample of data, and a
> retrain on next month's sample can produce a completely different set of rules with similar accuracy. The rules
> describe that sample, not the world.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{X}$, $\mathbf{y}$ | training features and labels, $y_i \in \{1, \dots, K\}$ | `(n, d)`, `(n,)` |
| $S$ | the set of training rows in a node; $m = \lvert S\rvert$ is how many | — |
| $p_k$ | share of class $k$ in the node: (rows of class $k$ in $S$) / $m$ | — |
| $G(S)$ | Gini impurity of the node | scalar |
| $H(S)$ | entropy of the node, in bits ($\log_2$) | scalar |
| $j$, $t$ | split feature (a column of $\mathbf{X}$) and threshold | int, scalar |
| $S_L$, $S_R$ | rows with $x_j \le t$ and with $x_j > t$; sizes $m_L$, $m_R$ | — |
| $I(\cdot)$ | either impurity, $G$ or $H$ | scalar |

**Step 1: two ways to measure "mixed".** For a node with class shares $p_1, \dots, p_K$:

$$G(S) = 1 - \sum_{k=1}^{K} p_k^2, \qquad H(S) = -\sum_{k=1}^{K} p_k \log_2 p_k \quad (\text{with } 0\log 0 = 0).$$

Gini has a direct reading: pick a row from the node at random and guess its class by drawing a random label with the
node's class shares. $\sum_k p_k^2$ is the chance the guess is right, so $G$ is the chance it is wrong. Entropy is the
average number of yes/no questions needed to identify the class (A6.4). For two classes with shares $(p, 1 - p)$:
$G = 1 - p^2 - (1 - p)^2 = 1 - p^2 - 1 + 2p - p^2 = 2p(1 - p)$, which peaks at 0.5, and $H$ peaks at 1 bit, both at $p = 0.5$. We use bits here because scikit-learn does;
the base only rescales $H$ by a constant, so it never changes which split wins.

**Step 2: the quality of a split.** A split sends $m_L$ rows left and $m_R$ right. Its cost is the children's impurity,
weighted by how many rows each child gets:

$$I_{\text{split}} = \frac{m_L}{m}\,I(S_L) + \frac{m_R}{m}\,I(S_R), \qquad \text{gain} = I(S) - I_{\text{split}}.$$

The weighting matters: a pure child holding 2 rows is worth far less than a pure child holding 200. The parent's
impurity $I(S)$ is the same for every candidate split, so minimizing $I_{\text{split}}$ and maximizing the gain choose the same split.

**Step 3: which thresholds to try.** Sort the node's rows by feature $j$: $x_{(1)} \le x_{(2)} \le \dots \le x_{(m)}$. Any threshold between two
neighbouring values gives the same partition, so only $m - 1$ cuts matter, one between each neighbouring pair, and only if
the two values differ (no threshold separates equal values). CART uses the midpoint $t = (x_{(i)} + x_{(i+1)})/2$. Sorting also makes
the search fast: walking along the sorted rows, the left child's class counts are a running (cumulative) sum and the
right child's are the total minus that, so all $m - 1$ candidates for one feature cost one sort plus one pass:
$O(m \log m)$ per feature per node.

**Step 4: why not just count mistakes?** The obvious impurity is the misclassification rate, $1 - \max_k p_k$. It fails as
a guide for growing trees. Take a parent with 400 rows of each class. Split A gives children $(300, 100)$ and $(100, 300)$.
Split B gives $(200, 400)$ and $(200, 0)$. Both leave 200 rows misclassified (error 0.25), so the error rate cannot tell
them apart. Gini can: split A costs $\frac12(0.375) + \frac12(0.375) = 0.375$, split B costs $\frac34 \cdot \frac49 + \frac14 \cdot 0 = 0.333$. Gini prefers B because its pure
child is finished: it needs no further splits. Gini and entropy are strictly concave in the class shares (A4.5: the
curve lies above its chords, so the impurity of an average is more than the average of the impurities). The parent's
shares are the size-weighted average of the children's, $\mathbf{p}_S = \frac{m_L}{m}\mathbf{p}_L + \frac{m_R}{m}\mathbf{p}_R$, so $I(S) \ge \frac{m_L}{m}I(S_L) + \frac{m_R}{m}I(S_R) = I_{\text{split}}$, with equality only
when the children's shares equal the parent's. So any split that moves the children's shares apart from the
parent's has positive gain. The error rate is flat over wide ranges,
so many useful splits have zero gain and the search would stall.

**Step 5: growing and predicting.** Apply Steps 2–3 at the root over all $d$ features, keep the best $(j, t)$, send the rows
left and right, and recurse. Stop at a node when (a) it is pure, (b) its depth equals `max_depth`, or (c) no split leaves
at least `min_samples_leaf` rows on each side. A leaf predicts the class shares of its training rows, and the
predicted class is the largest share. A fully grown tree (no limits) keeps splitting until every leaf is pure, so it
gets every training row right. It has memorized them.

**Step 6: regression trees.** For a numeric target, a leaf predicts the *mean* of its rows' targets, and the impurity
is the mean squared deviation from that mean (the node's variance), $I(S) = \frac1m\sum_{i \in S}(y_i - \bar{y}_S)^2$. The split search is identical. A
regression tree is a piecewise-constant function: a staircase. B3.3 builds boosting out of shallow regression trees,
and Exercise 2 asks you to implement them.

## Worked example by hand

Six points, two features, four negatives and two positives:

| Point | $x_1$ | $x_2$ | $y$ |
|---|---|---|---|
| P1 | 1 | 3 | 0 |
| P2 | 2 | 4 | 0 |
| P3 | 3 | 5 | 0 |
| P4 | 4 | 2 | 1 |
| P5 | 5 | 1 | 0 |
| P6 | 6 | 6 | 1 |

**Root impurity.** Shares $(4/6, 2/6)$. $G = 1 - (0.6667^2 + 0.3333^2) = 1 - (0.4444 + 0.1111) = 0.4444$.
$H = -(0.6667\log_2 0.6667 + 0.3333\log_2 0.3333) = -(0.6667 \times (-0.5850) + 0.3333 \times (-1.5850)) = 0.3900 + 0.5283 = 0.9183$ bits.

**Feature $x_1$.** Sorted by $x_1$ the labels read 0, 0, 0, 1, 0, 1. Try the cut at $t = 3.5$: the left child is $\{P1, P2, P3\}$,
all 0, so $G_L = 0$. The right child $\{P4, P5, P6\}$ has labels 1, 0, 1: $G_R = 1 - (1/9 + 4/9) = 0.4444$. Weighted: $\frac36 \cdot 0 + \frac36 \cdot 0.4444 = 0.2222$.
Gain: $0.4444 - 0.2222 = 0.2222$. The other four cuts, worked the same way:

| Cut on $x_1$ | Left labels | Right labels | $I_{\text{split}}$ (Gini) | Gain |
|---|---|---|---|---|
| 1.5 | 0 | 0,0,1,0,1 | $\frac56 \times 0.48 = 0.4000$ | 0.0444 |
| 2.5 | 0,0 | 0,1,0,1 | $\frac46 \times 0.5 = 0.3333$ | 0.1111 |
| **3.5** | 0,0,0 | 1,0,1 | $\frac36 \times 0.4444 = 0.2222$ | **0.2222** |
| 4.5 | 0,0,0,1 | 0,1 | $\frac46 \times 0.375 + \frac26 \times 0.5 = 0.4167$ | 0.0278 |
| 5.5 | 0,0,0,1,0 | 1 | $\frac56 \times 0.32 = 0.2667$ | 0.1778 |

**Feature $x_2$.** Sorted by $x_2$ (P5, P4, P1, P2, P3, P6) the labels read 0, 1, 0, 0, 0, 1. Its best cut is 5.5, with gain 0.1778 (the same
counts as the last row above). So the root split is $x_1 \le 3.5$, gain 0.2222. Entropy agrees: gain 0.4591 bits for $x_1 \le 3.5$, against
0.3167 for the runner-up.

**Second level.** The left child is pure: a leaf predicting 0. The right child $\{P4, P5, P6\}$ (labels 1, 0, 1) is searched
again. On $x_1$ (values 4, 5, 6) no cut isolates P5, which sits in the middle. On $x_2$ the values are 2, 1, 6, so the cut at 1.5 puts P5
alone on the left: both children are pure, gain $0.4444$, the maximum possible. Final tree: *if $x_1 \le 3.5$ predict 0; else if
$x_2 \le 1.5$ predict 0; else predict 1.* It classifies all six points correctly at depth 2.

```text
== worked example by hand ==
root counts (neg, pos) [4, 2]   Gini 0.4444   entropy 0.9183 bits
gini    x1: t=1.5: gain 0.0444  t=2.5: gain 0.1111  t=3.5: gain 0.2222  t=4.5: gain 0.0278  t=5.5: gain 0.1778
gini    x2: t=1.5: gain 0.0444  t=2.5: gain 0.0278  t=3.5: gain 0.0000  t=4.5: gain 0.0278  t=5.5: gain 0.1778
entropy x1: t=1.5: gain 0.1092  t=2.5: gain 0.2516  t=3.5: gain 0.4591  t=4.5: gain 0.0441  t=5.5: gain 0.3167
entropy x2: t=1.5: gain 0.1092  t=2.5: gain 0.0441  t=3.5: gain 0.0000  t=4.5: gain 0.0441  t=5.5: gain 0.3167
root split: x1 <= 3.5   right child [1, 0, 1] splits on x2 <= 1.5
scikit-learn's tree on the same six points:
|--- x1 <= 3.50
|   |--- class: 0
|--- x1 >  3.50
|   |--- x2 <= 1.50
|   |   |--- class: 0
|   |--- x2 >  1.50
|   |   |--- class: 1
```

The gain depends only on the *class counts* on each side, not on the feature values ($x_1$ and $x_2$ tie at 1.5 and 5.5),
which is why the search can work on cumulative counts.

## Key plot

![Two panels against max_depth from 1 to 10. Left: training accuracy rises from 0.93 to 1.0 by depth 7 while validation accuracy stays between 0.94 and 0.95. Right: training log-loss falls to 0 while validation log-loss jumps from 0.22 at depth 2 to 2.14 by depth 5](img/b3_1_depth_overfitting.png)

Deeper trees fit the training rows better and better until every leaf is pure, but validation accuracy does not
improve after depth 2. The right panel shows the real damage. A pure leaf predicts probability 0 or 1, so every
validation tumour that lands in a wrong pure leaf costs $-\log(10^{-15}) = 34.5$ nats of log-loss (after clipping), and the
validation log-loss is 10 times worse than at depth 2.

## From scratch in NumPy

**The data.** Breast cancer again (B2.1), with the same stratified split and malignant as the positive class, but now
all 30 features and *no standardization*: the ranges differ by four orders of magnitude, and the tree does not care.

```text
== data ==
train X (343, 30)  val X (113, 30)  (test 113 rows kept aside)
malignant share: train 0.3732  val 0.3717
feature ranges differ wildly (no scaling needed): 'mean area' 143.5..2501.0   'mean smoothness' 0.0526..0.1634
```

**Impurity and the split search.** Step 1 and Step 3 in code. `Y` is the one-hot label matrix, so a column sum gives class
counts and `np.cumsum` gives the left child's counts at every cut at once:

```python
def impurity(counts, criterion):
    """Gini or entropy (bits) of each row of class counts. counts (m, K) -> (m,)."""
    p = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    if criterion == "gini":
        return 1.0 - (p**2).sum(axis=1)
    return np.where(p > 0, -p * np.log2(p), 0.0).sum(axis=1)        # 0 log 0 = 0

def split_scores(x, Y, criterion, min_leaf):
    """Weighted child impurity for every cut on one feature. x (m,), Y (m, K) -> (m - 1,), (m - 1,)."""
    order = np.argsort(x, kind="stable")
    xs, Ys = x[order], Y[order]
    m = len(xs)
    left = np.cumsum(Ys, axis=0)[:-1]                  # (m - 1, K) class counts left of each cut
    right = Ys.sum(axis=0) - left
    n_left = np.arange(1, m)
    score = (n_left * impurity(left, criterion) + (m - n_left) * impurity(right, criterion)) / m
    valid = (xs[:-1] < xs[1:]) & (n_left >= min_leaf) & (m - n_left >= min_leaf)
    return np.where(valid, score, np.inf), (xs[:-1] + xs[1:]) / 2
```

`best_split` loops over the $d$ features and keeps the lowest score. `grow` is Step 5: a node is a dictionary holding its class
counts and, if it was split, `feature`, `threshold`, `gain`, `left` and `right`. `predict_proba_tree` sends each row down the tree.

```text
== from scratch: grow and predict ==
root: worst concave points <= 0.1424   Gini 0.4678 -> gain 0.3442
|--- worst concave points <= 0.14
|   |--- worst radius <= 17.48
|   |   |--- class: 0   counts [210, 7]
|   |--- worst radius >  17.48
|   |   |--- class: 1   counts [2, 13]
|--- worst concave points >  0.14
|   |--- concavity error <= 0.02
|   |   |--- class: 0   counts [2, 0]
|   |--- concavity error >  0.02
|   |   |--- class: 1   counts [1, 108]
depth-2 tree: train accuracy 0.9708   val accuracy 0.9469
fully grown tree: 14 leaves, depth 7   train accuracy 1.0000   val accuracy 0.9381
```

- The depth-2 tree is a readable rule: *tumours with few concave points and a worst radius of at most 17.48 are benign*
  (210 of the 217 training rows in that leaf are). It scores 0.947 on validation, within a few rows of anything below.
- The leaf under `concavity error <= 0.02` holds 2 rows. The split that created it peels 2 of the 3 benign rows off a node of
  111 to make a pure leaf: a split that fits two training rows, not a pattern. That is overfitting at work even at depth 2. `min_samples_leaf` exists to forbid it.
- The fully grown tree has 14 leaves and gets every training row right, but scores lower on validation.

**Gini versus entropy, and leaf size.**

```text
== from scratch: gini versus entropy, and min_samples_leaf ==
gini    max_depth    3  min_leaf  1:  7 leaves, depth  3   train 0.9767  val 0.9469
gini    max_depth None  min_leaf  1: 14 leaves, depth  7   train 1.0000  val 0.9381
gini    max_depth None  min_leaf  5: 10 leaves, depth  5   train 0.9767  val 0.9381
gini    max_depth None  min_leaf 20:  6 leaves, depth  4   train 0.9592  val 0.9558
entropy max_depth    3  min_leaf  1:  7 leaves, depth  3   train 0.9767  val 0.9558
entropy max_depth None  min_leaf  1: 11 leaves, depth  5   train 1.0000  val 0.9646
entropy max_depth None  min_leaf  5: 10 leaves, depth  5   train 0.9883  val 0.9735
entropy max_depth None  min_leaf 20:  5 leaves, depth  3   train 0.9446  val 0.9292
```

Validation ranges from 0.929 to 0.974, a spread of 5 rows out of 113: too little data to rank these settings (A6.2); use
cross-validation (B4.3) for choices this close. Gini and entropy are rarely far apart. What *is* reliable: every tree with
no depth or leaf-size limit reaches training accuracy 1.0, and that number tells you nothing.

## With scikit-learn

`DecisionTreeClassifier` implements the same CART algorithm. The lab checks it two ways. First, at every internal node of
our tree, a scikit-learn stump fitted on that node's rows finds the same impurity decrease, so our search finds the best split everywhere.
Second, it compares the whole trees.

```text
== with scikit-learn ==
gini    max_depth    1: best split at all  1 nodes OK   leaves mine  2 sklearn  2   same training leaves True   val predictions agree 1.0000
gini    max_depth    2: best split at all  3 nodes OK   leaves mine  4 sklearn  4   same training leaves True   val predictions agree 0.9912
gini    max_depth    3: best split at all  6 nodes OK   leaves mine  7 sklearn  7   same training leaves True   val predictions agree 0.9823
gini    max_depth    4: best split at all  9 nodes OK   leaves mine 10 sklearn 10   same training leaves True   val predictions agree 0.9823
gini    max_depth None: best split at all 13 nodes OK   leaves mine 14 sklearn 14   same training leaves True   val predictions agree 0.9912
entropy max_depth    1: best split at all  1 nodes OK   leaves mine  2 sklearn  2   same training leaves True   val predictions agree 1.0000
entropy max_depth    2: best split at all  3 nodes OK   leaves mine  4 sklearn  4   same training leaves True   val predictions agree 1.0000
entropy max_depth    3: best split at all  6 nodes OK   leaves mine  7 sklearn  7   same training leaves True   val predictions agree 0.9912
entropy max_depth    4: best split at all  8 nodes OK   leaves mine  9 sklearn  9   same training leaves False  val predictions agree 1.0000
entropy max_depth None: best split at all 10 nodes OK   leaves mine 11 sklearn 11   same training leaves True   val predictions agree 1.0000
every split matches the best split DecisionTreeClassifier finds on the same rows: OK
root threshold: sklearn 0.142350  mine 0.142350  (|diff| 4e-09: sklearn stores X as float32)
tie in the left child: worst radius <= 17.48   sends 217 rows left, 15 right   counts [210, 7] / [2, 13]
tie in the left child: worst area   <= 949.85  sends 217 rows left, 15 right   counts [210, 7] / [2, 13]
sklearn's pick there for random_state 0..4: ['worst area', 'worst area', 'worst area', 'worst radius', 'worst radius']
```

Every split is optimal and the leaf counts agree, yet the trees sometimes disagree on a validation row. The reason is
*ties*. Radius and area are almost the same measurement, so in the depth-2 tree's left child `worst radius <= 17.48` and
`worst area <= 949.85` send exactly the same training rows each way. Our code keeps the lower feature index; scikit-learn
tries the features in a random order set by `random_state`, so even its own trees differ between seeds. On the training
rows the choice changes nothing; on a new tumour with a large radius but a small area, it does. The one "same training
leaves False" row is two different splits with exactly equal gain. Fix `random_state`, or you cannot reproduce your model.

**Instability.** Refit the full tree on bootstrap resamples of the training rows (A6.2):

```text
== instability: the same tree on bootstrap resamples ==
resample 0: root worst concave points   <=   0.1436   next split on the left: worst radius           val accuracy 0.9469
resample 1: root worst perimeter        <= 113.6500   next split on the left: worst concave points   val accuracy 0.9027
resample 2: root worst perimeter        <= 106.6500   next split on the left: worst compactness      val accuracy 0.9469
resample 3: root worst area             <= 893.6500   next split on the left: worst concave points   val accuracy 0.9115
resample 4: root worst perimeter        <= 106.6500   next split on the left: worst concavity        val accuracy 0.9292
```

Five resamples of the same 343 rows give three different root features, four different second questions and a 4-point
spread in validation accuracy. That is high variance: the model depends heavily on which rows it saw. B3.2 averages many
such trees so the variance cancels out.

**The depth sweep** behind the key plot:

```text
== key plot: depth sweep ==
max_depth  1: accuracy train 0.9329  val 0.9381   log-loss train 0.2388  val 0.2461
max_depth  2: accuracy train 0.9708  val 0.9469   log-loss train 0.1239  val 0.2167
max_depth  3: accuracy train 0.9767  val 0.9469   log-loss train 0.0632  val 1.5879
max_depth  4: accuracy train 0.9883  val 0.9381   log-loss train 0.0339  val 1.8624
max_depth  5: accuracy train 0.9971  val 0.9381   log-loss train 0.0114  val 2.1410
max_depth  6: accuracy train 0.9971  val 0.9381   log-loss train 0.0040  val 2.1396
max_depth  7: accuracy train 1.0000  val 0.9381   log-loss train 0.0000  val 2.1396
max_depth  8: accuracy train 1.0000  val 0.9381   log-loss train 0.0000  val 2.1396
max_depth  9: accuracy train 1.0000  val 0.9381   log-loss train 0.0000  val 2.1396
max_depth 10: accuracy train 1.0000  val 0.9381   log-loss train 0.0000  val 2.1396
```

From depth 7 the tree stops growing (every leaf is pure), so nothing changes. Compare the best tree, 0.947 validation
accuracy and 0.217 log-loss, with logistic regression on five standardized features in B2.1: 0.982 and 0.086. One tree is
a weak model. Its value is as a building block.

## In practice

- **When to use it:** a single shallow tree (depth 2–4) when you need rules a person can read and check, or as a quick
  look at which features and thresholds matter. For accuracy, use an ensemble of trees (B3.2, B3.3), not one tree.
  Trees are also the building block of isolation forests for anomaly detection (B5.6).
- **Failure modes:**
  - *Overfitting.* Training accuracy near 1.0 with a validation gap, leaves holding 1–5 rows, many leaves. Limit
    `max_depth`, raise `min_samples_leaf`, or prune with `ccp_alpha` (cost-complexity pruning, which removes subtrees whose gain does not
    pay for their extra leaves); tune these by cross-validation (B4.3).
  - *Overconfident probabilities.* Pure leaves predict exactly 0 or 1, and log-loss explodes. Do not feed a single
    tree's `predict_proba` into a decision that needs calibrated probabilities (B4.1).
  - *Instability.* Retraining on new data changes the rules. Watch for it if people read the rules: compare the top splits between
    retrains, and prefer an ensemble plus the explanation tools of B3.4 when stability matters.
  - *Axis-aligned staircase boundaries.* A diagonal boundary (`x1 - x2 > 0`) needs many small steps. If you know such a
    relationship, add the feature (B1.3).
  - *Leakage looks like a great first split.* A root split with near-perfect gain on an ID, a timestamp or a "status
    after the event" column is a leak (B1.4), not a discovery.
  - *Bias towards many-valued features.* A feature with many distinct values offers many thresholds and wins more splits
    by chance, so impurity-based importance favours it (B3.4 shows this with a pure-noise feature).
- **Defaults that work:** `DecisionTreeClassifier(max_depth=3, random_state=0)` for a readable tree;
  `min_samples_leaf` of 5–20 rather than 1; Gini (the default; entropy is rarely different); `sklearn.tree.export_text` or
  `plot_tree` to read the tree. scikit-learn 1.9.1 trees accept missing values (`NaN`) and learn which side to send them
  (VERIFY as of 2026-10).

## Exercises

Save your work to `my-work/B/b3_1_decision_trees.md` (or `.py`) and ask for a review.

1. **By hand:** One feature, eight points: $x = 1, 2, \dots, 8$ with labels $y = 0, 0, 1, 0, 1, 1, 1, 1$. (a) Compute the root Gini and entropy (bits). (b)
   Compute the Gini gain for all seven cuts and find the best threshold. (c) Compute the misclassification rate of
   the children for the cuts at 2.5 and 4.5. What does it say about these two splits, and what does Gini say? (d) Compute the
   entropy gain for the same two cuts. Does entropy agree with Gini? (e) Now treat it as regression with targets
   $2, 4, 3, 10, 9, 14, 13, 15$ for $x = 1, \dots, 8$. Find the best single split by weighted variance (Step 6) and give the two leaf predictions.
2. **Code:** Extend the lab's tree to regression in a new file: impurity = variance of the targets in a node, leaf =
   mean, the same exhaustive split search with `max_depth` and `min_samples_leaf`. Train it on the diabetes data (B1.5, same
   split). (a) At depths 1–4, check every node's split against a `DecisionTreeRegressor` stump on the same rows, as the
   lab does, and check that your training predictions match `DecisionTreeRegressor` exactly. (b) Plot training and validation RMSE
   against depth from 1 to 12 and say which depth you would choose and why. (c) Compare your best tree's validation
   RMSE with B1.5's linear regression.
3. **Think:** A lending team wants to replace their logistic regression with a decision tree "because the regulator needs
   explanations". Their depth-12 tree has 380 leaves, gains 1 point of validation AUC, and after the monthly retrain the
   root split moved from `income` to `debt_to_income`. Write what you would tell them: is this tree an explanation, what
   are the risks, and what would you propose instead? Mention at least one way to measure stability.

## Takeaways

1. A decision tree asks threshold questions on one feature at a time; a leaf predicts its training rows' class shares
   (classification) or mean (regression). It needs no scaling and models interactions without extra features.
2. CART grows the tree greedily: at each node, try every feature and every midpoint between neighbouring values, and keep the split
   with the lowest weighted child impurity. Sorting plus cumulative counts makes that one pass per feature.
3. Gini $1 - \sum p_k^2$ and entropy $-\sum p_k\log_2 p_k$ almost always choose similar splits; the misclassification rate is a bad
   splitting criterion because it is flat and misses useful splits.
4. An unlimited tree memorizes the training data: training accuracy 1.0, pure leaves, 0/1 probabilities and exploding
   validation log-loss. Limit `max_depth`, raise `min_samples_leaf` or prune, and tune these on validation data.
5. Trees have high variance: small changes in the data, or a tie broken differently, change the rules. Averaging many
   trees (B3.2) and boosting shallow ones (B3.3) turn this into the strongest tabular models.

**What builds on this:** [B3.2](b3_2_random_forests.md) (averaging many randomized trees);
[B3.3](b3_3_gradient_boosting.md) (boosting shallow regression trees); [B3.4](b3_4_interpretability_shap.md) (impurity
importance and its bias, SHAP for trees); [B4.5](b4_5_bias_variance.md) (depth as the bias–variance dial);
[B5.6](b5_6_anomaly_detection.md) (isolation forests are random trees).

**References:** L. Breiman, J. Friedman, R. Olshen, C. Stone, *Classification and Regression Trees*, Wadsworth, 1984.
T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009, section 9.2.
scikit-learn User Guide, "Decision Trees" (VERIFY link: https://scikit-learn.org/stable/modules/tree.html).
