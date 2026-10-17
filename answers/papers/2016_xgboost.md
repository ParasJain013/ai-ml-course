# Answer notes: XGBoost: A Scalable Tree Boosting System (Chen, Guestrin, 2016)

Guide: [`papers/2016_xgboost.md`](../../papers/2016_xgboost.md). The answers follow the paper's section numbers (arXiv
1603.02754); notation as in B3.3. When reviewing a learner's answers, check the reasoning, not the wording.

## 1. The regularized objective (Section 2.1)

$\Omega(f) = \gamma T + \frac12\lambda\lVert\mathbf{w}\rVert^2$, where $T$ is the number of leaves and $\mathbf{w}$ the vector of leaf values. $\gamma$ charges a fixed cost per leaf, so it penalizes tree *size*: a
split must improve the loss by more than $\gamma$ to be worth making. $\lambda$ penalizes large leaf *values*, which shrinks the predictions of
leaves supported by few rows (or, for log-loss, by little curvature) towards zero. The full objective is the sum of
the training losses plus $\Omega$ for every tree. With $\Omega = 0$ it is ordinary gradient tree boosting, as the section notes.
Plain gradient boosting (Friedman, 2001) was already regularized, but outside the objective: shrinkage (the learning rate),
limits on tree depth or on rows per leaf, row subsampling (stochastic gradient boosting) and the number of trees. The paper's point is
to put a complexity penalty *into* the quantity that each tree optimizes, so leaf values and split choices account for it directly.

**What a strong answer adds:** $\lambda$ is the same idea as ridge regression (B2.2) applied to leaf values, and $\gamma$ acts like pruning.

## 2. Leaf weights and the structure score (Section 2.2)

Expand the loss of row $i$ around its current prediction $F_i$ to second order: $\ell(y_i, F_i + f(\mathbf{x}_i)) \approx \ell(y_i, F_i) + g_if(\mathbf{x}_i) + \frac12 h_if(\mathbf{x}_i)^2$. The first term does not depend on the new tree, so
drop it. A tree assigns the value $w_j$ to every row in leaf $j$, so grouping the rows by leaf:

$$\tilde{\mathcal{L}} = \sum_{j=1}^{T}\Big[G_jw_j + \tfrac12(H_j + \lambda)w_j^2\Big] + \gamma T.$$

For a fixed structure each leaf is an independent parabola in $w_j$ (it opens upward when $H_j + \lambda > 0$). The derivative $G_j + (H_j + \lambda)w_j = 0$ gives $w_j^* = -G_j/(H_j + \lambda)$.
Substituting: $G_j\left(-\frac{G_j}{H_j + \lambda}\right) + \frac12(H_j + \lambda)\frac{G_j^2}{(H_j + \lambda)^2} = -\frac{G_j^2}{H_j + \lambda} + \frac12\frac{G_j^2}{H_j + \lambda} = -\frac12\frac{G_j^2}{H_j + \lambda}$. Summing over leaves and adding $\gamma T$ gives the structure score; the section uses it as
the quality measure of a tree, like impurity in B3.1.

**Why second order.** Fitting trees to $-g$ alone (B3.3 Steps 2–3) finds good split *directions*, but it says nothing about how far to step in
each leaf: for squared error, $h = 1$ and the gradient step is already optimal, but for log-loss the right step depends on the
curvature, which is why Friedman's classifier needs a separate line search (the Newton leaf values of B3.3 Step 3). The second-order form gives leaf values and
split scores from the same two sums $G$ and $H$, for any twice-differentiable loss, so a single implementation covers regression,
classification, ranking and custom losses. A custom loss only needs to supply $g_i$ and $h_i$.

## 3. Gradients and Hessians for log-loss (Section 2.2, applied)

With $p_i = \sigma(F_i)$: $g_i = p_i - y_i$ and $h_i = p_i(1 - p_i)$ (B2.1 Steps 1 and 3). $h_i$ is largest (0.25) for rows the model is unsure about ($p \approx 0.5$) and
near zero for rows it is confident about. In $w^* = -G/(H + \lambda)$, $H$ is a weighted count of the leaf's rows, so uncertain rows count
almost fully and confidently predicted rows hardly count. A leaf full of rows with $p$ near 0 or 1 has a tiny $H$: without $\lambda$ its Newton step
$-G/H$ could be huge. $\lambda$ caps that, and so does `min_child_weight`, a lower bound on $H$ per leaf.

**What a strong answer adds:** for squared error, $h_i = 1$, so $H_j$ is simply the number of rows in the leaf and `min_child_weight` becomes a
minimum leaf size.

## 4. Shrinkage and column subsampling (Section 2.3)

Shrinkage multiplies each new tree's weights by a factor $\eta$ (the learning rate). The paper's reason is that it reduces the
influence of each individual tree and leaves room for future trees to improve the model (B3.3 Step 4). Column subsampling, borrowed from random forests, gives each tree (or
split) a random subset of the features. The paper gives two reasons: user feedback reports that it prevents overfitting
even more than row subsampling does, and it speeds up the parallel algorithm, since fewer columns are scanned. A strong answer links
this to B3.2: fewer shared features make the trees less correlated.

## 5. Approximate split finding (Sections 3.2–3.3)

The exact algorithm (Section 3.1) scans every distinct value of every feature, which needs the data sorted and in
memory. The approximate algorithm proposes a set of candidate thresholds per feature from percentiles of its
distribution, buckets the rows between candidates, sums $g$ and $h$ per bucket, and evaluates splits only at bucket boundaries.
*Global* proposals are made once per tree and reused for every node; *local* proposals are recomputed for each node after
it is split. The section's experiment shows that the global variant needs many more candidates (a finer quantile grid)
to match the exact algorithm's accuracy, while the local variant reaches it with a coarser grid, because local candidates adapt to the
rows that actually reach each node. Global is cheaper per candidate; local needs fewer candidates. Today's
histogram methods are the global variant with a fixed bin budget.
*Why weight by $h_i$:* the second-order objective of Section 2.2 can be rewritten as a weighted squared error,
$\sum_i \frac12 h_i\big(f(\mathbf{x}_i) - (-g_i/h_i)\big)^2$ plus a constant: each row behaves as if it had target $-g_i/h_i$ and weight $h_i$. Rows with larger $h_i$ matter more to the
objective, so candidate thresholds should split the *Hessian-weighted* mass evenly, not the row count. Section 3.3 gives a
quantile sketch that works with weights and comes with error guarantees (proved in the appendix); existing quantile sketches handled
only unweighted data.

## 6. Sparsity-aware split finding (Section 3.4)

For each candidate split, the algorithm accumulates $G$ and $H$ over the rows whose value is *present* (sorted by the feature), and tries
sending all rows with a missing value to the left child and then to the right child. The total $G$ and $H$ of the missing rows is the node total minus the sums over present rows. It keeps the
direction with the higher gain as that node's **default direction**; at prediction time, a missing value follows it.
Because the missing rows only ever enter as a single pair of totals, each scan visits only the non-missing entries, so the cost is linear in the number of non-missing
entries. That matters for one-hot encoded or other very sparse data, where most entries are absent or zero, and the section reports that the
sparsity-aware version is far faster on such data than a naive implementation.
*Compared with median imputation:* imputing puts every missing row at the median, so they always go wherever the
median goes, and the model cannot tell "missing" from "typical". The learned default direction can send missing rows
wherever they fit best at each split, which captures cases where missingness itself is informative (a bedroom count that is missing because
the survey failed in certain areas). Imputation also needs fitting on training rows only (B1.4) and is one more step to keep identical in serving.

## 7. System design (Section 4)

- **Column blocks (4.1):** the expensive part of split finding is sorting. The data is stored once in compressed column
  blocks, each column pre-sorted, and reused across all iterations, so split finding becomes linear scans that parallelize over
  features.
- **Cache-aware access (4.2):** scanning a sorted column fetches $g_i$ and $h_i$ in row order, which is random access in memory and causes cache
  misses; the exact algorithm prefetches gradient statistics into a per-thread buffer, and the approximate one chooses a block size that
  balances parallelism against cache misses.
- **Out-of-core blocks (4.3):** for data larger than memory, blocks live on disk; block compression (a CPU cost traded for disk
  bandwidth) and block sharding across several disks keep the disk from becoming the bottleneck while a separate thread pre-fetches blocks.

A practitioner rarely needs these details, but should know that they are why XGBoost (and its successors) scale to data
that does not fit in RAM, and why the `hist` method is used for large data.

## 8. The evidence (Sections 1 and 6)

The introduction argues from adoption: it counts how often XGBoost appeared in winning solutions of 2015 machine-learning
competitions (Kaggle and the KDD Cup). That shows popularity and usefulness in competitions, not a controlled comparison. Section 6
measures mostly **speed and scalability**: time per tree against other implementations on a single machine, the effect of the
sparsity-aware algorithm, out-of-core training as data grows, and distributed training against other distributed systems, on large datasets
(insurance claims, a particle-physics dataset, learning to rank, click logs). Accuracy (for example test AUC in the
classification comparison) is reported alongside, but the emphasis throughout is speed.
*What is missing for an accuracy claim:* the same hyperparameter-tuning budget for every method; several datasets of
different kinds; repeated runs or cross-validation with variability (A6.2, B4.3); and an ablation of the individual ingredients
($\lambda$, $\gamma$, second-order leaf values, column subsampling). The paper is a systems paper and argues convincingly that XGBoost is fast and scalable; it
does not, and does not try to, prove it is more accurate than a carefully tuned alternative.
