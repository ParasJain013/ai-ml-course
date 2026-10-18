# B5.2 Hierarchical clustering and DBSCAN: when clusters aren't round

**Depth** · Part B, week 5 (course week 11) · ~1.5 h · Lab:
[`labs/B/b5_2_hierarchical_dbscan.py`](../../labs/B/b5_2_hierarchical_dbscan.py)

## Why it matters

**k-means** (B5.1) assumes every cluster is a round blob around a centre, that you know how many there are, and that every point belongs somewhere. Real
data breaks all three: delivery addresses follow streets, sessions form long chains of similar behaviour, and a few rows are garbage that belong to no group.
**Hierarchical clustering** gives you the whole family of groupings at once, as a tree you can cut at any level, and some of its variants follow any shape.
**DBSCAN** finds clusters as dense regions of any shape, decides how many there are by itself, and labels the points in sparse regions as noise. Both
appear in interviews as "when would you not use k-means?", and DBSCAN's noise label is a cheap first anomaly detector (B5.6). Their failure modes are
just as specific, and this lesson measures them.

## Prerequisite check

- Can you compute a Euclidean distance matrix between points with NumPy broadcasting? If not, revisit [B2.3](b2_3_knn.md) and
  [A1.2](../A/a1_2_numpy_arrays.md).
- Do you know why features on different scales distort every distance, and how to standardize inside a pipeline? If not, revisit
  [B1.3](b1_3_feature_engineering.md).
- Can you run k-means and say what its inertia (within-cluster sum of squares) measures? If not, revisit [B5.1](b5_1_k_means.md).
- Do you remember how breadth-first search walks a graph with a queue? (No lesson for this one; it is the same BFS you would write for a dependency graph.)

## Intuition

**Hierarchical (agglomerative) clustering** starts with every point as its own cluster and repeatedly merges the two closest clusters until only one is
left. Recording each merge and the distance at which it happened gives a tree, drawn as a **dendrogram**: leaves at the bottom, a horizontal bar where
two clusters merge, the bar's height equal to the merge distance. To get $k$ clusters you cut the tree where it has $k$ branches, or you cut at a height
and take whatever branches cross it. You run it once and choose $k$ afterwards.

The only real decision is what "distance between two *clusters*" means. Points have one distance; groups of points have many. The **linkage** answers it:

- **Single linkage:** the distance between the two *closest* members. Two clusters are close if any pair of their points is close. It follows thin
  curved shapes, and for the same reason it **chains**: a line of stray points between two blobs is a bridge it will happily cross.
- **Complete linkage:** the distance between the two *farthest* members. Two clusters are close only if all their points are. It produces compact clusters
  of similar diameter and ignores bridges.
- **Ward linkage:** merge the pair whose merge increases the total within-cluster sum of squares the least. That is the quantity k-means minimizes
  (B5.1), so Ward behaves like a greedy, nested k-means: round clusters of similar size, a good default for blob-shaped data.

**DBSCAN** (density-based spatial clustering of applications with noise) asks a different question: where is the data *dense*? Draw a circle of radius
$\varepsilon$ around each point. A point whose circle holds at least `min_samples` points (itself included) is a **core point**: it sits inside a dense
region. Core points within $\varepsilon$ of each other belong to the same cluster, and so do chains of them, so a cluster can snake around any shape. A
non-core point inside a core point's circle is a **border point**: on the edge of a cluster, it joins it. Every other point is **noise**, labelled `-1`.
The number of clusters is an output, not an input.

> **Bridge:** DBSCAN is a connected-components query on a graph. Make every core point a node, add an edge between two cores within $\varepsilon$, and a
> cluster is a connected component, found with BFS exactly as you would find every service reachable from one node in a dependency graph. Border points
> are leaves attached to a component; noise points are isolated nodes. **Where it breaks:** your service graph has edges you declared. Here the edges come from
> $\varepsilon$, a distance threshold in the units of your features: rescale a column and the graph, and therefore every cluster, changes.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{X}$ | data matrix, one point per row | `(n, d)` |
| $\mathbf{x}_i$ | point $i$ | `(d,)` |
| $\lVert \mathbf{x}_i - \mathbf{x}_j \rVert$ | Euclidean distance between two points (B2.3) | scalar |
| $A$, $B$ | two clusters (sets of row indices) | sets |
| $n_A$, $n_B$ | number of points in $A$ and $B$ | int |
| $\boldsymbol{\mu}_A$ | mean (centroid) of the points in $A$ | `(d,)` |
| $D(A, B)$ | linkage distance between clusters $A$ and $B$ | scalar |
| $\varepsilon$ | DBSCAN neighbourhood radius (`eps`), in feature units | scalar |
| $m$ | DBSCAN's `min_samples`: points needed in a neighbourhood, the point itself included | int |
| $N_\varepsilon(\mathbf{x}_i)$ | the $\varepsilon$-neighbourhood: all points within $\varepsilon$ of $\mathbf{x}_i$, itself included | set |

Indices start at 1 in the math and 0 in the code.

**Agglomerative clustering.** Start with $n$ clusters $\{1\}, \dots, \{n\}$. Repeat $n - 1$ times: find the pair $(A, B)$ with the smallest $D(A, B)$,
replace both by $A \cup B$, and record the height $D(A, B)$. The three linkages:

$$D_{\text{single}}(A, B) = \min_{i \in A,\, j \in B} \lVert \mathbf{x}_i - \mathbf{x}_j \rVert, \qquad D_{\text{complete}}(A, B) = \max_{i \in A,\, j \in B} \lVert \mathbf{x}_i - \mathbf{x}_j \rVert.$$

For **Ward**, write $\text{SSE}(A) = \sum_{i \in A} \lVert \mathbf{x}_i - \boldsymbol{\mu}_A \rVert^2$ for a cluster's sum of squared distances to its own mean.
Merging $A$ and $B$ can only increase the total, because one mean must now serve both groups. The increase is

$$\Delta(A, B) = \text{SSE}(A \cup B) - \text{SSE}(A) - \text{SSE}(B) = \frac{n_A n_B}{n_A + n_B}\,\lVert \boldsymbol{\mu}_A - \boldsymbol{\mu}_B \rVert^2 .$$

Why this formula: the merged mean is the weighted average $\boldsymbol{\mu} = (n_A\boldsymbol{\mu}_A + n_B\boldsymbol{\mu}_B)/(n_A + n_B)$. Each point of $A$ is
now measured from $\boldsymbol{\mu}$ instead of $\boldsymbol{\mu}_A$, which adds $n_A\lVert\boldsymbol{\mu}_A - \boldsymbol{\mu}\rVert^2$ to $A$'s part of the sum
(write $\mathbf{x}_i - \boldsymbol{\mu} = (\mathbf{x}_i - \boldsymbol{\mu}_A) + (\boldsymbol{\mu}_A - \boldsymbol{\mu})$ and expand the square: $\sum_{i \in A}\lVert\mathbf{x}_i - \boldsymbol{\mu}\rVert^2 = \text{SSE}(A) + 2(\boldsymbol{\mu}_A - \boldsymbol{\mu})^\top\sum_{i \in A}(\mathbf{x}_i - \boldsymbol{\mu}_A) + n_A\lVert\boldsymbol{\mu}_A - \boldsymbol{\mu}\rVert^2$, and the middle sum is zero because the points of $A$ average to $\boldsymbol{\mu}_A$). Likewise for $B$. Since
$\boldsymbol{\mu}_A - \boldsymbol{\mu} = \frac{n_B}{n_A + n_B}(\boldsymbol{\mu}_A - \boldsymbol{\mu}_B)$ and $\boldsymbol{\mu}_B - \boldsymbol{\mu} = \frac{n_A}{n_A + n_B}(\boldsymbol{\mu}_B - \boldsymbol{\mu}_A)$, the two additions are
$n_A\frac{n_B^2}{(n_A + n_B)^2}\lVert\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\rVert^2 + n_B\frac{n_A^2}{(n_A + n_B)^2}\lVert\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\rVert^2 = \frac{n_A n_B (n_B + n_A)}{(n_A + n_B)^2}\lVert\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\rVert^2$, and cancelling one factor $(n_A + n_B)$ gives the formula above. The worked example checks it with numbers.

Ward merges the pair with the smallest $\Delta$. scipy and scikit-learn report the height as $D_{\text{Ward}} = \sqrt{2\,\Delta(A, B)}$, a rescaling chosen so that for
two single points ($n_A = n_B = 1$, $\Delta = \frac12\lVert\mathbf{x}_i - \mathbf{x}_j\rVert^2$) the height is just their distance. It changes the numbers on the
dendrogram's axis, not the order of merges.

**Cost.** The naive algorithm compares every pair of clusters at every step: $O(n^3)$ time. Library implementations are $O(n^2)$ time and memory (single
linkage via a minimum spanning tree; the others with the nearest-neighbour-chain algorithm). Either way, the $(n, n)$ distance matrix limits hierarchical
clustering to tens of thousands of points: 50,000 points need $50{,}000^2 \times 8$ bytes $= 20$ GB as a dense float64 matrix (scipy stores the half above the
diagonal, 10 GB). Beyond that, cluster a sample, or use k-means or DBSCAN.

**DBSCAN.** With $N_\varepsilon(\mathbf{x}_i) = \{\, j : \lVert \mathbf{x}_i - \mathbf{x}_j \rVert \le \varepsilon \,\}$ (which contains $i$ itself):

1. $\mathbf{x}_i$ is a **core point** if $|N_\varepsilon(\mathbf{x}_i)| \ge m$.
2. Two core points are in the same cluster if you can walk from one to the other in steps of length at most $\varepsilon$, landing only on core points.
3. A non-core point within $\varepsilon$ of some core point is a **border point** of that core's cluster. If it is near cores of two clusters, it joins whichever
   reaches it first; this is the only order-dependent part of DBSCAN, and it affects border points only.
4. Every other point is **noise**.

The algorithm is a BFS: take a core point not yet labelled, start a new cluster, push it on a queue; pop a point, label all its unlabelled neighbours with the
cluster, and push those that are themselves core. Border points get the label but are never pushed, so a cluster does not grow *through* its edge. With a
spatial index (a k-d tree or ball tree), finding neighbourhoods costs about $O(n \log n)$ in low dimensions; in high dimensions, or when $\varepsilon$ is large, it
degrades towards $O(n^2)$.

**Choosing $\varepsilon$: the k-distance curve.** Rule 1 says $\mathbf{x}_i$ is core exactly when its $m$-th nearest neighbour (counting itself as the first) is within
$\varepsilon$. So compute that **k-distance** for every point and sort it. Most points sit in dense regions and have small values; the curve bends sharply upward
where the sparse points start. An $\varepsilon$ just above the bend makes most points core and leaves the stragglers as noise. For $m$, a common start is
$m = 2d$ (VERIFY as of 2026-10: a heuristic from the DBSCAN literature, not a theorem), with larger $m$ for noisier data.

**Scoring a clustering against known labels.** On toy data we know the true groups, so the lab scores each method with the **adjusted Rand index** (ARI, derived in [B5.1](b5_1_k_means.md)):
in short, the share of point *pairs* on which two labellings agree (same group in both, or different groups in both), corrected so that random labels score about 0 and
identical partitions score 1, whatever the label numbers are. It is for checking methods on data with known structure; on real data you have no labels, which is
the whole point of clustering.

## Worked example by hand

**Linkage.** Five points on a line: $A = 0$, $B = 2$, $C = 5$, $D = 9$, $E = 10.5$. Pairwise distances that matter: $DE = 1.5$, $AB = 2$, $BC = 3$, $AC = 5$,
$CD = 4$, $CE = 5.5$, $AE = 10.5$.

*Single linkage.* Smallest distance $DE = 1.5$: merge. Next smallest is $AB = 2$: merge. Now three clusters $\{A, B\}$, $\{C\}$, $\{D, E\}$:
$D(\{A,B\}, C) = \min(5, 3) = 3$; $D(C, \{D,E\}) = \min(4, 5.5) = 4$; $D(\{A,B\}, \{D,E\}) = \min(9, 10.5, 7, 8.5) = 7$. Merge $ABC$ at 3. Last,
$D(\{A,B,C\}, \{D,E\}) = $ the closest pair, $CD = 4$. Heights: 1.5, 2, 3, 4.

*Complete linkage.* The first two merges are the same ($DE$ at 1.5, $AB$ at 2: for single points, min and max are the same distance). Then
$D(\{A,B\}, C) = \max(5, 3) = 5$; $D(C, \{D,E\}) = \max(4, 5.5) = 5.5$; so $ABC$ at 5. Last, the farthest pair $AE = 10.5$. Heights: 1.5, 2, 5, 10.5.

*Ward.* Again $DE$ at 1.5 and $AB$ at 2 (for two single points the Ward height equals their distance). Merging $\{A, B\}$ (mean 1) with $C$ (mean 5):
$\Delta = \frac{2 \cdot 1}{3}(5 - 1)^2 = 10.6667$, height $\sqrt{2 \times 10.6667} = 4.6188$. Merging $C$ with $\{D, E\}$ (mean 9.75): $\Delta = \frac{2}{3}(4.75)^2 = 15.04$,
height 5.485. So $ABC$ at 4.6188. The final merge, $\{A,B,C\}$ (mean 2.3333) with $\{D,E\}$: $\Delta = \frac{3 \cdot 2}{5}(7.4167)^2 = 66.01$, height 11.4899.

All three cut into two clusters give $ABC \mid DE$. They agree on *which* groups, but not on *how far apart* they are: single linkage says the two groups are 4 apart
(their closest points), complete says 10.5 (their farthest). On harder data that difference changes the groups too, as the lab shows.

```text
== worked example by hand: linkage ==
single   merges: DE@1.5000, AB@2.0000, ABC@3.0000, ABCDE@4.0000
complete merges: DE@1.5000, AB@2.0000, ABC@5.0000, ABCDE@10.5000
ward     merges: DE@1.5000, AB@2.0000, ABC@4.6188, ABCDE@11.4899
ward AB+C by hand: dSSE = (2*1/3) * (5 - 1)^2 = 10.6667, height sqrt(2 dSSE) = 4.6188
single   cut into 2 clusters: ABC | DE
complete cut into 2 clusters: ABC | DE
ward     cut into 2 clusters: ABC | DE
```

**DBSCAN.** Eight points on a line, $\varepsilon = 1.5$, $m = 3$: $0, 1, 2, 3.4, 6, 7, 8, 12$. Count each neighbourhood (itself included):

- $0$: $\{0, 1\}$, 2 points, not core. $1$: $\{0, 1, 2\}$, 3, **core**. $2$: $\{1, 2, 3.4\}$ ($3.4 - 2 = 1.4 \le 1.5$), 3, **core**. $3.4$: $\{2, 3.4\}$, 2, not core.
- $6$: $\{6, 7\}$, not core. $7$: $\{6, 7, 8\}$, **core**. $8$: $\{7, 8\}$, not core. $12$: $\{12\}$, not core.

Start at the first core point, 1: cluster 0 takes 0, 1, 2; 2 is core, so its neighbour 3.4 joins too. 0 and 3.4 are border points (in a core's
neighbourhood, but not core themselves). The next unlabelled core, 7, starts cluster 1 and takes 6 and 8 as border points. 12 is near no core: noise.

```text
== worked example by hand: DBSCAN ==
x:                [0.0, 1.0, 2.0, 3.4, 6.0, 7.0, 8.0, 12.0]
points within 1.5: [2, 3, 3, 2, 2, 3, 2, 1]
kind:             ['border', 'core', 'core', 'border', 'border', 'core', 'border', 'noise']
labels:           [0, 0, 0, 0, 1, 1, 1, -1]
```

Notice that 6 and 8, each with only one other point nearby, are in a cluster, while 12 is not: membership depends on being near a *core*, not on your own count.

## Key plot

![Two rows of scatter plots, two moons on top and concentric circles below, clustered by k-means, Ward, single linkage and DBSCAN: k-means and Ward split each shape with a straight-ish cut (ARI 0.25 and 0.56 on moons, about 0 on circles), while single linkage and DBSCAN recover both shapes (ARI 1.00)](img/b5_2_shapes_four_methods.png)

The two methods that think in terms of centres and squared distances (k-means and Ward) cut each shape into two halves, because two half-moons are rounder
than one moon. The two methods that only ask "is there a path of close points?" (single linkage and DBSCAN) follow the shapes. DBSCAN marks one stray point
on the moons as noise (grey cross).

## From scratch in NumPy

**Linkage.** The naive algorithm keeps a dictionary from cluster id to its rows, and at every step tries every pair:

```python
def cluster_distance(A, B, method):
    if method == "ward":
        gap = A.mean(axis=0) - B.mean(axis=0)
        delta_sse = len(A) * len(B) / (len(A) + len(B)) * float(gap @ gap)
        return np.sqrt(2 * delta_sse)
    D = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=-1))   # (n_a, n_b)
    return D.min() if method == "single" else D.max()
```

`linkage_scratch` writes scipy's format: one row per merge, `[cluster a, cluster b, height, size]`, where original points are clusters $0, \dots, n-1$ and merge
$i$ creates cluster $n + i$. On 40 random points (random floats, so no ties and a unique merge order) the whole table matches scipy, row for row:

```text
== from scratch: linkage matches scipy ==
single   40 points: merge table matches scipy.cluster.hierarchy.linkage: OK (last merge height 1.1951)
complete 40 points: merge table matches scipy.cluster.hierarchy.linkage: OK (last merge height 4.4025)
ward     40 points: merge table matches scipy.cluster.hierarchy.linkage: OK (last merge height 7.0518)
```

The last single-linkage merge is at 1.2 while the complete-linkage one is at 4.4: single linkage measures the gap between groups, complete linkage their span.

**DBSCAN.** The whole algorithm is neighbourhoods, a core mask and a BFS:

```python
D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))   # (n, n)
neighbours = [np.flatnonzero(row <= eps) for row in D]              # includes the point itself
is_core = np.array([len(nb) >= min_samples for nb in neighbours])
labels = np.full(len(X), -1)
cluster = 0
for start in range(len(X)):                    # seed clusters in index order, as scikit-learn does
    if not is_core[start] or labels[start] != -1:
        continue
    labels[start] = cluster
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for q in neighbours[p]:
            if labels[q] == -1:
                labels[q] = cluster
                if is_core[q]:                 # only cores pass the cluster on
                    queue.append(q)
    cluster += 1
```

Because both implementations seed clusters in index order, even the cluster numbers and the border-point assignments match scikit-learn exactly:

```text
== from scratch: DBSCAN matches scikit-learn ==
two moons eps=0.15 min_samples=5: 2 clusters, 497 core, 2 border, 1 noise; labels match sklearn.cluster.DBSCAN: OK
circles   eps=0.15 min_samples=5: 2 clusters, 497 core, 3 border, 0 noise; labels match sklearn.cluster.DBSCAN: OK
```

**Choosing $\varepsilon$.** The 5-distance of every point (distance to its 5th nearest, itself included). With $\varepsilon = 0.15$, everything below 0.15 is core; on
both datasets that is more than 99% of the points:

```text
== choosing eps: the k-distance curve ==
two moons 5-distance quantiles 50/90/95/99%: [0.0579, 0.0918, 0.1096, 0.136]  max 0.1680
circles   5-distance quantiles 50/90/95/99%: [0.0641, 0.0985, 0.1143, 0.1365]  max 0.1648
saved outputs/b5_2/k_distance.png
```

## With scikit-learn

`AgglomerativeClustering(n_clusters=k, linkage=...)` cuts the tree for you; `scipy.cluster.hierarchy.linkage` gives the full merge table, `dendrogram` draws it and
`fcluster(Z, t=k, criterion="maxclust")` cuts it. `DBSCAN(eps=..., min_samples=...)` exposes `labels_` and `core_sample_indices_`. All four methods on both shapes:

```text
== with scikit-learn: who finds the shapes ==
method    moons ARI circles ARI
k-means      0.2525     -0.0020
Ward         0.5587      0.0017
complete     0.4451     -0.0016
single       1.0000      1.0000
DBSCAN       0.9960      1.0000   (noise points count as their own group)
single linkage: sklearn AgglomerativeClustering matches scipy linkage + fcluster: OK
```

On the circles, k-means, Ward and complete linkage score about 0: no better than random labels. On clean, well-separated shapes, single linkage is perfect. DBSCAN
is perfect on the circles and loses 0.004 on the moons only because it calls one stray point noise, which is arguably correct.

**The failure modes.** Clean toy shapes flatter single linkage and DBSCAN. Three tests where they fail:

```text
== failure modes: chaining and varying density ==
blobs + bridge, single  : cluster sizes [311, 1], ARI on the blob points 0.0000
blobs + bridge, complete: cluster sizes [159, 153], ARI on the blob points 1.0000
blobs + bridge, ward    : cluster sizes [159, 153], ARI on the blob points 1.0000
three densities (std 0.3, 1, 2), DBSCAN eps=0.2: 8 clusters, noise share 0.560, ARI 0.5130
three densities (std 0.3, 1, 2), DBSCAN eps=0.5: 10 clusters, noise share 0.170, ARI 0.7494
three densities (std 0.3, 1, 2), DBSCAN eps=1.0: 2 clusters, noise share 0.018, ARI 0.5648
three densities, Ward with 3 clusters: ARI 0.9416
two moons with feature 2 x10, same eps: 24 clusters, noise share 0.662, ARI 0.0103
```

- **Chaining.** Two clean blobs of 150 points, plus 12 points in a line between them. Single linkage walks across the bridge, puts both blobs and the bridge in one
  cluster of 311, and makes the "second cluster" a single stray point. Complete and Ward ignore the bridge and recover the blobs perfectly.
- **Varying density.** Three blobs with standard deviations 0.3, 1 and 2. A small $\varepsilon$ finds the tight blob and shatters the loose one into noise and fragments; a large
  $\varepsilon$ lets the loose blob's outskirts reach the medium blob, and the two merge. No single $\varepsilon$ fits all three, while Ward, whose assumptions (round blobs) hold here, scores 0.94.
- **Units.** The same moons with the second feature multiplied by 10 (centimetres instead of decimetres, say). With the same $\varepsilon$, two-thirds of the points become
  noise and the rest shatter into 24 clusters. $\varepsilon$ is a distance in feature units; standardize first (B1.3), and recompute the k-distance curve after any
  change to the features.

## In practice

- **When to use it:**
  - *Hierarchical, Ward:* a few thousand rows, blob-like clusters, and you want to look at several values of $k$ from one run, or show stakeholders a
    dendrogram of how segments nest ("premium splits into frequent and occasional").
  - *Hierarchical, single:* shapes connected by thin, clean paths, with no noise. Rare in practice for that reason.
  - *Hierarchical, complete or average:* when you have only a distance matrix (edit distances between strings, a custom similarity between products); linkage needs
    distances, not coordinates. Ward needs coordinates (it uses means).
  - *DBSCAN:* low-dimensional data with arbitrary shapes and real noise, where you don't know $k$: geographic hotspots, sensor readings, a first pass at outliers.
- **Failure modes:**
  - *Single-linkage chaining:* one giant cluster plus singletons. Notice it from the cluster sizes; always print them.
  - *One $\varepsilon$ for clusters of different density:* the sparse cluster becomes noise, or the dense ones merge. Notice it from a noise share that jumps when you nudge
    $\varepsilon$, or from noise concentrated in one region. Try HDBSCAN.
  - *Unscaled features:* $\varepsilon$ and every linkage are dominated by the largest-unit column (B1.3).
  - *High dimensions:* distances concentrate (B2.3), so the k-distance curve has no bend and DBSCAN labels everything as one cluster or as noise. Reduce dimensions
    first (PCA, B5.4) or use another method.
  - *Memory:* hierarchical clustering on 100,000 rows tries to allocate tens of GB. Sample, or use k-means.
  - *Reading noise as anomalies uncritically:* DBSCAN noise means "not in a dense region at this $\varepsilon$", which is a property of the parameters as much as of the point.
- **Defaults that work:** standardize first; for DBSCAN start with `min_samples` about $2d$ (at least 4–5) and read $\varepsilon$ from the k-distance curve; for
  hierarchical start with Ward, then compare single and complete if the clusters look wrong. When densities differ, try `sklearn.cluster.HDBSCAN`
  (`min_cluster_size` is its main knob; it builds a hierarchy of DBSCAN clusterings over all $\varepsilon$ and keeps the most stable clusters). Library versions as pinned:
  scikit-learn 1.9.1, scipy 1.18.1 (VERIFY as of 2026-10).

## Exercises

Save your work to `my-work/B/b5_2_hierarchical_dbscan.md` (or `.py`) and ask for a review.

1. **By hand:** Six points on a line: $A = 1$, $B = 2$, $C = 4.5$, $D = 9$, $E = 10.2$, $F = 16$. (a) Run single and complete linkage to the end: list every merge and its
   height. (b) Cut each tree into 3 clusters and into 2 clusters. Where do the two linkages disagree, and why? (c) Compute the Ward height for merging $\{A, B\}$ with
   $C$. (d) Run DBSCAN with $\varepsilon = 1.5$, $m = 2$, and again with $\varepsilon = 2.6$, $m = 3$: label each point core, border or noise, and give the cluster labels.
   Explain why $D$ and $E$, the second-closest pair of points, are noise in the second run.
2. **Code:** Using the lab's three-density blobs (`make_blobs(n_samples=[200, 200, 200], centers=[[0, 0], [6, 0], [3, 5]], cluster_std=[0.3, 1.0, 2.0],
   random_state=0)`): (a) sweep DBSCAN's $\varepsilon$ from 0.1 to 2.0 with `min_samples=5`, report the number of clusters, the noise share and the ARI, and the noise share
   *per true blob* at the best $\varepsilon$; (b) run `HDBSCAN` with `min_cluster_size` 10, 20 and 50 and compare; (c) stretch the lab's two moons by 10 in the second
   feature, choose $\varepsilon$ as the 99th percentile of the 5-distance, and show what standardizing does. Is "the best $\varepsilon$" in (a) something you could find in practice?
3. **Think:** A ride-hailing company has 400,000 pickup locations (latitude, longitude) from last month and wants "hotspots" to position idle drivers. A colleague
   proposes Ward hierarchical clustering with $k = 50$; another proposes DBSCAN. Compare them for this data: what each would produce, what breaks (memory, shapes,
   noise, distance units on a sphere), how you would set the parameters, and what you would show the operations team.

## Takeaways

1. Agglomerative clustering merges the closest pair of clusters $n - 1$ times; the linkage (single = closest members, complete = farthest, Ward = smallest rise in
   within-cluster sum of squares) decides what "closest" means, and the dendrogram lets you choose $k$ afterwards.
2. Single linkage follows any connected shape and therefore chains across bridges; complete and Ward make compact, k-means-like clusters and ignore bridges but cannot follow curves.
3. DBSCAN: core points have at least $m$ neighbours within $\varepsilon$; clusters are connected groups of cores plus their border points; everything else is noise, and the number of clusters is an output.
4. $\varepsilon$ is a distance in feature units: standardize first, read it off the k-distance curve, and expect one $\varepsilon$ to fail when clusters differ in density (HDBSCAN is the usual fix).
5. Hierarchical clustering needs $O(n^2)$ memory, so it stops at tens of thousands of rows; DBSCAN scales further in low dimensions but loses its meaning in high ones.

**What builds on this:** [B5.3](b5_3_gmm_em.md) (soft clusters with a shape per cluster); [B5.5](b5_5_tsne_umap.md) (clusters you see in a 2-D map are not clusters in the
data); [B5.6](b5_6_anomaly_detection.md) (density-based anomaly scores, DBSCAN noise as a first detector); P3 (segmentation and anomalies on one dataset).

**References:** M. Ester, H.-P. Kriegel, J. Sander, X. Xu, "A density-based algorithm for discovering clusters in large spatial databases with noise", *KDD*, 1996.
J. H. Ward Jr., "Hierarchical grouping to optimize an objective function", *Journal of the American Statistical Association* 58, 1963. R. J. G. B. Campello,
D. Moulavi, J. Sander, "Density-based clustering based on hierarchical density estimates", *PAKDD*, 2013 (HDBSCAN). scikit-learn User Guide, "Clustering"
(VERIFY link: https://scikit-learn.org/stable/modules/clustering.html).
