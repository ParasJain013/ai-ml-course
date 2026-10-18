# B5.5 t-SNE and UMAP: what they show and how they mislead

**Core** · Part B, week 5 (course week 11) · ~1.5 h · Lab:
[`labs/B/b5_5_tsne_umap.py`](../../labs/B/b5_5_tsne_umap.py)

## Why it matters

Sooner or later someone will show you a 2-D scatter of embeddings or customers with neat coloured islands and
conclude "there are seven segments", "this group is twice as big", "these two are close, so they are similar".
t-SNE and UMAP make those pictures; they are the standard way to look at high-dimensional data, including the
embeddings of Parts E, F and I, and the most over-read plots in ML. They keep *which points are neighbours* and
throw away almost everything else: cluster sizes, distances between clusters, even whether the clusters exist.
This lesson shows what they compute, measures what they keep, and builds three datasets on which they lie.

## Prerequisite check

- Can you compute an entropy in nats and say why $e^{H}$ of a uniform distribution over $m$ outcomes is $m$? If not,
  revisit [A6.4](../A/a6_4_entropy_cross_entropy_kl.md).
- Can you say what the KL divergence $D_{\mathrm{KL}}(p \,\|\, q)$ measures, and that it is not symmetric? If not,
  revisit [A6.4](../A/a6_4_entropy_cross_entropy_kl.md).
- Can you find the $k$ nearest neighbours of a point and explain why distances lose contrast in high dimensions? If
  not, revisit [B2.3](b2_3_knn.md).
- Do you know what a projection onto the top two principal directions keeps? If not, revisit [A3.2](../A/a3_2_svd.md)
  and [B5.4](b5_4_pca.md).

## Intuition

**PCA** ([B5.4](b5_4_pca.md)) is a *linear* map: it rotates the data and drops all but two axes. Long distances
survive, so big gaps stay big, but two clusters that are separated along an axis it drops land on top of each
other. On the 64-pixel digits, PCA's 2-D picture is a smear in which most digits overlap.

**t-SNE** (t-distributed stochastic neighbour embedding) and **UMAP** (uniform manifold approximation and
projection) are *non-linear* and ask a different question: for every point, who are its near neighbours? They then
place points in 2-D so that each point keeps the same neighbours. Nothing else is asked for. Distances between
points that are not neighbours do not enter the objective (t-SNE) or enter only as a gentle push apart (UMAP).

That one design choice explains both why the pictures look so good and how they mislead:

- **Good:** local structure is kept very well. In the lab, about 58% of each digit's 10 nearest neighbours in 64-D
  are also among its 10 nearest in the t-SNE picture, against 12% for PCA. Clusters that are distinct in 64-D come
  out as distinct islands.
- **Bad:** each point's neighbourhood is normalized to the same effective size, so a tight cluster and a loose one
  come out the same size. Gaps between islands are set by a tug of war between attraction and repulsion, not by the
  real distances. And a method that is required to make neighbourhoods visible will make some where none exist.

> **Bridge:** a t-SNE map is like a schematic metro map drawn from the service graph: it keeps which stations are
> adjacent and is far more readable than a to-scale map. **Where it breaks:** everyone knows a metro map is
> schematic. A t-SNE plot looks like a to-scale scatter plot (it even has numbered axes), so readers measure sizes
> and gaps on it that the method never tried to preserve.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{X}$ | input data, one example per row | `(n, d)` |
| $\mathbf{x}_i$ | example $i$ | `(d,)` |
| $\mathbf{y}_i$ | the position of example $i$ in the picture (its 2-D *embedding*) | `(2,)` |
| $\sigma_i$ | width of the Gaussian around $\mathbf{x}_i$, chosen per point | scalar |
| $p_{j\mid i}$ | probability that $\mathbf{x}_i$ would pick $\mathbf{x}_j$ as its neighbour | scalar; row $i$ sums to 1 |
| $H_i$ | entropy of the row $p_{\cdot\mid i}$, in nats | scalar |
| $\mathrm{Perp}$ | **perplexity**, the target $e^{H_i}$ for every point | scalar |
| $p_{ij}$, $\mathbf{P}$ | symmetric input affinities; $\mathbf{P}$ sums to 1 | `(n, n)` |
| $q_{ij}$, $\mathbf{Q}$ | affinities between the points in the picture; $\mathbf{Q}$ sums to 1 | `(n, n)` |
| $C$ | t-SNE's cost, $D_{\mathrm{KL}}(\mathbf{P} \,\|\, \mathbf{Q})$ | scalar |
| $k$ | number of neighbours: UMAP's `n_neighbors`, and the neighbourhood size in the quality scores | int |
| $d_{ij}$, $\rho_i$, $w_{ij}$ | UMAP only: distance from $\mathbf{x}_i$ to $\mathbf{x}_j$, distance to $i$'s nearest neighbour, edge weight | scalars |

In the math, points are numbered from 1; in the code, rows from 0.

**Step 1: neighbour probabilities in the input.** Point $i$ picks a neighbour at random, preferring close ones with
a Gaussian weight:

$$p_{j\mid i} = \frac{\exp\!\left(-\lVert\mathbf{x}_i - \mathbf{x}_j\rVert^2 / 2\sigma_i^2\right)}{\sum_{l \ne i}\exp\!\left(-\lVert\mathbf{x}_i - \mathbf{x}_l\rVert^2 / 2\sigma_i^2\right)}, \qquad p_{i\mid i} = 0.$$

This is a softmax over negative squared distances. Small $\sigma_i$: almost all the probability goes to the
nearest point. Large $\sigma_i$: it spreads over many points.

**Step 2: perplexity sets $\sigma_i$.** How spread out row $i$ is, is measured by its entropy
$H_i = -\sum_j p_{j\mid i}\log p_{j\mid i}$. If the row were uniform over $m$ points, $H_i = \log m$ and
$e^{H_i} = m$ (A6.4). So $e^{H_i}$ is the *effective number of neighbours*, and t-SNE's **perplexity** is the value
it should take for every point. (The t-SNE paper writes $2^{H}$ with $H$ in bits: the same number.) Because
$e^{H_i}$ grows steadily with $\sigma_i$, a binary search finds the $\sigma_i$ that hits the target, separately for
each point. In a dense region the neighbours are close, so a small $\sigma_i$ already covers 30 of them; in a sparse
region $\sigma_i$ must be large. **This per-point normalization is the first lie:** every point is told it has 30
neighbours, whatever the density, so dense and sparse clusters are drawn the same size.

**Step 3: symmetrize.** $p_{j\mid i}$ and $p_{i\mid j}$ differ (different $\sigma$s), so t-SNE averages them and
rescales so that the whole matrix sums to 1:

$$p_{ij} = \frac{p_{j\mid i} + p_{i\mid j}}{2n}.$$

Each row of conditionals sums to 1, so the $n$ rows sum to $n$, the two transposes to $2n$, hence the $2n$.

**Step 4: affinities in the picture use a heavy tail.** For positions $\mathbf{y}_i$ in 2-D:

$$q_{ij} = \frac{\left(1 + \lVert\mathbf{y}_i - \mathbf{y}_j\rVert^2\right)^{-1}}{\sum_{k \ne l}\left(1 + \lVert\mathbf{y}_k - \mathbf{y}_l\rVert^2\right)^{-1}}.$$

Here $d$ is the distance between two points in the picture. The kernel $1/(1 + d^2)$ has the shape of the Student-t
distribution with one degree of freedom (the "t" in t-SNE). It decays like
$1/d^2$, far more slowly than a Gaussian's $e^{-d^2/2}$: at distance 3 the Gaussian is 0.011 and the t-kernel 0.1.
Why: in 64-D a point can have many points at the same moderate distance, and 2-D has no room for them all (the
**crowding problem**). The heavy tail lets moderately similar points be placed *far* away at little cost, which opens
clear gaps between groups. **That is the second lie:** the size of a gap between islands reflects this
convenience, not how far apart the groups are.

**Step 5: the cost.** t-SNE moves the $\mathbf{y}_i$ by gradient descent (A4.4) to minimize

$$C = D_{\mathrm{KL}}(\mathbf{P} \,\|\, \mathbf{Q}) = \sum_{i \ne j} p_{ij}\log\frac{p_{ij}}{q_{ij}}.$$

Look at one term. A pair that are neighbours in the input ($p_{ij}$ large) but far apart in the picture ($q_{ij}$
small) has a large $\log(p_{ij}/q_{ij})$ multiplied by a large $p_{ij}$: expensive. A pair that are far apart in the
input ($p_{ij} \approx 0$) but drawn close together ($q_{ij}$ large) contributes $p_{ij}\log(\cdot) \approx 0$:
almost free. So t-SNE fights hard to keep neighbours together and barely cares where non-neighbours go. Only the
shared denominator of $\mathbf{Q}$ pushes all points apart a little. (The gradient works like a spring between every
pair, pulling when $p_{ij} > q_{ij}$ and pushing when $p_{ij} < q_{ij}$; you do not need its formula to use t-SNE.)

**Step 6: UMAP, in one paragraph.** UMAP builds the same kind of neighbour graph from the $k$ = `n_neighbors` nearest
points only. Edge weights are $w_{ij} = \exp(-(d_{ij} - \rho_i)/\sigma_i)$, where $\rho_i$ is the distance to $i$'s
nearest neighbour (so the nearest neighbour gets weight 1) and UMAP's own $\sigma_i$ is chosen, like t-SNE's, so that
each point's weights sum to the same value ($\log_2 k$): the same per-point normalization. The two directed weights are
combined as $w_{ij} + w_{ji} - w_{ij}w_{ji}$, the probability that at least one of two independent edges exists. In
the picture, the affinity is $1/(1 + \alpha\lVert\mathbf{y}_i - \mathbf{y}_j\rVert^{2\beta})$, a t-like kernel whose
constants $\alpha, \beta$ (in this paragraph only) are set from `min_dist`, how tightly points may pack. The cost is a cross-entropy that rewards
keeping edges short *and* non-edges long, optimized by stochastic gradient descent that samples a few non-edges per
step. UMAP is faster on large data, but it has the same local-first character.

## Worked example by hand

Four points on a line, $x = 0, 1, 2, 4$ (so $d = 1$). Take point $i$ at $x = 0$ with $\sigma = 1$. Squared distances
to the others: $1, 4, 16$. Weights $e^{-d^2/2}$: $e^{-0.5} = 0.6065$, $e^{-2} = 0.1353$, $e^{-8} = 0.0003$; their sum
is $0.7422$. Divide by it:

$$p_{\cdot\mid i} = (0.8172,\ 0.1823,\ 0.0005).$$

Entropy: $H = -(0.8172\log 0.8172 + 0.1823\log 0.1823 + 0.0005\log 0.0005) = 0.1650 + 0.3103 + 0.0035 = 0.4788$ nats,
so the perplexity is $e^{0.4788} = 1.61$: the point effectively has 1.6 neighbours. With $\sigma = 2$ the weights
spread out and the perplexity rises to 2.48; with $\sigma = 0.5$ it falls to 1.02 (one neighbour). A target
perplexity of 2 therefore needs a $\sigma$ between 1 and 2; the binary search finds $\sigma = 1.40$.

**The heavy tail.** Kernels at distances 1, 2, 3, 4: Gaussian $e^{-d^2/2}$ gives $0.6065, 0.1353, 0.0111, 0.0003$;
Student-t $1/(1 + d^2)$ gives $0.5, 0.2, 0.1, 0.0588$. At distance 4 the t-kernel is about 175 times the Gaussian.

**$q$ in a picture.** Put the four points in 1-D at the same positions, $y = 0, 1, 2, 4$. The six pairs have
squared distances $1, 4, 16, 1, 9, 4$ and t-kernels $0.5, 0.2, 0.0588, 0.5, 0.1, 0.2$, which sum to 1.5588; counting
both orders $(i, j)$ and $(j, i)$, the denominator is $3.1176$. So $q_{12} = 0.5/3.1176 = 0.1604$ and
$q_{14} = 0.0588/3.1176 = 0.0189$ (the lab labels these pairs by position: `q_(0,1)` and `q_(0,4)`).

**Why the KL cost is lopsided.** A neighbour pair drawn far apart, $p = 0.2$, $q = 0.01$: $0.2\log(0.2/0.01) = +0.599$.
A distant pair drawn close, $p = 0.01$, $q = 0.2$: $0.01\log(0.01/0.2) = -0.030$, twenty times smaller in size (the
negative terms are balanced by positive ones elsewhere, because both $\mathbf{P}$ and $\mathbf{Q}$ sum to 1).

```text
== worked example by hand ==
point x=0, sigma=1: squared distances [1.0, 4.0, 16.0], weights [0.6065, 0.1353, 0.0003]
p_(j|0) for j = x=1, x=2, x=4: [0.8172, 0.1823, 0.0005]
entropy 0.4788 nats -> perplexity exp(H) = 1.6141
sigma=2.0: p = [0.5433, 0.3734, 0.0833], perplexity 2.4753
sigma=0.5: p = [0.9975, 0.0025, 0.0], perplexity 1.0175
binary search: sigma for perplexity 2 = 1.4021, p = [0.6719, 0.3133, 0.0148]
Student-t kernel 1/(1+d^2) for d = 1, 2, 3, 4: [0.5, 0.2, 0.1, 0.0588]
Gaussian kernel exp(-d^2/2)  for d = 1, 2, 3, 4: [0.6065, 0.1353, 0.0111, 0.0003]
sum over ordered pairs of 1/(1+d^2) = 3.1176;  q_(0,1) = 0.1604, q_(0,4) = 0.0189
KL term p log(p/q) with p=0.2, q=0.01: +0.5991
KL term p log(p/q) with p=0.01, q=0.2: -0.0300
```

**Trustworthiness by hand.** To score a picture we need a number for "are the neighbours in the picture real
neighbours?". The **trustworthiness** at $k$ penalizes every point $j$ that is among $i$'s $k$ nearest in the
picture but not in the input, by how far down $i$'s input ranking it sits:

$$T(k) = 1 - \frac{2}{nk(2n - 3k - 1)}\sum_{i=1}^{n}\ \sum_{j \in U_i(k)}\big(r(i, j) - k\big),$$

where $r(i, j)$ is $j$'s rank among $i$'s input neighbours (1 = nearest) and $U_i(k)$ is the set of $i$'s picture
neighbours that are not input neighbours. The fraction scales the worst possible penalty to 1, so $T = 1$ means no
intruders and $T$ near 0 means the picture's neighbourhoods are as bad as possible. Toy: points at $0, 1, 3, 7$,
drawn with the first and last swapped (0 drawn at 7 and 7 drawn at 0), $k = 1$. Point 0 now sits at 7; its picture
neighbour is the point at 3, its 2nd input neighbour: penalty $2 - 1 = 1$. Point 1's picture neighbour is the point
drawn at 0 (the real 7), its 3rd input neighbour: penalty 2. Point 3's picture neighbour is 1, its 1st: no penalty.
Point 7 (drawn at 0) has picture neighbour 1, its 2nd input neighbour: penalty 1. Total 4, and
$T(1) = 1 - \frac{2}{4 \cdot 1 \cdot (8 - 3 - 1)}\cdot 4 = 1 - 0.5 = 0.5$. A second, plainer score is
**kNN preservation**: the average share of each point's $k$ input neighbours that are also its $k$ picture
neighbours.

## Key plot

![Six panels: top row t-SNE, bottom row PCA. Left: a cluster ten times wider than another in 10-D is drawn the same size by t-SNE but ten times wider by PCA. Middle: three clusters whose centres are at 0, 10 and 100 are drawn evenly spaced by t-SNE, while PCA keeps cluster C ten times farther away. Right: pure Gaussian noise appears as dozens of small clumps in t-SNE at perplexity 2 and as one round cloud in PCA](img/b5_5_three_lies.png)

PCA (bottom row) tells the truth about sizes, gaps and noise in all three; t-SNE (top row) gets all three wrong. Yet on
real data (digits, below) PCA cannot separate what t-SNE separates. Neither is "right": they answer different
questions.

## From scratch in NumPy

**The data.** scikit-learn's bundled digits: 1,797 images of 8×8 pixels, so $\mathbf{X}$ has shape `(1797, 64)`, with
about 180 images of each digit.

```text
== data ==
digits X (1797, 64), labels 0-9, counts per digit [178, 182, 177, 183, 181, 182, 181, 179, 174, 180]
pixel values 0-16; median distance between two digits 49.09
```

**The affinities $\mathbf{P}$.** All 1,797 binary searches run at once as array operations. The search is on the
precision $\beta_i = 1/(2\sigma_i^2)$, which is what scikit-learn does: double $\beta$ until it is too large, then halve
the interval. One numerical detail matters: digits are about 49 units apart, so $e^{-\beta d^2}$ with $d^2$ in the
thousands underflows to 0. Subtracting each row's smallest squared distance before exponentiating changes nothing
after normalization (the factor cancels) and keeps the numbers in range, the same trick as a stable softmax
(A3.4).

```python
def tsne_affinities(X, perplexity, n_steps=100):
    D = pairwise_distances(X, squared=True)            # (n, n)
    np.fill_diagonal(D, np.inf)                        # a point is not its own neighbour
    D = D - D.min(axis=1, keepdims=True)               # p unchanged, no underflow
    target = np.log(perplexity)                        # target entropy in nats
    beta = np.ones(len(X)); lo = np.zeros(len(X)); hi = np.full(len(X), np.inf)
    for _ in range(n_steps):
        W = np.exp(-D * beta[:, None])
        P_cond = W / W.sum(axis=1, keepdims=True)      # row i is p_{.|i}
        H = entr(P_cond).sum(axis=1)                   # scipy.special.entr(p) = -p log p
        too_flat = H > target                          # sigma too large: raise beta
        lo = np.where(too_flat, beta, lo); hi = np.where(too_flat, hi, beta)
        beta = np.where(np.isinf(hi), beta * 2, (lo + hi) / 2)
    return (P_cond + P_cond.T) / (2 * len(X)), np.sqrt(1 / (2 * beta)), np.exp(H)
```

```text
== from scratch: t-SNE affinities P ==
P shape (1797, 1797), sum 1.000000, symmetric: True
perplexity of every p_(.|i) after the search: min 30.0000, max 30.0000 (target 30)
sigma per point: min 4.83, median 8.22, max 12.27 (dense regions get small sigmas)
max |P_mine - P_sklearn| = 1.01e-09 (largest entry 2.24e-04)
matches sklearn.manifold._t_sne._joint_probabilities: OK
```

The widths range from 4.8 to 12.3 pixels-units: points in sparse regions get Gaussians two and a half times wider,
so that every point ends up with exactly 30 effective neighbours. The match with scikit-learn's private
`_joint_probabilities` is checked to a relative 1e-3 because scikit-learn works in float32 and stops its search at an
entropy tolerance of 1e-5.

**Trustworthiness and kNN preservation.** Both are about 20 lines; the rank matrix `rank[i, j]` is filled by
scattering $1, \dots, n$ into the `argsort` order of each row of the distance matrix. The toy reproduces the hand
calculation, and the PCA picture of the digits is checked against scikit-learn:

```text
== from scratch: trustworthiness and kNN preservation ==
toy: same order -> trustworthiness(k=1) 1.0000;  0 and 7 swapped -> 0.5000
PCA 2-D of digits: trustworthiness(k=10) mine 0.8300, sklearn 0.8300
matches sklearn.manifold.trustworthiness: OK
```

## With scikit-learn and umap-learn

`sklearn.manifold.TSNE(perplexity=30, init="pca", random_state=0)` and
`umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)`, each `.fit_transform(X)` to shape `(1797, 2)`. Four scores
for each picture: trustworthiness and kNN preservation at $k = 10$ (local structure); the Spearman correlation
between the 45 distances among the ten digit centroids in 64-D and in the picture ("centroid rho", a crude measure of
global layout); and the **silhouette score** (B5.1) of the true digit labels in the picture.

```text
== with scikit-learn and umap-learn: digits ==
embedding              trust@10  kNN@10 centroid rho silhouette
PCA                      0.8300  0.1179       0.8146     0.1051
t-SNE perplexity=5       0.9911  0.5432       0.7771     0.4288
t-SNE perplexity=30      0.9925  0.5837       0.8161     0.5522
t-SNE perplexity=100     0.9905  0.5409       0.8290     0.5198
UMAP n_neighbors=5       0.9890  0.5111       0.4552     0.6126
UMAP n_neighbors=15      0.9892  0.4949       0.5387     0.6153
UMAP n_neighbors=50      0.9857  0.4504       0.6585     0.6141
64-D input (reference)   1.0000  1.0000       1.0000     0.1629
```

- **Local structure:** t-SNE and UMAP keep about 50–58% of each point's 10 nearest neighbours; PCA keeps 12%.
  Trustworthiness says the same (0.99 against 0.83).
- **Separation is exaggerated:** in the picture the digits' silhouette is 0.55 (t-SNE) to 0.62 (UMAP), far *higher*
  than in the original 64-D space (0.16). The islands look cleaner than the data is. Never report a cluster-quality
  score computed on a t-SNE or UMAP picture as a property of the data.
- **Global layout:** larger perplexity or `n_neighbors` keeps more of it (centroid rho 0.78 → 0.83 for t-SNE,
  0.46 → 0.66 for UMAP), at a small cost in local detail. UMAP's layout of the ten digits is the least faithful here,
  despite claims that UMAP "preserves global structure"; initialization matters at least as much as the method
  (Kobak & Linderman, 2021).
- The six pictures are saved to `outputs/b5_5/digits_grid.png`: the number and arrangement of islands visibly change
  with the setting.

**The three lies, measured.** The key plot's three datasets, with t-SNE, UMAP and PCA:

```text
== it lies 1: cluster sizes ==
cluster A: std 1, cluster B: std 10 (both 300 points). Spread = RMS distance to the cluster's centroid.
input 10-D           spread A   3.137   spread B  31.469   ratio B/A  10.03
PCA                  spread A   1.475   spread B  14.981   ratio B/A  10.16
t-SNE perplexity=30  spread A   9.950   spread B  10.185   ratio B/A   1.02
UMAP n_neighbors=15  spread A   1.277   spread B   1.264   ratio B/A   0.99

== it lies 2: distances between clusters ==
centres on one axis at 0 (A), 10 (B) and 100 (C): C is 10x farther from A than B is.
input 10-D           d(A,B)    10.06   d(A,C)    99.98   ratio d(A,C)/d(A,B)   9.93
PCA                  d(A,B)    10.06   d(A,C)    99.98   ratio d(A,C)/d(A,B)   9.94
t-SNE perplexity=30  d(A,B)    45.95   d(A,C)    90.75   ratio d(A,C)/d(A,B)   1.98
UMAP n_neighbors=15  d(A,B)    16.96   d(A,C)    16.22   ratio d(A,C)/d(A,B)   0.96
```

1. **Sizes:** a cluster ten times wider is drawn the same size by both methods (ratio 1.02 and 0.99). That is Step 2's
   per-point normalization: B's points get ten-times-wider Gaussians, so their neighbour sets look exactly like A's.
2. **Gaps:** C is ten times farther from A than B is. t-SNE shrinks that to 2, UMAP to about 1 (C is drawn as close to
   A as B is). That is Step 5: once two groups share no neighbours, the cost no longer cares how far apart they are.

```text
== it lies 3: clusters in pure noise ==
500 points from N(0, I) in 100-D. Clumpiness = median d(1st neighbour) / d(16th neighbour), lower = tighter clumps;
silhouette of k-means with k=6 run on that space.
input 100-D          clumpiness 0.9234   k-means silhouette 0.0079
PCA                  clumpiness 0.2169   k-means silhouette 0.3362
t-SNE perplexity=2   clumpiness 0.1064   k-means silhouette 0.3792
t-SNE perplexity=30  clumpiness 0.1678   k-means silhouette 0.3362
UMAP n_neighbors=5   clumpiness 0.1949   k-means silhouette 0.3639
```

3. **Noise:** 500 points of pure Gaussian noise. "Clumpiness" is the median ratio of the distance to a point's
   nearest neighbour to the distance to its 16th. In an even 2-D scatter, the number of points within a radius grows
   with the area, $\propto r^2$, so the 16th neighbour is about $\sqrt{16} = 4$ times as far as the 1st: a ratio of
   about 0.25. PCA's cloud gives 0.22. t-SNE at perplexity 2 halves it to 0.11: points sit in tight clumps of a few
   with empty space around them, exactly the "clusters" in the key plot. (In the 100-D input the ratio is 0.92: in
   high dimensions all distances are nearly equal, B2.3.) Note the last column too: k-means finds "clusters" with
   silhouette around 0.34 in *every* 2-D picture, PCA included, while in the 100-D data it is 0.008. A clustering
   score on a 2-D picture measures the picture.

## In practice

- **When to use it:** to *look* at high-dimensional data: are the labelled classes separable at all, is there
  structure the labels miss, are there mislabelled points sitting inside another class's island, did the embedding
  model (Parts E, F, I) put similar items together? Use it to generate hypotheses, then test them in the original
  space. To find clusters, cluster the original data or a PCA projection of it (B5.1, B5.2, B5.3), then *colour* the
  t-SNE or UMAP picture by those clusters.
- **Failure modes:**
  - *Reading cluster sizes, densities or gaps.* None is preserved (lies 1 and 2). Check them in the original space:
    per-cluster spread, centroid distances.
  - *Seeing clusters that are not there.* Low perplexity or `n_neighbors` makes clumps out of noise (lie 3). A
    structure is credible only if it survives several settings and seeds and shows up in the original space.
  - *Clustering on the picture,* or reporting a silhouette computed there. On clean data with real, well-separated
    groups (digits), clustering a t-SNE picture can work well (exercise 2), but the picture's separation is inflated
    (silhouette 0.55 against 0.16 in 64-D), and the same pipeline also "finds" clusters in noise. A cluster found on
    the picture is a hypothesis until it is confirmed in the original space and across seeds and settings.
  - *Comparing two plots* made with different settings, seeds or data: axes, orientation and scale are arbitrary.
  - *Running on raw, unscaled features.* Distances are dominated by the largest-scale feature, as for k-nearest
    neighbours (B2.3, B1.3). Standardize first, or reduce with PCA to about 50 dimensions, which also speeds things
    up.
  - *Using t-SNE to transform new data.* scikit-learn's `TSNE` has no `transform`; adding new points means rerunning
    the whole thing. UMAP has `transform`, but the result is an approximation.
- **Defaults that work:** standardize (or use the embedding as it is, for model embeddings with cosine distance:
  `metric="cosine"`); PCA to 50 dimensions when $d$ is large; t-SNE `perplexity=30`, `init="pca"`, `max_iter=1000`;
  UMAP `n_neighbors=15`, `min_dist=0.1`; set `random_state` and show at least two settings side by side. scikit-learn
  1.9.1 and umap-learn 0.5.12; t-SNE's defaults (`init="pca"`, `learning_rate="auto"`) changed in past releases, and
  setting `random_state` in UMAP forces a single thread (it warns). VERIFY (as of 2026-10).

## Exercises

Save your work to `my-work/B/b5_5_tsne_umap.md` (or `.py`) and ask for a review.

1. **By hand:** Three points on a line at $x = 0, 1, 3$. (a) For the point at 0 with $\sigma = 1$, compute
   $p_{j\mid 0}$ for the other two, the entropy in nats and the perplexity. (b) Repeat with $\sigma = 2$. If the target
   perplexity is 1.9, is the right $\sigma$ below 1, between 1 and 2, or above 2? (c) The three points are drawn at
   $y = 0, 1, 2$. Compute all $q_{ij}$ (ordered pairs). (d) Compute $p\log(p/q)$ for $p = 0.3, q = 0.03$ and for
   $p = 0.03, q = 0.3$, and explain in one sentence what the difference means for the picture.
2. **Code:** On the digits: (a) run k-means with $k = 10$ (B5.1) on the 64-D pixels, on PCA to 2 and to 10
   components, on t-SNE (perplexity 30) and on UMAP (`n_neighbors=15`, `min_dist=0.0`). For each, report the adjusted
   Rand index against the digit labels (`sklearn.metrics.adjusted_rand_score`: 1 = the same partition as the labels,
   about 0 = chance) and the silhouette of the k-means clusters both in the space you clustered and in the 64-D space.
   (b) Repeat the t-SNE clustering with `random_state` 0, 1 and 2 (with `init="pca"` and with `init="random"`) and report
   the ARI between the runs. (c) Run the same pipeline on 1,797 points of pure Gaussian noise in 64-D. Which of your
   numbers would you put in a report, and which not?
3. **Think:** A product manager shows you a UMAP of 2 million customer embeddings with seven well-separated islands.
   They propose seven customer segments, with budgets proportional to the islands' areas, and a "migration
   campaign" that nudges customers from each island to the nearest neighbouring island. What do you tell them about
   each of the three claims, and what analysis would you run instead before anyone spends money?

## Takeaways

1. t-SNE and UMAP keep *who is next to whom* (here about 55% of each point's 10 nearest neighbours, against 12% for
   PCA) and are the right tool to look at high-dimensional data.
2. Perplexity is $e^{H}$ of each point's neighbour distribution: the effective number of neighbours. Every point gets
   its own $\sigma_i$ to hit it, which is why dense and sparse clusters come out the same size.
3. The KL cost punishes separating neighbours and barely punishes anything else, and the heavy-tailed $q$ opens gaps
   for free: distances between islands mean little (a 10× gap became 2× in t-SNE and 1× in UMAP).
4. Low perplexity or `n_neighbors` makes clumps in pure noise, and any 2-D picture inflates cluster scores; trust a
   structure only if it survives several settings and seeds and shows up in the original space.
5. Cluster and measure in the original (or PCA-reduced) space, then use t-SNE or UMAP to *display* the result.

**What builds on this:** [B5.6](b5_6_anomaly_detection.md) (neighbour distances as anomaly scores); P3 (the PCA and
UMAP views of the segments); [E1.3](../E/e1_3_word2vec.md) (looking at word-embedding geometry);
[I2.1](../I/i2_1_embeddings_semantic_search.md) (looking at sentence embeddings before building search on them).

**References:** L. van der Maaten, G. Hinton, "Visualizing Data using t-SNE", *JMLR* 9, 2008. L. McInnes, J. Healy,
J. Melville, "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction", arXiv:1802.03426, 2018.
M. Wattenberg, F. Viégas, I. Johnson, "How to Use t-SNE Effectively", *Distill*, 2016 (VERIFY link:
https://distill.pub/2016/misread-tsne/). D. Kobak, G. C. Linderman, "Initialization is critical for preserving global
data structure in both t-SNE and UMAP", *Nature Biotechnology* 39, 2021 (VERIFY). J. Venna, S. Kaski,
"Neighborhood preservation in nonlinear projection methods: an experimental study", *ICANN*, 2001 (trustworthiness;
VERIFY). UMAP documentation, "How to Use UMAP" (VERIFY link: https://umap-learn.readthedocs.io/).
