# B5.4 PCA from scratch via SVD, explained variance

**Core** · Part B, week 5 (course week 11) · ~2 h · Lab:
[`labs/B/b5_4_pca.py`](../../labs/B/b5_4_pca.py)

## Why it matters

Real feature tables are wide and redundant: 64 pixels of which neighbours move together, 300 customer
metrics of which a dozen tell the story, 768-dimensional text embeddings. **PCA** finds the few directions
along which the data actually varies, so you can plot data in 2-D, compress it, denoise it, speed up a
distance-based model, or flag rows that the main directions cannot explain. It is the default first look at
any numeric dataset and the most common **dimensionality reduction** in production. Used carelessly it also
fails quietly: one feature in large units can take over every component, a PCA fitted on all rows leaks the
test set, and a refit can flip the sign of a component that a dashboard depends on.

## Prerequisite check

- Can you compute the covariance matrix of centred data, and say why it divides by $n - 1$? If not, revisit
  [A3.3](../A/a3_3_covariance_matrices.md).
- Can you say what $\mathbf{U}$, $\boldsymbol{\Sigma}$ and $\mathbf{V}$ are in a thin SVD, and why
  $\mathbf{A}^\top\mathbf{A} = \mathbf{V}\boldsymbol{\Sigma}^2\mathbf{V}^\top$? If not, revisit
  [A3.2](../A/a3_2_svd.md).
- Can you project a vector onto a unit vector and say what the leftover (the residual) is perpendicular to? If
  not, revisit [A2.4](../A/a2_4_projections_least_squares.md).
- Can you explain why a scaler must be fitted on training rows only? If not, revisit
  [B1.3](b1_3_feature_engineering.md) and [B1.4](b1_4_data_leakage.md).

## Intuition

Picture a cloud of points in 2-D shaped like a tilted cigar. Its centre is the mean. Draw a line through
the centre along the cigar: if you replace every point by its shadow on that line (its projection), you
lose a little (each point's distance to the line) but keep most of the picture, and each point is now one
number, its position along the line, instead of two. PCA chooses that line, and the next ones, by a single
rule: **keep as much of the spread as possible**. The first **principal component** is the direction of
largest variance; the second is the direction of largest variance among those perpendicular to the first;
and so on, until there are as many components as features. Nothing is lost until you decide to keep only
the first $k$.

Two goals give the same answer. "Keep the most variance" and "lose the least when reconstructing the
points from their shadows" are the same requirement, because each centred point's squared length splits,
by Pythagoras, into the part along the line plus the part off it. The total is fixed, so maximizing one
minimizes the other.

You already know the answer from Part A. In A3.3 the covariance matrix's eigenvectors were the axes of the
cloud and its eigenvalues the variances along them; in A3.2 the SVD of a matrix gave those same eigenvectors
without forming the covariance matrix. PCA is those two facts turned into a tool: centre the data, take its
SVD, keep the top $k$ right singular vectors.

> **Bridge:** PCA is like choosing which columns to put in a covering index: you want the few that serve
> most queries. **Where it breaks:** an index keeps original columns, while a principal component is a
> *new* column, a weighted mix of all the originals (every pixel gets a weight). "PC3" rarely means anything
> a stakeholder would name, and the mix changes whenever you refit.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{X}$ | data matrix, one example per row | `(n, d)` |
| $\boldsymbol{\mu}$ | column means of the training rows | `(d,)` |
| $\mathbf{X}_c$ | centred data, $\mathbf{X} - \boldsymbol{\mu}$ (row by row) | `(n, d)` |
| $\mathbf{S}$ | covariance matrix, $\frac{1}{n-1}\mathbf{X}_c^\top\mathbf{X}_c$ | `(d, d)` |
| $\mathbf{w}$ | any unit direction, $\lVert\mathbf{w}\rVert_2 = 1$ | `(d,)` |
| $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ | thin SVD of $\mathbf{X}_c$; $r = \min(n, d)$ | `(n, r)`, `(r, r)`, `(d, r)` |
| $\mathbf{v}_j$, $\sigma_j$ | $j$-th principal component (column $j$ of $\mathbf{V}$) and singular value | `(d,)`, scalar |
| $\lambda_j$ | variance along $\mathbf{v}_j$, $\sigma_j^2/(n-1)$ (the **explained variance**) | scalar |
| $k$ | number of components kept | int |
| $\mathbf{V}_k$ | the first $k$ columns of $\mathbf{V}$ | `(d, k)` |
| $\mathbf{Z}$ | **principal component scores**, $\mathbf{X}_c\mathbf{V}_k$ | `(n, k)` |
| $\hat{\mathbf{X}}$ | reconstruction from $k$ components | `(n, d)` |

Indices start at 1 in the math and 0 in the code: $\mathbf{v}_1$ is `components[0]`.

**Step 1: the variance along a direction.** From A3.3: the coordinates of the centred rows along a unit
$\mathbf{w}$ are $\mathbf{X}_c\mathbf{w}$, shape `(n,)`, they have mean 0, and their variance is
$\mathbf{w}^\top\mathbf{S}\mathbf{w}$.

**Step 2: why the top eigenvector wins** (A3.3 stated this; here is why). $\mathbf{S}$ is symmetric, so it
has orthonormal eigenvectors $\mathbf{v}_1, \dots, \mathbf{v}_d$ with eigenvalues
$\lambda_1 \ge \dots \ge \lambda_d \ge 0$ (A3.1). Orthonormal vectors form a coordinate system, so any unit
$\mathbf{w}$ can be written $\mathbf{w} = c_1\mathbf{v}_1 + \dots + c_d\mathbf{v}_d$ with $c_j = \mathbf{v}_j^\top\mathbf{w}$, and
its length squared is $c_1^2 + \dots + c_d^2 = 1$ (Pythagoras in the new coordinates). Apply $\mathbf{S}$, using
$\mathbf{S}\mathbf{v}_j = \lambda_j\mathbf{v}_j$ on each term: $\mathbf{S}\mathbf{w} = \sum_j c_j\lambda_j\mathbf{v}_j$. Dot with $\mathbf{w}$; every cross term
$\mathbf{v}_i^\top\mathbf{v}_j$ with $i \ne j$ is 0, so

$$\mathbf{w}^\top\mathbf{S}\mathbf{w} = \lambda_1c_1^2 + \lambda_2c_2^2 + \dots + \lambda_dc_d^2 \le \lambda_1(c_1^2 + \dots + c_d^2) = \lambda_1.$$

The variance along any direction is a weighted average of the eigenvalues, with weights $c_j^2$ that sum
to 1, so it can never exceed the largest one, and it equals $\lambda_1$ exactly when all the weight is on
$c_1$, that is $\mathbf{w} = \pm\mathbf{v}_1$. Among directions perpendicular to $\mathbf{v}_1$ ($c_1 = 0$)
the same argument gives a maximum of $\lambda_2$ at $\mathbf{v}_2$, and so on. **The principal components are
the eigenvectors of $\mathbf{S}$, in order of eigenvalue.**

**Step 3: the same thing from the SVD.** Take the thin SVD $\mathbf{X}_c = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$.
A3.2 showed $\mathbf{X}_c^\top\mathbf{X}_c = \mathbf{V}\boldsymbol{\Sigma}^2\mathbf{V}^\top$ (the $\mathbf{U}^\top\mathbf{U} = \mathbf{I}$
in the middle cancels). Divide by $n - 1$:

$$\mathbf{S} = \mathbf{V}\,\frac{\boldsymbol{\Sigma}^2}{n-1}\,\mathbf{V}^\top.$$

That is an eigendecomposition of $\mathbf{S}$: the columns of $\mathbf{V}$ are its eigenvectors and its
eigenvalues are $\lambda_j = \sigma_j^2/(n - 1)$. So the right singular vectors of the centred data are the
principal components, and we never have to form $\mathbf{S}$ (which squares the condition number,
[A3.4](../A/a3_4_numerical_stability.md)).

Why $\sigma_j^2/(n - 1)$ is literally a variance, without the eigen-argument: the scores along
$\mathbf{v}_j$ are $\mathbf{X}_c\mathbf{v}_j$. From $\mathbf{X}_c\mathbf{V} = \mathbf{U}\boldsymbol{\Sigma}$ (A3.2), column $j$ is
$\sigma_j\mathbf{u}_j$. Its mean is 0 (the columns of $\mathbf{X}_c$ sum to 0, so any combination of them
does too), its sum of squares is $\sigma_j^2\lVert\mathbf{u}_j\rVert^2 = \sigma_j^2$, and a variance is a sum of
squared deviations from the mean divided by $n - 1$.

**Step 4: scores and reconstruction.** Keep $k$ components. The scores are the coordinates of each row
along them:

$$\mathbf{Z} = \mathbf{X}_c\mathbf{V}_k = \mathbf{U}_k\boldsymbol{\Sigma}_k, \qquad (n, d)(d, k) \to (n, k).$$

Going back multiplies each score by its direction and adds the mean back:

$$\hat{\mathbf{X}} = \mathbf{Z}\mathbf{V}_k^\top + \boldsymbol{\mu}, \qquad (n, k)(k, d) \to (n, d).$$

$\hat{\mathbf{X}} - \boldsymbol{\mu} = \mathbf{U}_k\boldsymbol{\Sigma}_k\mathbf{V}_k^\top$ is the rank-$k$ truncation of $\mathbf{X}_c$, so by
A3.2 (Eckart–Young) no other $k$-dimensional projection reconstructs the centred training data better, and

$$\lVert\mathbf{X} - \hat{\mathbf{X}}\rVert_F^2 = \sum_{j > k}\sigma_j^2 = (n - 1)\sum_{j > k}\lambda_j.$$

In words: the total squared **reconstruction error** is exactly the variance you threw away (times $n - 1$).
The lab reports it per row, $\frac1n\lVert\mathbf{X} - \hat{\mathbf{X}}\rVert_F^2 = \frac{n-1}{n}\sum_{j>k}\lambda_j$.

**Step 5: explained-variance ratio.** The total variance is the sum of the feature variances, the trace of
$\mathbf{S}$, which equals the sum of its eigenvalues (A3.1). Component $j$'s share of it is

$$\text{ratio}_j = \frac{\lambda_j}{\sum_{i}\lambda_i} = \frac{\sigma_j^2}{\sum_i\sigma_i^2}.$$

The **cumulative** ratio $\sum_{j \le k}\text{ratio}_j$ is the "energy kept" of A3.2, computed on centred
data so the mean no longer dominates it.

**Step 6: new rows.** A new example $\mathbf{x}$ `(d,)` is centred with the *training* mean and projected on
the *training* components: $\mathbf{z} = \mathbf{V}_k^\top(\mathbf{x} - \boldsymbol{\mu})$, `(k,)`. Nothing is
refitted. That is what makes PCA a `fit`/`transform` step, with the same leakage rules as a scaler.

**Step 7: units.** $\mathbf{S}$ holds variances in squared units. A feature measured in thousands has
variance a million times that of the same feature in thousands of thousands, so it dominates
$\mathbf{w}^\top\mathbf{S}\mathbf{w}$. When features have different units, standardize first; PCA then
works on the correlation matrix (A3.3). When they share one unit (pixels, all in "ink"), standardizing
inflates near-constant columns and is usually not wanted.

## Worked example by hand

The four points of A3.3: $\mathbf{X} = \begin{pmatrix}1&4\\3&5\\4&4\\4&7\end{pmatrix}$ `(4, 2)`.

1. Mean $\boldsymbol{\mu} = (3, 5)$; centred rows $(-2, -1), (0, 0), (1, -1), (1, 2)$.
2. From A3.3, $\mathbf{S} = \begin{pmatrix}2&1\\1&2\end{pmatrix}$ with eigenvalues 3 and 1 and eigenvectors
   $\mathbf{v}_1 = (1, 1)/\sqrt2$, $\mathbf{v}_2 = (1, -1)/\sqrt2$. So the singular values of $\mathbf{X}_c$ are
   $\sigma_j = \sqrt{(n - 1)\lambda_j}$: $\sqrt{9} = 3$ and $\sqrt3 = 1.7321$.
3. Explained-variance ratio: $3/(3 + 1) = 0.75$ and $0.25$. PC1 keeps three quarters of the spread.
4. PC1 scores: $\mathbf{X}_c\mathbf{v}_1 = (-2 - 1, 0, 1 - 1, 1 + 2)/\sqrt2 = (-3, 0, 0, 3)/\sqrt2 = (-2.1213, 0, 0, 2.1213)$.
   Their variance is $(4.5 + 0 + 0 + 4.5)/3 = 3 = \lambda_1$ ✓.
5. Rank-1 reconstruction, row 1: $\boldsymbol{\mu} + z_1\mathbf{v}_1 = (3, 5) + \frac{-3}{\sqrt2}\cdot\frac{(1, 1)}{\sqrt2} = (3, 5) + (-1.5, -1.5) = (1.5, 3.5)$.
   Rows 2 and 3 have score 0 and land on the mean $(3, 5)$; row 4 lands on $(4.5, 6.5)$.
6. Errors: $(1, 4) - (1.5, 3.5) = (-0.5, 0.5)$, squared 0.5; row 2 exact, 0; $(4, 4) - (3, 5) = (1, -1)$, 2;
   $(4, 7) - (4.5, 6.5) = (-0.5, 0.5)$, 0.5. Total $3 = \sigma_2^2$ ✓ (Step 4).
7. A new point $(5, 5)$: centre with the training mean, $(2, 0)$; score $(2 + 0)/\sqrt2 = 1.4142$;
   reconstruction $(3, 5) + 1.4142\cdot(0.7071, 0.7071) = (4, 6)$.

```text
== worked example by hand ==
mean = [3.0, 5.0]   centred rows = [[-2.0, -1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 2.0]]
singular values of Xc = [3.0, 1.7321]   squared = [9.0, 3.0]
variances along the components = s^2/(n-1) = [3.0, 1.0]   (eigenvalues of S in A3.3: 3 and 1)
explained-variance ratio = [0.75, 0.25]
components (rows of Vt) = [[0.7071, 0.7071], [0.7071, -0.7071]]
PC1 scores Xc v1 = [-2.1213, 0.0, 0.0, 2.1213]   (= [-3, 0, 0, 3]/sqrt(2))
rank-1 reconstruction = [[1.5, 3.5], [3.0, 5.0], [3.0, 5.0], [4.5, 6.5]]
squared reconstruction error = 3.0000   = sigma_2^2 = 3.0000
new point [5.0, 5.0]: score 1.4142, reconstruction [4.0, 6.0]
```

## Key plot

![Left: the cumulative explained-variance ratio of the 64 digit pixels rises steeply, reaching 0.28 at 2 components, 0.74 at 10 and 0.96 at 30. Right: four test digits (0, 3, 5, 8) reconstructed from 2, 10 and 30 components next to the originals; at 2 they are grey blobs, at 10 recognizable, at 30 nearly identical to the original](img/b5_4_explained_variance_reconstructions.png)

Two components keep 28% of the variance and the reconstructions are blurred blobs that all look alike; ten
keep 74% and every digit is readable; thirty keep 96% and the reconstruction is hard to tell from the
original. The curve has no sharp elbow, which is normal for real data: where to stop depends on what the
components are for (exercise 2).

## From scratch in NumPy

**The data.** scikit-learn's bundled digits: 1,797 grey 8 × 8 images of handwritten digits, each flattened
to 64 pixel values from 0 to 16. A stratified 75/25 split; everything is fitted on the 1,347 training rows.

```text
== data ==
digits: X (1797, 64), pixel values 0..16, classes [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
train (1347, 64)   test (450, 64)
pixels that are 0 in every training image: 4 of 64
```

**Step 2, checked.** The lab draws 10,000 random unit directions in 64-D and computes the variance along
each, $\mathbf{w}^\top\mathbf{S}\mathbf{w}$:

```text
== no direction beats PC1 ==
largest eigenvalue of S (variance along PC1) = 178.7204
best of 10000 random unit directions = 65.3433   mean = 18.6455
mean over random directions = trace(S)/d = 18.7302
```

None comes close to $\lambda_1 = 178.7$. A random direction spreads its weight $c_j^2$ evenly over all 64
eigenvectors, so on average it sees the average eigenvalue, $\operatorname{tr}(\mathbf{S})/d = 18.7$. In
64 dimensions, random search for a good direction is hopeless (the **curse of dimensionality** of
[B2.3](b2_3_knn.md)); the eigenvectors are found by linear algebra, not by search.

**PCA in four functions.** Fit is "centre, SVD, fix the signs"; transform and inverse are one matrix product
each:

```python
def pca_fit(X, k):
    mean = X.mean(axis=0)                              # (d,)
    _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)   # s (r,), Vt (r, d)
    signs = np.sign(Vt[np.arange(len(s)), np.argmax(np.abs(Vt), axis=1)])
    Vt = Vt * signs[:, None]                           # largest-|entry| of each component positive
    var = s**2 / (len(X) - 1)                          # (r,) Step 3
    return mean, Vt[:k], var[:k], var[:k] / var.sum()  # components are ROWS of Vt

def pca_transform(X, mean, components):  # (m, d) -> (m, k)
    return (X - mean) @ components.T

def pca_inverse(Z, mean, components):    # (m, k) -> (m, d)
    return Z @ components + mean
```

Each singular vector's sign is arbitrary (A3.2: $\mathbf{u}_j$ and $\mathbf{v}_j$ can flip together). The
sign line fixes one convention, the same one scikit-learn uses, so refits on the same data give the same
components. `var.sum()` uses all $r$ singular values, so the ratio is a share of the *total* variance even
when you keep only $k$.

```text
== from scratch: PCA on digits ==
components (64, 64), orthonormal: True
explained variance (first 5) = [178.72, 162.655, 144.297, 98.687, 69.778]
explained-variance ratio (first 5) = [0.1491, 0.1357, 0.1204, 0.0823, 0.0582]
components needed for 80% of the variance: 13
components needed for 90% of the variance: 21
components needed for 95% of the variance: 28
components needed for 99% of the variance: 41
sum of variances = total variance trace(S): True
variance along PC1 = largest eigenvalue of S: True
k =  2: scores (1347, 2)  train error 856.7203 (dropped variance x (n-1)/n = 856.7203)  test error 868.6504
k = 10: scores (1347, 10)  train error 312.8325 (dropped variance x (n-1)/n = 312.8325)  test error 323.3769
k = 30: scores (1347, 30)  train error  48.8829 (dropped variance x (n-1)/n =  48.8829)  test error  51.8558
k = 64: scores (1347, 64)  train error   0.0000 (dropped variance x (n-1)/n =   0.0000)  test error   0.0000
```

- No single component dominates: PC1 explains 15%, and 28 components are needed for 95%. Handwriting has
  many independent ways to vary.
- The training reconstruction error equals the dropped variance to every printed digit (Step 4). The error
  is the mean over rows of the squared distance between an image and its reconstruction, in squared pixel
  units: at $k = 10$, a root-mean-square distance of $\sqrt{313} \approx 18$ ink units across the 64 pixels.
- Test rows reconstruct slightly worse than training rows: the components were chosen to fit the training
  rows, so unseen rows are a little off-plane. This gap is what makes reconstruction error usable as an
  anomaly score ([B5.6](b5_6_anomaly_detection.md)): rows unlike the training data reconstruct badly.
- At $k = 64$ nothing is dropped and reconstruction is exact: PCA with all components is only a rotation.

## With scikit-learn

```text
== with scikit-learn ==
sign flips needed to align with sklearn: 0 of 10
components, explained_variance_, explained_variance_ratio_, transform, inverse_transform match sklearn PCA: OK
PCA(n_components=0.95) keeps 28 components (ratio sum 0.9503)
```

`PCA(n_components=k)` stores the components as **rows** of `components_` `(k, d)`, the means in `mean_`,
$\lambda_j$ in `explained_variance_` (divided by $n - 1$, like ours) and the ratios in
`explained_variance_ratio_`. The check aligns signs before comparing, because another library or version
may pick the other sign; here none were needed because the from-scratch code copies scikit-learn's
convention. A float between 0 and 1, `PCA(n_components=0.95)`, keeps the fewest components that reach that
share. `svd_solver="full"` runs the exact LAPACK SVD; the default `"auto"` may switch to a randomized solver
on large inputs (VERIFY (as of 2026-10) the switching rule in the scikit-learn docs). Unlike
`TruncatedSVD` (A3.2), `PCA` always centres.

**Standardize or not.** The wine dataset has 13 chemical measurements in different units:

```text
== standardize or not (wine) ==
wine: X (178, 13); largest std proline = 314.9, smallest nonflavanoid_phenols = 0.124
raw           PC1 ratio 0.9981, PC2 ratio 0.0017;  PC1's largest weight: proline (0.9998)
standardized  PC1 ratio 0.3620, PC2 ratio 0.1921;  PC1's largest weight: flavanoids (0.4229)
```

On raw wine, "PC1 explains 99.8% of the variance" sounds like a discovery and is an artefact: proline's
standard deviation is about 2,500 times the smallest one's, so PC1 is proline alone (weight 0.9998). After
standardizing (B1.3), PC1 explains 36% and mixes several features. A ratio near 1 on raw data with mixed
units is a sign to check the scales, not a result.

**Fit on training rows only.** PCA does not use labels, so it cannot leak a target, but the components
still learn from whatever rows you give them:

```text
== fit on train only ==
k = 10: test reconstruction error, PCA fitted on train 323.3769  on train+test 316.6906  (2.1% optimistic)
k = 30: test reconstruction error, PCA fitted on train 51.8558  on train+test 48.6368  (6.2% optimistic)
```

Fitted on train and test together, the components bend towards the test rows, and the test reconstruction
error looks 2–6% better than it would be on genuinely new data. For a classifier on top, the effect on
accuracy is usually small; for anything that *scores* rows by reconstruction error (anomaly detection,
B5.6; drift monitoring, [J2.2](../J/j2_2_monitoring_drift.md)) it is exactly the number you report, so it
matters. The rule is the one from [B1.4](b1_4_data_leakage.md): every step that learns goes inside the
training folds.

**PCA inside a Pipeline.** Putting PCA in the pipeline refits it on each cross-validation training fold
automatically:

```text
== PCA inside a Pipeline ==
  k cum. ratio  CV accuracy    std
  2     0.2848       0.6013 0.0174
  5     0.5457       0.8552 0.0241
 10     0.7388       0.9235 0.0150
 20     0.8951       0.9391 0.0061
 30     0.9592       0.9517 0.0082
 64     1.0000       0.9621 0.0142
```

Two components already separate the ten digits 60% of the time (chance is 10%); ten reach 92%; all 64
reach 96%. Variance kept and accuracy rise together but not in proportion: PCA ranks directions by spread,
and spread is not the same as usefulness for the label. A direction with little variance can carry the one
stroke that separates a 3 from an 8, which is why the number of components should be tuned on the task
metric (exercise 2) rather than fixed by a variance rule.

## In practice

- **When to use it:** a first look at numeric data (scatter the first two scores, colour by a known label);
  compressing wide, correlated features before a distance-based or linear model (kNN, k-means in
  [B5.1](b5_1_k_means.md), logistic regression); speeding up t-SNE and UMAP by running them on the top
  ~50 components ([B5.5](b5_5_tsne_umap.md)); denoising (keep the top components, drop the noisy tail);
  reconstruction error as an anomaly score ([B5.6](b5_6_anomaly_detection.md)); shrinking embeddings before
  storage or search ([I2.1](../I/i2_1_embeddings_semantic_search.md)).
- **Failure modes:**
  - *Mixed units.* One large-unit feature becomes PC1 and the ratio is near 1 (the wine run). You notice a
    component dominated by one weight; standardize first.
  - *Fitted on all rows.* Test reconstruction errors and anomaly scores look better than they will in
    production. You notice a `fit` or `fit_transform` before the split; put PCA in the `Pipeline`.
  - *Sign flips and reordering on refit.* A retrain can flip $\mathbf{v}_j$ to $-\mathbf{v}_j$ or swap two
    components with nearly equal variance. A downstream model or a dashboard reading "PC1" breaks. You
    notice a feature whose effect inverts after a retrain; fix a sign convention, refit downstream models
    together with the PCA, and never let a human-facing label depend on component order.
  - *Non-linear structure.* PCA finds flat subspaces. Data on a curved surface (a spiral, a ring) needs
    many components, or a non-linear method (t-SNE/UMAP for plots, B5.5; autoencoders,
    [G1.1](../G/g1_1_autoencoders.md)).
  - *Interpretation.* Components are mixes of every feature; reading "PC2 = price sensitivity" from the top
    weights is a story, not a measurement. Report the weights, and check the story on held-out data.
  - *Before tree models.* Trees split on single features and do not need decorrelated inputs; rotating the
    features with PCA often makes them worse, because a split on one original feature becomes a diagonal
    cut. You notice it as a drop in validation score; leave tree inputs alone.
- **Defaults that work:** `make_pipeline(StandardScaler(), PCA(n_components=k), model)` when units differ,
  without the scaler when they share one unit; choose $k$ by cross-validating the downstream metric, and use
  `n_components=0.95` only as a starting point; `PCA(svd_solver="full")` up to tens of thousands of rows,
  the randomized solver (`svd_solver="randomized", random_state=SEED`) or `IncrementalPCA` (fits in
  mini-batches) when the matrix is large (scikit-learn 1.9.1; VERIFY (as of 2026-10)).

## Exercises

Save your work to `my-work/B/b5_4_pca.md` (or `.py`) and ask for a review.

1. **By hand:** $\mathbf{X} = \begin{pmatrix}2&1\\4&3\\6&5\\4&7\end{pmatrix}$ `(4, 2)`. (a) Centre it, compute
   $\mathbf{X}_c^\top\mathbf{X}_c$ and $\mathbf{S}$, and find the eigenvalues and unit eigenvectors of $\mathbf{S}$
   (make the largest entry of each eigenvector positive). (b) Give the explained-variance ratio of each
   component. (c) Compute the PC1 scores of the four rows, the rank-1 reconstruction of all four, and the
   total squared reconstruction error; check it against Step 4. (d) A new row $(6, 6)$ arrives. Give its PC1
   score, its reconstruction and its squared reconstruction error, using the training mean.
2. **Code:** On the lab's digits split, put `PCA` and `LogisticRegression(max_iter=2000)` in a `Pipeline`
   and tune `n_components` over $\{5, 10, 15, 20, 25, 30, 40, 50, 64\}$ with `GridSearchCV` and 5-fold
   stratified CV on the training rows only (`random_state=0`). Report mean and standard deviation of the CV
   accuracy for each $k$. Which $k$ is best, and which is the smallest $k$ within one standard deviation of
   the best? Which $k$ do the "90% of variance" and "95% of variance" rules pick, and how do they compare?
   Score your chosen pipeline on the test set once.
3. **Think:** A product team has 300 numeric features per customer (usage counts, spend, ticket counts, a
   few one-hot plan types). They want to run PCA, keep 20 components, feed them to their churn model, and
   put "PC1 = engagement" and "PC2 = price sensitivity" on a dashboard that is refreshed by refitting the
   PCA every month. What would you check or change before agreeing, and what could go wrong over the
   following months?

## Takeaways

1. PCA centres the data and takes its SVD; the principal components are the right singular vectors
   $\mathbf{v}_j$, which are the eigenvectors of the covariance matrix, and the variance along each is
   $\lambda_j = \sigma_j^2/(n - 1)$.
2. The variance along any unit direction is a weighted average of the eigenvalues, so no direction beats
   PC1; keeping the most variance and losing the least reconstruction error are the same goal.
3. With $k$ components, scores are $\mathbf{Z} = \mathbf{X}_c\mathbf{V}_k$ `(n, k)`, reconstruction is
   $\mathbf{Z}\mathbf{V}_k^\top + \boldsymbol{\mu}$, and the squared reconstruction error equals the dropped
   variance times $n - 1$.
4. PCA is scale-sensitive: standardize features with different units (raw wine's PC1 was proline alone,
   "99.8%"); and it is a fitted step, so it goes inside the `Pipeline`, fitted on training rows only.
5. Choose $k$ by the downstream task, not by a variance rule alone: on digits the 95% rule kept 28 of 64
   components, while accuracy kept climbing past it.

**What builds on this:** [B5.5](b5_5_tsne_umap.md) (t-SNE and UMAP, often run on the top PCA components);
[B5.6](b5_6_anomaly_detection.md) (reconstruction error as an anomaly score); [B6.1](b6_1_recommender_systems.md)
(matrix factorization, a low-rank model with missing entries); [E1.3](../E/e1_3_word2vec.md) and
[I2.1](../I/i2_1_embeddings_semantic_search.md) (projecting embeddings to 2-D, shrinking them for search);
[G1.1](../G/g1_1_autoencoders.md) (an autoencoder is a non-linear PCA with a bottleneck);
[J1.3](../J/j1_3_lora_qlora.md) (low-rank weight updates); P3 (PCA views of your segments).

**References:** K. Pearson, "On lines and planes of closest fit to systems of points in space",
*Philosophical Magazine* 2, 1901. H. Hotelling, "Analysis of a complex of statistical variables into principal
components", *Journal of Educational Psychology* 24, 1933. I. T. Jolliffe, *Principal Component Analysis*,
2nd ed., Springer, 2002. scikit-learn User Guide, "Decomposing signals in components" (VERIFY link:
https://scikit-learn.org/stable/modules/decomposition.html).
