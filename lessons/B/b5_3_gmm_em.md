# B5.3 Gaussian mixture models and the EM algorithm

**Depth** · Part B, week 5 (course week 11) · ~2 h · Lab:
[`labs/B/b5_3_gmm_em.py`](../../labs/B/b5_3_gmm_em.py)

## Why it matters

K-means (B5.1) gives every point exactly one cluster and assumes the clusters are round and of similar size. Real segments are often tilted, unequal and overlapping, and a customer who sits halfway
between two segments should be reported as "60/40", not forced into one. A **Gaussian mixture model** (GMM) describes the data as a weighted sum of Gaussians with their own shapes, and gives each point a
probability of belonging to each one. It is also a density model: $p(\mathbf{x})$ is a real number you can threshold for anomaly detection (B5.6). The algorithm that fits it, **EM** (expectation–maximization), is the
general recipe for models with hidden variables, and its lower-bound argument is the one that variational autoencoders (G1.2) are built on.

## Prerequisite check

- Can you write the density of a multivariate Gaussian and say what its covariance matrix does to the shape of the contours? If not, revisit [A5.2](../A/a5_2_common_distributions.md) and
  [A3.3](../A/a3_3_covariance_matrices.md).
- Can you apply Bayes' rule: posterior = prior × likelihood / evidence? If not, revisit [A5.3](../A/a5_3_bayes_theorem.md).
- Can you derive the maximum-likelihood mean and variance of a Gaussian from data? If not, revisit [A5.5](../A/a5_5_mle_map.md).
- Do you know why we compute $\log\sum_i e^{s_i}$ by subtracting the maximum first? If not, revisit [A3.4](../A/a3_4_numerical_stability.md).
- Can you run k-means (assign to the nearest centroid, recompute the centroids)? If not, revisit [B5.1](b5_1_k_means.md).

## Intuition

Imagine the data was made by a two-step process. First, a hidden die with $K$ unequal faces picks a component: "segment 2". Then a point is drawn from that component's Gaussian, which has its own centre,
spread and tilt. You see the points but not the die rolls. The die roll is a **latent variable**: part of the story, never observed.

If you knew every die roll, fitting would be easy: for each component, take its points and compute their mean and covariance (A5.5), and set its weight to the share of points it produced. If you knew the
components' parameters, you could guess each die roll with Bayes' rule: "a point here is 98% likely to come from component 1". EM does the obvious thing with these two half-solutions and alternates:

1. **E-step (expectation):** with the current parameters, compute for every point the probability that each component produced it. These probabilities are the point's **responsibilities**.
2. **M-step (maximization):** recompute each component's weight, mean and covariance as if every point belonged to it *in proportion to its responsibility*: a point with responsibility 0.3 counts as 0.3 of a point.

Repeat until nothing changes. This is k-means with soft assignments and shaped clusters. K-means does the same alternation, but with "100% to the nearest centroid" instead of probabilities, and with
every cluster forced to be a round blob of the same size. The lab shows that k-means is exactly the limit of EM when every component is a round Gaussian whose variance shrinks to zero.

Two properties make EM trustworthy and two make it awkward. Trustworthy: each round never lowers the likelihood of the data (the lab asserts it), and each step has a closed form, with no learning rate.
Awkward: it climbs to the nearest hill, not necessarily the highest one (a local maximum), and the likelihood can run off to infinity if a component shrinks onto a single point.

> **Bridge:** EM is like reconciling two systems where each one's state is only computable from the other's: a retry loop that alternates "recompute derived table B from A" and "recompute A from B" until a
> checksum stops changing. **Where it breaks:** a reconciliation loop usually has one correct answer; EM can settle into a different stable state depending on where it started, so you run it several times
> from different starts and keep the best (`n_init`).

## The math

**Symbols**

In this lesson only, $\pi_k$ is a mixture weight, not the prevalence of B4.4.

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{X}$ | data matrix, one example per row | `(n, d)` |
| $\mathbf{x}_i$ | example $i$ | `(d,)` |
| $K$ | number of mixture components | int |
| $\pi_k$ | weight of component $k$; $\pi_k \ge 0$, $\sum_k \pi_k = 1$ | scalar (all of them: `(K,)`) |
| $\boldsymbol{\mu}_k$ | mean of component $k$ | `(d,)` (all: `(K, d)`) |
| $\boldsymbol{\Sigma}_k$ | covariance of component $k$ | `(d, d)` (all: `(K, d, d)`) |
| $\boldsymbol{\theta}$ | all parameters $\{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}_{k=1}^K$ | — |
| $z_i$ | latent component of example $i$, a number in $1..K$ | int |
| $\gamma_{ik}$ | responsibility: $P(z_i = k \mid \mathbf{x}_i)$ | scalar (all: `(n, K)`, rows sum to 1) |
| $N_k$ | effective number of points in component $k$, $\sum_i \gamma_{ik}$ | scalar |
| $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})$ | Gaussian density at $\mathbf{x}$ (A5.2) | scalar |
| $\ell(\boldsymbol{\theta})$ | log-likelihood of the data | scalar |

Indices run from 1 in the math and from 0 in the code: component $k = 1$ is column `0` of `resp`.

**The model.** The two-step story says $P(z = k) = \pi_k$ and $p(\mathbf{x} \mid z = k) = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$. The density of a point, not knowing $z$, sums over the ways it could have been made
(the law of total probability, A5.3):
$$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k\, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k),\qquad
\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{\sqrt{(2\pi)^d \det\boldsymbol{\Sigma}}}\exp\!\Big(-\tfrac12(\mathbf{x} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\Big).$$
(The $\pi$ inside $(2\pi)^d$ is the number 3.14159…, not a weight.)

**Why there is no direct formula.** The log-likelihood of the $n$ points (independent, so their densities multiply and their logs add) is
$$\ell(\boldsymbol{\theta}) = \sum_{i=1}^{n} \log \sum_{k=1}^{K} \pi_k\, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k).$$
For a single Gaussian the log would cancel the exponential and the mean would come out as the sample mean (A5.5). Here the log sits outside a *sum* of exponentials, nothing cancels, and setting the
derivative to zero gives equations in which every component's parameters depend on all the others'. EM breaks that knot.

**E-step: Bayes' rule.** Fix the parameters. For point $i$, the prior is $P(z_i = k) = \pi_k$, the likelihood is $\mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$, and the evidence is $p(\mathbf{x}_i)$, the sum above. Bayes' rule
(A5.3) gives the posterior:
$$\gamma_{ik} = P(z_i = k \mid \mathbf{x}_i) = \frac{\pi_k\,\mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K}\pi_j\,\mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}.$$
Each row of $\boldsymbol{\Gamma}$ (the `(n, K)` matrix of $\gamma_{ik}$) sums to 1 because the numerators are divided by their own sum.

**M-step: weighted maximum likelihood.** If we knew each $z_i$, the log-likelihood would split into one piece per component: $\sum_i \sum_k [z_i = k]\big(\log\pi_k + \log\mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)\big)$, where $[z_i = k]$ is 1 or 0. We do not
know $z_i$, so EM replaces each indicator by its probability $\gamma_{ik}$ and maximizes
$$Q(\boldsymbol{\theta}) = \sum_{i=1}^{n}\sum_{k=1}^{K}\gamma_{ik}\Big(\log\pi_k + \log\mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)\Big),$$
with the $\gamma_{ik}$ held fixed. The log now sits directly on each Gaussian, so the pieces separate. Here is each update, derived in 1-D (the multivariate case follows the same steps with matrix
calculus and gives the formulas in the box).

*The mean.* In 1-D, $\log\mathcal{N}(x \mid \mu, \sigma^2) = -\tfrac12\log(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2}$. Only the last term contains $\mu$, and its derivative with respect to $\mu$ is $\frac{x - \mu}{\sigma^2}$ (chain rule: the derivative of $-(x - \mu)^2/2$ is $+(x - \mu)$). Set the derivative of $Q$ to zero:
$$\sum_i \gamma_{ik}\frac{x_i - \mu_k}{\sigma_k^2} = 0 \;\Rightarrow\; \sum_i \gamma_{ik}x_i = \mu_k\sum_i\gamma_{ik} \;\Rightarrow\; \mu_k = \frac{\sum_i \gamma_{ik}\,x_i}{N_k}.$$
A weighted average of all the points, each weighted by how much it belongs to component $k$.

*The variance.* Write $v = \sigma_k^2$ and differentiate with respect to $v$: $\frac{\partial}{\partial v}\big[-\tfrac12\log v - \frac{(x - \mu)^2}{2v}\big] = -\frac{1}{2v} + \frac{(x - \mu)^2}{2v^2}$. Set the weighted sum to zero and multiply by $2v^2$:
$$\sum_i \gamma_{ik}\big(-v + (x_i - \mu_k)^2\big) = 0 \;\Rightarrow\; \sigma_k^2 = \frac{\sum_i \gamma_{ik}(x_i - \mu_k)^2}{N_k}.$$

*The weights.* Only $\sum_i\sum_k\gamma_{ik}\log\pi_k = \sum_k N_k\log\pi_k$ involves them, and they must sum to 1. With a Lagrange multiplier (A4.5), maximize $\sum_k N_k\log\pi_k - \lambda(\sum_k\pi_k - 1)$: the
derivative $N_k/\pi_k - \lambda = 0$ gives $\pi_k = N_k/\lambda$; summing over $k$, $1 = n/\lambda$ (because $\sum_k N_k = \sum_i\sum_k\gamma_{ik} = \sum_i 1 = n$), so $\lambda = n$.

$$\boxed{\;N_k = \sum_i\gamma_{ik},\quad \pi_k = \frac{N_k}{n},\quad \boldsymbol{\mu}_k = \frac{1}{N_k}\sum_i \gamma_{ik}\,\mathbf{x}_i,\quad \boldsymbol{\Sigma}_k = \frac{1}{N_k}\sum_i \gamma_{ik}\,(\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^\top\;}$$

Shapes: $(\mathbf{x}_i - \boldsymbol{\mu}_k)$ is `(d,)`, so the outer product $(\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^\top$ is `(d, d)`, the same as $\boldsymbol{\Sigma}_k$. With every $\gamma_{ik}$ equal to 0 or 1 these are exactly the sample mean, the sample
covariance (dividing by the count, the MLE of A5.5) and the share of points.

**Why the log-likelihood never decreases.** For any probabilities $q_1, \dots, q_K$ over the components of a point, the log of an average is at least the average of the logs (**Jensen's inequality**, because $\log$ is concave:
$\log\frac{1 + 100}{2} = 3.92 > \frac{\log 1 + \log 100}{2} = 2.30$). Write $p(\mathbf{x}_i) = \sum_k q_k \cdot \frac{\pi_k\mathcal{N}_k}{q_k}$ as such an average, with $\mathcal{N}_k$ short for $\mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$:
$$\log p(\mathbf{x}_i) \;\ge\; \sum_k q_k \log\frac{\pi_k\mathcal{N}_k}{q_k}.$$
The right side is a **lower bound** on the log-likelihood. If $q_k = \gamma_{ik}$, then $\frac{\pi_k\mathcal{N}_k}{\gamma_{ik}} = p(\mathbf{x}_i)$ for every $k$ (rearrange the E-step formula), so the bound equals $\log p(\mathbf{x}_i)$ exactly. So: the E-step makes
the bound touch the log-likelihood at the current parameters; the M-step maximizes the bound over $\boldsymbol{\theta}$ with the $q$'s held at the old $\gamma$'s, which can only raise it (summed over $i$, the bound is
$\sum_i\sum_k\gamma_{ik}\log(\pi_k\mathcal{N}_k) - \sum_i\sum_k\gamma_{ik}\log\gamma_{ik}$: the first part is exactly $Q$, the second does not contain $\boldsymbol{\theta}$); and since the inequality holds for *any* $q$ and *any* parameters, the log-likelihood at the new
parameters is at least the bound there. Each round therefore ends no lower than it began. The same bound, with a neural network producing the $q$'s, is the **ELBO** of variational autoencoders (G1.2).

**Numerical stability.** In 50 dimensions a Gaussian density can be $e^{-800}$, which is 0.0 in float64, and the E-step becomes $0/0$. So the code works in logs: $\log\gamma_{ik} = s_{ik} - \log\sum_j e^{s_{ij}}$ with
$s_{ik} = \log\pi_k + \log\mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$, and $\log\sum_j e^{s_{ij}}$ computed by **log-sum-exp** (subtract the largest $s_{ij}$ first, A3.4). The log-density itself uses a Cholesky factor
$\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$: the log-determinant is $2\sum_j\log L_{jj}$, and the squared **Mahalanobis distance** is $\lVert\mathbf{L}^{-1}(\mathbf{x} - \boldsymbol{\mu})\rVert_2^2$, with no matrix inverse computed.

**Choosing K.** The likelihood always rises with more components (more parameters fit better), so it cannot pick $K$. The **BIC** (Bayesian information criterion) adds a penalty per parameter:
$$\text{BIC} = -2\,\ell(\hat{\boldsymbol{\theta}}) + p\log n,\qquad p = \underbrace{(K - 1)}_{\text{weights}} + \underbrace{Kd}_{\text{means}} + \underbrace{K\,\tfrac{d(d + 1)}{2}}_{\text{full covariances}},$$
and the lowest BIC wins. The weights have $K - 1$ free values because they sum to 1; a symmetric `(d, d)` covariance has $d(d + 1)/2$ free entries.

## Worked example by hand

Four points in 1-D, $x = (1, 2, 4, 5)$, two components started at $\mu_1 = 1$, $\mu_2 = 5$, $\sigma_1^2 = \sigma_2^2 = 1$, $\pi_1 = \pi_2 = 0.5$.

**E-step.** Equal weights and equal variances cancel in the ratio, leaving $\gamma_{i1} = \frac{e^{-(x_i - 1)^2/2}}{e^{-(x_i - 1)^2/2} + e^{-(x_i - 5)^2/2}} = \frac{1}{1 + e^{[(x_i - 1)^2 - (x_i - 5)^2]/2}}$.

- $x = 1$: exponent $(0 - 16)/2 = -8$, $\gamma = 1/(1 + e^{-8}) = 1/(1 + 0.000335) = 0.99966$.
- $x = 2$: exponent $(1 - 9)/2 = -4$, $\gamma = 1/(1 + 0.0183) = 0.98201$.
- $x = 4$ and $x = 5$: by symmetry $\gamma = 0.01799$ and $0.00034$.

**M-step for component 1.** $N_1 = 0.99966 + 0.98201 + 0.01799 + 0.00034 = 2.0000$, so $\pi_1 = 2/4 = 0.5$.
$\mu_1 = (0.99966 \cdot 1 + 0.98201 \cdot 2 + 0.01799 \cdot 4 + 0.00034 \cdot 5)/2 = 3.0373/2 = 1.5187$.
Squared deviations from 1.5187: $0.2690, 0.2317, 6.1571, 12.1197$; weighted sum $0.2689 + 0.2275 + 0.1108 + 0.0041 = 0.6113$; $\sigma_1^2 = 0.6113/2 = 0.3056$.
Component 2 is the mirror image: $\mu_2 = 6 - 1.5187 = 4.4813$, $\sigma_2^2 = 0.3056$.

The means moved towards the middle of their pairs (1 and 2, 4 and 5), and the variances shrank from 1 to 0.31 because each pair is tight. The points 2 and 4 pull each mean a little, through their
small responsibilities. The total log-likelihood rises from $-7.4114$ to $-5.7157$.

```text
== worked example by hand ==
responsibilities of component 1: [0.9997, 0.982, 0.018, 0.0003]
N_1 = 2.0000   pi_1 = 0.5000   mu_1 = 1.5187   sigma_1^2 = 0.3056
N_2 = 2.0000   pi_2 = 0.5000   mu_2 = 4.4813   sigma_2^2 = 0.3056
total log-likelihood: before -7.4114   after one EM step -5.7157
```

## Key plot

![Left: 1,200 points in three tilted clusters, coloured by their most likely component, with the fitted 1- and 2-standard-deviation ellipses following each cluster's tilt and size (weights 0.24, 0.25, 0.51). Right: the mean log-likelihood per point rises from -4.98 at iteration 1 to -3.45 by iteration 10 and stays flat after](img/b5_3_em_ellipses.png)

The ellipses are the Gaussians' contours at 1 and 2 standard deviations, drawn from the eigenvectors and eigenvalues of each fitted $\boldsymbol{\Sigma}_k$ (A3.3): the long axis points along the
largest-variance direction. EM recovered two tilted clusters with opposite slopes and one round one, and the weights 0.24 / 0.25 / 0.51 of the true 1:1:2 sizes. The right panel is the
monotone climb the math promised: most of the gain comes in the first ten iterations.

## From scratch in NumPy

**The data.** A 1-D sample of 900 points from three Gaussians (weights 0.5, 0.3, 0.2) and a 2-D sample of 1,200 points from three clusters, two of them tilted.

```text
== data ==
1-D: (900,) points from 3 components, sizes [450, 270, 180]
2-D: X (1200, 2), true component sizes [600, 300, 300]
```

**The log-density**, through the Cholesky factor:

```python
def log_gaussian(X, mean, cov):                    # X (n, d), mean (d,), cov (d, d) -> (n,)
    L = np.linalg.cholesky(cov)                    # cov = L L^T
    z = np.linalg.solve(L, (X - mean).T)           # (d, n) whitened deviations
    maha2 = np.sum(z**2, axis=0)                   # squared Mahalanobis distances, (n,)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + log_det + maha2)
```

**E-step and M-step**, the boxed formulas line for line:

```python
def e_step(X, weights, means, covs):
    log_joint = np.column_stack([np.log(w) + log_gaussian(X, m, c)
                                 for w, m, c in zip(weights, means, covs)])   # (n, K)
    log_px = logsumexp(log_joint, axis=1, keepdims=True)                      # (n, 1): log p(x_i)
    return np.exp(log_joint - log_px), float(np.mean(log_px))                 # Bayes' rule in logs

def m_step(X, resp, reg):
    nk = resp.sum(axis=0)                           # (K,)
    weights = nk / len(X)
    means = (resp.T @ X) / nk[:, None]              # (K, n) @ (n, d) -> (K, d)
    covs = np.empty((K, d, d))
    for k in range(K):
        D = X - means[k]                            # (n, d)
        covs[k] = (resp[:, k, None] * D).T @ D / nk[k] + reg * np.eye(d)   # (d, d)
    return weights, means, covs
```

`resp.T @ X` computes all $K$ weighted sums $\sum_i\gamma_{ik}\mathbf{x}_i$ at once: row $k$ of `resp.T` holds the weights of component $k$. The small `reg` (scikit-learn's `reg_covar`, $10^{-6}$) is added to every
diagonal so a covariance can never become singular; the lab's "degenerate component" cell shows why that matters. The function `fit_em` loops E, M, E, M… and records the log-likelihood each time, and
the lab asserts that it never decreases.

**1-D, from two different starts.** The same data, the same code, the same number of iterations; only the initial means differ.

```text
== from scratch: 1-D EM ==
start from random points: initial means [-2.166, -2.006, -1.67]; mean log-likelihood per point: iteration 1 -2.6550, 10 -2.2973, 100 -2.1670  (never decreased: OK)
start from quantiles    : initial means [-2.38, -0.15, 3.108]; mean log-likelihood per point: iteration 1 -2.3870, 10 -2.1660, 100 -2.0582  (never decreased: OK)
                  weight    mean     std
random points      0.378  -2.158   0.653
random points      0.261   0.704   2.027
random points      0.361   1.978   2.006
quantiles          0.503  -1.998   0.815
quantiles          0.298   1.004   0.480
quantiles          0.199   3.962   0.883
true               0.500  -2.000   0.800
true               0.300   1.000   0.500
true               0.200   4.000   1.000
```

The random start happened to put all three initial means inside the left cluster. EM climbed faithfully (the likelihood never decreased) but to a **local maximum**: one component covers the left
cluster and the other two share the right half as two wide Gaussians (std 2.0) that cannot separate the clusters at 1 and 4. The quantile start reaches a higher likelihood (−2.058 against
−2.167 per point) and recovers every true parameter to within sampling noise. Monotone does not mean optimal. This is why libraries run EM several times (`n_init`) and start it from k-means.

**2-D, full covariances:**

```text
== from scratch: 2-D EM ==
mean log-likelihood per point: iteration 1 -4.9822, 10 -3.4523, 100 -3.4495  (never decreased: OK)
weights [0.238, 0.253, 0.509]   (true [0.25, 0.25, 0.5])
adjusted Rand index of the hard assignments against the true components: 0.9308
points whose largest responsibility is below 0.9: 86 of 1200 (7.2%)
```

The **adjusted Rand index** ([B5.1](b5_1_k_means.md)) compares two groupings of the same points while ignoring the label names: 1 means identical, 0 means no better than chance. Taking each point's most likely component
("hard" assignment) agrees with the true components at 0.93; the misses are where the tilted cluster's tip touches the round one. That overlap is visible in the responsibilities: 86 points (7%) have no
component above 0.9. Those are exactly the customers you would report as "between segments", which k-means cannot do.

## With scikit-learn

`GaussianMixture` runs the same algorithm. To check the code rather than the luck of the start, give it the same starting weights, means and precisions (inverse covariances), `tol=0.0` so it runs all
100 iterations, and the same `reg_covar`:

```python
gm = GaussianMixture(n_components=3, covariance_type="full", max_iter=100, tol=0.0, reg_covar=1e-6,
                     weights_init=w_init, means_init=m_init, precisions_init=np.linalg.inv(c_init))
```

```text
== with scikit-learn: GaussianMixture ==
weights, means and covariances match sklearn GaussianMixture: OK
mean log-likelihood and responsibilities match gm.score and gm.predict_proba: OK
sklearn's own convergence check (tol=1e-3, default init): converged in 9 iterations
```

The tolerance is $10^{-6}$ relative: scikit-learn computes the same updates through Cholesky factors of the precision matrices, so the last few digits differ, and 100 iterations can amplify
float rounding slightly. `gm.score(X)` is the mean log-likelihood per point; `gm.predict_proba(X)` is the responsibility matrix; `gm.predict(X)` is its row-wise argmax. With its defaults (start from
k-means, stop when the bound improves by less than $10^{-3}$ per point) it converged in 9 iterations here (scikit-learn 1.9.1; VERIFY as of 2026-10).

**Choosing K with BIC**, from scratch and with `gm.bic`:

```text
== choosing K with BIC ==
 K  params       log L  BIC (mine)  BIC (sklearn)
 1       5     -5434.3     10904.1        10904.1
 2      11     -4361.1      8800.1         8800.1
 3      17     -4139.9      8400.3         8400.3
 4      23     -4140.1      8443.3         8443.3
 5      29     -4136.8      8479.2         8479.2
 6      35     -4132.6      8513.3         8513.3
lowest BIC at K = 3; from-scratch BIC matches g.bic: OK
```

From $K = 3$ on, the log-likelihood barely improves (−4139.9 to −4132.6 over three more components) while each component costs 6 parameters, that is $6\log 1200 = 42.5$ BIC points. The minimum is at the
true $K = 3$. ($K = 4$ even has a slightly lower log-likelihood than $K = 3$: each fit is a different local maximum, picked as the best of `n_init=3` starts.) On real data that is not a mixture of
Gaussians, BIC tends to keep adding components to approximate non-Gaussian shapes, so treat it as a guide, as with the elbow and the silhouette in B5.1.

**k-means is the hard limit.** Give every component weight 0.5 and the same spherical variance $s^2$. The E-step ratio becomes $\gamma_{i1} = 1/\big(1 + e^{(\lVert\mathbf{x}_i - \boldsymbol{\mu}_1\rVert^2 - \lVert\mathbf{x}_i - \boldsymbol{\mu}_2\rVert^2)/(2s^2)}\big)$: as $s^2 \to 0$
the exponent goes to $-\infty$ for the nearer mean, and the responsibility snaps to 1. The M-step mean then becomes the plain mean of the assigned points: k-means.

```text
== k-means as the limit of EM ==
shared variance  4.00: responsibility of the nearer mean (1.0) = 0.5987
shared variance  1.00: responsibility of the nearer mean (1.0) = 0.8320
shared variance  0.25: responsibility of the nearer mean (1.0) = 0.9983
shared variance  0.01: responsibility of the nearer mean (1.0) = 1.0000
adjusted Rand index on the tilted 2-D clusters: k-means 0.7457   GMM (full covariance) 0.9308
```

On the tilted clusters, k-means' round-cluster assumption costs it: it cuts the long cluster across its length (ARI 0.75 against the GMM's 0.93).

**Why `reg_covar` exists.** Put a component's mean exactly on one data point and shrink its standard deviation: its density at that point, and so the likelihood, grows without bound.

```text
== degenerate component ==
std  1.000: log density at its own mean =   -0.919
std  0.100: log density at its own mean =    1.384
std  0.010: log density at its own mean =    3.686
std  0.001: log density at its own mean =    5.989
```

Each factor of 10 adds $\log 10 = 2.30$. The maximum likelihood of a GMM is technically infinite, reached by a useless collapsed component. A floor on the variances (`reg_covar`), or duplicate-free
data and several restarts, keeps EM away from it.

## In practice

- **When to use it:** soft segmentation where "between segments" is a real answer; clusters that are elliptical, tilted or of unequal size; a density estimate $p(\mathbf{x})$, for example as an anomaly score
  $-\log p(\mathbf{x})$ (B5.6) or to sample synthetic points (`gm.sample`). Not for clusters with curved or ring shapes (two moons, circles): use DBSCAN or hierarchical clustering (B5.2).
- **Failure modes:**
  - *Local maxima.* Different starts, different answers. You notice by running several seeds and comparing the final log-likelihood and the assignments (adjusted Rand index). Fix: `n_init` ≥ 5, k-means initialization.
  - *Collapsing components.* A variance shrinking towards `reg_covar`, a weight near 0, or a log-likelihood that jumps late in training. Fix: `reg_covar`, fewer components, remove duplicate rows.
  - *Too many parameters.* Full covariances cost $K\,d(d + 1)/2$ parameters: 1,275 per component at $d = 50$. Use `covariance_type="diag"` (each feature its own variance, no tilt), `"tied"` (one shared covariance) or
    PCA first (B5.4), and let BIC compare them.
  - *Unscaled features.* A feature in dollars dominates one in years in every distance and covariance. Standardize first (B1.3), as for k-means.
  - *Reading responsibilities as truth.* They are probabilities *under the model*; if the clusters are not Gaussian, a confident 0.99 can still be wrong.
- **Defaults that work:** `StandardScaler` then `GaussianMixture(n_components=K, covariance_type="full", n_init=5, random_state=0)`, with $K$ chosen by BIC over a range and sanity-checked by looking at the
  segments; `"diag"` above about 20 features. Defaults: `init_params="kmeans"`, `tol=1e-3`, `max_iter=100`, `reg_covar=1e-6` (scikit-learn 1.9.1; VERIFY as of 2026-10). For a version that chooses the number of
  components itself, see `BayesianGaussianMixture`.

## Exercises

Save your work to `my-work/B/b5_3_gmm_em.md` (or `.py`) and ask for a review.

1. **By hand:** Three points in 1-D, $x = (0, 2, 6)$, two components with $\mu_1 = 0$, $\mu_2 = 6$, $\sigma_1^2 = \sigma_2^2 = 4$, $\pi_1 = \pi_2 = 0.5$. (a) Compute the three responsibilities $\gamma_{i1}$. (b) Do one M-step: $N_1$, $\pi_1$,
   $\mu_1$ and $\sigma_1^2$. (c) Recompute $\gamma_{21}$ (the point $x = 2$) if instead $\pi_1 = 0.2$, $\pi_2 = 0.8$, with everything else as at the start, and explain the change in one sentence. (d) Count the free
   parameters of a GMM with $K = 4$, $d = 3$ for full and for diagonal covariances.
2. **Code:** Write a from-scratch EM for **diagonal** covariances (each $\boldsymbol{\Sigma}_k$ is a diagonal matrix, so the log-density is a sum of 1-D log-densities and you never need a Cholesky factor). Check it against
   `GaussianMixture(covariance_type="diag")` with the same initialization, as the lab does for `"full"`. Then, on the lab's 2-D data, compute BIC for $K = 1..6$ with both covariance types and report which
   combination wins and why the diagonal model needs more components.
3. **Think:** A marketing team asks you to segment 50,000 customers on 8 standardized behaviour features. You fit a GMM with $K = 5$; 18% of customers have no responsibility above 0.6. The team
   wants "one segment per customer, for the CRM". What do you deliver, how do you explain the 18% to them, and what checks would you run before anyone acts on the segments?

## Takeaways

1. A GMM models the data as $p(\mathbf{x}) = \sum_k\pi_k\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$: clusters with their own size, shape and tilt, and a density you can evaluate.
2. EM alternates Bayes' rule (E-step: responsibilities $\gamma_{ik}$) and weighted maximum likelihood (M-step: weighted means, covariances and shares), each in closed form.
3. Every EM round raises a lower bound that touches the log-likelihood, so the log-likelihood never decreases; but it converges to a local maximum, so use several starts (`n_init`).
4. K-means is EM with equal round components whose variance shrinks to zero; on tilted clusters the GMM recovered the truth far better (ARI 0.93 against 0.75).
5. Work in logs (log-sum-exp, Cholesky log-determinants), keep `reg_covar` to stop components collapsing, standardize features, and pick $K$ and the covariance type with BIC plus judgement.

**What builds on this:** [B5.6](b5_6_anomaly_detection.md) (density-based anomaly scores); P3 (soft segments and their stability); [G1.2](../G/g1_2_vae.md) (the same lower bound, the ELBO, trains a VAE);
[B6.1](b6_1_recommender_systems.md) (alternating updates again, in alternating least squares).

**References:** A. P. Dempster, N. M. Laird, D. B. Rubin, "Maximum likelihood from incomplete data via the EM algorithm", *Journal of the Royal Statistical Society B* 39, 1977. C. M. Bishop, *Pattern
Recognition and Machine Learning*, Springer, 2006, chapter 9. G. Schwarz, "Estimating the dimension of a model", *Annals of Statistics* 6, 1978 (BIC). scikit-learn User Guide, "Gaussian mixture models"
(VERIFY link: https://scikit-learn.org/stable/modules/mixture.html).
