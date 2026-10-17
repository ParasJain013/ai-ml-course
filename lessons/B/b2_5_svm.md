# B2.5 Support vector machines: maximum margin, hinge loss and the kernel trick

**Depth** · Part B, week 2 (course week 08) · ~2 h · Lab: [`labs/B/b2_5_svm.py`](../../labs/B/b2_5_svm.py)

## Why it matters

**Support vector machines (SVMs)** were the strongest general-purpose classifiers of the late 1990s and 2000s. They
are still a good choice for small to medium datasets with a non-linear boundary, and a standard interview topic. Three
ideas from them are used far beyond SVMs. The **hinge loss** is a classification loss that stops caring once a point is
classified with enough room. It returns in ranking and metric learning, and margin-based losses appear in contrastive
training (G2.2). The **maximum margin** idea is that the boundary farthest from both classes generalizes best. The
**kernel trick** computes similarities in a huge feature space without ever building it, an idea that returns in
attention (F1.1) and Gaussian processes. This topic is Depth: skip it if the week runs long, but come back before
interviews.

## Prerequisite check

- Can you write a linear classifier's decision rule $\operatorname{sign}(\mathbf{w}^\top\mathbf{x} + b)$ and train one by gradient descent on a loss with an L2 penalty? If not, revisit
  [B2.1](b2_1_logistic_regression.md) and [B2.2](b2_2_regularization.md).
- Can you set up a Lagrangian for a constrained minimum and read off the stationarity conditions? If not, revisit
  [A4.5](../A/a4_5_convexity_lagrange.md).
- Can you compute the distance from a point to a line? If not, revisit [A2.4](../A/a2_4_projections_least_squares.md)
  (projections).
- Do you know why distance-based methods need scaled features? If not, revisit [B2.3](b2_3_knn.md).
- Do you know why the mean of a dataset's labels is not enough to judge a classifier? If not, revisit [B1.1](b1_1_ml_workflow.md).

## Intuition

If two classes can be separated by a straight line, there are infinitely many lines that do it. Logistic regression
picks one by maximizing likelihood. An SVM picks **the line farthest from the nearest points of both classes**: it
draws the widest possible street between the classes and puts the boundary down its middle. A wide street means
that a new point slightly off from the training points still lands on the correct side. The points on the edges of the
street are the **support vectors**. They alone determine the boundary: move any other point (without crossing the
street) and nothing changes.

Real data overlaps, so a perfect street does not exist. The **soft margin** allows some points inside the street or on
the wrong side, at a cost. The **hinge loss** measures that cost: 0 for a point outside the street on its own side, rising
linearly the further it intrudes. The hyperparameter $C$ sets the exchange rate: large $C$ means violations are expensive, so the
street narrows to fit the training data; small $C$ tolerates violations in exchange for a wider, more robust street.

For curved boundaries, SVMs use the **kernel trick**. It turns out that the SVM never needs the features themselves,
only dot products between pairs of points. So replace the dot product by a **kernel**, a similarity function that
equals a dot product in some richer feature space. The RBF kernel $e^{-\gamma\lVert\mathbf{x} - \mathbf{z}\rVert^2}$ corresponds to an infinite-dimensional space, yet costs one
distance to compute. The resulting classifier is a weighted vote of similarity bumps centred on the support vectors.
It is a smarter kNN (B2.3): it learns which training points matter and how much.

> **Bridge:** the support vectors are like the binding constraints in a capacity plan. Out of thousands of services,
> a handful sit at their limit and set the plan; the rest have slack, and you could change them without changing
> anything. **Where it breaks:** a capacity plan's constraints are given, while an SVM discovers which points are binding
> by solving the optimization. And with noisy data "a handful" can be most of the training set (174 of 180 points in one
> lab setting), and then the model is neither sparse nor cheap.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{X}$ | training features (standardized) | `(n, d)` |
| $y_i$ | label coded $-1$ or $+1$ (not 0/1) | scalar |
| $\mathbf{w}$, $b$ | weights and bias of the boundary $f(\mathbf{x}) = \mathbf{w}^\top\mathbf{x} + b = 0$ | `(d,)`, scalar |
| $y_if(\mathbf{x}_i)$ | functional margin: positive if correct, $\ge 1$ if outside the street | scalar |
| $\xi_i$ | slack: how far point $i$ violates the margin | scalar $\ge 0$ |
| $C$ | cost of violations (inverse regularization strength) | scalar $> 0$ |
| $\lambda$ | the same trade-off in our notation, $1/(Cn)$ | scalar |
| $\alpha_i$ | Lagrange multiplier (dual weight) of point $i$; $0 \le \alpha_i \le C$ | scalar |
| $k(\mathbf{x}, \mathbf{z})$ | kernel: a dot product in a feature space, $\boldsymbol{\phi}(\mathbf{x})^\top\boldsymbol{\phi}(\mathbf{z})$ | scalar |
| $\gamma$ | RBF kernel width parameter (larger = narrower bumps) | scalar $> 0$ |

**Step 1: distance to the boundary.** The vector $\mathbf{w}$ is perpendicular to the hyperplane $\mathbf{w}^\top\mathbf{x} + b = 0$ (A2.4). Moving from a point $\mathbf{x}$ straight to the
plane changes $f$ from $f(\mathbf{x})$ to 0. Along the unit direction $\mathbf{u} = \mathbf{w}/\lVert\mathbf{w}\rVert$, a step of length $s$ changes $f$ by $\mathbf{w}^\top(s\mathbf{u}) = s\lVert\mathbf{w}\rVert$, so $f$ changes by $\lVert\mathbf{w}\rVert$ per unit of distance. So the distance is $\lvert f(\mathbf{x})\rvert/\lVert\mathbf{w}\rVert$.

**Step 2: the hard-margin problem.** Scaling $\mathbf{w}$ and $b$ by the same factor does not move the plane, so fix the scale by requiring
$y_if(\mathbf{x}_i) \ge 1$ for all points, with equality for the closest ones. Those closest points are then at distance $1/\lVert\mathbf{w}\rVert$, and the street is
$2/\lVert\mathbf{w}\rVert$ wide. Maximizing the width means minimizing $\lVert\mathbf{w}\rVert$, or more conveniently $\frac12\lVert\mathbf{w}\rVert^2$:

$$\min_{\mathbf{w}, b}\ \tfrac12\lVert\mathbf{w}\rVert^2 \quad\text{subject to}\quad y_i(\mathbf{w}^\top\mathbf{x}_i + b) \ge 1 \ \text{ for all } i.$$

**Step 3: the soft margin and the hinge loss.** Allow violations $\xi_i \ge 0$, with $y_if(\mathbf{x}_i) \ge 1 - \xi_i$, and pay $C$ per unit:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}}\ \tfrac12\lVert\mathbf{w}\rVert^2 + C\sum_i \xi_i \quad\text{subject to}\quad y_if(\mathbf{x}_i) \ge 1 - \xi_i,\ \ \xi_i \ge 0.$$

For fixed $\mathbf{w}, b$, the cheapest slack is $\xi_i = 0$ if $y_if(\mathbf{x}_i) \ge 1$ and $\xi_i = 1 - y_if(\mathbf{x}_i)$ otherwise, that is $\xi_i = \max(0, 1 - y_if(\mathbf{x}_i))$, the **hinge loss**. Substituting it and dividing
by $Cn$ (which does not move the minimum):

$$\min_{\mathbf{w}, b}\ \frac{\lambda}{2}\lVert\mathbf{w}\rVert^2 + \frac1n\sum_i \max\big(0,\ 1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)\big), \qquad \lambda = \frac{1}{Cn}.$$

This is L2-regularized logistic regression (B2.2) with the log-loss $\log(1 + e^{-yf})$ replaced by the hinge $\max(0, 1 - yf)$. Both punish
wrong answers roughly linearly. The difference is near the boundary. The log-loss never reaches zero, so every point keeps
pulling a little. The hinge is exactly zero once $yf \ge 1$, so confidently correct points stop mattering. That is why only
the support vectors determine the solution, and why an SVM gives scores, not probabilities.

**Step 4: sub-gradient descent.** The hinge has a kink at $yf = 1$, so use a **sub-gradient**: any slope between the left and right
derivatives at the kink. Per point, the hinge's slope with respect to $\mathbf{w}$ is $-y_i\mathbf{x}_i$ if $y_if(\mathbf{x}_i) < 1$ and $\mathbf{0}$ otherwise. So

$$\mathbf{g}_{\mathbf{w}} = \lambda\mathbf{w} - \frac1n\sum_{i:\, y_if_i < 1} y_i\mathbf{x}_i, \qquad g_b = -\frac1n\sum_{i:\, y_if_i < 1} y_i.$$

With kinks, a fixed step size never settles: the active set flips back and forth. A decreasing step $\eta_t = 1/(\lambda t)$ converges (the
Pegasos schedule, which uses the $\lambda$-strong convexity of the penalty).

**Step 5: the dual and the support vectors (A4.5).** For the hard margin, the Lagrangian is
$\mathcal{L} = \frac12\lVert\mathbf{w}\rVert^2 - \sum_i\alpha_i\big[y_i(\mathbf{w}^\top\mathbf{x}_i + b) - 1\big]$, with multipliers $\alpha_i \ge 0$. Setting its derivatives to zero:

$$\frac{\partial\mathcal{L}}{\partial\mathbf{w}} = \mathbf{w} - \sum_i\alpha_iy_i\mathbf{x}_i = \mathbf{0} \ \Rightarrow\ \mathbf{w} = \sum_i\alpha_iy_i\mathbf{x}_i, \qquad \frac{\partial\mathcal{L}}{\partial b} = -\sum_i\alpha_iy_i = 0.$$

The weight vector is a weighted sum of training points. **Complementary slackness** (a constraint's multiplier is
zero unless the constraint is tight) says $\alpha_i > 0$ only for points with $y_if(\mathbf{x}_i) = 1$, the support vectors. In the soft-margin version, the
same steps add the bound $\alpha_i \le C$, and points inside the street sit at $\alpha_i = C$. Substituting $\mathbf{w}$ back gives the decision function

$$f(\mathbf{x}) = \sum_{i \in \text{SV}}\alpha_iy_i\,\mathbf{x}_i^\top\mathbf{x} + b,$$

and the dual problem (solved by libsvm inside `SVC`) also depends on the data only through the dot products $\mathbf{x}_i^\top\mathbf{x}_j$.

**Step 6: the kernel trick.** Replace every dot product $\mathbf{x}_i^\top\mathbf{x}$ by $k(\mathbf{x}_i, \mathbf{x}) = \boldsymbol{\phi}(\mathbf{x}_i)^\top\boldsymbol{\phi}(\mathbf{x})$. The SVM then runs as a linear SVM on the features $\boldsymbol{\phi}(\mathbf{x})$,
which you never compute. Example in 2-D: $k(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^\top\mathbf{z})^2 = (x_1z_1 + x_2z_2)^2 = x_1^2z_1^2 + 2x_1x_2z_1z_2 + x_2^2z_2^2$, which is the dot product of
$\boldsymbol{\phi}(\mathbf{x}) = (x_1^2, \sqrt2\,x_1x_2, x_2^2)$ with $\boldsymbol{\phi}(\mathbf{z})$. A linear boundary in $\boldsymbol{\phi}$-space is a quadratic curve (an ellipse or hyperbola) in the original space. The **RBF
(Gaussian) kernel** $k(\mathbf{x}, \mathbf{z}) = \exp(-\gamma\lVert\mathbf{x} - \mathbf{z}\rVert^2)$ is 1 for identical points and falls to 0 with distance, at a rate set by $\gamma$. Its $\boldsymbol{\phi}$ has infinitely many
dimensions, yet each kernel value costs one distance. With it, $f(\mathbf{x}) = \sum_{\text{SV}}\alpha_iy_ie^{-\gamma\lVert\mathbf{x}_i - \mathbf{x}\rVert^2} + b$: a sum of bumps centred on the support vectors.
Large $\gamma$ means narrow bumps and a wiggly boundary; small $\gamma$ means wide bumps and a nearly linear one.

**Cost.** Training a kernel SVM works with the $n \times n$ matrix of kernel values: roughly $O(n^2)$ to $O(n^3)$ time and $O(n^2)$ memory, so it
becomes impractical beyond tens of thousands of rows. Prediction costs one kernel per support vector. Linear SVMs (`LinearSVC`,
or `SGDClassifier(loss="hinge")`) scale to millions of rows.

## Worked example by hand

Four separable points in 2-D: $(0, 0)$ and $(1, 0)$ with $y = -1$; $(2, 2)$ and $(3, 2)$ with $y = +1$.

**Find the support vectors.** The closest opposite-class pair is $(1, 0)$ and $(2, 2)$, at distance $\sqrt{1^2 + 2^2} = \sqrt5 = 2.236$ (every other cross pair is farther:
$(0, 0)$–$(2, 2)$ is $\sqrt8$, $(1, 0)$–$(3, 2)$ is $\sqrt8$). With just these two points, the widest street is the one perpendicular to the segment
between them, so $\mathbf{w}$ points along $(2, 2) - (1, 0) = (1, 2)$: $\mathbf{w} = k(1, 2)$ for some $k > 0$.

**Solve for the scale.** Both support vectors sit on the street's edges: $f(2, 2) = +1$ and $f(1, 0) = -1$:

$$k(1 \cdot 2 + 2 \cdot 2) + b = 6k + b = 1, \qquad k(1 \cdot 1 + 2 \cdot 0) + b = k + b = -1.$$

Subtracting: $5k = 2$, so $k = 0.4$, $\mathbf{w} = (0.4, 0.8)$ and $b = -1 - 0.4 = -1.4$.

**Check.** $y_if(\mathbf{x}_i)$: $(0, 0)$: $-1 \times (-1.4) = 1.4$; $(1, 0)$: $-1 \times (0.4 - 1.4) = 1$; $(2, 2)$: $0.8 + 1.6 - 1.4 = 1$; $(3, 2)$: $1.2 + 1.6 - 1.4 = 1.4$. All $\ge 1$, so the other two points are
outside the street and are not support vectors. Street width: $2/\lVert\mathbf{w}\rVert = 2/\sqrt{0.16 + 0.64} = 2/0.894 = 2.236$, exactly the distance between the support vectors.

**Hinge losses for new positive points** ($y = +1$). At $(1.5, 1)$, the midpoint: $f = 0.6 + 0.8 - 1.4 = 0$, on the boundary, hinge $= 1$. At $(2, 1)$: $f = 0.2$,
correct side but inside the street, hinge $= 0.8$. At $(2.5, 2)$: $f = 1.2$, outside the street, hinge $= 0$.

**Kernel check.** $\mathbf{x} = (1, 2)$, $\mathbf{z} = (3, 1)$: $\mathbf{x}^\top\mathbf{z} = 5$, so $(\mathbf{x}^\top\mathbf{z})^2 = 25$. Explicitly, $\boldsymbol{\phi}(\mathbf{x}) = (1, 2\sqrt2, 4)$ and $\boldsymbol{\phi}(\mathbf{z}) = (9, 3\sqrt2, 1)$, and their dot product is
$9 + 2\sqrt2 \cdot 3\sqrt2 + 4 = 9 + 12 + 4 = 25$. RBF: $\lVert\mathbf{x} - \mathbf{z}\rVert^2 = 4 + 1 = 5$; $\gamma = 0.1$ gives $e^{-0.5} = 0.607$ ("fairly similar"), $\gamma = 1$ gives $e^{-5} = 0.0067$ ("unrelated").

```text
== worked example by hand ==
hand solution w = [0.4, 0.8]  b = -1.4   y * f(x) per point = [1.4, 1.0, 1.0, 1.4]
margin width 2 / |w| = 2.2361   distance between (1, 0) and (2, 2) = 2.2361
SVC(kernel='linear', C=1e6): w = [0.4, 0.8]  b = -1.4000  support vectors [[1.0, 0.0], [2.0, 2.0]]
point [1.5, 1.0] with y = +1: y f(x) = 0.0000   hinge loss max(0, 1 - y f) = 1.0000
point [2.0, 1.0] with y = +1: y f(x) = 0.2000   hinge loss max(0, 1 - y f) = 0.8000
point [2.5, 2.0] with y = +1: y f(x) = 1.2000   hinge loss max(0, 1 - y f) = 0.0000
phi(x) = [1.0, 2.8284, 4.0]   phi(z) = [9.0, 4.2426, 1.0]
kernel trick: (x . z)^2 = (5.0)^2 = 25.0   phi(x) . phi(z) = 25.0000
RBF kernel exp(-gamma |x - z|^2), |x - z|^2 = 5.0: gamma 0.1 -> 0.6065   gamma 1 -> 0.0067
```

`SVC` with a huge $C$ (so that violations are essentially forbidden, the hard margin) finds the same $\mathbf{w}$, $b$ and the same two support vectors.

## Key plot

![A 3-by-3 grid of RBF SVM decision regions on the two-moons data, C = 0.1, 1, 100 down the rows and gamma = 0.1, 1, 10 across the columns. Small gamma gives nearly straight boundaries; gamma = 10 wraps tightly around individual points; small C gives wide margins with most points circled as support vectors; C = 100 gives narrow margins. The best validation accuracy, 0.93, is at C = 1 with gamma = 1 or 10](img/b2_5_rbf_c_gamma_grid.png)

Across each row, larger $\gamma$ bends the boundary more, from almost straight at $\gamma = 0.1$ to islands around single points at $\gamma = 10$. Down each column,
larger $C$ narrows the dashed margins and makes the fit chase training points. The circled support vectors are many
when the margin is wide (small $C$) and few when it is narrow. The middle setting ($C = 1$, $\gamma = 1$) follows the moons without
chasing noise.

## From scratch in NumPy

**The data.** The five breast-cancer features and split from B2.1, standardized, with labels recoded to $-1$ (benign) and $+1$
(malignant):

```text
== data ==
train Z (343, 5)  val Z (113, 5)  (test 113 rows kept aside)   labels [-1.0, 1.0]
```

**A linear SVM by sub-gradient descent** (Step 4):

```python
for t in range(1, steps + 1):
    active = y * (Z @ w + b) < 1.0                                       # inside the margin or wrong
    grad_w = lam * w - (y[active, None] * Z[active]).sum(axis=0) / len(y)  # (d,)
    grad_b = -y[active].sum() / len(y)
    eta = 1.0 / (lam * t)                                                # decreasing step
    w, b = w - eta * grad_w, b - eta * grad_b
```

```text
== from scratch: linear SVM by sub-gradient descent ==
C = 1.0 -> lambda = 1/(C n) = 0.002915
step    100: objective 0.164016   max |w - w_svc| 1.6e+00
step   1000: objective 0.144502   max |w - w_svc| 9.2e-04
step  10000: objective 0.144497   max |w - w_svc| 7.7e-05
step 100000: objective 0.144496   max |w - w_svc| 9.6e-06
SVC optimum objective 0.144496
from scratch: w [2.1807, 0.7401, 0.5409, 1.3016, 0.0902]  b -0.2982
SVC linear:   w [2.1807, 0.7401, 0.5409, 1.3016, 0.0902]  b -0.2982
from-scratch linear SVM matches SVC(kernel='linear') to 1e-3: OK
training points with y f(x) <= 1 (on or inside the margin): 56   SVC support vectors: 56
val accuracy: from scratch 0.9823   SVC 0.9823
```

- After a fast start, the error in the weights shrinks only about tenfold per tenfold more steps (from step 1,000 on), the slow
  $1/t$ rate typical of sub-gradient methods. Compare B2.1's smooth log-loss, where gradient descent converged to $10^{-7}$ in 2,000 steps. Kinks cost speed. libsvm
  avoids the problem by solving the dual exactly.
- 56 of 343 training points (16%) are on or inside the margin, and they are exactly `SVC`'s support vectors. The other 287
  could be deleted without changing the model.

## With scikit-learn

```text
== with scikit-learn: LinearSVC and logistic regression ==
LinearSVC(loss='hinge')  w [2.1807, 0.7401, 0.5409, 1.3016, 0.0902]  b -0.2982  val acc 0.9823
LinearSVC() default      w [1.4204, 0.5122, 0.4254, 0.681, 0.0065]  b -0.2434  val acc 0.9823
LogisticRegression(C=1)  w [3.2104, 1.2383, 0.9008, 1.6782, 0.0956]  b -0.6906  val acc 0.9735
LinearSVC(loss='hinge') max |w - w_svc| 3.7e-07
decision_function for 3 val tumours (signed score, not a probability): [-2.7523, -0.7278, -2.4991]
```

- `LinearSVC()` with its defaults is **not** the textbook SVM. It minimizes the *squared* hinge loss and also penalizes the
  intercept (VERIFY as of 2026-10, scikit-learn 1.9.1). That is why its weights differ. With `loss="hinge"` it matches `SVC`
  here (its intercept penalty happens to be negligible on this data). Read the objective before comparing models.
- The SVM, the squared-hinge SVM and logistic regression all point in roughly the same direction (radius and concavity
  dominate) and get the same validation accuracy within one or two tumours. On data like this, the loss matters less
  than the features and the regularization strength.
- `decision_function` returns $f(\mathbf{x})$, a signed score. For probabilities, `SVC(probability=True)` fits a sigmoid to the scores (Platt scaling) with an
  internal 5-fold cross-validation. That is slow, and its `predict_proba` can disagree with `predict` near the boundary.

**The kernel trick on two moons.** Two interleaved half-circles with noise, where no straight line works:

```text
== kernel SVM on two moons ==
moons: train (180, 2)  val (60, 2)  (features already on comparable scales)
linear SVC on moons: val accuracy 0.8667
RBF C =   0.1  gamma =  0.1: train acc 0.7889  val acc 0.8500  support vectors 139 of 180
RBF C =   0.1  gamma =  1.0: train acc 0.8944  val acc 0.9167  support vectors 114 of 180
RBF C =   0.1  gamma = 10.0: train acc 0.9556  val acc 0.9167  support vectors 174 of 180
RBF C =   1.0  gamma =  0.1: train acc 0.8167  val acc 0.8667  support vectors  84 of 180
RBF C =   1.0  gamma =  1.0: train acc 0.9667  val acc 0.9333  support vectors  55 of 180
RBF C =   1.0  gamma = 10.0: train acc 0.9722  val acc 0.9333  support vectors  87 of 180
RBF C = 100.0  gamma =  0.1: train acc 0.9389  val acc 0.8833  support vectors  55 of 180
RBF C = 100.0  gamma =  1.0: train acc 0.9722  val acc 0.9000  support vectors  27 of 180
RBF C = 100.0  gamma = 10.0: train acc 0.9944  val acc 0.8833  support vectors  56 of 180
f(x) = sum over 55 support vectors of alpha_i y_i k(x_i, x) + b reproduces decision_function: OK
alpha_i y_i ranges over [-1.0000, 1.0000]  (|alpha_i| <= C = 1; 44 at the bound C)
```

- The linear SVM manages 0.87; the RBF kernel reaches 0.93.
- $C = 100$, $\gamma = 10$ has the best training accuracy (0.99) and one of the worst validation scores (0.88): the islands in the key plot's
  corner panel are overfitting. Small $\gamma$ with small $C$ underfits (0.79 on training).
- With 60 validation points, 0.93 against 0.92 is one point. Tune $C$ and $\gamma$ on a log grid by cross-validation (B4.3), not on one split.
- Recomputing $f(\mathbf{x}) = \sum\alpha_iy_ik(\mathbf{x}_i, \mathbf{x}) + b$ from `dual_coef_` (which stores $\alpha_iy_i$) and the 55 support vectors reproduces `decision_function`: Step 5 and Step 6 are
  literally how the model predicts. 44 of the 55 multipliers sit at the bound $C$: those points are inside the margin.

## In practice

- **When to use it:** small to medium tabular or vector data (up to roughly $10^4$–$10^5$ rows) with a non-linear boundary,
  where an RBF SVM is a strong, well-understood model; high-dimensional sparse data such as text, where a *linear* SVM
  (`LinearSVC`) is fast and accurate. For large tabular data, gradient-boosted trees (B3.3) usually win and need no scaling.
  For images, audio and text at scale, neural networks win.
- **Failure modes:**
  - *Unscaled features.* The RBF kernel is a distance, so the problems of B2.3 apply in full: one large-unit feature
    decides every kernel value. Standardize in a pipeline.
  - *Bad $C$ / $\gamma$.* Training accuracy near 1.0 with poor validation accuracy means $\gamma$ or $C$ is too large. Most points becoming
    support vectors with low training accuracy means the model is too constrained. Search both on a log grid (for example
    $10^{-3}$ to $10^3$) jointly, because they interact.
  - *Training time and memory.* Fitting slows sharply as $n$ grows ($n^2$ kernel values). Subsample, use a linear SVM, or
    approximate the kernel with random features (`RBFSampler`, `Nystroem`) followed by a linear model.
  - *Treating scores as probabilities.* `decision_function` is not calibrated, and `probability=True` is slow and can contradict `predict`. If you
    need probabilities, calibrate explicitly (B4.1) or use logistic regression.
  - *Imbalanced classes.* The margin is pulled towards the minority class. Use `class_weight="balanced"` and tune the
    threshold on the decision function (B4.4).
  - *Label coding by hand.* The math needs $y \in \{-1, +1\}$; scikit-learn handles this internally, but hand-written code with 0/1 labels
    gives a silently wrong hinge loss.
- **Defaults that work:** `make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))` (`gamma="scale"` sets
  $\gamma = 1/(d \cdot \operatorname{Var}[\mathbf{X}])$; VERIFY as of 2026-10, scikit-learn 1.9.1), then a cross-validated grid over `C` and `gamma`; `LinearSVC(C=1.0)`
  or `SGDClassifier(loss="hinge")` for large or sparse data.

## Exercises

Save your work to `my-work/B/b2_5_svm.md` (or `.py`) and ask for a review.

1. **By hand:** Points $(1, 1)$ and $(2, 0)$ with $y = -1$, and $(3, 3)$ and $(4, 2)$ with $y = +1$. (a) Find the closest opposite-class pairs (two pairs tie: what do
   they have in common?) and use them to find the hard-margin $\mathbf{w}$ and $b$, as in the worked example. Check all four
   constraints; which points are support vectors? (b) Give the street width. (c) For $C$-SVM with
   $\mathbf{w} = (0.5, 0.5)$, $b = -2.5$, compute the hinge loss of each of the four points and the objective $\frac12\lVert\mathbf{w}\rVert^2 + C\sum_i\xi_i$ for $C = 1$. (d) Show that $k(\mathbf{x}, \mathbf{z}) = (1 + \mathbf{x}^\top\mathbf{z})^2$ for
   2-D inputs is a dot product $\boldsymbol{\phi}(\mathbf{x})^\top\boldsymbol{\phi}(\mathbf{z})$, and write $\boldsymbol{\phi}$ (it has 6 components).
2. **Code:** Hinge versus log-loss, from scratch. On the lab's breast-cancer split (all 30 features, standardized,
   labels $\pm1$): (i) train your sub-gradient linear SVM and an L2 logistic regression by gradient descent, both with $\lambda = 1/(Cn)$ for
   $C \in \{0.01, 0.1, 1, 10\}$, and check each against `SVC(kernel="linear")` and `LogisticRegression`; (ii) for each $C$, report validation accuracy, the number of
   training points with non-zero loss gradient (hinge: $yf < 1$; logistic: all, but count those with $p(1 - p) > 0.01$), and the norm of
   $\mathbf{w}$; (iii) plot the two losses as functions of $yf$ on one axis. Explain in 3–5 sentences what the counts say about which
   points each model listens to.
3. **Think:** You must ship a classifier for 2 million customer records with 40 numeric features, retrained nightly, with
   prediction latency under 5 ms. A colleague's notebook shows an RBF SVM beating logistic regression by 2 points of
   accuracy on a 20,000-row sample. Decide whether to ship the SVM. Address training cost, prediction cost, whether the
   2 points are real, calibration, and the alternatives you would benchmark.

## Takeaways

1. An SVM chooses the separating hyperplane with the widest margin, $2/\lVert\mathbf{w}\rVert$; only the support vectors (points on or inside the margin)
   determine it.
2. The soft-margin SVM is L2-regularized hinge loss: $\frac{\lambda}{2}\lVert\mathbf{w}\rVert^2 + \frac1n\sum\max(0, 1 - yf)$ with $\lambda = 1/(Cn)$. Large $C$ means a narrow margin and a close fit; small $C$ a
   wide margin and more tolerance.
3. The hinge is exactly zero for confidently correct points, so the SVM ignores them. It gives scores, not probabilities, and its kinks
   make plain sub-gradient descent slow (about $1/t$).
4. The dual writes $f(\mathbf{x}) = \sum_{\text{SV}}\alpha_iy_i\mathbf{x}_i^\top\mathbf{x} + b$. Swapping the dot product for a kernel $k(\mathbf{x}_i, \mathbf{x})$ gives non-linear boundaries without computing features;
   RBF's $\gamma$ sets the bump width.
5. Scale features; tune $C$ and $\gamma$ together on a log grid; mind $O(n^2)$ training; read `LinearSVC`'s objective (squared hinge,
   penalized intercept) before comparing.

**What builds on this:** [B3.3](b3_3_gradient_boosting.md) (the usual winner on large tabular data);
[B4.1](b4_1_classification_metrics.md) (calibrating scores into probabilities); [B4.3](b4_3_cross_validation_tuning.md)
(tuning $C$ and $\gamma$); [B5.6](b5_6_anomaly_detection.md) (one-class SVM for anomalies); [C1.1](../C/c1_1_perceptron_mlp.md)
(the perceptron, the SVM's ancestor); [F1.1](../F/f1_1_self_attention.md) (attention weights as normalized kernel similarities).

**References:** C. Cortes, V. Vapnik, "Support-Vector Networks", *Machine Learning* 20, 1995. B. Boser, I. Guyon, V. Vapnik,
"A Training Algorithm for Optimal Margin Classifiers", *COLT*, 1992. S. Shalev-Shwartz, Y. Singer, N. Srebro, A. Cotter,
"Pegasos: Primal Estimated Sub-Gradient Solver for SVM", *Mathematical Programming* 127, 2011. C.-C. Chang, C.-J. Lin,
"LIBSVM: A Library for Support Vector Machines", *ACM TIST* 2(3), 2011. scikit-learn User Guide, "Support Vector Machines"
(VERIFY link: https://scikit-learn.org/stable/modules/svm.html).
