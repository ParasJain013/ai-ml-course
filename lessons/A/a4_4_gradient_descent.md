# A4.4 Gradient descent

**Core** · Part A, week 4 (course week 04) · ~2.5 h · Lab:
[`labs/A/a4_4_gradient_descent.py`](../../labs/A/a4_4_gradient_descent.py)

## Why it matters

Linear regression has a formula for its best weights ([A2.4](a2_4_projections_least_squares.md)).
Almost nothing else does: logistic regression, neural networks and transformers are trained by
starting from a guess and repeatedly stepping downhill along the negative gradient. That loop,
**gradient descent**, is the engine under every `optimizer.step()` you will ever call. Its one
critical setting, the **learning rate**, decides whether training converges, crawls for days, or
blows up into `nan`, and the reasons are not mysterious: they follow from the eigenvalues of the
loss's curvature, the same eigenvalues as in [A3.1](a3_1_eigenvalues_eigenvectors.md). This lesson fits
a line by gradient descent, matches the normal-equation answer to 4 decimals, and explains every
failure you will see in a loss curve.

## Prerequisite check

- Can you write the gradient of the MSE, $\frac{2}{n}\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$, and say why it
  is zero at the least-squares solution? If not, revisit [A4.2](a4_2_gradients_jacobians.md) and
  [A2.4](a2_4_projections_least_squares.md).
- Can you say why the gradient points uphill and what $-\nabla\mathcal{L}$ is? If not, revisit
  [A4.2](a4_2_gradients_jacobians.md).
- Can you find the eigenvalues of a symmetric 2×2 matrix and say what $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$
  means? If not, revisit [A3.1](a3_1_eigenvalues_eigenvectors.md).
- Do you know what standardizing a feature does (mean 0, standard deviation 1)? If not, revisit
  [A1.2](a1_2_numpy_arrays.md).

## Intuition

### Walking downhill in fog

You are on a hillside in thick fog and want to reach the bottom of the valley. You cannot see the
valley; you can only feel the slope under your feet. The sensible plan: feel which way is steepest
down, take a step that way, repeat. That is gradient descent:

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\,\nabla\mathcal{L}(\boldsymbol{\theta}).$$

The **learning rate** $\eta$ is your stride. With tiny strides you get there, eventually. With huge
strides you overshoot the valley floor and land higher up the opposite slope; if each overshoot is
bigger than the last, you climb out of the valley altogether and the loss explodes. Near the bottom
the slope flattens, so steps shrink on their own even with a fixed $\eta$, and the walk stops when the
gradient is (nearly) zero.

### Long, thin valleys are slow

If the valley is a round bowl, "steepest down" points at the bottom and you walk straight there. Most
real valleys are long and thin: steep walls, a gently sloping floor. Steepest down then points mostly
*across* the valley, towards the nearer wall ([A4.2](a4_2_gradients_jacobians.md)'s key plot). A stride
small enough not to bounce out of the steep direction is tiny compared with the long distance along
the floor, so you zig-zag across and crawl along. How long and thin the valley is, measured by the
ratio of its steepest to its gentlest curvature, decides how many steps you need. Rescaling the
features to similar ranges makes the valley rounder, which is why feature scaling speeds up training.

### Estimating the slope from a sample

Computing the exact gradient touches every training example. With a million examples that is a lot
of work for one step. **Stochastic gradient descent (SGD)** estimates the gradient from a small random
**mini-batch** of examples instead: a noisy but unbiased estimate, many times cheaper. You take many
more, wobblier steps, and the total work to get close is far lower. One pass through all the data is
an **epoch**.

> **Bridge:** a Kubernetes HorizontalPodAutoscaler adjusts replicas in proportion to how far a
> metric is from its target. Too aggressive a setting and it overshoots, scales down, overshoots
> again (flapping); too timid and it lags behind the load. A learning rate behaves the same way:
> too large oscillates or diverges, too small lags. **Where it breaks:** the autoscaler knows its
> target value; gradient descent does not know the minimum loss, only the local slope. And an
> autoscaler's metric does not change shape under it, while a loss surface can have many valleys
> ([A4.5](a4_5_convexity_lagrange.md)).

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\boldsymbol{\theta}$, $\mathbf{w}$ | parameters (here the weights of linear regression) | `(d,)` |
| $\mathbf{w}_k$ | the weights after $k$ steps; $\mathbf{w}_0$ is the starting guess | `(d,)` |
| $\mathbf{w}^\star$ | the minimizer (here the normal-equation solution) | `(d,)` |
| $\mathbf{e}_k$ | error $\mathbf{w}_k - \mathbf{w}^\star$ | `(d,)` |
| $\eta$ | learning rate | scalar, $> 0$ |
| $\mathcal{L}$ | MSE $\frac{1}{n}\lVert\mathbf{X}\mathbf{w} - \mathbf{y}\rVert^2$ | scalar |
| $\mathbf{H}$ | **Hessian**: matrix of second partial derivatives, $H_{jk} = \frac{\partial^2\mathcal{L}}{\partial w_j\,\partial w_k}$ | `(d, d)` |
| $\lambda_j$, $\mathbf{q}_j$ | eigenvalues and orthonormal eigenvectors of $\mathbf{H}$ (not the regularization $\lambda$) | scalar, `(d,)` |
| $\kappa$ | **condition number** of $\mathbf{H}$, $\lambda_{\max}/\lambda_{\min}$ | scalar, $\ge 1$ |
| $B$, $\mathcal{B}$ | batch size, and the set of row indices in one mini-batch | int, set |

**Why a small step goes downhill.** By the A4.2 linear approximation with the step
$\boldsymbol{\delta} = -\eta\nabla\mathcal{L}$:

$$\mathcal{L}(\mathbf{w} - \eta\nabla\mathcal{L}) \approx \mathcal{L}(\mathbf{w}) - \eta\,\nabla\mathcal{L}\cdot\nabla\mathcal{L} = \mathcal{L}(\mathbf{w}) - \eta\,\lVert\nabla\mathcal{L}\rVert^2.$$

The loss falls by about $\eta\lVert\nabla\mathcal{L}\rVert^2 > 0$ unless the gradient is already zero. "About"
is the catch: the approximation ignores curvature, and with a large $\eta$ the curvature term wins.
The rest of this section measures exactly when.

**One parameter, a perfect bowl.** Take $\mathcal{L}(w) = \frac{a}{2}(w - w^\star)^2$ with curvature $a > 0$
(its second derivative). The gradient is $a(w - w^\star)$, so one step gives

$$w_{k+1} - w^\star = w_k - w^\star - \eta\,a\,(w_k - w^\star) = (1 - \eta a)(w_k - w^\star).$$

Every step multiplies the error by the same factor $1 - \eta a$. Four regimes follow:

| $\eta$ | factor $1 - \eta a$ | behaviour |
|---|---|---|
| $0 < \eta < 1/a$ | between 0 and 1 | smooth, monotone approach |
| $\eta = 1/a$ | 0 | lands on $w^\star$ in one step |
| $1/a < \eta < 2/a$ | between $-1$ and 0 | overshoots, oscillates, still converges |
| $\eta > 2/a$ | below $-1$ | each overshoot is bigger: **diverges** |

So the largest stable learning rate is $2/a$: twice the reciprocal of the curvature.

**The MSE is a bowl too, with $d$ curvatures.** The Hessian is the Jacobian of the gradient. The MSE
gradient $\frac{2}{n}\mathbf{X}^\top\mathbf{X}\mathbf{w} - \frac{2}{n}\mathbf{X}^\top\mathbf{y}$ is a linear map of $\mathbf{w}$ plus a
constant, and a linear map's Jacobian is its matrix (A4.2), so

$$\mathbf{H} = \frac{2}{n}\mathbf{X}^\top\mathbf{X}, \quad\text{the same at every } \mathbf{w}.$$

It is symmetric and positive semi-definite ($\mathbf{v}^\top\mathbf{H}\mathbf{v} = \frac{2}{n}\lVert\mathbf{X}\mathbf{v}\rVert^2 \ge 0$, as for
covariance matrices in [A3.3](a3_3_covariance_matrices.md)), so its eigenvalues are real and
$\ge 0$. At the minimum the normal equations say $\frac{2}{n}\mathbf{X}^\top\mathbf{y} = \frac{2}{n}\mathbf{X}^\top\mathbf{X}\mathbf{w}^\star = \mathbf{H}\mathbf{w}^\star$;
substitute that into the gradient:

$$\nabla\mathcal{L}(\mathbf{w}) = \mathbf{H}\mathbf{w} - \mathbf{H}\mathbf{w}^\star = \mathbf{H}(\mathbf{w} - \mathbf{w}^\star) \quad\Longrightarrow\quad \mathbf{e}_{k+1} = \mathbf{e}_k - \eta\mathbf{H}\mathbf{e}_k = (\mathbf{I} - \eta\mathbf{H})\,\mathbf{e}_k.$$

The one-parameter factor $1 - \eta a$ has become a matrix. Use the eigenvectors of $\mathbf{H}$ (spectral
theorem, A3.1): write the error as $\mathbf{e}_k = \sum_j c_j\mathbf{q}_j$. Since
$(\mathbf{I} - \eta\mathbf{H})\mathbf{q}_j = (1 - \eta\lambda_j)\mathbf{q}_j$, each coordinate evolves on its own, exactly like
the one-parameter bowl:

$$c_j \leftarrow (1 - \eta\lambda_j)\,c_j \qquad\text{for each } j = 1, \dots, d.$$

The valley has one steepness per eigen-direction. Everything about gradient descent on a quadratic
follows from this line:

1. **Stability** needs $\lvert 1 - \eta\lambda_j\rvert < 1$ for *every* $j$, so $\eta < 2/\lambda_{\max}$. The
   steepest direction sets the speed limit.
2. **Speed** is set by the slowest direction. With $\eta < 2/\lambda_{\max}$, the factor for
   $\lambda_{\min}$ is at best about $1 - 2\lambda_{\min}/\lambda_{\max}$. The best fixed learning rate balances
   the two extremes: raising $\eta$ helps the flattest direction ($1 - \eta\lambda_{\min}$ falls) and hurts the
   steepest ($1 - \eta\lambda_{\max}$ heads towards $-1$), so the best choice makes them equal in size,
   $1 - \eta\lambda_{\min} = -(1 - \eta\lambda_{\max})$. Solving, $\eta^\star = \frac{2}{\lambda_{\min} + \lambda_{\max}}$, and both
   factors have size $1 - \frac{2\lambda_{\min}}{\lambda_{\min} + \lambda_{\max}} = \frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\max} + \lambda_{\min}} = \frac{\kappa - 1}{\kappa + 1}$
   (divide top and bottom by $\lambda_{\min}$).
3. **Steps needed** to shrink the error by a factor $F$: solve $\big(\frac{\kappa - 1}{\kappa + 1}\big)^k = 1/F$ by
   taking logs, $k = \ln F / \ln\frac{\kappa + 1}{\kappa - 1}$. For large $\kappa$,
   $\frac{\kappa + 1}{\kappa - 1} = 1 + \frac{2}{\kappa - 1}$ and $\ln(1 + x) \approx x$ for small $x$, so
   $k \approx \frac{\kappa}{2}\ln F$. **The step count grows linearly with the condition number.**
4. **Zig-zag:** with $\eta$ near $2/\lambda_{\max}$ the steep direction's factor is near $-1$, so that
   coordinate flips sign every step (across the valley) while the gentle one shrinks slowly (along it).

**Why scaling features helps.** For a design matrix $[\mathbf{1}, \mathbf{x}]$ with $x$ between 0 and 10,
$\mathbf{X}^\top\mathbf{X}$ has entries $n$, $\sum x_i$, $\sum x_i^2$ of very different sizes and the two columns
point in similar directions, so $\kappa$ is large. Centering $x$ makes the columns orthogonal; dividing by
the standard deviation gives them equal length. Then $\mathbf{H} = \frac{2}{n}\mathbf{X}^\top\mathbf{X} = 2\mathbf{I}$: $\kappa = 1$,
and gradient descent with $\eta = 1/2$ lands on the answer in one step. With more features, standardizing
does not make $\kappa = 1$ (correlated features still squash the bowl), but it removes the part of $\kappa$
that came from units.

**When to stop.** Three tests, usually combined: the gradient's length below a tolerance (the
minimum has $\nabla\mathcal{L} = \mathbf{0}$); the loss improving by less than a tolerance; and a maximum number of
steps, after which you raise or log rather than silently return.

**Stochastic gradients.** The MSE is an average of per-example losses, so its gradient is the average
of per-example gradients: $\nabla\mathcal{L} = \frac{1}{n}\sum_i\nabla\ell_i$ with $\ell_i = (\mathbf{x}_i^\top\mathbf{w} - y_i)^2$. A
mini-batch $\mathcal{B}$ of $B$ random rows gives the estimate $\mathbf{g} = \frac{1}{B}\sum_{i\in\mathcal{B}}\nabla\ell_i$, and in
code it is the same function applied to `X[idx], y[idx]`. Each example is equally likely to be in the
batch, so on average $\mathbf{g}$ equals $\nabla\mathcal{L}$ (it is **unbiased**), but any one batch is off by
random noise. Two consequences:

- With a constant $\eta$, SGD never settles: at $\mathbf{w}^\star$ the full gradient is zero but a batch's
  gradient is not, so the weights keep jittering in a cloud around $\mathbf{w}^\star$ whose size grows with
  $\eta$ and shrinks with $B$.
- To converge, shrink $\eta$ over time, for example $\eta_t = \eta_0/(1 + t/T)$. (The classical
  condition is that the step sizes add up to infinity while their squares add up to something
  finite; stated, not proved.) Shuffling the rows every epoch keeps batches from repeating a fixed,
  possibly sorted, order.

## Worked example by hand

**1. One parameter.** $\mathcal{L}(w) = (w - 3)^2$, so $a = 2$ (the $\frac{a}{2}$ form with $\frac{a}{2} = 1$), gradient
$2(w - 3)$, largest stable $\eta = 2/a = 1$. Start at $w_0 = 0$ (error $-3$).

- $\eta = 0.1$: factor $1 - 0.2 = 0.8$. Errors $-2.4, -1.92, -1.536, -1.2288$, so
  $w = 0.6, 1.08, 1.464, 1.7712$. Check the first: $w_1 = 0 - 0.1 \times 2(0 - 3) = 0.6$ ✓.
- $\eta = 0.5$: factor 0. $w_1 = 0 - 0.5 \times (-6) = 3$: done in one step.
- $\eta = 0.9$: factor $-0.8$. $w_1 = 0 + 5.4 = 5.4$ (overshoots to error $+2.4$), then $1.08$, $4.536$,
  $1.7712$: the same distances as $\eta = 0.1$, alternating sides.
- $\eta = 1.1$: factor $-1.2$. Errors $3.6, -4.32, 5.184, -6.2208$: $w = 6.6, -1.32, 8.184, -3.2208$. Diverging.

**2. The four-point line of A2.4.** $\mathbf{X} = \begin{pmatrix}1&0\\1&1\\1&2\\1&3\end{pmatrix}$ `(4, 2)`,
$\mathbf{y} = (1, 3, 4, 4)$; least squares gives $\mathbf{w}^\star = (1.5, 1.0)$. Start at $\mathbf{w}_0 = (0, 0)$, $\eta = 0.1$.

- Residuals $\mathbf{X}\mathbf{w}_0 - \mathbf{y} = (-1, -3, -4, -4)$; loss $\frac{1 + 9 + 16 + 16}{4} = 10.5$.
- $\mathbf{X}^\top\mathbf{r}$: the ones column gives $-1 - 3 - 4 - 4 = -12$; the $x$ column gives
  $0 - 3 - 8 - 12 = -23$. Gradient $\frac{2}{4}(-12, -23) = (-6, -11.5)$.
- Step: $\mathbf{w}_1 = (0, 0) - 0.1(-6, -11.5) = (0.6, 1.15)$. Predictions $(0.6, 1.75, 2.9, 4.05)$,
  residuals $(-0.4, -1.25, -1.1, 0.05)$, loss $\frac{0.16 + 1.5625 + 1.21 + 0.0025}{4} = 0.73375$. One step cut the
  loss from 10.5 to 0.73.
- Hessian $\mathbf{H} = \frac{2}{4}\mathbf{X}^\top\mathbf{X} = \frac12\begin{pmatrix}4&6\\6&14\end{pmatrix} = \begin{pmatrix}2&3\\3&7\end{pmatrix}$.
  Trace 9, determinant $14 - 9 = 5$, so $\lambda = \frac{9 \pm\sqrt{81 - 20}}{2} = \frac{9 \pm 7.8102}{2}$:
  $\lambda_{\max} = 8.4051$, $\lambda_{\min} = 0.5949$ (A3.1).
- Largest stable $\eta = 2/8.4051 = 0.2380$, so $\eta = 0.1$ is safe. $\kappa = 14.13$. The slow
  direction's factor is $1 - 0.1 \times 0.5949 = 0.9405$ per step: after 100 steps
  $0.9405^{100} \approx 0.0022$ of the original error remains in that direction. The starting error
  $(-1.5, -1)$ has a component of about $0.94$ along the slow eigenvector, and $0.94 \times 0.0022 \approx 0.002$:
  the distance the lab finds after 100 steps. The fast direction's factor, $1 - 0.84 = 0.16$, wiped out its
  component within a few steps.

```text
== worked example: one parameter ==
lr = 0.1  factor (1 - 2 lr) = 0.8   w after 1..4 steps = [0.6, 1.08, 1.464, 1.7712]
lr = 0.5  factor (1 - 2 lr) = 0.0   w after 1..4 steps = [3.0, 3.0, 3.0, 3.0]
lr = 0.9  factor (1 - 2 lr) = -0.8  w after 1..4 steps = [5.4, 1.08, 4.536, 1.7712]
lr = 1.1  factor (1 - 2 lr) = -1.2  w after 1..4 steps = [6.6, -1.32, 8.184, -3.2208]

== worked example: four points ==
loss at w = 0: 10.50000   gradient = [-6.0, -11.5]
after one step with lr = 0.1: w = [0.6, 1.15]   loss = 0.73375
Hessian (2/n) X^T X = [[2.0, 3.0], [3.0, 7.0]]   eigenvalues = [0.5949, 8.4051]
largest stable lr 2/lambda_max = 0.2380   condition number = 14.13
after  10 steps: w = [1.0418, 1.2146]   distance to (1.5, 1.0) = 0.505940
after 100 steps: w = [1.4982, 1.0009]   distance to (1.5, 1.0) = 0.002027
after 300 steps: w = [1.5, 1.0]   distance to (1.5, 1.0) = 0.000000
```

## Key plot

![Left: excess loss against step on a log scale for three learning rates; the too-small rate flattens near 0.4, the best fixed rate falls in a straight line to about 1e-10 after 1000 steps, and the too-large rate shoots off the top of the chart within a few dozen steps. Right: contour ellipses of the raw-feature MSE, long and thin, with a gradient-descent path that bounces across the valley for a few steps and then creeps along its floor towards the minimum](img/a4_4_learning_rates.png)

Left: the A2.4 line fit ($\kappa = 161$) with three learning rates. The best fixed rate is a straight
line on the log scale: the error shrinks by the same factor every step, as the math predicts. Too
small looks fine for 50 steps, then stalls: it fixed the steep direction and crawls along the gentle
one. Just 5% above $2/\lambda_{\max}$ diverges. Right: with $\eta = 0.8 \times$ the maximum, the steep
coordinate's factor is $1 - 1.6 = -0.6$: a few bounces across the valley that die out, then a long creep
along its floor.

## From scratch in NumPy

The data is A2.4's, regenerated with the same seed: 50 points around $y = 2 + 0.5x$, $x$ between 0 and 10,
with normal-equation weights $(1.7575, 0.5523)$. Batch gradient descent with the three stopping tests:

```python
def gradient_descent(
    X: FloatArray, y: FloatArray, lr: float, max_iters: int = MAX_ITERS, tol: float = GRAD_TOL
) -> tuple[FloatArray, list[float], int]:
    """Batch GD on the MSE from w = 0. X (n, d), y (n,) -> (w (d,), loss per step, iterations)."""
    w = np.zeros(X.shape[1])  # (d,)
    losses = [mse(w, X, y)]
    for k in range(1, max_iters + 1):
        g = mse_grad(w, X, y)  # (d,)
        if np.linalg.norm(g) < tol:
            return w, losses, k - 1
        w = w - lr * g
        losses.append(mse(w, X, y))
        if not np.isfinite(losses[-1]):
            raise FloatingPointError(f"diverged at step {k}: lr = {lr} is too large")
    raise RuntimeError(f"no convergence in {max_iters} steps (|grad| = {np.linalg.norm(g):.2e})")
```

The lab computes the Hessian's eigenvalues first and picks $\eta^\star = 2/(\lambda_{\min} + \lambda_{\max})$:

```text
== from scratch: batch gradient descent ==
Hessian eigenvalues = [0.4589, 74.0413]   condition number = 161.3
largest stable lr = 0.0270   best fixed lr = 0.0268
GD with lr = 0.0268: 1816 steps   w = [1.7575, 0.5523]   normal equations w = [1.7575, 0.5523]
GD matches the normal equations to 4 decimals: OK
|grad| at w = 0: 59.3   rate (kappa - 1)/(kappa + 1) = 0.9877   predicted steps to reach |grad| < 1e-8: about 1815
```

Gradient descent reaches the normal-equation weights (the check uses `atol=1e-6`, well inside 4
decimals), and the eigenvalue theory predicted the step count to within one. The gradient is
$\mathbf{H}\mathbf{e}_k$, so it shrinks at the same rate as the error.

Now standardize $x$ (subtract its mean $\mu$, divide by its standard deviation $s$) and run again:

```text
== from scratch: standardizing the feature ==
Hessian eigenvalues = [2.0, 2.0]   condition number = 1.0
GD with lr = 0.5000: 1 steps   w (standardized) = [4.6674, 1.6098]
converted back to the raw x: w = [1.7575, 0.5523]   matches normal equations: True
```

From 1,816 steps to 1. The model $\hat{y} = a + b\,\frac{x - \mu}{s}$ is the same line as
$\hat{y} = (a - b\mu/s) + (b/s)\,x$, which is how the lab converts back.

Mini-batch SGD shuffles the rows every epoch and steps on each batch of 10 (on the standardized
feature, starting from $\eta_0 = 0.5$):

```python
for epoch in range(epochs):
    lr = lr0 / (1 + epoch / 10) if decay else lr0
    order = gen.permutation(len(y))  # a fresh shuffle every epoch
    for start in range(0, len(y), batch_size):
        idx = order[start : start + batch_size]  # (batch_size,) row indices
        w = w - lr * mse_grad(w, X[idx], y[idx])  # gradient on the batch only
```

```text
== from scratch: mini-batch SGD ==
SGD, batch 10, 200 epochs, constant lr: w = [2.0198, 0.4889]   distance to normal equations = 2.7e-01   excess loss = 3.9e-02
SGD, batch 10, 200 epochs, decaying lr: w = [1.7516, 0.5527]   distance to normal equations = 5.9e-03   excess loss = 1.5e-05
```

With a constant learning rate SGD is still wandering after 1,000 updates, 0.27 away from the answer:
the noise cloud. Decaying the rate shrinks the cloud and gets 45 times closer. Neither matches to 4
decimals, and that is normal: SGD buys cheap steps with noise. (The "excess loss" of 0.039 is small
compared with the minimum MSE of 0.99, which is why SGD's answer is good enough for prediction.)

## With SciPy and scikit-learn

```text
== with scipy and scikit-learn ==
scipy.optimize.minimize(BFGS): w = [1.7575, 0.5523]   9 iterations   matches normal equations: OK
LinearRegression: w = [1.7575, 0.5523]
SGDRegressor (standardized x): w = [1.7574, 0.5524]   distance to normal equations = 9.2e-05
SGDRegressor agrees with the normal equations to about 3 decimals: OK
```

- `scipy.optimize.minimize(fun, x0, jac=grad, method="BFGS")` takes your loss and gradient and gets
  there in 9 iterations on the *raw* feature. BFGS builds an estimate of the Hessian from successive
  gradients and uses it to undo the valley's stretch, so the condition number hurts it far less. That
  is why classical models are fit with such methods ([B2.1](../B/b2_1_logistic_regression.md)), while
  deep networks, with millions of parameters and noisy mini-batch gradients, use SGD and its relatives
  ([C3.1](../C/c3_1_optimizers.md)).
- `LinearRegression` solves least squares directly (A2.4).
- `SGDRegressor` is mini-batch SGD with $B = 1$ and a decaying learning-rate schedule by default
  (`learning_rate="invscaling"`, `eta0=0.01`). Two defaults to know (VERIFY (as of 2026-10) for
  scikit-learn 1.9.1): it adds an L2 penalty unless `penalty=None`, and it stops early based on `tol`
  unless `tol=None`. Its per-example loss is $\frac12(y - \hat{y})^2$, half of ours, so its learning rates
  are not directly comparable with the lab's. Always give it standardized features.

## In practice

- **When to use it:** plain gradient descent to understand training and for small smooth problems;
  mini-batch SGD (and its descendants such as Adam, [C3.1](../C/c3_1_optimizers.md)) for anything
  large; a library solver (closed form, BFGS, L-BFGS) when the problem is small enough and convex.
- **Failure modes:**
  - *Learning rate too large:* the loss oscillates, then grows, then becomes `inf` or `nan`. You notice
    it in the first few dozen steps; divide $\eta$ by 3–10.
  - *Learning rate too small:* the loss falls, then flattens well above where it should be. You notice
    a curve that is still sloping down when training ends; multiply $\eta$ by 3.
  - *Unscaled features:* one feature in thousands, another in fractions, gives a huge $\kappa$, so the
    stable $\eta$ is tiny for the small-scale direction. You notice very slow convergence at the largest
    $\eta$ that does not diverge; standardize (fit the scaler on training data only,
    [B1.4](../B/b1_4_data_leakage.md)).
  - *Expecting SGD to match exactly:* with a constant $\eta$ it hovers. You notice the loss curve going
    flat but jittery; decay the learning rate, or average the last iterates.
  - *Unshuffled data:* batches sorted by date or label give biased, correlated gradients. You notice a
    sawtooth loss curve repeating every epoch.
- **Defaults that work:** standardize features; try learning rates on a log grid (0.001, 0.003, 0.01,
  0.03, 0.1, …) for a few hundred steps and keep the largest that decreases smoothly; plot the loss on a
  log scale; stop on gradient norm or loss change with a step cap; shuffle every epoch; batch sizes of 32
  to 256 for SGD.

## Exercises

Save your work to `my-work/A/a4_4_gradient_descent.md` (or `.py`) and ask for a review.

1. **By hand:** (a) For $\mathcal{L}(w) = 2(w + 1)^2$, find the curvature $a$, the range of learning rates that
   converge, and the one that converges in a single step. Starting from $w_0 = 1$, compute three steps
   with $\eta = 0.1$ and three with $\eta = 0.4$, and describe the difference. (b) For
   $\mathcal{L}(\mathbf{w}) = \frac12\mathbf{w}^\top\mathbf{H}\mathbf{w}$ with $\mathbf{H} = \operatorname{diag}(4, 1)$, find the largest stable learning
   rate, the best fixed learning rate and its per-step factor, and how many steps that rate needs to
   shrink the error by a factor of $10^6$. (c) Explain in two sentences why the answer to (b) gets
   worse as the 4 is replaced by 400.
2. **Code:** Generate $n = 40$ points with $x$ uniform on $[-3, 4]$ and $y = 1 - 2x + 0.5x^2$ plus
   Gaussian noise with standard deviation 1 (`SEED = 0`, draw $x$ first, then the noise). Fit
   $\hat{y} = w_0 + w_1x + w_2x^2$ by batch gradient descent: (a) on the raw columns $[1, x, x^2]$, and
   (b) with $x$ and $x^2$ each standardized. For each, compute the Hessian's condition number, use the best
   fixed learning rate, and report the number of steps to reach $\lVert\nabla\mathcal{L}\rVert < 10^{-8}$. Convert
   (b)'s weights back to the raw scale and check both against `np.linalg.lstsq` to 4 decimals. Is
   $\kappa = 1$ after standardizing? Explain why not.
3. **Think:** A nightly job retrains a linear model by gradient descent with $\eta = 0.01$, tuned a year
   ago. Yesterday someone added a new feature, a price in cents (values around 50,000), without
   scaling it; today the job fails with `nan` weights. Explain the failure with this lesson's
   eigenvalue argument. What two changes would you make so that the next unscaled feature cannot break
   training silently or loudly?

## Takeaways

1. Gradient descent repeats $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\nabla\mathcal{L}$; a small enough step lowers the loss
   by about $\eta\lVert\nabla\mathcal{L}\rVert^2$.
2. On a quadratic, each Hessian eigen-direction's error is multiplied by $1 - \eta\lambda_j$ per step:
   stable only if $\eta < 2/\lambda_{\max}$, and as slow as the flattest direction.
3. Steps needed grow linearly with the condition number $\kappa = \lambda_{\max}/\lambda_{\min}$; standardizing
   features shrinks $\kappa$ (here from 161 to 1, and 1,816 steps to 1).
4. SGD uses unbiased but noisy mini-batch gradients: far cheaper per step, never settles with a
   constant rate, converges with a decaying one. Shuffle every epoch.
5. Read loss curves: exploding means $\eta$ too large, flattening too early means too small or badly
   scaled features.

**What builds on this:** [A4.5](a4_5_convexity_lagrange.md) (when gradient descent is guaranteed to find
the best answer); [A5.5](a5_5_mle_map.md) (fitting distributions by gradient descent on the negative
log-likelihood); [B1.5](../B/b1_5_linear_regression.md) and [B2.1](../B/b2_1_logistic_regression.md)
(training real models); [C3.1](../C/c3_1_optimizers.md) (momentum and Adam fix the long-valley problem);
[C3.2](../C/c3_2_lr_schedules.md) (learning-rate schedules).
