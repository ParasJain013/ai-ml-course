# A4.4 Gradient descent

**Core** · Part A, week 4 (course week 04) · ~2.5 h · Lab:
[`labs/A/a4_4_gradient_descent.py`](../../labs/A/a4_4_gradient_descent.py)

## Why it matters

Linear regression has a formula for its best weights ([A2.4](a2_4_projections_least_squares.md));
almost nothing else does. Logistic regression, neural networks and every model after them are trained
by **gradient descent**: start somewhere, compute the gradient of the loss, take a small step against
it, repeat. One number, the **learning rate**, decides whether that loop converges in minutes, crawls
for days, or blows up into NaNs, and the scale of your input features decides how forgiving it is.
This lesson fits A2.4's line by gradient descent, matches the normal-equation weights to 4 decimal
places, and explains every one of those failure modes with the eigenvalues you met in week 3.

## Prerequisite check

- Can you write the MSE gradient $\frac{2}{n}\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$ and its Hessian
  $\frac{2}{n}\mathbf{X}^\top\mathbf{X}$ with their shapes? If not, revisit [A4.2](a4_2_gradients_jacobians.md).
- Can you fit a line with the normal equations and say what the residuals look like at the optimum? If
  not, revisit [A2.4](a2_4_projections_least_squares.md).
- Can you compute the eigenvalues of a symmetric 2×2 matrix from its trace and determinant? If not,
  revisit [A3.1](a3_1_eigenvalues_eigenvectors.md).
- Do you know what a condition number is? If not, revisit [A3.4](a3_4_numerical_stability.md).

## Intuition

### Walking downhill in fog

You are on a hillside in thick fog and want the bottom of the valley. You cannot see it, but you can
feel the slope under your feet. So: feel which way is steepest downhill (that is $-\nabla\mathcal{L}$,
[A4.2](a4_2_gradients_jacobians.md)), take a step that way, and repeat. The step length is the learning
rate $\eta$. Tiny steps are safe but you will be walking all day. Huge steps overshoot the valley floor
and land higher up the opposite side, and if each overshoot is bigger than the last, you leave the
valley entirely.

### Narrow valleys make it zig-zag

If the valley is a round bowl, "steepest downhill" points at the bottom and a few good steps get you
there. If it is a long narrow ravine (steep walls, gentle floor), steepest downhill points mostly
across the ravine, not along it. A step big enough to make progress along the floor overshoots the
walls; a step small enough for the walls barely moves you along the floor. The key plot shows exactly
this on A2.4's data. The cure is not a cleverer step but a rounder valley: **standardizing** the
features.

> **Bridge:** gradient descent is a proportional controller, like a Kubernetes Horizontal Pod
> Autoscaler that adds replicas in proportion to how far CPU is from its target. The learning rate is
> the gain: too low and the system lags for ages; too high and it overshoots, oscillates, and with
> enough gain the oscillation grows until something breaks. **Where it breaks:** the autoscaler chases
> a moving target with one noisy measurement; batch gradient descent works on a *fixed* loss surface
> with an exact gradient in every direction at once, so its behaviour can be predicted exactly from the
> surface's curvature, as below. (Mini-batch SGD brings the noise back.)

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\boldsymbol{\theta}$, $\mathbf{w}$ | parameters; for linear regression the weights (intercept first, via a ones column) | `(d,)` |
| $\mathbf{w}_k$ | the weights after $k$ steps | `(d,)` |
| $\mathbf{w}^\star$ | the minimizer (here: the normal-equation answer) | `(d,)` |
| $\eta$ | **learning rate** (step size) | scalar |
| $\mathcal{L}$ | loss, here the MSE $\frac{1}{n}\lVert\mathbf{X}\mathbf{w} - \mathbf{y}\rVert^2$ | scalar |
| $\mathbf{g}$ | the gradient $\nabla\mathcal{L}(\mathbf{w}_k)$ | `(d,)` |
| $\mathbf{H}$ | Hessian of the MSE, $\frac{2}{n}\mathbf{X}^\top\mathbf{X}$ | `(d, d)` |
| $\lambda_{\min}$, $\lambda_{\max}$ | smallest and largest eigenvalue of $\mathbf{H}$ (curvatures; not the regularization $\lambda$) | scalars |
| $\kappa$ | condition number of $\mathbf{H}$, $\lambda_{\max}/\lambda_{\min}$ | scalar |
| $a$ | curvature of a one-dimensional quadratic | scalar |
| $B$ | **batch size** of mini-batch SGD | int |

**The update.**

$$\mathbf{w}_{k+1} = \mathbf{w}_k - \eta\,\nabla\mathcal{L}(\mathbf{w}_k).$$

**Why it goes down (for small enough $\eta$).** The linear approximation of A4.2 with the step
$\boldsymbol{\delta} = -\eta\mathbf{g}$ gives

$$\mathcal{L}(\mathbf{w}_k - \eta\mathbf{g}) \approx \mathcal{L}(\mathbf{w}_k) + \mathbf{g}\cdot(-\eta\mathbf{g}) = \mathcal{L}(\mathbf{w}_k) - \eta\lVert\mathbf{g}\rVert^2,$$

which is lower whenever $\mathbf{g} \ne \mathbf{0}$. The approximation ignores curvature, and curvature is what
punishes a large $\eta$. To see exactly how, look at a quadratic, where nothing is approximated.

**One dimension.** Take $\mathcal{L}(w) = \frac{a}{2}(w - w^\star)^2$ with curvature $a > 0$ (its second
derivative). Its derivative is $a(w - w^\star)$, so one step gives

$$w_{k+1} - w^\star = w_k - \eta a(w_k - w^\star) - w^\star = (1 - \eta a)(w_k - w^\star).$$

The distance to the minimum is multiplied by the same factor $1 - \eta a$ every step. Four regimes:

| Learning rate | Factor $1 - \eta a$ | Behaviour |
|---|---|---|
| $0 < \eta < 1/a$ | between 0 and 1 | slides monotonically toward $w^\star$ (slowly if $\eta a$ is small) |
| $\eta = 1/a$ | 0 | lands on $w^\star$ in one step |
| $1/a < \eta < 2/a$ | between $-1$ and 0 | overshoots, alternating sides, but converges |
| $\eta > 2/a$ | below $-1$ | overshoots further each time: **diverges** |

So the largest stable learning rate is $2/a$: twice the reciprocal of the curvature.

**Linear regression is exactly such a quadratic, in $d$ directions.** The MSE gradient (A4.2) is
$\mathbf{H}\mathbf{w} - \frac{2}{n}\mathbf{X}^\top\mathbf{y}$, and the normal equations say $\mathbf{H}\mathbf{w}^\star = \frac{2}{n}\mathbf{X}^\top\mathbf{y}$
(multiply $\mathbf{X}^\top\mathbf{X}\mathbf{w}^\star = \mathbf{X}^\top\mathbf{y}$ by $\frac{2}{n}$). Subtract: $\nabla\mathcal{L}(\mathbf{w}) = \mathbf{H}(\mathbf{w} - \mathbf{w}^\star)$.
One step:

$$\mathbf{w}_{k+1} - \mathbf{w}^\star = (\mathbf{I} - \eta\mathbf{H})(\mathbf{w}_k - \mathbf{w}^\star).$$

$\mathbf{H}$ is symmetric, so it has orthonormal eigenvectors $\mathbf{q}_j$ with eigenvalues $\lambda_j \ge 0$
([A3.1](a3_1_eigenvalues_eigenvectors.md)). Write the error as a combination of them,
$\mathbf{w}_k - \mathbf{w}^\star = \sum_j c_j\mathbf{q}_j$. Since $\mathbf{H}\mathbf{q}_j = \lambda_j\mathbf{q}_j$, we get $(\mathbf{I} - \eta\mathbf{H})\mathbf{q}_j = (1 - \eta\lambda_j)\mathbf{q}_j$,
so one step turns the error into $\sum_j (1 - \eta\lambda_j)\,c_j\mathbf{q}_j$: **each eigen-direction
behaves like its own one-dimensional quadratic with curvature $\lambda_j$**. Two consequences:

1. *Stability is set by the steepest direction.* Every factor needs $\lvert 1 - \eta\lambda_j\rvert < 1$, so
   $\eta < 2/\lambda_{\max}$.
2. *Speed is set by the flattest direction.* With $\eta$ at most about $1/\lambda_{\max}$, the flattest
   direction shrinks by $1 - \eta\lambda_{\min} \approx 1 - \lambda_{\min}/\lambda_{\max} = 1 - 1/\kappa$ per step, so after $k$ steps
   by $(1 - 1/\kappa)^k$. For large $\kappa$, $(1 - 1/\kappa)^k \approx e^{-k/\kappa}$ (the compound-interest limit, stated;
   try $\kappa = 100$, $k = 100$: $0.99^{100} = 0.366$ against $e^{-1} = 0.368$). So shrinking an error by a factor
   $10^{-6}$ needs $e^{-k/\kappa} = 10^{-6}$, i.e. $k \approx \kappa\ln(10^6) \approx 14\kappa$ steps. (The best
   fixed rate, $\eta = 2/(\lambda_{\min} + \lambda_{\max})$, improves the factor to $\frac{\kappa - 1}{\kappa + 1}$, still about
   $\kappa/2$ times more steps than a round bowl.)

**Feature scale sets $\kappa$.** $\mathbf{H} = \frac{2}{n}\mathbf{X}^\top\mathbf{X}$ has entries
$\frac{2}{n}\sum_i x_{ij}x_{ik}$, so a feature measured in large units makes its curvature large, and a
feature that is far from zero correlates with the ones column. For A2.4's $x \in [0, 10]$,
$\kappa \approx 161$. Standardizing, $x \to (x - \bar{x})/s$ (mean 0, standard deviation 1 with
$s^2 = \frac{1}{n}\sum_i(x_i - \bar{x})^2$), makes the column orthogonal to the ones column
($\sum_i (x_i - \bar{x}) = 0$) and gives both columns $\frac{1}{n}\sum(\cdot)^2 = 1$, so $\mathbf{H} = 2\mathbf{I}$ and $\kappa = 1$:
a perfectly round bowl. (Also note $\kappa(\mathbf{H}) = \kappa(\mathbf{X})^2$, as in A3.4.) After fitting, undo the
scaling: $y = w_0 + w_1\frac{x - \bar{x}}{s}$ means slope $w_1/s$ and intercept $w_0 - \bar{x}\,w_1/s$.

**When to stop.** At the minimum the gradient is zero, so stop when $\lVert\mathbf{g}\rVert$ falls below a
tolerance; also cap the number of iterations, and stop with an error if the loss or gradient becomes
non-finite or huge.

**Mini-batch stochastic gradient descent (SGD).** Computing $\mathbf{g}$ touches all $n$ rows. With
millions of rows, use a random **mini-batch** of $B$ rows instead: the gradient of the batch's mean
loss, $\frac{2}{B}\mathbf{X}_B^\top(\mathbf{X}_B\mathbf{w} - \mathbf{y}_B)$. Averaged over all possible random batches it equals the
full gradient (it is an *unbiased* estimate; expectation is [A5.1](a5_1_random_variables_expectation.md)),
but each one is noisy. One pass over the shuffled data in batches is an **epoch**. Two facts follow from
the noise:

- With a *constant* $\eta$, SGD does not settle at $\mathbf{w}^\star$: even at $\mathbf{w}^\star$ a batch's gradient is
  not zero, so it keeps hopping around in a small region whose size grows with $\eta$.
- *Decaying* $\eta$ over time shrinks the hops and lets it converge. That is a **learning-rate schedule**
  ([C3.2](../C/c3_2_lr_schedules.md)).

Each SGD update costs $B/n$ of a full gradient, so for large $n$ you make far more progress per second,
which is why every deep-learning optimizer is a variant of it ([C3.1](../C/c3_1_optimizers.md)).

## Worked example by hand

A2.4's four points: $x = (0, 1, 2, 3)$, $\mathbf{y} = (1, 3, 4, 4)$, $\mathbf{X}$ `(4, 2)` with a ones column, and the
least-squares answer $\mathbf{w}^\star = (1.5, 1)$. Start at $\mathbf{w}_0 = (0, 0)$ with $\eta = 0.1$.

**Step 1.** A4.2 computed $\nabla\mathcal{L}(\mathbf{0}) = (-6, -11.5)$, so $\mathbf{w}_1 = (0, 0) - 0.1 \cdot (-6, -11.5) = (0.6, 1.15)$.
Predictions $0.6 + 1.15x = (0.6, 1.75, 2.9, 4.05)$, residuals (prediction − target)
$(-0.4, -1.25, -1.1, 0.05)$, loss $(0.16 + 1.5625 + 1.21 + 0.0025)/4 = 0.73375$, down from 10.5.

**Step 2.** $\mathbf{X}^\top\mathbf{r} = (\sum r_i, \sum x_ir_i) = (-2.7,\; 0 - 1.25 - 2.2 + 0.15) = (-2.7, -3.3)$, so
$\mathbf{g} = \frac{2}{4}(-2.7, -3.3) = (-1.35, -1.65)$ and $\mathbf{w}_2 = (0.6 + 0.135, 1.15 + 0.165) = (0.735, 1.315)$, loss
$0.45959$. The loss fell a lot in step 1 and much less in step 2: the fast direction is already done.

**How large can $\eta$ be?** $\mathbf{H} = \frac{2}{4}\mathbf{X}^\top\mathbf{X} = \frac12\begin{pmatrix}4&6\\6&14\end{pmatrix} = \begin{pmatrix}2&3\\3&7\end{pmatrix}$.
Trace 9, determinant $14 - 9 = 5$, so $\lambda = \frac{9 \pm \sqrt{81 - 20}}{2} = \frac{9 \pm 7.8102}{2}$: $\lambda_{\max} = 8.4051$,
$\lambda_{\min} = 0.5949$. Largest stable rate $2/8.4051 = 0.238$; $\kappa = 14.13$. With $\eta = 0.1$ the slow
direction shrinks by $1 - 0.1 \cdot 0.5949 = 0.9405$ per step, so a factor of $10^{-4}$ takes about
$\ln(10^{-4})/\ln(0.9405) \approx 150$ steps. Prediction for three rates: $0.01$ crawls, $0.2$ converges
(below 0.238), $0.25$ diverges (above it).

```text
== worked example: two steps by hand ==
step 0: w = [0.0, 0.0]  loss = 10.50000
step 1: gradient = [-6.0, -11.5]  w = [0.6, 1.15]  residuals = [-0.4, -1.25, -1.1, 0.05]  loss = 0.73375
step 2: gradient = [-1.35, -1.65]  w = [0.735, 1.315]  residuals = [-0.265, -0.95, -0.635, 0.68]  loss = 0.45959
Hessian (2/n) X^T X = [[2.0, 3.0], [3.0, 7.0]]   eigenvalues = [0.5949, 8.4051]
largest stable learning rate 2 / lambda_max = 0.2380   condition number = 14.13
with lr = 0.1, the slow direction shrinks by 1 - lr * lambda_min = 0.9405 per step

== worked example: three learning rates ==
least-squares answer for the 4 points: w = [1.5, 1.0]   loss = 0.2500
lr = 0.01  loss after 10 steps =     2.2063   after 100 =       0.3287   w = [1.034, 1.218]
lr = 0.2   loss after 10 steps =     0.2752   after 100 =       0.2500   w = [1.5, 1.0]
lr = 0.25  loss after 10 steps =    69.0539   after 100 = 2394576823.9217   w = [-10123.2031, -21615.6625]
```

A rate only 5% above the limit (0.25 against 0.238) does not "converge a bit worse": the loss reaches
$2 \times 10^9$ in 100 steps. Divergence is exponential.

## Key plot

![Left: loss minus its minimum on a log scale against iteration for three learning rates on A2.4's data; 0.002 flattens out, 0.025 falls steadily, 0.0305 shoots up. Right: loss contours form a long tilted ellipse; with learning rate 0.026 the path zig-zags across the valley, with 0.012 it moves smoothly but slowly along it](img/a4_4_learning_rates.png)

Left: on A2.4's 50 points (raw $x$, limit $2/\lambda_{\max} = 0.027$), all three rates drop fast at first (the
steep direction), then split. 0.002 nearly stops: its slow-direction factor is $1 - 0.002 \cdot 0.46 = 0.999$.
0.025 falls as a straight line on the log scale, which is the signature of the constant per-step factor.
0.0305 explodes. Right: the loss surface over (intercept, slope) is a long, narrow, tilted valley
($\kappa = 161$). At 0.026 each step overshoots across the valley (factor $1 - 0.026 \cdot 74 = -0.93$ in the
steep direction), so the path zig-zags; at 0.012 it does not zig-zag but creeps along the floor.
Neither reaches the star in the steps shown.

## From scratch in NumPy

The loop, with the three stopping rules (converged, diverged, out of budget):

```python
def gd_until_converged(
    X_: FloatArray, y_: FloatArray, lr_: float, tol: float = GRAD_TOL, max_iters: int = MAX_ITERS
) -> tuple[FloatArray, int]:
    """Batch GD on the MSE from w = 0 until ||gradient|| < tol. Returns (w (d,), iterations)."""
    w_ = np.zeros(X_.shape[1])
    for k in range(1, max_iters + 1):
        g = grad_mse(w_, X_, y_)
        if not np.all(np.isfinite(g)) or np.linalg.norm(g) > 1e12:
            raise FloatingPointError(f"gradient norm {np.linalg.norm(g):.1e} at iteration {k}: learning rate {lr_} is too large")
        if np.linalg.norm(g) < tol:
            return w_, k
        w_ -= lr_ * g
    raise RuntimeError(f"no convergence in {max_iters} iterations (gradient norm {np.linalg.norm(g):.2e})")
```

On the 50 points with raw $x$, against A2.4's normal-equation answer $(1.7575, 0.5523)$:

```text
== from scratch: batch gradient descent ==
Hessian eigenvalues = [0.4589, 74.0413]   condition number = 161.3   largest stable lr = 2 / lambda_max = 0.02701
lr = 0.025: converged in 1973 iterations   w = [1.7575, 0.5523]
matches the normal equations (A2.4) to 4 decimals: OK   (max difference 2.2e-10)
lr = 0.03 (above 2 / lambda_max): gradient norm 1.0e+12 at iteration 119 -> diverged
```

1,973 iterations for two parameters, because $\kappa = 161$ (the same 161.3 that A2.4 printed as
`cond(X^T X)`). Now standardize $x$ first:

```text
== from scratch: standardized feature ==
Hessian eigenvalues = [2.0, 2.0]   condition number = 1.00
lr = 0.5000: converged in 2 iterations   w (standardized) = [4.6674, 1.6098]   back on the x scale = [1.7575, 0.5523]
matches the normal equations: OK
with a typical lr = 0.1 instead: 115 iterations (raw x needed 1973)
```

$\mathbf{H} = 2\mathbf{I}$ exactly, as derived, so $\eta = 1/\lambda = 0.5$ lands on the answer in one step (the second
iteration only confirms the gradient is zero). That is a special case of a one-feature model; with many
correlated features standardizing helps a lot but cannot make $\kappa = 1$. The more honest comparison is
the last line: the same ordinary rate needs 17 times fewer steps on standardized data, and the safe range
of $\eta$ no longer depends on the units of $x$. In standardized units the intercept is the mean of $y$
(4.6674) and the slope is per standard deviation of $x$; converted back they match A2.4.

Mini-batch SGD on the standardized feature, batches of 10 reshuffled every epoch:

```text
== from scratch: mini-batch SGD ==
SGD, batch 10, 200 epochs, lr 0.1 constant           w = [4.6595, 1.628]   distance to the GD answer = 2.0e-02
SGD, batch 10, 200 epochs, lr 0.1 / (1 + epoch / 10) w = [4.6671, 1.61]   distance to the GD answer = 3.2e-04
200 epochs x 5 batches = 1000 updates, each using 10 of 50 rows
```

With a constant rate SGD hovers about 0.02 from the answer however long it runs; with a decaying rate it
gets 60 times closer. For 50 rows SGD is pointless (the full gradient is cheap); at 50 million rows it is
the only option.

## With PyTorch and scikit-learn

`torch.optim.SGD` with the full batch takes exactly the update $\mathbf{w} \leftarrow \mathbf{w} - \eta\mathbf{g}$, so from the same
start and for the same number of steps it reproduces our loop to rounding. scikit-learn's `SGDRegressor`
is a real mini-batch-of-one SGD with its own schedule and stopping rule:

```text
== with pytorch and scikit-learn ==
torch.optim.SGD, 1972 full-batch steps at lr 0.025: w = [1.7575, 0.5523]   identical to our loop: OK
SGDRegressor on the standardized feature: 107 epochs   w back on the x scale = [1.7581, 0.5521]   distance to the normal equations = 6.4e-04
```

(1,972 steps, because our loop counts the final iteration that only checks the gradient.) `SGDRegressor`
stops when the loss stops improving by more than `tol`, about $6 \times 10^{-4}$ from the exact answer. For
a model this small use `LinearRegression`; `SGDRegressor` is for data that do not fit in memory
(`partial_fit`). Its defaults (`learning_rate="invscaling"`, `eta0=0.01`, an L2 penalty unless you pass
`penalty=None`) are VERIFY (as of 2026-10) items; the lab sets them explicitly.

## In practice

- **When to use it:** whenever there is no closed form (logistic regression, neural networks), or the
  data are too large for one (mini-batch SGD). For ordinary least squares with modest $d$, `lstsq` is
  faster and exact; gradient descent here is for understanding and for checking.
- **Failure modes:**
  - *Learning rate too large:* the loss rises, oscillates or becomes `nan`/`inf`. You notice a loss curve
    going up, often within the first few steps; divide $\eta$ by 3–10.
  - *Learning rate too small:* the loss decreases slowly and steadily forever. You notice a curve that
    is still sloping when you stop; try $\eta$ several times larger.
  - *Unscaled features:* one feature in large units sets $\lambda_{\max}$, forcing a tiny $\eta$ under which the
    other weights barely move. You notice that some weights stop changing while others oscillate;
    standardize inputs (fit the scaler on training data only, [B1.4](../B/b1_4_data_leakage.md)).
  - *SGD on sorted data:* batches that are not shuffled each contain one region of the data, and the
    weights chase each region in turn. You notice a sawtooth loss curve that repeats every epoch; shuffle
    every epoch.
  - *Judging convergence on one noisy batch:* mini-batch losses jump around. Track a moving average,
    or the full loss on a fixed evaluation set.
  - *No stopping rule:* a fixed iteration count silently returns an unconverged answer. Check the gradient
    norm, and raise on non-finite values, as `gd_until_converged` does.
- **Defaults that work:** standardize every numeric feature; try learning rates on a log grid
  (`1e-4, 3e-4, 1e-3, …, 1`) and take the largest whose loss curve falls smoothly; plot the loss on a log
  scale; start from zero (linear models) or small random weights (networks, [C3.3](../C/c3_3_initialization.md));
  shuffle every epoch; use a decaying schedule for the last part of training.

## Exercises

Save your work to `my-work/A/a4_4_gradient_descent.md` (or `.py`) and ask for a review.

1. **By hand:** (a) For $\mathcal{L}(w) = 2(w - 3)^2$, find the curvature $a$ and the largest stable learning
   rate. Starting from $w_0 = 0$, take three steps with each of $\eta = 0.1, 0.25, 0.4, 0.6$, and classify each
   run as monotone, one-step, oscillating-convergent or divergent. (b) A two-parameter quadratic loss has
   Hessian $\begin{pmatrix}4&0\\0&1\end{pmatrix}$. What is the largest stable $\eta$? With $\eta = 0.4$, by what factor does
   the error shrink per step in each direction, and about how many steps does it take to shrink the
   slow direction's error 1,000-fold? Which direction oscillates?
2. **Code:** On A2.4's 50 points with raw $x$: (a) find, by bisection between 0.02 and 0.04, the largest
   learning rate for which batch GD converges, and compare with $2/\lambda_{\max}$. (b) Implement gradient
   descent with a **backtracking line search**: at each step start from $\eta = 1$ and halve it until
   $\mathcal{L}(\mathbf{w} - \eta\mathbf{g}) \le \mathcal{L}(\mathbf{w}) - \frac{1}{2}\eta\lVert\mathbf{g}\rVert^2$. Report the number of iterations and the
   number of loss evaluations needed to reach $\lVert\mathbf{g}\rVert < 10^{-6}$, compared with fixed $\eta = 0.025$,
   and check the final weights against the normal equations to 4 decimals. (c) Now ask the line search for
   $\lVert\mathbf{g}\rVert < 10^{-10}$. What happens, and why? (Hint: how far apart are float64 numbers near the
   loss value, and how big is the decrease the test demands?)
3. **Think:** A churn model trained nightly by mini-batch gradient descent starts producing `nan` loss
   after a teammate adds a new feature, `lifetime_revenue_cents` (values up to about $10^7$). Nothing else
   changed. Explain what happened in terms of the Hessian and the stable learning rate. How would you
   fix it, and what would you add to the pipeline so a change like this is caught before training, and
   during training if it slips through?

## Takeaways

1. Gradient descent repeats $\mathbf{w} \leftarrow \mathbf{w} - \eta\nabla\mathcal{L}(\mathbf{w})$; for small enough $\eta$ every step lowers the loss.
2. On a quadratic, each eigen-direction of the Hessian is multiplied by $1 - \eta\lambda_j$ per step: stable iff
   $\eta < 2/\lambda_{\max}$, and the slowest direction needs about $\kappa = \lambda_{\max}/\lambda_{\min}$ steps per factor $e$.
3. Diverging is exponential, not gradual: a rate a few percent too large reaches `inf` quickly.
4. Feature scale sets $\kappa$: standardizing turns a narrow valley into a round bowl, so the same rate works
   for any units and converges many times faster.
5. Mini-batch SGD uses a noisy but unbiased gradient from $B$ rows; with a constant rate it hovers near the
   answer, with a decaying rate it converges.

**What builds on this:** [A4.5](a4_5_convexity_lagrange.md) (when one valley floor is guaranteed);
[A5.5](a5_5_mle_map.md) (fitting by minimizing negative log-likelihood with this code);
[B1.5](../B/b1_5_linear_regression.md) and [B2.1](../B/b2_1_logistic_regression.md) (models trained this way);
[C3.1](../C/c3_1_optimizers.md) (momentum and Adam); [C3.2](../C/c3_2_lr_schedules.md) (learning-rate schedules).
