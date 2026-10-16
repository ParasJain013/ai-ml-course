# A4.2 Gradients, Jacobians and the shapes of matrix derivatives used in ML

**Core** · Part A, week 4 (course week 04) · ~2 h · Lab:
[`labs/A/a4_2_gradients_jacobians.py`](../../labs/A/a4_2_gradients_jacobians.py)

## Why it matters

A model has thousands to billions of parameters, and training needs to know, for every one of them,
which way to move it to lower the loss. The **gradient** packs all those answers into one array with
the same shape as the parameters. Every optimizer you will use updates parameters with it, and every
`loss.backward()` in PyTorch computes it. Most hand-derived gradients in ML come from three or four
patterns ($\mathbf{X}^\top\mathbf{r}$, $\mathbf{A}^\top(\mathbf{A}\mathbf{x} - \mathbf{b})$, $\mathbf{X}^\top\mathbf{G}$), and the
bugs are always the same: a transpose in the wrong place, a missing factor of 2 or $1/n$, a shape that
broadcasts silently into nonsense. This lesson derives the patterns and builds the checker that
catches those bugs.

## Prerequisite check

- Can you compute a partial derivative and check it with a central difference? If not, revisit
  [A4.1](a4_1_derivatives.md).
- Can you compute a dot product and say what it means when it is largest for unit vectors? If not,
  revisit [A1.5](a1_5_vectors.md).
- Can you state the shape of $\mathbf{X}^\top\mathbf{r}$ for $\mathbf{X}$ `(n, d)` and $\mathbf{r}$ `(n,)`, and
  say what a matrix does to a vector as a linear map? If not, revisit
  [A2.1](a2_1_matrices_as_linear_maps.md) and [A2.2](a2_2_matrix_multiplication_inverse.md).
- Can you write the least-squares loss $\lVert\mathbf{X}\mathbf{w} - \mathbf{y}\rVert^2$ and the normal
  equations? If not, revisit [A2.4](a2_4_projections_least_squares.md).

## Intuition

### All the slopes at once

[A4.1](a4_1_derivatives.md) asked "if I turn one knob, how fast does the loss move?" The **gradient**
asks it for every knob and stacks the answers into a vector:
$\nabla f = \big(\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}\big)$. The one-variable
rule $f(x + h) \approx f(x) + f'(x)h$ becomes, for a small step $\boldsymbol{\delta}$ in any direction,

$$f(\mathbf{x} + \boldsymbol{\delta}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x}) \cdot \boldsymbol{\delta}.$$

Each coordinate of the step contributes its own slope times its own size, and the contributions add.

### The gradient points uphill, steepest first

Which direction of a unit-length step raises $f$ the most? The rate of change along a unit direction
$\mathbf{u}$, the **directional derivative**, is $\nabla f \cdot \mathbf{u}$ (the linear approximation above with
$\boldsymbol{\delta} = h\mathbf{u}$, divided by $h$), and
for a unit vector $\mathbf{u}$ that dot product is $\lVert\nabla f\rVert\cos\theta$ ([A1.5](a1_5_vectors.md)),
largest when $\theta = 0$. So **the gradient points in the direction of steepest increase, and its
length is that steepest slope**. Minus the gradient is steepest *decrease*: the direction gradient
descent ([A4.4](a4_4_gradient_descent.md)) steps in. Steps perpendicular to the gradient
($\theta = 90°$) do not change $f$ at all, to first order, which means the gradient is perpendicular to
the **contour lines** (the lines of equal $f$) in the key plot. It does *not* point at the minimum,
only straight downhill from where you stand; that distinction explains why gradient descent zig-zags.

### Many outputs: the Jacobian

If a function returns a vector, say $m$ outputs from $n$ inputs, each output has its own gradient.
Stack them as rows and you get the **Jacobian**, a matrix of shape `(m, n)`. It plays the role of the
derivative: $f(\mathbf{x} + \boldsymbol{\delta}) \approx f(\mathbf{x}) + \mathbf{J}\boldsymbol{\delta}$. In the words of
[A2.1](a2_1_matrices_as_linear_maps.md), the Jacobian is the linear map that best imitates $f$ near
$\mathbf{x}$. Zoom in on any smooth function and it looks linear; the Jacobian is that linear map.

> **Bridge:** a Jacobian is like the sensitivity table a capacity planner builds: one row per output
> metric (latency, CPU, cost), one column per input knob (replicas, cache size, batch size), and each
> cell says "this metric moves this much per unit of this knob". **Where it breaks:** a planner's table
> is usually measured once and reused; a Jacobian is only valid near the point where it was computed
> and changes as soon as you move. Gradient descent recomputes it at every step for that reason.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $f:\mathbb{R}^n \to \mathbb{R}$ | a scalar function of a vector (a loss) | — |
| $\mathbf{x}$, $\boldsymbol{\delta}$, $\mathbf{u}$ | a point, a small step, a unit direction | `(n,)` |
| $\nabla f(\mathbf{x})$ | gradient: the partial derivatives $\partial f/\partial x_j$ | `(n,)`, same as $\mathbf{x}$ |
| $\mathbf{f}:\mathbb{R}^n \to \mathbb{R}^m$ | a vector function, outputs $f_1, \dots, f_m$ | — |
| $\mathbf{J}$ | Jacobian, $J_{ij} = \partial f_i/\partial x_j$ | `(m, n)` |
| $\mathbf{X}$, $\mathbf{y}$, $\mathbf{w}$ | data matrix, targets, weights | `(n, d)`, `(n,)`, `(d,)` |
| $\mathbf{r}$ | residuals $\mathbf{X}\mathbf{w} - \mathbf{y}$ | `(n,)` |
| $\mathcal{L}$ | mean squared error $\frac{1}{n}\lVert\mathbf{X}\mathbf{w} - \mathbf{y}\rVert^2$ | scalar |
| $\mathbf{A}$, $\mathbf{b}$ | a matrix and vector in $\lVert\mathbf{A}\mathbf{x} - \mathbf{b}\rVert^2$ | `(m, n)`, `(m,)` |
| $\mathbf{M}$ | a square matrix in the quadratic form $\mathbf{x}^\top\mathbf{M}\mathbf{x}$ | `(n, n)` |
| $\mathbf{W}$, $\mathbf{Z}$ | a layer's weights and outputs $\mathbf{Z} = \mathbf{X}\mathbf{W}$ | `(d_in, d_out)`, `(n, d_out)` |
| $\mathbf{G}$ | upstream gradient $\partial\mathcal{L}/\partial\mathbf{Z}$ | `(n, d_out)` |

Here $n$ means two things: the number of inputs of a generic $f$, and the number of examples in
$\mathbf{X}$. Which one is always clear from the formula.

**Two shape rules.** (1) The gradient of a scalar with respect to anything has the shape of that
thing: $\nabla_{\mathbf{w}}\mathcal{L}$ is `(d,)` like $\mathbf{w}$, and $\nabla_{\mathbf{W}}\mathcal{L}$ is `(d_in, d_out)`
like $\mathbf{W}$. That is the course convention ([STYLE_GUIDE](../../course/STYLE_GUIDE.md) §1.3) and
PyTorch's: `w.grad.shape == w.shape`. (2) The Jacobian of $\mathbf{f}:\mathbb{R}^n \to \mathbb{R}^m$ is
`(m, n)`: one row per output. A scalar $f$ has $m = 1$, and its Jacobian is the gradient laid out as
a `(1, n)` row.

**The method for every derivation below.** Write the function as a sum over indices, take the
partial derivative with respect to one entry with the A4.1 rules, then recognise the result as a
matrix product. It is mechanical and it never relies on memorised matrix-calculus identities.

**1. A linear function.** $f(\mathbf{w}) = \mathbf{a}^\top\mathbf{w} = \sum_k a_k w_k$. Differentiating in
$w_j$, every term with $k \ne j$ is a constant, so $\partial f/\partial w_j = a_j$ and $\nabla f = \mathbf{a}$.

**2. A quadratic form.** $f(\mathbf{x}) = \mathbf{x}^\top\mathbf{M}\mathbf{x} = \sum_i\sum_k x_i M_{ik} x_k$, a sum of
products. By the product rule (A4.1) the derivative of one term $x_i M_{ik} x_k$ in $x_j$ is
$\frac{\partial x_i}{\partial x_j} M_{ik} x_k + x_i M_{ik}\frac{\partial x_k}{\partial x_j}$, and $\frac{\partial x_i}{\partial x_j}$ is 1 when
$i = j$ and 0 otherwise. So the first part survives only for $i = j$ and the second only for $k = j$:

$$\frac{\partial f}{\partial x_j} = \sum_k M_{jk}x_k + \sum_i x_i M_{ij} = (\mathbf{M}\mathbf{x})_j + (\mathbf{M}^\top\mathbf{x})_j, \qquad \nabla f = (\mathbf{M} + \mathbf{M}^\top)\,\mathbf{x}.$$

(The second sum runs down column $j$ of $\mathbf{M}$, which is row $j$ of $\mathbf{M}^\top$.) If $\mathbf{M}$ is symmetric this is $2\mathbf{M}\mathbf{x}$, the matrix version of $(mx^2)' = 2mx$. Writing
$2\mathbf{M}\mathbf{x}$ for a non-symmetric $\mathbf{M}$ is a classic bug; the lab's checker catches it.

**3. Squared distance $\lVert\mathbf{A}\mathbf{x} - \mathbf{b}\rVert^2$.** Expand it like $(a - b)^2$, using
$\lVert\mathbf{v}\rVert^2 = \mathbf{v}^\top\mathbf{v}$ and $(\mathbf{A}\mathbf{x})^\top = \mathbf{x}^\top\mathbf{A}^\top$ ([A2.2](a2_2_matrix_multiplication_inverse.md)):

$$\lVert\mathbf{A}\mathbf{x} - \mathbf{b}\rVert^2 = (\mathbf{A}\mathbf{x} - \mathbf{b})^\top(\mathbf{A}\mathbf{x} - \mathbf{b}) = \mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x} - \mathbf{x}^\top\mathbf{A}^\top\mathbf{b} - \mathbf{b}^\top\mathbf{A}\mathbf{x} + \mathbf{b}^\top\mathbf{b} = \mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x} - 2\,\mathbf{b}^\top\mathbf{A}\mathbf{x} + \mathbf{b}^\top\mathbf{b}.$$

(The two cross terms $\mathbf{x}^\top\mathbf{A}^\top\mathbf{b}$ and $\mathbf{b}^\top\mathbf{A}\mathbf{x}$ are the same number, a scalar
equals its own transpose, hence the 2.) Term by term: the first is a quadratic form with the symmetric
matrix $\mathbf{A}^\top\mathbf{A}$, gradient $2\mathbf{A}^\top\mathbf{A}\mathbf{x}$ (pattern 2); the second is linear,
$\mathbf{a}^\top\mathbf{x}$ with $\mathbf{a} = -2\mathbf{A}^\top\mathbf{b}$ (pattern 1); the third does not involve $\mathbf{x}$. So

$$\nabla_{\mathbf{x}}\lVert\mathbf{A}\mathbf{x} - \mathbf{b}\rVert^2 = 2\mathbf{A}^\top\mathbf{A}\mathbf{x} - 2\mathbf{A}^\top\mathbf{b} = 2\mathbf{A}^\top(\mathbf{A}\mathbf{x} - \mathbf{b}).$$

Shape check: $\mathbf{A}^\top$ is `(n, m)`, the residual $\mathbf{A}\mathbf{x} - \mathbf{b}$ is `(m,)`, the product is
`(n,)` like $\mathbf{x}$ ✓. Read it as "residual, sent back through $\mathbf{A}^\top$": each input
coordinate collects the residuals of the outputs it influences, weighted by how much it influences them.

**4. Mean squared error.** With $\mathbf{A} = \mathbf{X}$, $\mathbf{b} = \mathbf{y}$, $\mathbf{x} = \mathbf{w}$ and the $\frac1n$ in front:

$$\mathcal{L}(\mathbf{w}) = \frac{1}{n}\lVert\mathbf{X}\mathbf{w} - \mathbf{y}\rVert^2, \qquad \nabla_{\mathbf{w}}\mathcal{L} = \frac{2}{n}\,\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}) = \frac{2}{n}\,\mathbf{X}^\top\mathbf{r}.$$

In code: `2 / n * X.T @ (X @ w - y)`, shapes `(d, n) @ (n,) -> (d,)`. Component $j$ is
$\frac{2}{n}\sum_i r_i X_{ij}$: the A4.1 result $2rx$ for one example, averaged over all examples. Setting
it to zero gives $\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}$, the normal equations of
[A2.4](a2_4_projections_least_squares.md): the least-squares solution is exactly where the gradient
vanishes. If the model has a separate bias $b$, then $\partial\mathcal{L}/\partial b = \frac{2}{n}\sum_i r_i$, the same
formula with a column of ones.

**5. A linear layer, $\mathbf{Z} = \mathbf{X}\mathbf{W}$.** In a network some later computation turns $\mathbf{Z}$
into a scalar loss. Suppose we already know $\mathbf{G} = \partial\mathcal{L}/\partial\mathbf{Z}$, shape `(n, d_out)` (computing
it is [A4.3](a4_3_chain_rule_graphs.md)'s job). Entry $W_{jk}$ affects only column $k$ of $\mathbf{Z}$,
through $Z_{ik} = \sum_{j'} X_{ij'}W_{j'k}$, so $\partial Z_{ik}/\partial W_{jk} = X_{ij}$. Each of those $n$
entries passes its own gradient $G_{ik}$ on, and the contributions add over the examples:

$$\frac{\partial\mathcal{L}}{\partial W_{jk}} = \sum_{i=1}^{n} G_{ik}\,X_{ij} = (\mathbf{X}^\top\mathbf{G})_{jk}, \qquad \nabla_{\mathbf{W}}\mathcal{L} = \mathbf{X}^\top\mathbf{G}.$$

Shape check: `(d_in, n) @ (n, d_out) -> (d_in, d_out)`, the shape of $\mathbf{W}$ ✓. This one line,
`dW = X.T @ dZ`, is in every backpropagation you will write ([C1.2](../C/c1_2_backprop_by_hand.md)).
A useful memory aid: $\mathbf{X}^\top\mathbf{G}$ is the only way to combine `(n, d_in)` and `(n, d_out)` into
`(d_in, d_out)`. Use it to *guess*, then verify with the checker: shapes can match by accident when
dimensions are equal.

**6. Jacobians you will meet.** For $\mathbf{f}(\mathbf{x}) = \mathbf{A}\mathbf{x}$, output $i$ is
$\sum_j A_{ij}x_j$, so $\partial f_i/\partial x_j = A_{ij}$: **the Jacobian of a linear map is its
matrix**. For an elementwise function $\mathbf{f}(\mathbf{x}) = (g(x_1), \dots, g(x_n))$ such as `tanh`,
output $i$ depends on $x_i$ only, so the Jacobian is diagonal with entries $g'(x_i)$. Multiplying by a
diagonal matrix is the same as multiplying elementwise, so code never builds it: `J @ v` becomes
`g_prime(x) * v`. That is why backpropagation through an activation is an elementwise product ($\odot$).

## Worked example by hand

**1. A gradient and what it predicts.** $f(\mathbf{x}) = x_1^2 + 3x_1x_2$ at $\mathbf{x} = (1, 2)$. Partial
derivatives: $\partial f/\partial x_1 = 2x_1 + 3x_2 = 8$, $\partial f/\partial x_2 = 3x_1 = 3$, so $\nabla f = (8, 3)$,
length $\sqrt{64 + 9} = \sqrt{73} \approx 8.544$.

- Slope in the unit direction $\mathbf{u} = (0.6, 0.8)$: $\nabla f \cdot \mathbf{u} = 4.8 + 2.4 = 7.2$. In the
  gradient's own direction it is the length, $8.544$, the steepest possible.
- Linear prediction for the step $\boldsymbol{\delta} = (0.01, -0.02)$: $f = 1 + 6 = 7$, and
  $7 + 8(0.01) + 3(-0.02) = 7.02$. The true value is $1.01^2 + 3(1.01)(1.98) = 1.0201 + 5.9994 = 7.0195$:
  off by $0.0005$, a second-order amount (it is $\delta_1^2 + 3\delta_1\delta_2$, the part a linear
  approximation leaves out).

**2. The MSE gradient.** $\mathbf{X} = \begin{pmatrix}1&1\\1&2\\1&3\end{pmatrix}$ `(3, 2)` (a ones column and
$x = 1, 2, 3$), $\mathbf{y} = (1, 2, 2)$, $\mathbf{w} = (0, 1)$: intercept 0, slope 1.

- Predictions $\mathbf{X}\mathbf{w} = (1, 2, 3)$, residuals $\mathbf{r} = (0, 0, 1)$, $\mathcal{L} = 1/3$.
- $\mathbf{X}^\top\mathbf{r}$: column 1 of $\mathbf{X}$ dotted with $\mathbf{r}$ is $0 + 0 + 1 = 1$; column 2 is
  $0 + 0 + 3 = 3$. So $\nabla\mathcal{L} = \frac{2}{3}(1, 3) = (0.6667, 2)$.
- Both components are positive, so increasing either the intercept or the slope makes the fit worse:
  the only miss is the last point, predicted 3 against a target of 2, and the slope's push is three
  times stronger because that point has $x = 3$.
- The normal equations give $\mathbf{w} = (0.6667, 0.5)$, where the gradient is $(0, 0)$.

**3. A Jacobian.** $\mathbf{f}(\mathbf{x}) = (x_1x_2,\; x_1 + x_2^2)$ at $(2, 3)$. Row 1 is the gradient of
$x_1x_2$, $(x_2, x_1) = (3, 2)$; row 2 is the gradient of $x_1 + x_2^2$, $(1, 2x_2) = (1, 6)$:
$\mathbf{J} = \begin{pmatrix}3&2\\1&6\end{pmatrix}$. Prediction for $\boldsymbol{\delta} = (0.01, 0.01)$:
$\mathbf{J}\boldsymbol{\delta} = (0.05, 0.07)$; the true change is $(0.0501, 0.0701)$.

```text
== worked example: gradient of x1^2 + 3 x1 x2 ==
gradient at (1, 2) by hand = [8.0, 3.0]   length = 8.544
directional derivative along (0.6, 0.8): grad . u = 7.2000   numerical = 7.2000
along the gradient's own direction: 8.5440 (= its length, the steepest)
f(x0 + delta) = 7.019500   f(x0) + grad . delta = 7.020000

== worked example: gradient of MSE ==
w = [0.0, 1.0]   predictions = [1.0, 2.0, 3.0]   residuals = [0.0, 0.0, 1.0]
MSE = 0.3333   X^T r = [1.0, 3.0]   gradient = (2/3) X^T r = [0.6667, 2.0]
at the least-squares w = [0.6667, 0.5]  the gradient is [0.0, 0.0]

== worked example: Jacobian of (x1 x2, x1 + x2^2) ==
J by hand at (2, 3) = [[3.0, 2.0], [1.0, 6.0]]
f(x + dx) - f(x) = [0.0501, 0.0701]   J @ dx = [0.05, 0.07]
```

## Key plot

![Contour lines of the mean squared error over intercept and slope for the three-point example, with red arrows of the negative gradient that cross every contour line at a right angle; the minimum at (0.6667, 0.5) is marked with a star](img/a4_2_gradient_contours.png)

The grey ellipses are contour lines of $\mathcal{L}(w_0, w_1)$ for the worked example; the star is the
minimum. Each red arrow is $-\nabla\mathcal{L}$ at its tail, scaled to the same length. Every arrow crosses
the contour line at a right angle, and many arrows do *not* point at the star: along the long, thin
valley they point mostly across it. Walking along such arrows zig-zags down the valley, which is
exactly the slow convergence [A4.4](a4_4_gradient_descent.md) explains with eigenvalues.

## From scratch in NumPy

A **gradient checker** compares an analytic gradient with a numerical one. The numerical gradient
nudges each entry of the parameter array in turn, whatever its shape; `np.ndindex` yields `(j,)` for a
vector and `(i, j)` for a matrix:

```python
def numerical_gradient(f: Callable[[FloatArray], float], x: FloatArray, h: float = H) -> FloatArray:
    """Central-difference gradient of a scalar function of an array of any shape. x (...) -> (...)."""
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + h
        f_plus = f(x)
        x[idx] = old - h
        f_minus = f(x)
        x[idx] = old  # restore: the caller's array must come back unchanged
        grad[idx] = (f_plus - f_minus) / (2 * h)
    return grad
```

It edits `x` in place to avoid copying a large array $2n$ times, and restores each entry before moving
on. The comparison uses a **relative error**, $\frac{\lVert\mathbf{a} - \mathbf{b}\rVert}{\lVert\mathbf{a}\rVert + \lVert\mathbf{b}\rVert}$,
because an absolute tolerance means nothing without knowing the gradient's size: $10^{-6}$ is tiny
for a gradient of size 1000 and huge for one of size $10^{-7}$. With central differences and $h = 10^{-6}$
a correct gradient gives about $10^{-9}$ to $10^{-11}$; a bug gives $10^{-2}$ or more. The lab checks
each pattern from "The math" on random data, plus two deliberate bugs:

```text
== from scratch: gradient checker ==
MSE: (2/n) X^T (Xw - y)              shape (3,)     rel error < 1e-8   PASS
MSE without the 2 (a typical bug)    shape (3,)     rel error 3.3e-01  FAIL
|Ax - b|^2: 2 A^T (Ax - b)           shape (3,)     rel error < 1e-8   PASS
w^T M w: (M + M^T) w                 shape (3,)     rel error < 1e-8   PASS
w^T M w: 2 M w (only if M symmetric) shape (3,)     rel error 1.7e-01  FAIL
Z = XW: dL/dW = X^T G                shape (3, 2)   rel error < 1e-8   PASS
numerical Jacobian at (2, 3) = [[3.0, 2.0], [1.0, 6.0]]   matches by hand: True
Jacobian of elementwise tanh is diagonal: True
Jacobian of v -> A v is A itself: True   shape (4, 3)
```

Two details. The missing-2 bug gives exactly $\frac{1 - 1/2}{1 + 1/2} = 0.33$: a relative error tells you
the *size* of the mistake too. And the linear-layer check uses the loss $\mathcal{L}(\mathbf{W}) = \sum_{ik}G_{ik}(\mathbf{X}\mathbf{W})_{ik}$,
`np.sum(G * (X @ W))`, the simplest loss whose gradient with respect to $\mathbf{Z}$ is exactly an
arbitrary $\mathbf{G}$; it tests the rule $\mathbf{X}^\top\mathbf{G}$ for *every* possible upstream gradient at once.
Tiny errors print as `< 1e-8` because their last digits vary between machines.

`numerical_jacobian` nudges each input and stores the change of the whole output vector as one
column, giving `(m, n)`.

## With SciPy

SciPy ships the same tools:

- `scipy.optimize.approx_fprime(x, f, eps, *args)`: a numerical gradient (forward differences).
- `scipy.optimize.check_grad(f, grad, x, *args)`: the norm of the difference between your gradient
  and `approx_fprime`'s. It is an *absolute* number, so judge it against the gradient's size.
- `scipy.differentiate.jacobian(f, x)`: a numerical Jacobian with error control. `f` must accept an
  array whose first axis is the input coordinate (VERIFY (as of 2026-10) the calling convention if you
  change the pinned SciPy 1.18.1).

```text
== with scipy ==
scipy.optimize.approx_fprime matches the analytic MSE gradient (rtol 1e-5): OK
scipy.optimize.check_grad = |analytic - forward-difference| < 1e-6: True
check_grad with the bug = 1.25  (large: the bug is caught)
scipy.differentiate.jacobian = [[3.0, 2.0], [1.0, 6.0]]  matches ours and the hand result: OK
```

`approx_fprime` agrees with the analytic gradient only to `rtol=1e-5` because forward differences are
less accurate (A4.1). Neither SciPy tool handles a matrix-shaped parameter; flatten it with
`W.ravel()` and reshape inside `f`, or use your own `numerical_gradient`.

## In practice

- **When to use it:** derive a gradient whenever you write a model or loss outside an autograd
  framework, or a custom backward pass inside one, and check it every time. In PyTorch,
  `torch.autograd.gradcheck` does the same job ([C2.1](../C/c2_1_tensors_autograd_modules.md)).
- **Failure modes:**
  - *Transposes:* `X @ r` instead of `X.T @ r` raises a shape error only if $n \ne d$. You notice a
    FAIL from the checker on a test with $n \ne d$; always test with distinct sizes (the lab uses 20, 3, 2).
  - *Missing constants:* a dropped 2 or $1/n$ still trains (it only rescales the learning rate),
    which is why it survives code review. You notice a constant relative error such as 0.33.
  - *Broadcasting:* `y` of shape `(n, 1)` and `X @ w` of shape `(n,)` give residuals of shape
    `(n, n)`. Nothing crashes; the gradient is garbage. You notice it with an `assert r.shape == (n,)`.
  - *Symmetric-only formulas:* $2\mathbf{M}\mathbf{x}$ for $\mathbf{x}^\top\mathbf{M}\mathbf{x}$ is only right if
    $\mathbf{M} = \mathbf{M}^\top$. Test with a non-symmetric random matrix.
  - *Checking at special points:* at $\mathbf{w} = \mathbf{0}$ or with all-equal inputs many wrong formulas
    give the right number. Check at random points.
- **Defaults that work:** float64, central differences with $h = 10^{-6}$, random inputs with
  unequal dimensions, relative error below $10^{-6}$ to pass, one deliberately broken gradient as a
  negative control.

## Exercises

Save your work to `my-work/A/a4_2_gradients_jacobians.md` (or `.py`) and ask for a review.

1. **By hand:** (a) Find $\nabla f$ for $f(\mathbf{x}) = (x_1 - 2x_2)^2 + x_1x_2$ and evaluate it at
   $(3, 1)$. (b) **Ridge regression** adds a penalty to the MSE:
   $\mathcal{L}(\mathbf{w}) = \frac{1}{n}\lVert\mathbf{X}\mathbf{w} - \mathbf{y}\rVert^2 + \lambda\lVert\mathbf{w}\rVert^2$. Derive
   $\nabla_{\mathbf{w}}\mathcal{L}$ with the patterns of "The math", state its shape, and evaluate it for the
   worked example's $\mathbf{X}$, $\mathbf{y}$, $\mathbf{w} = (0, 1)$ and $\lambda = 0.5$. (c) Write the
   Jacobian of $\mathbf{f}(x_1, x_2, x_3) = (x_1x_3,\; x_2^2 - x_1)$ at $(1, 2, 3)$ and give its shape.
2. **Code:** Multi-output linear regression predicts $k$ targets at once: $\mathbf{W}$ `(d, k)`,
   $\mathbf{Y}$ `(n, k)`, and
   $\mathcal{L}(\mathbf{W}) = \frac{1}{n}\sum_{i,c}\big((\mathbf{X}\mathbf{W})_{ic} - Y_{ic}\big)^2 + \lambda\sum_{j,c}W_{jc}^2$.
   (a) Derive $\nabla_{\mathbf{W}}\mathcal{L}$ using pattern 5 (what is $\mathbf{G}$ here?). (b) Implement it and
   check it with your own `numerical_gradient` (copy it from the lab) on random data with
   $n = 8, d = 3, k = 2$, `SEED = 0`. (c) Repeat with $d = k = 3$ and show that the *transposed*
   formula passes a shape assertion but fails the gradient check. (d) Solve $\nabla_{\mathbf{W}}\mathcal{L} = \mathbf{0}$
   for $\mathbf{W}$ with `np.linalg.solve`, check the gradient is zero there, and compare with
   scikit-learn's `Ridge(alpha=..., fit_intercept=False)` (work out which `alpha` matches your
   $\lambda$ and $n$).
3. **Think:** Your team's training code has a hand-written gradient for a model with 10 million
   parameters, and a forward pass takes 50 ms. How long would a full numerical gradient check take?
   Propose a check cheap enough to run on every pull request in CI. What kinds of bugs would your
   cheap check catch, and which could it miss?

## Takeaways

1. The gradient stacks every partial derivative into an array with the shape of the parameters;
   $f(\mathbf{x} + \boldsymbol{\delta}) \approx f(\mathbf{x}) + \nabla f \cdot \boldsymbol{\delta}$.
2. It points in the direction of steepest increase and is perpendicular to the contour lines; it does
   not point at the minimum.
3. The Jacobian of $\mathbb{R}^n \to \mathbb{R}^m$ is `(m, n)`, the best local linear map; a linear map's
   Jacobian is its matrix, and an elementwise function's is diagonal (an elementwise product in code).
4. Patterns to know: $\nabla\,\mathbf{a}^\top\mathbf{w} = \mathbf{a}$, $\nabla\,\mathbf{x}^\top\mathbf{M}\mathbf{x} = (\mathbf{M} + \mathbf{M}^\top)\mathbf{x}$,
   $\nabla\lVert\mathbf{A}\mathbf{x} - \mathbf{b}\rVert^2 = 2\mathbf{A}^\top(\mathbf{A}\mathbf{x} - \mathbf{b})$, MSE $\frac{2}{n}\mathbf{X}^\top\mathbf{r}$,
   linear layer $\mathbf{X}^\top\mathbf{G}$.
5. Check every hand-derived gradient numerically, in float64, at random points with unequal
   dimensions, using a relative error.

**What builds on this:** [A4.3](a4_3_chain_rule_graphs.md) (where $\mathbf{G}$ comes from: the chain
rule backward through a graph); [A4.4](a4_4_gradient_descent.md) (stepping along $-\nabla\mathcal{L}$);
[B1.5](../B/b1_5_linear_regression.md) and [B2.1](../B/b2_1_logistic_regression.md) (the gradients of
real models); [C1.2](../C/c1_2_backprop_by_hand.md) (`dW = X.T @ dZ` in every layer).
