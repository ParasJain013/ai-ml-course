# A4.3 The chain rule as a computational graph

**Core** · Part A, week 4 (course week 04) · ~2 h · Lab:
[`labs/A/a4_3_chain_rule_graphs.py`](../../labs/A/a4_3_chain_rule_graphs.py)

## Why it matters

A neural network's loss is a long chain of simple operations: multiply, add, apply an activation,
multiply again, compare with the target. Training needs the derivative of that one number with
respect to millions of parameters, and finite differences would need two full evaluations per
parameter ([A4.1](a4_1_derivatives.md)). The **chain rule**, organised as a walk backwards over a
**computational graph**, gets *all* of them for roughly the cost of one extra pass. That walk is
**backpropagation**, and every framework's autograd (PyTorch, JAX, TensorFlow) is an implementation of
it. When training misbehaves (gradients that vanish, explode, or come back `None`) you debug it by
reasoning about this graph.

## Prerequisite check

- Can you differentiate $x^3$, $e^x$ and $\log x$, and a product $u(x)v(x)$? If not, revisit
  [A4.1](a4_1_derivatives.md).
- Can you write the gradient of a scalar function of a vector, and the Jacobian of a vector function,
  with their shapes? If not, revisit [A4.2](a4_2_gradients_jacobians.md).
- Can you say what `X.T @ g` computes for `X` `(n, d)` and `g` `(n,)`, and its shape? If not, revisit
  [A2.1](a2_1_matrices_as_linear_maps.md).
- Can you check a gradient numerically and read a relative error? If not, revisit
  [A4.2](a4_2_gradients_jacobians.md).

## Intuition

### Rates multiply

If pressing a car's pedal 1 cm further adds 12 km/h, and each extra km/h raises fuel consumption by
0.08 L/100 km, then pressing the pedal 1 cm further raises consumption by $12 \times 0.08 = 0.96$
L/100 km. Rates of change along a chain
multiply. That is the whole chain rule: for $z = f(g(x))$, the sensitivity of $z$ to $x$ is (sensitivity
of $z$ to $g$) × (sensitivity of $g$ to $x$).

### Break the expression into a graph

Write $f = (x \cdot y + z)^2$ as three steps: $a = x \cdot y$, then $b = a + z$, then $f = b^2$. Each step is
a node that only knows its own inputs and its own **local derivative** (multiply: "the other input";
add: "1"; square: "2 × input"). The **forward pass** computes every node's value from left to right
and remembers them. The **backward pass** starts at the output with $\partial f/\partial f = 1$ and moves right
to left: each node takes the gradient arriving from its output side (the **upstream gradient**),
multiplies it by its local derivative, and passes the result to its inputs. The key plot shows the
numbers on exactly this graph.

### When a value is used twice, its gradients add

If $x$ feeds two nodes, wiggling $x$ moves $f$ through both, and the two effects add. So a node
receiving gradients from several consumers *sums* them. Forgetting that sum is the classic bug in a
hand-written backward pass.

> **Bridge:** a computational graph is the dependency graph of a build system (Nx, Turborepo, Make) or
> of a Terraform plan: a DAG, evaluated in topological order, each node computed once its inputs are
> ready, with intermediate results cached. The forward pass is exactly that. **Where it breaks:**
> build tools only run forward. Backpropagation then walks the same DAG in *reverse* topological order
> carrying numbers (sensitivities), multiplying along edges and *adding* where paths meet: closer to an
> impact analysis ("which outputs change, and by how much, if I edit this file") than to a build.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $g$, $f$ | functions chained as $f(g(x))$ | scalar → scalar |
| $u$ | the intermediate value $g(x)$ | scalar |
| $\bar{v}$ | the **upstream gradient** at node $v$: $\partial f/\partial v$, the final output's sensitivity to $v$ | same shape as $v$ |
| $\sigma(z)$ | logistic **sigmoid**, $1/(1 + e^{-z})$ | elementwise |
| $\mathbf{x}$, $\mathbf{w}$, $b$, $y$ | one example's features, weights, bias, target | `(d,)`, `(d,)`, scalar, scalar |
| $z$, $p$, $\ell$ | logit $\mathbf{w}\cdot\mathbf{x} + b$, prediction $\sigma(z)$, loss of one example | scalars |
| $\mathbf{X}$, $\mathbf{y}$ | a batch of $n$ examples and their targets | `(n, d)`, `(n,)` |
| $\mathbf{z}$, $\mathbf{p}$ | logits and predictions for the batch | `(n,)` |
| $\mathbf{J}$ | Jacobian of a node's output with respect to its input | `(m, p)` |

**The chain rule, one variable.** Let $u = g(x)$ and $f$ depend on $u$. Nudge $x$ by a small $\Delta x$.
From the meaning of a derivative ([A4.1](a4_1_derivatives.md)), $u$ moves by $\Delta u \approx g'(x)\,\Delta x$,
and then $f$ moves by $\Delta f \approx f'(u)\,\Delta u \approx f'(u)\,g'(x)\,\Delta x$. Divide by $\Delta x$:

$$\frac{df}{dx} = \frac{df}{du}\cdot\frac{du}{dx} = f'(g(x))\,g'(x).$$

Example: $e^{-x^2}$ is $f(u) = e^u$ with $u = -x^2$, so its derivative is $e^{u} \cdot (-2x) = -2x\,e^{-x^2}$.

**Several paths: add them.** If $x$ affects $f$ through intermediate values $u_1, \dots, u_k$, the
linear approximation of [A4.2](a4_2_gradients_jacobians.md) says
$\Delta f \approx \sum_i \frac{\partial f}{\partial u_i}\Delta u_i$, and each $\Delta u_i \approx \frac{du_i}{dx}\Delta x$. So

$$\frac{df}{dx} = \sum_{i=1}^{k}\frac{\partial f}{\partial u_i}\,\frac{du_i}{dx}.$$

One term per path. With a single path it is the one-variable rule.

**Local derivatives of the usual nodes.** Each was derived in A4.1 or follows from the chain rule:

| Node | Output | Local derivative(s) |
|---|---|---|
| add | $u + v$ | $1$ for each input |
| multiply | $u \cdot v$ | $v$ for $u$, $u$ for $v$ (the other input) |
| square | $u^2$ | $2u$ |
| exp, log | $e^u$, $\log u$ | $e^u$, $1/u$ |
| dot product | $\mathbf{w}\cdot\mathbf{x}$ | $\mathbf{x}$ for $\mathbf{w}$, $\mathbf{w}$ for $\mathbf{x}$ |
| sigmoid | $\sigma(z)$ | $\sigma(z)\,(1 - \sigma(z))$ |

The sigmoid line, step by step. Write $\sigma(z) = (1 + e^{-z})^{-1}$: a power $(\cdot)^{-1}$ of $u = 1 + e^{-z}$.
The power rule gives $-u^{-2}$, and $du/dz = -e^{-z}$ (chain rule again: $e^{v}$ with $v = -z$). Multiply:

$$\sigma'(z) = -(1 + e^{-z})^{-2}\cdot(-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2} = \underbrace{\frac{1}{1 + e^{-z}}}_{\sigma(z)}\cdot\underbrace{\frac{e^{-z}}{1 + e^{-z}}}_{1 - \sigma(z)}.$$

The last factor is $1 - \sigma(z)$ because $1 - \frac{1}{1 + e^{-z}} = \frac{1 + e^{-z} - 1}{1 + e^{-z}}$. So the
backward pass reuses the forward value $p = \sigma(z)$: no new exponential needed. Its largest value is
$0.25$, at $z = 0$; for large $\lvert z\rvert$ it is nearly 0, which is the "saturated sigmoid" that stops
learning ([C3.3](../C/c3_3_initialization.md)).

**Backpropagation, the algorithm.**

1. *Forward:* evaluate the nodes in an order where every input comes before its node (topological
   order). Store every value.
2. *Seed:* $\bar{f} = \partial f/\partial f = 1$; every other $\bar{v} = 0$.
3. *Backward:* visit the nodes in reverse order. For each node $v$ and each of its inputs $u$:
   $\bar{u} \mathrel{+}= \bar{v} \times \frac{\partial v}{\partial u}$ (upstream gradient times local derivative).
   The `+=` is the "add the paths" rule.

Each node is visited once forward and once backward, so a full gradient costs a small constant times
one forward pass, however many inputs there are. That is why **reverse mode** is used in ML: millions of
inputs (parameters), one output (the loss). (The alternative, **forward mode**, carries derivatives with
respect to *one* input forward through the graph, so it needs one pass per input; it wins only when
inputs are few and outputs many.) The price is memory: step 3 needs the values stored in step 1.

**Vectors: vector-Jacobian products.** When a node maps $\mathbf{u}$ `(p,)` to $\mathbf{v}$ `(m,)` with Jacobian
$\mathbf{J}$ `(m, p)`, apply the "add the paths" rule to each input coordinate: $u_j$ affects every $v_i$, so

$$\bar{u}_j = \sum_{i=1}^{m}\bar{v}_i\,\frac{\partial v_i}{\partial u_j} = \sum_i \bar{v}_i J_{ij} = (\mathbf{J}^\top\bar{\mathbf{v}})_j, \qquad \bar{\mathbf{u}} = \mathbf{J}^\top\bar{\mathbf{v}} \quad (p,).$$

Frameworks never build $\mathbf{J}$; each node knows how to compute $\mathbf{J}^\top\bar{\mathbf{v}}$ directly. For the two
nodes of a layer:

- *Linear:* $\mathbf{z} = \mathbf{X}\mathbf{w}$ has Jacobian $\mathbf{X}$ with respect to $\mathbf{w}$ (A4.2), so $\bar{\mathbf{w}} = \mathbf{X}^\top\bar{\mathbf{z}}$,
  `(d, n) @ (n,)` → `(d,)`. The same $\mathbf{X}^\top(\cdot)$ as in the MSE gradient, for the same reason.
- *Elementwise:* $\mathbf{p} = \sigma(\mathbf{z})$ has a diagonal Jacobian, so $\bar{\mathbf{z}} = \bar{\mathbf{p}} \odot \mathbf{p} \odot (1 - \mathbf{p})$,
  elementwise, `(n,)`.
- *Broadcast bias:* $b$ is added to every $z_i$ (a fan-out to $n$ places), so $\bar{b} = \sum_i \bar{z}_i$.

**The one-neuron model on a batch.** Forward: $\mathbf{z} = \mathbf{X}\mathbf{w} + b$ `(n,)`, $\mathbf{p} = \sigma(\mathbf{z})$ `(n,)`,
$\mathcal{L} = \frac{1}{n}\sum_i (p_i - y_i)^2$. Backward, one node at a time:

$$\bar{p}_i = \frac{2}{n}(p_i - y_i), \qquad \bar{z}_i = \bar{p}_i\,p_i(1 - p_i), \qquad \bar{\mathbf{w}} = \mathbf{X}^\top\bar{\mathbf{z}} \;(d,), \qquad \bar{b} = \sum_i \bar{z}_i.$$

(Squared error on a sigmoid is used here because you know it; for classification the course uses
cross-entropy instead, [A6.4](a6_4_entropy_cross_entropy_kl.md) and [B2.1](../B/b2_1_logistic_regression.md),
which also avoids the vanishing $p(1 - p)$ factor.)

## Worked example by hand

**1. $f = (x \cdot y + z)^2$ at $x = 3$, $y = -2$, $z = 4$.**

Forward: $a = x \cdot y = -6$; $b = a + z = -2$; $f = b^2 = 4$.

Backward, from the output:

| Node | Upstream × local | Gradient |
|---|---|---|
| $f$ | seed | $\bar{f} = 1$ |
| $b$ | $\bar{f} \times 2b = 1 \times (-4)$ | $\bar{b} = -4$ |
| $a$ | $\bar{b} \times 1$ | $\bar{a} = -4$ |
| $z$ | $\bar{b} \times 1$ | $\bar{z} = -4$ |
| $x$ | $\bar{a} \times y = -4 \times (-2)$ | $\bar{x} = 8$ |
| $y$ | $\bar{a} \times x = -4 \times 3$ | $\bar{y} = -12$ |

Check without the graph: $\partial f/\partial x = 2(xy + z)\cdot y = 2(-2)(-2) = 8$ ✓.

**2. Fan-out, $f = (x + y)\cdot x$ at $x = 3$, $y = 1$.** Forward: $q = x + y = 4$, $f = q \cdot x = 12$.
Backward: $\bar{q} = x = 3$ (multiply: the other input). $x$ has two paths: directly into the multiply,
local derivative $q = 4$, and through $q$, contributing $\bar{q} \times 1 = 3$. So $\bar{x} = 4 + 3 = 7$ and
$\bar{y} = \bar{q} \times 1 = 3$. Check: $f = x^2 + xy$, so $\partial f/\partial x = 2x + y = 7$ ✓.

**3. One neuron**, $\mathbf{x} = (1, 2)$, $\mathbf{w} = (0.5, -0.25)$, $b = 0$, target $y = 1$, loss $\ell = (p - y)^2$.
Forward: $z = 0.5 \cdot 1 - 0.25 \cdot 2 + 0 = 0$; $p = \sigma(0) = 1/(1 + 1) = 0.5$; $\ell = (0.5 - 1)^2 = 0.25$.
Backward: $\bar{p} = 2(p - y) = -1$; $\bar{z} = \bar{p}\cdot p(1 - p) = -1 \cdot 0.25 = -0.25$;
$\bar{\mathbf{w}} = \bar{z}\,\mathbf{x} = (-0.25, -0.5)$; $\bar{b} = \bar{z} = -0.25$. All negative: increasing any parameter
raises $z$, raises $p$ toward the target 1, and lowers the loss.

```text
== worked example: f = (x*y + z)^2 ==
node    value  df/dnode
x         3.0       8.0
y        -2.0     -12.0
z         4.0      -4.0
a        -6.0      -4.0
b        -2.0      -4.0
f         4.0       1.0
numerical gradient (x, y, z) = [8.0, -12.0, -4.0]   rel error 2.1e-11

== worked example: fan-out ==
q = 4.0  f = 12.0
df/dx = direct path 4.0 + path through q 3.0 = 7.0   df/dy = 3.0
check with f = x^2 + x y: df/dx = 2x + y = 7.0   df/dy = x = 3.0
numerical: [7.0, 3.0]

== worked example: one neuron ==
forward:  z = 0.0000  p = sigmoid(z) = 0.5000  loss = 0.2500
backward: dl/dp = -1.0000  dp/dz = 0.2500  dl/dz = -0.2500  dl/dw = [-0.25, -0.5]  dl/db = -0.2500
numerical (dw1, dw2, db) = [-0.25, -0.5, -0.25]   rel error 3.8e-11
sigmoid'(z) = s(1 - s) at z = [-3, 0, 2]: [0.0452, 0.25, 0.105]   numerical: [0.0452, 0.25, 0.105]
```

## Key plot

![The computational graph of f = (x*y + z)^2 with nodes x, y, z, a = x*y, b = a + z, f = b^2; each node shows its forward value in blue and its gradient in red, and each edge shows its local derivative](img/a4_3_graph_forward_backward.png)

Follow one red number back to its source: $\partial f/\partial x = 8$ is $\partial f/\partial a = -4$ times the local
derivative on the edge from $x$ to $a$, which is $y = -2$. Every gradient in the graph is the product of
the local derivatives along the path from $f$ back to that node.

## From scratch in NumPy

The lab writes each graph as a pair of functions: `forward` returns every intermediate value (the
**cache**), and `backward` consumes it. For graph 1:

```python
def graph1_backward(c: dict[str, float]) -> dict[str, float]:
    """Backward pass: upstream gradient times local derivative, from the output back to the inputs."""
    df_df = 1.0  # seed: df/df = 1
    df_db = df_df * 2 * c["b"]  # square: local derivative 2b
    df_da = df_db * 1.0  # add: local derivative 1 for each input
    df_dz = df_db * 1.0
    df_dx = df_da * c["y"]  # multiply: local derivative is the OTHER input
    df_dy = df_da * c["x"]
    return {"f": df_df, "b": df_db, "a": df_da, "z": df_dz, "x": df_dx, "y": df_dy}
```

The batched neuron is the same pattern with arrays, one line per node, shapes in the comments:

```python
def neuron_backward(cache: dict[str, FloatArray]) -> tuple[FloatArray, float]:
    """Gradients of the mean loss. Returns dw (d,) and db (scalar)."""
    X_, p_, y_ = cache["X"], cache["p"], cache["y"]
    n = len(y_)
    dp = 2.0 * (p_ - y_) / n  # (n,): d mean((p - y)^2) / dp_i
    dz = dp * p_ * (1.0 - p_)  # (n,): elementwise sigmoid, so its Jacobian is diagonal
    dw = X_.T @ dz  # (d,) = J^T dz, with J = X the Jacobian of z = X w with respect to w
    db = float(dz.sum())  # b is broadcast to every example, so its gradient sums
    return dw, db
```

```text
== from scratch: a batch of examples ==
loss = 0.298130   dw (3,) = [-0.1238, 0.0665, -0.0807]   db = 0.1279
check against numerical gradient: dw rel error 4.8e-11   db rel error 2.4e-11
```

And the reason reverse mode exists. The finite-difference gradient needs $2d$ forward passes; the
backward pass needs one forward and one backward pass whatever $d$ is (timings vary by machine, hence
the `~` lines):

```text
== from scratch: cost of a gradient ==
d =    10: finite differences need    20 forward passes, reverse mode 1 forward + 1 backward;  agree to 2e-10
~ d =    10: finite differences     0.15 ms   reverse mode  0.022 ms
d =   100: finite differences need   200 forward passes, reverse mode 1 forward + 1 backward;  agree to 7e-09
~ d =   100: finite differences     1.51 ms   reverse mode  0.034 ms
d =  1000: finite differences need  2000 forward passes, reverse mode 1 forward + 1 backward;  agree to 3e-08
~ d =  1000: finite differences    22.46 ms   reverse mode  0.071 ms
```

The finite-difference time grows about tenfold with each tenfold $d$; reverse mode barely moves. At a
billion parameters the gap is the difference between a training step and never finishing.

## With PyTorch

PyTorch records the graph while it runs the forward pass (each result remembers the operation that made
it, its `grad_fn`) and `.backward()` walks it in reverse, exactly as above
([C2.1](../C/c2_1_tensors_autograd_modules.md) covers the API properly):

```text
== with pytorch ==
graph recorded by autograd: PowBackward0 <- AddBackward0 <- MulBackward0
f = 4.0   df/dx = 8.0   df/dy = -12.0   df/dz = -4.0
matches the hand backward pass: OK
fan-out: df/dx = 7.0  df/dy = 3.0   (autograd adds the two paths too)
batched neuron: torch gradients match neuron_backward: OK
```

The recorded graph, read from the output back, is square ← add ← multiply: the three nodes of the key
plot. The `Backward0` names are internal and may change between versions (VERIFY (as of 2026-10)); the
structure will not.

## In practice

- **When to use it:** whenever you differentiate something with more than one step, by hand (small
  graphs, custom layers) or by reading what autograd will do. Draw the graph, label each edge with its
  local derivative, and the gradient of anything is "multiply along paths, add across paths".
- **Failure modes:**
  - *Overwriting instead of accumulating:* `grad[x] = ...` where it should be `+=`, so only the last
    path counts. You notice a gradient check that fails only for variables used more than once.
  - *Stale or missing cache:* the backward pass uses a value that was changed after the forward pass
    (an in-place update), so local derivatives are evaluated at the wrong point. PyTorch raises an
    error about a variable "modified by an inplace operation"; in your own code, a gradient check fails.
  - *Vanishing gradients:* local derivatives below 1 multiply along a long chain (sigmoid's is at most
    0.25), so early layers get almost nothing. You notice per-layer gradient norms falling by orders of
    magnitude toward the input ([C3.3](../C/c3_3_initialization.md), [E2.1](../E/e2_1_rnn_bptt.md)).
  - *Exploding gradients:* the same with factors above 1. You notice loss spikes or NaNs; clip the
    gradient norm and fix the initialization.
  - *Memory:* every intermediate value is stored for the backward pass, so training uses far more
    memory than inference. You notice out-of-memory errors only during training (exercise 3).
- **Defaults that work:** let autograd do it in real code; hand-derive only to understand or for a
  custom operation, and then always gradient-check it (A4.2). When you write a backward function, give
  it the forward's cache rather than recomputing, and assert every gradient has its parameter's shape.

## Exercises

Save your work to `my-work/A/a4_3_chain_rule_graphs.md` (or `.py`) and ask for a review.

1. **By hand:** (a) Draw the graph of $f = (x + y)(y - z)$, run the forward and backward passes at
   $x = 1$, $y = 2$, $z = 4$, and check $\partial f/\partial y$ by expanding $f$ and differentiating directly.
   Which variable has two paths? (b) One neuron with $\mathbf{x} = (2, 1)$, $\mathbf{w} = (1, -1)$, $b = -1$, target
   $y = 0$ and loss $(p - y)^2$: run the forward and backward passes. Then take one gradient descent step
   $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\nabla\ell$ with $\eta = 1$ on $\mathbf{w}$ and $b$, recompute the loss, and
   say whether it went down. (c) The **softplus** function is $\log(1 + e^{x})$. Use the chain rule to show
   its derivative is $\sigma(x)$.
2. **Code:** A one-hidden-unit network on a batch: $\mathbf{h} = \tanh(w_1\mathbf{x} + b_1)$,
   $\hat{\mathbf{y}} = w_2\mathbf{h} + b_2$, $\mathcal{L} = \frac{1}{n}\lVert\hat{\mathbf{y}} - \mathbf{y}\rVert^2$, with $\mathbf{x}, \mathbf{y}$ `(n,)` and
   four scalar parameters. Use $\tanh'(u) = 1 - \tanh^2(u)$. Write `forward` returning the loss and a
   cache, and `backward` returning the four gradients. Test on $\mathbf{x}$ = 8 points evenly spaced in
   $[-2, 2]$, $\mathbf{y} = \sin(\mathbf{x})$, at a parameter point of your choice; check against a numerical
   gradient and against PyTorch. Then set $w_1 = 10$ and explain what happens to the gradients of $w_1$
   and $b_1$, and why.
3. **Think:** A model fits on the GPU for inference with batch size 64, but training at batch size 64
   runs out of memory. Using the computational graph, explain why. Name three ways to cut training
   memory and what each costs (time, accuracy or something else).

## Takeaways

1. The chain rule: rates multiply along a path, $\frac{df}{dx} = \frac{df}{du}\frac{du}{dx}$; with several paths,
   add one product per path.
2. Backpropagation = forward pass storing values, then a reverse walk where each node does
   upstream gradient × local derivative, accumulating with `+=` at fan-outs.
3. Local derivatives are simple: add → 1, multiply → the other input, square → $2u$, sigmoid →
   $\sigma(1 - \sigma)$, linear $\mathbf{X}\mathbf{w}$ → $\mathbf{X}^\top\bar{\mathbf{z}}$, broadcast bias → sum.
4. With vectors, each node computes a vector-Jacobian product $\mathbf{J}^\top\bar{\mathbf{v}}$ without ever building $\mathbf{J}$.
5. Reverse mode costs about one extra pass for all parameters at once, which is why it trains every
   neural network; the price is storing activations, so training needs more memory than inference.

**What builds on this:** [A4.4](a4_4_gradient_descent.md) (using the gradient to train);
[C1.2](../C/c1_2_backprop_by_hand.md) (backprop through a whole small network);
[C1.3](../C/c1_3_autograd_engine.md) (a general autograd engine); [C2.1](../C/c2_1_tensors_autograd_modules.md)
(PyTorch autograd); [E2.1](../E/e2_1_rnn_bptt.md) (backprop through time and vanishing gradients).
