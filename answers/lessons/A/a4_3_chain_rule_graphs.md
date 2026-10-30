# Answer key: A4.3 The chain rule as a computational graph

Lesson: [`lessons/A/a4_3_chain_rule_graphs.md`](../../../lessons/A/a4_3_chain_rule_graphs.md)

## Exercise 1 (by hand)

**Solution (a).** Graph: $q_1 = x + y$, $q_2 = y - z$, $f = q_1 \cdot q_2$. $y$ feeds both $q_1$ and $q_2$, so it
has two paths.

- Forward: $q_1 = 3$, $q_2 = -2$, $f = -6$.
- Backward: $\bar{f} = 1$. Multiply node: $\bar{q}_1 = q_2 = -2$, $\bar{q}_2 = q_1 = 3$. Add node: $\bar{x} = \bar{q}_1 \cdot 1 = -2$.
  Subtract node ($q_2 = y - z$, local derivatives $+1$ and $-1$): $\bar{z} = \bar{q}_2 \cdot (-1) = -3$. $y$ collects
  both paths: $\bar{y} = \bar{q}_1 \cdot 1 + \bar{q}_2 \cdot 1 = -2 + 3 = 1$.
- Check: $f = xy - xz + y^2 - yz$, so $\partial f/\partial y = x + 2y - z = 1 + 4 - 4 = 1$ ✓.

**Solution (b).** Forward: $z = 1 \cdot 2 + (-1) \cdot 1 - 1 = 0$, $p = 0.5$, $\ell = (0.5 - 0)^2 = 0.25$. Backward:
$\bar{p} = 2(p - y) = 1$, $\bar{z} = 1 \cdot 0.5 \cdot 0.5 = 0.25$, $\bar{\mathbf{w}} = 0.25 \cdot (2, 1) = (0.5, 0.25)$,
$\bar{b} = 0.25$. Step with $\eta = 1$: $\mathbf{w} = (1 - 0.5, -1 - 0.25) = (0.5, -1.25)$, $b = -1 - 0.25 = -1.25$. New
$z = 0.5 \cdot 2 - 1.25 \cdot 1 - 1.25 = -1.5$, $p = \sigma(-1.5) \approx 0.1824$, $\ell \approx 0.0333$. The loss fell
from 0.25 to 0.033: the prediction moved toward the target 0.

**Solution (c).** $\log(1 + e^x)$ is $\log u$ with $u = 1 + e^x$: $\frac{d}{dx} = \frac{1}{u}\cdot e^x = \frac{e^x}{1 + e^x}$.
Divide top and bottom by $e^x$: $\frac{1}{e^{-x} + 1} = \sigma(x)$ ✓.

```text
== exercise 1 check ==
(a) q1 = x + y = 3.0  q2 = y - z = -2.0  f = -6.0
    df/dx = q2 = -2.0  df/dy = q2 + q1 = 1.0  df/dz = -q1 = -3.0
    numerical: [-2.0, 1.0, -3.0]
(b) z = 0.0  p = 0.5  loss = 0.25  dl/dz = 0.25  dl/dw = [0.5, 0.25]  dl/db = 0.25
    after one step (lr = 1): w = [0.5, -1.25]  b = -1.25  z = -1.5  p = 0.18243  loss = 0.03328
(c) numerical softplus' at [-2, 0, 3] = [0.1192, 0.5, 0.9526]   sigmoid = [0.1192, 0.5, 0.9526]
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Graph with three nodes and correct forward values | 2 |
| (a) Backward values $-2$, $1$, $-3$ with the two paths for $y$ added | 3 |
| (a) Check by expanding | 1 |
| (b) Forward and backward values | 3 |
| (b) Correct update, new loss, and the statement that it decreased | 2 |
| (c) Chain rule with $\log$ and $\exp$, simplified to $\sigma(x)$ | 3 |

**Likely mistakes:**

- Using only one path for $y$ in (a) ($\bar{y} = -2$ or $3$). This is the `=` versus `+=` bug.
- Forgetting the $-1$ local derivative of subtraction, giving $\bar{z} = +3$.
- In (b), stepping *up* the gradient ($\boldsymbol{\theta} + \eta\nabla\ell$): the loss rises. Gradient descent subtracts.
- In (c), stopping at $\frac{e^x}{1 + e^x}$ and not recognising it as $\sigma(x)$. Both are correct; the
  recognition is the point (softplus is a smooth ReLU whose slope is the sigmoid).

## Exercise 2 (code)

**Solution:** [`a4_3_chain_rule_graphs_ex2.py`](a4_3_chain_rule_graphs_ex2.py). The backward pass, one
line per node, in reverse order:

```python
dy_hat = 2.0 * (y_hat - y_) / n       # (n,)
dw2 = float(np.sum(dy_hat * h))       # y_hat = w2 * h; w2 is shared by all n examples -> sum
db2 = float(np.sum(dy_hat))           # broadcast scalar -> sum
dh = dy_hat * w2                      # (n,)
du = dh * (1.0 - h**2)                # tanh'(u) = 1 - tanh(u)^2, reusing the forward value h
dw1 = float(np.sum(du * x_))
db1 = float(np.sum(du))
```

```text
== exercise 2: a one-hidden-unit network ==
params (w1, b1, w2, b2) = [0.5, 0.1, -0.3, 0.2]  loss = 0.917038
  backward (dw1, db1, dw2, db2) = [0.4167, -0.11, -0.9616, 0.3571]
  rel error vs numerical 2.2e-11   vs torch 1.3e-16
  tanh'(u) = 1 - h^2 at the 8 points: [0.4869, 0.7006, 0.8994, 0.9982, 0.9433, 0.7655, 0.5485, 0.3592]
params (w1, b1, w2, b2) = [10.0, 0.1, -0.3, 0.2]  loss = 1.184345
  backward (dw1, db1, dw2, db2) = [0.0004, -0.0006, -2.0654, 0.3998]
  rel error vs numerical 3.0e-08   vs torch 8.5e-14
  tanh'(u) = 1 - h^2 at the 8 points: [0.0, 0.0, 0.0, 0.016, 0.0107, 0.0, 0.0, 0.0]
```

With $w_1 = 10$, $u = 10x + 0.1$ is large in absolute value at almost every point, $\tanh$ is flat there
(**saturated**), and its local derivative $1 - h^2$ is about 0. Every gradient that has to pass back
*through* the tanh ($w_1$ and $b_1$) is multiplied by that 0 and vanishes: $0.0004$ and $-0.0006$, against
$-2.07$ for $w_2$, which sits after the tanh. Gradient descent can barely change $w_1$ and $b_1$, so the
hidden unit is stuck. This is the vanishing-gradient problem in miniature, and why initialization scales
weights down ([C3.3](../../../lessons/C/c3_3_initialization.md)). The learner's parameter point will
differ; the saturation explanation is what is graded. (The numerical check is slightly worse at
$w_1 = 10$, $3 \times 10^{-8}$, because the tiny gradients are compared relatively.)

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| `forward` returns the loss and a cache with what `backward` needs | 2 |
| `backward` correct for all four parameters, with sums for the shared scalars | 4 |
| Uses $1 - h^2$ from the cache (does not recompute tanh) | 1 |
| Checked against a numerical gradient (relative error ≤ 1e-7) | 2 |
| Checked against PyTorch in float64 | 2 |
| Explains saturation: $1 - \tanh^2 \approx 0$ kills the gradients of $w_1$, $b_1$ but not $w_2$, $b_2$ | 3 |

**Likely mistakes:**

- `dw2 = dy_hat * h` without the sum: an `(n,)` array where a scalar is needed. A parameter shared by
  every example collects one contribution per example.
- Using $1 - \tanh(x)^2$ (of the input) instead of $1 - \tanh(u)^2$ (of the node's own input $u = w_1x + b_1$).
- Comparing with PyTorch in float32 (the default for `torch.tensor` of Python floats) and loosening the
  tolerance to make it pass. Build the tensors from float64 NumPy arrays.
- Explaining the small gradients as "the loss is already small". It is not (1.18); the gradient is small
  because of the flat tanh, while the loss is large.

## Exercise 3 (think)

**Strong answer:** Inference only runs the forward pass and can discard each layer's output as soon as
the next layer has consumed it, so memory holds roughly the weights plus one or two layers'
activations. Training must run the backward pass, and every node's local derivative needs values from
the forward pass (the cache): the input to every linear layer for $\mathbf{X}^\top\bar{\mathbf{z}}$, the output of every
sigmoid or tanh, and so on. All of these are kept for the whole batch until the backward pass reaches
them, so activation memory grows with depth × batch size × layer width. On top of that come the
gradients (one per parameter) and the optimizer's state (Adam keeps two more numbers per parameter,
[C3.1](../../../lessons/C/c3_1_optimizers.md)). Ways to cut it: (1) a smaller batch, possibly with
gradient accumulation over several small batches to keep the effective batch size (costs time, and
batch-dependent layers like batch norm behave differently); (2) **gradient checkpointing**: store only
some activations and recompute the rest during the backward pass (costs roughly one extra forward
pass of compute); (3) mixed precision, storing activations in 16-bit floats (halves activation memory,
needs care with numerical range); all three are taught in [C4.2](../../../lessons/C/c4_2_mixed_precision.md);
also acceptable: freezing layers so their gradients are not needed, or a smaller model.

**Weak answer:** "Training uses more memory because of the gradients." Weak because gradients are the
same size as the weights, which already fit; the dominant term is the stored activations, which scale
with the batch and are what the graph explanation is about. It also offers no remedies.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Explains that backward needs forward values, so activations are stored for the whole batch and depth | 4 |
| Mentions gradients and optimizer state as additional (smaller) terms | 1 |
| Three valid remedies | 3 |
| States a cost for each remedy | 2 |

## Scoring

Total 38 pts. Pass ≥ 70% (27 pts). Topic ids to revisit for each lost criterion: exercise 1 → A4.3
"The math" (chain rule, adding paths) and A4.1 (derivative rules); exercise 2 → A4.3 "From scratch"
and A4.2 (gradient checking); exercise 3 → A4.3 "Backpropagation, the algorithm" and "In practice".
