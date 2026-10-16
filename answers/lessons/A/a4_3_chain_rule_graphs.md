# Answer key: A4.3 The chain rule as a computational graph

Lesson: [`lessons/A/a4_3_chain_rule_graphs.md`](../../../lessons/A/a4_3_chain_rule_graphs.md)

## Exercise 1 (by hand)

**Solution (a).** Graph: $z \to p = \sigma(z) \to \mathcal{L} = -[y\log p + (1 - y)\log(1 - p)]$. With the log
row, $\partial\mathcal{L}/\partial p = -\frac{y}{p} + \frac{1 - y}{1 - p}$ (the second term: $\log(1 - p)$ is a log of
$1 - p$, whose derivative in $p$ is $-1$, so the chain rule gives $\frac{-1}{1 - p}$, and the minus sign
in front of the bracket flips it). Over a common denominator:

$$\frac{\partial\mathcal{L}}{\partial p} = \frac{-y(1 - p) + (1 - y)p}{p(1 - p)} = \frac{p - y}{p(1 - p)}.$$

The sigmoid row multiplies by $p(1 - p)$, which cancels the denominator:
$\frac{\partial\mathcal{L}}{\partial z} = \frac{p - y}{p(1 - p)}\cdot p(1 - p) = p - y$.

For the worked example $p = 0.5$, $y = 1$: $\bar{z} = -0.5$, $\bar{\mathbf{w}} = \bar{z}\,\mathbf{x} = (-0.5, -1)$,
$\bar{b} = -0.5$. With squared error $\bar{z}$ was $-0.25$.

When the neuron is confidently wrong ($y = 1$, $p \to 0$), log loss gives $\bar{z} = p - 1 \to -1$, the
strongest possible push. Squared error gives $\bar{z} = 2(p - 1)\,p(1 - p) \to 0$ because the sigmoid's
slope $p(1 - p)$ vanishes: the more wrong the neuron, the *less* it learns. Log loss's $\frac{1}{p(1 - p)}$
exactly cancels the saturation. (That is why classifiers use log loss, [B2.1](../../../lessons/B/b2_1_logistic_regression.md).)

**Solution (b).** Nodes $a = x + y = 3$, $c = y + z = 5$, $h = ac = 15$. Backward from $\bar{h} = 1$:
multiply swaps, $\bar{a} = c = 5$, $\bar{c} = a = 3$. Add copies: $\bar{x} = \bar{a} = 5$, $\bar{z} = \bar{c} = 3$,
and $y$ feeds both adds, so $\bar{y} = \bar{a} + \bar{c} = 8$. Check: $h = xy + xz + y^2 + yz$,
$\partial h/\partial y = x + 2y + z = 1 + 4 + 3 = 8$ ✓.

```text
== exercise 1 check ==
(a) p = 0.5   dL/dz = p - y = -0.5   dL/dw = [-0.5, -1.0]   dL/db = -0.5
    numerical (dL/dw1, dL/dw2, dL/db) = [-0.5, -1.0, -0.5]
    p = 0.5    y = 1: log loss dL/dz = -0.5000   squared error dL/dz = -0.2500
    p = 0.1    y = 1: log loss dL/dz = -0.9000   squared error dL/dz = -0.1620
    p = 0.001  y = 1: log loss dL/dz = -0.9990   squared error dL/dz = -0.0020
(b) a = x + y = 3.0  c = y + z = 5.0  h = 15.0   dh/dx = 5.0  dh/dy = 5.0 + 3.0 = 8.0  dh/dz = 3.0
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) $\partial\mathcal{L}/\partial p$ correct, including the sign of the $\log(1 - p)$ term | 2 |
| (a) Combines with $\sigma' = p(1 - p)$ to get $p - y$, every step shown | 3 |
| (a) Numbers $-0.5$, $(-0.5, -1)$, $-0.5$ | 2 |
| (a) Confidently-wrong comparison with the saturation argument | 2 |
| (b) Graph drawn, forward values correct | 2 |
| (b) $\bar{y} = 8$ by adding both paths, checked by expansion | 3 |

**Likely mistakes:**

- Dropping the inner $-1$ of $\log(1 - p)$, giving $\partial\mathcal{L}/\partial p = -\frac{y}{p} - \frac{1 - y}{1 - p}$ and a
  result that is not $p - y$.
- Getting $\bar{y} = 5$ or $3$ in (b): only one path counted. Whenever a variable appears twice, the
  gradients add.
- Saying "log loss is better because it is larger". The point is not size but that its gradient does not
  vanish when the sigmoid saturates.

## Exercise 2 (code)

**Solution:** [`a4_3_chain_rule_graphs_ex2.py`](a4_3_chain_rule_graphs_ex2.py). The backward pass, every
line one row of the lesson's table:

```python
d_yhat = 2.0 / n * (y_hat - y)          # (n,)
d_w2 = Hid.T @ d_yhat                   # (3,)   matmul: X^T G with H as the input
d_b2 = np.array(d_yhat.sum())           # ()     broadcast bias: sum over rows
d_H = np.outer(d_yhat, params["w2"])    # (n, 3) matmul: G W^T
d_A = d_H * (1.0 - Hid**2)              # (n, 3) tanh' = 1 - tanh^2, elementwise
d_W1 = X.T @ d_A                        # (2, 3)
d_b1 = d_A.sum(axis=0)                  # (3,)
```

```text
== exercise 2: gradient check of the two-layer network ==
W1  shape (2, 3)  rel error < 1e-8: True
b1  shape (3,)    rel error < 1e-8: True
w2  shape (3,)    rel error < 1e-8: True
b2  shape ()      rel error < 1e-8: True

== exercise 2: training with these gradients ==
step   0   loss = 5.1745
step  50   loss = 0.0299
step 100   loss = 0.0278
step 150   loss = 0.0266
step 200   loss = 0.0254
```

The learner's loss values will differ (different initialisation draws); what is graded is that every
gradient passes the check and the loss falls steadily. A loss that rises or turns into `nan` means a
sign error or a learning rate that is too large ([A4.4](../../../lessons/A/a4_4_gradient_descent.md)).

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| `forward` returns loss and a cache; no recomputation of forward values in `backward` | 2 |
| Correct output-layer gradients `d_w2`, `d_b2` | 2 |
| Correct `d_H` (uses $\mathbf{w}_2$, the "other input" rule) and the tanh step reusing `H` | 3 |
| Correct `d_W1` and `d_b1` (bias summed over rows) | 2 |
| Shapes in comments; asserts that each gradient has its parameter's shape | 2 |
| All four gradients checked with the copied checker, relative error, at random points | 3 |
| Training loop runs and the loss falls; printed every 50 steps | 2 |

**Likely mistakes:**

- `d_b1 = d_A.sum()` (summing over both axes) gives a scalar instead of `(3,)`.
- Using `np.tanh(A)` again in backward, or worse `1 - np.tanh(Hid) ** 2` (tanh applied twice).
- `d_H = d_yhat * params["w2"]` with shapes `(n,)` and `(3,)`: a broadcasting error for $n \ne 3$, and a
  silently wrong result when $n = 3$. Use `np.outer` or `d_yhat[:, None] * w2`.
- Checking only `W1`, the hardest one, and missing a bias bug.

## Exercise 3 (think)

**Strong answer:** Inference only needs each layer's output until the next layer has used it, so
memory holds the weights plus roughly one layer's activations at a time. Training needs the backward
pass, and every node's local derivative uses values from the forward pass (the inputs of each matmul
for $\mathbf{X}^\top\bar{\mathbf{Z}}$, the outputs of each tanh or sigmoid), so the forward pass must keep *every*
layer's activations for the whole batch until backward reaches them: memory grows with depth × batch
size × width. On top of that, training stores a gradient for every parameter (as large as the model)
and the optimizer's state (Adam keeps two more copies, [C3.1](../../../lessons/C/c3_1_optimizers.md)). Ways to cut
it: (1) smaller batches, which lowers activation memory but makes gradient estimates noisier and each
epoch slower (accumulate gradients over several small batches to keep the effective batch size);
(2) don't cache some activations and recompute them during the backward pass (the technique is called
gradient checkpointing, [C4.2](../../../lessons/C/c4_2_mixed_precision.md)), which trades about one extra forward
pass of compute for memory; (3) store activations in 16-bit floats (mixed precision), which halves
memory at some numerical risk. Any two with their costs is full marks.

**Weak answer:** "Training uses more memory because the model is updating its weights." Weak because
updating weights in place needs almost no extra memory; it misses the cached activations, which are the
main cost, and so cannot propose fixes that target them.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Identifies cached forward activations as the main extra memory, with the reason (backward needs them) | 4 |
| Mentions gradients and optimizer state as additional costs | 2 |
| Two concrete reductions | 2 |
| States the cost of each (noise/speed, recompute time, precision) | 2 |

## Scoring

Total 40 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → A4.3
"The math" (local derivatives, add over paths) and A4.1 (log rule); exercise 2 → A4.3 "From scratch"
and A4.2 (pattern 5, shapes); exercise 3 → A4.3 "Intuition" (bridge) and "In practice".
