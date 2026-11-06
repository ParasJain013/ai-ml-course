# Answer key: C1.3 An autograd engine from scratch

Lesson: [`lessons/C/c1_3_autograd_engine.md`](../../../lessons/C/c1_3_autograd_engine.md)

## Exercise 1 (by hand)

**Solution (a).** Nodes: $p = xy$, $q = e^x$, $s = p + q$, $f = \log s$. $x$ is used twice (by $p$ and by $q$).

*Topological order:* $x, y, p, q, s, f$ (or $y, x, q, p, s, f$; any order with each node after its inputs).

*Forward* at $x = 0$, $y = 2$: $p = 0$, $q = e^0 = 1$, $s = 1$, $f = \log 1 = 0$.

*Backward*, visiting $f, s, q, p, y, x$ with all gradients starting at 0:

| Visit | Rule | Adds |
|---|---|---|
| $f = \log s$ | $g_s \mathrel{+}= g_f / s$ | $g_s = 1/1 = 1$ |
| $s = p + q$ | $g_p \mathrel{+}= g_s$, $g_q \mathrel{+}= g_s$ | $g_p = 1$, $g_q = 1$ |
| $q = e^x$ | $g_x \mathrel{+}= q\,g_q$ | $g_x = 1 \times 1 = 1$ |
| $p = xy$ | $g_x \mathrel{+}= y\,g_p$, $g_y \mathrel{+}= x\,g_p$ | $g_x = 1 + 2 = 3$, $g_y = 0$ |

*Check:* $\partial f/\partial x = \frac{y + e^x}{xy + e^x} = \frac{2 + 1}{0 + 1} = 3$ ✓; $\partial f/\partial y = \frac{x}{xy + e^x} = 0$ ✓.

**(b)** With `=`, $g_x$ ends up holding only the contribution of whichever of $p$, $q$ runs *last*: 2 (from $p$) or 1 (from
$q$), depending on the order. Never 3. The bug is order-dependent, which is why it can pass one test and fail another.

**(c)** $\mathbf{c}$ `(2, 1)` is stretched across the 3 columns: $(\mathbf{X} \odot \mathbf{c})_{ij} = X_{ij}c_i$. $\mathcal{L} = \sum_{ij}X_{ij}c_i$.
$\mathbf{G}_{\mathbf{X}}$: $\partial\mathcal{L}/\partial X_{ij} = c_i$, so $\mathbf{G}_{\mathbf{X}} = \begin{pmatrix}2 & 2 & 2\\ -1 & -1 & -1\end{pmatrix}$, shape `(2, 3)`.
$\mathbf{G}_{\mathbf{c}}$: $c_i$ is used in all 3 entries of row $i$, so $\partial\mathcal{L}/\partial c_i = \sum_j X_{ij}$: $(1 + 0 + 2,\ -1 + 3 + 1) = (3, 3)$, shape `(2, 1)` (sum over the
stretched axis, keeping it).

```text
== exercise 1 check ==
(a) f = 0.0  df/dx = 3.0  df/dy = 0.0   formula: (y + e^x)/(xy + e^x) = 3, x/(xy + e^x) = 0
(c) dL/dX [[2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]] shape (2, 3)   dL/dc [[3.0], [3.0]] shape (2, 1)
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) Correct graph and a valid topological order | 2 |
| (a) Each node's contribution shown; both contributions to $g_x$ added; 3 and 0 | 3 |
| (a) Checked against the formula | 1 |
| (b) Order-dependent wrong value (1 or 2), with the reason | 2 |
| (c) $\mathbf{G}_{\mathbf{X}}$ with shape `(2, 3)` | 2 |
| (c) $\mathbf{G}_{\mathbf{c}}$ as row sums, shape `(2, 1)` not `(2,)` or `(2, 3)` | 2 |

**Likely mistakes:**

- Treating $\log$'s derivative as $1/f$ instead of $1/s$ (the input, not the output).
- Missing the path through $q = e^x$ and answering 2. Drawing the graph first prevents this.
- An invalid order (visiting $s$ before $q$ in the backward pass) and then wondering why $g_q$ is 0 when it is used.
- In (c), giving $\mathbf{G}_{\mathbf{c}}$ the shape `(2, 3)` (not unbroadcast) or `(2,)` (summed but the axis dropped). It must match $\mathbf{c}$.

## Exercise 2 (code)

**Solution:** [`c1_3_autograd_engine_ex2.py`](c1_3_autograd_engine_ex2.py). The new backward rules:

```python
# pow:  self.grad += k * self.data ** (k - 1) * out.grad
# tanh: self.grad += (1.0 - out.data ** 2) * out.grad
# .T:   self.grad += out.grad.T                          # transpose is its own inverse
# max:
idx = np.argmax(self.data, axis=axis)                                   # first maximum on ties
g = out.grad if keepdims else np.expand_dims(out.grad, axis)
mask = np.zeros_like(self.data)
np.put_along_axis(mask, np.expand_dims(idx, axis), 1.0, axis=axis)
self.grad += mask * g
```

```text
== exercise 2: gradient checks of the new operations ==
pow 3                        rel error < 1e-8  PASS
pow 2 with broadcast add     rel error < 1e-8  PASS
tanh                         rel error < 1e-8  PASS
transpose then matmul        rel error < 1e-8  PASS
max axis=1                   rel error < 1e-8  PASS
max axis=0 keepdims          rel error < 1e-8  PASS

== exercise 2: a tie in max ==
x = [[1.0, 3.0, 3.0]]  our gradient [[0.0, 1.0, 0.0]]  (all to the first maximum)
central differences [0.0, 0.5, 0.5]  (each tied entry looks like half)

== exercise 2: shifting by the row max ==
gradient [[1.0, 1.0, -3.0, 1.0], [1.0, 1.0, -3.0, 1.0]]
row sums of the gradient [0.0, 0.0]
```

**Why the check fails at a tie.** With two equal maxima, nudging either one *up* raises the max by $h$, and nudging it
*down* leaves the max unchanged (the other entry is still there). Central differences average the two: $(h - 0)/(2h) = 0.5$
for each tied entry. The function has a kink there: no single gradient exists, and any split that sums to 1 is a valid
subgradient. The code is right; the check is being asked about a point where the question has no single answer.

**Why the row sums are zero.** $\mathcal{L} = \sum_j (x_j - \max_k x_k)$ does not change if you add the same constant to every entry of a
row (both terms shift by it). So the directional derivative along $(1, 1, \dots, 1)$ is zero, and that directional derivative
is exactly the sum of the gradient's entries. This shift-invariance is what makes the max-subtraction trick in C1.4 safe:
subtracting the row max changes no gradient of a softmax-based loss.

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| `pow` and `tanh` with correct rules, reusing the forward output for tanh | 2 |
| `.T` backward transposes the gradient | 2 |
| `max` sends gradient to one position per reduced slice; handles `keepdims` both ways | 3 |
| Gradient checks with a random output weighting, including a broadcast case | 3 |
| Tie explained (kink, subgradient, why the check gives 0.5) | 2 |
| Zero row sum shown and explained by shift invariance | 2 |

**Likely mistakes:**

- Using `self.data == out.data` as the mask for `max`: at a tie it sends the full gradient to *both* entries (sum 2), which
  is not a subgradient.
- Forgetting to expand the reduced axis before multiplying by the mask when `keepdims=False`, so the gradient broadcasts
  along the wrong axis (or crashes for a non-square input). Test with a `(3, 4)` input, never a square one.
- Checking `max` with `f(...).sum()` and random inputs only: the code passes, but it was never tested with `keepdims=True` or
  axis 0.
- Implementing `tanh` as `(exp(x) - exp(-x)) / (exp(x) + exp(-x))` from existing operations: correct gradients, but it
  overflows for $\lvert x\rvert > 710$ (C1.4).

## Exercise 3 (think)

**Strong answer:** Count the arrays a forward pass creates and a backward rule needs. Input $\mathbf{X}$: $512 \times 784 \approx 0.40$M
numbers (the first matmul's backward reads it for $\mathbf{G}_{\mathbf{W}^{[1]}}$). Each hidden layer creates the matmul output, the bias-add output
$\mathbf{Z}$ (the ReLU reads it) and the ReLU output $\mathbf{H}$ (the next matmul reads it), each $512 \times 1024 \approx 0.52$M numbers: about 3.1M for the two hidden
layers. The logits are $512 \times 10$. Total about 3.5M numbers, 14 MB in float32. Our engine also allocates a zero gradient
for every node at creation, which roughly doubles it (about 28 MB), and the weights themselves are about 1.9M numbers (7.5 MB).
So at batch 512 the graph's activations are already larger than the model; they grow linearly with batch size and depth,
which is why large models run out of memory in training and not at inference. Under `torch.no_grad()`, PyTorch records
no graph: each intermediate tensor is freed as soon as the next operation has used it, so inference memory is about one
layer's activations at a time. The classic out-of-memory bug: an evaluation loop that runs without `no_grad()` *and*
keeps a reference to each batch's loss tensor (`total += loss` instead of `total += loss.item()`), which keeps every batch's
whole graph alive until the loop ends. Even without the leak, it wastes memory and time building graphs nobody uses.

**Weak answer:** "The graph stores all the activations, which uses a lot of memory. `no_grad()` turns off gradients so it
uses less memory at inference."

**Why it is weak:** no estimate (the question asks for one), no reasoning about which arrays are kept and why, and no
mechanism for the out-of-memory failure (references to the graph kept alive across batches).

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Lists the cached arrays (input, pre-activations, activations, logits) and why each is needed | 3 |
| A numeric estimate in the right range (about 10–30 MB, with what is counted stated) and comparison with the weights | 3 |
| Explains `no_grad()`: no graph recorded, intermediates freed immediately | 2 |
| Names the leak (keeping loss tensors, not `.item()`) or the wasted graph building | 2 |

## Scoring

Total 36 pts. Pass ≥ 70% (26 pts). Topic ids to revisit for each lost criterion: Exercise 1 (a)–(b) → C1.3 (node rule,
`+=`, topological order) and A4.3; (c) → C1.3 (broadcasting and unbroadcast), A1.2. Exercise 2 → C1.3 (tensor rules,
gradient checking), A4.2; the shift-invariance part → C1.4. Exercise 3 → C1.3 (what the graph keeps), C2.1 (`no_grad`),
C4.2 (activation memory).
