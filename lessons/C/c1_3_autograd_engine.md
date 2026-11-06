# C1.3 An autograd engine from scratch

**Core** · Part C, week C1 (course week 13) · ~3.5 h · Lab:
[`labs/C/c1_3_autograd_engine.py`](../../labs/C/c1_3_autograd_engine.py)

## Why it matters

In C1.2 you wrote the backward pass of one specific network by hand. Change the architecture (add a layer, a skip
connection, a different loss) and you would have to derive it again. An **autograd** engine removes that work: you
write only the forward computation, the engine records it as a graph, and one call to `backward()` gives the gradient
of the loss with respect to every input. PyTorch, JAX and TensorFlow are all built on this idea. Building a small one
yourself, first for single numbers and then for NumPy arrays, is the fastest way to understand what
`loss.backward()`, `.grad` and `zero_grad()` actually do, why gradients accumulate, and why a bias of shape `(d,)` gets
a gradient of shape `(d,)` even though it was added to a `(n, d)` matrix. This engine becomes the core of project P4.

## Prerequisite check

- Can you run a backward pass on a graph where one value is used twice, and say why the two contributions are added?
  If not, revisit [A4.3](../A/a4_3_chain_rule_graphs.md).
- Can you write the backward pass of a dense layer, $\mathbf{G}_{\mathbf{W}} = \mathbf{X}^\top\boldsymbol{\Delta}$, $\mathbf{G}_{\mathbf{X}} = \boldsymbol{\Delta}\mathbf{W}^\top$, and say why the bias gradient is a column sum?
  If not, revisit [C1.2](c1_2_backprop_by_hand.md).
- Can you predict the shape of `A + b` for `A` of shape `(3, 4)` and `b` of shape `(4,)`, and of `(3, 1) + (1, 4)`? If not,
  revisit [A1.2](../A/a1_2_numpy_arrays.md).
- Can you check a gradient with central differences and a relative error? If not, revisit
  [A4.2](../A/a4_2_gradients_jacobians.md).

## Intuition

**Record, then replay backwards.** Every time you compute something with the engine's numbers (`c = a * b`), the
result remembers two things: which values it was made from, and a small function that knows that operation's local
derivative ("for a product, each input's gradient is the other input times the output's gradient"). After the
forward pass you have a graph of these records, ending at the loss. `backward()` then visits the records in reverse
order and calls each small function, which pushes the gradient one step further back. No node needs to know anything
about the rest of the network; each one only knows its own local rule. That is what makes the engine general.

**The order matters.** A node may only pass its gradient back once that gradient is complete, meaning every node that
*uses* it has already added its share. Sorting the graph so that each node comes after all of its inputs (a
**topological order**) and walking it backwards guarantees exactly that.

**Add, never assign.** If a value is used twice (`x * x`, or a weight shared by every example in a batch), it receives
a gradient from each use, and they must be *added* (A4.3: "add where paths meet"). Write `grad = ...` instead of
`grad += ...` and the second contribution silently overwrites the first. This is the most common bug in hand-written
engines, and the lab shows it giving $dy/dx = 3$ for $y = x^2$ at $x = 3$. The same rule explains why PyTorch needs `zero_grad()`:
gradients keep adding up across `backward()` calls until you reset them.

**Broadcasting is a hidden "used many times".** When NumPy adds a bias of shape `(d,)` to a batch of shape `(n, d)`, it
behaves as if the bias were copied into every row. Each copy is a use, so on the way back the gradients of the $n$
copies are added: the bias gradient is the sum over rows. A tensor engine must undo every broadcast this way
("unbroadcast"), or the gradient comes back with the wrong shape.

> **Bridge:** the forward pass writes an append-only log of operations, like an event store or a database's write-ahead
> log, and `backward()` replays it in reverse, like rolling back a migration chain. **Where it breaks:** a rollback
> restores old state, while the backward pass computes something new (derivatives) and must *accumulate* into each
> entry rather than overwrite it. And the log is rebuilt from scratch on every forward pass, so the graph can differ
> from one batch to the next (an `if` in your model is fine). PyTorch calls this define-by-run.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $v$ | a node of the graph (a scalar `Value` or a `Tensor`) | any |
| $\mathcal{L}$ | the final scalar the gradients are taken of (`backward()` is called on it) | scalar |
| $\mathbf{G}_v$ | $\partial\mathcal{L}/\partial v$, stored in `v.grad`; the same shape as $v$ | shape of $v$ |
| $u = \mathrm{op}(v_1, \dots, v_m)$ | a node computed from inputs $v_1, \dots, v_m$ | — |
| $\mathbf{X}$, $\mathbf{b}$ | a batch and a bias in the broadcast examples | `(n, d)`, `(d,)` |
| $\mathbf{1}_n$ | a column of $n$ ones | `(n, 1)` |

**The node rule.** For $u = \mathrm{op}(v_1, \dots, v_m)$, when $\mathbf{G}_u$ is complete, each input receives

$$\mathbf{G}_{v_k} \mathrel{+}= (\text{local derivative of } u \text{ with respect to } v_k) \text{ applied to } \mathbf{G}_u.$$

For scalars "applied to" is a plain multiplication. This is A4.3's rule; the engine just stores it as a closure on
each node.

**Why `+=` is the chain rule.** Suppose $v$ feeds several nodes $u_1, \dots, u_r$. The multivariable chain rule (A4.3) gives
$\frac{\partial\mathcal{L}}{\partial v} = \sum_{s=1}^{r}\frac{\partial\mathcal{L}}{\partial u_s}\frac{\partial u_s}{\partial v}$: one term per use. Each $u_s$'s backward function adds its own term into `v.grad`, which starts at 0. After all
$r$ have run, `v.grad` holds the full sum. If $v$ is used twice *by the same node* (in $y = x \cdot x$), the multiply rule adds
"the other input times $\mathbf{G}_y$" twice, once for each argument slot: $x \cdot 1 + x \cdot 1 = 2x$. Correct.

**Topological order.** List the nodes so that every node appears after all of its inputs. Walking the list *backwards*,
every node that uses $v$ appears after $v$ in the list, so it is visited before $v$. By the time $v$ is visited, all its
users have added their terms and $\mathbf{G}_v$ is complete. The lab builds the list with a depth-first search that appends a node
only after all of its inputs have been appended (post-order). It uses an explicit stack rather than recursion,
because a graph for a whole batch is thousands of nodes deep and Python's default recursion limit is 1,000.

**Scalar local rules** (A4.3), as the lab's `Value` implements them, with $u$ the output:

| Operation | Backward |
|---|---|
| $u = a + b$ | $g_a \mathrel{+}= g_u$, $g_b \mathrel{+}= g_u$ |
| $u = ab$ | $g_a \mathrel{+}= b\,g_u$, $g_b \mathrel{+}= a\,g_u$ |
| $u = a^k$ ($k$ a constant) | $g_a \mathrel{+}= k\,a^{k-1}g_u$ |
| $u = e^a$ | $g_a \mathrel{+}= u\,g_u$ |
| $u = \log a$ | $g_a \mathrel{+}= g_u / a$ |
| $u = \tanh a$ | $g_a \mathrel{+}= (1 - u^2)\,g_u$ |
| $u = \mathrm{ReLU}(a)$ | $g_a \mathrel{+}= \mathbb{1}[a > 0]\,g_u$ |

Subtraction, negation and division need no rules of their own: $a - b = a + (-1)b$ and $a / b = a\,b^{-1}$, so the engine builds
them from the rules above and gets their gradients for free.

**Tensor rules.** A `Tensor` node holds an array, and $\mathbf{G}_v$ has the same shape. Elementwise operations use the same rules
with $\odot$ (elementwise product) in place of multiplication. The others:

| Operation | Forward shape | Backward |
|---|---|---|
| $\mathbf{U} = \mathbf{A}\mathbf{B}$ | `(n, d) @ (d, k) -> (n, k)` | $\mathbf{G}_{\mathbf{A}} \mathrel{+}= \mathbf{G}_{\mathbf{U}}\mathbf{B}^\top$, $\mathbf{G}_{\mathbf{B}} \mathrel{+}= \mathbf{A}^\top\mathbf{G}_{\mathbf{U}}$ (C1.2) |
| $s = \sum_{ij} A_{ij}$ | `(n, d) -> ()` | $\mathbf{G}_{\mathbf{A}} \mathrel{+}= g_s$ in every entry |
| $\mathbf{s} = $ `A.sum(axis=0)` | `(n, d) -> (d,)` | $\mathbf{G}_{\mathbf{A}} \mathrel{+}= $ the vector $\mathbf{G}_{\mathbf{s}}$ copied into every row |
| `A.mean(axis)` | as `sum` | as `sum`, divided by the number of entries averaged |
| `A.reshape(shape)` | same entries, new shape | $\mathbf{G}_{\mathbf{A}} \mathrel{+}= \mathbf{G}_{\mathbf{U}}$ reshaped back to `A`'s shape |

*Why sum's backward is a copy.* Each $A_{ij}$ appears in the sum once with coefficient 1, so $\partial s/\partial A_{ij} = 1$ and every entry gets $g_s$. For
`sum(axis=0)`, $s_j = \sum_i A_{ij}$, so $A_{ij}$ affects only $s_j$ and gets $(\mathbf{G}_{\mathbf{s}})_j$. Sum and copy are each other's backward: summing
forward means copying backward, and (next) copying forward means summing backward.

**Broadcasting and unbroadcast.** Take $\mathbf{Z} = \mathbf{X} + \mathbf{b}$ with $\mathbf{X}$ `(n, d)` and $\mathbf{b}$ `(d,)`. Entry by entry, $Z_{ij} = X_{ij} + b_j$: the bias entry $b_j$
appears in every row $i$. By the "add over uses" rule,

$$\frac{\partial\mathcal{L}}{\partial b_j} = \sum_{i=1}^{n}\frac{\partial\mathcal{L}}{\partial Z_{ij}}\cdot 1 \quad\Longrightarrow\quad \mathbf{G}_{\mathbf{b}} = \textstyle\sum_i (\mathbf{G}_{\mathbf{Z}})_{i,:} = \mathbf{1}_n^\top\mathbf{G}_{\mathbf{Z}}, \quad \text{shape } (d,).$$

That is the bias rule from C1.2, now derived for any broadcast. NumPy's broadcasting does two things to an input's
shape, and the backward pass undoes each by summing:

1. *It adds axes on the left.* `(d,)` meets `(n, d)` and is treated as `(1, d)`, then stretched. Undo: sum the gradient over
   each extra leading axis.
2. *It stretches axes of size 1.* `(n, 1)` meets `(n, d)` and its single column is copied $d$ times. Undo: sum over that axis,
   keeping it as size 1 (`keepdims=True`).

The lab's `unbroadcast(grad, shape)` does exactly these two steps, and every `+` and `*` in the engine calls it on both
inputs. For `*` the local derivative comes first: for $\mathbf{U} = \mathbf{A} \odot \mathbf{B}$, $\mathbf{A}$'s share is $\mathbf{B} \odot \mathbf{G}_{\mathbf{U}}$ (in the broadcast shape), then unbroadcast to $\mathbf{A}$'s
shape.

**Where backward starts.** `backward()` sets $\mathbf{G}_{\mathcal{L}} = 1$ and requires $\mathcal{L}$ to be a single number. The gradient of a non-scalar output is a
whole Jacobian (A4.2), which reverse mode never forms; to differentiate a vector you first reduce it to a scalar (a
sum, a mean, a loss). PyTorch has the same rule: `backward()` on a non-scalar tensor raises an error unless you pass
a gradient to start from.

## Worked example by hand

**1. A value used twice.** $e = (ab + b)^2$ at $a = 2$, $b = -3$. Nodes: $c = ab$, $d = c + b$, $e = d^2$.

*Topological order* (each node after its inputs): $b, a, c, d, e$ (also valid: $a, b, c, d, e$; the order among nodes that do
not depend on each other does not matter).

*Forward:* $c = 2 \times (-3) = -6$, $d = -6 + (-3) = -9$, $e = 81$.

*Backward*, visiting $e, d, c, a, b$, every gradient starting at 0:

| Visit | Rule | Adds |
|---|---|---|
| $e = d^2$ | $g_d \mathrel{+}= 2d \cdot g_e$ | $g_d = 2(-9)(1) = -18$ |
| $d = c + b$ | $g_c \mathrel{+}= g_d$, $g_b \mathrel{+}= g_d$ | $g_c = -18$, $g_b = -18$ |
| $c = ab$ | $g_a \mathrel{+}= b\,g_c$, $g_b \mathrel{+}= a\,g_c$ | $g_a = (-3)(-18) = 54$, $g_b = -18 + 2(-18) = -54$ |

*Check by algebra:* $e = (b(a + 1))^2 = b^2(a + 1)^2$, so $\partial e/\partial a = 2b^2(a + 1) = 2 \cdot 9 \cdot 3 = 54$ ✓ and $\partial e/\partial b = 2b(a + 1)^2 = 2(-3)(9) = -54$ ✓.

*With `=` instead of `+=`:* $d$'s rule sets $g_b = -18$; then $c$'s rule *replaces* it with $a\,g_c = -36$. The engine reports
$-36$: one path's contribution, lost. For $y = x \cdot x$ at $x = 3$, both contributions come from the same multiply node, and
assignment keeps one of them: $3$ instead of $6$.

**2. A broadcast bias.** $\mathbf{X} = \begin{pmatrix}1 & 2\\ 3 & 4\end{pmatrix}$, $\mathbf{b} = (10, 20)$, $\mathbf{Z} = \mathbf{X} + \mathbf{b}$, $\mathcal{L} = \sum_{ij}Z_{ij}^2$.

- Forward: $\mathbf{Z} = \begin{pmatrix}11 & 22\\ 13 & 24\end{pmatrix}$, $\mathcal{L} = 121 + 484 + 169 + 576 = 1350$.
- $\mathbf{G}_{\mathbf{Z}} = 2\mathbf{Z} = \begin{pmatrix}22 & 44\\ 26 & 48\end{pmatrix}$ (the square's local derivative, entry by entry).
- $\mathbf{G}_{\mathbf{X}} = \mathbf{G}_{\mathbf{Z}}$: $\mathbf{X}$ was not broadcast, so it gets the gradient as is, shape `(2, 2)`.
- $\mathbf{G}_{\mathbf{b}}$: $b_1$ was used in both rows of column 1, so $g_{b_1} = 22 + 26 = 48$; $g_{b_2} = 44 + 48 = 92$. Shape `(2,)`, like $\mathbf{b}$.
- Check one: $\mathcal{L}$ contains $b_1$ in $(1 + b_1)^2 + (3 + b_1)^2$, whose derivative is $2(11) + 2(13) = 48$ ✓.

```text
== worked example: e = (a b + b)^2 ==
topological order: ['b', 'a', 'c = a*b', 'd = c+b', 'e = d**2']
forward: c = -6.0  d = -9.0  e = 81.0
backward: de/dd = -18.0  de/dc = -18.0  de/da = 54.0  de/db = -54.0
check: de/da = 2 b^2 (a + 1) = 54   de/db = 2 b (a + 1)^2 = -54

== the = versus += bug ==
=   y = x * x at x = 3: dy/dx = 3.0  (true value 6.0)
=   e = (a b + b)^2: de/db = -36.0  (true value -54.0)
+=  y = x * x at x = 3: dy/dx = 6.0  (true value 6.0)
+=  e = (a b + b)^2: de/db = -54.0  (true value -54.0)
```

```text
== worked example: a broadcast bias ==
Z = X + b = [[11.0, 22.0], [13.0, 24.0]]   L = 1350.0
dL/dZ = 2 Z = [[22.0, 44.0], [26.0, 48.0]]
dL/dX = [[22.0, 44.0], [26.0, 48.0]]  (same shape as X)
dL/db = column sums of dL/dZ = [48.0, 92.0]  (shape (2,), same as b)
```

## Key plot

![Left: the training loss of the scalar engine for 100 steps lies exactly on top of the tensor engine's curve, which continues to 1,000 steps and falls to about 0.001 on a log scale. Right: the two half-moons with the curved decision boundary learned by the 2-8-8-1 ReLU network](img/c1_3_engines_moons.png)

Left: two engines that share nothing but the local rules, started from the same weights, produce the same loss at every
step (they agree to $10^{-10}$). The tensor engine is about a thousand times faster, so it can keep training. Right: the
boundary of the trained network bends around the moons, which no linear model can do (C1.1).

## From scratch in NumPy

### The scalar engine

A `Value` holds a number, its gradient, its inputs and a closure. Here are two operations and `backward()`; the lab has
all seven.

```python
class Value:
    def __init__(self, data, _prev=(), op=""):
        self.data, self.grad = float(data), 0.0
        self._prev, self._backward, self.op = _prev, (lambda: None), op

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad      # += : every use adds its share
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")
        def _backward():
            self.grad += out.data * out.grad        # d(e^a)/da = e^a, reuse the output
        out._backward = _backward
        return out

    def backward(self):
        order = topological_order(self)            # each node after its inputs
        self.grad = 1.0
        for node in reversed(order):
            node._backward()
```

The closure captures `self`, `other` and `out`, so each node's backward function has everything it needs without any
global state. The engine is checked once against central differences on an expression that uses every operation:

```text
== scalar engine: gradient check of every operation ==
analytic [0.5815, 0.5157, 0.6806]
numeric  [0.5815, 0.5157, 0.6806]
relative error < 1e-8: PASS
```

**Training an MLP with it.** A 2-8-8-1 ReLU network (105 parameters) on 100 points of scikit-learn's `make_moons`, with
labels $y = \pm1$ and the logistic loss $\log(1 + e^{-yz})$. That is B2.1's log-loss written for $\pm1$ labels: for $y = +1$ it is
$\log(1 + e^{-z}) = -\log\sigma(z)$, and for $y = -1$ it is $\log(1 + e^{z}) = -\log(1 - \sigma(z))$, because $1 - \sigma(z) = \sigma(-z)$. Each step builds the whole graph
again, zeroes every gradient, calls `backward()` and moves each weight against its gradient:

```python
for p in params:
    p.grad = 0.0                 # gradients accumulate across backward() calls
loss = scalar_loss(smlp, X_tr, y_tr)
loss.backward()
for p in params:
    p.data -= LR * p.grad
```

```text
== scalar engine: training an MLP on make_moons ==
parameters: 105
graph size for one loss: 21608 nodes
loss: step 0 0.5615  step 50 0.2751  step 99 0.2299
```

One loss over 100 examples is a graph of 21,608 scalar nodes, every one a Python object with its own closure. That is why
each step takes about half a second (the lab prints the time on a `~` line, since it varies by machine).

### The tensor engine

Same design, but `data` and `grad` are NumPy arrays and every operation works on whole arrays. The one new piece is
`unbroadcast`:

```python
def unbroadcast(grad, shape):
    """Sum a gradient down to `shape`, undoing NumPy broadcasting."""
    while grad.ndim > len(shape):                       # axes added on the left
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):                 # size-1 axes that were stretched
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad

def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, (self, other), "+")
    def _backward():
        self.grad += unbroadcast(out.grad, self.shape)
        other.grad += unbroadcast(out.grad, other.shape)
    out._backward = _backward
    return out

def __matmul__(self, other):
    out = Tensor(self.data @ other.data, (self, other), "@")       # (n, d) @ (d, k)
    def _backward():
        self.grad += out.grad @ other.data.T                       # (n, k) @ (k, d) -> (n, d)
        other.grad += self.data.T @ out.grad                       # (d, n) @ (n, k) -> (d, k)
    out._backward = _backward
    return out
```

**Every operation is gradient-checked** on random inputs, including the broadcast cases. The check uses
$\mathcal{L} = \sum(\mathbf{R} \odot f(\dots))$ with a fixed random $\mathbf{R}$, so that every output entry gets a different weight; checking `f(...).sum()` alone would
let some wrong gradients through (any error that sums to zero).

```text
== tensor engine: gradient check of every operation ==
add (3, 4) + (3, 4)                    rel error < 1e-8  PASS
add (3, 4) + (4,)      [broadcast]     rel error < 1e-8  PASS
add (3, 1) + (1, 4)    [broadcast]     rel error < 1e-8  PASS
add (3, 4) + scalar ()  [broadcast]    rel error < 1e-8  PASS
mul (3, 4) * (3, 1)    [broadcast]     rel error < 1e-8  PASS
mul (3, 4) * (4,)      [broadcast]     rel error < 1e-8  PASS
matmul (3, 4) @ (4, 2)                 rel error < 1e-8  PASS
sum all                                rel error < 1e-8  PASS
sum axis=0                             rel error < 1e-8  PASS
sum axis=1 keepdims                    rel error < 1e-8  PASS
mean all                               rel error < 1e-8  PASS
mean axis=0                            rel error < 1e-8  PASS
reshape (3, 4) -> (2, 6)               rel error < 1e-8  PASS
relu                                   rel error < 1e-8  PASS
exp                                    rel error < 1e-8  PASS
log (positive inputs)                  rel error < 1e-8  PASS
x used twice: x * x + x                rel error < 1e-8  PASS
dense layer relu(X W + b)              rel error < 1e-8  PASS
all tensor ops pass: OK
```

`log` is checked on positive inputs only (it is undefined elsewhere), and `relu` on random normal inputs, which are
never within $10^{-6}$ of the kink at 0 in practice; a check that straddles a kink fails for a correct gradient.

**Without `unbroadcast`.** If `__add__` passed `out.grad` straight back, the bias would receive a `(5, 4)` gradient. With
`+=` on a `(4,)` array, NumPy refuses. With a bias stored as `(1, 4)` and `grad = grad + g`, it silently broadcasts the
*gradient* up to `(5, 4)`, and the next weight update changes the bias's shape. That is how a shape bug turns into a
model that trains but is wrong.

```text
== without unbroadcast ==
bias shape (4,), correct grad [5.0, 5.0, 5.0, 5.0] (each bias entry was used 5 times)
adding the raw (5, 4) gradient to a (4,) bias fails: non-broadcastable output operand with shape (4,) doesn't match the broadcast shape (5,4)
with b of shape (1, 4) and `b.grad = b.grad + g`, b.grad silently becomes shape (5, 4)
```

**The same MLP, the same start.** The network is now four tensor operations per layer instead of thousands of scalar
ones:

```python
H = Tensor(X)
for l, (W, b) in enumerate(zip(Ws, bs)):
    H = H @ W + b                       # (n, d_out); the (d_out,) bias broadcasts over rows
    if l < len(Ws) - 1:
        H = H.relu()
z = H.reshape(-1)                       # (n,)
loss = ((Tensor(-y) * z).exp() + 1.0).log().mean()
```

```text
== tensor engine: the same MLP, same start ==
loss: step 0 0.5615  step 50 0.2751  step 99 0.2299
max |scalar loss - tensor loss| over 100 steps: < 1e-10
max |scalar weights - tensor weights| after 100 steps: < 1e-10
after 100 steps:  train accuracy 0.9100   test accuracy 0.8980
after 1000 steps: loss 0.0011  train accuracy 1.0000   test accuracy 0.9980
```

The two engines agree to $10^{-10}$ after 100 steps: the only differences are the order in which floating-point sums are
added. Then the tensor engine trains on to 1,000 steps in a fraction of a second, and the network classifies 998 of 1,000
fresh test points correctly. (The loss here, $\log(1 + e^{-yz})$, computed as `exp` then `log`, overflows if $\lvert z\rvert$ gets large. C1.4
replaces it with a stable fused operation.)

## With PyTorch

The same network, weights and batch in PyTorch (float64), with `softplus(-y z)`, which is PyTorch's stable
$\log(1 + e^{-yz})$. Every one of the six gradient arrays matches ours:

```text
== with PyTorch ==
initial loss: torch 0.561541   ours 0.561541
max |torch grad - our grad| over all 6 parameter arrays: < 1e-12
matches PyTorch autograd: OK
PyTorch, y = x * x at x = 3: dy/dx = 6.0 (it accumulates too)
```

What carries over to PyTorch, which you will use from C2.1 on:

| Our engine | PyTorch |
|---|---|
| `Value` / `Tensor` with `.data`, `.grad` | `torch.Tensor` with `requires_grad=True`, `.grad` |
| `_prev` and `_backward` closures | `.grad_fn` (a node of the recorded graph) |
| `loss.backward()` over a topological order | `loss.backward()` (the same idea, in C++) |
| `p.grad = 0` before each step | `optimizer.zero_grad()` |
| building a new graph every forward pass | define-by-run: the graph is rebuilt every call |
| `unbroadcast` | done for you inside every broadcasting operation |

## In practice

- **When to use it:** you will not use your own engine in production. You use what it taught you: reading a PyTorch
  error about shapes or in-place operations, writing a custom `torch.autograd.Function` when an operation needs a
  hand-written backward (C2.1), and predicting memory use (the graph keeps every intermediate array alive until
  `backward()` runs).
- **Failure modes:**
  - *`=` instead of `+=`.* Wrong gradients for any value used twice, which in a real network is every weight (shared across
    the batch). A gradient check on an expression that reuses a value catches it.
  - *Gradients not zeroed between steps.* They accumulate across `backward()` calls; the effective learning rate grows every
    step and the loss blows up. In PyTorch: a missing `zero_grad()`.
  - *Missing unbroadcast.* A crash at best; at worst a parameter whose shape changes after the first update. Assert
    `p.grad.shape == p.data.shape` after `backward()`.
  - *Changing `data` between forward and backward.* The closures read the cached arrays when `backward()` runs; if you
    modified one in place (`x.data *= 2`), the gradient is computed with the wrong values. PyTorch detects this with a
    version counter and raises an error about an in-place operation.
  - *Recursion depth.* A recursive topological sort fails on deep graphs (a long sequence, a big scalar graph). Use an
    explicit stack.
  - *`exp` and `log` from first principles.* $\log(1 + e^z)$ written as two operations overflows for large $z$; $\log$ of a
    probability that rounded to 0 is $-\infty$. Fuse such pairs into one stable operation (C1.4).
- **Defaults that work:** gradient-check every new operation in float64 on random inputs, with broadcast cases, a value
  used twice, and a random weighting of the output; compare a small model end to end against PyTorch; keep each
  operation's backward next to its forward so they change together.

## Exercises

Save your work to `my-work/C/c1_3_autograd_engine.md` (or `.py`) and ask for a review.

1. **By hand:** (a) For $f = \log(xy + e^x)$ at $x = 0$, $y = 2$, draw the graph ($p = xy$, $q = e^x$, $s = p + q$, $f = \log s$), give a
   topological order, and compute $\partial f/\partial x$ and $\partial f/\partial y$ with the node rules, showing what each node adds. Check against the
   formula. (b) What does an engine with `=` instead of `+=` report for $\partial f/\partial x$, and why? (c) For $\mathbf{X}$ `(2, 3)` $= \begin{pmatrix}1 & 0 & 2\\ -1 & 3 & 1\end{pmatrix}$, $\mathbf{c}$
   `(2, 1)` $= (2, -1)^\top$ and $\mathcal{L} = \sum_{ij}(\mathbf{X} \odot \mathbf{c})_{ij}$, compute $\mathbf{G}_{\mathbf{X}}$ and $\mathbf{G}_{\mathbf{c}}$ with their shapes.
2. **Code:** Add four operations to the lab's `Tensor` class, each with a gradient check that uses a random weighting of
   the output: `__pow__` with a constant exponent, `tanh`, `.T` (transpose of a 2-D tensor) and `max(axis, keepdims)`. For
   `max`, send the gradient only to the position of the maximum (ties: to the first one, as `np.argmax` does) and
   explain in a comment why a gradient check near a tie fails even when the code is right. Then show that
   `x - x.max(axis=1, keepdims=True)` has a zero gradient row sum when its output is summed along that axis, and say why.
3. **Think:** Your engine keeps every intermediate array alive until `backward()` runs. Estimate the memory held by the
   graph for one training step of a 784-1024-1024-10 ReLU MLP at batch size 512 in float32 (count each array a forward pass
   creates that a backward rule reads). Then explain what PyTorch's `torch.no_grad()` changes at inference time and why
   forgetting it in an evaluation loop over a large validation set can run out of memory.

## Takeaways

1. An autograd engine records each operation's inputs and local derivative during the forward pass, then visits the
   nodes in reverse topological order so that each node's gradient is complete before it is passed on.
2. Gradients are accumulated with `+=`: a value used several times (a shared weight, `x * x`, a broadcast bias) receives
   the sum of its uses' contributions. Assigning instead silently drops paths.
3. Because gradients accumulate across `backward()` calls too, they must be zeroed before every step; this is PyTorch's
   `zero_grad()`.
4. Broadcasting copies forward, so its backward sums: `unbroadcast` sums over added leading axes and stretched size-1
   axes, giving every gradient the shape of its tensor.
5. A scalar engine and a tensor engine built from the same local rules give the same numbers; the tensor engine is
   about a thousand times faster because the work happens inside NumPy, not in Python objects.

**What builds on this:** [C1.4](c1_4_stable_softmax_cross_entropy.md) (a fused, stable softmax cross-entropy operation for
this engine); [C2.1](c2_1_tensors_autograd_modules.md) (PyTorch autograd, checked against this engine, and custom
`autograd.Function`s); [C4.2](c4_2_mixed_precision.md) (activation memory and checkpointing); P4 (this engine as a tested
package, then an MLP on digits that matches PyTorch to $10^{-6}$).

**References:** A. Karpathy, *micrograd* (a scalar autograd engine in about 100 lines, the model for this lab's
`Value`) and "Neural Networks: Zero to Hero", lecture 1 (VERIFY link: https://karpathy.ai/zero-to-hero.html).
A. G. Baydin, B. A. Pearlmutter, A. A. Radul, J. M. Siskind, "Automatic differentiation in machine learning: a survey",
*Journal of Machine Learning Research*, 2018. A. Paszke et al., "PyTorch: an imperative style, high-performance deep
learning library", *NeurIPS*, 2019. NumPy documentation, "Broadcasting" (VERIFY link:
https://numpy.org/doc/stable/user/basics.broadcasting.html).
