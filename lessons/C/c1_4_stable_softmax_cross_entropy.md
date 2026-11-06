# C1.4 Softmax and cross-entropy, written to be numerically stable

**Core** · Part C, week C1 (course week 13) · ~1.5 h · Lab:
[`labs/C/c1_4_stable_softmax_cross_entropy.py`](../../labs/C/c1_4_stable_softmax_cross_entropy.py)

## Why it matters

Almost every classifier in this course ends the same way: the last layer produces one score per class (the
**logits**), **softmax** turns them into probabilities, and **cross-entropy** measures how much probability went to the
right class. Language models do this over a vocabulary of 50,000+ tokens at every position (F1.4), attention does it
over every pair of tokens (F1.1). Written straight from the textbook formula, softmax returns `nan` for logits like
`[1000, 1001, 1002]`, and in float32 the textbook cross-entropy returns `inf` once logits reach about 50. In a training
run that shows up as a loss that suddenly becomes `nan` and never recovers. This lesson writes both so they cannot
overflow, derives the one-line gradient that makes softmax and cross-entropy a perfect pair, $\mathbf{p} - \mathbf{y}$, and adds them to
your C1.3 engine as a single fused operation.

## Prerequisite check

- Can you compute the cross-entropy $H(p, q) = -\sum_x p(x)\log q(x)$ of two small distributions, and say why it equals $-\log q(\text{true class})$
  when $p$ is one-hot? If not, revisit [A6.4](../A/a6_4_entropy_cross_entropy_kl.md).
- Can you explain why minimizing log-loss is maximum likelihood? If not, revisit [A5.5](../A/a5_5_mle_map.md).
- Can you write logistic regression's loss from the logit as $\log(1 + e^z) - yz$, and say why it is computed that way? If not,
  revisit [B2.1](../B/b2_1_logistic_regression.md).
- Can you say what float64 can and cannot represent (the largest number, the smallest positive one) and why
  subtracting nearly equal numbers loses digits? If not, revisit [A3.4](../A/a3_4_numerical_stability.md).
- Can you add an operation with its own backward rule to your C1.3 `Tensor`? If not, revisit
  [C1.3](c1_3_autograd_engine.md).

## Intuition

**Softmax is "exponentiate, then normalize".** The network outputs any real numbers, one per class. To turn them into
probabilities you need them positive and summing to 1. Exponentiating makes them positive and keeps their order; dividing
by the total makes them sum to 1. A logit one unit higher than another gets $e \approx 2.7$ times the probability. The name
comes from being a smooth version of "max": with logits far apart, nearly all the probability goes to the largest.

**Only differences between logits matter.** Adding the same number to every logit multiplies every $e^{z_k}$ by the same
factor, which cancels in the division. So softmax(`[1000, 1001, 1002]`) is exactly softmax(`[0, 1, 2]`) and
softmax(`[-2, -1, 0]`). That is the whole fix: before exponentiating, subtract the largest logit. The largest
exponent becomes $e^0 = 1$, nothing can overflow, and at least one term of the sum is 1, so the sum cannot underflow to 0.

**Never take the log of a probability you computed.** Cross-entropy is $-\log p_{\text{true}}$. If the model is confidently wrong,
$p_{\text{true}}$ can be smaller than the smallest positive float and round to exactly 0, and $-\log 0 = \infty$. But $\log p$ simplifies
algebraically to "true logit minus log-sum-exp of all logits", which involves no tiny numbers at all. Compute that.

**The gradient is "prediction minus truth".** Pushing the loss down means raising the true class's logit and lowering the
others in proportion to how much probability they stole. The gradient with respect to the logits is exactly $\mathbf{p} - \mathbf{y}$:
the predicted probabilities minus the one-hot target. It is the same form as logistic regression (B2.1) and linear
regression (B1.5), and no coincidence: all three pair a model with its natural loss.

> **Bridge:** subtracting the max before exponentiating is like storing money in integer cents, or computing a
> percentage change from a baseline instead of from absolute values: you choose a representation where the numbers you
> work with are small, without changing the answer. **Where it breaks:** cents only avoid rounding; the max shift is
> exact only because softmax is invariant to adding a constant. Most functions are not (for example `exp` alone), so
> each one needs its own stable rewrite, and you have to prove it is the same function.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $K$ | number of classes | int |
| $\mathbf{z}$ | logits of one example | `(K,)` |
| $\mathbf{p} = \operatorname{softmax}(\mathbf{z})$ | predicted probabilities | `(K,)` |
| $y$, $\mathbf{y}$ | the true class index, and its one-hot vector ($y_k = 1$ for the true class, 0 elsewhere) | int, `(K,)` |
| $\ell$ | cross-entropy of one example | scalar |
| $m$ | $\max_k z_k$, the largest logit | scalar |
| $\operatorname{LSE}(\mathbf{z})$ | log-sum-exp, $\log\sum_k e^{z_k}$ | scalar |
| $\mathbf{Z}$, $\mathbf{P}$, $\mathbf{Y}$ | the same for a batch, one example per row | `(n, K)` |
| $\delta_{kj}$ | 1 if $k = j$, else 0 (the Kronecker delta) | — |

Indices run from 1 to $K$ in the math and from 0 to $K - 1$ in the code.

**Softmax.**

$$p_k = \frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}, \qquad k = 1, \dots, K.$$

Every $p_k > 0$ and $\sum_k p_k = 1$.

**Shift invariance.** For any constant $c$: $\frac{e^{z_k + c}}{\sum_j e^{z_j + c}} = \frac{e^c\,e^{z_k}}{e^c\sum_j e^{z_j}} = \frac{e^{z_k}}{\sum_j e^{z_j}}$. Choose $c = -m$:

$$\operatorname{softmax}(\mathbf{z}) = \operatorname{softmax}(\mathbf{z} - m), \qquad \text{all entries of } \mathbf{z} - m \le 0.$$

Now each $e^{z_k - m} \le 1$ (no overflow), and the largest is exactly $e^0 = 1$, so the denominator is at least 1 (no division by 0).

**Log-sum-exp.** Factor $e^m$ out of the sum:

$$\operatorname{LSE}(\mathbf{z}) = \log\Big(e^m\sum_k e^{z_k - m}\Big) = m + \log\sum_k e^{z_k - m}.$$

The sum is between 1 and $K$, so its log is between 0 and $\log K$: a small, safe number. LSE is a smooth maximum: it is always
between $m$ and $m + \log K$.

**Log-softmax and cross-entropy from logits.** Take the log of $p_k$: $\log p_k = z_k - \log\sum_j e^{z_j} = z_k - \operatorname{LSE}(\mathbf{z})$. With a one-hot target, cross-entropy
(A6.4) keeps only the true class's term:

$$\ell = -\sum_k y_k\log p_k = -\log p_y = \operatorname{LSE}(\mathbf{z}) - z_y.$$

No probability is ever computed, so none can round to 0. For a batch, $\mathcal{L} = \frac1n\sum_{i=1}^{n}\big(\operatorname{LSE}(\mathbf{z}_i) - z_{i, y_i}\big)$.

**The gradient, the short way.** Differentiate $\ell = \operatorname{LSE}(\mathbf{z}) - z_y$ with respect to one logit $z_k$:

- $\frac{\partial}{\partial z_k}\log\sum_j e^{z_j} = \frac{1}{\sum_j e^{z_j}}\cdot e^{z_k} = p_k$ (chain rule: derivative of $\log$ is one over its input, times the derivative of the
  sum, in which only the $j = k$ term depends on $z_k$).
- $\frac{\partial z_y}{\partial z_k} = 1$ if $k = y$, else 0, which is $y_k$.

So

$$\frac{\partial\ell}{\partial z_k} = p_k - y_k, \qquad \nabla_{\mathbf{z}}\ell = \mathbf{p} - \mathbf{y}, \qquad \text{and for the batch mean } \mathbf{G}_{\mathbf{Z}} = \tfrac1n(\mathbf{P} - \mathbf{Y}),\ (n, K).$$

Each row sums to $\sum_k p_k - \sum_k y_k = 1 - 1 = 0$: raising every logit equally changes nothing (shift invariance again).

**The gradient, the long way (through softmax's Jacobian).** This is what an engine that treated softmax and the log as
separate nodes would compute, and it shows why fusing them is simpler. First the Jacobian of softmax. Write
$p_k = e^{z_k}\cdot S^{-1}$ with $S = \sum_i e^{z_i}$, and note two facts. $\partial S/\partial z_j = e^{z_j}$, because only one term of the sum depends on $z_j$. And by
the power rule and the chain rule (A4.1, A4.3), $\partial(S^{-1})/\partial z_j = -S^{-2}\cdot\partial S/\partial z_j = -e^{z_j}/S^2$.

*Case $k = j$:* both factors of $e^{z_k}\cdot S^{-1}$ depend on $z_k$, so use the product rule (A4.1):

$$\frac{\partial p_k}{\partial z_k} = e^{z_k}\cdot S^{-1} + e^{z_k}\cdot\Big(-\frac{e^{z_k}}{S^2}\Big) = \frac{e^{z_k}}{S} - \Big(\frac{e^{z_k}}{S}\Big)^2 = p_k - p_k^2 = p_k(1 - p_k).$$

*Case $k \ne j$:* $e^{z_k}$ does not depend on $z_j$, so only the second factor changes: $\frac{\partial p_k}{\partial z_j} = e^{z_k}\cdot\Big(-\frac{e^{z_j}}{S^2}\Big) = -\frac{e^{z_k}}{S}\cdot\frac{e^{z_j}}{S} = -p_kp_j$. Both cases in one line:
$\frac{\partial p_k}{\partial z_j} = p_k(\delta_{kj} - p_j)$, a `(K, K)` matrix $\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$. Then the chain rule through $\ell = -\sum_k y_k\log p_k$, whose derivative with
respect to $p_k$ is $-y_k/p_k$:

$$\frac{\partial\ell}{\partial z_j} = \sum_k\Big(-\frac{y_k}{p_k}\Big)p_k(\delta_{kj} - p_j) = -\sum_k y_k\delta_{kj} + p_j\sum_k y_k = -y_j + p_j.$$

The $p_k$ from the Jacobian cancels the $1/p_k$ from the log, the same cancellation as in B2.1. The long way needs the full
Jacobian and a division by $p_k$ (which may be 0); the short way needs neither.

**Two classes are logistic regression.** With $K = 2$ and logits $(0, z)$ (by shift invariance, one logit can always be set to
0):

$$p_2 = \frac{e^z}{e^0 + e^z} = \frac{1}{e^{-z} + 1} = \sigma(z), \qquad \ell = \log(1 + e^z) - y\,z \quad (y \in \{0, 1\} \text{ marks class 2}),$$

which is exactly B2.1's log-loss from logits, and the gradient $p_2 - y = \sigma(z) - y$ is B2.1's gradient. Softmax regression with $K$
classes has one redundant logit for the same reason; that is why binary logistic regression uses a single logit.

**Where the naive versions break.** `exp` overflows to `inf` above $\log(\text{largest float})$: 709.78 in float64, 88.72 in float32. In
the opposite direction, a probability below the smallest positive float (about $5 \times 10^{-324}$ in float64, $1.4 \times 10^{-45}$ in float32)
rounds to 0, and its log is $-\infty$. For logits $s\,(1, 0, -1)$ with the true class the last one, the true probability is about
$e^{-2s}$, which reaches the float32 limit at $2s \approx 103$, so $s \approx 52$: the lab's key plot.

## Worked example by hand

**1. Softmax, loss and gradient.** $\mathbf{z} = (2, 1, 0)$, true class 1 (index 0), so $\mathbf{y} = (1, 0, 0)$.

- $e^{\mathbf{z}} = (e^2, e^1, e^0) = (7.3891, 2.7183, 1.0000)$, sum $= 11.1073$.
- $\mathbf{p} = (7.3891, 2.7183, 1)/11.1073 = (0.6652, 0.2447, 0.0900)$. Check: they sum to 1.
- $\ell = -\log 0.6652 = 0.4076$. From logits: $\operatorname{LSE} = \log 11.1073 = 2.4076$, and $\operatorname{LSE} - z_1 = 2.4076 - 2 = 0.4076$. Same.
- $\nabla_{\mathbf{z}}\ell = \mathbf{p} - \mathbf{y} = (0.6652 - 1,\ 0.2447,\ 0.0900) = (-0.3348, 0.2447, 0.0900)$. Gradient descent raises $z_1$ and lowers the other two, the
  second more than the third because it holds more of the wrong probability. The entries sum to 0.

**2. The same through the Jacobian.** $\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ has diagonal $p_k(1 - p_k)$: $0.6652 \times 0.3348 = 0.2227$, $0.2447 \times 0.7553 = 0.1848$,
$0.0900 \times 0.9100 = 0.0819$; off-diagonal $-p_kp_j$: $-0.6652 \times 0.2447 = -0.1628$, $-0.6652 \times 0.0900 = -0.0599$, $-0.2447 \times 0.0900 = -0.0220$. The upstream
gradient $\partial\ell/\partial\mathbf{p} = -\mathbf{y}/\mathbf{p} = (-1/0.6652, 0, 0) = (-1.5033, 0, 0)$. Multiply: $\partial\ell/\partial z_j = \sum_k(\partial\ell/\partial p_k)(\partial p_k/\partial z_j)$, only $k = 1$ is non-zero, so it is
$-1.5033$ times the first row: $(-1.5033 \times 0.2227,\ -1.5033 \times (-0.1628),\ -1.5033 \times (-0.0599)) = (-0.3348, 0.2447, 0.0900)$. Same as $\mathbf{p} - \mathbf{y}$.

**3. Large logits.** $\mathbf{z} = (1000, 1001, 1002)$. Naively $e^{1000}$ is `inf` and every $p_k = \infty/\infty$ = `nan`. Stably: $m = 1002$, $\mathbf{z} - m = (-2, -1, 0)$,
$e^{\mathbf{z} - m} = (0.1353, 0.3679, 1)$, sum 1.5032, $\mathbf{p} = (0.0900, 0.2447, 0.6652)$: example 1's probabilities in reverse order, because these are
example 1's logits reversed and shifted. $\operatorname{LSE} = 1002 + \log 1.5032 = 1002.4076$.

```text
== worked example: softmax, cross-entropy and p - y ==
exp(z) = [7.3891, 2.7183, 1.0]   sum = 11.1073
p = softmax(z) = [0.6652, 0.2447, 0.09]   sum = 1.0000
cross-entropy = -log p[0] = 0.4076
gradient dL/dz = p - y = [-0.3348, 0.2447, 0.09]   sum = 0.0000
```

## Key plot

![Cross-entropy of the true class against the logit scale s on log-log axes: the stable value rises in a straight line through s = 1000; the naive float32 curve stops at s = 52 and the naive float64 curve at s = 375, where the true class's probability rounds to zero](img/c1_4_naive_vs_stable.png)

The logits are $s\,(1, 0, -1)$ and the true class is the last one, so the model is confidently wrong and the true loss is about $2s$.
The naive formula agrees with the stable one until the true probability rounds to 0, then returns `inf`: at $s = 52$ in float32
and $s = 375$ in float64. Logits of 50 are not exotic in float32 training, especially early on or with a large learning rate.

## From scratch in NumPy

**The overflow.** The textbook formula on large and on very negative logits:

```text
== from scratch: naive softmax overflows ==
exp([1000, 1001, 1002]) = [inf, inf, inf]
naive softmax(z_big) = [nan, nan, nan]
naive softmax(z_neg) = [nan, nan, nan]  (exp underflows to 0: 0/0)
largest float64 = 1.7977e+308; exp overflows above log of that = 709.78
largest float32 = 3.4028e+38; exp overflows above 88.72
```

**The stable versions** are three short functions:

```python
def softmax(z):
    shifted = z - z.max(axis=-1, keepdims=True)      # largest entry is now 0
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)

def logsumexp(z):
    m = z.max(axis=-1)
    return m + np.log(np.exp(z - m[..., None]).sum(axis=-1))

def log_softmax(z):
    return z - logsumexp(z)[..., None]
```

`keepdims=True` and `m[..., None]` keep the max as a column so it broadcasts against every row (a `(n,)` max would be
broadcast across the wrong axis of a `(n, K)` batch, or crash).

```text
== from scratch: stable softmax, log-sum-exp, log-softmax ==
z_big: softmax [0.09, 0.2447, 0.6652]  logsumexp 1002.4076  log_softmax [-2.4076, -1.4076, -0.4076]
z_neg: softmax [0.6652, 0.2447, 0.09]  logsumexp -999.5924  log_softmax [-0.4076, -1.4076, -2.4076]
z_small: softmax [0.6652, 0.2447, 0.09]  logsumexp 2.4076  log_softmax [-0.4076, -1.4076, -2.4076]
matches scipy.special logsumexp, log_softmax and softmax: OK
softmax(z + 7) == softmax(z): True  (shift invariance)
```

**Cross-entropy from logits** is one line, `np.mean(logsumexp(Z) - Z[np.arange(n), y])`. Three ways of computing it on three
inputs:

```text
== from scratch: cross-entropy from logits ==
moderate [2, 1, 0], y=0              naive    0.4076   -log(stable softmax)    0.4076   from logits    0.4076
large [1000, 1001, 1002], y=2        naive       nan   -log(stable softmax)    0.4076   from logits    0.4076
confidently wrong [0, 800, 0], y=0   naive       inf   -log(stable softmax)       inf   from logits  800.0000
```

The middle column is the trap: a stable softmax followed by `-log` survives large logits but still fails when the model is
confidently wrong, because $p = e^{-800}$ rounds to 0. Only the version that never forms the probability gives the right
answer, 800. Those confidently wrong examples are exactly the ones whose gradient you most need.

**The gradient.** For the batch mean, $\mathbf{G}_{\mathbf{Z}} = (\mathbf{P} - \mathbf{Y})/n$, checked against central differences on a random `(5, 3)` batch, and the long way
through the Jacobian for worked example 1:

```text
== from scratch: the gradient p - y, checked ==
analytic row 0 [0.0455, 0.0272, -0.0727]   numeric row 0 [0.0455, 0.0272, -0.0727]
relative error over the batch < 1e-8: PASS
every row of the gradient sums to 0: True
softmax Jacobian for z = [2, 1, 0]:
[[ 0.2227 -0.1628 -0.0599]
 [-0.1628  0.1848 -0.022 ]
 [-0.0599 -0.022   0.0819]]
chain rule through the Jacobian: [-0.3348, 0.2447, 0.09]   p - y: [-0.3348, 0.2447, 0.09]
```

**Two classes.** Logits $(0, z)$ for 8 random examples: softmax's second column equals the sigmoid, and the loss and its
gradient equal B2.1's and scikit-learn's:

```text
== from scratch: two classes = logistic regression ==
softmax([0, z])[1] equals sigmoid(z): True
2-class cross-entropy 1.056188   B2.1 log-loss 1.056188   sklearn log_loss 1.056188
gradient wrt the logit z: (p1 - y) / n = [-0.0282, 0.1198, -0.0744] ...   B2.1's (sigmoid - y) / n = [-0.0282, 0.1198, -0.0744] ...
```

**A fused operation for the tensor engine.** The lab copies the C1.3 `Tensor` (the operations it needs) and adds
`cross_entropy` as a single node: the forward pass is the stable formula, the backward pass is $(\mathbf{P} - \mathbf{Y})/n$ with $\mathbf{P}$ recomputed
stably from the cached logits.

```python
def cross_entropy(self, y):
    """Mean cross-entropy of logits (n, K) against class indices y (n,), as ONE node."""
    n = self.data.shape[0]
    lse = logsumexp(self.data)                               # (n,) stable
    out = Tensor(np.mean(lse - self.data[np.arange(n), y]), (self,), "cross_entropy")
    def _backward():
        P = np.exp(self.data - lse[:, None])                 # (n, K) softmax
        P[np.arange(n), y] -= 1.0                            # P - Y without building Y
        self.grad += P / n * out.grad
    out._backward = _backward
    return out
```

It is compared with the same loss built from the engine's existing `exp`, `sum`, `log` and `*` operations:

```text
== from scratch: a fused cross-entropy op for the tensor engine ==
fused loss 1.316140   unfused loss 1.316140   numpy 1.316140
fused gradient vs numeric: rel error < 1e-8  PASS
graph nodes: fused 2, unfused 13
dense layer + fused loss, dW vs numeric: rel error < 1e-8  PASS
logits [1000, 1001, 1002], y = 2: fused loss 0.4076 grad [[0.09, 0.2447, -0.3348]]
                                 unfused loss inf grad [[nan, nan, nan]]
```

On ordinary logits the two agree. The fused version uses 2 graph nodes instead of 13 (less memory, fewer intermediate
arrays), and on large logits it is the only one that still works: the unfused graph overflows in its `exp` node and its
`nan` gradient would destroy every weight in one update. (You could add the max shift to the unfused graph, using the C1.3
exercise's `max`; the fused operation is still simpler and is what frameworks do.)

## With SciPy and PyTorch

`scipy.special.logsumexp`, `log_softmax` and `softmax` match ours to $10^{-12}$ (checked in the lab above). In PyTorch,
`torch.nn.functional.cross_entropy(logits, targets)` takes **logits** and integer class indices, applies log-softmax
internally, and returns the mean. Its gradient matches our fused operation:

```text
== with PyTorch ==
torch cross_entropy 1.316140   ours 1.316140
max |torch grad - fused grad| < 1e-12
matches torch.nn.functional.cross_entropy: OK
float32, logits [1000, 1001, 1002]: torch softmax [[0.09, 0.2447, 0.6652]]  cross_entropy(y=2) 0.4076
torch.exp(torch.tensor(89.0)) in float32 = inf
```

PyTorch's `softmax` and `cross_entropy` are stable even in float32; its bare `exp` is not, so the same care applies to
any formula you write yourself.

## In practice

- **When to use it:** softmax with cross-entropy for any single-label classification with $K \ge 2$ classes (one class per
  example). For two classes, or for multi-label problems where each label is a separate yes/no, use one logit per label
  with a sigmoid and binary cross-entropy from logits (PyTorch's `BCEWithLogitsLoss`), not a softmax.
- **Failure modes:**
  - *Softmax before `CrossEntropyLoss`.* PyTorch's loss applies log-softmax itself, so a softmax layer before it squashes
    twice: the loss still falls, but gradients are much smaller and the model trains slowly and stays underconfident.
    The last layer must output raw logits.
  - *`log(softmax(z))` written by hand.* Survives large logits but returns `inf` for confidently wrong predictions (the
    lab's middle column). Use `log_softmax`.
  - *Clipping probabilities* (`p.clamp(min=1e-7)`) to hide `inf`. It removes the symptom and the gradient: for a clipped
    example the gradient is 0, so the worst mistakes stop being corrected.
  - *Wrong axis.* Softmax over the batch axis instead of the class axis gives rows that do not sum to 1; check
    `p.sum(axis=-1)` once.
  - *`nan` loss.* Usually an overflow upstream (huge logits from a too-high learning rate) or a hand-written `exp`/`log`.
    Find the first non-finite tensor (PyTorch: `torch.autograd.set_detect_anomaly(True)` while debugging), then fix the
    cause rather than the loss.
- **Defaults that work:** last layer outputs logits; `F.cross_entropy(logits, y)` (or `nn.CrossEntropyLoss()`) with integer
  targets; `F.log_softmax` if you need log-probabilities; `F.softmax(logits, dim=-1)` only for reporting probabilities, never
  inside the loss. `CrossEntropyLoss` also accepts class probabilities as targets and has a `label_smoothing` option
  (VERIFY (as of 2026-11) the current signature in the PyTorch docs).

## Exercises

Save your work to `my-work/C/c1_4_stable_softmax_cross_entropy.md` (or `.py`) and ask for a review.

1. **By hand:** (a) For $\mathbf{z} = (0, \log 2, \log 5)$ and true class 2 (index 1), compute $\mathbf{p}$, the cross-entropy, and $\mathbf{p} - \mathbf{y}$, exactly (fractions
   and logs). (b) Compute $\operatorname{LSE}(500, 500, 498)$ by the max-subtraction formula to 4 decimals, and the softmax. (c) A batch of two
   examples has $\mathbf{P} = \begin{pmatrix}0.7 & 0.2 & 0.1\\ 0.1 & 0.3 & 0.6\end{pmatrix}$ and true classes 1 and 2 (indices 0 and 1). Write $\mathbf{G}_{\mathbf{Z}}$ for the *mean* loss. (d) Show that with
   logits $(z_1, z_2)$ the 2-class softmax gives $p_2 = \sigma(z_2 - z_1)$.
2. **Code:** Write, with gradient checks: (a) `cross_entropy_soft(Z, T)` for target rows $\mathbf{T}$ that are probability
   distributions (for example label smoothing: $1 - \varepsilon$ on the true class, $\varepsilon/(K - 1)$ elsewhere), whose gradient is $(\mathbf{P} - \mathbf{T})/n$;
   (b) `bce_with_logits(z, y)` for binary labels that stays finite at $z = \pm 1000$ in float32, checked against
   `torch.nn.functional.binary_cross_entropy_with_logits`; (c) `softmax_vjp(p, g)`, the product of an upstream gradient $\mathbf{g}$ with
   softmax's Jacobian *without forming it*, $\mathbf{p} \odot (\mathbf{g} - \mathbf{g}\cdot\mathbf{p})$, checked against `J.T @ g` with the explicit Jacobian.
3. **Think:** A training run's loss becomes `nan` after a few thousand steps. The model's last layer is
   `nn.Softmax(dim=1)` followed by a hand-written `-(y_onehot * torch.log(p)).sum(1).mean()`. A teammate proposes
   `torch.log(p + 1e-8)`. Explain what that patch fixes and what it breaks, what you would change instead, and how you
   would find out where the first non-finite value appears if the `nan` persists.

## Takeaways

1. Softmax exponentiates and normalizes; it is unchanged by adding a constant to every logit, so subtract the max before
   exponentiating and nothing overflows or divides by zero.
2. Log-sum-exp, $m + \log\sum_k e^{z_k - m}$, is the stable way to compute $\log\sum_k e^{z_k}$, and log-softmax is $\mathbf{z} - \operatorname{LSE}(\mathbf{z})$.
3. Cross-entropy from logits is $\operatorname{LSE}(\mathbf{z}) - z_y$; never take the log of a computed probability, which rounds to 0 for confident
   mistakes (at logits of about 52 in float32).
4. The gradient of cross-entropy with respect to the logits is $\mathbf{p} - \mathbf{y}$ (divided by $n$ for a batch mean); the Jacobian route
   gives the same answer the long way.
5. With two classes and logits $(0, z)$, softmax cross-entropy is exactly logistic regression's log-loss; frameworks fuse
   softmax and cross-entropy into one stable operation, so the model's last layer outputs logits.

**What builds on this:** [C2.1](c2_1_tensors_autograd_modules.md) (`nn.CrossEntropyLoss` in PyTorch);
[C2.2](c2_2_dataset_dataloader_training_loop.md) (the training loop that uses it); [C4.2](c4_2_mixed_precision.md)
(why losses are computed in float32 under mixed precision); [F1.1](../F/f1_1_self_attention.md) (softmax over attention
scores, with masking by $-\infty$); [F2.4](../F/f2_4_decoding.md) (temperature: softmax of $\mathbf{z}/\tau$); P4 (the fused loss in your engine).

**References:** J. S. Bridle, "Probabilistic interpretation of feedforward classification network outputs, with
relationships to statistical pattern recognition", in *Neurocomputing*, Springer, 1990 (the name "softmax"). I. Goodfellow,
Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016, sections 4.1 and 6.2.2. P. Blanchard, D. J. Higham, N. J. Higham,
"Accurately computing the log-sum-exp and softmax functions", *IMA Journal of Numerical Analysis*, 2021. PyTorch documentation,
`torch.nn.CrossEntropyLoss` (VERIFY link: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
