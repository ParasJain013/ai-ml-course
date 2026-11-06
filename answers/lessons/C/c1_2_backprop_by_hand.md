# Answer key: C1.2 Backpropagation by hand on a tiny network with real numbers

Lesson: [`lessons/C/c1_2_backprop_by_hand.md`](../../../lessons/C/c1_2_backprop_by_hand.md)

## Exercise 1 (by hand)

**Solution (a).** $\mathbf{x} = (1, 0.5)$, $\mathbf{y} = (0, 1)$, the lesson's weights, ReLU hidden units.

- $z^{[1]}_1 = 1 \times 0.1 + 0.5 \times (-0.2) + 0.1 = 0.1$; $z^{[1]}_2 = 1 \times 0.3 + 0.5 \times 0.4 - 0.1 = 0.4$. Both positive, so $\mathbf{h} = (0.1, 0.4)$.
- $z^{[2]}_1 = 0.1 \times 0.5 + 0.4 \times 0.3 + 0 = 0.17$; $z^{[2]}_2 = 0.1 \times (-0.4) + 0.4 \times 0.2 + 0.1 = 0.14$.
- $\mathbf{a} = (\sigma(0.17), \sigma(0.14)) = (0.54240, 0.53494)$.
- $\mathcal{L} = \frac12[(0.54240 - 0)^2 + (0.53494 - 1)^2] = \frac12[0.29420 + 0.21628] = 0.25524$.

**(b)**

- $\mathbf{a} - \mathbf{y} = (0.54240, -0.46506)$; $\mathbf{a}\odot(1 - \mathbf{a}) = (0.24820, 0.24878)$; $\boldsymbol{\delta}^{[2]} = (0.13462, -0.11570)$, shape `(1, 2)`.
- $\mathbf{G}_{\mathbf{W}^{[2]}} = \mathbf{h}^\top\boldsymbol{\delta}^{[2]} = \begin{pmatrix}0.1 \times 0.13462 & 0.1 \times (-0.11570)\\ 0.4 \times 0.13462 & 0.4 \times (-0.11570)\end{pmatrix} = \begin{pmatrix}0.01346 & -0.01157\\ 0.05385 & -0.04628\end{pmatrix}$, shape `(2, 2)`.
- $\mathbf{G}_{\mathbf{h}} = \boldsymbol{\delta}^{[2]}\mathbf{W}^{[2]\top}$: $g_{h_1} = 0.13462 \times 0.5 + (-0.11570)(-0.4) = 0.06731 + 0.04628 = 0.11359$; $g_{h_2} = 0.13462 \times 0.3 + (-0.11570) \times 0.2 = 0.04039 - 0.02314 = 0.01725$. Shape `(1, 2)`.
- ReLU's derivative is 1 for both units (both $z^{[1]} > 0$), so $\boldsymbol{\delta}^{[1]} = \mathbf{G}_{\mathbf{h}} = (0.11359, 0.01725)$, shape `(1, 2)`.
- $\mathbf{G}_{\mathbf{W}^{[1]}} = \mathbf{x}^\top\boldsymbol{\delta}^{[1]} = \begin{pmatrix}0.11359 & 0.01725\\ 0.05680 & 0.00862\end{pmatrix}$, shape `(2, 2)`.

**(c)** $\mathbf{W}^{[2]} \leftarrow \mathbf{W}^{[2]} - \mathbf{G}_{\mathbf{W}^{[2]}} = \begin{pmatrix}0.48654 & -0.38843\\ 0.24615 & 0.24628\end{pmatrix}$, $\mathbf{b}^{[2]} \leftarrow (0, 0.1) - (0.13462, -0.11570) = (-0.13462, 0.21570)$. New loss 0.21970 < 0.25524.

**(d)** If $z^{[1]}_1 < 0$, then $h_1 = 0$ and ReLU's derivative there is 0, so $\delta^{[1]}_1 = 0$. Every gradient that goes through unit 1 is
zero: the first column of $\mathbf{G}_{\mathbf{W}^{[1]}}$ (the weights *into* unit 1) and $g_{b^{[1]}_1}$. The first row of $\mathbf{G}_{\mathbf{W}^{[2]}}$ (the weights *out of* unit 1) is also zero,
because it is $h_1\boldsymbol{\delta}^{[2]}$ and $h_1 = 0$. If that happens for every input, the unit never receives a gradient again: a **dead ReLU**.
Its weights are frozen and the network has lost a unit.

```text
== exercise 1 check: ReLU hidden layer ==
(a) z1 [[0.1, 0.4]]  h [[0.1, 0.4]]  z2 [[0.17, 0.14]]  a [[0.5424, 0.53494]]  loss 0.25524
(b) delta2 [[0.13462, -0.1157]]  shape (1, 2)
(b) dW2 [[0.01346, -0.01157], [0.05385, -0.04628]]  shape (2, 2)
(b) dh [[0.11359, 0.01725]]  delta1 [[0.11359, 0.01725]]  shape (1, 2)
(b) dW1 [[0.11359, 0.01725], [0.0568, 0.00862]]  shape (2, 2)
(c) W2 [[0.48654, -0.38843], [0.24615, 0.24628]]  b2 [-0.13462, 0.2157]  loss 0.25524 -> 0.21970
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Forward pass with the right row/column convention; ReLU applied; loss to 4 decimals | 3 |
| (b) $\boldsymbol{\delta}^{[2]}$ with the sigmoid derivative from the output $\mathbf{a}$ | 2 |
| (b) $\mathbf{G}_{\mathbf{W}^{[2]}}$ as an outer product and $\mathbf{G}_{\mathbf{h}}$ summing over both outputs | 3 |
| (b) $\boldsymbol{\delta}^{[1]}$ with ReLU's derivative (1 here), $\mathbf{G}_{\mathbf{W}^{[1]}}$; every shape stated | 2 |
| (c) Step with the minus sign; new loss lower | 2 |
| (d) Names the zeroed gradients (into and out of the unit) and the dead-ReLU consequence | 2 |

**Likely mistakes:**

- Using the sigmoid derivative for the hidden layer out of habit. With ReLU the factor is 1 or 0, not $h(1 - h)$.
- Taking $\mathbf{G}_{\mathbf{h}}$ from one output only. Each hidden unit feeds both outputs; the two contributions must be added (here they have
  the same sign for unit 1 because both the weight and the delta flip sign).
- Reading $\mathbf{W}$ by rows when the convention is columns: $z^{[1]}_1$ uses the *first column* $(0.1, -0.2)$, not the first row $(0.1, 0.3)$.
- In (d), saying only the incoming weights are affected. The outgoing weights' gradients are also zero, because $h_1 = 0$.

## Exercise 2 (code)

**Solution:** [`c1_2_backprop_by_hand_ex2.py`](c1_2_backprop_by_hand_ex2.py). The backward loop is the dense-layer table
applied from the last layer down:

```python
D = 2 * (out - Y) / n                                     # delta of the linear output layer, (n, k)
for l in range(len(layers) - 1, -1, -1):
    grads[l]["W"] = Hs[l].T @ D                           # (d_l, n) @ (n, d_{l+1})
    grads[l]["b"] = D.sum(axis=0)                         # (d_{l+1},)
    if l > 0:
        D = (D @ layers[l]["W"].T) * (Zs[l - 1] > 0)      # through W, then through the ReLU below
```

The loss is the mean over examples of the summed squared error, so the output delta is $\frac{2}{n}(\hat{\mathbf{Y}} - \mathbf{Y})$ (the 2 comes from
differentiating the square; there is no $\frac12$ this time).

```text
== general MLP forward/backward ==
layer 1 dW shape (3, 4)  rel error < 1e-8
layer 1 db shape (4,)    rel error < 1e-8
layer 2 dW shape (4, 5)  rel error < 1e-8
layer 2 db shape (5,)    rel error < 1e-8
layer 3 dW shape (5, 2)  rel error < 1e-8
layer 3 db shape (2,)    rel error < 1e-8
gradient check 3-4-5-2 on a batch of 5: PASS

== training 1-32-32-1 on sin(3x) ==
step    0: train MSE 2.7254
step  500: train MSE 0.0829
step 1000: train MSE 0.0658
step 2000: train MSE 0.0419
step 5000: train MSE 0.0149
noise variance (the best possible MSE on average): 0.0100
saved outputs/c1_2/c1_2_ex2_sin_fit.png
```

The training MSE approaches the noise variance (0.01), which is as low as a model that does not memorize the noise can
go. A learner's numbers will differ with their initialization and learning rate; what matters is a passing check and
a loss that falls towards about 0.01–0.03.

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| Works for any list of widths; forward caches each layer's input and pre-activation | 3 |
| Backward loop: $\mathbf{X}^\top\boldsymbol{\Delta}$, column sums, $\boldsymbol{\Delta}\mathbf{W}^\top$, ReLU mask from the *pre-activation* of the layer below | 3 |
| Correct output delta for the chosen loss (factor 2 or $\frac12$, and $\frac1n$) | 2 |
| Gradient check on every parameter, float64, non-zero biases, relative error $< 10^{-6}$ | 3 |
| Training loss falls; plot of the fit; seed set | 2 |
| Shapes asserted or documented; readable | 1 |

**Likely mistakes:**

- Checking gradients with all-zero biases: a bug in the bias gradient can hide, and every ReLU is on or off for the same
  reason. Use random biases in the check.
- Masking with the current layer's pre-activation instead of the one below (off by one layer), or with the post-ReLU
  value `H > 0` (the same mask for ReLU, but a habit that breaks for other activations).
- A learning rate that is too large for the $\frac{2}{n}$ scaling: the loss becomes `nan` within a few steps. Halve it until the loss falls.
- Forgetting that the output layer is linear and applying ReLU there too, so the network cannot output $\sin < 0$.

## Exercise 3 (think)

**Strong answer:** Central differences need two forward passes per parameter, so one numerical gradient of a
1-million-parameter model costs about 2 million forward passes. Backprop costs one forward pass and a backward pass of
about twice the cost, so about 3 forward passes in total: the numerical gradient is roughly 700,000 times more expensive, and
it is also less accurate (its error depends on the step size: truncation error if $h$ is too large, rounding error if it is too small,
A4.2). It is not an option for training. You still compute numerical gradients (1) when you write a new layer, loss or
custom `autograd.Function` by hand, as a unit test on a tiny input, and (2) when debugging a training run whose
gradients look wrong (`nan`, exploding, or a model that will not learn), on a small slice of the model and a small batch.
Do it in float64 with $h \approx 10^{-6}$ and require a relative error below about $10^{-6}$ (below $10^{-8}$ is typical for smooth functions); check a
random subset of parameters if the layer is large; avoid points where the function has a kink (ReLU at exactly 0), which
give spurious failures.

**Weak answer:** "Numerical gradients are slower, so use backprop. You can use numerical gradients for testing."

**Why it is weak:** no estimate of how much slower (the question asks for one), no mention of accuracy, and "for testing"
is not concrete: no situation, precision or tolerance.

**Rubric (8 pts):**

| Criterion | Points |
|---|---|
| Cost estimate: ~2 forward passes per parameter (~2 million) vs ~3 for backprop | 3 |
| Mentions the accuracy problem of finite differences (step size trade-off) | 1 |
| Two concrete situations (new layer/loss, debugging) | 2 |
| Precision and tolerance: float64, $h \approx 10^{-6}$, relative error $< 10^{-6}$; kinks or subsampling mentioned | 2 |

## Scoring

Total 36 pts. Pass ≥ 70% (26 pts). Topic ids to revisit for each lost criterion: Exercise 1 forward → C1.1 (layer
convention); backward → C1.2 and A4.3 (chain rule, add over uses); (d) → C1.2 failure modes, C3.3. Exercise 2 → C1.2
(dense-layer table), A4.2 (gradient checking). Exercise 3 → A4.2, A4.3 (reverse mode cost).
