# Answer key: C1.1 Perceptron and multilayer perceptron

Lesson: [`lessons/C/c1_1_perceptron_mlp.md`](../../../lessons/C/c1_1_perceptron_mlp.md)

## Exercise 1 (by hand)

**Solution (a).** Points $\mathbf{x}_1 = (1, 0)$, $y = +1$; $\mathbf{x}_2 = (0, 1)$, $y = -1$; $\mathbf{x}_3 = (1, 1)$, $y = +1$. Start $\mathbf{w} = (0, 0)$, $b = 0$.

| Epoch | Point | Score $z$ | $yz \le 0$? | New $\mathbf{w}$, $b$ |
|---|---|---|---|---|
| 1 | 1 | $0$ | yes | $(1, 0)$, $1$ |
| 1 | 2 | $0 + 0 + 1 = 1$ | yes ($y = -1$) | $(1, 0) - (0, 1) = (1, -1)$, $0$ |
| 1 | 3 | $1 - 1 + 0 = 0$ | yes | $(1, -1) + (1, 1) = (2, 0)$, $1$ |
| 2 | 1 | $2 + 1 = 3$ | no | — |
| 2 | 2 | $0 + 1 = 1$ | yes | $(2, 0) - (0, 1) = (2, -1)$, $0$ |
| 2 | 3 | $2 - 1 + 0 = 1$ | no | — |
| 3 | 1, 2, 3 | $2$, $-1$, $1$ | no, no, no | converged: $\mathbf{w} = (2, -1)$, $b = 0$ |

**(b)** XNOR is "not XOR" = "AND, or not OR". Keep the hidden layer of the lesson (OR unit: weights $(1, 1)$, bias
$-0.5$; AND unit: weights $(1, 1)$, bias $-1.5$; step activations) and flip the output: $\mathbf{W}^{[2]} = (-1, 1)^\top$ (minus OR, plus
AND), $b^{[2]} = 0.5$. Check: $(0, 0)$: $\mathbf{h} = (0, 0)$, score $0.5$, output 1. $(0, 1)$ and $(1, 0)$: $\mathbf{h} = (1, 0)$, score $-0.5$, output 0.
$(1, 1)$: $\mathbf{h} = (1, 1)$, score $-1 + 1 + 0.5 = 0.5$, output 1. Any weights that pass all four checks are correct (for example
two hidden units "both off" and "both on", OR-ed together at the output).

**(c)** $\mathbf{W}^{[1]}\mathbf{W}^{[2]} = \begin{pmatrix}2 \cdot 1 + (-1) \cdot 2\\ 1 \cdot 1 + 1 \cdot 2\end{pmatrix} = \begin{pmatrix}0\\ 3\end{pmatrix}$, shape `(2, 1)`. At $\mathbf{x} = (1, -1)$: layer by layer,
$\mathbf{x}\mathbf{W}^{[1]} = (2 - 1,\ -1 - 1) = (1, -2)$, then $1 \cdot 1 + (-2) \cdot 2 = -3$; with the single matrix, $1 \cdot 0 + (-1) \cdot 3 = -3$. Same.
With a ReLU after layer 1: $(1, -2) \to (1, 0)$, output $1 \cdot 1 + 0 \cdot 2 = 1$, not $-3$.

```text
== exercise 1 check ==
(a) epoch 1 point 1: z=+0 wrong -> w=[1.0, 0.0] b=+1
(a) epoch 1 point 2: z=+1 wrong -> w=[1.0, -1.0] b=+0
(a) epoch 1 point 3: z=+0 wrong -> w=[2.0, 0.0] b=+1
(a) epoch 2 point 1: z=+3 correct
(a) epoch 2 point 2: z=+1 wrong -> w=[2.0, -1.0] b=+0
(a) epoch 2 point 3: z=+1 correct
(a) epoch 3 point 1: z=+2 correct
(a) epoch 3 point 2: z=-1 correct
(a) epoch 3 point 3: z=+1 correct
(a) converged: w=[2.0, -1.0] b=+0
(b) XNOR outputs [1.0, 0.0, 0.0, 1.0]  expected [1.0, 0.0, 0.0, 1.0]
(c) W1 W2 = [[0.0], [3.0]]  layer by layer [-3.0]  one matrix [-3.0]
(c) with ReLU: hidden [1.0, 0.0]  output [1.0]
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) Every score computed with the *current* weights; updates only when $yz \le 0$ (including $z = 0$); correct final $\mathbf{w}$, $b$ and epoch count | 4 |
| (b) Weights and biases that give the right output on all four inputs, each checked | 3 |
| (c) Product $\mathbf{W}^{[1]}\mathbf{W}^{[2]}$ with the right shape `(2, 1)`, both routes giving $-3$ | 3 |
| (c) ReLU case recomputed: 1, and the point that the collapse no longer holds | 2 |

**Likely mistakes:**

- Treating $z = 0$ as correct. The rule updates when $yz \le 0$; at the start every score is 0, so the first point always
  triggers an update. Skipping it changes every later number.
- Computing all three scores with the weights from the *start* of the epoch (as in batch gradient descent). The
  perceptron is online: each point sees the weights left by the previous one.
- Updating $\mathbf{w}$ but forgetting $b$, or adding $\eta\mathbf{x}$ regardless of the sign of $y$.
- Multiplying in the wrong order ($\mathbf{W}^{[2]}\mathbf{W}^{[1]}$ is `(2, 1) @ (2, 2)`, which does not even have matching shapes). Writing
  shapes next to each product catches this.
- For XNOR, flipping only the output bias and not the weights; check all four rows rather than one.

## Exercise 2 (code)

**Solution:** [`c1_1_perceptron_mlp_ex2.py`](c1_1_perceptron_mlp_ex2.py). The two constructions:

```python
# |x| = relu(x) + relu(-x): two hidden units, weights (1, -1), output weights (1, 1)
W_abs = [np.array([[1.0, -1.0]]), np.array([[1.0], [1.0]])]

# sin on [-pi, pi]: unit j is relu(x - t_j); its output weight is the change of slope at knot t_j
knots = np.linspace(lo, hi, k + 1)
slopes = np.diff(f(knots)) / np.diff(knots)               # (k,)
c = np.concatenate([[slopes[0]], np.diff(slopes)])        # (k,)
Ws = [np.ones((1, k)), c[:, None]];  bs = [-knots[:-1], np.array([f(knots[0])])]
```

Why the sin network works: on $[t_0, t_1]$ only unit 0 is on and the output is $f(t_0) + s_1(x - t_0)$, the first segment. At
each knot $t_j$ one more unit switches on and adds $(s_{j+1} - s_j)(x - t_j)$, which changes the slope from $s_j$ to $s_{j+1}$ without
moving the value at $t_j$ (the new term is 0 there).

```text
== mlp_forward ==
output shape for (5, 3) through 3 -> 4 -> 2: (5, 2)

== |x| with two ReLUs ==
max | net(x) - |x| | on [-3, 3]: 0.0e+00

== piecewise-linear sin with k hidden ReLUs ==
k =  4 hidden units: max error 0.21051  (bound h^2/8 = 0.30843)
k =  8 hidden units: max error 0.07038  (bound h^2/8 = 0.07711)  error ratio to k/2: 2.99
k = 16 hidden units: max error 0.01885  (bound h^2/8 = 0.01928)  error ratio to k/2: 3.73
k = 32 hidden units: max error 0.00479  (bound h^2/8 = 0.00482)  error ratio to k/2: 3.93
```

Doubling $k$ divides the error by about 4: linear interpolation with knot spacing $h = 2\pi/k$ has error at most
$h^2\max\lvert f''\rvert/8 = h^2/8$, so the error falls like $1/k^2$. For a smooth function of one variable that is cheap. The universal
approximation theorem's "enough units" hides how this scales with the input dimension: covering $d$ dimensions with
a grid of spacing $h$ needs on the order of $(1/h)^d$ pieces, which is why the theorem guarantees existence, not
efficiency, and why real networks rely on structure in the data rather than on tiling space.

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| `mlp_forward` loops over layers, applies the activation to all but the last, works for any depth | 2 |
| Shapes stated (docstring or comments); a clear error for mismatched shapes | 2 |
| $\lvert x\rvert$ network exact (error 0 on a grid), with weights explained | 2 |
| Interpolating network built from slopes and slope changes; values at the knots exact | 3 |
| Error table for $k = 4, 8, 16, 32$ and the "÷4 per doubling" ($1/k^2$) observation | 2 |
| Comment on what "enough units" means (existence vs efficiency, dimension) | 1 |

**Likely mistakes:**

- Applying ReLU to the output layer, so the network can never output a negative value (half of $\sin$ comes out as 0).
- Using the slopes themselves as output weights instead of the *changes* in slope: the slopes then add up and the curve
  shoots off.
- Measuring the error only at the knots (where it is 0 by construction). Use a grid much finer than the knots.
- Forgetting the output bias $f(t_0)$, which shifts the whole curve.

## Exercise 3 (think)

**Strong answer:** The likeliest bug is a missing non-linearity: if the layers have no activation (or the activation is
defined but never called, or applied only to the output), six layers collapse to one linear map and the network *is*
a logistic regression, so identical AUC is exactly what you would see. Other bugs with the same symptom: inputs not
reaching the hidden layers (a detached or zeroed tensor), or every ReLU dead so only biases pass through. The
non-bug explanation: the signal in these features really is close to linear (or the features were already
engineered for a linear model), so extra capacity buys nothing; churn data with a few strong, monotone drivers often
looks like that. To tell them apart in under an hour: print the model and check for activations between layers;
feed a batch and check that hidden activations are non-linear in the input (for ReLU, the fraction of units on is
neither 0 nor 1, and the network's output is not an exact linear function of the input: fit a linear regression from
inputs to the network's logits and look at $R^2$; an $R^2$ of 1.0 means the network is linear); train both models on a small
synthetic dataset with a known non-linear boundary (XOR-like) and confirm the MLP beats logistic regression there.
If both really are equally good, ship the logistic regression: cheaper to serve, easier to explain to the business
and to auditors, better calibrated out of the box, and less to maintain. Compare them with confidence intervals on the
AUC (A6.2) before claiming they are equal.

**Weak answer:** "The network is probably overfitting, so add dropout and more data. Deep learning should win, so tune it
more."

**Why it is weak:** identical AUC is not a symptom of overfitting; it does not consider the collapse bug, gives no way
to check, and assumes the bigger model must win instead of choosing the cheaper model when performance ties.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Names the missing-activation collapse (or another bug with the same symptom) and explains why it gives identical AUC | 3 |
| Names a genuine non-bug reason (signal is linear, features already engineered) | 2 |
| A concrete, fast diagnostic (inspect layers, linear fit of the logits, synthetic non-linear test) | 3 |
| Recommends the simpler model if they truly tie, with reasons (cost, explainability, calibration), and checks the tie with CIs | 2 |

## Scoring

Total 34 pts. Pass ≥ 70% (24 pts). Topic ids to revisit for each lost criterion: Exercise 1 (a) → C1.1 (perceptron
rule); (b) → C1.1 (hidden layers as features); (c) → C1.1 and A2.2 (matrix products). Exercise 2 → C1.1 (piecewise
linear networks, universal approximation), A1.2 (shapes). Exercise 3 → C1.1 (collapse without activations), B4.1 (AUC),
A6.2 (confidence intervals).
