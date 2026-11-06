# Answer key: C1.4 Softmax and cross-entropy, written to be numerically stable

Lesson: [`lessons/C/c1_4_stable_softmax_cross_entropy.md`](../../../lessons/C/c1_4_stable_softmax_cross_entropy.md)

## Exercise 1 (by hand)

**Solution (a).** $e^{\mathbf{z}} = (e^0, e^{\log 2}, e^{\log 5}) = (1, 2, 5)$, sum 8, so $\mathbf{p} = (\tfrac18, \tfrac14, \tfrac58)$. True class 2: $\ell = -\log\tfrac14 = \log 4 \approx 1.3863$.
$\mathbf{p} - \mathbf{y} = (\tfrac18, \tfrac14 - 1, \tfrac58) = (0.125, -0.75, 0.625)$; the entries sum to 0.

**(b)** $m = 500$; $\mathbf{z} - m = (0, 0, -2)$; $\sum e^{z_k - m} = 1 + 1 + e^{-2} = 2.1353$; $\operatorname{LSE} = 500 + \log 2.1353 = 500 + 0.7586 = 500.7586$.
Softmax $= (1, 1, 0.1353)/2.1353 = (0.4683, 0.4683, 0.0634)$.

**(c)** $\mathbf{Y} = \begin{pmatrix}1 & 0 & 0\\ 0 & 1 & 0\end{pmatrix}$, $\mathbf{G}_{\mathbf{Z}} = \frac12(\mathbf{P} - \mathbf{Y}) = \frac12\begin{pmatrix}-0.3 & 0.2 & 0.1\\ 0.1 & -0.7 & 0.6\end{pmatrix} = \begin{pmatrix}-0.15 & 0.10 & 0.05\\ 0.05 & -0.35 & 0.30\end{pmatrix}$, shape `(2, 3)`.

**(d)** $p_2 = \frac{e^{z_2}}{e^{z_1} + e^{z_2}}$. Divide top and bottom by $e^{z_2}$: $p_2 = \frac{1}{e^{z_1 - z_2} + 1} = \frac{1}{1 + e^{-(z_2 - z_1)}} = \sigma(z_2 - z_1)$. Only the difference of the
two logits matters.

```text
== exercise 1 check ==
(a) p [0.125, 0.25, 0.625] = [1/8, 1/4, 5/8]  loss 1.3863 = log 4  p - y [0.125, -0.75, 0.625]
(b) LSE 500.7586  softmax [0.4683, 0.4683, 0.0634]
(c) G_Z = (P - Y) / 2 = [[-0.15, 0.1, 0.05], [0.05, -0.35, 0.3]]
(d) softmax([z1, z2])[1] 0.197816  sigmoid(z2 - z1) 0.197816
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) Exact $\mathbf{p}$ (using $e^{\log a} = a$), loss $\log 4$, gradient with the right sign on the true class | 3 |
| (b) Max subtracted first; LSE and softmax to 4 decimals | 3 |
| (c) $\mathbf{P} - \mathbf{Y}$ with the $\frac1n$ for the mean; shape stated | 3 |
| (d) The algebra to $\sigma(z_2 - z_1)$ | 3 |

**Likely mistakes:**

- Computing $e^{500}$ on a calculator in (b) and getting an overflow, or rounding $e^{-2}$ to 0. The point of the formula is that
  the hand calculation is as easy as the computer's.
- Writing $\mathbf{y} - \mathbf{p}$ (the negative of the gradient) and then descending in the wrong direction. A quick sanity check: the
  true class's entry must be negative (raising that logit lowers the loss).
- Forgetting the $\frac12$ in (c), or dividing by $K$ instead of $n$.
- In (d), getting $\sigma(z_1 - z_2)$: that is $p_1$, not $p_2$.

## Exercise 2 (code)

**Solution:** [`c1_4_stable_softmax_cross_entropy_ex2.py`](c1_4_stable_softmax_cross_entropy_ex2.py). The three
functions:

```python
def cross_entropy_soft(Z, T):                      # T rows are distributions
    log_p = Z - logsumexp(Z)[:, None]
    return -np.mean(np.sum(T * log_p, axis=1))     # gradient (softmax(Z) - T) / n

def bce_with_logits(z, y):                         # log(1 + e^z) - y z, stable
    return np.maximum(z, 0) - y * z + np.log1p(np.exp(-np.abs(z)))

def softmax_vjp(p, g):                             # g^T (diag(p) - p p^T)
    return p * (g - g @ p)
```

*Why the soft-target gradient is still $\mathbf{P} - \mathbf{T}$:* $\ell = -\sum_k t_k(z_k - \operatorname{LSE}) = \operatorname{LSE}\sum_k t_k - \sum_k t_kz_k$; with $\sum_k t_k = 1$, $\partial\ell/\partial z_j = p_j - t_j$.
*Why the BCE rewrite is stable:* for $z \ge 0$, $\log(1 + e^z) = z + \log(1 + e^{-z})$; for $z < 0$ it is $\log(1 + e^z)$ directly. Both are $\max(z, 0) + \log(1 + e^{-\lvert z\rvert})$,
whose exponent is never positive. `log1p` keeps precision when $e^{-\lvert z\rvert}$ is tiny.
*Why the VJP formula:* $(\mathbf{g}^\top\mathbf{J})_j = \sum_k g_kp_k(\delta_{kj} - p_j) = g_jp_j - p_j\sum_k g_kp_k = p_j(g_j - \mathbf{g}\cdot\mathbf{p})$.

```text
== (a) cross-entropy with soft targets ==
label-smoothed targets, rows sum to [1.0, 1.0, 1.0, 1.0]
gradient (P - T)/n vs numeric: rel error < 1e-8  PASS
loss 2.270274   torch cross_entropy with probability targets 2.270274
note: torch label_smoothing=0.1 spreads eps over all K classes: 2.265534 = ours with that T 2.265534

== (b) binary cross-entropy with logits ==
float32 z [-1000.0, -5.0, 0.0, 5.0, 1000.0]  y [1.0, 0.0, 1.0, 1.0, 0.0]
ours  [1000.0, 0.0067, 0.6931, 0.0067, 1000.0]  dtype float32
torch [1000.0, 0.0067, 0.6931, 0.0067, 1000.0]
naive [inf, 0.006715324241667986, 0.6931471824645996, 0.006715324241667986, inf]
gradient (sigmoid(z) - y)/n vs numeric: rel error < 1e-8  PASS

== (c) softmax vector-Jacobian product ==
vjp     [-0.0993, 0.0143, 0.2142, -0.3648, 0.1824, 0.0532]
J.T @ g [-0.0993, 0.0143, 0.2142, -0.3648, 0.1824, 0.0532]
matches the explicit Jacobian: OK  (O(K) work and memory instead of O(K^2))
```

Note the label-smoothing convention: the exercise's $\varepsilon/(K - 1)$ on the wrong classes and PyTorch's `label_smoothing=ε`
(which puts $\varepsilon/K$ on *every* class, the true one included, so the true class gets $1 - \varepsilon + \varepsilon/K$) are different targets and
give different losses. Both are valid; a learner who compares with PyTorch must use the same convention.

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Soft-target loss from log-softmax, gradient $(\mathbf{P} - \mathbf{T})/n$ checked numerically | 3 |
| (a) Label-smoothed targets built correctly (rows sum to 1); convention difference with PyTorch noticed if compared | 1 |
| (b) Stable BCE (no positive exponent, `log1p`), finite at $\pm1000$ in float32, matches PyTorch | 3 |
| (b) Gradient $(\sigma(z) - y)/n$ checked | 1 |
| (c) VJP without forming $\mathbf{J}$, derived or explained, checked against the explicit Jacobian | 3 |
| Tests in float64 for gradient checks, float32 for the overflow test; readable, typed | 3 |

**Likely mistakes:**

- Computing BCE as `-(y * log(sigmoid(z)) + (1 - y) * log(1 - sigmoid(z)))` with a stable sigmoid: still `inf`, because
  $1 - \sigma(1000)$ is exactly 0 in floating point.
- Checking `bce_with_logits` only in float64 at $\pm1000$, where the naive form may survive more cases. The exercise asks for
  float32 because that is where training runs.
- Assuming PyTorch's `label_smoothing` matches a hand-built smoothed target and concluding one of them is wrong.
- In (c), forgetting that softmax's Jacobian is symmetric and then worrying about `J @ g` versus `J.T @ g` (they are equal here,
  but not for a general layer; write the transpose).

## Exercise 3 (think)

**Strong answer:** The `nan` almost certainly comes from `torch.log(p)` where `p` has rounded to exactly 0: a confidently
wrong prediction (logit gaps above about 50 in float32, or less in float16/bfloat16) makes the true class's probability
underflow, the log is `-inf`, `0 * -inf` in the masked sum is `nan`, and one `nan` in the loss puts `nan` in every gradient
and every weight. Adding `1e-8` makes the log finite, so the run continues, but it caps the loss at $-\log 10^{-8} \approx 18.4$ and, worse,
flattens the gradient: below $10^{-8}$ the loss no longer changes with the logit, so the examples the model gets most badly wrong
stop contributing gradient. It also biases every small probability. The real fix is to remove the softmax layer and
have the model output logits, then use `F.cross_entropy(logits, targets)` (integer targets, or `log_softmax` with soft
targets), which computes $\operatorname{LSE}(\mathbf{z}) - z_y$ and never forms the probability. Keep `F.softmax` only for reporting probabilities. If
the `nan` persists after that, the cause is upstream: logits that are themselves `inf`/`nan` from an exploding activation
or a too-large learning rate. To find it, turn on `torch.autograd.set_detect_anomaly(True)` for a short debugging run (it
names the operation whose backward produced the first `nan`), add forward hooks or assertions that check
`torch.isfinite` on each layer's output, log the max absolute logit and the gradient norm per step to see what grew
before the failure, and then fix the cause (lower the learning rate, add gradient clipping, check the inputs for bad
values), rather than masking the loss.

**Weak answer:** "Adding 1e-8 fixes the log(0) problem, so that is fine. If it still fails, lower the learning rate."

**Why it is weak:** it accepts a patch that hides the symptom and silently removes the gradient of the worst examples,
never names the real fix (logits plus a fused loss), and offers no method to locate a remaining `nan`.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Explains the mechanism: probability underflows to 0, log gives `-inf`, `nan` spreads to every weight | 2 |
| What the epsilon patch breaks: capped loss and vanishing gradient for the worst examples, bias | 2 |
| Real fix: logits out, `F.cross_entropy` / `log_softmax`, no softmax layer before the loss | 3 |
| A concrete way to localize a remaining `nan` (anomaly detection, finiteness checks, logging logit and gradient norms) and fixing the upstream cause | 3 |

## Scoring

Total 36 pts. Pass ≥ 70% (26 pts). Topic ids to revisit for each lost criterion: Exercise 1 (a), (c) → C1.4 (gradient
$\mathbf{p} - \mathbf{y}$); (b) → C1.4 (log-sum-exp), A3.4; (d) → C1.4 (two classes), B2.1. Exercise 2 → C1.4, B2.1 (stable log-loss), C1.3
(vector–Jacobian products). Exercise 3 → C1.4 (failure modes), C4.3 (debugging training), C4.2 (reduced precision).
