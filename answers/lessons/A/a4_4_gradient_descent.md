# Answer key: A4.4 Gradient descent

Lesson: [`lessons/A/a4_4_gradient_descent.md`](../../../lessons/A/a4_4_gradient_descent.md)

## Exercise 1 (by hand)

**Solution (a).** $\mathcal{L}(w) = 2(w - 3)^2 = \frac{4}{2}(w - 3)^2$, so the curvature is $a = 4$ (also
$\mathcal{L}'' = 4$) and the largest stable rate is $2/a = 0.5$. The derivative is $4(w - 3)$, so each step
multiplies $w - 3$ by $1 - 4\eta$. From $w_0 = 0$ the error starts at $-3$:

| $\eta$ | Factor | $w_1$, $w_2$, $w_3$ | Behaviour |
|---|---|---|---|
| 0.1 | $+0.6$ | 1.2, 1.92, 2.352 | monotone (below $1/a = 0.25$) |
| 0.25 | $0$ | 3, 3, 3 | one step ($\eta = 1/a$) |
| 0.4 | $-0.6$ | 4.8, 1.92, 3.648 | oscillating, converging (between $1/a$ and $2/a$) |
| 0.6 | $-1.4$ | 7.2, −2.88, 11.232 | divergent (above $2/a$) |

**Solution (b).** The eigenvalues of a diagonal matrix are its diagonal entries, 4 and 1, so
$\eta < 2/4 = 0.5$. With $\eta = 0.4$: steep direction factor $1 - 0.4 \cdot 4 = -0.6$ (oscillates, because it is
negative); slow direction factor $1 - 0.4 \cdot 1 = 0.6$ (monotone). Steps for a 1,000-fold shrink:
$0.6^k = 10^{-3}$, $k = \ln(10^{-3})/\ln(0.6) \approx 13.5$, so 14 steps. Here both directions shrink at the same
rate $\lvert 0.6\rvert$, because $\eta = 0.4 = 2/(\lambda_{\min} + \lambda_{\max})$ is exactly the best fixed rate, which
balances the two.

```text
== exercise 1 check ==
lr = 0.1  factor 1 - 4 lr = +0.6   w1, w2, w3 = [1.2, 1.92, 2.352]
lr = 0.25 factor 1 - 4 lr = +0.0   w1, w2, w3 = [3.0, 3.0, 3.0]
lr = 0.4  factor 1 - 4 lr = -0.6   w1, w2, w3 = [4.8, 1.92, 3.648]
lr = 0.6  factor 1 - 4 lr = -1.4   w1, w2, w3 = [7.2, -2.88, 11.232]
curvature 4.0: factor -0.6  steps for a 1000-fold shrink = 13.5
curvature 1.0: factor +0.6  steps for a 1000-fold shrink = 13.5
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) Curvature 4 and limit 0.5 | 2 |
| (a) Twelve iterates correct | 3 |
| (a) Each run classified correctly | 2 |
| (b) Limit 0.5 from the largest eigenvalue | 1 |
| (b) Factors $-0.6$ and $0.6$; the steep direction oscillates | 2 |
| (b) About 14 steps, with the logarithm shown | 2 |

**Likely mistakes:**

- Taking the curvature as 2 (the coefficient) instead of 4 (the second derivative): the limit comes out
  as 1, and $\eta = 0.6$ is wrongly called stable. Write the loss as $\frac{a}{2}(w - w^\star)^2$ first.
- Adding the gradient instead of subtracting it.
- Assuming the steep direction is the slow one. The *flat* direction (small curvature) is slow; the steep
  one is where instability starts.
- Rounding $k$ down to 13: $0.6^{13} \approx 1.3 \times 10^{-3}$ is not yet a 1,000-fold shrink.

## Exercise 2 (code)

**Solution:** [`a4_4_gradient_descent_ex2.py`](a4_4_gradient_descent_ex2.py). The line search:

```python
loss_w, lr = mse(w), 1.0
for _ in range(max_halvings):
    if mse(w - lr * g) <= loss_w - 0.5 * lr * gg:   # gg = ||g||^2: "sufficient decrease"
        break
    lr /= 2
else:
    raise RuntimeError(...)
w = w - lr * g
```

```text
== exercise 2a: the largest stable learning rate ==
bisection: largest converging lr ~ 0.02700   theory 2 / lambda_max = 0.02701

== exercise 2b: backtracking line search ==
fixed lr = 0.025  iterations =  1175   loss evaluations =   1174   w = [1.7575, 0.5523]  matches to 4 decimals: OK
backtracking      iterations =   418   loss evaluations =   2900   w = [1.7575, 0.5523]  matches to 4 decimals: OK

== exercise 2c: asking for ||g|| < 1e-10 ==
no convergence in 20000 iterations: gradient norm still 5.0e-08
loss values near the minimum are spaced 1.1e-16 apart; the required decrease 0.5 lr ||g||^2 at ||g|| = 1e-8 is at most 5.0e-17
```

(a) Bisection agrees with $2/\lambda_{\max}$ to the bisection tolerance: the eigenvalue analysis predicts the
real limit exactly, because the MSE is exactly quadratic.

(b) Backtracking needs about a third of the iterations, because it sometimes takes steps larger than
$2/\lambda_{\max}$ would allow forever: a single step can be safe even when repeating it would diverge. But it
spends several loss evaluations per iteration on the halvings, so in total work it is *not* cheaper here
(2,900 loss evaluations plus 418 gradients against 1,174 plus 1,175). Its real advantage is that you do
not need to know $\lambda_{\max}$ or tune $\eta$ at all, which matters when the curvature changes from place to
place (non-quadratic losses). The learner's counts may differ with a different stopping test or
halving factor; the comparison and its explanation are graded.

(c) The gradient norm stalls near $5 \times 10^{-8}$. The test needs
$\mathcal{L}(\mathbf{w} - \eta\mathbf{g}) \le \mathcal{L}(\mathbf{w}) - \frac12\eta\lVert\mathbf{g}\rVert^2$, but near the minimum the loss is about 0.99 and
float64 numbers there are $1.1 \times 10^{-16}$ apart, while the required decrease is below $10^{-16}$: it rounds
away, so the comparison degenerates into "is the new loss ≤ the old one *as rounded*", which accepts and
rejects steps more or less at random. Any method that judges progress by loss *values* can locate a
minimum only to about $\sqrt{\varepsilon}$ relative accuracy in $\mathbf{w}$ (a quadratic changes by $\delta^2$ when $\mathbf{w}$
moves by $\delta$); fixed-step GD uses only gradients, which keep shrinking linearly, so it reaches
$10^{-10}$. (The same reason is behind A4.1's U-curve: differences of nearly equal numbers.) A stopping
test at $10^{-6}$ to $10^{-8}$ is the practical choice.

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) A correct bisection with a robust "converges" test (gradient tolerance, divergence guard, cap) | 3 |
| (a) Result compared with $2/\lambda_{\max}$ and the agreement explained | 1 |
| (b) Correct Armijo backtracking (starts at 1, halves, sufficient-decrease test) with a halving cap | 3 |
| (b) Iterations *and* loss evaluations reported for both methods, weights checked | 2 |
| (b) A fair comparison: fewer iterations but not necessarily less work; no tuning needed | 2 |
| (c) Reports the stall and explains it with float spacing versus the required decrease | 3 |

**Likely mistakes:**

- A "converges" test that only checks for NaN after a fixed number of steps: rates just below the limit
  look like failures (they are slow) and rates just above look fine for a while (divergence starts slowly
  from a small error). Test the gradient norm, guard against blow-up, and allow enough iterations.
- A `while` loop without a halving cap: in (c) it can halve forever, or until `lr` underflows to 0.
- Comparing only iteration counts and concluding backtracking is 3 times faster.
- Explaining (c) as "the learning rate got too small". It is a symptom; the cause is that the loss can no
  longer resolve the decrease being asked for.

## Exercise 3 (think)

**Strong answer:** The Hessian of a linear model's MSE is $\frac{2}{n}\mathbf{X}^\top\mathbf{X}$ (for a logistic or neural
model, the same structure appears near the inputs). A feature with values up to $10^7$ adds a diagonal
entry of order $\frac{2}{n}\sum x^2 \approx 10^{14}$, so $\lambda_{\max}$ jumps by many orders of magnitude and the stable
learning rate $2/\lambda_{\max}$ falls to about $10^{-14}$. The learning rate tuned for the old features is now
far above the limit: the update overshoots, the error grows by a large factor every step, and within a
few steps the numbers overflow to `inf` and then `nan` (`inf - inf`). Fix: scale the feature
(standardize it, or use `log1p` for a heavy-tailed money amount, then standardize), with the scaler
fitted on training data only and saved with the model so serving applies the same transform. Before
training: a data-validation step on the feature table (schema with expected ranges or units per column,
checks on summary statistics against the previous run) and a unit test that the preprocessing output has
mean about 0 and standard deviation about 1 per column; review new features with a units check. During
training: stop on non-finite loss or gradients, log the gradient norm and loss every few steps and alert
on spikes, optionally clip the gradient norm as a guard (it prevents the blow-up but does not fix the
conditioning), and fail the nightly job rather than ship a model trained on a `nan` loss.

**Weak answer:** "Lower the learning rate until it stops producing `nan`." Weak because it treats the
symptom: a rate small enough for the new feature (about $10^{-14}$) makes every other weight effectively
frozen, so the model trains far worse, and nothing stops the next unscaled feature from doing it again.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Explains via the Hessian: the large feature raises $\lambda_{\max}$, so $\eta > 2/\lambda_{\max}$ and the updates diverge to `inf`/`nan` | 3 |
| Fix by scaling (standardize or log-transform), fitted on training data and shipped with the model | 2 |
| Pre-training checks (schema, ranges, distribution or scaling tests) | 2 |
| In-training guards (stop on non-finite values, monitor gradient norm, clipping as a guard) | 2 |
| Says why "lower the learning rate" alone is a bad fix | 1 |

## Scoring

Total 36 pts. Pass ≥ 70% (26 pts). Topic ids to revisit for each lost criterion: exercise 1 → A4.4
"The math" (one dimension, eigen-directions); exercise 2 → A4.4 "From scratch" and A4.1 (rounding of
differences); exercise 3 → A4.4 "Feature scale sets κ" and "In practice", and B1.4 (fitting scalers
without leakage).
