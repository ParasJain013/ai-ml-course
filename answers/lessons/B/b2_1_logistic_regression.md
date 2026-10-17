# Answer key: B2.1 Logistic regression: sigmoid, log-loss, gradient descent from scratch

Lesson: [`lessons/B/b2_1_logistic_regression.md`](../../../lessons/B/b2_1_logistic_regression.md)

## Exercise 1 (by hand)

**Solution (a).** $z = wx + b = (-1.5, -0.5, 1.5)$. $p = \sigma(z)$: $\sigma(-1.5) = 1/(1 + e^{1.5}) = 1/5.4817 = 0.1824$, $\sigma(-0.5) = 1/(1 + e^{0.5}) = 1/2.6487 = 0.3775$,
$\sigma(1.5) = 1 - \sigma(-1.5) = 0.8176$. Losses: $y = 0$ uses $-\log(1 - p)$: $-\log 0.8176 = 0.2014$ and $-\log 0.6225 = 0.4741$; $y = 1$ uses $-\log p$: $-\log 0.8176 = 0.2014$.
Mean $= 0.8769/3 = 0.2923$.

**(b)** $p - y = (0.1824, 0.3775, -0.1824)$. $\partial\mathcal{L}/\partial w = \frac13[0.1824(-1) + 0.3775(0) + (-0.1824)(2)] = \frac13(-0.5472) = -0.1824$.
$\partial\mathcal{L}/\partial b = \frac13(0.1824 + 0.3775 - 0.1824) = 0.1258$.

**(c)** $w = 1 - 0.5(-0.1824) = 1.0912$, $b = -0.5 - 0.5(0.1258) = -0.5629$. New $p = (0.1606, 0.3629, 0.8347)$, mean loss 0.2688. Lower than 0.2923: yes.

**(d)** $e^1 = 2.718$: each extra unit of $x$ multiplies the odds of $y = 1$ by about 2.7 (log-odds up by 1).

**(e)** Yes: any threshold between $x = 0$ and $x = 2$ (for example $x = 1$) puts both negatives on one side and the positive on
the other. So there is no finite minimum: gradient descent keeps growing $w$ (and pushing $b$ to keep the boundary in the
gap), the loss creeps towards 0 and never stops falling. After 10,000 more steps, $w = 7.6$ and the loss is 0.0004.

```text
== exercise 1 check ==
(a) z [-1.5, -0.5, 1.5]  p [0.1824, 0.3775, 0.8176]  per-example loss [0.2014, 0.4741, 0.2014]  mean 0.2923
(b) dL/dw -0.1824  dL/db 0.1258
(c) w 1.0912  b -0.5629  p [0.1606, 0.3629, 0.8347]  mean loss 0.2688
(d) odds ratio e^1 = 2.7183
(e) after    100 more steps: b -2.554  w 3.142  loss 0.033995
(e) after   1000 more steps: b -4.728  w 5.282  loss 0.003922
(e) after  10000 more steps: b -7.015  w 7.565  loss 0.000399
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) $z$, $p$ to 4 decimals, the right log term for each label, mean loss | 3 |
| (b) Both partial derivatives, with the $\frac1n$ and the sign of $p - y$ | 3 |
| (c) The step with the minus sign, new loss, and "went down" | 2 |
| (d) Odds ratio in words ("multiplies the odds", not "multiplies the probability") | 1 |
| (e) Separable: yes, with a threshold; weights grow without bound and the loss tends to 0 without a minimum | 3 |

**Likely mistakes:**

- Using $-\log p$ for every example, including the $y = 0$ ones. The loss for a negative is $-\log(1 - p)$.
- Writing the gradient as $\sum(y - p)x$ and then *adding* it: two sign errors that cancel, or one that does not. Check that
  the step lowers the loss.
- Forgetting the $\frac1n$, which gives a gradient three times too large (harmless for the direction, wrong for the number).
- Saying "the probability goes up by a factor of 2.7". It is the *odds* that multiply; probability cannot exceed 1.
- Answering (e) with "the loss will reach 0". It approaches 0 and never gets there; the point is that no finite $\mathbf{w}$ is the answer.

## Exercise 2 (code)

**Solution:** [`b2_1_logistic_regression_ex2.py`](b2_1_logistic_regression_ex2.py). The core pieces:

```python
def sigmoid(z):
    ez = np.exp(-np.abs(z))                       # never overflows
    return np.where(z >= 0, 1.0 / (1.0 + ez), ez / (1.0 + ez))

def mean_log_loss(z, y):
    return float(np.mean(np.logaddexp(0.0, z) - y * z))

def gradient(theta, A, y):
    return A.T @ (sigmoid(A @ theta) - y) / len(y)   # (k,)
```

Mini-batch GD shuffles the rows each epoch (`gen.permutation(n)`) and takes one step per batch of 32 rows (11 steps
per epoch for 343 rows), with the gradient averaged over the batch.

```text
== data and standardization ==
train (343, 6)  val (113, 6)

== gradient check ==
relative error 1.4e-10: OK

== full-batch GD versus scikit-learn ==
steps 2049  theta [-0.8168, 4.8585, 1.7987, 1.5642, 2.1219, 0.0426]
max |p_mine - p_sklearn| on val 2.3e-06
match to 3 decimals: OK

== mini-batch versus full batch, same examples processed ==
optimum train loss 0.1267; mini-batch makes 11 updates per epoch
epoch   1 (   343 examples): full batch 0.3336   mini-batch 0.1724
epoch   5 (  1715 examples): full batch 0.2096   mini-batch 0.1357
epoch  20 (  6860 examples): full batch 0.1543   mini-batch 0.1275
epoch 100 ( 34300 examples): full batch 0.1305   mini-batch 0.1270
```

For the same number of examples processed, mini-batch is far ahead early: one epoch gives 11 updates instead of 1. It
then stalls slightly above the optimum (0.1270 against 0.1267): with a fixed learning rate the noisy batch gradients keep
it jittering around the minimum. Shrinking the learning rate over time (C3.2) removes that floor. Full batch eventually
reaches the exact minimum (2,049 steps). This is why neural networks are trained with mini-batches.

The step count differs by one from the lab (2049 against 2050) only because this version counts the steps *taken*
before the stopping test and the lab counts loss evaluations. The weights are identical.

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| Stable sigmoid and loss from logits (no `log(sigmoid(...))`) | 2 |
| Vectorized gradient `A.T @ (p - y) / n`, shape `(d + 1,)`; no Python loop over rows | 2 |
| Gradient check with central differences and a relative error below $10^{-6}$ | 2 |
| Standardization with training statistics only; intercept not standardized | 2 |
| Full-batch GD matches `LogisticRegression(C=np.inf)` to 3 decimals on validation | 2 |
| Mini-batch with a reshuffle every epoch, compared at equal examples processed | 2 |
| Readability: type hints, shapes in docstrings, seeds, no global state; time of the first attempt recorded | 2 |

**Likely mistakes:**

- Comparing against `LogisticRegression()` with its default `C=1`: the probabilities will not match, because the default
  is penalized. Or using `penalty=None`, which works in 1.9.1 but raises a deprecation warning.
- Taking column 0 of `predict_proba` (that is $P(y = 0)$).
- Fitting the scaler on all rows, or standardizing the column of ones (its standard deviation is 0, giving `nan`).
- Comparing mini-batch and full batch per *step* rather than per example processed, which makes mini-batch look slow.
- Not reshuffling between epochs, so every epoch sees the same batches in the same order.
- Using `sum` instead of `mean` in the gradient: the learning rate then has to shrink with $n$, and the bound $1/L$ no longer applies.

## Exercise 3 (think)

**Strong answer:** 100% training accuracy is the symptom of the problem, not a recommendation. With 30 features the
training tumours are perfectly separable, so as $C$ grows the unpenalized optimum does not exist: the weights grow
(|w| from 3.5 at $C = 1$ to 98 at $C = 10^6$), training loss goes to zero, and the probabilities become overconfident. On the
validation rows log-loss gets 28 times worse (0.026 to 0.733): the model gives near-certain wrong answers to the tumours
on the wrong side of its plane, which for a triage tool means a malignant tumour flagged as "0.000% chance". Use a
penalized model (the default L2, or L1/elastic net, B2.2) inside a pipeline with the scaler, choose $C$ by stratified
cross-validation on the training data with log-loss (or another probability-aware metric) as the criterion, and check
on held-out data that the probabilities are calibrated (B4.1). Report the test score once, at the end. The threshold
is a decision about costs, not a model property: a missed malignant tumour (false negative) is far more costly than an
extra second reading (false positive), so the threshold should be set well below 0.5, chosen on validation data to
reach a required sensitivity (say 99%) and then checked for the workload it creates (B4.1, B4.4). The clinicians should
agree that trade-off; it should not be an engineering default. A model this accurate on 569 rows from one
institution also needs external validation before clinical use.

**Weak answer:** "C=1e6 overfits, so use the default C=1 instead. Thresholds can be tuned later."

**Why it is weak:** it names the right direction but explains nothing: not why the weights diverge (separation), not
what goes wrong for patients (overconfident wrong probabilities), not how to choose $C$ (cross-validation on a
probability metric), and it treats the threshold as an afterthought rather than the main clinical decision.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Identifies perfect separation and diverging weights as the reason for 100% training accuracy | 2 |
| Uses the lab's numbers: validation log-loss worsens as $C$ grows; overconfident wrong probabilities | 2 |
| Proposes a penalized model, with $C$ chosen by cross-validation on training data with a probability-aware metric | 2 |
| Threshold set from the asymmetric costs (missed cancer much worse), chosen on validation data for a target sensitivity | 3 |
| Mentions calibration, a final one-time test, or external validation | 1 |

## Scoring

Total 36 pts. Pass ≥ 70% (26 pts). Topic ids to revisit for each lost criterion: Exercise 1 (a)–(d) → B2.1 and
A4.2 (chain rule); (e) → B2.1 Step 7. Exercise 2 gradient or stability → B2.1, A3.4; standardization → B1.3, B1.4;
mini-batch → A4.4. Exercise 3 separation → B2.1, B2.2; thresholds and calibration → B4.1, B4.4; cross-validation → B4.3.
