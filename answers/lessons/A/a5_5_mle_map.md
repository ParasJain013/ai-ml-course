# Answer key: A5.5 MLE and MAP

Lesson: [`lessons/A/a5_5_mle_map.md`](../../../lessons/A/a5_5_mle_map.md)

## Exercise 1 (by hand)

**Solution (a).** $\log p(k \mid \lambda) = -\lambda + k\log\lambda - \log k!$. For counts $k_1, \dots, k_4$:

$$\mathcal{L}_{\text{NLL}}(\lambda) = \sum_{i=1}^4 \big(\lambda - k_i\log\lambda + \log k_i!\big) = 4\lambda - \Big(\sum_i k_i\Big)\log\lambda + \text{const}.$$

Derivative: $4 - \frac{\sum k_i}{\lambda} = 0$, so $\hat{\lambda} = \frac{\sum k_i}{4} = \frac{2 + 0 + 3 + 1}{4} = 1.5$: the sample mean. (Second derivative
$\sum k_i/\lambda^2 > 0$, so it is a minimum.) The $\log k_i!$ terms do not involve $\lambda$ and drop out.

**Solution (b).** Minimize $\frac{1}{2\sigma^2}\sum_i(x_i - \mu)^2 + \frac{\mu^2}{2\tau^2}$. Derivative:
$-\frac{1}{\sigma^2}\sum_i(x_i - \mu) + \frac{\mu}{\tau^2} = 0 \Rightarrow \mu\big(\frac{n}{\sigma^2} + \frac{1}{\tau^2}\big) = \frac{\sum x_i}{\sigma^2}$, so

$$\hat{\mu}_{\text{MAP}} = \frac{\sum_i x_i/\sigma^2}{n/\sigma^2 + 1/\tau^2}.$$

With $\sum x_i = 12$, $\sigma^2 = 4$, $n = 3$: $\tau^2 = 1$ gives $\frac{3}{0.75 + 1} = 1.714$; $\tau^2 = 100$ gives $\frac{3}{0.75 + 0.01} = 3.947$. The
MLE is 4. A tight prior ($\tau = 1$: "μ is probably within ±2 of 0") outweighs three noisy points and
pulls the estimate more than halfway to 0; a vague prior ($\tau = 10$) barely moves it. Equivalently,
$\hat{\mu}_{\text{MAP}}$ is a weighted average of the sample mean and the prior mean 0, with weights $n/\sigma^2$ (the data's
precision) and $1/\tau^2$ (the prior's precision).

**Solution (c).** $-\log p(\varepsilon_i) = \log(2b) + \frac{\lvert\varepsilon_i\rvert}{b}$, with $\varepsilon_i = y_i - \mathbf{x}_i^\top\mathbf{w}$. So
$\mathcal{L}_{\text{NLL}}(\mathbf{w}) = n\log(2b) + \frac1b\sum_i\lvert y_i - \mathbf{x}_i^\top\mathbf{w}\rvert$. The first term is constant in $\mathbf{w}$ and $\frac1b > 0$, so
minimizing the NLL means minimizing $\sum_i\lvert y_i - \mathbf{x}_i^\top\mathbf{w}\rvert$, equivalently its mean: the MAE. (For a constant
model the MAE minimizer is the median, which is why it resists outliers.)

```text
== exercise 1 check ==
(a) Poisson rate MLE = mean count = 1.5000
(b) tau^2 =   1.0: mu_MAP = (sum x / sigma^2) / (n / sigma^2 + 1 / tau^2) = 1.7143   (MLE 4.0)
(b) tau^2 = 100.0: mu_MAP = (sum x / sigma^2) / (n / sigma^2 + 1 / tau^2) = 3.9474   (MLE 4.0)
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Correct NLL, derivative, $\hat{\lambda} = 1.5$ | 4 |
| (b) Objective with both terms, derivative, general formula | 3 |
| (b) Both numbers and the interpretation (prior versus data precision) | 3 |
| (c) NLL of Laplace noise, constant dropped, conclusion MAE | 4 |

**Likely mistakes:**

- In (a), keeping $\prod$ instead of taking logs, then differentiating a product of four terms: correct in
  principle, error-prone in practice.
- In (b), forgetting the $\frac{1}{2\sigma^2}$ weight on the data term (writing $\sum(x_i - \mu)^2 + \mu^2/\tau^2$), which gives
  $\frac{12}{3 + 1} = 3$ instead of 1.714. The prior's pull depends on the *ratio* of noise and prior variances.
- In (c), concluding "minimizes $\sum\lvert r_i\rvert/b$, so $b$ matters": a positive constant factor never moves the minimizer.

## Exercise 2 (code)

**Solution:** [`a5_5_mle_map_ex2.py`](a5_5_mle_map_ex2.py).

```text
== exercise 2: data ==
n = 200   positives = 79

== exercise 2 (a, b): first 200 examples ==
MLE by GD (906 steps): (b, w) = [-0.8285, 2.2789]   LogisticRegression(C=inf): [-0.8285, 2.2789]
MAP by GD (228 steps), tau^2 = 0.05: (b, w) = [-0.5411, 1.0351]   LogisticRegression(C=0.05): [-0.5411, 1.0351]
matches scikit-learn: OK   weight shrank from 2.2789 to 1.0351 (55%)

== exercise 2 (c): first 20 examples ==
MLE by GD (3432 steps): (b, w) = [0.2445, 4.5973]   LogisticRegression(C=inf): [0.2445, 4.5973]
MAP by GD (152 steps), tau^2 = 0.05: (b, w) = [-0.1567, 0.2545]   LogisticRegression(C=0.05): [-0.1567, 0.2545]
matches scikit-learn: OK   weight shrank from 4.5973 to 0.2545 (94%)
```

**Why $C = \tau^2$.** MAP minimizes $\sum_i \ell_i + \frac{w^2}{2\tau^2}$ (sum of per-example log-losses plus the negative log
prior). scikit-learn minimizes $\frac12 w^2 + C\sum_i\ell_i$. Multiplying the MAP objective by $\tau^2 > 0$ gives
$\tau^2\sum_i\ell_i + \frac12 w^2$, the same function with $C = \tau^2$, so the minimizers agree. (In the gradient, the prior adds
$w/(n\tau^2)$ when the data term is the *mean* log-loss.)

**(c).** With 20 examples the MLE weight jumps to 4.6 (more than twice the true 2: small samples give
noisy, often inflated logistic weights, and with a few more points it could separate perfectly and
diverge), and the same prior now shrinks it by 94% instead of 55%. The prior's term is fixed while the
likelihood's grows with $n$, so the prior dominates small datasets. Note also that $\tau^2 = 0.05$ is a poor
prior for this problem: it says $w$ is probably within $\pm 2\tau \approx \pm 0.45$ of 0, while the true weight is 2. A
learner who points out that this prior is misspecified, and that in practice $C$ is chosen by
cross-validation (B4.3), earns the "interpretation" points in full. Exact numbers depend on draw order.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Data generated as specified; design matrix `(n, 2)` with a ones column | 2 |
| GD on the mean log-loss with the $\frac1n\mathbf{X}^\top(\mathbf{p} - \mathbf{y})$ gradient and a gradient-norm stop | 3 |
| MLE matches `LogisticRegression(C=np.inf)` with an assertion | 2 |
| Prior gradient on the weight only, scaled correctly for the mean loss | 3 |
| MAP matches `LogisticRegression(C=0.05)`; $C = \tau^2$ derived | 3 |
| (c) Shrinkage grows as $n$ falls, explained by a fixed prior versus a likelihood that scales with $n$ | 3 |

**Likely mistakes:**

- Penalizing the intercept too: GD and scikit-learn then disagree (scikit-learn's lbfgs solver does not
  penalize `intercept_`).
- Adding $w/\tau^2$ to the gradient of the *mean* loss instead of $w/(n\tau^2)$: equivalent to $C = n\tau^2$, so it matches
  `C=10` rather than `C=0.05` at $n = 200$.
- Reading $C$ as a regularization strength ("larger $C$, more regularization"): it is the reverse.
- Stopping GD after a fixed 1,000 steps: on 20 points the MLE needs over 3,000, and the comparison fails
  in the second decimal.
- `LogisticRegression(penalty=None)` raises a deprecation warning in 1.9.1; `C=np.inf` is the current way.

## Exercise 3 (think)

**Strong answer:** MSE is the NLL of Gaussian noise with one constant variance, symmetric around the
prediction. Gaussian tails are so thin that a 2-hour delivery when the model predicts 30 minutes (with
residual spread of maybe 8 minutes) is an "impossible" 11σ event, so the fit moves to reduce those
enormous squared errors: the 1% of disasters pull the predictions for everyone upwards (and towards
whatever features correlate with disasters), making typical predictions too long and the model
sensitive to which outliers happen to be in the training data. The noise is also skewed (deliveries can
be very late but not very early) and grows with distance. Alternatives: (1) **MAE**, the NLL of Laplace
noise: predicts the conditional median, far less affected by the 1%; (2) **Huber loss**, quadratic for
small errors and linear for large, a Gaussian core with Laplace-like tails; (3) **model the log of the
time** with MSE, i.e. log-normal noise: multiplicative, right-skewed errors, which matches durations;
(4) **quantile (pinball) loss** to predict, say, the 90th percentile, if the product promises an "arrives
by" time; (5) separate the two processes: a classifier for "something goes wrong" plus a regression for
normal deliveries. How to choose: from the product. The ETA shown to customers is judged by how often it
is badly wrong and in which direction (late is worse than early), so evaluate each option on the
business metric (median absolute error, share of orders more than 10 minutes late versus the promise)
on a held-out recent period, look at residual plots by distance and time of day, and pick the loss whose
implied distribution matches what the residuals show.

**Weak answer:** "Remove the outliers above 2 hours and keep MSE." Weak because those deliveries are
real and will happen in production (dropping them hides the problem rather than modelling it), it does
not name the noise model MSE assumes, and it offers no way to choose among losses.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Names the Gaussian, constant-variance, symmetric noise model behind MSE | 2 |
| Explains how rare huge errors distort the fit (squared penalty, thin tails) | 2 |
| At least two alternatives, each tied to its probability model | 4 |
| A selection method tied to the product (business metric, held-out data, residual plots) | 2 |

## Scoring

Total 40 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → A5.5
"The math" (NLL derivations, MAP) and A4.1 (derivatives); exercise 2 → A5.5 "With SciPy and
scikit-learn" (`C` versus $\lambda$), A4.4 (gradient descent) and A4.3 (the $\mathbf{p} - \mathbf{y}$ gradient); exercise 3 →
A5.5 "In practice" and B4.2 (choosing a metric that matches the cost of errors).
