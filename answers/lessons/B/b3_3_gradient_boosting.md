# Answer key: B3.3 Gradient boosting: fitting residuals from scratch, then XGBoost / LightGBM in practice

Lesson: [`lessons/B/b3_3_gradient_boosting.md`](../../../lessons/B/b3_3_gradient_boosting.md)

## Exercise 1 (by hand)

**Solution (a).** $F_0 = (2 + 2 + 6 + 10)/4 = 5$. Residuals $\mathbf{r} = (-3, -3, 1, 5)$. Stump cuts on the residuals:
- $x \le 1.5$: left $\{-3\}$ (SSE 0), right $\{-3, 1, 5\}$ (mean 1, SSE $16 + 0 + 16 = 32$): total 32.
- $x \le 2.5$: left $\{-3, -3\}$ (SSE 0), right $\{1, 5\}$ (mean 3, SSE $4 + 4 = 8$): total **8**.
- $x \le 3.5$: left $\{-3, -3, 1\}$ (mean $-5/3$, SSE $\frac{16}{9} + \frac{16}{9} + \frac{64}{9} = 10.67$), right $\{5\}$ (SSE 0): total 10.67.

Best: $x \le 2.5$, leaves $-3$ and $+3$. $\mathbf{F}_1 = 5 + 0.1 \times (-3, -3, 3, 3) = (4.7, 4.7, 5.3, 5.3)$.

**(b)** $g_i = F_0 - y_i = (3, 3, -1, -5)$, $h_i = 1$. Left: $G_L = 6$, $H_L = 2$; right: $G_R = -6$, $H_R = 2$. With $\lambda = 2$: $w_L = -6/(2 + 2) = -1.5$, $w_R = +1.5$ (the residual means $\mp 3$, halved
by the penalty). Gain $= \frac12\big[\frac{36}{4} + \frac{36}{4} - \frac{0^2}{6}\big] - 1 = \frac12(18) - 1 = 8 > 0$: the split is made.

**(c)** $F_0 = \log\frac{0.25}{0.75} = \log\frac13 = -1.0986$, so every $p = 0.25$. Pseudo-residuals $y - p$: positive $0.75$, negative $-0.25$. Newton leaf:
$\frac{0.75 + (-0.25)}{0.25 \times 0.75 + 0.25 \times 0.75} = \frac{0.5}{0.375} = 1.3333$ (on the logit scale; the leaf then adds $\eta \times 1.3333$ to both rows' logits).

**(d)** About twice as many trees. Each tree's contribution is scaled by $\eta$, so to cover the same total distance in function space the
number of steps scales roughly with $1/\eta$ (and in practice a little more, since the smaller steps also generalize better
and early stopping lets the model run longer).

```text
== exercise 1 check ==
(a) F0 5.0   residuals [-3.0, -3.0, 1.0, 5.0]   SSE by cut 1.5: 32.0000  2.5: 8.0000  3.5: 10.6667
    best cut 2.5: leaves -3.0 and 3.0   F1 = [4.7, 4.7, 5.3, 5.3]
(b) lambda 2: w_L = -1.5  w_R = 1.5   gain (gamma 1) = 8.0000
    XGBoost leaf weights [-1.5, -1.5, 1.5, 1.5]
(c) F0 = log(0.25/0.75) = -1.0986   pseudo-residuals: positive 0.75, negative -0.25   Newton leaf (0.75 + -0.25) / (2 x 0.1875) = 1.3333
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) $F_0 = 5$, residuals, SSE of all three cuts, best cut and $\mathbf{F}_1$ with the 0.1 factor | 4 |
| (b) $g$ with the right sign, $w = -G/(H + \lambda) = \mp 1.5$, gain 8, "split is made" | 4 |
| (c) $F_0 = -1.0986$, pseudo-residuals 0.75 and $-0.25$, Newton leaf 1.3333 | 4 |
| (d) About double, with the reason | 2 |

**Likely mistakes:**

- Fitting the second tree to $\mathbf{y}$ instead of the residuals, or forgetting to multiply the leaf values by $\eta$.
- Sign errors with $g$: XGBoost's $g$ is $F - y$ (the derivative of the loss), so $w^* = -G/(H + \lambda)$ comes out with the sign of the residual mean.
  Writing $g = y - F$ and keeping the minus sign flips every leaf.
- Leaving out $\frac12$ or $\gamma$ in the gain.
- In (c), starting from $F_0 = 0.25$ (a probability) instead of the log-odds, or using the gradient-step leaf value (the mean
  pseudo-residual, 0.25) instead of the Newton step.

## Exercise 2 (code)

**Solution:** [`b3_3_gradient_boosting_ex2.py`](b3_3_gradient_boosting_ex2.py). The loop:

```python
F = np.full(len(y), np.log(y.mean() / (1 - y.mean())))        # log-odds start
for _ in range(n_trees):
    residual = y - expit(F)                                    # pseudo-residuals (scikit-learn's formula in the file)
    p = y - residual
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=tie_breaker).fit(X, residual)
    leaf_of = tree.apply(X)
    values = {leaf: residual[leaf_of == leaf].mean() / (p * (1 - p))[leaf_of == leaf].mean()
              for leaf in np.unique(leaf_of)}                  # Newton step per leaf
    F += lr * np.array([values[leaf] for leaf in leaf_of])
```

```text
== data ==
train (343, 30)  val (113, 30)  positive rate 0.3732

== from scratch versus GradientBoostingClassifier (B2.1's five features) ==
F0 = log-odds of 0.3732 = -0.5186
predict_proba matches GradientBoostingClassifier at all 100 stages: OK

== validation log-loss: five features and all 30 ==
5 features : val log-loss after 10 / 50 / 100 trees 0.2916 / 0.1542 / 0.1492   best 0.1472 at  68   val accuracy at 100 0.9381
30 features: val log-loss after 10 / 50 / 100 trees 0.2399 / 0.0671 / 0.0701   best 0.0626 at  67   val accuracy at 100 0.9735
B2.1 logistic regression on the 5 features: val log-loss 0.0856, accuracy 0.9823
saved outputs/b3_3/ex2_classifier_logloss.png
```

**Which wins, and why.** On the same five features, logistic regression (0.086) beats boosting (0.147) clearly. The
malignancy signal in these features is close to linear in the log-odds, the training set is small (343 rows), and a
sum of depth-3 trees has to build a smooth linear trend out of steps, which costs variance. With all 30 features,
boosting reaches 0.063, better than B2.1's five-feature model; but that comparison changes two things at once, and B2.1
showed that an unpenalized logistic regression on 30 features hits perfect separation, so the fair comparison is a penalized
logistic regression on 30 features (B2.2), tuned by cross-validation (B4.3). The lesson: boosting does not win
automatically. It wins when the signal is non-linear and there are enough rows to learn the shape.

**A note on matching.** Getting identical trees needs two details beyond the algorithm: the shared tie-breaker, and computing
$y - p$ with the same floating-point formula as scikit-learn. On all 30 features, exact matching fails even then. Radius, perimeter and area
are near-copies, so some nodes contain rows whose residuals are equal to 15 digits, and the "best" split there is decided by
rounding noise. A strong answer notices this and explains it instead of loosening the tolerance until the test passes.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| $F_0$ = log-odds of the positive rate; pseudo-residuals $y - p$ | 3 |
| Newton leaf values $\sum r / \sum p(1 - p)$, applied per leaf with `apply` (or by editing leaf values) | 3 |
| Exact match with `GradientBoostingClassifier` at every stage (an assert) | 3 |
| Log-loss curves for 5 and 30 features with a labelled plot | 2 |
| Honest comparison with B2.1: logistic regression wins on 5 features, and why | 3 |
| Notes that 5 → 30 features is a confounded comparison (or fits a penalized LR on 30) | 2 |

**Likely mistakes:**

- Using the gradient-step leaf value (the mean residual) instead of the Newton step. The model still learns, but it cannot match
  scikit-learn and converges more slowly.
- Applying the sigmoid to the tree outputs instead of to the running logit $F$.
- Starting at $F_0 = 0$ or at the positive rate 0.37 instead of its log-odds.
- Reporting validation log-loss at the best round as if it were a fair estimate: the round was chosen on the same rows.
- Concluding "boosting is better" from the 30-feature number alone.

## Exercise 3 (think)

**Strong answer:**
(a) *Delivery time, 40 tabular features, 2 million rows:* gradient-boosted trees (LightGBM or XGBoost). It is mixed tabular data with
non-linear interactions (rush hour × weather × distance) and plenty of rows; boosting will be accurate, fast to train and
easy to iterate on. A linear model is the baseline to beat.
(b) *Support tickets from text:* a neural model: a fine-tuned transformer, or embeddings plus a linear classifier (Parts E–F). Trees on raw
text features are weak. A TF-IDF plus logistic regression baseline (E1.2) comes first.
(c) *Credit limits with applicant-facing reasons, 30,000 rows:* a regularized logistic regression or a scorecard, possibly a monotone-constrained boosted model with
SHAP reasons (B3.4) if it is clearly more accurate. Regulation and explanations dominate here, and 30,000 rows is modest.
(d) *Next year's demand with a growth trend:* not trees on raw time, because trees cannot extrapolate beyond the largest target they saw.
Use a model that captures the trend (linear trend, ARIMA or exponential smoothing, B6.4), or boosting on *differenced* or detrended targets with lag features.
Silent failures after deployment: drift in the inputs, so new values land in leaves fitted to different populations
(J2.2 monitoring); a feature computed differently online than in training (training–serving skew); predictions stuck
at the edge of the training range when the target trends; a leaked feature that is unavailable or different at serving time.
**Weak answer:** "Use XGBoost for (a), (c) and (d) because it wins on tabular data, and a neural network for text. XGBoost
can fail if you overfit, so use early stopping."
**Why weak:** it ignores extrapolation in (d) and the explanation requirement in (c), and its "failure" is a training
problem rather than a deployment failure.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| (a) Boosting with a reason | 1 |
| (b) Neural or text model with a baseline | 1 |
| (c) Weighs explainability and data size; a defensible choice | 2 |
| (d) Recognizes that trees cannot extrapolate; proposes a trend-aware approach | 3 |
| Two genuine deployment-time failure modes | 3 |

## Scoring

Total 40 pts. Pass ≥ 70% (28). Topic ids to revisit for each lost criterion: residual fitting and shrinkage (Ex 1a, 1d) →
B3.3; XGBoost leaf weights and gain (Ex 1b) → B3.3 and the paper guide; log-loss gradients (Ex 1c, Ex 2) → B2.1, B3.3;
comparing with baselines (Ex 2) → B2.2, B4.3; model choice and deployment failures (Ex 3) → B3.3, B6.4, J2.2.
