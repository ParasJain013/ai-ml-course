# Answer key: B2.2 Regularization: L1, L2 and elastic net, read as MAP priors

Lesson: [`lessons/B/b2_2_regularization.md`](../../../lessons/B/b2_2_regularization.md)

## Exercise 1 (by hand)

**Solution (a).** Least squares solves $\frac1n\mathbf{Z}^\top\mathbf{Z}\,\mathbf{w} = \frac1n\mathbf{Z}^\top(\mathbf{y} - \bar{y})$ (the normal equations divided by $n$). With $\frac1n\mathbf{Z}^\top\mathbf{Z} = \mathbf{I}$: $\mathbf{w} = (3, -0.5)$.

**(b)** With uncorrelated features, $\mathbf{z}_j^\top\mathbf{z}_k = 0$ for $k \ne j$, so $a_j = \frac1n\mathbf{z}_j^\top(\mathbf{y} - \bar{y} - \sum_{k \ne j}\mathbf{z}_kw_k) = \frac1n\mathbf{z}_j^\top(\mathbf{y} - \bar{y})$ whatever the other weights are. Each
coordinate's update does not depend on the others, so one sweep reaches the minimum. $c_j = 1$.
$\lambda = 1$: $w_1 = S(3, 1) = 2$, $w_2 = S(-0.5, 1) = 0$ (since $\lvert -0.5\rvert \le 1$). $\lambda = 4$: $S(3, 4) = 0$ and $S(-0.5, 4) = 0$, so $\mathbf{w} = (0, 0)$.

**(c)** Sum form: $(\mathbf{Z}^\top\mathbf{Z} + \lambda\mathbf{I})\mathbf{w} = \mathbf{Z}^\top(\mathbf{y} - \bar{y})$. Here $\mathbf{Z}^\top\mathbf{Z} = n\mathbf{I} = 10\mathbf{I}$ and $\mathbf{Z}^\top(\mathbf{y} - \bar{y}) = 10 \times (3, -0.5) = (30, -5)$, so $\mathbf{w} = (30/20, -5/20) = (1.5, -0.25)$.
Each weight is multiplied by $n/(n + \lambda) = 0.5$; neither reaches zero.

**(d)** $w_j = S(a_j, \lambda\rho)/(c_j + \lambda(1 - \rho)) = S(a_j, 0.5)/1.5$: $w_1 = 2.5/1.5 = 1.6667$, $w_2 = S(-0.5, 0.5)/1.5 = 0$.

**(e)** $\lambda_{\max} = \max_j\lvert a_j\rvert$ at $\mathbf{w} = \mathbf{0}$ $= \max(3, 0.5) = 3$.

```text
== exercise 1 check ==
(a) least squares w = [3.0, -0.5]
(b) lasso lambda 1.0: w = [2.0, 0.0]
(b) lasso lambda 4.0: w = [0.0, 0.0]
(c) ridge (sum form) lambda 10: w = [1.5, -0.25]
(d) elastic net lambda 1, rho 0.5: w = [1.6667, 0.0]
(e) lambda_max = max |a_j| = 3.0
    sklearn Lasso(1) [2. 0.]  Ridge(10) [ 1.5  -0.25]  ElasticNet(1, 0.5) [1.6667 0.    ]
```

The last line builds a concrete $\mathbf{Z}$ with exactly these statistics and confirms all three with scikit-learn.

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) Least squares from the scaled normal equations | 1 |
| (b) Both lasso solutions via soft-thresholding; the reason one sweep is exact (updates do not interact) | 4 |
| (c) Ridge in the sum form, with $\mathbf{Z}^\top\mathbf{Z} = n\mathbf{I}$ (not $\mathbf{I}$) | 3 |
| (d) Elastic-net update with threshold $\lambda\rho$ and denominator $1 + \lambda(1 - \rho)$ | 2 |
| (e) $\lambda_{\max} = 3$ | 2 |

**Likely mistakes:**

- In (c), using $\frac1n\mathbf{Z}^\top\mathbf{Z} = \mathbf{I}$ in the sum form and getting $w_1 = 3/11$. The ridge sum form uses $\mathbf{Z}^\top\mathbf{Z}$ itself, which is $n\mathbf{I}$. This is exactly the
  scaling trap between `Ridge` and `Lasso` that the lesson warns about.
- Soft-thresholding the negative coordinate the wrong way, getting $S(-0.5, 1) = -1.5$ or $+0.5$. The magnitude shrinks towards zero
  and stops there.
- Forgetting the $(1 - \rho)$ term in the elastic-net denominator, or thresholding at $\lambda$ instead of $\lambda\rho$.
- Saying one sweep is exact "because coordinate descent always converges in one sweep". It does not; it is exact here
  only because the features are uncorrelated.

## Exercise 2 (code)

**Solution:** [`b2_2_regularization_ex2.py`](b2_2_regularization_ex2.py). For each of 200 bootstrap samples of the 266
training rows, standardize with the sample's own statistics, fit the coordinate-descent lasso ($\alpha = 1$) and elastic net
($\alpha = 1$, $\rho = 0.5$), and record the weights.

```text
== bootstrap stability on diabetes ==
bootstrap sample 0 matches sklearn Lasso: OK
feature   lasso: % non-zero     mean      sd    enet: % non-zero     mean      sd
age                     72%     0.84    2.82                 92%     1.82    2.10
sex                    100%   -11.42    3.50                100%    -7.14    2.17
bmi                    100%    24.40    4.48                100%    17.79    2.50
bp                     100%    15.21    3.71                100%    11.59    2.07
s1                      86%    -6.16    4.81                 85%    -0.93    1.29
s2                      11%     0.16    2.00                 93%    -2.02    1.55
s3                      94%    -7.83    4.45                100%    -7.76    1.41
s4                      50%     3.05    5.34                100%     5.29    2.03
s5                     100%    24.53    4.49                100%    15.75    2.14
s6                      76%     2.49    3.25                100%     5.28    2.23
lasso s1/s2: both non-zero 0.04   only s1 0.83   only s2 0.07   neither 0.06
lasso: number of non-zero weights per fit ranges from 6 to 10
```

**Reading it.**

- Reliably selected by the lasso (100% of fits, with standard deviations well below the mean): `sex`, `bmi`, `bp`, `s5`. These
  are the features worth naming.
- `s4` is a coin flip (50%), and `age` and `s6` are in about three fits out of four. A single lasso fit that "drops `s4`"
  says almost nothing.
- `s1`/`s2`: the lasso usually keeps `s1` and drops `s2` (83% of fits), sometimes the reverse (7%), rarely both. The pair
  carries one piece of information (cholesterol), and the lasso hands it to one of them. Which one is a property of the
  sample, not of biology.
- The elastic net keeps almost every feature in almost every fit, with smaller standard deviations (about 1.3–2.5 against
  2.8–5.3): its weights are more stable, at the price of less sparsity. It also gives `s2` a small negative weight in
  93% of fits rather than letting `s1` take all the credit.
- The lasso's model size ranges from 6 to 10 features across resamples: "the lasso picked 8 features" is itself unstable.

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| Bootstrap done right: rows sampled with replacement, same size as the training set, seeded generator | 2 |
| Standardization refitted on each bootstrap sample (not the full training set's statistics reused) | 2 |
| Lasso and elastic net from scratch, one fit checked against scikit-learn | 2 |
| Table of selection frequency, mean and standard deviation per feature | 3 |
| Interpretation: which features are reliable, the `s1`/`s2` swap, the elastic net's stability | 3 |
| Readability: functions with type hints and shapes, no copy-paste between the two models | 2 |

**Likely mistakes:**

- Standardizing once with the full training statistics, then bootstrapping. It changes the numbers only a little here,
  but it is the habit that leaks in cross-validation (B1.4).
- Counting a weight as "selected" with `abs(w) > 1e-3` or similar. Coordinate descent gives exact zeros; use `!= 0`.
- Reporting only the mean weight. A mean of 0.16 for `s2` hides that it is 0 in 89% of fits and occasionally large.
- Using the default `Lasso(tol=1e-4)` for the check and getting differences around $10^{-3}$, then loosening the assertion instead of the tolerance.

## Exercise 3 (think)

**Strong answer:** The lasso can deliver a sparse *predictive* model, not a list of causes. With 400 features, 3,000 rows,
near-duplicate counters and one-hot groups, its selection among correlated features is close to arbitrary and changes
between resamples (Exercise 2 showed this on 10 features). So I would deliver: (1) a model chosen for prediction by
cross-validation, comparing ridge, lasso and elastic net inside a pipeline with the scaler (and the encoder), on a
probability metric such as log-loss; elastic net is the likely winner given the correlated groups. (2) A stability
analysis: selection frequency of each feature over bootstrap or cross-validation refits, reported at the level of
*groups* (all usage counters together, all dummies of one category together) rather than single columns. (3) A clear
statement of what this means: "customers with low recent usage are more likely to churn, holding the other features
fixed", which is an association in historical data. It does not mean that raising usage will reduce churn: usage may be a symptom
of dissatisfaction rather than its cause. To learn what to act on, they need an experiment (A6.3b) or at least a causal
analysis. I would also check the features for leakage (B1.4), since a "top driver" of churn is often a field written
after the customer decided to leave (a cancellation-page visit). For production, the choice is by validation score,
stability and cost: if a 30-feature elastic net is within noise of the 400-feature ridge, the smaller model is cheaper
to compute, monitor and explain.

**Weak answer:** "Fit `LassoCV`, take the 10 features with the largest absolute coefficients, and give them to the team as the drivers."

**Why it is weak:** it treats one fit's selection as stable and its coefficients as causal; it ignores correlated
groups (the lasso picks one counter of five), unscaled or one-hot features, leakage, and the difference between
prediction and intervention. The team would act on noise.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| States that the lasso's selection among correlated features is unstable; proposes a stability check (bootstrap or CV refits), ideally at group level | 3 |
| Distinguishes association from cause; says what would be needed to justify acting (experiment) | 3 |
| Chooses the production model by cross-validated score in a pipeline, with elastic net as a reasoned candidate | 2 |
| Mentions leakage among "top drivers", or the cost and simplicity argument for a sparse model | 2 |

## Scoring

Total 36 pts. Pass ≥ 70% (26 pts). Topic ids to revisit for each lost criterion: Exercise 1 → B2.2 (coordinate descent,
objectives), A5.5 (MAP); Exercise 2 → B2.2, A6.2 (bootstrap), B1.4 (refitting inside resamples); Exercise 3 → B2.2,
B1.4, B4.3, A6.3b.
