# Answer key: B5.3 Gaussian mixture models and the EM algorithm

Lesson: [`lessons/B/b5_3_gmm_em.md`](../../../lessons/B/b5_3_gmm_em.md)

## Exercise 1 (by hand)

**Solution (a).** Equal weights and variances cancel, so $\gamma_{i1} = 1/\big(1 + e^{[(x_i - 0)^2 - (x_i - 6)^2]/(2 \cdot 4)}\big)$.

- $x = 0$: exponent $(0 - 36)/8 = -4.5$; $e^{-4.5} = 0.01111$; $\gamma_{11} = 1/1.01111 = 0.98901$.
- $x = 2$: exponent $(4 - 16)/8 = -1.5$; $e^{-1.5} = 0.22313$; $\gamma_{21} = 1/1.22313 = 0.81757$.
- $x = 6$: exponent $(36 - 0)/8 = 4.5$; $\gamma_{31} = 1/(1 + 90.017) = 0.01099$.

**(b)** $N_1 = 0.98901 + 0.81757 + 0.01099 = 1.81757$; $\pi_1 = 1.81757/3 = 0.60586$.
$\mu_1 = (0.98901 \cdot 0 + 0.81757 \cdot 2 + 0.01099 \cdot 6)/1.81757 = (1.63514 + 0.06594)/1.81757 = 0.93590$.
Squared deviations from 0.93590: $0.87591$, $1.13231$, $25.64511$; weighted: $0.86628 + 0.92574 + 0.28184 = 2.07386$; $\sigma_1^2 = 2.07386/1.81757 = 1.14100$ (1.14097 unrounded).
The variance dropped from 4 to 1.14: component 1 now covers the two left points, which are close together.

**(c)** $\gamma_{21} = \frac{0.2\,e^{-4/8}}{0.2\,e^{-4/8} + 0.8\,e^{-16/8}} = \frac{1}{1 + 4e^{-1.5}} = \frac{1}{1 + 0.89252} = 0.52840$. The prior moved against component 1 by a factor of 4 in odds, so the point is
now almost a coin flip even though it is twice as close to $\mu_1$: the E-step is Bayes' rule, and the weights are the prior.

**(d)** Full: $(K - 1) + Kd + K\,d(d + 1)/2 = 3 + 12 + 24 = 39$. Diagonal: $3 + 12 + Kd = 3 + 12 + 12 = 27$.

```text
== exercise 1 check ==
(a) gamma_i1 = [0.98901, 0.81757, 0.01099]
(b) N_1 = 1.81757   pi_1 = 0.60586   mu_1 = 0.93590   sigma_1^2 = 1.14097
(c) with pi = (0.2, 0.8): gamma_21 = 0.52840
(d) K=4, d=3: full 39 parameters, diagonal 27
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) Three responsibilities, with the cancellation shown | 3 |
| (b) $N_1$, $\pi_1$, $\mu_1$, $\sigma_1^2$ (variance about the *new* mean) | 4 |
| (c) 0.528 and the prior-odds explanation | 3 |
| (d) 39 and 27 | 2 |

**Likely mistakes:**

- Forgetting the denominator in the E-step and reporting the raw densities as responsibilities (they do not sum to 1).
- Computing $\sigma_1^2$ around the old mean $\mu_1 = 0$ instead of the updated 0.936. The M-step updates the mean first and uses it in the covariance.
- Dividing by $n = 3$ instead of $N_1$ in the mean or variance.
- Counting $K$ free weights instead of $K - 1$, or $d^2$ entries per full covariance instead of $d(d + 1)/2$ (it is symmetric).

## Exercise 2 (code)

**Solution:** [`b5_3_gmm_em_ex2.py`](b5_3_gmm_em_ex2.py). The diagonal log-density is a sum over features of 1-D log-densities, computed on an `(n, K, d)` array; the M-step keeps the weighted mean and
replaces the outer product by per-feature weighted squared deviations.

```text
== data (as in the lab) ==
X (1200, 2)

== diagonal EM from scratch versus GaussianMixture(covariance_type='diag') ==
final mean log-likelihood -3.9551 (never decreased); weights, means, variances match sklearn: OK

== BIC: full versus diagonal, K = 1..6 ==
 K  full params  full BIC  diag params  diag BIC
 1            5   10904.1            4   10917.1
 2           11    8800.1            9   10416.7
 3           17    8400.3           14    9593.0
 4           23    8443.3           19    9198.5
 5           29    8479.2           24    8996.1
 6           35    8513.3           29    8867.0
lowest BIC: covariance_type='full', K = 3 (BIC 8400.3)
best K among diagonal models: 6
```

**What a strong answer says:**

- The full-covariance model with $K = 3$ wins (BIC 8400.3), the true structure. A diagonal Gaussian's ellipses are aligned with the axes, so it cannot represent a tilted cluster with one component.
  It approximates each tilted cluster by a staircase of several axis-aligned blobs, so its BIC keeps falling up to the largest $K$ tried (6) and is still worse there (8867.0) than the full model at 3.
- More diagonal components is not "more segments in the data": it is the model's shape restriction showing up as extra clusters. With correlated features, either use full covariances,
  or decorrelate first (PCA, B5.4) and then a diagonal model is reasonable.
- Diagonal models earn their place when $d$ is large: $Kd$ covariance parameters instead of $K\,d(d + 1)/2$.
- The from-scratch run reached a mean log-likelihood of −3.955, much worse than the full model's −3.450, which is expected for a model that cannot tilt.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Diagonal E-step in logs (log-sum-exp), shapes correct, no Cholesky or inverse needed | 4 |
| Diagonal M-step correct; `reg_covar` added; log-likelihood asserted non-decreasing | 3 |
| Matches `GaussianMixture(covariance_type="diag")` from the same initialization, with a justified tolerance | 3 |
| BIC for both types and $K = 1..6$ with correct parameter counts (checked against `bic()`) | 3 |
| Interpretation: why full $K = 3$ wins and why diagonal wants more components | 3 |

**Likely mistakes:**

- Passing variances instead of precisions to `precisions_init` (scikit-learn expects $1/\sigma^2$ for `"diag"`); the two runs then start from different places and do not match.
- Leaving sklearn's default `tol` on, so it stops early while the from-scratch loop runs 100 iterations; the parameters then differ in the third or fourth digit and the learner loosens the tolerance instead of
  finding the cause.
- Using the full-covariance parameter count for the diagonal BIC.
- Reading "diagonal BIC is lowest at $K = 6$" as "there are six segments".

## Exercise 3 (think)

**Strong answer:** Deliver both: a hard label (the most likely segment) for the CRM, *and* the responsibility vector and a confidence flag (for example "max responsibility < 0.6: mixed") stored alongside it. Explain the
18% plainly: these customers behave partly like two segments, so any single label for them is a coin flip, and campaigns targeted at "segment 3" should either exclude them or treat them as a separate
"mixed" group. Do not hide them by switching to k-means, which would give the same customers confident-looking labels without changing the underlying ambiguity. Before anyone acts: (1) stability: refit
with several seeds and on bootstrap samples, and compare assignments with the adjusted Rand index; segments that change between runs are not real; (2) choice of $K$ and covariance type by BIC *and*
by whether the segments are interpretable and different enough to warrant different actions; (3) profile each segment on the original (unstandardized) features and on outcomes not used in the fit
(revenue, churn) to check that they differ in ways the team cares about; (4) check that the features were standardized and that no single feature (or a few outliers) drives a segment; (5) set a review date,
because behaviour drifts and segments must be refitted, with a mapping from old to new segment ids.
**Weak answer:** "Take the argmax of the responsibilities and give each customer their segment. The 18% are noise."
**Why weak:** it throws away the information the GMM was chosen for, mislabels a large group with false confidence, calls ambiguity "noise", and runs no stability or usefulness check.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Hard label plus responsibilities or a confidence flag, with a reason | 3 |
| Plain-English explanation of mixed customers and what to do with them | 2 |
| Stability checks (seeds, bootstrap, adjusted Rand index) | 2 |
| Choice of $K$/covariance type and segment profiling on outcomes not used in the fit | 2 |
| Scaling, outliers, drift or refresh mentioned | 1 |

## Scoring

Total 38 pts. Pass ≥ 70% (27). Topic ids to revisit for each lost criterion: responsibilities and Bayes' rule (Ex 1a, 1c) → B5.3, A5.3; the M-step formulas (Ex 1b) → B5.3, A5.5; parameter counts and BIC
(Ex 1d, Ex 2) → B5.3; covariance shapes and decorrelation (Ex 2) → A3.3, B5.4; log-sum-exp (Ex 2) → A3.4; segment stability and scaling (Ex 3) → B5.1, B1.3.
