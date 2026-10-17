# Answer key: B2.5 Support vector machines: maximum margin, hinge loss and the kernel trick

Lesson: [`lessons/B/b2_5_svm.md`](../../../lessons/B/b2_5_svm.md)

## Exercise 1 (by hand)

**Solution (a).** Cross-class distances: $(1, 1)$–$(3, 3)$: $\sqrt{4 + 4} = 2.828$; $(1, 1)$–$(4, 2)$: $\sqrt{9 + 1} = 3.162$; $(2, 0)$–$(3, 3)$: $\sqrt{1 + 9} = 3.162$; $(2, 0)$–$(4, 2)$: $\sqrt{4 + 4} = 2.828$. Two pairs
tie at $\sqrt8$, and both differences are the same vector $(2, 2)$: the two pairs are parallel. So the street runs perpendicular to $(1, 1)$:
$\mathbf{w} = k(1, 1)$. From $f(3, 3) = 6k + b = 1$ and $f(1, 1) = 2k + b = -1$: $4k = 2$, $k = 0.5$, $\mathbf{w} = (0.5, 0.5)$, $b = -2$.
Check: $y_if(\mathbf{x}_i)$ for $(1, 1)$: $-(1 - 2) = 1$; $(2, 0)$: $-(1 - 2) = 1$; $(3, 3)$: $3 - 2 = 1$; $(4, 2)$: $3 - 2 = 1$. All four lie exactly on the margin. Geometrically, all four are
support vectors (each touches the street). In the dual the solution for $\boldsymbol{\alpha}$ is not unique when points tie like this, and libsvm
returns one with only two non-zero $\alpha_i$ (the lab check reports 2 support vectors). The boundary is the same either way.

**(b)** $2/\lVert\mathbf{w}\rVert = 2/\sqrt{0.5} = 2.828$, the distance between the parallel pairs.

**(c)** $\mathbf{w} = (0.5, 0.5)$, $b = -2.5$: $f = -1.5, -1.5, 0.5, 0.5$, so $y_if = 1.5, 1.5, 0.5, 0.5$. Hinge losses: $0, 0, 0.5, 0.5$. Objective $= \frac12(0.25 + 0.25) + 1 \times (0 + 0 + 0.5 + 0.5) = 0.25 + 1 = 1.25$,
against $0.25 + 0 = 0.25$ for the hard-margin solution: shifting the boundary towards the positives costs two margin violations.

**(d)** $(1 + \mathbf{x}^\top\mathbf{z})^2 = 1 + 2x_1z_1 + 2x_2z_2 + x_1^2z_1^2 + 2x_1x_2z_1z_2 + x_2^2z_2^2$. Match term by term with a dot product:
$\boldsymbol{\phi}(\mathbf{x}) = (1,\ \sqrt2\,x_1,\ \sqrt2\,x_2,\ x_1^2,\ \sqrt2\,x_1x_2,\ x_2^2)$. Check with $\mathbf{x} = (1, 2)$, $\mathbf{z} = (3, 1)$: $(1 + 5)^2 = 36$.

```text
== exercise 1 check ==
(a) cross distances (rows (1,1),(2,0); cols (3,3),(4,2)): [[2.8284, 3.1623], [3.1623, 2.8284]]
    hard margin w [0.5, 0.5] b -2.0: y f(x) = [1.0, 1.0, 1.0, 1.0]
    SVC(C=1e6): w [0.5, 0.5]  b -2.0000  support vectors 2
(b) width 2/|w| = 2.8284
(c) hinge losses [0.0, 0.0, 0.5, 0.5]   objective 0.5|w|^2 + C sum = 0.2500 + 1.0000 = 1.2500
(d) (1 + x.z)^2 = 36.0   phi(x).phi(z) = 36.0000
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) Cross distances, the tie noticed and resolved (same direction), $\mathbf{w}$ and $b$ from two margin equations, all four checks | 4 |
| (b) Width $2/\lVert\mathbf{w}\rVert$ | 1 |
| (c) Hinge losses with labels $\pm1$, objective with the $\frac12$ | 3 |
| (d) Expansion and all six components, including the $\sqrt2$ factors and the constant 1 | 4 |

**Likely mistakes:**

- Using 0/1 labels in the hinge: $\max(0, 1 - 0 \cdot f) = 1$ for every negative point.
- Writing the width as $1/\lVert\mathbf{w}\rVert$ (that is the distance from the boundary to one edge).
- Missing the $\sqrt2$ factors in $\boldsymbol{\phi}$, or forgetting the constant feature 1 that the "+1" in the kernel creates.
- Thinking the tie means "no unique answer". The boundary is unique (the problem is strictly convex in $\mathbf{w}$); only the
  dual weights are not.

## Exercise 2 (code)

**Solution:** [`b2_5_svm_ex2.py`](b2_5_svm_ex2.py). Sub-gradient descent with step $1/(\lambda t)$ and averaging over the second half
of the iterates (which removes most of the zig-zag around the kinks), and gradient descent on the L2 log-loss with the
B2.1 step bound plus $\lambda$. Both use $\lambda = 1/(Cn)$ and are checked against scikit-learn: the objective gap to `SVC` and the
probability gap to `LogisticRegression`.

```text
== hinge versus log-loss across C ==
     C | SVM: val acc  yf<1     |w|  obj gap | LR: val acc p(1-p)>.01     |w|   max dp
  0.01 |       0.9912    78   0.613  2.0e-07 |      0.9823        331   0.752  3.1e-08
   0.1 |       1.0000    37   1.356  5.4e-07 |      1.0000        208   1.668  1.2e-07
   1.0 |       0.9912    17   3.160  1.6e-06 |      1.0000         88   3.542  3.3e-07
  10.0 |       0.9558     6   7.765  1.5e-05 |      0.9912         43   8.447  3.2e-06
obj gap = our hinge objective minus SVC's (>= 0; small = converged); max dp = max |p_mine - p_sklearn| on val
```

The loss plot is saved to `outputs/b2_5/ex2_hinge_vs_logloss.png`.

**Interpretation.** The SVM listens only to the points with $yf < 1$: 78 of 343 at $C = 0.01$, down to 6 at $C = 10$. Every other
point has exactly zero gradient and could be deleted. Logistic regression's gradient is never exactly zero. Counting
points whose $p(1 - p)$ is non-negligible, it still listens to 43–331 points, many more than the SVM at every $C$. It keeps nudging the
boundary to make confident points even more confident. As $C$ grows, both models fit the separable training data more
tightly and $\lVert\mathbf{w}\rVert$ grows. The SVM at $C = 10$ rests on 6 points and its validation accuracy drops to 0.956: a boundary determined by a handful
of points is sensitive to them. The differences of one or two validation tumours between the models (0.991 against 0.982) are noise.

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| Sub-gradient SVM with labels $\pm1$, decreasing step, intercept unpenalized; converges (objective gap small against `SVC`) | 3 |
| L2 logistic regression by GD; matches `LogisticRegression(C)` probabilities | 2 |
| The $\lambda = 1/(Cn)$ mapping used for both | 2 |
| Table with validation accuracy, active-point counts and $\lVert\mathbf{w}\rVert$ for every $C$ | 3 |
| Loss plot with both losses (and ideally the 0-1 loss) against $yf$ | 2 |
| Interpretation: the SVM ignores points beyond the margin, logistic regression never quite does; the effect of $C$ | 2 |

**Likely mistakes:**

- A fixed learning rate for the SVM: the objective stalls well above the optimum and oscillates.
- Comparing weights rather than objectives at large $C$: the hinge objective is flat in some directions near the optimum, so
  weights can differ by $10^{-2}$ while the objectives agree to $10^{-5}$.
- Penalizing the bias in the SVM, which `SVC` does not do.
- Using $C$ directly as $\lambda$ (they are inverses, scaled by $n$).

## Exercise 3 (think)

**Strong answer:** Probably not, and certainly not on the strength of that notebook. *Training:* an RBF SVM's cost grows
roughly with $n^2$ in time and memory for the kernel matrix. Going from 20,000 to 2 million rows is 100 times the data and on the
order of 10,000 times the work: hours to days, not a nightly job. You could train on a subsample, but then you
are not using the data that might make another model better. *Prediction:* each prediction costs one kernel evaluation
per support vector. With noisy data the support vectors can be a large fraction of the training set (tens of thousands
of 40-dimensional distances per request). That is measurable against 5 ms but leaves little headroom, and the model
artefact is the support set itself, large and containing raw customer rows (a privacy consideration). *Is the gain real?* Two points on one
20,000-row sample with one split may be noise or a tuning artefact. Logistic regression may not have had non-linear features or
tuning. Compare with repeated cross-validation on the same folds, with confidence intervals (A6.2, B4.3), and both models properly
tuned. *Calibration:* SVM scores are not probabilities; if the business uses probabilities, add explicit calibration.
*Alternatives to benchmark:* gradient-boosted trees (B3.3), the usual winner on tabular data, which scale to millions of
rows, predict in microseconds and need no scaling; a linear model with engineered or spline features; and a kernel
approximation (`Nystroem` or `RBFSampler` plus a linear model) if the SVM's boundary really is what helps. Ship the
simplest model whose validated advantage survives the latency and retraining budget.

**Weak answer:** "The SVM is 2 points more accurate, so ship it. If it is slow, use a bigger machine."

**Why it is weak:** it takes a single-sample difference at face value, ignores the quadratic training cost and the
per-support-vector prediction cost, says nothing about probabilities, and does not consider the models that usually
win on this kind of data.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Training cost scaling ($n^2$ or worse) against a nightly retrain on 2M rows | 2 |
| Prediction cost depends on the number of support vectors; latency checked against 5 ms | 2 |
| Questions whether 2 points is real: repeated CV, tuning both models fairly, intervals | 2 |
| Calibration of scores | 1 |
| Benchmarks gradient boosting and a kernel approximation or feature-engineered linear model | 3 |

## Scoring

Total 36 pts. Pass ≥ 70% (26 pts). Topic ids to revisit for each lost criterion: Exercise 1 → B2.5 (Steps 2, 3, 6),
A4.5; Exercise 2 → B2.5 (Step 4), B2.1, B2.2; Exercise 3 → B2.5 (cost), B3.3, B4.3, B4.1.
