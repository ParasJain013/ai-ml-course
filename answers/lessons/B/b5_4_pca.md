# Answer key: B5.4 PCA from scratch via SVD, explained variance

Lesson: [`lessons/B/b5_4_pca.md`](../../../lessons/B/b5_4_pca.md)

## Exercise 1 (by hand)

**Solution (a).** Mean $\boldsymbol{\mu} = ((2 + 4 + 6 + 4)/4, (1 + 3 + 5 + 7)/4) = (4, 4)$. Centred rows:
$(-2, -3), (0, -1), (2, 1), (0, 3)$.
$\mathbf{X}_c^\top\mathbf{X}_c$: $\sum x_1^2 = 4 + 0 + 4 + 0 = 8$; $\sum x_1x_2 = 6 + 0 + 2 + 0 = 8$; $\sum x_2^2 = 9 + 1 + 1 + 9 = 20$,
so $\mathbf{X}_c^\top\mathbf{X}_c = \begin{pmatrix}8&8\\8&20\end{pmatrix}$ and $\mathbf{S} = \frac13\begin{pmatrix}8&8\\8&20\end{pmatrix} = \begin{pmatrix}2.6667&2.6667\\2.6667&6.6667\end{pmatrix}$.
Eigenvalues: easiest on $\mathbf{X}_c^\top\mathbf{X}_c$ (trace 28, determinant $160 - 64 = 96$):
$\lambda^2 - 28\lambda + 96 = (\lambda - 24)(\lambda - 4)$, so 24 and 4; divide by 3 for $\mathbf{S}$: $\lambda_1 = 8$, $\lambda_2 = 1.3333$.
Eigenvector for 24: $(8 - 24)a + 8b = 0 \Rightarrow b = 2a$, so $\mathbf{v}_1 = (1, 2)/\sqrt5 = (0.4472, 0.8944)$. Perpendicular:
$\mathbf{v}_2 = (2, -1)/\sqrt5 = (0.8944, -0.4472)$ (largest entry positive).

**(b)** Ratios $8/9.3333 = 0.8571$ and $1.3333/9.3333 = 0.1429$ (equivalently $24/28$ and $4/28$).

**(c)** Scores $\mathbf{X}_c\mathbf{v}_1 = (-2 - 6, 0 - 2, 2 + 2, 0 + 6)/\sqrt5 = (-8, -2, 4, 6)/\sqrt5 = (-3.5777, -0.8944, 1.7889, 2.6833)$.
Reconstruction $\boldsymbol{\mu} + z_i\mathbf{v}_1 = (4, 4) + \frac{z_i\sqrt5}{5}(1, 2)$, i.e. $(4, 4) + \frac{(-8, -2, 4, 6)_i}{5}(1, 2)$:
$(2.4, 0.8), (3.6, 3.2), (4.8, 5.6), (5.2, 6.4)$. Residuals $(-0.4, 0.2), (0.4, -0.2), (1.2, -0.6), (-1.2, 0.6)$, squared
$0.2 + 0.2 + 1.8 + 1.8 = 4.0$. Step 4: $\sigma_2^2 = (n - 1)\lambda_2 = 3 \times 1.3333 = 4$ ✓.

**(d)** Centred with the training mean: $(2, 2)$. Score $(2 + 4)/\sqrt5 = 2.6833$. Reconstruction $(4, 4) + \frac65(1, 2) = (5.2, 6.4)$.
Residual $(0.8, -0.4)$, squared error $0.64 + 0.16 = 0.8$.

```text
== exercise 1 check ==
(a) mean [4.0, 4.0]   Xc^T Xc [[8.0, 8.0], [8.0, 20.0]]   S [[2.6667, 2.6667], [2.6667, 6.6667]]
    eigenvalues of S [8.0, 1.3333]   v1 [0.4472, 0.8944]   v2 [0.8944, -0.4472]
(b) explained-variance ratio [0.8571, 0.1429]
(c) PC1 scores [-3.5777, -0.8944, 1.7889, 2.6833]   (= [-8, -2, 4, 6]/sqrt(5))
    rank-1 reconstruction [[2.4, 0.8], [3.6, 3.2], [4.8, 5.6], [5.2, 6.4]]
    squared error 4.0000   sigma_2^2 = (n-1) lambda_2 = 4.0000
(d) new point score 2.6833   reconstruction [5.2, 6.4]   squared error 0.8000
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) Correct centring, $\mathbf{X}_c^\top\mathbf{X}_c$, $\mathbf{S}$ with $n - 1$, eigenvalues 8 and 1.3333, unit eigenvectors | 4 |
| (b) Ratios 0.8571 and 0.1429 | 1 |
| (c) Scores, four reconstructed rows, total error 4 checked against $(n - 1)\lambda_2$ | 4 |
| (d) Training mean used; score 2.6833, reconstruction (5.2, 6.4), error 0.8 | 3 |

**Likely mistakes:**

- Forgetting to centre, so the "PC1" points at the mean $(4, 4)$ instead of along the cloud.
- Using unnormalized eigenvectors such as $(1, 2)$: the scores come out $\sqrt5$ times too large and the reconstruction is wrong.
- Reconstructing without adding the mean back.
- In (d), centring the new point with its own value (giving $(0, 0)$) or refitting the PCA with it included.
- Comparing the error 4 with $\lambda_2 = 1.33$ and concluding the formula is wrong: the identity is with $\sigma_2^2 = (n - 1)\lambda_2$.

## Exercise 2 (code)

**Solution:** [`b5_4_pca_ex2.py`](b5_4_pca_ex2.py).

```text
== data ==
train (1347, 64)   test (450, 64)

== grid search over n_components ==
  k  CV accuracy    std
  5       0.8552 0.0241
 10       0.9235 0.0150
 15       0.9361 0.0164
 20       0.9391 0.0061
 25       0.9466 0.0111
 30       0.9517 0.0082
 40       0.9614 0.0144
 50       0.9621 0.0142
 64       0.9621 0.0142
~ mean fit time per fold: k = 5 3.409 s, k = 64 0.164 s
best k = 50 (CV accuracy 0.9621);  smallest k within one std of the best: 30

== variance rules ==
90% rule keeps k = 21   (CV accuracy at the nearest grid value: 0.9391)
95% rule keeps k = 28   (CV accuracy at the nearest grid value: 0.9517)

== test set, scored once ==
chosen k = 30: test accuracy 0.9622
```

(The fit-time line varies from run to run.)

**What a strong answer says:**

- Accuracy rises quickly to about $k = 30$ and then flattens; 50 and 64 tie (0.9621), so the "best" $k$ is
  a tie broken by the grid order. The differences above $k = 30$ are within one fold standard deviation
  (about 0.01–0.015), so they are not reliable. The one-standard-error style rule picks the smallest $k$
  within one std of the best, $k = 30$ (less to store, compute and explain).
- The 90% rule ($k = 21$) costs about 2 points of accuracy; the 95% rule ($k = 28$) lands near the
  one-std choice. Neither rule knows the task: on another dataset 95% of the variance can be mostly noise,
  or can miss a low-variance direction that carries the label.
- On digits, PCA does not *improve* accuracy over all 64 pixels; its value here is compression (30 numbers
  instead of 64) at a small cost. A strong answer says that plainly.
- The test score (0.9622) was computed once, after the choice, and is consistent with the CV estimate.
- Fewer components did not mean faster fitting here: with $k = 5$ the logistic regression needs more
  iterations to converge on harder-to-separate data (942 lbfgs iterations at $k = 5$ against 98 at
  $k = 64$ when fitted once on the training rows). Timings are noisy and not the point.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| PCA inside the `Pipeline`, tuned with `GridSearchCV` on training rows only, stratified 5-fold with the seed | 4 |
| Mean and std of CV accuracy for every $k$ | 3 |
| Best $k$ and the smallest $k$ within one std identified, with the tie and noise discussed | 3 |
| 90% and 95% rules computed from the training fit and compared with the CV choice | 3 |
| Test set scored once, for the chosen pipeline; readable code with seeds | 3 |

**Likely mistakes:**

- Fitting PCA on all 1,797 rows (or on the training rows outside the pipeline) and then cross-validating only
  the classifier: the components saw the validation folds.
- Choosing $k$ from the test score, or printing the test score for every $k$.
- Computing the variance rules on train + test.
- Reading a 0.0007 difference between $k = 40$ and $k = 50$ as a real improvement.

## Exercise 3 (think)

**Strong answer:** Before agreeing: (1) **Scale and types.** Usage counts, spend and ticket counts have
different units and heavy tails; standardize (and probably log-transform counts) inside the pipeline,
otherwise the largest-unit feature becomes PC1 (as raw wine's proline did). One-hot plan columns are 0/1 and
mix poorly with continuous features in a variance-based method; consider keeping them out of the PCA and
passing them to the model directly. (2) **Does it help the model?** Compare the churn model on all 300
features against PCA-20 (and other $k$) under the same time-ordered validation (B4.3). If the model is
gradient boosting, PCA will likely hurt (trees split on single features) and is unnecessary; if it is a
linear or distance-based model it may help. (3) **Fit on the training window only**, inside the pipeline,
and refit the PCA and the churn model together, as one artefact with one version. Over the following
months: (4) **Monthly refits change the components.** Signs can flip and nearly-equal components can swap
order, so "PC1" next month may be minus this month's PC1 or a different mix; a dashboard tracking "PC1 =
engagement" would show jumps that are artefacts of the refit, and the churn model trained on last month's
components gets misaligned inputs if only the PCA is refitted. Fix a sign convention, align new components
to old ones (match by absolute dot product), or freeze the PCA between planned model releases. (5) **Labels
are stories.** "Engagement" and "price sensitivity" are readings of the top weights; each component mixes all
300 features. Put interpretable, hand-defined metrics on the dashboard (sessions per week, discount usage)
instead, and keep PCA scores as model inputs. (6) **Drift.** New features, new plan types, or a change in
how a metric is logged changes the covariance; monitor the explained-variance ratios and the reconstruction
error of new data (J2.2): a rising error means the new data no longer lies near the old subspace.
**Weak answer:** "PCA is a good idea because it reduces dimensionality and removes noise. Keep the
components that explain 95% of the variance and label them by their biggest weights."
**Why weak:** it ignores scaling and one-hot columns, never checks whether the churn model benefits, misses
refit instability (sign flips, reordering) and the leakage rule, and treats component labels as facts.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Scaling / skewed counts / one-hot columns addressed | 2 |
| Benefit measured on the downstream model with the right validation; tree models noted | 2 |
| Fit on training data inside a pipeline; PCA and model versioned together | 2 |
| Refit instability (sign flips, reordering) and its effect on the dashboard and the model | 2 |
| Interpretability of labels and drift monitoring | 2 |

## Scoring

Total 38 pts. Pass ≥ 70% (27). Topic ids to revisit for each lost criterion: covariance, centring and
$n - 1$ (Ex 1a) → A3.3; SVD and reconstruction error (Ex 1c) → A3.2, B5.4; pipelines and leakage
(Ex 2, Ex 3) → B1.4, B1.3; cross-validation and choosing by noise (Ex 2) → B4.3; monitoring (Ex 3) → J2.2.
