# B4.5 The bias–variance trade-off, learning curves and validation curves

**Core** · Part B, week 4 (course week 10) · ~1.5 h · Lab:
[`labs/B/b4_5_bias_variance.py`](../../labs/B/b4_5_bias_variance.py)

## Why it matters

When a model disappoints, the next step depends on *why*, and the wrong step wastes weeks. If the model is too simple, more data will not help: it needs
more capacity or better features. If it is too flexible, more capacity makes things worse: it needs more data, more regularization or an ensemble.
The **bias–variance** decomposition says precisely what those two failure modes are, and two cheap plots tell them apart: the **learning curve**
(error against training-set size) and the **validation curve** (error against a capacity setting). This lesson derives the decomposition, measures it by
simulation, and builds both curves from scratch, checked against scikit-learn. You will use the curves on every project, and in C4.3 to debug neural networks. "Your model has 99% training
accuracy and 80% validation accuracy; what do you do?" is a standard interview question.

## Prerequisite check

- Can you compute an expectation and a variance, and expand $\mathbb{E}[(X - c)^2]$? If not, revisit [A5.1](../A/a5_1_random_variables_expectation.md).
- Do you know why a fully grown tree overfits and averaging trees reduces variance? If not, revisit [B3.1](b3_1_decision_trees.md) and
  [B3.2](b3_2_random_forests.md).
- Can you run k-fold cross-validation and say what the fold-to-fold spread means? If not, revisit [B4.3](b4_3_cross_validation_tuning.md).
- Can you fit a polynomial by least squares (a linear model on the features $1, x, x^2, \dots$)? If not, revisit [B1.5](b1_5_linear_regression.md).

## Intuition

Imagine you could collect many different training sets from the same source and train the same kind of model on each. For a fixed test input, you would get many
different predictions.

- Their *average* might be off from the truth. That systematic miss is **bias**: the model family cannot represent the true pattern (a straight line through a curve),
  however much data it gets.
- They might *scatter* around their average. That scatter is **variance**: the model is so flexible that it chases the particular noise in each training set.
- Even a perfect model faces **noise**: the part of $y$ no feature explains. It is the irreducible floor.

Expected test error is exactly bias² plus variance plus noise. Simple models have high bias and low variance; flexible models the reverse. Somewhere in between, the sum
is smallest. That is the "trade-off": added capacity first removes bias faster than it adds variance, then the opposite.

You cannot draw many training sets in practice, but two plots reveal which term dominates:

- **Learning curve:** train on 5%, 10%, ..., 100% of the training data and plot training and validation error. A high-bias model's two curves meet early at a high error and stay
  flat: more data will not help. A high-variance model has low training error and a large gap that slowly shrinks: more data helps.
- **Validation curve:** sweep one capacity setting (tree depth, polynomial degree, penalty strength) and plot both errors. Training error keeps falling; validation error falls then rises.
  The bottom of the validation curve is the setting you want.

> **Bridge:** bias is a systematic error, like a clock that runs consistently five minutes slow; variance is jitter, like a clock that is right on average but
> wanders by minutes. You fix the first by changing the mechanism and the second by averaging or stabilizing. **Where it breaks:** with clocks you can measure each
> error directly against a reference. With a model you only have one training set and one validation score, so bias and variance must be inferred from how
> the errors change with data size and capacity.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $f(x)$ | the true function | scalar |
| $\varepsilon$ | noise, with $\mathbb{E}[\varepsilon] = 0$ and $\operatorname{Var}[\varepsilon] = \sigma^2$, independent of everything else | scalar |
| $y = f(x) + \varepsilon$ | an observed target | scalar |
| $\mathcal{D}$ | a random training set of $n$ examples | — |
| $\hat{f}_{\mathcal{D}}(x)$ | the prediction at $x$ of the model trained on $\mathcal{D}$ | scalar |
| $\bar{f}(x)$ | the average prediction over training sets, $\mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(x)]$ | scalar |

**Step 1: the decomposition at one input $x$.** Take the expected squared error over both the training set and the test noise, $\mathbb{E}\big[(y - \hat{f}_{\mathcal{D}}(x))^2\big]$. Write the error as a sum of three pieces by adding and
subtracting $f(x)$ and $\bar{f}(x)$:

$$y - \hat{f}_{\mathcal{D}}(x) = \underbrace{\big(y - f(x)\big)}_{\varepsilon} + \underbrace{\big(f(x) - \bar{f}(x)\big)}_{\text{a fixed number}} + \underbrace{\big(\bar{f}(x) - \hat{f}_{\mathcal{D}}(x)\big)}_{\text{mean 0 over } \mathcal{D}}.$$

Square and take expectations. Write the three pieces as $a = \varepsilon$, $b = f(x) - \bar{f}(x)$ and $c = \bar{f}(x) - \hat{f}_{\mathcal{D}}(x)$; then $(a + b + c)^2 = a^2 + b^2 + c^2 + 2ab + 2ac + 2bc$. The three cross products vanish in expectation:
$\mathbb{E}[ab] = b\,\mathbb{E}[\varepsilon] = 0$ ($b$ is a constant); $\mathbb{E}[ac] = \mathbb{E}[\varepsilon]\,\mathbb{E}[c] = 0$ ($\varepsilon$ is independent of the training set); $\mathbb{E}[bc] = b\,\mathbb{E}[c] = 0$ (by the definition of $\bar{f}$, $\mathbb{E}[c] = 0$).
And $\mathbb{E}[a^2] = \operatorname{Var}[\varepsilon] = \sigma^2$ because $\varepsilon$ has mean 0. What remains:

$$\mathbb{E}\big[(y - \hat{f}_{\mathcal{D}}(x))^2\big] = \underbrace{\sigma^2}_{\text{noise}} + \underbrace{\big(f(x) - \bar{f}(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_{\mathcal{D}}\big[(\hat{f}_{\mathcal{D}}(x) - \bar{f}(x))^2\big]}_{\text{variance}}.$$

Averaging over test inputs $x$ gives the same decomposition for the expected test MSE.

**Step 2: what moves each term.** More capacity (higher degree, deeper trees, weaker penalty) usually lowers bias and raises variance. More training data lowers variance (each fit is less sensitive to
any one row) but does not change the bias of the family. Averaging independent-ish models lowers variance (B3.2: the variance of an average of $M$ trees, each with variance $s^2$ and pairwise correlation $\rho$, is $\rho s^2 + (1 - \rho)s^2/M$); boosting lowers bias (B3.3). Regularization (B2.2) trades a little bias for less variance.
Nothing touches $\sigma^2$, except new features that explain more of $y$ (which redefines the noise).

**Step 3: the learning curve.** For training sizes $m_1 < m_2 < \dots$, train on the first $m$ rows of each fold's training part and record training and validation error. Training error at small $m$ is low (few points are easy to fit) and
rises towards the model's floor as $m$ grows. Validation error starts high and falls towards the same floor. The floor is roughly $\sigma^2 + \text{bias}^2$ in MSE, or its square root on an RMSE plot; the gap between the curves is mostly variance.

**Step 4: the validation curve.** For values $c_1, c_2, \dots$ of one capacity setting, record cross-validated training and validation error. Training error falls monotonically with capacity. Validation error
traces the U of Step 2. Choose the minimum, or the simplest setting within one fold-to-fold standard deviation of it, which is more robust.

A caveat for later: very large models trained to zero training error (deep networks, C-parts) can show "double descent", where test error falls again past the point of interpolation
(Belkin and colleagues, 2019). The decomposition still holds; the classic U is what you see for the classical models of Part B.

## Worked example by hand

One test input with true value $f(x_0) = 4$ and noise variance $\sigma^2 = 1$. The same model trained on four different training sets predicts $2.5, 3.5, 3.0, 3.0$ at $x_0$.

- Average prediction $\bar{f}(x_0) = 12/4 = 3.0$. **Bias²** $= (3.0 - 4.0)^2 = 1.0$.
- **Variance:** squared deviations from 3.0 are $0.25, 0.25, 0, 0$; mean $0.5/4 = 0.125$.
- **Expected squared error** $= 1.0 + 0.125 + 1.0 = 2.125$. The model is consistent (low variance) but consistently low (high bias): it needs a different model, not more data.

```text
== worked example by hand ==
predictions [2.5, 3.5, 3.0, 3.0]: mean 3.0  bias^2 (3.0 - 4.0)^2 = 1.0  variance 0.125
expected squared error = bias^2 + variance + noise = 1.0 + 0.125 + 1.0 = 2.125
```

## Key plot

![Bias squared, variance and measured test MSE against polynomial degree 0 to 10, on a log scale, with the noise floor at 0.09. Bias squared falls from 0.50 at degree 0 to near zero by degree 5; variance rises from 0.02 to 90 at degree 10; measured test MSE is lowest at degree 3 (0.11) and explodes beyond degree 8](img/b4_5_bias_variance.png)

300 simulated training sets of 30 noisy points from $\sin(2\pi x)$, fitted with polynomials of each degree. Bias² collapses once the degree is high enough to bend like a sine (degree 3);
variance creeps up and then explodes, because high-degree polynomials swing wildly between and beyond the training points. The measured test error (triangles) sits on bias² + variance + noise
at every degree: the decomposition is not an approximation.

## From scratch in NumPy

**The simulation.** For each degree, fit all 300 training sets with least squares on the Vandermonde matrix (columns $1, x, \dots, x^{\text{degree}}$, shape `(30, degree + 1)`), predict on a grid of 200 test points, and
measure Step 1's terms across the 300 fits:

```python
P = np.stack([predict_poly(fit_poly(train_x[s], train_y[s], deg), x_test) for s in range(N_SETS)])  # (300, 200)
avg = P.mean(axis=0)                                  # the average model f_bar at each test point
bias2 = np.mean((avg - f_true(x_test)) ** 2)          # averaged over test points
variance = np.mean(P.var(axis=0))
measured = np.mean((y_test_noisy - P) ** 2)           # actual MSE on fresh noisy test targets
```

```text
== from scratch: bias-variance decomposition ==
noise variance sigma^2 = 0.0900
degree   bias^2  variance sum + noise  test MSE train MSE
     0   0.4975    0.0190      0.6065    0.6033    0.5729
     1   0.2002    0.0211      0.3114    0.3108    0.2731
     2   0.2017    0.0461      0.3378    0.3367    0.2581
     3   0.0048    0.0180      0.1128    0.1122    0.0807
     4   0.0053    0.0262      0.1215    0.1211    0.0772
     5   0.0003    0.0323      0.1225    0.1230    0.0714
     6   0.0001    0.0764      0.1666    0.1663    0.0688
     7   0.0007    0.2489      0.3396    0.3384    0.0655
     8   0.0011    0.7811      0.8721    0.8706    0.0626
     9   0.0168    3.4196      3.5264    3.5193    0.0589
    10   0.6458   89.8251     90.5610   90.5501    0.0559
lowest test MSE at degree 3
```

- "sum + noise" and the measured test MSE agree to the third decimal at every degree: Step 1 in numbers.
- Degree 2 is no better than degree 1: $\sin(2\pi x)$ is antisymmetric about $x = 0.5$ (its value at $0.5 + t$ is minus its value at $0.5 - t$), while the extra term $(x - 0.5)^2$ is symmetric about 0.5, so it adds variance without removing bias.
- Training MSE falls steadily with degree, below the noise floor of 0.09 from degree 3 on. Some of that is expected: least squares with $p$ coefficients has an average training MSE of about $\sigma^2(1 - p/n)$, which is 0.078 for degree 3 ($p = 4$, $n = 30$). Training error *far* below $\sigma^2$, like 0.056 at degree 10, means fitting noise.
- The apparent rise in bias² at degrees 9–10 is a simulation artefact: with variance near 90, the average of 300 fits is itself uncertain (its standard error at each point is about $\sqrt{90/300} \approx 0.55$), and that
  uncertainty shows up in the "bias". More simulated sets shrink it.

**Learning curves.** California housing again (development rows, 5-fold CV). The rule matches scikit-learn's: for each fold, train on the *first* $m$ rows of the fold's training indices.

```text
== from scratch: learning curves ==
depth-2 tree (high bias):
      660 rows: train RMSE 0.8417   val RMSE 0.8728   gap 0.0311
     1320 rows: train RMSE 0.8451   val RMSE 0.8708   gap 0.0257
     2641 rows: train RMSE 0.8321   val RMSE 0.8702   gap 0.0381
     5283 rows: train RMSE 0.8541   val RMSE 0.8654   gap 0.0113
     9246 rows: train RMSE 0.8625   val RMSE 0.8662   gap 0.0037
    13209 rows: train RMSE 0.8597   val RMSE 0.8638   gap 0.0041
full-depth tree (high variance):
      660 rows: train RMSE 0.0000   val RMSE 0.8978   gap 0.8978
     1320 rows: train RMSE 0.0000   val RMSE 0.8562   gap 0.8562
     2641 rows: train RMSE 0.0000   val RMSE 0.8203   gap 0.8203
     5283 rows: train RMSE 0.0000   val RMSE 0.8143   gap 0.8143
     9246 rows: train RMSE 0.0000   val RMSE 0.7510   gap 0.7510
    13209 rows: train RMSE 0.0000   val RMSE 0.7384   gap 0.7384
```

- **High bias:** the depth-2 tree (four leaves) has almost no gap from 660 rows on, and its validation error barely moves from 0.873 to 0.864 with 20 times more data. Collecting data would be wasted
  effort; it needs capacity.
- **High variance:** the full tree memorizes every training row (training RMSE 0) and its validation error falls steadily with data (0.90 to 0.74) but the gap is still 0.74 at the end. More data helps, but
  slowly. Limiting depth (below) or averaging trees (B3.2) helps much faster.

**Validation curve** over the tree's `max_depth`:

```text
== from scratch: validation curve ==
max_depth  1: train RMSE 0.9630   val RMSE 0.9651 (+- 0.0097 across folds)
max_depth  2: train RMSE 0.8597   val RMSE 0.8638 (+- 0.0123 across folds)
max_depth  4: train RMSE 0.7412   val RMSE 0.7571 (+- 0.0047 across folds)
max_depth  6: train RMSE 0.6568   val RMSE 0.6936 (+- 0.0036 across folds)
max_depth  8: train RMSE 0.5656   val RMSE 0.6614 (+- 0.0136 across folds)
max_depth 10: train RMSE 0.4629   val RMSE 0.6597 (+- 0.0232 across folds)
max_depth 12: train RMSE 0.3527   val RMSE 0.6860 (+- 0.0294 across folds)
max_depth 15: train RMSE 0.2072   val RMSE 0.7148 (+- 0.0258 across folds)
max_depth 20: train RMSE 0.0613   val RMSE 0.7340 (+- 0.0298 across folds)
best max_depth by mean validation RMSE: 9
```

Training RMSE falls with every extra level; validation RMSE bottoms out around depth 8–10 (0.66) and then rises as the tree starts memorizing. Depth 8 (0.6614) is within one fold-to-fold standard deviation of depth 10 and of the best
depth, 9 (the lab prints a subset of depths), and is simpler, so it is a defensible choice too. Note that even the best single tree (0.66) is far behind B3.3's boosted model (0.47): the curve finds the best setting *within* a model family, not the best family.
The lab saves both curves to `outputs/b4_5/learning_validation_curves.png`.

## With scikit-learn

```text
== with scikit-learn ==
learning_curve matches for both models: OK
validation_curve matches: OK
```

- `learning_curve(model, X, y, train_sizes=[...], cv=cv, scoring="neg_root_mean_squared_error")` returns the absolute sizes and two arrays of shape `(sizes, folds)`. Scores are negated so that higher is always better:
  flip the sign for errors. Pass the same `cv` object everywhere so that the folds match.
- `validation_curve(model, X, y, param_name="max_depth", param_range=[...], cv=cv, scoring=...)` does the sweep. For pipelines, use the `step__param` name (`"tree__max_depth"`).
- `LearningCurveDisplay` and `ValidationCurveDisplay` draw the plots directly (scikit-learn 1.9.1; VERIFY as of 2026-10).

## In practice

- **When to use it:** read the curves:
  - *Both errors high and close together* (high bias): more data will not help. Add capacity (deeper trees, boosting, interactions), add features, or reduce regularization.
  - *Low training error, large gap* (high variance): add data if the validation curve is still falling, regularize, simplify (limit depth, raise `min_samples_leaf`), average (forests, bagging) or stop earlier.
  - *Validation error flat while training error falls:* you have extracted what the features offer; the remaining error is noise or missing information.
- **Failure modes:**
  - *Collecting data for a high-bias model:* the learning curve has already flattened at a bad level.
  - *Adding layers or trees to a high-variance model:* the gap grows.
  - *Training error far below the noise level* (often unknown, but sometimes bounded by label disagreement between annotators): overfitting.
  - *Curves from a single split:* noisy. Use cross-validated curves and look at the fold spread.
  - *Choosing capacity on the test set* via a validation curve drawn on it (B4.3).
- **Defaults that work:** cross-validated learning curves at 5–6 sizes on a log scale; validation curves over the one or two settings that control capacity; pick the minimum or the simplest
  setting within one standard deviation of it; re-check the learning curve before asking for more data.

## Exercises

Save your work to `my-work/B/b4_5_bias_variance.md` (or `.py`) and ask for a review.

1. **By hand:** (a) At a test point with $f(x_0) = 10$ and $\sigma^2 = 4$, five training sets give predictions $12, 8, 14, 6, 10$. Compute bias², variance and the expected squared error. (b) Another model gives $13, 13, 12, 14, 13$. Which model would you
   rather have, and what would you change about each? (c) Diagnose from the numbers and say what you would do: model A, training RMSE 4.1 and validation RMSE 4.3 at 1,000 rows, and 4.2 and 4.25 at 10,000
   rows; model B, training RMSE 0.5 and validation 3.8 at 1,000 rows, and 0.9 and 3.1 at 10,000 rows. (d) Why can a fully grown tree's training RMSE be 0 while the noise variance is not?
2. **Code:** (a) Repeat the lab's decomposition with k-nearest-neighbour regression (B2.3) on the same $\sin(2\pi x)$ problem for $k = 1, 2, 5, 10, 20, 30$ (with 30 training points, $k = 30$ predicts the mean everywhere). Compare each measured variance with $\sigma^2/k$, the
   variance of an average of $k$ independent noisy targets, and explain where and why they differ. (b) Draw cross-validated learning curves for `RandomForestRegressor(n_estimators=100, max_features=0.33, random_state=0)` on the lab's housing
   data and compare them with the full tree's: does the gap close faster, and why?
3. **Think:** A team's churn model has 99% training accuracy and 82% validation accuracy (base rate 20% churn). They plan to (i) collect a year more data, (ii) switch to a deeper model, and
   (iii) add 200 new features from a vendor. Which curves would you ask for before approving any of these, what would each curve need to show to justify each plan, and is accuracy even the right
   metric to read them with?

## Takeaways

1. Expected test error = noise + bias² + variance: bias is the average model's systematic miss, variance the scatter of models trained on different samples, noise the floor nothing removes.
2. More capacity lowers bias and raises variance; more data lowers variance only; averaging lowers variance; boosting lowers bias; regularization trades one for the other.
3. A learning curve diagnoses the problem: curves meeting at a high error mean high bias (more data will not help); a large gap that shrinks with data means high variance.
4. A validation curve chooses capacity: training error keeps falling, validation error bottoms out; choose the minimum or the simplest setting within one standard deviation of it.
5. Training error far below the noise level, or zero, is a symptom of fitting noise; read the curves cross-validated, never on the test set.

**What builds on this:** [C3.5](../C/c3_5_regularization.md) (regularizing neural networks); [C4.3](../C/c4_3_debugging_training.md) (learning curves
for debugging training runs); [B6.4](b6_4_arima_vs_boosting.md) (capacity choices for forecasting models); P2 and P3 (justifying capacity and data decisions).

**References:** S. Geman, E. Bienenstock, R. Doursat, "Neural networks and the bias/variance dilemma", *Neural Computation* 4, 1992. T. Hastie, R. Tibshirani,
J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009, sections 2.9 and 7.3. M. Belkin, D. Hsu, S. Ma, S. Mandal, "Reconciling modern machine-learning practice
and the classical bias–variance trade-off", *PNAS* 116, 2019. scikit-learn User Guide, "Validation curves: plotting scores to evaluate models" (VERIFY link:
https://scikit-learn.org/stable/modules/learning_curve.html).
