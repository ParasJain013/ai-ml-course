# B3.3 Gradient boosting: fitting residuals from scratch, then XGBoost / LightGBM in practice

**Core** · Part B, week 3 (course week 09) · ~3 h · Lab:
[`labs/B/b3_3_gradient_boosting.py`](../../labs/B/b3_3_gradient_boosting.py)

## Why it matters

For prediction on tabular data (churn, fraud, pricing, demand, credit risk), gradient-boosted trees are the model to
beat, and often the model that ships. XGBoost and LightGBM are what most teams reach for after a linear
baseline, and "why does boosting work, and how do you stop it overfitting?" is a standard interview question. The idea is
simple: start with a constant prediction, then repeatedly fit a small tree to the current mistakes and add a little
of it. In production the usual failures come from how it is driven, not from the algorithm: a learning rate too
high, early stopping on the test set, or trusting a model that has silently learned a leak.

## Prerequisite check

- Can you grow a regression tree, where a leaf predicts the mean of its rows? If not, revisit
  [B3.1](b3_1_decision_trees.md).
- Can you explain the gradient-descent update $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\nabla\mathcal{L}$ and what a learning rate that is too large does? If not, revisit
  [A4.4](../A/a4_4_gradient_descent.md).
- Can you derive $\partial\ell/\partial z = p - y$ for log-loss with a logit $z$? If not, revisit
  [B2.1](b2_1_logistic_regression.md).
- Can you say why a random forest uses deep trees (averaging reduces variance, not bias)? If not, revisit
  [B3.2](b3_2_random_forests.md).
- Do you know why the test set must not be used to choose settings? If not, revisit [B1.1](b1_1_ml_workflow.md).

## Intuition

A random forest builds deep trees independently and averages them. Boosting does the opposite: it builds *shallow*
trees *in sequence*, each one trained to fix what the previous ones got wrong.

Picture predicting house prices. Start with the dumbest model: predict the average price everywhere. Compute each
district's mistake, the **residual** $y - \hat{y}$. Fit a depth-3 tree to the *residuals*, not to the prices: it learns, say, "rich
coastal districts are under-predicted by about $150k". Add a *fraction* of that tree's output to the model, maybe 10%.
Recompute the residuals and fit another tree to what is still wrong. Repeat a few hundred times. Each tree is weak on
its own, but each one chips away at the remaining error.

Two settings control everything:

- **The learning rate $\eta$** (**shrinkage**): how much of each tree to add. With $\eta = 1$ the model trusts each small tree
  completely, chases the noise in the residuals, and overfits within a few rounds. With $\eta = 0.05$ each tree nudges the
  model, later trees can correct earlier ones, and the result generalizes better, at the cost of more trees.
- **The number of trees $M$**: too few underfits, too many slowly overfits. Choose it with **early stopping**: watch
  the error on held-out rows and stop when it has not improved for a while.

Why "gradient"? For squared error, the residual is exactly minus the gradient of the loss with respect to the
prediction. So each round is a gradient-descent step, except that the thing being updated is the prediction function itself,
not a parameter vector. That view lets boosting use any differentiable loss, log-loss for classification included, by
fitting each tree to minus the gradient of that loss (the **pseudo-residuals**).

> **Bridge:** boosting is like iterating on a failing test suite. Each pass, you look only at the tests still failing
> and write a small fix aimed at them, then rerun. **Where it breaks:** a code fix for a failing test is either right
> or wrong. Here every "fix" is learned from noisy data, so each one is applied only partially (the learning rate),
> and "all tests pass" (zero training error) is the danger sign, not the goal.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{X}$, $\mathbf{y}$ | training features and targets | `(n, d)`, `(n,)` |
| $F_m(\mathbf{x})$ | the model after $m$ trees: a prediction for one input | scalar |
| $\mathbf{F}_m$ | that model's predictions on the training rows | `(n,)` |
| $f_m$ | tree number $m$ (a function from $\mathbf{x}$ to a number) | — |
| $\eta$, $M$ | learning rate, number of trees | scalar, int |
| $\ell(y, F)$ | loss of one example | scalar |
| $r_i$, $\mathbf{g}$ | pseudo-residual $-g_i$ of row $i$; the vector of all $g_i$ | scalar, `(n,)` |
| $p_i$, $\sigma$ | predicted probability $\sigma(F_{m-1}(\mathbf{x}_i))$ for log-loss; the sigmoid | scalar |
| $\Omega(f)$ | XGBoost's penalty on one tree | scalar |
| $g_i$, $h_i$ | first and second derivative of $\ell(y_i, F)$ with respect to $F$, at $F = F_{m-1}(\mathbf{x}_i)$ | scalars |
| $T$, $w_j$, $I_j$ | number of leaves of a tree, value (weight) of leaf $j$, rows in leaf $j$ | int, scalar, set |
| $G_j$, $H_j$ | $\sum_{i \in I_j} g_i$ and $\sum_{i \in I_j} h_i$ | scalars |
| $\lambda$, $\gamma$ | XGBoost's L2 penalty on leaf weights and its cost per leaf | scalars |

**Step 1: the model is a sum.** $F_M(\mathbf{x}) = F_0 + \eta\sum_{m=1}^{M} f_m(\mathbf{x})$. Start from the best constant: for squared error the mean, $F_0 = \bar{y}$.

**Step 2: for squared error, residuals are negative gradients.** With $\ell = \frac12(y - F)^2$, $\frac{\partial\ell}{\partial F} = -(y - F)$. So the vector of residuals $\mathbf{y} - \mathbf{F}_{m-1}$, shape `(n,)`, is exactly
$-\mathbf{g}$. If we could move each training prediction freely, gradient descent would do $\mathbf{F}_m = \mathbf{F}_{m-1} - \eta\mathbf{g} = \mathbf{F}_{m-1} + \eta(\mathbf{y} - \mathbf{F}_{m-1})$. But that only changes the $n$
training predictions and says nothing about new inputs. So we fit a tree $f_m$ to the targets $-\mathbf{g}$: the closest
*function* to the gradient step that generalizes to new $\mathbf{x}$. Then $F_m = F_{m-1} + \eta f_m$. That is all of gradient boosting (Friedman, 2001):

1. $F_0 = \bar{y}$.
2. For $m = 1, \dots, M$: compute $r_i = y_i - F_{m-1}(\mathbf{x}_i)$; fit a depth-limited regression tree $f_m$ to $(\mathbf{X}, \mathbf{r})$; set $F_m = F_{m-1} + \eta f_m$.

**Step 3: other losses.** Replace residuals by **pseudo-residuals** $r_i = -g_i$. For binary log-loss with $F$ = the logit, B2.1
Step 3 gave $g = p - y$ with $p = \sigma(F)$, so $r_i = y_i - p_i$: the classifier fits trees to "label minus predicted probability", starting from
$F_0 = \log\frac{\bar{y}}{1 - \bar{y}}$, the log-odds of the positive rate. The tree's *splits* come from fitting $\mathbf{r}$; its *leaf values* are then recomputed with a Newton step,
$w_j = \sum_{i \in I_j} r_i \big/ \sum_{i \in I_j} p_i(1 - p_i)$, which uses the curvature (Step 6 shows where that comes from). Exercise 2 builds this.

**Step 4: why the learning rate helps.** A tree fitted to residuals fits their noise too. Adding all of it ($\eta = 1$) bakes that
noise into the model immediately, and later trees then fit residuals that are partly the negatives of earlier noise.
A small $\eta$ adds a little of each tree, so the noise of any single tree is diluted and later trees can correct it.
Empirically, a smaller $\eta$ with proportionally more trees almost always generalizes better, down to about 0.01–0.1, beyond
which the gains vanish while training time keeps growing. Halving $\eta$ roughly doubles the trees needed.

**Step 5: depth sets the interactions.** A depth-1 tree (a **stump**) uses one feature, so a sum of stumps is an
additive model: no interactions. A depth-3 tree can express a three-way interaction such as "rich *and* coastal *and*
old". Boosting uses shallow trees (depth 3–8) because each one only needs to be a weak improvement; the ensemble
supplies the complexity. This is the mirror image of B3.2: boosting reduces *bias* by adding trees, so it can afford
high-bias base trees, and it controls variance with $\eta$, early stopping and subsampling.

**Step 6: what XGBoost adds** (Chen and Guestrin, 2016; the paper guide goes deeper). It writes the objective for tree $m$
as the training loss plus a penalty on the tree, $\Omega(f) = \gamma T + \frac12\lambda\sum_j w_j^2$, and approximates the loss around the current predictions with a
second-order Taylor expansion (A4.1), $\ell(y_i, F + f) \approx \ell(y_i, F) + g_if + \frac12 h_i f^2$, where $f$ is short for the new tree's output $f(\mathbf{x}_i)$. Drop $\ell(y_i, F)$, which does not depend on the tree. Every row in leaf $j$ gets the same value $w_j$, so
grouping the rows by leaf, $\sum_i g_i f(\mathbf{x}_i) = \sum_j \big(\sum_{i \in I_j} g_i\big) w_j = \sum_j G_jw_j$, and likewise $\sum_i h_i f(\mathbf{x}_i)^2 = \sum_j H_jw_j^2$. With the penalty, the objective is a sum over leaves:

$$\sum_{j=1}^{T}\Big[G_jw_j + \tfrac12(H_j + \lambda)w_j^2\Big] + \gamma T.$$

Each bracket is a parabola in $w_j$. Setting its derivative $G_j + (H_j + \lambda)w_j$ to zero gives the best leaf value. Plugging it back in, $G_jw_j^* + \frac12(H_j + \lambda)w_j^{*2} = -\frac{G_j^2}{H_j + \lambda} + \frac12\frac{G_j^2}{H_j + \lambda} = -\frac12\frac{G_j^2}{H_j + \lambda}$, so:

$$w_j^* = -\frac{G_j}{H_j + \lambda}, \qquad \text{objective} = -\frac12\sum_{j=1}^{T}\frac{G_j^2}{H_j + \lambda} + \gamma T.$$

Splitting one leaf (with sums $G = G_L + G_R$, $H = H_L + H_R$) into a left and a right child replaces its term $-\frac12\frac{G^2}{H + \lambda}$ by the two children's terms and adds one leaf ($+\gamma$). The
drop in the objective is the split's gain:

$$\text{gain} = \frac12\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma.$$

For squared error, $g_i = F - y_i$ and $h_i = 1$, so $w_j^* = \frac{\sum_{i \in I_j}(y_i - F)}{|I_j| + \lambda}$: the mean residual of the leaf, shrunk towards zero by $\lambda$. With $\lambda = 0$ it is exactly the
leaf value of Step 2. For log-loss, $g_i = p_i - y_i$ and, since $\frac{d}{dF}\sigma(F) = \sigma(F)(1 - \sigma(F))$ (B2.1 Step 1), $h_i = p_i(1 - p_i)$; then $w_j^*$ is the Newton step of Step 3. The gain replaces Gini or variance as the split criterion, and
$\gamma$ makes a split pay for its extra leaf: a split with gain below zero is not made. These two ideas, the regularized objective and
second-order leaf values, are what the paper adds to plain gradient boosting. The rest of the paper is engineering:
approximate split finding with quantiles, learned default directions for missing values, and cache-aware parallel computation.

**Step 7: what LightGBM adds** (Ke and colleagues, 2017). It bins each feature into at most a few hundred values once
(histograms), so a split search scans bins instead of sorted rows; and it grows trees **leaf-wise** (split whichever
leaf has the largest gain anywhere in the tree, up to `num_leaves`) instead of level by level. Both make it fast; leaf-wise
growth can overfit small data, so `num_leaves` and `min_child_samples` matter.

## Worked example by hand

Four points, one feature: $x = 1, 2, 3, 4$ with $y = 1, 3, 6, 10$. Stumps (depth 1), $\eta = 0.5$.

**Start.** $F_0 = \bar{y} = 20/4 = 5$. Residuals $\mathbf{r} = (-4, -2, 1, 5)$. MSE $= (16 + 4 + 1 + 25)/4 = 11.5$.

**Round 1.** Fit a stump to $\mathbf{r}$. The three cuts and their summed squared errors (SSE):
- $x \le 1.5$: left $\{-4\}$, right $\{-2, 1, 5\}$ (mean 1.333): SSE $0 + (11.11 + 0.11 + 13.44) = 24.67$.
- $x \le 2.5$: left $\{-4, -2\}$ (mean $-3$, SSE 2), right $\{1, 5\}$ (mean 3, SSE 8): SSE **10**.
- $x \le 3.5$: left $\{-4, -2, 1\}$ (mean $-1.667$, SSE 12.67), right $\{5\}$: SSE 12.67.

Best: $x \le 2.5$, leaf values $-3$ and $+3$. Update with half of it: $\mathbf{F}_1 = 5 + 0.5 \times (-3, -3, 3, 3) = (3.5, 3.5, 6.5, 6.5)$. MSE $= (6.25 + 0.25 + 0.25 + 12.25)/4 = 4.75$.

**Round 2.** New residuals $\mathbf{y} - \mathbf{F}_1 = (-2.5, -0.5, -0.5, 3.5)$. Now the best cut is $x \le 3.5$ (SSE $2.667$, against 10 for 2.5 and 10.67 for 1.5): leaf values
$(-2.5 - 0.5 - 0.5)/3 = -1.1667$ and $3.5$. $\mathbf{F}_2 = \mathbf{F}_1 + 0.5 \times (-1.1667, -1.1667, -1.1667, 3.5) = (2.9167, 2.9167, 5.9167, 8.25)$, MSE 1.6875. The second tree went after the biggest remaining mistake, the point $x = 4$.

**XGBoost's leaf weight.** Take round 1 with XGBoost's formula. With $F = 5$: $g_i = F - y_i = (4, 2, -1, -5)$, $h_i = 1$. Left leaf: $G_L = 6$, $H_L = 2$. With $\lambda = 0$:
$w_L = -6/2 = -3$, the residual mean. With $\lambda = 1$: $w_L = -6/3 = -2$, shrunk towards zero. By symmetry $w_R = +3$ or $+2$.

```text
== worked example by hand ==
F0 = mean(y) = 5.0000   MSE 11.5000
round 1: residuals [-4.0, -2.0, 1.0, 5.0]   split x <= 2.5   leaf values [-3.0, 3.0]
         F1 = [3.5, 3.5, 6.5, 6.5]   MSE 4.7500
round 2: residuals [-2.5, -0.5, -0.5, 3.5]   split x <= 3.5   leaf values [-1.1667, 3.5]
         F2 = [2.9167, 2.9167, 5.9167, 8.25]   MSE 1.6875
GradientBoostingRegressor(n_estimators=2, learning_rate=0.5, max_depth=1) gives the same F2: OK
XGBoost, one stump, lambda 0.0: leaf weights [-3.0, -3.0, 3.0, 3.0]
XGBoost, one stump, lambda 1.0: leaf weights [-2.0, -2.0, 2.0, 2.0]
```

## Key plot

![Validation RMSE against boosting rounds on a log scale for learning rates 1.0, 0.3, 0.1 and 0.03. The 1.0 curve bottoms at 0.57 after 7 rounds and rises to 0.62; 0.3 bottoms at 0.49; 0.1 reaches 0.476 after about 700 rounds; 0.03 is still falling at 0.48 after 1,000 rounds](img/b3_3_learning_rate_curves.png)

XGBoost on California housing, one depth-6 tree per round. With $\eta = 1$ the validation error bottoms out after 7 trees and
then *rises*: the model is fitting noise. Each smaller learning rate starts slower and ends lower, until $\eta = 0.03$, which
needs more than 1,000 rounds to catch up. The best region is a learning rate of 0.03–0.1 and however many trees early stopping
picks. (These curves watch the validation set only to illustrate; for real model selection, use the early-stopping setup below.)

## From scratch in NumPy

**The data.** California housing (1990 census): 20,640 districts, 8 features, target = median house value in units of
$100,000, capped at 5.00001 (992 districts sit at the cap). The lab downloads a copy that matches
`sklearn.datasets.fetch_california_housing` except that 207 bedroom counts are missing; for scikit-learn's
`GradientBoostingRegressor`, which refuses `NaN`, they are filled with the *training* median (B1.2, B1.4). XGBoost and LightGBM get the raw
values with `NaN`.

```text
== data ==
data/california_housing/housing.tgz: checksum OK, 20640 districts, 8 features
missing AveBedrms: 207 rows (filled with the training median 1.0491 for sklearn)
target: median house value in $100,000s, mean 2.0686, capped at 5.00001 (992 districts at the cap)
train (12384, 8)  val (4128, 8)  (test 4128 rows kept aside)
   MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude
0  8.3252      41.0    6.9841     1.0238       322.0    2.5556     37.88    -122.23
1  8.3014      21.0    6.2381     0.9719      2401.0    2.1098     37.86    -122.22
2  7.2574      52.0    8.2881     1.0734       496.0    2.8023     37.85    -122.24
```

**The boosting loop.** Step 2 in code. The base learner is scikit-learn's `DecisionTreeRegressor` (you built one in B3.1's
Exercise 2), so the loop itself is the from-scratch part:

```python
def fit_boosting(X, y, n_trees, lr, max_depth):
    """Gradient boosting for squared error. X (n, d), y (n,) -> (F0, trees)."""
    f0 = float(y.mean())
    F = np.full(len(y), f0)                        # (n,) current predictions
    trees = []
    tie_breaker = np.random.RandomState(SEED)      # mirrors scikit-learn's tie-breaking, see the lab
    for _ in range(n_trees):
        residual = y - F                           # (n,) negative gradient of (1/2)(y - F)^2
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=tie_breaker).fit(X, residual)
        F += lr * tree.predict(X)                  # a small step in the direction the tree points
        trees.append(tree)
    return f0, trees
```

Predicting after $m$ trees is `f0 + lr * cumsum(tree predictions)[m - 1]`, so one pass gives the whole curve.

```text
== from scratch: boosting on California housing ==
~ fitted 200 trees in 4.8 s
F0 = 2.0626: train RMSE 1.1472  val RMSE 1.1576
after   1 trees: train RMSE 1.0875  val RMSE 1.0983
after  10 trees: train RMSE 0.7983  val RMSE 0.8200
after  50 trees: train RMSE 0.5528  val RMSE 0.5945
after 100 trees: train RMSE 0.5054  val RMSE 0.5610
after 200 trees: train RMSE 0.4618  val RMSE 0.5333
```

The constant model has a typical error of $116k (RMSE 1.16). Each tree is small, and the first one cuts the RMSE by only
5%, but 200 of them bring it to 0.53. Validation is still improving at 200 trees; the training error falls faster, and the gap
between them (0.46 against 0.53) grows as the model starts to fit the training noise.

## With scikit-learn, XGBoost and LightGBM

**`GradientBoostingRegressor`** runs exactly the loop above: start at the mean, fit depth-3 regression trees to the
residuals, add 10% of each. When two splits tie, it picks between them using a random feature order drawn from one shared generator (B3.1); the lab shares one the same way, so
it can demand identical trees:

```text
== with scikit-learn: GradientBoostingRegressor ==
val RMSE: mine 0.5333   sklearn 0.5333
predictions match GradientBoostingRegressor at every one of the 200 stages: OK
```

In scikit-learn 1.9.1, `GradientBoostingRegressor`'s `criterion` parameter is deprecated and has no effect (VERIFY as of 2026-10).
For larger data, scikit-learn's `HistGradientBoostingRegressor` is its LightGBM-style, histogram-based version.

**The learning-rate sweep** behind the key plot:

```text
== key plot: learning rate ==
learning rate 1.0 : best val RMSE 0.5747 at round    7   val RMSE at round 1000: 0.6213
learning rate 0.3 : best val RMSE 0.4937 at round  168   val RMSE at round 1000: 0.4954
learning rate 0.1 : best val RMSE 0.4757 at round  694   val RMSE at round 1000: 0.4764
learning rate 0.03: best val RMSE 0.4799 at round 1000   val RMSE at round 1000: 0.4799
saved lessons/B/img/b3_3_learning_rate_curves.png
```

**XGBoost and LightGBM in practice.** The standard recipe: a low learning rate (0.05), a generous upper limit on rounds,
row and column subsampling (0.8 each: each tree sees 80% of the rows and of the features, which decorrelates trees as in B3.2), and **early stopping**
after 100 rounds without improvement. Early stopping needs held-out rows, and they must not be the validation rows you then report: the stopping point is
tuned on them, so their score would be optimistic. The lab carves 20% of the training rows off as an early-stopping set
and keeps the validation set for comparing models:

```python
xgb_model = xgb.XGBRegressor(n_estimators=5000, learning_rate=0.05, max_depth=6, subsample=0.8,
                             colsample_bytree=0.8, early_stopping_rounds=100, eval_metric="rmse")
xgb_model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)

lgb_model = lgb.LGBMRegressor(n_estimators=5000, learning_rate=0.05, num_leaves=63, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, verbose=-1)
lgb_model.fit(X_fit, y_fit, eval_X=X_es, eval_y=y_es, callbacks=[lgb.early_stopping(100, verbose=False)])
```

```text
== XGBoost and LightGBM with early stopping ==
training rows split for early stopping: fit 9907  early-stopping 2477
model                                        val RMSE  trees
predict the training mean                      1.1576      -
linear regression (B1.5)                       0.7509      -
random forest (B3.2), max_features 0.33        0.5230    200
from-scratch boosting, depth 3, lr 0.1         0.5333    200
XGBoost, lr 0.05, early stopping               0.4724    788
LightGBM, lr 0.05, early stopping              0.4694    426
~ training time: random forest 4.1 s   XGBoost 2.3 s   LightGBM 0.9 s (1 thread each)
XGBoost features by total gain: MedInc, Latitude, AveOccup, Longitude
```

- The tuned boosters beat linear regression by 37% in RMSE (0.47 against 0.75: about $47k against $75k). House prices depend on location in a
  very non-linear way (latitude and longitude only matter in combination), and trees capture that without hand-made features.
- XGBoost and LightGBM land within 0.003 of each other. That is typical: on most tabular problems the choice between
  them matters far less than the features and the validation design. Early stopping picked 788 and 426 trees; the number of trees is
  not comparable across libraries with different tree shapes (leaf-wise versus depth-wise growth).
- Both used the raw `NaN` bedroom counts. At every split they learn a default direction for missing values, the
  paper's "sparsity-aware split finding". No imputation step to leak or to get wrong.
- Timings start with `~` because they vary between machines. LightGBM's histograms make it the fastest here, and the gap
  grows with data size. XGBoost's default `tree_method` is also histogram-based in recent versions (VERIFY as of 2026-10).

## In practice

- **When to use it:** the first serious model for tabular data, after a linear baseline: mixed numeric and categorical
  features, missing values, non-linear effects, thousands to hundreds of millions of rows. It is usually the most
  accurate model there without heavy tuning. Prefer something else for images, audio and free text (neural
  networks, Parts C–F), when you need smooth extrapolation beyond the training range (trees predict flat outside it), or
  when the model must be a readable formula (logistic regression, B2.1).
- **Failure modes:**
  - *Learning rate too high.* The validation curve bottoms out within a few dozen rounds and rises (key plot). Lower $\eta$, raise the
    round limit, use early stopping.
  - *Early stopping on the test or reporting set.* The stopping point is a tuned setting; the score on the set that
    chose it is optimistic. Use a separate early-stopping slice or cross-validation (B4.3).
  - *Leaks are found first.* Boosting greedily exploits the strongest signal, so a leaked column (B1.4) jumps to the
    top of the importance list. Treat a surprisingly dominant feature as a leak until proved otherwise (P2 M2 asks you to
    check).
  - *Overfitting small data.* Deep trees, many leaves (LightGBM's leaf-wise growth) and hundreds of rows do not mix.
    Shrink `max_depth` or `num_leaves`, raise `min_child_weight` / `min_child_samples`, add `reg_lambda`.
  - *Probabilities.* Boosted classifiers trained on log-loss are often reasonably calibrated, but check (B4.1),
    especially after class weighting (B4.4).
- **Defaults that work** (XGBoost 3.4, LightGBM 4.7; VERIFY as of 2026-10): `learning_rate` 0.03–0.1 with early stopping
  (patience 50–200 rounds) on a held-out slice; XGBoost `max_depth` 4–8 or LightGBM `num_leaves` 15–127; `subsample` and
  `colsample_bytree` around 0.8; tune `min_child_weight` / `min_child_samples` and `reg_lambda` next. Retrain on all
  training rows with the chosen number of rounds, or keep the early-stopped model; either is defensible if the test set
  stays untouched.

## Exercises

Save your work to `my-work/B/b3_3_gradient_boosting.md` (or `.py`) and ask for a review.

1. **By hand:** Four points $x = 1, 2, 3, 4$ with $y = 2, 2, 6, 10$. (a) Compute $F_0$ and the residuals, fit the best stump to the residuals (show the SSE of
   each cut) and compute $\mathbf{F}_1$ with $\eta = 0.1$. (b) For the same first round, compute XGBoost's leaf weights with $\lambda = 2$ and the gain of the chosen split
   with $\lambda = 2$, $\gamma = 1$. Would the split be made? (c) A binary classifier starts from a positive rate of 0.25. Give $F_0$, the pseudo-residuals for a
   positive and a negative example, and the Newton leaf value for a leaf holding one positive and one negative example
   (both still at $F_0$). (d) You halve the learning rate. Roughly how should the number of trees change, and why?
2. **Code:** Build gradient boosting for binary classification from scratch on breast cancer (B2.1's split and its
   five features): $F_0$ = log-odds of the training positive rate; each round, fit a
   `DecisionTreeRegressor(max_depth=3, min_samples_leaf=5)` to the pseudo-residuals $y - p$, replace its leaf values with the Newton
   step $\sum r / \sum p(1 - p)$, and add $\eta = 0.1$ times the result. Match `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
   max_depth=3, min_samples_leaf=5, random_state=0)`'s `predict_proba` on the validation rows at every stage (share a
   `np.random.RandomState(0)` tie-breaker as the lab does). Then refit your model on all 30 features. Plot validation log-loss against
   rounds for both, and compare with B2.1's logistic regression (validation log-loss 0.0856 on the five features). Which wins, and why?
3. **Think:** For each case, say whether you would start with gradient-boosted trees, a neural network or a linear
   model, and why: (a) predicting delivery time from 40 order, restaurant and weather features, 2 million rows; (b)
   classifying support tickets from their text; (c) a credit-limit model that must show applicants which factors lowered
   their score, 30,000 rows; (d) forecasting next year's electricity demand when demand has grown every year. Then name
   two ways a gradient-boosted model can fail silently after deployment.

## Takeaways

1. Gradient boosting builds $F_M = F_0 + \eta\sum_m f_m$: each shallow tree is fitted to the current residuals (for squared error, the negative
   gradient), and only a fraction $\eta$ of it is added.
2. For any differentiable loss, trees are fitted to pseudo-residuals $-g_i$; for log-loss that is $y - p$, with Newton-step leaf values
   $\sum r/\sum p(1 - p)$.
3. The learning rate and the number of trees trade off: small $\eta$ and many trees generalize better, and early stopping on
   held-out rows (never the reporting set) chooses the number.
4. XGBoost adds a regularized objective ($\gamma T + \frac12\lambda\lVert\mathbf{w}\rVert^2$) and second-order leaf values $w^* = -G/(H + \lambda)$ with a matching split gain; LightGBM adds
   histograms and leaf-wise growth. Both handle missing values natively.
5. On tabular data, tuned boosting is usually the strongest model (here 0.47 RMSE against 0.75 for linear regression), but it
   cannot extrapolate, and it will find any leak first.

**What builds on this:** [B3.4](b3_4_interpretability_shap.md) (explaining the boosted model with SHAP);
[B4.3](b4_3_cross_validation_tuning.md) (tuning boosting with cross-validation and Bayesian search);
[B4.4](b4_4_class_imbalance.md) (boosting with imbalanced classes); [B6.4](b6_4_arima_vs_boosting.md) (boosting on lag
features for forecasting); the XGBoost paper guide ([`papers/2016_xgboost.md`](../../papers/2016_xgboost.md)); P2 M2 (the challenger
model) and P12.

**References:** J. H. Friedman, "Greedy function approximation: a gradient boosting machine", *Annals of Statistics* 29(5),
2001. T. Chen, C. Guestrin, "XGBoost: A Scalable Tree Boosting System", *KDD*, 2016 (arXiv 1603.02754). G. Ke et al.,
"LightGBM: A Highly Efficient Gradient Boosting Decision Tree", *NeurIPS*, 2017. R. K. Pace, R. Barry, "Sparse spatial
autoregressions", *Statistics & Probability Letters* 33, 1997 (the source of the California housing data). XGBoost
documentation, "Introduction to Boosted Trees" (VERIFY link: https://xgboost.readthedocs.io/en/stable/tutorials/model.html).
