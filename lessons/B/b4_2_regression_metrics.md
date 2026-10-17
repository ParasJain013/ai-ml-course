# B4.2 Regression metrics, and choosing a metric that matches the real cost of errors

**Core** · Part B, week 4 (course week 10) · ~1 h · Lab:
[`labs/B/b4_2_regression_metrics.py`](../../labs/B/b4_2_regression_metrics.py)

## Why it matters

A metric is a statement about which errors you care about. RMSE says "one big miss is worse than several small ones";
MAE says "every unit of error costs the same"; MAPE says "a miss of 10 units matters more on a small item than on a large one".
Pick the wrong one and you optimize the wrong model: a demand forecast tuned on MAPE quietly under-orders, and a price model
tuned on RMSE is pushed around by a few mansions. The same logic sets a classifier's threshold: the cost of a false positive
and of a false negative fix the threshold, not habit. This lesson builds the four standard regression metrics from scratch, shows each one's
blind spot on real data, and turns error costs into a threshold. P2 asks you to justify your metric and threshold from costs,
and interviewers ask "which metric would you use, and why?" about almost every project.

## Prerequisite check

- Can you compute RMSE and $R^2$ for a linear regression? If not, revisit [B1.5](b1_5_linear_regression.md).
- Do you know the difference between the mean and the median of a skewed sample? If not, revisit
  [A6.1](../A/a6_1_descriptive_statistics_eda.md).
- Can you read a confusion matrix and say what a false positive is? If not, revisit
  [B4.1](b4_1_classification_metrics.md).
- Can you compute an expected value from probabilities and payoffs? If not, revisit
  [A5.1](../A/a5_1_random_variables_expectation.md).

## Intuition

Every regression metric averages the errors $e_i = y_i - \hat{y}_i$ after passing them through a penalty, and the penalty's shape is the
statement:

- **MAE** (mean absolute error) charges $|e|$: linear. Doubling an error doubles its cost. Reported in the target's units.
- **RMSE** (root mean squared error) charges $e^2$, then takes the square root to return to the target's units. Squaring makes large errors
  dominate: one error of 30 costs as much as nine errors of 10 in the mean square.
- **MAPE** (mean absolute percentage error) charges $|e|/|y|$: relative. Missing by 2 units on an item that sold 1 is a 200% error;
  on an item that sold 100 it is 2%. Undefined when $y = 0$.
- **$R^2$** compares the squared error with that of always predicting the mean: 1 is perfect, 0 is no better than the mean, negative is worse.

A consequence that surprises many people: the best constant prediction differs by metric. For squared error it is the
**mean** of the targets; for absolute error it is the **median**. A model trained on one loss learns, for each input, an estimate of that
quantity, the conditional mean or the conditional median. When the target is skewed (prices, incomes, waiting times) the two differ, and so do the
models and their metric scores.

For decisions, start from money, not from a metric. If a false negative (a missed fraud) costs five times as much as a
false positive (a review), the classification threshold should be low; if the costs are equal, it should be near 0.5.
For calibrated probabilities there is a formula for it.

> **Bridge:** choosing MAE versus RMSE is like choosing between average latency and a tail metric such as p99. RMSE
> behaves like a tail metric: a few slow requests (large errors) drive it. **Where it breaks:** p99 ignores everything below the
> 99th percentile, while RMSE still counts every error, just weighted by its size. If you truly care only about the worst
> cases, report a high quantile of $|e|$ directly.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{y}$, $\hat{\mathbf{y}}$ | targets and predictions | `(n,)` |
| $\mathbf{e}$ | errors $\mathbf{y} - \hat{\mathbf{y}}$ | `(n,)` |
| $\bar{y}$ | mean of the targets | scalar |
| $c$ | a constant prediction | scalar |
| $c_{FP}$, $c_{FN}$ | cost of one false positive and one false negative | scalars |
| $p$ | a calibrated predicted probability that $y = 1$ | scalar |
| $t$ | classification threshold | scalar |

**Step 1: the four metrics.**

$$\text{MAE} = \frac1n\sum_i|e_i|, \qquad \text{RMSE} = \sqrt{\frac1n\sum_i e_i^2}, \qquad \text{MAPE} = \frac1n\sum_i\frac{|e_i|}{|y_i|}, \qquad R^2 = 1 - \frac{\sum_i e_i^2}{\sum_i(y_i - \bar{y})^2}.$$

RMSE is never smaller than MAE: $\text{RMSE}^2 - \text{MAE}^2 = \frac1n\sum_i e_i^2 - \big(\frac1n\sum_i|e_i|\big)^2$ is the variance of the numbers $|e_i|$ (A5.1: variance = mean of squares minus square of mean), which is never negative. It is zero, and the two
metrics equal, only when every error has the same size. A large ratio RMSE / MAE means a few large errors.

**Step 2: the best constant for squared error is the mean.** Minimize $f(c) = \sum_i(y_i - c)^2$. Its derivative is $f'(c) = -2\sum_i(y_i - c) = -2\big(\sum_i y_i - nc\big)$, which is zero at $c = \frac1n\sum_i y_i = \bar{y}$ (and $f'' = 2n > 0$, so a minimum).

**Step 3: the best constant for absolute error is the median.** Minimize $g(c) = \sum_i|y_i - c|$. Away from the data points, each term has slope $-1$ if $y_i > c$
(moving $c$ up shrinks that error) and $+1$ if $y_i < c$. So $g'(c) = (\text{number of } y_i \text{ below } c) - (\text{number above } c)$. While more points lie above $c$, raising $c$ lowers the cost; once more lie below, raising it raises the cost. The minimum is where
the counts balance: the median. One huge target moves the mean a lot and the median not at all, which is why MAE is called robust to outliers.

**Step 4: $R^2$ is relative to the evaluation set.** $R^2 = 1 - \text{MSE}/\operatorname{Var}(y)$, where both are computed on the same rows. The same model has a higher $R^2$ on a set with more spread in $y$, so
$R^2$ values from different datasets are not comparable. And $R^2$ can be negative: predicting the *training* mean on validation rows already gives a slightly negative value.

**Step 5: MAPE is asymmetric.** For a positive target, under-forecasting can cost at most 100% ($\hat{y} = 0$ gives $|e|/y = 1$), while over-forecasting is unbounded. Rows with
small $y$ dominate the average. Minimizing MAPE therefore pulls forecasts down. If you need a relative metric, prefer the
weighted version WAPE $= \sum_i|e_i| / \sum_i|y_i|$ (total error relative to total volume), which is defined whenever the total is positive.

**Step 6: a threshold from costs.** For one row with calibrated probability $p$ of being positive: flagging it costs $c_{FP}$ if it is negative, so its expected
cost is $(1 - p)\,c_{FP}$; not flagging costs $c_{FN}$ if it is positive, expected $p\,c_{FN}$. Flag when flagging is cheaper:

$$(1 - p)\,c_{FP} < p\,c_{FN} \iff c_{FP} < p\,(c_{FP} + c_{FN}) \iff p > t^* = \frac{c_{FP}}{c_{FP} + c_{FN}}.$$

Equal costs give $t^* = 0.5$; a miss five times as costly gives $t^* = 1/6 = 0.167$. The formula needs calibrated probabilities (B4.1). In practice, also pick the threshold
empirically: compute the average cost $\frac1n(c_{FP}\,\text{FP}(t) + c_{FN}\,\text{FN}(t))$ on validation rows for every $t$ and take the minimum.

## Worked example by hand

Five targets $y = (10, 12, 14, 16, 48)$ and two models. A is exact on three rows, nearly exact on one, and misses the outlier: $\hat{y}_A = (11, 12, 14, 16, 20)$. B is off by 6 everywhere: $\hat{y}_B = (16, 18, 20, 22, 42)$.

- **MAE.** A: $(1 + 0 + 0 + 0 + 28)/5 = 5.8$. B: $(6 \times 5)/5 = 6.0$. A wins.
- **RMSE.** A: $\sqrt{(1 + 784)/5} = \sqrt{157} = 12.53$. B: $\sqrt{(36 \times 5)/5} = 6.0$. B wins, by a factor of two.
- **MAPE.** A: $(0.1 + 0 + 0 + 0 + 28/48)/5 = 0.1367$. B: $(6/10 + 6/12 + 6/14 + 6/16 + 6/48)/5 = 0.4057$. A wins easily: B's misses are large *relative to the small targets*.
- **$R^2$.** $\bar{y} = 20$, and $\sum(y - \bar{y})^2 = 100 + 64 + 36 + 16 + 784 = 1000$. A: $1 - 785/1000 = 0.215$. B: $1 - 180/1000 = 0.82$.

Which model is better depends entirely on whether the 48 is a real, costly case you must get right (RMSE, $R^2$) or a fluke you would rather
not chase (MAE). **Best constants:** the mean 20 gives MAE 11.2 and RMSE 14.14; the median 14 gives MAE 8.4 but RMSE 15.36. Each wins its own metric.

```text
== worked example by hand ==
model A: errors [-1.0, 0.0, 0.0, 0.0, 28.0]   MAE 5.8000  RMSE 12.5300  MAPE 0.1367  R^2 0.2150
model B: errors [-6.0, -6.0, -6.0, -6.0, 6.0]   MAE 6.0000  RMSE 6.0000  MAPE 0.4057  R^2 0.8200
mean of y 20.0, median 14.0:   constant = mean: MAE 11.2000  RMSE 14.1421   constant = median: MAE 8.4000  RMSE 15.3623
```

## Key plot

![Average cost per row against the decision threshold for the Adult income model. With equal costs the curve is shallow with a minimum at 0.45, near the theoretical 0.5; when a false negative costs five times a false positive the minimum moves to 0.15, near the theoretical 0.167, and the default 0.5 costs 75% more](img/b4_2_cost_threshold.png)

The B4.1 logistic regression on Adult, scored on validation rows. With equal costs, the best threshold (0.45) is close to the formula's 0.5 and the
curve is flat around it. When a miss costs five times a false alarm, the best threshold drops to 0.15, near the formula's
0.167. Staying at the default 0.5 then costs 0.554 per row instead of 0.316: 75% more. The dotted lines are $c_{FP}/(c_{FP} + c_{FN})$.

## From scratch in NumPy

**The metrics.** Each is one line on the error vector (Step 1):

```python
def mae(y, y_hat):  return float(np.mean(np.abs(y - y_hat)))
def rmse(y, y_hat): return float(np.sqrt(np.mean((y - y_hat) ** 2)))
def mape(y, y_hat): return float(np.mean(np.abs(y - y_hat) / np.abs(y)))   # undefined if any y == 0
def r2(y, y_hat):   return float(1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))
```

**MAE and RMSE pick different winners.** California housing (B3.3's data and features), with two
`HistGradientBoostingRegressor` models that differ only in the loss they minimize:

```text
== from scratch: MAE and RMSE pick different winners ==
trained on squared_error : MAE 0.3215  RMSE 0.4856  MAPE 0.1784  R^2 0.8240   mean prediction 2.0709
trained on absolute_error: MAE 0.3168  RMSE 0.5019  MAPE 0.1659  R^2 0.8120   mean prediction 2.0172
validation target: mean 2.0788  median 1.8265
```

Each model wins the metric it was trained on: the squared-error model has the lower RMSE, the absolute-error model the lower MAE (and MAPE).
House prices are right-skewed (mean 2.08 against median 1.83, in $100,000s), and as Steps 2–3 predict, the absolute-error model predicts lower
on average (2.02 against 2.07): it aims at conditional medians. Neither is "more accurate". The question is which kind of error your use case pays for.

**MAPE and near-zero targets.** Synthetic daily unit sales of 2,000 slow-moving item-days, and the best possible forecast: the true expected
sales of each item-day.

```text
== from scratch: MAPE and near-zero targets ==
sales: 101 zero days, 269 days with 2 units or fewer, of 2000
perfect-expectation forecast, nonzero days only: MAE 3.3119  MAPE 0.2880
   actual 1  forecast 9.68  -> absolute percentage error 8.7
   actual 1  forecast 5.77  -> absolute percentage error 4.8
   actual 1  forecast 5.77  -> absolute percentage error 4.8
forecast x 0.5: MAE 10.7949  MAPE 0.4884
forecast x 0.8: MAE 5.0816  MAPE 0.2975
forecast x 1.0: MAE 3.3119  MAPE 0.2880
forecast x 1.2: MAE 5.0733  MAPE 0.4021
sklearn MAPE with zeros included (it divides by max(|y|, eps)): 2.220e+14
```

- Even the perfect forecast has a MAPE of 29%, and single rows contribute errors of 870%: an item expected to sell 9.7 that sold 1. These are
  ordinary Poisson fluctuations, not forecasting failures.
- Step 5 in numbers: forecasting 20% too high or 20% too low costs about the same MAE (5.07 against 5.08), but over-forecasting raises
  MAPE to 0.40 while under-forecasting raises it only to 0.30. A team rewarded on MAPE learns to under-order.
- With the 101 zero days included, MAPE is meaningless. scikit-learn divides by a tiny `eps` instead of zero and returns $2 \times 10^{14}$.

**A cost-weighted threshold.** Step 6, empirically: for 1,001 thresholds, count FP and FN on the validation rows and average the cost.

```python
flagged = s[None, :] >= thresholds[:, None]              # (T, n) one row per threshold
fp = np.sum(flagged & (y[None, :] == 0), axis=1)
fn = np.sum(~flagged & (y[None, :] == 1), axis=1)
cost = (c_fp * fp + c_fn * fn) / len(y)                  # (T,)
```

```text
== cost-weighted threshold ==
c_FP 1, c_FN 1: best threshold 0.453 (cost 0.1510/row)   theory c_FP/(c_FP+c_FN) = 0.500 (cost 0.1535)   threshold 0.5: 0.1535
c_FP 1, c_FN 5: best threshold 0.147 (cost 0.3158/row)   theory c_FP/(c_FP+c_FN) = 0.167 (cost 0.3195)   threshold 0.5: 0.5544
flag nobody: cost per row = c_FN x prevalence = 0.2393 x c_FN
```

The empirical optimum and the formula are close because logistic regression is well calibrated (B4.1). Their costs differ by 1–2%, which is
within the noise of one validation set. With a poorly calibrated model the formula would be off and only the empirical search would work. The
empirical threshold is itself tuned on validation rows, so its cost there is slightly optimistic; confirm it on the test set once.

## With scikit-learn

```text
== with scikit-learn ==
mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score match: OK
r2_score of predicting the validation mean: 0.0000;  of predicting the training mean: -0.0002
```

- `root_mean_squared_error` is a function of its own in scikit-learn 1.9.1; the older `mean_squared_error(squared=False)` was removed (VERIFY as of 2026-10).
- `mean_absolute_percentage_error` returns a fraction (0.17 = 17%), not a percentage, and does not raise on zeros: it divides by a tiny epsilon. Check for zeros yourself.
- `HistGradientBoostingRegressor(loss="absolute_error")` or `loss="quantile", quantile=0.9` trains directly for the metric you report.
  For P2's threshold, `sklearn.model_selection.TunedThresholdClassifierCV` searches thresholds inside cross-validation for a scorer you supply
  (VERIFY as of 2026-10); the from-scratch cost curve above is the same idea.

## In practice

- **When to use it:**
  - *RMSE* when large errors are disproportionately costly, and to match a model trained on squared error.
  - *MAE* when cost grows linearly with the error, or the data has outliers you do not want to chase. It is reported in the target's units, so stakeholders can read it.
  - *WAPE* rather than MAPE for a relative error on demand-like data; avoid MAPE whenever targets can be near zero.
  - *$R^2$* only next to an absolute metric, and only to compare models on the *same* rows.
  - For asymmetric costs (stock-outs cost more than overstock), train and evaluate with the quantile (pinball) loss at the quantile
    $c_{\text{under}}/(c_{\text{under}} + c_{\text{over}})$, the regression version of Step 6.
- **Failure modes:**
  - *Optimizing a metric nobody pays for.* Ask what one unit of error costs, and whether that cost is linear, quadratic or relative.
  - *MAPE with small targets.* Huge, unstable, biased towards under-forecasting.
  - *Comparing $R^2$ across datasets or time periods* with different target spread.
  - *Thresholds at 0.5 by habit.* Compute the cost curve; at a 5:1 cost ratio the default cost 75% more here.
  - *Using the formula threshold with an uncalibrated model.* Calibrate first (B4.1), or search empirically.
- **Defaults that work:** report MAE and RMSE together with a baseline (predict the training mean or median); state the metric's cost
  assumption in one sentence in the README or model card; choose thresholds from an explicit cost ratio on validation folds.

## Exercises

Save your work to `my-work/B/b4_2_regression_metrics.md` (or `.py`) and ask for a review.

1. **By hand:** Targets $y = (2, 4, 6, 8, 100)$. (a) Find the constant that minimizes MAE and the one that minimizes RMSE, and compute both metrics for both constants.
   (b) A forecast is $\hat{y} = (1, 5, 6, 9, 80)$. Compute MAE, RMSE, MAPE and WAPE. (c) A classifier's calibrated probability for a transaction is 0.08. Reviewing a
   transaction costs 4 and a missed fraud costs 60. Should it be reviewed? What is the threshold? (d) A forecast is 10. Compute the absolute percentage error when the actual
   value is 5 and when it is 15 (both are misses of 5 units). Which direction of miss does MAPE punish more, what is the largest error an under-forecast of a
   positive target can score, and what does that do to a model trained to minimize MAPE?
2. **Code:** On California housing (the lab's split), train `HistGradientBoostingRegressor` with `loss="quantile"` at quantiles 0.1, 0.5 and 0.9, and with
   `loss="squared_error"`. Implement the pinball loss from scratch, $\ell_\tau(y, \hat{y}) = \max(\tau(y - \hat{y}), (\tau - 1)(y - \hat{y}))$, and check it against `sklearn.metrics.mean_pinball_loss`. For each model report MAE, RMSE, the pinball loss at
   each of the three quantiles, and the share of validation targets below its prediction. Then report how often the true value falls between the 0.1 and 0.9 predictions.
3. **Think:** A grocery chain's forecasting team is rewarded on MAPE of daily item-level sales, computed on days with nonzero sales. Stores complain about empty
   shelves for slow-moving items. Explain the link, propose a metric (or metrics) that matches what a stock-out and an overstock actually cost, and say how you would
   change the model's training loss to match.

## Takeaways

1. MAE, RMSE and MAPE penalize errors linearly, quadratically and relatively; RMSE ≥ MAE, and a large ratio means a few large errors.
2. The best constant for squared error is the mean and for absolute error the median, so models trained on each estimate conditional means or medians, and they differ on skewed targets.
3. MAPE explodes near zero and rewards under-forecasting; use WAPE or an absolute metric instead.
4. $R^2$ compares with predicting the mean on the same rows: it can be negative, and it is not comparable across datasets.
5. With calibrated probabilities, flag a case when $p > c_{FP}/(c_{FP} + c_{FN})$; in practice, pick the threshold by minimizing the average cost on validation rows.

**What builds on this:** [B4.3](b4_3_cross_validation_tuning.md) (scoring inside cross-validation);
[B4.4](b4_4_class_imbalance.md) (thresholds for rare positives); [B6.3](b6_3_time_series_forecasting.md) and
[B6.4](b6_4_arima_vs_boosting.md) (forecast metrics); [J2.2](../J/j2_2_monitoring_drift.md) (tracking error metrics in production);
P2 (justifying the metric and threshold from costs).

**References:** R. J. Hyndman, A. B. Koehler, "Another look at measures of forecast accuracy", *International Journal of
Forecasting* 22, 2006. C. Elkan, "The foundations of cost-sensitive learning", *IJCAI*, 2001. T. Gneiting, "Making and evaluating point
forecasts", *Journal of the American Statistical Association* 106, 2011. scikit-learn User Guide, "Regression metrics"
(VERIFY link: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics).
