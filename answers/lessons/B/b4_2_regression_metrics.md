# Answer key: B4.2 Regression metrics, and choosing a metric that matches the real cost of errors

Lesson: [`lessons/B/b4_2_regression_metrics.md`](../../../lessons/B/b4_2_regression_metrics.md)

## Exercise 1 (by hand)

**Solution (a).** $y = (2, 4, 6, 8, 100)$. The MAE-optimal constant is the median, 6; the RMSE-optimal constant is the mean, $120/5 = 24$.
- Constant 6: absolute errors $4, 2, 0, 2, 94$: MAE $= 102/5 = 20.4$; squared $16, 4, 0, 4, 8836$: RMSE $= \sqrt{8860/5} = \sqrt{1772} = 42.10$.
- Constant 24: absolute errors $22, 20, 18, 16, 76$: MAE $= 152/5 = 30.4$; squared $484, 400, 324, 256, 5776$: RMSE $= \sqrt{7240/5} = \sqrt{1448} = 38.05$.

Each constant wins its own metric. The outlier drags the mean to 24, a value no typical row is near.

**(b)** $\hat{y} = (1, 5, 6, 9, 80)$, errors $1, -1, 0, -1, 20$. MAE $= 23/5 = 4.6$. RMSE $= \sqrt{(1 + 1 + 0 + 1 + 400)/5} = \sqrt{80.6} = 8.978$. MAPE $= (1/2 + 1/4 + 0 + 1/8 + 20/100)/5 = 1.075/5 = 0.215$.
WAPE $= 23/120 = 0.1917$. MAPE is above WAPE because the small targets (2, 4) carry the largest relative errors.

**(c)** $t^* = 4/(4 + 60) = 0.0625$. $0.08 > 0.0625$, so review it. Check: reviewing costs $(1 - 0.08) \times 4 = 3.68$ in expectation; not reviewing costs $0.08 \times 60 = 4.80$.

**(d)** Forecast 10. Actual 5: $|5 - 10|/5 = 1.0$. Actual 15: $|15 - 10|/15 = 0.333$. The same 5-unit miss costs three times as much when the forecast was too high.
An under-forecast of a positive target scores at most 1.0 (forecast 0); an over-forecast has no upper limit. A model trained on MAPE is
therefore pushed to forecast low.

```text
== exercise 1 check ==
(a) constant = median 6: MAE 20.4000  RMSE 42.0951
(a) constant = mean 24: MAE 30.4000  RMSE 38.0526
(b) MAE 4.6000  RMSE 8.9778  MAPE 0.2150  WAPE 0.1917
(c) threshold 0.0625; expected cost if reviewed 3.68, if not 4.80
(d) forecast 10: actual 5 -> APE 1.0000; actual 15 -> APE 0.3333; under-forecast of 0 for any positive actual -> APE 1.0
```

**Rubric (13 pts):**

| Criterion | Points |
|---|---|
| (a) Median and mean as the optimal constants, with all four metric values | 4 |
| (b) MAE, RMSE, MAPE, WAPE correct | 3 |
| (c) $t^* = 0.0625$, decision "review", expected costs shown | 3 |
| (d) 1.0 vs 0.333, bound of 1.0 for under-forecasts, and the downward bias | 3 |

**Likely mistakes:**

- Saying the mean minimizes MAE. It minimizes squared error; the median minimizes absolute error.
- Forgetting the square root in RMSE, or averaging the square roots of the squared errors (which gives MAE).
- Dividing by the forecast instead of the actual in MAPE.
- In (c), comparing 0.08 with 0.5, or writing the threshold as $c_{FN}/(c_{FP} + c_{FN})$.

## Exercise 2 (code)

**Solution:** [`b4_2_regression_metrics_ex2.py`](b4_2_regression_metrics_ex2.py). The pinball loss charges $\tau|e|$ when the target is above
the prediction and $(1 - \tau)|e|$ when it is below, so with $\tau = 0.9$ predicting too low is nine times as costly as predicting too high, and the
best prediction is the 90th percentile:

```python
def pinball(y, y_hat, tau):
    e = y - y_hat
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))
```

```text
== data ==
train 12384  val 4128

== pinball loss check ==
toy: y [1.0, 2.0, 3.0], prediction 2 -> pinball at 0.9 = 0.3333 (0.9 x 1 above, 0.1 x 1 below, averaged over 3 rows)
matches sklearn.metrics.mean_pinball_loss: OK

== four models ==
model               MAE    RMSE  pin@0.1  pin@0.5  pin@0.9  y < pred
squared_error    0.3215  0.4856   0.1576   0.1608   0.1639    0.5564
quantile 0.1     0.4841  0.7421   0.0645   0.2421   0.4196    0.1519
quantile 0.5     0.3168  0.5019   0.1338   0.1584   0.1830    0.5044
quantile 0.9     0.5479  0.7246   0.4503   0.2739   0.0975    0.8505
share of validation targets between the 0.1 and 0.9 predictions: 0.6999 (nominal 0.80)
rows where the 0.1 prediction is above the 0.9 prediction (quantile crossing): 0.0012
```

What a strong answer reads from the table:

- Each model is best on its own loss: the squared-error model on RMSE, the 0.5-quantile model on MAE and pinball@0.5 (pinball at 0.5 is half the MAE),
  and the 0.1 and 0.9 models on their pinball losses. The quantile models are bad point forecasts (high MAE and RMSE); that is not their job.
- The share of targets below each prediction lands near its quantile: 0.15 for 0.1, 0.50 for 0.5, 0.85 for 0.9. The squared-error model sits at 0.56:
  it predicts conditional means, which lie above the medians of a right-skewed target.
- The 0.1–0.9 band covers 70% of validation targets, not the nominal 80%. Quantile models trained on the same rows they are judged on overfit, so
  their intervals are too narrow on new data. Calibrate them on held-out rows (conformal prediction is the standard tool) before promising coverage.
- Quantile crossing (0.1 above 0.9) happens on 0.12% of rows because the models are trained separately; sort the predictions or use a model
  that fits all quantiles jointly if it matters.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Pinball loss from scratch, matching `mean_pinball_loss` (an assert) | 3 |
| Four models trained on the lab's split with the right losses | 3 |
| Table with MAE, RMSE, three pinball losses and the "below prediction" share | 3 |
| Coverage of the 0.1–0.9 band computed and compared with 80% | 3 |
| Interpretation: each model wins its own loss; mean vs median; undercoverage and why | 4 |

**Likely mistakes:**

- Writing the pinball loss with the two slopes swapped, which makes the 0.9 model's loss lowest for the 0.1 model.
- Reporting coverage on the training rows (it will look close to 80%).
- Judging the quantile models by RMSE and calling them "worse".
- Using `loss="absolute_error"` and `quantile=0.5` and expecting different models: they are the same loss.

## Exercise 3 (think)

**Strong answer:** MAPE on nonzero days rewards under-forecasting (Step 5): over-forecasts of slow movers are punished hardest, and zero-sales days, often caused by an
empty shelf, are excluded entirely, so a stock-out *improves* the metric by removing a day. The model therefore learns to forecast low for slow movers, and
the stores run out. The real costs are asymmetric and in money: a stock-out loses the margin on unmet demand (plus some customer goodwill), and overstock costs
holding, waste for perishables, and markdowns. A matching setup: (1) forecast a quantile, not the mean. Order up to the quantile
$\tau = c_{\text{under}}/(c_{\text{under}} + c_{\text{over}})$ (the newsvendor result), for example $\tau \approx 0.8$ when a lost sale costs four times an overstocked unit. Train with the pinball loss at
that $\tau$ and evaluate with it. (2) Report WAPE (or MAE) per item group for accuracy, a bias metric (total forecast minus total sales) and a service level: the share of item-days without a stock-out.
(3) Include zero days, and treat sales on stock-out days as censored demand (true demand was at least what sold), not as zero demand.
**Weak answer:** "MAPE is a standard metric; the shelves are empty because of supply problems. Maybe switch to RMSE."
**Why weak:** it misses MAPE's downward bias and the exclusion of zero days, and RMSE is still symmetric, so it does not encode the cost asymmetry either.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| MAPE's bias towards under-forecasting, and the effect of excluding zero days | 3 |
| Identifies the costs as asymmetric and monetary (stock-out vs overstock) | 2 |
| Proposes a quantile/pinball target with $\tau$ from the cost ratio | 3 |
| Supporting metrics (WAPE, bias, service level) or censored demand | 2 |

## Scoring

Total 39 pts. Pass ≥ 70% (28). Topic ids to revisit for each lost criterion: metric definitions and optimal constants (Ex 1a, 1b) → B4.2;
cost-based thresholds (Ex 1c) → B4.2; MAPE asymmetry and quantile loss (Ex 1d, Ex 2, Ex 3) → B4.2; calibration of intervals → B4.1.
