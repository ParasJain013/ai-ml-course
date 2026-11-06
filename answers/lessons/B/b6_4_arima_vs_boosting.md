# Answer key: B6.4 ARIMA and exponential smoothing versus gradient boosting on lag features

Lesson: [`lessons/B/b6_4_arima_vs_boosting.md`](../../../lessons/B/b6_4_arima_vs_boosting.md)

## Exercise 1 (by hand)

**Solution.**

**(a)** $\alpha = 0.2$, $\ell_0 = 100$.

| $t$ | forecast | actual | error | new level $0.2\,y_t + 0.8\,\ell_{t-1}$ |
|---|---|---|---|---|
| 1 | 100 | 100 | 0 | $20 + 80 = 100$ |
| 2 | 100 | 110 | +10 | $22 + 80 = 102$ |
| 3 | 102 | 90 | −12 | $18 + 81.6 = 99.6$ |
| 4 | 99.6 | 120 | +20.4 | $24 + 79.68 = 103.68$ |

SSE $= 0 + 100 + 144 + 416.16 = 660.16$. Next forecast: 103.68 (and the same for every later step).

**(b)** Weights $\alpha(1 - \alpha)^j$: $0.2$, $0.16$, $0.128$, $0.1024$ on $y_T, \dots, y_{T-3}$; they sum to $0.5904$, so $1 - 0.5904 = 0.4096 = 0.8^4$ is left on
everything older (including $\ell_0$). With $\alpha = 0.2$ the memory is long: the last four values carry only 59% of the weight.

**(c)** $\hat y_{T+1} = 200 + 3 + 10 = 213$; $\hat y_{T+2} = 200 + 6 - 5 = 201$; $\hat y_{T+3} = 200 + 9 - 5 = 204$. Damped, $h = 3$:
$200 + (0.9 + 0.81 + 0.729) \times 3 - 5 = 200 + 7.317 - 5 = 202.317$.

**(d)** Last change $y'_T = 54 - 50 = 4$. $\hat y'_{T+1} = 1 + 0.6 \times 4 = 3.4$, so $\hat y_{T+1} = 57.4$. $\hat y'_{T+2} = 1 + 0.6 \times 3.4 = 3.04$, so
$\hat y_{T+2} = 60.44$. (The changes tend to $c/(1 - \phi) = 2.5$: with a constant, the forecast keeps a slope of 2.5 per step.)

```text
== exercise 1 check ==
(a) t=1: forecast 100, error +0, level 100
(a) t=2: forecast 100, error +10, level 102
(a) t=3: forecast 102, error -12, level 99.6
(a) t=4: forecast 99.6, error +20.4, level 103.68
(a) SSE 660.16; next forecast 103.68
(b) weights 0.2, 0.16, 0.128, 0.1024; older 0.4096
(c) [213, 201, 204]; damped h=3: 202.317
(d) changes 3.4, 3.04 -> forecasts 57.4, 60.44
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Forecast at $t$ is the level *before* seeing $y_t$; all levels and the SSE | 4 |
| (b) Weights and the leftover $0.8^4$ | 3 |
| (c) Seasonal term matched to the horizon; damped sum of powers | 3 |
| (d) Model on the changes, with the constant; changes added back to the level | 4 |

**Likely mistakes:**

- Using the updated level as the forecast of the same $t$ (zero error by construction; the forecast must come *before* the value).
- In (c), applying the first seasonal term to every horizon, or multiplying the damped slope by $\phi^h$ only instead of summing $\phi + \dots + \phi^h$.
- In (d), applying $\phi$ to the level (54) instead of the change, which gives an AR(1) on levels, a different model.

## Exercise 2 (code)

**Solution:** [`b6_4_arima_vs_boosting_ex2.py`](b6_4_arima_vs_boosting_ex2.py).

```text
== (a) SARIMA order by AIC on fold 0, then by walk-forward MAE ==
(p,q)=(1,1) AIC 230.3
(p,q)=(2,2) AIC 230.8
(p,q)=(2,1) AIC 231.1
(p,q)=(1,2) AIC 231.6
(p,q)=(0,2) AIC 233.3
(p,q)=(0,1) AIC 235.3
(p,q)=(2,0) AIC 243.1
(p,q)=(1,0) AIC 243.7
(p,q)=(0,0) AIC 277.8
top three by AIC, walk-forward MAE per fold:
(p,q)=(1,1): 0.424 1.058 0.513 0.289 0.273  mean 0.511
(p,q)=(2,2): 0.422 1.061 0.490 0.296 0.265  mean 0.507
(p,q)=(2,1): 0.424 1.054 0.492 0.300 0.264  mean 0.507
the lab's (1,1): mean 0.511

== (b) Holt-Winters: damped versus undamped trend ==
damped=False: 0.282 1.193 0.429 0.213 0.223  mean 0.468
damped=True : 0.288 1.151 0.464 0.211 0.176  mean 0.458

== (c) the 2001 forecasts ==
Holt-Winters                       MAE 0.223
Holt-Winters damped                MAE 0.176
SARIMA(1,1,1)(0,1,1)12             MAE 0.273
LightGBM on lags                   MAE 1.109
LightGBM on 12-month differences   MAE 0.425
saved outputs/b6_4/ex2_forecasts_2001.png
```

**What a strong answer says:**

- AIC picks (1,1), the lab's order, but the top four are within 1.3 AIC points of each other: a difference that small does not separate models
  (a rule of thumb is that differences below about 2 are not meaningful). The walk-forward means of the top three are within 0.004 ppm, far below the
  fold-to-fold spread (0.26 to 1.06). Conclusion: the ranking does not really disagree; it is a tie. Any of the three is fine, and the simplest (1,1)
  is a reasonable choice. Models without an MA term are clearly worse by AIC (243 and above): the MA terms matter.
- Damping changes little on average (0.458 against 0.468) and the folds split 3–2. With a 12-month horizon and a steady trend, damping hardly bends the
  forecast; it matters at long horizons or after a trend change. No evidence here to prefer either.
- The plot shows LightGBM on the raw level flattening below the actual 2001 peak (it cannot exceed its training maximum), while the differenced
  version follows the rise.
- Model selection was done on fold 0's training data (AIC) and on walk-forward MAE; the strong answer does not pick the order with the lowest *test* MAE
  and then report that MAE as if it were unbiased.

**Rubric (15 pts):**

| Criterion | Points |
|---|---|
| Nine SARIMA fits with AIC on fold 0's training data only | 3 |
| Top three compared by walk-forward MAE on all folds | 3 |
| Correct reading: differences are within noise; AIC and walk-forward agree that it is a tie | 3 |
| Damped vs undamped per fold, with a cautious conclusion | 3 |
| Plot saved under `outputs/`, labelled, showing the extrapolation failure | 3 |

**Likely mistakes:**

- Fitting the AIC grid on the full series (including the test years), then evaluating walk-forward: the order was chosen with test data.
- Declaring (2,2) "better" because 0.507 < 0.511.
- Comparing AIC across models with different differencing ($d$ or $D$): the likelihoods are of different series and are not comparable. Here $d = D = 1$
  for all nine, so the comparison is valid.

## Exercise 3 (think)

**Strong answer:** Use both, with a global LightGBM as the main model and per-series ETS (or seasonal naive) as a baseline and fallback. *Why global:*
20,000 short, noisy series share patterns (seasonality, price and promotion response, holiday effects) that one model can learn across series, and the
known 8-week price and promotion plan is exactly the kind of non-linear driver boosting uses well; fitting and monitoring 20,000 ARIMA models is also
hard to operate. *Setup:* one row per (series, week); target sales in week $t$; features with lags of at least 8 weeks (lag 8, 9, 12, 52, rolling means
ending 8 weeks ago), calendar (week of year, holidays), the planned price and discount and promotion type for week $t$, price relative to the product's usual
price, product and store attributes (category, size, region) and the series id or target-encoded statistics computed on the training period only (B1.3).
*Scale:* series differ by orders of magnitude, so either predict a scaled target (sales divided by the series' recent mean, computed from data at least 8
weeks old) or use a loss that suits counts (Tweedie or Poisson objective in LightGBM) and weight the series by revenue if that is what matters.
*Direct multi-horizon:* one model per horizon, or one model with the horizon as a feature and lags $\ge$ horizon. *New products:* no lags exist; the global
model can still predict from attributes, price and the sales of similar products (category averages), which per-series models cannot; switch on lag
features as history accumulates. *Slow sellers:* many zeros make MAE favour predicting 0; use a Poisson/Tweedie objective, evaluate with a scaled error or
the replenishment cost, and consider an intermittent-demand method such as Croston's as a baseline. *Decide by* a walk-forward backtest over the last year
with weekly origins and the real 8-week horizon, comparing the global model, per-series ETS and seasonal naive on a scale-free metric (MASE or weighted
error) and on the business outcome (stock-outs and overstock simulated from the replenishment rule), broken down by segment (fast, slow, new).
**Weak answer:** "ARIMA for each product, because it is the standard forecasting model."
**Why weak:** ignores the known drivers, the operational cost of 20,000 models, new products without history and intermittent zeros; no evaluation plan.

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| A justified choice (global model plus classical baseline/fallback) | 2 |
| Global model set-up: target scaling or count loss, lags $\ge$ 8 weeks, drivers known in advance | 3 |
| New-product strategy | 2 |
| Slow sellers / intermittent demand handled | 2 |
| Walk-forward backtest at the real horizon with a scale-free metric and business outcome, by segment | 3 |

## Scoring

Total 41 pts. Pass ≥ 70% (29). Topic ids to revisit for each lost criterion: SES, Holt–Winters, ARIMA (Ex 1, Ex 2) → B6.4; lags, horizons and walk-forward
(Ex 2, Ex 3) → B6.3; target encoding without leakage (Ex 3) → B1.3; information criteria (Ex 2a) → B5.3; boosting (Ex 3) → B3.3.
