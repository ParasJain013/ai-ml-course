# Answer key: B6.3 Time-series forecasting: trend, seasonality, lag features, naive baselines, walk-forward validation

Lesson: [`lessons/B/b6_3_time_series_forecasting.md`](../../../lessons/B/b6_3_time_series_forecasting.md)

## Exercise 1 (by hand)

**Solution.** Series (months 1–8): $20, 30, 25, 45, 24, 34, 29, 49$; $m = 4$, so the 2×4 weights are $\tfrac18, \tfrac14, \tfrac14, \tfrac14, \tfrac18$.

**(a)** Month 3 uses months 1–5: $(\tfrac12 \cdot 20 + 30 + 25 + 45 + \tfrac12 \cdot 24)/4 = (10 + 30 + 25 + 45 + 12)/4 = 122/4 = 30.5$; detrended $25 - 30.5 = -5.5$.
Month 4 uses months 2–6: $(15 + 25 + 45 + 24 + 17)/4 = 126/4 = 31.5$; detrended $45 - 31.5 = +13.5$. Month 4 of the cycle is the peak.

**(b)** Truth $28, 38, 33, 53$.
- Naive (49 for all): errors $21, 11, 16, 4$; MAE $52/4 = 13$.
- Seasonal naive (months 5–8: $24, 34, 29, 49$): errors $4, 4, 4, 4$; MAE $4$.
- 4-month moving average ($(24 + 34 + 29 + 49)/4 = 34$ for all): errors $6, 4, 1, 19$; MAE $30/4 = 7.5$.
Seasonal naive wins and is exactly one cycle's growth (4) too low: a lag model with growth would do better.

**(c)** Horizon 2: at origin $T$, the newest known value is $y_T$, which is 2 months older than the target $y_{T+2}$, so every input must be at
least 2 months old.
- $y_{t-1}$: **no** (1 month old). $y_{t-2}$: **yes**. $y_{t-4}$: **yes**. Mean of $y_{t-2}, \dots, y_{t-5}$: **yes** (newest is 2 months old).
- Mean of $y_{t-1}, \dots, y_{t+2}$: **no**: it contains the target itself and two future months; it is a centred smoother.
- Month of $t$: **yes**: the calendar is known in advance.

**(d)** `s.shift(2).rolling(3).mean()` at row $t$ is the mean of $y_{t-2}, y_{t-3}, y_{t-4}$: **safe** for horizon 2. `s.rolling(3).mean().shift(1)` is the
mean of $y_{t-1}, y_{t-2}, y_{t-3}$: it includes $y_{t-1}$, so it is safe only for horizon 1. The check uses a series whose values equal their month
numbers, so the output shows which months were averaged:

```text
== exercise 1 check ==
(a) month 3: trend 30.50, detrended -5.50
(a) month 4: trend 31.50, detrended +13.50
(b) naive              [49.0, 49.0, 49.0, 49.0] MAE 13.00
(b) seasonal naive     [24.0, 34.0, 29.0, 49.0] MAE 4.00
(b) moving average (4) [34.0, 34.0, 34.0, 34.0] MAE 7.50
(d) row 9 of s.shift(2).rolling(3).mean(): 7.0 (mean of months 6, 7, 8)
    row 9 of s.rolling(3).mean().shift(1): 8.0 (mean of months 7, 8, 9: month 9 is t-1)
```

(Python's row 9 is month 10, so "months 7, 8, 9" are $t - 3, t - 2, t - 1$.)

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) 2×4 weights with halves at both ends; both trend values and detrended values | 3 |
| (b) All three forecasts and MAEs | 3 |
| (c) All six features classified correctly, with the reason "at least $h$ old" | 4 |
| (d) Correct answer with the months each expression averages | 4 |

**Likely mistakes:**

- Using a plain 4-month average (no half weights): it is not centred on a month, so the trend is shifted by half a month.
- Seasonal naive taken from the wrong year (months 1–4 instead of 5–8).
- Marking "month of $t$" as leakage: calendar features are known for any future date.
- In (d), assuming `.shift(1)` after rolling makes everything "lagged enough": the shift has to be at least the horizon.

## Exercise 2 (code)

**Solution:** [`b6_3_time_series_forecasting_ex2.py`](b6_3_time_series_forecasting_ex2.py).

```text
== (a) direct versus recursive 12-month forecasts ==
method               1997   1998   1999   2000   2001    mean
seasonal naive      1.109  2.914  1.611  1.136  1.493   1.652
direct (lab)        0.680  1.196  0.650  0.819  0.454   0.760
recursive 1-step    0.599  0.814  0.620  0.567  0.440   0.608
recursive MAE by horizon h=1..12: 0.21 0.28 0.39 0.38 0.52 0.70 0.79 0.88 1.02 0.85 0.69 0.57
direct    MAE by horizon h=1..12: 0.74 0.77 0.85 0.73 0.50 0.73 0.87 0.85 0.93 0.86 0.70 0.59

== (b) MASE against in-sample seasonal naive ==
seasonal naive     MASE per fold 0.885 2.333 1.248 0.875 1.153  mean 1.299
direct (lab)       MASE per fold 0.543 0.958 0.504 0.631 0.350  mean 0.597
recursive 1-step   MASE per fold 0.478 0.652 0.480 0.436 0.340  mean 0.477

== (c) the leakage test on the one-step features ==
min_age=1: OK (every feature is at least 1 month old)
min_age=12 fails: features of 1995-12 use data newer than 12 months: ['lag_1', 'lag_2']
```

**What a strong answer says:**

- On this series the recursive forecast is better on average (0.61 against 0.76) and in every year. The by-horizon rows show why: the one-step
  model uses last month's value, so its first months are very accurate (0.21 at $h = 1$), and its error grows with $h$ as forecasts are fed back in
  (1.02 at $h = 9$). The direct model cannot use anything newer than 12 months, so its error is flat (about 0.5–0.9) at every horizon. Recursive
  wins here because CO2 is smooth and almost linear, so the compounding of errors is mild. On noisier series, or with a non-linear model, errors
  compound faster and direct (or one model per horizon) often wins. Neither strategy is better in general; test both under walk-forward.
- Feeding the forecast back is not leakage: at each step only the history up to the origin plus the model's own forecasts are used. Using the
  *actual* value of month $h - 1$ to forecast month $h$ would be leakage for a 12-month forecast (that is a one-step-ahead evaluation, a different question).
- MASE: seasonal naive scores 1.30 on average out of sample. It is above 1 because its out-of-sample error is larger than its in-sample error
  (the in-sample scale averages the growth of 1959–1996, about 1.2 ppm a year, while the test years grew faster, 1.65 ppm a year on average, with 2.85 in 1998). Both models are well below 1 (0.60 and 0.48). MASE makes the results
  comparable to other series in other units.
- The leakage test does exactly its job: the one-step features are fine for a 1-month horizon and illegal for a 12-month one, and the test names the
  offending columns. A feature set is not "leaky" or "safe" in itself, only relative to a horizon.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Recursive loop correct: the model is fitted once per fold; each step's features are computed from history + earlier forecasts only | 5 |
| Per-fold MAEs for both strategies, same folds as the lab | 2 |
| MASE with the in-sample seasonal-naive scale from each fold's training data | 3 |
| Leakage test run twice and both results explained relative to the horizon | 3 |
| Interpretation: error by horizon, when each strategy wins, and why the result is series-specific | 3 |

**Likely mistakes:**

- Using the true $y_{t-1}$ at every step of the 12-month forecast ("recursive" with actuals): the MAE drops to the one-step level and is fiction.
- Building month dummies with `pd.get_dummies` on a single row (it creates only one column, so the design matrix has the wrong width); build all 11
  columns explicitly, as the solution does.
- Scaling MASE by the *test* data's seasonal-naive error, or by the whole series: the scale must come from the training months.
- Concluding "recursive is always better": the result depends on the series.

## Exercise 3 (think)

**Strong answer:** *Baselines first:* seasonal naive with $m = 7$ (same weekday last week), the mean of the same weekday over the last 4 weeks, and the
same day last year aligned by weekday (for yearly patterns), each per warehouse. *Features and minimum ages* (horizon 14 days): lags of orders at
14, 21, 28 and 364 days and rolling means ending 14 days ago (at least 14 days old); calendar (weekday, month, holiday flags, days to the next holiday:
known in advance, any age); the promotion *calendar* (known 14 days ahead by construction; use only what was planned at the origin, not promotions
added later); weather: use the *forecast* available at the origin, not the observed weather, or leave it out if no archived forecasts exist (training
on observed weather and serving with forecasts creates a train-serve mismatch). *Validation:* walk-forward over the last 6–12 months with weekly
origins, horizon 14 days, refitting each time; a gap if the most recent day's orders are finalized late; report the error separately by horizon day
and by warehouse. *Metric:* MAE in orders per day (operations thinks in orders and staff), plus MASE against seasonal naive; possibly a quantile
forecast (for example the 90th percentile) since under-staffing costs more than over-staffing (B4.2). *Ways the history misleads:* stock-outs and
capacity caps **censor** demand (orders recorded on a sold-out day are less than the demand); closures and outages show zeros that are not low
demand; promotions inflate days that will not repeat; a new warehouse has too little history (use the other warehouses, scaled, or a pooled model
with a warehouse feature); and changes such as a new delivery area are structural breaks. Flag or remove such days, or add features for them.
**Weak answer:** "Use all the features, train XGBoost with a random 80/20 split and report R²."
**Why weak:** no baseline, features of unknown age (observed weather and promotions leak), a random split, a metric operations cannot use, and no
awareness of censored demand.

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| Seasonal-naive (weekly) baseline and at least one other | 2 |
| Every feature with a correct minimum age; promotions and weather handled as known-at-origin | 3 |
| Walk-forward with a 14-day horizon and refits; error by horizon | 2 |
| Metric in operational units, with a skill score or asymmetric cost discussion | 2 |
| Two concrete ways the data misleads (censoring, closures, promotions, breaks, new sites) | 3 |

## Scoring

Total 42 pts. Pass ≥ 70% (30). Topic ids to revisit for each lost criterion: decomposition and baselines (Ex 1a–b) → B6.3; feature ages and
leakage (Ex 1c–d, Ex 2c, Ex 3) → B6.3 and B1.4; walk-forward (Ex 2, Ex 3) → B4.3; metrics and asymmetric costs (Ex 2b, Ex 3) → B4.2.
