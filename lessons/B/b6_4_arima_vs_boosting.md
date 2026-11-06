# B6.4 ARIMA and exponential smoothing versus gradient boosting on lag features

**Depth** · Part B, week 6 (course week 12) · ~1.5 h · Lab:
[`labs/B/b6_4_arima_vs_boosting.py`](../../labs/B/b6_4_arima_vs_boosting.py)

## Why it matters

When a team needs a forecast, two camps form. One reaches for the classical statistical models, **exponential smoothing** and **ARIMA**,
which have been the workhorses of forecasting for decades and fit a single series in a second. The other turns the series into a table
of lag features (B6.3) and trains gradient boosting (B3.3), which wins many forecasting competitions with many related series. Both camps
are right, about different problems. An interviewer will ask "would you use ARIMA or XGBoost for this?" and wants the conditions, not a
favourite. This lesson builds simple exponential smoothing from scratch, fits Holt–Winters and seasonal ARIMA with statsmodels, and races
them against LightGBM under B6.3's walk-forward folds in three settings, so you can say from your own numbers when each one wins.

## Prerequisite check

- Can you build lag features that are at least as old as the horizon, and validate walk-forward? If not, revisit
  [B6.3](b6_3_time_series_forecasting.md).
- Can you say what a regression tree predicts in a leaf, and how gradient boosting adds trees? If not, revisit
  [B3.3](b3_3_gradient_boosting.md) (and B3.1 for the leaf).
- Can you sum a geometric series, $1 + r + r^2 + \dots = 1/(1 - r)$ for $\lvert r\rvert < 1$? If not, the step is shown below; for practice with
  sums and limits revisit [A4.1](../A/a4_1_derivatives.md).
- Do you know what maximum likelihood estimation does? If not, revisit [A5.5](../A/a5_5_mle_map.md).

## Intuition

**Exponential smoothing** forecasts with a weighted average of the past in which recent values count most and older values count
geometrically less. One number, $\alpha$, sets how fast the memory fades. A small $\alpha$ means a long memory: the forecast moves slowly and
ignores noise. A large $\alpha$ means it chases the latest value. **Holt–Winters** keeps three such running averages at once: one for the
level, one for the slope (trend) and one for each position in the season, each updated a little at every step. It is like keeping an
exponentially weighted moving average of your API latency (an EWMA in a monitoring dashboard), plus one for its daily slope, plus one per
hour of the day.

**ARIMA** describes how a series moves from one step to the next. First **difference** the series until it has no trend
($y_t - y_{t-1}$: the change, not the level). Then model the change as a linear function of recent changes (**autoregressive**, AR) and of
recent surprises, the errors the model made in the last steps (**moving average**, MA). A seasonal ARIMA does the same at the seasonal lag
(this month versus the same month last year).

Both are **local** models: one model per series, a handful of parameters, fitted to that series' own history. They extrapolate trends
naturally (a slope is a slope) and need little data. They cannot easily use other information, such as promotions or weather, and they
assume the series' structure is roughly linear.

**Gradient boosting on lag features** is a **global** model: a big non-linear function of whatever features you give it. It can learn that
demand jumps on promotion days, or explodes above 24 °C, and it can be trained on thousands of related series at once (every store, every
product), sharing patterns between them. But a tree ensemble predicts by averaging training targets in its leaves, so it **cannot predict a
value outside the range of targets it has seen**. A rising series breaks it unless you model the *change* instead of the level.

> **Bridge:** exponential smoothing is the EWMA you already use for latency dashboards and load-balancer weights: `avg = a * sample + (1 - a)
> * avg`. **Where it breaks:** in monitoring you pick `a` by feel; here $\alpha$ is *fitted*, by minimizing past one-step forecast errors, and a
> plain EWMA has no notion of trend or season, so it always lags a rising series. Holt–Winters adds those.

## The math

**Symbols** (in this lesson only, $\alpha$, $\beta$, $\gamma$ are smoothing weights, not a significance level, and $\phi$, $\theta$ are ARIMA
coefficients)

| Symbol | Meaning | Shape |
|---|---|---|
| $y_t$ | the series at time $t = 1, \dots, n$ | scalar |
| $\ell_t$, $b_t$, $s_t$ | level, trend (slope per step) and seasonal component after seeing $y_t$ | scalars |
| $\alpha, \beta, \gamma$ | smoothing weights for level, trend, season, each in $[0, 1]$ | scalars |
| $m$, $h$, $T$ | season length, horizon, forecast origin (as in B6.3) | ints |
| $\hat y_{T+h \mid T}$ | forecast of $y_{T+h}$ made at $T$ | scalar |
| $y'_t$ | the differenced series, $y_t - y_{t-1}$ | scalar |
| $\varepsilon_t$ | the one-step forecast error ("innovation") at $t$ | scalar |
| $\phi_j$, $\theta_j$ | AR and MA coefficients | scalars |

**Step 1: simple exponential smoothing (SES).** Start with $\ell_0 = y_1$. After each new value, update the level:
$$\ell_t = \alpha\,y_t + (1 - \alpha)\,\ell_{t-1}, \qquad \hat y_{T+h \mid T} = \ell_T \text{ for every } h.$$
*Why "exponential".* Substitute the update into itself: $\ell_T = \alpha y_T + (1-\alpha)\ell_{T-1} = \alpha y_T + (1-\alpha)[\alpha y_{T-1} + (1-\alpha)\ell_{T-2}]$,
and so on:
$$\ell_T = \alpha y_T + \alpha(1-\alpha)\,y_{T-1} + \alpha(1-\alpha)^2\,y_{T-2} + \dots + (1-\alpha)^T\ell_0 .$$
The weights $\alpha, \alpha(1-\alpha), \alpha(1-\alpha)^2, \dots$ shrink by a factor $(1 - \alpha)$ per step back. With $\alpha = 0.5$ they are $0.5, 0.25, 0.125,
\dots$. They add up to 1 (with $r = 1 - \alpha$: $\alpha(1 + r + r^2 + \dots) = \alpha/(1 - r) = \alpha/\alpha = 1$), so the level is a proper weighted
average. $\alpha = 1$ gives the naive forecast (all weight on the last value); $\alpha \to 0$ gives the mean of a long history.

*Fitting $\alpha$.* The one-step forecast of $y_t$ is $\ell_{t-1}$, so the one-step error is $\varepsilon_t = y_t - \ell_{t-1}$. Choose the
$\alpha \in [0, 1]$ that minimizes $\sum_t \varepsilon_t^2$. There is one parameter, so a 1-D search (`scipy.optimize.minimize_scalar`) is enough.

**Step 2: Holt–Winters (additive).** Add a slope $b_t$ and seasonal terms $s_t$, each with its own smoothing weight:
$$\begin{aligned}
\ell_t &= \alpha\,(y_t - s_{t-m}) + (1 - \alpha)(\ell_{t-1} + b_{t-1}) && \text{level: the deseasonalized value, blended with the previous level moved one step along the slope}\\
b_t &= \beta\,(\ell_t - \ell_{t-1}) + (1 - \beta)\,b_{t-1} && \text{slope: the latest change in level, blended with the previous slope}\\
s_t &= \gamma\,(y_t - \ell_t) + (1 - \gamma)\,s_{t-m} && \text{season: what is left after the level, blended with the same season last cycle}
\end{aligned}$$
$$\hat y_{T+h \mid T} = \ell_T + h\,b_T + s_{T+h-m} \quad (h \le m).$$
Each line is SES applied to one component. The forecast continues the level along the slope and adds the matching seasonal effect. A
**damped trend** multiplies the slope by $\phi^h$ ($0 < \phi < 1$) so that long-horizon forecasts flatten out instead of rising forever. The
parameters $\alpha, \beta, \gamma$ and the starting values are fitted by minimizing the one-step errors (statsmodels does this).

**Step 3: ARIMA($p, d, q$).** The three numbers are three operations.

- **I, integrated, $d$:** difference $d$ times. $d = 1$: model $y'_t = y_t - y_{t-1}$. A linear trend becomes a constant after one difference, so the
  differenced series has a stable mean: it is (roughly) **stationary**, meaning its mean, variance and correlations do not drift over time. ARIMA's
  AR and MA parts assume stationarity.
- **AR, autoregressive, $p$:** the (differenced) series depends linearly on its own last $p$ values:
  $y'_t = c + \phi_1 y'_{t-1} + \dots + \phi_p y'_{t-p} + \varepsilon_t$.
- **MA, moving average, $q$:** it also depends on the last $q$ errors: $\dots + \theta_1\varepsilon_{t-1} + \dots + \theta_q\varepsilon_{t-q}$. (This
  "moving average" is not B6.3's rolling mean; the name is historical.) A negative $\theta_1$ means "after a positive surprise, expect part of it to
  reverse".

The **seasonal** version, SARIMA($p, d, q$)($P, D, Q$)$_m$, adds the same three operations at lag $m$: $D = 1$ means also take
$y_t - y_{t-m}$ (this month minus the same month last year), and $P$, $Q$ are AR and MA terms at lags $m, 2m, \dots$. The lab uses
SARIMA(1,1,1)(0,1,1)$_{12}$: difference once and seasonally once, then one AR term, one MA term and one seasonal MA term. The coefficients are
fitted by **maximum likelihood** (A5.5) assuming Gaussian errors; statsmodels writes the model in state-space form and computes the likelihood
with a Kalman filter, a recursion that updates a running estimate one step at a time (much like the smoothing updates above; you do not need its details here). To forecast, set future errors to 0 (their expected value), compute the differenced series forward, and undo the
differencing by adding the changes back up.

Choosing $(p, d, q)$: $d$ from whether the series has a trend (differencing until it does not); $p$ and $q$ by comparing information criteria
such as AIC or BIC (B5.3 used BIC) over a small grid, or simply by comparing walk-forward errors of a few candidates, which is what you should
trust most.

**Step 4: why trees cannot extrapolate.** A regression tree predicts, in each leaf, the mean of the training targets in that leaf (B3.1). A
boosted ensemble adds up such leaf values, and each tree was fitted to residuals of training targets, so its prediction stays within roughly
the range of targets seen in training. If CO2 has never been above 371.65 ppm in the training data, no tree ensemble trained on the level will
forecast 373. The standard fix is to change the target to something whose range does not drift: the change $y_t - y_{t-12}$ (here "this month
minus the same month last year") or a ratio. Forecast the change, then add it back: $\hat y_t = y_{t-12} + \widehat{(y_t - y_{t-12})}$.

## Worked example by hand

**SES** with $\alpha = 0.5$ on $y = 10, 12, 11, 15$, starting from $\ell_0 = 10$:

| $t$ | forecast $\ell_{t-1}$ | actual $y_t$ | error | new level $0.5\,y_t + 0.5\,\ell_{t-1}$ |
|---|---|---|---|---|
| 1 | 10 | 10 | 0 | $5 + 5 = 10$ |
| 2 | 10 | 12 | +2 | $6 + 5 = 11$ |
| 3 | 11 | 11 | 0 | $5.5 + 5.5 = 11$ |
| 4 | 11 | 15 | +4 | $7.5 + 5.5 = 13$ |

Sum of squared one-step errors: $0 + 4 + 0 + 16 = 20$. The forecast for $t = 5, 6, \dots$ is 13, flat: SES has no slope, so it does not continue the
rise.

**ARIMA(1,1,0)** with $\phi = 0.5$ and no constant, on the same series. Differences: $12 - 10 = 2$, $11 - 12 = -1$, $15 - 11 = 4$. The model says the
next change is half the last one: $\hat y'_5 = 0.5 \times 4 = 2$, so $\hat y_5 = 15 + 2 = 17$. Then $\hat y'_6 = 0.5 \times 2 = 1$ (the forecast change
feeds the next step), so $\hat y_6 = 17 + 1 = 18$. The changes decay towards 0 and the forecast levels off; with a constant $c$ it would keep a slope.

```text
== worked example by hand ==
t=1: forecast 10, actual 10, error +0, new level 0.5*10 + 0.5*10 = 10
t=2: forecast 10, actual 12, error +2, new level 0.5*12 + 0.5*10 = 11
t=3: forecast 11, actual 11, error +0, new level 0.5*11 + 0.5*11 = 11
t=4: forecast 11, actual 15, error +4, new level 0.5*15 + 0.5*11 = 13
SSE of the one-step errors: 20; forecast for t=5, 6, ...: 13 (flat)
differences [2.0, -1.0, 4.0]; ARIMA(1,1,0) with phi=0.5: d5 = 2 -> y5 = 17; d6 = 1 -> y6 = 18
```

## Key plot

![Two bar charts of skill against seasonal naive. Left, CO2 12 months ahead: Holt-Winters and SARIMA have the highest skill, about 0.6 to 0.7, on both the long and the short history; LightGBM on raw lags is lowest after SES. Right, daily demand with known drivers: LightGBM with promotion and temperature features has the highest skill, 0.61, ahead of SARIMAX with the same drivers, 0.48, and Holt-Winters and SARIMA without them, about 0.2](img/b6_4_who_wins.png)

On a single clean, trending series (left) the classical models win, and their lead is largest on the short history. When the future depends on
known drivers with a non-linear effect (right), boosting with those features wins, and the classical models without them barely beat the
baseline.

## From scratch in NumPy

**The data.** Three settings, each with the walk-forward splitter of B6.3:

```text
== data ==
CO2 long: 526 months 1958-03..2001-12; short: 120 months from 1992-01
synthetic daily demand: 1095 days 2022-01-01..2024-12-30; mean 334.0, promo days 137, days above 24 C 292
(the temperature of a future day is treated as known: in real use it would be the weather forecast)
```

The daily series is **synthetic** so that we know what drives it: a flat level of 300, a weekly pattern (weekends busier), +80 on promotion days
(about 1 in 10, planned in advance), a heat effect $2\max(T - 24, 0)^2$ that is zero below 24 °C and grows fast above it, and Gaussian noise
with standard deviation 15. It stands in for the kind of retail or delivery series where external drivers matter; on a real one, check that the
driver values you train on are the ones you would have had at forecast time (the weather *forecast*, the promotion *plan*).

**SES.** `ses_one_step` is the loop of Step 1; `fit_ses` minimizes the squared one-step errors over $\alpha \in [0, 1]$:

```text
== from scratch: simple exponential smoothing ==
alpha=0.3: one-step forecasts match statsmodels SimpleExpSmoothing (known initial level): OK
fitted alpha on a noisy wandering level: mine 0.0656, statsmodels 0.0656
fitted alpha on CO2: 1.0000 (SES has no trend or season, so it copies the last value: naive)
```

On a level that wanders slowly under a lot of noise, the best $\alpha$ is small (0.066): average over many past values to cancel the noise. On
CO2, which rises steadily, any averaging makes the level lag behind, so the best SES is $\alpha = 1$, the naive forecast. SES is the right tool only
for series without trend or season; it is here to show the mechanism.

## With statsmodels and LightGBM

Seven contenders forecast the CO2 years 1997–2001, each from the end of the previous year: seasonal naive, SES, Holt–Winters
(`ExponentialSmoothing(trend="add", seasonal="add", seasonal_periods=12)`), SARIMA(1,1,1)(0,1,1)$_{12}$ (`SARIMAX`), B6.3's linear model on lags,
LightGBM on the same lag features with the raw level as target, and LightGBM on the 12-month differences (Step 4):

```text
== CO2, long series (466-514 training months) ==
MAE in ppm per test year; skill = 1 - MAE / MAE(seasonal naive)
method                               1997   1998   1999   2000   2001    mean  skill
seasonal naive                      1.109  2.914  1.611  1.136  1.493   1.652  0.000
SES (from scratch)                  1.967  2.418  2.056  1.883  1.919   2.049 -0.240
Holt-Winters additive               0.282  1.193  0.429  0.213  0.223   0.468  0.717
SARIMA(1,1,1)(0,1,1)12              0.424  1.058  0.513  0.289  0.273   0.511  0.691
linear on lags (B6.3)               0.680  1.196  0.650  0.819  0.454   0.760  0.540
LightGBM on lags                    0.960  2.282  1.410  0.958  1.109   1.344  0.187
LightGBM on 12-month differences    0.525  1.819  0.789  0.559  0.425   0.823  0.502
LightGBM on raw lags, 2001: forecast max 370.98 ppm, highest value in its training data 371.65, actual max 373.80
Holt-Winters fitted on 1958-2000: alpha 0.574, beta 0.010, gamma 0.128; final trend 0.1303 ppm/month
SARIMA fitted on 1958-2000: ar.L1 0.274, ma.L1 -0.594, ma.S.L12 -0.854, sigma2 0.093
```

- **Holt–Winters wins** (MAE 0.47, skill 0.72), with SARIMA close behind (0.51). Their fitted parameters read sensibly: Holt–Winters keeps a slow
  slope ($\beta = 0.01$, a long memory for the trend) of 0.13 ppm a month, 1.56 a year; SARIMA's seasonal MA coefficient of $-0.85$ says "most of a
  surprise in one month's seasonal pattern does not repeat next year".
- **LightGBM on the raw level fails** exactly as Step 4 predicts. Its highest forecast for 2001 is 370.98 ppm, below the highest value in its training
  data (371.65), while the truth reached 373.80: it cannot go higher than what it has seen. Modelling the 12-month change instead fixes that (MAE
  1.34 → 0.82), but it still trails the classical models and the linear lag model.
- **Part of the classical advantage is information.** Holt–Winters and SARIMA update with every month up to the origin, while the lag models use
  nothing newer than 12 months (B6.3's direct strategy). B6.3's exercise 2 lets you measure how much of that gap a recursive model closes. A fair comparison
  states what each model is allowed to see.

**A short history.** The same contest, with the series starting in 1992, so each fold trains on 5–9 years:

```text
== CO2, short series (60-108 training months) ==
MAE in ppm per test year
method                               1997   1998   1999   2000   2001    mean  skill
seasonal naive                      1.109  2.914  1.611  1.136  1.493   1.652  0.000
SES (from scratch)                  1.967  2.418  2.056  1.883  1.919   2.049 -0.240
Holt-Winters additive               0.522  0.950  1.184  0.329  0.326   0.662  0.599
SARIMA(1,1,1)(0,1,1)12              0.487  0.991  0.673  0.321  0.283   0.551  0.666
linear on lags (B6.3)               0.461  2.231  0.572  1.079  0.680   1.004  0.392
LightGBM on lags                    0.994  2.407  1.560  0.810  1.191   1.392  0.157
LightGBM on 12-month differences    0.696  1.293  0.670  0.447  1.311   0.884  0.465
```

SARIMA (4 parameters) barely notices the shorter history (0.51 → 0.55). The lag models lose more: the linear model's error rises by a third
(0.76 → 1.00), with a bad 1998, because with lags up to 24 months the first fold has only 36 complete training rows. With little data, few
parameters and a built-in notion of trend win.

**Known drivers.** On the synthetic daily demand, 8 folds of 14 days, LightGBM gets lags of at least 14 days plus weekday, promotion flag and
temperature. The classical side gets a fair chance too: `SARIMAX(..., exog=[promo, temp])` is regression on the drivers with SARIMA errors, the
standard classical way to use external information (but linear in each driver):

```text
== synthetic daily demand with known drivers ==
MAE in orders per day, 8 folds of 14 days (fold start dates)
method                              09-10  09-24  10-08  10-22  11-05  11-19  12-03  12-17    mean  skill
seasonal naive                     38.109 26.159 23.884 39.963 31.017 44.180 25.862 29.448  32.328  0.000
Holt-Winters additive              32.975 19.002 23.564 33.514 25.799 29.274 17.999 19.394  25.190  0.221
SARIMA(1,0,0)(1,0,1)7              28.149 25.432 26.237 35.093 26.574 28.182 19.183 21.989  26.355  0.185
SARIMAX + promo, temp (linear)     17.783 11.080 16.064 17.961 14.083 17.047 19.879 20.068  16.745  0.482
LightGBM on lags + promo, temp     12.494  8.898 10.817 16.589  7.862 12.468 14.708 17.048  12.610  0.610
(the noise alone has std 15, so even a perfect model has MAE about 12.0)
```

Without the drivers, no model can know which future day has a promotion, so Holt–Winters and SARIMA only beat the baseline by about 20%.
SARIMAX with the drivers halves the error, but it can only fit a straight line in temperature, while the true effect is zero below 24 °C and
curved above. LightGBM learns the curve and the weekday pattern and gets to 12.6, close to the floor of 12.0 that the noise alone imposes (for
Gaussian noise with standard deviation $\sigma$, the mean absolute value is $\sigma\sqrt{2/\pi}$). It wins in every one of the 8 folds.

## In practice

- **When classical models win:** one or a few series, each with its own history; short histories (tens to a few hundred points); smooth trend and
  stable seasonality; no useful external drivers; when you need prediction intervals cheaply (both families give them from their error model) and
  a model a planner can read. Start with ETS (the state-space form of exponential smoothing, which can choose additive or multiplicative components
  automatically) and seasonal ARIMA.
- **When boosting wins:** many related series (thousands of products or stores, trained as one global model with series id or attributes as
  features); external drivers known at forecast time (promotions, prices, holidays, weather forecasts) with non-linear or interacting effects;
  long histories. Most winning entries in the M5 retail forecasting competition were LightGBM models of this kind (VERIFY (as of 2026-11) the
  competition summary in Makridakis, Spiliotis and Assimakopoulos, *International Journal of Forecasting*, 2022).
- **Failure modes:**
  - *Trees on a trending level.* Forecasts flatten at the training maximum. Model differences or ratios, or detrend first.
  - *ARIMA on a non-stationary series.* Forgetting to difference gives forecasts that drift back to an old mean. Plot the differenced series; use
    $d = 1$ (and $D = 1$ for a strong season) when in doubt.
  - *Comparing models that saw different information.* A model that updates with the latest month against a direct model with 12-month-old lags
    (above). State the information set of each.
  - *Driver leakage.* Training on observed weather and serving with forecast weather; training on promotions that were added after the fact.
  - *Over-trusting one series.* Rankings differ by series. Compare on your own series under walk-forward, with fold-by-fold results.
  - *Optimizer warnings.* statsmodels may warn that a fit did not converge on some folds; check the parameters and the walk-forward error rather
    than ignoring or blindly trusting the fit.
- **Defaults that work:** seasonal naive as the bar; `ExponentialSmoothing(trend="add", damped_trend=True, seasonal="add")` and a small seasonal
  ARIMA as classical candidates; LightGBM on lags ≥ horizon plus calendar and drivers, with a differenced target if the series trends; choose by
  walk-forward MAE and fold-by-fold wins, not by in-sample fit (statsmodels 0.15.0, LightGBM 4.7.0; VERIFY (as of 2026-11)). Dedicated libraries
  such as `statsforecast` (fast ETS/ARIMA on many series), `sktime` and `darts` wrap both families (not pinned in this course; VERIFY (as of 2026-11)).

## Exercises

Save your work to `my-work/B/b6_4_arima_vs_boosting.md` (or `.py`) and ask for a review.

1. **By hand:** (a) Run SES with $\alpha = 0.2$ and $\ell_0 = 100$ on $y = 100, 110, 90, 120$: give each one-step forecast, error and level, the SSE,
   and the forecast for the next step. (b) Write out the weights SES with $\alpha = 0.2$ puts on $y_T, y_{T-1}, y_{T-2}, y_{T-3}$, and the weight left on
   everything older. (c) A Holt–Winters model has $\ell_T = 200$, $b_T = 3$ and seasonal terms for the next three months $+10, -5, -5$. Give the
   forecasts for $h = 1, 2, 3$, and for $h = 3$ with a damped trend $\phi = 0.9$ (forecast $\ell_T + (\phi + \phi^2 + \dots + \phi^h)\,b_T + s$).
   (d) For an ARIMA(1,1,0) with $c = 1$ and $\phi = 0.6$, the last two values are 50 and 54. Forecast the next two values.
2. **Code:** On the lab's CO2 long series and folds: (a) fit SARIMA models with orders $(p, 1, q)(0, 1, 1)_{12}$ for $p, q \in \{0, 1, 2\}$ on the first fold's
   training data, rank them by AIC, then compare the top three by walk-forward MAE over all five folds. Does the AIC ranking agree with the walk-forward
   ranking? (b) Holt–Winters with `damped_trend=True` versus `False`: compare MAE per fold. (c) Plot the 2001 forecasts of Holt–Winters, SARIMA and both
   LightGBM variants against the actual values, and save the figure to `outputs/`.
3. **Think:** A retailer wants weekly sales forecasts, 8 weeks ahead, for 20,000 product–store combinations, to drive replenishment. Most products have
   2–3 years of history, many sell fewer than 5 units a week, new products appear every month, and the price and promotion plan is known 8 weeks ahead.
   Would you use per-series ETS/ARIMA, a global LightGBM, or both? Explain how you would set up the global model (target, features, how series of
   different scales are handled), what you would do for new products and for slow sellers with many zero weeks, and how you would decide between the
   approaches.

## Takeaways

1. SES is an exponentially weighted average, $\ell_t = \alpha y_t + (1 - \alpha)\ell_{t-1}$, with weights $\alpha(1-\alpha)^j$; Holt–Winters runs the same update for
   level, slope and season, and forecasts $\ell_T + h\,b_T + s_{T+h-m}$.
2. ARIMA($p, d, q$) differences the series $d$ times until it is stationary, then models each change from the last $p$ changes and the last $q$ errors;
   SARIMA repeats this at the seasonal lag. Parameters come from maximum likelihood.
3. Tree ensembles cannot forecast outside the range of their training targets; on a trending series, model the change ($y_t - y_{t-m}$) and add it back.
4. On one clean, trending series the classical models won (Holt–Winters MAE 0.47 against 0.82 for the best LightGBM), and their lead held on a 5–9 year
   history. With known, non-linear drivers, LightGBM won (12.6 against 16.7 for SARIMAX with the same drivers, near the noise floor of 12.0).
5. Choose by walk-forward error on your own series, with seasonal naive as the bar, and make sure every contender sees the same information.

**What builds on this:** [J2.2](../J/j2_2_monitoring_drift.md) (monitoring forecast error and drift after deployment); P12 (a forecasting capstone
option); [E2.1](../E/e2_1_rnn_bptt.md) (recurrent networks: a third option for sequences, rarely the first choice for tabular forecasting).

**References:** R. J. Hyndman, G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd edition, OTexts, 2021, chapters 8 (exponential smoothing)
and 9 (ARIMA) (VERIFY link: https://otexts.com/fpp3/). C. C. Holt, "Forecasting seasonals and trends by exponentially weighted moving averages", ONR
memorandum, 1957 (reprinted in *International Journal of Forecasting* 20(1), 2004). P. R. Winters, "Forecasting sales by exponentially weighted moving
averages", *Management Science* 6(3), 1960. G. E. P. Box, G. M. Jenkins, *Time Series Analysis: Forecasting and Control*, 1970. S. Makridakis,
E. Spiliotis, V. Assimakopoulos, "M5 accuracy competition: Results, findings, and conclusions", *International Journal of Forecasting* 38(4), 2022
(VERIFY). statsmodels documentation, "Time Series analysis tsa" (VERIFY link: https://www.statsmodels.org/stable/tsa.html).
