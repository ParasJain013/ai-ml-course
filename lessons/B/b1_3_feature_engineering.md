# B1.3 Feature engineering

**Core** · Part B, week 1 (course week 07) · ~2.5 h · Lab:
[`labs/B/b1_3_feature_engineering.py`](../../labs/B/b1_3_feature_engineering.py)

## Why it matters

A model sees only the columns you give it. A linear model in particular can only multiply each
column by a weight and add the results, so an effect that is not a column (the evening rush, a
weekend, a bike being slower per kilometre than a car) is invisible to it. **Feature engineering** turns
raw data into columns that expose those effects: scaled numbers, encoded categories, date parts,
interactions. In the lab the same linear regression goes from 6.3 to 3.3 minutes of error on
delivery times, close to the noise floor, with no change to the model. The second half of the topic
is just as important in production: every transformation that learns something (a mean, a list of
categories) must be fitted on training data and applied identically at prediction time, which is
what scikit-learn's `Pipeline` and `ColumnTransformer` are for.

## Prerequisite check

- Can you explain why gradient descent is slow when features have very different scales? If not,
  revisit [A4.4](../A/a4_4_gradient_descent.md).
- Can you say why a matrix with linearly dependent columns makes $\mathbf{X}^\top\mathbf{X}$ singular? If not, revisit
  [A2.3](../A/a2_3_linear_systems_rank.md).
- Can you predict the shape of `(n, 1) == (1, k)` under NumPy broadcasting? If not, revisit
  [A1.2](../A/a1_2_numpy_arrays.md).
- Can you explain why an imputer's fill value is learned on the training rows only? If not, revisit
  [B1.2](b1_2_data_cleaning.md).

## Intuition

### A linear model adds up columns

A linear model predicts $\hat{y} = w_1x_1 + w_2x_2 + \dots + b$. Each column gets one weight, so each column can only
push the prediction up or down in proportion to its value. Take delivery time and the hour of the order.
Deliveries are slow from 17:00 to 19:59 and normal otherwise. Given the hour as a number, the model can only
learn "later is slower" or "later is faster", one straight line through all 24 hours. Given a column "is it
rush hour" (1 or 0), it can learn "+8 minutes in the rush" exactly. Feature engineering is this move,
repeated: decide what shape each effect has, then build a column with that shape.

Four kinds of transformation cover most tabular work.

**Scaling.** Features measured in different units (kilometres, item counts, pounds) have very different
ranges. Anything that compares or penalizes features as numbers cares: gradient descent (A4.4 and B1.5),
distances (k-nearest neighbours, B2.3), and regularization (B2.2), which penalizes large weights and so
penalizes features with small units. **Standardization** maps each column to mean 0 and standard deviation 1;
**min–max scaling** maps the training range to [0, 1]. Tree models (B3) split on thresholds and do not care.

**Categorical encoding.** A category (city, vehicle) is not a number, and numbering it (Bristol = 1, Leeds = 2,
London = 3) invents an order and equal spacing that do not exist. **One-hot encoding** gives each category its
own 0/1 column, so the model can learn one separate shift per category. Two practical details: a category
never seen in training must not crash the model at prediction time (the usual choice is all zeros), and a
column per category plus an intercept is redundant (the "dummy variable trap", in the math below).

**Dates and times.** A timestamp as a raw number (seconds since 1970) means almost nothing to a model. What
matters is usually a part of it: day of week, hour, month, a holiday flag, and durations such as "days since
the restaurant opened" measured from the moment of the prediction. Cyclic quantities (month, hour) wrap
around: December is next to January. Encoding the month as the pair $(\sin\theta, \cos\theta)$ places the 12 months on
a circle, so neighbouring months are always close.

**Interactions and shapes.** If bikes take 4 minutes per kilometre and cars 2, the effect of distance depends
on the vehicle. A linear model with a distance column and a vehicle one-hot can only shift the line per
vehicle, not change its slope. A product column, "distance if the vehicle is a bike, else 0", gives each
vehicle its own slope. Similarly, if new restaurants are slow at first and the effect fades, a column shaped
like that fade (here $e^{-\text{days}/60}$) works far better than raw days. Knowing the shape comes from domain
knowledge and from plotting residuals against candidate features.

### Fit on training rows, apply everywhere: the Pipeline

Several of these steps *learn* from data: the scaler learns means and standard deviations, the encoder
learns the list of categories, the imputer (B1.2) learns fill values. They follow the same pattern as a
model: `fit` on training rows, `transform` any rows. scikit-learn makes the pattern composable:

- A **`ColumnTransformer`** applies a different transformer to each group of columns and concatenates the
  results side by side.
- A **`Pipeline`** chains steps: the ColumnTransformer, then the model. `pipeline.fit(train_df, y)` fits
  every step on the training rows, in order; `pipeline.predict(new_df)` applies the fitted steps to raw
  new rows.

> **Bridge:** a scikit-learn `Pipeline` is middleware: every request (row) passes through the same ordered
> steps. **Where it breaks:** middleware has no `fit`; a pipeline step *learns* from the training data, and
> learning from the test data is leakage (B1.4).

The other reason for the pipeline is **train/serve skew**: features computed one way in a training notebook
and another way in the production service (a different rounding, a different list of categories, "today"
instead of the order time). Shipping the fitted pipeline as one object means production runs exactly the
code and the fitted values that training used.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{x}$ | one numeric column of the training data | `(n,)` |
| $\mu$, $\sigma$ | its mean and standard deviation (divide by $n$), learned on training rows | scalars |
| $z$ | a standardized value, $(x - \mu)/\sigma$ | scalar |
| $c_1, \dots, c_K$ | the categories seen in training, sorted | — |
| $\mathbf{e}(c)$ | one-hot vector of category $c$: 1 in the position of $c$, 0 elsewhere | `(K,)` |
| $\mathbf{X}$ | the full feature matrix after the transformations | `(n, d)` |
| $m$ | month number, 1 to 12 | int |
| $\theta$ | the month as an angle, $2\pi(m - 1)/12$ | scalar |
| $v_{\text{bike}}$ | 1 if the vehicle is a bike, else 0 (same for car, scooter) | scalar |

**Standardization and min–max.** With $\mu$ and $\sigma$ from the training rows,

$$z = \frac{x - \mu}{\sigma}, \qquad x_{\text{minmax}} = \frac{x - \min(\mathbf{x})}{\max(\mathbf{x}) - \min(\mathbf{x})}.$$

The same $\mu, \sigma, \min, \max$ are used for validation, test and production rows, so those rows will not have mean
exactly 0 or range exactly [0, 1]; a new value outside the training range gives $x_{\text{minmax}} > 1$ or $< 0$. scikit-learn's
`StandardScaler` uses the divide-by-$n$ standard deviation (`ddof=0`). Standardizing does not change what a linear
model can fit, only the units of its weights: $wz = \frac{w}{\sigma}(x - \mu)$, so a weight per standard deviation $w$ becomes $w/\sigma$
per raw unit (B1.5 uses this to convert back).

**One-hot and the dummy variable trap.** One-hot encoding replaces a category column by $K$ columns, row $i$ getting
$\mathbf{e}(c_i)$. A linear model then predicts

$$\hat{y} = b + \sum_{j=1}^{K} w_j\,[c_i = c_j] + \dots = b + w_{c_i} + \dots,$$

one shift per category. But every row has exactly one 1 among the $K$ columns, so the $K$ columns add up to the
column of ones that carries the intercept $b$. The columns are linearly dependent, $\mathbf{X}$ has rank one less than its
number of columns, and $\mathbf{X}^\top\mathbf{X}$ is singular (A2.3): adding 5 to $b$ and subtracting 5 from every $w_j$ gives identical
predictions, so the weights are not unique. Two fixes: drop one category's column (`drop="first"`), making that
category the baseline the others are compared with; or keep all $K$ columns and use a solver that copes
(`LinearRegression` uses a least-squares solver that picks one of the equivalent solutions; regularization,
B2.2, also makes it unique). The predictions are the same either way.

**Cyclic encoding.** Map month $m$ to the angle $\theta = 2\pi(m - 1)/12$ and use the two columns $(\sin\theta, \cos\theta)$. Every month is
a point on the unit circle, and neighbouring months are always the same distance apart: $2\sin(\pi/12) = 0.518$,
including December and January. As raw numbers 12 and 1 are 11 apart. Why two columns: $\cos\theta$ alone gives the same
value to February and December (both $\cos 30°$); the pair is unique. A linear model on $(\sin\theta, \cos\theta)$ can fit any
smooth yearly cycle with one peak, $a\sin\theta + b\cos\theta = r\cos(\theta - \varphi)$ (the angle-difference formula, with $r = \sqrt{a^2 + b^2}$, $\cos\varphi = b/r$, $\sin\varphi = a/r$),
with the peak month learned through $\varphi$.

**Interaction.** With columns $x_{\text{km}}\,v_{\text{bike}}$, $x_{\text{km}}\,v_{\text{car}}$, $x_{\text{km}}\,v_{\text{scooter}}$, the model is

$$\hat{y} = \dots + w_{\text{bike}}\,x_{\text{km}}v_{\text{bike}} + w_{\text{car}}\,x_{\text{km}}v_{\text{car}} + w_{\text{scooter}}\,x_{\text{km}}v_{\text{scooter}},$$

and for a bike order only the first term is non-zero: the slope per kilometre is $w_{\text{bike}}$. Each vehicle gets its own slope.

## Worked example by hand

**Scaling.** Training values $2, 4, 6, 8$. Mean 5, deviations $-3, -1, 1, 3$, squared $9, 1, 1, 9$, average $20/4 = 5$, so
$\sigma = \sqrt5 = 2.236$. Standardized: $-3/2.236 = -1.342$, $-0.447$, $0.447$, $1.342$. Min–max: $(x - 2)/6 = 0, 0.333, 0.667, 1$. A new value 10
standardizes to $5/2.236 = 2.236$ and min–max scales to $8/6 = 1.333$, outside [0, 1].

**One-hot.** Colours `red, blue, red, green`. Sorted categories: `blue, green, red`. Rows: red $\to (0, 0, 1)$, blue $\to (1, 0, 0)$,
red $\to (0, 0, 1)$, green $\to (0, 1, 0)$. An unseen `purple` $\to (0, 0, 0)$.

**Date.** Saturday 17 October 2026 at 18:30. Day of week 5 (Monday = 0), so weekend = 1; hour 18, so rush = 1.
Month 10: $\theta = 2\pi \times 9/12 = 3\pi/2 = 4.712$, $(\sin\theta, \cos\theta) = (-1, 0)$. If the restaurant opened on 1 September 2026, days since
opening = 30 (all of September) + 16 = 46.

**Interaction.** 3.0 km by bike: $(\text{bike\_km}, \text{car\_km}, \text{scooter\_km}) = (3.0, 0, 0)$.

```text
== worked example by hand ==
x = [2.0, 4.0, 6.0, 8.0]   mean = 5.0   deviations = [-3.0, -1.0, 1.0, 3.0]   sd (divide by n) = sqrt(5.0) = 2.2361
standardized = [-1.3416, -0.4472, 0.4472, 1.3416]   min-max = [0.0, 0.3333, 0.6667, 1.0]
new value 10 -> standardized 2.2361   min-max 1.3333 (outside [0, 1]: the scaler only knows the training range)
categories (sorted) = ['blue', 'green', 'red']
one-hot = [[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]]   'purple' -> [0, 0, 0]
2026-10-17 18:30:00: day of week = 5 (Saturday, Monday = 0)   weekend = 1   hour = 18   rush = 1
month = 10: angle 2*pi*(month - 1)/12 = 4.7124   (sin, cos) = [-1.0, 0.0]
December (12) and January (1): cos = 0.8660 and 1.0000   raw month numbers 12 and 1 are 11 apart
days since opening (2026-09-01) = 46
interaction: distance 3.0 km on a bike -> [bike_km, car_km, scooter_km] = [3.0, 0.0, 0.0]
```

## Key plot

![Horizontal bar chart of validation MAE in minutes for the same linear model with growing feature sets: numeric only 6.33, plus one-hot city and vehicle 5.31, plus date features 3.48, plus distance-by-vehicle interaction 3.27, with a dashed line at the noise floor 3.19](img/b1_3_features_vs_error.png)

Same model, same data; only the columns change. The date features (weekend, rush hour, month, restaurant
newness) give the biggest drop, and the interaction takes the error to within 0.08 minutes of the noise floor,
the error even the true formula would make.

## From scratch in NumPy

**The data.** 3,434 synthetic food-delivery orders from 2025 (few at night), with raw columns `order_time`,
`opened_date` (when the restaurant opened), `city`, `vehicle`, `distance_km`, `n_items`, and the target `minutes`. The
generator builds delivery time from a city base, a per-kilometre time that depends on the vehicle, 1.5
minutes per item, +5 at weekends, +8 in the evening rush, a winter bump, a slow first few months for new
restaurants, and Gaussian noise with standard deviation 4. A random 60/20/20 split:

```text
== data ==
orders: 3434 rows   train / val / test: 2060 / 687 / 687
           order_time opened_date        city  vehicle  distance_km  n_items  minutes
0 2025-11-07 11:28:00  2023-12-17      London      car         2.19        1     31.3
1 2025-08-21 11:47:00  2024-03-06     Bristol     bike         2.69        1     26.8
2 2025-07-06 13:33:00  2023-12-09  Manchester  scooter         6.07        3     39.7
3 2025-04-09 11:19:00  2023-06-26     Bristol  scooter         1.58        3     22.8
minutes: mean 35.18  sd 9.58
```

**Transformers with `fit` and `transform`.** The course avoids classes except when the topic is about one;
here it is, because the fit/transform object is the thing to learn. Each `fit` stores what it learned in
attributes ending in `_` (scikit-learn's convention); `transform` only uses them:

```python
class StandardScalerScratch:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)                     # (d,)
        sd = X.std(axis=0)                              # (d,) ddof=0
        self.scale_ = np.where(sd == 0, 1.0, sd)        # a constant column stays 0, not NaN
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_           # (n, d)


class OneHotScratch:
    def fit(self, X):
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]   # sorted, per column
        return self

    def transform(self, X):
        blocks = [(X[:, [j]] == cats[None, :]).astype(np.float64)            # (n, 1) == (1, k) -> (n, k)
                  for j, cats in enumerate(self.categories_)]
        return np.hstack(blocks)
```

An unseen category matches none of the stored categories, so its block is all zeros, for free.

```text
== from scratch: scalers and one-hot ==
standard scaler: mean_ [3.417, 2.9782]   scale_ [1.7706, 1.3875]
train after scaling: mean [0.0, 0.0]   sd [1.0, 1.0]
val after scaling (train statistics): mean [-0.0793, -0.0241]   sd [1.0195, 0.9906]
one-hot categories: [['Bristol', 'Leeds', 'London', 'Manchester'], ['bike', 'car', 'scooter']]   width 7
unseen city 'York' on a bike -> [[0, 0, 0, 0, 1, 0, 0]]
intercept + all 4 city columns: shape (2060, 5)   rank 4   (the dummy variable trap: the city columns sum to the ones column)
```

The training rows come out with mean 0 and standard deviation 1 exactly; the validation rows, scaled with
the *training* statistics, come out close but not exact, as they should. The rank check shows the dummy
variable trap: 5 columns, rank 4.

**Dates, interaction, the full matrix.** `date_features` returns five columns (weekend, rush, month sine,
month cosine, $e^{-\text{days}/60}$), `distance_by_vehicle` three, and `build_matrix` stacks everything: 2 scaled numeric + 7 one-hot
+ 5 date + 3 interaction = 17 columns. Then the same `LinearRegression` is fitted on the first 2, 9, 14 and 17
columns:

```text
== from scratch: dates, interaction, full matrix ==
feature matrix: train (2060, 17)   val (687, 17)
first train row: [-1.1391, -0.705, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -0.5, -0.866, 0.0027, 1.4, 0.0, 0.0]
numeric only                2 columns   val MAE 6.329 min   val R^2 0.3540
+ one-hot city, vehicle     9 columns   val MAE 5.310 min   val R^2 0.5493
+ date features            14 columns   val MAE 3.479 min   val R^2 0.7983
+ distance x vehicle       17 columns   val MAE 3.275 min   val R^2 0.8283
mean baseline: val MAE 7.802 min
noise floor: the target has N(0, 4^2) noise, so even the true formula has MAE about 4 * sqrt(2 / pi) = 3.192 min
```

Each group of columns exposes effects the previous set could not express. With all 17, the validation MAE is
3.27 minutes, against a floor of $4\sqrt{2/\pi} = 3.19$: the mean absolute value of $\mathcal{N}(0, 4^2)$ noise, which no model can
predict. The mean baseline (B1.1) is 7.8. On real data there is no known floor, so you compare with the
baseline and look at residual plots instead.

## With scikit-learn

The same preprocessing as one object. `FunctionTransformer` wraps a plain function (no fitting) so it can sit
in a `ColumnTransformer`; giving it `feature_names_out` keeps the output columns named:

```python
preprocess = ColumnTransformer([
    ("num", StandardScaler(), ["distance_km", "n_items"]),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city", "vehicle"]),
    ("date", FunctionTransformer(date_features, feature_names_out=...), ["order_time", "opened_date"]),
    ("inter", FunctionTransformer(distance_by_vehicle, feature_names_out=...), ["distance_km", "vehicle"]),
])
model = Pipeline([("prep", preprocess), ("reg", LinearRegression())])
model.fit(train, y_tr)            # every step fitted on the training rows only
model.predict(one_raw_order)      # raw DataFrame in, minutes out
```

```text
== with scikit-learn ==
StandardScaler matches StandardScalerScratch: OK
MinMaxScaler matches MinMaxScalerScratch: OK
OneHotEncoder(handle_unknown='ignore') matches OneHotScratch, unseen 'York' included: OK
ColumnTransformer output (687, 17) equals build_matrix: OK
pipeline val MAE 3.275 min   (from scratch 3.275)
feature names: ['num__distance_km', 'num__n_items', 'cat__city_Bristol', 'cat__city_Leeds', 'cat__city_London', 'cat__city_Manchester', 'cat__vehicle_bike', 'cat__vehicle_car', 'cat__vehicle_scooter', 'date__weekend', 'date__rush', 'date__month_sin', 'date__month_cos', 'date__newness', 'inter__bike_km', 'inter__car_km', 'inter__scooter_km']
raw order from an unseen city -> predicted 47.99 min
```

- The scikit-learn scalers and encoder match the from-scratch classes, and the ColumnTransformer's output is
  identical to `build_matrix` column for column: same order, same values.
- `pipeline.predict` takes a raw DataFrame, including an order from York, a city the model never saw. It does
  not crash (all four city columns are 0). Do not read much into the 47.99: with all four city columns plus an
  intercept, how the city effect is split between the intercept and the city weights is arbitrary (the dummy
  variable trap again), so "no city" gets an arbitrary share. In production, count unseen categories and
  alert when they appear (J2.2).
- `get_feature_names_out()` names every column, which you need to read the weights (B1.5) and for
  explanations later (B3.4).

## In practice

- **When to use it:** for every tabular model. Start with the simple encoding of each column (scale numbers,
  one-hot categories, basic date parts), fit a baseline model, then add features where residual plots or
  domain knowledge show an effect the model cannot express. Linear models and k-NN need much more of this
  than tree ensembles, which find thresholds and interactions themselves (B3).
- **Failure modes:**
  - *Transformers fitted on all rows.* Mild leakage for scalers, severe for anything that uses the target
    (B1.4). You notice when preprocessing code calls `fit` outside a pipeline. Fit only through `Pipeline.fit`.
  - *Train/serve skew.* A feature computed differently in production: "days since signup" using today's date
    in a training query but the order time in the service; a category list that changed. You notice when
    production predictions drift from offline ones on the same inputs. Ship the fitted pipeline, and compute
    time features relative to the prediction time, never the time the dataset was built.
  - *Unseen categories.* An encoder fitted without `handle_unknown="ignore"` raises on the first new city in
    production. Set it, and monitor how often it happens.
  - *High-cardinality categories* (thousands of product ids). One-hot explodes the width and most columns are
    nearly always 0. Group rare categories (`min_frequency`), use hashing, or target encoding fitted inside
    cross-validation (it uses the target, so it leaks if done naively; B1.4).
  - *Cyclic features as plain numbers.* Month 12 and month 1 look far apart. Use sine and cosine, or one-hot.
  - *Too many hand-made features on little data.* Each column is another weight to estimate; validation error
    rises while training error falls (B4.5). Add features one group at a time and keep the ones that help on
    validation.
- **Defaults that work:** `StandardScaler` for numeric columns (`ddof=0`); `OneHotEncoder(handle_unknown="ignore")`
  for categories (`sparse_output=False` when the data is small; the default output is a sparse matrix);
  weekday, hour, month sine/cosine and holiday flags for timestamps; one `Pipeline` per model, saved and
  loaded as one artifact. `OneHotEncoder` defaults to `handle_unknown="error"` and `drop=None` (VERIFY as of
  2026-10, scikit-learn 1.9.1).

## Exercises

Save your work to `my-work/B/b1_3_feature_engineering.md` (or `.py`) and ask for a review.

1. **By hand:** (a) Standardize the training values $1, 3, 5, 7, 9$ (divide-by-$n$ standard deviation) and min–max scale
   them; then transform the new values 0 and 12 with both. (b) One-hot encode the training plans `pro, basic, team,
   basic` and then the new rows `team`, `enterprise`. (c) For Thursday 31 December 2026 at 23:15, give day of week
   (Monday = 0), weekend flag, rush flag (17:00–19:59), month sine and cosine, and days since a restaurant opened on 1
   December 2026. (d) Compute the distance between the cyclic encodings of December and January, and between June
   and July. (e) Write the 4-column design matrix (intercept plus one-hot) for the four training plans and show that
   its columns are linearly dependent.
2. **Code:** Copy the lab's data generator and pipeline. (a) Write a `CyclicEncoder` transformer class with `fit`,
   `transform` and `get_feature_names_out` that turns an integer column with a given period (hour: 24, month: 12)
   into sine and cosine columns, and test it on hours 0, 6, 12, 18 and 23. (b) Replace the rush flag by each of: the
   raw hour as a number, one-hot hour, and your cyclic hour, inside the pipeline, and report the validation MAE of
   each. (c) Explain the ranking from the shape of the true hour effect in the generator.
3. **Think:** A churn model was trained a year ago from a SQL query that computed
   `days_since_last_login = CURRENT_DATE - last_login_date` at the moment the query ran. In production the service
   computes the same feature at prediction time. The model's offline AUC was 0.82; live, it performs much worse.
   Explain what went wrong, how you would confirm it, and how you would design feature computation so that
   training and serving cannot disagree.

## Takeaways

1. A linear model can only weight the columns it is given; feature engineering builds columns with the
   shape of each effect (flags, bands, products, decays).
2. Standardize numeric features for anything that compares them as numbers (gradient descent, distances,
   regularization); trees do not need it.
3. One-hot encode categories, handle unseen ones (all zeros), and remember that $K$ one-hot columns plus an
   intercept are linearly dependent (drop one, or regularize).
4. Take dates apart (weekday, hour, month as sine/cosine, durations from the prediction time) and give
   interacting effects their own product columns.
5. Every transformer that learns from data is fitted on training rows and shipped with the model: a
   `ColumnTransformer` inside a `Pipeline` prevents both leakage and train/serve skew.

**What builds on this:** [B1.4](b1_4_data_leakage.md) (preprocessing fitted on all rows);
[B1.5](b1_5_linear_regression.md) (scaling and gradient descent, reading weights);
[B2.2](b2_2_regularization.md) (why regularization needs scaled features);
[B2.3](b2_3_knn.md) (distances need scaling); [B3.1](b3_1_decision_trees.md) (trees that do not);
[B6.3](b6_3_time_series_forecasting.md) (lag and calendar features); P2 (the M1 pipeline).

**References:** scikit-learn User Guide, "Pipelines and composite estimators" and "Preprocessing data"
(VERIFY link: https://scikit-learn.org/stable/modules/compose.html). A. Zheng and A. Casari, *Feature
Engineering for Machine Learning*, O'Reilly, 2018. D. Sculley et al., "Hidden Technical Debt in Machine
Learning Systems", NeurIPS 2015 (training/serving skew and pipeline debt).
