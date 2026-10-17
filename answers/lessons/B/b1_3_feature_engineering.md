# Answer key: B1.3 Feature engineering

Lesson: [`lessons/B/b1_3_feature_engineering.md`](../../../lessons/B/b1_3_feature_engineering.md)

## Exercise 1 (by hand)

**Solution (a).** Mean $= 25/5 = 5$; deviations $-4, -2, 0, 2, 4$; squares sum to 40; variance $40/5 = 8$; $\sigma = \sqrt8 = 2.828$.
Standardized: $-1.414, -0.707, 0, 0.707, 1.414$. Min–max $(x - 1)/8$: $0, 0.25, 0.5, 0.75, 1$. New value 0: $z = -5/2.828 = -1.768$, min–max
$-1/8 = -0.125$. New value 12: $z = 7/2.828 = 2.475$, min–max $11/8 = 1.375$. Both fall outside the training range of the
min–max scaler, which is expected and not an error.

**(b)** Sorted categories `basic, pro, team`. `pro` $\to (0, 1, 0)$, `basic` $\to (1, 0, 0)$, `team` $\to (0, 0, 1)$, `basic` $\to (1, 0, 0)$. New
`team` $\to (0, 0, 1)$; `enterprise` (unseen) $\to (0, 0, 0)$ with `handle_unknown="ignore"`, or an error with the default.

**(c)** 31 December 2026 is a Thursday: day of week 3, weekend 0. Hour 23: rush 0. Month 12: $\theta = 2\pi \times 11/12 = 330°$,
$\sin\theta = -0.5$, $\cos\theta = 0.866$. Days since 1 December 2026: 30.

**(d)** December is $(-0.5, 0.866)$, January $(0, 1)$: distance $\sqrt{0.25 + 0.018} = 0.518$. June ($\theta = 150°$) is $(0.5, -0.866)$, July
($180°$) is $(0, -1)$: distance $\sqrt{0.25 + 0.018} = 0.518$. Every pair of neighbouring months is $2\sin(\pi/12) = 0.518$ apart.

**(e)** Columns (intercept, basic, pro, team): `pro` $(1, 0, 1, 0)$, `basic` $(1, 1, 0, 0)$, `team` $(1, 0, 0, 1)$, `basic` $(1, 1, 0, 0)$. In every row
basic + pro + team $= 1$ = intercept, so column 1 minus the other three is the zero vector: the columns are linearly
dependent and the rank is 3, not 4.

```text
== exercise 1 check ==
(a) mean 5.0  sd sqrt(8.0) = 2.8284  z [-1.4142, -0.7071, 0.0, 0.7071, 1.4142]  min-max [0.0, 0.25, 0.5, 0.75, 1.0]
    new 0.0: z -1.7678  min-max -0.1250
    new 12.0: z 2.4749  min-max 1.3750
(b) categories ['basic', 'pro', 'team']: pro -> [0, 1, 0], basic -> [1, 0, 0], team -> [0, 0, 1], basic -> [1, 0, 0], team -> [0, 0, 1], enterprise -> [0, 0, 0]
(c) 2026-12-31 23:15:00: day of week 3 (Thursday)  weekend 0  rush 0  (sin, cos) [-0.5, 0.866]  days since 2026-12-01: 30
(d) |Dec - Jan| = 0.5176   |Jun - Jul| = 0.5176   2 sin(pi/12) = 0.5176
(e) design matrix [[1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0]]   rank 3 of 4 columns   col0 - col1 - col2 - col3 = [0.0, 0.0, 0.0, 0.0]
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) $\sigma$ with divide-by-$n$, both scalings, and the two new values with the *training* statistics | 3 |
| (b) Encodings in sorted category order; unseen category as all zeros (or an error, stated) | 2 |
| (c) Weekday, flags, sine/cosine of month 12 and the day count | 3 |
| (d) Both distances equal (0.518), with the reason | 3 |
| (e) The matrix and the explicit linear dependence (or rank 3) | 3 |

**Likely mistakes:**

- Using $n - 1$ for $\sigma$ (gives 3.162): not what `StandardScaler` does.
- Re-fitting the scaler on the new values, which makes 0 and 12 come out as $\pm 1$.
- Using $\theta = 2\pi m/12$ (with $m$ from 1): still a valid circle, just rotated; accept it if consistent.
- In (e), saying "the matrix is singular": it is not square. The point is that its *columns* are dependent, so
  $\mathbf{X}^\top\mathbf{X}$ is singular.

## Exercise 2 (code)

**Solution:** [`b1_3_feature_engineering_ex2.py`](b1_3_feature_engineering_ex2.py).

```text
== exercise 2 (a): CyclicEncoder ==
names ['hour_sin', 'hour_cos']
hours [0, 6, 12, 18, 23] -> [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0], [-0.2588, 0.9659]]
hour 0 and hour 24 encode identically: OK

== exercise 2 (b): four encodings of the hour ==
raw hour (one number)    17 columns   val MAE 3.991 min
cyclic hour (sin, cos)   18 columns   val MAE 3.772 min
one-hot hour             40 columns   val MAE 3.286 min
rush flag (lab)          17 columns   val MAE 3.275 min
noise floor 4 * sqrt(2 / pi) = 3.192 min
mean minutes by hour (train): {0: 31.4, 1: 28.8, 2: 30.6, 3: 30.0, 4: 33.4, 5: 33.2, 6: 35.7, 7: 35.9, 8: 37.2, 9: 28.6, 10: 34.9, 11: 34.0, 12: 34.4, 13: 33.5, 14: 32.4, 15: 34.1, 16: 33.0, 17: 41.7, 18: 41.4, 19: 40.8, 20: 33.3, 21: 33.8, 22: 33.0, 23: 29.5}
```

**(c) Why this ranking.** The generator's hour effect is a *step*: +8 minutes from 17:00 to 19:59, nothing otherwise
(the by-hour means show it: about 41 at 17–19, 33 to 35 around them; the night hours are noisy because few orders
happen then).

- *Raw hour* gives one slope across the day. A step in the middle of the range is not a line, so most of the
  effect is lost (3.99).
- *Cyclic hour* gives a smooth curve with one peak, $a\sin\theta + b\cos\theta$. It can put the peak in the evening but
  cannot be flat all day and jump for three hours (3.77).
- *One-hot hour* can fit any shape, including the step (3.29), at the cost of 23 extra weights, each estimated
  from the orders in one hour. With fewer rows, or a smoother true effect, it would overfit.
- *The rush flag* has exactly the right shape and one weight (3.275). It wins because we know the shape;
  on real data you would find it by plotting mean error by hour, then decide between one-hot and a flag.

Cyclic encoding is not wrong here, it is the wrong *shape*: it is the right choice when an effect rises and
falls smoothly around the clock.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| `CyclicEncoder` with `fit`/`transform`/`get_feature_names_out`, correct sine/cosine, period as a parameter | 4 |
| Works inside a `ColumnTransformer` (inherits `BaseEstimator`, `TransformerMixin`, or equivalent) | 2 |
| Tests on the given hours, including the wrap-around (0 and 24, or 23 next to 0) | 2 |
| Four encodings compared in the same pipeline on the same split | 4 |
| (c) Explanation tied to the shape of the true effect, and the over-fitting cost of one-hot | 4 |

**Likely mistakes:**

- A plain class without `get_params`: `ColumnTransformer` fails with "Cannot clone object ...". Inherit from
  `BaseEstimator`, and store `__init__` arguments unchanged under the same names.
- Learning something in `__init__` or changing the period in `fit`: scikit-learn's cloning assumes `__init__` only
  stores parameters.
- Comparing encodings on different splits or with different other features, which mixes several effects in one
  comparison.
- Concluding "cyclic encoding is bad": it depends on the shape of the effect.

## Exercise 3 (think)

**Strong answer:** The feature was computed differently in training and in serving (train/serve skew). In the
training query, `CURRENT_DATE` was the day the query ran, which for most rows was long after the moment a
prediction would have been made, so `days_since_last_login` measured "days between last login and the dataset
build". For customers who later churned, that gap includes the time after they stopped using the product, so the
feature was partly computed from the future: target leakage through time (B1.4), which inflated the offline AUC.
In production the feature is measured at prediction time and has a different distribution and a weaker link to
the label. To confirm: recompute the training feature "as of" each row's prediction date and compare its
distribution with the old one and with live values (they should match live); retrain and re-evaluate with the
corrected feature (expect a lower, honest AUC that matches live performance); compare the distribution of the
feature in training data against production logs (a drift check, J2.2). Design fixes: every training row has an
explicit prediction timestamp, and every feature is computed "as of" that timestamp with the same code used in
serving (one feature function or a feature store with point-in-time joins); no `NOW()` or `CURRENT_DATE` in
training queries; log the features the service actually used and train future models from those logs; add a test
that computes features for a few sample entities through both paths and asserts they are equal.

**Weak answer:** "The data changed since last year, so retrain the model on recent data." Weak because it misses
that the feature was defined differently in the two paths; retraining with the same query reproduces the same
leak and the same gap.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Identifies train/serve skew and that `CURRENT_DATE` made the training feature depend on time after the prediction | 3 |
| Connects it to leakage and the inflated offline score | 2 |
| A concrete way to confirm (point-in-time recomputation, distribution comparison, re-evaluation) | 2 |
| Design fix: as-of timestamps and one shared feature code path or logged features, with a test | 3 |

## Scoring

Total 40 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → B1.3 "The math"
and A2.3 (rank); exercise 2 → B1.3 "Intuition" (shapes of effects) and "With scikit-learn"; exercise 3 → B1.3 "In
practice" (train/serve skew), B1.4 (temporal leakage) and J2.2 (drift monitoring).
