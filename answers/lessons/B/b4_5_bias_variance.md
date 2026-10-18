# Answer key: B4.5 The bias–variance trade-off, learning curves and validation curves

Lesson: [`lessons/B/b4_5_bias_variance.md`](../../../lessons/B/b4_5_bias_variance.md)

## Exercise 1 (by hand)

**Solution (a).** Predictions $12, 8, 14, 6, 10$: mean 10, so bias² $= (10 - 10)^2 = 0$. Squared deviations $4, 4, 16, 16, 0$; variance $= 40/5 = 8$. Expected squared error $= 0 + 8 + 4 = 12$.

**(b)** Predictions $13, 13, 12, 14, 13$: mean 13, bias² $= 9$; deviations $0, 0, 1, 1, 0$, variance $0.4$; expected error $9 + 0.4 + 4 = 13.4$. The first model is slightly better in expectation (12 against 13.4), but the two fail differently.
For the first, reduce variance: average several models, regularize, or add data. For the second, reduce bias: more capacity or better features (or, since it is a constant offset here, fix the systematic error).
A strong answer adds that, in production, the second model's consistent error may be easier to detect and correct.

**(c)** Model A: the training and validation errors are close (gap 0.2, then 0.05) and barely improve with 10 times more data. That is high bias: add capacity or features; more data will not help.
Model B: a large gap (3.3, then 2.2), training error rising and validation error falling as data grows. That is high variance: more data is helping and would help more; also regularize, simplify or average.

**(d)** The training labels contain the noise too: $y_i = f(x_i) + \varepsilon_i$. A fully grown tree puts each training row in its own leaf (or a pure one) and reproduces $y_i$ exactly, noise included. Training RMSE 0 means
it memorized the noise. On new rows the noise is different, so the memorized part becomes error: the variance term.

```text
== exercise 1 check ==
(a) first model: mean 10  bias^2 0  variance 8  expected error 12
(b) second model: mean 13  bias^2 9  variance 0.4  expected error 13.4
```

**Rubric (13 pts):**

| Criterion | Points |
|---|---|
| (a) Bias² 0, variance 8, total 12 | 3 |
| (b) Bias² 9, variance 0.4, total 13.4; a different remedy for each model | 3 |
| (c) A = high bias, B = high variance, each with a matching action | 4 |
| (d) Labels contain noise; memorizing it shows up as variance on new data | 3 |

**Likely mistakes:**

- Computing the variance around the true value 10 instead of around the mean prediction (for the second model that gives 9.4, mixing bias into variance).
- Using $n - 1$ in the variance: acceptable if stated, but Step 1 uses the expectation, so divide by $n$ for these "population" numbers.
- Reading model B's *rising* training error as getting worse: it is the gap closing, which is healthy.

## Exercise 2 (code)

**Solution:** [`b4_5_bias_variance_ex2.py`](b4_5_bias_variance_ex2.py).

```text
== (a) k-nearest-neighbour regression: bias and variance by k ==
  k   bias^2  variance sigma^2/k   total
  1   0.0012    0.0966    0.0900  0.1878
  2   0.0030    0.0547    0.0450  0.1477
  5   0.0185    0.0317    0.0180  0.1402
 10   0.0539    0.0254    0.0090  0.1693
 20   0.1877    0.0370    0.0045  0.3148
 30   0.4975    0.0190    0.0030  0.6065

== (b) learning curves: random forest versus full tree ==
full-depth tree:
      660 rows: train RMSE 0.0000   val RMSE 0.8978   gap 0.8978
     1320 rows: train RMSE 0.0000   val RMSE 0.8562   gap 0.8562
     2641 rows: train RMSE 0.0000   val RMSE 0.8203   gap 0.8203
     5283 rows: train RMSE 0.0000   val RMSE 0.8143   gap 0.8143
     9246 rows: train RMSE 0.0000   val RMSE 0.7510   gap 0.7510
    13209 rows: train RMSE 0.0000   val RMSE 0.7384   gap 0.7384
random forest:
      660 rows: train RMSE 0.2362   val RMSE 0.6277   gap 0.3915
     1320 rows: train RMSE 0.2224   val RMSE 0.5895   gap 0.3671
     2641 rows: train RMSE 0.2022   val RMSE 0.5650   gap 0.3628
     5283 rows: train RMSE 0.2020   val RMSE 0.5365   gap 0.3344
     9246 rows: train RMSE 0.1947   val RMSE 0.5119   gap 0.3172
    13209 rows: train RMSE 0.1880   val RMSE 0.5001   gap 0.3121
```

**(a) What a strong answer explains:** small $k$ is low bias and high variance, large $k$ the reverse, and the total is lowest at $k = 5$. The variance is always *above* $\sigma^2/k$ because $\sigma^2/k$ counts only the label noise of
$k$ fixed neighbours. Here the training $x$ values are random too, so which neighbours are found, and their true $f$ values, change from one training set to the next. That adds variance. The extra part grows with $k$ because farther neighbours
have more varied $f$ values. At $k = 30$ the prediction is the mean of all 30 targets: variance $= \sigma^2/30 + \operatorname{Var}[f(X)]/30 = 0.003 + 0.5/30 = 0.0197$, matching the measured 0.0190, and the row is identical to the lab's degree-0 polynomial (the same model).

**(b)** The forest's validation RMSE is far lower than the tree's at every size (0.50 against 0.74 with all rows), and its gap is less than half (0.31 against 0.74): averaging decorrelated trees removes
much of the variance (B3.2). The gap still shrinks only slowly with data, because each tree is still fully grown; the forest's remaining error is mostly the variance that correlated trees cannot cancel, plus bias.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| kNN decomposition by simulation for all six k (same setup as the lab) | 4 |
| Comparison with $\sigma^2/k$, explaining the extra variance from random x positions | 3 |
| Identifies the best k and the bias–variance shape | 2 |
| Cross-validated learning curves for the forest and the tree on the same folds | 4 |
| Interpretation of the forest's smaller gap and lower error (averaging) | 3 |

**Likely mistakes:**

- Using a fixed design (the same $x$ values in every training set) and then claiming variance equals $\sigma^2/k$ exactly, without noticing the difference from the random-design case.
- Measuring "variance" across test points instead of across training sets.
- Drawing the forest's learning curve with different folds from the tree's.

## Exercise 3 (think)

**Strong answer:** 99% training against 82% validation accuracy is a large gap: high variance. But accuracy is a poor lens at 20% churn (B4.1): predicting "no churn" is 80% accurate, so 82%
is barely better than the baseline. Ask for log-loss or average precision as well. Curves to request: (1) a cross-validated learning curve, using the right splitter (grouped by customer, time-ordered if churn is measured over time, B4.3); (2) a validation curve for the main capacity
setting (depth, number of trees, regularization). What justifies each plan: (i) more data, if the validation curve on the learning plot is still clearly falling at the current size; if it has flattened, a year of data
mostly buys staleness. (ii) A deeper model only if the curves show high bias (training and validation both poor and close together). With a 17-point gap, a deeper model would most likely make things worse;
regularization, early stopping or an ensemble is the indicated move. (iii) 200 vendor features add capacity and new information. If the gap is the problem, they add variance unless the features carry
real signal; test with a learning curve before and after adding them, on the same folds, and check them for leakage (B1.4) and for cost and stability in production.
**Weak answer:** "Approve all three; more data, a bigger model and more features all improve accuracy."
**Why weak:** it ignores the diagnosis. The gap says variance, so a deeper model and 200 features are likely to hurt. And it never questions accuracy as the metric.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Diagnoses high variance from the gap | 2 |
| Questions accuracy at a 20% base rate; proposes better metrics | 2 |
| Asks for cross-validated learning and validation curves with the right splitter | 2 |
| States what each curve must show to justify each of the three plans | 3 |
| Mentions leakage or production cost of the vendor features | 1 |

## Scoring

Total 39 pts. Pass ≥ 70% (28). Topic ids to revisit for each lost criterion: the decomposition (Ex 1a–b, 1d, Ex 2a) → B4.5, A5.1; reading curves (Ex 1c, Ex 3) → B4.5;
kNN and forests (Ex 2) → B2.3, B3.2; metrics and splitters (Ex 3) → B4.1, B4.3.
