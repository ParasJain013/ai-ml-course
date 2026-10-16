# Answer key: A5.1 Random variables, expectation, variance and covariance

Lesson: [`lessons/A/a5_1_random_variables_expectation.md`](../../../lessons/A/a5_1_random_variables_expectation.md)

## Exercise 1 (by hand)

**Solution (a).** $\mathbb{E}[X] = 0(0.5) + 1(0.3) + 4(0.2) = 1.1$. $\mathbb{E}[X^2] = 0 + 1(0.3) + 16(0.2) = 3.5$.
$\operatorname{Var}[X] = 3.5 - 1.1^2 = 3.5 - 1.21 = 2.29$, $\sigma = \sqrt{2.29} = 1.5133$.

**Solution (b).** $\mathbb{E}[Y] = 3 - 2(1.1) = 0.8$; $\operatorname{Var}[Y] = (-2)^2(2.29) = 9.16$ (the shift 3 does not matter, the sign
of $-2$ disappears in the square). Directly: $Y$ takes $3, 1, -5$ with probabilities $0.5, 0.3, 0.2$, so
$\mathbb{E}[Y] = 1.5 + 0.3 - 1.0 = 0.8$ ✓.

**Solution (c).** Independent, so variances add: $\operatorname{Var}[X_1 + X_2] = 2.29 + 2.29 = 4.58$.
$\operatorname{Cov}[X_1, X_1 + X_2] = \operatorname{Cov}[X_1, X_1] + \operatorname{Cov}[X_1, X_2] = 2.29 + 0 = 2.29$.

**Solution (d).** Marginals: $P(X = 1) = 0.1 + 0.4 = 0.5$, $P(Y = 1) = 0.2 + 0.4 = 0.6$. $\mathbb{E}[XY] = P(X = 1, Y = 1) = 0.4$.
$\operatorname{Cov} = 0.4 - 0.5 \times 0.6 = 0.1$. $\operatorname{Var}[X] = 0.5 - 0.25 = 0.25$, $\operatorname{Var}[Y] = 0.6 - 0.36 = 0.24$, correlation
$0.1/\sqrt{0.25 \times 0.24} = 0.1/0.2449 = 0.408$. Not independent: $0.4 \ne 0.5 \times 0.6 = 0.3$ (one failing cell is
enough; and a nonzero covariance already rules out independence).

```text
== exercise 1 check ==
(a) E[X] = 1.1   E[X^2] = 3.5   Var[X] = 2.29   sd = 1.5133
(b) E[3 - 2X] = 0.8 (rule) = 0.8 (table)   Var = 9.16
(c) Var[X1 + X2] = 4.58   Cov[X1, X1 + X2] = 2.29
(d) E[X] = 0.5  E[Y] = 0.6  E[XY] = 0.4  Cov = 0.1  corr = 0.4082
(d) P(X=1, Y=1) = 0.4 vs P(X=1) P(Y=1) = 0.3: not independent
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) $\mathbb{E}[X] = 1.1$, $\mathbb{E}[X^2] = 3.5$, $\operatorname{Var} = 2.29$, $\sigma = 1.513$ | 3 |
| (b) $\mathbb{E}[Y] = 0.8$ both ways; $\operatorname{Var}[Y] = 9.16$ with the $a^2$ rule | 3 |
| (c) $4.58$ (variances add because independent) and $2.29$ (covariance split by linearity) | 3 |
| (d) Marginals, $\operatorname{Cov} = 0.1$, correlation $0.408$ | 3 |
| (d) Independence correctly rejected with a reason | 2 |

**Likely mistakes:**

- $\operatorname{Var}[3 - 2X] = -2 \times 2.29$ or $3 + 4 \times 2.29$: forgetting that the shift drops out and the factor
  is squared. A variance is never negative; a negative answer is a red flag.
- $\operatorname{Var}[X] = \mathbb{E}[X^2] - \mathbb{E}[X] = 3.5 - 1.1$: subtracting the mean instead of its square.
- In (c), writing $\operatorname{Var}[X_1 + X_2] = 4\operatorname{Var}[X]$: that is $\operatorname{Var}[2X]$ (one spinner counted twice), not the sum of
  two independent spinners. The two differ because $X_1$ and $X_2$ partly cancel each other's deviations.
- In (d), checking independence only through the covariance in the wrong direction ("covariance is
  small, so independent"). Here it is nonzero, so they are dependent; but zero would not have proved
  independence.

## Exercise 2 (code)

**Solution:** [`a5_1_random_variables_expectation_ex2.py`](a5_1_random_variables_expectation_ex2.py).

```text
== exercise 2 (a): exact, by enumerating 36 outcomes ==
E[M]      = 161/36    = 4.4722
E[L]      = 91/36     = 2.5278
Var[M]    = 2555/1296 = 1.9715
Cov[M, L] = 1225/1296 = 0.9452
E[ML] = 49/4 = E[die1] E[die2] = 3.5 * 3.5 = 12.25

== exercise 2 (b): simulation ==
        n     E[M]     E[L]   Var[M]  Cov[M,L]
      100   4.6700   2.7800   2.1223    1.0479
    10000   4.4764   2.5122   1.9702    0.9288
  1000000   4.4735   2.5290   1.9718    0.9459
    exact   4.4722   2.5278   1.9715    0.9452
own estimators match np.cov: OK
```

(c) $\operatorname{Cov}[M, L] > 0$ because both are driven by the same two dice: two high dice give a high max *and* a
high min, two low dice give low values of both. The shortcut: $M \cdot L$ is always the product of the two
dice (the larger times the smaller is the same as die1 times die2), so
$\mathbb{E}[ML] = \mathbb{E}[X_1X_2] = \mathbb{E}[X_1]\,\mathbb{E}[X_2] = 12.25$ by independence, and
$\operatorname{Cov}[M, L] = 12.25 - \frac{161}{36}\cdot\frac{91}{36} = \frac{1225}{1296}$. Also: $M + L = X_1 + X_2$, so $\mathbb{E}[L] = 7 - \mathbb{E}[M]$.

The simulated values at $n = 100$ are off in the first decimal; at $10^6$ in the third. Learners' numbers
depend on their draw order; what is graded is the pattern (errors shrinking roughly like $1/\sqrt{n}$) and
the exact fractions.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Exact values by enumeration (all four), ideally as fractions | 4 |
| Vectorized simulation: one `(n, 2)` draw, `max`/`min` along `axis=1`, no Python loop over rows | 3 |
| Own estimators with `ddof=1`, checked against `np.cov` with `assert_allclose` | 3 |
| Table across the three sample sizes, compared with the exact values | 2 |
| (c) Sign of the covariance explained; $ML = X_1X_2$ shortcut found | 3 |
| Readability: type hints, docstrings with shapes, seed at the top | 1 |

**Likely mistakes:**

- Calling `np.cov(np.stack([m, low], axis=1))`: with `(n, 2)` input and the default `rowvar=True`,
  NumPy treats each of the $n$ rows as a variable and returns an `(n, n)` matrix (or runs out of memory
  at $n = 10^6$). Pass `rowvar=False` or stack as rows.
- Drawing the two dice with two different generators seeded identically, so die2 equals die1:
  $\operatorname{Cov}[M, L]$ comes out as $\operatorname{Var}[X] = 2.92$.
- Enumerating only the 21 unordered pairs and weighting them equally: $(1, 2)$ happens twice as often
  as $(1, 1)$.
- Comparing floats from the enumeration with `==` instead of a tolerance (or `Fraction`).

## Exercise 3 (think)

**Strong answer:** A correlation of 0.02 only says there is no *linear* trend between the two series
across the week; it does not show independence, and even independence would not make the error-rate
alert redundant. Ways it could be wrong: (1) the relationship can be non-linear or rare: most minutes
have low latency and zero errors, while a handful of incidents have both spike together; a few dozen
incident minutes out of 10,080 barely move a correlation computed over all of them, and an incident is
exactly when the alert matters. (2) Some failures raise errors *without* raising latency (fast failures:
a dependency returns 500 immediately, an auth bug rejects everything), so latency alerting would miss
them entirely; that is an argument for keeping the alert regardless of $\rho$. (3) p99 and error rate can
move in opposite directions in some incidents (timeouts counted as errors are removed from the latency
percentile) and in the same direction in others, cancelling in a single number. (4) One week may not
contain the failure modes that matter. To find out: plot the scatter and the two time series around
past incidents, compute the correlation within incident windows only, look at the post-mortems for
incidents caught by each alert, and check how many minutes have errors above threshold while latency is
normal. Decision: keep the error-rate alert; alerts exist to catch distinct failure modes, and the
question to ask is "which incidents would we miss without it", not "are the series correlated".

**Weak answer:** "Correlation 0.02 is basically zero, so the two metrics are independent and one alert
is enough." Weak because zero correlation is not independence, because the average over a quiet week
hides the rare minutes that matter, and because the purpose of an alert is coverage of failure modes,
which a correlation coefficient cannot measure.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| States that zero correlation measures only linear association, not independence | 3 |
| At least one concrete mechanism (rare joint spikes, fast failures, opposite movements) | 3 |
| A concrete check (scatter, incident windows, post-mortem coverage) | 2 |
| A clear decision tied to the purpose of the alert | 2 |

## Scoring

Total 40 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → A5.1
"The math" (variance rules, covariance linearity); exercise 2 → A5.1 "From scratch" and A3.3 (`np.cov`,
`rowvar`, `ddof`); exercise 3 → A5.1 "In practice" (zero correlation versus independence) and A6.1
(plot before you summarize).
