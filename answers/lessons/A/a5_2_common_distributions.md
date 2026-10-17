# Answer key: A5.2 Common distributions

Lesson: [`lessons/A/a5_2_common_distributions.md`](../../../lessons/A/a5_2_common_distributions.md)

## Exercise 1 (by hand)

**Solution (a).** $\binom52 = \frac{5 \cdot 4}{2} = 10$; $P(X = 2) = 10 \times 0.3^2 \times 0.7^3 = 10 \times 0.09 \times 0.343 = 0.3087$.
Mean $np = 1.5$, variance $np(1 - p) = 1.5 \times 0.7 = 1.05$.

**Solution (b).** $P(0) = e^{-1.5} = 0.2231$. $P(X \ge 1) = 1 - P(0) = 0.7769$ (the complement is one term instead
of infinitely many). $P(1) = 0.2231 \times 1.5 = 0.3347$, $P(2) = 0.3347 \times \frac{1.5}{2} = 0.2510$, so
$P(X \le 2) = 0.2231 + 0.3347 + 0.2510 = 0.8088$.

**Solution (c).** Running totals $(0.1, 0.7, 1.0)$; return the first class with $u < F_j$. $u = 0.05$ → class 0;
$0.65$ → class 1; $0.71$ → class 2; $u = 0.1$ → class 1, because $0.1 < 0.1$ is false. (Over continuous $u$ the
boundary has probability zero, so either convention is correct as long as it is applied consistently;
the code's `side="right"` is this one.)

**Solution (d).** $L_{11} = \sqrt1 = 1$, $L_{21} = 0.6/1 = 0.6$, $L_{22} = \sqrt{1 - 0.36} = 0.8$:
$\mathbf{L} = \begin{pmatrix}1 & 0\\0.6 & 0.8\end{pmatrix}$. $\mathbf{x} = \mathbf{L}\mathbf{z} = (1,\ 0.6 + 0.8) = (1, 1.4)$. Check:
$\mathbf{L}\mathbf{L}^\top = \begin{pmatrix}1 & 0.6\\0.6 & 0.36 + 0.64\end{pmatrix} = \boldsymbol{\Sigma}$ ✓.

```text
== exercise 1 check ==
(a) P(X = 2) = 0.3087   mean = 1.50   variance = 1.05
(b) P(0) = 0.2231   P(>= 1) = 0.7769   P(<= 2) = 0.8088
(c) cumulative = [0.1, 0.7, 1.0]  classes for u = 0.05, 0.65, 0.71, 0.1: [0, 1, 2, 1]
(d) L = [[1.0, 0.0], [0.6, 0.8]]   L z for z = (1, 1): [1.0, 1.4]   L L^T = [[1.0, 0.6], [0.6, 1.0]]
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) $0.3087$ with $\binom52 = 10$ shown; mean 1.5, variance 1.05 | 3 |
| (b) $P(0)$, the complement for $P(\ge 1)$, and $P(\le 2) = 0.809$ | 4 |
| (c) All four classes, including the boundary case with a stated rule | 3 |
| (d) $\mathbf{L}$ correct, $\mathbf{x} = (1, 1.4)$, and the $\mathbf{L}\mathbf{L}^\top$ check | 4 |

**Likely mistakes:**

- Forgetting $\binom{n}{k}$ in (a) and reporting $0.3^2 \times 0.7^3 = 0.0309$: that is the probability of one specific
  order of successes, not of "exactly two".
- Computing $P(X \ge 1)$ as $P(1)$.
- Writing $\mathbf{x} = \mathbf{L}^\top\mathbf{z} = (1.6, 0.8)$ in (d): the right factor multiplies from the left; with
  $\mathbf{L}^\top$ the sample has covariance $\mathbf{L}^\top\mathbf{L}$, which is a different matrix.
- Taking $L_{22} = 1 - 0.36$ (forgetting the square root).

## Exercise 2 (code)

**Solution:** [`a5_2_common_distributions_ex2.py`](a5_2_common_distributions_ex2.py). The Poisson sampler is
the categorical sampler applied to a pmf table built with $P(k) = P(k - 1)\cdot\lambda/k$; the exponential is the
inverse CDF of $F(t) = 1 - e^{-\text{rate}\cdot t}$ (solve $u = 1 - e^{-\text{rate}\cdot t}$ for $t$, and note $1 - u$ is uniform too).

```text
== exercise 2 (a, b): samplers ==
Poisson(4): mean = 3.9975   variance = 3.9995   (exact 4, 4)
Exponential(4): mean = 0.2488   variance = 0.0620   (exact 0.25, 0.0625)

== exercise 2 (c): counts from exponential gaps ==
  k     gaps  sampler    scipy
  0   0.0182   0.0178   0.0183
  1   0.0731   0.0751   0.0733
  2   0.1467   0.1452   0.1465
  3   0.1941   0.1945   0.1954
  4   0.1962   0.1963   0.1954
  5   0.1556   0.1585   0.1563
  6   0.1040   0.1034   0.1042
  7   0.0608   0.0578   0.0595
  8   0.0304   0.0298   0.0298
gap counts vs scipy    largest gap = 0.0013
sampler vs scipy       largest gap = 0.0022
both below 3/sqrt(n) = 0.0095: OK
```

Events with exponential gaps form a **Poisson process**: the number of events in an interval of length 1
is Poisson with the same rate. Learners' exact numbers depend on draw order; the agreement within
$3/\sqrt{n}$ is what is graded. A loop-based version (draw gaps one at a time until the time passes 1) is
correct but slow; 100,000 repeats should still finish in seconds if vectorized as here.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Poisson pmf table by recurrence, truncated by tail mass (not a fixed $k_{\max}$ chosen by eye) | 3 |
| Inverse-CDF lookup with `searchsorted` and the last-entry guard; only `gen.random()` used | 3 |
| Exponential sampler avoids `log(0)`; mean and variance checked | 2 |
| Counting events in $[0, 1)$ from cumulative gaps, with enough gaps per row (checked) | 3 |
| Three-way comparison with a stated, justified tolerance | 3 |
| Readability: type hints, docstrings with shapes, `SEED = 0`, one generator passed around | 2 |

**Likely mistakes:**

- `-np.log(gen.random(size))`: `gen.random()` can return exactly 0.0, giving `inf`. Rare, but a
  sampler that occasionally returns infinity is a production bug. Use `1 - gen.random()`.
- Computing the pmf with `rate**k / factorial(k)` for large $k$: overflows or loses precision for large
  rates; the recurrence avoids both.
- Using rate as the exponential's *mean* (`-log(u) * rate`): gaps of mean 4 instead of 0.25, giving a
  Poisson(0.25) count.
- A fixed number of gaps per row that is too small (say 8 for rate 4): rows that run out before time 1
  undercount the tail, and the gap check fails at $k \ge 8$.

## Exercise 3 (think)

**Strong answer:**

| Situation | Distribution | Assumption likely to fail | How to detect it |
|---|---|---|---|
| (i) 5xx per minute | Poisson (rate $\lambda$ per minute) | Constant rate and independence: errors come in bursts during incidents and deploys | Variance far above the mean (overdispersion); clusters in the time series; more zero-minutes *and* more huge minutes than Poisson predicts |
| (ii) Ad click | Bernoulli($p$) | One $p$ for everyone: click rates vary by user, placement and time; one user's clicks are correlated | Per-segment rates that differ a lot; variance of clicks per user above $np(1 - p)$ |
| (iii) Daily revenue | Not Gaussian: positive and right-skewed (log-normal is a common start) | Gaussian symmetry and thin tails; weekly seasonality; occasional huge orders | Histogram on a log axis; skew; mean far above the median; day-of-week pattern |
| (iv) Plan choice | Categorical($p_1, \dots, p_4$) | Fixed probabilities: they shift with pricing pages, campaigns, region | Compare plan shares across time windows and segments (a chi-square test, A6.3a) |
| (v) Sensor noise | Gaussian($0$, $\sigma^2$) | Mean zero (sensor drift/bias) and constant $\sigma$ (noise grows with temperature); occasional outliers from glitches | Compare with a reference sensor: mean residual not 0, residual spread changing with the reading, residuals beyond 4σ |

The common thread: name the distribution from the process, then name the assumption in it (constant
rate, independence, symmetry, fixed probabilities), because that is what production violates.

**Weak answer:** "(i) Poisson, (ii) Bernoulli, (iii) Gaussian, (iv) categorical, (v) Gaussian." Weak
because it picks Gaussian for a skewed positive quantity and lists no failing assumption or detection
method, which is the actual question.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Five reasonable distributions; revenue not modelled as Gaussian without comment | 3 |
| A specific, plausible failing assumption for each | 4 |
| A concrete detection method for each (statistic or plot) | 3 |

## Scoring

Total 40 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → A5.2
"The math" (binomial, Poisson, inverse CDF, multivariate sampling) and A3.4 (Cholesky); exercise 2 →
A5.2 "From scratch"; exercise 3 → A5.2 "In practice" and A6.1 (looking at distributions in data).
