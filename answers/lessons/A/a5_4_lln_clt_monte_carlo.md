# Answer key: A5.4 Law of large numbers, central limit theorem and Monte Carlo estimation

Lesson: [`lessons/A/a5_4_lln_clt_monte_carlo.md`](../../../lessons/A/a5_4_lln_clt_monte_carlo.md)

## Exercise 1 (by hand)

**Solution (a).** A click is Bernoulli(0.3), $\sigma = \sqrt{0.3 \times 0.7} = \sqrt{0.21} = 0.458$. Standard error $0.458/\sqrt{n}$:
$0.0458$ at $n = 100$, $0.0046$ at $n = 10{,}000$. For a half-width of 0.01: $n = (1.96 \times 0.458/0.01)^2 = 8{,}067.4$, so
**8,068** impressions (round up). Without knowing $p$ in advance, the worst case $\sigma = 0.5$ gives 9,604.

**Solution (b).** Total CPU time $T = \sum_{i=1}^{400} X_i$: mean $400 \times 20 = 8{,}000$ ms, variance $400 \times 15^2 = 90{,}000$,
standard deviation 300 ms (variances add, not standard deviations). $z = (8{,}600 - 8{,}000)/300 = 2$, so
$P(T > 8.6\,\text{s}) \approx 1 - \Phi(2) = 0.0228$. The assumption to doubt first is independence: requests
share caches, garbage-collection pauses, noisy neighbours and traffic bursts, so their costs are
positively correlated, which makes the total's variance larger than $400\sigma^2$ and the tail probability
larger than 2.3%. (Second: heavy tails in per-request cost, which make $n = 400$ less "large".)

**Solution (c).** $x = 2u = 0.2, 0.6, 1.0, 1.4, 1.8$; $x^2 = 0.04, 0.36, 1.00, 1.96, 3.24$; mean $6.6/5 = 1.32$; times
$(b - a) = 2$: estimate $2.64$. Exact: $\int_0^2 x^2\,dx = 8/3 = 2.667$. (These $u$ are evenly spaced, so this is
really the midpoint rule; with random $u$ the error would typically be larger.)

```text
== exercise 1 check ==
(a) n =    100: standard error = sqrt(0.3 * 0.7 / n) = 0.0458
(a) n = 10,000: standard error = sqrt(0.3 * 0.7 / n) = 0.0046
(a) impressions for +-0.01: (1.96 * sqrt(0.21) / 0.01)^2 = 8067.4
(b) total: mean = 8000 ms   sd = 300 ms   z = 2.00   P(total > 8.6 s) = 0.0228
(c) x = [0.2, 0.6, 1.0, 1.4, 1.8]   x^2 = [0.04, 0.36, 1.0, 1.96, 3.24]   mean = 1.32   estimate = 2 * mean = 2.64   exact 8/3 = 2.6667
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Both standard errors and $n \approx 8{,}068$ (rounded up) | 4 |
| (b) Mean 8,000 ms, sd 300 ms from added variances, $z = 2$, $P \approx 0.023$ | 4 |
| (b) Names independence (correlated request costs) as the assumption to doubt, with a reason | 2 |
| (c) Transforms to $[0, 2]$, multiplies by $b - a = 2$, gets 2.64 versus 2.667 | 4 |

**Likely mistakes:**

- Adding standard deviations in (b): $400 \times 15 = 6{,}000$ ms, which makes the event look certain not to
  happen. Variances add; standard deviations add only for perfectly correlated parts.
- Forgetting the $(b - a)$ factor in (c) and reporting 1.32.
- Rounding $n$ down in (a): 8,067 gives a half-width slightly above 0.01.
- Using $\sigma = 0.3$ (the probability) instead of $\sqrt{p(1 - p)}$.

## Exercise 2 (code)

**Solution:** [`a5_4_lln_clt_monte_carlo_ex2.py`](a5_4_lln_clt_monte_carlo_ex2.py).

```text
== exercise 2 (a, b): Monte Carlo against quadrature ==
quad: E[M] = 1.538753   (density integrates to 1.000000; quad error estimate 1.9e-10)
         n   estimate  +-1.96 SE   |error|  covers
     1,000    1.53776    0.03493   0.00099    True
    10,000    1.53730    0.01157   0.00145    True
   100,000    1.54017    0.00364   0.00142    True
 1,000,000    1.53760    0.00115   0.00116   False

== exercise 2 (c): coverage at n = 1,000 ==
2000 runs: interval contained E[M] in 0.951 of runs

== exercise 2 (d): repetitions for 3 decimals ==
sd of M = 0.5873   n for a 95% half-width of 0.0005: 5,300,036
```

The density $10\,\varphi(x)\Phi(x)^9$ comes from $P(M \le x) = P(\text{all ten} \le x) = \Phi(x)^{10}$ and differentiating
(chain rule, A4.3). Checking that the density integrates to 1 is a cheap guard against a wrong formula.
One of four intervals in (a) misses, by a hair; that is the 5% the method allows, and (c) shows the
long-run coverage is 95.1%, as promised. (d) The standard deviation of $M$ is about 0.587, so a 95%
half-width of $0.0005$ (enough to trust the third decimal) needs $(1.96 \times 0.587/0.0005)^2 \approx 5.3$ million
repetitions: a second or two vectorized, but a `(5.3M, 10)` float64 array is 424 MB, so on a 15 GB laptop
it fits, and in chunks it is safer. Learners' estimates depend on their draws; the coverage near 0.95 and
error bars that contain 1.5388 most of the time are what is graded. A half-width of 0.001 (about 1.3
million) is also accepted if justified.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Vectorized `(n, 10)` draw, `max(axis=1)`, mean and standard error with `ddof=1` | 3 |
| Error bars at four sizes, shrinking about $\sqrt{10}\times$ per step | 2 |
| Quadrature with the correct max density, and a sanity check (area 1 or a known case) | 4 |
| Coverage experiment, result near 0.95 and interpreted | 3 |
| Sample-size calculation from the estimated sd, with a memory or time estimate | 2 |
| Readability: `SEED = 0`, type hints, docstrings, `~` on timing lines | 2 |

**Likely mistakes:**

- Using `np.max(draws)` without `axis=1`: one number, the maximum of everything.
- Density $\varphi(x)\Phi(x)^9$ without the factor 10 (it integrates to 0.1), or $\Phi(x)^{10}$ itself (a CDF, not a density).
- Calling one interval that misses "a bug": an interval that always contains the truth is too wide.
- Integrating `quad` over a finite range such as `[-5, 5]` is fine here, but `[0, inf]` silently drops the
  negative part of the maximum's distribution.

## Exercise 3 (think)

**Strong answer:** The interval $\bar{x} \pm 1.96\,s/\sqrt{n}$ assumes independent draws from one distribution
and that $n$ is large enough for the CLT; with heavy tails, neither may hold. Problems: (1) **heavy tails
and small $n$**: with a few 30 s timeouts among 200 requests, the mean and $s$ are dominated by those few
values; another run with one more or one fewer timeout changes the mean by $30{,}000/200 = 150$ ms, far
beyond ±12 ms, so the interval's coverage is much worse than 95%. Indeed, if the run had timeouts, an
interval of ±12 ms is suspicious: $s$ would be in the seconds. (2) **Correlation**: requests in a load test
share warm or cold caches, connection pools, GC pauses and autoscaling events, so the effective sample
size is smaller than 200. (3) **The wrong quantity**: users and SLOs care about percentiles and the timeout
rate, not the mean; a mean of 180 ms can hide a p99 of several seconds. (4) **Warm-up and drift**: the first
requests (cold start) and later ones come from different distributions. Report instead: p50, p95, p99
and the timeout/error rate, each with a bootstrap interval (A6.2), the number of requests, the test
duration and concurrency; plot the latency histogram on a log axis and latency over time. Change the
test: run longer (thousands of requests per configuration), discard or report a warm-up period
separately, repeat the whole test several times on different occasions and compare runs (between-run
variation is the honest error bar), and track timeouts as their own metric.

**Weak answer:** "The CLT says the mean is Gaussian for $n > 30$, so with 200 requests the ±12 ms interval
is reliable; maybe add more requests to make it smaller." Weak because it applies the "$n > 30$" rule to
heavy-tailed data, ignores correlation, and never questions whether the mean is the right metric.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Heavy tails make $n = 200$ too small / the interval unreliable, with a concrete reason | 3 |
| Correlation between requests reduces the effective sample size | 2 |
| Percentiles and timeout rate (with intervals) instead of or alongside the mean | 3 |
| A concrete change to the test (longer, warm-up, repeated runs) | 2 |

## Scoring

Total 40 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → A5.4
"The math" (standard error, CLT for sums, integrals as expectations) and A5.1 (variances add);
exercise 2 → A5.4 "From scratch" (Monte Carlo with error bars, coverage); exercise 3 → A5.4 "In practice"
and A6.2 (bootstrap intervals for percentiles).
