# Answer key: A6.2 Confidence intervals and the bootstrap

Lesson: [`lessons/A/a6_2_confidence_intervals_bootstrap.md`](../../../lessons/A/a6_2_confidence_intervals_bootstrap.md)

## Exercise 1 (by hand)

**Solution (a).** Mean $= 7.0/5 = 1.40$ s. Deviations $-0.2, -0.6, 0.1, -0.3, 1.0$; squares $0.04, 0.36, 0.01, 0.09, 1.00$, sum $1.50$.
$s = \sqrt{1.5/4} = \sqrt{0.375} = 0.612$; $\mathrm{SE} = 0.612/\sqrt5 = 0.274$. t interval: $1.40 \pm 2.776 \times 0.274 = 1.40 \pm 0.760 = [0.64, 2.16]$ s.
The $z$ interval would be $1.40 \pm 0.537$: its half-width is 29% smaller, so it would claim more precision
than five values can give.

**(b)** Indices $4, 0, 0, 2, 1$ give $2.4, 1.2, 1.2, 1.5, 0.8$: mean $7.1/5 = 1.42$, median (sorted $0.8, 1.2, 1.2, 1.5, 2.4$) $1.2$.
Indices $3, 3, 1, 4, 4$ give $1.1, 1.1, 0.8, 2.4, 2.4$: mean $7.8/5 = 1.56$, median (sorted $0.8, 1.1, 1.1, 2.4, 2.4$) $1.1$.

**(c)** The bootstrap SE of a mean is $\hat{\sigma}/\sqrt{n}$ with the divide-by-$n$ standard deviation:
$\hat{\sigma} = \sqrt{1.5/5} = 0.548$, so $0.548/\sqrt5 = 0.245$, against $s/\sqrt{n} = 0.274$ (ratio $\sqrt{4/5} = 0.894$).

**(d)** $\hat{p} = 0$. Wald: $0 \pm 1.96\sqrt{0 \times 1/120} = [0, 0]$, which claims certainty. Rule of three: upper bound
$3/120 = 0.025$. Wilson with $z^2/m = 3.8416/120 = 0.0320$: centre $= (0 + 3.8416/240)/1.0320 = 0.0155$, half-width
$= \frac{1.96}{1.0320}\sqrt{0 + 3.8416/(4 \times 120^2)} = 1.899 \times 0.00817 = 0.0155$, so $[0, 0.031]$. For the release notes: "0 failures in
120 attempts; the failure rate is very likely below about 3% (95% Wilson interval 0% to 3.1%)". Never
"0% failure rate".

```text
== exercise 1 check ==
(a) mean 1.4000   s 0.6124   SE 0.2739   t* 2.776   t interval [0.6396, 2.1604]   z half-width 0.5368 vs t half-width 0.7604
(b) indices [4, 0, 0, 2, 1] -> [2.4, 1.2, 1.2, 1.5, 0.8]   mean 1.42   median 1.20
(b) indices [3, 3, 1, 4, 4] -> [1.1, 1.1, 0.8, 2.4, 2.4]   mean 1.56   median 1.10
(c) exact bootstrap SE = sqrt(mean((x - mean)^2)) / sqrt(n) = 0.2449
(d) 0/120: Wald [0, 0]   rule of three 0.0250   Wilson [0.0000, 0.0310]
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Mean, $s$ with $n - 1$, SE, t interval $[0.64, 2.16]$, and the comparison with $z$ | 4 |
| (b) Both resamples written out with correct means and medians | 2 |
| (c) $\hat{\sigma}/\sqrt{n} = 0.245$ with the divide-by-$n$ variance, and why it is below $s/\sqrt{n}$ | 3 |
| (d) Wald $[0, 0]$ and why it is wrong; rule of three 0.025; Wilson $\approx [0, 0.031]$ | 4 |
| (d) A sensible release-note sentence that does not claim 0% | 1 |

**Likely mistakes:**

- Using $n$ instead of $n - 1$ for $s$ in (a), or $t^*$ for the wrong degrees of freedom ($n$ instead of $n - 1$).
- Treating the bootstrap indices as 1-based and picking the wrong values in (b).
- In (c), answering $s/\sqrt{n}$: the bootstrap resamples from the empirical distribution, whose variance divides by $n$.
- In (d), applying the Wald formula and reporting "0% ± 0%", or forgetting that the Wilson interval
  needs no successes to be informative.

## Exercise 2 (code)

**Solution:** [`a6_2_confidence_intervals_bootstrap_ex2.py`](a6_2_confidence_intervals_bootstrap_ex2.py).

```text
== exercise 2 (a): intervals for one accuracy ==
classifier 1: 419/500 correct = 0.8380
Wald      [0.8057, 0.8703]
Wilson    [0.8031, 0.8677]
bootstrap [0.8040, 0.8700]

== exercise 2 (b): difference between two classifiers on the same examples ==
accuracy 1 = 0.8380   accuracy 2 = 0.8580   agreement = 0.8840   observed difference = 0.0200
paired   (resample examples): [-0.0100, 0.0500]   SE 0.0150
unpaired (wrong):             [-0.0240, 0.0640]   SE 0.0226
corr of the two correctness vectors = 0.5520

== exercise 2 (c): coverage of the paired interval ==
500 repetitions: the paired 95% interval contained the true difference 0.02 in 0.956 of them   (Monte Carlo SE 0.010)
```

(a) At $n = 500$ and $p \approx 0.84$ all three intervals agree to about 0.003: Wald is fine away from 0 and 1 with this
many examples. (b) To get 90% agreement and a 0.02 gain, classifier 2 must differ from classifier 1 on 10%
of examples, being right on 6% where 1 was wrong and wrong on 4% where 1 was right ($0.06 - 0.04 = 0.02$). The
paired interval resamples *examples* and recomputes both accuracies on the same rows, which is how the
data was produced (both models scored the same test set). Its SE is 0.015 against 0.023 for the unpaired
one, because the two correctness vectors are strongly correlated (0.55): $\operatorname{Var}[a_2 - a_1] = \operatorname{Var}[a_2] + \operatorname{Var}[a_1] - 2\operatorname{Cov}[a_2, a_1]$
(A5.1), and the unpaired resampling throws the covariance away. Hard examples are hard for both models;
pairing cancels that shared difficulty. Both intervals include 0 here, so 500 examples cannot settle a
0.02 gain; the paired one is the honest and the more precise one. (c) Coverage 0.956 is within noise of
0.95. Learners' exact numbers depend on how they built the second classifier; the conclusions are what
is graded.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| (a) Wald, Wilson and bootstrap intervals, correctly computed | 3 |
| (b) A second classifier with ~90% agreement and ~0.88 accuracy, built from the first | 3 |
| (b) Paired bootstrap with shared indices, and the unpaired version for contrast | 4 |
| (b) Explains why paired is narrower (covariance, shared example difficulty) | 3 |
| (c) Coverage simulation near 0.95, with its Monte Carlo error | 2 |
| Readability: vectorized `(B, n)` resampling, `SEED = 0`, docstrings with shapes | 1 |

**Likely mistakes:**

- Flipping a random 10% of classifier 1's results: that makes classifier 2 *worse* (about 0.79), not 0.88.
- Drawing classifier 2 independently with `rng.random(n) < 0.88`: agreement then is about 0.77, not 0.9,
  and the point of pairing disappears.
- Concluding "classifier 2 is better" because its point estimate is higher, or "no difference" because
  the interval includes 0.

## Exercise 3 (think)

**Strong answer:** A p99 from 300 requests is decided by the 3 slowest requests, so it is very noisy: the
bootstrap of such a p99 has only a handful of distinct values and an interval that easily spans ±30%
or more. An alert on a 20% hour-over-hour rise will fire constantly on noise for small services, and
people will learn to ignore it (alert fatigue), which is worse than no alert. Also, comparing two noisy
numbers doubles the noise: the difference of two hourly p99s varies more than either. Instead: (1)
compute p99 over a window with enough requests (for example 3,000+ requests, so about 30 values lie
above the p99; a rolling window rather than a fixed hour); (2) for small services alert on a less extreme
percentile (p90/p95) or on the share of requests above an SLO threshold, a proportion with a Wilson
interval and a well-understood noise level; (3) alert when the interval of the new window lies above the
SLO or the baseline, not when the point estimate moves; (4) require the condition to persist (two
windows in a row) and set thresholds from historical false-alarm rates (backtest the rule on past data).
Report p99 with its interval and request count on the dashboard so people see the noise.

**Weak answer:** "A 20% rise in p99 is significant, so the alert is fine; maybe use 30% to be safe." Weak
because it never asks how noisy a p99 from 300 requests is; changing the threshold without measuring
the noise just trades one arbitrary false-alarm rate for another.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Explains that a p99 from ~300 requests rests on ~3 values and is very noisy | 3 |
| Predicts frequent false alarms (and alert fatigue), and that differences of noisy numbers are noisier | 2 |
| Proposes a better quantity (larger windows, lower percentile, share above an SLO) | 3 |
| Proposes interval-based or persistence-based alerting, or backtesting the rule | 2 |

## Scoring

Total 40 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → A6.2 "The
math" (t interval, bootstrap SE, Wilson); exercise 2 → A6.2 "From scratch" and A5.1 (variance of a
difference); exercise 3 → A6.2 "In practice" (extreme statistics) and J2.2 (alerting).
