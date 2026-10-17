# Answer key: A6.5 Bayesian inference

Lesson: [`lessons/A/a6_5_bayesian_inference.md`](../../../lessons/A/a6_5_bayesian_inference.md)

## Exercise 1 (by hand)

**Solution (a).** Add successes to $a$ and failures to $b$: Beta(3 + 9, 7 + 11) = Beta(12, 18). Mean $12/30 = 0.4$. Mode
$(12 - 1)/(12 + 18 - 2) = 11/28 = 0.3929$. MLE $9/20 = 0.45$. The prior is worth $a + b = 10$ pseudo-observations against
$n = 20$ real ones, so the weight on the prior mean is $10/30 = 1/3$: $\frac13 \times 0.3 + \frac23 \times 0.45 = 0.1 + 0.3 = 0.4$. The posterior
mean sits between the prior's 0.3 and the data's 0.45, closer to the data because there is more of it.

**(b)** The posterior predictive probability of one success is the posterior mean: 0.4 (average $p$ over the
posterior, $\int p\,\pi(p \mid \text{data})\,dp = \mathbb{E}[p]$). Not the MLE 0.45, and not the mode.

**(c)** X: Beta(4 + 1, 1 + 0) = Beta(5, 1), mean $5/6 = 0.8333$. Y: Beta(4 + 360, 1 + 40) = Beta(364, 41), mean $364/405 = 0.8988$.
Y ranks higher. With raw rates X (1/1 = 100%) would rank first on a single review; the prior pulls X's estimate
towards the prior mean $4/5 = 0.8$ because one review is weak evidence, while Y's 400 reviews barely move.

**(d)** A 95% confidence interval comes from a procedure that, over repeated samples, covers the fixed true value
95% of the time; this particular interval either contains it or not. A 95% credible interval says that, given the
prior, the model and this data, the parameter has probability 0.95 of lying in [0.31, 0.52]: a direct probability
statement, conditional on the prior.

```text
== exercise 1 check ==
(a) posterior Beta(12, 18)   mean 0.4000   mode 0.3929   MLE 0.4500
(a) weight on prior 0.3333:  0.3333 * 0.30 + 0.6667 * 0.45 = 0.4000
(b) P(next email opened) = posterior mean = 0.4000
(c) X: Beta(4, 1) + 1/1 -> Beta(5, 1)   mean 0.8333
(c) Y: Beta(4, 1) + 360/400 -> Beta(364, 41)   mean 0.8988
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Beta(12, 18), mean 0.4, mode 0.393, MLE 0.45 | 3 |
| (a) Weight 1/3 from pseudo-counts 10 versus 20, and the weighted average checked | 2 |
| (b) 0.4, with the reason (predictive = posterior mean) | 2 |
| (c) Both posteriors and means; Y ranks higher; why X is pulled to 0.8 | 4 |
| (d) Long-run coverage of a procedure versus probability given prior and data | 3 |

**Likely mistakes:**

- Adding $n$ instead of the failures $n - k$ to $b$ (Beta(12, 27)).
- Using the mode formula for the mean, or $(a + k)/(a + b + n - 2)$.
- In (b), answering 0.45 (the plug-in MLE ignores the prior you just used).
- In (d), saying a confidence interval "has a 95% chance of containing the true value": that is the credible
  interval's reading, and it is exactly the confusion the exercise is about.

## Exercise 2 (code)

**Solution:** [`a6_5_bayesian_inference_ex2.py`](a6_5_bayesian_inference_ex2.py).

```text
== exercise 2(a): Bayesian A/B versus the z-test ==
P(p_B > p_A) = 0.9507   mean lift 0.0249   95% credible [-0.0047, 0.0545]
one-sided z-test: z = 1.649   p = 0.0496   1 - p = 0.9504

== exercise 2(b): exact P(p_B > p_A) on a grid ==
exact 0.9502   Monte Carlo 0.9507   difference +0.0005   MC standard error 0.0005
OK: within 4 standard errors

== exercise 2(c): A/A experiments ==
P(p_B > p_A) > 0.95 in 63 of 1000 A/A experiments = 0.063   (one-sided alpha 0.05; +-2 se = [0.036, 0.064])
```

(a) With flat priors and 1,000 users per arm, $P(p_B > p_A) = 0.95$ is almost exactly $1 - p$ of the one-sided z-test
(0.9504). That is not a coincidence: with flat priors and large samples both posteriors are close to normal with the
same centres and spreads that the z-test uses, so the two calculations are nearly the same integral. They answer
different questions ("probability B is better, given this data and a flat prior" versus "how surprising is this data
if A = B"), but here they give the same number. The credible interval for the lift includes 0, matching a two-sided
test that does not reject at 0.05.

(b) The exact value is $\int_0^1 f_B(x)\,F_A(x)\,dx$: for each possible $p_B = x$, the probability that $p_A$ is below it,
weighted by how likely that $x$ is. Monte Carlo agrees within one standard error,
$\sqrt{0.95 \times 0.05 / 200{,}000} = 0.0005$.

(c) About 5% of A/A experiments (here 6.3%, inside the ±2 standard-error band 0.036 to 0.064 for a true 5%) cross
0.95, although there is no difference. A threshold of $P(B > A) > 0.95$ with flat priors behaves like a one-sided test
at $\alpha = 0.05$: it does not escape false positives, and peeking at it repeatedly inflates them just as it does
for p-values.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| `ab_bayes` correct: posteriors Beta(a + k, b + n − k), lift = draws of $p_B - p_A$, quantile interval | 4 |
| (a) Values printed and compared with the one-sided p-value; explains why they nearly agree | 3 |
| (b) Exact value by integration or grid; checked against Monte Carlo with its standard error | 4 |
| (c) A/A simulation as specified; about 5%; related to a one-sided $\alpha = 0.05$ | 4 |
| Readability: shapes in comments, `SEED = 0`, one generator passed through | 1 |

**Likely mistakes:**

- Sorting the A and B draws separately before subtracting (or comparing their quantiles): the lift needs the draws
  paired as drawn, independently.
- Comparing with the two-sided p-value (0.099) and concluding the Bayesian answer is "twice as confident".
- Checking Monte Carlo against the exact value without a tolerance, or with an absolute tolerance unrelated to
  the number of draws.
- In (c), drawing new A/A data but reusing the same posterior draws, or creating a new generator with `SEED` inside
  the loop (every experiment then gets the same random numbers).

## Exercise 3 (think)

**Strong answer:** Centre the prior on a measured rate, not a guess: the positive-feedback rate of sellers in their
first months, ideally in the same category (an empirical-Bayes prior, fitted to past data, for example by matching
the mean and variance of established sellers' early rates). Its strength $a + b$ is how many reviews it is worth;
choose it from how much sellers' true rates actually vary (a small spread between sellers justifies a strong prior),
and keep it modest (of the order of 10 to 20 reviews) so that real evidence takes over quickly. Effects on new sellers:
a genuinely excellent seller is ranked below established sellers with the same true rate until their reviews
accumulate, which costs them sales and therefore reviews, a feedback loop that can keep good newcomers down; a
genuinely bad seller is protected by the prior and shown to more buyers for longer, which costs buyers. If the prior
centre comes from sellers who survived, it is biased upwards (survivorship bias). One prior for every category is
wrong when categories differ (electronics may have lower rates than books); use per-category priors, but check that
small categories have enough data and that the category itself is not a proxy for a protected attribute of sellers
(region, language). To check fairness: simulate sellers with known true rates and measure how many reviews it takes
for each to reach their correct rank; compare ranking errors for new versus established sellers on held-out later
data; monitor exposure and sales of new sellers over time; and consider an exploration policy (showing new sellers
to some buyers on purpose) so the ranking can learn about them.

**Weak answer:** "Use Beta(1, 1) so it is neutral, and the ranking will be fair because the prior is the same for
everyone." Weak because Beta(1, 1) centres new sellers on 50%, far below typical rates, so it pushes them down; "the
same prior" is not the same as "fair outcomes"; and it says nothing about strength, categories or how to check.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Prior centre from data on comparable (new) sellers; strength chosen and justified as pseudo-counts | 3 |
| Effects on excellent and bad new sellers, including the exposure feedback loop | 3 |
| Per-category priors, with the small-category and proxy caveats | 2 |
| A concrete check: simulation, held-out comparison of new versus established, exposure monitoring | 2 |

## Scoring

Total 40 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → A6.5 "The math"
(conjugate update, mean as a weighted average, predictive) and A6.2 (what a confidence interval means); exercise 2
→ A6.5 "From scratch" (Bayesian A/B) and A6.3b (one-sided tests, peeking); exercise 3 → A6.5 "In practice" and
A6.1 (confounding by category).
