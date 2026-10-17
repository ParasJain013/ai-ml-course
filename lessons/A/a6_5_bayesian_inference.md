# A6.5 Bayesian inference

**Depth** · Part A, week 6 (course week 06) · ~1 h · Lab:
[`labs/A/a6_5_bayesian_inference.py`](../../labs/A/a6_5_bayesian_inference.py)

## Why it matters

A new product has one review, five stars; another has 400 reviews averaging 4.7. Which should rank higher?
A new ad has 2 clicks in 3 impressions; is its click rate 67%? Maximum likelihood (A5.5) says yes, and is
badly wrong. **Bayesian inference** treats the unknown rate as a quantity with a probability distribution,
starts from a **prior** belief, and updates it with the data into a **posterior**. The result is a whole
distribution: an estimate that is sensible with little data, an interval with the direct reading people
want ("95% probability the rate is between…"), and answers to questions like "how likely is B better than
A?". This lesson does it for the one case with a clean formula, the Beta–Binomial model, which already
covers click rates, conversion rates, error rates and ratings. It is a Depth topic: skip it if the week
runs long, but come back before B2.4 (naive Bayes) and Part H.

## Prerequisite check

- Can you apply Bayes' theorem, $P(H \mid E) = P(E \mid H)P(H)/P(E)$, and explain what the prior and the likelihood are? If not,
  revisit [A5.3](a5_3_bayes_theorem.md).
- Can you write the binomial pmf and say what $\binom{n}{k}$ counts? If not, revisit [A5.2](a5_2_common_distributions.md).
- Do you know the MLE of a Bernoulli rate ($k/n$) and what a MAP estimate is? If not, revisit
  [A5.5](a5_5_mle_map.md).
- Can you say what a 95% confidence interval does and does not mean? If not, revisit
  [A6.2](a6_2_confidence_intervals_bootstrap.md).

## Intuition

### A belief that updates

In A5.3 the unknown was a yes/no fact (has the disease or not), and Bayes' theorem turned a prior
probability into a posterior one. Now the unknown is a *number*, a conversion rate $p$ somewhere between 0 and
1. Instead of one prior probability we need a prior *density* over all values of $p$: which rates are plausible
before seeing data. Bayes' theorem then works value by value: for every candidate $p$, multiply its prior
density by the likelihood of the data at that $p$ (how probable the observed clicks would be if the rate were
$p$), and rescale so the result integrates to 1. Rates that explain the data well gain weight; rates that
explain it badly lose it.

### Pseudo-counts

For a rate, there is a choice of prior that makes this arithmetic trivial: the **Beta distribution**.
A Beta($a$, $b$) prior behaves as if you had already seen $a$ successes and $b$ failures: the posterior mean and the
update rule are exactly those of adding $a$ and $b$ to your counts (the math below shows it). Observing $k$ successes in $n$ trials then gives a Beta($a + k$, $b + n - k$)
posterior: add the new successes to $a$ and the new failures to $b$. The prior is a set of **pseudo-counts**,
and updating is counting. A prior and likelihood that combine into a posterior of the same family like this
are called **conjugate**.

The posterior mean $(a + k)/(a + b + n)$ is a compromise between the prior's guess and the data's: with 0 data it is
the prior mean; with a lot of data it approaches $k/n$, the MLE. The prior's total $a + b$ says how many
observations' worth of conviction it carries. Two clicks in three impressions with a Beta(1, 1) prior
gives $(1 + 2)/(2 + 3) = 0.6$, not 0.67; with a Beta(2, 18) prior (an expected 10% click rate, worth 20
impressions), it gives $(2 + 2)/(20 + 3) = 0.17$.

> **Bridge:** this is how you would sort products by rating if you thought about it as an engineer: seed
> every product's counters with a few "average" ratings, so one five-star review cannot put a new product
> at the top, and the seed washes out as real reviews arrive. That seed is a prior. **Where it breaks:** the
> seed is a modelling decision with consequences: a seed worth 1,000 reviews makes the ranking ignore real
> data for months. The arithmetic also assumes every observation comes from the same fixed rate; if the
> rate drifts (seasonality, a product change), old counts mislead and you need to discount them.

### A different kind of interval

The posterior is a distribution over $p$, so you can read probabilities off it directly: "given the data and
the prior, there is a 95% probability that $p$ lies between 0.29 and 0.90" (a **credible interval**), or "a 77%
probability that $p > 0.5$". A confidence interval (A6.2) cannot be read this way; its 95% is about the method.
The price is the prior: different priors give different answers, so the prior must be stated and
defended. With a lot of data and a mild prior, the two kinds of interval nearly coincide.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\theta$ | the unknown rate (a probability), treated as a random variable | scalar in $[0, 1]$ |
| $\mathcal{D}$ | the data: $k$ successes in $n$ independent trials | — |
| $p(\theta)$ | prior density | function |
| $p(\mathcal{D} \mid \theta)$ | likelihood (binomial) | function of $\theta$ |
| $p(\theta \mid \mathcal{D})$ | posterior density | function |
| $a, b$ | Beta parameters, both > 0 (pseudo-counts) | scalars |
| $B(a, b)$ | the Beta function, the normalizing constant of the Beta density | scalar |
| $\propto$ | "proportional to": equal up to a factor that does not depend on $\theta$ | — |

The intuition above called the rate $p$; from here on it is $\theta$, so that $p(\cdot)$ can stay the name of a density, as in the
style guide. The lab's printed output keeps the everyday names (`p`, `p_A`, `p_B`) for the rate.

**Bayes' theorem for a parameter.**

$$p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\,p(\theta)}{p(\mathcal{D})}, \qquad p(\mathcal{D}) = \int_0^1 p(\mathcal{D} \mid \theta)\,p(\theta)\,d\theta.$$

The denominator does not depend on $\theta$; it only makes the posterior integrate to 1. So the working form is
**posterior ∝ likelihood × prior**, followed by normalizing.

**The Beta distribution.** For $\theta \in [0, 1]$,

$$p(\theta) = \frac{\theta^{a-1}(1 - \theta)^{b-1}}{B(a, b)}, \qquad \mathbb{E}[\theta] = \frac{a}{a + b}, \qquad \text{mode} = \frac{a - 1}{a + b - 2}\ (a, b > 1), \qquad \operatorname{Var}[\theta] = \frac{ab}{(a + b)^2(a + b + 1)}$$

(moments stated, not derived). Beta(1, 1) is the uniform distribution on $[0, 1]$; Beta(2, 2) is a gentle hump at 0.5;
Beta(20, 20) is a narrow hump at 0.5; Beta(2, 18) is concentrated near 0.1. The variance shrinks as $a + b$ grows: more
pseudo-counts, more certainty.

**Conjugate update.** The binomial likelihood is $\binom{n}{k}\theta^k(1 - \theta)^{n-k}$ (A5.2). Multiply by the prior and drop every factor that does
not involve $\theta$ (the binomial coefficient and $B(a, b)$):

$$p(\theta \mid \mathcal{D}) \propto \theta^k(1 - \theta)^{n-k}\cdot\theta^{a-1}(1 - \theta)^{b-1} = \theta^{(a + k) - 1}(1 - \theta)^{(b + n - k) - 1}.$$

The right side has exactly the shape of a Beta($a + k$, $b + n - k$) density, and a density's shape fixes its normalizing
constant, so

$$\theta \mid \mathcal{D} \sim \mathrm{Beta}(a + k,\ b + n - k).$$

Exponents add, so updating one observation at a time or all at once gives the same posterior.

**Posterior mean as a weighted average.** Write the posterior mean with $w = \frac{a + b}{a + b + n}$:

$$\frac{a + k}{a + b + n} = \frac{a + b}{a + b + n}\cdot\frac{a}{a + b} + \frac{n}{a + b + n}\cdot\frac{k}{n} = w \cdot \text{prior mean} + (1 - w) \cdot \text{MLE}.$$

(Check: the first product is $\frac{a}{a + b + n}$, the second $\frac{k}{a + b + n}$; they add to the left side.) The weight on the prior is its share of
all the counts, real and pseudo. This pull of the estimate toward the prior is called **shrinkage**; ridge
regression's pull toward 0 (A5.5) is the same idea.

**MAP.** The posterior mode $\frac{a + k - 1}{a + b + n - 2}$ is the MAP estimate of A5.5. With the flat prior Beta(1, 1) it is $k/n$: MAP equals
MLE, because a flat prior adds nothing to the log-likelihood.

**Credible interval.** The equal-tailed 95% credible interval runs from the 2.5th to the 97.5th percentile of the
posterior: `stats.beta(a + k, b + n - k).ppf([0.025, 0.975])`.

**Posterior predictive.** The probability that the *next* trial succeeds averages the success probability $\theta$ over
the posterior: $P(\text{next} = 1 \mid \mathcal{D}) = \int_0^1 \theta\,p(\theta \mid \mathcal{D})\,d\theta = \mathbb{E}[\theta \mid \mathcal{D}] = \frac{a + k}{a + b + n}$. With a flat prior this is $\frac{k + 1}{n + 2}$, Laplace's "rule of
succession". The number of successes in the next $m$ trials follows the **beta-binomial** distribution: draw $\theta$ from the
posterior, then draw a Binomial($m$, $\theta$). It is wider than plugging in a single estimate of $\theta$, because it includes
the uncertainty about $\theta$.

**When there is no formula.** With any other prior, compute the posterior on a grid: evaluate prior × likelihood at
thousands of values of $\theta$ and divide by the sum times the grid spacing. That is the from-scratch method in the lab,
and it works for any prior on one parameter. For many parameters, grids are too big, and practice moves to
sampling methods (Markov chain Monte Carlo) or approximations (variational inference, G1.2).

## Worked example by hand

**Prior Beta(2, 2), data 3 successes in 4 trials.**

1. Posterior: Beta($2 + 3$, $2 + 1$) = Beta(5, 3).
2. Posterior mean: $5/8 = 0.625$. Prior mean $2/4 = 0.5$, MLE $3/4 = 0.75$. Weight on the prior $w = 4/(4 + 4) = 0.5$, so
   $0.5 \times 0.5 + 0.5 \times 0.75 = 0.625$ ✓.
3. MAP: $(5 - 1)/(5 + 3 - 2) = 4/6 = 0.667$, between the posterior mean and the MLE.
4. Probability that the next trial succeeds: 0.625.
5. Sequentially: start at Beta(2, 2); observe 1 → Beta(3, 2); 0 → Beta(3, 3); 1 → Beta(4, 3); 1 → Beta(5, 3). The same posterior.
6. From the lab: the 95% credible interval is $[0.29, 0.90]$ (four observations say little), and $P(\theta > 0.5 \mid \mathcal{D}) = 0.77$.

```text
== worked example by hand ==
prior Beta(2, 2): mean 0.5000   data 3/4: MLE 0.7500
posterior Beta(2 + 3, 2 + 1) = Beta(5, 3): mean 5/8 = 0.6250   mode (MAP) 4/6 = 0.6667
weighted average: 0.50 * prior mean 0.5 + 0.50 * MLE 0.75 = 0.6250
P(next trial succeeds) = posterior mean = 0.6250
95% credible interval: [0.2904, 0.9010]   P(p > 0.5 | data) = 0.7734
  after observing 1: Beta(3, 2)
  after observing 0: Beta(3, 3)
  after observing 1: Beta(4, 3)
  after observing 1: Beta(5, 3)
```

## Key plot

![Left: posterior densities of the conversion rate under a Beta(2, 2) prior after 0, 1, 5, 20, 50 and 200 observations; the flat hump of the prior becomes a tall narrow peak near 0.27, close to the true rate 0.3. Right: on a log x-axis, the MLE k/n jumps between 0 and 0.75 in the first few observations while the posterior means under Beta(1, 1) and Beta(2, 2) move less, and the Beta(20, 20) posterior mean stays near 0.5 longest; all converge towards 0.3 by 200 observations](img/a6_5_beta_binomial_updates.png)

Left: each observation narrows the posterior; after 200 the plausible rates span about 0.2 to 0.33. Right: the
MLE starts at 0 (first observation a failure) and swings to 0.75; the posterior means start at the prior's 0.5
and move only as far as the data justifies. A stronger prior (Beta(20, 20)) resists longer.

## From scratch in NumPy

**Grid posterior.** Prior times likelihood on 10,001 values of $\theta$, normalized. The likelihood is computed in logs
and shifted by its maximum before exponentiating, because $\theta^{53}(1 - \theta)^{147}$ underflows to 0 for most $\theta$ (A3.4):

```python
def grid_posterior(prior: FloatArray, k: int, n: int) -> FloatArray:
    """Normalized posterior density on theta. prior (N_GRID,) -> (N_GRID,)."""
    log_lik = special.xlogy(k, theta) + special.xlogy(n - k, 1 - theta)  # log of p^k (1-p)^(n-k), 0 log 0 = 0
    unnorm = prior * np.exp(log_lik - log_lik.max())  # subtract the max: no underflow for large n
    return unnorm / (unnorm.sum() * d_theta)  # (N_GRID,) integrates to 1
```

```text
== from scratch: grid posterior ==
3/4 with Beta(2, 2): largest |grid - Beta(5, 3) density| = 5.33e-15
  grid mean 0.6250   grid 95% interval [0.2904, 0.9010]
all 200 observations, flat prior: largest |grid - Beta(54, 148)| = 3.62e-13
  first 10 observations (3/10), flat prior: mean 0.3333   95% interval [0.1092, 0.6097]
  first 10 observations (3/10), step prior, p > 0.5 3x as plausible: mean 0.3771   95% interval [0.1161, 0.6593]
```

The grid reproduces the conjugate formula to rounding error, which confirms the derivation. The last two lines use
a prior with no formula (rates above 0.5 three times as plausible as rates below): with only 10 observations it
moves the posterior mean from 0.33 to 0.38. The grid handles any one-parameter prior.

**Posterior versus MLE as data arrives.** 200 simulated observations with true rate 0.3, and the posterior mean
under three priors at checkpoints:

```text
== from scratch: posterior versus MLE as data arrives ==
   n   k     MLE    Beta(1, 1)    Beta(2, 2)  Beta(20, 20)  95% width (flat)
   0   0       -        0.5000        0.5000        0.5000            0.9500
   1   0  0.0000        0.3333        0.4000        0.4878            0.8293
   2   1  0.5000        0.5000        0.5000        0.5000            0.8114
   5   3  0.6000        0.5714        0.5556        0.5111            0.6591
  10   3  0.3000        0.3333        0.3571        0.4600            0.5005
  20   7  0.3500        0.3636        0.3750        0.4500            0.3886
  50  12  0.2400        0.2500        0.2593        0.3556            0.2317
 100  25  0.2500        0.2549        0.2596        0.3214            0.1678
 200  53  0.2650        0.2673        0.2696        0.3042            0.1216
one-at-a-time updates over all 200: Beta(54, 148)   batch update: Beta(54, 148)
```

After one failure the MLE is 0, a rate that says "this never converts"; every posterior mean stays sensible. By $n = 200$
the weak priors agree with the MLE to within 0.005, while Beta(20, 20), worth 40 observations, still pulls toward 0.5.
Here that pull happens to land closer to the true 0.3, because this sample's rate (0.265) is below the truth; a
strong prior centred on a *wrong* value would pull the other way just as hard. The credible interval shrinks
roughly like $1/\sqrt{n}$ (0.50 at $n = 10$, 0.17 at $n = 100$), as every standard error in this course does.

**A Bayesian A/B comparison.** The A6.3a example (30/200 against 50/200) with flat priors. Draw 200,000 values from
each posterior and compare them pairwise:

```text
== from scratch: Bayesian A/B comparison ==
A: 30/200 -> Beta(31, 171)   B: 50/200 -> Beta(51, 151)
P(p_B > p_A | data) = 0.9938   expected lift = 0.0990   95% credible interval for the lift = [0.0212, 0.1771]
P(lift > 0.05 | data) = 0.8933   (is the lift at least 5 points?)
for comparison, the chi-square test of A6.3a gave p = 0.0124
```

The Bayesian summary answers the question a product manager actually asks, "how likely is B better, and by how
much?", directly: 99.4% that B is better, 89% that it is better by at least 5 points. The p-value of 0.012 answers a
different question (how surprising the data would be if A and B were equal). They are not in conflict, and
$1 - 0.994$ is not the p-value. Note that the Bayesian probabilities are only as good as the priors and the model,
and that peeking at a posterior and stopping when $P(B > A)$ crosses 95% inflates false wins just as it does for p-values
(A6.3b).

## With SciPy

```text
== with scipy ==
stats.beta(5, 3): mean 0.6250   interval(0.95) [0.2904, 0.901]   matches the grid (to 1e-4, the grid spacing): OK
next 10 trials: posterior predictive mean 6.2500   sd 2.1651   plug-in Binomial(10, 0.75): mean 7.5000   sd 1.3693
P(0 or 1 successes in the next 10): posterior predictive 0.0175   plug-in 3.0e-05
simulated predictive (draw p from the posterior, then 10 trials): mean 6.2545   OK
```

- `stats.beta(a, b)` is a frozen distribution with `.pdf`, `.mean()`, `.interval(0.95)`, `.ppf`, `.sf` and `.rvs`;
  NumPy's `rng.beta(a, b, size)` samples it. VERIFY (as of 2026-10), SciPy 1.18.1.
- `stats.betabinom(m, a, b)` is the posterior predictive for the next $m$ trials. Compared with plugging in the MLE
  (Binomial(10, 0.75)), it is centred lower (shrinkage) and much wider (sd 2.17 against 1.37): it admits that 4
  observations leave the rate uncertain. The plug-in says 0 or 1 successes in the next 10 is essentially impossible
  (3 in 100,000); the predictive says 1.75%. Plug-in predictions are overconfident.
- Libraries for general Bayesian models (PyMC, NumPyro, Stan) are not part of this course's pinned stack; the
  grid method covers one parameter, and Part G covers the variational approach.

## In practice

- **When to use it:** estimating rates from small counts (click-through of new items, error rates of rare
  endpoints, per-segment conversion), ranking items with very different amounts of data, reporting "probability B
  beats A" alongside a test, and anywhere prior knowledge is real and worth using (last quarter's rate as a prior
  for this quarter's).
- **Failure modes:**
  - *An overconfident prior.* Beta(500, 500) ignores 200 observations. You notice posteriors that barely move
    when the data clearly disagrees. Check the prior's pseudo-count total $a + b$ against your sample size, and
    show the results under a weaker prior too (a sensitivity check).
  - *A prior chosen after seeing the data* to get the answer you wanted. Fix the prior in advance, as with
    pre-registration.
  - *Non-exchangeable data.* Rates that drift over time or differ by segment pooled into one Beta. You notice
    posteriors that are confidently wrong for recent data. Model segments separately or discount old counts.
  - *Reading the credible interval without the prior.* "95% probability" is conditional on the prior and the
    model. Report them.
  - *Peeking at $P(B > A)$.* Stopping when it first crosses 95% inflates false wins, just like peeking at p-values.
- **Defaults that work:** Beta(1, 1) when you know nothing; a weak informative prior with $a + b$ of about 2 to 10
  centred on a historical rate when you know something; posterior mean for ranking; equal-tailed 95% credible
  intervals; posterior predictive (not plug-in) for forecasting counts.

## Exercises

Save your work to `my-work/A/a6_5_bayesian_inference.md` (or `.py`) and ask for a review.

1. **By hand:** (a) An email campaign's open rate has a Beta(3, 7) prior (from past campaigns). The first 20 emails
   produce 9 opens. Give the posterior, its mean, its mode, the MLE, and show that the mean is the weighted average
   of the prior mean and the MLE (give the weight). (b) What is the probability that the next email is opened?
   (c) Two products: X has 1 review, 5 stars; Y has 400 reviews, 4.7 stars on average. Treat a review of 4 or 5 stars
   as a "success": X has 1/1, and say Y has 360/400. With a Beta(4, 1) prior (a typical product gets mostly good
   reviews), compute each product's posterior mean. Which ranks higher? (d) Explain in two sentences the
   difference between "95% confidence interval [0.31, 0.52]" and "95% credible interval [0.31, 0.52]".
2. **Code:** Write `ab_bayes(k_a, n_a, k_b, n_b, a=1, b=1, draws, gen)` returning $P(p_B > p_A)$, the expected lift and a 95%
   credible interval for the lift. (a) Apply it to 120/1,000 against 145/1,000 and compare $P(p_B > p_A)$ with the
   one-sided p-value of the two-proportion z-test. (b) Compute $P(p_B > p_A)$ exactly on a grid (or by numerical
   integration) and check your Monte Carlo value within its error. (c) Simulate 1,000 A/A experiments (both rates 0.12,
   1,000 users each, `SEED = 0`) and report how often $P(p_B > p_A) > 0.95$. Relate that number to a significance level.
3. **Think:** A marketplace ranks new sellers by their positive-feedback rate. A product manager proposes a Bayesian
   prior so that new sellers are not ranked first or last on a handful of reviews. Discuss how you would choose the
   prior (where its centre and strength come from), what could go wrong for new sellers who are genuinely
   excellent or genuinely bad, whether the same prior should apply to every category, and how you would check that
   the ranking is fair to new sellers. `[RAI]`

## Takeaways

1. Bayesian inference updates a prior density over a parameter into a posterior: posterior ∝ likelihood × prior, then
   normalize.
2. With a Beta($a$, $b$) prior and $k$ successes in $n$ trials the posterior is Beta($a + k$, $b + n - k$): the prior acts as $a$ successes and $b$
   failures of pseudo-data, and updating is counting.
3. The posterior mean $\frac{a + k}{a + b + n}$ is a weighted average of the prior mean and the MLE, with the prior weighted by its share of
   the counts; the effect is shrinkage, strongest when data is scarce.
4. A credible interval is a direct probability statement about the parameter, conditional on the prior and model; a
   confidence interval is a statement about the method.
5. Grids handle any one-parameter prior; posterior predictive distributions include the uncertainty about the
   parameter and are wider than plug-in predictions.

**What builds on this:** [B2.4](../B/b2_4_naive_bayes.md) (Laplace smoothing is a Beta/Dirichlet prior);
[B4.3](../B/b4_3_cross_validation_tuning.md) (Bayesian optimization of hyperparameters);
[B5.3](../B/b5_3_gmm_em.md) (latent variables and posterior responsibilities); [G1.2](../G/g1_2_vae.md)
(approximate posteriors with KL); [H1.3](../H/h1_3_q_learning.md) (exploration with uncertain value estimates).

**References:** A. Gelman, J. B. Carlin, H. S. Stern, D. B. Dunson, A. Vehtari and D. B. Rubin, *Bayesian Data Analysis*,
3rd ed., CRC Press, 2013 (chapter 2 covers the Beta–Binomial model). A. B. Downey, *Think Bayes*, 2nd ed., O'Reilly,
2021.
