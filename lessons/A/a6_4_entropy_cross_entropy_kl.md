# A6.4 Entropy, cross-entropy and KL divergence

**Core** · Part A, week 6 (course week 06) · ~2 h · Lab:
[`labs/A/a6_4_entropy_cross_entropy_kl.py`](../../labs/A/a6_4_entropy_cross_entropy_kl.py)

## Why it matters

Every classifier in this course, from logistic regression in Part B to the LLMs in Part F, is trained by
minimizing **cross-entropy**. Variational autoencoders (G1.2) and RLHF (H2.3) add a **KL divergence**
term to their objectives, drift monitors compare distributions with it, and LLM quality is reported as
perplexity, which is cross-entropy in disguise. These three quantities come from one idea, the
"surprise" of an outcome, and once you have that idea the loss values you see in training logs start to
mean something: why a loss of 0.69 on a balanced binary problem means "the model learned nothing", why a
loss cannot go below a floor set by the labels, and why one confident mistake can wreck an average.

## Prerequisite check

- Can you turn the log of a product into a sum of logs, and do you know $\log 1 = 0$ and $\log x < 0$ for $0 < x < 1$?
  If not, revisit [A5.5](a5_5_mle_map.md) ("Why negative log").
- Can you compute an expectation $\mathbb{E}[g(X)] = \sum_x p(x)g(x)$ for a discrete distribution? If not, revisit
  [A5.1](a5_1_random_variables_expectation.md).
- Do you know that log-loss is the negative log-likelihood of a Bernoulli model? If not, revisit
  [A5.5](a5_5_mle_map.md).
- Can you write the categorical distribution and the Gaussian density? If not, revisit
  [A5.2](a5_2_common_distributions.md).

## Intuition

### Surprise

How surprised should you be by an outcome? If it was certain, not at all; the rarer it was, the more
surprised. And if two independent things happen, the surprise should add: a coin landing heads and a
die showing 6 should be as surprising as each one, added together. The one function that does this is the
logarithm of $1/p$, the **surprisal** $-\log p$: it is 0 when $p = 1$, grows as $p$ shrinks, and because the probability of
two independent events is a product, $-\log(p_1p_2) = -\log p_1 - \log p_2$, so surprises add.

### Entropy: average surprise

**Entropy** is the average surprisal of a distribution, the surprise you expect *before* you see the
outcome. A fair coin: each side has probability $\frac12$, surprisal $\log_2 2 = 1$ **bit**, so the entropy is 1 bit. A coin
that lands heads 90% of the time: heads is barely surprising (0.15 bits), tails very (3.3 bits), but tails is
rare, and the average is 0.47 bits. A two-headed coin: 0 bits, no uncertainty at all.

Bits have a concrete meaning: entropy in bits is the average number of yes/no questions you need to identify
the outcome, using the best possible questioning strategy, and equivalently the shortest possible average
length, in bits, of a code that records the outcomes (Shannon, 1948). 8 equally likely outcomes need 3
questions ("is it in the first half?" three times): 3 bits. With the natural log, the unit is the **nat**;
1 nat = 1.443 bits. ML code uses nats.

### Cross-entropy: the cost of believing the wrong thing

Now suppose outcomes come from a distribution $p$, but you *believe* they come from $q$, and you measure your
surprise with your beliefs: $-\log q(x)$. The average surprise is the **cross-entropy** $H(p, q)$. It is never less
than $H(p)$, and it equals $H(p)$ only when $q = p$: wrong beliefs always cost extra surprise on average.

This is exactly the classification loss. For one training example, "reality" is the true label (a
distribution that puts probability 1 on the correct class), and the model's belief is its predicted
distribution $q$. The cross-entropy between them is $-\log q(\text{true class})$: the model's surprise at the truth. Training
minimizes the average surprise over the dataset.

### KL divergence: the extra cost

The **KL divergence** (Kullback–Leibler, 1951) is the extra: $D_{\mathrm{KL}}(p \,\|\, q) = H(p, q) - H(p)$, how much more surprised you are on
average because you used $q$ instead of the truth $p$. It is 0 only when $q = p$, otherwise positive, and it is
**not symmetric**: using $q$ when the truth is $p$ costs a different amount than the reverse. In particular, if $q$
gives probability 0 to something $p$ allows, the cost is infinite (you were infinitely surprised by something
that happened), while the reverse can be finite. KL is a "divergence", not a distance.

> **Bridge:** compression. A compressor tuned for English text compresses English to near its entropy. Feed
> it JSON and the output is larger, because its model of the data ($q$, English) does not match the data ($p$,
> JSON): the extra bits per character are the KL divergence. Cross-entropy is the size you actually get.
> **Where it breaks:** real compressors adapt their model as they read, and they code whole contexts, not
> independent symbols; the bridge is the single-symbol, fixed-model case. And ML losses are in nats,
> computed on probabilities a model outputs, not on a file size.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $x$ | an outcome (a class, a symbol) out of $K$ possibilities | — |
| $p$, $q$ | two distributions over the same outcomes; $p(x)$ the probability of $x$ | `(K,)` each |
| $H(p)$ | entropy of $p$ | scalar ≥ 0 |
| $H(p, q)$ | cross-entropy of $q$ relative to $p$ | scalar |
| $D_{\mathrm{KL}}(p \,\|\, q)$ | KL divergence of $q$ from $p$ | scalar ≥ 0 |
| $y_i$ | true class of example $i$ (1 to $K$) | int |
| $\mathbf{p}_i$ | model's predicted probabilities for example $i$ | `(K,)` |
| $\mathcal{L}$ | mean cross-entropy loss over $n$ examples | scalar |
| $I(X; Y)$ | mutual information between $X$ and $Y$ | scalar ≥ 0 |

`log` is the natural log, so everything is in nats unless written $\log_2$ (bits). Divide nats by $\ln 2 = 0.6931$ to get
bits. By convention $0 \log 0 = 0$ (an impossible outcome contributes nothing; it is the limit of $t\log t$ as $t \to 0$).

**Definitions.**

$$H(p) = -\sum_x p(x)\log p(x), \qquad H(p, q) = -\sum_x p(x)\log q(x), \qquad D_{\mathrm{KL}}(p \,\|\, q) = \sum_x p(x)\log\frac{p(x)}{q(x)}.$$

Each is an expectation under $p$ (A5.1): of the surprisal $-\log p(X)$, of $-\log q(X)$, and of the difference $\log p(X) - \log q(X)$. Since
$\log\frac{p}{q} = \log p - \log q$, subtracting the first two gives

$$H(p, q) = H(p) + D_{\mathrm{KL}}(p \,\|\, q).$$

**Entropy limits.** $H(p) \ge 0$ (each $-\log p(x) \ge 0$ because $p(x) \le 1$), with 0 only for a certain outcome. For $K$ equally likely
outcomes $H = -K \cdot \frac1K\log\frac1K = \log K$, the maximum over all distributions on $K$ outcomes (follows from the next result
with $q$ uniform).

**KL is never negative (Gibbs' inequality).** The one fact needed is $\log t \le t - 1$ for all $t > 0$, with equality only at $t = 1$
(the line $t - 1$ is tangent to $\log t$ at $t = 1$, and $\log$ is concave, so it lies below its tangent; A4.5). Apply it with
$t = q(x)/p(x)$, summing over the $x$ with $p(x) > 0$:

$$-D_{\mathrm{KL}}(p \,\|\, q) = \sum_x p(x)\log\frac{q(x)}{p(x)} \le \sum_x p(x)\Big(\frac{q(x)}{p(x)} - 1\Big) = \sum_x q(x) - \sum_x p(x) \le 1 - 1 = 0.$$

The last step: both sums run only over the $x$ with $p(x) > 0$, so $\sum_x p(x) = 1$, while $\sum_x q(x)$ over those $x$ is at most 1 (it
misses whatever mass $q$ puts elsewhere). So $D_{\mathrm{KL}} \ge 0$, with equality only when $q(x) = p(x)$ everywhere. Consequently $H(p, q) \ge H(p)$.

**Cross-entropy as the classification loss.** For example $i$, the "true distribution" is one-hot: probability 1 on
class $y_i$, 0 elsewhere. Its cross-entropy with the prediction $\mathbf{p}_i$ keeps only one term of the sum:

$$H(\text{one-hot}_{y_i}, \mathbf{p}_i) = -\log p_{i, y_i}, \qquad \mathcal{L} = -\frac1n\sum_{i=1}^n \log p_{i, y_i}.$$

This is the categorical negative log-likelihood (A5.5); for $K = 2$ it is the familiar log-loss
$-[y\log p + (1 - y)\log(1 - p)]$. Two consequences:

- **The floor.** For a model that predicts the same $q$ for every example, the mean loss is the cross-entropy between
  the label distribution $p$ and $q$, which is $H(p) + D_{\mathrm{KL}}(p \,\|\, q)$. Its minimum is at $q = p$, and the minimum value is $H(p)$, the
  entropy of the labels. On a balanced binary problem that is $\log 2 = 0.693$: a training log stuck at 0.693 means the
  model predicts 50/50 for everything. A model that uses the features can go below that, but the label noise
  that no feature explains sets a floor above 0.
- **Confident mistakes dominate.** $-\log 0.01 = 4.6$, $-\log 10^{-6} = 13.8$, $-\log 0 = \infty$. One example predicted with 0.01 on its true class
  costs as much as 13 examples predicted at 0.7.

**Mutual information.** How much does knowing $X$ tell you about $Y$? Compare the joint distribution with the one
you would have if they were independent:

$$I(X; Y) = D_{\mathrm{KL}}\big(p(x, y) \,\|\, p(x)p(y)\big) = H(Y) - H(Y \mid X),$$

where $H(Y \mid X) = \sum_x p(x)\,H(Y \mid X = x)$ is the average entropy of $Y$ left after seeing $X$ (the second form is stated; the lab
checks it numerically). $I = 0$ exactly when $X$ and $Y$ are independent, unlike correlation, which only detects linear
relations (A6.1). Feature selection and decision trees (B3.1, information gain) use it.

**KL between Gaussians.** For continuous densities, sums become integrals. One case has a closed form that VAEs
use (G1.2): $p = \mathcal{N}(\mu, \sigma^2)$ and $q = \mathcal{N}(0, 1)$. From the Gaussian density (A5.2), $\log p(x) = -\frac12\log(2\pi) - \log\sigma - \frac{(x - \mu)^2}{2\sigma^2}$ and
$\log q(x) = -\frac12\log(2\pi) - \frac{x^2}{2}$; the $\frac12\log(2\pi)$ terms cancel in the difference:

$$\log p(x) - \log q(x) = -\log\sigma - \frac{(x - \mu)^2}{2\sigma^2} + \frac{x^2}{2}.$$

Take the expectation under $p$ (so the middle term becomes $\frac{\sigma^2}{2\sigma^2} = \frac12$), using $\mathbb{E}[(X - \mu)^2] = \sigma^2$ and $\mathbb{E}[X^2] = \operatorname{Var}[X] + \mathbb{E}[X]^2 = \sigma^2 + \mu^2$ (A5.1):

$$D_{\mathrm{KL}}\big(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, 1)\big) = -\log\sigma - \frac12 + \frac{\sigma^2 + \mu^2}{2} = \frac12\big(\sigma^2 + \mu^2 - 1 - \log\sigma^2\big).$$

It is 0 only at $\mu = 0, \sigma = 1$, grows like $\mu^2/2$ as the mean moves, and blows up as $\sigma \to 0$ (the $-\log\sigma^2$ term).

**Forward and reverse KL.** When you fit a simple $q$ to a complicated $p$, the direction matters:

- Minimizing $D_{\mathrm{KL}}(p \,\|\, q)$ ("forward") averages $\log(p/q)$ over $p$: wherever $p$ has mass, $q$ must not be tiny, or the cost explodes. The
  fit **covers** all of $p$'s mass, even if that puts $q$'s mass where $p$ has none. Maximum likelihood is this direction
  (fitting $q$ to data drawn from $p$).
- Minimizing $D_{\mathrm{KL}}(q \,\|\, p)$ ("reverse") averages over $q$: wherever $q$ has mass, $p$ must not be tiny. The fit **seeks a mode**: it
  sits inside one region of $p$ and ignores the rest. Variational inference (the VAE's objective) and the KL
  penalty in RLHF use this direction.

## Worked example by hand

**1. Coins.** Fair: $H = -2 \times 0.5\log 0.5 = \log 2 = 0.6931$ nats $= 1$ bit. A 0.9/0.1 coin: $-0.9\log 0.9 - 0.1\log 0.1 = 0.0948 + 0.2303 = 0.3251$ nats
$= 0.3251/0.6931 = 0.469$ bits. Most of the entropy comes from the rare outcome: small probability, large surprisal.

**2. Three outcomes.** $p = (0.5, 0.25, 0.25)$ and the uniform $q = (\frac13, \frac13, \frac13)$.

- $H(p) = 0.5\log 2 + 2 \times 0.25\log 4 = 0.5\log 2 + \log 2 = 1.5\log 2 = 1.0397$ nats $= 1.5$ bits. (Questions: "is it the first?" settles it
  half the time; otherwise one more question: $0.5 \times 1 + 0.5 \times 2 = 1.5$.)
- $H(p, q) = -\sum p(x)\log\frac13 = \log 3 = 1.0986$ nats (each term has the same $\log\frac13$, and the $p(x)$ sum to 1).
- $D_{\mathrm{KL}}(p \,\|\, q) = 1.0986 - 1.0397 = 0.0589$. Directly: $0.5\log\frac{0.5}{1/3} + 2 \times 0.25\log\frac{0.25}{1/3} = 0.5\log 1.5 + 0.5\log 0.75 = 0.2027 - 0.1438 = 0.0589$ ✓.
- $D_{\mathrm{KL}}(q \,\|\, p) = \frac13\big(\log\frac{1/3}{0.5} + 2\log\frac{1/3}{0.25}\big) = \frac13(\log\frac23 + 2\log\frac43) = \frac13(-0.4055 + 0.5754) = 0.0566$. Not the same.

**3. A classifier's loss.** Three examples with true classes 1, 2, 3 (0, 1, 2 in code) and predicted
probabilities $(0.7, 0.2, 0.1)$, $(0.1, 0.6, 0.3)$, $(0.2, 0.2, 0.6)$. Losses: $-\log 0.7 = 0.3567$, $-\log 0.6 = 0.5108$, $-\log 0.6 = 0.5108$; mean $0.4594$.
Had the first example been predicted with 0.01 on its true class, its loss alone would be 4.605.

**4. The floor.** 30% of labels are positive. The best constant prediction is 0.3, with loss
$H(0.3) = -(0.3\log 0.3 + 0.7\log 0.7) = 0.3612 + 0.2497 = 0.6109$ nats.

**5. A Gaussian KL.** $\mu = 1$, $\sigma = 0.5$: $\frac12(0.25 + 1 - 1 - \log 0.25) = \frac12(0.25 + 1.3863) = 0.8181$.

```text
== worked example by hand ==
fair coin: H = log 2 = 0.6931 nats = 1.0000 bits
0.9/0.1 coin: -0.9 log 0.9 - 0.1 log 0.1 = 0.0948 + 0.2303 = 0.3251 nats = 0.4690 bits
H(p) = 1.0397 nats (1.5000 bits)   H(p, q) = log 3 = 1.0986   KL(p || q) = 0.0589   H(p) + KL = 1.0986
KL(q || p) = (1/3)(log(2/3) + 2 log(4/3)) = 0.0566   (not equal to KL(p || q))
per-example loss -log q(true class) = [0.3567, 0.5108, 0.5108]   mean = 0.4594
a confident mistake, q(true) = 0.01: loss = 4.6052
30% positives, predict 0.3 for everyone: loss = H(0.3) = 0.6109 nats
KL(N(1, 0.5^2) || N(0, 1)) = 0.5 * (0.25 + 1 - 1 - log 0.25) = 0.8181
```

## Key plot

![Left: the binary entropy curve in bits, an arch from 0 at P(heads) = 0 up to 1 bit at 0.5 and back to 0 at 1, with points marked at 0.5 (1.000 bits) and 0.9 (0.469 bits). Right: a target density with two equal bumps at minus 2.5 and plus 2.5; the Gaussian that minimizes forward KL is a wide bell centred at 0 covering both bumps, and the one that minimizes reverse KL is a narrow bell sitting on the right bump only](img/a6_4_entropy_kl_direction.png)

Left: entropy is largest when you are most unsure. Right: asked to approximate two bumps with one bell,
forward KL spreads the bell over both (and puts most of its mass in the valley where the target has none);
reverse KL commits to one bump and ignores the other.

## From scratch in NumPy

**The three quantities.** Masking with `p > 0` implements $0\log 0 = 0$; a zero in $q$ where $p > 0$ makes the cross-entropy
and KL infinite:

```python
def kl_divergence(p: FloatArray, q: FloatArray, base: float = np.e) -> float:
    """KL(p || q) = sum p log(p / q). p, q (K,) -> scalar >= 0."""
    nz = p > 0
    if np.any(q[nz] == 0):
        return float("inf")
    return float(np.sum(p[nz] * np.log(p[nz] / q[nz])) / np.log(base))
```

```text
== from scratch: entropy, cross-entropy, KL ==
p3 vs uniform: H(p) = 1.0397   H(p, q) = 1.0986   KL(p||q) = 0.0589   KL(q||p) = 0.0566   H(p, q) - H(p) - KL = 2.2e-16
random 5-class: H(p) = 1.1123   H(p, q) = 2.0755   KL(p||q) = 0.9632   KL(q||p) = 1.5669   H(p, q) - H(p) - KL = 0.0e+00
entropy of K equally likely outcomes = log K: K = 2, 4, 8 -> [1.0, 2.0, 3.0] bits
a certain outcome [1, 0, 0]: H = 0.0000
q gives 0 to an outcome p allows: KL(p || q) = inf   KL(q || p) = 0.6931
smallest KL over 10,000 random pairs of 4-class distributions: 0.000446   (never negative)
```

The identity $H(p, q) = H(p) + \mathrm{KL}$ holds to rounding error; KL's asymmetry is small for similar distributions (0.059 against
0.057) and large for different ones (0.96 against 1.57); a zero in $q$ gives infinity in one direction and a finite
number in the other; and 10,000 random pairs never produce a negative KL.

**Cross-entropy as a loss, and its floor.** The mean of $-\log q(\text{true class})$ is the mean cross-entropy against one-hot
targets. The lab then draws 10,000 binary labels with a 30% positive rate and tries every constant prediction on a
grid:

```text
== from scratch: cross-entropy as a loss ==
mean -log q(true) = 0.4594   mean H(onehot, q) = 0.4594   (the same number)
labels: 0.2966 positive   best constant prediction on a 0.05 grid = 0.30   its loss = 0.6080   entropy of the labels = 0.6080
predicting 0.5 for everyone costs log 2 = 0.6931; the extra over the entropy is KL = 0.0852
```

The best constant prediction is the label frequency, and its loss is the entropy of the labels (0.608 here,
because this sample has 29.7% positives rather than exactly 30%). Predicting 50/50 costs $\log 2$, and the excess is
exactly the KL divergence from the label distribution to $(0.5, 0.5)$. This is the baseline any real model must beat.

**Mutual information, continuous KL and the two directions.** Mutual information is a KL divergence on a joint
table; Gaussian KLs are checked against a numerical integral on a grid; and the lab fits a single Gaussian to a
two-bump density by searching a grid of $(\mu, \sigma)$ for the smallest KL in each direction:

```text
== from scratch: mutual information and continuous KL ==
I(X; Y) = 0.2504 nats   H(Y) = 0.6474   H(Y | X) = 0.3970   H(Y) - H(Y | X) = 0.2504
independent table (outer product of the marginals): I = 0.0000
KL(N(1.0, 0.5^2) || N(0, 1)): closed form 0.8181   numerical 0.8181
KL(N(0.0, 2.0^2) || N(0, 1)): closed form 0.8069   numerical 0.8068
KL(N(3.0, 1.0^2) || N(0, 1)): closed form 4.5000   numerical 4.5000
target: two bumps at -2.5 and 2.5 (sd 0.7 each); mean 0, sd 2.5962
best Gaussian by forward KL(p || q): mu = 0.00, sd = 2.57   KL = 0.6182
best Gaussian by reverse KL(q || p): mu = 2.53, sd = 0.69   KL = 0.6941
```

The forward fit has (to grid precision) the target's own mean and standard deviation: minimizing forward KL over
Gaussians is moment matching, which is what maximum likelihood does. The reverse fit sits on one bump with
nearly that bump's width. (By symmetry, $\mu = -2.53$ is equally good; the grid search returns the first minimum it
finds.)

## With SciPy and scikit-learn

```text
== with scipy and scikit-learn ==
stats.entropy(p) = H(p)                        library 1.0397   mine 1.0397   OK
stats.entropy(p, base=2) = H(p) in bits        library 1.5000   mine 1.5000   OK
stats.entropy(p, q) = KL(p || q), not H(p, q)  library 0.0589   mine 0.0589   OK
special.rel_entr(p, q).sum() = KL              library 0.0589   mine 0.0589   OK
log_loss(y, probs) = mean CE                   library 0.4594   mine 0.4594   OK
mutual_info_score on 1,000 rows built from the table: 0.2504   mine 0.2504   OK
quad of KL(N(1, 0.25) || N(0, 1)): 0.818147   closed form 0.818147   OK
```

- `scipy.stats.entropy(p)` is the entropy; with a second argument, `stats.entropy(p, q)` is the **KL
  divergence**, not the cross-entropy. This trips people up; the cross-entropy is
  `stats.entropy(p) + stats.entropy(p, q)`. It also normalizes its inputs to sum to 1 silently. VERIFY (as of
  2026-10), SciPy 1.18.1.
- `scipy.special.rel_entr(p, q)` returns the elementwise terms $p\log(p/q)$ with the $0\log 0$ convention; `special.xlogy` gives
  $x\log y$ safely.
- `sklearn.metrics.log_loss(y, probs)` is the mean cross-entropy against one-hot labels, in nats (it clips
  probabilities away from 0 and 1). `sklearn.metrics.mutual_info_score(x, y)` computes the mutual information from
  two label arrays, in nats.
- PyTorch's `nn.CrossEntropyLoss` takes **logits**, not probabilities, and computes the log-softmax in a
  numerically stable way; C1.4 builds that from scratch.

## In practice

- **When to use it:** cross-entropy is the loss for any classifier or language model; read training curves
  against the label-entropy floor ($\log K$ for $K$ balanced classes). KL measures how far one distribution is from a
  reference: a predicted from a target distribution (distillation), a policy from a reference model (RLHF),
  an encoder's posterior from the prior (VAEs), today's feature distribution from last month's (drift). Mutual
  information screens features for non-linear dependence on the target.
- **Failure modes:**
  - *$\log 0$.* A probability of exactly 0 on a true class gives an infinite loss, and NaN gradients. You notice `inf` or
    `nan` in the loss. Work with logits and log-softmax (C1.4) or clip probabilities.
  - *Overconfident models.* A handful of confident mistakes dominate the mean loss even when accuracy is high. You
    notice a loss that rises while accuracy stays flat (common late in training). Look at the per-example loss
    distribution; consider label smoothing or calibration (B4.1).
  - *Wrong direction or wrong function.* `stats.entropy(p, q)` used as cross-entropy; KL arguments swapped. You notice
    numbers that do not satisfy $H(p, q) = H(p) + \mathrm{KL}$; check that identity in a unit test.
  - *Zero bins in histogram KL.* Comparing two histograms (for drift) where one has an empty bin gives infinity. You
    notice an `inf` drift score; add a small count to every bin or use a bounded measure (the Jensen–Shannon
    divergence, or the population stability index, J2.2).
  - *Mixing nats and bits.* A paper's "bits per character" against your nats per token. Divide nats by $\ln 2$, and
    remember tokens and characters are different units.
- **Defaults that work:** nats everywhere in code; compare every loss with the constant-prediction baseline
  $H(\text{labels})$; compute losses from logits; add-one smoothing for histogram KLs.

## Exercises

Save your work to `my-work/A/a6_4_entropy_cross_entropy_kl.md` (or `.py`) and ask for a review.

1. **By hand:** (a) Compute in bits the entropy of a fair 4-sided die, of the distribution $(0.5, 0.25, 0.125, 0.125)$,
   and of $(0.97, 0.01, 0.01, 0.01)$. Explain the order in words. (b) For $p = (0.5, 0.5)$ and $q = (0.9, 0.1)$, compute in nats $H(p)$,
   $H(p, q)$, $D_{\mathrm{KL}}(p \,\|\, q)$ and $D_{\mathrm{KL}}(q \,\|\, p)$. (c) A 4-class classifier outputs $(0.1, 0.6, 0.2, 0.1)$ for an example of class 2 and
   $(0.25, 0.25, 0.25, 0.25)$ for an example of class 4. Give each loss and the mean. What loss does a model that always
   outputs the uniform distribution get, and what does that tell you when you see it in a training log?
   (d) Compute $D_{\mathrm{KL}}(\mathcal{N}(0, 0.5^2) \,\|\, \mathcal{N}(0, 1))$ and $D_{\mathrm{KL}}(\mathcal{N}(2, 1) \,\|\, \mathcal{N}(0, 1))$ with the closed form.
2. **Code:** Write `cross_entropy_from_logits(Z, y)` for a batch of logits $\mathbf{Z}$ `(n, K)` and integer labels $\mathbf{y}$ `(n,)` that never
   computes $\log$ of a softmax directly: use $\log\operatorname{softmax}(\mathbf{z})_k = z_k - \log\sum_j e^{z_j}$ with the max subtracted before exponentiating.
   Check it against `sklearn.metrics.log_loss` on random logits (`SEED = 0`, $n = 1{,}000$, $K = 5$) and show that it returns
   a finite, correct value for logits such as $(1000, 0, -1000)$ where the naive version returns `nan` or `inf`. Then
   compute the mutual information between each of three synthetic features and a binary label (one independent,
   one linearly related, one related through $y = 1$ when $\lvert x\rvert > 1$) by binning the features into 10 quantile bins,
   and compare it with the absolute Pearson correlation. Which feature does correlation miss?
3. **Think:** A language model's validation loss is 2.30 nats per token. A colleague says the model "is 10 times
   better than random" because $e^{2.30} \approx 10$. Explain what $e^{\text{loss}}$ (the perplexity) means, what the loss of a uniformly
   random model over a vocabulary of 50,000 tokens would be, what lower bound the loss can never go below, and why
   comparing the losses of two models with different tokenizers can mislead.

## Takeaways

1. Surprisal is $-\log p$; entropy $H(p)$ is its average, the irreducible uncertainty of a distribution: 1 bit for a fair coin,
   $\log K$ for $K$ equal outcomes, 0 for a certain one.
2. Cross-entropy $H(p, q)$ is the average surprise when the data comes from $p$ and you predict with $q$;
   $H(p, q) = H(p) + D_{\mathrm{KL}}(p \,\|\, q)$, and $D_{\mathrm{KL}} \ge 0$ with equality only at $q = p$.
3. The classification loss is the mean of $-\log q(\text{true class})$: cross-entropy against one-hot labels, the same as the
   negative log-likelihood. Its floor is the entropy of the labels ($\log 2$ for balanced binary).
4. KL is not symmetric and is infinite when $q$ rules out what $p$ allows; minimizing forward KL covers all of $p$, reverse KL
   locks onto one mode.
5. $D_{\mathrm{KL}}(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, 1)) = \frac12(\sigma^2 + \mu^2 - 1 - \log\sigma^2)$, the VAE's regularizer; mutual information is a KL that detects any dependence, not
   only linear.

**What builds on this:** [B2.1](../B/b2_1_logistic_regression.md) (log-loss for logistic regression);
[B3.1](../B/b3_1_decision_trees.md) (information gain = mutual information); [C1.4](../C/c1_4_stable_softmax_cross_entropy.md)
(stable softmax cross-entropy); [F2.2](../F/f2_2_pretraining.md) (language-model loss and perplexity);
[G1.2](../G/g1_2_vae.md) (the KL term of the VAE); [H2.3](../H/h2_3_rlhf.md) (the KL penalty to a reference model);
[J2.2](../J/j2_2_monitoring_drift.md) (distribution drift).

**References:** C. E. Shannon, "A Mathematical Theory of Communication", *Bell System Technical Journal* 27, 1948.
S. Kullback and R. A. Leibler, "On Information and Sufficiency", *Annals of Mathematical Statistics* 22(1), 1951.
T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley, 2006.
