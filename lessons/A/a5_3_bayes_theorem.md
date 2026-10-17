# A5.3 Conditional probability and Bayes' theorem

**Core** · Part A, week 5 (course week 05) · ~1.5 h · Lab:
[`labs/A/a5_3_bayes_theorem.py`](../../labs/A/a5_3_bayes_theorem.py)

## Why it matters

A fraud model catches 99% of fraud and flags only 1% of honest transactions. A transaction is
flagged. Is it fraud? Most engineers say "99% likely"; if fraud is 1 in 1,000 transactions, the
answer is about 9%. The gap between those two numbers is the **base rate**, and ignoring it is the
most common reasoning error in ML evaluation: it is why a classifier with excellent accuracy can still
bury a review team in false alarms ([B4.1](../B/b4_1_classification_metrics.md), [B4.4](../B/b4_4_class_imbalance.md)).
**Bayes' theorem** is the rule for turning "how likely is the evidence if the hypothesis is true" into
"how likely is the hypothesis given the evidence". It is also the engine of naive Bayes
([B2.4](../B/b2_4_naive_bayes.md)) and of the MAP estimates in [A5.5](a5_5_mle_map.md).

## Prerequisite check

- Can you build a joint probability table for two yes/no variables and read the marginals off its
  row and column sums? If not, revisit [A5.1](a5_1_random_variables_expectation.md).
- Can you simulate a Bernoulli outcome with `rng.random(n) < p`? If not, revisit
  [A5.2](a5_2_common_distributions.md).
- Can you use a boolean mask to select rows (`x[mask]`) and combine masks with `&`, `|` and `~`? If not,
  revisit [A1.2](a1_2_numpy_arrays.md).
- Can you read a plot with a log-scale x-axis? If not, revisit [A1.4](a1_4_matplotlib.md).

## Intuition

### Conditioning is zooming in

$P(A)$ is the fraction of all outcomes in which $A$ happens. $P(A \mid B)$, "the probability of $A$ given
$B$", is the fraction of the outcomes *where $B$ happened* in which $A$ also happens. Conditioning throws
away every outcome where $B$ did not happen and recomputes fractions in what is left.

> **Bridge:** conditioning is a `WHERE` clause. $P(\text{disease})$ is
> `SELECT AVG(disease) FROM people`; $P(\text{disease} \mid \text{positive})$ is
> `SELECT AVG(disease) FROM people WHERE positive`. The lab literally does this with a boolean mask.
> **Where it breaks:** a query runs on the rows you have. Probability is about the process that
> generates rows, so a `WHERE` over 12 matching rows gives a noisy estimate (A5.1), and a condition that
> never occurred in your table still has a probability.

### The two conditionals are not the same

$P(\text{positive} \mid \text{disease})$ asks: among sick people, how many test positive? That is a property
of the test (its **sensitivity**), measured in a lab on known sick people. $P(\text{disease} \mid \text{positive})$
asks: among people who test positive, how many are sick? That is what a patient wants to know, and it
depends on how many sick people there are to begin with. Confusing the two is called the **prosecutor's
fallacy** ("the chance of this DNA match if innocent is one in a million, so the chance he is innocent
is one in a million").

### Count people, not probabilities

The easiest way to get Bayes right is to imagine 10,000 people. A disease affects 1% of them; a test
catches 90% of sick people and wrongly flags 9% of healthy ones.

- 100 are sick; 90 of them test positive.
- 9,900 are healthy; 9% of them, 891, test positive anyway.
- 981 people test positive, and only 90 of them are sick: $90/981 = 9.2\%$.

The test is decent, but the 891 false alarms from the huge healthy group swamp the 90 real cases. A
positive result moved the probability from 1% to 9%: real evidence, but far from proof. That is the
**medical-test trap**, and the key plot shows how it depends on the base rate.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $A, B$ | events (things that do or do not happen) | — |
| $P(A \cap B)$ | probability that both happen (also written $P(A, B)$) | scalar |
| $P(A \mid B)$ | conditional probability of $A$ given $B$ | scalar |
| $\bar{A}$ | the complement, "not $A$"; $P(\bar{A}) = 1 - P(A)$ | — |
| $D$, $H$ | "has the disease", "healthy" ($H = \bar{D}$) | events |
| $+$, $-$ | positive and negative test result | events |
| $H_1, \dots, H_K$ | competing hypotheses, exactly one of which is true | events |
| sensitivity | $P(+ \mid D)$, the **true positive rate** (recall) | scalar |
| specificity | $P(- \mid H)$, the true negative rate; $1 - $ specificity is the **false positive rate** | scalar |
| $\mathrm{LR}_+$ | likelihood ratio of a positive result, $P(+ \mid D)/P(+ \mid H)$ | scalar |

**Conditional probability.** Restrict to $B$ and renormalize:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \qquad P(B) > 0.$$

**Product rule.** Multiply both sides by $P(B)$; and since $A \cap B = B \cap A$, the same holds the other way round:

$$P(A \cap B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A).$$

**Independence**, in these terms: $A$ and $B$ are independent when $P(A \mid B) = P(A)$, which by the product
rule is the same as $P(A \cap B) = P(A)P(B)$ (the A5.1 definition). Learning $B$ tells you nothing about $A$.

**Law of total probability.** $B$ either happens together with $A$ or together with $\bar{A}$, never both, so
the two probabilities add:

$$P(B) = P(B \cap A) + P(B \cap \bar{A}) = P(B \mid A)P(A) + P(B \mid \bar{A})P(\bar{A}).$$

With $K$ hypotheses that cover every case and cannot overlap, $P(B) = \sum_{k=1}^K P(B \mid H_k)P(H_k)$.

**Bayes' theorem.** Take the two forms of the product rule, $P(A \mid B)P(B) = P(B \mid A)P(A)$, and divide by $P(B)$:

$$P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)} = \frac{P(B \mid A)\,P(A)}{P(B \mid A)P(A) + P(B \mid \bar{A})P(\bar{A})}.$$

The pieces have names that the rest of the course uses:

- $P(A)$, the **prior**: what you believed before the evidence (the base rate).
- $P(B \mid A)$, the **likelihood**: how probable the evidence is if $A$ is true.
- $P(B)$, the **evidence** or normalizer: how probable the evidence is overall.
- $P(A \mid B)$, the **posterior**: what you believe after seeing $B$.

For the medical test: $P(D \mid +) = \frac{\text{sens}\cdot P(D)}{\text{sens}\cdot P(D) + (1 - \text{spec})\,(1 - P(D))}$. The
denominator is "all the ways to test positive": true positives plus false positives.

**With $K$ hypotheses.** $P(H_k \mid B) = \frac{P(B \mid H_k)P(H_k)}{\sum_j P(B \mid H_j)P(H_j)}$. Compute
"prior × likelihood" for every hypothesis, then divide by their sum so the posteriors add to 1. This is
exactly what a naive Bayes classifier does for each class (B2.4).

**The odds form.** The **odds** of an event are $P/(1 - P)$: probability 0.2 is odds $0.2/0.8 = 0.25$ ("1 to 4").
Write Bayes for $D$ and for $H$ given the same evidence $+$; both have the same denominator $P(+)$, so it
cancels in the ratio:

$$\underbrace{\frac{P(D \mid +)}{P(H \mid +)}}_{\text{posterior odds}} = \underbrace{\frac{P(+ \mid D)}{P(+ \mid H)}}_{\mathrm{LR}_+} \times \underbrace{\frac{P(D)}{P(H)}}_{\text{prior odds}}.$$

Evidence multiplies the odds by its likelihood ratio. Convert back with $P = \text{odds}/(1 + \text{odds})$. The
likelihood ratio is the test's real strength: $\mathrm{LR}_+ = 10$ means "a positive result makes the disease
ten times more likely, in odds, than it was".

**Several pieces of evidence.** If two tests' results are **conditionally independent** given the true
status (once you know whether the person is sick, one test's result tells you nothing about the
other's), then $P(+_1, +_2 \mid D) = P(+_1 \mid D)P(+_2 \mid D)$, and the odds simply get multiplied by both
likelihood ratios. Equivalently, yesterday's posterior becomes today's prior. When the errors are
correlated (both tests fooled by the same thing), multiplying overstates the evidence. Naive Bayes
makes exactly this assumption about features, which is what "naive" refers to.

## Worked example by hand

**1. The medical test**, base rate 1%, sensitivity 0.90, specificity 0.91.

- Natural frequencies (10,000 people): 100 sick → 90 positive; 9,900 healthy → 891 positive;
  $P(D \mid +) = 90/981 = 0.0917$.
- Formula: $P(+) = 0.9 \times 0.01 + 0.09 \times 0.99 = 0.009 + 0.0891 = 0.0981$; $P(D \mid +) = 0.009/0.0981 = 0.0917$.
- Odds: $\mathrm{LR}_+ = 0.9/0.09 = 10$. Prior odds $0.01/0.99 = 1/99$. Posterior odds $10/99 = 0.1010$, so
  $P = \frac{10/99}{1 + 10/99} = \frac{10}{109} = 0.0917$ ✓.
- A second, conditionally independent positive: odds $\frac{10}{99} \times 10 = \frac{100}{99}$, so $P = \frac{100}{199} = 0.5025$. Two
  positives make it a coin flip, not a certainty.
- A negative result: $P(D \mid -) = \frac{P(- \mid D)P(D)}{P(-)} = \frac{0.1 \times 0.01}{0.1 \times 0.01 + 0.91 \times 0.99} = \frac{0.001}{0.9019} = 0.00111$.
  A negative makes the disease about 9 times less likely than the 1% base rate.

**2. Three hypotheses.** Machines A, B, C make 50%, 30% and 20% of the parts; their defect rates are 1%,
2% and 5%. A part is defective. Which machine made it?

- Prior × likelihood: A $0.5 \times 0.01 = 0.005$; B $0.3 \times 0.02 = 0.006$; C $0.2 \times 0.05 = 0.010$.
- Evidence: $P(\text{defect}) = 0.005 + 0.006 + 0.010 = 0.021$.
- Posteriors: A $0.005/0.021 = 0.2381$; B $0.2857$; C $0.4762$. Machine C makes the fewest parts but is the most
  likely culprit, because its likelihood is high enough to beat its smaller prior.

```text
== worked example: natural frequencies (10,000 people) ==
sick = 100, of whom positive = 90   healthy = 9900, of whom positive = 891
P(disease | positive) = 90 / (90 + 891) = 0.0917

== worked example: Bayes' formula and the odds form ==
P(+) = 0.9 * 0.01 + 0.09 * 0.99 = 0.0981   P(D | +) = 0.009 / 0.0981 = 0.0917
likelihood ratio LR+ = 0.9 / 0.09 = 10.0   prior odds = 1/99 = 0.0101   posterior odds = 0.1010   -> probability 0.0917
second independent positive: prior 0.0917 -> posterior 0.5025   (odds 0.1010 x 10 = 1.0101)
P(D | negative) = 0.1 * 0.01 / (0.1 * 0.01 + 0.91 * 0.99) = 0.00111

== worked example: which machine made the defective part? ==
P(machine and defect) = [0.005, 0.006, 0.01]   P(defect) = 0.0210
P(machine | defect) = [0.2381, 0.2857, 0.4762]   (prior was [0.5, 0.3, 0.2])
```

## Key plot

![P(disease given a positive test) against the base rate on a log scale from 0.0001 to 0.5, for three tests; each curve is an S shape that stays near zero for rare conditions and approaches one for common ones; the 90%/91% test gives 0.09 at a 1% base rate, the 99%/99% test gives 0.5 there, and the 99.9%/99.9% test gives 0.5 at a 0.1% base rate; hollow markers from a million simulated people sit on the curves](img/a5_3_posterior_vs_base_rate.png)

Read it at the 1% line: the decent test gives 0.09, a 99%-accurate test gives 0.50, and only the
99.9% test makes a positive convincing. Each curve crosses 0.5 where the prior odds equal $1/\mathrm{LR}_+$:
a test is only "more likely right than wrong" on a positive when its likelihood ratio beats the
odds against the condition. The dotted line is the prior; the gap between it and a curve is what the
test adds.

## From scratch in NumPy

The simulator draws a disease status for every person, then one test result whose probability depends
on that status:

```python
def simulate_population(
    n: int, base_rate: float, sens: float, spec: float, gen: np.random.Generator
) -> tuple[BoolArray, BoolArray]:
    """Draw disease status and one test result per person. -> (disease (n,), positive (n,))."""
    disease = gen.random(n) < base_rate  # (n,) Bernoulli(base_rate)
    u = gen.random(n)  # one uniform per person for the test
    positive = np.where(disease, u < sens, u < 1.0 - spec)  # (n,) the test's behaviour depends on status
    return disease, positive
```

Conditioning is a mask: `(disease & positive).sum() / positive.sum()` is "among the positives, the
fraction who are sick". The formula is three lines:

```python
def posterior(prior: float | FloatArray, sens: float, spec: float) -> float | FloatArray:
    """Bayes: P(D | +) = P(+ | D) P(D) / P(+), with P(+) from the law of total probability."""
    p_pos = sens * prior + (1.0 - spec) * (1.0 - prior)  # P(+) = P(+|D)P(D) + P(+|H)P(H)
    return sens * prior / p_pos
```

```text
== data ==
population: 1,000,000 people   with disease: 9,996   tested positive: 97,850

== from scratch: counting the simulated population ==
P(disease | positive): counted 0.0916   formula 0.0917   (standard error of the count ~ 0.0009)
P(positive | disease): counted 0.8969   (sensitivity 0.9)
P(positive | healthy): counted 0.0898   (1 - specificity 0.09)
P(disease | negative): counted 0.00114   formula 0.00111
P(disease | two positives): counted 0.5086   formula 0.5025
counted and formula agree within sampling noise: OK
```

The counted and formula answers agree to within the noise A5.1 predicts for a fraction estimated from
97,850 positives ($\sqrt{p(1 - p)/n} \approx 0.0009$). The two-positives estimate rests on only the ~16,000 people
who tested positive twice, so it wanders more (0.5086 against 0.5025, about one standard error). The
second test is simulated independently of the first given the disease status, which is precisely the
conditional-independence assumption the formula makes.

## With scikit-learn

A test is a classifier, and the classification metrics of [B4.1](../B/b4_1_classification_metrics.md)
are these conditional probabilities under other names:

```text
== with scikit-learn: the confusion matrix says the same thing ==
confusion matrix: TN = 901,119  FP = 88,885  FN = 1,031  TP = 8,965
precision_score = TP / (TP + FP) = 0.0916   (our counted P(D | +) = 0.0916)
recall_score = TP / (TP + FN) = 0.8969   specificity = TN / (TN + FP) = 0.9102
precision_score equals our count: OK
```

| Metric (B4.1) | Conditional probability | Depends on the base rate? |
|---|---|---|
| Recall, sensitivity, true positive rate | $P(+ \mid D)$ | no |
| Specificity, true negative rate | $P(- \mid H)$ | no |
| False positive rate | $P(+ \mid H)$ | no |
| Precision, positive predictive value | $P(D \mid +)$ | **yes** |

So recall and false positive rate describe the model, while precision describes the model *in a
particular population*. A model's precision measured on a balanced test set says nothing about its
precision in production, where positives may be 100 times rarer. (`sklearn.metrics.confusion_matrix`
returns `[[TN, FP], [FN, TP]]` for labels `[0, 1]`; `.ravel()` flattens it in that order. VERIFY (as of
2026-10), scikit-learn 1.9.1.)

## In practice

- **When to use it:** any time you turn a classifier score or a test result into a decision, compute
  the precision you will actually get at the production base rate, not the one on your evaluation set.
  Any time you combine several signals, ask whether they are conditionally independent.
- **Failure modes:**
  - *Base-rate neglect.* "The alert is 99% accurate, so a page is almost surely real." You notice it when
    on-call engineers learn to ignore pages. Compute $P(\text{incident} \mid \text{alert})$ from the real incident rate.
  - *Evaluating on a balanced set.* Training and test sets are often rebalanced to 50/50 (B4.4), which
    makes precision look far better than in production. You notice a drop in precision after launch with
    no change in recall. Report precision at the deployment base rate, or report recall and false positive
    rate, which do not depend on it.
  - *Base rates that shift.* Fraud rises during a holiday campaign; a disease spreads. The same model's
    precision changes with no change to the model. You notice precision moving while recall and false
    positive rate stay put.
  - *Double-counting correlated evidence.* Two signals that both fire on VPN users are treated as
    independent, so their likelihood ratios multiply and the posterior is overconfident. You notice that
    cases flagged by both are not much more often real than cases flagged by one.
  - *Swapping the conditionals.* Reading $P(\text{evidence} \mid \text{hypothesis})$ as $P(\text{hypothesis} \mid \text{evidence})$. You notice
    it by asking "out of how many?" and checking which group is in the denominator.
- **Defaults that work:** reason with natural frequencies ("out of 10,000…") when explaining a result to
  anyone; quote likelihood ratios for tests and signals; whenever a metric depends on the base rate, say
  which base rate; simulate the population when the formula gets complicated, then check one against the
  other.

## Exercises

Save your work to `my-work/A/a5_3_bayes_theorem.md` (or `.py`) and ask for a review.

1. **By hand:** 20% of email is spam. A filter flags 95% of spam and 2% of legitimate mail. (a) Draw the
   natural-frequency table for 1,000 emails. (b) Compute $P(\text{spam} \mid \text{flagged})$ and
   $P(\text{spam} \mid \text{not flagged})$ with the formula. (c) Compute $\mathrm{LR}_+$ and check (b) with the odds form.
   (d) The same filter is deployed where only 1% of email is spam. Recompute $P(\text{spam} \mid \text{flagged})$ and
   say in one sentence what changed and what did not.
2. **Code:** Extend the simulator to two tests: test A (sensitivity 0.90, specificity 0.91) and test B
   (sensitivity 0.80, specificity 0.97), base rate 1%, $10^6$ people, `SEED = 0`. (a) With the tests
   conditionally independent, estimate $P(D \mid A+, B+)$ by counting and compare with the odds form.
   (b) Now make their errors correlated: 5% of healthy people have a condition $C$ that makes *both*
   tests positive with probability 0.5 each (independently, given $C$); healthy people without $C$ test
   positive with probability 0.0684 on A and 0.00526 on B. Check by simulation that the overall false
   positive rates are still about 0.09 and 0.03. Estimate $P(D \mid A+, B+)$ again and compare with what the
   odds form (which assumes independence) predicts. Explain the difference.
3. **Think:** A fraud model has recall 0.99 and a false positive rate of 0.01, measured on a balanced
   test set. Fraud is 0.1% of transactions. The product team wants to automatically block every
   flagged transaction. What fraction of blocked transactions would be fraud? What would you recommend
   instead, and what numbers would you want to see before deciding?

## Takeaways

1. $P(A \mid B) = P(A \cap B)/P(B)$: conditioning restricts to the cases where $B$ happened; in code it is a mask.
2. Bayes: posterior = likelihood × prior / evidence, with the evidence from the law of total probability.
   For $K$ hypotheses, compute prior × likelihood for each and normalize.
3. $P(+ \mid D)$ is not $P(D \mid +)$. With a rare condition, most positives are false even for a good test;
   count 10,000 imaginary people to see it.
4. Odds form: posterior odds = likelihood ratio × prior odds; conditionally independent evidence
   multiplies, correlated evidence must not.
5. Recall and false positive rate describe a classifier; precision also depends on the base rate, so
   evaluate it at the production base rate.

**What builds on this:** [A5.5](a5_5_mle_map.md) (MAP: Bayes applied to model parameters);
[A6.5](a6_5_bayesian_inference.md) (posterior distributions over parameters);
[B2.4](../B/b2_4_naive_bayes.md) (naive Bayes classifiers);
[B4.1](../B/b4_1_classification_metrics.md) (precision, recall and ROC curves);
[B4.4](../B/b4_4_class_imbalance.md) (imbalanced classes and threshold choice).
