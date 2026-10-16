# Answer key: A5.3 Conditional probability and Bayes' theorem

Lesson: [`lessons/A/a5_3_bayes_theorem.md`](../../../lessons/A/a5_3_bayes_theorem.md)

## Exercise 1 (by hand)

**Solution (a).** Out of 1,000 emails:

| | Flagged | Not flagged | Total |
|---|---|---|---|
| Spam | $200 \times 0.95 = 190$ | 10 | 200 |
| Legitimate | $800 \times 0.02 = 16$ | 784 | 800 |
| Total | 206 | 794 | 1,000 |

**Solution (b).** $P(\text{spam} \mid \text{flagged}) = \frac{0.95 \times 0.2}{0.95 \times 0.2 + 0.02 \times 0.8} = \frac{0.19}{0.206} = 0.9223$ (the table: $190/206$).
$P(\text{spam} \mid \text{not flagged}) = \frac{0.05 \times 0.2}{0.05 \times 0.2 + 0.98 \times 0.8} = \frac{0.01}{0.794} = 0.0126$ ($10/794$).

**Solution (c).** $\mathrm{LR}_+ = 0.95/0.02 = 47.5$. Prior odds $0.2/0.8 = 0.25$; posterior odds $47.5 \times 0.25 = 11.875$;
probability $11.875/12.875 = 0.9223$ ✓.

**Solution (d).** $\frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.02 \times 0.99} = \frac{0.0095}{0.0293} = 0.3242$. The filter's sensitivity, false
positive rate and likelihood ratio did not change; the base rate fell 20-fold (prior odds from 0.25 to
0.0101), so two flagged emails in three are now legitimate.

```text
== exercise 1 check ==
(a) 1,000 emails: spam 200 -> flagged 190, not 10; legit 800 -> flagged 16, not 784
(b) P(spam | flagged) = 0.9223   P(spam | not flagged) = 0.0126
(c) LR+ = 47.5   posterior odds = 47.5 x 0.25 = 11.8750 -> 0.9223
(d) at 1% spam: P(spam | flagged) = 0.3242
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) All four cells and the margins | 3 |
| (b) Both posteriors with the law of total probability in the denominator | 4 |
| (c) $\mathrm{LR}_+ = 47.5$ and the odds conversion back to 0.922 | 3 |
| (d) 0.324, and the sentence separating "the filter" from "the population" | 4 |

**Likely mistakes:**

- Answering (b) with 0.95 ("the filter is 95% accurate"): the prosecutor's fallacy.
- Using $P(\text{flagged}) = 0.95$ or $0.2$ as the denominator instead of $0.206$.
- Treating odds as probabilities: reporting 11.875 or $0.25 \times 47.5 = $ "a probability above 1" without
  converting with $o/(1 + o)$.
- In (d), saying "the filter got worse". It did not; its precision did, because precision depends on the
  base rate.

## Exercise 2 (code)

**Solution:** [`a5_3_bayes_theorem_ex2.py`](a5_3_bayes_theorem_ex2.py).

```text
== exercise 2 (a): conditionally independent tests ==
people positive on both: 9,798
P(D | A+, B+): counted 0.7365   odds form 0.7293   (LR_A = 10.00, LR_B = 26.67)

== exercise 2 (b): correlated false positives ==
false positive rate A = 0.0904 (target 0.09)   B = 0.0303 (target 0.03)
P(A+, B+ | healthy): counted 0.01303   independence predicts 0.00270   exact 0.01284
P(D | A+, B+): counted 0.3569   odds form (assumes independence) 0.7293
exact with the correlated model: 0.3616   share of healthy double-positives with C: 0.9708
```

(a) Prior odds $\frac{1}{99}$, times $\mathrm{LR}_A = 10$ and $\mathrm{LR}_B = 0.8/0.03 = 26.67$, gives odds $2.694$, probability
$0.729$; the count (0.737, from about 9,800 double positives, standard error ≈ 0.0045) agrees.

(b) The marginal false positive rates are unchanged ($0.05 \times 0.5 + 0.95 \times 0.0684 = 0.09$ and
$0.05 \times 0.5 + 0.95 \times 0.00526 = 0.03$), so each test on its own looks exactly as before. But a healthy person
is now positive on both with probability $0.05 \times 0.25 + 0.95 \times 0.0684 \times 0.00526 = 0.0128$, almost five times the
$0.09 \times 0.03 = 0.0027$ that independence assumes, and 97% of those double false positives have $C$. The
true posterior drops to 0.36, half what the odds form says. The second test adds little evidence for
exactly the people it is supposed to help with, because it is fooled by the same thing as the first.
Learners' counted values depend on their draw order; the size of the gap (about 0.73 against about
0.36) is what is graded.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Two tests simulated with separate uniforms, conditionally independent given status | 3 |
| (a) Counted posterior and odds-form posterior, agreeing within noise | 3 |
| (b) Correlated model implemented as specified; marginal FP rates checked (≈ 0.09, 0.03) | 4 |
| (b) Posterior counted and compared with the independence formula | 3 |
| Explanation: joint false-positive probability is far above the product; the evidence is double-counted | 3 |

**Likely mistakes:**

- Reusing the same uniform `u` for both tests: then a healthy person positive on A (u < 0.09) is
  automatically positive on B whenever u < 0.03, a much stronger (and different) correlation than asked.
- Giving condition $C$ to sick people too, which changes the sensitivities.
- Checking only the posterior in (b) and not the marginal rates, so the reader cannot tell whether the
  change comes from correlation or from a mistakenly worse test.
- Concluding "the formula is wrong". Bayes' theorem is fine; the conditional-independence *assumption*
  used to multiply the likelihood ratios is what fails.

## Exercise 3 (think)

**Strong answer:** Recall and false positive rate do not depend on the base rate, so they carry over
from the balanced test set; precision does not. Per million transactions: 1,000 are fraud, of which 990
are flagged; 999,000 are honest, of which 1% = 9,990 are flagged. So
$P(\text{fraud} \mid \text{flagged}) = 990/10{,}980 = 9.0\%$: about ten honest customers are blocked for every fraudster
caught. Auto-blocking every flag would block about 1% of all honest transactions, which is a large
revenue and support cost and a churn risk. Recommendations: (1) do not hard-block on this score alone;
use tiers: block only at a much higher threshold where precision is acceptable, send the middle band to
step-up verification (3-D Secure, an SMS code) or manual review, and let the rest through; (2) choose
thresholds from the cost of each error (average fraud loss versus cost of a false block), not from
accuracy; (3) re-measure precision on recent, unbalanced production-like data, and monitor it, because the
fraud rate shifts. Numbers to see first: precision and recall at several thresholds measured at the real
base rate (a precision–recall curve, B4.1), the cost of a fraud loss and of a false block, the review
team's capacity per day, and how much the fraud rate varies by segment and season.

**Weak answer:** "The model is 99% accurate, so blocking flagged transactions is fine; we would only
annoy 1% of customers." Weak because it reads recall as precision, never computes that 91% of blocks are
wrong, and treats "1% of customers" as small without comparing it with the 0.1% fraud it is meant to stop.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Computes $P(\text{fraud} \mid \text{flagged}) \approx 9\%$ (formula or natural frequencies) | 4 |
| Explains why the balanced test set misleads (precision depends on the base rate) | 2 |
| A concrete alternative (tiers, step-up verification, cost-based threshold) | 2 |
| Names the numbers needed (precision at real base rate, costs, review capacity, drift) | 2 |

## Scoring

Total 40 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → A5.3
"The math" (Bayes, odds form); exercise 2 → A5.3 "From scratch" and the conditional-independence
paragraph; exercise 3 → A5.3 "With scikit-learn" (which metrics depend on the base rate) and B4.1.
