# Answer key: B2.4 Naive Bayes: Gaussian and multinomial, with a spam filter

Lesson: [`lessons/B/b2_4_naive_bayes.md`](../../../lessons/B/b2_4_naive_bayes.md)

## Exercise 1 (by hand)

**Solution (a).** "cash prize at lunch" uses four vocabulary words. Counts from the worked example: ham has cash 0, prize 0,
at 1, lunch 3 ($N_{\text{ham}} = 9$); spam has cash 1, prize 1, at 0, lunch 0 ($N_{\text{spam}} = 7$). With $\alpha = 1$, $V = 10$:

- ham: $\frac35 \cdot \frac{1}{19} \cdot \frac{1}{19} \cdot \frac{2}{19} \cdot \frac{4}{19} = \frac{3 \cdot 8}{5 \cdot 19^4} = \frac{24}{651{,}605} = 3.683 \times 10^{-5}$
- spam: $\frac25 \cdot \frac{2}{17} \cdot \frac{2}{17} \cdot \frac{1}{17} \cdot \frac{1}{17} = \frac{8}{5 \cdot 17^4} = \frac{8}{417{,}605} = 1.916 \times 10^{-5}$
- $P(\text{spam} \mid \text{message}) = 1.916/(1.916 + 3.683) = 0.342$.

**(b)** $\alpha = 0.1$: ham $\frac35 \cdot \frac{0.1}{10} \cdot \frac{0.1}{10} \cdot \frac{1.1}{10} \cdot \frac{3.1}{10} = 2.046 \times 10^{-6}$; spam $\frac25 \cdot \frac{1.1}{8} \cdot \frac{1.1}{8} \cdot \frac{0.1}{8} \cdot \frac{0.1}{8} = 1.182 \times 10^{-6}$; $P(\text{spam}) = 0.366$. It
moved slightly *towards* spam. Each class has two words it never saw, and less smoothing makes unseen words much
less likely in both classes. Spam has fewer words in total (7 against 9), so the pseudo-count is a bigger share of its
total: an unseen word gets $0.1/8 = 0.0125$ in spam against $0.1/10 = 0.01$ in ham. With so little data, the answer depends
on $\alpha$, which is why $\alpha$ is tuned.

**(c)** lunch: $\log\frac{1/17}{4/19} = \log 0.2794 = -1.275$ (evidence for ham). win: $\log\frac{3/17}{1/19} = \log 3.353 = +1.210$ (evidence for spam).

**(d)** Bias $\log\frac{0.4}{0.6} = -0.405$. "win": $-0.405 + 1.210 = 0.804$, $P(\text{spam}) = 0.691$. "win win win": $-0.405 + 3 \times 1.210 = 3.224$, $P(\text{spam}) = 0.962$. Repeating a word
adds no new information, but naive Bayes treats each occurrence as an independent draw and counts it again. The
confidence grows with repetition (and with correlated words), which is the overconfidence in the lesson's key plot.

```text
== exercise 1 check ==
alpha 1.0: ham 24/651605 = 3.6832e-05   spam 8/417605 = 1.9157e-05   P(spam) 0.3422
alpha 0.1: ham 1023/500000000 = 2.0460e-06   spam 121/102400000 = 1.1816e-06   P(spam) 0.3661
(c) weight lunch -1.2751   weight win 1.2098   bias log(0.4/0.6) -0.4055
(d) 'win' x 1: log-odds 0.8044   P(spam) 0.6909
(d) 'win' x 3: log-odds 3.2240   P(spam) 0.9617
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) Correct counts, smoothed fractions with $N_c + \alpha V$ in the denominator, both joint scores, normalized posterior | 4 |
| (b) Recomputed with $\alpha = 0.1$, direction stated, and a reason that refers to the different class totals | 3 |
| (c) Both weights with the right sign | 2 |
| (d) Both log-odds, and the explanation that repetition is counted as independent evidence | 3 |

**Likely mistakes:**

- Denominator $N_c + \alpha$ instead of $N_c + \alpha V$: the probabilities no longer sum to 1.
- Counting "a" in the spam total (it has one letter and is not in the vocabulary), giving $N_{\text{spam}} = 8$.
- Forgetting the prior, or using $P(\text{spam}) = 0.5$.
- Reporting the joint scores as the posterior. They must be normalized by their sum.
- In (b), saying "less smoothing always makes the answer more extreme". Here it moved towards the middle.

## Exercise 2 (code)

**Solution:** [`b2_4_naive_bayes_ex2.py`](b2_4_naive_bayes_ex2.py). A `Vectorizer` whose vocabulary is rebuilt from the
training part of every fold; multinomial NB with log-sum-exp; $\alpha$ chosen by 5-fold stratified CV log-loss; then the
validation report and the threshold. For the number variant, digit runs become the word `numtoken` (a placeholder like
`<num>` would be split by the tokenizer's word rule, since `<` and `>` are not word characters).

```text
== from-scratch spam filter ==
[words] CV log-loss by alpha: 0.01: 0.1465  0.03: 0.1227  0.1: 0.1018  0.3: 0.0897  1.0: 0.0969  3.0: 0.1574
[words] alpha 0.3: val accuracy 0.9856  precision 0.9586  recall 0.9329  log-loss 0.1128
[words] threshold 0.892520 (FP allowed 0 of 965 ham): FP 0  spam recall 0.9262
[words] matches CountVectorizer + MultinomialNB: OK
[numbers -> one token] CV log-loss by alpha: 0.01: 0.1331  0.03: 0.1093  0.1: 0.0900  0.3: 0.0823  1.0: 0.0860  3.0: 0.1104
[numbers -> one token] alpha 0.3: val accuracy 0.9901  precision 0.9600  recall 0.9664  log-loss 0.0890
[numbers -> one token] threshold 0.856500 (FP allowed 0 of 965 ham): FP 0  spam recall 0.9463
```

- $\alpha = 0.3$ minimizes CV log-loss. Too little smoothing ($0.01$) makes rare words decisive, too much ($3$) washes them out.
- "At most 1 in 1,000" on 965 ham messages means zero false positives. The threshold sits just above the highest-scoring ham
  message, and still catches 93% of spam (95% with the number token). But this threshold was chosen and evaluated on the
  same 965 messages, so it is optimistic. Confirm it on the test set, and expect to need far more ham messages to
  estimate a 0.1% rate reliably (with 965 messages, 0 errors is consistent with a true rate up to about 0.3%, the "rule of three", A6.2).
- Mapping numbers to one token helps on every metric: phone numbers, prices and short codes are individually rare but
  collectively a strong spam signal. One token pools their evidence. The 0.4-point accuracy gain is 5 messages, so it
  would need confirming with cross-validation, but CV log-loss agrees (0.082 against 0.090).

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Own tokenizer, vocabulary and count matrix (sparse), with unseen words ignored | 3 |
| Log-space NB with smoothing and log-sum-exp; checked against scikit-learn | 3 |
| $\alpha$ tuned by stratified CV on training messages only, with the vocabulary rebuilt inside each fold | 3 |
| Validation accuracy, precision, recall and log-loss reported | 2 |
| Threshold meeting the false-positive constraint, with recall at that threshold, and a note that it is optimistic | 3 |
| Tokenizer variant tried and judged with appropriate caution | 2 |

**Likely mistakes:**

- Building the vocabulary once on all training messages and then cross-validating: a mild leak (words seen only in a
  validation fold are in the vocabulary), and a habit to avoid.
- Tuning $\alpha$ on accuracy, which barely moves between values. Log-loss is far more sensitive to smoothing.
- Choosing the threshold as "the smallest $t$ with FP rate ≤ 0.1%" but computing the FP rate over *all* messages instead of over ham messages.
- Using `<num>` as the placeholder and not noticing that the tokenizer turns it into `num`. That happens to work, but by accident.

## Exercise 3 (think)

**Strong answer:** Build the naive Bayes (or TF-IDF + logistic regression) baseline first: it takes an hour, needs no
GPU, and with 600 emails over 12 queues (about 50 per queue, probably far fewer for rare queues) a model with few
effective parameters per class is a sensible start. Measure it properly: stratified cross-validation (the data is too
small for a single split to mean much), macro-averaged F1 and per-queue recall (B4.1), and a confusion matrix to see which
queues get mixed up. Then try stronger options in order of cost: logistic regression on TF-IDF; a pretrained sentence
embedding with logistic regression on top (I2.1); then an LLM with a few labelled examples in the prompt (I1.2), compared on the
same folds. Fine-tuning a large model on 600 emails is unlikely to pay off before those. Switch when a candidate beats
the baseline by more than the cross-validation noise, at an acceptable cost and latency. The "95% confident" rule is
the real issue. Naive Bayes's probabilities are badly overconfident (the lesson's key plot: most messages beyond 0.999),
so a 0.95 threshold on raw naive Bayes output would auto-send far more answers than are actually 95% correct. Whatever
the model, calibrate its probabilities on held-out data (Platt scaling or isotonic regression, B4.1). Then choose the
auto-send threshold from a reliability curve, so that the emails sent automatically really are right about 95% of the
time, and measure what share of emails clears that bar. Log every auto-sent decision and sample them for human review,
because the email mix will drift.

**Weak answer:** "Naive Bayes is fine because text is its classic use. Train it, and auto-send when `predict_proba` > 0.95."

**Why it is weak:** it skips evaluation on so little data (no cross-validation, no per-queue metrics), never compares
alternatives, and uses naive Bayes's raw probabilities for an automatic action, which is exactly where its
overconfidence does harm.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Starts with a cheap baseline and justifies it by data size and cost | 2 |
| Evaluation fit for 600 rows and 12 classes: stratified CV, per-class or macro metrics, confusion matrix | 2 |
| A path of stronger candidates and a switching rule based on a gain beyond CV noise, cost and latency | 2 |
| Identifies naive Bayes's overconfidence as a problem for the 95% rule; calibrates; picks the threshold from a reliability curve | 3 |
| Monitoring or human review of auto-sent answers | 1 |

## Scoring

Total 38 pts. Pass ≥ 70% (27 pts). Topic ids to revisit for each lost criterion: Exercise 1 → B2.4, A5.3; Exercise 2 →
B2.4, B4.3 (cross-validation), B4.1 (thresholds), B1.4; Exercise 3 → B2.4, B4.1 (calibration, macro metrics), E1.2, I2.1.
