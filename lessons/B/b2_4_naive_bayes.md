# B2.4 Naive Bayes: Gaussian and multinomial, with a spam filter

**Core** · Part B, week 2 (course week 08) · ~1.5 h · Lab:
[`labs/B/b2_4_naive_bayes.py`](../../labs/B/b2_4_naive_bayes.py)

## Why it matters

**Naive Bayes** is Bayes' theorem (A5.3) turned into a classifier by one bold simplification. Training is counting, so
it fits 3,000 text messages in about 2 milliseconds, works with tiny datasets, and handles vocabularies of hundreds of
thousands of words. It was the engine of the first effective spam filters and is still the fastest reasonable baseline
for text classification (E1.2). Two things make it worth understanding properly. Its smoothing trick (never let a
probability be exactly zero) appears again in language models. And it shows a gap that matters in production: a
classifier can rank well (high accuracy) while its probabilities are badly wrong (99.9999% sure of the wrong answer).
If you feed those probabilities into a decision or a threshold, you need to know that.

## Prerequisite check

- Can you apply Bayes' theorem, $P(A \mid B) = P(B \mid A)P(A)/P(B)$, and explain what the prior and the likelihood are? If not, revisit
  [A5.3](../A/a5_3_bayes_theorem.md).
- Do you know what conditional independence means? If not, revisit [A5.3](../A/a5_3_bayes_theorem.md).
- Can you write the density of a normal distribution and its log? If not, revisit [A5.2](../A/a5_2_common_distributions.md).
- Can you explain why the MLE of a category's probability is its observed frequency, and what a prior does to it? If
  not, revisit [A5.5](../A/a5_5_mle_map.md).
- Can you read a logistic regression's log-odds as a sum of weighted features? If not, revisit
  [B2.1](b2_1_logistic_regression.md).

## Intuition

We want $P(\text{spam} \mid \text{message})$. Bayes' theorem turns it around:

$$P(\text{spam} \mid \text{message}) \propto P(\text{message} \mid \text{spam}) \times P(\text{spam}).$$

$P(\text{spam})$ is easy: the fraction of training messages that are spam (13%). $P(\text{message} \mid \text{spam})$ is hopeless as it stands: almost every message
is unique, so we have no data on the probability of that exact message. The **naive** step: *pretend that, once you
know the class, the words are independent of each other*. Then the probability of the message is the product of the
probabilities of its words, and each word's probability is easy to estimate: how often "prize" appears in spam,
divided by the number of words in spam.

The assumption is false. "Claim" and "prize" travel together in spam, and "see you later" is a phrase, not three
independent draws. But the classifier often works anyway, because classifying only needs the right *winner*, not the
right probability. Each word contributes a vote, the log of $P(\text{word} \mid \text{spam})/P(\text{word} \mid \text{ham})$, and the votes are added. Correlated words get
counted several times, so the total is too extreme. A message is called "spam with probability 0.999999" when 0.95
would be honest. The sign of the total, which decides the class, is usually still right.

One trap needs fixing. If "lunch" never appeared in a spam message in the training set, its estimated probability given spam is
0, and a single "lunch" makes the whole product zero, whatever else the message says. **Laplace smoothing** adds a small
pseudo-count (usually 1) to every word in every class: "assume every word has been seen once more than it was".

> **Bridge:** naive Bayes is a scoring rule built from per-feature lookup tables, like a fraud score that adds a
> precomputed risk weight per merchant, per country and per device. Training fills the tables with log-ratios by
> counting (`GROUP BY class, word`); prediction is a sum of lookups. **Where it breaks:** a hand-built rules engine lets a
> person avoid counting the same evidence twice. Naive Bayes cannot: if three features are all proxies for "new device",
> it adds all three weights in full, which is why its scores are overconfident even when its decisions are good.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $K$, $c$ | number of classes; a class index | int |
| $V$ | vocabulary size (number of distinct training words) | int |
| $\mathbf{x}$ | one message as word counts: $x_w$ = times word $w$ occurs | `(V,)` |
| $\mathbf{X}$ | count matrix for $n$ messages (mostly zeros: stored sparse) | `(n, V)` |
| $\pi_c$ | prior $P(y = c)$, estimated as the class's share of training rows | scalar |
| $N_{cw}$ | total count of word $w$ in training messages of class $c$ | scalar |
| $N_c$ | total count of all words in class $c$: $\sum_w N_{cw}$ | scalar |
| $\theta_{cw}$ | $P(\text{word } w \mid \text{class } c)$, the smoothed estimate | matrix `(K, V)` |
| $\alpha$ | smoothing pseudo-count (Laplace: $\alpha = 1$) | scalar |
| $\mu_{cj}$, $\sigma^2_{cj}$ | Gaussian NB: mean and variance of feature $j$ in class $c$ | `(K, d)` each |

**Step 1: Bayes' theorem for a class.**

$$P(y = c \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y = c)\,\pi_c}{P(\mathbf{x})}, \qquad P(\mathbf{x}) = \sum_{c'} P(\mathbf{x} \mid y = c')\,\pi_{c'}.$$

The denominator is the same for every class, so for the prediction we only compare the numerators.

**Step 2: the naive assumption.** Features are independent given the class (**conditional independence**):
$P(\mathbf{x} \mid y = c) = \prod_j P(x_j \mid y = c)$. One model per feature, instead of one model of all features jointly.

**Step 3, multinomial NB (word counts).** Treat a message of $L$ words as $L$ independent draws from a class-specific bag
of words, where word $w$ comes up with probability $\theta_{cw}$. Then $P(\mathbf{x} \mid c) \propto \prod_w \theta_{cw}^{x_w}$: a word that occurs twice contributes its probability
twice. The proportionality constant counts the possible word orders. It does not depend on $c$, so it cancels in Step 1.

**Step 4: estimating $\theta$ by counting, with smoothing.** The MLE of a category's probability is its frequency (A5.5):
$\hat\theta_{cw} = N_{cw}/N_c$. That gives 0 for any word never seen in class $c$. Laplace smoothing adds $\alpha$ to every count:

$$\theta_{cw} = \frac{N_{cw} + \alpha}{N_c + \alpha V}.$$

The denominator grows by $\alpha V$ so that the probabilities still sum to 1 over the vocabulary: $\sum_w (N_{cw} + \alpha) = N_c + \alpha V$. In A5.5 terms, this is
the MAP estimate under a prior (a Dirichlet distribution) that says "every word is possible in every class". Words seen
at test time but never in training are simply ignored: they are not in the vocabulary.

**Step 5: work in logs.** A product of hundreds of probabilities like 0.001 underflows to 0.0 in floating point. Take logs:

$$\log P(y = c \mid \mathbf{x}) = \underbrace{\log\pi_c + \sum_w x_w\log\theta_{cw}}_{s_c \text{, the joint log score}} - \log\sum_{c'} e^{s_{c'}}.$$

For a batch this is one sparse matrix product: $\mathbf{S} = \mathbf{X}\,(\log\boldsymbol{\Theta})^\top + \log\boldsymbol{\pi}$, shape `(n, V)(V, K)` → `(n, K)`. The last term, the log of a sum of
exponentials, is computed with **log-sum-exp**: subtract the largest $s_c$ before exponentiating, so nothing overflows
(`scipy.special.logsumexp`; the same trick makes softmax stable in C1.4).

**Step 6: it is a linear classifier.** For two classes, the posterior log-odds are

$$\log\frac{P(\text{spam} \mid \mathbf{x})}{P(\text{ham} \mid \mathbf{x})} = \underbrace{\log\frac{\pi_{\text{spam}}}{\pi_{\text{ham}}}}_{\text{bias}} + \sum_w x_w\underbrace{\log\frac{\theta_{\text{spam},w}}{\theta_{\text{ham},w}}}_{\text{weight of word } w}.$$

This has exactly the form of a logistic regression on word counts (B2.1): a bias plus a weighted sum, read as log-odds.
The difference is how the weights are chosen. Naive Bayes is **generative**: it models how each class produces data,
and sets each weight by counting that word on its own. Logistic regression is **discriminative**: it chooses all weights
together to predict the label. If two words always appear together, logistic regression can split the credit between them.
Naive Bayes gives each its full log-ratio, so the evidence is counted twice. This is the source of the overconfidence.

**Step 7, Gaussian NB (continuous features).** Model each feature in each class as a normal distribution with its own mean and
variance (MLE: the class's sample mean and variance, dividing by $n_c$). The per-feature log density is
$-\frac12\left[\log(2\pi\sigma^2_{cj}) + (x_j - \mu_{cj})^2/\sigma^2_{cj}\right]$, and the joint log score is $\log\pi_c$ plus the sum over features. scikit-learn adds a tiny floor to every
variance ($10^{-9}$ times the largest feature variance) so a constant feature does not divide by zero.

## Worked example by hand

Five training messages, three ham and two spam. The tokenizer keeps words of two or more letters, so "a" is dropped:

| Message | Class |
|---|---|
| win cash now | spam |
| win a free prize now | spam |
| lunch now | ham |
| see you at lunch | ham |
| free for lunch | ham |

Vocabulary: at, cash, for, free, lunch, now, prize, see, win, you ($V = 10$). Word totals: ham $N_{\text{ham}} = 2 + 4 + 3 = 9$, spam $N_{\text{spam}} = 3 + 4 = 7$.
Priors $\pi_{\text{ham}} = 3/5$, $\pi_{\text{spam}} = 2/5$.

Classify **"free lunch now win"**. Smoothed probabilities, $(N_{cw} + 1)/(N_c + 10)$:

| Word | ham count | $P(w \mid \text{ham})$ | spam count | $P(w \mid \text{spam})$ |
|---|---|---|---|---|
| free | 1 | 2/19 = 0.1053 | 1 | 2/17 = 0.1176 |
| lunch | 3 | 4/19 = 0.2105 | 0 | 1/17 = 0.0588 |
| now | 1 | 2/19 = 0.1053 | 2 | 3/17 = 0.1765 |
| win | 0 | 1/19 = 0.0526 | 2 | 3/17 = 0.1765 |

Joint scores: ham $= \frac35 \times \frac{2 \cdot 4 \cdot 2 \cdot 1}{19^4} = 0.6 \times \frac{16}{130{,}321} = 7.366 \times 10^{-5}$; spam $= \frac25 \times \frac{2 \cdot 1 \cdot 3 \cdot 3}{17^4} = 0.4 \times \frac{18}{83{,}521} = 8.621 \times 10^{-5}$. Normalize:
$P(\text{spam} \mid \text{message}) = 8.621/(8.621 + 7.366) = 0.539$. A close call. The spam words "win" and "now" just outweigh "lunch" and the prior's lean towards ham.

Without smoothing, $P(\text{win} \mid \text{ham}) = 0/9$ and $P(\text{lunch} \mid \text{spam}) = 0/7$: both joint scores are 0, and the classifier cannot answer at all.

```text
== worked example by hand ==
vocabulary (10): ['at', 'cash', 'for', 'free', 'lunch', 'now', 'prize', 'see', 'win', 'you']   ('a' is dropped: one letter)
ham: prior 0.6  word count 9  P(w | ham) = (count + 1) / (9 + 10): free 1->0.1053  lunch 3->0.2105  now 1->0.1053  win 0->0.0526
      log score = log 0.6 + sum log P(w | ham) = -9.5160   (product 7.3664e-05)
spam: prior 0.4  word count 7  P(w | spam) = (count + 1) / (7 + 10): free 1->0.1176  lunch 0->0.0588  now 2->0.1765  win 2->0.1765
      log score = log 0.4 + sum log P(w | spam) = -9.3588   (product 8.6206e-05)
P(spam | 'free lunch now win') = 0.5392
without smoothing: P(win | ham) = 0/9 and P(lunch | spam) = 0/7, so both classes score 0 (log = -inf)
```

## Key plot

![Histogram, log-scale counts, of naive Bayes log-odds of spam for validation messages: ham messages spread from about -60 to 0 with a peak near -10, spam messages from about 0 to +60; only a small share falls in the grey band between -6.9 and +6.9 where the probability is between 0.001 and 0.999](img/b2_4_nb_log_odds.png)

The two classes are well separated: the decision at log-odds 0 is right for 98.7% of messages. But very few messages land
in the grey band where the model admits any doubt. A log-odds of 25, the median for spam, means $P(\text{ham}) \approx 10^{-11}$. That is not a
believable probability for a text message.

## From scratch in NumPy

**The data.** The SMS Spam Collection (Almeida, Gómez Hidalgo and Yamakami, 2011): 5,574 real text messages labelled ham
or spam, downloaded once into `data/sms_spam/` and checked against a SHA-256 checksum, then split 60/20/20, stratified:

```text
== data ==
data/sms_spam/SMSSpamCollection: checksum OK, 5574 messages, spam share 0.1340
example ham:  'Ok lar... Joking wif u oni...'
example spam: 'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Te'...
train 3346  val 1114  (test 1114 kept aside)   spam: train 449  val 149
majority-class baseline (always ham) on val: accuracy 0.8662
```

Always answering "ham" is already 86.6% accurate, so accuracy alone says little here. Watch spam recall (the share of spam
caught) and precision (the share of flagged messages that really are spam). B4.1 covers these metrics properly.

**Vocabulary, counts, fit, predict.** The vocabulary comes from training messages only. The count matrix is sparse:
99.8% of its entries are zero.

```python
def fit_multinomial_nb(Xc, y, alpha):
    """Log priors (K,) and smoothed log P(word | class) (K, V) from a count matrix Xc (n, V)."""
    classes = np.unique(y)
    log_prior = np.log(np.array([np.mean(y == c) for c in classes]))                      # (K,)
    word_counts = np.vstack([np.asarray(Xc[y == c].sum(axis=0)).ravel() for c in classes])  # (K, V)
    smoothed = word_counts + alpha
    log_theta = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))             # (K, V)
    return log_prior, log_theta

def predict_log_proba_nb(Xc, log_prior, log_theta):
    """Log posteriors via Bayes' rule, normalized with log-sum-exp. Xc (n, V) -> (n, K)."""
    joint = Xc @ log_theta.T + log_prior                       # (n, K)
    return joint - logsumexp(joint, axis=1, keepdims=True)
```

```text
== from scratch: multinomial naive Bayes ==
vocabulary 6646 words   train count matrix (3346, 6646) with 44595 non-zeros (0.20% filled)
priors: P(ham) 0.8658  P(spam) 0.1342
val words not in the training vocabulary (ignored): 1185 of 15558
val accuracy 0.9874   confusion: TN 962  FP 3  FN 11  TP 138
spam precision 0.9787  spam recall 0.9262
most spam-indicating words: ['claim', 'prize', '150p', 'uk', 'tone', 'co', 'guaranteed', '18', '500', 'cs']
most ham-indicating words:  ['gt', 'lt', 'he', 'she', 'lor', 'later', 'll', 'da', 'too', 'much']
```

- 98.7% accuracy: 3 ham messages wrongly flagged and 11 spam messages missed out of 149. For a spam filter, flagging a real
  message costs more than missing a spam, so this balance is roughly what you want. The threshold (here 0.5) is the knob for it (B4.1).
- 8% of validation words were never seen in training and are ignored. This is the vocabulary problem that subword
  tokenizers solve (E1.4, F1.3).
- The top words are what a person would guess: prizes, prices in pence (`150p`), "guaranteed". The ham words include
  `gt` and `lt`: the collection encodes `>` and `<` as `&gt;` and `&lt;`, which only appear in personal messages. A model happily
  learns such artefacts, and in production they break silently if the encoding changes.

## With scikit-learn

```text
== with scikit-learn ==
CountVectorizer vocabulary and counts identical to ours: OK
MultinomialNB(alpha=1) log-probabilities identical to ours: OK
naive Bayes:          val accuracy 0.9874   val log-loss 0.1206
logistic regression:  val accuracy 0.9856   val log-loss 0.0669
naive Bayes puts 0.8187 of val messages below 0.001 or above 0.999; accuracy on those 0.9923
most confident mistake: true spam, P(spam) 6.9e-13: 'Hello darling how are you today? I would love to have a chat, why dont you tell '
~ MultinomialNB fit time 2.44 ms
```

- `CountVectorizer()` uses the same tokenizer rule by default (lowercase, then words of 2+ characters), so the vocabulary,
  the counts and every log-probability match exactly.
- Naive Bayes and logistic regression on the same counts are equally accurate (the 0.2-point difference is 2 messages). But
  naive Bayes's log-loss is almost twice as bad. Its probabilities are too extreme: 82% of messages get a probability
  beyond 0.001 or 0.999, and the 0.8% of those that are wrong cost a lot each. A spam message that reads like a personal
  one ("Hello darling…") gets $P(\text{spam}) = 7 \times 10^{-13}$.
- If you need ranking or a yes/no decision, naive Bayes is fine. If you need probabilities (expected-cost decisions,
  combining with other signals), calibrate them (B4.1) or use logistic regression.

**Gaussian naive Bayes** on the wine data, for continuous features:

```text
== from scratch: Gaussian naive Bayes on wine ==
class means of proline: [1116.9, 539.8, 650.2]
val accuracy 0.9722   matches GaussianNB (means, variances, log-probabilities): OK
within class 0, |correlation| between features: median 0.2320  max 0.8034 (naive Bayes assumes 0)
```

Thirteen features, a mean and a variance per class each, and 97% validation accuracy with no tuning and no scaling. Each
feature has its own variance, so units do not matter, unlike kNN (B2.3). Within a class, the features are far from
independent (one pair correlates at 0.80), yet the classifier still does well. Classification only needs the right winner.

## In practice

- **When to use it:** a first baseline for text classification (spam, topic, intent) and for any problem with many
  sparse count features; very small training sets (each weight is estimated on its own, so it needs little data);
  when training must be instant or incremental (`partial_fit` just adds counts); when you need a fast, transparent
  model whose weights are readable log-ratios.
- **Failure modes:**
  - *Overconfident probabilities.* Log-loss much worse than a discriminative model with the same accuracy; predicted
    probabilities piled up at 0 and 1. Calibrate (B4.1) or switch to logistic regression if the probabilities are used.
  - *Correlated or duplicated features.* Adding a copy of a feature doubles its vote and pushes the model further
    from the truth. Remove redundant features, or prefer logistic regression.
  - *No smoothing, or the wrong amount.* $\alpha = 0$ gives `-inf` log-probabilities and crashes, or warnings. A large $\alpha$ washes out rare but telling
    words. Tune $\alpha$ by cross-validation (0.01–1 is the usual range).
  - *Vocabulary drift.* New spam words, new encodings and new slang are ignored if they are not in the vocabulary. Monitor
    the share of unknown tokens and retrain regularly.
  - *Wrong event model.* `MultinomialNB` expects counts (or TF-IDF), `BernoulliNB` binary present/absent features,
    `GaussianNB` continuous roughly normal features. Gaussian NB on heavy-tailed features (incomes, counts) fits badly:
    log-transform them first.
  - *Priors from a sample that does not match production.* If training was balanced 50/50 but production is 2% positive,
    the posterior is off by the prior ratio. Adjust `class_prior` or the threshold.
- **Defaults that work:** `make_pipeline(CountVectorizer(), MultinomialNB())` (or `TfidfVectorizer`) with `alpha` tuned over
  `[0.01, 0.1, 0.5, 1.0]`; `GaussianNB()` with defaults for continuous features. Check scikit-learn 1.9.1's
  `CountVectorizer` defaults before relying on them (VERIFY as of 2026-10).

## Exercises

Save your work to `my-work/B/b2_4_naive_bayes.md` (or `.py`) and ask for a review.

1. **By hand:** Use the worked example's five training messages. (a) Classify **"cash prize at lunch"** with $\alpha = 1$: give both joint scores as
   fractions and $P(\text{spam} \mid \text{message})$. (b) Repeat with $\alpha = 0.1$. Which way did the probability move, and why? (c) Write the naive Bayes weight
   $\log(\theta_{\text{spam},w}/\theta_{\text{ham},w})$ for "lunch" and for "win" with $\alpha = 1$. (d) The message "win win win" has the same words as "win" three times over. Compute
   the log-odds of spam for "win" and for "win win win". What does this say about naive Bayes's confidence?
2. **Code:** Build `my_spam_filter.py` with your own tokenizer and multinomial naive Bayes (no `CountVectorizer`, no
   `MultinomialNB` in the model; you may use them only to check). On the lab's split: (i) tune $\alpha$ over
   `[0.01, 0.03, 0.1, 0.3, 1, 3]` by 5-fold stratified cross-validation on the training messages, using log-loss; (ii) report
   validation accuracy, spam precision, spam recall and log-loss for the chosen $\alpha$; (iii) find the threshold on $P(\text{spam})$ that keeps
   false positives at most 1 in 1,000 ham messages on the validation set, and report the spam recall at that threshold.
   Then try one tokenizer change (keep 1-character tokens, or map every number to a single token `<num>`) and report
   whether it helps.
3. **Think:** A support team wants to auto-route customer emails to one of 12 queues. They have 600 labelled emails.
   A colleague proposes naive Bayes "because it is old and simple"; another wants to fine-tune a large language
   model. Make the case for which you would build first and why, what you would measure, and what would make you switch.
   Address the probability issue: the router will auto-send an answer when it is "more than 95% confident".

## Takeaways

1. Naive Bayes applies Bayes' theorem with the assumption that features are independent given the class:
   $P(c \mid \mathbf{x}) \propto \pi_c\prod_j P(x_j \mid c)$. Training is counting (multinomial) or computing class means and variances (Gaussian).
2. Laplace smoothing $(N_{cw} + \alpha)/(N_c + \alpha V)$ keeps unseen words from zeroing a whole class; it is a MAP estimate under a prior that every word is possible.
3. Compute in logs and normalize with log-sum-exp. The posterior log-odds are a linear function of the counts, with weights
   $\log(\theta_{\text{spam},w}/\theta_{\text{ham},w})$: a linear classifier whose weights are set by counting, not by optimizing.
4. Because correlated words each vote in full, naive Bayes is overconfident. Accuracy can match logistic regression while
   log-loss is twice as bad, so do not use its probabilities raw.
5. Use it as a fast, strong baseline for text and sparse counts, and with small data. Match the event model to the
   features, tune $\alpha$, and monitor vocabulary drift.

**What builds on this:** [B4.1](b4_1_classification_metrics.md) (precision, recall, calibration);
[B4.4](b4_4_class_imbalance.md) (priors and thresholds when classes are rare); [B5.3](b5_3_gmm_em.md) (Gaussian mixtures:
a generative model with hidden classes); [E1.1](../E/e1_1_text_preprocessing.md) and [E1.2](../E/e1_2_bow_tfidf.md)
(tokenizing and the TF-IDF text baseline); [C1.4](../C/c1_4_stable_softmax_cross_entropy.md) (log-sum-exp in softmax).

**References:** T. A. Almeida, J. M. Gómez Hidalgo, A. Yamakami, "Contributions to the Study of SMS Spam Filtering: New
Collection and Results", *ACM DocEng*, 2011 (the dataset). A. McCallum, K. Nigam, "A Comparison of Event Models for Naive
Bayes Text Classification", *AAAI Workshop on Learning for Text Categorization*, 1998. P. Domingos, M. Pazzani, "On the
Optimality of the Simple Bayesian Classifier under Zero-One Loss", *Machine Learning* 29, 1997. A. Ng, M. Jordan, "On
Discriminative vs. Generative Classifiers: A Comparison of Logistic Regression and Naive Bayes", *NeurIPS*, 2001.
scikit-learn User Guide, "Naive Bayes" (VERIFY link: https://scikit-learn.org/stable/modules/naive_bayes.html).
