# Answer key: A6.4 Entropy, cross-entropy and KL divergence

Lesson: [`lessons/A/a6_4_entropy_cross_entropy_kl.md`](../../../lessons/A/a6_4_entropy_cross_entropy_kl.md)

## Exercise 1 (by hand)

**Solution (a).** Fair 4-sided die: $\log_2 4 = 2$ bits. $(0.5, 0.25, 0.125, 0.125)$: $0.5 \times 1 + 0.25 \times 2 + 2 \times 0.125 \times 3 = 0.5 + 0.5 + 0.75 = 1.75$ bits
(surprisals in bits are $-\log_2 0.5 = 1$, $-\log_2 0.25 = 2$, $-\log_2 0.125 = 3$). $(0.97, 0.01, 0.01, 0.01)$: $0.97 \times 0.0439 + 3 \times 0.01 \times 6.644 = 0.0426 + 0.1993 = 0.242$ bits.
The order (2 > 1.75 > 0.24) follows how predictable the outcome is: uniform is the most uncertain (the maximum, $\log_2 K$);
the second is skewed but every outcome is plausible; the third is almost certain, and the rare outcomes, though
very surprising, are too rare to add much on average.

**(b)** $H(p) = \log 2 = 0.6931$. $H(p, q) = -0.5\log 0.9 - 0.5\log 0.1 = 0.0527 + 1.1513 = 1.2040$. $D_{\mathrm{KL}}(p \,\|\, q) = 1.2040 - 0.6931 = 0.5108$ (directly:
$0.5\log\frac{0.5}{0.9} + 0.5\log\frac{0.5}{0.1} = -0.2939 + 0.8047 = 0.5108$). $D_{\mathrm{KL}}(q \,\|\, p) = 0.9\log 1.8 + 0.1\log 0.2 = 0.5290 - 0.1609 = 0.3681$. Different, as expected.

**(c)** Class 2 with probability 0.6: $-\log 0.6 = 0.5108$. Class 4 with 0.25: $-\log 0.25 = 1.3863$. Mean 0.9486. A model that always outputs
the uniform distribution gets $\log 4 = 1.3863$ on every example. Seeing a training loss stuck at about $\log K$ (1.386 for 4 classes,
2.303 for 10, 0.693 for 2) means the model is predicting uniform: nothing has been learned yet (a learning rate too
small or too large, labels shuffled relative to inputs, dead activations). If the classes are imbalanced the floor of a
"know nothing" model is the label entropy, lower than $\log K$.

**(d)** $\frac12(0.25 + 0 - 1 - \log 0.25) = \frac12(-0.75 + 1.3863) = 0.3181$. $\frac12(1 + 4 - 1 - \log 1) = 2$.

```text
== exercise 1 check ==
(a) H([0.25, 0.25, 0.25, 0.25]) = 2.0000 bits
(a) H([0.5, 0.25, 0.125, 0.125]) = 1.7500 bits
(a) H([0.97, 0.01, 0.01, 0.01]) = 0.2419 bits
(b) H(p) = 0.6931   H(p, q) = 1.2040   KL(p||q) = 0.5108   KL(q||p) = 0.3681
(c) losses 0.5108 and 1.3863   mean 0.9486   uniform model: log 4 = 1.3863
(d) KL(N(0.0, 0.5^2) || N(0, 1)) = 0.3181
(d) KL(N(2.0, 1.0^2) || N(0, 1)) = 2.0000
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) 2, 1.75 and 0.24 bits, with an explanation of the order | 3 |
| (b) All four values, KL computed or checked both ways | 4 |
| (c) Both losses, the mean, $\log 4$ for the uniform model and what it means in a training log | 4 |
| (d) 0.318 and 2 from the closed form | 3 |

**Likely mistakes:**

- Mixing bases: computing in nats in (a), or in bits in (b). State the unit on every number.
- In (b), writing $H(p, q)$ as $-\sum q\log p$ (the arguments are not interchangeable).
- In (c), averaging probabilities first and then taking the log ($-\log((0.6 + 0.25)/2) = 0.86$), which is not the mean loss.
- In (d), using $\sigma$ instead of $\sigma^2$ in the formula (giving $\frac12(0.5 - 1 - \log 0.5)$).

## Exercise 2 (code)

**Solution:** [`a6_4_entropy_cross_entropy_kl_ex2.py`](a6_4_entropy_cross_entropy_kl_ex2.py).

```text
== exercise 2: stable cross-entropy from logits ==
random logits: mine 2.865892   sklearn log_loss 2.865892   OK
logits (1000, 0, -1000), classes [0, 1]: naive nan   stable 500.0000   (exact: (0 + 1000) / 2 = 500)

== exercise 2: mutual information versus correlation ==
independent  mutual information = 0.0002 nats   |Pearson r| = 0.0067
linear       mutual information = 0.1864 nats   |Pearson r| = 0.5607
|x| > 1      mutual information = 0.4851 nats   |Pearson r| = 0.0031
```

The stable version subtracts the row maximum before `exp`, so the largest exponent is $e^0 = 1$ and nothing overflows;
$\log\sum_j e^{z_j - m} \ge 0$ is always finite. For the row $(1000, 0, -1000)$, class 0 has log-probability $\approx 0$ and class 1 has
$\approx -1000$, so the mean loss is 500. The naive version computes $e^{1000} = \infty$, then $\infty/\infty = $ `nan`. The mutual information
of the $\lvert x\rvert > 1$ feature is the largest of the three, while its Pearson correlation is 0: the dependence is symmetric
(large positive and large negative $x$ both give label 1), so no straight line captures it. Binning makes mutual
information depend on the number of bins; 10 quantile bins is a reasonable default, and the independent feature's
0.0002 shows the small positive bias that binned estimates always have.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Log-softmax with the max subtracted; never takes `log(softmax)` | 4 |
| Matches `log_loss` on random logits | 2 |
| Extreme logits: naive fails (`nan`/`inf`), stable gives 500 | 3 |
| Three synthetic features as specified; MI by quantile binning with `mutual_info_score` or own code | 4 |
| Correct reading: correlation misses the symmetric dependence; MI's binning bias noted | 2 |
| Readability: shapes in docstrings, `SEED = 0`, warnings handled deliberately | 1 |

**Likely mistakes:**

- Subtracting the global maximum of the whole matrix instead of the per-row maximum (works for the test case,
  fails when rows have very different scales).
- Calling `log_loss` with logits instead of probabilities.
- Using equal-width bins for the Gaussian features: the outer bins are nearly empty and the estimate is noisier.
- Concluding that the $\lvert x\rvert > 1$ feature is useless because its correlation is 0.

## Exercise 3 (think)

**Strong answer:** Perplexity $e^{\mathcal{L}}$ is the exponential of the average cross-entropy per token: $e^{2.30} \approx 10$ means the model is,
on average, as uncertain as if it were choosing uniformly among about 10 tokens at each step. It is not "10 times
better than random": a uniformly random model over 50,000 tokens has loss $\log 50{,}000 = 10.82$ nats and perplexity
50,000, so this model is far better than random (a perplexity 5,000 times smaller). The loss can never go below the
entropy of the text itself (per token, under the true distribution of language); we do not know that number, so a
loss of 2.30 does not tell us how close to the floor we are, only that it is above it. Comparisons across tokenizers
mislead because "per token" changes meaning: a tokenizer with longer tokens has fewer, harder-to-predict tokens per
sentence, so its per-token loss is higher even for an equally good model. Compare in a tokenizer-independent unit
(bits per character or per byte: total nats over the text divided by the number of characters, times $1/\ln 2$), on the
same evaluation text.

**Weak answer:** "Perplexity 10 means the model is right 10% of the time, so it is 10 times better than a random guess
among 10." Weak because perplexity is not an accuracy, the random baseline is the vocabulary size, not 10, and it
ignores the tokenizer issue.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Perplexity as the effective number of equally likely choices; $e^{\mathcal{L}}$ | 2 |
| Uniform baseline: $\log 50{,}000 = 10.8$ nats, perplexity 50,000 | 3 |
| The floor is the entropy of the data, unknown, and above 0 | 2 |
| Tokenizer dependence and a fix (bits per character/byte on the same text) | 3 |

## Scoring

Total 40 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → A6.4 "The math"
(definitions, Gaussian KL) and A5.5 (negative log-likelihood); exercise 2 → A6.4 "From scratch" and C1.4 (stable
softmax); exercise 3 → A6.4 "Cross-entropy as the classification loss" and F2.2 (perplexity).
