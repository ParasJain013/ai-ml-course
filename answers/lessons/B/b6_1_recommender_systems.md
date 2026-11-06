# Answer key: B6.1 Recommender systems: collaborative filtering and matrix factorization from scratch, implicit feedback

Lesson: [`lessons/B/b6_1_recommender_systems.md`](../../../lessons/B/b6_1_recommender_systems.md)

## Exercise 1 (by hand)

**Solution (a).** Centred rows: Ben $(0.25, 1.25, 0.25, -1.75)$, Dee $(1.333, 0, 1.333, -2.667)$.
Dot product $= 0.25 \times 1.333 + 1.25 \times 0 + 0.25 \times 1.333 + (-1.75)(-2.667) = 0.333 + 0 + 0.333 + 4.667 = 5.333$.
$\lVert\mathbf{c}_{\text{Ben}}\rVert = \sqrt{4.75} = 2.179$; $\lVert\mathbf{c}_{\text{Dee}}\rVert = \sqrt{1.778 + 0 + 1.778 + 7.111} = \sqrt{10.667} = 3.266$.
$s_{\text{Ben,Dee}} = 5.333/(2.179 \times 3.266) = 5.333/7.117 = 0.749$.

**(b)** Item 2 was rated by Ana, Ben and Cy. Similarities to Dee: Ana $0.878$ (from the lesson, cosine is symmetric), Ben $0.749$, and Cy:
dot $= 1.333(-1.25) + 0 + 1.333(-1.25) + (-2.667)(2.75) = -1.667 - 1.667 - 7.333 = -10.667$; $\lVert\mathbf{c}_{\text{Cy}}\rVert = \sqrt{10.75} = 3.279$;
$s = -10.667/(3.266 \times 3.279) = -0.996$. The two most similar are Ana and Ben. Their centred ratings of item 2 are $0.667$ and $1.25$:
$$\hat r = 3.667 + \frac{0.878 \times 0.667 + 0.749 \times 1.25}{0.878 + 0.749} = 3.667 + \frac{0.585 + 0.936}{1.627} = 3.667 + 0.935 = 4.602 .$$
(Had you used all three, Cy's $-0.996 \times -0.25 = +0.249$ would join the numerator and $0.996$ the denominator: $3.667 + 1.770/2.623 = 4.342$.)

**(c)** Item A: $b_A = (20 - 4 \times 3.5)/(25 + 4) = 6/29 = 0.207$ (plain mean 5.0). Item B: $b_B = (1700 - 400 \times 3.5)/(25 + 400) = 300/425 = 0.706$
(plain mean 4.25). Rank B higher: four perfect ratings are weak evidence (a few fans), 400 ratings averaging 4.25 are strong evidence. The damping
says "with 4 ratings, believe about $4/29 = 14\%$ of the observed deviation; with 400, believe $400/425 = 94\%$".

**(d)** $\mathbf{p}_u\cdot\mathbf{q}_i = 0.5 \times 0.4 + 0 = 0.2$; $\hat r = 3.5 + 0.2 + 0.3 + 0.2 = 4.2$; $e = 2 - 4.2 = -2.2$.
$b_u = 0.2 + 0.1(-2.2 - 0.02) = -0.022$; $b_i = 0.3 + 0.1(-2.2 - 0.03) = 0.077$.
$\mathbf{p}_u = (0.5, 0) + 0.1[-2.2(0.4, 0.2) - 0.1(0.5, 0)] = (0.5, 0) + 0.1(-0.93, -0.44) = (0.407, -0.044)$.
$\mathbf{q}_i = (0.4, 0.2) + 0.1[-2.2(0.5, 0) - 0.1(0.4, 0.2)] = (0.4, 0.2) + 0.1(-1.14, -0.02) = (0.286, 0.198)$ (old $\mathbf{p}_u$).
New prediction $3.5 - 0.022 + 0.077 + (0.407 \times 0.286 - 0.044 \times 0.198) = 3.555 + 0.116 - 0.009 = 3.663$, down from 4.2 towards 2.

```text
== exercise 1 check ==
(a) s(Ben, Dee) = 0.749
(b) s(Dee, Ana/Ben/Cy) = [0.878, 0.749, -0.996]
    Dee, item 2: 4.602
(c) b_A = 0.2069, b_B = 0.7059
(d) r_hat 4.20, e -2.20; b_u -0.0220, b_i 0.0770, p_u [0.407, -0.044], q_i [0.286, 0.198]; new r_hat 3.6627
```

**Rubric (15 pts):**

| Criterion | Points |
|---|---|
| (a) Centred rows used; dot product over all items with missing = 0; $s = 0.749$ | 3 |
| (b) Cy's similarity computed and correctly excluded as not among the top 2; weighted average with $\lvert s\rvert$ in the denominator; 4.60 | 4 |
| (c) Both damped biases; B ranked higher with the evidence argument | 3 |
| (d) Error sign right; all four updates use the *old* values; new prediction 3.66 | 5 |

**Likely mistakes:**

- Computing cosine on the *raw* ratings (all positive, so every similarity is high, e.g. Ana–Cy becomes positive) instead of centred ratings.
- Picking neighbours by $\lvert s\rvert$ and so including Cy among the "two most similar". That is a defensible design (opposites carry information),
  but it is not the rule the lesson and the lab use; say which rule you used.
- Updating $\mathbf{p}_u$ first and then using the *new* $\mathbf{p}_u$ in the $\mathbf{q}_i$ update.
- Sign error on $e$: with $\hat r > r$, $e < 0$ and every term must move the prediction *down*.
- Ranking A first because "5.0 > 4.25": forgetting that a mean of four ratings is noisy (A5.4, B2.2).

## Exercise 2 (code)

**Solution:** [`b6_1_recommender_systems_ex2.py`](b6_1_recommender_systems_ex2.py) (copies the lab's data split, bias model and MF; it also
checks exercise 1).

```text
== data and the lab's time split ==
train+validation 95,285, test 4,715
bias model test RMSE 1.0424

== (a) user-based CF: centred on the user mean versus on the bias model ==
m=20   user-mean centred 1.0315   bias-model centred 1.0152
m=40   user-mean centred 1.0291   bias-model centred 1.0128
m=100  user-mean centred 1.0374   bias-model centred 1.0156

== (b) item-based CF on the bias-model residuals ==
m=20   item-based 1.0056
m=40   item-based 1.0068
m=100  item-based 1.0101

== (c) bootstrap CI: best CF versus matrix factorization ==
best CF variant: item-based, m=20, test RMSE 1.0056
MF test RMSE 1.0023 (the lab printed 1.0023)
RMSE(best CF) - RMSE(MF): 0.0034, 95% CI [-0.0051, 0.0119]
```

**What a strong answer says:**

- Centring on the bias model removes *both* the user's generosity and the film's general appeal before measuring similarity, so CF only has
  to model what is personal. It improves user-based CF at every $m$ (1.029 → 1.013 at $m = 40$).
- Item-based CF on the same residuals is better still (1.006 at $m = 20$). Item similarities are estimated from many raters per pair of
  popular films and are more stable than user similarities, which is one reason Amazon-style "people who bought X also bought Y" is item-based.
- MF is 0.003 better than the best CF, but the paired bootstrap interval $[-0.005, 0.012]$ includes 0: **on this test set the two are
  indistinguishable**. MF's real advantages are elsewhere: memory ($52{,}500$ numbers against a $1682 \times 1682$ similarity matrix), prediction cost
  (one dot product), and vectors that can be reused (retrieval in B6.2, item embeddings).
- The bootstrap must be **paired**: resample test ratings once and score both models on the same rows; resampling each model separately would
  ignore that they make correlated errors and give a much wider interval.
- Choosing "the best CF variant" by its test RMSE and then using that same test set for the comparison slightly favours CF (selection on the
  test set). A careful version picks $m$ and the variant on the validation block; say so.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Bias-model centring implemented (residuals 0 where missing; prediction adds back $\mu + b_u + b_i$) and reported for three $m$ | 4 |
| Item-based CF implemented correctly (column similarities; neighbours are items the user rated) | 4 |
| Paired bootstrap of the RMSE difference with 1,000 resamples; interval reported | 4 |
| Correct conclusion (no reliable difference) and at least one reason to prefer MF anyway | 2 |
| Same split and training rows as the lab; seeds fixed; runs in seconds | 2 |

**Likely mistakes:**

- Filling missing residuals with the bias model's prediction instead of 0 (then every user looks similar to every other through invented values).
- In item-based CF, taking neighbours among items *other users* rated instead of items *this user* rated: you can only average this user's own
  opinions of similar items.
- Unpaired bootstrap; or bootstrapping users when the test unit is a rating (both are defensible if stated, but they answer different questions).
- Declaring MF "better" because 1.0023 < 1.0056, without an interval.

## Exercise 3 (think)

**Strong answer:** The colleague's plan has four problems. (1) *Signal:* clicks on a job board are cheap and noisy (curiosity, misleading
titles); the business outcome is applications, and saves are in between. Use a weighted implicit signal (apply > save > click), and treat
impressions without a click as weak negatives (the user saw it and passed) and never-shown jobs as *unknown*, sampled as negatives at most.
(2) *Target and metric:* RMSE on clicks is meaningless for a top-of-feed ranking; evaluate recall@k and NDCG@k (B6.2) on what users applied to
next, plus coverage. (3) *Split:* a random 20% of clicks leaks the future; split by time (train on months 1–20, validate on 21–22, test on
23–24), because jobs and candidates both change. (4) *Cold start:* new jobs arrive hourly and have no interactions, so pure MF cannot score
them. Use a hybrid: a content model on job text and candidate profiles (embeddings, I2.1) for retrieval, with MF/CF signals added as features
once interactions exist, and a fresh-job boost or exploration slot. Plan: candidate retrieval from content and interaction embeddings, a ranking
model on top (B6.2), offline time-split evaluation, then an A/B test on applications per user and hires if measurable. `[RAI]` risks:
(a) historical clicks encode bias (for example certain roles shown or clicked more by one gender), so the recommender can steer groups away from
well-paid jobs; measure exposure and application rates of each job category by candidate group where lawful and available, and audit
outcomes (B4.6); (b) feedback loops starve new or small employers; measure exposure share by employer size and job age; (c) candidate
profiles are personal data; minimize what the model sees and do not use protected attributes as features. Hiring is a high-stakes domain
(regulated in several jurisdictions; VERIFY (as of 2026-11) the rules that apply to you), so document the system (model card, J2.5).
**Weak answer:** "Use implicit ALS instead of SGD and tune the number of factors." It accepts the colleague's framing and only swaps the optimizer.
**Why weak:** it does not address the signal, the split, the metric, cold start or any harm; tuning cannot fix an evaluation that leaks.

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| Better signal than raw clicks; non-clicks treated as unknown or weak negatives, not as dislikes | 2 |
| Time-based split, explained | 2 |
| Ranking metric (recall@k / NDCG@k) on the business outcome instead of RMSE | 2 |
| Concrete cold-start strategy for new jobs (content features / hybrid / exploration) | 3 |
| At least one `[RAI]` risk with a concrete measurement | 3 |

## Scoring

Total 43 pts. Pass ≥ 70% (31). Topic ids to revisit for each lost criterion: cosine and centring (Ex 1a–b, Ex 2) → A1.5 and B6.1;
damped means and L2 (Ex 1c) → B2.2; SGD updates (Ex 1d) → A4.4 and A4.3; paired bootstrap (Ex 2c) → A6.2; time split (Ex 3) → B1.4;
ranking metrics and two-stage systems (Ex 3) → B6.2; fairness audit (Ex 3) → B4.6.
