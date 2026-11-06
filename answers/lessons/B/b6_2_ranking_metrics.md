# Answer key: B6.2 Ranking metrics (precision@k, NDCG) and the retrieve-then-rank pattern

Lesson: [`lessons/B/b6_2_ranking_metrics.md`](../../../lessons/B/b6_2_ranking_metrics.md)

## Exercise 1 (by hand)

**Solution.** $\mathcal{R} = \{X, Y, Z\}$, binary relevance.

*System 1* (X, a, b, Y, c): hits at positions 1 and 4.
- P@5 $= 2/5 = 0.4$; R@5 $= 2/3 = 0.667$.
- AP@5 $= (1/1 + 2/4)/\min(5, 3) = 1.5/3 = 0.5$.
- DCG@5 $= 1/\log_2 2 + 1/\log_2 5 = 1 + 0.431 = 1.431$. IDCG@5 (three relevant items on top) $= 1 + 1/1.585 + 1/2 = 2.131$. NDCG@5 $= 0.671$.

*System 2* (a, X, Y, b, c): hits at positions 2 and 3.
- P@5 $= 0.4$; R@5 $= 0.667$ (the same two items are in the list).
- AP@5 $= (1/2 + 2/3)/3 = 1.167/3 = 0.389$.
- DCG@5 $= 1/1.585 + 1/2 = 0.631 + 0.5 = 1.131$; NDCG@5 $= 1.131/2.131 = 0.531$.

P@5 and R@5 tie: both lists contain the same two relevant items, and these metrics ignore order inside the top 5. AP and NDCG prefer
System 1, which puts a relevant item first; both reward position 1 heavily (AP through P@1 = 1, NDCG through the undiscounted first slot).

*Graded* (X = 3, Y = Z = 1): IDCG $= 3 + 1/1.585 + 1/2 = 4.131$. System 1: $3 + 1/2.322 = 3.431$, NDCG $0.831$. System 2: $3/1.585 + 1/2 = 1.893 + 0.5 = 2.393$,
NDCG $0.579$. System 1 is still better and the gap widens (0.25 instead of 0.14), because the item it puts first is now the most valuable one.

```text
== exercise 1 check ==
system 1: P@5 0.4000  R@5 0.6667  AP@5 0.5000  NDCG@5 0.6714
system 2: P@5 0.4000  R@5 0.6667  AP@5 0.3889  NDCG@5 0.5307
system 1, graded (X=3, Y=Z=1): NDCG@5 0.8305
system 2, graded (X=3, Y=Z=1): NDCG@5 0.5792
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| P@5 and R@5 correct for both (denominators $k$ and $\lvert\mathcal{R}\rvert$) | 2 |
| AP@5 correct for both, with the $\min(k, \lvert\mathcal{R}\rvert)$ denominator | 3 |
| NDCG@5: discounts $\log_2(j + 1)$; IDCG from all three relevant items; both values | 4 |
| Explains why P/R tie and AP/NDCG do not | 2 |
| Graded NDCG for both and the effect on the comparison | 3 |

**Likely mistakes:**

- Discounting position $j$ by $\log_2 j$ (position 1 then divides by 0) instead of $\log_2(j + 1)$.
- Building IDCG from the two relevant items that were *shown* (IDCG $= 1.631$), giving System 1 NDCG $0.877$: Z is relevant and must be in the ideal list.
- Dividing AP by the number of hits (2) instead of $\min(k, \lvert\mathcal{R}\rvert) = 3$, which gives $0.75$ and $0.583$: a list that finds only some of the
  relevant items should not be able to score 1. State the convention if you use another one.
- Using $2^{\mathrm{rel}} - 1$ for the graded part without saying so (it gives other numbers: gains 7, 1, 1).

## Exercise 2 (code)

**Solution:** [`b6_2_ranking_metrics_ex2.py`](b6_2_ranking_metrics_ex2.py) (copies the lab's pipeline, silences its printing, then adds the
three analyses).

```text
== (a) MRR@10, catalogue coverage and popularity of what is recommended ==
system                  MRR@10  coverage  mean watchers of recommended
popularity              0.1009    0.0547                         409.7
retrieval only (MF)     0.1586    0.3258                         255.1
retrieve + rerank       0.2454    0.4096                         224.6

== (b) NDCG@10 by user activity ==
light   users 313 (history 15-36): retrieval 0.1247  rerank 0.1713  gain +0.0466
medium  users 314 (history 37-108): retrieval 0.0911  rerank 0.1355  gain +0.0443
heavy   users 316 (history 109-732): retrieval 0.0731  rerank 0.1199  gain +0.0468

== (c) ablation on the validation users: drop sim_to_last5 ==
all 11 features        validation NDCG@10 0.1368
without sim_to_last5   validation NDCG@10 0.1252
```

**What a strong answer says:**

- MRR@10 rises from 0.159 to 0.245 (by 55%): the first relevant film appears higher, or appears at all, for many users. Do not read
  $1/0.245 \approx 4$ as "the average position of the first hit": MRR averages $1/\text{rank}$ and counts 0 when the top 10 has no hit. Popularity shows only
  5.5% of the catalogue to anyone (the same few films for everyone); the personalized systems cover 33% and 41%. Reranking both improves accuracy and
  recommends *less* popular films (225 against 255 watchers on average), so here accuracy and diversity did not trade off. That is a finding, not a
  rule: check it every time.
- By activity, the gain from reranking is about the same for every group (+0.044 to +0.047). Light users have the *highest* NDCG: with a short
  history, their next films are more predictable from the most popular and most recent titles; heavy users have already watched the obvious films.
  The interesting result is that the reranker does not only help the users with rich histories.
- Without `sim_to_last5`, validation NDCG@10 drops from 0.137 to 0.125, almost back to the retrieval order (0.124 in the lab): most of the reranker's
  value is the recency signal. The ablation was run on validation users, as model choices must be.
- A strong answer adds intervals: the groups have about 313 users each, so per-group differences of 0.003 are noise; a bootstrap per group would show it.

**Rubric (15 pts):**

| Criterion | Points |
|---|---|
| MRR@10 correct (first hit only, 0 if none) | 3 |
| Coverage computed over the full catalogue from the top-10 lists; mean popularity of recommended items | 3 |
| Activity groups from the history only (not the test block), with sizes reported | 3 |
| Ablation on validation users, not on the test block | 3 |
| Interpretation: accuracy vs coverage; who gains; what the ablation says, with a note on noise | 3 |

**Likely mistakes:**

- Defining activity groups with the test interactions (every user has 5, so the groups would be meaningless) or with the full data (leakage of the future).
- Computing coverage over the 100 candidates instead of the 10 shown.
- Running the ablation on the test block and then "choosing" the feature set: this is selection on the test set.
- Reading "light users have higher NDCG" as "the model is better for light users" without noting the task is easier for them.

## Exercise 3 (think)

**Strong answer:** Split the problem in two and measure each stage. *Labels:* build an eval set of real user questions with the chunk(s) that
answer each one, from three sources: questions where users clicked or quoted a source, a sample annotated by domain experts, and (as a supplement,
checked by people) LLM-generated questions for known chunks (I2.4). *Retriever:* measure **recall@20** (is a correct chunk among the 20?) and recall@5.
If recall@20 is low, a reranker cannot help; fix chunking, the embedding model or add hybrid keyword search first. If recall@20 is high but
recall@5 low, the right chunk is retrieved but ranked too low, which is exactly what a reranker fixes. *Reranker:* measure **recall@5, MRR@5 or
NDCG@5** on the same candidate lists before and after reranking, with a paired bootstrap over questions, and the end metric: answer correctness
judged on the same questions (I2.4). *Number of candidates:* plot retriever recall@$N$ for $N$ = 10, 20, 50, 100 and the reranker's top-5 quality and
latency at each; pick the smallest $N$ where recall flattens and latency fits the budget. *Trade-offs:* a cross-encoder runs the model once per (query,
chunk) pair, so cost and latency grow linearly with $N$ (for example 20 pairs at a few ms each on a GPU; much slower on CPU; VERIFY (as of 2026-11)
for the model you choose); batch the pairs, cache by query, and set a timeout that falls back to the retrieval order. The reranker is worth it if the
end-to-end answer accuracy improves by more than its noise and the p95 latency stays within the product's budget.
**Weak answer:** "Add the cross-encoder; rerankers are always better than embeddings, and check that answers look better."
**Why weak:** no separate retriever and reranker metrics, no labels, no way to tell whether the problem is recall or ordering, no latency or cost.

**Rubric (11 pts):**

| Criterion | Points |
|---|---|
| Separates retriever recall@N from top-k quality after reranking, and says what each diagnosis implies | 3 |
| A concrete plan for labels | 2 |
| Chooses $N$ from a recall@N curve and latency | 2 |
| Latency and cost of the cross-encoder, with mitigations | 2 |
| End-to-end metric and a significance check before shipping | 2 |

## Scoring

Total 40 pts. Pass ≥ 70% (28). Topic ids to revisit for each lost criterion: metric definitions (Ex 1, Ex 2a) → B6.2; precision and recall
(Ex 1) → B4.1; time blocks and leakage (Ex 2b) → B1.4; bootstrap (Ex 2) → A6.2; retrieval and RAG evals (Ex 3) → I2.3, I2.4.
