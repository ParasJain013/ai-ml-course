# Answer key: B5.5 t-SNE and UMAP: what they show and how they mislead

Lesson: [`lessons/B/b5_5_tsne_umap.md`](../../../lessons/B/b5_5_tsne_umap.md)

## Exercise 1 (by hand)

**Solution (a).** Point at 0, $\sigma = 1$. Squared distances to the points at 1 and 3: $1$ and $9$. Weights
$e^{-d^2/2}$: $e^{-0.5} = 0.6065$ and $e^{-4.5} = 0.0111$; sum $0.6176$. So $p_{\cdot\mid 0} = (0.9820,\ 0.0180)$.
Entropy $H = -(0.982\log 0.982 + 0.018\log 0.018) = 0.0178 + 0.0723 = 0.0901$ nats; perplexity $e^{0.0901} = 1.094$:
effectively one neighbour.

**(b)** $\sigma = 2$: weights $e^{-1/8} = 0.8825$ and $e^{-9/8} = 0.3247$; sum $1.2072$; $p = (0.7311,\ 0.2689)$;
$H = 0.2292 + 0.3530 = 0.5822$ nats; perplexity $1.790$. Perplexity grows with $\sigma$, and even $\sigma = 2$ gives
only 1.79, so a target of 1.9 needs $\sigma$ **above 2**. (With only two other points the perplexity can never exceed
2, reached as $\sigma \to \infty$.)

**(c)** Drawn at $y = 0, 1, 2$: squared distances $1$ (pairs 1–2 and 2–3) and $4$ (pair 1–3); t-kernels $0.5, 0.5, 0.2$.
Ordered pairs count each twice: denominator $2(0.5 + 0.5 + 0.2) = 2.4$. So $q_{12} = q_{21} = q_{23} = q_{32} = 0.5/2.4 = 0.2083$ and
$q_{13} = q_{31} = 0.2/2.4 = 0.0833$; the six values sum to 1.

**(d)** $0.3\log(0.3/0.03) = 0.3\log 10 = +0.6908$; $0.03\log(0.03/0.3) = -0.0691$. Separating a real neighbour pair
costs ten times more than drawing a distant pair close together, so t-SNE keeps neighbourhoods and lets the
positions of non-neighbours (and so the gaps and arrangement of clusters) be almost anything.

```text
== exercise 1 check ==
(a/b) sigma=1.0: weights [0.6065, 0.0111], p [0.982, 0.018], H 0.0901 nats, perplexity 1.0943
(a/b) sigma=2.0: weights [0.8825, 0.3247], p [0.7311, 0.2689], H 0.5822 nats, perplexity 1.7900
(c) sum of 1/(1+d^2) over ordered pairs 2.4000; q_01 0.2083, q_02 0.0833, q_12 0.2083
(d) p=0.3, q=0.03: p log(p/q) = +0.6908
(d) p=0.03, q=0.3: p log(p/q) = -0.0691
```

(The check script labels pairs by 0-based index: `q_01` is $q_{12}$, `q_02` is $q_{13}$.)

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) Weights, normalized $p$, entropy in nats and perplexity $\approx 1.09$ | 3 |
| (b) $\sigma = 2$ values and the conclusion "above 2", with the monotonicity reason | 3 |
| (c) Denominator over ordered pairs (2.4) and all $q_{ij}$ | 3 |
| (d) Both KL terms and the one-sentence consequence (local kept, global free) | 3 |

**Likely mistakes:**

- Using distances instead of squared distances in the Gaussian ($e^{-3/2}$ instead of $e^{-9/2}$).
- Taking perplexity as $2^{H}$ with $H$ in nats (gives 1.06 instead of 1.09). Either base works only if it matches the
  log used for $H$.
- Summing the $q$ denominator over unordered pairs only (1.2), which makes the $q_{ij}$ sum to 2.
- Answering "between 1 and 2" in (b) by interpolating the target between the two $\sigma$s instead of comparing it with
  the perplexities.

## Exercise 2 (code)

**Solution:** [`b5_5_tsne_umap_ex2.py`](b5_5_tsne_umap_ex2.py).

```text
== (a) k-means (k=10) on digits in each space ==
space                  ARI vs digit  silhouette (own space)  silhouette (64-D)
raw 64-D                     0.6657                  0.1825             0.1825
PCA 2                        0.3927                  0.3937             0.0819
PCA 10                       0.6517                  0.2639             0.1813
t-SNE 2 (perp 30)            0.8826                  0.6391             0.1634
UMAP 2 (nn 15, md 0)         0.8213                  0.8216             0.1603

== (b) t-SNE seed stability ==
init='pca'     ARI between seeds [1.0, 1.0, 1.0]   ARI vs digit per seed [0.8826, 0.8826, 0.8826]
init='random'  ARI between seeds [0.9936, 0.9976, 0.9935]   ARI vs digit per seed [0.8804, 0.8793, 0.8781]

== (c) the same pipeline on pure Gaussian noise (1797, 64) ==
space                   silhouette (own space)  silhouette (64-D)
raw 64-D                                0.0124             0.0124
PCA 2                                   0.3212             0.0018
PCA 10                                  0.0699             0.0106
t-SNE 2 (perp 30)                       0.3254            -0.0111
UMAP 2 (nn 15, md 0)                    0.3411            -0.0027
```

**What a strong answer says:**

- **The surprise, reported honestly:** on digits, k-means on the t-SNE picture recovers the digit labels *better*
  (ARI 0.88) than k-means on the raw pixels (0.67) or on PCA-10 (0.65). Digit classes are real, well separated, and
  not round in 64-D; k-means assumes round clusters (B5.1), and t-SNE unfolds each class into a compact island. So
  "never cluster on t-SNE" is too strong; "never *trust* a clustering found on t-SNE without checks" is right.
- **The silhouette in the picture is not evidence:** it is 0.64 (t-SNE) and 0.82 (UMAP), against 0.16–0.18 when the
  *same* clusters are scored in 64-D. On noise, every 2-D picture gives 0.32–0.34 while the 64-D silhouette of those
  clusters is about 0. A cluster score computed in the picture would look "moderate structure" even for noise.
- **Seed stability with `init="pca"` is a fake test:** the PCA initialization and the Barnes–Hut optimizer are
  deterministic, so the seed does nothing (ARI 1.0 between runs). With `init="random"` the clusterings still agree
  closely (ARI ≈ 0.99): on digits the islands are stable. A stability check must change something that matters
  (seed with random init, perplexity, a bootstrap resample of the rows).
- **What goes in a report:** ARI against labels when labels exist (here 0.88 for t-SNE + k-means, with the caveat that
  it is one dataset), the silhouette in the *original* space, and the stability across random inits and settings. Not:
  any silhouette computed in the 2-D picture. PCA-2 is bad for clustering here (ARI 0.39): two components keep too
  little of the digits' structure.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Five spaces built correctly (PCA fitted on the data, t-SNE and UMAP with the stated settings, `random_state` set) | 3 |
| ARI vs labels and silhouettes in own space and in 64-D for each | 3 |
| Seed comparison, including the observation that `init="pca"` makes the seed irrelevant | 3 |
| Noise run with the silhouettes in both spaces | 3 |
| Interpretation: t-SNE clustering can work on clean data, picture silhouettes are inflated, what to report | 4 |

**Likely mistakes:**

- Reporting only the silhouette in the picture and concluding the t-SNE clusters are "excellent".
- Concluding from the identical `init="pca"` runs that t-SNE is "perfectly stable".
- Comparing ARI between runs without fixing the number of clusters, or computing ARI on cluster ids as if they were
  labels (`accuracy` of cluster id against digit), which depends on the arbitrary numbering.
- Standardizing the digits pixels without noticing that edge pixels that are almost always 0 then get blown up.

## Exercise 3 (think)

**Strong answer:** Seven islands is a hypothesis, not a finding. (1) *Seven segments:* the number of islands
depends on `n_neighbors`, `min_dist` and the seed, and UMAP can make islands out of noise; I would cluster in the
original embedding space (or a PCA reduction of it) with k-means or a Gaussian mixture, choose $k$ with silhouette and
stability across bootstrap resamples (B5.1, B5.3), and check that the segments also differ on business features the
team cares about (spend, tenure, churn). (2) *Budgets by island area:* area on a UMAP plot does not measure segment
size or spread, because every point's neighbourhood is normalized (in the lesson's lab a cluster ten times wider was
drawn the same size). Count the customers in each segment and measure its value directly. (3) *Migration to the
nearest island:* distances between islands carry almost no information (a 10× gap was drawn as 1× by UMAP), so
"nearest neighbouring island" is an artefact of the layout; compute centroid distances in the original space, and,
more importantly, ask whether moving customers between segments is causal at all; that needs an experiment (A6.3b),
not a picture. I would still use the UMAP plot to *display* the validated segments, coloured by segment, and show two
or three settings side by side so nobody reads the layout as a map. With 2 million points I would also subsample for
the plot and say so.
**Weak answer:** "UMAP preserves global structure better than t-SNE, so the islands and their distances are reliable;
seven segments is a good choice and areas are a reasonable proxy for size."
**Why weak:** it repeats a contested claim, ignores that both methods normalize density and distort gaps, and proposes
spending money on properties the method does not preserve.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Number of islands depends on settings/seed; validate clusters in the original space with stability checks | 3 |
| Island area is not segment size or spread; count and measure directly | 2 |
| Distances between islands are not meaningful; nearest island is an artefact | 2 |
| Causal question behind "migration" needs an experiment; proposes how the UMAP plot should still be used | 3 |

## Scoring

Total 38 pts. Pass ≥ 70% (27). Topic ids to revisit for each lost criterion: affinities, perplexity and the KL
asymmetry (Ex 1) → B5.5, A6.4; clustering and silhouette (Ex 2) → B5.1, B5.5; PCA as a baseline (Ex 2) → B5.4; what
the picture preserves and how to validate segments (Ex 3) → B5.5, B5.1; experiments for causal claims (Ex 3) → A6.3b.
