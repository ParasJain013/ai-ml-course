# Answer key: B5.1 k-means from scratch, and choosing k (elbow, silhouette)

Lesson: [`lessons/B/b5_1_k_means.md`](../../../lessons/B/b5_1_k_means.md)

## Exercise 1 (by hand)

**Solution (a).** Points $0, 2, 3, 9, 11$; centroids $c_1 = 0$, $c_2 = 2$.

- *Iteration 1.* Squared distances to $(c_1, c_2)$: 0: $(0, 4)$ → 1; 2: $(4, 0)$ → 2; 3: $(9, 1)$ → 2; 9: $(81, 49)$ → 2; 11: $(121, 81)$ → 2.
  Update: $c_1 = 0$, $c_2 = (2 + 3 + 9 + 11)/4 = 6.25$. Inertia $= 0 + 4.25^2 + 3.25^2 + 2.75^2 + 4.75^2 = 18.0625 + 10.5625 + 7.5625 + 22.5625 = 58.75$.
- *Iteration 2.* 2: $(4, 18.06)$ → 1; 3: $(9, 10.56)$ → 1; 9: $(81, 7.56)$ → 2; 11: → 2. Update: $c_1 = 5/3 = 1.667$, $c_2 = 10$.
  Inertia $= 1.667^2 + 0.333^2 + 1.333^2 + 1^2 + 1^2 = 2.778 + 0.111 + 1.778 + 2 = 6.667$.
- *Iteration 3.* Nothing changes (3 is $1.333$ from $c_1$ and 7 from $c_2$): converged, $\{0, 2, 3\}, \{9, 11\}$, $J = 6.667$.

**(b)** Cuts after 0, 2, 3, 9: inertias $58.75$, $36.667$ ($\{0,2\}$: 2; $\{3,9,11\}$: mean $7.667$, $21.78 + 1.78 + 11.11 = 34.67$), $6.667$ and $45$
($\{0,2,3,9\}$: mean 3.5, $12.25 + 2.25 + 0.25 + 30.25 = 45$; $\{11\}$: 0). The cut after 3 is the global minimum, and (a) reached it despite the poor start.

**(c)** $D^2 = 0, 4, 9, 81, 121$, sum 215. Probabilities: 0 → 0; 2 → $4/215 = 0.0186$; 3 → $0.0419$; 9 → $0.3767$; 11 → $0.5628$.
About 94% of the time the second centroid lands in the right-hand group.

**(d)** Point 3 is in $\{0, 2, 3\}$: $a = (3 + 1)/2 = 2$; $b = (6 + 8)/2 = 7$; $s = (7 - 2)/7 = 0.714$.

**(e)** Contingency table (rows: first labeling, columns: second): $\begin{pmatrix}2 & 1\\ 0 & 2\end{pmatrix}$. Index $= \binom22 + \binom12 + \binom02 + \binom22 = 1 + 0 + 0 + 1 = 2$.
Row sums 3, 2: $\binom32 + \binom22 = 3 + 1 = 4$. Column sums 2, 3: $1 + 3 = 4$. $\binom52 = 10$. Expected $= 4 \times 4/10 = 1.6$; max $= (4 + 4)/2 = 4$.
$\text{ARI} = (2 - 1.6)/(4 - 1.6) = 0.4/2.4 = 0.1667$. Moving one point out of five drops the ARI from 1 to 0.17.

```text
== exercise 1 check ==
iteration 1: labels [0, 1, 1, 1, 1] -> centroids [0.0, 6.25], inertia 58.75
iteration 2: labels [0, 0, 0, 1, 1] -> centroids [1.6666666666666667, 10.0], inertia 6.66667
iteration 3: labels [0, 0, 0, 1, 1] -> centroids [1.6666666666666667, 10.0], inertia 6.66667
(b) split after 0: inertia 58.7500
(b) split after 2: inertia 36.6667
(b) split after 3: inertia 6.6667
(b) split after 9: inertia 45.0000
(c) k-means++ after picking 0: D^2 = [0.0, 4.0, 9.0, 81.0, 121.0], P(pick 11) = 0.5628
(d) silhouette of 3: a = 2, b = 7, s = 0.7143
    ARI([0,0,0,1,1], [0,0,1,1,1]) = 0.1667
```

**Rubric (15 pts):**

| Criterion | Points |
|---|---|
| (a) Every assignment, centroid and inertia correct, and convergence recognized | 4 |
| (b) Four cut inertias; conclusion that (a) is the global minimum | 3 |
| (c) Probabilities from $D^2$ (squared, normalized; the chosen point has 0) | 3 |
| (d) $a$ excludes the point itself, $b$ uses the other cluster, $s = 0.714$ | 2 |
| (e) Contingency table, the three pair counts, ARI $= 0.1667$ | 3 |

**Likely mistakes:**

- Using plain distances instead of squared ones in k-means++ (probabilities $2/25, 3/25, 9/25, 11/25$): the $D^2$ weighting is the whole point.
- Including the point itself (distance 0) in $a$ and dividing by 3 instead of 2, giving $a = 1.333$.
- Inertia computed with distances, not squared distances.
- In (e), comparing labels position by position ("4 of 5 agree, so 0.8"): ARI counts pairs and corrects for chance.

## Exercise 2 (code)

**Solution:** [`b5_1_k_means_ex2.py`](b5_1_k_means_ex2.py) (from-scratch k-means copied from the lab; scikit-learn only for the scores and the data).

```text
== wine: raw versus standardized ==
X (178, 13), cultivars [59, 71, 48]
largest feature std: proline 314.0; smallest: nonflavanoid_phenols 0.124
raw           k=3: inertia    2370689.7  silhouette 0.5711  ARI vs cultivar 0.3711  sizes [47, 62, 69]
standardized  k=3: inertia       1277.9  silhouette 0.2849  ARI vs cultivar 0.8975  sizes [51, 62, 65]
raw centroids, proline column: [458.2, 728.3, 1195.1]

== choosing k on standardized wine ==
 k  inertia silhouette boot ARI mean boot ARI min
 2   1659.0     0.2683        0.6415       0.2979
 3   1277.9     0.2849        0.9307       0.8039
 4   1175.4     0.2614        0.7592       0.5693
 5   1110.6     0.2063        0.7071       0.5177
 6   1051.2     0.1861        0.5667       0.4482
 7    984.5     0.1975        0.6578       0.4710
 8    945.3     0.1318        0.4982       0.3710
```

**What a strong answer says:**

- On raw features, proline has a standard deviation of 314 against 0.12 for the smallest feature, so squared distances are almost entirely proline
  differences. k-means cuts proline into three bands (centroids 458, 728, 1195) and ignores the other 12 features: ARI with the cultivars is only 0.37.
- The raw silhouette (0.57) is *higher* than the standardized one (0.28). That does not make the raw clustering better: the silhouette is computed in the
  space you clustered in, and cutting one stretched feature into bands gives tidy-looking clusters in that space. Silhouettes are comparable only
  between clusterings of the same features with the same scaling. ARI against the known cultivars (available here, not in real P3 work) shows the
  standardized clustering recovers them far better (0.90).
- On standardized data, $k = 3$ has the highest silhouette (0.285) and is by far the most stable (bootstrap ARI mean 0.93, minimum 0.80; every other $k$ has a
  mean below 0.76). The elbow is soft: the drop is 381 from 2 to 3, then 103, 65, 59: a mild bend at 3. Report $k = 3$ on the strength of silhouette and
  stability together, and say the elbow alone would not decide it.
- Standardizing on all 178 rows is fine here: there is no held-out set in an exploratory clustering. In a pipeline that later assigns new rows, the scaler is
  fitted once and reused (B1.3).

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Uses own k-means (k-means++ seeding, `n_init=10`); runs and is reproducible with `SEED = 0` | 3 |
| Raw vs standardized: inertia, silhouette and ARI reported for both | 3 |
| Explains the raw result with the proline scale and the centroids | 3 |
| Notes that silhouettes are not comparable across scalings | 2 |
| Table for $k = 2..8$ with bootstrap ARI comparing on the original rows | 3 |
| Chooses $k$ with stated evidence from more than one signal | 2 |

**Likely mistakes:**

- Concluding the raw clustering is better "because its silhouette is higher".
- Comparing a bootstrap refit with the full-data labels only on the resampled rows (with duplicates), or comparing labels with `==` instead of ARI.
- Forgetting that inertia values on raw and standardized data are on different scales and cannot be compared at all.
- Using the cultivar labels to choose $k$ and then claiming the clustering "found" them; in an unsupervised project no such labels exist.

## Exercise 3 (think)

**Strong answer:** Not as requested, at least not without checking. A silhouette of 0.18 at $k = 5$ and a smooth elbow mean the data has no strong
five-group structure; the five segments would be slices of a continuum, and the boundaries would be partly arbitrary. What I would check: (1) stability:
fit on several samples of, say, 50,000 customers with different seeds and bootstrap resamples and compute ARI; if five segments come back with ARI
near 1, they are at least reproducible, even if not "natural"; (2) whether the $k = 2$ split is trivial (for example "active vs inactive", driven by
one or two features), which would explain its higher silhouette without making it useful; (3) the feature profile of each segment (mean of each
standardized feature, size), checking the features were scaled sensibly and that outliers do not own a segment; (4) a 2-D view (PCA, B5.4) to show the
shape of the data. What I would show: five segments with sizes, profiles and plain-English names, together with the stability numbers and a plain
statement that customers near the boundaries could equally fall in the neighbouring segment. What they are for: targeting and messaging tests, where a
mis-assigned customer costs little, with an A/B test to see whether segment-specific campaigns actually beat a single one (A6.3b). What they are not:
evidence that five distinct customer types exist, or a basis for decisions with real consequences for individuals (pricing, credit) without
further validation. If $k = 5$ is unstable, propose fewer segments, or a continuous score (such as a spend or engagement percentile) instead.
**Weak answer:** "The silhouette says 2, so I'd tell marketing that the data only has two segments." Or: "Run k-means with $k = 5$ and send the labels."
**Why weak:** the first treats one noisy statistic as the truth and ignores what the business needs; the second ships unvalidated, possibly
unstable segments without telling anyone how soft they are.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Reads 0.18 and a smooth elbow as weak structure, not as "wrong k" | 2 |
| Stability check (seeds and resamples, ARI) on a sample of the 2 million | 3 |
| Profiles, sizes, outliers and scaling checked before naming segments | 2 |
| Says what segments can and cannot be used for, and proposes validating their value (A/B test) | 3 |

## Scoring

Total 41 pts. Pass ≥ 70% (29). Topic ids to revisit for each lost criterion: Lloyd steps, k-means++, silhouette, ARI (Ex 1, Ex 2) → B5.1;
feature scaling (Ex 2) → B1.3; distances (Ex 1, Ex 2) → B2.3; bootstrap (Ex 2, Ex 3) → A6.2; experiments on segments (Ex 3) → A6.3b.
