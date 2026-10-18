# Answer key: B5.2 Hierarchical clustering and DBSCAN: when clusters aren't round

Lesson: [`lessons/B/b5_2_hierarchical_dbscan.md`](../../../lessons/B/b5_2_hierarchical_dbscan.md)

## Exercise 1 (by hand)

Points: $A = 1$, $B = 2$, $C = 4.5$, $D = 9$, $E = 10.2$, $F = 16$. Neighbouring gaps: $AB = 1$, $BC = 2.5$, $CD = 4.5$, $DE = 1.2$, $EF = 5.8$.

**Solution (a).** *Single linkage.* $AB$ at 1, then $DE$ at 1.2. Next candidates: $\{A,B\}$–$C$ $= \min(3.5, 2.5) = 2.5$; $C$–$\{D,E\} = 4.5$; $\{D,E\}$–$F = 5.8$:
merge $ABC$ at 2.5. Then $\{A,B,C\}$–$\{D,E\}$ $=$ closest pair $CD = 4.5$, against $\{D,E\}$–$F = 5.8$: merge $ABCDE$ at 4.5. Last, $F$ joins at $EF = 5.8$.
Heights: 1, 1.2, 2.5, 4.5, 5.8.

*Complete linkage.* $AB$ at 1, $DE$ at 1.2. Then $\{A,B\}$–$C = \max(3.5, 2.5) = 3.5$; $C$–$\{D,E\} = \max(4.5, 5.7) = 5.7$; $\{D,E\}$–$F = \max(7, 5.8) = 7$: merge
$ABC$ at 3.5. Then $\{A,B,C\}$–$\{D,E\}$ $= AE = 9.2$; $\{D,E\}$–$F = 7$; $\{A,B,C\}$–$F = AF = 15$: merge $DEF$ at 7. Last, $ABC$–$DEF$ $= AF = 15$.
Heights: 1, 1.2, 3.5, 7, 15.

**(b)** Three clusters: both give $ABC \mid DE \mid F$. Two clusters: single gives $ABCDE \mid F$, complete gives $ABC \mid DEF$. Single linkage only looks at the
closest gap between groups: $C$ to $D$ is 4.5, smaller than $E$ to $F$ at 5.8, so $F$ is the most isolated point and is left alone. Complete linkage looks at the
diameter the merged group would have: $\{D, E, F\}$ spans 7, while joining $ABC$ with $DE$ would span 9.2, so it keeps the left group compact and attaches $F$ to $DE$.

**(c)** Means $1.5$ and $4.5$, sizes 2 and 1: $\Delta = \frac{2 \cdot 1}{3}(4.5 - 1.5)^2 = 6$, height $\sqrt{2 \times 6} = \sqrt{12} = 3.4641$ (and it is Ward's third merge, since
$C$–$\{D,E\}$ gives $\sqrt{2 \cdot \frac23 \cdot 5.1^2} = 5.89$).

**(d)** $\varepsilon = 1.5$, $m = 2$: $A$ and $B$ are core (each has the other within 1), $C$ has no neighbour within 1.5 (nearest 2.5) and is noise, $D$ and $E$ are core
(1.2 apart), $F$ is noise. Labels $[0, 0, -1, 1, 1, -1]$. With $m = 2$ every point that has any neighbour within $\varepsilon$ is core; DBSCAN then equals single linkage cut at height $\varepsilon$, with singletons as noise.

$\varepsilon = 2.6$, $m = 3$: neighbourhood sizes $A$ 2, $B$ 3 ($A$, $B$, $C$), $C$ 2, $D$ 2, $E$ 2, $F$ 1. Only $B$ is core; $A$ and $C$ are border points of its cluster; $D$, $E$ and
$F$ are noise. Labels $[0, 0, 0, -1, -1, -1]$. $D$ and $E$ are 1.2 apart, closer than $B$ and $C$, but a pair is not dense enough: each neighbourhood holds 2 points, fewer than
$m = 3$, and no core point is within reach. DBSCAN's notion of a cluster is "a region holding at least $m$ points within $\varepsilon$", not "points that are close".

```text
== exercise 1 check ==
single   merge heights: [1.0, 1.2, 2.5, 4.5, 5.8]
         cut into 3: ABC | DE | F
         cut into 2: ABCDE | F
complete merge heights: [1.0, 1.2, 3.5, 7.0, 15.0]
         cut into 3: ABC | DE | F
         cut into 2: ABC | DEF
ward     first three merge heights: [1.0, 1.2, 3.4641]   by hand sqrt(2 * 2/3 * 3^2) = 3.4641
DBSCAN eps=1.5 min_samples=2: labels [0, 0, -1, 1, 1, -1]  ['core', 'core', 'noise', 'core', 'core', 'noise']
DBSCAN eps=2.6 min_samples=3: labels [0, 0, 0, -1, -1, -1]  ['border', 'core', 'border', 'noise', 'noise', 'noise']
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Single linkage: all five merges and heights | 3 |
| (a) Complete linkage: all five merges and heights (the $DEF$ merge at 7 before the last) | 3 |
| (b) Both cuts, and the disagreement explained by closest gap vs diameter | 3 |
| (c) Ward height 3.4641 from $\Delta = \frac{n_A n_B}{n_A + n_B}\lVert\Delta\mu\rVert^2$ and $\sqrt{2\Delta}$ | 2 |
| (d) Both DBSCAN runs labelled correctly, with core/border/noise, and why $D$, $E$ are noise | 3 |

**Likely mistakes:**

- Using single-point distances after a merge (for example $AB$–$C$ as the distance from $C$ to "the cluster's first point") instead of the min or max over all pairs.
- In complete linkage, merging $ABC$ with $DE$ before $DE$ with $F$ by looking at the closest pair: that is single linkage's rule.
- Forgetting the factor 2 and the square root in the Ward height ($\Delta = 6$ is the rise in SSE; the height scipy reports is 3.4641), or using the sum instead of the mean of $\{A, B\}$.
- Not counting the point itself in its neighbourhood, which makes every count one smaller (scikit-learn's `min_samples` includes the point).
- Calling $A$ or $C$ in the second run core because they have a neighbour: border points are in a core's neighbourhood but lack $m$ points in their own.

## Exercise 2 (code)

**Solution:** [`b5_2_hierarchical_dbscan_ex2.py`](b5_2_hierarchical_dbscan_ex2.py).

```text
== (a) one eps for three densities ==
 eps clusters  noise     ARI
 0.1        5  0.745  0.2628
 0.3        9  0.415  0.6219
 0.5       10  0.170  0.7494
 0.7        6  0.050  0.8731
 0.9        3  0.027  0.5610
 1.1        2  0.013  0.5661
 1.3        2  0.007  0.5683
 1.5        2  0.005  0.5689
 2.0        1  0.003  0.0000
best ARI over the grid 0.8731 at eps = 0.7
  true blob 0 (std 0.3): noise share 0.000
  true blob 1 (std 1.0): noise share 0.030
  true blob 2 (std 2.0): noise share 0.120

== (b) HDBSCAN ==
min_cluster_size=10: 3 clusters, noise share 0.112, ARI 0.8466
min_cluster_size=20: 3 clusters, noise share 0.150, ARI 0.8158
min_cluster_size=50: 3 clusters, noise share 0.227, ARI 0.8175

== (c) units: stretched moons ==
raw moons              eps (99th pct of 5-distance) 0.1360: 2 clusters, noise 0.002, ARI 0.9960
feature 2 x10          eps (99th pct of 5-distance) 0.7440: 1 clusters, noise 0.004, ARI 0.0000
x10 then standardized  eps (99th pct of 5-distance) 0.2249: 2 clusters, noise 0.000, ARI 1.0000
```

**What a strong answer says:**

- (a) The ARI peaks at 0.87 with $\varepsilon = 0.7$, which finds 6 clusters: the loose blob (std 2) keeps a main cluster but sheds a few small fragments, and 12% of
  it is noise, while the tight blob loses nothing. Smaller $\varepsilon$ shatters the loose blob further. From 0.9 up, the loose blob's outskirts reach the medium blob at
  $(6, 0)$ and the two merge into one cluster, so the ARI falls to about 0.56. **The "best $\varepsilon$" was chosen with the true labels**, which real data does not have. In
  practice you would read $\varepsilon$ from the k-distance curve, which has no single bend here because the three blobs each have their own scale.
- (b) HDBSCAN finds 3 clusters at every `min_cluster_size` tried, with no $\varepsilon$ at all, and an ARI of 0.82–0.85: slightly below the label-tuned DBSCAN, far above
  any $\varepsilon$ you could pick without labels. It labels 11–23% of points as noise, mostly the loose blob's outskirts; a strong answer says whether that is acceptable for
  the use case, rather than calling the ARI a win or a loss.
- (c) Multiplying feature 2 by 10 inflates every k-distance (the vertical noise is now 10 times larger), so the k-distance rule picks $\varepsilon = 0.74$. But the horizontal
  gaps between the moons' tips and the other moon were not stretched, and they are smaller than 0.74, so DBSCAN bridges the moons into one cluster (ARI 0). The rule
  adapted $\varepsilon$ to the scale, but not to the *shape* the rescaling changed. Standardizing restores comparable units and a perfect ARI.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| (a) Sweep with clusters, noise share and ARI; per-blob noise at the best $\varepsilon$ | 4 |
| (a) States that the best $\varepsilon$ needed the labels, and what you would do instead | 3 |
| (b) HDBSCAN run at three settings, compared honestly (number of clusters, noise, ARI) | 3 |
| (c) Stretch, k-distance rule and standardization shown; why the stretched run merges the moons | 4 |
| Code quality: seeds fixed, no loops where broadcasting works, results printed with labels | 2 |

**Likely mistakes:**

- Reporting the best ARI over the sweep as "DBSCAN's score" without saying it used the answer.
- Treating noise points as wrongly clustered and dropping them before computing the ARI, which makes every high-noise setting look good.
- Standardizing with statistics computed separately per cluster or on part of the data; for unsupervised work on a fixed dataset, fit the scaler on all rows used for clustering, but in a pipeline that will score new rows, fit it once and reuse it.
- Forgetting that `k_distance` must count the point itself, so the 5-distance is column index 4 of the sorted distance matrix (column 0 is the zero self-distance).

## Exercise 3 (think)

**Strong answer:** Ward first. Hierarchical clustering needs the pairwise distances: $400{,}000^2/2 \times 8$ bytes $\approx 640$ GB. It will not run; you would have to
sample (say 20,000 pickups) and then assign the rest to the nearest cluster mean, which makes it k-means in all but name. Ward with $k = 50$ also forces every pickup
into one of 50 round, similar-sized zones, including the scattered suburban and rural ones, and splits elongated hotspots (an airport terminal road, a stadium
perimeter, a nightlife street) or merges two nearby ones to keep sizes even. It answers "divide the city into 50 zones", not "where are the hotspots".

DBSCAN matches the question: a hotspot is a place with many pickups close together, and most pickups should be noise. Practical setup: (1) distances on a sphere:
degrees of longitude shrink with latitude (about $\cos(\text{lat})$), so either project to metres (a local UTM zone) or use `metric="haversine"` on radians with a ball
tree and $\varepsilon$ in radians ($\varepsilon = 150 / 6{,}371{,}000$ for 150 m); (2) $\varepsilon$ from the business (a walk of about 100–200 m that a driver can cover) and checked
on the k-distance curve; `min_samples` from what a hotspot means ("at least 30 pickups a month", or scale to hourly rates); (3) density varies a lot between the centre
and the suburbs, so one $\varepsilon$ will either merge the whole centre into one blob or miss suburban hotspots: try HDBSCAN, or cluster the centre and the suburbs
separately; (4) demand moves through the day and week, so cluster per time slot (weekday mornings, Friday nights) rather than the whole month at once; (5) check
stability: rerun on two different weeks and match the hotspots. For the operations team: a map of hotspot outlines (or centroids with radius) per time slot, each with
its expected pickups per hour, plus the share of all pickups that fall in hotspots. Then test the policy, not the clustering: an A/B test (A6.3b) of positioning
drivers at hotspots against the current practice, measured by rider wait time.
**Weak answer:** "DBSCAN is better because it finds any shape and handles noise; set eps with the elbow method and plot the clusters."
**Why weak:** right conclusion, no reasons that come from this data: it ignores the memory wall for hierarchical clustering, latitude/longitude distances, varying
density across the city, time of day, and how the result would be used and validated.

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| Hierarchical memory/time wall quantified (or sampling and its consequences) | 2 |
| What Ward/$k = 50$ produces versus what "hotspot" means (forced assignment, round equal zones) | 2 |
| Distances on a sphere handled (projection or haversine), $\varepsilon$ in real units | 2 |
| Parameters from the business and the k-distance curve; varying density addressed (HDBSCAN or splitting) | 3 |
| Time slots, stability check, and an output plus validation the operations team can use | 3 |

## Scoring

Total 42 pts. Pass ≥ 70% (30). Topic ids to revisit for each lost criterion: linkage merges and the dendrogram (Ex 1a–c, Ex 3) → B5.2; DBSCAN core/border/noise and
$\varepsilon$ (Ex 1d, Ex 2) → B5.2; scaling and units (Ex 2c) → B1.3; distances and their concentration (Ex 2c, Ex 3) → B2.3; the k-means objective behind Ward (Ex 1c, Ex 3) → B5.1;
testing the policy (Ex 3) → A6.3b.
