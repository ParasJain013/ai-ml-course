# Answer key: A1.5 Vectors

Lesson: [`lessons/A/a1_5_vectors.md`](../../../lessons/A/a1_5_vectors.md)

## Exercise 1 (by hand)

**Solution.** $\mathbf{a} = (3, 4, 0)$, $\mathbf{b} = (4, 3, 0)$, $\mathbf{c} = (0, 0, 2)$.

- Dot products: $\mathbf{a}\cdot\mathbf{b} = 12 + 12 + 0 = 24$; $\mathbf{a}\cdot\mathbf{c} = 0 + 0 + 0 = 0$.
- L2 norms: $\lVert\mathbf{a}\rVert = \sqrt{9 + 16} = 5$, $\lVert\mathbf{b}\rVert = 5$,
  $\lVert\mathbf{c}\rVert = \sqrt{4} = 2$.
- L1 norms: $3 + 4 = 7$, $4 + 3 = 7$, $2$.
- Cosines: $\cos(\mathbf{a}, \mathbf{b}) = 24/(5 \cdot 5) = 0.96$; $\cos(\mathbf{a}, \mathbf{c}) = 0/(5 \cdot 2) = 0$
  (at right angles: they share no non-zero feature).
- Distances: $\lVert\mathbf{a} - \mathbf{b}\rVert = \lVert(-1, 1, 0)\rVert = \sqrt2 \approx 1.4142$;
  $\lVert\mathbf{a} - \mathbf{c}\rVert = \lVert(3, 4, -2)\rVert = \sqrt{9 + 16 + 4} = \sqrt{29} \approx 5.3852$.

Scaling $\mathbf{b}$ by 10, $10\mathbf{b} = (40, 30, 0)$:

| Quantity | $\mathbf{b}$ | $10\mathbf{b}$ | Changes? |
|---|---|---|---|
| $\mathbf{a}\cdot\mathbf{b}$ | 24 | 240 | yes, ×10 |
| L2 norm | 5 | 50 | yes, ×10 |
| L1 norm | 7 | 70 | yes, ×10 |
| cosine with $\mathbf{a}$ | 0.96 | 0.96 | **no** (direction unchanged) |
| distance to $\mathbf{a}$ | 1.4142 | $\sqrt{37^2 + 26^2} = \sqrt{2045} \approx 45.2217$ | yes, a lot |

Unit-vector identity: $\hat{\mathbf{a}} = (0.6, 0.8, 0)$, $\hat{\mathbf{b}} = (0.8, 0.6, 0)$,
$\hat{\mathbf{a}} - \hat{\mathbf{b}} = (-0.2, 0.2, 0)$, squared length $0.04 + 0.04 = 0.08$, and
$2 - 2 \times 0.96 = 0.08$. ✓

```text
== exercise 1 check ==
  b: a.b=24  |b|_2=5  |b|_1=7  cos=0.9600  |a-b|=1.4142
10b: a.10b=240  |10b|_2=50  |10b|_1=70  cos=0.9600  |a-10b|=45.2217
a.c=0  |c|_2=2  cos(a,c)=0.0000  |a-c|=5.3852
|a_hat - b_hat|^2 = 0.08  2 - 2cos = 0.08
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| Both dot products correct | 2 |
| All six norms correct | 2 |
| Both cosines correct, with $\cos(\mathbf{a}, \mathbf{c}) = 0$ interpreted | 2 |
| Both distances correct | 2 |
| Scaling: identifies that cosine is unchanged and everything else changes | 2 |
| Unit-vector identity verified with numbers | 2 |

**Likely mistakes:**

- Forgetting the square root in the L2 norm ($\lVert\mathbf{a}\rVert = 25$).
- Taking the L1 norm without absolute values (fine here, wrong for $\mathbf{a} - \mathbf{c}$).
- Saying the cosine changes when $\mathbf{b}$ is scaled because "the dot product changed": both the
  numerator and the denominator grow by 10.
- Computing the distance with L1 instead of L2.

## Exercise 2 (code)

**Solution:** [`a1_5_vectors_ex2.py`](a1_5_vectors_ex2.py). The search function:

```python
def search(query: FloatArray, M: FloatArray, k: int) -> list[tuple[int, float]]:
    """Top-k rows of M by cosine similarity to query (d,), best first. M (m, d)."""
    q_hat = query / np.linalg.norm(query)  # (d,)
    M_hat = M / np.linalg.norm(M, axis=1, keepdims=True)  # (m, d)
    scores = M_hat @ q_hat  # (m,)
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in order]
```

```text
== exercise 2: search ==
query 'cheap outdoor cooking': [('camping stove', 0.986), ('garden hose', 0.8461), ('budget tent', 0.8411)]
matches NearestNeighbors(metric='cosine'): OK

== exercise 2: dot product vs cosine ==
raw dot top-1: 'luxury tent' dot=0.9000 |doc|=1.2728
cosine  top-1: 'garden hose' cos=0.9942 |doc|=0.9220
luxury tent: dot=0.9000 cos=0.7809;  garden hose: dot=0.8300 cos=0.9942
```

For the query "cheap outdoor", `[0.1, 0.9, 0, 0, 0]`, the raw dot product prefers "luxury tent"
because its vector is longer (the large `price` feature adds length), not because it matches better;
cosine prefers "garden hose", whose *direction* is almost identical to the query's (cheap and
outdoor). Any working example where the dot-product winner has a larger norm and a smaller cosine
earns full marks.

**Rubric (15 pts):**

| Criterion | Points |
|---|---|
| At least 12 vectors with named features, shape `(m, d)` stated | 2 |
| `search` correct, no Python loop over rows, sorted best first | 4 |
| Checked against `NearestNeighbors(metric="cosine")` with asserts (indices and scores, `1 - distance`) | 3 |
| A concrete query where dot-product and cosine top-1 differ, with numbers | 3 |
| Correct explanation (vector length / norm) | 2 |
| Type hints, docstring with shapes | 1 |

**Likely mistakes:**

- Comparing their scores to `NearestNeighbors` distances directly, forgetting it returns $1 - \cos$.
- `np.argsort(scores)[:k]`: ascending order, returning the *least* similar documents.
- Normalizing `M` with `np.linalg.norm(M)` (the whole matrix) instead of per row.
- A "difference" example that is really a tie, broken by sort order.

## Exercise 3 (think)

**Strong answer:** Embeddings from the old and new model live in different vector spaces. Even when
both are, say, 768-dimensional, feature $j$ means unrelated things in each, so the cosine between a
new-model query vector and an old-model product vector is close to meaningless. Queries are embedded
with the new model, so products still on old vectors are ranked by noise: some never surface, others
surface for unrelated queries. Detection: store `embedding_model` and `embedding_version` on every
row and alert when a search runs across mixed versions; monitor offline retrieval metrics on a fixed
labelled query set (recall@k, I2.4) after every model change; watch online signals like click-through
on search results by product age or by version. Migration: treat it like a schema migration. Add a
new vector column (or table) for the new model, backfill all 2 million products in batches (with the
cost and time estimated first), build the new index, evaluate the new search on the labelled query
set against the old one, then switch reads with a flag, keeping the old column for rollback until
the new one is proven. Never mix versions in one index.

**Weak answer:** "The new model is worse; roll back." Weak because it misdiagnoses the cause (the
new model may be better), and rolling back leaves the newly embedded products broken in the other
direction. It also proposes no way to detect the problem or to migrate.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Explains that vectors from different models are not comparable (different spaces) | 3 |
| Explains the observed effect on ranking | 1 |
| Detection: version metadata plus an offline eval set and/or online metrics | 3 |
| Migration: full re-embed into a new column/index, evaluate, switch, keep rollback | 3 |

## Scoring

Total 37 pts. Pass ≥ 70% (26 pts). Topic ids to revisit for each lost criterion: exercise 1 →
A1.5 "The math"; exercise 2 → A1.5 "From scratch" and A1.2 (axes and `keepdims`); exercise 3 →
A1.5 "In practice", and later I2.1, I2.2 and I2.4.
