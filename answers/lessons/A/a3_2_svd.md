# Answer key: A3.2 Singular value decomposition

Lesson: [`lessons/A/a3_2_svd.md`](../../../lessons/A/a3_2_svd.md)

## Exercise 1 (by hand)

**Solution (a).** $\mathbf{A} = \begin{pmatrix}2&2\\-1&1\end{pmatrix}$, so
$\mathbf{A}^\top\mathbf{A} = \begin{pmatrix}2&-1\\2&1\end{pmatrix}\begin{pmatrix}2&2\\-1&1\end{pmatrix} = \begin{pmatrix}4 + 1 & 4 - 1\\ 4 - 1 & 4 + 1\end{pmatrix} = \begin{pmatrix}5&3\\3&5\end{pmatrix}$.

- Eigenvalues: $\operatorname{tr} = 10$, $\det = 25 - 9 = 16$, $\lambda^2 - 10\lambda + 16 = (\lambda - 8)(\lambda - 2)$.
  So $\sigma_1 = \sqrt8 = 2\sqrt2 \approx 2.8284$, $\sigma_2 = \sqrt2 \approx 1.4142$.
- Eigenvectors: $\mathbf{v}_1 = (1, 1)/\sqrt2$ (for 8), $\mathbf{v}_2 = (1, -1)/\sqrt2$ (for 2).
- $\mathbf{u}_1 = \mathbf{A}\mathbf{v}_1/\sigma_1 = \frac{(4, 0)/\sqrt2}{2\sqrt2} = (1, 0)$;
  $\mathbf{u}_2 = \mathbf{A}\mathbf{v}_2/\sigma_2 = \frac{(0, -2)/\sqrt2}{\sqrt2} = (0, -1)$.
- $\sigma_1\sigma_2 = 2\sqrt2\cdot\sqrt2 = 4$ and $\lvert\det\mathbf{A}\rvert = \lvert 2 \cdot 1 - 2 \cdot (-1)\rvert = 4$ ✓.

Flipping the sign of a $\mathbf{v}_j$ *together with* its $\mathbf{u}_j$ is equally correct.

**Solution (b).** $\mathbf{A}_1 = \sigma_1\mathbf{u}_1\mathbf{v}_1^\top = 2\sqrt2\,(1, 0)^\top\frac{(1, 1)}{\sqrt2} = \begin{pmatrix}2&2\\0&0\end{pmatrix}$.
$\mathbf{A} - \mathbf{A}_1 = \begin{pmatrix}0&0\\-1&1\end{pmatrix}$, squared Frobenius norm $1 + 1 = 2 = \sigma_2^2$ ✓.

**Solution (c).** $\sigma_1/\sigma_2 = 2\sqrt2/\sqrt2 = 2$.

**Solution (d).** $k(n + d + 1) = 40 \times (1000 + 800 + 1) = 72{,}040$ numbers, against
$1000 \times 800 = 800{,}000$: 9.0%.

```text
== exercise 1 check ==
A^T A = [[5.0, 3.0], [3.0, 5.0]]  singular values = [2.8284, 1.4142]  (2 sqrt 2 = 2.8284 )
V (columns) = [[0.7071, 0.7071], [0.7071, -0.7071]]  U (columns) = [[1.0, 0.0], [0.0, -1.0]]
rank-1 approximation = [[2.0, 2.0], [0.0, 0.0]]  squared Frobenius error = 2.0
condition number = 2.0  |det A| = 4.0 = s1 * s2 = 4.0
storage of a rank-40 approximation of a 1000 x 800 matrix: 72040 numbers = 9.0%
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) $\mathbf{A}^\top\mathbf{A}$ correct (not $\mathbf{A}\mathbf{A}^\top$) | 2 |
| (a) $\sigma_1 = 2\sqrt2$, $\sigma_2 = \sqrt2$ as square roots of the eigenvalues | 2 |
| (a) $\mathbf{v}_1, \mathbf{v}_2$ unit length; $\mathbf{u}_j = \mathbf{A}\mathbf{v}_j/\sigma_j$ correct | 3 |
| (a) Determinant check | 1 |
| (b) $\mathbf{A}_1$ correct and the error equals $\sigma_2^2$ | 3 |
| (c) Condition number 2 | 1 |
| (d) 72,040 numbers, 9.0% | 2 |

**Likely mistakes:**

- Taking the eigenvalues of $\mathbf{A}^\top\mathbf{A}$ as the singular values, forgetting the square root
  ($\sigma = 8$ and 2 instead of $2\sqrt2$ and $\sqrt2$). The determinant check catches it ($16 \ne 4$).
- Computing the eigenvalues of $\mathbf{A}$ itself. For this non-symmetric matrix they are complex
  ($\operatorname{tr} = 3$, $\det = 4$, $9 - 16 < 0$), which is the whole reason for the SVD.
- Not normalizing $\mathbf{v}_j$ before computing $\mathbf{u}_j = \mathbf{A}\mathbf{v}_j/\sigma_j$, which gives
  a $\mathbf{u}_j$ of length $\sqrt2$.
- Choosing the signs of $\mathbf{u}_j$ independently of $\mathbf{v}_j$ (for example $\mathbf{u}_2 = (0, 1)$ with
  $\mathbf{v}_2 = (1, -1)/\sqrt2$), so $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top \ne \mathbf{A}$. Always reconstruct.
- Storage as $k(n + d)$ (forgetting the $k$ singular values; acceptable if stated) or as $k \cdot n \cdot d$.

## Exercise 2 (code)

**Solution:** [`a3_2_svd_ex2.py`](a3_2_svd_ex2.py).

```python
def truncate(U: FloatArray, s: FloatArray, Vt: FloatArray, k: int) -> FloatArray:
    """Rank-k reconstruction. U (n, r), s (r,), Vt (r, d) -> (n, d)."""
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k]

U_y, s_y, Vt_y = np.linalg.svd(Y, full_matrices=False)
err_vs_clean = [np.linalg.norm(truncate(U_y, s_y, Vt_y, k) - L) / np.linalg.norm(L) for k in ks]
```

```text
== exercise 2: low-rank signal plus noise ==
L (100, 80) rank 3   Y rank 80
top 6 singular values of Y = [103.3409, 94.3423, 78.1867, 18.2258, 17.788, 17.5173]
relative error of the raw noisy Y against L:      0.5657
relative error of the rank-1  truncation against L: 0.7707
relative error of the rank-2  truncation against L: 0.4966
relative error of the rank-3  truncation against L: 0.1403
relative error of the rank-4  truncation against L: 0.1815
relative error of the rank-10 truncation against L: 0.3157
relative error of the rank-20 truncation against L: 0.4257
best k against the clean matrix: 3   k chosen by the 90% energy rule: 21 (energy at k=3: 0.773)
saved outputs/a3_2_answers/ex2_denoise.png
```

The spectrum has three large singular values (78 to 103) and then a flat floor of noise (about 18 and
falling slowly). Truncating at $k = 3$ cuts the error against $\mathbf{L}$ from 0.566 (the raw noisy
matrix) to 0.140, because the discarded layers are almost pure noise. Every extra layer beyond 3 adds
mostly noise back, so the error rises again. The 90% energy rule picks $k = 21$: noise is spread over
*all* 80 directions, so together the 77 noise layers hold a large share of the total energy (here
about 23%), and an energy threshold keeps many of them. The rule measures how much of $\mathbf{Y}$ you
keep, not how much of the *signal*; the gap in the spectrum is the better guide here. In a real
problem you have no clean $\mathbf{L}$, so you choose $k$ with the spectrum's gap or a held-out error
(hide some entries, fit on the rest, measure error on the hidden ones).

**Rubric (15 pts):**

| Criterion | Points |
|---|---|
| Data built as specified (`default_rng(0)`, P then Q then noise, shapes stated) | 2 |
| Thin SVD with `full_matrices=False`; truncation correct (uses `Vt` rows) | 3 |
| Singular values plotted on a log scale, with the gap at 3 pointed out | 2 |
| Error against the clean $\mathbf{L}$ for $k = 1..20$, plotted with the noisy-$\mathbf{Y}$ reference line | 3 |
| Best $k = 3$ identified, and why error rises after it (adding noise back) | 2 |
| Energy rule's $k$ computed, and the disagreement explained (noise spreads energy over all directions) | 3 |

**Likely mistakes:**

- Measuring the error against $\mathbf{Y}$ instead of $\mathbf{L}$: that error falls monotonically to 0 at
  $k = 80$ and teaches nothing about denoising.
- Treating `Vt` as if it were $\mathbf{V}$: `U[:, :k] @ np.diag(s[:k]) @ Vt.T[:k]` has the right shape
  `(k, 80)` because `Vt` is square here, so it runs and returns garbage. (`Vt[:, :k]` at least fails
  with a shape error.) Reconstructing at $k = 80$ and comparing with $\mathbf{Y}$ catches both.
- Plotting singular values on a linear scale, where the noise floor looks like zero and the gap is
  less clear.
- Concluding "always keep 90% energy" or "always pick the elbow". The point is that $k$ is a modelling
  choice that needs a criterion tied to the goal.

## Exercise 3 (think)

**Strong answer:** Measure the task, not the reconstruction. Build an evaluation set of real queries
with known relevant items (or use logged clicks) and compare recall@10 and nDCG of exact nearest
neighbours at 768 dimensions with the same at 128 after projection. Also measure the latency and storage
actually saved with the real index type (HNSW or IVFFlat in pgvector), because index build time,
memory and recall knobs change with dimension. Look at the singular value spectrum of the embedding
matrix: if it decays slowly, 128 dimensions will discard real information.

Mechanics: the projection is $\mathbf{z} = \mathbf{V}_k^\top\mathbf{x}$ with $\mathbf{V}_k$ `(768, 128)` from the
SVD of the item matrix. Every new item *and every query* must go through the same $\mathbf{V}_k$, so it
becomes a versioned artifact deployed with the search service. If you center before the SVD (PCA), the
same stored mean must be subtracted. Cosine similarity is not preserved exactly: decide whether to
re-normalize after projecting, and test both.

Over time: new items and queries drift away from the distribution $\mathbf{V}_k$ was fit on (new product
categories), and recall degrades silently; re-fitting changes $\mathbf{V}_k$ (including sign flips), which
means re-projecting and re-indexing all million items in one migration, never mixing versions in one
index. Monitor recall on a fixed evaluation set and the reconstruction error of new embeddings.

Choosing $k$: sweep $k \in \{64, 96, 128, 192, 256, 384\}$, plot recall against storage and latency, and
pick the smallest $k$ within an agreed recall tolerance (for example 1% below the 768-dimension
baseline). Alternatives worth comparing: product quantization or halfvec storage (VERIFY (as of 2026-10)
which pgvector versions support it), or an embedding model that natively supports shorter vectors.

**Weak answer:** "Keep 95% of the variance, that will be fine; SVD is the optimal compression." Weak
because Eckart–Young optimality is about reconstruction error, not search quality; it says nothing
about projecting queries, versioning $\mathbf{V}_k$, drift or re-indexing, and it measures no task metric.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Evaluates retrieval quality (recall/nDCG on real queries) against the 768-d baseline, plus real latency and storage | 3 |
| New items and queries projected with the same versioned $\mathbf{V}_k$ (and mean, if centered); similarity effects noted | 3 |
| Drift and re-fitting risks (silent degradation, sign flips, full re-index) with monitoring | 2 |
| Chooses $k$ by a sweep against a task metric, not by an energy threshold alone | 2 |

## Scoring

Total 39 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → A3.2
"The math" and "Worked example", A3.1 (eigenvalues of a symmetric matrix); exercise 2 → A3.2 "With NumPy
and scikit-learn" (shapes, `Vt`) and "In practice" (choosing $k$); exercise 3 → A3.2 "In practice",
later B5.4 (PCA) and I2.1 (embeddings and semantic search).
