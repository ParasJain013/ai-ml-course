# Answer key: A2.2 Matrix multiplication, transpose, identity and inverse

Lesson: [`lessons/A/a2_2_matrix_multiplication_inverse.md`](../../../lessons/A/a2_2_matrix_multiplication_inverse.md)

## Exercise 1 (by hand)

**Solution.** $\mathbf{P} = \begin{pmatrix}1&2&0\\0&1&1\end{pmatrix}$ `(2, 3)`,
$\mathbf{Q} = \begin{pmatrix}2&1\\0&1\\1&0\end{pmatrix}$ `(3, 2)`.

(a) $\mathbf{P}\mathbf{Q}$ is `(2, 3) @ (3, 2)` → `(2, 2)`. Each entry is a row of $\mathbf{P}$ dotted with a
column of $\mathbf{Q}$:

- $(1, 2, 0)\cdot(2, 0, 1) = 2$, $(1, 2, 0)\cdot(1, 1, 0) = 3$,
- $(0, 1, 1)\cdot(2, 0, 1) = 1$, $(0, 1, 1)\cdot(1, 1, 0) = 1$.

$\mathbf{P}\mathbf{Q} = \begin{pmatrix}2&3\\1&1\end{pmatrix}$. $\mathbf{Q}\mathbf{P}$ is `(3, 2) @ (2, 3)` → `(3, 3)`:
rows of $\mathbf{Q}$ dotted with columns of $\mathbf{P}$ give
$\mathbf{Q}\mathbf{P} = \begin{pmatrix}2&5&1\\0&1&1\\1&2&0\end{pmatrix}$ (for example, the top-middle entry is
$(2, 1)\cdot(2, 1) = 5$).

(b) $\mathbf{Q}^\top = \begin{pmatrix}2&0&1\\1&1&0\end{pmatrix}$, $\mathbf{P}^\top = \begin{pmatrix}1&0\\2&1\\0&1\end{pmatrix}$, and
$\mathbf{Q}^\top\mathbf{P}^\top = \begin{pmatrix}2 + 0 + 0 & 0 + 0 + 1\\ 1 + 2 + 0 & 0 + 1 + 0\end{pmatrix} = \begin{pmatrix}2&1\\3&1\end{pmatrix}$,
which is $(\mathbf{P}\mathbf{Q})^\top$. ✓

(c) $\det(\mathbf{P}\mathbf{Q}) = 2 \cdot 1 - 3 \cdot 1 = -1$, so
$(\mathbf{P}\mathbf{Q})^{-1} = \frac{1}{-1}\begin{pmatrix}1&-3\\-1&2\end{pmatrix} = \begin{pmatrix}-1&3\\1&-2\end{pmatrix}$.
Check: $\begin{pmatrix}2&3\\1&1\end{pmatrix}\begin{pmatrix}-1&3\\1&-2\end{pmatrix} = \begin{pmatrix}-2 + 3 & 6 - 6\\ -1 + 1 & 3 - 2\end{pmatrix} = \mathbf{I}$. ✓

(d) $\mathbf{Q}\mathbf{P}\mathbf{x}$ means "first $\mathbf{P}$, then $\mathbf{Q}$". $\mathbf{P}$ maps 3-D vectors to
2-D vectors, so whatever comes out of it lies in a 2-dimensional space; $\mathbf{Q}$ can then only
produce combinations of its 2 columns. So every output of $\mathbf{Q}\mathbf{P}$ lies in a plane inside
3-D space: the map squashes 3-D space onto a plane, loses information (some non-zero $\mathbf{x}$ goes
to $\mathbf{0}$, because $\mathbf{P}$ has 3 columns in 2-D, so they are dependent) and cannot be undone.
Its rank is at most 2 < 3, so it is singular.

```text
== exercise 1 check ==
PQ = [[2.0, 3.0], [1.0, 1.0]] (2, 2)   QP = [[2.0, 5.0, 1.0], [0.0, 1.0, 1.0], [1.0, 2.0, 0.0]] (3, 3)
Q^T P^T = [[2.0, 1.0], [3.0, 1.0]]  == (PQ)^T: True
det(PQ) = -1.0   (PQ)^-1 = [[-1.0, 3.0], [1.0, -2.0]]   PQ (PQ)^-1 = [[1.0, 0.0], [0.0, 1.0]]
rank(QP) = 2  det(QP) = 0.0
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) $\mathbf{P}\mathbf{Q}$ and $\mathbf{Q}\mathbf{P}$ correct, with shapes | 4 |
| (b) $\mathbf{Q}^\top\mathbf{P}^\top$ correct and compared with $(\mathbf{P}\mathbf{Q})^\top$ | 2 |
| (c) inverse correct and verified by multiplication | 3 |
| (d) argument that the output lives in at most 2 dimensions, so $\mathbf{Q}\mathbf{P}$ is not invertible | 5 |

**Likely mistakes:**

- Multiplying row by row (row of $\mathbf{P}$ times row of $\mathbf{Q}$). The rule is row times column.
- Writing $(\mathbf{P}\mathbf{Q})^\top = \mathbf{P}^\top\mathbf{Q}^\top$: check the shapes, `(3, 2) @ (2, 3)` is
  `(3, 3)`, not `(2, 2)`.
- For the inverse, negating the diagonal instead of the off-diagonal, or forgetting to divide by a
  negative determinant.
- In (d), computing $\det(\mathbf{Q}\mathbf{P})$ by brute force (allowed for a check, but the question
  asked for the reason) or arguing "it is not square", which is false: it is `(3, 3)`.

## Exercise 2 (code)

**Solution:** [`a2_2_matrix_multiplication_inverse_ex2.py`](a2_2_matrix_multiplication_inverse_ex2.py).
Repeated squaring reads $k$ in binary: $\mathbf{M}^{13} = \mathbf{M}^{8}\mathbf{M}^{4}\mathbf{M}^{1}$
because $13 = 1101_2$. Keep squaring a `base` ($\mathbf{M}, \mathbf{M}^2, \mathbf{M}^4, \dots$) and multiply
it into the result whenever the current bit is 1. That is about $\log_2 k$ squarings plus at most
$\log_2 k$ multiplications, instead of $k - 1$.

```python
def matpow(M: FloatArray, k: int) -> tuple[FloatArray, int]:
    """M^k for square M (m, m) and k >= 0 by repeated squaring. Returns (M^k, number of products)."""
    ...
    result = np.eye(m)
    base = M.astype(float)
    products = 0
    first = True  # multiplying the identity by base is free: just copy
    while k > 0:
        if k & 1:  # this bit of k is set: include the current power of M
            if first:
                result, first = base.copy(), False
            else:
                result = matmul_loops(result, base)
                products += 1
        k >>= 1
        if k:  # square only if there are bits left
            base = matmul_loops(base, base)
            products += 1
    return result, products
```

```text
== exercise 2: matpow by repeated squaring ==
k =  0: matches np.linalg.matrix_power: OK   products:  0 (naive: 0)
k =  1: matches np.linalg.matrix_power: OK   products:  0 (naive: 0)
k =  2: matches np.linalg.matrix_power: OK   products:  1 (naive: 1)
k =  5: matches np.linalg.matrix_power: OK   products:  3 (naive: 4)
k =  8: matches np.linalg.matrix_power: OK   products:  3 (naive: 7)
k = 13: matches np.linalg.matrix_power: OK   products:  5 (naive: 12)
k = 64: matches np.linalg.matrix_power: OK   products:  6 (naive: 63)
F^30 = [[1346269.0, 832040.0], [832040.0, 514229.0]]
F_30 = top-right entry = 832040

== exercise 2: k = 100 in float64 ==
F_100 exact (Python int) = 354224848179261915075
F_100 from float64       = 354224848179261931520
difference               = 16445
2**53 = 9007199254740992  (largest range where float64 holds every integer exactly)
int64 F^100 top-right = 3736710778780434371 (wrapped around: wrong)
```

**Why Fibonacci:** $\mathbf{F}(F_{j+1}, F_j) = (F_{j+1} + F_j, F_{j+1}) = (F_{j+2}, F_{j+1})$: one
multiplication advances the pair by one step. So
$\mathbf{F}^k = \begin{pmatrix}F_{k+1} & F_k\\ F_k & F_{k-1}\end{pmatrix}$ (true for $k = 1$; multiply by
$\mathbf{F}$ to go from $k$ to $k + 1$), and the top-right entry is $F_k$.

**What goes wrong at $k = 100$:** $F_{100} \approx 3.5 \times 10^{20}$ is far beyond $2^{53} \approx
9 \times 10^{15}$, the range in which `float64` (53 significant bits) represents every integer
exactly, so the result is right to about 16 significant digits but the last digits are rounding noise
(off by 16,445 here). With `int64` it is worse: the maximum is about $9.2 \times 10^{18}$, and
`np.linalg.matrix_power` on integers silently wraps around to a wrong number. Python's `int` is exact
at any size. Full marks for identifying the float64 precision limit; mentioning the int64 overflow is a
bonus within the same criterion.

**Rubric (15 pts):**

| Criterion | Points |
|---|---|
| Repeated squaring correct for $k = 0$ and $k = 1$ (identity, $\mathbf{M}$) and general $k$ | 4 |
| Uses the from-scratch `matmul_loops` for every product | 1 |
| Product count reported and compared with $k - 1$ (about $2\log_2 k$) | 2 |
| Checked against `np.linalg.matrix_power` with `assert_allclose` on several $k$ | 2 |
| $F_{30} = 832040$ and the reason the top-right entry is $F_k$ | 3 |
| $k = 100$: explains the float64 exact-integer limit ($2^{53}$) with the observed error | 2 |
| Type hints, docstring with shapes, input validation (square, $k \ge 0$) | 1 |

**Likely mistakes:**

- Squaring once too often at the end of the loop (harmless for the result, wrong for the count).
- Returning `M` for $k = 0$ instead of the identity.
- Starting `result` as `M` instead of the identity, which computes $\mathbf{M}^{k+1}$.
- Reading $\mathbf{F}^k$'s top-*left* entry, which is $F_{k+1}$.
- Claiming the $k = 100$ answer "overflows" in float64. It does not overflow (float64 reaches about
  $10^{308}$); it loses *precision*.

## Exercise 3 (think)

**Strong answer:** The two orders are $\mathbf{W}\mathbf{S}\mathbf{x}$ ("convert units, then mix") and
$\mathbf{S}\mathbf{W}\mathbf{x}$ ("mix, then scale the *mixed* outputs"). Matrix multiplication is not
commutative, so they are equal only for special cases: if all three unit constants are the same,
$\mathbf{S} = s\mathbf{I}$ commutes with everything and the swap is safe; if $\mathbf{W}$ is itself
diagonal (no mixing), the swap is safe too. Entry-wise, $(\mathbf{W}\mathbf{S})_{ij} = W_{ij}s_j$ while
$(\mathbf{S}\mathbf{W})_{ij} = s_iW_{ij}$, so they agree only where $W_{ij} = 0$ or $s_i = s_j$. Three
different unit conversions with a general learned $\mathbf{W}$ will give different numbers, and the
learned $\mathbf{W}$ was fitted on converted inputs, so only the original order is correct. "Same
multiplications" is wrong: the same numbers are multiplied, but in a different pattern.

For serving, precompute the single matrix $\mathbf{C} = \mathbf{W}\mathbf{S}$ `(3, 3)` once at load time
and apply `X @ C.T` to each batch: one product instead of two, by associativity. Before merging any
change to the pipeline, add a golden test: a fixed set of real input rows with outputs recorded from
the current pipeline, asserted with `np.testing.assert_allclose` against the new pipeline, plus a
randomized test over many inputs. That test fails immediately for the swap.

**Weak answer:** "It is the same multiplications, just reordered, so the result is the same up to
rounding." Weak because it treats matrix multiplication like multiplying numbers; it is not
commutative. It also offers no test.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| States that $\mathbf{W}\mathbf{S} \ne \mathbf{S}\mathbf{W}$ in general (not commutative) | 3 |
| Identifies when it is safe ($\mathbf{S} = s\mathbf{I}$, or $\mathbf{W}$ diagonal / entries with $s_i = s_j$) | 2 |
| Precompute $\mathbf{C} = \mathbf{W}\mathbf{S}$ and apply once (associativity) | 2 |
| A concrete test (golden outputs / property test with `assert_allclose`) before merging | 3 |

## Scoring

Total 39 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → A2.2
"The math" (product, transpose, inverse) and A2.3 (rank) for (d); exercise 2 → A2.2 "From scratch" and,
for the precision question, A3.4; exercise 3 → A2.2 "Intuition: order matters" and "Rules".
