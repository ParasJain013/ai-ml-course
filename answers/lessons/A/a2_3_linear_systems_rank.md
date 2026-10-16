# Answer key: A2.3 Systems of linear equations and rank

Lesson: [`lessons/A/a2_3_linear_systems_rank.md`](../../../lessons/A/a2_3_linear_systems_rank.md)

## Exercise 1 (by hand)

**Solution (a).** Augmented matrix of $2x_1 + x_2 - x_3 = 8$, $-3x_1 - x_2 + 2x_3 = -11$,
$-2x_1 + x_2 + 2x_3 = -3$:

1. Column 1: entries $2, -3, -2$; the largest absolute value is $\lvert -3\rvert$ in $R_2$. Swap $R_1$ and $R_2$:
   $[-3, -1, 2 \mid -11]$, $[2, 1, -1 \mid 8]$, $[-2, 1, 2 \mid -3]$.
2. $R_2 \leftarrow R_2 - \frac{2}{-3}R_1 = R_2 + \frac23 R_1$: $(0,\ 1 - \frac23,\ -1 + \frac43 \mid 8 - \frac{22}{3}) = (0, \frac13, \frac13 \mid \frac23)$.
   $R_3 \leftarrow R_3 - \frac{-2}{-3}R_1 = R_3 - \frac23 R_1$: $(0,\ 1 + \frac23,\ 2 - \frac43 \mid -3 + \frac{22}{3}) = (0, \frac53, \frac23 \mid \frac{13}{3})$.
3. Column 2, rows 2–3: entries $\frac13, \frac53$; swap $R_2$ and $R_3$ so the pivot is $\frac53$.
4. $R_3 \leftarrow R_3 - \frac{1/3}{5/3}R_2 = R_3 - \frac15 R_2$: $(0, 0,\ \frac13 - \frac{2}{15} \mid \frac23 - \frac{13}{15}) = (0, 0, \frac15 \mid -\frac15)$.

   $$\begin{pmatrix}-3 & -1 & 2 & \mid & -11\\ 0 & \tfrac53 & \tfrac23 & \mid & \tfrac{13}{3}\\ 0 & 0 & \tfrac15 & \mid & -\tfrac15\end{pmatrix}$$

5. Back substitution: $x_3 = (-\frac15)/(\frac15) = -1$; $x_2 = (\frac{13}{3} - \frac23(-1))/\frac53 = 5 / \frac53 = 3$;
   $x_1 = (-11 + x_2 - 2x_3)/(-3) = (-11 + 3 + 2)/(-3) = 2$.
6. Check: $4 + 3 + 1 = 8$ ✓; $-6 - 3 - 2 = -11$ ✓; $-4 + 3 - 2 = -3$ ✓. $\mathbf{x} = (2, 3, -1)$.

**Solution (b).** $\mathbf{M} = \begin{pmatrix}1&2&3\\2&4&6\\1&0&1\end{pmatrix}$. $R_2 \leftarrow R_2 - 2R_1$ gives
$(0, 0, 0)$; $R_3 \leftarrow R_3 - R_1$ gives $(0, -2, -2)$. Swap so the zero row is last: pivots in
columns 1 and 2, so $\operatorname{rank}(\mathbf{M}) = 2$ (row 2 is twice row 1).

- $\mathbf{b} = (1, 2, 0)$: the same operations give $2 - 2 \cdot 1 = 0$ for the zero row (consistent, since
  $b_2 = 2b_1$) and $0 - 1 = -1$ for the third row. Equations: $x_1 + 2x_2 + 3x_3 = 1$ and
  $-2x_2 - 2x_3 = -1$. Rank 2 < 3 unknowns: infinitely many. Free $x_3 = t$: $x_2 = \frac12 - t$,
  $x_1 = 1 - 2(\frac12 - t) - 3t = -t$. So $\mathbf{x} = (0, \tfrac12, 0) + t(-1, -1, 1)$. Check the
  direction: $\mathbf{M}(-1, -1, 1) = (-1 - 2 + 3,\ -2 - 4 + 6,\ -1 + 0 + 1) = \mathbf{0}$ ✓.
- $\mathbf{b} = (1, 3, 0)$: the zero row now reads $0 = 3 - 2 = 1$. No solution; $\operatorname{rank}([\mathbf{M} \mid \mathbf{b}]) = 3 > 2$.

Any correct particular solution earns full marks (e.g. $t = -\frac12$ gives $(\frac12, 1, -\frac12)$);
the null-space direction must be a multiple of $(-1, -1, 1)$.

```text
== exercise 1 check ==
x = [2.0, 3.0, -1.0]
rank(M) = 2
b = [1.0, 2.0, 0.0]: rank([M|b]) = 2
b = [1.0, 3.0, 0.0]: rank([M|b]) = 3
  t =  0.0: M (x_p + t n) = [1.0, 2.0, 0.0]
  t =  1.0: M (x_p + t n) = [1.0, 2.0, 0.0]
  t = -2.5: M (x_p + t n) = [1.0, 2.0, 0.0]
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Correct pivot choices (swap to $-3$, then to $\frac53$), each matrix shown | 3 |
| (a) Correct echelon form and back substitution, $\mathbf{x} = (2, 3, -1)$ | 3 |
| (a) Checked in the original equations | 1 |
| (b) rank 2 with the reason | 2 |
| (b) first $\mathbf{b}$: infinitely many, written as $\mathbf{x}_p + t\,\mathbf{n}$, verified | 3 |
| (b) second $\mathbf{b}$: no solution, with the contradictory row or the rank comparison | 2 |

**Likely mistakes:**

- Skipping the pivot swaps (the question asked for partial pivoting); the answer is still right, but
  the method is not, and fractions get uglier.
- Sign errors when subtracting a negative multiple ($R_2 - \frac{2}{-3}R_1$ is an *addition*).
- Checking the answer in the *transformed* equations rather than the originals, which hides earlier
  arithmetic slips.
- Giving only one solution for the consistent singular system, or a null-space vector that was not
  checked ($\mathbf{M}\mathbf{n} = \mathbf{0}$).

## Exercise 2 (code)

**Solution:** [`a2_3_linear_systems_rank_ex2.py`](a2_3_linear_systems_rank_ex2.py). Reduced row echelon
form makes each pivot 1 and clears its column above and below, so each pivot row reads
"pivot variable + (multiples of free variables) = $b$ entry":

```python
def solve_general(A: FloatArray, b: FloatArray, tol: float = 1e-10) -> tuple[FloatArray, FloatArray] | None:
    """All solutions of A x = b as (x_p (d,), N (d, d - rank)), or None if there are none. A (m, d), b (m,)."""
    m, d = A.shape
    R, pivots = rref(np.column_stack([A, b]), tol=tol)
    if d in pivots:  # a pivot in the b column: a row reads 0 = non-zero
        return None
    free = [j for j in range(d) if j not in pivots]
    x_p = np.zeros(d)
    for i, pc in enumerate(pivots):  # free variables = 0: each pivot variable equals its row's b entry
        x_p[pc] = R[i, d]
    N = np.zeros((d, len(free)))
    for k, fc in enumerate(free):  # set one free variable to 1, the others to 0, b to 0
        N[fc, k] = 1.0
        for i, pc in enumerate(pivots):
            N[pc, k] = -R[i, fc]
    return x_p, N
```

```text
== exercise 2: solve_general ==
A_s, b_inf: x_p = [-2.0, 8.0, 0.0]  N (3, 1)  A(x_p + N t) == b for random t: OK  null-space dim matches scipy: OK
  N = [[1.0], [-2.0], [1.0]]
A_s, b_none: no solution   (np rank(A) = 2, rank([A|b]) = 3)
random (4, 6): x_p = [-0.4173, 0.7457, 1.8018, -0.9482, 0.0, 0.0]  N (6, 2)  A(x_p + N t) == b for random t: OK  null-space dim matches scipy: OK
rank-2 (6, 4), b in column space: x_p = [-573.3087, -477.2854, 0.0, 0.0]  N (4, 2)  A(x_p + N t) == b for random t: OK  null-space dim matches scipy: OK
rank-2 (6, 4), random b: no solution   (np rank(A) = 2, rank([A|b]) = 3)
```

The $\mathbf{A}_s$ case reproduces the lesson's $(-2, 8, 0) + t(1, -2, 1)$ exactly. The large particular
solution for the rank-2 `(6, 4)` matrix is correct but a warning sign: setting free variables to 0 can
produce huge entries when the pivot columns are nearly parallel. `np.linalg.lstsq` returns the
smallest solution instead. A strong submission notices this. Note also that a random `(6, 4)` right-hand
side is almost never in a 2-dimensional column space, so "no solution" is the expected answer there;
the test must build a consistent $\mathbf{b}$ as `A @ something`.

**Rubric (15 pts):**

| Criterion | Points |
|---|---|
| RREF correct (pivot 1, cleared above and below, with a tolerance) | 3 |
| Detects inconsistency (pivot in the $\mathbf{b}$ column) and returns `None` | 2 |
| Particular solution with free variables 0 | 2 |
| Null-space basis built from free columns, shape `(d, d - rank)` | 3 |
| Tests: all required cases, random $\mathbf{t}$, `assert_allclose`, null-space dimension vs `scipy.linalg.null_space` | 3 |
| Consistent $\mathbf{b}$ constructed for the rank-deficient test (not a random $\mathbf{b}$ expected to work) | 1 |
| Type hints, docstring with shapes | 1 |

**Likely mistakes:**

- No tolerance: `1e-17` treated as a pivot, so the rank comes out too high and `N` is too small.
- Clearing only *below* the pivots (row echelon, not reduced), then reading `x_p` straight from the
  last column, which is wrong unless you back-substitute.
- Sign of the null-space entries: the pivot variable is *minus* the free column's entry, because the
  row says $x_{\text{pivot}} + R_{i,\text{free}}\,x_{\text{free}} = 0$.
- Comparing `N` with `scipy.linalg.null_space(A)` entry by entry. Null-space bases are not unique
  (scipy's are orthonormal); compare dimensions, and check $\mathbf{A}\mathbf{N} \approx \mathbf{0}$.

## Exercise 3 (think)

**Strong answer:** The five one-hot columns of `city` add up, in every row, to exactly 1, the
intercept column. So the intercept column is a linear combination (the sum) of the five city columns:
the 6 columns of $\mathbf{X}$ have rank at most 5. This is the "dummy variable trap". Then
$\mathbf{X}^\top\mathbf{X}$ `(6, 6)` has rank 5, is singular, and the normal equations have infinitely
many solutions: you can add any constant $c$ to the intercept and subtract $c$ from all five city
coefficients without changing a single prediction (that direction, $(1, -1, -1, -1, -1, -1)$ on those six
weights, is the null space).

`np.linalg.solve(X.T @ X, X.T @ y)` either raises `LinAlgError: Singular matrix` or, if rounding makes
the matrix look barely non-singular, returns enormous, arbitrary coefficients that change completely
with tiny data changes. `LinearRegression` uses a least-squares solver that does not need an invertible
$\mathbf{X}^\top\mathbf{X}$ and returns one valid solution (the minimum-norm one); the predictions are fine,
but the individual coefficients for the intercept and the cities are not meaningful on their own.

Fixes: (1) **drop one category** (`OneHotEncoder(drop="first")` or `pd.get_dummies(drop_first=True)`):
rank becomes full; the intercept is the baseline city's level and each city coefficient is the
difference from that baseline, all else equal. (2) **drop the intercept** (`fit_intercept=False`) and
keep all five columns: each city coefficient is that city's own level. Both give identical
predictions. (3) Alternatively regularize (ridge, [B2.2](../../../lessons/B/b2_2_regularization.md)),
which makes the solution unique but shrinks the coefficients toward a compromise rather than giving
either clean interpretation.

**Weak answer:** "One-hot encoding is the right way to handle categories, so the model is fine; maybe
scale the features." Weak because it misses that the columns are linearly dependent, does not predict
what the solvers do, and scaling does not change rank.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Identifies the dependence: one-hot columns sum to the intercept column; rank 5 of 6 | 3 |
| Consequence: $\mathbf{X}^\top\mathbf{X}$ singular, infinitely many solutions, coefficients not identifiable | 2 |
| What `solve` and `LinearRegression` each do | 2 |
| Two valid fixes, with the meaning of the coefficients under each | 3 |

## Scoring

Total 39 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → A2.3
"Gaussian elimination with partial pivoting" and "The decision rule"; exercise 2 → A2.3 "Null space and
the shape of the solution set", and A3.2 for orthonormal null-space bases; exercise 3 → A2.3 "Rank"
and B1.3 (one-hot encoding).
