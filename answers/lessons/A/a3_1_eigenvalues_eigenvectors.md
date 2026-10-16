# Answer key: A3.1 Eigenvalues and eigenvectors

Lesson: [`lessons/A/a3_1_eigenvalues_eigenvectors.md`](../../../lessons/A/a3_1_eigenvalues_eigenvectors.md)

## Exercise 1 (by hand)

**Solution (a).** $\mathbf{A} = \begin{pmatrix}5&2\\2&2\end{pmatrix}$: $\operatorname{tr} = 7$,
$\det = 5 \cdot 2 - 2 \cdot 2 = 6$. Characteristic polynomial
$\lambda^2 - 7\lambda + 6 = (\lambda - 6)(\lambda - 1)$, so $\lambda_1 = 6$, $\lambda_2 = 1$.

- $\lambda_1 = 6$: $\mathbf{A} - 6\mathbf{I} = \begin{pmatrix}-1&2\\2&-4\end{pmatrix}$, first row
  $-v_1 + 2v_2 = 0$, so $\mathbf{v}_1 = (2, 1)$. Check: $\mathbf{A}(2, 1) = (12, 6) = 6(2, 1)$ ✓.
- $\lambda_2 = 1$: $\mathbf{A} - \mathbf{I} = \begin{pmatrix}4&2\\2&1\end{pmatrix}$, $2v_1 + v_2 = 0$, so
  $\mathbf{v}_2 = (1, -2)$. Check: $\mathbf{A}(1, -2) = (5 - 4, 2 - 4) = (1, -2)$ ✓.
- Trace $6 + 1 = 7$ ✓, determinant $6 \cdot 1 = 6$ ✓, $(2, 1)\cdot(1, -2) = 0$ ✓ (orthogonal, as the
  matrix is symmetric). Any non-zero multiples, such as $(2, 1)/\sqrt5$, are equally correct.

**Solution (b).** Without normalizing, $\mathbf{x}_{k} = \mathbf{A}\mathbf{x}_{k-1}$:

| $k$ | $\mathbf{x}_k$ | Rayleigh quotient $\mathbf{x}_k^\top\mathbf{A}\mathbf{x}_k / \mathbf{x}_k^\top\mathbf{x}_k$ |
|---|---|---|
| 1 | $(5, 2)$ | $(5,2)\cdot(29,14) / 29 = 173/29 = 5.96552$ |
| 2 | $(29, 14)$ | $(29,14)\cdot(173,86) / 1037 = 6221/1037 = 5.99904$ |
| 3 | $(173, 86)$ | $(173,86)\cdot(1037,518) / 37325 = 223949/37325 = 5.99997$ |

The direction approaches $(2, 1)$ ($173/86 \approx 2.01$) and the quotient approaches 6. The error
shrinks fast because $\lambda_2/\lambda_1 = 1/6$.

**Solution (c).** $\mathbf{A}^2$: $\mathbf{A}^2\mathbf{v} = \mathbf{A}(\lambda\mathbf{v}) = \lambda^2\mathbf{v}$, so
36 and 1. $\mathbf{A} + 3\mathbf{I}$: $(\mathbf{A} + 3\mathbf{I})\mathbf{v} = \lambda\mathbf{v} + 3\mathbf{v}$, so
9 and 4. $\mathbf{A}^{-1}$: multiply $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ by $\mathbf{A}^{-1}$ and divide
by $\lambda$ to get $\mathbf{A}^{-1}\mathbf{v} = \frac{1}{\lambda}\mathbf{v}$, so $\frac16$ and 1. In each case
the same $\mathbf{v}$ satisfies the new eigen-equation, so the eigenvectors are unchanged.

```text
== exercise 1 check ==
trace = 7.0  det = 6.0  eigh values = [1.0, 6.0]
A @ [2.0, 1.0] = [12.0, 6.0] = 6 * [2.0, 1.0]: True
A @ [1.0, -2.0] = [1.0, -2.0] = 1 * [1.0, -2.0]: True
A^1 x0 = [5.0, 2.0]       Rayleigh quotient = 5.96552
A^2 x0 = [29.0, 14.0]     Rayleigh quotient = 5.99904
A^3 x0 = [173.0, 86.0]    Rayleigh quotient = 5.99997
eigenvalues of A^2: [1.0, 36.0]  A + 3I: [4.0, 9.0]  A^-1: [0.1667, 1.0]
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Characteristic polynomial and eigenvalues 6, 1 | 3 |
| (a) An eigenvector for each, verified by multiplying | 2 |
| (a) Trace, determinant and orthogonality checks | 2 |
| (b) Three correct iterates | 2 |
| (b) Three Rayleigh quotients (to 3+ decimals) and the comment that they approach 6 | 2 |
| (c) 36 and 1; 9 and 4; 1/6 and 1, each with the one-line reason | 3 |

**Likely mistakes:**

- Sign slip in the polynomial: writing $\lambda^2 + 7\lambda + 6$, which gives $-6, -1$. The trace
  check catches it (eigenvalues must sum to $+7$).
- Solving $(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}$ and getting only $\mathbf{v} = \mathbf{0}$:
  that means $\lambda$ was wrong (the matrix was not singular). The zero vector is never an eigenvector.
- Using $\mathbf{x}_k^\top\mathbf{x}_{k+1} / \mathbf{x}_k^\top\mathbf{x}_k$ with a mismatched pair of vectors
  in the Rayleigh quotient, or dividing by $\lVert\mathbf{x}\rVert$ instead of $\lVert\mathbf{x}\rVert^2$.
- For $\mathbf{A} + 3\mathbf{I}$, thinking the eigenvalues multiply by 3. Adding a multiple of
  $\mathbf{I}$ *shifts* them; exercise 2 builds on exactly this.

## Exercise 2 (code)

**Solution:** [`a3_1_eigenvalues_eigenvectors_ex2.py`](a3_1_eigenvalues_eigenvectors_ex2.py). The core
is power iteration with one line changed: instead of `y = M @ x`, solve `(M - shift I) y = x`.

```python
shifted = M - shift * np.eye(d)  # eigenvalues lambda_j - shift, same eigenvectors
...
y = np.linalg.solve(shifted, x)  # y = shifted^-1 x, without forming the inverse
```

Why it works: $\mathbf{M} - s\mathbf{I}$ has eigenvalues $\lambda_j - s$ (exercise 1c), and its inverse has
eigenvalues $1/(\lambda_j - s)$ with the same eigenvectors. The largest of those in absolute value
belongs to the $\lambda_j$ *closest to* $s$, so power iteration on the inverse finds it. The Rayleigh
quotient is taken with the original $\mathbf{M}$ so it returns $\lambda_j$ directly.

```text
== exercise 2: inverse iteration ==
eigh eigenvalues (ascending): [-2.0037, -0.988, 0.2761, 0.4913, 1.5001, 1.8151]
shift   0.0: lambda = 0.2761  iterations =   23  ratio for the inverted matrix = 0.562  matches eigh: OK
shift  -1.0: lambda = -0.9880  iterations =    4  ratio for the inverted matrix = 0.012  matches eigh: OK
shift   1.4: lambda = 1.5001  iterations =   10  ratio for the inverted matrix = 0.241  matches eigh: OK
shift   1.0: lambda = 1.5001  iterations =  804  ratio for the inverted matrix = 0.983  matches eigh: OK
```

The convergence ratio of the inverted matrix is
$\frac{1/\lvert\lambda_{\text{2nd closest}} - s\rvert}{1/\lvert\lambda_{\text{closest}} - s\rvert} = \frac{\lvert\lambda_{\text{closest}} - s\rvert}{\lvert\lambda_{\text{2nd closest}} - s\rvert}$.
A shift right next to an eigenvalue ($-1.0$ next to $-0.988$) makes the numerator tiny: 4 iterations.
A shift almost exactly halfway between two eigenvalues ($1.0$ sits $0.5001$ from 1.5001 and $0.5087$
from 0.4913) makes the ratio nearly 1: 804 iterations. The learner's shifts will differ; the pattern
is what is graded. (This is the idea behind the Rayleigh-quotient iteration used in real solvers,
which updates the shift every step. Not required.)

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| `inverse_iteration` uses `np.linalg.solve` on $\mathbf{M} - s\mathbf{I}$ each step, never `inv` | 3 |
| Normalizes every step and has a convergence test that tolerates sign flips, plus an iteration cap that raises | 2 |
| Returns the eigenvalue of $\mathbf{M}$ (Rayleigh quotient with $\mathbf{M}$, or $s + 1/\mu$) | 2 |
| Matrix built as in the lab with `default_rng(0)`; four shifts tested | 1 |
| Each result checked against `eigh` with sign alignment and `assert_allclose` | 3 |
| Iteration counts explained with the ratio $\lvert\lambda_{\text{closest}} - s\rvert / \lvert\lambda_{\text{2nd}} - s\rvert$ | 3 |

**Likely mistakes:**

- Returning the dominant eigenvalue $\mu$ of the *inverted* matrix as if it were $\lambda$. The
  relation is $\mu = 1/(\lambda - s)$, so $\lambda = s + 1/\mu$.
- Choosing `shift` exactly equal to an eigenvalue: $\mathbf{M} - s\mathbf{I}$ is singular and `solve`
  raises. In floating point this practically never happens with an arbitrary shift, and a nearly
  singular solve is harmless here (the error lies along the wanted eigenvector), but the code should
  catch `LinAlgError` and nudge the shift rather than crash.
- Comparing eigenvectors without aligning signs, then "fixing" the failure with a huge tolerance.
- Computing `np.linalg.inv(shifted)` once outside the loop. It works, and for a fixed shift it is
  even a reasonable optimization, but the exercise asks for `solve`; the production version factors
  once (`scipy.linalg.lu_factor`, [A3.4](../../../lessons/A/a3_4_numerical_stability.md)) and reuses it.

## Exercise 3 (think)

**Strong answer:** Power iteration's error shrinks like $\lvert\lambda_2/\lambda_1\rvert^k$. For a
transition matrix $\lambda_1 = 1$, so after 100 iterations the leftover error is about $\lvert\lambda_2\rvert^{100}$.
Explanation 1: the redesign made the chain **slow-mixing**: pages split into groups with few links
between them (for example a new checkout funnel that rarely links back), which pushes
$\lvert\lambda_2\rvert$ close to 1. With $\lvert\lambda_2\rvert = 0.99$, $0.99^{100} \approx 0.37$: the
result after 100 steps still depends heavily on the starting vector and on small nightly changes,
so it jumps around. Explanation 2: the chain became **periodic** or **reducible**: a group of pages
that only link among themselves in a cycle gives an eigenvalue $-1$ or a second eigenvalue equal to
1, so there is no unique steady state, or the iteration oscillates forever. (Also acceptable: dead-end
pages with no outgoing links, which break the rows-sum-to-1 assumption.)

Detection: compute the change between the last two iterates and fail the job if it is above a
tolerance; log the number of iterations needed; compute $\lvert\lambda_2\rvert$ directly once in a while
(`scipy.sparse.linalg.eigs` for the top two, or `np.linalg.eigvals` at 50 pages) and alert when it
approaches 1; check that every row sums to 1 and that the graph is strongly connected. Changes: iterate
to a tolerance with a cap instead of a fixed count; add a small teleport probability (mix $\mathbf{P}$
with a uniform matrix, as PageRank does), which guarantees a unique steady state and bounds
$\lvert\lambda_2\rvert$ by the mixing weight; and at 50 pages simply solve for the steady state
directly as a linear system instead of iterating.

**Weak answer:** "The data is noisy after the redesign; average the last seven nights." Weak because
it treats a numerical convergence failure as statistical noise. Averaging hides the symptom, gives no
way to detect it, and leaves the pipeline returning an unconverged vector.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Links convergence to $\lvert\lambda_2/\lambda_1\rvert^k$ with $\lambda_1 = 1$ for a transition matrix | 2 |
| Two plausible eigenvalue explanations (slow mixing / $\lvert\lambda_2\rvert \to 1$; periodic, reducible or dead ends) | 3 |
| Automated detection (convergence test, iteration count, $\lvert\lambda_2\rvert$ monitor, row-sum and connectivity checks) | 3 |
| A concrete fix (tolerance plus cap, teleportation, or a direct solve) | 2 |

## Scoring

Total 38 pts. Pass ≥ 70% (27 pts). Topic ids to revisit for each lost criterion: exercise 1 → A3.1
"The math" (characteristic polynomial, power iteration) and A2.3 (null space); exercise 2 → A3.1
"From scratch" and A2.2 (why `solve`, not `inv`); exercise 3 → A3.1 "Intuition" and "In practice".
