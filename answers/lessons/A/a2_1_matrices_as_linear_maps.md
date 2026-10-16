# Answer key: A2.1 Matrices as linear maps

Lesson: [`lessons/A/a2_1_matrices_as_linear_maps.md`](../../../lessons/A/a2_1_matrices_as_linear_maps.md)

## Exercise 1 (by hand)

**Solution.** $\mathbf{M} = \begin{pmatrix}0 & 2\\ 1 & 0\end{pmatrix}$.

(a) The columns: $\mathbf{M}\mathbf{e}_1 = (0, 1)$ and $\mathbf{M}\mathbf{e}_2 = (2, 0)$. The unit square's
bottom edge (along $\mathbf{e}_1$) now points straight up with length 1; its left edge (along
$\mathbf{e}_2$) now points right with length 2. The square becomes a 2 × 1 rectangle lying along the
x-axis, but "flipped": going counter-clockwise around the original square ($\mathbf{e}_1$ then
$\mathbf{e}_2$) now goes clockwise. The map swaps the two coordinates and doubles the new first one.

(b) $\mathbf{x} = (3, -1)$. Row view: row 1 $(0, 2)\cdot(3, -1) = -2$; row 2 $(1, 0)\cdot(3, -1) = 3$.
Column view: $3(0, 1) + (-1)(2, 0) = (-2, 3)$. Both give $(-2, 3)$.

(c) $\det\mathbf{M} = 0 \cdot 0 - 2 \cdot 1 = -2$. Size 2: every area doubles (the unit square becomes a
2 × 1 rectangle). Negative sign: orientation flips (a mirror image), as seen in (a). Non-zero: the map
can be undone.

(d)
- `(10, 2) @ M.T`: `(10, 2) @ (2, 2)` → `(10, 2)` (apply $\mathbf{M}$ to 10 points stored as rows).
- `(2, 10) @ M`: error: 10 columns on the left, 2 rows on the right.
- `M @ (2,)`: `(2,)`.
- `(4, 3) @ (3, 2) @ (4, 2)`: the first product gives `(4, 2)`; then `(4, 2) @ (4, 2)` is an error (2
  columns on the left, 4 rows on the right).

```text
== exercise 1 check ==
M e1 = [0.0, 1.0]   M e2 = [2.0, 0.0]
row view: [-2.0, 3.0]   column view: [-2.0, 3.0]
det(M) = -2.0
[(10, 2), (2, 2)] -> (10, 2)
[(2, 10), (2, 2)] -> error
[(2, 2), (2,)] -> (2,)
[(4, 3), (3, 2), (4, 2)] -> error
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) Both column images correct and the square's image described or sketched | 2 |
| (b) Both views computed and agree on $(-2, 3)$ | 3 |
| (c) det = −2, with both the size (area doubles) and the sign (flip) interpreted | 3 |
| (d) All four shapes correct, with the reason for each error | 4 (1 each) |

**Likely mistakes:**

- Reading the *rows* as the images of the basis vectors ($\mathbf{e}_1 \to (0, 2)$). Rows are what
  you dot with; columns are where $\mathbf{e}_j$ lands.
- Calling $\det = -2$ "shrinks" or "invalid" because it is negative. A negative determinant is a flip
  plus a scaling by its absolute value.
- In (d), applying the shape rule to `(4, 3) @ (3, 2) @ (4, 2)` as if it checked only the first and
  last sizes. Each adjacent pair must match, after the previous product is formed.
- Forgetting that `M @ (2,)` returns a 1-D `(2,)`, not `(2, 1)`.

## Exercise 2 (code)

**Solution:** [`a2_1_matrices_as_linear_maps_ex2.py`](a2_1_matrices_as_linear_maps_ex2.py). The key
is to build the matrix from the images of $\mathbf{e}_1$ (angle 0 → angle $2\phi$) and $\mathbf{e}_2$
(angle 90° → angle $2\phi - 90°$):

$$\mathbf{F}_\phi = \begin{pmatrix}\cos 2\phi & \sin 2\phi\\ \sin 2\phi & -\cos 2\phi\end{pmatrix},$$

using $\cos(2\phi - 90°) = \sin 2\phi$ and $\sin(2\phi - 90°) = -\cos 2\phi$.

```python
def reflection(phi_deg: float) -> FloatArray:
    """Reflect across the line through the origin at angle phi to the x-axis. -> (2, 2)."""
    two_phi = math.radians(2.0 * phi_deg)
    col1 = [math.cos(two_phi), math.sin(two_phi)]  # image of e1
    col2 = [math.cos(two_phi - math.pi / 2), math.sin(two_phi - math.pi / 2)]  # image of e2
    return np.array(col1 + col2).reshape(2, 2, order="F")  # columns are the images
```

```text
== exercise 2: reflection matrix ==
phi =   0.0: F = [[1.0, 0.0], [0.0, -1.0]]  det = -1.0000  all checks OK
phi =  30.0: F = [[0.5, 0.866], [0.866, -0.5]]  det = -1.0000  all checks OK
phi =  45.0: F = [[0.0, 1.0], [1.0, 0.0]]  det = -1.0000  all checks OK
phi =  90.0: F = [[-1.0, 0.0], [0.0, 1.0]]  det = -1.0000  all checks OK
phi = 123.0: F = [[-0.4067, -0.9135], [-0.9135, 0.4067]]  det = -1.0000  all checks OK
```

Sanity anchors a reviewer can check at a glance: $\phi = 0$ is the mirror in the x-axis
$\operatorname{diag}(1, -1)$, and $\phi = 45°$ swaps the coordinates. The checks in the solution apply
`pts @ F.T` twice to 100 random points (rows) and compare with the originals, compare row lengths
before and after, and test a point on the mirror line.

**Rubric (15 pts):**

| Criterion | Points |
|---|---|
| Matrix built from the images of $\mathbf{e}_1$ and $\mathbf{e}_2$ (as asked), and correct | 4 |
| Reflect-twice-is-identity checked with asserts on several angles | 2 |
| Lengths preserved, checked with asserts | 2 |
| Points on the line unchanged, checked | 2 |
| det = −1 checked | 1 |
| Plot of a non-symmetric shape before and after, mirror line shown, labelled axes and a takeaway title | 2 |
| Type hints, docstring, points stored as rows and transformed with `@ F.T` (or a stated alternative) | 2 |

**Likely mistakes:**

- Putting the images in the rows instead of the columns, which gives the transpose. For a reflection
  the matrix is symmetric, so this happens to give the right answer here; it would not for a rotation.
  Ask yourself whether you built it by columns.
- Testing only $\phi = 0$ and $45°$, where many wrong formulas also pass (for example, swapping the
  sign of the off-diagonal). Test an angle like 123°.
- Degrees vs radians: `math.cos(60)` is the cosine of 60 *radians*.
- Asserting floats with `==`.

## Exercise 3 (think)

**Strong answer:** With `d = 64` and batches of 1 to 256 rows, a `(d, n)` matrix reaches the model's
first layer, `X @ W` with `W` `(64, k)`. It **crashes** whenever $n \ne 64$: `(64, n) @ (64, k)` fails
because $n \ne 64$ columns meet 64 rows. It **runs silently** when $n = 64$ exactly: `(64, 64) @ (64, k)`
is valid, but row $i$ is now "feature $i$ across 64 examples", so the model returns 64 predictions,
each a meaningless mix of one feature's values across different requests. Worse, every request's
prediction depends on the other 63 requests in the batch. Batches of exactly 64 rows may be common if
the batcher fills to a power of two, so this can affect a real share of traffic while health checks
(which often send 1 row) keep crashing loudly and look like a different bug.

Checks: assert the shape at the boundary where data enters the model (`assert X.ndim == 2 and
X.shape[1] == N_FEATURES`), ideally with a schema that names the columns (a DataFrame with expected
column names, or a pydantic model on the request) so a transpose cannot pass; log or reject batches
whose shape is unexpected. Tests: a unit test with a **non-square** batch (e.g. 5 × 64) *and* a
square one (64 × 64) whose expected predictions were computed row by row (each row predicted alone
must equal its prediction inside the batch), which catches both failure modes; and a property test
over random batch sizes 1 to 256.

**Weak answer:** "It will always throw a shape error, so we'll see it in the logs." Weak because the
$n = d$ case runs without any error, which is exactly the dangerous one, and it proposes no check or
test.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Crash case identified ($n \ne 64$) with the shape reasoning | 2 |
| Silent case identified ($n = 64$) and what the wrong output means | 3 |
| Shape/schema validation at the model boundary | 2 |
| A test that would catch it: non-square batch and a row-by-row vs batch consistency check | 3 |

## Scoring

Total 37 pts. Pass ≥ 70% (26 pts). Topic ids to revisit for each lost criterion: exercise 1 → A2.1
"Intuition: the columns tell you everything" and "The shape rule"; exercise 2 → A2.1 "The four
classic 2×2 maps" and A1.2 (assertions on arrays); exercise 3 → A2.1 "Many points at once" and
"In practice".
