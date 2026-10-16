# Answer key: A1.2 NumPy arrays

Lesson: [`lessons/A/a1_2_numpy_arrays.md`](../../../lessons/A/a1_2_numpy_arrays.md)

## Exercise 1 (by hand)

**Solution (shapes).** Align from the right, pad the shorter shape with 1s on the left, then each
axis must match or be 1.

| | Padded | Axis by axis (left to right) | Result |
|---|---|---|---|
| (a) `(4, 1) + (3,)` | `(4, 1)` vs `(1, 3)` | 4/1 → 4; 1/3 → 3 | `(4, 3)` |
| (b) `(2, 3) * (2,)` | `(2, 3)` vs `(1, 2)` | last axis 3 vs 2, neither is 1 | **error** |
| (c) `(5, 1, 4) - (3, 1)` | `(5, 1, 4)` vs `(1, 3, 1)` | 5/1 → 5; 1/3 → 3; 4/1 → 4 | `(5, 3, 4)` |
| (d) `X - X.mean(axis=1)`, `X` `(6, 3)` | `(6, 3)` vs `(1, 6)` (the mean is `(6,)`) | last axis 3 vs 6 | **error** |
| (e) same with `keepdims=True` | `(6, 3)` vs `(6, 1)` | 6/6 → 6; 3/1 → 3 | `(6, 3)` |

(d) is the classic bug: you wanted to subtract each row's mean, and the `(6,)` vector lines up with
the *columns*. If `X` were square, say `(6, 6)`, it would not even fail; it would subtract the wrong
means silently.

**Solution (distances).** $\mathbf{A} = [[1, 1], [2, 3]]$, $\mathbf{B} = [[1, 1], [4, 5], [1, 4]]$.

- $\lVert\mathbf{a}_i\rVert^2$: $1 + 1 = 2$; $4 + 9 = 13$.
- $\lVert\mathbf{b}_j\rVert^2$: $1 + 1 = 2$; $16 + 25 = 41$; $1 + 16 = 17$.
- $\mathbf{A}\mathbf{B}^\top$: row 1: $1\cdot1 + 1\cdot1 = 2$, $4 + 5 = 9$, $1 + 4 = 5$; row 2:
  $2 + 3 = 5$, $8 + 15 = 23$, $2 + 12 = 14$.

$$\mathbf{D}^2 = \begin{bmatrix} 2 + 2 - 4 & 2 + 41 - 18 & 2 + 17 - 10 \\ 13 + 2 - 10 & 13 + 41 - 46 & 13 + 17 - 28 \end{bmatrix}
= \begin{bmatrix} 0 & 25 & 9 \\ 5 & 8 & 2 \end{bmatrix}, \quad
\mathbf{D} = \begin{bmatrix} 0 & 5 & 3 \\ \sqrt5 \approx 2.2361 & \sqrt8 \approx 2.8284 & \sqrt2 \approx 1.4142 \end{bmatrix}.$$

Sanity check: $\mathbf{a}_1 = \mathbf{b}_1$, so $D_{11} = 0$. Code check:

```text
== exercise 1 check ==
(4, 1) op (3,) -> (4, 3)
(2, 3) op (2,) -> error
(5, 1, 4) op (3, 1) -> (5, 3, 4)
(6, 3) op (6,) -> error
(6, 3) op (6, 1) -> (6, 3)
|a|^2: [2.0, 13.0]  |b|^2: [2.0, 41.0, 17.0]
A @ B.T: [[2.0, 9.0, 5.0], [5.0, 23.0, 14.0]]
D^2: [[0.0, 25.0, 9.0], [5.0, 8.0, 2.0]]
D:   [[0.0, 5.0, 3.0], [2.2361, 2.8284, 1.4142]]
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a)–(e) correct, 1 pt each | 5 |
| Reasons given for (b) and (d) failing (which axis, which sizes) | 1 |
| Squared lengths of rows of A and B correct | 2 |
| $\mathbf{A}\mathbf{B}^\top$ correct, shape `(2, 3)` | 2 |
| Final $\mathbf{D}$ correct, with the square roots taken | 2 |

**Likely mistakes:**

- Aligning shapes from the **left**: then `(4, 1) + (3,)` looks like "4 vs 3, error". Always align
  from the right.
- In (c), forgetting that the result has the larger number of axes (3), not 2.
- Writing $\mathbf{A}\mathbf{B}^\top$ as `(3, 2)` by multiplying in the wrong order; the shape rule
  `(2, 2) @ (2, 3) → (2, 3)` is the check.
- Stopping at $\mathbf{D}^2$ and forgetting the square root.

## Exercise 2 (code)

**Solution:** [`a1_2_numpy_arrays_ex2.py`](a1_2_numpy_arrays_ex2.py). The core:

```python
def row_normalize(X: FloatArray) -> FloatArray:
    """Scale each row of X to unit L2 length; all-zero rows stay zero. X (n, d) -> (n, d)."""
    lengths = np.sqrt((X**2).sum(axis=1, keepdims=True))  # (n, 1)
    safe = np.where(lengths == 0.0, 1.0, lengths)  # (n, 1): avoid 0 / 0 -> nan
    return X / safe  # (n, d) / (n, 1) broadcasts over columns
```

Two details carry the marks. `keepdims=True` (or `[:, None]`) makes the lengths `(n, 1)`, so the
division broadcasts across columns; without it, `(n, d) / (n,)` fails (or, if $n = d$, divides the
wrong way silently). And replacing zero lengths with 1 makes `0 / 1 = 0`, so zero rows stay zero
instead of `0 / 0 = nan`. Real output:

```text
== exercise 2: row_normalize ==
[[0.6, 0.8], [0.0, 0.0], [0.7071, -0.7071]]
matches loop version: OK
matches sklearn normalize: OK
non-zero rows have length 1, zero row stays zero: True
~ vectorized 3.15 ms, loop 104.3 ms, speed-up 33x
```

The speed-up varies by machine (expect roughly 20–60×).

**Rubric (15 pts):**

| Criterion | Points |
|---|---|
| Correct result with no Python loop in `row_normalize` | 4 |
| Zero rows handled without `nan` (and without a warning) | 3 |
| Shapes right: lengths `(n, 1)`, output `(n, d)`; shape asserted | 2 |
| Checked against a loop version **and** `sklearn.preprocessing.normalize` with `assert_allclose` | 3 |
| Asserts unit length of non-zero rows | 1 |
| Timing done fairly (same input, both versions) and reported | 1 |
| Type hints, docstring with shapes | 1 |

**Likely mistakes:**

- `X / np.linalg.norm(X, axis=1)`: shape `(n,)` against `(n, d)` fails for $n \ne d$.
- `X / np.linalg.norm(X)` with no axis: divides everything by the length of the whole matrix.
- Adding a small epsilon (`X / (lengths + 1e-12)`): avoids `nan`, but every non-zero row is now
  very slightly shorter than 1; acceptable in production, but it does not satisfy "length 1" at
  `rtol=1e-12`. Half marks on the zero-row criterion.
- Using `np.errstate` to hide the divide-by-zero warning and then `np.nan_to_num`: works but hides
  the reason; prefer handling the zero case explicitly.
- Timing the first call only, which includes one-off overheads. Minor.

## Exercise 3 (think)

**Strong answer:** `y_hat - y` broadcasts `(n, 1)` against `(n,)` (padded to `(1, n)`) into an
`(n, n)` matrix whose entry `[i, j]` is $\hat{y}_i - y_j$: every prediction minus *every* target.
`np.mean` then averages $n^2$ squared differences instead of $n$. That loss is minimized when every
$\hat{y}_i$ equals the mean of all targets, so training pushes the model toward predicting a constant,
which is why it "converges" and is poor. Even a perfect model gets a large loss: in the demo, with
$\hat{y} = y = (1, 2, 3, 4)$ the buggy MSE is 2.5 instead of 0. No error is raised because the shapes
are legal under broadcasting; NumPy cannot know you meant elementwise. To make this fail loudly: assert
`y_hat.shape == y.shape` inside the loss function (or `squeeze` explicitly with a check), make the
model's output shape part of its contract and unit-test it, and add a test that a perfect prediction
gives loss 0. Libraries help too: PyTorch's `MSELoss` warns when the input and target sizes differ
(C2.1), and scikit-learn's metrics check shapes, which is one reason to use them.

**Weak answer:** "The model is too simple or needs more training; try a bigger learning rate." This is
weak because it diagnoses the model instead of the measurement, never inspects the shapes, and would
lead to hours of tuning a model whose loss function is wrong.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| States the `(n, n)` result and what entry `[i, j]` contains | 3 |
| Explains why the model drifts to a constant (or why a perfect model has non-zero loss) | 2 |
| Explains why no error: broadcasting rules allow it | 2 |
| Concrete prevention: shape asserts in the loss, tests (perfect prediction → 0), library losses | 3 |

## Scoring

Total 37 pts. Pass ≥ 70% (26 pts). Topic ids to revisit for each lost criterion: exercise 1 →
A1.2 "Broadcasting" and "The norm trick"; exercise 2 → A1.2 "Axes" and `keepdims`; exercise 3 →
A1.2 "Failure modes", and later B1.5 (the MSE loss in linear regression).
