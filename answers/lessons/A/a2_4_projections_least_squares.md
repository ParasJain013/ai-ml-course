# Answer key: A2.4 Projections and least squares

Lesson: [`lessons/A/a2_4_projections_least_squares.md`](../../../lessons/A/a2_4_projections_least_squares.md)

## Exercise 1 (by hand)

**Solution (a).** $\mathbf{a} = (1, 1, 0)$, $\mathbf{b} = (3, 1, 2)$. $\mathbf{a}^\top\mathbf{b} = 3 + 1 + 0 = 4$,
$\mathbf{a}^\top\mathbf{a} = 2$, so $c = 2$ and $\mathbf{p} = 2\mathbf{a} = (2, 2, 0)$.
$\mathbf{r} = \mathbf{b} - \mathbf{p} = (1, -1, 2)$, and $\mathbf{r}\cdot\mathbf{a} = 1 - 1 + 0 = 0$ ✓.
Pythagoras: $\lVert\mathbf{b}\rVert^2 = 9 + 1 + 4 = 14$, $\lVert\mathbf{p}\rVert^2 = 4 + 4 = 8$,
$\lVert\mathbf{r}\rVert^2 = 1 + 1 + 4 = 6$, and $8 + 6 = 14$ ✓.

**Solution (b).** Points $(1, 2), (2, 3), (3, 5)$.

- $\mathbf{X} = \begin{pmatrix}1&1\\1&2\\1&3\end{pmatrix}$ `(3, 2)`, $\mathbf{y} = (2, 3, 5)$.
- $\mathbf{X}^\top\mathbf{X} = \begin{pmatrix}3 & 6\\ 6 & 14\end{pmatrix}$ ($n = 3$, $\sum x = 6$, $\sum x^2 = 1 + 4 + 9 = 14$).
- $\mathbf{X}^\top\mathbf{y} = (2 + 3 + 5,\ 2 + 6 + 15) = (10, 23)$.
- $\det = 42 - 36 = 6$, so $\mathbf{w} = \frac16\begin{pmatrix}14 & -6\\ -6 & 3\end{pmatrix}(10, 23) = \frac16(140 - 138,\ -60 + 69) = \frac16(2, 9) = (\frac13, 1.5)$.
- Fitted values: $\frac13 + 1.5 = 1.8333$, $3.3333$, $4.8333$; residuals $(\frac16, -\frac13, \frac16) \approx (0.1667, -0.3333, 0.1667)$.
- Orthogonality: $\sum r_i = \frac16 - \frac13 + \frac16 = 0$ ✓; $\sum x_i r_i = \frac16 - \frac23 + \frac12 = 0$ ✓.
- Means: $\bar{x} = 2$, $\bar{y} = \frac{10}{3}$, and $\frac13 + 1.5 \cdot 2 = \frac{10}{3}$ ✓.

```text
== exercise 1 check ==
p = [2.0, 2.0, 0.0]  r = [1.0, -1.0, 2.0]  r.a = 0.0  |b|^2 = 14 = |p|^2 + |r|^2 = 8 + 6
X^T X = [[3.0, 6.0], [6.0, 14.0]]  X^T y = [10.0, 23.0]  w = [0.3333, 1.5]
residuals = [0.1667, -0.3333, 0.1667]  sum = 0.0  sum x*r = 0.0
line at mean x: 3.3333  mean y: 3.3333
```

**Rubric (13 pts):**

| Criterion | Points |
|---|---|
| (a) $\mathbf{p}$ and $\mathbf{r}$ correct, orthogonality checked | 3 |
| (a) Pythagoras verified with numbers | 1 |
| (b) $\mathbf{X}$ with the ones column, $\mathbf{X}^\top\mathbf{X}$, $\mathbf{X}^\top\mathbf{y}$ correct | 3 |
| (b) $\mathbf{w} = (\frac13, 1.5)$ | 2 |
| (b) residuals, both orthogonality checks, and the line through the means | 4 |

**Likely mistakes:**

- Leaving out the column of ones, which fits a line through the origin.
- $\mathbf{X}^\top\mathbf{X}$ with $\sum x$ where $\sum x^2$ belongs, or $n$ forgotten in the top-left.
- Projecting $\mathbf{a}$ onto $\mathbf{b}$ instead of $\mathbf{b}$ onto $\mathbf{a}$ (the formula is not
  symmetric: the denominator is the length of the direction you project *onto*).
- Rounding $\frac13$ to 0.33 early, so the orthogonality checks come out as 0.01 instead of 0.

## Exercise 2 (code)

**Solution:** [`a2_4_projections_least_squares_ex2.py`](a2_4_projections_least_squares_ex2.py). The
only new idea is the design matrix: the model is still *linear in the weights*, so a quadratic in $x$
is least squares with the columns $1, x, x^2$:

```python
def poly_design(x: FloatArray, degree: int) -> FloatArray:
    """Columns 1, x, x^2, ..., x^degree. x (n,) -> (n, degree + 1)."""
    return np.column_stack([x**k for k in range(degree + 1)])
```

```text
== exercise 2: quadratic least squares ==
X (40, 3)  w = [1.1307, -1.9557, 0.5273]  true = [1.0, -2.0, 0.5]
matches np.linalg.lstsq and np.polyfit (reversed): OK
X^T residual = [0.0, 0.0, 0.0]  orthogonal to all three columns: OK
P (40, 40)  idempotent and symmetric: OK   trace(P) = 3.0 (= number of columns)

== exercise 2: straight line vs quadratic residuals ==
line w = [3.445, -1.5245]  SSE line = 230.19   SSE quadratic = 45.60
line residuals still sum to 0.0 and are orthogonal to x: 0.0
line residuals at the smallest, middle and largest x: [2.2067, -1.8827, 5.168]
```

The point of the last part: the straight line's residuals still sum to zero and are orthogonal to $x$
(projection guarantees that), so a summary like "mean residual 0, no linear trend" looks clean. Yet
they are positive at both ends and negative in the middle: a U shape, the curvature the line cannot
represent. Only the residual plot shows it. (`trace(P) = 3` is a bonus fact: a projection matrix's
trace equals the dimension it projects onto. Not required.)

**Rubric (15 pts):**

| Criterion | Points |
|---|---|
| Data generated as specified with `SEED = 0` and `default_rng`; shapes stated | 1 |
| `(40, 3)` design matrix with columns $1, x, x^2$ | 2 |
| Normal equations solved from scratch (with `solve`, not `inv`) | 2 |
| Checked against `lstsq` and `polyfit`, with polyfit's reversed order handled | 3 |
| Orthogonality to all three columns asserted | 2 |
| Projection matrix idempotent and symmetric, asserted | 2 |
| Both residual plots, and the U-shape explained as missing curvature that summary statistics hide | 3 |

**Likely mistakes:**

- Comparing with `np.polyfit` without reversing: `polyfit` returns $(w_2, w_1, w_0)$.
- Building $\mathbf{P}$ with `np.linalg.inv` (acceptable at this size, but the habit is wrong) or
  comparing `P @ P == P` with `==`.
- Concluding from SSE alone that the quadratic is "better": with more columns the SSE can only go down
  on the training data. The residual *pattern* is the evidence; held-out error is the test
  ([B4.5](../../../lessons/B/b4_5_bias_variance.md)).

## Exercise 3 (think)

**Strong answer:** Least squares minimizes $\sum_i r_i^2$. A reading 1,000 times too large has a
residual roughly 1,000 times larger than normal, so its *squared* residual is about $10^6$ times
larger: one row contributes more to the loss than all 999 good rows together. The fit moves to shrink
that one residual. The intercept shifts toward the outlier (at the mean voltage, by roughly the
outlier's residual divided by $n$, which can still be huge), and the slope tilts toward it by an
amount that grows with how far its voltage is from the mean voltage (its **leverage**). A point at an
extreme voltage can swing the slope badly; a point at the mean voltage mostly moves the intercept. The
calibration is then wrong for every future reading, not just the bad one.

Detection before deploy: validate inputs (range checks against physical limits of the sensor; reject or
quarantine impossible readings); validate the fit (compare new $w_0, w_1$ with yesterday's and alert on a
relative change above a threshold; check the largest absolute residual against a robust scale such as
the median absolute residual, flagging anything beyond, say, 5 robust standard deviations); evaluate the
new calibration on a fixed reference set of trusted readings and block the deploy if its error rises.
Changes to the fit: use a robust method (Huber loss, which is quadratic for small residuals and linear
for large ones; RANSAC; or Theil–Sen), or remove points with large robust z-scores and refit, logging
what was removed. Least absolute deviations ($\sum\lvert r_i\rvert$) is another option.

**Weak answer:** "With 1,000 points, one bad value averages out, so the effect is negligible." Weak
because it confuses averaging with squared loss: a single extreme point is *not* diluted, it dominates.
It proposes no detection.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Explains why squaring lets one row dominate (with a magnitude argument) | 3 |
| Describes the effect on intercept and slope, including leverage | 2 |
| Automated detection before deployment (input validation, coefficient drift, residual check, reference set) | 3 |
| A concrete change to the fit (robust loss, RANSAC/Theil–Sen, outlier removal with logging) | 2 |

## Scoring

Total 38 pts. Pass ≥ 70% (27 pts). Topic ids to revisit for each lost criterion: exercise 1 → A2.4
"The math" and "Worked example"; exercise 2 → A2.4 "From scratch" and "Key plot" (residual plots);
exercise 3 → A2.4 "In practice", and later B1.5 and B4.2 (regression metrics and robust losses).
