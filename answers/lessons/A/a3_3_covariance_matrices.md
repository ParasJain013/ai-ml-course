# Answer key: A3.3 Covariance matrices

Lesson: [`lessons/A/a3_3_covariance_matrices.md`](../../../lessons/A/a3_3_covariance_matrices.md)

## Exercise 1 (by hand)

**Solution (a).** $\mathbf{X} = \begin{pmatrix}1&2\\2&1\\3&6\end{pmatrix}$. Means $\bar{x}_1 = 6/3 = 2$,
$\bar{x}_2 = 9/3 = 3$. Centered: $\mathbf{X}_c = \begin{pmatrix}-1&-1\\0&-2\\1&3\end{pmatrix}$.
$\mathbf{X}_c^\top\mathbf{X}_c = \begin{pmatrix}1 + 0 + 1 & 1 + 0 + 3\\ 1 + 0 + 3 & 1 + 4 + 9\end{pmatrix} = \begin{pmatrix}2&4\\4&14\end{pmatrix}$.
With $n - 1 = 2$: $\mathbf{S} = \begin{pmatrix}1&2\\2&7\end{pmatrix}$. With $n = 3$: $\begin{pmatrix}0.6667&1.3333\\1.3333&4.6667\end{pmatrix}$.

**Solution (b).** $R_{12} = 2/(\sqrt1\cdot\sqrt7) = 2/\sqrt7 \approx 0.7559$.

**Solution (c).** $\mathbf{X}_c\mathbf{w} = \frac{1}{\sqrt2}(-1 - 1,\ 0 - 2,\ 1 + 3) = \frac{1}{\sqrt2}(-2, -2, 4)$, i.e.
$(-1.4142, -1.4142, 2.8284)$. Sum of squares $(4 + 4 + 16)/2 = 12$; divided by $n - 1 = 2$: **6**.
Quadratic form: $\mathbf{w}^\top\mathbf{S}\mathbf{w} = \frac12(1, 1)\begin{pmatrix}1&2\\2&7\end{pmatrix}(1, 1)^\top = \frac12(1 + 2 + 2 + 7) = 6$ ✓.

**Solution (d).** $\operatorname{tr} = 8$, $\det = 7 - 4 = 3$, $\lambda^2 - 8\lambda + 3 = 0$, $\lambda = 4 \pm\sqrt{13}$,
i.e. $0.3944$ and $7.6056$. Both positive, so $\mathbf{S}$ is positive definite (no combination of the two
features is constant on these three rows).

```text
== exercise 1 check ==
means = [2.0, 3.0]  centered = [[-1.0, -1.0], [0.0, -2.0], [1.0, 3.0]]
covariance (n - 1) = [[1.0, 2.0], [2.0, 7.0]]   (n) = [[0.6667, 1.3333], [1.3333, 4.6667]]
correlation = 0.7559
projections onto w = [-1.4142, -1.4142, 2.8284]  variance = 6.0  w^T S w = 6.0
eigenvalues of S = [0.3944, 7.6056]  (4 -/+ sqrt 13 = [0.3944, 7.6056] )
```

**Rubric (13 pts):**

| Criterion | Points |
|---|---|
| (a) Means and centered matrix | 2 |
| (a) $\mathbf{S}$ with $n - 1$ and with $n$ | 3 |
| (b) Correlation $2/\sqrt7$ | 1 |
| (c) Projection route and quadratic-form route both give 6 | 4 |
| (d) Eigenvalues $4 \pm \sqrt{13}$, with the positive-definite conclusion and why | 3 |

**Likely mistakes:**

- Forgetting to center: $\mathbf{X}^\top\mathbf{X}/2 = \begin{pmatrix}7&11\\11&20.5\end{pmatrix}$ is not a covariance.
- Computing $\mathbf{X}_c\mathbf{X}_c^\top$ `(3, 3)` instead of $\mathbf{X}_c^\top\mathbf{X}_c$ `(2, 2)`; the shape tells you.
- In (c), dividing the sum of squares by $n$ while $\mathbf{S}$ used $n - 1$, so the two routes disagree
  (4 versus 6). Both routes must use the same divisor.
- Using $\mathbf{w} = (1, 1)$ without normalizing, which gives 24 and 24: consistent, but it is the variance
  of $x_1 + x_2$, not of the data along a unit direction.

## Exercise 2 (code)

**Solution:** [`a3_3_covariance_matrices_ex2.py`](a3_3_covariance_matrices_ex2.py).

*On paper.* Expand the centered outer product and use $\sum_i\mathbf{x}_i = n\bar{\mathbf{x}}$:

$$\sum_i(\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^\top = \sum_i\mathbf{x}_i\mathbf{x}_i^\top - \bar{\mathbf{x}}\Big(\sum_i\mathbf{x}_i\Big)^\top - \Big(\sum_i\mathbf{x}_i\Big)\bar{\mathbf{x}}^\top + n\bar{\mathbf{x}}\bar{\mathbf{x}}^\top = \sum_i\mathbf{x}_i\mathbf{x}_i^\top - n\bar{\mathbf{x}}\bar{\mathbf{x}}^\top.$$

The left side is $\mathbf{X}_c^\top\mathbf{X}_c$, so dividing by $n - 1$ gives $\mathbf{S}$. In exact arithmetic the
one-pass formula is correct.

*The failure.* With an offset of $10^8$, each $x^2$ is about $10^{16}$ and the sum over $10^5$ rows about
$10^{21}$. float64 keeps about 16 significant digits, so each of the two big terms carries an absolute
rounding error around $10^{5}$, while the quantity we want, their difference ($n - 1$ times the
covariance), is also about $10^{5}$. Subtracting two nearly equal large numbers cancels the correct
digits and leaves the rounding error (**catastrophic cancellation**, [A3.4](../../../lessons/A/a3_4_numerical_stability.md)).
The result is not even a valid covariance: a negative variance and a negative eigenvalue.

*The fix.* Covariance does not change when every row is shifted by the same vector, so subtract a shift
$\mathbf{K}$ close to the data (the first row seen) from every chunk before accumulating. The sums then
stay at the scale of the spread, not of the offset. Welford's or Chan's update formulas, which carry a
running mean and merge chunk statistics, are equally good answers.

```python
def cov_streaming_shifted(blocks):
    n, sx, sxx, K = 0, None, None, None
    for B in blocks:
        if K is None:
            K = B[0].copy()          # (d,), close to the mean in scale
        Bk = B - K
        n += Bk.shape[0]
        sx = Bk.sum(axis=0) if sx is None else sx + Bk.sum(axis=0)
        sxx = Bk.T @ Bk if sxx is None else sxx + Bk.T @ Bk
    mean_k = sx / n
    return (sxx - n * np.outer(mean_k, mean_k)) / (n - 1)
```

```text
== exercise 2: covariance in one pass over chunks ==
offset    0e+00: max |error| naive = 4.4e-16   shifted = 2.2e-16
offset    1e+04: max |error| naive = 3.6e-07   shifted = 6.7e-16
offset    1e+08: max |error| naive = 3.6e+01   shifted = 1.4e-12
   naive S = [[3.9322, -14.4181, 1.3107], [-14.4181, -34.0791, -18.3503], [1.3107, -18.3503, 0.0]]
   true  S = [[1.0057, 0.6027, -0.0007], [0.6027, 1.9913, -0.4999], [-0.0007, -0.4999, 0.5031]]   naive eigenvalues = [-45.4073, 0.8757, 14.3847]
shifted one-pass covariance matches np.cov at every offset: OK
```

(The tiny error magnitudes vary slightly between machines; the pattern does not.) Offsets like these
are realistic: Unix timestamps are about $1.7 \times 10^9$, and prices stored in paise or cents reach
$10^8$ quickly.

**Rubric (15 pts):**

| Criterion | Points |
|---|---|
| Algebra showing the one-pass formula equals $\mathbf{X}_c^\top\mathbf{X}_c/(n-1)$ | 3 |
| `cov_streaming_naive` makes one pass, keeps only $n$, $\sum\mathbf{x}$, $\sum\mathbf{x}\mathbf{x}^\top$; shapes stated | 3 |
| Tested at three offsets against `np.cov(X, rowvar=False)` | 2 |
| Failure explained as cancellation of two large nearly equal sums (with the magnitude argument) | 3 |
| A one-pass fix (shift, Welford or Chan) that matches `np.cov` at offset $10^8$ | 4 |

**Likely mistakes:**

- Shifting by the *global* mean: that needs a first pass to compute it, which the exercise rules out.
  Any value near the data (the first row) is enough.
- Comparing against `np.cov(X)` without `rowvar=False`, getting a `(100000, 100000)` matrix or a memory
  error.
- Blaming "float precision in general" without saying *why* this formula is fragile: the two-pass
  centered formula on the same data is accurate; the subtraction of two huge sums is the problem.
- Using float32 accumulators ("to save memory"), which fails already at an offset of about $10^3$.

## Exercise 3 (think)

**Strong answer:** Covariance is in the product of the two features' units, so its size mostly reflects
the units, not the strength of the relationship. Price in rupees (values around $10^7$) times distance in
metres (around $10^3$) produces covariances many orders of magnitude larger than, say, age in years times
number of rooms, even if the latter pair is more tightly related. Change price to lakhs and the
"strongest pair" changes. Show the **correlation matrix** instead (dimensionless, between −1 and 1), as a
heatmap with values and a fixed −1 to 1 colour scale, alongside scatter plots of the top pairs. Also
consider a rank (Spearman) correlation, since prices are skewed and a few luxury listings can dominate
Pearson's.

Two things even the correlation matrix cannot tell them (any two): whether the relationship is causal
(proximity to the metro and price may both be driven by neighbourhood); non-linear relationships (price
may fall with distance and then flatten, or relate to age in a U-shape, with correlation near 0); effects
that only appear after controlling for other features (area and rooms are correlated with each other,
so their separate effects on price are confounded); whether it holds on new data or other segments
(correlations computed on all cities can reverse within a city: Simpson's paradox); and how robust it is
(one data-entry error can create or destroy a correlation; with small $n$ it is noisy).

**Weak answer:** "Covariance shows how related variables are, so yes, the largest covariance is the
strongest relationship; maybe also check the p-value." Weak because it ignores units, which is the
whole error, and offers nothing beyond a significance test.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Identifies units/scale as the reason covariance magnitudes are not comparable, with an example | 4 |
| Proposes the correlation matrix (and/or standardization) plus a visual check | 3 |
| Two valid limitations (causation, non-linearity, confounding, Simpson's paradox, outliers or small $n$) | 3 |

## Scoring

Total 38 pts. Pass ≥ 70% (27 pts). Topic ids to revisit for each lost criterion: exercise 1 → A3.3
"The math" and "Worked example", A3.1 (eigenvalues); exercise 2 → A3.3 "The math" (one-pass algebra) and
A3.4 (floating point and cancellation); exercise 3 → A3.3 "Intuition: units matter" and "In practice",
later A6.1 (correlation versus causation).
