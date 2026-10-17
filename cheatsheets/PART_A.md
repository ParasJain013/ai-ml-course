# Part A cheat sheet: Foundations

Two pages for the whole of Part A. Details, more code and more pitfalls are in the weekly sheets:
[A1 Python scientific stack](w01_a1.md) · [A2 Linear algebra I](w02_a2.md) ·
[A3 Linear algebra II](w03_a3.md) · [A4 Calculus & optimization](w04_a4.md) ·
[A5 Probability](w05_a5.md) · [A6 Statistics & information theory](w06_a6.md).

## Formulas
| Name | Formula | Shapes |
|---|---|---|
| Dot, norm, cosine | $\mathbf{a}\cdot\mathbf{b} = \sum_j a_jb_j$; $\lVert\mathbf{a}\rVert = \sqrt{\mathbf{a}\cdot\mathbf{a}}$; $\cos = \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\lVert\mathbf{b}\rVert}$ | `(d,)` → scalar |
| Pairwise distance | $D_{ij}^2 = \lVert\mathbf{a}_i\rVert^2 + \lVert\mathbf{b}_j\rVert^2 - 2\,\mathbf{a}_i\cdot\mathbf{b}_j$ | `(n, m)` |
| Standardize | $z = (x - \mu)/\sigma$ per column, statistics from training data | `(n, d)` |
| Matrix product | $(\mathbf{A}\mathbf{B})_{ij} = \sum_p A_{ip}B_{pj}$; "do $\mathbf{B}$, then $\mathbf{A}$"; $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top\mathbf{A}^\top$ | `(n, d) @ (d, k)` |
| Apply to rows | $\mathbf{X}\mathbf{A}^\top$ or $\mathbf{X}\mathbf{W}$ | `(n, d)` → `(n, m)` |
| Rank, solutions | $\mathbf{A}\mathbf{x} = \mathbf{b}$: none / one / $\infty$ (null space dim $d - r$) | |
| Least squares | $\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}$; residual ⟂ every column; $\mathbf{w} = \mathbf{X}^+\mathbf{y}$ | `(d,)` |
| Eigen | $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$; $\sum\lambda = \operatorname{tr}$, $\prod\lambda = \det$; symmetric $\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ | |
| SVD | $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$; best rank $k$ keeps top $k$, error $\sum_{j>k}\sigma_j^2$ | |
| Covariance | $\mathbf{S} = \frac{1}{n-1}\mathbf{X}_c^\top\mathbf{X}_c$, PSD; PCA axes = eigenvectors of $\mathbf{S}$ | `(d, d)` |
| Condition number | $\kappa = \sigma_{\max}/\sigma_{\min}$; lose $\log_{10}\kappa$ digits; $\kappa(\mathbf{X}^\top\mathbf{X}) = \kappa^2$ | |
| Derivative check | central difference $\frac{f(x+h) - f(x-h)}{2h}$, $h \approx 10^{-6}$, float64 | |
| Gradient patterns | $\nabla\lVert\mathbf{A}\mathbf{x} - \mathbf{b}\rVert^2 = 2\mathbf{A}^\top(\mathbf{A}\mathbf{x} - \mathbf{b})$; MSE $\frac{2}{n}\mathbf{X}^\top\mathbf{r}$ | same as $\mathbf{w}$ |
| Backprop, linear layer | $\mathbf{Z} = \mathbf{X}\mathbf{W}$: $\bar{\mathbf{W}} = \mathbf{X}^\top\bar{\mathbf{Z}}$, $\bar{\mathbf{X}} = \bar{\mathbf{Z}}\mathbf{W}^\top$; sigmoid + log loss: $p - y$ | |
| Gradient descent | $\boldsymbol\theta \leftarrow \boldsymbol\theta - \eta\nabla\mathcal{L}$; stable iff $\eta < 2/\lambda_{\max}(\mathbf{H})$; steps $\propto\kappa$ | |
| Lagrange / KKT | $\nabla f = \lambda\nabla h$; $\mu \ge 0$, $\mu g = 0$; L2 penalty = norm constraint | |
| Expectation, variance | $\mathbb{E}$ linear; $\operatorname{Var}[aX + b] = a^2\operatorname{Var}[X]$; $\operatorname{Var}[X \pm Y] = \operatorname{Var}X + \operatorname{Var}Y \pm 2\operatorname{Cov}$ | |
| Distributions | Bernoulli $p$; binomial $np$, $np(1-p)$; Poisson mean = var $= \lambda$; Gaussian 68/95/99.7 | |
| Sampling $\mathcal{N}(\boldsymbol\mu, \boldsymbol\Sigma)$ | $\boldsymbol\mu + \mathbf{L}\mathbf{z}$, $\mathbf{L}\mathbf{L}^\top = \boldsymbol\Sigma$ (rows: `Z @ L.T`) | `(n, d)` |
| Bayes | $P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}$; posterior odds $= \mathrm{LR} \times$ prior odds | |
| CLT, standard error | $\bar{X} \approx \mathcal{N}(\mu, \sigma^2/n)$; 95% bar $\pm 1.96\,s/\sqrt{n}$; 100× data = one more digit | |
| MLE / MAP | min $-\sum\log p(x_i \mid \boldsymbol\theta)$; Gaussian → MSE, Bernoulli → log loss; Gaussian prior → ridge, $\lambda = \sigma^2/\tau^2$ | |
| Robust summaries | median, IQR, $1.4826\,\mathrm{MAD}$; Tukey fences $Q_1 - 1.5\,\mathrm{IQR}$, $Q_3 + 1.5\,\mathrm{IQR}$ | |
| Intervals | t: $\bar x \pm t_{n-1}s/\sqrt n$; bootstrap percentiles; Wilson for proportions | |
| Tests | p-value $= P(\text{as extreme} \mid H_0)$; permutation on pooled labels; Welch t; $\chi^2 = \sum(O - E)^2/E$ | |
| Multiple tests | $1 - (1-\alpha)^m$; Bonferroni $\alpha/m$, Holm (FWER), BH $p_{(k)} \le \frac{k}{m}\alpha$ (FDR) | |
| A/B sample size | $n = (z_{1-\alpha/2} + z_{1-\beta})^2(p_Aq_A + p_Bq_B)/\delta^2$ per arm | |
| Information | $H(p) = -\sum p\log p$; $H(p, q) = H(p) + D_{\mathrm{KL}}(p\,\|\,q)$; $D_{\mathrm{KL}} \ge 0$, asymmetric | |
| Beta–Binomial | Beta($a$, $b$) + $k$/$n$ → Beta($a + k$, $b + n - k$); mean shrinks the MLE to $\frac{a}{a+b}$ | |

## Code
```python
rng = np.random.default_rng(SEED)                            # one seed, generators passed around
Z = (X - X.mean(axis=0)) / X.std(axis=0)                     # (n, d), per column
S = rows_q @ rows_m.T                                         # cosine on unit rows
x = np.linalg.solve(A, b); w, *_ = np.linalg.lstsq(X, y, rcond=None)   # never inv(A) @ b
vals, vecs = np.linalg.eigh(S)                                # ascending; vectors are COLUMNS
U, s, Vt = np.linalg.svd(A, full_matrices=False); A_k = (U[:, :k] * s[:k]) @ Vt[:k]
np.cov(X, rowvar=False); x.var(ddof=1)                        # (d, d); pass ddof explicitly
grad = 2 / n * X.T @ (X @ w - y); w -= lr * grad              # assert residual shape (n,)
mu + rng.standard_normal((n, d)) @ np.linalg.cholesky(Sigma).T
est, se = fx.mean(), fx.std(ddof=1) / np.sqrt(n)              # Monte Carlo with error bar
stats.bootstrap((x,), np.mean, method="BCa", rng=rng).confidence_interval
stats.ttest_ind(b, a, equal_var=False); multipletests(p, method="holm")[0]
stats.entropy(p, q); stats.beta(a + k, b + n - k).interval(0.95)
```

## Defaults and rules of thumb
- Data is `(n, d)` with rows as examples; write the shape in a comment on every line that changes it.
- Floats: `assert_allclose`, never `==`; float64 for checks; standardize features before fitting.
- `solve`/`lstsq`/QR/Cholesky instead of inverses or normal equations on ill-conditioned data.
- Check every hand-written gradient numerically; tune learning rates on a log grid.
- Reason with counts of imaginary people for conditional probabilities; quote rates at the production base rate.
- Every reported number gets an interval and a sample size; every p-value an effect size.
- Plot before summarizing; median and high percentiles for skewed data.
- Fix the metric, sample size and analysis before the experiment; one look or a calibrated sequential plan.

## Pitfalls
- Silent broadcasting: `(n, 1) - (n,)` → `(n, n)`; `Z @ L` instead of `Z @ L.T`; a missing `.T` on square matrices.
- `np.std` vs pandas `.std()` (ddof 0 vs 1); `np.cov(X)` without `rowvar=False` is `(n, n)`.
- `vecs[0]` is not an eigenvector; uncentered data in PCA; one-hot + intercept → singular $\mathbf{X}^\top\mathbf{X}$.
- Learning rate above $2/\lambda_{\max}$ diverges; local minima and saddles satisfy $\nabla f = 0$ too.
- Zero correlation read as independence; $P(+ \mid D)$ read as $P(D \mid +)$; "CLT makes the data Gaussian".
- `log(0)` in a loss; $\log(\text{softmax})$ instead of log-softmax; mixing bits and nats.
- Bootstrapping rows of grouped data; permuting within groups; peeking; 20 metrics with no correction.
- Simpson's paradox from a changed mix; a prior chosen after seeing the data.

## Interview one-liners
- **Vectorization:** whole-array operations run in compiled code, 10–100× faster than Python loops.
- **Matrix:** a linear map; its columns are where the basis vectors go; the product is composition.
- **Rank:** the number of independent directions a map can reach.
- **SVD:** rotate, scale, rotate; truncating it gives the best low-rank approximation.
- **Condition number:** how much the problem amplifies input error.
- **Gradient:** steepest increase, same shape as the parameters; backprop is the reverse-mode chain rule.
- **Convexity:** every local minimum is global.
- **Bayes' rule:** prior × likelihood, normalized; base rates dominate rare events.
- **MLE:** parameters that make the data most probable; common losses are negative log-likelihoods.
- **Regularization:** a prior on the weights (L2 = Gaussian) or, equivalently, a constraint on their size.
- **Confidence interval:** a procedure that covers the truth 95% of the time over repeated samples.
- **p-value:** how surprising the data is if nothing changed; never the probability that nothing changed.
- **Cross-entropy:** the classification loss; KL to the labels plus a constant.
- **Posterior:** what you believe after the data; with little data it leans on the prior.
