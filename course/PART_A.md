# Part A. Foundations — detailed syllabus

Course weeks 1–6. Weekly detail for Part A of [`COURSE_OUTLINE.md`](COURSE_OUTLINE.md), which stays
the source of truth: the topic ids, Core/Depth labels, project P1 and checkpoint CP1 below are taken
from it unchanged. Written 2026-10-16.

**Goal:** rebuild the math every later part relies on, from intuition and always in code, and become
fluent with NumPy, pandas and matplotlib.

## At a glance

| Week | Course wk | Theme | Topics | Hours | Project | GPU |
|---|---|---|---|---|---|---|
| A1 | 1 | Python scientific stack | A1.1–A1.5 | 12 | — | no |
| A2 | 2 | Linear algebra I: matrices | A2.1–A2.4 | 12 | — | no |
| A3 | 3 | Linear algebra II: decompositions | A3.1–A3.4 | 12 | P1 M0 | no |
| A4 | 4 | Calculus & optimization | A4.1–A4.5 | 12.5 | P1 M1 | no |
| A5 | 5 | Probability | A5.1–A5.5 | 12 | P1 M2 | no |
| A6 | 6 | Statistics & information theory | A6.1–A6.5 | 13 | **P1 due** · CP1 | no |

About 74 hours in total. Each week's hours include about 1 hour for the weekly quiz and cheat sheet.
Topic hours cover reading the lesson and doing its lab and exercises.

**How the hours are counted.** *Reading* is working through the lessons, including the worked
examples by hand. *Coding* is the labs and the code exercises. *Project* is P1.

**If a week runs long,** drop its Depth topic first (A3.4, A4.5, A6.5) and come back to it later.
Skipping A4.5 means skipping B2.5 (SVMs) as well, because SVMs use Lagrange multipliers.

**Libraries this part needs** (Step 2 pins the versions): `numpy`, `pandas`, `matplotlib`, `scipy`,
`scikit-learn` (only for its bundled sample image in A3.2), `ipykernel` (to run `# %%` cells in VS
Code). `sqlite3` comes with Python.

**Hardware:** everything in Part A runs on the laptop CPU in seconds. No week needs a GPU. If your
local set-up ever breaks, every lab also runs unchanged on a free Colab or Kaggle CPU notebook.

---

## A1 · Python scientific stack (course week 1)

**Learning objectives**

- I can run any lab in this course with `uv run`, and step through its `# %%` cells in VS Code.
- I can predict the shape of a NumPy expression, including broadcasting, before running it.
- I can replace a Python loop over array elements with a vectorized NumPy expression and measure the speed-up.
- I can translate a SQL query with `WHERE`, `GROUP BY`, `JOIN` and `LEFT JOIN` into pandas and get the same rows.
- I can make line, scatter and histogram plots with labelled axes, and spot a misleading chart.
- I can compute a dot product, L2 norm and cosine similarity by hand and in code, and explain what cosine similarity measures.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| A1.1 | The lab workflow: `uv`, Jupyter vs `# %%` cells, seeds, how every lab is run | Core | 1 |
| A1.2 | NumPy arrays: shapes, indexing, broadcasting, vectorizing loops away | Core | 3 |
| A1.3 | pandas: DataFrames, selection, groupby, joins (mapped to SQL), missing values | Core | 3 |
| A1.4 | matplotlib: line, scatter, histogram, subplots, plots that don't mislead | Core | 2 |
| A1.5 | Vectors: addition, dot product, norms, cosine similarity; the geometry of embeddings | Core | 2 |

**Hours:** reading 4 · coding 7 · project 0 · quiz and cheat sheet 1 · **total 12**

**Labs**

- **A1.1:** a lab "health check": print Python and library versions, set every random seed, and
  show that two runs give identical random numbers. This is the template every later lab starts from.
- **A1.2:** a pairwise Euclidean distance matrix three ways: nested Python loops, broadcasting, and
  the `‖a‖² + ‖b‖² − 2a·b` trick. Time all three and check they agree. Also write column
  standardization (subtract the mean, divide by the standard deviation) and a broadcasting shape
  predictor that you test against NumPy's real result.
- **A1.3:** load a small two-table dataset into Python's built-in `sqlite3` *and* into pandas. Run
  five SQL queries (filter, `GROUP BY` with aggregates, inner join, `LEFT JOIN` with NULLs,
  `HAVING`) and write the pandas equivalent of each, asserting the results are equal. Then write a
  group-by mean with plain Python dictionaries and compare it with `groupby().mean()`.
- **A1.4:** histogram binning from scratch (bin edges and counts) checked against `np.histogram`.
  Draw one deliberately misleading chart (truncated y-axis, cherry-picked range) and its honest version side by side.
- **A1.5:** dot product, L1 and L2 norms and cosine similarity in pure Python, then in NumPy. Then a
  brute-force "nearest neighbour" search over a small set of hand-made word vectors, which is the
  same operation a vector database runs in Part I.

**Project milestone:** none.

**Prerequisites:** none (start of the course). Within the week, A1.5 builds on A1.2.

**Resources** (optional)

- uv documentation, "Getting started": https://docs.astral.sh/uv/ (VERIFY link)
- pandas, "Comparison with SQL": https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html (VERIFY link)
- NumPy, "the absolute basics for beginners": https://numpy.org/doc/stable/user/absolute_beginners.html (VERIFY link)

**GPU:** no.

**Used later in:** every lab (A1.1–A1.4); embeddings and semantic search in E1.3, I2.1 and I2.2 (A1.5).

---

## A2 · Linear algebra I: matrices (course week 2)

**Learning objectives**

- I can describe what a 2×2 matrix does to the plane (rotate, scale, shear, project) by looking at its columns.
- I can state the shape rule `(n, d) @ (d, k) → (n, k)` and use it to catch bugs before running code.
- I can multiply small matrices and invert a 2×2 matrix by hand, and verify both in NumPy.
- I can solve a linear system by Gaussian elimination and tell, from the rank, whether it has zero, one or infinitely many solutions.
- I can derive the normal equations `XᵀX w = Xᵀy` from the idea of projection, and fit a line with them.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| A2.1 | Matrices as linear maps: what multiplying by a matrix does to space; shape rules | Core | 3 |
| A2.2 | Matrix multiplication, transpose, identity and inverse, by hand then in NumPy | Core | 3 |
| A2.3 | Systems of linear equations, rank, and when a solution does not exist | Core | 2.5 |
| A2.4 | Projections and least squares: the normal equations, the first "model fit" | Core | 2.5 |

**Hours:** reading 4.5 · coding 6.5 · project 0 · quiz and cheat sheet 1 · **total 12**

**Labs**

- **A2.1:** apply rotation, scaling, shear and projection matrices to a grid of points and to a
  unit square, and plot before and after. Write a `check_shapes(*arrays)` helper that predicts the
  shape of a chain of matrix products, or explains why it fails.
- **A2.2:** matrix multiplication as a triple loop, then as rows-times-columns dot products,
  checked against `@`. Gauss–Jordan inversion for a 3×3 matrix, checked by `A @ A_inv ≈ I` and
  against `np.linalg.inv`.
- **A2.3:** Gaussian elimination with partial pivoting from scratch, plus a rank function based on
  row reduction. Check them against `np.linalg.solve` and `np.linalg.matrix_rank` on three systems:
  one with a single solution, one with no solution, and one with infinitely many.
- **A2.4:** project a vector onto a line and onto a plane, and check the leftover part is orthogonal.
  Fit a straight line to noisy data with the normal equations from scratch, compare with
  `np.linalg.lstsq`, and plot the fit and its residuals.

**Project milestone:** none.

**Prerequisites:** A1.2 (NumPy shapes), A1.5 (dot products and norms).

**Resources** (optional)

- 3Blue1Brown, "Essence of linear algebra", chapters 1–7: https://www.3blue1brown.com/topics/linear-algebra (VERIFY link)
- Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* (free PDF), chapters 2–3: https://mml-book.github.io/ (VERIFY link)
- MIT 18.06 Linear Algebra (Gilbert Strang), lectures 1–3 and 14–16: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/ (VERIFY link)

**GPU:** no.

**Used later in:** every model with a weight matrix; linear regression in B1.5 (A2.4); the shapes in
every PyTorch layer in Parts C–F (A2.1).

---

## A3 · Linear algebra II: decompositions (course week 3)

**Learning objectives**

- I can explain an eigenvector as a direction the matrix only stretches, and find the dominant one with power iteration.
- I can describe the SVD as "rotate, scale, rotate", and use a truncated SVD to compress a matrix while measuring the error.
- I can compute a covariance matrix from data, explain each entry, and say why it is symmetric and positive semi-definite.
- I can explain why `np.linalg.solve` is preferred over computing `inv(A) @ b`, using the condition number. *(Depth)*

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| A3.1 | Eigenvalues and eigenvectors: directions a matrix only stretches | Core | 3 |
| A3.2 | Singular value decomposition: any matrix as rotate, scale, rotate; the engine of PCA | Core | 3.5 |
| A3.3 | Covariance matrices: what they encode, why they are symmetric and positive semi-definite | Core | 2 |
| A3.4 | Numerical stability: condition numbers, QR and Cholesky, why you never call `inv()` | Depth | 1.5 |

**Hours:** reading 4 · coding 6 · project 1 · quiz and cheat sheet 1 · **total 12**

**Labs**

- **A3.1:** power iteration from scratch to find the largest eigenvalue and its eigenvector, checked
  against `np.linalg.eig`. Plot what a 2×2 matrix does to the unit circle and draw its eigenvectors on top.
- **A3.2:** SVD from scratch by running power iteration on `AᵀA` and deflating, checked against
  `np.linalg.svd` (allowing for sign flips). Compress the grayscale version of scikit-learn's bundled
  sample image at ranks 5, 20 and 50, and plot the reconstruction error against the rank.
- **A3.3:** a covariance matrix from scratch, checked against `np.cov` (watch the `ddof` argument).
  Plot 2-D correlated data with the covariance matrix's eigenvectors drawn as arrows, scaled by the
  square roots of the eigenvalues.
- **A3.4 (Depth):** QR by Gram–Schmidt and a Cholesky factorization from scratch, checked against
  NumPy. Solve an ill-conditioned system with `inv`, `solve` and `lstsq`, and compare the errors
  against the condition number.

**Project milestone: P1 M0, pick the dataset (1 h).** Choose a public tabular dataset with at least
1,000 rows, a mix of numeric and categorical columns, and some missing values. Record its source and
licence, and write 3–5 questions you want it to answer. Candidates (VERIFY availability): Palmer
Penguins, UCI Adult (Census Income) via OpenML, or an export from a city open-data portal.

**Prerequisites:** A2.1–A2.4.

**Resources** (optional)

- 3Blue1Brown, "Essence of linear algebra", chapter 14 (eigenvectors and eigenvalues): https://www.3blue1brown.com/topics/linear-algebra (VERIFY link)
- *Mathematics for Machine Learning*, chapter 4 (matrix decompositions): https://mml-book.github.io/ (VERIFY link)
- fast.ai, "Computational Linear Algebra" (for A3.4): https://github.com/fastai/numerical-linear-algebra (VERIFY link)

**GPU:** no.

**Used later in:** PCA in B5.4 (A3.2, A3.3); the multivariate Gaussian in A5.2 (A3.3); LoRA's
low-rank updates in J1.3 (A3.2); the matrix-factorization recommender in B6.1 (A3.2).

---

## A4 · Calculus & optimization (course week 4)

**Learning objectives**

- I can explain a derivative as a local slope and check any derivative I derive with a finite difference.
- I can write the gradient of a scalar function of a vector, state its shape, and derive the gradient of mean squared error with respect to the weights.
- I can draw a small expression as a computational graph and push gradients backward through it with the chain rule.
- I can fit linear regression by gradient descent, show it reaches the same weights as the normal equations, and diagnose a learning rate that is too small or too large.
- I can solve a small constrained problem with a Lagrange multiplier and say what makes a function convex. *(Depth)*

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| A4.1 | Derivatives and partial derivatives: slope intuition, checked with finite differences | Core | 1.5 |
| A4.2 | Gradients, Jacobians and the shapes of matrix derivatives used in ML | Core | 2 |
| A4.3 | The chain rule as a computational graph: the seed of backpropagation | Core | 2 |
| A4.4 | Gradient descent: learning rate, convergence, fitting a line by GD and matching A2.4 | Core | 2.5 |
| A4.5 | Convexity, constrained optimization and Lagrange multipliers | Depth | 1.5 |

**Hours:** reading 3.5 · coding 6 · project 2 · quiz and cheat sheet 1 · **total 12.5**

**Labs**

- **A4.1:** forward-difference and central-difference derivatives from scratch. Plot the error
  against the step size on a log scale and explain the U-shape (truncation error versus rounding error).
- **A4.2:** a general numerical gradient checker for any `f: ℝⁿ → ℝ`, and a numerical Jacobian for
  `f: ℝⁿ → ℝᵐ`. Use them to verify your hand-derived gradients of MSE and of `‖Ax − b‖²`.
- **A4.3:** forward and backward passes by hand for a small expression graph, for example
  `f = (x·y + z)²` and then a one-neuron model with a sigmoid. Write each node's local derivative
  explicitly and check the result with the A4.2 gradient checker. (You build a general autograd
  engine later, in C1.3.)
- **A4.4:** batch gradient descent and mini-batch SGD for linear regression from scratch. Match the
  A2.4 normal-equation weights to 4 decimal places, and plot loss curves for a learning rate that is
  too small, one that works and one that diverges.
- **A4.5 (Depth):** solve a small constrained problem by hand with a Lagrange multiplier, then check
  it with `scipy.optimize.minimize`. Plot a convex and a non-convex function and show where gradient
  descent ends up from different starting points.

**Project milestone: P1 M1, load and clean (2 h).** A script that loads the raw data into pandas
and writes a cleaned copy into `data/`. A data dictionary (column, type, meaning, % missing). A
cleaning log that justifies every decision: dropped rows, imputation, type fixes, outliers kept or
removed. Running the script twice must give the same output.

**Prerequisites:** A1.2, A2.1 (shapes), A2.4 (normal equations, for the GD comparison).

**Resources** (optional)

- Parr & Howard, "The Matrix Calculus You Need For Deep Learning": https://explained.ai/matrix-calculus/ (VERIFY link)
- 3Blue1Brown, "Essence of calculus", chapters 1–4: https://www.3blue1brown.com/topics/calculus (VERIFY link)
- *Mathematics for Machine Learning*, chapters 5 (vector calculus) and 7 (continuous optimization): https://mml-book.github.io/ (VERIFY link)

**GPU:** no.

**Used later in:** every model trained by gradient descent from B1.5 on; backpropagation in C1.2 and
the autograd engine in C1.3 (A4.3); optimizers in C3.1 (A4.4); SVMs in B2.5 (A4.5).

---

## A5 · Probability (course week 5)

**Learning objectives**

- I can estimate an expectation, variance and covariance by simulation and confirm them against the formulas.
- I can name the right distribution for a situation (counts, yes/no, categories, measurements) and sample from it using only a uniform random number generator.
- I can apply Bayes' theorem to a base-rate problem and explain why a positive result from an accurate test can still mean the condition is unlikely.
- I can explain the law of large numbers and the central limit theorem, and use Monte Carlo to estimate a quantity with an error bar.
- I can show that minimizing squared error is Gaussian maximum likelihood, that log-loss is Bernoulli maximum likelihood, and that L2 regularization is a Gaussian prior.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| A5.1 | Random variables, expectation, variance and covariance, by simulation first | Core | 2 |
| A5.2 | Common distributions: Bernoulli, binomial, Poisson, uniform, Gaussian, multivariate Gaussian, categorical | Core | 1.5 |
| A5.3 | Conditional probability and Bayes' theorem: base rates and the medical-test trap | Core | 1.5 |
| A5.4 | Law of large numbers, central limit theorem and Monte Carlo estimation | Core | 1.5 |
| A5.5 | MLE and MAP: squared error as Gaussian MLE, log-loss as Bernoulli MLE, L2 as a Gaussian prior | Core | 2.5 |

**Hours:** reading 3.5 · coding 5.5 · project 2 · quiz and cheat sheet 1 · **total 12**

**Labs**

- **A5.1:** simulate dice, coins and a two-variable process. Estimate the mean, variance and
  covariance, and compare them with the exact values. Show how far the estimates wander as the
  sample size changes.
- **A5.2:** samplers built only on `rng.random()`: Bernoulli, binomial, categorical (inverse CDF),
  and Gaussian (Box–Muller). Overlay their histograms on `scipy.stats` densities. Sample a 2-D
  multivariate Gaussian by transforming standard normals with a matrix, and check its sample
  covariance against the target (links to A3.3).
- **A5.3:** a population simulator for a medical test. Compute `P(disease | positive)` by counting
  simulated people, compare it with Bayes' formula, and plot how it changes with the base rate.
- **A5.4:** a running-mean plot for the law of large numbers. Show the central limit theorem with
  sums of a very skewed variable. Estimate π and a 1-D integral by Monte Carlo, with error bars that
  shrink like `1/√n`.
- **A5.5:** Gaussian MLE for the mean and variance in closed form *and* by gradient descent on the
  negative log-likelihood (reusing your A4.4 code). Bernoulli MLE. A numerical check that the MSE
  minimizer equals the Gaussian NLL minimizer. MAP with a Gaussian prior, compared with the
  closed-form ridge solution.

**Project milestone: P1 M2, explore (2 h).** 6–8 plots that follow the A1.4 rules, each with a
one-sentence takeaway. Turn your M0 questions into 2–3 testable hypotheses and **write them down
before you run any test**. This is pre-registration, and A6.3 explains why it matters.

**Prerequisites:** A1.2, A1.4, A3.3 (covariance), A4.4 (gradient descent, for A5.5).

**Resources** (optional)

- Brown University, "Seeing Theory" (interactive probability): https://seeing-theory.brown.edu/ (VERIFY link)
- Harvard Stat 110 (Joe Blitzstein), lectures and handouts: https://projects.iq.harvard.edu/stat110 (VERIFY link)
- *Mathematics for Machine Learning*, chapter 6 (probability and distributions): https://mml-book.github.io/ (VERIFY link)

**GPU:** no.

**Used later in:** every loss function in the course (A5.5); naive Bayes in B2.4 (A5.3); regularization
as MAP priors in B2.2 (A5.5); Gaussian mixtures in B5.3 (A5.2); sampling in decoding (F2.4) and in
diffusion (G1.4) (A5.2).

---

## A6 · Statistics & information theory (course week 6)

**Teaching order this week:** A6.1 → A6.2 → A6.3 first, because P1's final milestone uses them. Then
the P1 write-up, then A6.4 and A6.5.

**Learning objectives**

- I can summarize a variable robustly (median, IQR, MAD), and explain why correlation alone cannot show causation.
- I can build a bootstrap confidence interval from scratch and explain what "95% confidence" does and does not mean.
- I can choose and run a t-test, a chi-square test or a permutation test, state its assumptions, report an effect size, and correct for multiple comparisons.
- I can explain two ways an A/B test goes wrong (peeking, many metrics) and show each by simulation.
- I can compute entropy, cross-entropy and KL divergence by hand for small distributions, and explain why a classifier's loss is a cross-entropy.
- I can update a Beta prior with Binomial data and explain how the posterior differs from the MLE. *(Depth)*

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| A6.1 | Descriptive statistics and EDA: robust summaries, correlation versus causation | Core | 1.5 |
| A6.2 | Confidence intervals and the bootstrap | Core | 2 |
| A6.3 | Hypothesis testing: p-values, t-test, chi-square, multiple comparisons, A/B tests and their pitfalls | Core | 2.5 |
| A6.4 | Entropy, cross-entropy and KL divergence: what classifiers and VAEs minimize | Core | 2 |
| A6.5 | Bayesian inference: conjugate priors with the Beta–Binomial example | Depth | 1 |

**Hours:** reading 3.5 · coding 5.5 · project 3 · quiz, cheat sheet and CP1 self-check 1 · **total 13**

**Labs**

- **A6.1:** median, IQR, MAD and trimmed mean from scratch, checked against pandas/NumPy. Pearson
  and Spearman correlation from scratch, checked against `scipy.stats`. Anscombe's quartet, typed
  in by hand: identical summary statistics, four very different plots.
- **A6.2:** percentile bootstrap confidence intervals from scratch for the mean and the median,
  compared with the t-interval from `scipy.stats`. A coverage simulation: draw 1,000 samples from a
  known distribution and count how often each method's 95% interval contains the true mean.
- **A6.3:** a permutation test for a difference in means from scratch. Welch's t-test from scratch,
  checked against `scipy.stats.ttest_ind(equal_var=False)`. A chi-square test of independence from
  scratch, checked against `scipy.stats.chi2_contingency`. Two failure simulations: 20 tests on pure
  noise (count the false positives, then apply Bonferroni and Holm), and an A/B test where you
  "peek" every day and stop at the first p < 0.05.
- **A6.4:** entropy, cross-entropy and KL divergence from scratch, checked against
  `scipy.stats.entropy`. Verify numerically that cross-entropy = entropy + KL, and show that KL is
  not symmetric. Compute the cross-entropy loss of a few hand-made classifier predictions.
- **A6.5 (Depth):** a Beta–Binomial update from scratch. Plot prior → posterior as data arrives,
  and compare the posterior mean with the MLE for small and large samples.

**Project milestone: P1 due, test and report (3 h).** Run your pre-registered tests with their
assumptions checked. Report bootstrap confidence intervals and effect sizes, not only p-values, and
apply a multiple-comparison correction. Write the report in the project README, including what the
data *cannot* tell you (confounders, correlation versus causation, sampling bias). P1 needs **≥ 70%**
on its rubric to pass CP1.

**Prerequisites:** A1.3–A1.4, A5.1–A5.5 (A6.2–A6.3 rely on the CLT from A5.4; A6.4 relies on A5.5).

**Resources** (optional)

- *OpenIntro Statistics* (free textbook), chapters 5–6: https://www.openintro.org/book/os/ (VERIFY link)
- Christopher Olah, "Visual Information Theory": https://colah.github.io/posts/2015-09-Visual-Information/ (VERIFY link)
- Allen Downey, *Think Stats* (free online): https://greenteapress.com/wp/think-stats/ (VERIFY link)

**GPU:** no.

**Used later in:** evaluation and cross-validation in B4 (A6.2, A6.3); A/B testing and drift
detection in J2.2 (A6.3); the cross-entropy loss of every classifier from B2.1 to the LLMs in F
(A6.4); the VAE objective in G1.2 (A6.4); the KL penalty in RLHF in H2.3 (A6.4).

---

## You're ready for Part B when…

This is checkpoint **CP1**. Answer each item without notes. The answers belong in your quiz reviews,
not here.

1. You can predict the result shape of `(32, 10) @ (10, 3)`, and of a `(5, 1)` array plus a `(3,)`
   array, and explain why `(10, 3) @ (32, 10)` fails.
2. You can write a pairwise distance matrix with no Python loops.
3. You can translate a SQL `GROUP BY` with a `LEFT JOIN` into pandas, and say what happens to the NULLs.
4. You can explain in one sentence each what an eigenvector is, what the SVD does, and why PCA (B5.4) is built on the SVD.
5. You can derive the gradient of mean squared error for linear regression with respect to the
   weights, give its shape, and confirm it with a numerical gradient check.
6. You can fit a line with both the normal equations and gradient descent and get the same answer,
   and recognize a diverging learning rate from its loss curve.
7. You can solve a medical-test base-rate problem with Bayes' theorem and explain the result to a non-specialist.
8. You can derive the Gaussian MLE for the mean and say why that makes squared error the "Gaussian" loss.
9. You can build a bootstrap confidence interval and say what "95%" refers to.
10. You can explain what goes wrong when you run many tests at α = 0.05, and how to correct for it.
11. You can compute the entropy of a fair and of a biased coin, and explain why cross-entropy is never smaller than entropy.
12. **P1 scores ≥ 70%** on its rubric, and your Part A weekly quizzes average **≥ 80%**.

If two or more items fail, take one catch-up week on the topic ids your reviews flagged before you
start Part B. That is the outline's checkpoint rule.
