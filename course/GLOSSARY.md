# Glossary

Every term in `COURSE_OUTLINE.md`, with a one-line definition and the topic id where it is taught
(the first id is the main lesson; later ids are where it comes back in depth). Written 2026-10-16.
Lessons use these words with these meanings; a lesson that introduces a new term adds it here in the
same commit. Notation follows `STYLE_GUIDE.md`.

Jump to: [A](#a) · [B](#b) · [C](#c) · [D](#d) · [E](#e) · [F](#f) · [G](#g) · [H](#h) · [I](#i) ·
[J](#j) · [K](#k) · [L](#l) · [M](#m) · [N](#n) · [O](#o) · [P](#p) · [Q](#q) · [R](#r) · [S](#s) ·
[T](#t) · [U](#u) · [V](#v) · [W](#w) · [X–Z](#xz)

## A

| Term | Definition | Taught in |
|---|---|---|
| A/B test | A randomized experiment that splits users between two variants and tests whether a metric differs. | A6.3b |
| Accuracy | The fraction of predictions that are correct; misleading when classes are imbalanced. | B4.1 |
| Activation function | The elementwise non-linearity applied after a layer's linear map (ReLU, sigmoid, tanh, GELU). | C1.1 |
| Actor–critic | An RL method with a policy (actor) and a learned value function (critic) that reduces the variance of policy gradients. | H2.2 |
| Adam | An optimizer that scales each parameter's step by running estimates of the gradient's mean and squared magnitude. | C3.1 |
| AdamW | Adam with weight decay applied directly to the weights instead of through the gradient. | C3.1 |
| Advantage | How much better an action is than the policy's average in that state: $A(s, a) = Q(s, a) - V(s)$. | H2.2 |
| Affine map | A linear map followed by a shift, $f(\mathbf{x}) = \mathbf{A}\mathbf{x} + \mathbf{b}$; a dense layer before its activation. | A2.1, C1.1 |
| Agent (LLM) | An LLM in a loop that chooses tools, observes their results and decides the next step until a task is done. | I3.1 |
| Alerting | Automatically notifying a person when a monitored metric crosses a threshold. | J2.2 |
| Alignment | Making a model's behaviour match what its developers and users intend, including refusing harmful requests. | F3.3 |
| Alternative hypothesis | $H_1$: the claim a test looks for evidence of, such as "the two means differ". | A6.3a |
| Anomaly detection | Finding examples that differ markedly from the bulk of the data. | B5.6 |
| Anscombe's quartet | Four small datasets with identical means, variances, correlation and regression line but completely different plots. | A6.1 |
| ARIMA | AutoRegressive Integrated Moving Average: a classical time-series model built from lags, differencing and past errors. | B6.4 |
| Array (NumPy) | A block of numbers of one dtype with a shape; NumPy's core data structure. | A1.2 |
| Artifact (MLOps) | A file produced by a run, such as a model, a plot or a dataset snapshot, stored with the run that made it. | J1.1 |
| Associativity (matrix product) | $(\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$: the bracketing changes the cost, not the result. | A2.2 |
| Attention | A mechanism that computes a weighted average of values, with weights from how well a query matches each key. | E2.4, F1.1 |
| Augmentation | Creating modified training examples (flips, crops, colour shifts) so a model learns invariances and overfits less. | D2.2 |
| Augmented matrix | A matrix with extra columns appended, such as $[\mathbf{A} \mid \mathbf{b}]$ or $[\mathbf{A} \mid \mathbf{I}]$, so row operations act on both parts. | A2.2, A2.3 |
| Autocast | PyTorch's context manager that runs selected operations in a lower-precision dtype for mixed precision. | C4.2 |
| Autoencoder | A network trained to reconstruct its input through a narrow bottleneck, learning a compressed representation. | G1.1 |
| Autograd | Automatic differentiation: recording operations on a graph and applying the chain rule backward to get gradients. | C1.3, C2.1 |
| Axis | One dimension of an array; axis 0 runs down the rows, axis 1 across the columns. | A1.2 |

## B

| Term | Definition | Taught in |
|---|---|---|
| Back substitution | Solving an upper-triangular system from the last equation up. | A2.3 |
| Backpropagation | Computing every parameter's gradient by applying the chain rule backward through the network. | C1.2 |
| Backpropagation through time (BPTT) | Backpropagation through an RNN unrolled over the time steps of a sequence. | E2.1 |
| Backward stable | An algorithm whose computed answer is the exact answer to a problem within rounding of yours; its residual stays at rounding level (`np.linalg.solve`). | A3.4 |
| Bag-of-words | Representing a text by the counts of its words, ignoring their order. | E1.2 |
| Bagging | Bootstrap aggregating: training models on bootstrap resamples and averaging their predictions. | B3.2 |
| Base rate | The prior probability of a condition in the population, before any evidence. | A5.3 |
| Baseline | The simplest reasonable model or rule that a real model must beat to be worth deploying. | B1.1 |
| Batching (serving) | Grouping several requests into one model call to raise throughput. | J2.1, I4.3 |
| BatchNorm | Normalizing each feature over the batch, then applying a learned scale and shift. | C3.4 |
| Bayes' theorem | $P(A \mid B) = P(B \mid A)\,P(A) / P(B)$: how to update a belief with evidence. | A5.3 |
| Bayesian inference | Treating parameters as random variables and computing their posterior distribution given data. | A6.5 |
| Bayesian optimization | Hyperparameter search that fits a model of the score and picks the next trial where it looks most promising. | B4.3 |
| BCa interval | Bias-corrected and accelerated bootstrap interval: percentiles shifted and stretched to correct for bias and skew; SciPy's default. | A6.2 |
| Beam search | Decoding that keeps the $k$ most probable partial sequences at each step. | E2.3, F2.4 |
| Bellman equation | The recursive relation between a state's value and the values of the states that follow it. | H1.1 |
| Benjamini–Hochberg procedure | Reject the $k$ smallest of $m$ p-values, $k$ the largest with $p_{(k)} \le k\alpha/m$; controls the false discovery rate. | A6.3b |
| Bernoulli distribution | The distribution of a single yes/no outcome with success probability $p$. | A5.2 |
| BERT | An encoder-only transformer pretrained with masked language modelling. | F2.1 |
| Bessel's correction | Dividing by $n - 1$ instead of $n$ in a sample variance or covariance so it is unbiased; `ddof=1` in NumPy and pandas. | A3.3, A5.1 |
| Beta-binomial distribution | The number of successes in $m$ trials when the rate itself is Beta-distributed; the posterior predictive of the Beta–Binomial model. | A6.5 |
| Beta distribution | A density on $[0, 1]$, $\propto \theta^{a-1}(1 - \theta)^{b-1}$, with mean $a/(a + b)$; the conjugate prior for a rate. | A6.5 |
| Bias (statistical) | The systematic gap between an estimator's or model's expected value and the truth. | B4.5 |
| Bias (fairness) | Systematically different model behaviour or errors across groups of people. | B4.6 |
| Bias–variance trade-off | Test error splits into bias (too simple), variance (too sensitive to the training set) and noise. | B4.5 |
| Binomial distribution | The number of successes in $n$ independent Bernoulli trials. | A5.2 |
| Bit | The unit of information with $\log_2$: the surprise of one fair coin flip. | A6.4 |
| Block bootstrap | Resampling blocks of consecutive rows (or whole users) so that correlated data keeps its dependence. | A6.2 |
| Bonferroni correction | Testing each of $m$ hypotheses at $\alpha / m$ to control the chance of any false positive. | A6.3b |
| Boolean mask | An array of True/False values that selects the matching entries of an array or rows of a DataFrame. | A1.2, A1.3 |
| Bootstrap | Estimating the uncertainty of a statistic by recomputing it on resamples drawn with replacement. | A6.2 |
| Box–Muller transform | Turning two uniform numbers into two independent standard normals through a random radius $\sqrt{-2\log u_1}$ and angle $2\pi u_2$. | A5.2 |
| BPE (byte-pair encoding) | A subword tokenizer that repeatedly merges the most frequent adjacent pair of symbols. | F1.3 |
| Breakdown point | The share of values that can be replaced by garbage before a statistic can be pushed arbitrarily far: 0 for the mean, 50% for the median. | A6.1 |
| Broadcasting | NumPy's rule for combining arrays of different shapes by virtually repeating size-1 axes. | A1.2 |

## C

| Term | Definition | Taught in |
|---|---|---|
| Cache (backprop) | Values saved during the forward pass (inputs, activations) that the backward pass needs to compute local derivatives. | A4.3 |
| Caching (LLM) | Reusing earlier results instead of paying for a new model call; see exact, prompt and semantic caching. | I4.4 |
| Calibration | Whether predicted probabilities match observed frequencies (of cases predicted 0.8, about 80% are positive). | B4.1 |
| Canary rollout | Sending a small share of live traffic to a new model version before a full rollout. | J2.4 |
| Catastrophic cancellation | Losing most significant digits by subtracting two large, nearly equal floating-point numbers. | A3.4, A3.3 |
| Categorical distribution | The distribution of one draw from $K$ classes with probabilities $p_1, \dots, p_K$. | A5.2 |
| Categorical encoding | Turning category labels into numbers: one-hot, ordinal or target encoding. | B1.3 |
| Cauchy distribution | A heavy-tailed distribution with no mean; its running sample mean never settles. | A5.4 |
| Causal masking | Blocking attention to future positions so each token sees only earlier ones. | F1.1 |
| Centering | Subtracting each column's mean so every feature has mean 0; required before computing covariance or PCA. | A3.3 |
| Central difference | $(f(x + h) - f(x - h)) / 2h$: a derivative estimate whose truncation error shrinks like $h^2$. | A4.1 |
| Central limit theorem (CLT) | The mean of many independent draws is approximately Gaussian, whatever the original distribution. | A5.4 |
| Chain rule | The derivative of a composition is the product of the derivatives of its parts. | A4.3 |
| Chained assignment | Assigning through two indexing steps (`df[m]["c"] = v`); does nothing under copy-on-write. | A1.3 |
| Characteristic polynomial | $\det(\mathbf{A} - \lambda\mathbf{I})$, a polynomial in $\lambda$ whose roots are the eigenvalues of $\mathbf{A}$. | A3.1 |
| Chat template | The exact text format that wraps system, user and assistant messages for a given model. | F3.1 |
| Chebyshev's inequality | $P(\lvert X - \mu\rvert \ge \varepsilon) \le \sigma^2/\varepsilon^2$: a bound on straying from the mean that holds for any distribution. | A5.4 |
| Checkpointing (gradient) | Saving memory by recomputing some activations during the backward pass instead of storing them. | C4.2 |
| Chinchilla (compute-optimal) | The finding that, for a fixed compute budget, parameters and training tokens should grow together. | F2.3 |
| Chi-square test | A test of whether observed counts in categories differ from the counts expected under independence. | A6.3a |
| Cholesky decomposition | Writing a symmetric positive-definite matrix as $\mathbf{L}\mathbf{L}^\top$ with $\mathbf{L}$ lower triangular. | A3.4 |
| Chord | The straight segment joining two points on a function's graph. | A4.5 |
| Chunking | Splitting documents into pieces small enough to embed and retrieve. | I2.1 |
| CI for ML | Continuous integration that also runs data tests, model tests and evals before a change can merge. | J2.3 |
| Citation (RAG) | Pointing each claim in a generated answer to the retrieved source it came from. | I2.3 |
| Class imbalance | One class is much rarer than the others, so accuracy and default thresholds mislead. | B4.4 |
| Class weights | Weighting the loss so errors on rare classes cost more. | B4.4 |
| Classifier-free guidance | Steering a diffusion model by mixing its conditional and unconditional predictions. | G2.4 |
| Cleaning log | A record of every data-cleaning decision, its reason and how many rows it touched. | B1.2 |
| CLIP | A model that trains image and text encoders contrastively so matching pairs have similar embeddings. | G2.2 |
| Cohen's d | Difference in means divided by the pooled standard deviation; a standardized effect size. | A6.3a |
| Collaborative filtering | Recommending items from the behaviour of similar users or items, without item content. | B6.1 |
| Column space | The set of all vectors $\mathbf{A}\mathbf{x}$: every output a matrix can reach. | A2.3 |
| ColumnTransformer | A scikit-learn step that applies a different transformer to each group of columns and joins the results. | B1.3 |
| Commutative | Describes an operation whose order does not matter; matrix multiplication is not commutative. | A2.2 |
| Complementary slackness | The KKT condition $\mu g(\mathbf{x}) = 0$: a multiplier is zero unless its inequality constraint is active. | A4.5 |
| Computational graph | A graph of operations whose edges carry values forward and gradients backward. | A4.3 |
| Concept drift | The relationship between inputs and the target changes over time. | J2.2 |
| Condition number | How much a matrix amplifies relative error when you solve with it; large means unstable. | A3.4, A4.4 |
| Conditional independence | $A$ and $B$ are independent once $C$ is known: $P(A, B \mid C) = P(A \mid C)\,P(B \mid C)$. | A5.3, B2.4 |
| Conditional probability | $P(A \mid B)$: the probability of $A$ given that $B$ happened. | A5.3 |
| Confidence interval | A range computed from data by a method that contains the true value in a stated share (say 95%) of repeated samples. | A6.2 |
| Confounder | A variable that affects both a supposed cause and its effect, creating correlation without causation. | A6.1 |
| Confusion matrix | A table of true versus predicted classes. | B4.1 |
| Conjugate prior | A prior that gives a posterior in the same family, such as Beta prior with Binomial data. | A6.5 |
| Constrained optimization | Minimizing a function subject to equality or inequality constraints. | A4.5 |
| Context assembly | Choosing, ordering and formatting retrieved chunks into the prompt within a token budget. | I2.3 |
| Contrastive learning | Training embeddings so matching pairs are close and non-matching pairs are far apart. | G2.2 |
| Convexity | A function is convex if every chord lies above it; then any local minimum is global. | A4.5 |
| Convolution | Sliding a small kernel over an input and taking a weighted sum at each position. | D1.1 |
| Coordinate descent | Minimizing one parameter at a time with the others held fixed, cycling until nothing changes; how the lasso is fitted. | B2.2 |
| Copy-on-write | pandas 3 behaviour: every selection acts as a new object, so changing it never changes the original. | A1.3 |
| Correlation | A standardized measure (−1 to 1) of how two variables move together; not evidence of causation. | A6.1 |
| Correlation matrix | The `(d, d)` matrix of pairwise correlations, $R_{jk} = S_{jk}/(s_j s_k)$: the covariance matrix of the standardized features. | A3.3 |
| Cosine decay | A learning-rate schedule that falls along a half cosine curve. | C3.2 |
| Cosine distance | $1 - \cos(\mathbf{a}, \mathbf{b})$; smaller means more similar. | A1.5, B2.3 |
| Cosine similarity | The cosine of the angle between two vectors: $\mathbf{a}\cdot\mathbf{b} / (\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert)$. | A1.5 |
| Covariance | $\mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$: how two variables vary together. | A5.1 |
| Covariance matrix | The `(d, d)` matrix of pairwise covariances between features; symmetric and positive semi-definite. | A3.3 |
| Coverage | The share of repeated samples in which an interval method contains the true value; checked by simulation. | A6.2 |
| Cramér's V | $\sqrt{\chi^2/(N(\min(r, c) - 1))}$: an effect size for a table of counts, from 0 (independent) to 1. | A6.3a |
| Credible interval | A range that contains a parameter with a stated posterior probability, given the prior and the data. | A6.5 |
| Cross-encoder | A model that reads a query and a document together to score their relevance; used for reranking. | I2.3 |
| Cross-entropy | $H(p, q) = -\sum_x p(x) \log q(x)$: the expected cost of coding $p$ with $q$; the classification loss. | A6.4, C1.4 |
| Cross-validation | Estimating generalization by training and scoring on several train/validation splits of the data. | B4.3 |
| Cumulative distribution function (CDF) | $F(x) = P(X \le x)$; climbs from 0 to 1, in steps for discrete variables. | A5.2 |
| Curse of dimensionality | In high dimensions points are all far apart and distances lose contrast. | B2.3 |
| CutMix | Augmentation that pastes a patch of one image into another and mixes the labels by area. | D2.2 |
| Cyclic encoding | Encoding a periodic value (hour, month) as the sine and cosine of an angle, so the ends of the cycle are neighbours. | B1.3 |

## D

| Term | Definition | Taught in |
|---|---|---|
| Data cleaning | Fixing types, missing values, outliers and duplicates before modelling. | B1.2 |
| Data drift | The distribution of model inputs changes over time. | J2.2 |
| Data leakage | Information unavailable at prediction time reaches training or evaluation, inflating scores. | B1.4 |
| Data minimization | Collecting and keeping only the personal data a task needs. `[RAI]` | B1.2 |
| Data parallel | Copying the model to several devices, each processing part of the batch, then averaging gradients. | C4.4 |
| Data versioning | Tracking exact dataset snapshots so any model can be traced to the data that trained it. | J1.2 |
| DataFrame | pandas' in-memory table: named columns (Series) that share one row index. | A1.3 |
| DataLoader | PyTorch's iterator that batches, shuffles and loads examples from a Dataset. | C2.2 |
| Dataset (PyTorch) | A class with `__len__` and `__getitem__` that returns one example. | C2.2 |
| Datasheet | Documentation of a dataset's motivation, collection, composition and recommended uses. `[RAI]` | J2.5 |
| DBSCAN | Density-based clustering that grows clusters from dense regions and marks sparse points as noise. | B5.2 |
| DCGAN | A GAN that uses convolutional generator and discriminator networks. | G1.3 |
| ddof | "Delta degrees of freedom": the divisor is $n - $ `ddof`; `np.var` defaults to 0, `np.cov` and pandas to 1. | A3.3 |
| DDP | Distributed Data Parallel: PyTorch's multi-process data-parallel training. | C4.4 |
| DDPM | Denoising Diffusion Probabilistic Model: a diffusion model trained to predict the noise added at each step. | G1.4 |
| Decision boundary | The set of inputs where a classifier switches class; for logistic regression the hyperplane $\mathbf{w}^\top\mathbf{x} + b = 0$ ($p = 0.5$). | B2.1 |
| Decision tree | A model that splits the data by feature thresholds into leaves with simple predictions. | B3.1 |
| Decoder-only model | A transformer with causal self-attention that predicts the next token (GPT). | F2.1 |
| Decoding | Turning a language model's next-token distributions into text. | F2.4 |
| Deepfake | Realistic synthetic media of a real person, made without their consent or to deceive. `[RAI]` | G2.4 |
| Deflation | Removing a found eigen- or singular component from a matrix so an iterative method can find the next one. | A3.2 |
| Degrees of freedom | The number of independent pieces of information left for estimating spread; $n - 1$ for a sample variance, the parameter of the t and chi-square distributions. | A6.2, A6.3a |
| Demographic parity | A fairness criterion: the positive-prediction rate is equal across groups. `[RAI]` | B4.6 |
| Dense retrieval | Retrieval by nearest neighbours in embedding space. | I2.1 |
| Density (histogram) | A bin's count divided by $n$ times the bin width, so the bars' areas sum to 1. | A1.4 |
| Depthwise separable convolution | A convolution split into a per-channel spatial filter and a 1×1 channel mix, using far fewer parameters. | D1.4 |
| Derivative | The instantaneous rate of change of a function; its local slope. | A4.1 |
| Descriptive statistics | Numbers that summarize data: centre, spread, shape. | A6.1 |
| Design matrix | The `(n, d)` matrix of examples a linear model multiplies, usually with a column of ones for the intercept. | A2.4, B1.5 |
| Determinant | The factor by which a square matrix scales area or volume (negative if it flips orientation); $ad - bc$ for a 2×2; 0 means singular. | A2.1 |
| Dice coefficient | Segmentation overlap: twice the intersection divided by the sum of the two areas. | D2.4 |
| Differential privacy | A guarantee that an algorithm's output changes very little whether or not any one person's data is included. `[RAI]` | J2.5 |
| Diffusion model | A generative model that learns to reverse a gradual noising process. | G1.4 |
| Dimensionality reduction | Mapping data to fewer dimensions while keeping its structure (PCA, t-SNE, UMAP). | B5.4 |
| Directional derivative | The rate of change of $f$ when moving from $\mathbf{x}$ along a unit vector $\mathbf{u}$: $\nabla f \cdot \mathbf{u}$. | A4.2 |
| Discount factor | $\gamma \in [0, 1]$: how much an RL agent values future rewards relative to immediate ones. | H1.1 |
| Discriminative model | A model of $p(y \mid \mathbf{x})$ directly, fitted to predict the label (logistic regression, SVMs). | B2.4 |
| Distributed training | Training one model across several devices or machines. | C4.4 |
| Docker image | A packaged, versioned file system and start command that runs the same everywhere. | J2.4 |
| Dot product | $\mathbf{a}\cdot\mathbf{b} = \sum_j a_j b_j$. | A1.5 |
| DPO | Direct Preference Optimization: tuning a model on preference pairs with a classification-style loss, no RL loop. | F3.2 |
| DQN | Deep Q-Network: Q-learning with a neural network, a replay buffer and a target network. | H1.4 |
| Dropout | Randomly zeroing activations during training so the network cannot rely on any one unit. | C3.5 |
| dtype | The element type of an array or column, such as `float64`, `int64` or `str`. | A1.2 |
| Dual problem (SVM) | The SVM rewritten in terms of one multiplier $\alpha_i$ per training point; it uses the data only through dot products, which enables kernels. | B2.5 |
| Dummy variable trap | One-hot columns for every category plus an intercept column: the columns are linearly dependent. | A2.3, B1.3 |
| Dynamic programming (RL) | Computing exact values or policies from a known model of the environment by repeated Bellman updates. | H1.2 |

## E

| Term | Definition | Taught in |
|---|---|---|
| Early stopping | Stopping training when the validation score stops improving. | C3.5 |
| Eckart–Young theorem | The truncated SVD $\mathbf{A}_k$ is the best rank-$k$ approximation of $\mathbf{A}$ in Frobenius (and spectral) norm. | A3.2 |
| EDA | Exploratory data analysis: summarizing and plotting data to understand it before modelling. | A6.1 |
| Effect size | How large a difference is, not just whether it is statistically significant. | A6.3a |
| EfficientNet | A CNN family that scales depth, width and resolution together. | D1.4 |
| Eigendecomposition | Writing a square matrix as $\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$ (eigenvectors as columns, eigenvalues on the diagonal); $\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ when symmetric. | A3.1 |
| Eigenvalue, eigenvector | $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$: a direction $\mathbf{v}$ the matrix only stretches, by $\lambda$. | A3.1 |
| Elastic net | Regularization with both L1 and L2 penalties. | B2.2 |
| ELBO | Evidence lower bound: the objective a VAE maximizes, reconstruction minus a KL term. | G1.2 |
| Elbow method | Choosing $k$ for k-means where adding clusters stops reducing inertia much. | B5.1 |
| EM algorithm | Expectation–maximization: alternating soft assignments and parameter updates to fit latent-variable models. | B5.3 |
| Embedding | A dense vector that represents an item (word, document, image) so similar items are close. | E1.3, I2.1 |
| Empirical CDF | The fraction of observed values at or below each $x$; an estimate of the CDF. | A5.2 |
| Empirical risk | The average loss of a model on a given set of rows; an estimate of the true risk. | B1.1 |
| Encoder–decoder | An architecture where an encoder reads the input and a decoder generates the output conditioned on it. | E2.3, F2.1 |
| Entropy | $H(p) = -\sum_x p(x) \log p(x)$: the average surprise of a distribution. | A6.4 |
| Epoch | One full pass of training over every example in the training set. | A4.4 |
| Epsilon-greedy | Exploring by taking a random action with probability $\epsilon$, otherwise the best-known action. | H1.3 |
| Equalized odds | A fairness criterion: true-positive and false-positive rates are equal across groups. `[RAI]` | B4.6 |
| Euclidean distance | The straight-line distance between two points, $\lVert\mathbf{a} - \mathbf{b}\rVert_2$. | A1.2, A1.5, B2.3 |
| Evals | Systematic tests that score an ML or LLM system against a fixed set of cases. | I2.4, I3.3 |
| Evidence (Bayes) | $P(B)$, the normalizer in Bayes' theorem, computed with the law of total probability. | A5.3 |
| Exact test | A test that computes the p-value by listing every possible relabelling instead of sampling or approximating. | A6.3a |
| Expectation | $\mathbb{E}[X]$: the probability-weighted average of a random variable. | A5.1 |
| Experiment tracking | Recording each training run's parameters, metrics and artifacts so runs can be compared and reproduced. | J1.1 |
| Explained variance | The share of total variance captured by each principal component. | B5.4 |
| Exploding gradients | Gradients that grow exponentially through layers or time steps, making training diverge. | C3.3, E2.1 |
| Exponential distribution | Waiting times between events at a constant rate $r$; density $re^{-rt}$, mean $1/r$. | A5.2 |
| Exponential smoothing | Forecasting with exponentially decaying weights on past observations. | B6.4 |

## F

| Term | Definition | Taught in |
|---|---|---|
| F1 score | The harmonic mean of precision and recall. | B4.1 |
| Fairness | The absence of unjustified differences in a model's outcomes or errors across groups. `[RAI]` | B4.6 |
| Faithfulness | Whether a generated answer is supported by the retrieved context it claims to use. | I2.4 |
| False discovery rate (FDR) | The expected share of false positives among all rejected hypotheses. | A6.3b |
| False positive rate | $P(\text{flagged} \mid \text{actually negative})$, equal to $1 -$ specificity. | A5.3, B4.1 |
| Family-wise error rate (FWER) | The probability of at least one false positive among a family of tests; $1 - (1 - \alpha)^m$ for $m$ independent true nulls. | A6.3b |
| Fan-out | One value used by several later operations; in the backward pass the gradients from each use add up. | A4.3 |
| Fancy indexing | Indexing an array with a list or array of positions; always returns a copy. | A1.2 |
| fastText | Word embeddings built from character n-grams, so unseen words still get vectors. | E1.4 |
| Feature engineering | Building model inputs from raw data: scaling, encoding, dates, interactions. | B1.3 |
| Feature extraction (transfer) | Using a frozen pretrained network's outputs as features for a new small model. | D2.1 |
| Feature subsampling | Each tree or split considers only a random subset of features (random forests). | B3.2 |
| Few-shot prompting | Including a few worked examples in the prompt. | I1.2 |
| Figure and Axes (matplotlib) | A Figure is the whole image; each Axes inside it is one plot with its own axes, title and legend. | A1.4 |
| Fine-tuning | Continuing to train a pretrained model on a new task's data. | D2.1, J1.3 |
| Finite differences | Approximating a derivative by $(f(x + h) - f(x - h)) / 2h$; used to check gradients. | A4.1 |
| Fixed-horizon test | An experiment analysed once, at a sample size fixed in advance. | A6.3b |
| FlashAttention | An exact attention algorithm that avoids storing the full attention matrix by working in tiles. | F2.5 |
| Forward and reverse KL | Minimizing $D_{\mathrm{KL}}(p \,\|\, q)$ makes $q$ cover all of $p$'s mass; minimizing $D_{\mathrm{KL}}(q \,\|\, p)$ makes it lock onto one mode. | A6.4, G1.2 |
| Forward difference | $(f(x + h) - f(x)) / h$: a derivative estimate whose truncation error shrinks like $h$. | A4.1 |
| Forward mode | Automatic differentiation that pushes derivatives from inputs to outputs; one pass per input direction. | A4.3 |
| Forward substitution | Solving a lower-triangular system from the first equation down. | A3.4 |
| Free variable | An unknown of a linear system that can take any value; there are $d - \operatorname{rank}(\mathbf{A})$ of them. | A2.3 |
| Freedman–Diaconis rule | A histogram bin width of $2\,\mathrm{IQR}\,n^{-1/3}$, robust to outliers. | A1.4 |
| Frobenius norm | $\lVert\mathbf{A}\rVert_F = \sqrt{\sum_{ij} A_{ij}^2} = \sqrt{\sum_j \sigma_j^2}$: the matrix as one long vector. | A3.2 |
| FSDP | Fully Sharded Data Parallel: sharding parameters, gradients and optimizer state across devices. | C4.4 |
| Full rank | A matrix whose rank equals the smaller of its two dimensions. | A2.3 |
| Function calling | See tool use. | I1.4 |

## G

| Term | Definition | Taught in |
|---|---|---|
| Gamma distribution | A positive, right-skewed distribution; the mean of $n$ Exponential(1) draws is Gamma with shape $n$ and scale $1/n$. | A5.4 |
| GAN | Generative adversarial network: a generator and a discriminator trained against each other. | G1.3 |
| Garden of forking paths | The many analysis choices (metrics, subgroups, outlier rules) that inflate false positives when chosen after seeing the data; also p-hacking. | A6.3b |
| Gaussian (normal) distribution | The bell-shaped distribution $\mathcal{N}(\mu, \sigma^2)$. | A5.2 |
| Gaussian elimination | Solving a linear system with row operations that clear the entries below the diagonal, then back substitution. | A2.3 |
| Gaussian mixture model (GMM) | A density modelled as a weighted sum of Gaussians; soft clustering. | B5.3 |
| Gaussian naive Bayes | Naive Bayes for continuous features: one normal distribution per class and feature. | B2.4 |
| Gauss–Jordan elimination | Inverting a matrix by row-reducing $[\mathbf{M} \mid \mathbf{I}]$ to $[\mathbf{I} \mid \mathbf{M}^{-1}]$. | A2.2 |
| Generalization | How well a model performs on new data from the same distribution, not just its training data. | B1.1 |
| Generalization gap | The difference between a model's error on new data and on its training data. | B1.1, B4.5 |
| Generative model | A model of how each class produces the data, $p(\mathbf{x} \mid y)\,p(y)$, turned into $p(y \mid \mathbf{x})$ by Bayes' theorem (naive Bayes, GMMs); in Part G, a model that samples new data. | B2.4, G1.1 |
| Geometric mean | $\exp$ of the mean of $\log x$; for data whose log is symmetric it sits near the median. | A6.1 |
| Gibbs' inequality | $D_{\mathrm{KL}}(p \,\|\, q) \ge 0$, with equality only when $q = p$; so $H(p, q) \ge H(p)$. | A6.4 |
| Golden set | A curated, versioned set of inputs with known good outputs used to evaluate a system. | I2.4 |
| GPT | A decoder-only transformer pretrained on next-token prediction. | F1.4, F2.1 |
| Grad-CAM | A heat map of which image regions most influenced a CNN's prediction, from gradients of the last conv layer. | D2.5 |
| Gradient | The vector of partial derivatives; it points in the direction of steepest increase. | A4.2 |
| Gradient accumulation | Summing gradients over several small batches before one optimizer step to simulate a larger batch. | C4.2 |
| Gradient boosting | Building an ensemble by fitting each new tree to the gradient (residuals) of the current ensemble's loss. | B3.3 |
| Gradient checking | Comparing a hand-written gradient against finite differences by relative error before trusting it. | A4.2 |
| Gradient clipping | Rescaling gradients whose norm exceeds a threshold. | E2.1 |
| Gradient descent | Repeatedly stepping parameters against the gradient: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\nabla\mathcal{L}$. | A4.4 |
| Gram–Schmidt | Building an orthonormal basis by removing from each column its parts along the earlier ones; the classical form loses orthogonality on ill-conditioned matrices, the modified form less so. | A3.4 |
| Grid approximation | Computing a posterior by evaluating prior × likelihood on many parameter values and normalizing. | A6.5 |
| Grid search | Trying every combination of hyperparameter values from a grid. | B4.3 |
| Group k-fold | Cross-validation that keeps all rows of a group (a user, a patient) in the same fold. | B4.3 |
| Group sequential design | An experiment with a few planned interim looks, each with a stricter threshold so the total false-positive rate stays at $\alpha$. | A6.3b |
| Grouped-query attention (GQA) | Attention where groups of query heads share key and value heads, shrinking the KV cache. | F2.5 |
| GRU | Gated recurrent unit: an RNN cell with update and reset gates. | E2.2 |
| Guardrail metric | An experiment metric that must not get worse (latency, errors), checked alongside the primary metric. | A6.3b |
| Guardrails | Checks on an LLM system's inputs and outputs that block or fix unsafe or invalid content. `[RAI]` | I4.1 |
| Gymnasium | The standard Python API for RL environments (`reset`, `step`). | H1.3 |

## H

| Term | Definition | Taught in |
|---|---|---|
| Hallucination | A model stating something fluent and confident that is false or unsupported. `[RAI]` | F3.3 |
| He initialization | Weight initialization with variance $2 / d_{\text{in}}$, suited to ReLU layers. | C3.3 |
| Hessian | The matrix of second partial derivatives $H_{ij} = \partial^2 f / \partial x_i \partial x_j$; its eigenvalues give the curvature in each direction. | A4.4, A4.5 |
| Hierarchical clustering | Building a tree of clusters by repeatedly merging (or splitting) groups. | B5.2 |
| Hilbert matrix | $H_{ij} = 1/(i + j - 1)$: a standard example of a severely ill-conditioned matrix. | A3.4 |
| Hinge loss | $\max(0, 1 - y z)$: the SVM loss that penalizes points inside the margin. | B2.5 |
| Histogram | A chart of how many values fall into each of a set of equal-width intervals (bins). | A1.4 |
| HNSW | Hierarchical Navigable Small World: a graph index for fast approximate nearest-neighbour search. | I2.2 |
| Holdout set | Data kept aside and used only for the final evaluation. | B1.1 |
| Holm procedure | Step-down FWER correction: compare sorted p-values with $\alpha/m, \alpha/(m - 1), \dots$ and stop at the first failure; never rejects less than Bonferroni. | A6.3b |
| Householder reflection | A reflection that zeroes a column below its first entry; the stable way to compute QR (`np.linalg.qr`). | A3.4 |
| Hybrid search | Combining keyword (full-text) and vector search results. | I2.2 |
| Hyperparameter | A setting chosen before training (learning rate, depth), not learned from data. | B4.3 |
| Hypothesis test | A procedure that decides whether data are surprising enough under a null hypothesis to reject it. | A6.3a |

## I

| Term | Definition | Taught in |
|---|---|---|
| Idempotent | A matrix with $\mathbf{P}^2 = \mathbf{P}$: applying it twice is the same as once, like every projection. | A2.4 |
| Identity matrix | $\mathbf{I}$: ones on the diagonal, zeros elsewhere; $\mathbf{A}\mathbf{I} = \mathbf{A}$. | A2.2 |
| i.i.d. | Independent and identically distributed: draws that do not influence each other and share one distribution. | A5.4 |
| Implicit feedback | User signals such as clicks and views, which show interest but not dislike. | B6.1 |
| Imputation | Filling missing values with an estimate (mean, median, a model), learned on training rows only. | B1.2 |
| Independence (random variables) | $p(x, y) = p(x)\,p(y)$ for all values: knowing one tells nothing about the other; implies zero covariance, not the reverse. | A5.1 |
| Index (pandas) | The row labels of a DataFrame or Series; arithmetic aligns rows by these labels. | A1.3 |
| Indirect prompt injection | Malicious instructions hidden in content the model reads (a web page, an email, a tool result). `[RAI]` | I4.2 |
| Inference server (LLM) | A service optimized to run an LLM for many requests (batching, KV cache management). | J2.1 |
| Initialization | How weights are set before training; it controls whether signals vanish or explode. | C3.3 |
| Inner join | A join that keeps only rows whose key appears in both tables. | A1.3 |
| Instance segmentation | Labelling each pixel with the object instance it belongs to. | D2.4 |
| Instruction tuning (SFT) | Supervised fine-tuning on instruction–response pairs so a base model follows instructions. | F3.1 |
| Interaction feature | A feature built from two others (often their product) so the effect of one can depend on the other. | B1.3 |
| Interpretability | Methods that explain what a model relies on and why it made a prediction. | B3.4 |
| Interquartile range (IQR) | $Q(0.75) - Q(0.25)$: the width of the middle half of the data; a robust spread. | A6.1 |
| Inverse matrix | $\mathbf{A}^{-1}$ with $\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$; exists only for square full-rank matrices. | A2.2 |
| Inverse-CDF sampling | Draw $u$ uniform on $[0, 1)$ and return $F^{-1}(u)$; for classes, the first one whose running total exceeds $u$. | A5.2 |
| IoU | Intersection over union: the overlap of two boxes or masks divided by their union. | D2.3 |
| Isolation forest | Anomaly detection that scores points by how few random splits isolate them. | B5.6 |
| IVFFlat | An index that clusters vectors and searches only the nearest clusters. | I2.2 |

## J

| Term | Definition | Taught in |
|---|---|---|
| Jacobian | The matrix of all partial derivatives of a vector-valued function; shape `(m, n)` for $f: \mathbb{R}^n \to \mathbb{R}^m$. | A4.2 |
| Jitter | A small multiple of the identity added to a matrix that should be positive definite so Cholesky succeeds despite rounding. | A3.4 |
| Joint distribution | The probabilities $p(x, y)$ of every pair of values of two random variables. | A5.1 |
| JSON schema | A machine-readable description of the structure a JSON value must have. | I1.3 |
| Jupyter | An interactive notebook environment that runs code in cells. | A1.1 |

## K

| Term | Definition | Taught in |
|---|---|---|
| k-anonymity | A table property: every combination of quasi-identifier values is shared by at least k rows. | B1.2 |
| k-fold cross-validation | Splitting data into $k$ folds and validating on each in turn. | B4.3 |
| k-means | Clustering that alternates assigning points to the nearest centroid and recomputing centroids. | B5.1 |
| k-nearest neighbours (kNN) | Predicting from the labels of the $k$ closest training examples. | B2.3 |
| Kernel (convolution) | The small weight array slid over the input in a convolution. | D1.1 |
| Kernel trick | Computing inner products in a high-dimensional feature space through a kernel function, without the features. | B2.5 |
| KKT conditions | Karush–Kuhn–Tucker conditions for $\min f$ s.t. $g \le 0$: $\nabla f + \mu\nabla g = \mathbf{0}$, $g \le 0$, $\mu \ge 0$, $\mu g = 0$. | A4.5 |
| KL divergence | $D_{\mathrm{KL}}(p \,\|\, q) = \sum_x p(x) \log (p(x) / q(x))$: the extra cost of using $q$ for $p$; not symmetric. | A6.4 |
| KL penalty (RLHF) | A term that keeps the tuned policy close to the reference model. | H2.3 |
| Kolmogorov–Smirnov statistic | The largest gap between an empirical CDF and a reference CDF (or between two empirical CDFs). | A5.2, A6.3a |
| KS test | Kolmogorov–Smirnov test: compares two distributions by the largest gap between their CDFs. | A6.3a, J2.2 |
| KV cache | Storing past keys and values during generation so each new token costs one step, not a full recompute. | F2.4 |

## L

| Term | Definition | Taught in |
|---|---|---|
| L1 regularization (lasso) | Penalizing $\lambda\lVert\mathbf{w}\rVert_1$; pushes weights to exactly zero. | B2.2 |
| L2 regularization (ridge) | Penalizing $\lambda\lVert\mathbf{w}\rVert_2^2$; shrinks weights; equals a Gaussian prior. | B2.2, A5.5 |
| Lagrange multiplier | A variable that turns a constrained optimization into an unconstrained one at a stationary point. | A4.5 |
| Lag feature | A past value of a time series used as a feature for predicting a later value. | B6.3 |
| Lagrangian | $f(\mathbf{x}) - \lambda(h(\mathbf{x}) - c)$: the objective plus multiplier times constraint, whose stationary points solve a constrained problem. | A4.5 |
| Laplace distribution | Density $\frac{1}{2b}e^{-\lvert x - \mu\rvert/b}$: sharper peak and heavier tails than a Gaussian; its NLL is the absolute error. | A5.5, B2.2 |
| Laplace smoothing | Adding a pseudo-count $\alpha$ (usually 1) to every count so no estimated probability is zero: $(N_{cw} + \alpha)/(N_c + \alpha V)$. | B2.4 |
| Latency | The time from a request to its response; reported as percentiles (p50, p95). | I4.3 |
| Latent diffusion | Diffusion run in an autoencoder's latent space instead of pixel space. | G2.4 |
| Law of large numbers | The sample mean converges to the expectation as the sample grows. | A5.4 |
| Law of total probability | $P(B) = \sum_k P(B \mid H_k)\,P(H_k)$ over hypotheses that cover every case without overlapping. | A5.3 |
| LayerNorm | Normalizing each example's features to zero mean and unit variance, then scaling and shifting. | C3.4 |
| Learning curve | Training and validation score plotted against training-set size (or epochs). | B4.5 |
| Learning rate | $\eta$: the step size of gradient descent. | A4.4 |
| Least privilege | Giving each tool or credential only the permissions its task needs. `[RAI]` | I4.2 |
| Least squares | Fitting by minimizing the sum of squared residuals. | A2.4 |
| Left join | A join that keeps every row of the left table, with missing values where the right table has no match. | A1.3 |
| LeNet | An early CNN for digit recognition (convolution, pooling, fully connected). | D1.3 |
| Lie factor | The size of an effect shown in a chart divided by its size in the data (Tufte); about 1 when honest. | A1.4 |
| LightGBM | A fast gradient-boosting library with histogram-based, leaf-wise tree growth. | B3.3 |
| Likelihood | The probability (or density) of the observed data as a function of the parameters, $p(\text{data} \mid \boldsymbol{\theta})$. | A5.3, A5.5 |
| Likelihood ratio | $P(\text{evidence} \mid H_1)/P(\text{evidence} \mid H_2)$: the factor by which the evidence multiplies the odds. | A5.3 |
| Linear combination | A weighted sum of vectors, $c_1\mathbf{v}_1 + \dots + c_k\mathbf{v}_k$. | A2.1 |
| Linear congruential generator (LCG) | A simple pseudo-random number generator, $x_{k+1} = (a x_k + c) \bmod m$. | A1.1 |
| Linear independence | A set of vectors none of which is a linear combination of the others; otherwise they are linearly dependent. | A2.3 |
| Linear map | A function $f(\mathbf{x}) = \mathbf{A}\mathbf{x}$ that preserves addition and scaling. | A2.1 |
| Linear regression | Predicting a number as a weighted sum of features plus a bias. | B1.5 |
| Linearity of expectation | $\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]$ for any random variables, dependent or not. | A5.1 |
| LLM-as-judge | Using an LLM with a rubric to grade outputs; it has position, length and self-preference biases. | I2.4 |
| Local derivative | The derivative of one node's output with respect to its own inputs; backprop multiplies local derivatives along paths. | A4.3 |
| Lockfile | A file recording the exact resolved version of every dependency (here `uv.lock`). | A1.1 |
| Log-odds (logit) | $\log\frac{p}{1 - p}$; the linear score of a logistic regression. The inverse of the sigmoid. | B2.1 |
| Log-sum-exp | Computing $\log\sum_i e^{s_i}$ stably by subtracting $\max_i s_i$ first; used to normalize log-probabilities and in softmax. | B2.4, C1.4 |
| Logistic regression | A linear model whose output goes through a sigmoid to give a probability. | B2.1 |
| Log-loss | Binary cross-entropy: $-[y \log p + (1 - y)\log(1 - p)]$. | B2.1, A5.5 |
| Long context | Handling inputs of many thousands of tokens, limited by attention's cost and the KV cache. | F2.5 |
| LoRA | Low-Rank Adaptation: fine-tuning by learning a low-rank update $\mathbf{B}\mathbf{A}$ added to frozen weights. | J1.3 |
| Loss scaling | Multiplying the loss before backward in fp16 so small gradients do not underflow. | C4.2 |
| Low-rank approximation | Replacing a matrix by one of rank $k$ that is close to it, usually the truncated SVD; stores $k(n + d + 1)$ numbers. | A3.2 |
| LR range test | Increasing the learning rate over one short run to find the range where loss falls fastest. | C3.2 |
| LSTM | Long short-term memory: an RNN cell with input, forget and output gates and a cell state. | E2.2 |
| LU factorization | Writing a square matrix (with rows reordered) as lower-triangular times upper-triangular; what `np.linalg.solve` computes. | A2.3, A3.4 |

## M

| Term | Definition | Taught in |
|---|---|---|
| Machine epsilon | The gap between 1 and the next float: $\varepsilon \approx 2.2 \times 10^{-16}$ in float64, $1.2 \times 10^{-7}$ in float32; the relative rounding error per operation. | A3.4, A4.1 |
| Mahalanobis distance | $\sqrt{(\mathbf{x} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})}$: distance from the mean in standard deviations along each eigen-direction of $\boldsymbol{\Sigma}$. | A5.2 |
| Manhattan distance | The L1 distance $\sum_j \lvert u_j - v_j\rvert$: distance along a city grid. | B2.3 |
| Mann–Whitney U test | A rank-based test of whether values in one group tend to be larger than in the other. | A6.3a |
| mAP | Mean average precision: the detection metric averaging precision over recall levels, classes and IoU thresholds. | D2.3 |
| MAP estimation | Maximum a posteriori: choosing parameters that maximize likelihood times prior. | A5.5 |
| Marginal distribution | The distribution of one variable on its own, found by summing the joint distribution over the others. | A5.1 |
| Markov chain | A random process that moves between states with probabilities that depend only on the current state, given by a transition matrix. | A3.1, H1.1 |
| Masked language modelling | Predicting hidden tokens from both sides of the context (BERT's objective). | F2.1 |
| Matrix | A rectangular table of numbers of shape `(m, d)`, read as a linear map from $d$-vectors to $m$-vectors. | A2.1 |
| Matrix factorization | Approximating a matrix as a product of two low-rank matrices (users × factors, factors × items). | B6.1 |
| Matrix multiplication | $(\mathbf{A}\mathbf{B})_{ij} = \sum_k A_{ik} B_{kj}$; shapes `(n, d) @ (d, k) -> (n, k)`. | A2.2 |
| Maximum margin | The SVM's goal: the separating boundary farthest from the nearest points of each class. | B2.5 |
| MCP | Model Context Protocol: an open protocol for exposing tools, resources and prompts to LLM applications. | I3.2 |
| MDP | Markov decision process: states, actions, transition probabilities, rewards and a discount factor. | H1.1 |
| Mean squared error (MSE) | The average squared difference between predictions and targets. | B4.2, A5.5 |
| Median | The middle value after sorting (mean of the two middle values for even $n$); breakdown point 50%. | A6.1 |
| Median absolute deviation (MAD) | The median of $\lvert x_i - \text{median}\rvert$; times 1.4826 it estimates $\sigma$ robustly. | A6.1 |
| Medical-test trap | A positive result from an accurate test for a rare condition is still most likely a false positive, because of the base rate. | A5.3 |
| Memory (agent) | State an agent carries across steps or sessions: the conversation, notes or retrieved facts. | I3.1 |
| Mini-batch | A small random subset of the training data used to estimate the gradient for one step. | A4.4 |
| Minimum detectable effect (MDE) | The smallest true effect an experiment of a given size detects with the planned power. | A6.3b |
| Minimum-norm solution | Of all least-squares solutions of an underdetermined or collinear system, the one with the smallest length; what `lstsq`, the pseudo-inverse and gradient descent from zero return. | B1.5, A2.4 |
| Missing indicator | A 0/1 column marking where a value was missing, so a model can use the fact of missingness. | B1.2 |
| Missing values | Absent entries in data; handled by dropping, imputing or flagging. | A1.3, B1.2 |
| Mixed precision | Training with 16-bit floats for most operations and 32-bit where precision matters. | C4.2 |
| Mixup | Augmentation that blends two images and their labels with the same weight. | D2.2 |
| MLE | Maximum likelihood estimation: choosing parameters that make the observed data most probable. | A5.5 |
| MLP | Multilayer perceptron: stacked fully connected layers with non-linear activations. | C1.1 |
| MobileNet | A CNN family built on depthwise separable convolutions for phones and edge devices. | D1.4 |
| Mode collapse | A GAN generator producing only a few kinds of output. | G1.3 |
| Model card | Documentation of a model's purpose, data, metrics, per-group performance, limits and risks. `[RAI]` | B4.6, J2.5 |
| Model registry | A versioned store of trained models with their lineage and stage (staging, production). | J1.2 |
| Model routing | Sending each request to the cheapest model that can handle it. | I4.3 |
| Moderation | Classifying content as harmful or not, to block or flag it. `[RAI]` | I4.1 |
| Momentum | Accumulating a running average of gradients to smooth and speed up descent. | C3.1 |
| Monitoring | Tracking a live model's inputs, outputs, quality and costs over time. | J2.2 |
| Monte Carlo estimation | Estimating an expectation by averaging over random samples. | A5.4 |
| MRR | Mean reciprocal rank: the average of 1 / (rank of the first relevant result). | I2.4 |
| Multi-agent system | Several LLM agents with different roles coordinated by an orchestrator or by messages. | I3.4 |
| Multi-head attention | Running several attention operations in parallel on different learned projections and concatenating them. | F1.1 |
| Multi-query attention (MQA) | Attention where all query heads share one key head and one value head. | F2.5 |
| Multinomial naive Bayes | Naive Bayes for count features (word counts): each class is a bag of words with its own word probabilities. | B2.4 |
| Multiple comparisons | Running many tests inflates the chance of false positives unless corrected. | A6.3b |
| Multivariate Gaussian | $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$: a Gaussian over vectors, with mean `(d,)` and covariance `(d, d)`. | A5.2 |
| Mutual information | $I(X; Y) = H(Y) - H(Y \mid X)$: how much knowing $X$ reduces uncertainty about $Y$; 0 only under independence. | A6.4, B3.1 |

## N

| Term | Definition | Taught in |
|---|---|---|
| Nat | The unit of information with the natural log; 1 nat = 1.443 bits. | A6.4 |
| Negative log-likelihood (NLL) | $-\sum_i \log p(x_i \mid \boldsymbol{\theta})$; minimizing it is MLE, and most training losses are an NLL. | A5.5 |
| n-gram | A sequence of $n$ consecutive tokens. | E1.1 |
| Naive Bayes | A classifier that applies Bayes' theorem assuming features are independent given the class. | B2.4 |
| NaN | "Not a number": the float value NumPy and pandas use for a missing number; NaN is not equal to itself. | A1.3 |
| NDCG | Normalized discounted cumulative gain: a ranking metric that rewards relevant items near the top. | B6.2 |
| Nearest-neighbour search | Finding the stored vectors most similar to a query vector. | A1.5, I2.1 |
| Negative sampling | Training embeddings by contrasting each true pair with a few random "negative" pairs. | E1.3 |
| Non-maximum suppression (NMS) | Removing overlapping detection boxes, keeping the highest-scoring one. | D2.3 |
| Norm | The length of a vector: L2 $\sqrt{\sum_j x_j^2}$, L1 $\sum_j \lvert x_j \rvert$. | A1.5 |
| Normal equations | $\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}$: the closed-form least-squares solution. | A2.4 |
| Normalization (layers) | Rescaling activations to a stable range inside a network (BatchNorm, LayerNorm). | C3.4 |
| Normalization (text) | Lower-casing, Unicode normalization and similar cleanup before tokenizing. | E1.1 |
| Null distribution | The distribution a test statistic would have if the null hypothesis were true. | A6.3a |
| Null hypothesis | $H_0$: the sceptical default a test tries to reject, such as "no difference". | A6.3a |
| Null space | All vectors $\mathbf{n}$ with $\mathbf{A}\mathbf{n} = \mathbf{0}$; adding one to a solution of $\mathbf{A}\mathbf{x} = \mathbf{b}$ gives another solution. | A2.3 |
| Numerical stability | Computing so that rounding errors and overflow do not ruin the result. | A3.4, C1.4 |

## O

| Term | Definition | Taught in |
|---|---|---|
| Object detection | Finding each object in an image as a class label and a bounding box. | D2.3 |
| Observability | Tracing, logging and metrics that let you see what a system did and why. | I4.5 |
| Odds | $P/(1 - P)$: probability 0.2 is odds 0.25, "1 to 4". | A5.3 |
| Odds ratio | $e^{w_j}$ in logistic regression: the factor by which the odds multiply when feature $j$ rises by one unit, others fixed. | B2.1 |
| One-cycle schedule | A learning rate that rises then falls over one cycle of training. | C3.2 |
| One-hot encoding | Representing a category as a vector with a 1 in its position and 0 elsewhere. | B1.3 |
| One-sided test | A test that counts only deviations in one pre-chosen direction; a two-sided test counts both. | A6.3a |
| Online evals | Scoring a live system's real traffic, often by sampling and judging. | I4.5 |
| ONNX | Open Neural Network Exchange: a portable model format that many runtimes can execute. | J2.1 |
| Optimizer | The rule that turns gradients into parameter updates (SGD, Adam). | C3.1 |
| Order statistic | $x_{(k)}$, the $k$-th smallest value of a sample. | A6.1 |
| Orthogonal | At right angles: $\mathbf{a}\cdot\mathbf{b} = 0$. | A2.4 |
| Orthogonal matrix | A square matrix $\mathbf{Q}$ with orthonormal columns, $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$; it rotates or reflects without changing lengths. | A3.1 |
| Out-of-bag error | A random forest's error on the rows each tree did not see in its bootstrap sample. | B3.2 |
| Outer product | $\mathbf{u}\mathbf{v}^\top$: a column times a row, giving a matrix of rank 1. | A2.1, A2.2 |
| Outlier | A value far from the bulk of the data. | A6.1, B1.2 |
| Overdispersion | Count data whose variance is larger than its mean, a sign that a Poisson model is too simple. | A5.2 |
| Overfitting | Fitting noise in the training data, so training error is low but test error is high. | B4.5 |

## P

| Term | Definition | Taught in |
|---|---|---|
| Partial correlation | The correlation of two variables after removing the linear effect of a third from each. | A6.1 |
| Pearson correlation | $r$: how well a straight line describes two variables, from $-1$ to $1$. | A6.1 |
| Peeking | Checking an experiment repeatedly and stopping at the first significant result, which inflates false positives. | A6.3b |
| Percentile bootstrap | A confidence interval from the $\alpha/2$ and $1 - \alpha/2$ quantiles of a statistic over bootstrap resamples. | A6.2 |
| Perfect separation | A feature that splits the classes exactly, so the logistic-regression MLE weights grow without bound. | A5.5, B2.1 |
| Plug-in principle | Using the empirical distribution of the sample in place of the unknown population; the basis of the bootstrap. | A6.2 |
| Point estimate | A single number computed from data as the best guess of an unknown quantity. | A6.2 |
| Posterior | $P(\text{hypothesis} \mid \text{evidence})$ or $p(\boldsymbol{\theta} \mid \text{data})$: the belief after seeing the evidence. | A5.3, A6.5 |
| Posterior predictive | The distribution of future data averaged over the posterior, wider than plugging in one estimate. | A6.5 |
| Power (statistical) | The probability that a test rejects $H_0$ when a given real effect exists; $1 - \beta$. | A6.3a, A6.3b |
| Pre-registration | Writing down the hypotheses, metrics, sample size and analysis before seeing the data. | A6.3b |
| Prior | $P(\text{hypothesis})$ or $p(\boldsymbol{\theta})$: the belief before seeing the evidence. | A5.3, A5.5, A6.5 |
| Probability density function (pdf) | Probability per unit of $x$ for a continuous variable; probabilities are areas under it, and it can exceed 1. | A5.2 |
| Probability mass function (pmf) | $P(X = x)$ for each value of a discrete variable; the values add up to 1. | A5.2 |
| Problem statement | The framing of an ML task: unit of prediction, target, prediction time, action, metric and baseline. | B1.1 |
| Prosecutor's fallacy | Mistaking $P(\text{evidence} \mid \text{hypothesis})$ for $P(\text{hypothesis} \mid \text{evidence})$. | A5.3 |
| Pseudonymization | Replacing an identifier with a token (for example a keyed hash, HMAC) so records can be joined but not directly identified; still personal data. | B1.2 |
| p-value | The probability of data at least as extreme as observed if the null hypothesis were true. | A6.3a |
| Padding (convolution) | Adding a border of zeros around the input to control the output size. | D1.1 |
| Partial dependence | The average prediction as one feature varies, with the others left as observed. | B3.4 |
| Partial derivative | The derivative with respect to one variable, holding the others fixed. | A4.1 |
| Partial pivoting | Choosing the row with the largest absolute entry in the current column as the pivot, to keep elimination accurate. | A2.2, A2.3 |
| PCA | Principal component analysis: projecting data onto the directions of greatest variance. | B5.4 |
| Perceptron | A single linear unit followed by a step function; the first neural network. | C1.1 |
| Period (random number generator) | The number of draws before a generator's sequence starts repeating. | A1.1 |
| Permutation importance | The drop in score when one feature's values are shuffled. | B3.4 |
| Permutation test | A hypothesis test that builds the null distribution by shuffling labels. | A6.3a |
| Perplexity | $\exp$(average cross-entropy per token); lower means the model is less surprised. | A6.4, F2.2 |
| pgvector | A PostgreSQL extension that stores vectors and runs similarity search. | I2.2 |
| PII | Personally identifiable information: data that identifies a person. `[RAI]` | B1.2, I4.1 |
| Pipeline (scikit-learn) | A chain of preprocessing steps and a model fitted and applied as one unit. | B1.3 |
| Pivot | The entry used to clear the rest of its column during elimination; the number of pivots is the rank. | A2.2, A2.3 |
| Poisson distribution | Counts of events in a fixed interval, with mean rate $\lambda$. | A5.2 |
| Policy | The agent's rule for choosing actions: $\pi(a \mid s)$. | H1.1 |
| Policy gradient | Improving a policy by gradient ascent on expected return, using sampled episodes. | H2.1 |
| Policy iteration | Alternating policy evaluation and greedy policy improvement until the policy stops changing. | H1.2 |
| Pooling | Downsampling a feature map by taking the max or mean over small windows. | D1.2 |
| Population standard deviation | $\sqrt{\frac{1}{n}\sum_i (x_i - \bar{x})^2}$; NumPy's default (`ddof=0`). | A1.2, A6.2 |
| Positional encoding | Information about token position added to a transformer's inputs or attention (sinusoidal, RoPE). | F1.2 |
| Positive definite | A symmetric matrix with $\mathbf{w}^\top\mathbf{C}\mathbf{w} > 0$ for every $\mathbf{w} \ne \mathbf{0}$; all eigenvalues positive; has a Cholesky factor. | A3.3, A3.4 |
| Positive semi-definite (PSD) | A symmetric matrix with $\mathbf{w}^\top\mathbf{C}\mathbf{w} \ge 0$ for every $\mathbf{w}$; all eigenvalues $\ge 0$. Every covariance matrix is PSD. | A3.3 |
| Power iteration | Finding the dominant eigenvector by repeating $\mathbf{x} \leftarrow \mathbf{A}\mathbf{x}/\lVert\mathbf{A}\mathbf{x}\rVert$; converges at rate $\lvert\lambda_2/\lambda_1\rvert$. | A3.1 |
| PPO | Proximal policy optimization: a policy-gradient method with a clipped objective that limits each update. | H2.2 |
| PR-AUC | Area under the precision–recall curve; informative when positives are rare. | B4.1 |
| Precision | Of the predicted positives, the share that are truly positive. | B4.1 |
| Precision@k | The share of the top $k$ recommended or retrieved items that are relevant. | B6.2 |
| Preference tuning | Tuning a model on human (or AI) preferences between pairs of outputs (RLHF, DPO). | F3.2 |
| Pretraining | Training a model on a large general corpus with a self-supervised objective. | F2.2 |
| Projected gradient descent | Gradient descent that maps each iterate back onto the feasible set after the step. | A4.5 |
| Projection | The closest point to a vector within a subspace; the residual is orthogonal to the subspace. | A2.4 |
| Projection matrix | $\mathbf{P} = \mathbf{A}(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top$, which sends any vector to its projection onto the column space of $\mathbf{A}$. | A2.4 |
| Prompt caching | Reusing the provider's computation for an identical prompt prefix, at lower cost and latency. | I4.4 |
| Prompt injection | Input crafted to make an LLM ignore its instructions and follow the attacker's. `[RAI]` | I4.2 |
| Prompting | Writing the instructions, context and examples an LLM receives. | I1.2 |
| Pseudo-counts | The reading of a Beta($a$, $b$) prior as $a$ earlier successes and $b$ failures added to the data. | A6.5 |
| Pseudo-inverse | $\mathbf{X}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^\top$ (invert the nonzero singular values); $\mathbf{X}^+\mathbf{y}$ is the minimum-norm least-squares solution. | A3.2 |
| Pseudo-random number generator | A deterministic algorithm whose output looks random; the seed sets its starting state. | A1.1 |
| PSI | Population stability index: a drift score comparing binned distributions of a feature. | J2.2 |

## Q

| Term | Definition | Taught in |
|---|---|---|
| Q-learning | Learning action values by moving $Q(s, a)$ toward $r + \gamma \max_{a'} Q(s', a')$. | H1.3 |
| QLoRA | LoRA on top of a base model quantized to 4 bits, so fine-tuning fits on a small GPU. | J1.3 |
| QR decomposition | Writing a matrix as an orthogonal matrix times an upper-triangular one. | A3.4 |
| Quadratic form | $\mathbf{w}^\top\mathbf{A}\mathbf{w}$, a scalar; with a covariance matrix it is the variance of the data projected on $\mathbf{w}$. | A3.3, A4.2 |
| Quantile | $Q(p)$: the value below which a share $p$ of the data lies; the median is $Q(0.5)$. | A6.1 |
| Quantization | Storing weights (and sometimes activations) in fewer bits to save memory and time. | F3.4, J2.1 |
| Quasi-identifier | A column that is harmless alone but identifies people in combination with others (birth date, postcode, sex). | B1.2 |
| Quasi-Newton method | An optimizer that builds an estimate of the curvature from successive gradients to take better-scaled steps (lbfgs, scikit-learn's default for logistic regression). | B2.1 |

## R

| Term | Definition | Taught in |
|---|---|---|
| RAG | Retrieval-augmented generation: retrieving relevant documents and giving them to an LLM as context. | I2.3 |
| Random forest | An ensemble of decision trees trained on bootstrap samples with feature subsampling. | B3.2 |
| Random search | Sampling hyperparameter combinations at random; usually beats grid search for the same budget. | B4.3 |
| Random variable | A quantity whose value is determined by a random outcome. | A5.1 |
| Rank | The number of linearly independent rows or columns of a matrix. | A2.3 |
| Rate limit | A provider's cap on requests or tokens per minute. | I1.1 |
| Rayleigh quotient | $\mathbf{x}^\top\mathbf{A}\mathbf{x}/\mathbf{x}^\top\mathbf{x}$: the eigenvalue estimate for an approximate eigenvector $\mathbf{x}$. | A3.1 |
| RBF kernel | The Gaussian kernel $e^{-\gamma\lVert\mathbf{x} - \mathbf{z}\rVert^2}$: a similarity that is 1 for identical points and decays with distance at a rate set by $\gamma$. | B2.5 |
| Recall | Of the true positives, the share the model finds. | B4.1 |
| Recall@k | The share of relevant items that appear in the top $k$ results. | I2.4 |
| Receptive field | The region of the input that affects one unit of a feature map. | D1.2 |
| Recommender system | A system that ranks items for each user. | B6.1 |
| Red-teaming | Deliberately attacking a model to find harmful or unsafe behaviour before users do. `[RAI]` | F3.3 |
| Reduced row echelon form | Row echelon form in which every pivot is 1 and is the only non-zero entry in its column. | A2.3 |
| Refusal | A model declining a request it should not fulfil. `[RAI]` | F3.3 |
| Regression suite | A fixed set of tests run on every change to catch quality that got worse. | I3.3 |
| Regularization | Anything that discourages overfitting: penalties, dropout, early stopping, augmentation. | B2.2, C3.5 |
| REINFORCE | The basic policy-gradient algorithm: weight log-probability gradients by the return. | H2.1 |
| Relative error | $\lVert\mathbf{a} - \mathbf{b}\rVert / (\lVert\mathbf{a}\rVert + \lVert\mathbf{b}\rVert)$: a scale-free size of a difference; the gradient-check metric. | A4.2 |
| Reparameterization trick | Writing $z = \mu + \sigma\epsilon$ with $\epsilon \sim \mathcal{N}(0, 1)$ so gradients flow through sampling. | G1.2 |
| Replay buffer | A store of past transitions sampled at random to train DQN on less correlated data. | H1.4 |
| Reranking | Re-scoring retrieved candidates with a stronger model before using the top few. | I2.3 |
| Residual | Observed minus fitted value, $y_i - \hat{y}_i$; for a projection, $\mathbf{b} - \mathbf{p}$. | A2.4, B1.5 |
| Residual connection | Adding a block's input to its output, $\mathbf{x} + f(\mathbf{x})$, so gradients flow through deep networks. | D1.3, F1.2 |
| Residual plot | Residuals plotted against fitted values or a feature; any pattern means the model is missing something. | A2.4, B4.2 |
| ResNet | A CNN family built from residual blocks. | D1.3 |
| Resampling | Changing class frequencies by over-sampling the rare class or under-sampling the common one. | B4.4 |
| Retention (data) | How long data is kept before deletion; part of privacy governance. `[RAI]` | J2.5 |
| Retrieve-then-rank | A two-stage system: a cheap retriever finds candidates and a stronger model ranks them. | B6.2, I2.3 |
| Return | The (discounted) sum of rewards from a time step on: $G_t = \sum_k \gamma^k r_{t+k+1}$. | H1.1 |
| Reverse mode | Automatic differentiation that pushes derivatives from the output back to every input in one pass; backpropagation. | A4.3 |
| Reward hacking | An agent maximizing the reward signal in a way that defeats the designer's intent. `[RAI]` | H2.4 |
| Reward model | A model trained on human preferences to score outputs; the reward in RLHF. | H2.3 |
| RLHF | Reinforcement learning from human feedback: a reward model from preferences, then RL (usually PPO) against it. | F3.2, H2.3 |
| RNN | Recurrent neural network: applies the same cell at each time step, carrying a hidden state. | E2.1 |
| Robust statistic | A summary that barely moves when a few values are wrong or extreme, such as the median. | A6.1 |
| Robust z-score | $(x - \text{median})/(1.4826\,\mathrm{MAD})$; values with $\lvert z\rvert > 3.5$ are flagged for a look. | A6.1 |
| ROC-AUC | Area under the ROC curve: the probability a random positive is scored above a random negative. | B4.1 |
| Rollback | Returning to the previous model version after a bad deployment. | J2.4 |
| RoPE | Rotary positional embeddings: encoding position by rotating query and key vectors. | F1.2 |
| Rotation matrix | $\begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}$: turns the plane by $\theta$ counter-clockwise; determinant 1. | A2.1 |
| Rounding error | Error from storing numbers with finite precision; in a finite difference it grows like $\varepsilon\lvert f\rvert / h$. | A4.1 |
| Row echelon form | The staircase shape elimination produces: each row's first non-zero entry is right of the row above's, zero rows last. | A2.3 |
| Row operations | Swapping two rows, scaling a row, or subtracting a multiple of one row from another; they never change a system's solutions. | A2.2, A2.3 |
| Rule of succession | With a flat prior, the probability that the next trial succeeds after $k$ of $n$ is $(k + 1)/(n + 2)$. | A6.5 |
| Rule of three | After 0 events in $n$ trials, $3/n$ is roughly the upper end of a 95% interval for the rate. | A6.2 |

## S

| Term | Definition | Taught in |
|---|---|---|
| Safety (LLM) | Preventing harmful outputs and misuse of a model. `[RAI]` | F3.3 |
| Sample ratio mismatch (SRM) | A split of users between arms that differs from the planned one by more than chance; a sign the experiment is broken. | A6.3b |
| Sample standard deviation | $\sqrt{\frac{1}{n-1}\sum_i (x_i - \bar{x})^2}$; pandas' default (`ddof=1`). | A1.2, A6.2 |
| Scaled dot-product attention | $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k})\,\mathbf{V}$. | F1.1 |
| Scaling (features) | Putting features on comparable ranges: standardization or min–max scaling. | B1.3 |
| Scaling laws | Empirical power laws relating loss to parameters, data and compute. | F2.3 |
| Seasonality | A repeating pattern in a time series at a fixed period (daily, weekly, yearly). | B6.3 |
| Secant | The line through two points of a graph; its slope approaches the derivative as the points merge. | A4.1 |
| Seed | The starting value of a random number generator; fixing it makes runs repeatable. | A1.1 |
| Segmentation | Labelling every pixel of an image. | D2.4 |
| Self-attention | Attention where queries, keys and values all come from the same sequence. | F1.1 |
| Semantic caching | Reusing an earlier answer when a new query's embedding is close enough to a past one. | I4.4 |
| Semantic search | Finding documents by meaning through embeddings, not exact words. | I2.1 |
| Semantic segmentation | Labelling each pixel with a class, without separating instances. | D2.4 |
| Sensitivity | $P(\text{positive} \mid \text{condition present})$, the true positive rate; the same as recall. | A5.3, B4.1 |
| Sentinel value | A placeholder such as −999, 0 or 1900-01-01 that a source system writes for "unknown"; a disguised missing value. | B1.2 |
| seq2seq | A model that maps an input sequence to an output sequence with an encoder and a decoder. | E2.3 |
| Series | One pandas column: a 1-D array with an index. | A1.3 |
| SGD | Stochastic gradient descent: gradient descent using a random mini-batch per step. | A4.4, C3.1 |
| Shadow deployment | Running a new model on live traffic without serving its outputs, to compare it safely. | J2.4 |
| Shadow price | The rate at which the optimal value changes as a constraint is relaxed; the value of the Lagrange multiplier. | A4.5 |
| SHAP | SHapley Additive exPlanations: per-prediction feature attributions based on Shapley values. | B3.4 |
| Shape | The size of an array along each axis, written as a tuple such as `(n, d)`. | A1.2 |
| Shear | A linear map that slides points parallel to an axis by an amount proportional to their distance from it. | A2.1 |
| Shrinkage | Pulling an estimate toward a prior value, more strongly when there is little data. | A6.5 |
| Sigmoid | $\sigma(z) = 1 / (1 + e^{-z})$: squashes a number into $(0, 1)$. | B2.1 |
| Significance level | $\alpha$: the false-positive rate accepted in advance; reject $H_0$ when $p < \alpha$. | A6.3a |
| Silhouette score | A clustering score comparing each point's distance to its own cluster with the nearest other cluster. | B5.1 |
| Simpson's paradox | A comparison that holds in every subgroup and reverses when the groups are pooled, because the mix differs. | A6.1 |
| Singular matrix | A square matrix with no inverse: dependent columns, determinant 0. | A2.2, A2.3 |
| Singular value, singular vector | In $\mathbf{A}\mathbf{v}_j = \sigma_j\mathbf{u}_j$: $\sigma_j \ge 0$ is how much $\mathbf{A}$ stretches the input direction $\mathbf{v}_j$ into the output direction $\mathbf{u}_j$. | A3.2 |
| Skewness | $\mathbb{E}[((X - \mu)/\sigma)^3]$: lopsidedness; 0 if symmetric, positive with a long right tail. | A5.4, A6.1 |
| Skip-gram | word2vec's objective: predict context words from the centre word. | E1.3 |
| Soft margin | An SVM margin that allows violations, each paid for through the hinge loss at a price set by $C$. | B2.5 |
| Soft-thresholding | $S(a, t) = \operatorname{sign}(a)\max(\lvert a\rvert - t, 0)$: shrink towards zero by $t$ and stop at zero; the lasso's coordinate update. | B2.2 |
| Softmax | $\operatorname{softmax}(\mathbf{z})_k = e^{z_k} / \sum_j e^{z_j}$: turns logits into probabilities. | C1.4 |
| Spearman correlation | $\rho$: Pearson correlation of the ranks; measures monotonic trend and resists outliers. | A6.1 |
| Specification gaming | Satisfying the literal objective in an unintended way; the general form of reward hacking. `[RAI]` | H2.4 |
| Specificity | $P(\text{negative} \mid \text{condition absent})$, the true negative rate. | A5.3, B4.1 |
| Spectral theorem | A real symmetric matrix has real eigenvalues and an orthonormal basis of eigenvectors: $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$. | A3.1 |
| Standard basis vectors | $\mathbf{e}_1, \dots, \mathbf{e}_d$: a 1 in one position and 0 elsewhere; the columns of $\mathbf{I}$. | A2.1 |
| Standard deviation | The square root of the variance, in the units of the variable: the typical distance from the mean. | A5.1 |
| Standard error | The standard deviation of an estimate across repeated samples; $\sigma/\sqrt{n}$ for a sample mean. | A5.1, A5.4 |
| Standard normal | $\mathcal{N}(0, 1)$; every Gaussian is $\mu + \sigma Z$ for a standard normal $Z$. | A5.2 |
| Standardization | Shifting and scaling a feature to mean 0 and standard deviation 1. | A1.2, B1.3 |
| Steady state (Markov chain) | A probability vector $\boldsymbol{\pi}$ the transition matrix leaves unchanged: an eigenvector with eigenvalue 1. | A3.1 |
| Stratified k-fold | k-fold cross-validation that keeps the class proportions in every fold. | B4.3 |
| Stratified split | A train/validation/test split made within each class, so every set keeps the class proportions. | B1.1 |
| Streaming | Sending a response token by token as it is generated. | I1.1, I4.3 |
| Strictly convex | Convex with chords strictly above the graph between distinct points; has at most one minimizer. | A4.5 |
| Stride | How many positions a convolution or pooling window moves at each step. | D1.1 |
| Structured output | Constraining or validating an LLM's output to match a schema. | I1.3 |
| Sturges' rule | A histogram bin count of $\lceil \log_2 n \rceil + 1$. | A1.4 |
| Sub-gradient | A slope that stays below a convex function at a kink (any value between the left and right derivatives); used to minimize the hinge loss or L1 penalty. | B2.5 |
| Sum of squared residuals | $\sum_i (y_i - \hat{y}_i)^2$, the quantity least squares minimizes. | A2.4 |
| Support vector | A training point on or inside an SVM's margin; only these have non-zero dual weights and determine the boundary. | B2.5 |
| Support vector machine (SVM) | A maximum-margin classifier, optionally with kernels. | B2.5 |
| Surprisal | $-\log p(x)$: the surprise of an outcome; it adds over independent events. | A6.4 |
| SVD | Singular value decomposition: $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$, rotate, scale, rotate. | A3.2 |
| Symmetric matrix | A square matrix equal to its transpose, such as $\mathbf{X}^\top\mathbf{X}$. | A2.2 |
| System of linear equations | Equations $\mathbf{A}\mathbf{x} = \mathbf{b}$ that are linear in the unknowns $\mathbf{x}$. | A2.3 |
| System prompt | Instructions given to an LLM that set its role and rules for the conversation. | I1.1 |

## T

| Term | Definition | Taught in |
|---|---|---|
| t distribution | The distribution of $(\bar{X} - \mu)/(s/\sqrt{n})$ for Gaussian data, with heavier tails than the normal; gives the t interval. | A6.2 |
| Target encoding | Replacing a category by the mean target of its rows; must be computed out of fold, or it leaks the label. | B1.4 |
| Test set | Rows used once, at the end, to report the chosen model's performance (also holdout set). | B1.1 |
| Training set | Rows a model's parameters are fitted on. | B1.1 |
| Train/serve skew | A feature computed differently in training and in production, so the model sees different inputs live. | B1.3, J2.2 |
| Train/test contamination | Leakage through a fitting step (scaler, selector, encoder) that learned from the evaluation rows. | B1.4 |
| True risk | A model's expected loss on new data from the same source; estimated by the empirical risk on held-out rows. | B1.1 |
| t-SNE | A non-linear 2-D embedding that preserves local neighbourhoods; distances between clusters mean little. | B5.5 |
| t-test | A test of whether two means differ, using the t distribution (Welch's version allows unequal variances). | A6.3a |
| T5 | An encoder–decoder transformer that casts every task as text to text. | F2.1 |
| Target leakage | A feature that encodes the target or is only known after the outcome. | B1.4 |
| Target network | A slowly updated copy of the Q-network used to compute stable DQN targets. | H1.4 |
| Taylor's formula | $f(x + h) \approx f(x) + f'(x)h + \frac{1}{2}f''(x)h^2 + \dots$: a function near a point as a polynomial in the step. | A4.1 |
| Teacher forcing | Feeding the true previous token, not the model's own prediction, to the decoder during training. | E2.3 |
| Temperature | Dividing logits by $\tau$ before softmax; higher is more random, lower is more deterministic. | F2.4 |
| Temporal leakage | Using information from the future when predicting the past, often through a random split of time-ordered data. | B1.4 |
| TF-IDF | Term frequency times inverse document frequency: word weights that downweight common words. | E1.2 |
| Thin SVD | The SVD keeping only $r = \min(n, d)$ columns of $\mathbf{U}$: shapes `(n, r)(r, r)(r, d)`; `full_matrices=False`. | A3.2 |
| Threshold tuning | Choosing the probability cut-off for a positive prediction to match the costs of errors. | B4.4 |
| Throughput | Work done per unit time (examples per second, tokens per second). | C4.1, I4.3 |
| Time-series cross-validation | Validation that always trains on the past and tests on the following period. | B4.3, B6.3 |
| Token | The unit a language model reads and writes: a word, subword or byte sequence. | E1.1, F1.3, I1.1 |
| Tokenization | Splitting text into tokens and mapping them to integer ids. | E1.1, F1.3 |
| Tool use | An LLM requesting a function call with arguments; the application runs it and returns the result. | I1.4 |
| Top-k sampling | Sampling the next token only from the $k$ most probable tokens. | F2.4 |
| Top-p (nucleus) sampling | Sampling from the smallest set of tokens whose probabilities sum to at least $p$. | F2.4 |
| Trace | $\operatorname{tr}(\mathbf{A}) = \sum_i A_{ii}$, the sum of the diagonal; equals the sum of the eigenvalues. | A3.1 |
| Tracing | Recording each step of a request (model calls, tools, retrieval) with timing, as linked spans. | I4.5 |
| Train/validation/test split | Separate data for fitting, for choosing models and settings, and for the final unbiased estimate. | B1.1 |
| Trajectory (agent) | The sequence of steps, tool calls and observations an agent took on a task. | I3.3 |
| Transfer learning | Reusing a model pretrained on one task as the starting point for another. | D2.1 |
| Transformer | An architecture of stacked self-attention and MLP blocks with residuals and normalization. | F1.2 |
| Transpose | $\mathbf{A}^\top$: rows become columns. | A2.2 |
| Trend | The long-term direction of a time series. | B6.3 |
| Trimmed mean | The mean after dropping a fixed share of the smallest and largest values. | A6.1 |
| Truncated SVD | Keeping only the top $k$ singular values and vectors; `sklearn.decomposition.TruncatedSVD` computes it without centering. | A3.2 |
| Truncation error | Error from cutting off the Taylor series; in a finite difference it shrinks as $h$ shrinks. | A4.1 |
| Tukey's fences | $[Q_1 - 1.5\,\mathrm{IQR},\ Q_3 + 1.5\,\mathrm{IQR}]$: values outside are flagged as possible outliers. | A6.1 |
| Type I and type II errors | A false positive (rejecting a true $H_0$, rate $\alpha$) and a false negative (missing a real effect, rate $\beta$). | A6.3a |

## U

| Term | Definition | Taught in |
|---|---|---|
| Unbiased estimator | An estimator whose average over repeated samples equals the true value. | A5.1 |
| U-Net | An encoder–decoder CNN with skip connections, used for segmentation and in diffusion models. | D2.4, G1.4 |
| UMAP | A non-linear dimensionality reduction that preserves local structure, faster than t-SNE. | B5.5 |
| Underfitting | A model too simple to capture the pattern, so both training and test error are high. | B4.5 |
| Uniform distribution | Every value in a range is equally likely. | A5.2 |
| Unit vector | A vector of length 1: $\mathbf{a} / \lVert\mathbf{a}\rVert_2$. | A1.5 |
| uv | The Python package and environment manager this course uses. | A1.1 |

## V

| Term | Definition | Taught in |
|---|---|---|
| VAE | Variational autoencoder: an autoencoder whose latent space is a distribution, trained on the ELBO. | G1.2 |
| Validation curve | Training and validation score plotted against one hyperparameter. | B4.5 |
| Validation set | Held-out rows used to compare models and settings; optimistic after many comparisons. | B1.1 |
| Value function | The expected return from a state, $V^\pi(s)$, or from a state and action, $Q^\pi(s, a)$. | H1.1 |
| Value iteration | Repeatedly applying the Bellman optimality update until values converge. | H1.2 |
| Vanishing gradients | Gradients that shrink exponentially through layers or time steps, so early parts stop learning. | C3.3, E2.1 |
| Variance | $\operatorname{Var}[X] = \mathbb{E}[(X - \mu)^2]$: the spread of a random variable. | A5.1 |
| Vectorization | Replacing Python loops with whole-array operations. | A1.2 |
| Vector–Jacobian product (VJP) | $\bar{\mathbf{y}}^\top\mathbf{J}$: what one reverse-mode step computes, without forming $\mathbf{J}$. | A4.3 |
| View (NumPy) | An array that shares memory with another; basic slicing returns one. | A1.2 |
| Virtual environment | A project-local Python installation with its own packages (`.venv/`). | A1.1 |
| Vision-language model (VLM) | A model that takes images and text as input and generates text. | G2.3 |
| Vision Transformer (ViT) | A transformer that treats image patches as tokens. | G2.1 |

## W

| Term | Definition | Taught in |
|---|---|---|
| Walk-forward validation | Repeatedly training on data up to a time and testing on the next period, moving forward. | B6.3 |
| Warmup | Starting with a small learning rate and increasing it over the first steps. | C3.2 |
| Watermarking | Embedding a detectable signal in generated content to mark it as synthetic. `[RAI]` | G2.4 |
| Weight decay | Shrinking weights a little at each step; equivalent to L2 regularization for plain SGD. | C3.5 |
| Welch's t-test | The two-sample t-test that does not assume equal variances; the default. | A6.3a |
| Wilson interval | A confidence interval for a proportion that stays inside $[0, 1]$ and works near 0 and 1. | A6.2 |
| word2vec | Learning word embeddings by predicting words from their context (skip-gram, CBOW). | E1.3 |
| Workflow (vs agent) | A fixed sequence of LLM and code steps, chosen when the path is known in advance. | I3.1 |

## X–Z

| Term | Definition | Taught in |
|---|---|---|
| Xavier (Glorot) initialization | Weight initialization with variance $2 / (d_{\text{in}} + d_{\text{out}})$, suited to tanh layers. | C3.3 |
| XGBoost | A widely used gradient-boosting library with regularized trees. | B3.3 |
| Yates' correction | A continuity correction SciPy applies by default to 2×2 chi-square tests; it makes them slightly conservative. | A6.3a |
| Zero-shot classification | Classifying into labels the model was not trained on, for example by comparing CLIP image and text embeddings. | G2.2 |
