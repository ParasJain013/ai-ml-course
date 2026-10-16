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
| A/B test | A randomized experiment that splits users between two variants and tests whether a metric differs. | A6.3 |
| Accuracy | The fraction of predictions that are correct; misleading when classes are imbalanced. | B4.1 |
| Activation function | The elementwise non-linearity applied after a layer's linear map (ReLU, sigmoid, tanh, GELU). | C1.1 |
| Actor–critic | An RL method with a policy (actor) and a learned value function (critic) that reduces the variance of policy gradients. | H2.2 |
| Adam | An optimizer that scales each parameter's step by running estimates of the gradient's mean and squared magnitude. | C3.1 |
| AdamW | Adam with weight decay applied directly to the weights instead of through the gradient. | C3.1 |
| Advantage | How much better an action is than the policy's average in that state: $A(s, a) = Q(s, a) - V(s)$. | H2.2 |
| Agent (LLM) | An LLM in a loop that chooses tools, observes their results and decides the next step until a task is done. | I3.1 |
| Alerting | Automatically notifying a person when a monitored metric crosses a threshold. | J2.2 |
| Alignment | Making a model's behaviour match what its developers and users intend, including refusing harmful requests. | F3.3 |
| Anomaly detection | Finding examples that differ markedly from the bulk of the data. | B5.6 |
| ARIMA | AutoRegressive Integrated Moving Average: a classical time-series model built from lags, differencing and past errors. | B6.4 |
| Artifact (MLOps) | A file produced by a run, such as a model, a plot or a dataset snapshot, stored with the run that made it. | J1.1 |
| Attention | A mechanism that computes a weighted average of values, with weights from how well a query matches each key. | E2.4, F1.1 |
| Augmentation | Creating modified training examples (flips, crops, colour shifts) so a model learns invariances and overfits less. | D2.2 |
| Autocast | PyTorch's context manager that runs selected operations in a lower-precision dtype for mixed precision. | C4.2 |
| Autoencoder | A network trained to reconstruct its input through a narrow bottleneck, learning a compressed representation. | G1.1 |
| Autograd | Automatic differentiation: recording operations on a graph and applying the chain rule backward to get gradients. | C1.3, C2.1 |

## B

| Term | Definition | Taught in |
|---|---|---|
| Backpropagation | Computing every parameter's gradient by applying the chain rule backward through the network. | C1.2 |
| Backpropagation through time (BPTT) | Backpropagation through an RNN unrolled over the time steps of a sequence. | E2.1 |
| Bag-of-words | Representing a text by the counts of its words, ignoring their order. | E1.2 |
| Bagging | Bootstrap aggregating: training models on bootstrap resamples and averaging their predictions. | B3.2 |
| Base rate | The prior probability of a condition in the population, before any evidence. | A5.3 |
| Baseline | The simplest reasonable model or rule that a real model must beat to be worth deploying. | B1.1 |
| Batching (serving) | Grouping several requests into one model call to raise throughput. | J2.1, I4.3 |
| BatchNorm | Normalizing each feature over the batch, then applying a learned scale and shift. | C3.4 |
| Bayes' theorem | $P(A \mid B) = P(B \mid A)\,P(A) / P(B)$: how to update a belief with evidence. | A5.3 |
| Bayesian inference | Treating parameters as random variables and computing their posterior distribution given data. | A6.5 |
| Bayesian optimization | Hyperparameter search that fits a model of the score and picks the next trial where it looks most promising. | B4.3 |
| Beam search | Decoding that keeps the $k$ most probable partial sequences at each step. | E2.3, F2.4 |
| Bellman equation | The recursive relation between a state's value and the values of the states that follow it. | H1.1 |
| Bernoulli distribution | The distribution of a single yes/no outcome with success probability $p$. | A5.2 |
| BERT | An encoder-only transformer pretrained with masked language modelling. | F2.1 |
| Bias (statistical) | The systematic gap between an estimator's or model's expected value and the truth. | B4.5 |
| Bias (fairness) | Systematically different model behaviour or errors across groups of people. | B4.6 |
| Bias–variance trade-off | Test error splits into bias (too simple), variance (too sensitive to the training set) and noise. | B4.5 |
| Binomial distribution | The number of successes in $n$ independent Bernoulli trials. | A5.2 |
| Bonferroni correction | Testing each of $m$ hypotheses at $\alpha / m$ to control the chance of any false positive. | A6.3 |
| Bootstrap | Estimating the uncertainty of a statistic by recomputing it on resamples drawn with replacement. | A6.2 |
| BPE (byte-pair encoding) | A subword tokenizer that repeatedly merges the most frequent adjacent pair of symbols. | F1.3 |
| Broadcasting | NumPy's rule for combining arrays of different shapes by virtually repeating size-1 axes. | A1.2 |

## C

| Term | Definition | Taught in |
|---|---|---|
| Caching (LLM) | Reusing earlier results instead of paying for a new model call; see exact, prompt and semantic caching. | I4.4 |
| Calibration | Whether predicted probabilities match observed frequencies (of cases predicted 0.8, about 80% are positive). | B4.1 |
| Canary rollout | Sending a small share of live traffic to a new model version before a full rollout. | J2.4 |
| Categorical distribution | The distribution of one draw from $K$ classes with probabilities $p_1, \dots, p_K$. | A5.2 |
| Categorical encoding | Turning category labels into numbers: one-hot, ordinal or target encoding. | B1.3 |
| Causal masking | Blocking attention to future positions so each token sees only earlier ones. | F1.1 |
| Central limit theorem (CLT) | The mean of many independent draws is approximately Gaussian, whatever the original distribution. | A5.4 |
| Chain rule | The derivative of a composition is the product of the derivatives of its parts. | A4.3 |
| Chat template | The exact text format that wraps system, user and assistant messages for a given model. | F3.1 |
| Checkpointing (gradient) | Saving memory by recomputing some activations during the backward pass instead of storing them. | C4.2 |
| Chinchilla (compute-optimal) | The finding that, for a fixed compute budget, parameters and training tokens should grow together. | F2.3 |
| Chi-square test | A test of whether observed counts in categories differ from the counts expected under independence. | A6.3 |
| Cholesky decomposition | Writing a symmetric positive-definite matrix as $\mathbf{L}\mathbf{L}^\top$ with $\mathbf{L}$ lower triangular. | A3.4 |
| Chunking | Splitting documents into pieces small enough to embed and retrieve. | I2.1 |
| CI for ML | Continuous integration that also runs data tests, model tests and evals before a change can merge. | J2.3 |
| Citation (RAG) | Pointing each claim in a generated answer to the retrieved source it came from. | I2.3 |
| Class imbalance | One class is much rarer than the others, so accuracy and default thresholds mislead. | B4.4 |
| Class weights | Weighting the loss so errors on rare classes cost more. | B4.4 |
| Classifier-free guidance | Steering a diffusion model by mixing its conditional and unconditional predictions. | G2.4 |
| CLIP | A model that trains image and text encoders contrastively so matching pairs have similar embeddings. | G2.2 |
| Collaborative filtering | Recommending items from the behaviour of similar users or items, without item content. | B6.1 |
| Computational graph | A graph of operations whose edges carry values forward and gradients backward. | A4.3 |
| Concept drift | The relationship between inputs and the target changes over time. | J2.2 |
| Condition number | How much a matrix amplifies relative error when you solve with it; large means unstable. | A3.4 |
| Conditional probability | $P(A \mid B)$: the probability of $A$ given that $B$ happened. | A5.3 |
| Confidence interval | A range computed from data by a method that contains the true value in a stated share (say 95%) of repeated samples. | A6.2 |
| Confusion matrix | A table of true versus predicted classes. | B4.1 |
| Conjugate prior | A prior that gives a posterior in the same family, such as Beta prior with Binomial data. | A6.5 |
| Constrained optimization | Minimizing a function subject to equality or inequality constraints. | A4.5 |
| Context assembly | Choosing, ordering and formatting retrieved chunks into the prompt within a token budget. | I2.3 |
| Contrastive learning | Training embeddings so matching pairs are close and non-matching pairs are far apart. | G2.2 |
| Convexity | A function is convex if every chord lies above it; then any local minimum is global. | A4.5 |
| Convolution | Sliding a small kernel over an input and taking a weighted sum at each position. | D1.1 |
| Correlation | A standardized measure (−1 to 1) of how two variables move together; not evidence of causation. | A6.1 |
| Cosine decay | A learning-rate schedule that falls along a half cosine curve. | C3.2 |
| Cosine similarity | The cosine of the angle between two vectors: $\mathbf{a}\cdot\mathbf{b} / (\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert)$. | A1.5 |
| Covariance | $\mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$: how two variables vary together. | A5.1 |
| Covariance matrix | The `(d, d)` matrix of pairwise covariances between features; symmetric and positive semi-definite. | A3.3 |
| Cross-encoder | A model that reads a query and a document together to score their relevance; used for reranking. | I2.3 |
| Cross-entropy | $H(p, q) = -\sum_x p(x) \log q(x)$: the expected cost of coding $p$ with $q$; the classification loss. | A6.4, C1.4 |
| Cross-validation | Estimating generalization by training and scoring on several train/validation splits of the data. | B4.3 |
| Curse of dimensionality | In high dimensions points are all far apart and distances lose contrast. | B2.3 |
| CutMix | Augmentation that pastes a patch of one image into another and mixes the labels by area. | D2.2 |

## D

| Term | Definition | Taught in |
|---|---|---|
| Data cleaning | Fixing types, missing values, outliers and duplicates before modelling. | B1.2 |
| Data drift | The distribution of model inputs changes over time. | J2.2 |
| Data leakage | Information unavailable at prediction time reaches training or evaluation, inflating scores. | B1.4 |
| Data minimization | Collecting and keeping only the personal data a task needs. `[RAI]` | B1.2 |
| Data parallel | Copying the model to several devices, each processing part of the batch, then averaging gradients. | C4.4 |
| Data versioning | Tracking exact dataset snapshots so any model can be traced to the data that trained it. | J1.2 |
| DataLoader | PyTorch's iterator that batches, shuffles and loads examples from a Dataset. | C2.2 |
| Dataset (PyTorch) | A class with `__len__` and `__getitem__` that returns one example. | C2.2 |
| Datasheet | Documentation of a dataset's motivation, collection, composition and recommended uses. `[RAI]` | J2.5 |
| DBSCAN | Density-based clustering that grows clusters from dense regions and marks sparse points as noise. | B5.2 |
| DCGAN | A GAN that uses convolutional generator and discriminator networks. | G1.3 |
| DDP | Distributed Data Parallel: PyTorch's multi-process data-parallel training. | C4.4 |
| DDPM | Denoising Diffusion Probabilistic Model: a diffusion model trained to predict the noise added at each step. | G1.4 |
| Decision tree | A model that splits the data by feature thresholds into leaves with simple predictions. | B3.1 |
| Decoder-only model | A transformer with causal self-attention that predicts the next token (GPT). | F2.1 |
| Decoding | Turning a language model's next-token distributions into text. | F2.4 |
| Deepfake | Realistic synthetic media of a real person, made without their consent or to deceive. `[RAI]` | G2.4 |
| Demographic parity | A fairness criterion: the positive-prediction rate is equal across groups. `[RAI]` | B4.6 |
| Dense retrieval | Retrieval by nearest neighbours in embedding space. | I2.1 |
| Depthwise separable convolution | A convolution split into a per-channel spatial filter and a 1×1 channel mix, using far fewer parameters. | D1.4 |
| Derivative | The instantaneous rate of change of a function; its local slope. | A4.1 |
| Descriptive statistics | Numbers that summarize data: centre, spread, shape. | A6.1 |
| Dice coefficient | Segmentation overlap: twice the intersection divided by the sum of the two areas. | D2.4 |
| Differential privacy | A guarantee that an algorithm's output changes very little whether or not any one person's data is included. `[RAI]` | J2.5 |
| Diffusion model | A generative model that learns to reverse a gradual noising process. | G1.4 |
| Dimensionality reduction | Mapping data to fewer dimensions while keeping its structure (PCA, t-SNE, UMAP). | B5.4 |
| Discount factor | $\gamma \in [0, 1]$: how much an RL agent values future rewards relative to immediate ones. | H1.1 |
| Distributed training | Training one model across several devices or machines. | C4.4 |
| Docker image | A packaged, versioned file system and start command that runs the same everywhere. | J2.4 |
| Dot product | $\mathbf{a}\cdot\mathbf{b} = \sum_j a_j b_j$. | A1.5 |
| DPO | Direct Preference Optimization: tuning a model on preference pairs with a classification-style loss, no RL loop. | F3.2 |
| DQN | Deep Q-Network: Q-learning with a neural network, a replay buffer and a target network. | H1.4 |
| Dropout | Randomly zeroing activations during training so the network cannot rely on any one unit. | C3.5 |
| Dynamic programming (RL) | Computing exact values or policies from a known model of the environment by repeated Bellman updates. | H1.2 |

## E

| Term | Definition | Taught in |
|---|---|---|
| Early stopping | Stopping training when the validation score stops improving. | C3.5 |
| EDA | Exploratory data analysis: summarizing and plotting data to understand it before modelling. | A6.1 |
| Effect size | How large a difference is, not just whether it is statistically significant. | A6.3 |
| EfficientNet | A CNN family that scales depth, width and resolution together. | D1.4 |
| Eigenvalue, eigenvector | $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$: a direction $\mathbf{v}$ the matrix only stretches, by $\lambda$. | A3.1 |
| Elastic net | Regularization with both L1 and L2 penalties. | B2.2 |
| ELBO | Evidence lower bound: the objective a VAE maximizes, reconstruction minus a KL term. | G1.2 |
| Elbow method | Choosing $k$ for k-means where adding clusters stops reducing inertia much. | B5.1 |
| EM algorithm | Expectation–maximization: alternating soft assignments and parameter updates to fit latent-variable models. | B5.3 |
| Embedding | A dense vector that represents an item (word, document, image) so similar items are close. | E1.3, I2.1 |
| Encoder–decoder | An architecture where an encoder reads the input and a decoder generates the output conditioned on it. | E2.3, F2.1 |
| Entropy | $H(p) = -\sum_x p(x) \log p(x)$: the average surprise of a distribution. | A6.4 |
| Epsilon-greedy | Exploring by taking a random action with probability $\epsilon$, otherwise the best-known action. | H1.3 |
| Equalized odds | A fairness criterion: true-positive and false-positive rates are equal across groups. `[RAI]` | B4.6 |
| Evals | Systematic tests that score an ML or LLM system against a fixed set of cases. | I2.4, I3.3 |
| Expectation | $\mathbb{E}[X]$: the probability-weighted average of a random variable. | A5.1 |
| Experiment tracking | Recording each training run's parameters, metrics and artifacts so runs can be compared and reproduced. | J1.1 |
| Explained variance | The share of total variance captured by each principal component. | B5.4 |
| Exploding gradients | Gradients that grow exponentially through layers or time steps, making training diverge. | C3.3, E2.1 |
| Exponential smoothing | Forecasting with exponentially decaying weights on past observations. | B6.4 |

## F

| Term | Definition | Taught in |
|---|---|---|
| F1 score | The harmonic mean of precision and recall. | B4.1 |
| Fairness | The absence of unjustified differences in a model's outcomes or errors across groups. `[RAI]` | B4.6 |
| Faithfulness | Whether a generated answer is supported by the retrieved context it claims to use. | I2.4 |
| fastText | Word embeddings built from character n-grams, so unseen words still get vectors. | E1.4 |
| Feature engineering | Building model inputs from raw data: scaling, encoding, dates, interactions. | B1.3 |
| Feature extraction (transfer) | Using a frozen pretrained network's outputs as features for a new small model. | D2.1 |
| Feature subsampling | Each tree or split considers only a random subset of features (random forests). | B3.2 |
| Few-shot prompting | Including a few worked examples in the prompt. | I1.2 |
| Fine-tuning | Continuing to train a pretrained model on a new task's data. | D2.1, J1.3 |
| Finite differences | Approximating a derivative by $(f(x + h) - f(x - h)) / 2h$; used to check gradients. | A4.1 |
| FlashAttention | An exact attention algorithm that avoids storing the full attention matrix by working in tiles. | F2.5 |
| FSDP | Fully Sharded Data Parallel: sharding parameters, gradients and optimizer state across devices. | C4.4 |
| Function calling | See tool use. | I1.4 |

## G

| Term | Definition | Taught in |
|---|---|---|
| GAN | Generative adversarial network: a generator and a discriminator trained against each other. | G1.3 |
| Gaussian (normal) distribution | The bell-shaped distribution $\mathcal{N}(\mu, \sigma^2)$. | A5.2 |
| Gaussian mixture model (GMM) | A density modelled as a weighted sum of Gaussians; soft clustering. | B5.3 |
| Generalization | How well a model performs on new data from the same distribution, not just its training data. | B1.1 |
| Golden set | A curated, versioned set of inputs with known good outputs used to evaluate a system. | I2.4 |
| GPT | A decoder-only transformer pretrained on next-token prediction. | F1.4, F2.1 |
| Grad-CAM | A heat map of which image regions most influenced a CNN's prediction, from gradients of the last conv layer. | D2.5 |
| Gradient | The vector of partial derivatives; it points in the direction of steepest increase. | A4.2 |
| Gradient accumulation | Summing gradients over several small batches before one optimizer step to simulate a larger batch. | C4.2 |
| Gradient boosting | Building an ensemble by fitting each new tree to the gradient (residuals) of the current ensemble's loss. | B3.3 |
| Gradient clipping | Rescaling gradients whose norm exceeds a threshold. | E2.1 |
| Gradient descent | Repeatedly stepping parameters against the gradient: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\nabla\mathcal{L}$. | A4.4 |
| Grid search | Trying every combination of hyperparameter values from a grid. | B4.3 |
| Group k-fold | Cross-validation that keeps all rows of a group (a user, a patient) in the same fold. | B4.3 |
| Grouped-query attention (GQA) | Attention where groups of query heads share key and value heads, shrinking the KV cache. | F2.5 |
| GRU | Gated recurrent unit: an RNN cell with update and reset gates. | E2.2 |
| Guardrails | Checks on an LLM system's inputs and outputs that block or fix unsafe or invalid content. `[RAI]` | I4.1 |
| Gymnasium | The standard Python API for RL environments (`reset`, `step`). | H1.3 |

## H

| Term | Definition | Taught in |
|---|---|---|
| Hallucination | A model stating something fluent and confident that is false or unsupported. `[RAI]` | F3.3 |
| He initialization | Weight initialization with variance $2 / d_{\text{in}}$, suited to ReLU layers. | C3.3 |
| Hierarchical clustering | Building a tree of clusters by repeatedly merging (or splitting) groups. | B5.2 |
| Hinge loss | $\max(0, 1 - y z)$: the SVM loss that penalizes points inside the margin. | B2.5 |
| HNSW | Hierarchical Navigable Small World: a graph index for fast approximate nearest-neighbour search. | I2.2 |
| Holdout set | Data kept aside and used only for the final evaluation. | B1.1 |
| Hybrid search | Combining keyword (full-text) and vector search results. | I2.2 |
| Hyperparameter | A setting chosen before training (learning rate, depth), not learned from data. | B4.3 |
| Hypothesis test | A procedure that decides whether data are surprising enough under a null hypothesis to reject it. | A6.3 |

## I

| Term | Definition | Taught in |
|---|---|---|
| Identity matrix | $\mathbf{I}$: ones on the diagonal, zeros elsewhere; $\mathbf{A}\mathbf{I} = \mathbf{A}$. | A2.2 |
| Implicit feedback | User signals such as clicks and views, which show interest but not dislike. | B6.1 |
| Indirect prompt injection | Malicious instructions hidden in content the model reads (a web page, an email, a tool result). `[RAI]` | I4.2 |
| Inference server (LLM) | A service optimized to run an LLM for many requests (batching, KV cache management). | J2.1 |
| Initialization | How weights are set before training; it controls whether signals vanish or explode. | C3.3 |
| Instance segmentation | Labelling each pixel with the object instance it belongs to. | D2.4 |
| Instruction tuning (SFT) | Supervised fine-tuning on instruction–response pairs so a base model follows instructions. | F3.1 |
| Interpretability | Methods that explain what a model relies on and why it made a prediction. | B3.4 |
| Inverse matrix | $\mathbf{A}^{-1}$ with $\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$; exists only for square full-rank matrices. | A2.2 |
| IoU | Intersection over union: the overlap of two boxes or masks divided by their union. | D2.3 |
| Isolation forest | Anomaly detection that scores points by how few random splits isolate them. | B5.6 |
| IVFFlat | An index that clusters vectors and searches only the nearest clusters. | I2.2 |

## J

| Term | Definition | Taught in |
|---|---|---|
| Jacobian | The matrix of all partial derivatives of a vector-valued function; shape `(m, n)` for $f: \mathbb{R}^n \to \mathbb{R}^m$. | A4.2 |
| JSON schema | A machine-readable description of the structure a JSON value must have. | I1.3 |
| Jupyter | An interactive notebook environment that runs code in cells. | A1.1 |

## K

| Term | Definition | Taught in |
|---|---|---|
| k-fold cross-validation | Splitting data into $k$ folds and validating on each in turn. | B4.3 |
| k-means | Clustering that alternates assigning points to the nearest centroid and recomputing centroids. | B5.1 |
| k-nearest neighbours (kNN) | Predicting from the labels of the $k$ closest training examples. | B2.3 |
| Kernel (convolution) | The small weight array slid over the input in a convolution. | D1.1 |
| Kernel trick | Computing inner products in a high-dimensional feature space through a kernel function, without the features. | B2.5 |
| KL divergence | $D_{\mathrm{KL}}(p \,\|\, q) = \sum_x p(x) \log (p(x) / q(x))$: the extra cost of using $q$ for $p$; not symmetric. | A6.4 |
| KL penalty (RLHF) | A term that keeps the tuned policy close to the reference model. | H2.3 |
| KS test | Kolmogorov–Smirnov test: compares two distributions by the largest gap between their CDFs. | J2.2 |
| KV cache | Storing past keys and values during generation so each new token costs one step, not a full recompute. | F2.4 |

## L

| Term | Definition | Taught in |
|---|---|---|
| L1 regularization (lasso) | Penalizing $\lambda\lVert\mathbf{w}\rVert_1$; pushes weights to exactly zero. | B2.2 |
| L2 regularization (ridge) | Penalizing $\lambda\lVert\mathbf{w}\rVert_2^2$; shrinks weights; equals a Gaussian prior. | B2.2, A5.5 |
| Lagrange multiplier | A variable that turns a constrained optimization into an unconstrained one at a stationary point. | A4.5 |
| Lag feature | A past value of a time series used as a feature for predicting a later value. | B6.3 |
| Latency | The time from a request to its response; reported as percentiles (p50, p95). | I4.3 |
| Latent diffusion | Diffusion run in an autoencoder's latent space instead of pixel space. | G2.4 |
| Law of large numbers | The sample mean converges to the expectation as the sample grows. | A5.4 |
| LayerNorm | Normalizing each example's features to zero mean and unit variance, then scaling and shifting. | C3.4 |
| Learning curve | Training and validation score plotted against training-set size (or epochs). | B4.5 |
| Learning rate | $\eta$: the step size of gradient descent. | A4.4 |
| Least privilege | Giving each tool or credential only the permissions its task needs. `[RAI]` | I4.2 |
| Least squares | Fitting by minimizing the sum of squared residuals. | A2.4 |
| LeNet | An early CNN for digit recognition (convolution, pooling, fully connected). | D1.3 |
| LightGBM | A fast gradient-boosting library with histogram-based, leaf-wise tree growth. | B3.3 |
| Linear map | A function $f(\mathbf{x}) = \mathbf{A}\mathbf{x}$ that preserves addition and scaling. | A2.1 |
| Linear regression | Predicting a number as a weighted sum of features plus a bias. | B1.5 |
| LLM-as-judge | Using an LLM with a rubric to grade outputs; it has position, length and self-preference biases. | I2.4 |
| Logistic regression | A linear model whose output goes through a sigmoid to give a probability. | B2.1 |
| Log-loss | Binary cross-entropy: $-[y \log p + (1 - y)\log(1 - p)]$. | B2.1, A5.5 |
| Long context | Handling inputs of many thousands of tokens, limited by attention's cost and the KV cache. | F2.5 |
| LoRA | Low-Rank Adaptation: fine-tuning by learning a low-rank update $\mathbf{B}\mathbf{A}$ added to frozen weights. | J1.3 |
| Loss scaling | Multiplying the loss before backward in fp16 so small gradients do not underflow. | C4.2 |
| LR range test | Increasing the learning rate over one short run to find the range where loss falls fastest. | C3.2 |
| LSTM | Long short-term memory: an RNN cell with input, forget and output gates and a cell state. | E2.2 |

## M

| Term | Definition | Taught in |
|---|---|---|
| mAP | Mean average precision: the detection metric averaging precision over recall levels, classes and IoU thresholds. | D2.3 |
| MAP estimation | Maximum a posteriori: choosing parameters that maximize likelihood times prior. | A5.5 |
| Masked language modelling | Predicting hidden tokens from both sides of the context (BERT's objective). | F2.1 |
| Matrix factorization | Approximating a matrix as a product of two low-rank matrices (users × factors, factors × items). | B6.1 |
| Matrix multiplication | $(\mathbf{A}\mathbf{B})_{ij} = \sum_k A_{ik} B_{kj}$; shapes `(n, d) @ (d, k) -> (n, k)`. | A2.2 |
| Maximum margin | The SVM's goal: the separating boundary farthest from the nearest points of each class. | B2.5 |
| MCP | Model Context Protocol: an open protocol for exposing tools, resources and prompts to LLM applications. | I3.2 |
| MDP | Markov decision process: states, actions, transition probabilities, rewards and a discount factor. | H1.1 |
| Mean squared error (MSE) | The average squared difference between predictions and targets. | B4.2, A5.5 |
| Memory (agent) | State an agent carries across steps or sessions: the conversation, notes or retrieved facts. | I3.1 |
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
| Multiple comparisons | Running many tests inflates the chance of false positives unless corrected. | A6.3 |
| Multivariate Gaussian | $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$: a Gaussian over vectors, with mean `(d,)` and covariance `(d, d)`. | A5.2 |

## N

| Term | Definition | Taught in |
|---|---|---|
| n-gram | A sequence of $n$ consecutive tokens. | E1.1 |
| Naive Bayes | A classifier that applies Bayes' theorem assuming features are independent given the class. | B2.4 |
| NDCG | Normalized discounted cumulative gain: a ranking metric that rewards relevant items near the top. | B6.2 |
| Negative sampling | Training embeddings by contrasting each true pair with a few random "negative" pairs. | E1.3 |
| Non-maximum suppression (NMS) | Removing overlapping detection boxes, keeping the highest-scoring one. | D2.3 |
| Norm | The length of a vector: L2 $\sqrt{\sum_j x_j^2}$, L1 $\sum_j \lvert x_j \rvert$. | A1.5 |
| Normal equations | $\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}$: the closed-form least-squares solution. | A2.4 |
| Normalization (layers) | Rescaling activations to a stable range inside a network (BatchNorm, LayerNorm). | C3.4 |
| Normalization (text) | Lower-casing, Unicode normalization and similar cleanup before tokenizing. | E1.1 |
| Numerical stability | Computing so that rounding errors and overflow do not ruin the result. | A3.4, C1.4 |

## O

| Term | Definition | Taught in |
|---|---|---|
| Object detection | Finding each object in an image as a class label and a bounding box. | D2.3 |
| Observability | Tracing, logging and metrics that let you see what a system did and why. | I4.5 |
| One-cycle schedule | A learning rate that rises then falls over one cycle of training. | C3.2 |
| One-hot encoding | Representing a category as a vector with a 1 in its position and 0 elsewhere. | B1.3 |
| Online evals | Scoring a live system's real traffic, often by sampling and judging. | I4.5 |
| ONNX | Open Neural Network Exchange: a portable model format that many runtimes can execute. | J2.1 |
| Optimizer | The rule that turns gradients into parameter updates (SGD, Adam). | C3.1 |
| Out-of-bag error | A random forest's error on the rows each tree did not see in its bootstrap sample. | B3.2 |
| Outlier | A value far from the bulk of the data. | B1.2 |
| Overfitting | Fitting noise in the training data, so training error is low but test error is high. | B4.5 |

## P

| Term | Definition | Taught in |
|---|---|---|
| p-value | The probability of data at least as extreme as observed if the null hypothesis were true. | A6.3 |
| Padding (convolution) | Adding a border of zeros around the input to control the output size. | D1.1 |
| Partial dependence | The average prediction as one feature varies, with the others left as observed. | B3.4 |
| Partial derivative | The derivative with respect to one variable, holding the others fixed. | A4.1 |
| PCA | Principal component analysis: projecting data onto the directions of greatest variance. | B5.4 |
| Perceptron | A single linear unit followed by a step function; the first neural network. | C1.1 |
| Permutation importance | The drop in score when one feature's values are shuffled. | B3.4 |
| Permutation test | A hypothesis test that builds the null distribution by shuffling labels. | A6.3 |
| Perplexity | $\exp$(average cross-entropy per token); lower means the model is less surprised. | F2.2 |
| pgvector | A PostgreSQL extension that stores vectors and runs similarity search. | I2.2 |
| PII | Personally identifiable information: data that identifies a person. `[RAI]` | B1.2, I4.1 |
| Pipeline (scikit-learn) | A chain of preprocessing steps and a model fitted and applied as one unit. | B1.3 |
| Poisson distribution | Counts of events in a fixed interval, with mean rate $\lambda$. | A5.2 |
| Policy | The agent's rule for choosing actions: $\pi(a \mid s)$. | H1.1 |
| Policy gradient | Improving a policy by gradient ascent on expected return, using sampled episodes. | H2.1 |
| Policy iteration | Alternating policy evaluation and greedy policy improvement until the policy stops changing. | H1.2 |
| Pooling | Downsampling a feature map by taking the max or mean over small windows. | D1.2 |
| Positional encoding | Information about token position added to a transformer's inputs or attention (sinusoidal, RoPE). | F1.2 |
| PPO | Proximal policy optimization: a policy-gradient method with a clipped objective that limits each update. | H2.2 |
| PR-AUC | Area under the precision–recall curve; informative when positives are rare. | B4.1 |
| Precision | Of the predicted positives, the share that are truly positive. | B4.1 |
| Precision@k | The share of the top $k$ recommended or retrieved items that are relevant. | B6.2 |
| Preference tuning | Tuning a model on human (or AI) preferences between pairs of outputs (RLHF, DPO). | F3.2 |
| Pretraining | Training a model on a large general corpus with a self-supervised objective. | F2.2 |
| Projection | The closest point to a vector within a subspace; the residual is orthogonal to the subspace. | A2.4 |
| Prompt caching | Reusing the provider's computation for an identical prompt prefix, at lower cost and latency. | I4.4 |
| Prompt injection | Input crafted to make an LLM ignore its instructions and follow the attacker's. `[RAI]` | I4.2 |
| Prompting | Writing the instructions, context and examples an LLM receives. | I1.2 |
| PSI | Population stability index: a drift score comparing binned distributions of a feature. | J2.2 |

## Q

| Term | Definition | Taught in |
|---|---|---|
| Q-learning | Learning action values by moving $Q(s, a)$ toward $r + \gamma \max_{a'} Q(s', a')$. | H1.3 |
| QLoRA | LoRA on top of a base model quantized to 4 bits, so fine-tuning fits on a small GPU. | J1.3 |
| QR decomposition | Writing a matrix as an orthogonal matrix times an upper-triangular one. | A3.4 |
| Quantization | Storing weights (and sometimes activations) in fewer bits to save memory and time. | F3.4, J2.1 |

## R

| Term | Definition | Taught in |
|---|---|---|
| RAG | Retrieval-augmented generation: retrieving relevant documents and giving them to an LLM as context. | I2.3 |
| Random forest | An ensemble of decision trees trained on bootstrap samples with feature subsampling. | B3.2 |
| Random search | Sampling hyperparameter combinations at random; usually beats grid search for the same budget. | B4.3 |
| Random variable | A quantity whose value is determined by a random outcome. | A5.1 |
| Rank | The number of linearly independent rows or columns of a matrix. | A2.3 |
| Rate limit | A provider's cap on requests or tokens per minute. | I1.1 |
| Recall | Of the true positives, the share the model finds. | B4.1 |
| Recall@k | The share of relevant items that appear in the top $k$ results. | I2.4 |
| Receptive field | The region of the input that affects one unit of a feature map. | D1.2 |
| Recommender system | A system that ranks items for each user. | B6.1 |
| Red-teaming | Deliberately attacking a model to find harmful or unsafe behaviour before users do. `[RAI]` | F3.3 |
| Refusal | A model declining a request it should not fulfil. `[RAI]` | F3.3 |
| Regression suite | A fixed set of tests run on every change to catch quality that got worse. | I3.3 |
| Regularization | Anything that discourages overfitting: penalties, dropout, early stopping, augmentation. | B2.2, C3.5 |
| REINFORCE | The basic policy-gradient algorithm: weight log-probability gradients by the return. | H2.1 |
| Reparameterization trick | Writing $z = \mu + \sigma\epsilon$ with $\epsilon \sim \mathcal{N}(0, 1)$ so gradients flow through sampling. | G1.2 |
| Replay buffer | A store of past transitions sampled at random to train DQN on less correlated data. | H1.4 |
| Reranking | Re-scoring retrieved candidates with a stronger model before using the top few. | I2.3 |
| Residual connection | Adding a block's input to its output, $\mathbf{x} + f(\mathbf{x})$, so gradients flow through deep networks. | D1.3, F1.2 |
| ResNet | A CNN family built from residual blocks. | D1.3 |
| Resampling | Changing class frequencies by over-sampling the rare class or under-sampling the common one. | B4.4 |
| Retention (data) | How long data is kept before deletion; part of privacy governance. `[RAI]` | J2.5 |
| Retrieve-then-rank | A two-stage system: a cheap retriever finds candidates and a stronger model ranks them. | B6.2, I2.3 |
| Return | The (discounted) sum of rewards from a time step on: $G_t = \sum_k \gamma^k r_{t+k+1}$. | H1.1 |
| Reward hacking | An agent maximizing the reward signal in a way that defeats the designer's intent. `[RAI]` | H2.4 |
| Reward model | A model trained on human preferences to score outputs; the reward in RLHF. | H2.3 |
| RLHF | Reinforcement learning from human feedback: a reward model from preferences, then RL (usually PPO) against it. | F3.2, H2.3 |
| RNN | Recurrent neural network: applies the same cell at each time step, carrying a hidden state. | E2.1 |
| ROC-AUC | Area under the ROC curve: the probability a random positive is scored above a random negative. | B4.1 |
| Rollback | Returning to the previous model version after a bad deployment. | J2.4 |
| RoPE | Rotary positional embeddings: encoding position by rotating query and key vectors. | F1.2 |

## S

| Term | Definition | Taught in |
|---|---|---|
| Safety (LLM) | Preventing harmful outputs and misuse of a model. `[RAI]` | F3.3 |
| Scaled dot-product attention | $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k})\,\mathbf{V}$. | F1.1 |
| Scaling (features) | Putting features on comparable ranges: standardization or min–max scaling. | B1.3 |
| Scaling laws | Empirical power laws relating loss to parameters, data and compute. | F2.3 |
| Seasonality | A repeating pattern in a time series at a fixed period (daily, weekly, yearly). | B6.3 |
| Seed | The starting value of a random number generator; fixing it makes runs repeatable. | A1.1 |
| Segmentation | Labelling every pixel of an image. | D2.4 |
| Self-attention | Attention where queries, keys and values all come from the same sequence. | F1.1 |
| Semantic caching | Reusing an earlier answer when a new query's embedding is close enough to a past one. | I4.4 |
| Semantic search | Finding documents by meaning through embeddings, not exact words. | I2.1 |
| Semantic segmentation | Labelling each pixel with a class, without separating instances. | D2.4 |
| seq2seq | A model that maps an input sequence to an output sequence with an encoder and a decoder. | E2.3 |
| SGD | Stochastic gradient descent: gradient descent using a random mini-batch per step. | A4.4, C3.1 |
| Shadow deployment | Running a new model on live traffic without serving its outputs, to compare it safely. | J2.4 |
| SHAP | SHapley Additive exPlanations: per-prediction feature attributions based on Shapley values. | B3.4 |
| Sigmoid | $\sigma(z) = 1 / (1 + e^{-z})$: squashes a number into $(0, 1)$. | B2.1 |
| Silhouette score | A clustering score comparing each point's distance to its own cluster with the nearest other cluster. | B5.1 |
| Skip-gram | word2vec's objective: predict context words from the centre word. | E1.3 |
| Softmax | $\operatorname{softmax}(\mathbf{z})_k = e^{z_k} / \sum_j e^{z_j}$: turns logits into probabilities. | C1.4 |
| Specification gaming | Satisfying the literal objective in an unintended way; the general form of reward hacking. `[RAI]` | H2.4 |
| Stratified k-fold | k-fold cross-validation that keeps the class proportions in every fold. | B4.3 |
| Streaming | Sending a response token by token as it is generated. | I1.1, I4.3 |
| Stride | How many positions a convolution or pooling window moves at each step. | D1.1 |
| Structured output | Constraining or validating an LLM's output to match a schema. | I1.3 |
| Support vector machine (SVM) | A maximum-margin classifier, optionally with kernels. | B2.5 |
| SVD | Singular value decomposition: $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$, rotate, scale, rotate. | A3.2 |
| System prompt | Instructions given to an LLM that set its role and rules for the conversation. | I1.1 |

## T

| Term | Definition | Taught in |
|---|---|---|
| t-SNE | A non-linear 2-D embedding that preserves local neighbourhoods; distances between clusters mean little. | B5.5 |
| t-test | A test of whether two means differ, using the t distribution (Welch's version allows unequal variances). | A6.3 |
| T5 | An encoder–decoder transformer that casts every task as text to text. | F2.1 |
| Target leakage | A feature that encodes the target or is only known after the outcome. | B1.4 |
| Target network | A slowly updated copy of the Q-network used to compute stable DQN targets. | H1.4 |
| Teacher forcing | Feeding the true previous token, not the model's own prediction, to the decoder during training. | E2.3 |
| Temperature | Dividing logits by $\tau$ before softmax; higher is more random, lower is more deterministic. | F2.4 |
| Temporal leakage | Using information from the future when predicting the past, often through a random split of time-ordered data. | B1.4 |
| TF-IDF | Term frequency times inverse document frequency: word weights that downweight common words. | E1.2 |
| Threshold tuning | Choosing the probability cut-off for a positive prediction to match the costs of errors. | B4.4 |
| Throughput | Work done per unit time (examples per second, tokens per second). | C4.1, I4.3 |
| Time-series cross-validation | Validation that always trains on the past and tests on the following period. | B4.3, B6.3 |
| Token | The unit a language model reads and writes: a word, subword or byte sequence. | E1.1, F1.3, I1.1 |
| Tokenization | Splitting text into tokens and mapping them to integer ids. | E1.1, F1.3 |
| Tool use | An LLM requesting a function call with arguments; the application runs it and returns the result. | I1.4 |
| Top-k sampling | Sampling the next token only from the $k$ most probable tokens. | F2.4 |
| Top-p (nucleus) sampling | Sampling from the smallest set of tokens whose probabilities sum to at least $p$. | F2.4 |
| Tracing | Recording each step of a request (model calls, tools, retrieval) with timing, as linked spans. | I4.5 |
| Train/validation/test split | Separate data for fitting, for choosing models and settings, and for the final unbiased estimate. | B1.1 |
| Trajectory (agent) | The sequence of steps, tool calls and observations an agent took on a task. | I3.3 |
| Transfer learning | Reusing a model pretrained on one task as the starting point for another. | D2.1 |
| Transformer | An architecture of stacked self-attention and MLP blocks with residuals and normalization. | F1.2 |
| Transpose | $\mathbf{A}^\top$: rows become columns. | A2.2 |
| Trend | The long-term direction of a time series. | B6.3 |

## U

| Term | Definition | Taught in |
|---|---|---|
| U-Net | An encoder–decoder CNN with skip connections, used for segmentation and in diffusion models. | D2.4, G1.4 |
| UMAP | A non-linear dimensionality reduction that preserves local structure, faster than t-SNE. | B5.5 |
| Underfitting | A model too simple to capture the pattern, so both training and test error are high. | B4.5 |
| Uniform distribution | Every value in a range is equally likely. | A5.2 |
| uv | The Python package and environment manager this course uses. | A1.1 |

## V

| Term | Definition | Taught in |
|---|---|---|
| VAE | Variational autoencoder: an autoencoder whose latent space is a distribution, trained on the ELBO. | G1.2 |
| Validation curve | Training and validation score plotted against one hyperparameter. | B4.5 |
| Value function | The expected return from a state, $V^\pi(s)$, or from a state and action, $Q^\pi(s, a)$. | H1.1 |
| Value iteration | Repeatedly applying the Bellman optimality update until values converge. | H1.2 |
| Vanishing gradients | Gradients that shrink exponentially through layers or time steps, so early parts stop learning. | C3.3, E2.1 |
| Variance | $\operatorname{Var}[X] = \mathbb{E}[(X - \mu)^2]$: the spread of a random variable. | A5.1 |
| Vectorization | Replacing Python loops with whole-array operations. | A1.2 |
| Vision-language model (VLM) | A model that takes images and text as input and generates text. | G2.3 |
| Vision Transformer (ViT) | A transformer that treats image patches as tokens. | G2.1 |

## W

| Term | Definition | Taught in |
|---|---|---|
| Walk-forward validation | Repeatedly training on data up to a time and testing on the next period, moving forward. | B6.3 |
| Warmup | Starting with a small learning rate and increasing it over the first steps. | C3.2 |
| Watermarking | Embedding a detectable signal in generated content to mark it as synthetic. `[RAI]` | G2.4 |
| Weight decay | Shrinking weights a little at each step; equivalent to L2 regularization for plain SGD. | C3.5 |
| word2vec | Learning word embeddings by predicting words from their context (skip-gram, CBOW). | E1.3 |
| Workflow (vs agent) | A fixed sequence of LLM and code steps, chosen when the path is known in advance. | I3.1 |

## X–Z

| Term | Definition | Taught in |
|---|---|---|
| Xavier (Glorot) initialization | Weight initialization with variance $2 / (d_{\text{in}} + d_{\text{out}})$, suited to tanh layers. | C3.3 |
| XGBoost | A widely used gradient-boosting library with regularized trees. | B3.3 |
| Zero-shot classification | Classifying into labels the model was not trained on, for example by comparing CLIP image and text embeddings. | G2.2 |
