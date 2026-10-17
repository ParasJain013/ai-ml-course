# AI/ML Course Outline

Source of truth for the course structure. Part syllabi (`course/PART_*.md`) add the weekly detail and
must not contradict this file. Written 2026-10-16.

## Assumptions

- **Budget.** 36 weeks at ~12 hrs/week ≈ 430 hours. A normal week holds 4–5 topics (~2 hrs each,
  lesson + lab) plus 2–4 hrs of project work; weeks with a project deadline carry fewer topics.
- **Target role.** "Job-ready" means an applied ML / AI engineer who builds, evaluates and ships
  models and LLM systems, not a research scientist. That is why Applied LLM engineering (I) and
  MLOps (J) are Core-heavy and scheduled before Generative models (G) and RL (H), which are mostly
  Depth for that role.
- **Hardware.** Every lab runs on the laptop CPU at default settings. GPU work (CNN fine-tuning, the
  small GPT at full size, diffusion, LoRA/QLoRA) runs on free Colab or Kaggle GPUs, sized to fit a
  single free session. Free-tier quotas: VERIFY (as of 2026-10).
- **LLM access in Parts I, J, L.** One paid LLM API key with a small budget (tens of US dollars over
  the whole course; VERIFY (as of 2026-10)) plus a small open-weights model that runs locally on CPU
  as the free fallback. Postgres + pgvector run in Docker, which you already know.
- **Ids.** Part letters are names, not the order you study them. `B3.2` = Part B, week 3 of that
  part, topic 2. **Core** = job-essential, **Depth** = mastery. `[RAI]` marks a Responsible AI
  topic (Part K is woven in; see its section).

## Changes from the requested plan, and why

| Change | Why |
|---|---|
| Order is A → B → C → D → E → F → **I → J** → G → H → L | I and J are the most job-critical parts after the fundamentals. Putting them before G and H means that if time slips, the cut lands on Depth-heavy parts. It also lines up with the planned web refresh of Parts I and J around week 24 (see `PROMPTS.md`). |
| A: 5 → **6** weeks | Your math is rusty and every later part reads its formulas. Linear algebra gets two weeks instead of one. |
| D: 3 → **2** weeks | Vision Transformers need attention, which is not taught until E2/F1, so ViT moves to G2 next to CLIP (whose image encoder is a ViT). Detection and segmentation are taught at the level an applied engineer uses them: metrics, pretrained models, fine-tuning. |
| RLHF is split between F3 and H2 | DPO needs no RL, so instruction tuning and DPO stay in F3. The reward-model + PPO pipeline is taught properly in H2.3, after PPO exists. F3.2 gives the overview and names H2.3 as the full treatment. |
| Total stays 36 weeks | B (6), C (4), E (2), F (3), I (4), J (2), G (2), H (2) and L (3) keep the requested counts. |

## Course map

| Course weeks | Part | Weeks | Project due | Checkpoint |
|---|---|---|---|---|
| 1–6 | A. Foundations | 6 | P1 (wk 6) | CP1 (wk 6) |
| 7–12 | B. Classical ML | 6 | P2 (wk 10), P3 (wk 12) | CP2 (wk 12) |
| 13–16 | C. Deep learning | 4 | P4 (wk 14) | CP3 (wk 16) |
| 17–18 | D. Computer vision | 2 | P5 (wk 18) | |
| 19–20 | E. NLP & sequence models | 2 | | |
| 21–23 | F. Transformers & LLMs | 3 | P6 (wk 22) | CP4 (wk 23) |
| 24–27 | I. Applied LLM engineering | 4 | P7 (wk 25), P8 (wk 26) | |
| 28–29 | J. MLOps & production | 2 | P9 (wk 28) | CP5 (wk 29) |
| 30–31 | G. Generative models & multimodal | 2 | P10 (wk 31) | |
| 32–33 | H. Reinforcement learning | 2 | P11 (wk 33) | |
| 34–36 | L. Capstone & interview prep | 3 | P12 (wk 36) | CP6 (wk 36) |
| — | K. Responsible AI | woven in | | |

## Dependency map

```text
A Foundations
└─► B Classical ML ─────────────────────────────┐
    └─► C Deep learning                         │
        ├─► D Computer vision ───────────┐      │
        ├─► E NLP & sequences            │      │
        │   └─► F Transformers & LLMs ───┼──┐   │
        │       ├─► I Applied LLM eng. ──┼──┼───┼─► J MLOps ──┐
        │       │                        ▼  │   │             │
        │       └──────────────────► G Generative & multimodal│
        └─────────────────────────► H RL ◄──┘ (H2.3 RLHF)     │
                                                              ▼
                       B, C, I, J (required) + D–H (optional) ─► L Capstone & interview
```

Hard prerequisites, part by part:

| Part | Needs | Specifically |
|---|---|---|
| B | A | A2–A3 (least squares, SVD), A4 (gradient descent), A5–A6 (MLE, testing, cross-entropy) |
| C | A, B | A4 chain rule, B2.1 logistic regression, B4 evaluation |
| D | C | C2 PyTorch, C3 training mechanics |
| E | C | C2–C3; A1.5 cosine similarity for embeddings |
| F | E | E2.4 attention, E1.3 embeddings, C4 GPU training |
| I | F | F2.4 decoding, E1.3/I2.1 embeddings; B4 evaluation thinking for evals |
| J | B, I | B4 metrics for monitoring; I2.4/I3.3 evals for the P9 comparison; F3.1 SFT for LoRA |
| G | C, D, F | A6.4 KL for VAEs; D1 CNNs; F1 attention for ViT and CLIP |
| H | C, F | C3 optimizers; F3 for the RLHF link |
| L | B, C, I, J | plus whichever of D–H the capstone uses |

---

## Part A. Foundations (6 weeks, course weeks 1–6)

**Goal:** rebuild the math every later part relies on, from intuition and always in code, and become
fluent with NumPy, pandas and matplotlib.

**A1. Python scientific stack**
- A1.1 Core: the lab workflow — `uv`, Jupyter vs `# %%` cells, seeds, how every lab in this course is run.
- A1.2 Core: NumPy arrays — shapes, indexing, broadcasting and vectorizing loops away.
- A1.3 Core: pandas — DataFrames, selection, groupby, joins (mapped to SQL you already write), missing values.
- A1.4 Core: matplotlib — line, scatter, histogram, subplots, and plots that don't mislead.
- A1.5 Core: vectors — addition, dot product, norms, cosine similarity; the geometry behind embeddings.

**A2. Linear algebra I: matrices**
- A2.1 Core: matrices as linear maps — what multiplying by a matrix does to space, and shape rules `(n, d) @ (d, k)`.
- A2.2 Core: matrix multiplication, transpose, identity and inverse, done by hand then in NumPy.
- A2.3 Core: systems of linear equations, rank, and when a solution does not exist.
- A2.4 Core: projections and least squares — the normal equations, the first "model fit".

**A3. Linear algebra II: decompositions**
- A3.1 Core: eigenvalues and eigenvectors — directions a matrix only stretches.
- A3.2 Core: singular value decomposition — any matrix as rotate, scale, rotate; the engine of PCA.
- A3.3 Core: covariance matrices — what they encode and why they are symmetric and positive semi-definite.
- A3.4 Depth: numerical stability — condition numbers, QR and Cholesky, why you never call `inv()` in practice.

**A4. Calculus & optimization**
- A4.1 Core: derivatives and partial derivatives — slope intuition, checked with finite differences.
- A4.2 Core: gradients, Jacobians and the shapes of matrix derivatives used in ML.
- A4.3 Core: the chain rule as a computational graph — the seed of backpropagation.
- A4.4 Core: gradient descent — learning rate, convergence, fitting a line by GD and matching A2.4.
- A4.5 Depth: convexity, constrained optimization and Lagrange multipliers (needed for B2.5 SVMs).

**A5. Probability**
- A5.1 Core: random variables, expectation, variance and covariance, by simulation first.
- A5.2 Core: common distributions — Bernoulli, binomial, Poisson, uniform, Gaussian, multivariate Gaussian, categorical.
- A5.3 Core: conditional probability and Bayes' theorem — base rates and the medical-test trap.
- A5.4 Core: law of large numbers, central limit theorem and Monte Carlo estimation.
- A5.5 Core: MLE and MAP — why squared error is Gaussian MLE, log-loss is Bernoulli MLE, and L2 is a Gaussian prior.

**A6. Statistics & information theory**
- A6.1 Core: descriptive statistics and EDA — robust summaries, correlation versus causation.
- A6.2 Core: confidence intervals and the bootstrap.
- A6.3 Core: hypothesis testing — p-values, t-test, chi-square, multiple comparisons, A/B tests and their pitfalls.
  (Taught as two lessons: A6.3a single tests, A6.3b multiple comparisons and A/B tests; see PART_A.md.)
- A6.4 Core: entropy, cross-entropy and KL divergence — what classifiers and VAEs actually minimize.
- A6.5 Depth: Bayesian inference — conjugate priors with the Beta–Binomial example.

## Part B. Classical ML (6 weeks, course weeks 7–12)

**Goal:** frame, build, evaluate and explain classical models end to end, without leakage, and know
when a gradient-boosted tree beats a neural net.

**B1. Workflow & data**
- B1.1 Core: the ML workflow — framing, baselines, train/validation/test splits, and what "generalize" means.
- B1.2 Core: data cleaning — types, missing values, outliers, duplicates; `[RAI]` PII handling and data minimization.
- B1.3 Core: feature engineering — scaling, categorical encoding, dates, interactions, scikit-learn Pipelines and ColumnTransformer.
- B1.4 Core: data leakage — target leakage, train/test contamination, temporal leakage, and how to catch each.
- B1.5 Core: linear regression from scratch (closed form and GD) versus scikit-learn.

**B2. Linear models & simple classifiers**
- B2.1 Core: logistic regression — sigmoid, log-loss, gradient descent from scratch.
- B2.2 Core: regularization — L1, L2 and elastic net, read as MAP priors from A5.5.
- B2.3 Core: k-nearest neighbours — distance metrics and the curse of dimensionality.
- B2.4 Core: naive Bayes — Gaussian and multinomial, with a spam-filter example.
- B2.5 Depth: support vector machines — maximum margin, hinge loss and the kernel trick.

**B3. Trees & ensembles**
- B3.1 Core: decision trees — Gini and entropy splits built from scratch.
- B3.2 Core: random forests — bagging, feature subsampling, out-of-bag error.
- B3.3 Core: gradient boosting — fitting residuals from scratch, then XGBoost / LightGBM in practice.
- B3.4 Core: interpretability — permutation importance, partial dependence and SHAP values.

**B4. Evaluation**
- B4.1 Core: classification metrics — confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC, calibration.
- B4.2 Core: regression metrics, and choosing a metric that matches the real cost of errors.
- B4.3 Core: cross-validation (k-fold, stratified, group, time-series) and hyperparameter tuning (grid, random, Bayesian).
- B4.4 Core: class imbalance — class weights, resampling, threshold tuning.
- B4.5 Core: the bias–variance trade-off, learning curves and validation curves.
- B4.6 Core: `[RAI]` fairness — group metrics (demographic parity, equalized odds), auditing a model, and a first model card.

**B5. Unsupervised learning**
- B5.1 Core: k-means from scratch, and choosing k (elbow, silhouette).
- B5.2 Depth: hierarchical clustering and DBSCAN — when clusters aren't round.
- B5.3 Depth: Gaussian mixture models and the EM algorithm.
- B5.4 Core: PCA from scratch via SVD (A3.2), explained variance.
- B5.5 Core: t-SNE and UMAP — what they show and how they mislead.
- B5.6 Core: anomaly detection — statistical rules, isolation forest, density-based scores.

**B6. Recommenders, time series**
- B6.1 Core: recommender systems — collaborative filtering and matrix factorization from scratch, implicit feedback.
- B6.2 Depth: ranking metrics (precision@k, NDCG) and the retrieve-then-rank pattern.
- B6.3 Core: time-series forecasting — trend, seasonality, lag features, naive baselines, walk-forward validation.
- B6.4 Depth: ARIMA / exponential smoothing versus gradient boosting on lag features.

## Part C. Deep learning (4 weeks, course weeks 13–16)

**Goal:** understand neural networks from the gradient up, then train them reliably in PyTorch on
CPU and on a free GPU.

**C1. Neural networks from scratch**
- C1.1 Core: perceptron and multilayer perceptron — neurons, activations, why depth helps.
- C1.2 Core: backpropagation by hand on a tiny network with real numbers.
- C1.3 Core: an autograd engine from scratch — scalar first, then tensors.
- C1.4 Core: softmax and cross-entropy, written to be numerically stable.

**C2. PyTorch**
- C2.1 Core: tensors, autograd and `nn.Module`, checked against your C1.3 engine.
- C2.2 Core: Dataset, DataLoader and the training loop; train versus eval mode.
- C2.3 Core: saving, loading, reproducibility and device handling.

**C3. Making training work**
- C3.1 Core: optimizers — SGD, momentum, Adam and AdamW from scratch.
- C3.2 Core: learning-rate schedules — warmup, cosine decay, one-cycle, the LR range test.
- C3.3 Core: initialization — Xavier and He, vanishing and exploding gradients.
- C3.4 Core: normalization — BatchNorm and LayerNorm.
- C3.5 Core: regularization — dropout, weight decay, early stopping.

**C4. GPU training & debugging**
- C4.1 Core: training on a Colab/Kaggle GPU — moving work to the device, throughput, memory.
- C4.2 Depth: mixed precision (fp16/bf16, autocast, loss scaling), gradient accumulation and checkpointing.
- C4.3 Core: debugging training runs — overfit one batch, read loss curves, watch gradient norms, the common bugs.
- C4.4 Depth: distributed training concepts — data parallel, DDP, FSDP (concepts only, no multi-GPU needed).

## Part D. Computer vision (2 weeks, course weeks 17–18)

**Goal:** build convolutional networks from scratch, then solve real vision tasks with pretrained models.

**D1. Convolutional networks**
- D1.1 Core: convolution from scratch — kernels, stride, padding and the output-shape formula.
- D1.2 Core: pooling, receptive fields and a CNN in PyTorch.
- D1.3 Core: architectures from LeNet to ResNet — what each added, and why residual connections work.
- D1.4 Depth: efficient architectures — depthwise separable convolutions (MobileNet, EfficientNet).

**D2. Applied vision**
- D2.1 Core: transfer learning — feature extraction versus fine-tuning with pretrained weights.
- D2.2 Core: data augmentation — geometric and colour transforms, Mixup and CutMix.
- D2.3 Core: object detection — boxes, IoU, non-maximum suppression, mAP, fine-tuning a pretrained detector.
- D2.4 Core: segmentation — semantic versus instance, U-Net, Dice and IoU.
- D2.5 Depth: `[RAI]` error analysis for vision — Grad-CAM, per-group errors and dataset bias.

## Part E. NLP & sequence models (2 weeks, course weeks 19–20)

**Goal:** represent text as numbers and model sequences, ending at the attention mechanism that
transformers are built on.

**E1. Text representation**
- E1.1 Core: text preprocessing — normalization, word and character tokenization, vocabularies, n-grams.
- E1.2 Core: bag-of-words and TF-IDF with a linear classifier, the baseline every text model must beat.
- E1.3 Core: word2vec (skip-gram with negative sampling) from scratch, and embedding geometry.
- E1.4 Depth: subword and pretrained static embeddings (fastText, GloVe); `[RAI]` bias in embeddings.

**E2. Sequence models**
- E2.1 Core: recurrent networks and backpropagation through time; why gradients vanish.
- E2.2 Core: LSTM and GRU — what the gates do.
- E2.3 Core: sequence-to-sequence encoder–decoder models and teacher forcing.
- E2.4 Core: attention (Bahdanau-style) — letting the decoder look back, the bridge to transformers.

## Part F. Transformers & LLMs (3 weeks, course weeks 21–23)

**Goal:** build and train a transformer from scratch and understand how modern LLMs are pretrained,
decoded and tuned.

**F1. The transformer from scratch**
- F1.1 Core: scaled dot-product self-attention, multi-head attention and causal masking, from scratch.
- F1.2 Core: the transformer block — residuals, LayerNorm, MLP, positional encodings (sinusoidal, RoPE).
- F1.3 Core: BPE tokenization from scratch, compared with a production tokenizer.
- F1.4 Core: training a small GPT — tiny on CPU, scaled up on a free GPU.

**F2. Pretraining & inference**
- F2.1 Core: encoder, decoder and encoder–decoder models — BERT (masked LM) versus GPT (causal LM) versus T5.
- F2.2 Core: pretraining — data pipelines, objective, perplexity, compute cost.
- F2.3 Depth: scaling laws — loss versus parameters, data and compute; the compute-optimal trade-off.
- F2.4 Core: decoding — greedy, beam search, temperature, top-k, top-p, and the KV cache.
- F2.5 Depth: efficient attention and long context — FlashAttention idea, multi-query and grouped-query attention.

**F3. Post-training**
- F3.1 Core: instruction tuning (SFT) — chat templates, data format, what changes in the model.
- F3.2 Core: preference tuning — the RLHF pipeline at overview level and DPO implemented on a tiny model (full RLHF in H2.3).
- F3.3 Core: `[RAI]` safety and alignment — hallucination, harmful outputs, refusals, red-teaming.
- F3.4 Core: running open models with Hugging Face `transformers` on CPU — loading, generating, quantized weights.

## Part I. Applied LLM engineering (4 weeks, course weeks 24–27)

**Goal:** build LLM applications that are measured, secure, fast and affordable, the way you build
backend services. All library and API facts here get a VERIFY (as of <date>) flag.

**I1. Talking to models**
- I1.1 Core: LLM APIs — messages, system prompts, tokens, streaming, rate limits, retries (an external-service client, with honest differences).
- I1.2 Core: prompting — clear instructions, examples, reasoning, prompts versioned and tested like code.
- I1.3 Core: structured output — JSON schemas, Pydantic validation, repair and retry.
- I1.4 Core: tool use (function calling) — the request/execute/respond loop.

**I2. Embeddings & RAG**
- I2.1 Core: embedding models and semantic search — chunking strategies and similarity search.
- I2.2 Core: Postgres + pgvector — schema, HNSW / IVFFlat indexes, hybrid search with full-text search.
- I2.3 Core: the RAG pipeline — retrieval, reranking, context assembly, citations.
- I2.4 Core: evals for RAG — golden sets, recall@k and MRR for retrieval, faithfulness and answer quality, LLM-as-judge and its biases.

**I3. Agents & MCP**
- I3.1 Core: agents — the tool loop, planning, memory, and when a plain workflow is better than an agent.
- I3.2 Core: the Model Context Protocol — servers, tools and resources; build a small MCP server.
- I3.3 Core: agent evals — task success, trajectory checks, regression suites that run in CI.
- I3.4 Depth: multi-agent patterns and orchestration.

**I4. LLM systems in production**
- I4.1 Core: `[RAI]` guardrails — input and output checks, PII redaction, moderation.
- I4.2 Core: `[RAI]` security — direct and indirect prompt injection, data exfiltration, least-privilege tools.
- I4.3 Core: cost and latency — token accounting, model routing, streaming, batching.
- I4.4 Core: caching — exact, prompt and semantic caching, and what each saves.
- I4.5 Core: observability — tracing, logging prompts safely, online evals and feedback loops.

## Part J. MLOps & production (2 weeks, course weeks 28–29)

**Goal:** take any model from notebook to a monitored, versioned, tested service, and know when
fine-tuning beats prompting.

**J1. Tracking, versioning, fine-tuning**
- J1.1 Core: experiment tracking — runs, parameters, metrics, artifacts (MLflow or similar).
- J1.2 Core: data and model versioning — dataset snapshots, a model registry, lineage.
- J1.3 Core: LoRA and QLoRA — low-rank updates from scratch, then parameter-efficient fine-tuning on a free GPU.
- J1.4 Core: prompt versus RAG versus fine-tune — a decision framework with cost and quality evidence.

**J2. Serving & operating**
- J2.1 Core: model serving — a FastAPI service, batching, ONNX export and quantization; LLM inference servers.
- J2.2 Core: monitoring and drift — data and concept drift, PSI and KS tests, alerting.
- J2.3 Core: CI for ML — data tests, model tests and evals as merge gates in GitHub Actions.
- J2.4 Core: deployment — Docker images, a cloud target, shadow and canary rollout, rollback.
- J2.5 Core: `[RAI]` documentation and governance — model cards, datasheets, privacy (differential privacy idea, retention).

## Part G. Generative models & multimodal (2 weeks, course weeks 30–31)

**Goal:** understand how models generate images and connect vision with language.

**G1. Generative models**
- G1.1 Core: autoencoders — bottlenecks, denoising, learned representations.
- G1.2 Core: variational autoencoders — the ELBO and the reparameterization trick (uses A6.4 KL).
- G1.3 Depth: GANs — the generator/discriminator game, mode collapse, DCGAN.
- G1.4 Core: diffusion models — forward noising, the denoising objective, a tiny DDPM sampler from scratch.

**G2. Multimodal**
- G2.1 Core: Vision Transformers — images as patch tokens (moved here from D; needs F1).
- G2.2 Core: CLIP — contrastive image–text training and zero-shot classification.
- G2.3 Depth: vision-language models — how image features enter an LLM; using one through an API.
- G2.4 Depth: latent diffusion, text conditioning and classifier-free guidance; `[RAI]` deepfakes and watermarking.

## Part H. Reinforcement learning (2 weeks, course weeks 32–33)

**Goal:** learn the RL toolkit from MDPs to PPO and see exactly how it is used to tune LLMs.

**H1. Value-based RL**
- H1.1 Core: MDPs — states, actions, rewards, returns, value functions, the Bellman equations.
- H1.2 Core: dynamic programming — value iteration and policy iteration on a gridworld.
- H1.3 Core: Q-learning — tabular, epsilon-greedy exploration.
- H1.4 Depth: DQN — function approximation, replay buffer, target network.

**H2. Policy-based RL**
- H2.1 Core: policy gradients — REINFORCE and baselines.
- H2.2 Depth: actor–critic and PPO — advantage estimation and the clipped objective.
- H2.3 Core: RLHF in full — reward model, PPO with a KL penalty, and why DPO (F3.2) removes the RL loop.
- H2.4 Depth: `[RAI]` reward hacking and specification gaming.

## Part L. Capstone & interview prep (3 weeks, course weeks 34–36)

**Goal:** ship one end-to-end system you can defend in an interview, and rehearse the interview loop.

**L1. Capstone build**
- L1.1 Core: scoping and a design doc — problem, metric, data, baseline, architecture, risks.
- L1.2 Core: building the system with its eval harness first.

**L2. Ship it**
- L2.1 Core: deploy, monitor and document — public endpoint, dashboards, model card, cost report.
- L2.2 Core: ML system design interviews — the framework, worked through on several cases.

**L3. Interview loop**
- L3.1 Core: ML coding interviews — implementing core algorithms from scratch under time pressure.
- L3.2 Core: fundamentals and LLM-systems review with mock interviews.
- L3.3 Core: portfolio and project storytelling — the README, the demo, the 2-minute walkthrough.

## Part K. Responsible AI (woven in)

Every `[RAI]` topic above, grouped by theme. From P2 on, every project ships a model card.

| Theme | Where it is taught |
|---|---|
| Fairness | B4.6 group metrics and audits (applied in P2); D2.5 dataset bias in vision; E1.4 bias in embeddings |
| Privacy | B1.2 PII and data minimization; I4.1 PII redaction; I4.2 data exfiltration; J2.5 differential privacy idea and retention |
| Safety | F3.3 alignment and red-teaming; I4.1 guardrails; I4.2 prompt injection; G2.4 deepfakes and watermarking; H2.4 reward hacking |
| Documentation | B4.6 first model card; J2.5 model cards and datasheets; L2.1 capstone model card |

---

## Portfolio projects

Each project escalates on the ones before it. Briefs go in `projects/`, rubrics in `answers/projects/`.

| # | Project | Due (part, course wk) | What you build | GPU |
|---|---|---|---|---|
| P1 | EDA + statistics report | A6, wk 6 | Clean a public dataset in pandas, explore it with honest plots, run 2–3 hypothesis tests with bootstrap CIs, and write a report that states its caveats. | no |
| P2 | End-to-end tabular model | B4, wk 10 | Pipeline, cross-validated and tuned gradient boosting versus a logistic-regression baseline, a leakage audit, SHAP explanations, a fairness audit and a model card. | no |
| P3 | Unsupervised project | B6, wk 12 | Customer segmentation plus anomaly detection on one dataset: PCA/UMAP views, cluster-stability checks, and a short write-up a non-ML reader can act on. | no |
| P4 | Neural net + autograd from scratch | C2, wk 14 | A tensor autograd engine and MLP in NumPy, trained on a small image set; gradients and accuracy match PyTorch. | no |
| P5 | CV transfer learning | D2, wk 18 | Fine-tune a pretrained CNN on a small dataset you collect or pick, with an augmentation ablation, error analysis and Grad-CAM. | helpful |
| P6 | Small GPT from scratch | F1–F2, wk 22 | Your BPE tokenizer and decoder-only transformer trained on a small corpus, with loss curves and samples under several decoding strategies. | helpful |
| P7 | RAG system with an eval suite | I2, wk 25 | Question answering over a document set using Postgres + pgvector, with a golden set, retrieval and answer metrics, and a report on what each change did to the scores. | no |
| P8 | Tool-using agent with evals | I3, wk 26 | An agent that uses 3+ tools (one exposed through your own MCP server), a task-success eval suite run in CI, and a prompt-injection test set. | no |
| P9 | LoRA fine-tune versus prompted baseline | J1, wk 28 | LoRA/QLoRA fine-tune of a small open model on one narrow task, compared with a prompted (and few-shot) baseline on the same eval set, including cost and latency. | required (Colab/Kaggle) |
| P10 | Generative model | G1–G2, wk 31 | A VAE and a tiny DDPM on the same small image dataset, compared on samples and a simple quality metric, plus CLIP zero-shot as a bonus. | helpful |
| P11 | RL agent in Gymnasium | H2, wk 33 | Tabular Q-learning on a small environment, then DQN or PPO on CartPole or LunarLander, with learning curves across seeds. | helpful |
| P12 | Capstone, deployed end to end | L1–L2, wk 36 | A system of your choice using at least two parts (for example an LLM product or a tabular model plus LLM layer): design doc, evals in CI, Docker deployment, monitoring, model card, cost report. | depends |

## Landmark papers

Each paper gets a reading guide in `papers/` in the week shown.

| Week (topic) | Paper | Authors | Year |
|---|---|---|---|
| B3 (B3.3) | XGBoost: A Scalable Tree Boosting System | Chen, Guestrin | 2016 |
| D1 (D1.3) | Deep Residual Learning for Image Recognition | He, Zhang, Ren, Sun | 2015 (arXiv; CVPR 2016) |
| E1 (E1.3) | Efficient Estimation of Word Representations in Vector Space | Mikolov, Chen, Corrado, Dean | 2013 |
| F1 (F1.1) | Attention Is All You Need | Vaswani et al. | 2017 |
| F2 (F2.3) | Training Compute-Optimal Large Language Models ("Chinchilla") | Hoffmann et al. | 2022 |
| F3 (F3.2) | Training language models to follow instructions with human feedback ("InstructGPT") | Ouyang et al. | 2022 |
| I2 (I2.3) | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Lewis et al. | 2020 |
| J1 (J1.3) | LoRA: Low-Rank Adaptation of Large Language Models | Hu et al. | 2021 |
| G1 (G1.4) | Denoising Diffusion Probabilistic Models | Ho, Jain, Abbeel | 2020 |
| H1 (H1.4) | Human-level control through deep reinforcement learning ("DQN") | Mnih et al. | 2015 |

## Checkpoints

A checkpoint is passed only when every criterion holds. If one fails, take one catch-up week on the
topic ids your reviews flagged before moving on.

| CP | When | Pass criteria |
|---|---|---|
| CP1 | End of A (wk 6) | Without notes: predict matrix-product shapes, compute a gradient of mean squared error, solve a Bayes base-rate problem, derive the Gaussian MLE. P1 scores ≥ 70% on its rubric. Week quizzes average ≥ 80%. |
| CP2 | End of B (wk 12) | Implement logistic regression with GD from scratch in 30 min. Explain three kinds of leakage with an example each. Pick and justify a metric for an imbalanced problem. P2 and P3 ≥ 70%. |
| CP3 | End of C (wk 16) | P4 gradients match PyTorch to within 1e-6. Given three broken loss curves, name the likely cause of each and the fix. Explain why Adam, warmup and normalization help. |
| CP4 | End of F (wk 23) | Write causal self-attention from scratch in 45 min with correct shapes. P5 and P6 ≥ 70%; P6 beats a bigram baseline on validation loss. Explain BERT versus GPT and four decoding strategies. |
| CP5 | End of J (wk 29) | P7, P8 and P9 ≥ 70%, with eval suites that run in CI. Defend the P9 fine-tune-or-prompt decision with numbers. One LLM-systems mock interview at "lean hire" or better. |
| CP6 | End of L (wk 36) | P12 deployed, monitored, documented and ≥ 70%. One ML-fundamentals mock and one system-design mock at "lean hire" or better. |

## Fast path: interview-ready by ~week 24

Study only these Core topics, in this order. Everything else becomes post-offer Depth. Week budgets
assume the full 12–15 hrs.

| Weeks | Topics | Projects |
|---|---|---|
| 1–4 | A1.2–A1.5, A2.1–A2.2, A2.4, A3.2, A4.1–A4.4, A5.1–A5.5, A6.1–A6.4 | P1 (lighter: one test, one CI) |
| 5–9 | B1.1–B1.5, B2.1–B2.3, B3.1–B3.4, B4.1–B4.6, B5.1, B5.4, B6.3 | P2 |
| 10–12 | C1.1–C1.4, C2.1–C2.3, C3.1–C3.5, C4.1, C4.3 | P4 |
| 13–14 | D1.1, D1.3, D2.1–D2.2; E1.2–E1.3, E2.4 | — |
| 15–17 | F1.1–F1.4, F2.1, F2.2, F2.4, F3.1–F3.2 | P6 (CPU-sized) |
| 18–21 | I1.1–I1.4, I2.1–I2.4, I3.1–I3.3, I4.1–I4.4 | P7, P8 |
| 22–23 | J1.3–J1.4, J2.1–J2.4 | P7 deployed as a mini capstone |
| 24 | L2.2, L3.1–L3.3 | mocks |

## If you fall 3 weeks behind

Cut in this order until you are back on schedule. Never cut leakage (B1.4), evaluation (B4), evals
(I2.4, I3.3), security (I4.2), or P2, P7, P8 and P12.

1. **Every Depth topic not yet studied** (about 1 week saved): B2.5, B5.2, B5.3, B6.2, B6.4,
   C4.2, C4.4, D1.4, D2.5, E1.4, F2.3, F2.5, I3.4.
2. **G from 2 weeks to 1** (1 week saved): keep G1.1, G1.2, G1.4 and G2.2; P10 becomes the VAE only.
3. **H from 2 weeks to 1** (1 week saved): keep H1.1, H1.3, H2.1 and H2.3; P11 becomes tabular Q-learning plus REINFORCE on CartPole.

If you are still behind after that, shorten L3 to one mock and fold L3.3 into L2.1.
