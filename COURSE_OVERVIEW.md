# What I'm Learning: AI/ML Engineering Course Overview

*Status as of 2 November 2026.*

## In one paragraph

This is a structured, self-paced course of about 36 weeks at 10–15 hours a week, roughly 430
hours in total. It aims to turn me from a backend engineer (TypeScript, NestJS, PostgreSQL, Docker,
Terraform) into a job-ready **applied ML / AI engineer**: someone who builds, evaluates and ships
models and LLM-based systems, not a research scientist. It starts by rebuilding the math, then
covers classical ML, deep learning, computer vision, NLP, transformers and LLMs, applied LLM
engineering (RAG, agents, evals), MLOps, generative models and reinforcement learning. It ends with a
deployed capstone project and interview preparation. Along the way I build 12 portfolio projects.

## How it is taught

Every topic follows the same pattern:

1. **Intuition first**, then the minimum math needed, with every symbol defined.
2. **A worked example by hand** with small numbers.
3. **Build it from scratch in NumPy**, then use the standard library (scikit-learn, PyTorch, Hugging
   Face) and check that the two give the same result.
4. **A runnable lab**: every topic has a Python lab that runs on my laptop's CPU in under 5 minutes.
   GPU-heavy work runs on free Colab or Kaggle GPUs.
5. **Exercises, a weekly quiz and a cheat sheet**, plus reading guides for 10 landmark papers.

My solutions are reviewed against a rubric by an AI reviewer acting as a strict senior engineer. It
checks correctness, data leakage, bad train/test splits, misleading metrics and production risks.

Each topic is labelled **Core** (needed for the job) or **Depth** (for mastery). Responsible AI
(fairness, privacy, safety, documentation) is taught inside the relevant parts, not as a separate
block.

## What the course covers

| Weeks | Part | What I learn | Portfolio project |
|---|---|---|---|
| 1–6 | **A. Foundations** | Python data stack (NumPy, pandas, matplotlib); linear algebra (matrices, least squares, eigenvectors, SVD); calculus and gradient descent; probability (distributions, Bayes, MLE/MAP); statistics (confidence intervals, bootstrap, hypothesis tests, A/B tests); entropy and KL divergence | P1: EDA + statistics report |
| 7–12 | **B. Classical ML** | ML workflow, data cleaning, feature engineering, **data leakage**; linear and logistic regression, regularization, kNN, naive Bayes, SVMs; decision trees, random forests, gradient boosting (XGBoost/LightGBM), SHAP; evaluation metrics, cross-validation, class imbalance, fairness audits; clustering, PCA, t-SNE/UMAP, anomaly detection; recommenders and time-series forecasting | P2: end-to-end tabular model with a fairness audit and model card · P3: customer segmentation + anomaly detection |
| 13–16 | **C. Deep learning** | Neural networks and backpropagation from scratch; my own autograd engine; PyTorch; optimizers (SGD, Adam, AdamW), learning-rate schedules, initialization, normalization, regularization; GPU training, mixed precision, debugging training runs | P4: neural net + autograd engine from scratch that matches PyTorch |
| 17–18 | **D. Computer vision** | Convolution from scratch, CNN architectures up to ResNet; transfer learning, data augmentation, object detection, segmentation, Grad-CAM | P5: fine-tuned image classifier with error analysis |
| 19–20 | **E. NLP & sequence models** | Tokenization, TF-IDF, word2vec from scratch; RNNs, LSTMs, seq2seq, attention | — |
| 21–23 | **F. Transformers & LLMs** | Self-attention and the transformer from scratch; BPE tokenizer; training a small GPT; BERT vs GPT vs T5; pretraining, scaling laws, decoding strategies, KV cache; instruction tuning, DPO, safety; running open models locally | P6: small GPT from scratch |
| 24–27 | **I. Applied LLM engineering** | LLM APIs, prompting, structured output, tool calling; embeddings, **Postgres + pgvector**, RAG pipelines and RAG evals; agents, Model Context Protocol (MCP), agent evals in CI; guardrails, prompt-injection security, cost, latency, caching, observability | P7: RAG system with an eval suite · P8: tool-using agent with evals and an MCP server |
| 28–29 | **J. MLOps & production** | Experiment tracking, data and model versioning; LoRA/QLoRA fine-tuning; prompt vs RAG vs fine-tune decisions; model serving (FastAPI, ONNX, quantization), drift monitoring, CI for ML, Docker deployment with canary releases and rollback, model cards and governance | P9: LoRA fine-tune vs prompted baseline, compared on quality, cost and latency |
| 30–31 | **G. Generative models & multimodal** | Autoencoders, VAEs, GANs, diffusion models (DDPM from scratch); Vision Transformers, CLIP, vision-language models, latent diffusion | P10: VAE vs tiny diffusion model |
| 32–33 | **H. Reinforcement learning** | MDPs, Bellman equations, Q-learning, DQN; policy gradients, PPO; **RLHF in full** (how LLMs are tuned with RL) | P11: RL agent in Gymnasium |
| 34–36 | **L. Capstone & interview prep** | Design doc, building with evals first, deploying, monitoring and documenting a full system; ML system-design interviews, ML coding interviews, mock interviews, portfolio storytelling | P12: capstone deployed end to end |

**Order.** Applied LLM engineering (I) and MLOps (J) come before generative models (G) and RL (H)
on purpose. They are the most job-relevant parts, so if time slips, the cut falls on the
mastery-heavy material.

**Checkpoints.** There are six pass/fail checkpoints, at weeks 6, 12, 16, 23, 29 and 36. For
example: implement logistic regression from scratch in 30 minutes, write causal self-attention in
45 minutes, or defend a fine-tune-vs-prompt decision with numbers. Each one also requires the
relevant projects to score at least 70% on their rubric.

## How much of the material is ready

The course material is generated one week at a time. **Weeks 1–12 of 36 are ready**, which is
one third of the course:

| | Ready | Total |
|---|---|---|
| Weeks of material | 12 (weeks 1–12) | 36 |
| Lessons (each with a runnable lab and an answer key) | 59 | 154 |
| Weekly quizzes | 12 | 36 |
| Weekly cheat sheets | 12 | 36 |
| Part cheat sheets | 2 (Parts A and B) | 11 |
| Project briefs | 3 (P1, P2, P3) | 12 |
| Paper reading guides | 1 (XGBoost) | 10 |

In terms of parts:

- **Part A, Foundations (weeks 1–6): complete.**
- **Part B, Classical ML (weeks 7–12): complete**, including the Part B cheat sheet. Part C (deep
  learning, from week 13) is next to be generated.
- **Parts C to L (weeks 13–36): planned in detail but not yet generated.** The syllabus for every
  week is fixed, so what is coming is known; only the lessons, labs and quizzes still need to be
  written.

What already exists for the whole course:

- The full outline, a detailed week-by-week syllabus for every part, a style guide and a glossary.
- A manifest that tracks all 625 course artifacts, so generation picks up exactly where it stopped.
- Pinned Python dependencies for every part, from NumPy to PyTorch, Hugging Face and diffusers.

Not done yet:

- **The per-part quality review.** Ready material has been generated and its labs run, but the
  independent review pass planned for each completed part has not been run on Part A or Part B yet.
- **The library refresh** planned for around week 24. Parts I and J use fast-moving tools and APIs,
  so they will be checked against current library versions before I study them.

## My own progress

<!-- Fill in: which week you are on, which projects you have submitted, checkpoint results. -->

- Currently studying: week __
- Projects submitted: __
- Checkpoints passed: __
