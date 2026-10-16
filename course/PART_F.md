# Part F. Transformers & LLMs — detailed syllabus

Course weeks 21–23. Weekly detail for Part F of [`COURSE_OUTLINE.md`](COURSE_OUTLINE.md), which stays
the source of truth: the topic ids, Core/Depth labels, project P6, checkpoint CP4 and the three papers
below are taken from it unchanged. Written 2026-10-16.

**Goal:** build and train a transformer from scratch and understand how modern LLMs are pretrained,
decoded and tuned.

Part F is where the course turns from "neural networks in general" into "the model behind every LLM
product you will build in Part I". The order is deliberate: you write every piece of a GPT yourself in
F1 and train it, F2 explains what changes when the same model is made a thousand times bigger and how
text is generated from it, and F3 covers the post-training that turns a text predictor into an
assistant. RLHF is split as the outline says: F3.2 gives the pipeline at overview level and implements
DPO; the reward model plus PPO is taught in full in H2.3, after PPO exists.

## At a glance

| Week | Course wk | Theme | Topics | Hours | Project | GPU |
|---|---|---|---|---|---|---|
| F1 | 21 | The transformer from scratch | F1.1–F1.4 | 13 | P6 M1 | helpful (F1.4) |
| F2 | 22 | Pretraining & inference | F2.1–F2.5 | 13 | **P6 due** | helpful |
| F3 | 23 | Post-training | F3.1–F3.4 | 12.5 | P6 review fixes; **CP4** | helpful |

About 38.5 hours in total. Each week's hours include about 1 hour for the weekly quiz and cheat sheet
(2 hours in F3, which also holds the CP4 self-test). Topic hours cover reading the lesson and doing its
lab and exercises.

**How the hours are counted.** *Reading* is working through the lessons and the papers, including the
worked examples by hand. *Coding* is the labs and the code exercises. *Project* is P6.

**Teaching order.** In F1 and F3 the topics are not taught in id order where a lab needs another
topic first: F2 starts with pretraining (F2.2) and decoding (F2.4) because P6, due that week, needs
perplexity and decoding strategies; F3 starts with running open models (F3.4) because the SFT and DPO
labs load one. The ids stay as the outline defines them.

**If a week runs long,** drop the Depth topics first, as the outline's catch-up plan says: F2.3
(scaling laws) and F2.5 (efficient attention). Read the Chinchilla reading guide's summary even if you
skip F2.3, because "how many tokens per parameter" comes up in interviews. On the fast path, keep
F1.1–F1.4, F2.1, F2.2, F2.4 and F3.1–F3.2, and build P6 at its CPU size.

**Libraries this part needs** (Step 2 pins the versions): `numpy`, `torch` (CPU build on the laptop),
`matplotlib`, `tiktoken` (to compare your BPE with GPT-2's tokenizer in F1.3), `tokenizers` and
`transformers` from Hugging Face (F1.3, F2.1, F2.4, F3.4), `datasets` (F2.2, F3.1, F3.2), and `trl`
(only to check your SFT and DPO against a reference implementation in F3.1–F3.2). Optionally
`llama-cpp-python` for running a quantized GGUF model in F3.4. All of these APIs move fast, `trl` most
of all: VERIFY (as of 2026-10).

**Models** (downloaded into the Hugging Face cache, never committed): GPT-2 small (124M) for decoding
and perplexity comparisons; a small BERT (for example `bert-base-uncased` or a distilled one) and
`t5-small` for F2.1; a small open base model and its instruct version in the 100M–500M range (for
example the SmolLM2 or Qwen2.5 families) for F3. Model names, sizes and licences change quickly: VERIFY
(as of 2026-10). On a 15 GB laptop, stay at or below about 0.5B parameters in fp32 (about 2 GB of
weights, plus optimizer state when training).

**Datasets** (all download into `data/`, never committed): your P6 corpus from E2 M0 (1–10 MB of
plain text) for every F1 lab, F2.2 and F2.3; IMDB again for the F2.1 feature comparison; a small
public instruction dataset (for example `databricks-dolly-15k`, a few hundred rows) for F3.1; a small
public preference dataset with chosen/rejected pairs (a few hundred rows of one from the Hugging Face
Hub) for F3.2; and hand-written prompt sets for F3.3. TinyStories is a good larger corpus for the
P6 `GPU_SCALE` run. VERIFY each source, its licence and its contents when the lab is generated.

**Hardware:** every lab runs on the laptop CPU at default settings: models of 0.5–2M parameters you
train yourself, block sizes of 64–128 tokens, a few thousand steps at most, and pretrained models no
larger than about 0.5B used only for inference or a few dozen training steps. Intel integrated
graphics are not used. Each lab has a `GPU_SCALE` block with the full-size settings for a free Colab
or Kaggle GPU; how to use those is under "Running Part F on a free GPU" at the end.

---

## F1 · The transformer from scratch (course week 21)

**Learning objectives**

- I can write scaled dot-product attention `softmax(Q Kᵀ / √d_k + M) V` with every shape stated for a
  batch, `(B, T, d_k)` in and `(B, T, d_v)` out, and compute it by hand for 3 tokens with `d_k = 2`.
- I can explain why the scores are divided by `√d_k`, and show the softmax saturating without it.
- I can implement multi-head attention from scratch, including the reshape `(B, T, d) → (B, h, T, d/h)`
  and back, and match PyTorch's `scaled_dot_product_attention` to within 1e-6.
- I can build a causal mask, explain how it lets one forward pass train on every position at once
  (teacher forcing in parallel, from E2.3), and show that removing it lets the model cheat.
- I can show that attention without positional information is permutation-equivariant, and
  implement sinusoidal encodings and RoPE, and check that RoPE scores depend only on relative offset.
- I can assemble a pre-LayerNorm transformer block (attention, MLP, two residual paths), explain why
  pre-LN trains more stably than post-LN in deep stacks, and count its parameters (about `12·d²` per
  block, plus `V·d` for embeddings).
- I can train a byte-level BPE tokenizer from scratch, encode and decode with an exact round trip, and
  explain why the same text costs different numbers of tokens in different languages and tokenizers.
- I can train a small GPT on my corpus: overfit one batch first, then train with AdamW, warmup and
  cosine decay, and gradient clipping, and report validation loss against the E2 bigram baseline in
  the same unit (nats per character).

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| F1.1 | Scaled dot-product self-attention, multi-head attention and causal masking, from scratch (+ *Attention Is All You Need*) | Core | 3 |
| F1.2 | The transformer block: residuals, LayerNorm, MLP, positional encodings (sinusoidal, RoPE) | Core | 2.5 |
| F1.3 | BPE tokenization from scratch, compared with a production tokenizer | Core | 2.5 |
| F1.4 | Training a small GPT: tiny on CPU, scaled up on a free GPU | Core | 3 |

**Hours:** reading 4 · coding 7 · project 1 · quiz and cheat sheet 1 · **total 13**

**Labs**

- **F1.1:** single-head attention from scratch in NumPy, starting from the scaled dot-product scoring
  you wrote at the end of E2.4, now with `Q`, `K` and `V` all projected from the same sequence. Assert
  every shape. Add a causal mask (upper triangle set to `-inf` before the softmax, never after), then
  multi-head attention with the split-heads reshape and an output projection. Port it to PyTorch and
  check it against `torch.nn.functional.scaled_dot_product_attention` (with `is_causal=True`) and
  `nn.MultiheadAttention` after copying weights across (its packed `in_proj_weight` layout: VERIFY in
  the docs). Then three experiments: a histogram of softmax outputs with and without the `√d_k`
  scaling at `d_k = 256`; a permutation test that shuffles the input tokens and shows the outputs are
  shuffled the same way (no notion of order yet); and a "cheating" test where the mask is removed and
  a next-token loss collapses to near zero on training data while generation is garbage. Bridge: one
  attention head is like a soft database lookup, query against keys returning a weighted mix of
  values; the analogy breaks because every row matches every key a little, and the "index" is learned.
- **F1.2:** the rest of the block. LayerNorm from scratch (C3.4) checked against `nn.LayerNorm`; the
  MLP with GELU and the 4× hidden width; residual connections; pre-LN and post-LN variants. Sinusoidal
  positional encodings, plotted as a heatmap, and RoPE from scratch (rotate pairs of query/key
  dimensions by a position-dependent angle), with a test that `score(q at i, k at j)` depends only on
  `i − j`. Stack 12 blocks of each LayerNorm variant on random data and plot gradient norms per layer
  at initialization: the post-LN stack needs warmup to survive, the pre-LN stack does not (C3.2,
  C3.3). Finish by writing a `count_params` function and checking it against the formula and against
  `sum(p.numel() ...)`.
- **F1.3:** byte-level BPE from scratch: start from the 256 byte values, count adjacent pairs, merge
  the most frequent, repeat until the target vocabulary size, and save the merge list. Write `encode`
  (apply merges in learned order) and `decode`, and test the round trip on your corpus plus awkward
  inputs (emoji, accented names, code, numbers, leading spaces). Add special tokens such as
  `<|endoftext|>` that are never produced by merges. Train a vocabulary of about 1,000–4,000 tokens on
  your P6 corpus, then train the Hugging Face `tokenizers` BPE on the same text and compare merges and
  token counts. Compare with GPT-2's tokenizer through `tiktoken`: tokens per character on English,
  on another language you know, on Python code and on numbers. Note what this means for cost and
  context length later (I4.3), and why arithmetic on numbers is hard for models whose tokenizer splits
  "1234" unpredictably. CPU default: train on 1 MB, a few seconds to a minute with a simple pair count.
- **F1.4:** a decoder-only GPT in PyTorch from your F1.1–F1.3 parts: token and position embeddings
  (or RoPE), `n_layer` pre-LN blocks, a final LayerNorm and a language-model head with tied weights.
  A data loader that samples random contiguous blocks from the tokenized train split. Following C4.3,
  first overfit a single batch to near-zero loss, then train with AdamW (C3.1), linear warmup plus
  cosine decay (C3.2), weight decay on matrices only, and gradient clipping (E2.1). Log train and
  validation loss, convert validation loss to nats per character (total nats divided by the number of
  characters, so token-level and character-level models can be compared), and sample with
  temperature. CPU default: 4 layers, `d = 128`, 4 heads, block size 128, BPE vocabulary of about
  2,000, roughly 1–2M parameters, about 2,000 steps in a few minutes. `GPU_SCALE`: 6 layers, `d = 384`,
  6 heads, block size 256 (about 10M parameters) with mixed precision (C4.2): fp16 with a gradient
  scaler on a T4, which has no bf16 support (VERIFY the GPU you are given).

**Paper:** *Attention Is All You Need* (Vaswani et al., 2017), with its reading guide in `papers/`.
Read sections 3.1–3.5 (the architecture, attention, positional encoding) and the "Why self-attention"
comparison with recurrent layers. Note what has changed since: modern GPTs are decoder-only, use
pre-LN, and mostly use RoPE rather than sinusoidal encodings; the reading guide flags each one. About
1 hour, counted in F1.1.

**Project milestone: P6 M1, tokenizer and first training run (1 h).** On the corpus and split you
chose in E2 M0: train your F1.3 BPE tokenizer on the training split only, save it, and record its
tokens-per-character on validation text. Run the F1.4 GPT at CPU size on the corpus, show that it
overfits one batch, and record the first validation loss in nats per character next to the two M0
baselines. This is the skeleton P6 grows from in F2.

**Prerequisites:** E2.4 (scaled dot-product attention and masking, the starting point), E2.3 (teacher
forcing), E2.1 (gradient clipping, the character model and bigram baseline), E1.1 (vocabularies,
special tokens), E1.3 (embedding layers); A1.2 (broadcasting), A2.1–A2.2 (shape rules, matrix
products), A6.4 (cross-entropy, nats), C1.4 (stable softmax and cross-entropy), C2.1–C2.3 (`nn.Module`,
training loop, device handling), C3.1 (AdamW), C3.2 (warmup and cosine decay), C3.3 (initialization),
C3.4 (LayerNorm), C3.5 (dropout and weight decay), C4.2 (mixed precision, for `GPU_SCALE`), C4.3
(overfit one batch, reading loss curves). Within the week: F1.2 needs F1.1; F1.4 needs F1.1–F1.3.

**Resources** (optional)

- Andrej Karpathy, *Neural Networks: Zero to Hero*, the "Let's build GPT" and "Let's build the GPT
  Tokenizer" videos: https://karpathy.ai/zero-to-hero.html (VERIFY link)
- *The Annotated Transformer* (Harvard NLP), the paper rebuilt line by line in PyTorch:
  https://nlp.seas.harvard.edu/annotated-transformer/ (VERIFY link)
- Vaswani et al., "Attention Is All You Need" (2017): https://arxiv.org/abs/1706.03762 (VERIFY link)

**GPU:** helpful for F1.4 only. F1.1–F1.3 are small and run on the CPU in seconds. F1.4's CPU default
gives a model that clearly beats the bigram baseline but produces rough samples; the `GPU_SCALE` run
on a free T4 gives noticeably better text in roughly 15–30 minutes (VERIFY on your session). Keep the
from-scratch NumPy attention for checks only, and use the PyTorch model for every training run.

**Used later in:** attention and causal masks in every model from here on, in ViT and CLIP (G2.1–G2.2)
and in CP4's 45-minute attention test (F1.1); the block and parameter count in F2.3 scaling and F2.5
memory maths (F1.2); token counting for cost and context limits in I1.1 and I4.3, and chunking in I2.1
(F1.3); the training loop that F2.2 scales, F3.1 fine-tunes and F3.2 preference-tunes (F1.4); P6.

---

## F2 · Pretraining & inference (course week 22)

**Learning objectives**

- I can describe a pretraining data pipeline (collection, deduplication, quality filtering,
  tokenization, packing into fixed-length blocks with document separators) and name one failure each
  step prevents.
- I can compute perplexity as `exp(mean negative log-likelihood per token)`, explain why perplexities
  from different tokenizers are not comparable, and convert to bits per byte or nats per character.
- I can estimate training compute with `C ≈ 6·N·D` FLOPs (N parameters, D tokens) and turn it into
  GPU-hours for a stated throughput.
- I can explain the difference between encoder-only (BERT, masked LM), decoder-only (GPT, causal LM)
  and encoder–decoder (T5) models by their attention masks and training objectives, and say which to
  use for classification, generation and translation.
- I can implement greedy decoding, beam search, temperature, top-k and top-p sampling from scratch on a
  logits vector, and say what each does to diversity and repetition.
- I can implement a KV cache, show it gives identical outputs to the uncached model, and compute its
  memory: `2 × n_layers × n_kv_heads × d_head × seq_len × bytes` per sequence.
- (Depth) I can state the shape of a scaling law (loss falling as a power law in parameters, data and
  compute), the Chinchilla result that parameters and tokens should grow roughly together, and fit a
  power law to my own small runs.
- (Depth) I can explain multi-query and grouped-query attention as a KV-cache saving, and the
  FlashAttention idea (tiling plus an online softmax so the `T × T` score matrix is never stored).

**Topics** (in teaching order)

| Id | Topic | Level | Hours |
|---|---|---|---|
| F2.2 | Pretraining: data pipelines, objective, perplexity, compute cost | Core | 2 |
| F2.4 | Decoding: greedy, beam search, temperature, top-k, top-p, and the KV cache | Core | 2.5 |
| F2.1 | Encoder, decoder and encoder–decoder models: BERT (masked LM) versus GPT (causal LM) versus T5 | Core | 1.5 |
| F2.3 | Scaling laws: loss versus parameters, data and compute; the compute-optimal trade-off (+ the Chinchilla paper) | Depth | 1.5 |
| F2.5 | Efficient attention and long context: the FlashAttention idea, multi-query and grouped-query attention | Depth | 1 |

**Hours:** reading 3.5 · coding 5 · project 3.5 · quiz and cheat sheet 1 · **total 13**

This is a project-deadline week, so the two Depth topics are kept short; if P6 runs late, they are the
first to move to the catch-up list.

**Labs**

- **F2.2:** turn your F1.4 data loading into a small but real pretraining pipeline: exact
  deduplication by hash of normalized lines or paragraphs, a near-duplicate check with MinHash on
  shingles (from scratch, small), two or three quality heuristics (length, ratio of letters to symbols,
  repeated lines), then tokenization into one long token stream with `<|endoftext|>` between
  documents, packed into fixed blocks and written as a NumPy memmap. Print how many tokens each step
  removes. Compute perplexity for your P6 model on validation text, and for GPT-2 small on the same
  text; show why the raw perplexities cannot be compared and convert both to bits per byte. Estimate
  the FLOPs of your P6 run with `6·N·D`, time it, and compute the achieved FLOPs per second; compare
  with a published GPU peak (VERIFY the number) to get a rough utilization. Bridge: the data pipeline
  is an ETL job, and dedup and filtering are the data-quality checks you would write for any
  warehouse; the honest difference is that here the "consumer" memorizes duplicates.
- **F2.4:** decoding from scratch on top of your P6 model, operating on a logits vector: greedy;
  temperature; top-k; top-p (nucleus) with the sort-and-cumulative-sum trick; a repetition penalty;
  and beam search with length normalization (extending your E2.3 version to a GPT). For a fixed prompt,
  plot the next-token distribution before and after each transform. Check greedy and beam search
  against Hugging Face `generate` on GPT-2 small (deterministic, so the token ids must match exactly),
  and sampling against it for the same seed where the API allows (VERIFY). Then add a KV cache to your
  GPT: cache keys and values per layer, feed only the new token each step, assert identical logits to
  the uncached model, and plot tokens per second against sequence length with and without the cache.
  Finish with the memory formula applied to a 7B-class model at a 32k context (VERIFY the
  architecture numbers you use) to see why the cache, not the weights, often limits batch size.
- **F2.1:** the three architectures by their masks: draw the bidirectional, causal and cross-attention
  mask matrices for one example and implement each as a flag in your F1.1 attention. Implement BERT's
  masked-LM corruption from scratch (15% of tokens; of those 80% `[MASK]`, 10% random, 10% unchanged)
  and T5's span corruption on a sentence, and print the inputs and targets. Use a pretrained BERT for
  fill-mask and `t5-small` for a text-to-text task through `transformers`. Then a frozen-feature
  comparison on 2,000 IMDB reviews: BERT's `[CLS]` or mean-pooled embedding versus GPT-2's last-token
  embedding, each with logistic regression, against your E1.2 TF-IDF baseline. Report the honest
  result and explain why a bidirectional encoder usually wins for classification. CPU default: 2,000
  reviews, inference only, a few minutes.
- **F2.3 (Depth):** a tiny scaling study on your P6 corpus: train 5–6 GPTs from about 50K to 2M
  parameters for a fixed number of tokens, plot final validation loss against parameters on log–log
  axes, and fit `L(N) = E + A / N^α` with `scipy.optimize.curve_fit`. Then fix a compute budget, trade
  model size against tokens, and plot the resulting curve to find its minimum: a toy isoFLOP curve.
  Compare your exponent with the published ones and say why it differs (tiny models, one corpus, few
  steps). CPU default: very small models, about 4 minutes in total. `GPU_SCALE`: 8 sizes up to about
  20M parameters, two compute budgets.
- **F2.5 (Depth):** add `n_kv_heads` to your attention so the same code does multi-head, grouped-query
  and multi-query attention; compare KV-cache size and generation speed for the three settings on
  your P6 model. Implement the FlashAttention idea for one head in NumPy: process keys and values in
  tiles, keep a running maximum and running sum for an online softmax, and show the result matches
  naive attention to within 1e-6 while never building the full `T × T` matrix. The point is the
  algorithm, not the speed; for speed, show that PyTorch's `scaled_dot_product_attention` picks a fused
  kernel on a GPU (VERIFY which backends are available on CPU and on T4). Close with a short note on
  long-context methods (sliding-window attention, RoPE scaling) and the "lost in the middle" effect,
  which matters for RAG context assembly in I2.3.

**Paper:** *Training Compute-Optimal Large Language Models* ("Chinchilla", Hoffmann et al., 2022),
with its reading guide in `papers/`. Read the introduction, the three approaches to estimating the
optimal model size, and the comparison of Chinchilla with Gopher. Take away the result that, for a
fixed compute budget, parameters and training tokens should be scaled in roughly equal proportion,
which the paper's fits put at about 20 tokens per parameter (check the exact figure in the paper), and
that many models trained before it were too large for their data. Also note why production models are
now often trained well past that ratio: inference cost favours a smaller model trained longer. About 1
hour, counted in F2.3. If you skip F2.3, read only the reading guide's one-page summary.

**Project milestone: P6 due, small GPT from scratch (3.5 h).** Finish and submit P6 (brief in
`projects/`, rubric in `answers/projects/`). It must contain: your F1.3 BPE tokenizer trained on the
training split; your F1.4 decoder-only transformer trained at least at CPU size (the `GPU_SCALE` run is
encouraged and is what the rubric's "sample quality" line expects); train and validation loss curves;
validation loss in nats per character next to the bigram and LSTM baselines from M0 (CP4 requires P6
to beat the bigram); perplexity and bits per byte from F2.2; samples from the same prompt under
greedy, beam search, temperature 0.7 and 1.2, top-k and top-p, with a paragraph on what each strategy
did; generation speed with and without your KV cache; a short README and a model card (training data
and its licence, intended use, limitations, the fact that it will produce plausible nonsense). Then
ask for a review with `Review my-work/<your P6 folder>`.

**Prerequisites:** F1.1–F1.4 (every lab builds on your GPT), P6 M0–M1; E1.2 (TF-IDF baseline), E2.3
(greedy and beam search, encoder–decoder); A5.2 (categorical distribution, sampling), A6.4
(cross-entropy, entropy, nats versus bits), B2.1 (logistic regression for the frozen-feature probe),
B4.1 (classification metrics), C4.1 (GPU throughput and memory), C4.2 (mixed precision), C4.4 (data
parallel and FSDP concepts, for the compute discussion). Within the week: F2.4 needs F2.2's perplexity
code only for its report; F2.3 needs F2.2 (FLOPs); F2.5 needs F2.4 (the KV cache).

**Resources** (optional)

- Hugging Face, *LLM Course* (formerly the NLP Course), chapters on transformer models and
  tokenizers: https://huggingface.co/learn/llm-course (VERIFY link)
- Hugging Face blog, "How to generate text: using different decoding methods for language generation
  with Transformers": https://huggingface.co/blog/how-to-generate (VERIFY link)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022):
  https://arxiv.org/abs/2203.15556 (VERIFY link)

**GPU:** helpful. F2.1, F2.4 and F2.5 run on the CPU in minutes. The GPU pays off for the P6
`GPU_SCALE` training run (see F1.4) and the F2.3 `GPU_SCALE` sweep. Kaggle's longer sessions suit the
sweep; Colab is fine for a single P6 run. Save checkpoints to Drive or `/kaggle/working` during
training so a dropped session does not cost the run, and bring back only metrics, plots and samples.

**Used later in:** perplexity and bits per byte as an eval metric in J1.3 and P9 (F2.2); choosing an
encoder model for embeddings and reranking in I2.1–I2.3, and an encoder–decoder versus decoder-only
argument in system-design interviews (F2.1); temperature, top-p and the KV cache when you call LLM APIs
and reason about latency and cost in I1.1 and I4.3, and in LLM inference servers in J2.1 (F2.4);
"bigger model or more data?" decisions and fine-tune-versus-prompt economics in J1.4 (F2.3);
long-context limits and "lost in the middle" in RAG context assembly in I2.3 (F2.5).

---

## F3 · Post-training (course week 23)

**Learning objectives**

- I can load an open model and its tokenizer with Hugging Face `transformers` on CPU, generate with
  explicit decoding settings, and estimate its memory from parameter count and dtype.
- I can explain what weight quantization (int8, 4-bit) trades off, run a quantized model on CPU, and
  measure the change in speed, memory and output quality on a few prompts.
- I can explain what supervised fine-tuning (SFT) changes: the same next-token objective on
  instruction–response pairs, formatted with a chat template, with the loss masked to the response.
- I can implement SFT with prompt-token loss masking from scratch, and get the same loss as a
  reference trainer on the same batch.
- I can draw the RLHF pipeline (SFT model, preference data, reward model, PPO with a KL penalty to the
  reference model), say what each stage is for, and point to H2.3 for the full treatment.
- I can write the DPO loss, explain each term (policy and reference log-probabilities of the chosen and
  rejected responses, scaled by β), implement it from scratch, and show the chosen-minus-rejected
  margin growing during training.
- I can define hallucination, harmful output, refusal and over-refusal, build a small red-team and
  factuality test set, run it against a model, and report the results with honest caveats.

**Topics** (in teaching order)

| Id | Topic | Level | Hours |
|---|---|---|---|
| F3.4 | Running open models with Hugging Face `transformers` on CPU: loading, generating, quantized weights | Core | 2 |
| F3.1 | Instruction tuning (SFT): chat templates, data format, what changes in the model | Core | 2.5 |
| F3.2 | Preference tuning: the RLHF pipeline at overview level and DPO implemented on a tiny model (full RLHF in H2.3) (+ the InstructGPT paper) | Core | 3 |
| F3.3 | `[RAI]` Safety and alignment: hallucination, harmful outputs, refusals, red-teaming | Core | 2 |

**Hours:** reading 3.5 · coding 6 · project 1 · quiz, cheat sheet and CP4 2 · **total 12.5**

**Labs**

- **F3.4:** load a small base model and its instruct version (about 100M–500M parameters; VERIFY the
  current names) with `AutoTokenizer` and `AutoModelForCausalLM`. Print the parameter count, dtype and
  memory, and check it against `parameters × bytes per parameter`. Generate with explicit settings and
  connect each `generate` argument to your F2.4 code. Apply the instruct model's chat template with
  `apply_chat_template` and print the raw string so you can see the special tokens. Compare base and
  instruct outputs on the same five prompts. Then quantization: apply PyTorch dynamic int8
  quantization to the linear layers, or load a 4-bit GGUF version through `llama-cpp-python`
  (both CPU-friendly; `bitsandbytes` 4-bit is mainly a GPU path and returns in J1.3; VERIFY the
  current CPU support of each), and measure tokens per second, memory and how much the outputs change.
  Bridge: a model checkpoint is a large immutable artifact you pin by revision, like a Docker image
  digest; the difference is that "the same version" can still give different outputs under different
  decoding settings.
- **F3.1:** SFT twice. First from scratch on your own F1.4 GPT: invent a tiny chat format with
  special tokens (`<|user|>`, `<|assistant|>`, `<|end|>`), generate a synthetic instruction set
  (for example "reverse: abc" → "cba", "upper: hello" → "HELLO", "count the vowels in …"), and
  fine-tune with the loss masked to assistant tokens (labels set to `-100` elsewhere). Show the model
  learns to stop at `<|end|>` and follows instructions it never saw in that exact form, and show what
  goes wrong without the mask. Then the realistic version: SFT the small base model from F3.4 on a few
  hundred rows of a public instruction dataset with your own training loop, and check that your
  per-batch loss matches `trl`'s SFT trainer on the same batch and settings (VERIFY its API and
  defaults). Compare base and SFT outputs on held-out prompts. CPU default: 200 examples, sequences up
  to 128 tokens, a few dozen steps, under 5 minutes. `GPU_SCALE`: 2,000 examples, 512 tokens, 1 epoch.
  Note why full fine-tuning even a 0.5B model is heavy (weights plus AdamW state is roughly 16 bytes
  per parameter), which is the motivation for LoRA in J1.3.
- **F3.2:** the RLHF pipeline first, as a diagram and a short read: what a reward model is (the
  Bradley–Terry model, `P(chosen ≻ rejected) = σ(r_chosen − r_rejected)`), what PPO optimizes, and why
  the KL penalty to the reference model exists; the implementation waits for H2.3. Then DPO from
  scratch: a function for the summed log-probability of a response given a prompt (masked to response
  tokens, the same masking as F3.1), the DPO loss
  `−log σ(β[(log π_θ(y_w|x) − log π_ref(y_w|x)) − (log π_θ(y_l|x) − log π_ref(y_l|x))])` with a frozen
  reference copy, and logging of the implicit rewards, their margin and accuracy. CPU default: your
  F3.1-tuned tiny GPT with a synthetic preference (for example, for the same instruction, prefer the
  correct answer over a plausible wrong one, or a short answer over a padded one), so you can see
  exactly what DPO pushes and check it worked. Then compute the DPO loss for one batch of a public
  preference dataset on the F3.4 small model and check it against `trl`'s DPO trainer on the same
  batch and β (VERIFY its API). Experiment: vary β and watch how far the policy drifts from the
  reference (KL on held-out prompts). `GPU_SCALE`: DPO on the small model with about 1,000 preference
  pairs, starting from your F3.1 SFT checkpoint.
- **F3.3:** a small, benign safety evaluation of the F3.4 instruct model and your F3.1 SFT model. Build
  three prompt sets of 20–30 items each: (1) factuality, with questions that have known answers plus
  questions about made-up entities and false premises, to measure fabrication; (2) policy probes using
  harmless stand-ins, such as trying to extract a "secret password" from the system prompt, role-play
  jailbreak framings around it, and requests for personal details of a fictional person; and (3)
  over-refusal probes, safe requests that merely sound risky ("how do I kill a Python process?").
  Grade every answer by hand against a written rubric, and report the rate in each category with the
  number of items, so it is clear how uncertain a rate from 25 prompts is (A6.2). Then try one
  mitigation, a system prompt or a few SFT examples, and re-run. Write a half-page red-team report in
  the model-card style you have used since B4.6. This lab is the seed of the guardrail and prompt-
  injection work in I4.1–I4.2 and the injection test set in P8.

**Paper:** *Training language models to follow instructions with human feedback* ("InstructGPT",
Ouyang et al., 2022), with its reading guide in `papers/`. Read the introduction, the three-step method
(SFT, reward model, PPO) and its figure, and the results on human preference and on truthfulness and
toxicity. Note the result that a much smaller tuned model was preferred over a far larger untuned one,
and the "alignment tax" discussion. The reading guide maps each step to F3.1, F3.2 and H2.3 and adds a
one-page note on the DPO paper (Rafailov et al., 2023), which your lab implements. About 1 hour,
counted in F3.2.

**Project milestone:** none due. Spend the project hour fixing what your P6 review flagged, so P6
reaches 70% on its rubric for CP4. If your P5 review (Part D) is still below 70%, fix that first.

**Checkpoint CP4 (end of Part F).** From the outline, passed only when every criterion holds:

- Write causal self-attention from scratch in 45 minutes with correct shapes (do it in `my-work/`,
  timed, without notes, then ask for a review).
- P5 and P6 at least 70% on their rubrics; P6 beats the bigram baseline on validation loss.
- Explain BERT versus GPT and four decoding strategies.

If one fails, take one catch-up week on the topic ids your reviews flagged before starting Part I.

**Prerequisites:** F1.4 (your GPT and training loop), F2.1 (decoder-only models), F2.2 (objective and
log-likelihoods), F2.4 (decoding settings); E1.1 (special tokens); A5.5 (likelihood), A6.2
(confidence intervals, for small-sample rates), A6.4 (KL divergence, for the KL penalty and DPO's
derivation), B4.6 (model cards and auditing), C1.4 (log-softmax), C2.3 (saving and loading,
reproducibility), C3.1 (AdamW memory). Within the week: F3.1 needs F3.4; F3.2 needs F3.1; F3.3 needs
F3.4 and uses the F3.1 model.

**Resources** (optional)

- Ouyang et al., "Training language models to follow instructions with human feedback" (2022):
  https://arxiv.org/abs/2203.02155 (VERIFY link)
- Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
  (2023): https://arxiv.org/abs/2305.18290 (VERIFY link)
- Hugging Face TRL documentation (SFT and DPO trainers): https://huggingface.co/docs/trl (VERIFY link)

**GPU:** helpful, never required. Every CPU default works on your own tiny GPT or uses a pretrained
model of at most about 0.5B for inference and a few dozen training steps. A free T4 is useful for the
`GPU_SCALE` SFT and DPO runs on the small pretrained model. Load it in fp16 on the T4 (no bf16); if
memory runs out, cut the sequence length and batch size and use gradient accumulation (C4.2) rather
than reaching for LoRA, which is J1.3's job.

**Used later in:** loading and running open models as the free fallback in Part I, and in serving in
J2.1 (F3.4); chat templates and message roles behind every LLM API in I1.1 (F3.1); SFT as the base for
LoRA/QLoRA fine-tuning in J1.3 and P9 (F3.1); the full RLHF pipeline with a reward model and PPO in
H2.3, and reward hacking in H2.4 (F3.2); guardrails in I4.1, prompt injection in I4.2, the injection
test set in P8, and the safety section of every model card (F3.3).

---

## Running Part F on a free GPU

Free-tier GPUs, quotas and session limits change often: VERIFY (as of 2026-10).

- **Colab:** open the lab, then Runtime → Change runtime type → a GPU (usually a T4). Check with
  `torch.cuda.is_available()` and `torch.cuda.get_device_name()`. Upload the lab file or clone your
  repo, set the `GPU_SCALE` values, and save checkpoints and results to Google Drive every few hundred
  steps, because the VM is wiped when the session ends or disconnects.
- **Kaggle:** create a notebook, then Settings → Accelerator → GPU, and turn Internet on to download
  datasets and Hugging Face models (Kaggle asks for phone verification first). Save outputs to
  `/kaggle/working`. Kaggle's weekly GPU quota and longer sessions make it the better place for the
  P6 `GPU_SCALE` run and the F2.3 sweep; spend Colab on the shorter F3 runs.
- **Both:** every lab already picks `device = "cuda" if torch.cuda.is_available() else "cpu"`
  (C2.3), so the same file runs in both places. On a T4 use fp16 autocast with a gradient scaler; bf16
  needs a newer GPU. Hugging Face models download into the session's cache each time, so pick small
  ones. Bring back only results (metrics, loss curves, samples, eval tables), never datasets, model
  weights or checkpoints: they are not committed.

---

## You're ready for Part I when…

Part I (Applied LLM engineering, course weeks 24–27) comes next in the course order, before G and H.
This self-check covers CP4 and the Part F ideas Part I builds on. Answer each item without notes. The
answers belong in your quiz reviews, not here.

1. You can write causal multi-head self-attention from scratch in 45 minutes, with every shape
   correct and the mask applied before the softmax (CP4).
2. You can draw a pre-LN transformer block, say what each residual path and LayerNorm is for, and
   estimate a model's parameter count from `n_layer`, `d` and vocabulary size.
3. You can explain how BPE builds its vocabulary, and why token counts (and so cost and context
   usage) differ across languages, code and numbers.
4. Your P6 model beats the bigram baseline in nats per character, P5 and P6 are at 70% or more, and
   you can explain why perplexities from different tokenizers cannot be compared.
5. You can explain BERT versus GPT versus T5 by their masks and objectives, and say which you would use
   for classification, embeddings and generation.
6. You can explain greedy, beam search, temperature, top-k and top-p, and say which settings you would
   choose for extracting structured data and for brainstorming (CP4 asks for four).
7. You can explain what a KV cache stores, compute its size for a stated model and context length, and
   say why it affects latency and how many requests a server can batch.
8. You can say what SFT changes and what it does not, why the loss is masked to the response, and what
   a chat template is.
9. You can draw the RLHF pipeline, write the DPO loss and explain what β controls.
10. You can define hallucination and over-refusal, and explain why a rate measured on 25 prompts needs
    a confidence interval before anyone acts on it.

If two or more items fail, revisit the topic ids your quiz reviews flagged before you start I1.
Items 6, 7 and 8 matter most for Part I: every API call there sets decoding parameters, pays for
tokens and sends messages through a chat template.
