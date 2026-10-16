# Part E. NLP & sequence models — detailed syllabus

Course weeks 19–20. Weekly detail for Part E of [`COURSE_OUTLINE.md`](COURSE_OUTLINE.md), which stays
the source of truth: the topic ids, Core/Depth labels and the paper below are taken from it unchanged.
Written 2026-10-16.

**Goal:** represent text as numbers and model sequences, ending at the attention mechanism that
transformers are built on.

Part E is deliberately short and pointed. Everything in it either becomes a baseline you must beat
later (TF-IDF, bag-of-words), an idea that transformers reuse directly (embeddings, encoder–decoder,
teacher forcing, attention), or the problem that transformers were invented to solve (the sequential,
vanishing-gradient RNN). Subword tokenization (BPE) is **not** here: it is F1.3, where it feeds the GPT.

## At a glance

| Week | Course wk | Theme | Topics | Hours | Project | GPU |
|---|---|---|---|---|---|---|
| E1 | 19 | Text representation | E1.1–E1.4 | 11 | — | no |
| E2 | 20 | Sequence models | E2.1–E2.4 | 13 | P6 M0 (optional head start) | helpful |

About 24 hours in total. Each week's hours include about 1 hour for the weekly quiz and cheat sheet.
Topic hours cover reading the lesson and doing its lab and exercises.

**How the hours are counted.** *Reading* is working through the lessons and the paper, including the
worked examples by hand. *Coding* is the labs and the code exercises. *Project* is the P6 head start.

**No project is due in Part E** (see the course map). That makes these two of the lighter weeks in the
course. Use the slack in two ways: catch up on anything your Part C or D reviews flagged, and start P6
(the small GPT, due in course week 22) with milestone M0 in E2. M0 is not graded on its own; it moves
an hour of F-week work forward.

**If a week runs long,** drop E1.4 (the only Depth topic) first, as the outline's catch-up plan says.
On the fast path, only E1.2, E1.3 and E2.4 are kept; if you are on it, read the E2.1–E2.3 lessons for
intuition and skip their labs, because E2.4's lab builds on the E2.3 encoder–decoder (the lab ships a
working one you can import).

**Libraries this part needs** (Step 2 pins the versions): `numpy`, `scikit-learn` (TF-IDF and the
linear classifier, to check your from-scratch versions), `torch` (CPU build on the laptop),
`matplotlib`, and `gensim` (only to load pretrained GloVe and fastText vectors in E1.4 and to compare
with your word2vec in E1.3). gensim's downloader names and the available pretrained vector sets are
fast-moving: VERIFY (as of 2026-10).

**Datasets** (all download into `data/`, never committed): the IMDB movie-review sentiment dataset
(E1.1–E1.2, and the sentiment classifier in E2.2), `text8` (the first 100 MB of cleaned English
Wikipedia, a standard word2vec benchmark; the CPU default uses a 5 MB slice) for E1.3, pretrained
`glove-wiki-gigaword-100` and a small fastText model through gensim for E1.4, the "tiny Shakespeare"
character corpus (about 1 MB) for E2.1, and synthetic data generated inside the labs for E2.2–E2.4
(sequence reversal and date-format translation, so you can see exactly what the model must learn).
VERIFY each source when the lab is generated.

**Hardware:** every lab runs on the laptop CPU at default settings: small vocabularies (≤ 20,000
words), corpus slices of a few MB, hidden sizes of 64–256 and 1–3 epochs. Each lab has a `GPU_SCALE`
block with the full-size settings for a free Colab or Kaggle GPU. How to use those is under "Running
Part E on a free GPU" at the end.

---

## E1 · Text representation (course week 19)

**Learning objectives**

- I can write a text-normalization and tokenization function (lowercasing, Unicode normalization,
  punctuation handling, word and character tokens) and name a case where each step destroys information.
- I can build a vocabulary with a minimum-frequency cut-off and `<unk>` / `<pad>` tokens, and turn a
  batch of texts into a padded `(batch, seq_len)` integer tensor.
- I can compute the TF-IDF matrix of a 3-document toy corpus by hand, and implement TF-IDF from scratch
  in NumPy that matches scikit-learn's `TfidfVectorizer` to within 1e-6 (with the same settings).
- I can train a bag-of-words / TF-IDF + logistic-regression sentiment classifier, report it with the
  right metrics, and read its top positive and negative weights.
- I can explain why one-hot vectors carry no notion of similarity, and what the distributional
  hypothesis ("a word is known by the company it keeps") says.
- I can implement skip-gram with negative sampling from scratch, derive its gradient for one
  (centre, context, negative) triple, and check it with finite differences.
- I can find nearest neighbours by cosine similarity, solve analogies with vector arithmetic, and say
  when analogy results are misleading.
- (Depth) I can explain how fastText handles words it has never seen, load pretrained GloVe vectors,
  and measure and partly remove a gender direction in them.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| E1.1 | Text preprocessing: normalization, word and character tokenization, vocabularies, n-grams | Core | 2 |
| E1.2 | Bag-of-words and TF-IDF with a linear classifier, the baseline every text model must beat | Core | 2.5 |
| E1.3 | word2vec (skip-gram with negative sampling) from scratch, and embedding geometry (+ the word2vec paper) | Core | 4 |
| E1.4 | Subword and pretrained static embeddings (fastText, GloVe); `[RAI]` bias in embeddings | Depth | 1.5 |

**Hours:** reading 3.5 · coding 6.5 · project 0 · quiz and cheat sheet 1 · **total 11**

**Labs**

- **E1.1:** a `normalize(text)` function (Unicode NFKC, lowercasing, whitespace, an optional rule for
  numbers and URLs) and two tokenizers: a regex word tokenizer and a character tokenizer. Run both on
  awkward inputs (contractions, hyphens, emoji, "U.S.A.", accented names) and print what each does.
  Build a `Vocab` class (token → id, id → token, `min_freq`, `<unk>`, `<pad>`, `encode`/`decode`) and
  an n-gram extractor. Plot word frequency against rank on a log–log scale for IMDB and see Zipf's
  law, which is why vocabularies need a cut-off. Finally, pad and batch a few reviews into a tensor
  with a mask; you will reuse this in every E2 lab. Bridge: a vocabulary is a lookup table with a
  fallback key, much like an enum with an `UNKNOWN` member.
- **E1.2:** count vectors and TF-IDF from scratch with a SciPy sparse matrix (raw counts, then
  `tf × idf` with smoothing, then L2 row normalization), checked against `CountVectorizer` and
  `TfidfVectorizer` with matching flags. Train your B2.1 logistic regression and scikit-learn's on
  IMDB with unigrams, then unigrams + bigrams; report accuracy and F1 with a proper validation split
  (fit the vectorizer on the training set only: the B1.4 leakage rule, applied to text). Print the
  20 most positive and negative weights and three misclassified reviews, and explain why "not good"
  breaks the unigram model. Also compare with multinomial naive Bayes (B2.4). Record your best
  validation score: **every text model in E2 and later must beat this number** or justify why not.
  CPU default: 10,000 reviews, a few seconds of training.
- **E1.3:** word2vec skip-gram with negative sampling from scratch in NumPy. Generate
  (centre, context) pairs with a sliding window, subsample frequent words, draw negatives from the
  unigram distribution raised to the 3/4 power, and write the loss and the gradients for the two
  embedding matrices `W_in, W_out` of shape `(V, d)` yourself. Check one gradient step against
  finite differences (A4.1), then re-implement the model in PyTorch with `nn.Embedding` and check the
  loss agrees on the same batch. Train on a 5 MB `text8` slice with `d = 100`. Inspect the result:
  nearest neighbours by cosine similarity (A1.5), the "king − man + woman" analogy, and a 2-D PCA
  (B5.4) plot of a few word groups (countries and capitals, numbers, verbs). Then train gensim's
  `Word2Vec` with the same settings and compare neighbours. CPU default: 5 MB, 1 epoch, a few
  minutes. `GPU_SCALE`: full `text8`, 5 epochs, in the PyTorch version.
- **E1.4 (Depth):** split a word into fastText character n-grams by hand and show how a vector is
  built for an unseen word such as a misspelling. Load pretrained GloVe and fastText vectors through
  gensim, and compare them with your E1.3 embeddings on the same neighbour and analogy queries. Then
  the `[RAI]` part: find a gender direction from definitional pairs ("he"−"she", "man"−"woman"),
  project occupation words onto it and plot them, run a small WEAT-style association test, and apply
  the "neutralize" step of hard debiasing. Show that bias measured by the projection drops but a
  classifier trained on the debiased vectors can still recover gender from neighbours: debiasing
  geometry is not the same as removing bias.

**Paper:** *Efficient Estimation of Word Representations in Vector Space* (Mikolov, Chen, Corrado,
Dean, 2013), with its reading guide in `papers/`. Read the introduction, the two model architectures
(CBOW and skip-gram) and the analogy evaluation. Note that negative sampling, which your lab
implements, comes from the follow-up paper (Mikolov et al., 2013, "Distributed Representations of
Words and Phrases and their Compositionality"); the reading guide covers how the two fit together.
About 1 hour, counted in E1.3.

**Project milestone:** none this week.

**Prerequisites:** A1.2 (broadcasting and vectorizing), A1.5 (dot product, norms, cosine similarity),
A4.1 (finite-difference gradient checks), A4.3 (chain rule), A5.2 (categorical distribution, for
sampling negatives), A6.4 (cross-entropy), B1.4 (leakage: fit the vectorizer on train only), B2.1
(logistic regression), B2.4 (multinomial naive Bayes), B4.1 (precision, recall, F1), B5.4 (PCA, for
the embedding plot), C1.4 (stable softmax and cross-entropy), C2.1 (`nn.Module`, `nn.Embedding`
builds on it). Within the week: E1.2 and E1.3 need E1.1; E1.4 needs E1.3; the E1.4 fairness part
reuses the group-metric thinking from B4.6.

**Resources** (optional)

- Stanford CS224N, *Natural Language Processing with Deep Learning*, lecture notes on word vectors:
  https://web.stanford.edu/class/cs224n/ (VERIFY link)
- Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft), chapters "Vector Semantics and
  Embeddings" and "Naive Bayes, Text Classification": https://web.stanford.edu/~jurafsky/slp3/ (VERIFY link)
- Bolukbasi et al., "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word
  Embeddings" (2016): https://arxiv.org/abs/1607.06520 (VERIFY link)

**GPU:** no. Every E1 lab runs on the CPU in a few minutes. Only the optional `GPU_SCALE` run of E1.3
(full `text8`, 5 epochs) benefits from a GPU, and it also finishes on the CPU if you leave it running.

**Used later in:** the token-embedding layer of every model in E2, F1 and beyond (E1.1, E1.3); the
TF-IDF baseline in hybrid search and as the "must beat" baseline in I2.2 (E1.2); embedding models and
cosine-similarity search in I2.1 and P7 (E1.3); subword ideas, which BPE generalizes in F1.3 (E1.4);
the embedding-bias audit as a model-card section from here on (E1.4).

---

## E2 · Sequence models (course week 20)

**Learning objectives**

- I can write the recurrence `h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)` with every shape stated, and
  unroll it by hand for three time steps with small numbers.
- I can implement an RNN forward pass and backpropagation through time from scratch, and match
  PyTorch autograd's gradients to within 1e-6.
- I can explain why gradients vanish or explode over long sequences (repeated multiplication by
  `W_hh` and the tanh derivative), show it in a gradient-norm plot, and fix explosion with gradient clipping.
- I can write the LSTM and GRU update equations, say in one sentence what each gate does, and explain
  why the LSTM's additive cell-state path helps gradients survive.
- I can implement an LSTM cell from scratch that matches `torch.nn.LSTMCell` given the same weights,
  and handle variable-length batches with padding and masking (or packed sequences).
- I can build a sequence-to-sequence encoder–decoder, train it with teacher forcing, decode it
  greedily, and explain exposure bias.
- I can explain the information bottleneck of a fixed-size context vector, and show it as accuracy
  falling with sequence length.
- I can implement Bahdanau (additive) attention from scratch: scores, softmax weights, and the
  context vector, with shapes, and plot an attention map that shows the alignment the model learned.
- I can state how dot-product attention differs from additive attention, and say what transformers
  keep from this week and what they throw away (the recurrence).

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| E2.1 | Recurrent networks and backpropagation through time; why gradients vanish | Core | 3 |
| E2.2 | LSTM and GRU: what the gates do | Core | 2.5 |
| E2.3 | Sequence-to-sequence encoder–decoder models and teacher forcing | Core | 2.5 |
| E2.4 | Attention (Bahdanau-style): letting the decoder look back, the bridge to transformers | Core | 3 |

**Hours:** reading 4 · coding 7 · project 1 · quiz and cheat sheet 1 · **total 13**

**Labs**

- **E2.1:** a vanilla RNN from scratch in NumPy: forward over a sequence of shape `(T, batch, d_in)`,
  then BPTT by hand, keeping every `h_t` for the backward pass. Check the gradients against PyTorch
  autograd on the same weights, the same discipline as your C1.3 engine. Then a character-level
  language model on tiny Shakespeare in PyTorch (`nn.RNN`): train it, sample from it with a
  temperature, and compare its validation loss with a character bigram baseline (counting, no
  training). Plot the gradient norm with respect to `h_0` against sequence length to watch it
  vanish, then crank up the weight scale (C3.3) to watch it explode, and add
  `clip_grad_norm_`. Also run truncated BPTT and explain the trade-off. CPU default: hidden size 128,
  sequence length 64, 2,000 steps, a few minutes. `GPU_SCALE`: hidden 512, 2 layers, 20,000 steps.
- **E2.2:** an LSTM cell and a GRU cell from scratch in PyTorch (only `nn.Linear` and elementwise
  operations), each checked against `nn.LSTMCell` / `nn.GRUCell` after copying the weights across
  (watch the gate ordering in PyTorch's weight layout: VERIFY in the docs). Swap them into the E2.1
  character model and compare loss curves for RNN, LSTM and GRU at the same parameter count. A
  synthetic long-memory task ("copy the first token after 50 steps of noise") on which the vanilla
  RNN fails and the LSTM succeeds; plot the forget-gate activations to see what it learned. Finally,
  an LSTM sentiment classifier on IMDB with padding, masking and `pack_padded_sequence`, compared
  with your E1.2 TF-IDF baseline. Expect it **not** to beat TF-IDF clearly on the CPU budget; the
  lesson explains why, and that is the honest result. `GPU_SCALE`: full IMDB, pretrained GloVe
  initialization from E1.4, 5 epochs.
- **E2.3:** a GRU encoder–decoder on two synthetic tasks: reversing a string of digits, and
  translating human-written dates ("16 Oct 2026", "October 16th, 2026") into ISO format
  ("2026-10-16"). Implement `<sos>` / `<eos>` handling, teacher forcing with a ratio you can change,
  greedy decoding, and exact-match accuracy. Plot accuracy against input length to show the
  fixed-context bottleneck. Compare teacher forcing ratios 1.0, 0.5 and 0.0 on training speed and
  test accuracy, and show an example of exposure bias (one early error derails the rest). Write a
  short beam-search decoder (beam width 3) and compare with greedy; F2.4 returns to decoding in
  depth. CPU default: 10,000 pairs, 3 epochs, a few minutes. `GPU_SCALE`: an English–French
  sentence-pair set (for example the Tatoeba pairs used in the PyTorch seq2seq tutorial; VERIFY the
  source), 20 epochs.
- **E2.4:** add Bahdanau attention to the E2.3 decoder. Write it from scratch: the additive score
  `vᵀ tanh(W_s s_{t-1} + W_h h_i)`, masked softmax over the encoder positions (padding must get zero
  weight), and the context vector `Σ α_i h_i`, with an assertion on every shape. Retrain on the date
  task and on long reversals, and plot accuracy against input length next to the E2.3 model: the
  bottleneck curve flattens. Plot attention heatmaps: for reversal you should see an anti-diagonal,
  for dates the model should look at the year when writing the year. Then swap in dot-product and
  scaled dot-product scoring (`qᵀk / √d`) and compare; this is exactly the formula F1.1 builds
  self-attention from. Exercise: remove the mask and find the bug it causes.

**Project milestone: P6 M0, corpus and baseline (1 h, optional head start).** P6 (the small GPT) is
due in course week 22. Pick its training corpus now: a single plain-text source of about 1–10 MB
(tiny Shakespeare, a public-domain book collection, or your own notes; check the licence). Split it
into train and validation by contiguous blocks, not random lines, so the same passage cannot appear
in both (the B1.4 leakage check, applied to text). Train and record two baselines on validation loss
(nats per character): the character bigram from E2.1 and your E2.2 LSTM. CP4 requires P6 to beat the
bigram baseline, so write the number down.

**Prerequisites:** E1.1 (vocabularies, padding, masks), E1.3 (embedding layers); A2.1 (shape rules),
A4.3 (chain rule, for BPTT), C1.3 (your autograd engine, the model for BPTT checks), C1.4 (stable
softmax and cross-entropy), C2.1–C2.2 (`nn.Module`, the training loop), C3.1 (Adam), C3.3
(initialization and exploding / vanishing gradients), C3.4 (LayerNorm, mentioned as the transformer's
choice), C4.3 (reading loss curves and gradient norms). Within the week: E2.2 needs E2.1; E2.3 needs
E2.2; E2.4 needs E2.3.

**Resources** (optional)

- Christopher Olah, "Understanding LSTM Networks" (blog post, 2015):
  https://colah.github.io/posts/2015-08-Understanding-LSTMs/ (VERIFY link)
- PyTorch tutorial, "NLP From Scratch: Translation with a Sequence to Sequence Network and
  Attention": https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html (VERIFY link)
- Bahdanau, Cho, Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate"
  (2014): https://arxiv.org/abs/1409.0473 (VERIFY link)

**GPU:** helpful, never required. Every lab has a CPU default that finishes in under 5 minutes.
RNNs are sequential, so a GPU helps less than it did for CNNs at small sizes; it pays off for the
`GPU_SCALE` runs of E2.2 (full IMDB) and E2.3 (real translation), where PyTorch's fused cuDNN LSTM is
much faster than the CPU. Your from-scratch cells are slow on any device; use them for checking, and
the `nn.LSTM` / `nn.GRU` modules for anything bigger.

**Used later in:** gradient clipping in every transformer training run in F1.4 and P6 (E2.1); the
character language model, bigram baseline and sampling with temperature, which P6 scales up (E2.1);
the encoder–decoder pattern behind T5 and the "encoder versus decoder" comparison in F2.1 (E2.3);
teacher forcing, which is exactly how GPT is trained with a causal mask (E2.3 → F1.1); beam search
and greedy decoding in F2.4 (E2.3); scaled dot-product attention and masking in F1.1 (E2.4); the
attention maps you will inspect again in transformers and in G2 (E2.4); the "why not RNNs" argument
in system-design interviews (L3).

---

## Running Part E on a free GPU

Free-tier GPUs, quotas and session limits change often: VERIFY (as of 2026-10).

- **Colab:** open the lab, then Runtime → Change runtime type → a GPU (usually a T4). Check with
  `torch.cuda.is_available()`. Upload the lab file or clone your repo, set the `GPU_SCALE` values,
  and save results to Google Drive, because the VM is wiped when the session ends.
- **Kaggle:** create a notebook, then Settings → Accelerator → GPU, and turn Internet on to download
  `text8`, IMDB and the GloVe vectors (Kaggle asks for phone verification first). Save outputs to
  `/kaggle/working`. Kaggle's GPU quota is weekly; Part E needs very little of it, so this is a good
  week to save quota for P6.
- **Both:** every lab already picks `device = "cuda" if torch.cuda.is_available() else "cpu"`
  (C2.3), so the same file runs in both places. Use `nn.LSTM` / `nn.GRU` (not your from-scratch
  cells) for the `GPU_SCALE` runs, so cuDNN does the work. Bring back only results (metrics, plots,
  attention maps), never datasets, pretrained vectors or checkpoints.

---

## You're ready for Part F when…

Part E has no checkpoint of its own; the attention items below are tested at CP4 (end of Part F).
Answer each item without notes. The answers belong in your quiz reviews, not here.

1. You can tokenize a messy sentence two ways, build a vocabulary with `<unk>` and `<pad>`, and turn
   three sentences of different lengths into a padded tensor and mask.
2. You can compute TF-IDF for a tiny corpus by hand, and you know your IMDB TF-IDF baseline score.
3. You can explain why "not good" breaks a unigram model and what bigrams or a sequence model change.
4. You can write the skip-gram negative-sampling loss for one example, say what the two embedding
   matrices are, and explain why cosine similarity (not Euclidean distance) is the usual comparison.
5. You can name two ways analogy results mislead, and one reason geometric debiasing does not remove bias.
6. You can unroll an RNN for three steps with stated shapes and explain, in terms of repeated matrix
   products, why its gradients vanish or explode, and what clipping fixes and does not fix.
7. You can draw an LSTM cell, say what the forget, input and output gates do, and why the cell state helps.
8. You can explain teacher forcing, exposure bias, and why a single context vector limits seq2seq.
9. You can write additive and scaled dot-product attention from scratch with correct shapes and a
   padding mask, and read an attention heatmap.
10. You can say in two sentences what a transformer keeps from E2.4 and what it removes, and why that
    makes it train in parallel.

If two or more items fail, revisit the topic ids your quiz reviews flagged before you start F1.
Item 9 is the one that matters most: F1.1 starts from it on day one.
