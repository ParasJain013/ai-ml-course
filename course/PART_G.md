# Part G. Generative models & multimodal — detailed syllabus

Course weeks 30–31. Weekly detail for Part G of [`COURSE_OUTLINE.md`](COURSE_OUTLINE.md), which stays
the source of truth: the topic ids, Core/Depth labels, project P10 and the DDPM paper below are taken
from it unchanged. Written 2026-10-16.

**Goal:** understand how models generate images and connect vision with language.

Part G comes after Parts I and J in the course order, so you arrive with a trained GPT (P6), a working
RAG system and agent (P7, P8) and a LoRA fine-tune (P9). That changes how this part is taught. G1 is
the generative-model family tree, built from scratch on small images so every piece runs on the
laptop: autoencoders, then VAEs (which turn the autoencoder into a probabilistic model you can sample
from), then GANs, then diffusion, the approach behind today's image generators. G2 connects images to
the transformers you already know: ViT turns an image into tokens, CLIP puts images and text in one
embedding space, a vision-language model feeds those image tokens into an LLM, and latent diffusion
puts a VAE, a diffusion model and a text encoder together into a text-to-image system. By the end you
can read the architecture diagram of a modern image generator or multimodal LLM and name what every
box does and which week you built it in.

## At a glance

| Week | Course wk | Theme | Topics | Hours | Project | GPU |
|---|---|---|---|---|---|---|
| G1 | 30 | Generative models | G1.1–G1.4 | 13 | P10 M1 | helpful (G1.4) |
| G2 | 31 | Multimodal | G2.1–G2.4 | 12.5 | **P10 due** | helpful |

About 25.5 hours in total. Each week's hours include about 1 hour for the weekly quiz and cheat sheet.
Topic hours cover reading the lesson and doing its lab and exercises.

**How the hours are counted.** *Reading* is working through the lessons and the paper, including the
worked examples by hand. *Coding* is the labs and the code exercises. *Project* is P10.

**Teaching order.** Both weeks are taught in id order. Each topic in G1 builds on the one before it
(the VAE is an autoencoder with a probabilistic bottleneck; diffusion reuses the denoising idea from
G1.1), and G2.4 needs the VAE from G1.2, the sampler from G1.4 and the text encoder from G2.2.

**If a week runs long,** drop the Depth topics first: G1.3 (GANs), G2.3 (vision-language models) and
G2.4 (latent diffusion). If you skip G2.4, still read its `[RAI]` section on deepfakes and
watermarking (about 20 minutes), because it is the only place the course teaches that safety theme.
If you are 3 weeks behind, the outline's catch-up plan cuts Part G to one week: keep G1.1, G1.2, G1.4
and G2.2, and P10 becomes the VAE only. Part G is not on the fast path to interview-readiness.

**Libraries this part needs** (Step 2 pins the versions): `numpy`, `torch` (CPU build on the laptop),
`torchvision` (MNIST, Fashion-MNIST and CIFAR-10 loaders, image transforms), `matplotlib`, `scipy`
(`scipy.linalg.sqrtm` for the Fréchet score in P10), `pillow`, `transformers` (pretrained ViT, CLIP and
a small vision-language model), and `diffusers` (only for the optional Stable Diffusion run in G2.4,
on a free GPU). Optionally `open_clip_torch` for more CLIP checkpoints. `transformers` and `diffusers`
APIs change often, the vision-language classes most of all: VERIFY (as of 2026-10).

**Models** (downloaded into the Hugging Face cache, never committed): a pretrained ViT such as
`google/vit-base-patch16-224` (about 86M parameters) for the G2.1 linear probe; CLIP ViT-B/32, for
example `openai/clip-vit-base-patch32` (about 150M parameters), for G2.2 and G2.4; a small open
vision-language model in the 250M–2B range (for example the SmolVLM or Qwen2-VL families) for G2.3 on
CPU; and, on a free GPU only, a small Stable Diffusion checkpoint for G2.4. The vision-language model
you call through an API in G2.3 is whichever model your Part I API key gives you that accepts images.
Model names, sizes, licences and which ones accept images change quickly: VERIFY (as of 2026-10).

**Datasets** (all download into `data/`, never committed): MNIST and Fashion-MNIST (28×28 grayscale,
about 30 MB each) for every from-scratch generative lab and the P10 default; CIFAR-10 (32×32 colour,
about 170 MB) for the `GPU_SCALE` runs and CLIP zero-shot; synthetic 2-D point clouds (two moons, a
ring of 8 Gaussians) generated in code for the toy GAN and toy diffusion experiments; and 20–30 images
you choose yourself (receipts, charts, photos you took) for the G2.3 vision-language eval. VERIFY each
source, its licence and its contents when the lab is generated.

**Hardware:** every lab runs on the laptop CPU at default settings: 28×28 images, models of 0.1–2M
parameters you train yourself for a few epochs on a subset, toy 2-D data where a full run needs more
compute than 5 minutes allows, and pretrained models of at most about 150M parameters used only for
inference and linear probes (the CPU vision-language model is the one exception, run on a handful of
images). CPU samples from the image DDPM will be recognizably digit- or garment-shaped but rough; that
is expected, and the `GPU_SCALE` run is where they get clean. How to use a free GPU is under "Running
Part G on a free GPU" at the end.

---

## G1 · Generative models (course week 30)

**Learning objectives**

- I can build a linear autoencoder in NumPy, train it with gradient descent, and show that its
  decoder spans the same subspace as the top-k principal components from PCA (B5.4), checked by the
  angles between the two subspaces.
- I can build a convolutional autoencoder and a denoising autoencoder in PyTorch, state the shape at
  every layer from `(B, 1, 28, 28)` to the bottleneck and back, and use reconstruction error as an
  anomaly score (B5.6).
- I can write the VAE objective, the evidence lower bound
  `ELBO = E_q(z|x)[log p(x|z)] − KL(q(z|x) ‖ p(z))`, say what each term rewards, and derive the
  closed-form KL between a diagonal Gaussian and `N(0, I)`, then check it against a Monte Carlo
  estimate (A5.4, A6.4).
- I can explain why you cannot backpropagate through sampling `z ~ N(μ, σ²)` directly, implement the
  reparameterization `z = μ + σ ⊙ ε` with `ε ~ N(0, I)`, and show that its gradient estimate has far
  lower variance than the score-function estimate (the one REINFORCE uses in H2.1).
- I can train a VAE with a 2-D latent, plot the latent space coloured by class, decode a grid of
  latent points, and explain what β in a β-VAE trades off and what posterior collapse looks like.
- I can write the GAN minimax objective, explain why the generator is trained with the
  non-saturating loss in practice, show mode collapse on a mixture of 8 Gaussians, and name two fixes.
- I can write the DDPM forward process in closed form, `x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε`, compute
  `ᾱ_t` by hand for the first 3 steps of a linear schedule, and explain why the model is trained to
  predict the noise `ε` with a plain mean-squared error.
- I can implement DDPM training and the ancestral sampler from scratch, first on 2-D points and then
  on 28×28 images with a small U-Net conditioned on the timestep, and explain why sampling needs `T`
  network calls while a VAE or GAN needs one.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| G1.1 | Autoencoders: bottlenecks, denoising, learned representations | Core | 2 |
| G1.2 | Variational autoencoders: the ELBO and the reparameterization trick | Core | 3 |
| G1.3 | GANs: the generator/discriminator game, mode collapse, DCGAN | Depth | 2 |
| G1.4 | Diffusion models: forward noising, the denoising objective, a tiny DDPM sampler from scratch (+ *Denoising Diffusion Probabilistic Models*) | Core | 3.5 |

**Hours:** reading 4.5 · coding 6 · project 1.5 · quiz and cheat sheet 1 · **total 13**

**Labs**

- **G1.1:** a linear autoencoder in NumPy (encoder `W_e` of shape `(d, k)`, decoder `W_d` of shape
  `(k, d)`, no activations, squared-error loss) trained by gradient descent on centred Fashion-MNIST
  vectors (`d = 784`, `k = 16`). Compare it with PCA from your B5.4 code: the reconstruction errors
  match, and the principal angles between the decoder's column space and the top-16 components are
  near zero, but the individual decoder vectors are not the principal components (the autoencoder
  finds the subspace, not the ordered axes). Then a convolutional autoencoder in PyTorch: strided
  convolutions down to a `(B, 16)` bottleneck and transposed convolutions back up (D2.4), with every
  shape asserted. Add Gaussian noise to the inputs only and train a denoising autoencoder; show its
  outputs on noisy test images. Finish with two uses of the learned code: k-nearest-neighbour
  classification on the 16-dimensional codes against raw pixels (B2.3), and reconstruction error as
  an anomaly score on MNIST digits fed to the Fashion-MNIST model (B5.6). Bridge: the bottleneck is
  like a lossy compression codec you learn from data; the analogy breaks because nothing forces the
  code space to be smooth, which is why you cannot sample new images from it, and why G1.2 exists.
- **G1.2:** first in NumPy, the closed-form KL for a diagonal Gaussian,
  `KL = ½ Σ (μ² + σ² − log σ² − 1)`, checked against a Monte Carlo estimate with 100,000 samples
  (A6.4, A5.4). Then the gradient-variance experiment on a one-dimensional toy: estimate
  `∂/∂μ E[f(z)]` with the reparameterization trick and with the score-function estimator, and plot
  the variance of each over 1,000 repeats. Then a VAE in PyTorch on MNIST: an MLP or small CNN encoder
  outputting `μ` and `log σ²` (shape `(B, k)` each), the reparameterized sample, a decoder, and a loss
  of binary cross-entropy summed over pixels (Bernoulli likelihood, A5.5) plus the KL term. With
  `k = 2`, plot the latent means of the test set coloured by digit and decode a 15×15 grid of latent
  points. With `k = 16`, generate samples from `N(0, I)`, interpolate between two test images in latent
  space, and sweep β in `{0.1, 1, 4}`, plotting reconstruction loss against KL. Show posterior collapse
  by giving the decoder an easy shortcut and watching the KL fall to zero. Connect the ELBO to the EM
  view from B5.3 if you studied it.
- **G1.3:** a GAN on 2-D toy data first: a small MLP generator mapping `N(0, I)` noise to points and a
  discriminator classifying real versus fake, trained in alternating steps. Plot the generated points
  every few hundred steps on a ring of 8 Gaussians; show mode collapse (the generator covering 1–3 of
  the 8 modes), then reduce it with a lower learning rate for the generator and minibatch features or
  a larger batch. Compare the original minimax generator loss `log(1 − D(G(z)))` with the
  non-saturating `−log D(G(z))` by plotting their gradients against `D(G(z))`. Then a small DCGAN on
  MNIST (transposed-convolution generator, strided-convolution discriminator, BatchNorm, LeakyReLU,
  Adam with `β₁ = 0.5`) for a few epochs on a subset, saving a sample grid each epoch. Note honestly
  that GANs are now rarely the first choice for image generation, but the adversarial idea survives
  (in discriminator losses for VAE decoders, as in G2.4's latent-diffusion autoencoder).
- **G1.4:** DDPM from scratch, in two stages. Stage one, on 2-D two-moons data: a linear β schedule
  (`β₁ = 1e-4` to `β_T = 0.02`), precomputed `α_t`, `ᾱ_t`, the closed-form forward noising, an MLP
  that takes `(x_t, t)` and predicts `ε`, the loss `‖ε − ε_θ(x_t, t)‖²` on a random `t` per example,
  and the ancestral sampler
  `x_{t−1} = (1/√α_t) · (x_t − (β_t / √(1 − ᾱ_t)) · ε_θ(x_t, t)) + σ_t · z`. Plot the forward process
  turning the moons into a Gaussian blob, and the reverse process turning a blob back into moons; this
  trains in under a minute and shows the method works before images are involved. Stage two, on 28×28
  Fashion-MNIST or MNIST: a small U-Net (your D2.4 U-Net with sinusoidal timestep embeddings from F1.2
  added to each block), trained for a few epochs on a subset; sample 16 images and plot the
  denoising trajectory at 10 timesteps. Compare `T = 1000` with `T = 100` and with fewer sampling steps
  by striding (a first look at why DDIM-style samplers exist). CPU default: about 1M parameters, a few
  minutes of training, rough but recognizable samples. `GPU_SCALE`: CIFAR-10 at 32×32, a U-Net of
  about 10–35M parameters, tens of thousands of steps with fp16 autocast and a gradient scaler on a T4
  (C4.2), plus an exponential moving average of the weights for sampling.

**Paper:** *Denoising Diffusion Probabilistic Models* (Ho, Jain, Abbeel, 2020), with its reading guide
in `papers/`. Read sections 1–3 and 4.1–4.2 (the forward and reverse processes, the simplified
noise-prediction loss and the sampling algorithm), and look at Algorithms 1 and 2, which are the two
functions you write in the G1.4 lab. Skim the variational-bound derivation in the appendix with your
G1.2 ELBO next to it: a DDPM is trained on an ELBO too. About 1 hour, counted in G1.4.

**Project milestone: P10 M1, evaluation harness and VAE (1.5 h).** Pick the P10 dataset
(Fashion-MNIST by default; CIFAR-10 if you will do the `GPU_SCALE` runs) and fix its train/test split.
Train a small CNN classifier on it (your D1.2 code) and write the "simple quality metric" P10 needs: a
Fréchet distance between Gaussians fitted to the classifier's penultimate-layer features of real and
generated images, `d² = ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2(Σ_r Σ_g)^½)`, plus the classifier's prediction
entropy on generated samples. Validate the metric before trusting it: real-versus-real test halves
should score near zero, and real-versus-noise and real-versus-blurred should score clearly worse, in
that order. Then train your G1.2 VAE on the dataset and record its samples and scores. Note in the
write-up that this is not FID: published FID uses Inception-v3 features on large sample sets, so your
numbers are only comparable with each other.

**Prerequisites:** B5.4 (PCA via SVD, for the linear autoencoder), A3.2 (SVD), B5.6 (anomaly scores),
B2.3 (k-NN), A5.2 (Gaussian and multivariate Gaussian distributions), A5.4 (Monte Carlo estimation),
A5.5 (likelihoods behind squared error and binary cross-entropy), A6.4 (KL divergence, the core of
G1.2), B5.3 (EM and the ELBO view; Depth, helpful but not required), A4.3 (chain rule, for why
sampling blocks gradients), C1.4 (stable cross-entropy), C2.1–C2.3 (PyTorch, training loop, device
handling), C3.1 (Adam), C3.2 (LR schedules), C3.4 (BatchNorm, for DCGAN), C4.1–C4.2 (GPU and mixed
precision, for `GPU_SCALE`), C4.3 (debugging training runs, GANs need it most), D1.1–D1.2
(convolutions, output-shape formula), D2.4 (transposed convolutions and the U-Net), F1.2 (sinusoidal
encodings, reused as timestep embeddings). Within the week: G1.2 needs G1.1; G1.4 uses the denoising
idea from G1.1 and the ELBO from G1.2.

**Resources** (optional)

- Kingma and Welling, "Auto-Encoding Variational Bayes" (2013): https://arxiv.org/abs/1312.6114
  (VERIFY link)
- Ho, Jain and Abbeel, "Denoising Diffusion Probabilistic Models" (2020):
  https://arxiv.org/abs/2006.11239 (VERIFY link)
- Lilian Weng, "What are Diffusion Models?" (blog post with the derivations in one place):
  https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ (VERIFY link)

**GPU:** helpful for G1.4 only, never required. G1.1–G1.3 train in seconds to a few minutes on the CPU
at 28×28. The image DDPM is the one model in this week where compute visibly limits quality: the CPU
default gives rough samples, and the `GPU_SCALE` run on a free T4 gives clean Fashion-MNIST samples in
roughly 20–40 minutes and usable CIFAR-10 samples in a few hours, which suits Kaggle's longer sessions
(VERIFY on your session). Save checkpoints often, because a sampler run needs only the weights.

**Used later in:** the VAE as the image compressor in latent diffusion (G2.4) (G1.2); the DDPM
sampler and noise-prediction loss in G2.4 (G1.4); the reparameterization versus score-function
gradient contrast in REINFORCE in H2.1 (G1.2); P10; generative-model questions in the L3 mock
interviews.

---

## G2 · Multimodal (course week 31)

**Learning objectives**

- I can turn an image of shape `(B, C, H, W)` into patch tokens of shape `(B, N, P²·C)` with
  `N = (H/P)·(W/P)`, compute `N` for a 224×224 image with 16×16 patches (196), and show that a linear
  patch embedding equals a convolution with kernel size and stride `P`.
- I can build a small Vision Transformer from my F1.2 block (patch embedding, a class token, learned
  position embeddings, encoder blocks without a causal mask), train it on small images, and explain
  why it loses to a CNN of similar size on small data but wins with large-scale pretraining
  (inductive bias versus data).
- I can write CLIP's symmetric contrastive loss: a `(B, B)` similarity matrix of normalized image and
  text embeddings divided by a temperature `τ`, with cross-entropy over rows and over columns where
  the diagonal is the correct pair, and compute it by hand for a batch of 3.
- I can do zero-shot classification with a pretrained CLIP by embedding one text prompt per class,
  measure how prompt wording changes accuracy, and compare zero-shot with a linear probe on the same
  features.
- I can explain how a vision-language model feeds an image into an LLM (vision encoder, projector,
  visual tokens placed among the text tokens), count how many tokens an image costs, and say what is
  trained in each stage.
- I can call a vision-language model through an API and a small open one on CPU, extract structured
  data from images with schema validation (I1.3), and measure accuracy and object hallucination on a
  small golden set (I2.4).
- I can explain latent diffusion as three trained parts (a VAE that compresses images, a denoiser that
  works in the latent space, a text encoder that conditions it through cross-attention), and implement
  classifier-free guidance, `ε̂ = ε_θ(x_t, ∅) + w · (ε_θ(x_t, c) − ε_θ(x_t, ∅))`, on my own model.
- I can explain `[RAI]` how image generators enable deepfakes, what provenance metadata and invisible
  watermarks can and cannot prove, and show a simple watermark failing under common edits.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| G2.1 | Vision Transformers: images as patch tokens | Core | 2.5 |
| G2.2 | CLIP: contrastive image–text training and zero-shot classification | Core | 2.5 |
| G2.3 | Vision-language models: how image features enter an LLM; using one through an API | Depth | 1.5 |
| G2.4 | Latent diffusion, text conditioning and classifier-free guidance; `[RAI]` deepfakes and watermarking | Depth | 2 |

**Hours:** reading 3.5 · coding 5 · project 3 · quiz and cheat sheet 1 · **total 12.5**

**Labs**

- **G2.1:** patchify from scratch with `reshape` and `permute`, asserting shapes, and an "unpatchify"
  that round-trips exactly; then show that a `Linear(P²·C, d)` on patches and a
  `Conv2d(C, d, kernel_size=P, stride=P)` give the same output after copying weights across. Build a
  tiny ViT for 28×28 Fashion-MNIST (patch size 4, so 49 tokens plus a class token, `d = 64`, 4 heads,
  4 encoder blocks reusing your F1.2 code with the causal mask removed) and train it for a few epochs;
  train your D1.2 CNN with a similar parameter count for the same time and compare accuracy and
  learning curves. Plot the cosine similarity between the learned position embeddings as a 7×7 grid
  per patch: neighbouring patches end up similar although nothing told the model about 2-D layout.
  Then load a pretrained ViT through `transformers`, extract class-token features for a few thousand
  CIFAR-10 images on the CPU, and fit a logistic-regression linear probe (B2.1); compare with your
  D2.1 ResNet features. `GPU_SCALE`: the tiny ViT on CIFAR-10 with stronger augmentation (D2.2) and
  more epochs, or fine-tuning the pretrained ViT end to end.
- **G2.2:** CLIP's loss from scratch in NumPy: L2-normalize image and text embeddings, form the
  `(B, B)` logit matrix `I · Tᵀ / τ`, and take the mean of the row-wise and column-wise cross-entropy
  with the diagonal as targets; check it against a hand computation for a batch of 3 and against a
  PyTorch version. Show what temperature does to the softmax over a row. Then train a tiny CLIP on the
  CPU: a small CNN image encoder on MNIST paired with short captions generated from templates ("a
  handwritten seven", "the digit 7", "a photo of the number seven"), a small text encoder (your F1
  transformer blocks over your F1.3 tokenizer, or mean-pooled word embeddings), both projected to a
  shared 64-dimensional space. Classify the test set zero-shot from prompts it never saw exactly. Then
  load pretrained CLIP ViT-B/32 on the CPU: zero-shot accuracy on 1,000 CIFAR-10 test images with the
  bare class name, with "a photo of a {class}." and with an ensemble of several prompts; a linear probe
  on the same image features; and text-to-image retrieval measured with recall@k (I2.4). Bridge: this
  is the embedding search you built on pgvector in I2.2, except the query and the documents come from
  different modalities and meet in one vector space; it breaks because CLIP embeddings are trained for
  short captions, so long or compositional queries ("a red cube on a blue sphere") retrieve poorly.
- **G2.3:** first the plumbing, on shapes alone: take CLIP patch features `(B, 49, 768)` from ViT-B/32,
  pass them through a two-layer MLP projector into an LLM's embedding width (your P6 GPT's `d`, or a
  small open model's), and splice them into the token sequence between text embeddings; count the
  resulting context length and the cost per image. Read how one open model handles high-resolution
  images (tiling, or pooling patch tokens) and how many tokens that adds (VERIFY in its model card).
  Then run a small open vision-language model on the CPU through `transformers` on 5–10 of your
  images: captioning and a few questions. Then the same task through your Part I API: send 20–30
  images (receipts or charts work well), extract fields into a Pydantic schema with validation and
  retry (I1.3), score field accuracy against a hand-labelled golden set, and count hallucinated
  objects or numbers. Record tokens, cost and latency per image (I4.3). Keep personal data out of the
  images you send (B1.2).
- **G2.4:** classifier-free guidance on your own models, on the CPU. Make the G1.4 MNIST DDPM
  class-conditional by adding a learned class embedding to the timestep embedding, and replace the
  label with a "null" class 10% of the time during training. Sample with guidance weights `w` in
  `{0, 1, 3, 7}` and plot sample grids next to your P10 classifier's accuracy and prediction entropy
  on them: higher `w` gives cleaner, more on-class but less varied samples. Then a tiny latent
  diffusion: encode MNIST with your G1.2 VAE (`k = 16`) and train a small MLP denoiser on the latents
  instead of the pixels; decode its samples and compare training time and sample quality with the
  pixel-space model. Read a real latent-diffusion pipeline's components (VAE, U-Net or transformer
  denoiser, CLIP-style text encoder, scheduler) and map each to the week you built it. `[RAI]`: add an
  invisible watermark to a generated image (a simple frequency-domain or least-significant-bit
  scheme), detect it, then show detection failing after JPEG compression, resizing and cropping;
  write one paragraph on what provenance metadata (C2PA content credentials) and model-level
  watermarks such as SynthID claim to offer and their limits (VERIFY (as of 2026-10)), and on consent
  and non-consensual imagery. Optional `GPU_SCALE`: a small Stable Diffusion checkpoint through
  `diffusers` in fp16 on a T4, sweeping guidance scale and step count and swapping the scheduler
  (VERIFY the checkpoint's licence and the current pipeline API).

**Project milestone: P10 due, generative model (3 h).** On the dataset and split from M1: train the
G1.4 DDPM (CPU default or `GPU_SCALE`) and compare it with your M1 VAE on sample grids, your Fréchet
score and prediction entropy (both validated in M1), number of parameters, training time and sampling
time per image. Include one ablation (number of sampling steps for the DDPM, or β for the VAE) and say
honestly which model is better on which measure and why. Bonus: CLIP zero-shot classification on the
dataset's test set with at least two prompt templates, against your M1 classifier. Ship a model card
as every project does from P2 on, with a misuse section on generated images (G2.4). Reference notes
go in `answers/projects/`.

**Prerequisites:** F1.1–F1.2 (multi-head attention and the transformer block, reused for ViT and the
CLIP text encoder), F1.3 (tokenization), F2.1 (encoder versus decoder models, cross-attention in T5),
E2.4 (attention from a decoder to an encoder's states, the idea behind text conditioning), D1.1–D1.3
(convolutions and the CNN baselines), D2.1 (transfer learning and linear probes), D2.2 (augmentation),
E1.3 (negative sampling, the first contrastive objective you met), A1.5 (cosine similarity), C1.4
(cross-entropy), B2.1 (logistic regression for the linear probe), I1.1 (LLM APIs), I1.3 (structured
output), I2.1–I2.2 (embeddings and vector search), I2.4 (golden sets, recall@k, judging outputs),
I4.3 (cost and latency), F3.3 (hallucination), F3.4 (running open models on CPU), B1.2 (PII);
G1.2 (the VAE) and G1.4 (the DDPM and sampler) from last week. Within the week: G2.2 uses G2.1's
image encoder idea; G2.3 needs G2.1–G2.2; G2.4 needs G1.2, G1.4 and G2.2.

**Resources** (optional)

- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
  (2020): https://arxiv.org/abs/2010.11929 (VERIFY link)
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP,
  2021): https://arxiv.org/abs/2103.00020 (VERIFY link)
- Hugging Face Diffusion Models course (free, covers `diffusers`, guidance and latent diffusion):
  https://github.com/huggingface/diffusion-models-class (VERIFY link)

**GPU:** helpful, never required. G2.1's tiny ViT, the tiny CLIP and the classifier-free-guidance and
latent-diffusion labs all train on the CPU at 28×28. Pretrained ViT and CLIP inference on a few
thousand images takes a few minutes on the CPU; the open vision-language model is slow on the CPU, so
keep it to 5–10 images there and run the full 20–30 image eval through the API or on a free GPU. Stable
Diffusion is the only thing in this part that is impractical without a GPU, which is why it is the
optional `GPU_SCALE` step; on a T4 load it in fp16.

**Used later in:** multimodal options for the capstone (L1.1), such as adding image input to an LLM
product; CLIP-style embeddings for image search in a RAG system (I2); vision-language and
text-to-image questions in ML system-design interviews (L2.2); the deepfake and watermarking material
in every model card for a generative system.

---

## Running Part G on a free GPU

Free-tier GPUs, quotas and session limits change often: VERIFY (as of 2026-10).

- **Colab:** open the lab, then Runtime → Change runtime type → a GPU (usually a T4). Check with
  `torch.cuda.is_available()` and `torch.cuda.get_device_name()`. Upload the lab file or clone your
  repo, set the `GPU_SCALE` values, and save checkpoints and sample grids to Google Drive every few
  thousand steps, because the VM is wiped when the session ends or disconnects. Colab suits the
  shorter runs: the Fashion-MNIST DDPM, the ViT fine-tune and the Stable Diffusion sweep.
- **Kaggle:** create a notebook, then Settings → Accelerator → GPU, and turn Internet on to download
  datasets and Hugging Face models (Kaggle asks for phone verification first). Save outputs to
  `/kaggle/working`. Kaggle's longer sessions make it the better place for the CIFAR-10 DDPM, which
  can need a few hours; run it as a background "Save & Run All" job so it survives closing the tab.
- **Both:** every lab already picks `device = "cuda" if torch.cuda.is_available() else "cpu"`
  (C2.3), so the same file runs in both places. On a T4 use fp16 autocast with a gradient scaler;
  bf16 needs a newer GPU. Keep an exponential moving average of the DDPM weights and sample from it.
  Bring back only results (sample grids, loss curves, score tables), never datasets, model weights
  or checkpoints: they are not committed.

---

## You're ready for Part H when…

Part H (Reinforcement learning, course weeks 32–33) comes next. It does not build on Part G's models,
so this self-check covers Part G itself and P10. Answer each item without notes. The answers belong in
your quiz reviews, not here.

1. You can explain why a plain autoencoder cannot generate new images and what the VAE adds to make
   sampling possible.
2. You can write the ELBO, say what each term does, write the closed-form Gaussian KL, and explain the
   reparameterization trick and why it gives lower-variance gradients than the score-function
   estimator (you will meet the latter again as REINFORCE in H2.1).
3. You can write the GAN objective, explain mode collapse, and say why the non-saturating generator
   loss is used.
4. You can write the DDPM forward process in closed form, the noise-prediction loss and one step of
   the sampler, and explain why diffusion sampling is slow and what fewer-step samplers trade away.
5. You can compute the number of patch tokens for an image and patch size, and explain when a ViT
   beats a CNN and when it does not.
6. You can write CLIP's contrastive loss for a batch, explain the temperature, and do zero-shot
   classification from text prompts.
7. You can draw how a vision-language model turns an image into tokens an LLM reads, and estimate what
   an image costs in context and money.
8. You can explain latent diffusion's three parts and what the classifier-free guidance weight trades
   off, and say what watermarks and provenance metadata can and cannot prove.
9. P10 is at 70% or more on its rubric, your quality metric was validated before use, and you can say
   honestly which of your two models is better on which measure.

If two or more items fail, revisit the topic ids your quiz reviews flagged before you start H1.
Item 2 matters most for Part H: the reparameterization-versus-score-function contrast is the cleanest
way into policy gradients.
