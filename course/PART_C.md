# Part C. Deep learning — detailed syllabus

Course weeks 13–16. Weekly detail for Part C of [`COURSE_OUTLINE.md`](COURSE_OUTLINE.md), which stays
the source of truth: the topic ids, Core/Depth labels, project P4 and checkpoint CP3 below are taken
from it unchanged. Written 2026-10-16.

**Goal:** understand neural networks from the gradient up, then train them reliably in PyTorch on
CPU and on a free GPU.

## At a glance

| Week | Course wk | Theme | Topics | Hours | Project | GPU |
|---|---|---|---|---|---|---|
| C1 | 13 | Neural networks from scratch | C1.1–C1.4 | 12 | P4 M1 | no |
| C2 | 14 | PyTorch | C2.1–C2.3 | 12 | **P4 due** | no |
| C3 | 15 | Making training work | C3.1–C3.5 | 11.5 | — | no (helpful for full-size runs) |
| C4 | 16 | GPU training & debugging | C4.1–C4.4 | 10.5 | CP3 | required for C4.1–C4.2 |

About 46 hours in total. Each week's hours include about 1 hour for the weekly quiz and cheat sheet
(1.5 hours in C4, which also holds the CP3 self-check). Topic hours cover reading the lesson and doing
its lab and exercises.

**How the hours are counted.** *Reading* is working through the lessons, including the worked
examples by hand. *Coding* is the labs and the code exercises. *Project* is P4.

**C4 is lighter on purpose.** It leaves about 1.5 hours of slack to go back over whatever the CP3
self-check or your P4 review flags, before Part D builds on all of it.

**If a week runs long,** drop its Depth topic first (C4.2, then C4.4) and come back to it later.
These are the only Depth topics in Part C, and both are on the outline's cut list. Skipping C4.2
means reading its lesson before J1.3 (QLoRA uses reduced precision); skipping C4.4 means reading it
before F2.2 (pretraining compute).

**Libraries this part needs** (Step 2 pins the versions): `torch` (CPU build), `torchvision` (only to
download Fashion-MNIST), `numpy`, `matplotlib`, `scikit-learn` (the bundled `load_digits` images and
`make_moons`), `ipykernel`.

**Datasets.** scikit-learn's digits (1,797 8×8 images) is bundled with the library, so P4 and most of
C1 need no download. Fashion-MNIST (60,000 28×28 training images, roughly 30 MB; VERIFY size)
downloads into `data/` through `torchvision` from C2.2 on. Lab defaults use subsets so every lab runs
on the laptop CPU in under 5 minutes; each lab's `GPU_SCALE` block gives the full-size settings.

**Hardware.** C1–C3 run on the laptop CPU. C4.1 and C4.2 are about GPUs, so their measurements need a
free Colab or Kaggle GPU; the labs still run on CPU (with the GPU-only cells skipped) so they stay
runnable locally.

**Running a lab on a free GPU** (used in C4, and optional in C3):

- **Colab:** *Runtime → Change runtime type → T4 GPU* (the usual free GPU; VERIFY (as of 2026-10)).
  Check it with `!nvidia-smi`. Clone the repo (or upload the lab file) and run it with
  `!python labs/C/<lab>.py`. Use Colab's preinstalled CUDA build of PyTorch: do **not** install the
  course's pinned CPU build there. Mount Google Drive and write checkpoints to it, because the
  session's disk is wiped when it disconnects.
- **Kaggle:** *Settings → Accelerator → GPU T4 ×2* or *P100*, and turn *Internet* on to clone the
  repo. GPU access needs a phone-verified account, and the weekly GPU quota is about 30 hours
  (VERIFY both, as of 2026-10). Outputs saved in `/kaggle/working` survive only if you save a
  version of the notebook.
- On either, set the lab's `GPU_SCALE` flag, and print `torch.__version__` and
  `torch.cuda.get_device_name()` at the top so your results say what they ran on.

---

## C1 · Neural networks from scratch (course week 13)

**Learning objectives**

- I can explain why a single perceptron cannot learn XOR and give hand-set weights for a two-layer network that does.
- I can explain why a stack of linear layers with no activation is still one linear layer, and what a ReLU network's output looks like (piecewise linear).
- I can run a forward and backward pass by hand on a 2-2-2 network with real numbers, and get the same gradients my code and a numerical gradient check give.
- I can write the backward pass of a dense layer in matrix form, `dW = Xᵀ δ`, and state every shape for a batch of `n` examples.
- I can build a scalar autograd engine that handles a value used twice, and a NumPy tensor engine whose backward pass handles broadcasting.
- I can compute softmax and cross-entropy for logits like `[1000, 1001, 1002]` without overflow, and derive that the gradient of cross-entropy with respect to the logits is `p − y`.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| C1.1 | Perceptron and multilayer perceptron: neurons, activations, why depth helps | Core | 2 |
| C1.2 | Backpropagation by hand on a tiny network with real numbers | Core | 2 |
| C1.3 | An autograd engine from scratch: scalar first, then tensors | Core | 3.5 |
| C1.4 | Softmax and cross-entropy, written to be numerically stable | Core | 1.5 |

**Hours:** reading 3.5 · coding 5.5 · project 2 · quiz and cheat sheet 1 · **total 12**

**Labs**

- **C1.1:** the perceptron learning rule from scratch on linearly separable 2-D points, with the
  decision boundary plotted as it learns. Run it on XOR and watch it never converge. Then a two-layer
  network with hand-set weights that solves XOR, and its boundary. Plot sigmoid, tanh and ReLU with
  their derivatives, and show numerically that `W₂(W₁x)` equals one matrix `(W₂W₁)x`. Finally, plot
  the output of small random ReLU networks along a line and count the linear pieces as you add
  width versus depth.
- **C1.2:** a 2-2-2 network with sigmoid units and fixed starting weights. Do the forward pass, loss
  and every backward step on paper first, then in code that prints each intermediate value so you
  can compare line by line. Check all gradients with your A4.2 gradient checker, take one gradient
  descent step and show the loss falls. Then redo it for a batch with matrices, deriving
  `dW = Xᵀ δ`, `db = Σ δ` and `dX = δ Wᵀ` with their shapes.
- **C1.3:** a scalar `Value` class (in the style of Karpathy's micrograd) supporting `+`, `*`,
  `**`, `exp`, `log`, `tanh` and `relu`, with `backward()` running over a topological sort. Write
  the bug first: assign gradients with `=` instead of `+=`, and show it gives a wrong answer for
  `y = x * x`. Train a tiny MLP on `make_moons` with it. Then a `Tensor` class on NumPy arrays with
  `add`, `mul`, `matmul`, `sum`, `mean`, `reshape`, `relu`, `exp` and `log`, including an
  "unbroadcast" step that sums gradients back to each input's shape. Gradient-check every op on
  random inputs, including broadcast cases like `(n, d) + (d,)`.
- **C1.4:** naive softmax on logits `[1000, 1001, 1002]` (it overflows), then the max-subtraction
  fix, log-sum-exp and log-softmax, checked against `scipy.special.logsumexp` and
  `scipy.special.log_softmax`. Cross-entropy computed directly from logits. Derive `∂L/∂z = p − y`
  by hand for 3 classes, check it numerically, and add it to your tensor engine as one fused op.
  Show that the 2-class case is exactly the log-loss of logistic regression from B2.1.

**Project milestone: P4 M1, the engine passes its checks (2 h).** Turn the C1.3 tensor engine and
the C1.4 loss into a small package under `my-work/` with a test file. Every op, and a 2-layer MLP
end to end, passes a numerical gradient check in float64 (max absolute difference ≤ 1e-6). Include
the broadcast cases and a node used twice in one graph.

**Prerequisites:** A2.1 (shapes), A4.2 (gradient checker), A4.3 (chain rule as a graph), A4.4
(gradient descent), A5.5 (log-loss as MLE), A6.4 (cross-entropy), B2.1 (logistic regression).
Within the week: C1.3 builds on C1.2, and C1.4's fused op plugs into C1.3's engine.

**Resources** (optional)

- Andrej Karpathy, "Neural Networks: Zero to Hero", lecture 1 (building micrograd): https://karpathy.ai/zero-to-hero.html (VERIFY link)
- 3Blue1Brown, "Neural networks" series, chapters 1–4: https://www.3blue1brown.com/topics/neural-networks (VERIFY link)
- Stanford CS231n course notes, "Backpropagation, Intuitions": https://cs231n.github.io/optimization-2/ (VERIFY link)

**GPU:** no. Everything is NumPy on tiny data and runs in seconds.

**Used later in:** every neural network in the course. Backprop through time in E2.1 (C1.2);
the PyTorch autograd comparison in C2.1 (C1.3); softmax in attention (F1.1) and temperature in
decoding (F2.4) (C1.4).

---

## C2 · PyTorch (course week 14)

**Teaching order this week:** C2.1 → C2.2 first, because P4's PyTorch comparison uses them. Then
C2.3, then finish P4.

**Learning objectives**

- I can create tensors with a chosen dtype and device, tell a view from a copy, and predict broadcasting results the same way as in NumPy.
- I can explain what `requires_grad`, `.backward()`, `.grad`, `zero_grad()`, `torch.no_grad()` and `.detach()` each do, and show that gradients accumulate if I forget to zero them.
- I can reproduce my C1.3 engine's gradients in PyTorch to within 1e-6, and write a custom `torch.autograd.Function` that passes `gradcheck`.
- I can write a `Dataset`, wrap it in a `DataLoader`, and write a training loop and an evaluation loop that I can reuse unchanged for any model in this course.
- I can explain what `model.train()` and `model.eval()` change, and why evaluation runs under `torch.no_grad()`.
- I can save a checkpoint with model and optimizer state, resume from it, and get the same loss curve as a run that was never interrupted.
- I can write device-agnostic code that runs unchanged on CPU and on a GPU.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| C2.1 | Tensors, autograd and `nn.Module`, checked against your C1.3 engine | Core | 2.5 |
| C2.2 | Dataset, DataLoader and the training loop; train versus eval mode | Core | 2.5 |
| C2.3 | Saving, loading, reproducibility and device handling | Core | 1.5 |

**Hours:** reading 2 · coding 4.5 · project 4.5 · quiz and cheat sheet 1 · **total 12**

**Labs**

- **C2.1:** NumPy-to-PyTorch translations of your Part A array code (shapes, broadcasting, views
  versus copies with `.view`, `.reshape` and `.clone`). Rebuild your C1.3 test expressions in
  PyTorch in float64 and assert the gradients equal your engine's. Show `.grad` accumulating across
  two `backward()` calls. Write your own `Linear` module with `nn.Parameter`, compare it with
  `nn.Linear`, and inspect `parameters()` and `state_dict()`. Write your ReLU as a custom
  `torch.autograd.Function` and pass `torch.autograd.gradcheck`.
- **C2.2:** a minimal data loader from scratch (shuffle with a seeded generator, batch, collate,
  drop or keep the last partial batch), compared with `torch.utils.data.DataLoader` on the same
  seed. A `Dataset` for a CSV file and one for Fashion-MNIST via `torchvision`. Write
  `train_one_epoch()` and `evaluate()` functions (the ones every later lab reuses) and train an MLP
  on a Fashion-MNIST subset. Show that a model containing `nn.Dropout` gives different outputs in
  train and eval mode (dropout itself is taught in C3.5).
- **C2.3:** save and load a checkpoint (model `state_dict`, optimizer state, epoch, RNG states).
  Train 4 epochs straight and 2 + resume + 2, and plot both loss curves on top of each other: on CPU
  with the seeds set they should be identical. Seed Python, NumPy and PyTorch, give the `DataLoader`
  a `generator`, and turn on `torch.use_deterministic_algorithms(True)`. A `get_device()` helper
  that picks `cuda`, then `mps`, then `cpu`. Load checkpoints with `torch.load(..., weights_only=True)`
  and explain why loading an untrusted pickle is a security risk (the default changed in recent
  releases; VERIFY (as of 2026-10)). Checkpoints go to a gitignored folder and are never committed.

**Project milestone: P4 due, an MLP from your own engine (4.5 h).** Using your M1 engine, train an
MLP (for example 64 → 128 → 10) with mini-batch SGD that you wrote yourself on scikit-learn's digits
images, with a fixed train/validation/test split. Build the identical model in PyTorch, copy your
initial weights into it, and feed both the same batches in the same order. Required:

- On at least 5 batches, every parameter's gradient matches PyTorch's with a max absolute
  difference ≤ 1e-6. Compare in float64: in float32, rounding alone can exceed 1e-6.
- The two loss curves overlap, and test accuracy matches PyTorch's to within 1 percentage point.
- A README that explains the engine's design, how the gradient checks work, and one bug you hit and
  how you found it.

Stretch: the same comparison on a Fashion-MNIST subset. Aim for **≥ 70%** on its rubric (the
course's usual project bar); the 1e-6 gradient match is a CP3 criterion.

**Prerequisites:** C1.1–C1.4 (P4 M1 in particular), A1.2 (NumPy shapes), B1.1 (train/validation/test
splits), B4.1 (accuracy and the confusion matrix).

**Resources** (optional)

- PyTorch tutorials, "Learn the Basics": https://pytorch.org/tutorials/beginner/basics/intro.html (VERIFY link)
- PyTorch docs, "Reproducibility": https://pytorch.org/docs/stable/notes/randomness.html (VERIFY link)
- PyTorch tutorials, "Saving and Loading Models": https://pytorch.org/tutorials/beginner/saving_loading_models.html (VERIFY link)

**GPU:** no. The digits MLP trains in seconds on CPU, and a Fashion-MNIST MLP takes well under a minute
an epoch.

**Used later in:** every PyTorch lab from C3 to H (C2.1, C2.2); early stopping in C3.5 and the
model registry in J1.2 (C2.3); the GPU code in C4.1 (C2.3's device handling).

---

## C3 · Making training work (course week 15)

**Learning objectives**

- I can write SGD, momentum, Adam and AdamW as small classes with a `step()` method that match `torch.optim` to within floating-point error over 100 steps.
- I can explain what Adam's bias correction fixes, and the difference between L2 regularization and AdamW's decoupled weight decay.
- I can implement linear warmup with cosine decay and one-cycle schedules, and use an LR range test to choose a learning rate.
- I can explain, with the variance argument, why Xavier initialization suits tanh and He suits ReLU, and diagnose vanishing or exploding gradients from per-layer statistics.
- I can implement BatchNorm and LayerNorm forward and backward, say what each normalizes over, and explain why BatchNorm behaves differently in train and eval mode.
- I can show overfitting on a small training set, reduce it with dropout, weight decay and early stopping, and read the train/validation curves.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| C3.1 | Optimizers: SGD, momentum, Adam and AdamW from scratch | Core | 2.5 |
| C3.2 | Learning-rate schedules: warmup, cosine decay, one-cycle, the LR range test | Core | 2 |
| C3.3 | Initialization: Xavier and He, vanishing and exploding gradients | Core | 2 |
| C3.4 | Normalization: BatchNorm and LayerNorm | Core | 2 |
| C3.5 | Regularization: dropout, weight decay, early stopping | Core | 2 |

**Hours:** reading 4 · coding 6.5 · project 0 · quiz and cheat sheet 1 · **total 11.5**

**Labs**

- **C3.1:** SGD, SGD with momentum, Adam and AdamW from scratch, each run side by side with its
  `torch.optim` counterpart on the same parameters and gradients for 100 steps and asserted equal.
  Plot each optimizer's path on an ill-conditioned quadratic and on the Rosenbrock function. Turn
  Adam's bias correction off and plot the size of the first few steps. Train the same model with
  Adam plus L2 and with AdamW, and compare the final weight norms.
- **C3.2:** step decay, cosine decay, linear warmup + cosine, and one-cycle as plain Python functions
  of the step number, plotted and checked against `torch.optim.lr_scheduler`. An LR range test from
  scratch: raise the learning rate exponentially over about 100 batches, plot loss against learning
  rate on a log axis, and pick a value. Train a Fashion-MNIST MLP with a constant learning rate and
  with warmup + cosine, and compare the curves.
- **C3.3:** a 20-layer MLP with tanh and then ReLU. Plot the standard deviation of activations and
  the gradient norm at every layer for four initializations: N(0, 1), N(0, 0.01²), Xavier and He.
  Write the Xavier and He initializers from scratch and compare their weight statistics with
  `nn.init`. Work the variance argument by hand for a layer with 4 inputs first.
- **C3.4:** BatchNorm forward and backward from scratch, including the running mean and variance
  used in eval mode, checked against `nn.BatchNorm1d` (outputs and gradients). LayerNorm from scratch
  checked against `nn.LayerNorm`. Reproduce the classic bug: forget `model.eval()` and predict one
  example at a time. Train the deep MLP from C3.3 with and without BatchNorm and compare.
- **C3.5:** inverted dropout from scratch, with a check that the expected activation is unchanged,
  compared with `nn.Dropout`. Train on only 1,000 Fashion-MNIST images until the model clearly
  overfits, then add dropout, weight decay and early stopping (patience, restoring the best
  checkpoint with your C2.3 code) one at a time. Plot train and validation curves for each, the
  deep-learning version of the B4.5 learning curves.

**Project milestone:** none.

**Prerequisites:** C2.1–C2.3, A3.3 (variance and covariance, for C3.3), A4.4 (gradient descent),
A5.1 (variance of a sum, for C3.3), B2.2 (L2 regularization as a prior, for C3.5), B4.3
(hyperparameter tuning), B4.5 (bias–variance and learning curves).

**Resources** (optional)

- Zhang, Lipton, Li & Smola, *Dive into Deep Learning* (free online), the optimization chapter and the batch-normalization section: https://d2l.ai/ (VERIFY link)
- Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014), arXiv:1412.6980: https://arxiv.org/abs/1412.6980 (VERIFY link)
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW, 2017), arXiv:1711.05101: https://arxiv.org/abs/1711.05101 (VERIFY link)

**GPU:** no. Every lab uses MLPs on Fashion-MNIST subsets and runs on CPU in a few minutes. A GPU
only helps if you want the `GPU_SCALE` full-dataset runs; use the Colab or Kaggle steps at the top of
this file.

**Used later in:** AdamW and warmup + cosine in the small GPT (F1.4) and LoRA fine-tuning (J1.3)
(C3.1, C3.2); vanishing gradients in RNNs (E2.1) (C3.3); BatchNorm in ResNet (D1.3) and LayerNorm in
the transformer block (F1.2) (C3.4); dropout and early stopping in every later project (C3.5).

---

## C4 · GPU training & debugging (course week 16)

**Teaching order this week:** C4.1 → C4.3 → C4.2 → C4.4. Debugging (C4.3) is Core and is what CP3
tests, so it comes before the two Depth topics, which you can drop if the week runs long.

**Learning objectives**

- I can run the same training script on CPU and on a Colab or Kaggle GPU, and measure throughput (examples per second) correctly, including `torch.cuda.synchronize()` before stopping a timer.
- I can estimate a model's training memory by hand (weights, gradients, optimizer state and activations) and check the estimate with `torch.cuda.max_memory_allocated()`.
- I can run the "overfit one batch" test, check that the starting loss is about `ln(number of classes)`, and log per-layer gradient norms.
- I can look at a broken loss curve, name its likely cause, and say how to confirm and fix it.
- I can explain why fp16 training needs loss scaling and bf16 usually does not, and show that gradient accumulation over 4 micro-batches gives the same gradient as one batch 4 times larger. *(Depth)*
- I can explain data parallelism as averaging per-shard gradients, and how much memory DDP, ZeRO-style sharding and FSDP each save. *(Depth)*

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| C4.1 | Training on a Colab/Kaggle GPU: moving work to the device, throughput, memory | Core | 2.5 |
| C4.3 | Debugging training runs: overfit one batch, read loss curves, watch gradient norms, the common bugs | Core | 3 |
| C4.2 | Mixed precision (fp16/bf16, autocast, loss scaling), gradient accumulation and checkpointing | Depth | 2 |
| C4.4 | Distributed training concepts: data parallel, DDP, FSDP (concepts only, no multi-GPU needed) | Depth | 1.5 |

**Hours:** reading 3 · coding 6 · project 0 · quiz, cheat sheet and CP3 self-check 1.5 · **total 10.5**

**Labs**

- **C4.1:** a matrix-multiply benchmark from scratch: time growing square matmuls on CPU and GPU,
  compute the achieved FLOP/s, and show what goes wrong if you forget to synchronize before timing.
  Measure the cost of host-to-device copies, with and without `pin_memory`, and the effect of
  `num_workers`. Train your Fashion-MNIST MLP at several batch sizes and plot examples per second
  and peak GPU memory against batch size. Estimate memory by hand first (for Adam: weights + gradients
  + two optimizer states, about 16 bytes per parameter in float32, plus activations) and compare it
  with `torch.cuda.max_memory_allocated()`. On the laptop the lab runs the CPU cells only.
- **C4.3:** a gradient-norm logger built with hooks that records each layer's gradient norm every
  step. The overfit-one-batch test on a healthy model. Then a "bug zoo": 8 short training runs, each
  with one planted bug, that you diagnose from their loss curves and gradient logs before reading
  the explanation in the answer key. The bugs: no `zero_grad()`; softmax applied before
  `CrossEntropyLoss`; learning rate 100× too high; `model.eval()` never called; labels shuffled
  separately from inputs; a `(n, 1)` versus `(n,)` broadcasting bug in the loss; a layer whose
  parameters were never passed to the optimizer; and validation data leaking into training. Finish
  with `torch.autograd.set_detect_anomaly(True)` on a run that produces NaN.
- **C4.2 (Depth):** show fp16's limits with `torch.finfo` and with gradients like 1e-8 that underflow
  to zero, then loss scaling by hand on a toy model. Compare fp16, bf16 and fp32 ranges. Train with
  `torch.autocast` and a gradient scaler on the GPU (fp16 on a T4, which has no fast bf16; VERIFY
  (as of 2026-10)), and with bf16 autocast on the laptop CPU. Gradient accumulation from scratch:
  assert that 4 micro-batches of 32 with the loss divided by 4 give the same gradient as one batch of
  128, and explain why BatchNorm breaks the equality. Activation checkpointing with
  `torch.utils.checkpoint` on a deep MLP: measure the memory saved and the extra time on the GPU.
- **C4.4 (Depth):** simulate data parallelism in one process: split a batch across K "workers",
  compute each shard's gradient, average them, and assert the result equals the full-batch gradient.
  Ring all-reduce from scratch on K simulated arrays, counting how much data each worker sends. Run
  real DDP with the `gloo` backend in 2 CPU processes on the laptop (VERIFY it works on Windows; if
  not, run it on Kaggle). By hand, a memory table for a 7-billion-parameter model trained with Adam:
  plain data parallel versus sharding the optimizer state, then the gradients, then the weights (the
  ZeRO stages that FSDP implements).

**Project milestone:** none. P4 is done; this week ends with the CP3 self-check below.

**Prerequisites:** C2.1–C2.3 (device handling in particular), C3.1–C3.5, A1.2 (broadcasting, for
the `(n, 1)` bug), B1.4 (leakage, for the bug zoo), B4.5 (learning curves). C4.2 also needs C3.4
(BatchNorm) for the accumulation caveat.

**Resources** (optional)

- Andrej Karpathy, "A Recipe for Training Neural Networks" (2019): http://karpathy.github.io/2019/04/25/recipe/ (VERIFY link)
- PyTorch docs, "Automatic Mixed Precision package (torch.amp)": https://pytorch.org/docs/stable/amp.html (VERIFY link)
- Hugging Face, "The Ultra-Scale Playbook: Training LLMs on GPU Clusters" (for C4.4): https://huggingface.co/spaces/nanotron/ultrascale-playbook (VERIFY link)

**GPU:** required for C4.1 and C4.2, which are about measuring GPU behaviour; not needed for C4.3,
and helpful for C4.4. On **Colab**, switch the runtime to a T4 GPU and run the C4.1 and C4.2 labs with
`GPU_SCALE` on; each needs well under an hour of GPU time. On **Kaggle**, choose *GPU T4 ×2* if you
want to try C4.4's DDP on two real GPUs: write the training code to a `.py` file and launch it with
`!torchrun --nproc_per_node=2 your_script.py`, because DDP does not run from notebook cells. Free-tier
GPU types and quotas: VERIFY (as of 2026-10).

**Used later in:** GPU training for P5 in D2, the small GPT in F1.4, and P9 in J1.3 (C4.1); bf16 and
quantized weights in F3.4 and QLoRA in J1.3 (C4.2); debugging every training run you do from here on
(C4.3); pretraining compute in F2.2 and scaling laws in F2.3 (C4.4).

---

## You're ready for Part D when…

This is checkpoint **CP3**. Answer each item without notes. The answers belong in your quiz reviews,
not here. Items in **bold** are the outline's CP3 pass criteria; the others are self-checks that
point to the topic to revisit.

1. You can do the forward and backward pass of a one-hidden-layer network on paper, and give the
   shapes of `dW`, `db` and `dX` for a batch of `n` examples.
2. You can explain why `+=`, not `=`, is the right way to store gradients in an autograd engine,
   using `y = x * x`.
3. You can write a numerically stable `log_softmax` from memory and say why subtracting the max does
   not change the answer.
4. **Your P4 gradients match PyTorch's to within 1e-6** in float64.
5. You can write a PyTorch training loop and evaluation loop from memory, including `zero_grad()`,
   `model.train()`, `model.eval()` and `torch.no_grad()`, and say what breaks if each is missing.
6. **You can explain why Adam usually trains faster than plain SGD, what its bias correction does, and
   how AdamW's weight decay differs from L2.**
7. **You can explain why warmup helps**, and sketch warmup + cosine decay.
8. You can explain why He initialization uses `2 / fan_in`, and what happens to a 20-layer ReLU
   network initialized with N(0, 0.01²).
9. **You can explain why normalization helps training**, what BatchNorm and LayerNorm each normalize
   over, and why BatchNorm needs eval mode.
10. **Given three broken loss curves** (for example: flat at `ln(10)`, rising to NaN, and a training
    loss that falls while the validation loss rises), you can name the likely cause of each and the
    fix.
11. You can estimate the training memory of a model with Adam from its parameter count, and say
    what you would try first if it does not fit on the GPU.
12. P4 scores ≥ 70% on its rubric (the course's usual project bar), and your Part C weekly quizzes
    average ≥ 80%.

If two or more items fail, take one catch-up week on the topic ids your reviews flagged before you
start Part D. That is the outline's checkpoint rule.
