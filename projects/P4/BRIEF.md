# P4 Neural net + autograd from scratch

Part C · starts week 13 · due course week 14 · GPU: no · ~6.5 h of project time

## The goal

Turn the tensor autograd engine you built in C1.3 (plus C1.4's fused cross-entropy) into a small, tested Python
package, then use it to train a multilayer perceptron on scikit-learn's handwritten digits with mini-batch gradient
descent that you write yourself. Build the identical model in PyTorch, give it the same starting weights and the same
batches in the same order, and show that the two agree: every gradient to within $10^{-6}$, the same loss curve, and the same
test accuracy. The question it answers: *do you understand what `loss.backward()` and `optimizer.step()` do well enough to
reproduce them exactly?*

## Why it matters for the job

Every deep-learning job assumes you can debug a training run, and most training bugs are gradient bugs: a wrong
transpose, a missing $\frac1n$, a broadcast that summed over the wrong axis, gradients never zeroed. An interviewer who reads this
project learns that you can build the machinery under PyTorch, test numerical code properly (gradient checks that could
actually fail, in the right precision), and compare two implementations rigorously instead of eyeballing a loss curve. It
is also a common interview exercise in its own right: "implement backprop for an MLP" or "write a tiny autograd".

## Data

- **scikit-learn's digits** (the default and the requirement): 1,797 grayscale images of handwritten digits, 8×8 pixels with
  values 0–16, 10 classes, bundled with scikit-learn (`sklearn.datasets.load_digits`), so nothing is downloaded. It is a
  copy of the test set of the UCI "Optical Recognition of Handwritten Digits" dataset (VERIFY licence on the UCI page, as
  of 2026-11; it is distributed with scikit-learn under its BSD licence).
- **Fashion-MNIST** (stretch only): 70,000 28×28 images of clothing in 10 classes, downloaded through `torchvision` into
  `data/fashion_mnist/` (gitignored) as in C2.2. MIT licence (VERIFY, as of 2026-11).

Use a fixed, stratified train/validation/test split with `SEED = 0` (for example 60/20/20, as in B1.1), and scale the
pixels to $[0, 1]$ by dividing by 16. Choose anything that needs choosing (learning rate, epochs, width) on the validation
set; touch the test set once, at the end. Never commit data, weights or checkpoints.

## Milestones

| Milestone | Week | Deliverable | Hours |
|---|---|---|---|
| M1 The engine passes its checks | 13 | `my-work/P4/tinygrad_course/` package (the name is yours to choose): `Tensor` with `+`, `*`, `@`, `sum`, `mean`, `reshape`, `relu`, `exp`, `log`, unbroadcasting, a fused `cross_entropy`, and `backward()`; `tests/test_engine.py` with a numerical gradient check of every op in float64 (max absolute difference ≤ $10^{-6}$), broadcast cases, a node used twice, and a 2-layer MLP end to end | 2 |
| Due: MLP on digits, matched against PyTorch | 14 | `train.py` (your MLP and your mini-batch SGD on your engine), `compare_torch.py` (the identical PyTorch model, same initial weights and batches), `figures/loss_curves.png`, `README.md`, `MODEL_CARD.md` | 4.5 |

C2.1 and C2.2 (week 14) teach the PyTorch you need for the comparison; do them before the due milestone. Commit after each
milestone and ask for a review (`Review my-work/P4/tests/test_engine.py` after M1).

## Requirements

1. **A package with tests.** The engine lives in an importable package under `my-work/P4/`, separate from the scripts that
   use it. `uv run pytest my-work/P4` runs every test and passes. No test depends on PyTorch except the comparison tests.
2. **Gradient checks that can fail.** Every operation is checked against central differences in float64 with a max
   absolute difference ≤ $10^{-6}$ (and you report the relative error too). Use non-square shapes, random non-zero biases, and a
   scalar loss that weights the outputs randomly (`(f(x) * R).sum()`), so that wrong transposes and errors that sum to zero
   are caught. Avoid kinks (ReLU inputs within $10^{-4}$ of 0). Required cases: `(n, d) + (d,)`, `(n, 1) * (1, d)`, a scalar added
   to a matrix, `sum` and `mean` with and without an axis, a value used twice in one graph, and a 2-layer MLP with the fused
   cross-entropy end to end.
3. **A test that proves a test works.** Introduce one deliberate bug (for example `=` instead of `+=` in `__mul__`, or a
   missing unbroadcast) on a branch or behind a flag, show that your test suite fails on it, and say so in the README.
4. **Your own training loop.** An MLP of at least 64 → 128 → 10 (ReLU hidden layer, logits out, the fused cross-entropy),
   He-initialized with a seeded generator, trained by mini-batch SGD that you write (shuffle each epoch with a seeded
   generator, step, zero the gradients). The batch order is generated once, stored, and given to both implementations.
5. **An identical PyTorch model.** Same architecture, initial weights copied from yours (mind PyTorch's `nn.Linear` weight
   layout, `(out, in)`, versus the course's `(in, out)`), the same batches in the same order, `torch.optim.SGD` with the same
   learning rate. Everything in float64 for the comparison.
6. **The match.** On at least 5 batches, every parameter's gradient matches PyTorch's with a max absolute difference
   ≤ $10^{-6}$ (this is the CP3 criterion). Over the whole run the two loss curves overlap (plot them on the same axes and
   report the largest gap), and the final test accuracies differ by at most 1 percentage point. Report the test accuracy
   with a 95% bootstrap confidence interval (A6.2), next to a baseline: logistic regression (B2.1) on the same split.
7. **Why float64.** In the README, explain why the $10^{-6}$ comparison is done in float64 (rounding in float32 and the precision
   central differences need) and report what the gap looks like if you run the PyTorch side in float32.
8. **README** (500–1,000 words): the engine's design (what a node stores, how `backward()` orders the graph, how
   broadcasting is undone, why the loss is one fused node), how the gradient checks work and what they caught, the
   comparison results with the figure, and **one bug you hit and how you found it** (what you saw, the hypothesis, the
   test that confirmed it, the fix).
9. **Model card** (`MODEL_CARD.md`, `[RAI]`), following §6.10 of `course/STYLE_GUIDE.md`, short (half a page): the intended
   use is teaching and verification, not production; out-of-scope uses (any real handwriting recognition: 8×8 digits from a
   few dozen writers in the 1990s do not represent real users); the data's origin and size; test accuracy with its CI;
   per-class accuracy as *Metrics by group* (which digits are confused, from the confusion matrix, B4.1); known
   limitations.

*Fast path:* requirements 1, 2, 4, 5, 6 and 8 unchanged; requirement 3 may be a short paragraph showing the failing test
output; the model card may be a few lines.

## Deliverables

All in `my-work/P4/`:

- the engine package and `tests/test_engine.py` (M1)
- `train.py`, `compare_torch.py`, `figures/loss_curves.png`, `README.md`, `MODEL_CARD.md` (due)

Ask for a review after M1 (`Review my-work/P4/tests/test_engine.py`) and of the whole project at the end
(`Review my-work/P4/README.md`).

## How it is graded

The full rubric with its scoring guide is in `answers/projects/P4/REFERENCE.md` (you do not open it). Criteria and
weights, out of 100:

| Criterion | Weight |
|---|---|
| Engine design and code quality (package, closures, unbroadcast, fused loss) | 15 |
| Gradient tests: coverage, float64, tests that can fail, the deliberate-bug demonstration | 20 |
| Own training loop: initialization, seeded shuffling, SGD, zeroing | 10 |
| PyTorch twin: same weights, layout, batches and optimizer | 10 |
| The match: ≤ $10^{-6}$ gradients on ≥ 5 batches, overlapping curves, accuracy within 1 point, with CI and baseline | 20 |
| README: design, checks, float64 reasoning, one bug found and fixed | 15 |
| Model card | 10 |

**Pass ≥ 70%.** The $10^{-6}$ gradient match is also a criterion of checkpoint CP3 (week 16).

## Stretch goals

- Repeat the comparison on a Fashion-MNIST subset (for example 10,000 training images, 784 → 256 → 10), using a `Dataset`
  and `DataLoader` from C2.2 to produce the shared batches.
- Add momentum or Adam to your optimizer (C3.1 previews it) and match `torch.optim.SGD(momentum=0.9)` or `torch.optim.Adam`
  step for step.
- Add a second hidden layer and dropout with a seeded mask shared with PyTorch, and keep the match.
- Time one training step of your engine against PyTorch on CPU and explain the difference.

## Topics used

A1.2, A4.2, A6.2, B1.1, B2.1, B4.1, C1.1, C1.2, C1.3, C1.4, C2.1, C2.2
