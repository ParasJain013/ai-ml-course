# P4 reference notes and rubric

Brief: [`projects/P4/BRIEF.md`](../../../projects/P4/BRIEF.md). These notes are for the reviewer. The reference run
[`reference_run.py`](reference_run.py) trains the brief's minimal model (64 → 128 → 10, ReLU, SGD with learning rate 0.1,
batch 32, 30 epochs, split 60/20/20 stratified with `random_state=0`) with a hand-written NumPy backward pass and a PyTorch
twin fed the same batches. It is not the learner's solution (the learner must use their own engine); it fixes what the
numbers should look like.

## What a strong submission looks like

- **An engine that reads like C1.3, packaged.** A `Tensor` class whose operations each create one node with its inputs and
  a closure; `backward()` over an iterative topological sort; `+=` in every backward rule; `unbroadcast` called by every
  broadcasting operation; the cross-entropy as one fused node with backward $(\mathbf{P} - \mathbf{Y})/n$. Imports cleanly from the scripts
  (no copy-pasted engine in `train.py`).
- **Tests that could fail.** Float64; non-square shapes; random non-zero biases; a random weighting of the output in the
  scalar loss; inputs kept away from ReLU's kink; the required broadcast cases; a value used twice; an end-to-end MLP check.
  The deliberate-bug demonstration shows the failing test output, so the reviewer can see the suite catches what it claims
  to.
- **A twin that really is identical.** Initial weights generated once (NumPy) and copied into PyTorch, with the layout
  handled: `nn.Linear` stores its weight as `(out, in)`, so the learner either transposes when copying or uses plain
  tensors with `x @ W + b`. The batch order is a stored list of index arrays used by both loops. The same learning rate,
  no momentum, no weight decay, float64 on both sides.
- **A comparison reported as numbers.** The max absolute gradient difference per parameter on at least 5 batches (the
  first 5 is fine; better, some early and some late), the largest gap between the loss curves, both test accuracies, a
  bootstrap CI and the logistic regression baseline on the same split.
- **A README with a real bug story.** Something like: "my bias gradients were 32 times too large; the gradient check passed
  with batch size 1 and failed with 32; I had applied the mean's $\frac1n$ in the loss and again in the bias rule." Specific
  symptom, hypothesis, test, fix.
- **A model card that does not oversell.** Teaching use only; 8×8 digits are a toy; per-class accuracy and the most common
  confusions from the confusion matrix.

## Reference approach and expected numbers

Real output of `uv run python answers/projects/P4/reference_run.py` (NumPy 2.5.4, PyTorch 2.14.1, CPU):

```text
== data ==
train (1077, 64)  val (360, 64)  test (360, 64)

== numpy training, with PyTorch fed the same batches ==
epoch  1: train loss numpy 1.7448  torch 1.7448   val acc numpy 0.7694
epoch  5: train loss numpy 0.3401  torch 0.3401   val acc numpy 0.9306
epoch 10: train loss numpy 0.1820  torch 0.1820   val acc numpy 0.9583
epoch 20: train loss numpy 0.1060  torch 0.1060   val acc numpy 0.9694
epoch 30: train loss numpy 0.0735  torch 0.0735   val acc numpy 0.9694
max |grad numpy - grad torch| over the first 5 batches: < 1e-12
max |param numpy - param torch| after 30 epochs: < 1e-10
test accuracy: numpy 0.9694   torch 0.9694

== float32 versus float64 ==
first batch: max |float64 grad - float32 grad| = 3.7e-08
same batches for 30 epochs, float32 torch vs float64 numpy: worst grad gap 1.7e-07, batches over 1e-6: 0 of 1020
test accuracy float32 0.9694
```

What to expect from the learner (ranges, because their split, width, learning rate and epochs may differ):

- **Gradient match in float64:** typically $10^{-15}$ to $10^{-12}$. Anything above about $10^{-9}$ in float64 means the two models are *not* computing
  the same thing (a different weight layout, a different loss reduction, a bias missing on one side), even though it passes
  the $10^{-6}$ bar. Ask about it.
- **Loss curves:** identical to 4 decimals for the whole run when everything matches; the parameter gap after training is
  around $10^{-10}$ or smaller.
- **Test accuracy:** about 95–98% for a 64 → 128 → 10 MLP on this data with a reasonable learning rate (0.05–0.5) and 20–50
  epochs; the reference gets 96.9%. With 360 test images, one image is 0.28 points, so a 95% bootstrap CI is roughly ±1.5–2
  points. Logistic regression on the same split typically lands within a couple of points of the MLP (digits are almost
  linearly separable in 64 pixels), so a learner who claims the MLP "clearly beats" it should show non-overlapping CIs or a
  paired comparison.
- **Float32:** on this small network the float32 gradients stay within about $2 \times 10^{-7}$ of float64 for the whole run, so a learner
  who runs PyTorch in float32 may still pass $10^{-6}$. The reason for float64 in the brief is margin (only a factor of ~5 here,
  and float32 rounding grows with layer width, depth and the size of activations) and the numerical gradient checks, which
  need float64 because central differences with $h = 10^{-6}$ lose about half the available digits. A strong README says this
  with their own measured numbers rather than claiming float32 "fails".

## Rubric

| Criterion | Weight | Full marks | Partial | Zero |
|---|---|---|---|---|
| Engine design and code quality | 15 | Importable package; one node per op with closures; iterative topological sort; `+=` everywhere; unbroadcast in every broadcasting op; fused cross-entropy; type hints and shapes in docstrings | Works but copy-pasted into scripts, recursive sort, or the loss built from unfused ops | Does not run, or uses PyTorch/autograd internally |
| Gradient tests | 20 | Every op checked in float64 with random output weighting, non-square shapes, random biases, kinks avoided; required broadcast, used-twice and end-to-end cases; deliberate bug shown failing | Checks present but with `sum()` only, square shapes, float32, or missing required cases; no bug demonstration | No numerical checks |
| Own training loop | 10 | Seeded He initialization; seeded per-epoch shuffle stored and reused; SGD written by hand; gradients zeroed each step; split fixed, test touched once | One of these missing (unseeded shuffle, gradients not zeroed, tuned on test) | Training uses PyTorch or a library optimizer for the engine side |
| PyTorch twin | 10 | Same weights (layout handled), batches, order, learning rate, float64; no hidden differences (momentum, weight decay, `reduction="sum"`) | Twin differs in one way that is acknowledged | Not identical and not acknowledged |
| The match | 20 | ≤ $10^{-6}$ on ≥ 5 batches, per parameter, reported; loss curves overlap with the largest gap reported; test accuracies within 1 point; CI and logistic-regression baseline | Match shown on fewer batches, curves only eyeballed, or no CI/baseline | No match, or match claimed without numbers |
| README | 15 | Design explained (nodes, order, unbroadcast, fusion); how the checks work; float64 reasoning with measured numbers; a specific bug story (symptom, hypothesis, test, fix) | Generic description; bug story vague ("I fixed some shape errors") | Missing |
| Model card | 10 | §6.10 sections, short; intended and out-of-scope use; data origin; accuracy with CI; per-class accuracy and confusions; limitations | Present but generic or missing per-class results | Missing |

Score each criterion as full (100% of its weight), partial (40–70%, reviewer's judgement, stated in the review) or zero.
**Pass ≥ 70.** Separately, record whether the CP3 criterion (gradients within $10^{-6}$ of PyTorch) is met.

## Common mistakes and how to spot them in a review

- **Weight layout.** Copying a `(in, out)` matrix into `nn.Linear.weight` (which is `(out, in)`) without transposing: for
  64 → 128 the shapes refuse, but for a square layer they silently accept and the twin is a different model. Look for
  `.T` or `torch.nn.functional.linear` and check one forward output matches before any training.
- **Different loss reductions.** The engine averages over the batch and PyTorch sums (or vice versa): gradients differ by
  exactly the batch size. A ratio of 32 between the two is this bug.
- **Unshared batch order.** Each side shuffles with its own generator; the curves look similar but never identical. Ask how
  the batch indices are passed.
- **Gradient checks that cannot fail.** `f(x).sum()` as the loss with symmetric inputs; square matrices only; zero biases;
  float32. Ask the learner to break `__matmul__` (swap the transpose) and show the test fails.
- **Comparing at the end only.** Matching test accuracy is weak evidence: two different models can both get 97%. The
  per-batch gradient match is the real test.
- **Dead ReLUs or a too-high learning rate** giving a plateau the learner blames on the engine. Ask for the fraction of active
  hidden units after training and the loss in the first 50 steps.
- **Test set used for tuning**, or the split not stratified or not seeded (B1.1).
- **A committed `.npz` of weights or the dataset.** Not allowed; ask for removal from history.

## Production questions to ask in the review

- Your engine and PyTorch agree to $10^{-12}$ in float64. In production you would train in bfloat16 on a GPU. What would you
  compare then, and what tolerance would you accept?
- A colleague adds a custom layer to a PyTorch model with a hand-written `autograd.Function`. What test would you require in
  CI before it is merged?
- Your engine is about how many times slower than PyTorch on CPU, and where does the time go? What would you change first
  (fusing operations, avoiding Python overhead per node, preallocating gradients)?
- How would you check that a model in production is the same model you validated (weights, preprocessing, numerics), and
  what could make two "identical" models disagree on a few inputs?
