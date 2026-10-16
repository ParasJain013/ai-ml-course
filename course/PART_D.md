# Part D. Computer vision — detailed syllabus

Course weeks 17–18. Weekly detail for Part D of [`COURSE_OUTLINE.md`](COURSE_OUTLINE.md), which stays
the source of truth: the topic ids, Core/Depth labels, project P5 and the paper below are taken from
it unchanged. Written 2026-10-16.

**Goal:** build convolutional networks from scratch, then solve real vision tasks with pretrained models.

Vision Transformers are not in this part. They need attention (F1), so the outline moved them to
G2.1, next to CLIP.

## At a glance

| Week | Course wk | Theme | Topics | Hours | Project | GPU |
|---|---|---|---|---|---|---|
| D1 | 17 | Convolutional networks | D1.1–D1.4 | 12.5 | P5 M0 | helpful (D1.3 only) |
| D2 | 18 | Applied vision | D2.1–D2.5 | 15 | **P5 due** | helpful |

About 27.5 hours in total. Each week's hours include about 1 hour for the weekly quiz and cheat sheet.
Topic hours cover reading the lesson and doing its lab and exercises.

**How the hours are counted.** *Reading* is working through the lessons and the paper, including the
worked examples by hand. *Coding* is the labs and the code exercises. *Project* is P5.

**Why D2 is a heavy week.** Five topics and a project deadline land in one 15-hour week, the top of
your 10–15 hour budget. To spread the load, P5 starts in D1 with milestone M0 (dataset and baseline).
If D2 still runs long, push D2.4 (segmentation) into the following weekend before you cut anything
from P5.

**If a week runs long,** drop its Depth topics first (D1.4, D2.5), as the outline's catch-up plan
says. One exception: **P5 needs Grad-CAM**, which is taught in D2.5. If you skip D2.5, still do its
Grad-CAM lab (about 45 minutes) and leave the dataset-bias experiment for later.

**Libraries this part needs** (Step 2 pins the versions): `torch` and `torchvision` (CPU build on the
laptop; Colab and Kaggle come with CUDA builds preinstalled), `torchmetrics` (only to check your mAP
in D2.3), `numpy`, `matplotlib`, `pillow`, `scikit-learn` (the logistic-regression head in D2.1).
torchvision's detection models, `transforms.v2` (including `MixUp` and `CutMix`) and pretrained-weight
enums are fast-moving APIs: VERIFY (as of 2026-10).

**Datasets** (all download into `data/`, never committed): MNIST and Fashion-MNIST, CIFAR-10, the
Oxford-IIIT Pet dataset (class labels *and* segmentation masks, used in D2.1 and D2.4), and the
Penn-Fudan pedestrian dataset (170 images with boxes and masks, used in D2.3). All are available
through `torchvision.datasets` or a direct download; VERIFY each source when the lab is generated.

**Hardware:** every lab runs on the laptop CPU at default settings: small images (32×32 to 128×128),
data subsets of a few thousand images, and 1–3 epochs. Each lab has a `GPU_SCALE` block with the
full-size settings for a free Colab or Kaggle GPU. How to use those is under "Running Part D on a
free GPU" at the end.

---

## D1 · Convolutional networks (course week 17)

**Learning objectives**

- I can compute a 2-D convolution of a small image with a 3×3 kernel by hand, with and without padding and stride.
- I can state the output-shape formula `⌊(n + 2p − k) / s⌋ + 1`, use it to predict every layer's
  shape in a CNN, and confirm it against PyTorch.
- I can implement a multi-channel `conv2d` forward pass from scratch in NumPy and match `torch.nn.functional.conv2d` to within 1e-5.
- I can count the parameters of a convolutional layer and explain why a CNN needs far fewer parameters than an MLP on the same image.
- I can compute the receptive field of a unit after a stack of convolutions and poolings.
- I can build and train a small CNN in PyTorch that beats my Part C MLP on Fashion-MNIST.
- I can explain what LeNet, AlexNet, VGG and ResNet each added, and why a residual connection lets a deep network train when a plain one of the same depth does not.
- I can write a ResNet basic block from scratch in PyTorch, including the projection shortcut when shapes change.
- (Depth) I can explain a depthwise separable convolution and compute how many parameters and multiply–adds it saves over a standard convolution.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| D1.1 | Convolution from scratch: kernels, stride, padding and the output-shape formula | Core | 3 |
| D1.2 | Pooling, receptive fields and a CNN in PyTorch | Core | 2.5 |
| D1.3 | Architectures from LeNet to ResNet: what each added, and why residual connections work (+ the ResNet paper) | Core | 3.5 |
| D1.4 | Efficient architectures: depthwise separable convolutions (MobileNet, EfficientNet) | Depth | 1.5 |

**Hours:** reading 4 · coding 6.5 · project 1 · quiz and cheat sheet 1 · **total 12.5**

**Labs**

- **D1.1:** apply hand-made kernels (box blur, Gaussian blur, Sobel edge, sharpen) to a grayscale
  image with a naive double loop, so you see what a kernel "looks for". Then write a multi-channel
  `conv2d(x, w, b, stride, padding)` for input `(N, C_in, H, W)` and weights `(C_out, C_in, k, k)`
  twice: with nested loops, and vectorized with `im2col` plus one matrix multiplication (the A2.2
  shape rule doing real work). Check both against `F.conv2d` across a grid of stride and padding
  values, and check the output shapes against the formula. Finally, write the backward pass for the
  weights and inputs and compare it with PyTorch autograd, which links this lab to your C1.3 engine.
- **D1.2:** max pooling and average pooling forward and backward from scratch (max pooling routes the
  gradient only to the winning position), checked against PyTorch. A receptive-field calculator that
  takes a list of `(kernel, stride)` layers, checked by back-propagating a one-pixel gradient through
  a real network and measuring the non-zero region. Then a LeNet-style CNN in PyTorch on
  Fashion-MNIST, compared with your Part C MLP on accuracy and parameter count.
- **D1.3:** write a ResNet `BasicBlock` from scratch (two 3×3 convs, BatchNorm, the identity or
  1×1 projection shortcut). Build a "plain" 20-layer CNN and a ResNet-20 that differ only by the
  shortcuts, train both on a CIFAR-10 subset, and plot training loss and per-layer gradient norms
  (the C4.3 tooling). The point to see: the plain net trains worse, even on the *training* set.
  Also count the parameters of LeNet-5, VGG-16 and ResNet-18 from their definitions and compare with
  the torchvision models. CPU default: 5,000 images, 3 epochs. `GPU_SCALE`: full CIFAR-10, 56 layers,
  30 epochs, which shows the degradation gap much more clearly.
- **D1.4 (Depth):** a depthwise separable block from scratch (`groups=C_in` depthwise conv, then a
  1×1 pointwise conv). A function that counts parameters and multiply–adds for standard versus
  separable convolutions, checked against your hand calculation. Time both blocks on the laptop CPU,
  and compare torchvision's `mobilenet_v3_small` with `resnet18` on parameters, latency and
  published ImageNet accuracy (VERIFY the accuracy figures from the torchvision docs).

**Paper:** *Deep Residual Learning for Image Recognition* (He, Zhang, Ren, Sun, 2015 arXiv; CVPR 2016),
with its reading guide in `papers/`. Read sections 1, 3 and 4.1 and the figure showing training error
of plain versus residual networks; your D1.3 lab reproduces that figure at small scale. About 1.5 hours,
counted in D1.3.

**Project milestone: P5 M0, dataset and baseline (1 h).** Choose the P5 dataset: 3–10 classes and a
few hundred to a few thousand images, either collected yourself or a public subset (for example
Oxford-IIIT Pet breeds or Oxford Flowers 102). Write a short data card: source, licence, class
counts, how you split train/validation/test, and any risk of the same subject appearing in two
splits (the B1.4 leakage check, applied to images). Run a majority-class baseline and record it.

**Prerequisites:** A1.2 (broadcasting), A2.1–A2.2 (matrix products and shape rules, for `im2col`),
A4.3 (chain rule, for the conv backward pass), C1.3 (your autograd engine), C2.1–C2.2 (`nn.Module`,
training loop), C3.3 (He initialization), C3.4 (BatchNorm, used in every ResNet block), C4.3 (reading
loss curves and gradient norms). Within the week: D1.2 needs D1.1; D1.3 needs D1.2; D1.4 needs D1.1.

**Resources** (optional)

- Stanford CS231n course notes, "Convolutional Neural Networks": https://cs231n.github.io/convolutional-networks/ (VERIFY link)
- Dumoulin & Visin, "A guide to convolution arithmetic for deep learning" (2016): https://arxiv.org/abs/1603.07285 (VERIFY link)
- *Dive into Deep Learning*, chapters on CNNs and modern CNNs: https://d2l.ai/ (VERIFY link)

**GPU:** helpful, for D1.3 only. D1.1, D1.2 and D1.4 run on the CPU in a minute or two. D1.3 runs
on the CPU at its default size in a few minutes; its `GPU_SCALE` run (ResNet-56 on full CIFAR-10)
takes a T4 well under an hour (VERIFY on the day).

**Used later in:** every vision model in D2 and P5 (D1.1–D1.3); residual connections in every
transformer block in F1.2 (D1.3); the CNN encoders in the autoencoder, VAE, DCGAN and DDPM U-Net in
G1 (D1.1–D1.3); the comparison with patch tokens in G2.1 (D1.1); edge-friendly models when you
serve with ONNX in J2.1 (D1.4).

---

## D2 · Applied vision (course week 18)

**Learning objectives**

- I can explain the difference between feature extraction and fine-tuning, and choose between them from dataset size and how similar it is to ImageNet.
- I can fine-tune a pretrained torchvision CNN with the right input normalization, a frozen-then-unfrozen schedule and a lower learning rate for the backbone.
- I can implement random crop, flip, colour jitter, Mixup and CutMix from scratch, apply them to the training set only, and measure their effect with an ablation.
- I can compute IoU between two boxes by hand and implement IoU, non-maximum suppression and average precision from scratch, matching torchvision and torchmetrics.
- I can explain mAP@0.5 versus mAP@[0.5:0.95] and what a detector's precision–recall curve shows.
- I can fine-tune a pretrained detector on a small dataset with torchvision and read its outputs (boxes, labels, scores).
- I can explain semantic versus instance segmentation, build a small U-Net with skip connections, and compute Dice and IoU from scratch.
- I can produce a Grad-CAM heatmap from scratch using PyTorch hooks and use it, together with per-group error rates, to find a model that relies on a shortcut instead of the object.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| D2.1 | Transfer learning: feature extraction versus fine-tuning with pretrained weights | Core | 2 |
| D2.2 | Data augmentation: geometric and colour transforms, Mixup and CutMix | Core | 2 |
| D2.3 | Object detection: boxes, IoU, non-maximum suppression, mAP, fine-tuning a pretrained detector | Core | 3 |
| D2.4 | Segmentation: semantic versus instance, U-Net, Dice and IoU | Core | 2.5 |
| D2.5 | `[RAI]` Error analysis for vision: Grad-CAM, per-group errors and dataset bias | Depth | 1.5 |

**Hours:** reading 3.5 · coding 7.5 · project 3 · quiz and cheat sheet 1 · **total 15**

**Labs**

- **D2.1:** on an Oxford-IIIT Pet subset at 128×128, compare three strategies with a pretrained
  `resnet18`: (a) frozen backbone, extract the 512-d embeddings once, and train your B2.1 logistic
  regression on them; (b) train only a new final layer in PyTorch; (c) fine-tune the last residual
  stage, then the whole network, with a 10× smaller learning rate on the backbone. Also train the
  same network from random initialization. Plot validation accuracy against training time for all
  four. Show what happens when you forget the ImageNet mean/std normalization. CPU default: 1,000
  images, embeddings cached to `data/`, 2 epochs. `GPU_SCALE`: full dataset, 224×224, 10 epochs.
- **D2.2:** random resized crop, horizontal flip and brightness/contrast jitter from scratch on
  tensors, then Mixup and CutMix from scratch, including the mixed labels and the loss for soft
  targets. Visualize a grid of augmented images with their mixed labels, and check your Mixup and
  CutMix against `torchvision.transforms.v2` (VERIFY the API). A small ablation on the D2.1 set: no
  augmentation, basic, basic + Mixup. This is the template for the P5 ablation. Exercise: find the
  bug in a pipeline that augments the validation set.
- **D2.3:** box IoU from scratch, vectorized for `(N, 4)` against `(M, 4)` boxes and checked against
  `torchvision.ops.box_iou`. Greedy NMS from scratch, checked against `torchvision.ops.nms`. Average
  precision from scratch: match predictions to ground truth at an IoU threshold, build the
  precision–recall curve (B4.1), integrate it, and average over classes and thresholds; check it
  against `torchmetrics` on a toy set of predictions. Then fine-tune torchvision's
  `fasterrcnn_mobilenet_v3_large_320_fpn` (VERIFY the name) on Penn-Fudan, plot predictions before
  and after, and report mAP. CPU default: 1 epoch on 60 images at 320 px, a few minutes.
  `GPU_SCALE`: the full set, a ResNet-50 FPN backbone, 10 epochs.
- **D2.4:** pixel accuracy, IoU and Dice from scratch, with a toy example where pixel accuracy is 95%
  and the model is useless (the B4.4 imbalance lesson, per pixel). A small U-Net from scratch in
  PyTorch: encoder, decoder with transposed convolutions, and the skip connections that carry
  detail. Train it on Oxford-IIIT Pet masks (foreground versus background) at 64×64, then remove the
  skip connections and compare the masks. Finally, run a pretrained Mask R-CNN in inference only on a
  few images to see instance masks. CPU default: 64×64, 1,000 images, 3 epochs. `GPU_SCALE`:
  128×128, the full dataset, 20 epochs.
- **D2.5 (Depth):** Grad-CAM from scratch with forward and backward hooks on the last convolutional
  stage, checked visually on the D2.1 model. Then a dataset-bias experiment: build a training set in
  which one class almost always has a particular background colour, train a classifier, and show
  that test accuracy drops when the correlation is removed, that Grad-CAM lights up the background,
  and that a per-group error table (class × background) exposes it. This links to the B4.6 fairness
  audit: the same per-group table, applied to images.

**Project milestone: P5 due, CV transfer learning (3 h).** On your M0 dataset: fine-tune a
pretrained CNN and compare it with the M0 baseline and a frozen-feature baseline (D2.1). Run an
augmentation ablation with at least three settings (none, basic, basic + Mixup or CutMix), with
3 seeds each if you have a GPU session, and report mean and spread rather than the best run. Do error
analysis on the test set: confusion matrix, the 10 most confident mistakes, Grad-CAM on correct and
wrong predictions, and per-group errors for any grouping your data allows (lighting, background,
source). Ship a model card (B4.6). Keep weights out of git. P5 needs **≥ 70%** on its rubric to count
towards CP4 (end of Part F).

**Prerequisites:** D1.1–D1.3; B1.4 (leakage, for splits and augmentation), B2.1 (logistic regression
for the frozen-feature head), B4.1 (precision, recall, PR curves, for mAP), B4.4 (class imbalance, for
pixel metrics), B4.6 (per-group metrics and model cards), C2.3 (saving, loading, device handling),
C3.2 (learning-rate schedules), C3.5 (regularization: augmentation is one), C4.1 (training on a free
GPU). Within the week: D2.2 builds on D2.1; D2.5 uses the D2.1 model.

**Resources** (optional)

- PyTorch tutorial, "Transfer Learning for Computer Vision": https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html (VERIFY link)
- PyTorch / torchvision tutorial, "TorchVision Object Detection Finetuning" (uses Penn-Fudan): https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html (VERIFY link)
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (2016): https://arxiv.org/abs/1610.02391 (VERIFY link)

**GPU:** helpful. Every lab has a CPU default that finishes in under 5 minutes. P5 is where a GPU
pays off: an augmentation ablation with 3 settings × 3 seeds is 9 fine-tuning runs, which on a free
T4 takes minutes per run at 224×224 and is impractical on the laptop at that size.

**Used later in:** the CLIP image encoder and ViT comparison in G2.1–G2.2 (D2.1); fine-tuning as a
general pattern, and the prompt-versus-fine-tune decision, in J1.3–J1.4 (D2.1); the U-Net inside the
DDPM in G1.4 (D2.4); the per-group audit in every model card from here on (D2.5); a vision-based
capstone option in P12 (all of D2).

---

## Running Part D on a free GPU

Free-tier GPUs, quotas and session limits change often: VERIFY (as of 2026-10).

- **Colab:** open the lab, then Runtime → Change runtime type → a GPU (usually a T4). Check with
  `torch.cuda.is_available()`. Upload the lab file or clone your repo, set the `GPU_SCALE` values,
  and save checkpoints and results to Google Drive, because the VM is wiped when the session ends.
- **Kaggle:** create a notebook, then Settings → Accelerator → GPU, and turn Internet on to download
  datasets and pretrained weights (Kaggle asks for phone verification first). Save outputs to
  `/kaggle/working` so they survive as notebook output. Kaggle gives a fixed weekly GPU quota, so
  check how much is left before starting the P5 ablation.
- **Both:** every lab already picks `device = "cuda" if torch.cuda.is_available() else "cpu"`
  (C2.3), so the same file runs in both places. Turn on mixed precision (C4.2) for the P5 runs. Bring
  back only results (metrics, plots, the model card), never weights, datasets or checkpoints.

---

## You're ready for Part E when…

Part D has no checkpoint of its own; P5 is assessed at CP4 (end of Part F). Answer each item without
notes. The answers belong in your quiz reviews, not here.

1. Given a 32×32×3 input, you can write down the output shape and parameter count of
   `Conv2d(3, 16, kernel_size=5, stride=1, padding=2)` followed by `MaxPool2d(2)`, and explain both numbers.
2. You can write a `conv2d` forward pass from scratch in 20 minutes and say how `im2col` turns it into one matrix multiplication.
3. You can compute the receptive field of three stacked 3×3 convolutions and say why VGG preferred them to one 7×7.
4. You can explain why a 56-layer plain network has *higher training* error than a 20-layer one, and how a residual connection fixes it.
5. You can say when you would freeze a pretrained backbone and when you would fine-tune it, and name two mistakes that make fine-tuning look worse than it is.
6. You can explain why augmentation must never touch the validation or test set, and what Mixup does to the labels.
7. You can compute the IoU of two boxes by hand, describe NMS in four steps, and explain what mAP@0.5 measures.
8. You can explain why pixel accuracy misleads on segmentation and what Dice and IoU fix, and what the U-Net skip connections carry.
9. You can use Grad-CAM and a per-group error table to tell a model that recognises the object from one that recognises the background.
10. **P5 is submitted** with its ablation, error analysis, Grad-CAM and model card, and its review scores **≥ 70%**.

If two or more items fail, revisit the topic ids your reviews flagged before you start Part E.
Part E does not depend on Part D (see the dependency map), so a weak spot here can also be fixed in
parallel with E1, but fix it before Part G, which builds on D1.
