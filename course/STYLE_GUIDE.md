# Style guide

How every artifact in this course is written: notation, code conventions, file names and the exact
templates. Written 2026-10-16. `COURSE_OUTLINE.md` and `PART_*.md` decide *what* is taught; this file
decides *how it looks*. When a lesson and this guide disagree, the lesson is wrong. When this guide
and the outline disagree, stop and fix one of them.

Contents: [1 Notation](#1-notation) · [2 Writing](#2-writing) · [3 Code](#3-code) ·
[4 Plots](#4-plots) · [5 File naming](#5-file-naming) · [6 Templates](#6-templates)

---

## 1. Notation

### 1.1 Rules that never bend

1. **Every vector and matrix gets its shape the first time it appears in a lesson**, written as a
   NumPy tuple: "$\mathbf{X}$, shape `(n, d)`". Repeat the shape whenever a formula changes it.
2. **Rows are examples.** A data matrix is `(n, d)`: `n` examples, `d` features, one example per row.
   This matches NumPy, pandas, scikit-learn and PyTorch, so the math and the code read the same way.
3. **Vectors are columns in the math, 1-D arrays in the code.** $\mathbf{w}$ is a column vector of
   shape `(d, 1)` on paper and a NumPy array of shape `(d,)` in code. When the difference matters
   (broadcasting bugs), the lesson says so explicitly.
4. **Define every symbol before or right after its first use**, in words, with its shape and units.
   A "Symbols" table opens the math section of every lesson.
5. **`log` is the natural log.** Entropy and cross-entropy are in nats unless a lesson says bits
   ($\log_2$).
6. **Indices start at 1 in the math and 0 in the code.** Say so the first time it matters in a lesson.

### 1.2 Typography

| Kind | Math | Code | Example |
|---|---|---|---|
| Scalar | lowercase italic | lowercase | $x$, $\eta$, `lr` |
| Vector | bold lowercase | lowercase | $\mathbf{x}$, `x` (shape `(d,)`) |
| Matrix | bold uppercase | uppercase | $\mathbf{W}$, `W` (shape `(d, k)`) |
| Tensor (3-D or more) | bold uppercase, shape stated | uppercase | $\mathbf{A}$, shape `(B, T, T)` |
| Set or dataset | calligraphic | `snake_case` | $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ |
| Random variable | uppercase italic | — | $X \sim \mathcal{N}(\mu, \sigma^2)$ |
| Estimate | hat | `_hat` suffix | $\hat{y}$, `y_hat` |
| Element | subscript | indexing | $x_j$ = `x[j-1]`, $W_{ij}$ = `W[i-1, j-1]` |
| Example *i* | subscript $i$ on a bold symbol | row | $\mathbf{x}_i$ = `X[i-1]` |
| Layer *l* | superscript in brackets | list index | $\mathbf{W}^{[l]}$ = `Ws[l-1]` |
| Time step *t* | subscript $t$ | `t` | $\mathbf{h}_t$ |

Markdown math uses `$...$` inline and `$$...$$` for display equations (GitHub and VS Code both render
them). Never put a formula in a code block unless it is code.

### 1.3 Reserved symbols

These mean the same thing in every lesson. A lesson that needs a symbol for something else picks a
different letter, or redefines it in a boxed note ("In this lesson only, $T$ is the number of
diffusion steps").

**Data and models**

| Symbol | Meaning | Shape / type | Code name |
|---|---|---|---|
| $n$ | number of examples | int | `n` |
| $d$ | number of input features | int | `d` |
| $K$ | number of classes | int | `k` / `n_classes` |
| $\mathbf{X}$ | design (data) matrix | `(n, d)` | `X` |
| $\mathbf{x}_i$ | one example | `(d,)` | `X[i]` |
| $\mathbf{y}$ | targets | `(n,)` | `y` |
| $\hat{\mathbf{y}}$ | predictions | `(n,)` | `y_hat` |
| $\mathbf{w}$ | weight vector | `(d,)` | `w` |
| $b$ | bias (intercept) | scalar | `b` |
| $\mathbf{W}$ | weight matrix of a layer | `(d_in, d_out)` | `W` |
| $\mathbf{z}$ | logits (pre-activation) | `(n,)` or `(n, K)` | `z` / `logits` |
| $\mathbf{p}$ | predicted probabilities | `(n, K)` | `p` / `probs` |
| $\boldsymbol{\theta}$ | all trainable parameters | — | `params` |
| $\sigma(\cdot)$ | logistic sigmoid | elementwise | `sigmoid` |
| $\operatorname{softmax}(\cdot)$ | softmax over the last axis | — | `softmax` |

**Training**

| Symbol | Meaning | Code name |
|---|---|---|
| $\mathcal{L}$ | total (mean) loss over a batch or dataset | `loss` |
| $\ell$ | loss of one example | `loss_i` |
| $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ | gradient; always the same shape as $\boldsymbol{\theta}$ | `grad` / `dW` |
| $\eta$ | learning rate | `lr` |
| $\lambda$ | regularization strength | `lam` (`alpha` in scikit-learn) |
| $B$ | batch size | `batch_size` |
| $E$ | number of epochs | `epochs` |

**Linear algebra**

| Symbol | Meaning |
|---|---|
| $\mathbf{A}^\top$ | transpose |
| $\mathbf{A}^{-1}$ | inverse (written, almost never computed; see A3.4) |
| $\mathbf{I}_d$ | identity, shape `(d, d)` |
| $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ | SVD; for $\mathbf{A}$ `(n, d)`: $\mathbf{U}$ `(n, r)`, $\boldsymbol{\Sigma}$ `(r, r)`, $\mathbf{V}$ `(d, r)` (thin SVD, $r = \min(n, d)$) |
| $\lambda_j$, $\mathbf{v}_j$ | eigenvalue and eigenvector (context makes clear this is not the regularization $\lambda$) |
| $\lVert\mathbf{x}\rVert_2$, $\lVert\mathbf{x}\rVert_1$ | L2 and L1 norms |
| $\mathbf{x}^\top\mathbf{y}$ or $\mathbf{x}\cdot\mathbf{y}$ | dot product |
| $\odot$ | elementwise (Hadamard) product |

**Probability and information**

| Symbol | Meaning |
|---|---|
| $P(A)$ | probability of an event |
| $p(x)$ | probability mass or density function |
| $p(y \mid \mathbf{x})$ | conditional |
| $\mathbb{E}[X]$, $\operatorname{Var}[X]$, $\operatorname{Cov}[X, Y]$ | expectation, variance, covariance |
| $\mathcal{N}(\mu, \sigma^2)$, $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | Gaussian, multivariate Gaussian ($\boldsymbol{\Sigma}$ `(d, d)`) |
| $\mu$, $\sigma$, $\sigma^2$ | mean, standard deviation, variance |
| $\bar{x}$, $s$ | sample mean, sample standard deviation |
| $H(p)$ | entropy |
| $H(p, q)$ | cross-entropy |
| $D_{\mathrm{KL}}(p \,\|\, q)$ | KL divergence |
| $\alpha$ | significance level (statistics); context makes clear when it is not |

**Deep learning and transformers**

| Symbol | Meaning | Code name |
|---|---|---|
| $T$ | sequence length (tokens) | `T` / `seq_len` |
| $V$ | vocabulary size | `vocab_size` |
| $d_{\text{model}}$ | model (residual stream) width | `d_model` |
| $n_h$ | number of attention heads | `n_heads` |
| $d_k$ | head dimension, $d_{\text{model}} / n_h$ | `d_head` |
| $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ | queries, keys, values, each `(B, n_h, T, d_k)` | `q, k, v` |
| $C, H, W$ | image channels, height, width; an image batch is `(B, C, H, W)` | `C, H, W` |

$\mathbf{V}$ (values) and $V$ (vocabulary) clash with the SVD's $\mathbf{V}$; in Parts F and G the SVD
is not used, so the transformer meaning wins there.

**Reinforcement learning**

| Symbol | Meaning |
|---|---|
| $s, a, r$ | state, action, reward |
| $\gamma$ | discount factor |
| $\pi(a \mid s)$ | policy |
| $V^\pi(s)$, $Q^\pi(s, a)$ | state value, action value |
| $G_t$ | return from step $t$ |
| $A_t$ | advantage |

### 1.4 Shapes in the code

Every from-scratch function states its shapes in its docstring or in a comment on the line where a
shape changes:

```python
def predict_proba(X: FloatArray, w: FloatArray, b: float) -> FloatArray:
    """Logistic regression probabilities. X (n, d), w (d,) -> (n,)."""
    z = X @ w + b          # (n,)
    return sigmoid(z)      # (n,)
```

Labs check shapes with `assert x.shape == (n, d), x.shape` at the points a bug is likely.

---

## 2. Writing

- **Audience:** a strong software engineer whose math is rusty. Never skip a step in a derivation; a
  "clearly" or "it is easy to see" is a bug.
- **Order inside every topic:** intuition → minimal math → worked example by hand with small numbers →
  code. From scratch in NumPy first, then the standard library, and show the two agree.
- **Backend bridges:** use an analogy from TypeScript, NestJS, PostgreSQL, Docker or Terraform when it
  is honest, in a callout that also says where it breaks:
  > **Bridge:** a scikit-learn `Pipeline` is middleware: every request (row) passes through the same
  > ordered steps. **Where it breaks:** middleware has no `fit`; a pipeline step *learns* from the
  > training data, and learning from the test data is leakage (B1.4).
- **Labels:** every topic heading carries **Core** or **Depth**, and `[RAI]` when it is a Responsible
  AI topic. Depth topics may be skipped; Core may not.
- **Fast-moving facts** (model names, prices, library APIs, benchmark numbers, free-tier quotas) carry
  `VERIFY (as of YYYY-MM)`. Library versions are the ones pinned in `pyproject.toml`.
- **Citations:** only real papers, books and docs, with authors and year; links marked `(VERIFY
  link)`. Never invent a URL, a quote or a number from a paper.
- **Cross-references** use topic ids that exist in `manifest.json`, linked to the lesson file:
  `[B1.4](../B/b1_4_data_leakage.md)`. Projects are `P1`–`P12`, checkpoints `CP1`–`CP6`.
- **Terms** come from `GLOSSARY.md`. The first use of a glossary term in a lesson is in **bold**. A
  new term is added to the glossary in the same commit.
- **Tone:** plain, direct, no flattery, no hype ("powerful", "revolutionary"). If something is wrong,
  say it is wrong and why.
- **Spelling:** consistent within a file; the course uses US spelling for code words (`normalize`,
  `color` in matplotlib) and does not fight library names.
- **Length:** a lesson is 1,500–3,500 words plus code. If it needs more, split the topic (`B3.3a`,
  `B3.3b`) and update the manifest.

---

## 3. Code

### 3.1 Environment and running

- Python 3.12, managed by `uv`. All libraries are pinned in `pyproject.toml`; never `pip install`
  inside a lab. A lab that needs an optional group says so at the top (`uv sync --group local-llm`).
- Run a lab from the repository root: `uv run python labs/A/a1_2_numpy_arrays.py`. In VS Code, open
  the file and use "Run Cell" on the `# %%` markers (the `ipykernel` from the project venv).
- **Budget:** with default settings every lab finishes on the laptop CPU (Intel Core Ultra 5 135U,
  15 GB RAM, no NVIDIA GPU) in **under 5 minutes**, and prints its elapsed time at the end.
- **`GPU_SCALE`:** a block near the top documents the full-size settings for a free Colab or Kaggle
  GPU. It is `False` by default; flipping it to `True` is the only change needed on Colab/Kaggle.
- **Data** downloads into `data/<dataset_name>/` (gitignored), is idempotent (skip if present), and
  prints where it came from. Model weights go to the Hugging Face cache or `data/`, never into git.
- **Secrets** (LLM API keys) come from environment variables loaded from `.env`; a lab that needs one
  fails fast with a one-line message naming the variable.
- **Services** (Postgres + pgvector) are started by `docker compose` files in `labs/<Part>/`; a lab
  that needs one checks the connection first and fails fast with the command to start it.

### 3.2 Conventions

| Topic | Rule |
|---|---|
| Seeds | `SEED = 0` at the top. NumPy: `rng = np.random.default_rng(SEED)` passed around explicitly; never `np.random.seed` or the legacy `np.random.*` functions. PyTorch: `torch.manual_seed(SEED)`; Python: `random.seed(SEED)`. scikit-learn: `random_state=SEED` on every estimator and splitter. |
| Determinism | Printed results must be identical run to run on the same machine. Round printed floats (`f"{x:.4f}"`). Timings and anything else that varies are printed on lines that start with `~` so the expected-output check ignores them. |
| Type hints | Every function has type hints. Arrays use `FloatArray = npt.NDArray[np.float64]` (or `IntArray`); tensors use `torch.Tensor`. |
| Docstrings | One line, imperative, with shapes: `"""Compute pairwise distances. A (n, d), B (m, d) -> (n, m)."""` |
| Names | Match the notation table (§1.3): `X`, `y`, `w`, `b`, `lr`, `n`, `d`. Upper-case module constants (`N_SAMPLES`, `EPOCHS`). No single-letter names outside math. |
| From scratch vs library | The from-scratch function comes first, the library call second, then a check: `np.testing.assert_allclose(mine, theirs, rtol=1e-6, atol=1e-8)` followed by `print("matches <library>: OK")`. Tolerances looser than `1e-6` are justified in a comment. |
| Structure | Small pure functions; no classes unless the topic is about one (`nn.Module`, an autograd `Tensor`). No hidden global state except the constants block. |
| Imports | Standard library, third-party, then nothing else. Labs are self-contained: they never import from another lab. When a later topic reuses earlier code, it copies the function with a comment `# from labs/C/c1_3_autograd_engine.py`. |
| Printing | Each cell that produces checked output starts with `section("name")`, which prints `== name ==`. The lesson's expected-output blocks quote these sections. |
| Device | `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`; never hard-code `"cuda"`. |
| Errors | Fail fast with a clear message; no bare `except`. |
| Style | Formatted as `ruff format` would (88 columns). |

### 3.3 The lab header (copied into every lab)

```python
# %% [markdown]
# # B1.4 Data leakage — lab
# Lesson: lessons/B/b1_4_data_leakage.md
# Run: uv run python labs/B/b1_4_data_leakage.py   (CPU, defaults: ~40 s)
# GPU: no

# %%
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:  # plain `uv run`: draw to files, never open windows
    matplotlib.use("Agg")
    # matplotlib 3.11 warns when plt.show() runs under Agg; under `uv run` it is a harmless no-op
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()


def repo_root() -> Path:
    """Find the course folder (works under `uv run` and in VS Code cells)."""
    here = Path(globals().get("__file__", Path.cwd())).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("Run labs from inside the ai-ml-course folder.")


ROOT = repo_root()
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "b1_4"            # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"       # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
GPU_SCALE = False
N_SAMPLES = 50_000 if GPU_SCALE else 2_000
```

Labs that use PyTorch add `import torch`, `torch.manual_seed(SEED)` and `DEVICE` to the header. The
last cell of every lab is:

```python
# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
```

### 3.4 Answer-key code

Code solutions in `answers/` follow the same conventions and are run before they are committed; the
answer key quotes their real output.

---

## 4. Plots

- **One key plot per lesson**, saved by the lab to `lessons/<Part>/img/<id>_<what>.png` (committed,
  under ~150 KB) and embedded in the lesson. Other plots go to `outputs/<id>/` (gitignored).
- matplotlib only, object-oriented API: `fig, ax = plt.subplots(figsize=(6, 4), dpi=120)`.
- Every axis has a label with units; every plot has a title that states the takeaway ("Test error
  rises after depth 6"), not just the variables.
- Colours: matplotlib's default `tab10` cycle, which is colourblind-friendlier than ad-hoc colours.
  Never encode meaning with colour alone; add markers or line styles.
- Axes start at zero for bar charts. Log scales are labelled "(log scale)". No 3-D plots unless the
  topic is about 3-D.
- Legends do not cover data (`loc="best"` or outside). `fig.tight_layout()` before saving.
- Save with `fig.savefig(path, dpi=120, bbox_inches="tight")`, then `plt.show()` (a no-op under
  `uv run`, inline in VS Code) and `plt.close(fig)`.

---

## 5. File naming

Topic ids are written `B1.4` in prose. In file names they become **lowercase with an underscore**,
`b1_4`, followed by a short snake-case slug from the topic title. Part folders are the upper-case
letter. Week files use the course week, zero-padded, then the part week.

| Artifact | Path | Example |
|---|---|---|
| Lesson | `lessons/<P>/<id>_<slug>.md` | `lessons/B/b1_4_data_leakage.md` |
| Lesson images | `lessons/<P>/img/<id>_<what>.png` | `lessons/B/img/b1_4_leak_vs_clean_auc.png` |
| Lab | `labs/<P>/<id>_<slug>.py` | `labs/B/b1_4_data_leakage.py` |
| Answer key (topic) | `answers/lessons/<P>/<id>_<slug>.md` | `answers/lessons/B/b1_4_data_leakage.md` |
| Answer code (if any) | `answers/lessons/<P>/<id>_<slug>_ex<k>.py` | `answers/lessons/B/b1_4_data_leakage_ex2.py` |
| Weekly quiz | `quizzes/w<NN>_<pw>.md` | `quizzes/w10_b4.md` |
| Quiz answers | `answers/quizzes/w<NN>_<pw>.md` | `answers/quizzes/w10_b4.md` |
| Weekly cheat sheet | `cheatsheets/w<NN>_<pw>.md` | `cheatsheets/w10_b4.md` |
| Part cheat sheet | `cheatsheets/PART_<P>.md` | `cheatsheets/PART_B.md` |
| Master cheat sheet | `cheatsheets/MASTER.md` | |
| Project brief | `projects/P<k>/BRIEF.md` | `projects/P2/BRIEF.md` |
| Project reference notes + rubric | `answers/projects/P<k>/REFERENCE.md` | `answers/projects/P2/REFERENCE.md` |
| Paper reading guide | `papers/<year>_<short_name>.md` | `papers/2017_attention_is_all_you_need.md` |
| Paper answer notes | `answers/papers/<year>_<short_name>.md` | `answers/papers/2017_attention_is_all_you_need.md` |
| Learner's exercise | `my-work/<P>/<id>_<slug>.md` or `.py` | `my-work/B/b1_4_data_leakage.md` |
| Learner's project | `my-work/P<k>/...` | `my-work/P12/DESIGN.md` |
| Learner's quiz | `my-work/quizzes/w<NN>_<pw>.md` | `my-work/quizzes/w10_b4.md` |
| Review | `reviews/<same relative name as in my-work>.md` | `reviews/B/b1_4_data_leakage.md` |
| QA report | `course/qa/PART_<P>.md` | `course/qa/PART_B.md` |

`manifest.json` holds the exact path of every generated artifact; when in doubt, it wins. If a topic
is split, the halves become `b3_3a_...` and `b3_3b_...`.

---

## 6. Templates

Each template below is exact: headings, their order and their wording are fixed. Text in `<angle
brackets>` is filled in; text in *italics* is guidance that does not appear in the output.

### 6.1 Lesson (`lessons/<P>/<id>_<slug>.md`)

````markdown
# <id> <Topic title>

**Core** | **Depth** · `[RAI]` *(only if it is one)* · Part <P>, week <pw> (course week <NN>) ·
~<h> h · Lab: [`labs/<P>/<id>_<slug>.py`](../../labs/<P>/<id>_<slug>.py)

## Why it matters

*2–4 sentences: the job problem this solves, and what goes wrong in production without it.*

## Prerequisite check

*3–5 quick questions the learner should answer in their head before starting, each with the topic
id to revisit if they can't. No answers here.*

- Can you <…>? If not, revisit [<id>](<path>).

## Intuition

*The idea in plain words and a picture or analogy, before any symbol. Backend bridge callout here if
one is honest, with "Where it breaks".*

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|

*The minimal math, every step shown, every new symbol defined, every shape stated.*

## Worked example by hand

*Small numbers (2–4 examples, 2–3 features) worked step by step so the learner can follow with pen
and paper. The lab reproduces these exact numbers.*

## Key plot

![<alt text stating the takeaway>](img/<id>_<what>.png)

*One or two sentences on what to see in it.*

## From scratch in NumPy

*The from-scratch code, explained in chunks that match the lab's cells.*

```text
<expected output, pasted from the real lab run>
```

## With <library>

*The same thing with scikit-learn / PyTorch / Hugging Face, and the check that both agree.*

```text
<expected output, pasted from the real lab run>
```

## In practice

- **When to use it:** <…>
- **Failure modes:** <…, each with how you would notice it>
- **Defaults that work:** <…, with VERIFY (as of YYYY-MM) for library defaults that change>

## Exercises

*Exactly three, no solutions (they live in `answers/lessons/`). Save your work to
`my-work/<P>/<id>_<slug>.md` (or `.py`) and ask for a review.*

1. **By hand:** <…>
2. **Code:** <…>
3. **Think:** <a judgement or production question with no single right number>

## Takeaways

1. <…>
2. <…>
3. <…>
4. <…>
5. <…>

**What builds on this:** <topic ids with one phrase each>
````

### 6.2 Lab (`labs/<P>/<id>_<slug>.py`)

A `.py` file of `# %%` cells, in this order:

1. **Header** (§3.3): title, lesson link, run command and runtime, GPU need, imports, seeds, paths,
   `section()`.
2. **`GPU_SCALE` block:** every size knob (rows, epochs, model width, image size) with its CPU default
   and its full-size value, plus a comment naming the Colab/Kaggle runtime it was sized for.
3. **Data cell:** load or generate the data; print its shape and a few rows or statistics.
4. **One cell per lesson section**, in lesson order: worked example (reproducing the hand-computed
   numbers), from scratch, library, comparison check.
5. **Key plot cell:** saves `lessons/<P>/img/<id>_<what>.png`.
6. **"Try this" cell:** 2–3 commented-out variations to experiment with (not graded, not exercises).
7. **Done cell:** prints `== done ==` and the elapsed time.

A lab never contains an exercise solution.

### 6.3 Answer key (`answers/lessons/<P>/<id>_<slug>.md`)

````markdown
# Answer key: <id> <Topic title>

Lesson: [`lessons/<P>/<id>_<slug>.md`](../../../lessons/<P>/<id>_<slug>.md)

## Exercise 1 (by hand)

**Solution:** *every step, same notation as the lesson.*

**Rubric (<points> pts):**
| Criterion | Points |
|---|---|

**Likely mistakes:** *what a rusty-math engineer will get wrong, and why.*

## Exercise 2 (code)

**Solution:** *code (or a link to `<id>_<slug>_ex2.py`), run, with its real output.*

```text
<real output>
```

**Rubric (<points> pts):** *table, as above; includes correctness, shapes, tests, readability.*

**Likely mistakes:** <…>

## Exercise 3 (think)

**Strong answer:** <…>
**Weak answer:** <… and why it is weak>
**Rubric (<points> pts):** *table, as above.*

## Scoring

Total <N> pts. Pass ≥ 70%. Topic ids to revisit for each lost criterion: <…>
````

### 6.4 Weekly quiz (`quizzes/w<NN>_<pw>.md`)

````markdown
# Week <NN> quiz: <part week> <theme>

Topics: <ids>. Write your answers in `my-work/quizzes/w<NN>_<pw>.md`, then ask for a review.
No notes for Part 1. Target: ≥ 80% (checkpoint rule).

## Part 1: Self-test (15 questions)

*5 concept questions, 5 calculation or shape questions (small numbers, by hand), 5 code-reading
questions ("what does this print?", "find the bug"). Each is tagged with its topic id.*

1. `[<id>]` <question>
…
15. `[<id>]` <question>

## Part 2: Interview questions (8)

*Questions a real interviewer asks, from definition to "how would you debug…" and "what would you
do in production…". Answer each in 3–6 sentences, as you would out loud.*

1. `[<id>]` <question>
…
8. `[<id>]` <question>
````

### 6.5 Quiz answers (`answers/quizzes/w<NN>_<pw>.md`)

````markdown
# Week <NN> quiz answers: <part week> <theme>

## Part 1: Self-test

1. **<answer>.** <one-to-three-line explanation>. *Revisit: <id>.*
…

## Part 2: Interview questions

### 1. <question>

**Strong answer:** <what a hire-level candidate says>
**Weak answer:** <a plausible answer that would not pass>
**Why the weak one fails:** <…>
**Follow-up an interviewer would ask:** <…>
…

## Scoring

Self-test: 1 point each (15). Interview: 0 / 1 / 2 each against the strong answer (16).
Total 31; pass ≥ 25 (80%).
````

### 6.6 Weekly cheat sheet (`cheatsheets/w<NN>_<pw>.md`)

Fits one printed A4 page (about 60 lines). No prose paragraphs.

````markdown
# Week <NN> cheat sheet: <part week> <theme>

## Formulas
| Name | Formula | Shapes |
|---|---|---|

## Code
```python
# the 5–10 lines you will actually type, library versions as pinned
```

## Defaults and rules of thumb
- <…>

## Pitfalls
- <…>

## Interview one-liners
- **<term>:** <one sentence>
````

Part cheat sheets (`cheatsheets/PART_<P>.md`) use the same headings, 1–2 pages, and link the weekly
sheets.

### 6.7 Project brief (`projects/P<k>/BRIEF.md`)

````markdown
# P<k> <Project name>

Part <P> · starts week <NN> · due course week <NN> · GPU: no | helpful | required ·
~<h> h of project time

## The goal

*What you build and the question it answers, in 3–5 sentences.*

## Why it matters for the job

*What an interviewer learns about you from this project.*

## Data

*Dataset options with source, size, licence (VERIFY availability), and how to download into `data/`.*

## Milestones

| Milestone | Week | Deliverable | Hours |
|---|---|---|---|

## Requirements

*Numbered, checkable requirements. Includes a model card from P2 on (`[RAI]`).*

## Deliverables

*Exact files in `my-work/P<k>/`: code, README report, figures, model card.*

## How it is graded

*The rubric's criteria and weights (the rubric itself, with the scoring guide, is in
`answers/projects/P<k>/REFERENCE.md`). Pass ≥ 70%.*

## Stretch goals

- <…>

## Topics used

<ids>
````

### 6.8 Project reference notes (`answers/projects/P<k>/REFERENCE.md`)

````markdown
# P<k> reference notes and rubric

## What a strong submission looks like
## Reference approach and expected numbers *(ranges, not one number; real runs only)*
## Rubric *(criterion · weight · full marks · partial · zero)*
## Common mistakes and how to spot them in a review
## Production questions to ask in the review
````

### 6.9 Paper reading guide (`papers/<year>_<short_name>.md`)

````markdown
# <Title> (<Authors>, <year>)

Read in week <NN> with <topic id>. Link: <arXiv or venue> (VERIFY link). Time: ~<h> h.

## Why read it
## Before you start *(topic ids you need)*
## How to read it *(which sections to read closely, skim, or skip, in what order)*
## Key ideas *(4–6 bullets, in this course's notation)*
## The one equation or figure to understand
## Questions while reading *(5–8; answers in `answers/papers/`)*
## What changed since *(what later work kept, replaced or disproved; VERIFY (as of YYYY-MM))*
## Connection to the course *(topic ids and projects)*
````

Paper answer notes (`answers/papers/<year>_<short_name>.md`) answer the "Questions while reading"
in order, quoting the paper by section number rather than by long quotation.

### 6.10 Model card (`my-work/P<k>/MODEL_CARD.md`) `[RAI]`

Every project from P2 on ships one (taught in B4.6, extended in J2.5). Adapted from Mitchell et al.,
"Model Cards for Model Reporting" (2019), and covering every section the P2 brief requires. Added
2026-10-17: B4.6 and P2 refer to this template.

````markdown
# Model card: <model name>

## Model details
*Type and library versions, version or commit, training date, owner and contact.*

## Intended use
*The decision the prediction supports, who acts on it, and the population it is meant for.*

## Out-of-scope uses
*Uses the model must not be put to, and why.*

## Data
*Source and licence, time period, how rows were sampled, who is in the data and who is missing;
preprocessing and excluded columns (leakage, PII).*

## Metrics
*The metric and why it matches the costs; test score with a 95% bootstrap CI next to the baselines.*

## Decision threshold
*The threshold, the costs it was chosen from, and the confusion matrix at that threshold.*

## Metrics by group
*Per group of each audited attribute: size, base rate, selection rate, TPR, FPR, with CIs for small
groups; the gaps between groups and what they mean for the intended use.*

## Known limitations
*Where the model is known or expected to fail: subpopulations, drift, extrapolation, data quality.*

## Ethical considerations
*Harms if it is wrong, sensitive attributes and proxies, who was consulted, mitigations taken or rejected.*
````
