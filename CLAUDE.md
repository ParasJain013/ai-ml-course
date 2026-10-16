# AI/ML Course

A self-study course, generated once and then studied. Claude generates the material (lessons, labs,
quizzes, cheat sheets, project briefs, answer keys) and later reviews the learner's work.
The prompts that drive generation are in `course/PROMPTS.md`.

## Learner

- Professional software engineer: TypeScript, NestJS, Next.js, PostgreSQL, Docker, Terraform.
- Python: comfortable. Math: rusty. Rebuild linear algebra, calculus, probability and statistics
  from intuition, always paired with code.
- Goal: job-ready AI/ML engineer with complete knowledge: classical ML, deep learning, computer
  vision, NLP, LLMs, generative models, reinforcement learning, applied LLM engineering, MLOps.
- Studies 10–15 hrs/week, ~36 weeks.
- Hardware: Windows laptop, Intel Core Ultra 5 135U, 15 GB RAM, Intel integrated graphics, **no
  NVIDIA GPU**. Free Google Colab / Kaggle GPUs are available for anything that needs a GPU.

## Teaching principles

- Intuition → minimal math (define every symbol) → worked example by hand with small numbers → code.
- Build from scratch in NumPy first, then the standard library (scikit-learn / PyTorch / Hugging
  Face), and show the two agree.
- Bridge from backend engineering where the analogy is honest; say where it breaks.
- Label every topic **Core** (job-essential) or **Depth** (mastery).
- Flag fast-moving facts (model names, prices, library APIs, benchmarks) as
  "VERIFY (as of <date>)". Never invent citations, URLs or paper results.
- No flattery. Wrong is wrong; explain why.

## Layout

```
course/        outline, part syllabi, style guide, glossary, manifest.json, PROMPTS.md, qa/
lessons/<part>/  one .md per topic
labs/<part>/     one runnable .py per topic (# %% cells)
quizzes/       one per week
cheatsheets/   per week, per part, MASTER.md
projects/      one brief per project
papers/        reading guides
interview/     question bank, system-design cases
answers/       ALL solutions, answer keys and rubrics (the learner does not open this)
my-work/       the learner's solutions
reviews/       Claude's reviews of the learner's work, plus PROGRESS.md
data/          downloaded datasets (gitignored, never committed)
```

## Generation rules

- Source of truth: `course/COURSE_OUTLINE.md`, `course/PART_*.md`, `course/STYLE_GUIDE.md`,
  `course/GLOSSARY.md`. Never contradict them; if one is wrong, stop and say so.
- `course/manifest.json` lists every artifact with status `todo` / `done` / `verified`. Always
  resume from the first item that is not done, and update statuses as you go.
- Labs are `.py` files with `# %%` cells. With default settings each runs on this laptop's CPU in
  under 5 minutes (small data, few epochs); a `GPU_SCALE` block at the top documents the
  full-size settings for Colab/Kaggle.
- A lab is not done until `uv run` executes it cleanly AND the expected outputs printed in the
  lesson match the real output.
- Datasets download into `data/` (gitignored). Never commit datasets, model weights or checkpoints.
- Solutions, answer keys and rubrics go ONLY under `answers/`. Never put a solution in `lessons/`,
  `quizzes/` or `projects/`.
- Use notation from `STYLE_GUIDE.md` and terms from `GLOSSARY.md`; add new terms to the glossary.
- Cross-reference only by topic ids that exist in the manifest.
- Commit after each completed week, with a plain-English message a beginner could follow.

## Reviewing my work

When I ask for a review of a file in `my-work/`:

- Read my file, the matching key and rubric in `answers/`, and the lesson or brief it belongs to.
- Act as a strict senior reviewer: score against the rubric; list what is correct, what is wrong
  and why, methodology problems (leakage, a wrong split, a misleading metric), and what would
  break in production.
- Run my code if it is code. Do not rewrite it; point to the line and let me fix it, unless I ask
  for the solution.
- End with the 1–2 topic ids to revisit. Save the review to `reviews/<same name>.md` and append one
  line to `reviews/PROGRESS.md` (date, item, score, weak topics).
- Never show me anything from `answers/` beyond what the review needs, unless I ask for the
  solution.
