# Generation prompts

Run these in Claude Code with this folder (`D:\personal\ai-ml-course`) open, in order.
`CLAUDE.md` already carries the learner profile and the rules, so the prompts do not repeat them.
Start a fresh session for each step (and for each week in Step 3) to keep context clean.

Suggested models: the strongest available (Opus) for Steps 1–4 and reviews; a faster one
(Sonnet) is fine for Step 5.

---

## Step 1A — Course outline (once)

```text
Design the outline of my complete AI/ML course, following CLAUDE.md.

The course must cover all of these parts, in an order that respects dependencies (adjust the week counts if you have good reason, and explain why):

A. Foundations (~5 wks): Python scientific stack (NumPy, pandas, matplotlib, Jupyter); linear algebra; calculus & optimization; probability & statistics (distributions, Bayes, MLE/MAP, hypothesis testing); information theory basics (entropy, cross-entropy, KL).
B. Classical ML (~6 wks): ML workflow, data cleaning, feature engineering, leakage. Supervised: linear/logistic regression, regularization, kNN, naive Bayes, SVM, decision trees, random forests, gradient boosting. Evaluation: metrics, cross-validation, hyperparameter tuning, class imbalance, bias–variance. Unsupervised: k-means, hierarchical, DBSCAN, Gaussian mixtures/EM, PCA, t-SNE/UMAP, anomaly detection. Also: recommender systems, time-series forecasting, interpretability (SHAP).
C. Deep learning (~4 wks): perceptron, backprop, autograd from scratch; PyTorch; optimizers, LR schedules, initialization, normalization, regularization; GPU training, mixed precision, debugging training runs.
D. Computer vision (~3 wks): convolutions from scratch, CNN architectures through ResNet, transfer learning, augmentation, object detection and segmentation, Vision Transformers.
E. NLP & sequence models (~2 wks): text preprocessing, TF-IDF, word2vec, RNN/LSTM/GRU, seq2seq, attention.
F. Transformers & LLMs (~3 wks): transformer from scratch, BPE tokenization, train a small GPT, BERT vs GPT, pretraining, scaling laws, decoding strategies, instruction tuning, RLHF/DPO.
G. Generative models & multimodal (~2 wks): autoencoders, VAEs, GANs, diffusion models, CLIP and vision-language models.
H. Reinforcement learning (~2 wks): MDPs, dynamic programming, Q-learning, DQN, policy gradients, PPO, connection to RLHF.
I. Applied LLM engineering (~4 wks): LLM APIs, prompting, structured output, tool use, embeddings, RAG with Postgres + pgvector, agents, MCP, evals (mandatory), guardrails, security (prompt injection), cost, latency, caching.
J. MLOps & production (~2 wks): experiment tracking, data/model versioning, LoRA/QLoRA fine-tuning, model serving, monitoring and drift, CI for ML, deployment.
K. Responsible AI (woven in, not a separate block): fairness, privacy, safety, documentation (model cards). Say where each piece is taught.
L. Capstone & interview prep (~3 wks).

For the outline output:
- Each part: goal, weeks, and its topics with ids (B3.2 = part B, week 3 of that part, topic 2), each labelled Core or Depth, each with one line saying what it is.
- 10–12 escalating portfolio projects placed in the parts where they belong, including at least: an EDA + statistics report; an end-to-end tabular ML model; an unsupervised project; a neural net + autograd from scratch; a CV transfer-learning project; a small GPT from scratch; a generative model; an RL agent in Gymnasium; a RAG system with an eval suite; a tool-using agent with evals; a LoRA fine-tune versus a prompted baseline; and a capstone deployed end to end.
- 6–10 landmark papers to read across the course (title, authors, year; mark "VERIFY" where unsure), each placed at the week it fits.
- A dependency map of the parts; 6 checkpoints with pass criteria; a "fast path" listing which Core topics make me interview-ready by ~week 24; and what to cut if I fall 3 weeks behind.

Before writing, list your assumptions in 3–5 bullets. Don't write the weekly detail yet.
Save to course/COURSE_OUTLINE.md and commit.
```

## Step 1B — Detail one part (once per part: A, B, C … L)

```text
Using course/COURSE_OUTLINE.md, write the detailed syllabus for Part {{LETTER}}: {{PART_NAME}}.

For each week in this part:
- Learning objectives as testable "I can…" statements.
- Topics in teaching order, with ids, Core/Depth labels, and estimated hours.
- Hours split between reading, coding and project.
- Labs: the specific from-scratch implementations I'll write.
- The project milestone due this week, if any.
- Prerequisite topic ids (including from earlier parts).
- 1–3 optional free resources (courses, docs, papers), each marked "VERIFY link".
- GPU needed? (no / helpful / required) and how to do it on free Colab or Kaggle.

End with a short "you're ready for the next part when…" self-check.
Save to course/PART_{{LETTER}}.md and commit.
```

## Step 2 — Consistency foundations (once)

```text
Read course/COURSE_OUTLINE.md and every course/PART_*.md. Don't write course content yet. Create:

1. course/STYLE_GUIDE.md: math notation (scalars, vectors, matrices, always state shapes like (n, d)), code conventions (seeds, type hints, plotting style), the exact templates for a lesson, lab, answer key, weekly quiz, cheat sheet, project brief and paper reading guide, and file naming.
   Lesson template: why it matters → prerequisite check → intuition, math, worked example → one key plot → from-scratch code then library code → in practice (when to use, failure modes, defaults) → 3 exercises without solutions → 5 takeaways + what builds on this.
2. course/GLOSSARY.md: every term in the outline, a one-line definition, and the topic id where it's taught.
3. course/manifest.json: one entry per artifact. Lesson + lab + answer key per topic; quiz + quiz answers + cheat sheet per week; cheat sheet per part; brief + reference notes per project; reading guide per paper. Each entry has id, type, path, week, depends_on, status "todo".
4. Add every library the course uses to pyproject.toml with pinned versions (look up current versions on the web; use the CPU build of PyTorch); run `uv sync` to prove it installs.

Report the artifact counts by type and commit.
```

## Step 3 — Generate one week (repeat until the manifest is done)

```text
Generate the next week in course/manifest.json that isn't done.

For each topic in that week, in order:
1. The lesson in lessons/, following the template exactly. Exercises have no solutions.
2. The lab in labs/. Run it, then paste the REAL outputs into the lesson's expected-output blocks.
3. The answer key in answers/lessons/: worked solutions for every exercise (run the code ones), a grading rubric, and the mistakes I'm likely to make.
Then for the week as a whole:
4. quizzes/<week>.md: 15 self-test questions + 8 interview questions. Answers in answers/quizzes/<week>.md, with a strong and a weak answer for each interview question.
5. cheatsheets/<week>.md: one printed page.
6. Any project brief or paper reading guide the manifest places in this week; reference notes go to answers/projects/ or answers/papers/.

Before marking anything done, reread each lesson as me (rusty math, strong engineer): can every step be followed without a leap? Fix the gaps. If a topic is too big, split it and update the manifest.
Finally, update the glossary and manifest, re-run all labs for the week, and commit.
```

## Step 4 — QA pass (once per part, after all its weeks are done)

```text
QA pass on Part {{X}}. Act as a skeptical reviewer, not the author.
- Run every lab in labs/{{X}}/ in a clean environment; fix failures.
- Check that every expected output in the lessons matches the real lab output.
- Re-derive every worked example by hand; check the shapes/dimensions of every formula.
- Check that prerequisites are taught before they're used and that every cross-reference id exists.
- Check that the answer keys actually solve the exercises (run the code answers).
- Check that no solution leaked outside answers/.
- Check notation and structure against STYLE_GUIDE.md.
Write what you found and fixed to course/qa/PART_{{X}}.md, set the part's manifest items to "verified", and commit.
```

## Step 5 — Revision material (once, at the end)

```text
Every part is verified. Create:
- cheatsheets/PART_<X>.md for each part (1–2 pages each).
- cheatsheets/MASTER.md: the whole course in ~10 pages, for interview revision.
- interview/QUESTION_BANK.md: every interview question from all quizzes, deduplicated and grouped by area; answers in answers/interview/.
- interview/ML_SYSTEM_DESIGN.md: 10 ML/LLM system-design cases; answer outlines in answers/interview/.
- Update README.md: how to study the course week by week, with a link to everything and the fast path to interview-readiness.
Commit.
```

---

# While studying

## Review (after each exercise, quiz or project milestone)

```text
Review my-work/{{file name}}
```

`CLAUDE.md` tells Claude how to review, where to save it, and how to log progress.

## Refresh (once, around week 24, before Part I)

```text
Refresh Parts I and J. Search the web for the current versions of every library, API and model these parts use; update the lessons, labs, answer keys and pinned versions where anything has changed; re-run every lab; note what changed in course/qa/REFRESH.md; commit.
```

## Interactive quiz (optional, any time)

```text
Quiz me on {{topic ids}} one question at a time. Grade each answer, explain, and adapt the difficulty. After 10 questions, give my score and weak topic ids, and append them to reviews/PROGRESS.md.
```

## Mock interview (final weeks)

```text
Run a mock AI/ML engineer interview: {{ML fundamentals / deep learning / LLM systems / ML system design / project deep-dive on <project>}}, about 45 minutes.

Ask one question at a time and wait for my answer. Ask follow-ups like a real interviewer, probing where my answer is vague. Don't teach during the interview. At the end, give a hiring-style assessment: a score for each area, strongest and weakest answers, what a strong candidate would have said, and the topic ids to study before the next mock. Save it to reviews/mock-<date>.md.
```
