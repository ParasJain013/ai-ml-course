# Part L. Capstone & interview prep — detailed syllabus

Course weeks 34–36. Weekly detail for Part L of [`COURSE_OUTLINE.md`](COURSE_OUTLINE.md), which stays
the source of truth: the topic ids, Core labels, project P12 and checkpoint CP6 below are taken from it
unchanged. Written 2026-10-16.

**Goal:** ship one end-to-end system you can defend in an interview, and rehearse the interview loop.

Part L teaches little that is new. It turns 33 weeks of topics into two things an employer can check:
a deployed system with evidence that it works (P12), and your ability to explain, design and code
under interview conditions. The order is deliberate: L1 scopes the capstone and builds its eval
harness before the system, the way Part I taught you to work; L2 ships it and, while it runs, teaches
the system-design interview, which asks you to design exactly this kind of system on a whiteboard; L3
rehearses the rest of the loop (coding, fundamentals, the project deep-dive) and closes with CP6.

## At a glance

| Week | Course wk | Theme | Topics | Hours | Project | GPU |
|---|---|---|---|---|---|---|
| L1 | 34 | Capstone build | L1.1–L1.2 | 14 | P12 M1 (design doc), M2 (eval harness + first working version) | depends on capstone |
| L2 | 35 | Ship it | L2.1–L2.2 | 14 | P12 M3 (deployed, monitored, documented); system-design mock | depends on capstone |
| L3 | 36 | Interview loop | L3.3, L3.1, L3.2 | 14 | **P12 due**; **CP6** | no |

About 42 hours in total, at the top of your 10–15 hour budget, because P12 is the largest project of
the course. Each week's hours include about 1 hour for the weekly quiz and cheat sheet (2 hours in L3,
which also holds the CP6 self-test). Topic hours cover reading the lesson and doing its lab and
exercises.

**How the hours are counted.** *Reading* is working through the lessons, the system-design cases and
the revision material. *Coding* is the labs, the timed coding drills and the code exercises.
*Project* is P12. *Mocks* (L2 and L3) are counted under the topic they belong to, L2.2 or L3.2.

**P12's due date.** The outline lists P12 as "L1–L2, wk 36": it is built in L1 and L2 and submitted in
week 36, after L3.3 adds the README, demo and walkthrough. This syllabus follows that reading.

**Teaching order.** In L3 the topics are not taught in id order: L3.3 (portfolio and storytelling)
comes first, because P12 is due that week and the project deep-dive mock in L3.2 needs your
2-minute walkthrough. L3.1 (coding drills) comes next and L3.2 (review and mocks) closes the week with
CP6. The ids stay as the outline defines them.

**If a week runs long,** there are no Depth topics to drop in Part L. The outline's catch-up plan
says: shorten L3 to one mock and fold L3.3 into L2.1. Never cut P12's evals in CI, its monitoring or
its model card: they are what separates a capstone from a demo, and CP6 checks them. Cut scope from
the product instead (fewer features, one user flow).

**On the fast path,** L2.2 and L3.1–L3.3 are studied in fast-path week 24 with P7 deployed as a mini
capstone (see the outline). When you come back to the full course, P12 replaces that mini capstone.

**Libraries this part needs** (Step 2 pins the versions): nothing new for the drills (`numpy` only, by
design). The capstone uses what its parts already use, typically `fastapi` and `uvicorn` (J2.1),
`pydantic` (I1.3), `psycopg` and `pgvector` (I2.2), `scikit-learn` or `lightgbm` (B), your LLM
provider's SDK (I1.1), `mlflow` (J1.1), `pytest` and GitHub Actions (J2.3), Docker (J2.4), and
`prometheus-client` or plain structured logs for monitoring (J2.2, I4.5). Add any new library to
`pyproject.toml` with a pinned version; LLM SDKs and model names move fast: VERIFY (as of 2026-10).

**Data and models:** whatever your capstone needs, downloaded into `data/` and the Hugging Face cache,
never committed. Commit the golden eval set (it is small, hand-labelled and the most valuable file in
the repo) unless it contains personal data, in which case commit a redacted version (B1.2, I4.1).

**Money:** the LLM API budget for the whole course is tens of US dollars (VERIFY (as of 2026-10)). Set a
hard spend limit with your provider before the endpoint goes public, cap requests per client, and keep
the local open-weights model (F3.4) as the fallback. Cloud free tiers for hosting a container change
often: VERIFY (as of 2026-10) before you pick one.

**Hardware:** every lab and drill runs on the laptop CPU. Whether P12 needs a GPU depends on the
capstone you choose; "Running Part L on a free GPU" at the end says how to keep any GPU work inside a
free Colab or Kaggle session and serve the result on CPU.

---

## L1 · Capstone build (course week 34)

**Learning objectives**

- I can write a 2–4 page design doc for an ML system that states the user and the problem, one
  primary metric tied to the real cost of errors (B4.2), the data and its leakage risks (B1.4), a
  baseline it must beat, the architecture, what could go wrong, and what is out of scope.
- I can choose between a rules baseline, a classical model, prompting, RAG and fine-tuning for my
  problem and defend the choice with the J1.4 framework and numbers from P7–P9, not with taste.
- I can size the work: estimate cost per request and latency from token counts and model prices (I4.3),
  and cut scope until the build fits about 15 hours of project time.
- I can build an eval harness before the system: a versioned golden set of at least 50 labelled cases,
  a runner that scores any implementation against it, per-slice results, and a pass/fail threshold
  that a CI job enforces (I2.4, I3.3, J2.3).
- I can get a first end-to-end version working behind that harness, record its score next to the
  baseline's in an experiment tracker (J1.1), and say which failure category to fix first from the
  error analysis, not from a hunch.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| L1.1 | Scoping and a design doc: problem, metric, data, baseline, architecture, risks | Core | 2.5 |
| L1.2 | Building the system with its eval harness first | Core | 3 |

**Hours:** reading 2 · coding 3.5 · project 7.5 · quiz and cheat sheet 1 · **total 14**

**Choosing the capstone.** The outline asks for a system that uses at least two parts. Pick one you
could explain to a hiring manager in one sentence, and that reuses code you already have: there are
only about 15 hours of project time across L1 and L2. Three shapes that fit, with what each needs:

| Shape | Example | Parts used | GPU |
|---|---|---|---|
| LLM product with evals (**default**) | A question-answering or triage assistant over a document set you care about, with tools and citations, grown from P7 and P8 | I, J, plus B or F for one component | no |
| Tabular model plus LLM layer | A gradient-boosted model (from P2) that scores records, with an LLM that explains each decision in plain language and is checked for faithfulness | B, I, J | no |
| Fine-tuned specialist | A LoRA model (from P9) for one narrow task, served quantized on CPU, compared with the prompted baseline in production | F, J, I | required for training only |

The default is the LLM product because it is the most common shape in applied AI engineer roles,
it needs no GPU, and its evals carry straight over from P7 and P8. Pick a domain where you can label
the golden set yourself in about two hours; a domain you know from work (backend logs, API docs,
support tickets with personal data removed) beats a famous dataset everyone else uses.

**Labs**

- **L1.1:** a design-doc template filled in for a worked example (a RAG assistant over a public
  docs site), then a back-of-envelope sizing script. The script takes requests per day, average input
  and output tokens, cache hit rate and per-token prices (passed in as arguments, because prices
  change: VERIFY (as of 2026-10)) and prints monthly cost, p50/p95 latency estimates from measured
  tokens per second, and the break-even point where self-hosting a small open model beats the API.
  Run it for your own capstone and paste the numbers into your design doc. Bridge: this is the same
  capacity-planning doc you would write before adding a service to a backend; it breaks in one place,
  because quality is a measured distribution, not a requirement you can meet by design, so the doc
  must name the eval that decides whether the system works.
- **L1.2:** an eval harness from scratch, with no eval framework. A `GoldenCase` Pydantic model (input,
  expected output or reference facts, slice tags such as "needs two documents" or "should refuse"); a
  JSONL golden set with a version hash; a runner that takes any `predict(input) -> output` callable,
  runs the cases with a concurrency limit and retries (I1.1), and stores outputs; scorers for exact
  match, a numeric tolerance, retrieval recall@k and MRR (I2.4) and an LLM-as-judge scorer with a
  fixed rubric and a position-swap check for its bias; a report that prints the score per slice with a
  bootstrap 95% CI (A6.2), so a 2-point "improvement" on 50 cases is not mistaken for progress; and a
  `pytest` entry point that fails when the score drops below a threshold or below the last release by
  more than the CI's width. Test it on two toy systems, a constant baseline and a keyword-matching
  one, and check that the harness ranks them correctly. CPU default: the toy systems need no API; the
  LLM judge can use the local open model from F3.4.

**Project milestones**

- **P12 M1, design doc (3 h).** `projects/P12` has the brief; your doc goes in `my-work/P12/DESIGN.md`.
  It names the primary metric and its target, the baseline, the architecture as a diagram, the cost
  and latency estimates from the L1.1 script, the top five risks with a mitigation each (include at
  least one `[RAI]` risk: a fairness slice, personal data, or prompt injection), and a cut list of
  features you will drop first if time runs out. Ask for a review (`Review my-work/P12/DESIGN.md`)
  before you write code; changing the metric later is expensive.
- **P12 M2, eval harness and first version (4.5 h).** Label the golden set (at least 50 cases, with
  10 or more adversarial or out-of-scope ones), wire your L1.2 harness into the capstone repo with a
  GitHub Actions job (J2.3), score the baseline, then build the simplest version of the system that
  runs end to end and score it. Log both runs in MLflow (J1.1). End the week with an error analysis:
  the 10 worst cases grouped by cause.

**Prerequisites:** J1.4 (prompt versus RAG versus fine-tune), I2.4 (RAG evals and LLM-as-judge
biases), I3.3 (agent evals and regression suites in CI), J2.3 (CI gates), J1.1–J1.2 (tracking and
versioning), I4.3 (token cost and latency), I4.2 (prompt injection, for the risk list), I1.1 and I1.3
(API clients, retries, structured output); B1.1 (baselines and splits), B1.4 (leakage), B4.2 (metrics
that match the cost of errors), A6.2 (bootstrap CIs); P7, P8 and P9 (code and numbers you will reuse).
Within the week: L1.2 needs L1.1's metric.

**Resources** (optional)

- Google, *Rules of Machine Learning: Best Practices for ML Engineering* (Martin Zinkevich), on
  starting with a simple baseline and a metric you trust:
  https://developers.google.com/machine-learning/guides/rules-of-ml (VERIFY link)
- Hamel Husain, "Your AI Product Needs Evals", on building evals before features:
  https://hamel.dev/blog/posts/evals/ (VERIFY link)
- Sculley et al., "Hidden Technical Debt in Machine Learning Systems" (NeurIPS 2015), for the risk
  section of your design doc: https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
  (VERIFY link)

**GPU:** depends on the capstone. The labs and the default capstone need none. A fine-tuned
specialist needs a free GPU for the LoRA training run only (see the end of this file); do that run in
L1 so L2 can be spent on serving it.

---

## L2 · Ship it (course week 35)

**Learning objectives**

- I can deploy my system as a Docker image to a public HTTPS endpoint with a health check, an API key
  or rate limit, a spend cap and a one-command rollback to the previous image (J2.4).
- I can show a monitoring dashboard with request rate, error rate, p50/p95 latency, cost per day and
  one quality signal (an online eval, user feedback or a drift statistic), and say what alert fires
  when each crosses its threshold (J2.2, I4.5).
- I can write a model card for my system that a reviewer could use to decide whether to deploy it:
  intended use, out-of-scope uses, eval results per slice with CIs, known failure modes, `[RAI]`
  fairness and privacy notes, and how it is monitored (J2.5).
- I can write a cost report from real logged usage: cost per request, the share from each component,
  and the effect of caching and model routing (I4.3, I4.4), with a projection at 10× traffic.
- I can run an ML system design interview from a one-line prompt in 45 minutes using a fixed
  framework: clarify the goal and constraints, define the metric (offline and online), the data and
  labels, a baseline, the model or LLM pipeline, serving and its latency and cost budget, evaluation
  and rollout, monitoring and failure modes.
- I can do the back-of-envelope numbers in that interview without a calculator: QPS from daily
  users, storage for an embedding index, tokens and cost per day, and KV-cache memory per request
  (F2.4).

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| L2.1 | Deploy, monitor and document: public endpoint, dashboards, model card, cost report | Core | 3 |
| L2.2 | ML system design interviews: the framework, worked through on several cases | Core | 3.5 |

**Hours:** reading 2.5 · coding 3 · mock 1 · project 6.5 · quiz and cheat sheet 1 · **total 14**

L2.2's 3.5 hours include the 1-hour system-design mock (45 minutes plus reading the assessment) at the end of the week.

**Labs**

- **L2.1:** the operating pieces, built small and from scratch before you use them on P12. A FastAPI
  middleware that logs one structured JSON line per request (request id, latency, status, model,
  input and output tokens, cost, cache hit, a hash of the prompt instead of its text, as I4.5 says);
  a script that turns those logs into the dashboard numbers (rates, latency percentiles, cost per
  day) and a PSI drift check on one input feature or on input length (J2.2) against the golden set's
  distribution; a sampled online eval that re-scores 5% of live traffic with your L1.2 judge; and a
  model-card generator that fills a Markdown template from the latest harness report, so the card's
  numbers can never drift from the eval's. CPU default: replay 1,000 synthetic requests through a
  stub model in under a minute. Bridge: this is the observability you already run on backend services
  (structured logs, RED metrics, alerts); it breaks because a model can return 200 OK, fast and cheap,
  and still be wrong, so the quality signal is the metric that matters most and the only one your
  APM tool cannot give you.
- **L2.2:** a back-of-envelope calculator and three worked cases. The calculator holds the handful of
  numbers you should know cold (seconds per day, bytes per float32 and float16, tokens per English
  word as a rule of thumb, the KV-cache formula from F2.4) and checks your mental arithmetic. The
  worked cases, each solved with the framework from `interview/ML_SYSTEM_DESIGN.md`: a
  retrieve-then-rank feed or recommender (B6.1–B6.2), a fraud or anomaly detector with delayed labels
  and class imbalance (B4.4, B5.6, J2.2), and an LLM support assistant with RAG, tools, guardrails and
  a cost budget (I2–I4). For each, write the 45-minute answer as a one-page outline first, then
  compare it with the answer outline in `answers/interview/` and note what you missed.

**Project milestone: P12 M3, deployed, monitored and documented (6.5 h).** Fix the top error category
from M2 and show the score change with its CI. Deploy the Docker image to the public endpoint, with
the eval job as a merge gate (a pull request that lowers the score cannot merge) and a smoke test
that runs against the live endpoint after each deploy. Put the L2.1 logging and dashboard in place,
generate the model card, and write the cost report from at least a few hundred real or replayed
requests. Add a security test set of at least 10 prompt-injection cases (I4.2) to the harness and
report how many succeed. Ask for a review of the deployment and the model card before L3.

**System-design mock.** At the end of the week, run the mock interview prompt from `PROMPTS.md` with
"ML system design" and a case you have not seen, for 45 minutes, and save it to
`reviews/mock-<date>.md`. CP6 needs this mock at "lean hire" or better; if it falls short, repeat it
in L3.2 with a different case.

**Prerequisites:** L1.1–L1.2 and P12 M1–M2; J2.1 (serving with FastAPI), J2.2 (drift and alerting),
J2.4 (Docker, a cloud target, canary and rollback), J2.5 (model cards and datasheets), I4.3–I4.5 (cost,
caching, observability), I4.1–I4.2 (guardrails and prompt injection); for L2.2, B6.1–B6.2
(recommenders, retrieve-then-rank), B4.1 and B4.4 (metrics and imbalance), B5.6 (anomaly detection),
I2.1–I2.3 (embeddings, pgvector, the RAG pipeline), F2.4 (KV cache), B4.6 (fairness, for the model
card). Within the week: L2.2 can be read in parallel with L2.1; the mock needs all of L2.2.

**Resources** (optional)

- Chip Huyen, *Machine Learning Interviews Book* (free online), the ML system design chapters:
  https://huyenchip.com/ml-interviews-book/ (VERIFY link)
- Mitchell et al., "Model Cards for Model Reporting" (2019): https://arxiv.org/abs/1810.03993
  (VERIFY link)
- Eugene Yan, "Patterns for Building LLM-based Systems & Products", for the LLM case:
  https://eugeneyan.com/writing/llm-patterns/ (VERIFY link)

**GPU:** depends on the capstone, and is never needed for serving. Serve the default capstone and the
tabular shape on CPU. Serve a fine-tuned specialist as a merged, quantized model on CPU (J2.1); if its
latency is too high on the free tier, say so in the cost report and compare with the prompted
baseline, which is itself a finding interviewers like.

---

## L3 · Interview loop (course week 36)

**Learning objectives**

- I can explain P12 in 2 minutes (problem, approach, the number that shows it works, the hardest
  problem and what I would do next) and in 10 minutes with the architecture diagram, and answer
  "why not X?" for each major choice with evidence from my evals.
- I can write each project README so that a reviewer who reads only its first screen knows what the
  project does, the headline result with its baseline, and how to run it in one command.
- I can rewrite each portfolio project as one or two resume bullets that state what I built, the
  measured result and the scale, without claiming more than the evidence shows.
- I can implement, from scratch in NumPy and under the time limits in the drill list below, the core
  algorithms interviewers ask for, with correct shapes and a test that checks them against
  scikit-learn or PyTorch.
- I can answer ML fundamentals questions (bias–variance, regularization, metrics, leakage,
  backprop, optimizers, attention) and LLM-systems questions (RAG, evals, agents, cost, security)
  out loud in under 3 minutes each, at the level of a "strong answer" in the quiz keys.
- I can pass an ML-fundamentals mock interview at "lean hire" or better (CP6), and name the topic ids
  my mocks flagged and what I did about each.

**Topics** (in teaching order)

| Id | Topic | Level | Hours |
|---|---|---|---|
| L3.3 | Portfolio and project storytelling: the README, the demo, the 2-minute walkthrough | Core | 2 |
| L3.1 | ML coding interviews: implementing core algorithms from scratch under time pressure | Core | 3.5 |
| L3.2 | Fundamentals and LLM-systems review with mock interviews | Core | 4.5 |

**Hours:** reading 3 · coding 4 · mocks 3 · project 2 · quiz, cheat sheet and CP6 2 · **total 14**

The 3 mock hours (three mocks of about an hour each) are counted in L3.2's 4.5. The coding drills are
tight at 3.5 hours: if they run over, redo only the slowest drill at the end of the week, not three.

**Labs**

- **L3.3:** a portfolio check script that walks your project folders and reports, for each of P1–P12,
  whether the README has a one-line summary, a headline metric next to its baseline, a run command, a
  model card link (P2 onward, per Part K) and a results figure, and whether the run command actually
  works from a clean checkout. Then the storytelling drills: write the 2-minute P12 walkthrough, record
  yourself giving it, and cut it until it fits; write a 1-minute version for P7 or P8 and one for a
  from-scratch project (P4 or P6); and prepare one STAR-format story (situation, task, action, result)
  for each of "a model that failed", "a trade-off you made under constraints" and "a time evals
  changed your mind", drawn from your projects, not invented. Bridge: this is the same as presenting
  a backend design review; it breaks because ML interviewers probe how you know it works, so every
  claim needs the eval number and the baseline it beat.
- **L3.1:** a timed drill file of from-scratch implementations, each a stub with hidden tests that
  compare against the library version, run with a timer. The time limits are targets, not the
  checkpoint (except where CP2 and CP4 set them):

  | Drill | Target | From |
  |---|---|---|
  | Logistic regression with gradient descent, plus log-loss | 30 min (CP2) | B2.1 |
  | k-means with k-means++ initialization | 25 min | B5.1 |
  | Decision-tree best split with Gini impurity | 25 min | B3.1 |
  | kNN classifier, vectorized distances (no Python loop over pairs) | 15 min | B2.3 |
  | PCA via SVD, with explained variance | 15 min | B5.4 |
  | ROC-AUC from scores and labels, without scikit-learn | 20 min | B4.1 |
  | Stable softmax and cross-entropy, and their gradient | 15 min | C1.4 |
  | A 2-layer MLP forward and backward pass, checked with finite differences | 35 min | C1.2–C1.3 |
  | Adam update step | 10 min | C3.1 |
  | Causal multi-head self-attention with shapes asserted | 45 min (CP4) | F1.1 |
  | Top-k and top-p sampling from logits | 15 min | F2.4 |
  | One BPE merge step on a toy corpus | 20 min | F1.3 |

  Do each drill once cold, note your time and the bug that cost you most, then redo the slowest three
  at the end of the week. Talk out loud while you code, as you would in the interview. CPU default:
  every test runs in seconds on small arrays.

**Project milestone: P12 due, capstone deployed end to end (2 h).** Finish and submit P12 (brief in
`projects/P12`, rubric in `answers/projects/`). The submission is the repository with the design doc,
the eval harness running in CI, the live endpoint, the dashboard (a screenshot is fine if it is
private), the model card, the cost report and the README from L3.3, plus a short demo (a recording or
a scripted run of three requests, one of them an injection attempt that the system handles). Then
`Review my-work/P12` and fix what the review flags before CP6.

**Mocks (L3.2).** Revise from `cheatsheets/MASTER.md` and the questions your past reviews flagged in
`reviews/PROGRESS.md`, using `interview/QUESTION_BANK.md` for self-tests. Then run the mock interview
prompt from `PROMPTS.md` three times, with a day between them if you can: "ML fundamentals" (required
for CP6), "project deep-dive on P12", and "LLM systems" or a second "ML system design" if the L2 mock
was below "lean hire". Save each to `reviews/mock-<date>.md`. Between mocks, study only the topic ids
the last mock named; one or two topics fixed properly beat a full reread.

**Checkpoint CP6 (end of Part L).** From the outline, passed only when every criterion holds:

- P12 is deployed, monitored, documented, and scores 70% or more on its rubric.
- One ML-fundamentals mock and one system-design mock are at "lean hire" or better.

If a criterion fails, take one catch-up week on the topic ids your reviews flagged, as the outline's
checkpoint rule says, then rerun only the failed mock or fix only the failed rubric lines.

**Prerequisites:** P12 M1–M3 and L2.1–L2.2; for L3.1, B2.1, B2.3, B3.1, B4.1, B5.1, B5.4, C1.2–C1.4,
C3.1, F1.1, F1.3, F2.4 (each drill's source topic, as listed); for L3.2, every checkpoint CP1–CP5 and
the quiz keys' strong answers; for L3.3, L2.1 (model card and cost report) and P1–P11. Within the
week: L3.2's project deep-dive needs L3.3's walkthrough.

**Resources** (optional)

- Chip Huyen, *Machine Learning Interviews Book*, the chapters on the interview process and ML
  fundamentals questions: https://huyenchip.com/ml-interviews-book/ (VERIFY link)
- *Tech Interview Handbook*, the behavioral interview section (STAR stories) and resume guide:
  https://www.techinterviewhandbook.org/ (VERIFY link)

**GPU:** no. The drills and mocks run on the CPU; P12 is already deployed by this week.

---

## Running Part L on a free GPU

Only the fine-tuned specialist capstone needs a GPU, and only for training. Everything else in Part L
runs on the laptop.

- **Plan the run in L1.** Reuse your P9 notebook: the same small open model, LoRA or QLoRA settings
  sized for one free T4 session, and the same eval set. Free-tier quotas and session limits change:
  VERIFY (as of 2026-10). Kaggle's weekly GPU quota and longer sessions make it the better choice for a
  training run; Colab is fine for a short re-run.
- **Make the run restartable.** Save adapter checkpoints to `/kaggle/working` (or a mounted drive on
  Colab) every few hundred steps, so a dropped session costs minutes, not the run.
- **Bring back only the adapter and the numbers.** Download the LoRA adapter (megabytes, not the base
  model), the eval results and the training curves. Merge and quantize on the laptop for CPU serving
  (J1.3, J2.1). Adapters, merged weights and checkpoints stay out of git, as `CLAUDE.md` says; store
  them in a model registry (J1.2) or a release asset and record the version in the model card.
- **Serve on CPU.** If the free hosting tier cannot hold the model in memory, deploy the prompted
  baseline as the public endpoint and document the fine-tuned model's offline results and local
  latency. That is an honest result, not a failed capstone.

---

## You're ready to apply for AI engineer roles when…

Part L is the last part. This self-check covers CP6 and what a hiring loop for an applied ML or AI
engineer actually tests. Answer each item without notes; the answers belong in your reviews, not here.

1. CP6 holds: P12 is live, monitored and documented at 70% or more, and your ML-fundamentals and
   system-design mocks are at "lean hire" or better.
2. You can give the 2-minute walkthrough of P12 without notes, and answer "how do you know it works?"
   with the eval, its size, the baseline and the confidence interval.
3. Every portfolio project you list has a README whose first screen states what it does, the headline
   number against a baseline, and a run command that works from a clean checkout.
4. Your resume has one or two bullets per listed project with a measured result, and nothing on it you
   could not defend under a follow-up question.
5. You can implement logistic regression in 30 minutes and causal self-attention in 45 (CP2 and CP4
   again, now cold), and at least eight of the twelve L3.1 drills within their targets.
6. You can run a system-design answer end to end in 45 minutes, with the metric, a baseline, the
   numbers and the failure modes, for both a classical ML case and an LLM case.
7. You can explain, with numbers from your own projects, when you would prompt, use RAG or fine-tune
   (P7–P9), and what evals, guardrails and monitoring you would put around each.
8. You have three true STAR stories from your projects and can tell each in under 2 minutes.
9. You have a written list of your weak topic ids from `reviews/PROGRESS.md`, and a plan to keep
   studying them while you apply.

If items 1–5 hold, start applying now, and keep doing one mock a week; interviews themselves are the
best practice left, and waiting until everything feels perfect costs more than it saves. If two or
more of items 1–5 fail, take the catch-up week on the flagged topic ids first. Before you apply, run
the Refresh prompt from `PROMPTS.md` once more on Parts I and J if more than a few months have passed:
model names, prices and LLM library APIs in your projects date quickly, and an interviewer will notice.
