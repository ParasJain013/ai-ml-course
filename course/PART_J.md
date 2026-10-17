# Part J. MLOps & production — detailed syllabus

Course weeks 28–29. Weekly detail for Part J of [`COURSE_OUTLINE.md`](COURSE_OUTLINE.md), which stays
the source of truth: the topic ids, Core/Depth labels, project P9, checkpoint CP5 and the LoRA paper
below are taken from it unchanged. Written 2026-10-16.

**Goal:** take any model from notebook to a monitored, versioned, tested service, and know when
fine-tuning beats prompting.

Part J is the part where your backend experience pays the most, and also where it can mislead you the
most. Docker, CI, deployments, health checks and rollbacks are things you already do; what is new is
that a model can return `200 OK` with the wrong answer, that its inputs drift away from the data it was
trained on without any code changing, and that "the build" includes a dataset and a training run that
must be reproducible. J1 makes experiments reproducible and comparable (tracking, versioning, a
registry), then teaches LoRA and QLoRA and uses them for P9, the fine-tune-versus-prompt comparison.
J2 turns a model into a service and operates it: serving, drift monitoring, CI gates, deployment with
canaries and rollback, and the documentation and privacy work that goes with shipping.

Every topic in Part J is Core. There is no Depth topic to drop, so if time runs short use the outline's
catch-up plan, which cuts elsewhere.

## At a glance

| Week | Course wk | Theme | Topics | Hours | Project | GPU |
|---|---|---|---|---|---|---|
| J1 | 28 | Tracking, versioning, fine-tuning | J1.1–J1.4 | 13.5 | **P9 due** | required (P9, J1.3 `GPU_SCALE`) |
| J2 | 29 | Serving & operating | J2.1–J2.5 | 14 | P7/P8 CI gates, P9 review fixes; **CP5** | no (helpful for J2.1's GPU inference server) |

About 27.5 hours in total. Each week's hours include about 1 hour for the weekly quiz and cheat sheet;
J2 adds about 1.5 hours for the CP5 self-test and the LLM-systems mock interview that CP5 requires.
Topic hours cover reading the lesson and doing its lab and exercises.

**How the hours are counted.** *Reading* is working through the lessons and the paper, including the
worked examples by hand. *Coding* is the labs and the code exercises. *Project* is P9 in J1 and the
P7/P8/P9 CP5 work in J2.

**Teaching order** follows the ids in both weeks. In J1, tracking (J1.1) and versioning (J1.2) come
first because every P9 run is logged and every P9 eval set is a versioned snapshot. In J2, CI (J2.3)
comes before deployment (J2.4) because the deploy pipeline is the CI pipeline with one more job.

**On the fast path,** the outline keeps J1.3–J1.4 and J2.1–J2.4 and deploys P7 as a mini capstone.
Read the J1.1 and J1.2 lessons even then: the labs in J1.3 and later log to MLflow.

**Libraries this part needs** (Step 2 pins the versions): `mlflow` (J1.1, J1.2), `dvc` (J1.2),
`torch` (CPU build on the laptop), `transformers`, `datasets`, `peft` and `trl` (J1.3, P9),
`bitsandbytes` (QLoRA, GPU only), `fastapi`, `uvicorn` and `httpx` (J2.1, J2.4), `onnx` and
`onnxruntime` (J2.1), `scipy` (the KS test in J2.2), `pytest` (J2.3), plus `pandas`, `numpy`,
`scikit-learn` and `matplotlib` from earlier parts. Optionally `llama-cpp-python` or Ollama for a CPU
LLM inference server in J2.1, and `vllm` on a free GPU. MLflow's registry API (stages versus aliases),
`peft` and `trl` argument names, `bitsandbytes` platform support and GitHub Actions runner images all
change between versions: VERIFY (as of 2026-10). These are exactly the facts the week-24 refresh in
`PROMPTS.md` re-checks.

**Models** (downloaded into the Hugging Face cache, never committed): GPT-2 small (124M) or your own
F1.4 GPT for the CPU LoRA lab; for P9, a small open instruct model in the 0.5B–1.5B range (for example
from the Qwen2.5 or SmolLM2 families) for LoRA/QLoRA on a free GPU, and the same model prompted as the
baseline; the paid API model from Part I as a second, stronger prompted baseline; a small quantized
GGUF model for the CPU inference server in J2.1. Names, sizes and licences change quickly: VERIFY (as
of 2026-10). Adapters are small (a few MB) but they are still model weights: keep them in the MLflow
artifact store or on Google Drive, never in git.

**Datasets** (all download into `data/`, never committed): your P2 tabular dataset again for J1.1,
J1.2, J2.1 and J2.2 (if it has no time column, J2.2 injects drift synthetically); the P9 task dataset,
a public dataset for one narrow task with a few thousand labelled rows at most (for example extracting
fields from short texts into JSON, or classifying support tickets into a fixed label set), picked at
P9 M0; the P7 golden set and P8 task suite from Part I for the CI gates in J2.3. VERIFY each source,
its licence and its contents when the lab is generated.

**Accounts:** a GitHub repository for J2.3 and J2.4 (a fork of your project repo or a new repo for
P9; GitHub Actions minutes on public repos are free, private-repo limits: VERIFY (as of 2026-10)); a
free Hugging Face account for the default deployment target in J2.4. A cloud account (for example
Google Cloud Run with Terraform) is optional and may need a billing card even inside a free tier:
VERIFY (as of 2026-10).

**Hardware:** every lab runs on the laptop CPU at default settings. J1.3's CPU default trains a LoRA
adapter on a model of at most about 125M parameters for a few dozen steps; the QLoRA path and the real
P9 training run need a CUDA GPU, because `bitsandbytes` 4-bit kernels are a CUDA path (VERIFY current
CPU support). Docker Desktop on Windows (WSL 2 backend), which you already use, runs the J2 services.
How to use the free GPUs is under "Running Part J on a free GPU" at the end.

---

## J1 · Tracking, versioning, fine-tuning (course week 28)

**Learning objectives**

- I can log a run's parameters, metrics (per step and final), artifacts and code version to MLflow,
  and find the best of 20 runs from the UI and from code, sorted by a validation metric.
- I can make a training run reproducible: same data snapshot, same code commit, same seeds, same
  library versions, and show that re-running it gives the same metric to within a stated tolerance.
- I can snapshot a dataset by content hash, explain why a file name or a date is not a version, and
  restore the exact data a registered model was trained on.
- I can register a model, move a `champion` alias from version 3 to version 4 and back, and trace a
  registered model to its run, commit and data hash (lineage).
- I can explain LoRA as `W' = W + (α/r)·B·A` with `W` of shape `(d_out, d_in)`, `B` of shape
  `(d_out, r)` and `A` of shape `(r, d_in)`, say why `B` starts at zero, and compute the trainable
  parameter count by hand for a stated model, rank and set of target layers.
- I can implement a LoRA layer from scratch, merge it into the base weights, and show the merged and
  unmerged models give the same outputs to within 1e-5.
- I can explain what QLoRA adds (a frozen 4-bit base model, NF4, double quantization, paged optimizers)
  and estimate the GPU memory for full fine-tuning, LoRA and QLoRA of a 1B-parameter model.
- I can run a QLoRA fine-tune on a free GPU and compare it with zero-shot and few-shot prompting on the
  same held-out eval set, with bootstrap confidence intervals, cost per 1,000 requests and p50/p95
  latency.
- I can choose between prompting, RAG and fine-tuning for a stated problem, and defend the choice with
  a quality-versus-cost table rather than a slogan.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| J1.1 | Experiment tracking: runs, parameters, metrics, artifacts (MLflow) | Core | 1.5 |
| J1.2 | Data and model versioning: dataset snapshots, a model registry, lineage | Core | 2 |
| J1.3 | LoRA and QLoRA: low-rank updates from scratch, then parameter-efficient fine-tuning on a free GPU (+ the LoRA paper) | Core | 3.5 |
| J1.4 | Prompt versus RAG versus fine-tune: a decision framework with cost and quality evidence | Core | 1.5 |

**Hours:** reading 3 · coding 5.5 · project 4 · quiz and cheat sheet 1 · **total 13.5**

**Labs**

- **J1.1:** first a tracker from scratch in about 50 lines: a `Run` context manager that creates
  `runs/<run_id>/`, writes `params.json`, appends one JSON line per `log_metric(name, value, step)`,
  copies artifact files in, and records the git commit hash, a dirty-tree flag, the Python and
  library versions and the seed. Write a `compare(runs, metric)` function that prints a sorted table.
  Then the same with MLflow on a local file store (`mlflow ui` on localhost): a 20-run random search
  over gradient-boosting hyperparameters on your P2 dataset (B3.3, B4.3), logging the CV mean and
  standard deviation, a validation curve plot and the fitted pipeline as an artifact. Query the best
  run with `mlflow.search_runs` and reload its model. Finish with a reproducibility test: re-run the
  best configuration twice and assert the metric matches. Bridge: a run is like a CI build record
  (inputs, commit, logs, artifacts); the analogy breaks because you keep and compare *all* runs,
  including the failed ones, and the question is "which inputs caused the better number?", not "did
  it pass?".
- **J1.2:** data versioning from scratch: a `snapshot(path)` function that hashes every file
  (SHA-256), writes a manifest with row counts and a schema summary, and returns a short content id;
  show that renaming a file keeps the id and changing one cell changes it. Freeze the train,
  validation and test splits as one snapshot, so no later run can re-split and leak (B1.4). Then the
  same with DVC: `dvc init`, `dvc add data/<file>`, a local DVC remote outside the repo, and commit
  only the small `.dvc` pointer file (the data itself stays gitignored, as `CLAUDE.md` requires).
  Then the model registry: register three MLflow models from J1.1, tag each with its data snapshot
  id, point a `champion` alias at one, load "the champion" by alias in a tiny script, move the alias,
  and roll back (MLflow aliases versus the older stages API: VERIFY). Write a `lineage(model_name,
  version)` function that prints model → run → commit → data id → split sizes. Bridge: a dataset
  hash is a container image digest and an alias is a mutable tag like `latest`; the analogy breaks
  because two "identical" datasets can still differ in what they mean (a label definition changed),
  which no hash catches, so the manifest records the label definition too.
- **J1.3:** LoRA from scratch in PyTorch. Start with the motivation from A3.2 and B5.4: fully
  fine-tune a tiny model (your F1.4 GPT at CPU size) for a few hundred steps on a new style of text,
  compute `ΔW = W_after − W_before` for one attention projection, take its SVD and plot the singular
  values: most of the change sits in a few directions. Then write `LoRALinear(nn.Module)` wrapping a
  frozen `nn.Linear`: `A` initialized small and random, `B` at zero, output
  `x Wᵀ + (α/r)·x Aᵀ Bᵀ`, assert every shape, and check that before training the wrapped model's
  outputs equal the base model's exactly. Write `inject_lora(model, target_names, r, alpha)`, count
  trainable parameters (compare with the formula `r·(d_in + d_out)` per adapted layer), train only
  the adapters on the same task, and compare validation loss and step time with the full fine-tune.
  Write `merge()` and show merged outputs match unmerged to within 1e-5, then explain why a merged
  adapter adds zero inference latency. Run a rank sweep (`r` = 1, 2, 4, 8, 16) and log it to MLflow.
  Then the library: apply `peft` LoRA with the same `r`, `alpha` and target modules to GPT-2 small,
  copy your `A` and `B` values into its adapter, and check the outputs agree (the `peft` module and
  attribute names: VERIFY). QLoRA, part one, from scratch in NumPy: block-wise absmax quantization of a
  weight matrix to 4 bits (blocks of 64), dequantize, and measure the error; compare uniform 4-bit
  levels with the NF4 levels from the QLoRA paper on Gaussian-shaped weights and show NF4's lower error.
  Part two runs on a free GPU: load the P9 base model in 4-bit with `bitsandbytes`, attach a LoRA
  adapter, and run SFT (F3.1) with `trl`, with the loss masked to the response. A memory table from
  the lesson and from `torch.cuda.max_memory_allocated()`: full fine-tune versus LoRA versus QLoRA.
  CPU default: GPT-2 small or the F1.4 GPT, LoRA `r = 8` on the attention projections, 50–100 steps,
  a few minutes. `GPU_SCALE`: the P9 model (0.5B–1.5B) in 4-bit NF4 on a T4 with fp16 compute (no
  bf16 on a T4; VERIFY the GPU you get), sequence length 512, batch size 4 with gradient accumulation
  (C4.2).
- **J1.4:** a decision lab, not a training lab. Write `cost_per_1k(option)` from measured inputs:
  prompt tokens, output tokens, price per token (VERIFY (as of 2026-10) for the API you use), and for
  the self-hosted fine-tune, GPU hourly cost divided by measured throughput, plus the one-off training
  and eval cost amortized over a stated request volume. Feed it the real numbers from P7 (RAG) and P9
  (prompted, few-shot, QLoRA). Plot quality (with bootstrap CIs, A6.2) against cost per 1,000
  requests, mark the Pareto front, and compute the break-even volume at which the fine-tune becomes
  cheaper than few-shot prompting. Then write a one-page decision framework you can reuse: missing
  *knowledge* points to RAG, a missing *behaviour or format* points to fine-tuning, an unclear spec
  points to better prompts and evals first; and list what changes the answer (data volume, latency
  budget, privacy, how often the knowledge changes, who maintains it).

**Paper:** *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021), with its reading
guide in `papers/`. Read the abstract, section 4 (the method and why `B` starts at zero), the
parameter-count discussion and the rank experiments in section 7, which ask how small `r` can be. The
reading guide connects section 7 to your SVD plot from the J1.3 lab. About 1 hour, counted in J1.3.

**Project milestone: P9 due, LoRA fine-tune versus prompted baseline (4 h).**

- *M0, task, data and baselines (start of the week, 1 h):* pick one narrow task with an
  unambiguous metric (exact-match JSON fields, or macro-F1 over a fixed label set). Snapshot the data
  with J1.2 and freeze a held-out eval set of at least 200 examples that is never used for training or
  prompt tuning. Reuse your I1.3 structured-output harness and I2.4 eval thinking. Run and log three
  baselines: the base instruct model zero-shot, the same model few-shot (5 examples), and the Part I
  API model zero-shot.
- *M1, fine-tune (2 h, on a free GPU):* QLoRA on the training split, logged to MLflow from the
  notebook (log to a file store on Google Drive or Kaggle working storage, then copy the `mlruns`
  folder back), with at least two configurations (for example two ranks or two learning rates).
  Register the best adapter in your local registry with its data snapshot id.
- *M2, comparison and report (1 h):* one table with every option's score and 95% CI on the same eval
  set, cost per 1,000 requests and p50/p95 latency (J1.4), an error analysis of 20 examples where the
  options disagree, a check that the fine-tune did not wreck general ability (perplexity on a small
  general text sample, F2.2, before and after), and a recommendation with the volume at which it
  would flip. A first model card draft (J2.5 finishes it). Brief in `projects/`, rubric in
  `answers/projects/`.

**Prerequisites:** B1.1 (splits, baselines), B1.4 (leakage, frozen splits), B3.3 (the J1.1 sweep
model), B4.1–B4.3 (metrics, cross-validation and tuning), A6.2 (bootstrap CIs), A3.2 (SVD) and B5.4
(low-rank structure), A2.1–A2.2 (shape rules), A1.1 (seeds), C2.1–C2.3 (`nn.Module`, training loop,
saving and reproducibility), C3.1 (AdamW), C4.1 (GPU training), C4.2 (mixed precision and gradient
accumulation), C4.3 (debugging runs), F1.2 (parameter counting), F1.4 (your GPT), F2.2 (perplexity),
F3.1 (SFT, chat templates, loss masking), F3.4 (loading open models and quantized weights), I1.2
(prompting), I1.3 (structured output), I2.3 and I2.4 (RAG and its evals, for J1.4), I4.3 (token cost
and latency). Within the week: J1.2 needs J1.1; J1.4 needs J1.3's P9 numbers.

**Resources** (optional)

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021):
  https://arxiv.org/abs/2106.09685 (VERIFY link)
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023):
  https://arxiv.org/abs/2305.14314 (VERIFY link)
- Hugging Face PEFT documentation: https://huggingface.co/docs/peft (VERIFY link)

**GPU:** required for the QLoRA half of J1.3 and for P9 M1; everything else in J1 runs on the CPU.
Use a free T4 on Kaggle (better weekly quota for a 30–60 minute run) or Colab. At 0.5B–1.5B
parameters in 4-bit, a few thousand short examples for one or two epochs fits one free session
(VERIFY on your session); if memory runs out, cut the sequence length, then the batch size with more
accumulation steps, before switching to a smaller model. Save the adapter and the `mlruns` folder to
Drive or `/kaggle/working` every epoch. Run the P9 evals for the API and few-shot baselines on the
laptop, since they need no GPU.

**Used later in:** the run and registry discipline in every later project, the capstone and the CI
gate in J2.3, which compares a candidate against the registry champion (J1.1, J1.2); serving the P9
adapter in J2.1 and its model card in J2.5 (J1.3); fine-tune-or-prompt questions in L2.2 system design
interviews and the CP5 defence (J1.4); the capstone's design doc in L1.1 (J1.4).

---

## J2 · Serving & operating (course week 29)

**Learning objectives**

- I can serve a model behind a FastAPI endpoint with Pydantic request and response schemas, the model
  loaded once at startup, and separate liveness and readiness checks.
- I can implement dynamic micro-batching (maximum batch size, maximum wait), and show its effect on
  throughput and p50/p95 latency under a load test.
- I can export a PyTorch model to ONNX, check that ONNX Runtime's outputs match PyTorch to within
  1e-5, quantize it to int8 and report the speed and accuracy change.
- I can explain what an LLM inference server adds over a plain endpoint (continuous batching, paged KV
  cache, an OpenAI-compatible API) and run one locally on the CPU.
- I can compute the population stability index and the two-sample KS statistic by hand for a small
  example and in code, and explain the difference between data drift and concept drift and how to
  detect each when labels arrive late.
- I can set drift alert thresholds that account for testing many features at once, and say what I
  would do when an alert fires.
- I can write data tests, model behaviour tests (invariance, directional, minimum functionality) and
  an eval threshold, and run them as required checks on a pull request in GitHub Actions.
- I can build a small, reproducible Docker image for a model service, deploy it to a free public
  target, and run a shadow and a canary rollout with an automatic rollback rule.
- I can write a model card and a datasheet, explain the randomized-response and Laplace mechanisms
  and what ε controls, and state a retention policy for logged inputs and predictions.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| J2.1 | Model serving: a FastAPI service, batching, ONNX export and quantization; LLM inference servers | Core | 2.5 |
| J2.2 | Monitoring and drift: data and concept drift, PSI and KS tests, alerting | Core | 2 |
| J2.3 | CI for ML: data tests, model tests and evals as merge gates in GitHub Actions | Core | 2 |
| J2.4 | Deployment: Docker images, a cloud target, shadow and canary rollout, rollback | Core | 2 |
| J2.5 | `[RAI]` Documentation and governance: model cards, datasheets, privacy (differential privacy idea, retention) | Core | 1.5 |

**Hours:** reading 3 · coding 7 · project 1.5 · quiz and cheat sheet 1 · CP5 self-test and mock
interview 1.5 · **total 14**

**Labs**

- **J2.1:** a FastAPI service for your P2 champion, loaded from the J1.2 registry by alias at
  startup: `POST /predict` with a Pydantic schema (validation errors return 422, not a stack trace),
  `GET /healthz` (process alive) and `GET /readyz` (model loaded), the model version in every response
  and in a structured log line. Then micro-batching from scratch with `asyncio`: requests go into a
  queue, a background task collects up to `max_batch` items or waits at most `max_wait_ms`, runs one
  vectorized prediction and resolves each request's future. Write a load script with `httpx`
  (concurrency 1, 8, 32) and plot throughput and p50/p95 latency for batching off and on, at two
  `max_wait_ms` values: batching raises throughput and costs latency at low load. Then ONNX: export a
  small PyTorch model (your C-part MLP or the D-part CNN) with `torch.onnx.export` and a dynamic batch
  axis, run it in ONNX Runtime, assert outputs match to within 1e-5, apply dynamic int8 quantization
  and compare file size, latency and accuracy (exporter API: VERIFY, it has changed). Finally an LLM
  inference server on the CPU: run a small quantized GGUF model with the `llama.cpp` server (or
  Ollama), call its OpenAI-compatible endpoint with the I1.1 client you already wrote, and measure
  time to first token and tokens per second; tie the numbers back to the KV cache (F2.4). Bridge: this
  is a normal stateless HTTP service; the analogy breaks at batching (a GPU or a vectorized CPU kernel
  is far cheaper per item in batches, which an ordinary web service does not care about) and at
  memory (one model can fill the whole machine, so you scale by replicas of a large process, not by
  threads).
- **J2.2:** drift monitoring on your P2 dataset. Split it by time (or, if there is no time column,
  build a "production" sample and inject three kinds of change: a shifted mean on one numeric feature,
  a new dominant category, and a changed relationship between a feature and the label with unchanged
  inputs). Write PSI from scratch (bin edges from the reference data's quantiles, a small epsilon for
  empty bins) and the two-sample KS statistic from scratch (the largest gap between the two empirical
  CDFs), and check both against `scipy.stats.ks_2samp` and a hand calculation on 10 values; use the
  chi-square test (A6.3a) for categoricals. Show that with 30 features and α = 0.05 you get false
  alarms on undrifted data, then fix it with a Bonferroni correction or a PSI threshold (the common
  0.1/0.25 rule of thumb, and why it is only a convention). Show that input drift tests miss the third
  change entirely, and catch it with performance monitoring on delayed labels: a rolling metric with a
  bootstrap CI (A6.2). Write a `monitor(reference, window)` job that outputs a report and an alert
  list, and a short runbook: what an alert means, who looks, when to retrain. Add an LLM column to the
  runbook from I4.5 (prompt length, refusal rate, eval score on sampled traffic). Bridge: this is
  application monitoring with dashboards and alerts; it breaks because nothing in the service is
  failing, so error rates and latency stay green while the model gets worse.
- **J2.3:** CI for the P9 repository (or your P2 repository) on GitHub Actions. Data tests with
  `pytest`: schema and types, value ranges, null rates, no duplicate ids, no overlap between train and
  test ids (B1.4), and a check that the data snapshot id matches the manifest. Model tests: a
  minimum-functionality set (cases the model must get right), invariance tests (changing a name or a
  harmless field must not change the prediction) and directional tests (a feature change that should
  move the score in one direction), in the style of the CheckList behavioural-testing approach.
  A regression gate: the candidate must not be worse than the registry champion by more than a stated
  margin on the frozen eval set, with the CI from the bootstrap so noise does not block merges. Then
  wire your P7 golden set and P8 task suite in as gates: run a small fixed subset on every pull request
  and the full suite nightly, cache model responses by a hash of prompt, model and parameters so
  re-runs are cheap, keep the API key in repository secrets, and set a cost cap per run. Make the
  workflow a required status check on the main branch and open a pull request that deliberately fails
  each gate once. Bridge: the same GitHub Actions you already write; it breaks because ML tests are
  statistical, so thresholds need margins and confidence intervals, and LLM evals cost money per run.
- **J2.4:** deployment. A multi-stage `Dockerfile` for the J2.1 service: CPU-only PyTorch or ONNX
  Runtime wheels, a non-root user, a pinned base image by digest, the lockfile installed with `uv`, and
  the model pulled at startup by registry alias rather than baked in (and the trade-off: baking in is
  more reproducible and starts faster; pulling lets you roll back without a rebuild). Measure and cut
  the image size. Deploy to the default free target, a Hugging Face Space with the Docker SDK (free
  CPU tier, sleeps when idle; VERIFY (as of 2026-10)), from the J2.3 workflow on merge to main.
  Optionally, the same image on Google Cloud Run with Terraform, which you already know. Then rollout
  patterns locally with Docker Compose: a small router in front of two model versions that (1) in
  shadow mode sends every request to v1, mirrors it to v2, returns only v1's answer and logs both
  predictions for comparison, and (2) in canary mode sends 10% of traffic to v2, compares error rate,
  latency and prediction distribution (J2.2's PSI) between the two, and rolls back automatically by
  setting the weight to zero when a rule trips. Roll back a second way by moving the registry alias
  back (J1.2). Bridge: blue–green and canary are deployment patterns you know; the model-specific
  part is the rollback rule, which watches prediction quality and distribution, not only HTTP errors.
- **J2.5:** finish the P9 model card (intended use, out-of-scope uses, training data and snapshot id,
  eval results with CIs, per-group or per-slice results where the task allows, known failure modes
  from the P9 error analysis, safety findings in the F3.3 style, licence of the base model) and write a
  datasheet for the P9 dataset (motivation, composition, collection, labelling, known biases, PII,
  licence, maintenance). Then the differential privacy idea from scratch: randomized response for a
  yes/no survey question, with a simulation showing how the estimate's error grows as ε shrinks;
  the Laplace mechanism for a count query (noise with scale `sensitivity / ε`, using A5.2); and the
  DP-SGD recipe at concept level (clip each example's gradient, add Gaussian noise), with what it costs
  in accuracy. Finish with a retention and logging policy for the J2.1 service: which fields are
  logged, PII redaction before logging (I4.1), how long prediction logs and LLM prompts are kept, who
  can read them, and how a deletion request is handled, including the honest point that data already
  used for training is not removed from the weights by deleting the row. Regulation such as the EU AI
  Act is mentioned only as "know it exists and where it applies": VERIFY (as of 2026-10).

**Project milestone: CP5 project work (1.5 h).** P7 and P8 eval suites running as GitHub Actions
merge gates (done in the J2.3 lab), the P9 model card and datasheet finished (J2.5), and any fixes from
your P7, P8 and P9 reviews, so that all three can score 70% or more for CP5. Optional and useful for
the fast path: deploy P7 as a service with the J2.4 pipeline.

**Prerequisites:** C2.3 (saving and loading, device handling), I1.3 (Pydantic schemas), I1.1 (the API
client, reused against the local inference server), F2.4 (KV cache), F3.4 (quantized weights), D1.4
(efficient models, for the ONNX discussion), I4.3 (batching, latency and cost), A5.2 (distributions,
the Laplace distribution), A6.2 (bootstrap CIs), A6.3a–A6.3b (hypothesis tests, chi-square, multiple
comparisons), B1.2 (PII and data minimization), B1.4 (leakage tests), B4.1–B4.2 (metrics to monitor),
B4.6 (the first model card, fairness slices), B5.6 (anomaly detection as alerting), B6.3 (time-based
splits), F3.3 (safety findings), I2.4 and I3.3 (the eval suites that become gates), I4.1 (PII
redaction), I4.2 (least-privilege secrets), I4.5 (observability and logging prompts safely), J1.1–J1.4
(tracking, registry, P9). Within the week: J2.2 and J2.4 use the J2.1 service; J2.4 extends the J2.3
workflow.

**Resources** (optional)

- Goku Mohandas, *Made With ML* (MLOps course, free): https://madewithml.com (VERIFY link)
- Mitchell et al., "Model Cards for Model Reporting" (2019): https://arxiv.org/abs/1810.03993 (VERIFY
  link); Gebru et al., "Datasheets for Datasets": https://arxiv.org/abs/1803.09010 (VERIFY link)
- Sculley et al., "Hidden Technical Debt in Machine Learning Systems" (NeurIPS 2015):
  https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems (VERIFY link)

**GPU:** no. Every J2 lab runs on the laptop CPU, including the LLM inference server with a small
quantized model. Helpful only if you want to try `vllm` in J2.1: run it in a Colab or Kaggle T4
session against a small model and compare its throughput under concurrent requests with the CPU
server (vLLM's supported GPUs and install steps: VERIFY). The notebook VM cannot be reached as a public
endpoint in a supported way, so measure from inside the notebook rather than deploying from it.

**Used later in:** the capstone's deployment, monitoring, model card and cost report in L2.1 and P12
(all of J2); serving, monitoring, canaries and CI gates as building blocks in every L2.2 system design
case (J2.1–J2.4); model cards for every later project, including P10 and P11 (J2.5).

---

## Running Part J on a free GPU

Only J1.3's QLoRA half and P9 M1 need a GPU. Free-tier GPUs, quotas and session limits change often:
VERIFY (as of 2026-10).

- **Kaggle (preferred for P9):** create a notebook, then Settings → Accelerator → GPU, and turn
  Internet on to download the base model and dataset (Kaggle asks for phone verification first).
  The weekly GPU quota and longer sessions suit a 30–60 minute QLoRA run. Write the adapter, the eval
  predictions and the `mlruns` folder to `/kaggle/working`, then download them.
- **Colab:** Runtime → Change runtime type → a GPU (usually a T4). Check with
  `torch.cuda.is_available()` and `torch.cuda.get_device_name()`. Mount Google Drive and point MLflow
  at a file store there (`mlflow.set_tracking_uri("file:/content/drive/MyDrive/mlruns")`), so runs
  survive a disconnect; save the adapter there every epoch.
- **Both:** install `bitsandbytes`, `peft` and `trl` at the pinned versions from `pyproject.toml`
  (a mismatch with the preinstalled `transformers` is the most common failure; VERIFY). On a T4 use
  fp16 compute for the 4-bit model; bf16 needs a newer GPU. Gated models need a Hugging Face token,
  which goes in the notebook's secrets, never in the code. Run the eval set in the same session with
  the merged or attached adapter, and record GPU time per run for J1.4's cost table. Bring back only
  adapters (to the MLflow artifact store or Drive, not git), metrics, logs and eval tables; never
  datasets, base-model weights or checkpoints.

---

## You're ready for Part G when…

Part G (Generative models & multimodal, course weeks 30–31) comes next in the course order. This
self-check covers CP5 and the Part J habits every later part relies on. Answer each item without
notes. The answers belong in your quiz reviews, not here.

1. P7, P8 and P9 each score 70% or more on their rubrics, and the P7 and P8 eval suites run as
   required checks on pull requests in GitHub Actions (CP5).
2. You can defend your P9 fine-tune-or-prompt decision with the numbers: scores with CIs, cost per
   1,000 requests, latency, and the request volume at which the answer flips (CP5).
3. You have done one LLM-systems mock interview with the mock-interview prompt and been rated "lean
   hire" or better (CP5).
4. Given any of your models, you can name the commit, data snapshot and run it came from, and roll
   the served version back without rebuilding the image.
5. You can write the LoRA update with its shapes, count its trainable parameters for a stated layer
   and rank, and say what QLoRA adds and why it fits on a free GPU.
6. You can explain why micro-batching raises throughput and costs latency, and what an LLM inference
   server does that a plain FastAPI endpoint does not.
7. You can compute PSI and the KS statistic on a small example, and explain why input drift tests
   cannot catch concept drift.
8. You can describe a shadow and a canary rollout and the rule that triggers an automatic rollback.
9. You can list what goes in a model card and a datasheet, and explain ε in the Laplace mechanism.

If two or more items fail, revisit the topic ids your quiz reviews flagged before you start G1.
Items 4 and 5 matter most for what comes next: Part G's training runs are logged and registered the
same way, and P12 in Part L reuses all of J2.
