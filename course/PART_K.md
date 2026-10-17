# Part K. Responsible AI — detailed syllabus

Woven into Parts B, D, E, F, I, J, G, H and L. Weekly detail for Part K of
[`COURSE_OUTLINE.md`](COURSE_OUTLINE.md), which stays the source of truth: every topic id, Core/Depth
label and placement below is taken from its Part K table and the `[RAI]` tags, unchanged. Written
2026-10-16.

**Goal:** make fairness, privacy, safety and documentation habits you apply in every project, not a
chapter you read once, and be able to defend those choices in an interview.

**How this file is organised.** Part K has no weeks of its own, so it is organised by the **host
week** where the outline teaches each `[RAI]` piece, and each section gives the requested fields for
the Responsible AI part of that week only.

**Who owns what.** The host part's file (`PART_B.md`, `PART_D.md`, …) owns each topic's lesson, lab
and total hours; this file only gathers the `[RAI]` thread, summarises the `[RAI]` labs those files
specify, and adds no study time. If a host file and this one ever disagree, the host file wins and
this file is corrected in that part's QA pass (Step 4).

## At a glance

| Host week | Course wk | `[RAI]` topic | Theme | Level | `[RAI]` hours (topic + project share) | Project | GPU |
|---|---|---|---|---|---|---|---|
| B1 | 7 | B1.2 PII handling, data minimization | Privacy | Core | 0.75 + 0.25 | P2 M0 | no |
| B4 | 10 | B4.6 group fairness metrics, audit, first model card | Fairness, Documentation | Core | 2 + 1.5 | **P2 due** | no |
| D2 | 18 | D2.5 Grad-CAM, per-group errors, dataset bias | Fairness | Depth | 1.5 + 1 | **P5 due** | no (helpful for P5) |
| E1 | 19 | E1.4 bias in embeddings | Fairness | Depth | 0.75 + 0 | — | no |
| F3 | 23 | F3.3 hallucination, harmful output, refusals, red-teaming | Safety | Core | 2 + 0 | CP4 | helpful |
| I4 | 27 | I4.1 guardrails, PII redaction, moderation; I4.2 prompt injection, exfiltration, least privilege | Safety, Privacy | Core | 4.5 + 0.75 | P7 + P8 hardening | no |
| J2 | 29 | J2.5 model cards, datasheets, differential privacy idea, retention | Documentation, Privacy | Core | 1.5 + 0.5 | P9 card · CP5 | no |
| G2 | 31 | G2.4 deepfakes and watermarking | Safety | Depth | 0.75 + 0.25 | **P10 due** | helpful |
| H2 | 33 | H2.4 reward hacking and specification gaming | Safety | Depth | 1.5 + 0.25 | **P11 due** | no |
| L1–L2 | 34–35 | L2.1 capstone model card (plus the L1.1 risk section) | Documentation, all themes | Core | 0.25 + 2 | P12 | no |

About **22 hours** across the course, all already counted inside the host weeks. *Topic* hours are the
`[RAI]` share of the lesson and lab; where a host topic is only partly `[RAI]` (B1.2, E1.4, G2.4, L2.1)
the share is an estimate of that part. *Project* hours are the share of that week's project work that
is Responsible AI (the fairness audit, the model card's limitations section, and so on).

**Core path.** B1.2, B4.6, F3.3, I4.1, I4.2, J2.5 and L2.1 are Core and on the fast path; they are
about 16 of the 22 hours. D2.5, E1.4, G2.4 and H2.4 are Depth and are the first `[RAI]` pieces to cut
if you fall behind (the outline's cut list already names D2.5 and E1.4). Never cut I4.2: the outline
forbids it.

**The running thread: one model card per project.** From P2 on, every project ships a model card
built from the `STYLE_GUIDE.md` template that B4.6 introduces. Each card grows with what you have
learned by then:

| Project | `[RAI]` content its model card must have | Learned in |
|---|---|---|
| P2 tabular model | Metrics by group for the sensitive attribute, the threshold choice and its fairness trade-off, excluded PII columns | B1.2, B4.6 |
| P3 unsupervised | Whether any cluster is a proxy for a sensitive attribute; who could be harmed by acting on a segment | B4.6 |
| P4 autograd + MLP | Intended use (teaching), data source, limitations; short is fine | B4.6 |
| P5 CV transfer learning | Per-group error table, Grad-CAM evidence of shortcuts, dataset source and its bias | D2.5 |
| P6 small GPT | Training data and its licence, what the model must not be used for, sample failures | F3.3 |
| P7 RAG system (system card) | PII in the corpus, faithfulness rate with its CI, what the system does when retrieval finds nothing | F3.3, I2.4 |
| P8 tool-using agent (system card) | Prompt-injection test results, each tool's permissions, actions that need confirmation | F3.3, I4.2 |
| P9 LoRA fine-tune | Fine-tuning data card (datasheet), base-model licence, safety re-test after tuning | J2.5 |
| P10 generative model | Misuse section, the watermark you added and how easily it is removed | G2.4 |
| P11 RL agent | The reward you specified, and any reward hacking you saw and how you fixed it | H2.4 |
| P12 capstone | All four themes, plus retention and monitoring commitments | L2.1 |

**Libraries this thread needs** (Step 2 pins the versions): `fairlearn` (only to check your
from-scratch metrics), `gensim` (E1.4, already in Part E) and `presidio-analyzer` (optional
comparison in I4.1). Library APIs here change between versions: VERIFY
(as of 2026-10) when Step 2 pins them.

**Hardware:** every `[RAI]` lab runs on the laptop CPU in under 5 minutes. The only places a GPU
helps are the host projects around them (P5, the F3 `GPU_SCALE` runs, P10).

---

## B1 · Privacy: PII and data minimization (course week 7)

**Learning objectives**

- I can list the direct identifiers (name, email, phone, ID numbers) and quasi-identifiers (birth date, postcode, gender) in a dataset and explain why the combination of quasi-identifiers can re-identify people.
- I can decide, for each PII column, whether to drop, hash (with a secret salt) or coarsen it, and justify the decision by what the model actually needs.
- I can explain why hashing a join key is pseudonymization, not anonymization.

**Topics** (in teaching order)

| Id | Topic | Level | Hours |
|---|---|---|---|
| B1.2 | Data cleaning; `[RAI]` PII handling and data minimization | Core | 2 (about 0.75 `[RAI]`) |

**Hours (`[RAI]` share):** reading 0.25 · coding 0.5 · project 0.25 · **total 1**

**Labs** (the PII pass inside the B1.2 lab in `PART_B.md`)

- A PII pass that drops direct identifiers, hashes a join key with a salted SHA-256 and coarsens a
  birth date to an age band. Then a uniqueness check from scratch: count how many rows are unique on
  the remaining quasi-identifiers (a k-anonymity count), before and after coarsening.

**Project milestone:** P2 M0 lists the columns you exclude for leakage or PII reasons, one line each
(`PART_B.md`).

**Prerequisites:** A1.3 (pandas), A6.1 (EDA).

**Resources** (optional)

- GDPR Article 5, "Principles relating to processing of personal data" (data minimisation, storage limitation): https://gdpr-info.eu/art-5-gdpr/ (VERIFY link)
- Microsoft Presidio documentation (PII detection and anonymization): https://microsoft.github.io/presidio/ (VERIFY link)

**GPU:** no.

**Used later in:** PII redaction in I4.1, data exfiltration in I4.2, retention and differential
privacy in J2.5.

---

## B4 · Fairness and the first model card (course week 10)

**Learning objectives**

- I can compute demographic parity difference, equal-opportunity difference and equalized-odds difference from predictions and a sensitive attribute, by hand on a 2×2 example and in code.
- I can explain, with a numeric example, why demographic parity and equalized odds usually cannot both hold when base rates differ between groups.
- I can show how a single threshold and per-group thresholds trade one fairness metric against another and against overall accuracy.
- I can write a model card with intended use, out-of-scope uses, metrics by group, and known limitations.
- I can explain why dropping the sensitive column does not make a model fair (proxies).

**Topics** (in teaching order)

| Id | Topic | Level | Hours |
|---|---|---|---|
| B4.6 | `[RAI]` Fairness: group metrics (demographic parity, equalized odds), auditing a model, a first model card | Core | 2 |

**Hours (`[RAI]` share):** reading 0.75 · coding 1.25 · project 1.5 · **total 3.5**

**Labs** (the B4.6 lab in `PART_B.md`)

- Demographic parity difference, equal-opportunity difference and equalized-odds difference from
  scratch on UCI Adult, checked against `fairlearn.metrics`.
- Single threshold versus per-group thresholds: plot each fairness metric and accuracy as the
  thresholds move.
- A model card for the model from the `STYLE_GUIDE.md` template.

**Project milestone: P2 due.** The `[RAI]` part: the fairness audit on the sensitive attribute and
the model card with intended use, metrics by group and known limitations (about 1.5 of P2's 4 hours).

**Prerequisites:** B1.2 (PII), B4.1 (confusion matrix, TPR, FPR), B4.4 (threshold tuning); A5.3
(conditional probability), A6.2 (bootstrap CIs for per-group metrics on small groups).

**Resources** (optional)

- Barocas, Hardt, Narayanan, *Fairness and Machine Learning: Limitations and Opportunities* (free online book): https://fairmlbook.org (VERIFY link)
- Hardt, Price, Srebro, "Equality of Opportunity in Supervised Learning" (2016): https://arxiv.org/abs/1610.02413 (VERIFY link)
- Fairlearn user guide: https://fairlearn.org (VERIFY link)

**GPU:** no.

**Used later in:** every model card from P2 on; per-group errors in D2.5; bias tests in E1.4; model
cards and datasheets in J2.5 and L2.1.

---

## D2 · Fairness in vision: dataset bias and shortcuts (course week 18)

**Learning objectives**

- I can implement Grad-CAM with forward and backward hooks and read a heatmap honestly (it shows where the gradient points, not "what the model thinks").
- I can build a per-group error table (class × group) for an image model and use it to find a shortcut the overall accuracy hides.
- I can explain why a spurious correlation in training data produces a model that fails when the correlation is removed.

**Topics** (in teaching order)

| Id | Topic | Level | Hours |
|---|---|---|---|
| D2.5 | `[RAI]` Error analysis for vision: Grad-CAM, per-group errors and dataset bias | Depth | 1.5 |

**Hours (`[RAI]` share):** reading 0.5 · coding 1 · project 1 · **total 2.5**

**Labs** (the D2.5 lab in `PART_D.md`)

- Grad-CAM from scratch on the last convolutional stage of the D2.1 model.
- A planted-bias dataset where one class almost always has one background colour: show the accuracy
  drop when the correlation is removed, Grad-CAM lighting up the background, and the per-group error
  table exposing it. The same per-group table as B4.6, applied to images.

**Project milestone: P5 due.** The `[RAI]` part: Grad-CAM on correct and wrong predictions, per-group
errors for any grouping your data allows, and the model card (about 1 of P5's 3 hours).

**Prerequisites:** B4.6 (per-group metrics, model cards), D2.1 (the fine-tuned model), C2.1
(autograd, for hooks).

**Resources** (optional)

- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (2016): https://arxiv.org/abs/1610.02391 (VERIFY link)
- Geirhos et al., "Shortcut Learning in Deep Neural Networks" (2020): https://arxiv.org/abs/2004.07780 (VERIFY link)
- Buolamwini, Gebru, "Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification" (FAT* 2018): https://proceedings.mlr.press/v81/buolamwini18a.html (VERIFY link)

**GPU:** no for the D2.5 lab (CPU default in under 5 minutes). Helpful for P5 as a whole; see
"Running Part D on a free GPU" in `PART_D.md`.

**Used later in:** the per-group audit in every model card from P5 on; CLIP's zero-shot behaviour
on groups in G2.2.

---

## E1 · Fairness in text: bias in embeddings (course week 19)

**Learning objectives**

- I can find a bias direction in an embedding space from definitional word pairs and project other words onto it.
- I can run a small WEAT-style association test and explain what its effect size measures.
- I can show that "neutralizing" a direction lowers the measured projection but does not remove the information, and explain why that matters for any model built on the vectors.

**Topics** (in teaching order)

| Id | Topic | Level | Hours |
|---|---|---|---|
| E1.4 | Subword and pretrained static embeddings (fastText, GloVe); `[RAI]` bias in embeddings | Depth | 1.5 (about 0.75 `[RAI]`) |

**Hours (`[RAI]` share):** reading 0.25 · coding 0.5 · project 0 · **total 0.75**

**Labs** (the `[RAI]` part of the E1.4 lab in `PART_E.md`)

- A gender direction from definitional pairs, occupation words projected onto it and plotted, a
  WEAT-style association test, and the "neutralize" step of hard debiasing.
- Show that a classifier trained on the debiased vectors can still recover gender from neighbours:
  debiasing the geometry is not the same as removing bias.

**Project milestone:** none.

**Prerequisites:** E1.3 (word2vec and embedding geometry), A1.5 (cosine similarity, projections),
A2.4 (projections), B2.1 (logistic regression for the recovery classifier), B4.6 (fairness framing).

**Resources** (optional)

- Bolukbasi et al., "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings" (2016): https://arxiv.org/abs/1607.06520 (VERIFY link)
- Caliskan, Bryson, Narayanan, "Semantics derived automatically from language corpora contain human-like biases" (2017): https://arxiv.org/abs/1608.07187 (VERIFY link)
- Gonen, Goldberg, "Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But do not Remove Them" (2019): https://arxiv.org/abs/1903.03862 (VERIFY link)

**GPU:** no.

**Used later in:** bias in contextual models and LLM outputs (F3.3), embedding-based retrieval in
I2.1 (the same vectors decide what a RAG system finds), CLIP in G2.2.

---

## F3 · Safety and alignment: hallucination, refusals, red-teaming (course week 23)

**Learning objectives**

- I can define hallucination, harmful output, refusal and over-refusal, with an example of each.
- I can build a small red-team and factuality test set with a written grading rubric, run it against a model, and grade it by hand.
- I can report a failure rate from 20–30 prompts with its count and a confidence interval, and say why it should not be quoted as a precise number.
- I can try one mitigation (a system prompt or a few SFT examples), re-run the same set, and say whether the change is bigger than the noise.

**Topics** (in teaching order)

| Id | Topic | Level | Hours |
|---|---|---|---|
| F3.3 | `[RAI]` Safety and alignment: hallucination, harmful outputs, refusals, red-teaming | Core | 2 |

**Hours (`[RAI]` share):** reading 0.5 · coding 1.5 · project 0 · **total 2**

**Labs** (the F3.3 lab in `PART_F.md`)

- Three benign prompt sets of 20–30 items each against the F3.4 instruct model and your F3.1 SFT
  model: factuality (known answers, made-up entities, false premises), policy probes with harmless
  stand-ins (extracting a "secret password" from the system prompt, role-play framings, personal
  details of a fictional person), and over-refusal probes ("how do I kill a Python process?").
- Hand grading against a rubric, rates with counts and CIs, one mitigation and a re-run, and a
  half-page red-team report in model-card style. Keep the prompt sets: they seed I4.1, I4.2 and P8.

**Project milestone:** none due (CP4 week).

**Prerequisites:** F3.1 (SFT), F3.4 (running open models); A6.2 (confidence intervals), B4.6 (model
cards and auditing).

**Resources** (optional)

- Ganguli et al., "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned" (2022): https://arxiv.org/abs/2209.07858 (VERIFY link)
- Lin, Hilton, Evans, "TruthfulQA: Measuring How Models Mimic Human Falsehoods" (2021): https://arxiv.org/abs/2109.07958 (VERIFY link)

**GPU:** helpful, never required. The lab runs on CPU with a model of about 0.5B parameters or less;
a free T4 only speeds up generating the answers. On Colab or Kaggle, load the model in fp16 and run
the same file; bring back only the graded tables.

**Used later in:** guardrails in I4.1, prompt injection in I4.2, the injection test set in P8,
reward hacking in H2.4, and the safety section of every model card.

---

## I4 · Guardrails and LLM security (course week 27)

**Learning objectives**

- I can put input and output checks around an LLM call as a pipeline (validate, redact, call, check, log) and say which checks are deterministic and which are probabilistic.
- I can implement PII redaction for emails, phone numbers and card numbers (with a Luhn check) from scratch, measure its precision and recall on a labelled set, and compare it with a library.
- I can measure a guardrail's false-positive rate on benign traffic, and explain why a guardrail that blocks 5% of good requests is a product bug.
- I can demonstrate direct and indirect prompt injection against a small RAG or tool-using app, including data exfiltration through a tool call or a rendered link.
- I can apply least privilege to an agent's tools (read-only by default, allowlists, human confirmation for side effects) and measure attack success before and after.
- I can explain why prompt injection has no equivalent of the parameterized query, so defences reduce risk but do not remove it.

**Topics** (in teaching order)

| Id | Topic | Level | Hours |
|---|---|---|---|
| I4.1 | `[RAI]` Guardrails: input and output checks, PII redaction, moderation | Core | 2 |
| I4.2 | `[RAI]` Security: direct and indirect prompt injection, data exfiltration, least-privilege tools | Core | 2.5 |

**Hours (`[RAI]` share):** reading 1.25 · coding 3.25 · project 0.75 · **total 5.25**

**Labs** (the I4.1 and I4.2 labs in `PART_I.md`)

- **I4.1:** a guardrail pipeline as middleware around P7, each check returning allow, block or
  transform. PII detection from scratch with regular expressions plus validation (the Luhn checksum
  for card numbers, email and phone formats), scored for precision and recall on 100 hand-labelled
  lines and compared with Presidio (VERIFY). Redaction by masking and by reversible pseudonymization
  (`<PERSON_1>`), so the provider never sees the raw values. Output checks: schema validation, a
  moderation check (VERIFY what exists), and a groundedness check on citations. Measure how many good
  P7 answers each check wrongly blocks and the latency each adds.
- **I4.2:** attack your own systems only. A canary secret in P8's system prompt and a "write" tool in
  its toolset, then 30 attacks: direct injection, role-play and encoding tricks, indirect injection in
  a P7 document and in a tool result, exfiltration through a Markdown image URL, and cross-user write
  attempts. Add defences one at a time (spotlighting untrusted content, a stricter system prompt, an
  injection classifier, a link allowlist, and the architectural ones: a `SELECT`-only Postgres role
  with row-level security, human confirmation for writes, no secrets in context) and re-measure attack
  success and P8 task success after each. Sketch the dual-LLM pattern. Bridge: SQL injection with no
  parameterized queries, because instructions and data share one channel.

**Project milestone: P7 and P8 hardening (`PART_I.md`).** The `[RAI]` part: on P8, the I4.2 defences
with the injection set and task suite re-run and both rates reported before and after; on P7, the I4.1
guardrails; update both system cards (about 0.75 of the 1 hour).

**Prerequisites:** F3.3 (red-team sets, over-refusal), B1.2 (PII), B4.1 (precision and recall of each
guardrail), I1.3 (schema checks), I1.4 and I3.1 (tools), I2.3 (RAG pipeline and citations), I2.4
(evals), I3.3 (the P8 task suite); P7 and P8 themselves; A6.2 (CIs on attack success rates).

**Resources** (optional)

- OWASP Top 10 for Large Language Model Applications: https://genai.owasp.org/ (VERIFY link)
- Greshake et al., "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection" (2023): https://arxiv.org/abs/2302.12173 (VERIFY link)
- Simon Willison's prompt-injection series: https://simonwillison.net/series/prompt-injection/ (VERIFY link)

**GPU:** no. API calls or a small local model on CPU.

**Used later in:** observability and safe prompt logging in I4.5, evals as CI gates in J2.3, the
capstone's security section in L2.1, LLM-systems mock interviews in L3.2.

---

## J2 · Documentation, governance and privacy (course week 29)

**Learning objectives**

- I can write a model card and a datasheet for a dataset, and say what each is for and who reads it.
- I can explain the definition of ε-differential privacy in words, implement the Laplace mechanism and randomized response from scratch, and plot the privacy–utility trade-off as ε changes.
- I can explain what DP-SGD adds to a training step (per-example gradient clipping and noise) and why it costs accuracy.
- I can explain why deleting a row does not remove its influence from a model already trained on it.
- I can write a retention policy for training data, prompts and logs, and connect it to data minimization.

**Topics** (in teaching order)

| Id | Topic | Level | Hours |
|---|---|---|---|
| J2.5 | `[RAI]` Documentation and governance: model cards, datasheets, privacy (differential privacy idea, retention) | Core | 1.5 |

**Hours (`[RAI]` share):** reading 0.5 · coding 1 · project 0.5 · **total 2**

**Labs** (the J2.5 lab in `PART_J.md`)

- Finish the P9 model card (intended use, out-of-scope uses, training data and snapshot id, eval
  results with CIs, per-slice results, failure modes, safety findings in the F3.3 style, base-model
  licence) and write a datasheet for the P9 dataset.
- Randomized response with a simulation of error against ε, the Laplace mechanism for a count query
  (scale `sensitivity / ε`), and DP-SGD at concept level.
- A retention and logging policy for the J2.1 service: fields logged, redaction before logging (I4.1),
  retention periods for prediction logs and prompts, access, and deletion requests. Regulation such as
  the EU AI Act only as "know it exists and where it applies": VERIFY (as of 2026-10).

**Project milestone: P9's card and datasheet (0.5 h).** P9 is due in J1 (week 28); update its model
card and add the datasheet. Checkpoint CP5 is this week.

**Prerequisites:** B1.2 (data minimization), B4.6 (first model card), A5.2 (Laplace and Gaussian
distributions), C3.1 (the optimizer step DP-SGD modifies), F3.3 (safety findings), I4.1 (redaction),
I4.5 (prompt logging), J1.2 (lineage), J1.3 (the P9 fine-tune), J2.1 (the service whose logs the
policy covers).

**Resources** (optional)

- Gebru et al., "Datasheets for Datasets" (2018): https://arxiv.org/abs/1803.09010 (VERIFY link)
- Dwork, Roth, *The Algorithmic Foundations of Differential Privacy* (free PDF): https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf (VERIFY link)
- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework (VERIFY link)

**GPU:** no.

**Used later in:** every model card from here on, and the capstone's documentation in L2.1.

---

## G2 · Deepfakes and watermarking (course week 31)

**Learning objectives**

- I can explain the misuse risks of image and text generators and the difference between detection, watermarking and provenance metadata.
- I can add a simple invisible watermark to generated images, detect it, and measure how well it survives JPEG compression, resizing and cropping.
- I can say what provenance metadata (C2PA content credentials) and model-level watermarks claim to offer, and where they fail.

**Topics** (in teaching order)

| Id | Topic | Level | Hours |
|---|---|---|---|
| G2.4 | Latent diffusion, text conditioning and classifier-free guidance; `[RAI]` deepfakes and watermarking | Depth | 2 (about 0.75 `[RAI]`) |

**Hours (`[RAI]` share):** reading 0.25 · coding 0.5 · project 0.25 · **total 1**

**Labs** (the `[RAI]` part of the G2.4 lab in `PART_G.md`)

- An invisible watermark (a simple frequency-domain or least-significant-bit scheme) on a generated
  image, a detector for it, and detection failing after JPEG compression, resizing and cropping.
- One paragraph on what C2PA content credentials and model-level watermarks such as SynthID claim and
  their limits (VERIFY (as of 2026-10)), and on consent and non-consensual imagery.

**Project milestone: P10 due.** Add a misuse section and the watermark result to its model card
(about 0.25 h).

**Prerequisites:** G1.4 (DDPM samples), A6.3a (hypothesis tests, for the detector).

**Resources** (optional)

- Kirchenbauer et al., "A Watermark for Large Language Models" (2023), the text-side counterpart: https://arxiv.org/abs/2301.10226 (VERIFY link)
- C2PA (Coalition for Content Provenance and Authenticity) specifications: https://c2pa.org (VERIFY link)

**GPU:** helpful only for generating more DDPM samples; the watermark lab itself is CPU and can use
any images. On Colab or Kaggle, generate samples with the P10 `GPU_SCALE` settings and bring back the
PNGs, not the weights.

**Used later in:** the capstone if it generates media; system-design answers in L2.2.

---

## H2 · Reward hacking and specification gaming (course week 33)

**Learning objectives**

- I can explain specification gaming with two real examples and say how it differs from a bug in the RL algorithm.
- I can build an environment with a misspecified reward, show an agent exploiting it, and fix the reward so the intended behaviour returns.
- I can measure reward-model overoptimization in RLHF and compare two mitigations: the KL penalty from H2.3 and a reward-model ensemble.

**Topics** (in teaching order)

| Id | Topic | Level | Hours |
|---|---|---|---|
| H2.4 | `[RAI]` Reward hacking and specification gaming | Depth | 1.5 |

**Hours (`[RAI]` share):** reading 0.5 · coding 1 · project 0.25 · **total 1.75**

**Labs** (the H2.4 lab in `PART_H.md`)

- A gridworld that rewards touching a respawning coin instead of reaching the goal: Q-learning (H1.3)
  circles the coin forever. Fix the reward, then find a subtler misspecification (a step penalty so
  harsh that walking into the pit early is optimal).
- Back to the `β = 0` run from H2.3: find what the policy exploited in the reward model and plot the
  gap between reward-model score and true-task accuracy during training (Goodhart's law in one plot).
  Mitigate with the KL penalty and with an ensemble of two reward models, and report how much each
  closes the gap. Write a "known failure modes" section for the H2.3 model in model-card style.

**Project milestone: P11 due.** State the reward you used in its model card, and any hacking you saw
(about 0.25 h).

**Prerequisites:** H1.3 (Q-learning), H2.3 (RLHF with a KL penalty), F3.2 (reward models, DPO), A6.4
(KL divergence).

**Resources** (optional)

- Amodei et al., "Concrete Problems in AI Safety" (2016): https://arxiv.org/abs/1606.06565 (VERIFY link)
- Gao, Schulman, Hilton, "Scaling Laws for Reward Model Overoptimization" (2022): https://arxiv.org/abs/2210.10760 (VERIFY link)
- Krakovna et al., "Specification gaming: the flip side of AI ingenuity" (DeepMind blog, 2020): https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/ (VERIFY link)

**GPU:** no.

**Used later in:** explaining why LLM evals and reward models can be gamed (I2.4's LLM-as-judge
biases, revisited in L3.2 mocks).

---

## L1–L2 · The capstone's Responsible AI (course weeks 34–35)

**Learning objectives**

- I can write the risk section of a design doc: who could be harmed, which of the four themes apply, and the test or monitor that covers each risk.
- I can ship a capstone model card that covers fairness, privacy, safety and documentation, with evidence for each claim (an eval, a test in CI, a monitor).
- I can answer a system-design interview question's "what could go wrong" part in two minutes, naming concrete checks rather than principles.

**Topics** (in teaching order)

| Id | Topic | Level | Hours |
|---|---|---|---|
| L1.1 | Scoping and a design doc: problem, metric, data, baseline, architecture, risks | Core | 2.5 (about 0.5 `[RAI]`) |
| L2.1 | Deploy, monitor and document: public endpoint, dashboards, model card, cost report | Core | 3 (about 1.75 `[RAI]`) |

**Hours (`[RAI]` share):** reading 0.25 · coding 0 · project 2 · **total 2.25**

**Labs:** no new from-scratch implementation. Reuse your own code: the B4.6 group metrics, the F3.3
and I4.2 test sets, the I4.1 redactor and the J2.2 drift monitor, wired into P12's eval suite in CI
where they apply.

**Project milestone: P12.** The L1.1 design doc names at least one `[RAI]` risk (a fairness slice, personal data or prompt
injection) with its mitigation (week 34). The final model card
covers all four themes, with retention and monitoring commitments (week 35). P12 needs ≥ 70% for CP6.

**Prerequisites:** B4.6, F3.3, I4.1, I4.2, J2.5; plus D2.5, E1.4, G2.4 or H2.4 if the capstone uses
those parts.

**Resources** (optional)

- Mitchell et al., "Model Cards for Model Reporting" (2019): https://arxiv.org/abs/1810.03993 (VERIFY link)
- Hugging Face Hub documentation, "Model Cards": https://huggingface.co/docs/hub/model-cards (VERIFY link)

**GPU:** no for the `[RAI]` work; the capstone itself depends on what you build.

---

## You're ready for the next part when…

Part K has no next part of its own. Use the items for each theme when its last host week ends, and
the whole list before CP6 and before interviews. Answer each item without notes; the answers belong
in your quiz reviews, not here.

1. **Privacy.** Given a table, you can name its direct and quasi-identifiers, decide drop, hash or
   coarsen for each, and explain why a hashed ID is still personal data (B1.2).
2. **Fairness.** You can compute demographic parity and equalized-odds differences by hand from two
   confusion matrices, and explain why both cannot generally hold when base rates differ (B4.6).
3. **Fairness.** You can explain why removing a sensitive column does not remove bias, using a proxy
   from P2 or the E1.4 result (B4.6, E1.4).
4. **Fairness.** You can show an overall-accuracy number hiding a per-group failure, with the table
   that exposes it (D2.5).
5. **Safety.** You can report a hallucination or refusal rate from 25 prompts with its confidence
   interval, and say what decision that interval can and cannot support (F3.3).
6. **Safety and privacy.** You can draw a guardrail pipeline, say which checks are probabilistic, and
   quote your redactor's precision and recall and your guardrail's false-positive rate (I4.1).
7. **Safety.** You can demonstrate an indirect prompt injection that exfiltrates data, explain why it
   is not solvable like SQL injection, and list the defences that cut its success rate in your
   I4.2 lab, with the numbers (I4.2).
8. **Documentation and privacy.** You can say what a model card and a datasheet each contain, explain
   ε in differential privacy in one sentence, and write a retention rule for prompt logs (J2.5).
9. **Depth themes.** You can explain how an image watermark is detected and why simple edits remove it (G2.4), and give one
   example of reward hacking with a fix (H2.4).
10. **Every project from P2 on** has a model card whose limitations section names a real failure you
    measured, not a generic disclaimer.

If two or more items fail, revisit the topic ids your quiz reviews flagged. Items 2, 5 and 7 matter
most for interviews: they are the Responsible AI questions an applied ML or LLM engineer is most
often asked to answer with numbers.
