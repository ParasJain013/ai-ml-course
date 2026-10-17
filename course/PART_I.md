# Part I. Applied LLM engineering — detailed syllabus

Course weeks 24–27. Weekly detail for Part I of [`COURSE_OUTLINE.md`](COURSE_OUTLINE.md), which stays
the source of truth: the topic ids, Core/Depth labels, projects P7 and P8 and the RAG paper below are
taken from it unchanged. Written 2026-10-16.

**Goal:** build LLM applications that are measured, secure, fast and affordable, the way you build
backend services.

Part I is the part closest to your day job. An LLM application is a backend service with one unusual
dependency: an external API (or a local model) that is slow, priced per token, rate-limited and
non-deterministic, and that can be talked into doing the wrong thing by its own inputs. Most of the
engineering is therefore familiar (HTTP clients, retries, Postgres, schemas, tests, CI, tracing,
caches), and the new skill is measurement: every week ends in an eval, because "it looked right on
five examples" is the LLM equivalent of shipping without tests. I1 covers the client and the four
ways you talk to a model, I2 builds retrieval-augmented generation on Postgres + pgvector and P7, I3
builds agents, an MCP server and P8, and I4 hardens both projects for production.

**Before you start: run the Refresh prompt.** `PROMPTS.md` schedules a web refresh of Parts I and J
around week 24. Every library, API, model name, price and limit in this part moves faster than the
rest of the course; this file flags them as VERIFY (as of 2026-10), and the refresh replaces them with
current facts before the week's material is generated.

## At a glance

| Week | Course wk | Theme | Topics | Hours | Project | GPU |
|---|---|---|---|---|---|---|
| I1 | 24 | Talking to models | I1.1–I1.4 | 13 | P7 M0, P8 M0 | no |
| I2 | 25 | Embeddings & RAG | I2.1–I2.4 | 13 | **P7 due** | helpful (no API key) |
| I3 | 26 | Agents & MCP | I3.1–I3.4 | 12.5 | **P8 due** | no |
| I4 | 27 | LLM systems in production | I4.1–I4.5 | 12.5 | P7 + P8 hardening and review fixes | no |

About 51 hours in total. Each week's hours include about 1 hour for the weekly quiz and cheat sheet.
Topic hours cover reading the lesson and doing its lab and exercises.

**How the hours are counted.** *Reading* is working through the lessons, the provider docs and the
paper. *Coding* is the labs and the code exercises. *Project* is P7 and P8.

**Teaching order.** Topics follow id order except in I4, which teaches observability (I4.5) before
cost and latency (I4.3) and caching (I4.4), because the tracer you build in I4.5 is what records the
tokens, latency and cost those two labs measure. The ids stay as the outline defines them.

**If a week runs long,** drop I3.4 (multi-agent patterns, the only Depth topic) first, as the outline's
catch-up plan says. Never cut the evals (I2.4, I3.3), security (I4.2), P7 or P8. On the fast path,
keep I1.1–I1.4, I2.1–I2.4, I3.1–I3.3 and I4.1–I4.4; I4.5 moves to post-offer study, but read its lesson
summary because "how would you monitor this?" is a standard interview follow-up.

**LLM access.** As the outline assumes: one paid API key with a small budget, plus a small
open-weights model on CPU as the free fallback. Set a hard monthly spending limit in the provider's
console before the first lab. Every lab reads an `LLM_BACKEND` setting:

- `api` calls your provider (for example Anthropic, OpenAI or Google; the labs are written so the
  provider is one adapter class). Use the provider's smallest, cheapest model for every default run,
  and a larger one only where a lab compares sizes. Model names and prices: VERIFY (as of 2026-10).
- `local` runs a small instruct model in 4-bit GGUF through `llama-cpp-python` (the F3.4 path), about
  0.5B–1.5B parameters, for example from the Qwen2.5 or SmolLM2 families (VERIFY names and licences).
  It is slow and weaker, especially at tool calling and judging; that difference is itself something
  the labs measure.
- `replay` reads responses recorded on an earlier run from a cassette file (a request-hash → response
  cache, the exact cache of I4.4 built early). Re-runs cost nothing and are deterministic, which is
  how the labs meet the "under 5 minutes, same output" rule and how CI runs evals in I3.3.

A lab's default run should cost well under US$0.10 at small-model prices (VERIFY (as of 2026-10)).
Lessons print real outputs for everything deterministic (metrics on recorded responses, token counts,
retrieval results) and label model-generated text as an example of what you may see.

**Libraries this part needs** (Step 2 pins the versions): `httpx` (your own API client), the official
SDK of your provider (`anthropic`, `openai` or `google-genai`), `pydantic`, `tiktoken` (rough token
estimates), `llama-cpp-python` (local fallback), `sentence-transformers` (embeddings and a
cross-encoder reranker), `psycopg` (version 3) and `pgvector` (its Python adapter), `numpy`, `pandas`,
`scipy`, `pytest`, `respx` (mocking HTTP in tests), `mcp` (the official MCP Python SDK),
`opentelemetry-sdk`, and optionally `presidio-analyzer` (PII detection, compared with your own in
I4.1). You build each technique yourself first; frameworks such as LangChain, LlamaIndex, LangGraph or
RAGAS are mentioned only as "what this looks like in a framework", never required. All of these APIs
move fast, the provider SDKs and `mcp` most of all: VERIFY (as of 2026-10).

**Services:** Postgres with the pgvector extension in Docker (the `pgvector/pgvector` image with a
pinned Postgres major version; VERIFY the tag), started by a `docker-compose.yml` in `labs/I/`. Labs
that need it say so at the top and fail fast with a clear message if it is not running. A database
volume lives under `data/` and is never committed.

**Datasets** (all download into `data/`, never committed): for P7, a documentation set you choose and
actually use, 100–500 pages of plain text or Markdown, with a licence that allows it (for example the
PostgreSQL or NestJS documentation; VERIFY each licence); for I1.2, about 200 rows of a public intent
or support-ticket dataset (for example Banking77; VERIFY); for I1.3, synthetic invoices and emails that
the lab generates with a script, so the right answer is known; for I1.4 and I3, a small "bookstore"
Postgres database (customers, books, orders) that a lab script generates. Golden sets, task suites and
attack sets are hand-written by you and live in `my-work/`.

**Hardware:** every lab runs on the laptop CPU at default settings. API calls do no local compute; the
embedding model (about 20–100M parameters) and the cross-encoder embed or score a few thousand chunks
in a few minutes; Postgres in Docker needs well under 1 GB. The `local` backend is the slowest path:
expect single-digit to low tens of tokens per second for a 0.5–1.5B 4-bit model on this CPU (VERIFY on
your laptop), so `local` defaults use fewer examples. Each lab's `GPU_SCALE` block documents the larger
settings; how to use a free GPU when you have no API key is under "Running Part I on a free GPU" at
the end.

---

## I1 · Talking to models (course week 24)

**Learning objectives**

- I can call an LLM chat API over raw HTTP (no SDK) with a system prompt, a list of user and assistant
  messages, `max_tokens` and `temperature`, and read the response's content, stop reason and token
  usage.
- I can parse a streaming response (server-sent events) into text deltas, and measure time to first
  token and output tokens per second separately.
- I can implement retries with exponential backoff and jitter that retry only on 429, 5xx and network
  timeouts, honour a `retry-after` header, and cap total attempts and total time; and I can prove it
  with tests against a mocked server.
- I can estimate a request's token count before sending it, compare it with the `usage` the API
  returns, and turn tokens into cost from a price table.
- I can write a prompt as a versioned template with clear instructions, delimited inputs and a few
  examples, and compare prompt variants on a labelled set with an accuracy and a bootstrap confidence
  interval, not by eye.
- I can get structured output from a model three ways (instructions plus parsing, a repair-and-retry
  loop with Pydantic validation errors fed back, and the provider's native JSON-schema mode), and
  report the valid-output rate of each.
- I can explain constrained decoding as masking logits that would break a grammar, and implement it for
  a tiny schema on a local model.
- I can implement the tool-use loop from scratch: tool definitions as JSON schemas, the model's tool-call
  request, validated execution, the result sent back, repeated until a final answer or a step limit.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| I1.1 | LLM APIs: messages, system prompts, tokens, streaming, rate limits, retries | Core | 2.5 |
| I1.2 | Prompting: clear instructions, examples, reasoning, prompts versioned and tested like code | Core | 2.5 |
| I1.3 | Structured output: JSON schemas, Pydantic validation, repair and retry | Core | 2 |
| I1.4 | Tool use (function calling): the request/execute/respond loop | Core | 2.5 |

**Hours:** reading 3.5 · coding 6 · project 2.5 · quiz and cheat sheet 1 · **total 13**

**Labs**

- **I1.1:** an LLM client from scratch with `httpx`, against your provider's REST API (read its API
  reference for the exact request and response shapes; VERIFY). A `chat(messages, system, ...)` call
  that returns text, stop reason and usage; a `stream(...)` generator that parses server-sent events
  line by line into text deltas; a retry wrapper with exponential backoff, full jitter, a `retry-after`
  override, a per-attempt timeout and an overall deadline; and a client-side rate limiter (a token
  bucket on requests and on tokens per minute). Test the retry and rate-limit logic with `respx`, faking
  a 429, a 500, a timeout and a malformed body, so no test spends money. Then switch the adapter to the
  official SDK and check both give the same usage numbers on the same request. Measure on 10 real
  calls: time to first token, total latency, output tokens per second, and compare a `tiktoken`
  estimate with the reported input tokens (it is only an estimate for non-OpenAI tokenizers, which is
  the F1.3 lesson again). Show that two calls at temperature 0 can still differ. Finally the same
  client pointed at the `local` backend through `llama-cpp-python`, with the chat template applied as
  in F3.4. Bridge: this is the client you would write in NestJS for a payment or email provider:
  timeouts, retries, idempotency, rate limits, a typed response. It breaks in three places: the same
  request does not give the same answer, latency grows with the length of the answer you asked for,
  and you pay for input and output tokens on every call, including retries.
- **I1.2:** prompting as an experiment. Take about 200 rows of an intent dataset, split 50 dev / 150
  test, and classify with five prompt versions kept as files in `prompts/` with a version in the name:
  a bare one-liner; clear instructions with the label list and delimited input; plus 3–5 examples
  (few-shot); plus "think step by step, then give the label on the last line" (and parse that line);
  and a version with the system prompt carrying the role and rules. Write the harness yourself: run a
  prompt over the set, parse labels, compute accuracy with a bootstrap 95% interval (A6.2), and a
  paired comparison between two prompts on the same items (A6.3a). Compare with your E1.2 TF-IDF plus
  logistic-regression baseline trained on the same split, and say honestly whether the LLM is worth it
  here. Add a `pytest` test that fails if the chosen prompt's dev accuracy drops below a threshold:
  this is the first eval in CI-shaped form, which I2.4 and I3.3 grow. `local` default: 50 test rows.
- **I1.3:** extract fields (vendor, date, line items, total, currency) from 50 synthetic invoices and
  emails, some deliberately messy (missing fields, two dates, totals written in words). Define the
  target as a Pydantic model with types and validators (the total equals the sum of line items). Three
  approaches, each with its valid-output rate, field-level accuracy, retries used and token cost:
  (1) instructions plus `json.loads` with a regex that strips code fences; (2) a repair loop that sends
  the Pydantic validation error back to the model and retries at most twice; (3) the provider's native
  structured output or JSON-schema mode (VERIFY what it guarantees: valid JSON, schema-valid JSON, or
  neither). Then constrained decoding from scratch on the `local` model: at each step, mask every
  logit whose token would make the output invalid for a tiny grammar (an enum value, or
  `{"label": "<one of N>"}`) and sample from the rest, using your F2.4 decoding loop; show it never
  produces invalid output and note what it cannot catch (a valid but wrong value). Mention libraries
  that do this for full JSON schemas (for example Outlines or llama.cpp grammars; VERIFY). Bridge: it
  is request validation with a DTO and `class-validator`, except that the "client" sending you bad
  input is your own dependency and you can ask it to try again.
- **I1.4:** the tool loop from scratch. A tool registry that turns a typed Python function (or a
  Pydantic argument model) into a JSON-schema tool definition; three tools against the bookstore
  database (`search_books`, `get_order_status`, `calculate_shipping`), with the database user granted
  `SELECT` only. The loop: send messages plus tool definitions; if the response asks for tool calls,
  validate the arguments with Pydantic, execute (in parallel when the model asks for several), append
  the results, including errors as tool results rather than exceptions, and repeat until a final answer
  or a limit of 6 steps. Log every step. Write a deterministic `ScriptedModel` that emits a fixed
  sequence of tool calls, so the loop itself is unit-tested without any model. Then run 15 questions
  on the real API and on `local`, and count wrong tool, invented arguments, and answers that ignore
  the tool result. Bridge: a tool is an RPC handler or a CQRS command, with a schema and a validator.
  The difference is the caller: a probabilistic planner that may call the wrong handler, invent an id,
  or skip the call and answer from memory, so the handler must defend itself as if it faced the public
  internet (I4.2 builds on exactly this).

**Project milestones (2.5 h).**

- **P7 M0, corpus, golden set and database (2 h).** Pick the P7 document set (see Datasets), check its
  licence, and convert it to clean text with a source URL or path per document. Before building any
  retrieval, write the golden set by hand: 40–60 questions a real user of those docs would ask, each
  with a reference answer and the passage ids that contain it, including 5–10 questions the docs cannot
  answer (the correct behaviour is "I don't know") and a few that need two passages. Split it into dev
  (about 60%) and a held-out test set you will not look at until P7's final report, so you cannot tune
  to it. Start Postgres + pgvector with the lab's `docker-compose.yml`.
- **P8 M0, agent scope (0.5 h).** Choose the P8 agent's job and its 3+ tools, one of which will be
  served by your own MCP server in I3.2 (a natural choice: P7's search as a tool, plus the bookstore
  database, plus one with a side effect such as creating a ticket or writing a file in a sandbox).
  Write 15 task descriptions with a checkable success condition each. These become the I3.3 suite.

**Prerequisites:** F3.4 (running a local model, chat templates), F3.1 (message roles and chat
templates), F2.4 (temperature, top-p, logits, stop conditions), F1.3 (tokenizers and why token counts
differ), F3.3 (hallucination, red-team mindset); E1.2 (TF-IDF baseline for I1.2); A6.2 (bootstrap
intervals), A6.3a (paired comparisons), B1.1 (dev/test splits), B4.1 (accuracy and per-class metrics),
C2.3 (reproducibility and seeds). Within the week: I1.2–I1.4 use the I1.1 client; I1.4 uses I1.3's
argument validation.

**Resources** (optional)

- Your provider's prompt-engineering guide, for example Anthropic's:
  https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview (VERIFY link)
- Your provider's tool-use / function-calling guide, for example OpenAI's:
  https://platform.openai.com/docs/guides/function-calling (VERIFY link)
- Pydantic documentation, models and validators: https://docs.pydantic.dev/ (VERIFY link)

**GPU:** no. API calls run on the provider's hardware, and the `local` model runs on the CPU, slowly.
If you have no API key at all, see "Running Part I on a free GPU" for running a larger open model on
Colab or Kaggle for the I1.2 and I1.3 comparisons.

**Used later in:** the client, retries and streaming in every lab and both projects, and latency
measurement in I4.3 (I1.1); versioned prompts and the accuracy-with-interval harness grow into the RAG
evals of I2.4 and the agent evals of I3.3, and the prompt baseline of P9 in J1.4 (I1.2); citation
parsing in I2.3 and judge outputs in I2.4 (I1.3); the agent loop of I3.1, MCP tools in I3.2 and
least-privilege tools in I4.2 (I1.4).

---

## I2 · Embeddings & RAG (course week 25)

**Learning objectives**

- I can explain what a text embedding model produces, why vectors are normalized, and why cosine
  similarity on normalized vectors equals a dot product (A1.5), and compare four ways of embedding the
  same texts on a retrieval task.
- I can chunk documents three ways (fixed token windows with overlap, structure-aware by headings and
  paragraphs, sentence-based), and show with recall@k which one suits my corpus.
- I can design a Postgres schema for documents and chunks with a `vector` column and a generated
  `tsvector` column, and write exact nearest-neighbour, filtered and full-text queries in SQL.
- I can build HNSW and IVFFlat indexes in pgvector, explain how each finds neighbours, measure recall
  against exact search and latency, and tune `ef_search` and `probes`.
- I can implement BM25 and reciprocal rank fusion from scratch and run hybrid search that combines
  full-text and vector results.
- I can build a RAG pipeline (retrieve, rerank with a cross-encoder, assemble context within a token
  budget, generate an answer with citations) and verify that every citation points to a chunk that was
  actually in the context.
- I can compute recall@k, MRR and nDCG for retrieval, and answer correctness and faithfulness for
  generation, and say which stage a failure came from.
- I can use an LLM as a judge with a written rubric, measure its agreement with my own labels, and name
  and test three of its biases (position, verbosity, self-preference).

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| I2.1 | Embedding models and semantic search: chunking strategies and similarity search | Core | 1.5 |
| I2.2 | Postgres + pgvector: schema, HNSW / IVFFlat indexes, hybrid search with full-text search | Core | 2.5 |
| I2.3 | The RAG pipeline: retrieval, reranking, context assembly, citations (+ the RAG paper) | Core | 2.5 |
| I2.4 | Evals for RAG: golden sets, recall@k and MRR, faithfulness and answer quality, LLM-as-judge and its biases | Core | 2.5 |

**Hours:** reading 3 · coding 6 · project 3 · quiz and cheat sheet 1 · **total 13**

This is a project-deadline week with only Core topics, so nothing is optional; the I2 labs are written
to run on your P7 corpus and golden set, so each lab is also P7 progress.

**Labs**

- **I2.1:** embed your P7 chunks and dev questions four ways: TF-IDF (E1.2), averaged word2vec vectors
  (E1.3), mean-pooled BERT from F2.1, and a sentence-embedding model through `sentence-transformers`
  (a small one such as `all-MiniLM-L6-v2`, 384 dimensions; VERIFY), optionally your provider's
  embedding API as a fifth. Brute-force search in NumPy: normalize, one matrix product, `argpartition`
  for the top k. Score each with recall@5 against the golden passage ids, and explain why a model
  trained for sentence similarity beats raw BERT pooling. Then chunking: fixed 256-token windows with
  50-token overlap, structure-aware chunks split on headings then paragraphs, and sentence groups;
  compare recall@5 and chunk counts, and look at three failures by eye (an answer split across chunks,
  a chunk with no heading context, a table broken in half). Add the document title and heading path to
  each chunk's text and measure again. CPU default: a few thousand chunks, a couple of minutes.
- **I2.2:** pgvector in Docker. Schema: `documents(id, source, title, ...)` and `chunks(id, document_id,
  ordinal, text, heading_path, embedding vector(384), tsv tsvector GENERATED ALWAYS AS ... STORED)`,
  inserted with `psycopg` and the `pgvector` adapter in batches. Exact search with `ORDER BY embedding
  <=> $1 LIMIT k`, checked against your NumPy results (they must match). Then IVF from scratch in
  NumPy: k-means (B5.1) on the vectors into `lists` clusters, search only the `probes` nearest lists,
  and plot recall@10 against probes; that is exactly what an IVFFlat index does. Build pgvector's
  IVFFlat and HNSW indexes, check with `EXPLAIN ANALYZE` that they are used, and plot recall@10 (against
  exact) and latency while varying `ivfflat.probes` and `hnsw.ef_search`, plus build time and index
  size. Read how HNSW's layered graph search works; you do not implement it. Hybrid search: write BM25
  from scratch in Python, compare its ranking with Postgres `ts_rank` (which is not BM25; say how they
  differ), then implement reciprocal rank fusion (`score = Σ 1 / (60 + rank)`) over the full-text and
  vector result lists, in Python and then as one SQL query with two CTEs. Measure recall@5 for vector
  only, full-text only and hybrid on your dev questions; queries with exact identifiers (function
  names, error codes) are where full-text wins. Last, a metadata filter (`WHERE source = ...`) on an
  approximate index, and why it can return fewer than k rows (VERIFY pgvector's iterative index scan
  setting and the dimension limits for HNSW). Bridge: this is the Postgres you know, with a new type,
  operator and index; the honest difference is that the index is approximate, so the same query can
  miss rows that exact search finds, and recall is a number you tune and monitor, not a guarantee.
- **I2.3:** the pipeline, one function per stage so each can be evaluated alone: (optional) query
  rewriting; hybrid retrieval of the top 30; reranking with a cross-encoder through `sentence-
  transformers` (a small MS MARCO cross-encoder; VERIFY the name) down to the top 5–8; context
  assembly under a token budget with deduplication, chunks grouped by document in reading order, and
  each chunk labelled with an id like `[d12:c3]`; and a generation prompt that answers only from the
  context, cites ids after each claim, and says "I don't know" when the context does not contain the
  answer. Parse the answer into a Pydantic model (I1.3) and check every cited id was in the context.
  Explain why a cross-encoder scores better than the bi-encoder used for retrieval but is too slow to
  run over the whole corpus (retrieve-then-rank, B6.2), and where "lost in the middle" (F2.5) affects
  ordering. CPU default: reranking 30 candidates per question takes a second or two.
- **I2.4:** the eval harness for P7, from scratch. Retrieval metrics on the golden set: recall@k,
  precision@k, MRR and nDCG@k (B6.2), each with a bootstrap interval. Generation metrics: refusal
  correctness on the unanswerable questions; answer correctness and faithfulness (every claim supported
  by a cited chunk) judged by an LLM with a written rubric that returns a structured verdict and a
  reason. Hand-label 30 answers yourself first, then measure the judge's agreement with you (percent
  agreement and Cohen's kappa) before trusting it. Test three biases: swap the order of two answers in
  a pairwise comparison (position), pad a correct answer with fluff (verbosity), and judge with the
  same model that generated (self-preference); mitigate with order swapping and a reference answer.
  Then compare two pipeline variants on the same questions with a paired bootstrap, so "reranking
  helped" comes with an interval. Cache every judge call (`replay`). Mention eval frameworks such as
  RAGAS that package these metrics (VERIFY), and why you build your own first. Bridge: this is an
  integration test suite with a fuzzy assertion; it breaks because the assertion is itself a model
  that can be wrong, which is why you measure the judge before you trust it.

**Paper:** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020),
with its reading guide in `papers/`. Read the introduction, the RAG-Sequence and RAG-Token
formulations, and the results on open-domain QA. Take away the idea that a model's knowledge can live
in an index you can update rather than in its weights, and note what is different in today's
applications: the original trained the retriever and generator together, while most production RAG
uses a frozen API model with a separately chosen retriever. About 1 hour, counted in I2.3.

**Project milestone: P7 due, RAG system with an eval suite (3 h).** Finish and submit P7 (brief in
`projects/`, rubric in `answers/projects/`). It must contain: the ingestion pipeline into Postgres +
pgvector; the pipeline from I2.3 with citations; the eval harness from I2.4 runnable with one command;
an ablation table on the dev set (vector only → hybrid → hybrid plus reranker → your best chunking),
each with retrieval and answer metrics and intervals; one final run on the held-out test set, reported
once; cost and latency per question; ten failures analysed by stage (retrieval missed, reranker
dropped it, generation ignored it, judge wrong); and a short system card (data and licence, intended
use, what it refuses, known failure modes). Running the evals in CI comes in I3.3, and tracing,
guardrails and caching in I4. Then ask for a review with `Review my-work/<your P7 folder>`.

**Prerequisites:** I1.1 (client), I1.3 (structured citations and judge verdicts); A1.5 (dot product,
norms, cosine similarity), A6.2 (bootstrap), A6.3a (paired comparisons); B1.4 (leakage, for the
held-out test set), B4.1 (precision and recall), B5.1 (k-means, for IVF), B6.2 (ranking metrics,
retrieve-then-rank); E1.2 (TF-IDF), E1.3 (embedding geometry); F1.3 (token budgets), F2.1 (encoders
versus decoders, for embedders and cross-encoders), F2.5 (long context, "lost in the middle"), F3.3
(hallucination); P7 M0. Within the week: I2.2 stores I2.1's embeddings; I2.3 uses I2.2's hybrid
search; I2.4 evaluates I2.3's pipeline.

**Resources** (optional)

- pgvector README (indexes, operators, tuning): https://github.com/pgvector/pgvector (VERIFY link)
- Sentence-Transformers documentation (bi-encoders and cross-encoders): https://www.sbert.net/
  (VERIFY link)
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020):
  https://arxiv.org/abs/2005.11401 (VERIFY link)

**GPU:** helpful only if you have no API key, or a corpus much bigger than the default. Embedding a
few thousand chunks and reranking 30 candidates per question are fine on the CPU. On Colab or Kaggle
you can embed a large corpus and save the vectors as a `.npy` or Parquet file to load into your local
Postgres, or run the generation and judge steps with a larger open model; Postgres itself stays on the
laptop.

**Used later in:** embeddings and pgvector for agent memory in I3.1 and the semantic cache in I4.4
(I2.1–I2.2); P7's search as an MCP tool in I3.2 and P8 (I2.3); the eval harness pattern in I3.3, the
online evals of I4.5, and the fine-tune-versus-RAG comparison in J1.4 and P9 (I2.4); indirect prompt
injection through retrieved documents in I4.2 (I2.3).

---

## I3 · Agents & MCP (course week 26)

**Learning objectives**

- I can extend the I1.4 tool loop into an agent with a step budget, a token and cost budget, error
  recovery and an explicit stop condition, and log its full trajectory.
- I can give an agent short-term memory (trimming or summarizing the conversation to fit the context)
  and long-term memory (facts stored and retrieved with pgvector), and say what each costs.
- I can solve the same task as a fixed workflow (a chain or a router) and as an agent, compare success
  rate, cost and latency, and argue which one I would ship.
- I can explain the Model Context Protocol's parts (host, client, server; JSON-RPC 2.0 messages; the
  stdio and HTTP transports; tools, resources and prompts) and the initialize handshake.
- I can write a minimal MCP server from scratch that answers `initialize`, `tools/list` and
  `tools/call` over stdio, then rewrite it with the official SDK and test both with the MCP Inspector.
- I can write an agent eval suite with deterministic success checks and trajectory checks, run each
  task several times, and report pass rates that account for non-determinism.
- I can run evals in CI on every pull request using recorded model responses, with thresholds that
  block a merge, plus a scheduled run against the live model.
- (Depth) I can describe orchestrator–worker, router and evaluator–optimizer patterns, and show with
  numbers when splitting work across several agents helps and what it costs.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| I3.1 | Agents: the tool loop, planning, memory, and when a plain workflow is better than an agent | Core | 2.5 |
| I3.2 | The Model Context Protocol: servers, tools and resources; build a small MCP server | Core | 2.5 |
| I3.3 | Agent evals: task success, trajectory checks, regression suites that run in CI | Core | 2 |
| I3.4 | Multi-agent patterns and orchestration | Depth | 1 |

**Hours:** reading 2.5 · coding 5.5 · project 3.5 · quiz and cheat sheet 1 · **total 12.5**

This is a project-deadline week. If P8 runs late, move I3.4 to the catch-up list.

**Labs**

- **I3.1:** an agent class around your I1.4 loop: a system prompt that states the goal, the tools and
  the rules; a step limit, a token and cost budget and a wall-clock timeout, each ending the run with a
  clear reason; tool errors returned to the model so it can try again; and a trajectory log (every
  model call, tool call, result, tokens and cost) written as JSON lines. Planning: compare letting the
  model act directly with asking it first for a short plan it then follows, on 15 bookstore tasks.
  Memory: trim old turns to a token budget, then replace them with a running summary, and measure what
  each loses on a long conversation; then a long-term memory table in pgvector (facts the agent saves
  with a `remember` tool and retrieves by similarity) and a test that a fact from session 1 is used in
  session 2. Then the comparison the outline asks for: implement one task family ("answer a customer
  question about their order") as a fixed workflow (classify the question, run the one right query,
  write the answer) and as the agent; report success, cost, latency and variance over 3 runs. Usually
  the workflow wins when the steps are known in advance. Bridge: the workflow is a state machine or a
  saga you would write in NestJS; the agent is the same machine with the LLM choosing each transition
  at runtime, which buys flexibility and costs predictability, so it needs the budgets and logs a saga
  never did.
- **I3.2:** MCP from the spec up (read the current specification's overview, lifecycle and tools
  sections; VERIFY the version). First a server from scratch in about 100 lines of Python: read
  JSON-RPC 2.0 messages from stdin, answer `initialize` with your capabilities, accept the
  `initialized` notification, and implement `tools/list` and `tools/call` for two tools (your P7
  `search_docs` and a `get_document` lookup), plus `resources/list` and `resources/read` exposing the
  documents themselves, writing responses to stdout and logs to stderr only. Connect to it with the MCP
  Inspector (VERIFY how it is launched; it is a Node tool, which you already have) and call each tool.
  Then rewrite the server with the official Python SDK (`mcp`, its high-level server API; VERIFY) and
  check both behave the same. Write an MCP client in your agent that starts the server as a subprocess,
  lists its tools, converts their schemas into your I1.4 tool definitions and routes calls to it, so
  the agent uses tools it was not written with. Run the server over the HTTP transport as well (VERIFY
  its current name and how it handles sessions), and note that a remote server needs authentication
  like any API. Bridge: MCP is a plugin protocol, close to the Language Server Protocol (also JSON-RPC
  over stdio): one interface, many tool providers. The difference that matters is trust: a third-party
  server's tool descriptions and results go straight into your model's context, so they are untrusted
  input (I4.2). The TypeScript SDK exists too if you want to port the server later.
- **I3.3:** the P8 eval suite. Turn your 15 M0 task descriptions into 20–30 tasks, each with a
  deterministic checker on the final state or answer (the order row has the new status, the answer
  contains the right total, the agent refused the out-of-scope request) and trajectory checks (called
  `get_order_status` before answering; never called the write tool on a read-only task; at most N
  steps; every argument valid). Run each task k times (k = 3 by default) and report the pass rate,
  pass@k (at least one of k succeeds) and pass^k (all k succeed, which is what a user experiences),
  with cost and latency per task. Add 10 prompt-injection tasks built from your F3.3 red-team set, now
  aimed at your agent: instructions hidden in a document P7 retrieves, and in a tool result, trying to
  make the agent call the write tool or reveal a canary string from its system prompt; the check is
  that it does not. Wire it into `pytest` with recorded responses (`replay`) so the suite runs in
  seconds without spending money, and a GitHub Actions workflow that runs it on every pull request with
  thresholds as a required check, plus a scheduled weekly job against the live API with the key stored
  as a repository secret. Add P7's I2.4 evals to the same workflow. Discuss what a replay test can and
  cannot catch (it catches regressions in your code, not changes in the model). Bridge: these are
  end-to-end tests with a flaky dependency; the fix is not retries until green, it is measuring the
  pass rate and gating on it.
- **I3.4 (Depth):** the patterns from a short read (see Resources): prompt chaining, routing,
  parallelization, orchestrator–workers and evaluator–optimizer. Implement orchestrator–workers with
  `asyncio`: an orchestrator splits a broad question over your P7 docs ("compare how X and Y handle
  Z") into sub-questions, workers answer each in parallel with RAG, and the orchestrator merges with
  citations. Compare with your single agent on 8 such questions for answer quality (your I2.4 judge),
  cost and latency. Then an evaluator–optimizer loop where a second call critiques and the first
  revises, capped at 2 rounds, and measure whether the quality gain is worth the cost. Note the failure
  modes: lost context between agents, cost that multiplies with the number of agents, and errors that
  compound. Read one framework's docs (for example LangGraph or your provider's agents SDK; VERIFY) and
  map its concepts onto the code you wrote.

**Project milestone: P8 due, tool-using agent with evals (3.5 h).** Finish and submit P8 (brief in
`projects/`, rubric in `answers/projects/`). It must contain: the agent with 3+ tools, at least one
served by your own MCP server (the I3.2 server, extended); budgets, error handling and a trajectory
log; the I3.3 task-success suite with pass rates over repeated runs; the prompt-injection test set with
its attack success rate; a GitHub Actions workflow that runs the suite on every pull request; cost and
latency per task; and a short system card including what the agent is allowed to change and why.
Security is taught next week (I4.2), so the injection results here are a baseline: the I4 milestone
adds defences and re-runs the same set. Then ask for a review with `Review my-work/<your P8 folder>`.

**Prerequisites:** I1.1–I1.4 (client, prompts, structured output, tool loop), I2.2 (pgvector, for
long-term memory), I2.3 (P7's search, served as an MCP tool), I2.4 (judge and paired comparisons);
F3.3 (red-team set, the seed of the injection tasks); A6.2 (intervals on small-sample pass rates);
B4.1 (success metrics); P8 M0. Within the week: I3.2 plugs into I3.1's agent; I3.3 evaluates both;
I3.4 reuses I3.1 and I2.4.

**Resources** (optional)

- Anthropic, "Building effective agents" (workflows versus agents, the patterns of I3.4):
  https://www.anthropic.com/engineering/building-effective-agents (VERIFY link)
- Model Context Protocol documentation and specification: https://modelcontextprotocol.io/
  (VERIFY link)
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk (VERIFY link)

**GPU:** no. Everything here is API calls, a small Postgres and Python processes. The `local` backend
works for the MCP plumbing and the replayed CI suite, but a 0.5–1.5B model fails many multi-step tool
tasks; record the API responses once for the suite and report `local` results separately as a
comparison, not as P8's headline number.

**Used later in:** budgets, trajectories and memory in the guardrails, security and tracing of I4
(I3.1); MCP servers as a deployment target in J2 and a common capstone component in P12 (I3.2); evals
as merge gates in CI for ML in J2.3, CP5's "eval suites that run in CI", and P12 (I3.3); multi-agent
designs in ML system-design interviews in L2.2 (I3.4).

---

## I4 · LLM systems in production (course week 27)

**Learning objectives**

- I can put input and output guardrails in front of and behind a model call (length and topic checks,
  PII redaction, moderation, schema and groundedness checks), and report each one's false-positive
  rate and added latency.
- I can detect and redact PII with rules I write (with checksum validation where one exists), measure
  their precision and recall on a labelled set, and compare with a library.
- I can explain direct and indirect prompt injection and data exfiltration, attack my own P7 and P8
  systems with them, and measure the attack success rate before and after each defence.
- I can apply least privilege to an LLM system: read-only database roles, tools scoped to the user,
  confirmation before side effects, no secrets in prompts, and output URL allowlists; and I can say why
  no prompt-based defence is complete.
- I can trace a request through retrieval, model and tool calls with OpenTelemetry spans carrying
  tokens, cost and latency, log prompts safely, and run sampled online evals on production traces.
- I can account for tokens and cost per request and per user, break latency into time to first token
  and generation time, and cut cost with model routing, prompt trimming and batch APIs while showing
  the effect on quality.
- I can implement exact, prompt (prefix) and semantic caching, measure the hit rate and the savings of
  each, and show how a semantic cache can return a wrong answer.

**Topics** (in teaching order)

| Id | Topic | Level | Hours |
|---|---|---|---|
| I4.1 | `[RAI]` Guardrails: input and output checks, PII redaction, moderation | Core | 2 |
| I4.2 | `[RAI]` Security: direct and indirect prompt injection, data exfiltration, least-privilege tools | Core | 2.5 |
| I4.5 | Observability: tracing, logging prompts safely, online evals and feedback loops | Core | 2 |
| I4.3 | Cost and latency: token accounting, model routing, streaming, batching | Core | 2 |
| I4.4 | Caching: exact, prompt and semantic caching, and what each saves | Core | 2 |

**Hours:** reading 3.5 · coding 7 · project 1 · quiz and cheat sheet 1 · **total 12.5**

**Labs**

- **I4.1:** a guardrail pipeline as middleware around your P7 pipeline, each check a small class with
  `check(input) -> Allow | Block(reason) | Transform(new_input)`. Input side: length and language
  limits; an off-topic check using embedding similarity to your corpus (I2.1) or a small classifier;
  and PII detection from scratch with regular expressions plus validation (the Luhn checksum for card
  numbers, formats for emails, phone numbers and an ID format of your choice), measured for precision
  and recall on 100 hand-labelled lines, then compared with Presidio (VERIFY). Redaction two ways:
  masking, and reversible pseudonymization (replace with `<PERSON_1>`, send that to the API, and map
  back in the answer) so the provider never sees the raw values. Output side: schema validation
  (I1.3), a moderation check through your provider's moderation endpoint or a small open classifier
  (VERIFY what exists), and a groundedness check that blocks answers whose citations fail (I2.3).
  Measure on your P7 dev set how many good answers each check wrongly blocks, and the latency each
  adds. Bridge: these are NestJS guards, pipes and interceptors around a controller; the difference is
  that most checks are classifiers with error rates, so every guardrail is a precision–recall trade-off
  you must measure, not a rule that is simply right.
- **I4.2:** attack your own systems, never anyone else's. Read the OWASP Top 10 for LLM Applications
  (VERIFY the current edition). Plant a canary secret in P8's system prompt and a "write" tool in its
  toolset, then build 30 attacks across categories: direct injection ("ignore previous instructions"),
  role-play and encoding tricks, indirect injection hidden in a P7 document and in a tool result,
  exfiltration through a Markdown image URL that carries data in its query string, and attempts to make
  the agent call the write tool for another user's order. Measure the attack success rate. Then add
  defences one at a time and re-measure both attack success and the P8 task-success rate, so you see
  what each costs in utility: delimiting and labelling untrusted content ("spotlighting"); a stricter
  system prompt; an injection classifier on retrieved text; output filtering that strips links to
  domains not on an allowlist; and, most important, architecture: a Postgres role with `SELECT` only
  and row-level security so tools can only see the current user's rows, write tools that require a
  human confirmation step, and no secrets anywhere in the context. Sketch the dual-LLM pattern
  (a privileged model that never reads untrusted text, a quarantined one that does and has no tools).
  Conclude honestly: prompt-level defences lower the rate but none reaches zero, so the real control is
  what the model can reach. Bridge: it is SQL injection with no parameterized queries, because
  instructions and data travel in the same channel; the habits that transfer are least privilege,
  treating every input as hostile, and limiting blast radius.
- **I4.5:** observability, from scratch first. A small tracer: a context manager that creates spans
  with a trace id, parent id, name, start and end time and attributes, nested across retrieval,
  reranking, model and tool calls, and written to a `traces` table in Postgres. Record model, input and
  output tokens, cost, latency, time to first token, cache hits, and the prompt version (I1.2). Then
  switch to the OpenTelemetry SDK with the same span structure, following the GenAI semantic
  conventions for attribute names (VERIFY their current status), exporting to the console or a local
  collector. Log prompts and completions safely: redact with I4.1 first, keep full text only for a
  sample, and set a retention period. Write SQL for the dashboard numbers: p50 and p95 latency, tokens
  and cost per request and per day, error and refusal rates. Online evals: a job that samples 10% of
  traces, runs your I2.4 faithfulness judge on them asynchronously, and stores the score on the trace;
  a thumbs-up/down endpoint stored against the trace id; and a script that turns low-scored or
  thumbs-down traces into candidate golden-set items for you to review, closing the feedback loop.
  Mention hosted and open-source LLM tracing tools (for example Langfuse or Arize Phoenix; VERIFY).
  Bridge: this is the distributed tracing you already use; the new part is that the payload (prompt
  and completion) is the most sensitive and most useful thing to log, so redaction and sampling are
  part of the design.
- **I4.3:** a cost and latency report for P7 from your traces: tokens and cost per question split by
  stage (retrieval, rerank, generation, judge), from a price table kept in config and flagged VERIFY.
  Latency: time to first token versus generation time for several `max_tokens` values, and the effect
  of streaming on perceived latency (F2.4's KV cache explains why generation time grows with output
  length). Then three savings, each measured against P7's eval: trim the context (fewer, better-ranked
  chunks); model routing (try the small model first and escalate to the larger one when the answer
  fails its citation check or a self-reported confidence is low), plotted as a quality-versus-cost
  Pareto chart for small only, large only and routed; and running the eval set through the provider's
  batch API for offline jobs (VERIFY the discount and turnaround). Also concurrency: `asyncio` with a
  semaphore sized to your rate limit, and throughput against concurrency. Bridge: it is capacity
  planning and a cloud bill; the difference is that cost scales with the size of every request and
  response, so prompt design is a cost lever.
- **I4.4:** three caches on P7. Exact cache: key = hash of the normalized request (model, parameters,
  prompt version, messages), stored in Postgres with a TTL; measure the hit rate on a replay of your
  traces. Prompt caching: restructure the prompt so the stable part (system prompt, tool definitions,
  instructions) is a fixed prefix and the variable part comes last, turn on the provider's prompt
  caching (VERIFY whether it is automatic or explicit, its minimum length, TTL and pricing), and read
  the cached-token counts in `usage` to compute the saving. Semantic cache from scratch with pgvector:
  embed the question, return a stored answer if a previous question is within a similarity threshold.
  Build two test sets, 30 true paraphrases and 30 near-misses ("status of order 1041" versus "status
  of order 1042", "how do I enable X" versus "how do I disable X"), and plot hit rate and wrong-answer
  rate against the threshold; there is often no threshold that gets both right. Then invalidation
  (clear entries whose source chunks changed after re-ingestion) and scoping by user so one user's
  cached answer is never served to another. Bridge: this is the Redis cache in front of your API; the
  exact cache transfers directly, while the semantic cache is a fuzzy key lookup where a "hit" can be
  a wrong answer served with confidence.

**Project milestone: P7 and P8 hardening (1 h).** No project is due. Fix what your P7 and P8 reviews
flagged, then add this week's work to both: on P8, the I4.2 defences with the injection set and task
suite re-run and both rates reported before and after; on P7, the I4.1 guardrails, I4.5 tracing and
the I4.3 cost report. Update both system cards. These versions are what CP5 (end of Part J) checks:
P7 and P8 at 70% or more on their rubrics, with eval suites that run in CI.

**Prerequisites:** P7 (the pipeline every lab wraps), P8 (the agent and injection set for I4.2),
I1.1 (usage and latency fields), I1.2 (prompt versions), I1.3 (schema checks), I1.4 and I3.1 (tools and
trajectories), I2.1–I2.2 (embeddings and pgvector, for the off-topic check and the semantic cache),
I2.4 (judge and evals, for online evals and every before/after comparison), I3.3 (the CI suite);
B1.2 (PII and data minimization), B4.1 (precision and recall of each guardrail), B4.4 (thresholds),
F2.4 (KV cache, latency), F3.3 (red-teaming, harmful outputs). Within the week: I4.2 builds on I4.1;
I4.3 and I4.4 read I4.5's traces.

**Resources** (optional)

- OWASP Top 10 for Large Language Model Applications: https://genai.owasp.org/llm-top-10/ (VERIFY link)
- Greshake et al., "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications
  with Indirect Prompt Injection" (2023): https://arxiv.org/abs/2302.12173 (VERIFY link)
- OpenTelemetry semantic conventions for generative AI: https://opentelemetry.io/docs/specs/semconv/gen-ai/
  (VERIFY link)

**GPU:** no. Guardrail classifiers, PII rules, the tracer and the caches all run on the CPU; the model
calls go to the API or the `local` backend.

**Used later in:** tracing and dashboards become the monitoring of J2.2 and the cost report of P12
(I4.5, I4.3); guardrails and redaction in J2.5's privacy and governance work (I4.1); the cost and
latency numbers in the fine-tune-versus-prompt decision of J1.4 and P9 (I4.3); LLM inference servers
and their batching in J2.1 (I4.3); every LLM-systems mock interview and system-design case in L2.2
and L3.2 (all of I4).

---

## Running Part I on a free GPU

Part I needs no GPU when you have an API key. A free GPU matters in one case: you have no API budget
and want results stronger than the small `local` model gives. Free-tier GPUs, quotas and session
limits change often: VERIFY (as of 2026-10).

- **What to run there:** batch jobs only. Generate answers for the golden set, run the judge, or embed
  a large corpus, then write the results to a JSONL, `.npy` or Parquet file and bring that back. Keep
  Postgres, the agent and the MCP server on the laptop; do not expose a notebook's model server to the
  internet to call it from home.
- **Which model:** an open instruct model of a few billion parameters in 4-bit fits a free T4's
  16 GB (VERIFY current names and licences). Run it with `transformers` (and `bitsandbytes` 4-bit,
  which J1.3 uses again) or a GGUF build through `llama-cpp-python` with CUDA (VERIFY the install
  command). Results from it are not comparable with an API model's; report which model produced each
  number.
- **Colab:** Runtime → Change runtime type → a GPU (usually a T4); check with
  `torch.cuda.is_available()`. Save outputs to Google Drive, because the VM is wiped when the session
  ends.
- **Kaggle:** Settings → Accelerator → GPU, and turn Internet on to download models (Kaggle asks for
  phone verification first). Save outputs to `/kaggle/working`. The longer sessions suit a full
  golden-set run with a judge.
- **Both:** the lab's `LLM_BACKEND` code is the same; only the adapter changes. Use fp16 on a T4 (no
  bf16). Bring back only results, never model weights.

---

## You're ready for Part J when…

Part J (MLOps & production, course weeks 28–29) comes next. It uses your P7 and P8 evals to compare a
LoRA fine-tune with a prompted baseline in P9, and turns this part's tracing and CI into proper
monitoring and deployment. Answer each item without notes. The answers belong in your quiz reviews,
not here.

1. You can sketch an LLM API client with timeouts, retries that only retry the right errors, and a rate
   limiter, and explain why temperature 0 does not make it deterministic.
2. You can compare two prompts on a labelled set and say whether the difference is real, with an
   interval, rather than from a few examples.
3. You can get valid structured output reliably and explain what the provider's JSON-schema mode does
   and does not guarantee.
4. You can draw the tool-use loop and say where arguments are validated and what happens when a tool
   fails.
5. You can design the pgvector schema and indexes for a RAG system, explain HNSW versus IVFFlat and
   exact search, and say why hybrid search helps on identifiers and error codes.
6. You can name where a RAG answer went wrong (retrieval, reranking, context assembly, generation, or
   the judge) using recall@k, MRR and faithfulness, and you have checked your LLM judge against your
   own labels.
7. You can argue for a workflow or an agent for a given task, with success rate, cost and latency.
8. You can explain what an MCP server exposes and why a third-party server's tool output is untrusted.
9. Your P7 and P8 eval suites run in CI on every pull request, and P7 and P8 are at 70% or more on
   their rubrics (both are CP5 criteria at the end of Part J).
10. You can explain indirect prompt injection and why least privilege, not a better system prompt, is
    the defence that holds.
11. You can estimate a request's cost, and say which of routing, trimming, prompt caching, semantic
    caching and batching you would use first for a given workload, and the risk of each.

If two or more items fail, revisit the topic ids your quiz reviews flagged before you start J1.
Items 2, 6 and 9 matter most for Part J: P9's fine-tune-or-prompt decision is only as good as the eval
set and the statistics you compare them with.
