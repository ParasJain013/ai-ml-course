# P3 Unsupervised project

Part B · starts week 11 · due course week 12 · GPU: no · ~5 h of project time

## The goal

Take a table of raw transactions with no labels, turn it into one row per customer, and find out what kinds of
customers there are and which ones look wrong. You engineer per-customer features, look at the data through PCA
and UMAP, segment it with k-means, check that the segments survive a change of seed and a resample of the
customers, name each segment from its feature profile, and flag unusual customers with two anomaly detectors whose
top cases you inspect by hand. The deliverable is a short report a marketing or operations lead with no ML
background could act on. The question it answers: *which groups of customers should be treated differently, which
records need a human to look at them, and how sure are you?*

## Why it matters for the job

Most data in a company has no label. Segmentation, deduplication, fraud triage and monitoring all start with an
unsupervised analysis, and the hard part is not calling `KMeans`: it is building features that mean something,
resisting the temptation to read structure into a pretty 2-D plot, showing that the segments are stable rather
than an accident of the seed, and saying plainly what the analysis cannot tell you. An interviewer who reads this
project learns whether you can do that, and whether you can explain clusters to someone who will spend money on
them.

## Data

Requirements: **raw events (transactions, orders, sessions) for at least 1,000 entities** that you aggregate
yourself into one row per entity (customer, store, device), with a timestamp, so that recency and tenure features
are possible. It must be public, with its source and licence recorded. Candidates (VERIFY availability, size and
licence before you commit, as of 2026-10):

- **UCI Online Retail II**, the default. All transactions of a UK-based online retailer from December 2009 to
  December 2011: invoice number, product code and description, quantity, unit price, customer id, country and
  timestamp; about a million rows and several thousand customers with an id (VERIFY the counts). Licence CC BY 4.0
  (VERIFY). Known quirks: many rows have no customer id; invoices starting with `C` are cancellations with negative
  quantities; some stock codes are postage, fees or manual adjustments rather than products; a few customers are
  wholesalers whose spend is orders of magnitude above the rest. It is distributed as an `.xlsx` file, and the
  course environment has no Excel reader, so convert it to CSV once with a temporary dependency:
  `uv run --with openpyxl python my-work/P3/download.py` (this does not change `pyproject.toml`).
- **UCI Wholesale customers**: annual spending of 440 clients of a wholesale distributor in six product categories,
  with channel and region. Already one row per client, so it does not exercise the aggregation step and is small
  for anomaly detection; acceptable only on the fast path, and say so in the report.
- **Another public event log** that meets the requirements (website sessions, bike-share trips per station,
  e-commerce orders such as the Olist dataset on Kaggle; VERIFY its licence, some are non-commercial only). Ask
  for a review of M0 before going further.

**How to download:** `my-work/P3/download.py` saves the raw data under `data/p3/` (gitignored), skips the download
if the file exists, checks a SHA-256 checksum you record on the first download, and prints the source and fetch
date. Never commit the data, the per-customer table or a fitted model. Customer ids are pseudonymous but still
personal data (`[RAI]`): never print them in the report; refer to customers by row number or a salted hash.

## Milestones

| Milestone | Week | Deliverable | Hours |
|---|---|---|---|
| M0 Data and features | 11 | `features.py`: cleaning rules (missing ids, cancellations, non-product codes, extreme prices), each with the number of rows it removes; one row per customer with at least 6 features (recency, frequency, monetary value, plus at least three of your own, e.g. tenure, average basket, distinct products, return share); the transform you chose for skewed features and why; `DATA.md` with source, licence, period, who is in the data and who is not | 1 |
| M1 Views and first clusters | 11 | `explore.py`: PCA with the explained-variance ratio and a 2-D view; UMAP at two `n_neighbors` settings; k-means for $k = 2$ to $10$ with the elbow (inertia) and silhouette plots; `FIRST_READING.md`: your first reading of the segments in a few lines, and which $k$ you will take forward and why | 1 |
| Due: Stability, anomalies, report | 12 | `segment.py`: stability across 10 seeds and 20 bootstrap resamples (ARI), segment profiles and names; `anomalies.py`: two detectors from B5.6, their agreement, the top 20 inspected by hand; `SEGMENT_CARD.md`; `README.md` report | 3 |

Commit after each milestone.

## Requirements

1. **Features that mean something.** Every feature is defined in one sentence a non-ML reader understands
   ("days since the customer's last purchase on 2011-12-09"). Each cleaning rule says how many rows it removes. The
   reference date for recency is fixed and stated. `uv run python my-work/P3/features.py` rebuilds the table from
   the raw file; all randomness in the project uses `SEED = 0`.
2. **Scaling chosen on purpose.** Skewed spend and count features are log-transformed (or the choice not to is
   argued), then standardized (B1.3). The README shows what happens to the k-means segments without the log
   transform, in one sentence and one number (for example the size of the smallest segment).
3. **Honest views.** PCA: the explained-variance ratio of the first two components and the loadings of the top
   features on each. UMAP: two `n_neighbors` settings side by side, with the B5.5 caveats written under the figure
   (sizes and distances between groups mean little; structure appears even in noise).
4. **Choosing $k$.** Elbow and silhouette plots for $k = 2$ to $10$ (B5.1), `n_init` of at least 10. The chosen $k$
   is justified by the plots *and* by whether the segments are usable (a segment of 12 customers is not a marketing
   segment), and the README says where the two disagree.
5. **Stability.** For the chosen $k$: ARI between the assignments from 10 different seeds, and between the
   full-data clustering and clusterings of 20 bootstrap resamples (compare on the customers each resample
   contains). Report the median and minimum ARI, and say which segments are stable and which are not. A segment
   that changes membership from run to run is reported as unstable, not named.
6. **Segment profiles.** For each stable segment: size, the median of each feature in original units (not z-scores),
   and a short name and description a marketing lead would recognize ("recent, frequent, mid-spend regulars"),
   plus one concrete action you would suggest and how its effect could be measured (an A/B test, A6.3b).
7. **Anomalies.** Two detectors from B5.6 (for example an isolation forest and the k-nearest-neighbour distance),
   each fitted on the standardized features. Report the overlap of their top 50, and inspect the top 20 of each by
   hand: for each, which raw feature values make it unusual and whether it looks like a data error, a real but rare
   customer (a wholesaler), or something worth investigating. No labels exist, so no PR-AUC; say how you would get
   a few labels if this went into production.
8. **Segment card** (`SEGMENT_CARD.md`, `[RAI]`). The model card of §6.10 in `course/STYLE_GUIDE.md`, adapted to an
   unsupervised model: *Decision threshold* becomes the anomaly cut-off (how many cases a reviewer can look at per
   week) and *Metrics* becomes silhouette, stability (ARI) and segment sizes. *Metrics by group*: segment composition
   by country (or another attribute your data has), and whether a segment is mostly one country. *Ethical
   considerations*: what a segment must not be used for (for example, different prices or credit terms for a segment
   that is a proxy for a country or a protected group), and the privacy of customer ids.
9. **The report** (`README.md`, 800–1,500 words, written for a non-ML reader): what was analysed and over what period,
   each segment with its size, description and suggested action, the anomalies worth a human look, how stable the
   findings are in plain words ("9 of 10 reruns put the same customers in the same segment"), and a section *What
   this analysis cannot tell you* (no causality, no future behaviour, one retailer, one period, customers without an
   id excluded).

*Fast path* (if you are on the outline's fast track): requirements 1–7 and 9 unchanged; one UMAP setting; stability over
seeds only (no bootstrap); the segment card may be half a page.

## Deliverables

All in `my-work/P3/`:

- `download.py`, `features.py`, `DATA.md` (M0)
- `explore.py`, `figures/pca.png`, `figures/umap.png`, `figures/elbow_silhouette.png`, `FIRST_READING.md` (M1)
- `segment.py`, `anomalies.py`, `figures/*.png`, `SEGMENT_CARD.md`, `README.md` (due)

Ask for a review after each milestone (`Review my-work/P3/FIRST_READING.md`, and so on), and of the whole project at
the end (`Review my-work/P3/README.md`).

## How it is graded

The full rubric with its scoring guide is in `answers/projects/P3/REFERENCE.md` (you do not open it). Criteria and
weights, out of 100:

| Criterion | Weight |
|---|---|
| Data cleaning and per-customer features, reproducible | 15 |
| Scaling and transforms chosen on purpose | 10 |
| PCA and UMAP views, read honestly | 10 |
| Choice of $k$: elbow, silhouette and usability | 10 |
| Stability across seeds and resamples | 15 |
| Segment profiles, names and actions | 10 |
| Anomalies: two detectors, top 20 inspected | 15 |
| Segment card and report for a non-ML reader | 15 |

**Pass ≥ 70%.** P3 ≥ 70% is part of checkpoint CP2.

## Stretch goals

- Fit a Gaussian mixture (B5.3) with the number of components chosen by BIC, and compare its hard assignments with
  k-means by ARI. Which customers does it treat as uncertain (no responsibility above 0.6)?
- Run DBSCAN or HDBSCAN (B5.2) on the UMAP embedding *and* on the standardized features, and explain why the two
  results differ.
- Split the data in time: build features on the first year, cluster, then assign the second year's customers to the
  nearest centroid. How many customers change segment, and what does that say about using the segments for a year?
- Write a `pytest` file that checks `features.py` on a hand-made 10-row transaction table, including a cancellation
  and a row without a customer id.

## Topics used

A6.2, A6.3b, B1.2, B1.3, B2.3, B5.1, B5.4, B5.5, B5.6 (stretch: B5.2, B5.3)
