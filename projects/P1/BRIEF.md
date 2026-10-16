# P1 EDA + statistics report

Part A · starts week 03 · due course week 06 · GPU: no · ~8 h of project time

## The goal

Pick a public tabular dataset, clean it with a script you can rerun, explore it with honest plots, and
answer two or three questions about it with proper statistics: hypothesis tests whose assumptions you
checked, bootstrap confidence intervals and effect sizes, and a correction for running several tests.
The deliverable is a short report a colleague could act on, which says clearly what the data supports
and what it cannot tell you. The question it answers is your own: you write it down in week 3, before
you have looked closely at the data.

## Why it matters for the job

Most ML work starts with a dataset nobody has cleaned and a stakeholder question that is vaguer than it
sounds. An interviewer who reads this project learns whether you can turn raw data into a reproducible
clean table, whether your plots inform or mislead, whether you know the difference between "the
difference is statistically significant" and "the difference is large enough to matter", and whether
you state the limits of your own analysis. These are the habits that later stop you from shipping a
model trained on leaked or biased data (P2).

## Data

Requirements for the dataset (from the Part A syllabus): **at least 1,000 rows, a mix of numeric and
categorical columns, and some missing values**. It must be public, and you must record its source and
licence. Candidates (VERIFY availability and licence before you commit to one, as of 2026-10):

- **UCI Adult (Census Income)**, via OpenML. Census records with demographic and work columns and an
  income label; tens of thousands of rows; missing values are coded as `?` in a few categorical
  columns. Source: UCI Machine Learning Repository / OpenML (VERIFY link and licence). Note that it
  contains sensitive attributes (sex, race, native country): if your questions touch them, say in the
  report what the data's collection and labels can and cannot support (`[RAI]`).
- **Ames Housing** (De Cock, 2011). House sales with about 80 numeric and categorical columns and
  structured missing values ("no garage" recorded as missing). About 2,900 rows. Several mirrors exist;
  use one that states the source and licence (VERIFY).
- **An export from a city open-data portal** (service requests, inspections, permits, traffic counts).
  Often messy in realistic ways: inconsistent categories, dates as text, duplicates. Take a fixed date
  range so your numbers are reproducible, and save the exact query or export URL (VERIFY the portal's
  licence terms).

Palmer Penguins appears in the syllabus as a candidate, but it has 344 rows, below the 1,000-row
requirement; use it only to practise. Avoid datasets with no missing values (most "clean" teaching
datasets): cleaning is part of the project.

**How to download:** write `my-work/P1/download.py` that saves the raw file under `data/p1/raw/`
(gitignored), skips the download if the file is already there, and prints the source URL and the
date it was fetched. For OpenML datasets, `sklearn.datasets.fetch_openml(name=..., version=...,
as_frame=True, data_home="data/p1/openml")` is one route (VERIFY the dataset name and version on
openml.org). Never commit the data.

## Milestones

| Milestone | Week | Deliverable | Hours |
|---|---|---|---|
| M0 Pick the dataset | 03 | `DATASET.md`: name, source URL, licence, size, why you chose it, 3–5 questions you want it to answer | 1 |
| M1 Load and clean | 04 | `download.py`, `clean.py` writing `data/p1/clean.csv`; `data_dictionary.md`; `cleaning_log.md` | 2 |
| M2 Explore and pre-register | 05 | `explore.py` and 6–8 figures in `figures/`, each with a one-sentence takeaway; `hypotheses.md` committed **before** any test is run | 2 |
| Due: Test and report | 06 | `analysis.py` running the pre-registered tests; `README.md`, the report | 3 |

Commit after each milestone; the commit history is part of the evidence for pre-registration.

## Requirements

1. **Reproducible pipeline.** From a fresh clone with the pinned environment, `uv run python
   my-work/P1/download.py`, then `clean.py`, `explore.py` and `analysis.py` regenerate every number and
   figure in the report. Running `clean.py` twice gives byte-identical output. All randomness (bootstrap)
   uses `SEED = 0` via `np.random.default_rng`.
2. **Data dictionary** (`data_dictionary.md`): one row per column of the *cleaned* table with name, type,
   meaning, unit, allowed values or range, and % missing before and after cleaning.
3. **Cleaning log** (`cleaning_log.md`): every decision with its reason and its effect on the row count:
   rows dropped (and why), duplicates, type fixes, how missing values were handled per column
   (dropped, imputed with what, or kept as a category), and outliers kept or removed. "Dropped rows with
   missing values" without counts and reasons does not pass.
4. **Exploration**: 6–8 figures that follow the A1.4 rules (labelled axes with units, a title that
   states the takeaway, bar charts from zero, no misleading scales), each with a one-sentence takeaway in
   the report. At least one figure shows a distribution, one a relationship between two numeric
   columns, one a comparison across a categorical column, and one a correlation matrix of the numeric
   columns (A3.3) with a sentence on what it does and does not show.
5. **Pre-registration** (`hypotheses.md`): 2–3 hypotheses, each stated as a testable claim with the
   null and alternative hypothesis, the variable(s), the test you will use, the significance level, the
   correction for multiple comparisons, and the smallest effect you would consider practically
   meaningful. Committed before `analysis.py` exists. Changes after that are allowed only as a dated
   "deviations" section that says what changed and why.
6. **Tests done properly**: for each hypothesis, the assumptions of the chosen test checked and
   reported (for example group sizes, skew, equal variances, expected cell counts); a permutation test as
   a check wherever a parametric test's assumptions are doubtful.
7. **Uncertainty and size, not just p-values**: for each hypothesis, a bootstrap confidence interval for
   the effect (A6.2) and an effect size in the data's units (difference in means or medians, difference in
   proportions) or a standardized one, plus the p-value and the multiple-comparison-corrected decision
   (A6.3).
8. **The report** (`README.md`): question, data (source, licence, what a row is, time range), cleaning
   summary, the figures with their takeaways, results table (hypothesis, test, effect, 95% CI, p-value,
   corrected decision), conclusions in plain language, and a section **"What this data cannot tell us"**
   covering at least: correlation versus causation and the likely confounders, how the data was sampled
   and who is missing from it, and measurement or labelling issues. 1,000–2,000 words.

*Fast path* (if you are on the outline's fast track): one hypothesis, one test and one bootstrap CI, with
requirements 1–4 and 8 unchanged.

## Deliverables

All in `my-work/P1/`:

- `DATASET.md` (M0)
- `download.py`, `clean.py`, `data_dictionary.md`, `cleaning_log.md` (M1)
- `explore.py`, `figures/*.png`, `hypotheses.md` (M2)
- `analysis.py`, `README.md` (due)

Ask for a review after each milestone (`Review my-work/P1/DATASET.md`, and so on), and of the whole
project at the end (`Review my-work/P1/README.md`).

## How it is graded

The full rubric with its scoring guide is in `answers/projects/P1/REFERENCE.md` (you do not open it).
Criteria and weights, out of 100:

| Criterion | Weight |
|---|---|
| Dataset choice and questions (M0) | 5 |
| Reproducible cleaning pipeline | 15 |
| Data dictionary and cleaning log | 10 |
| Exploration: honest, informative figures with takeaways | 15 |
| Pre-registration done before testing | 10 |
| Tests: right choice, assumptions checked | 15 |
| Confidence intervals, effect sizes and multiple-comparison correction | 15 |
| Report: clarity, conclusions that match the evidence, limitations | 15 |

**Pass ≥ 70%.** P1 ≥ 70% is part of checkpoint CP1.

## Stretch goals

- Compare the percentile bootstrap with a BCa interval (`scipy.stats.bootstrap`) on your skewest variable.
- Run a power analysis: how large a difference could your tests reliably detect with your sample sizes?
- Write a small `pytest` file that checks the cleaned table's invariants (no duplicate keys, ranges, no
  missing values where the dictionary says none).
- Repeat one analysis on a second time period or region and discuss whether the result holds.

## Topics used

A1.3, A1.4, A3.3, A5.1, A5.4, A6.1, A6.2, A6.3
