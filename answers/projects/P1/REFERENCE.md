# P1 reference notes and rubric

Brief: [`projects/P1/BRIEF.md`](../../../projects/P1/BRIEF.md). These notes are for the reviewer. The
dataset is the learner's choice, so there is no single reference solution; the review reruns the
learner's scripts and checks the numbers against their own report.

## What a strong submission looks like

- **A dataset and questions that fit together.** The questions in `DATASET.md` are about the population
  the data describes, answerable with the columns available, and specific ("Do hours worked per week
  differ between people with and without a degree?"), not "explore the data". The learner noticed at
  least one limitation of the source at M0 (for example, a census extract from one year, or a city
  export that only contains reported incidents).
- **A pipeline that runs from scratch.** `download.py` → `clean.py` → `explore.py` → `analysis.py` with
  no manual steps, no absolute paths, no notebook-only state. Running `clean.py` twice gives identical
  output (sorted rows, fixed column order, no timestamps in the file). The cleaned file lives under
  `data/` and is not committed.
- **Cleaning decisions that are explained and counted.** The log reads like a changelog: "Dropped 24
  exact duplicate rows (0.05%). `workclass` '?' (5.6%) kept as an explicit category 'Unknown' because
  the missingness is informative; see figure 3." Imputation choices are justified per column and done
  with statistics computed from the data, not from guessed constants. Outliers are inspected, not
  removed by a blanket rule.
- **Figures that each answer a question.** Titles state the takeaway; axes have units; distributions of
  skewed variables use a log scale or are shown both ways; group comparisons show the spread (box or
  violin plots, or points), not just bar heights of means; the correlation matrix uses a fixed −1 to 1
  colour scale and the text says it measures only linear association (A3.3).
- **Real pre-registration.** `hypotheses.md` is committed before `analysis.py` (check `git log --
  my-work/P1/`). Each hypothesis names the test, α, the correction and the smallest meaningful effect.
  Any later change is listed as a dated deviation.
- **Results reported as size plus uncertainty.** A results table with effect, 95% bootstrap CI, p-value
  and corrected decision for every hypothesis. With large samples, almost everything is "significant";
  the strong report says whether the effect is large enough to matter, using the threshold set at
  pre-registration.
- **Honest limits.** The "What this data cannot tell us" section names specific confounders for each
  finding, describes who is missing from the data and how it was collected, and does not slip into causal
  language ("X causes Y", "X leads to Y") anywhere in the report.

## Reference approach and expected numbers

There are no fixed expected numbers, because each learner picks their own dataset. The reviewer
reproduces the learner's numbers instead:

1. Run the four scripts in order in a clean checkout with `uv run`. They should finish in seconds to a
   minute on the laptop. Any error, missing file or hard-coded path is a reproducibility finding.
2. Run `clean.py` twice and compare file hashes (`certutil -hashfile` on Windows, `sha256sum` elsewhere).
3. Check that every number in the README's results table matches what `analysis.py` prints (to the
   reported precision). A mismatch means the report was edited by hand or the scripts changed after.
4. Recompute one bootstrap CI independently (`scipy.stats.bootstrap` with `method="percentile"` and the
   learner's statistic) and check it agrees with theirs up to Monte Carlo noise (the endpoints should
   differ by well under one CI-width-tenth with ≥ 5,000 resamples).
5. Recompute the corrected decisions by hand from the raw p-values (Bonferroni: compare each p with
   α/m; Holm: sort and compare the k-th smallest with α/(m − k + 1)).

Sanity checks that hold for any dataset: bootstrap CI widths shrink roughly like $1/\sqrt{n}$, so a group
of 40 rows next to a group of 4,000 should show a much wider interval; a percentile bootstrap CI for a
difference in means should be close to the Welch t-interval when both groups are large and not extremely
skewed; with tens of thousands of rows, p-values below $10^{-6}$ are common and say little on their own.

## Rubric

| Criterion | Weight | Full marks | Partial | Zero |
|---|---|---|---|---|
| Dataset choice and questions (M0) | 5 | Meets the size, mixed-types and missing-values requirements; source and licence recorded; 3–5 specific, answerable questions; one limitation noted | Requirements met but questions vague or not answerable with the columns | Requirements not met (e.g. < 1,000 rows, no missing values) or no source |
| Reproducible cleaning pipeline | 15 | Scripts run end to end from a fresh clone; `clean.py` idempotent (identical hashes); seeds fixed; data not committed | Runs with one or two manual fixes (a path, an install), or output not byte-identical but equivalent | Does not run, depends on notebook state, or data committed to git |
| Data dictionary and cleaning log | 10 | Every column documented with type, unit, range and % missing before/after; every decision with reason and row counts | Most columns or decisions covered; some reasons or counts missing | Missing, or decisions not explained |
| Exploration | 15 | 6–8 figures meeting the A1.4 rules, the four required kinds present, each with a correct one-sentence takeaway; correlation matrix interpreted correctly | Figures mostly correct, one or two rule violations or weak takeaways, or one required kind missing | Fewer than 4 usable figures, misleading figures, or no takeaways |
| Pre-registration | 10 | Committed before any test (git history shows it); null/alternative, test, α, correction and minimum meaningful effect for each hypothesis; deviations documented | Hypotheses written but incomplete (no correction or no minimum effect), or timing unclear | Written after testing, or hypotheses changed to fit results without saying so |
| Tests and assumptions | 15 | Test suited to each hypothesis and data type; assumptions checked with evidence; permutation test as a check where assumptions are doubtful; code correct | Reasonable tests but assumptions asserted rather than checked, or one test poorly matched | Wrong tests (e.g. t-test on a categorical outcome, paired test on independent groups) or results not reproducible |
| CIs, effect sizes, correction | 15 | Bootstrap CI and effect size for every hypothesis; correction applied correctly; practical significance discussed against the pre-registered threshold | CIs or effect sizes for some hypotheses only, or correction applied but misread | p-values only, or no correction with multiple tests |
| Report | 15 | Clear structure; conclusions match the evidence, in plain language; "What this data cannot tell us" names specific confounders, sampling issues and measurement issues; no causal language | Clear but generic limitations, or one overclaim | Conclusions not supported by the results, causal claims from observational data, or no limitations section |

Score each criterion as full (100% of its weight), partial (40–70%, reviewer's judgement, stated in the
review) or zero. **Pass ≥ 70.**

## Common mistakes and how to spot them in a review

- **Dropping every row with any missing value.** Look at the cleaning log's row counts; on Adult-style
  data this silently removes a non-random subset (people with unknown occupation) and biases every later
  comparison. Ask what the dropped rows look like.
- **Imputing with statistics from the whole dataset, then testing on it.** Not leakage in the ML sense
  yet, but mean imputation shrinks variance and makes tests overconfident. Check whether imputed values
  were included in the tests and whether the log says so.
- **Pre-registration written after the fact.** Compare commit dates of `hypotheses.md` and
  `analysis.py`, and look for hypotheses that suspiciously match the strongest pattern in the figures.
- **Testing many things, reporting the significant ones.** Count the tests in `analysis.py`, not in the
  README; the correction must use the real number.
- **Significance treated as importance.** "Highly significant (p < 0.001)" for a difference of 0.3 hours
  a week. Ask for the effect size against the pre-registered threshold.
- **Wrong unit of analysis.** Repeated measurements of the same entity (several inspections of one
  restaurant) treated as independent rows inflate the sample size. Ask what a row is.
- **Assumption checks that cannot fail.** A normality test on 30,000 rows rejects for any tiny skew, and
  a t-test on large samples is robust to it anyway. A strong learner explains why they did or did not rely
  on the check.
- **Misleading figures.** Truncated bar axes, a log scale without a label, a correlation heatmap with an
  auto-scaled colour bar, pie charts with many slices. Point to the figure and the A1.4 rule.
- **Causal language.** Search the README for "causes", "leads to", "drives", "because of", "impact of".
- **Absolute paths or notebook-only steps.** Run the scripts from a fresh clone; any `C:\Users\...` path
  fails.

## Production questions to ask in the review

- If this dataset were refreshed monthly, which steps of `clean.py` would break first, and what checks
  (schema, ranges, missing-value rates) would you run before trusting a new month?
- Which of your findings would you expect to change if the data came from a different year, region or
  collection method, and how would you find out?
- A product manager wants a single number from this report for a slide. Which number, with what
  uncertainty, and what sentence of caveat must go with it?
- If someone wanted to act on your strongest finding, what experiment would establish whether the
  relationship is causal?
- Does the dataset contain personal or sensitive attributes? Who could be harmed by a careless reading of
  your results, and what in the report guards against it?
