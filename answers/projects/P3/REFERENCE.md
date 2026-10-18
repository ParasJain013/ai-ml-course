# P3 reference notes and rubric

Brief: [`projects/P3/BRIEF.md`](../../../projects/P3/BRIEF.md). These notes are for the reviewer. The learner may
choose the dataset, and an unsupervised project has no single right answer, so the review reruns the learner's
scripts and checks the numbers against their own report; the notes below add what to expect on the default
dataset, UCI Online Retail II.

## What a strong submission looks like

- **Cleaning rules that are counted and argued.** `features.py` prints, for each rule, how many rows it removes:
  rows without a customer id (a large share of the file; they cannot be segmented and the report says so in
  *What this analysis cannot tell you*), cancellations (invoices starting with `C`), non-product stock codes (postage,
  bank charges, manual adjustments), zero or negative prices. A strong learner decides what to do with cancellations
  on purpose: dropped, or turned into a *return share* feature, which is often one of the most informative ones.
- **A per-customer table built without the future.** Recency is measured from a fixed, stated reference date (the
  day after the last invoice, or the end of the period). If the learner uses a time split (stretch goal), features
  for the first period use only first-period rows.
- **Transforms chosen from the data.** Monetary value, frequency and basket size are heavily right-skewed with a few
  wholesale customers far above the rest. A strong learner shows a histogram, applies `log1p` (or a rank/quantile
  transform) before `StandardScaler`, and shows what happens without it: k-means on raw standardized spend tends to
  spend one or more clusters on a handful of extreme customers, and PCA's first component becomes "how big is this
  customer". The learner reports this with a number (the smallest segment's size with and without the log).
- **Views read with the B5.5 caveats.** PCA with the explained-variance ratio and loadings ("PC1 is overall activity:
  frequency, monetary and distinct products load together"); UMAP at two settings with a sentence that group sizes
  and gaps in the picture are not evidence. The learner does not choose $k$ from the UMAP plot.
- **$k$ chosen from evidence and usability.** Elbow and silhouette for $k = 2$–$10$, `n_init ≥ 10`. On RFM-style
  features the elbow is usually soft and silhouette often peaks at small $k$ (2 or 3); a strong learner says so, and
  picks a $k$ in the 3–6 range only if the extra segments are stable and actionable, stating the trade-off.
- **Stability measured, not asserted.** ARI across seeds and bootstrap resamples, median and minimum, with the
  bootstrap comparison done on the customers each resample contains (or by assigning all customers to the
  resample's centroids). The learner reports which segments are stable, for example by the share of each segment's
  customers that stay together across runs.
- **Profiles in original units.** Medians per segment in days, orders and pounds, not z-scores; names a marketing lead
  recognizes; one action per segment with a way to measure its effect (A6.3b).
- **Anomalies triaged by hand.** Two detectors, the overlap of their top 50, and a table of the top 20 with the raw
  values that make each one unusual and a verdict (data error / real but rare / investigate). On Online Retail II
  the top cases usually include very large wholesale buyers, customers with a single huge order (sometimes
  cancelled straight away), and customers with a high return share; a strong learner recognizes that "anomalous"
  here mostly means "very large", and says whether that is useful to the reader.
- **A segment card and a report a non-ML reader can act on.** The *What this analysis cannot tell you* section names
  the excluded customers without ids, the single retailer and period, the absence of causal claims, and the fact
  that segments describe past behaviour.

## Reference approach and expected numbers

No reference run accompanies these notes: the UCI host was unreachable from the environment that generated them,
so **no number below was measured**. At the first P3 review, the reviewer runs the learner's scripts (and, if time
allows, a minimal reference: RFM plus return share, `log1p`, `StandardScaler`, `KMeans(k, n_init=10,
random_state=0)` for $k = 2$–$10$, `IsolationForest(random_state=0)`) and checks the learner's numbers against that
run, not against this page.

What can be said without a run (VERIFY each on the downloaded data, as of 2026-10):

- **Size.** Online Retail II has roughly a million transaction rows and several thousand distinct customer ids
  over two years; a large minority of rows (on the order of a fifth) have no customer id. A per-customer table with
  fewer than about 1,000 rows, or more than about 10,000, suggests a cleaning or grouping bug (for example grouping
  by invoice instead of by customer, or reading the two sheets of the file separately and keeping only one).
- **Two sheets.** The `.xlsx` holds one sheet per year (VERIFY). A learner who reads only the first sheet has half
  the data and a recency feature computed from the wrong reference date.
- **Skew.** Expect monetary value to span several orders of magnitude; without a transform, PCA's PC1 explains a
  large share of variance driven by a few customers, and k-means silhouettes look high for the wrong reason (one
  giant cluster plus a few outliers).
- **Silhouette.** Values on real customer features are usually modest (roughly 0.2–0.45). A silhouette above about
  0.7 on many features usually means one feature dominates or a few extreme points are being separated from
  everyone else; ask the learner which.
- **Stability.** With `n_init=10`, ARI across seeds for a well-chosen small $k$ is typically close to 1; bootstrap ARI is
  lower. A median bootstrap ARI below about 0.6 means the segmentation is not a finding.

## Rubric

| Criterion | Weight | Full marks | Partial | Zero |
|---|---|---|---|---|
| Data and features | 15 | Every cleaning rule counted; ≥ 6 per-customer features, each defined in a sentence; fixed reference date; rebuilds from the raw file with `SEED = 0`; data not committed; ids not printed | Runs, but a rule uncounted, a feature undefined, or the reference date implicit | Does not run, wrong grain (per invoice), data or ids committed |
| Scaling and transforms | 10 | Skew shown; log (or rank) transform then standardization, argued; the effect of skipping the log shown with a number | Transform applied without justification, or effect not shown | No scaling, or k-means on raw units |
| PCA and UMAP views | 10 | Explained-variance ratio and loadings read in words; two UMAP settings; B5.5 caveats stated; no conclusion drawn from cluster sizes or gaps in the 2-D view | Views present but read uncritically, or loadings missing | Missing, or $k$ chosen from the UMAP picture |
| Choice of $k$ | 10 | Elbow and silhouette for $k = 2$–$10$ with `n_init ≥ 10`; choice justified by both and by segment usability; disagreement acknowledged | One of the plots missing, or choice by plot alone | No justification |
| Stability | 15 | ARI over 10 seeds and 20 bootstrap resamples, median and minimum, computed on matching customers; unstable segments reported as such | Seeds only, or bootstrap compared on mismatched rows, or numbers without interpretation | Missing |
| Segment profiles and actions | 10 | Sizes, medians in original units, recognizable names, one measurable action each | Profiles in z-scores, or names without actions | Missing or contradicted by the numbers |
| Anomalies | 15 | Two detectors on standardized features; top-50 overlap; top 20 inspected by hand with raw values and a verdict; how labels would be obtained | Detectors run but triage thin, or only one detector | Missing, or anomalies reported without inspection |
| Segment card and report | 15 | Card with all adapted §6.10 sections incl. composition by country and misuse risks; report for a non-ML reader, numbers match the scripts, *What this analysis cannot tell you* present | Card or report generic, or written for an ML audience | Missing, or numbers that do not match the code |

Score each criterion as full (100% of its weight), partial (40–70%, reviewer's judgement, stated in the review) or
zero. **Pass ≥ 70.**

## Common mistakes and how to spot them in a review

- **Grouping by invoice instead of customer, or keeping rows without an id as one giant "NaN customer".** Check the
  number of rows of the per-customer table and whether any id is missing.
- **Cancellations summed into monetary value without a decision.** Negative quantities silently reduce spend; some
  customers end with negative monetary value. Ask what the learner decided and why.
- **Scaling before the log, or no log at all.** Look at the feature histograms; ask for the smallest segment's size.
- **Choosing $k$ from the UMAP picture** ("I can see five blobs"). B5.5: apparent clusters appear even in noise.
- **Silhouette treated as the answer.** It favours compact, separated, similar-sized blobs; real segments may not be
  that. Ask whether the chosen segments are usable and stable.
- **Stability from one seed comparison**, or ARI between bootstrap clusterings computed on different customers.
- **Profiles in z-scores.** "Segment 2 has frequency +1.3" means nothing to a marketing lead.
- **Anomalies equated with fraud.** Most top anomalies on this data are large legitimate customers or data errors;
  ask for the verdict column.
- **Segments that are really countries.** If one segment is almost entirely one country, the features (or a country
  feature) are driving it; ask whether that was intended, and about the misuse risk in the card.
- **Customer ids in the report or committed CSVs.** `[RAI]`: ask for them to be removed from history if committed.

## Production questions to ask in the review

- The segmentation is refreshed every month. How do you keep segment 3 meaning the same thing from one month to
  the next (fixed centroids, relabelling by matching, or a classifier trained on the first segmentation)?
- A new customer makes their first purchase today. Which segment are they in, and how much should anyone trust that?
- The anomaly list goes to an operations team that can review 30 cases a week. How do you set the cut-off, and how
  do you use their verdicts to measure and improve the detector (precision@30 over time, B5.6)?
- Marketing wants to offer a discount only to the "at risk" segment. How would you show that the discount changes
  their behaviour, rather than just that they were different to begin with?
- What would make you retire this segmentation?
