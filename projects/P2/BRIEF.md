# P2 End-to-end tabular model

Part B · starts week 07 · due course week 10 · GPU: no · ~9 h of project time

## The goal

Take a tabular classification problem from a raw table to a documented model you could hand to a team. You
frame the problem (who acts on the prediction, and what an error costs), build one scikit-learn `Pipeline` from
the raw dataframe to a prediction, compare a regularized logistic regression with a gradient-boosted challenger,
tune and cross-validate both properly, report the test score once with a confidence interval, and pick a decision
threshold from your costs. Along the way you audit every feature for leakage, explain the model with SHAP values,
audit its errors across a sensitive attribute, and write a model card. The question it answers: *is this model
good enough, and fair enough, for the use you defined, and how do you know?*

## Why it matters for the job

This is the shape of most applied ML work, and of many take-home assignments. An interviewer who reads it
learns whether you frame before you fit, whether your evaluation would survive contact with production (no
leakage, the right split, one look at the test set), whether you can say why a metric and a threshold fit a
business cost, whether you can explain a model to a non-specialist, and whether you check who it fails. The
pipeline, the leakage audit and the model card are the parts most candidates skip, and the parts reviewers
ask about.

## Data

Requirements: a **binary classification** table with **at least 5,000 rows**, a mix of numeric and categorical
columns, some missing values, and **a sensitive attribute** (for example sex, age or race) you can audit in week 10.
It must be public, with its source and licence recorded. Candidates (VERIFY availability and licence before you
commit, as of 2026-10):

- **UCI Adult (Census Income)**, the default. Records from a 1994 US census extract; predict whether income exceeds
  50K USD a year. About 48,000 rows (train and test files combined), 14 columns, missing values coded `?` in
  `workclass`, `occupation` and `native-country`; sensitive attributes `sex` and `race`. Via OpenML:
  `fetch_openml(name="adult", version=2, as_frame=True, data_home="data/p2/openml")` (VERIFY the name, version
  and row count on openml.org). If you used it in P1, you already know its quirks; the modelling questions are
  new. Note that the data, its income threshold and its categories reflect one country and year: say so in the
  model card.
- **UCI Bank Marketing**. Predict whether a client subscribes to a term deposit after a phone campaign; about
  45,000 rows; sensitive attribute `age` (banded) or `marital`. Its documentation warns that the `duration`
  column (call length) is known only after the call and should be dropped for a realistic model (VERIFY in the
  dataset's description): a ready-made leakage case for your audit.
- **Another public table** that meets the requirements, if you have a reason (a domain you know, a problem from
  work you can describe with public data). Ask for a review of M0 before going further.

**How to download:** `my-work/P2/download.py` saves the raw data under `data/p2/` (gitignored), skips the download
if the file exists, and prints the source and fetch date. Never commit the data or a fitted model.

## Milestones

| Milestone | Week | Deliverable | Hours |
|---|---|---|---|
| M0 Frame the problem | 07 | `FRAMING.md`: target, unit of prediction, prediction time, who acts and how, cost of each error type, metric and why, the B1.1 baseline score on a validation split; columns excluded for leakage or PII, one line each | 1 |
| M1 Pipeline and baseline | 08 | `pipeline.py`: one `Pipeline` from raw dataframe to prediction (cleaning, encoding, scaling, regularized logistic regression), scored on validation against the M0 baseline; `LEAKAGE_AUDIT.md` v1: for every feature, when its value is known relative to the prediction time | 2 |
| M2 The challenger | 09 | Gradient boosting (XGBoost or LightGBM) in the same pipeline, compared with M1 on validation; a SHAP summary plot and the top five features explained in plain English; `LEAKAGE_AUDIT.md` updated (is any top feature a leak you missed?) | 2 |
| Due: Evaluate, audit, document | 10 | Both models cross-validated and tuned with the right splitter; the test score reported once, with a bootstrap CI; metric and threshold justified from the M0 costs; fairness audit on the sensitive attribute; `MODEL_CARD.md`; `README.md` report | 4 |

Commit after each milestone. B4 (week 10) is the heaviest week of Part B: if you can, start the M3
cross-validation in week 9.

## Requirements

1. **Framing before fitting.** `FRAMING.md` exists in a commit before any model code. It names the prediction
   time, the decision the prediction drives, an explicit cost (or cost ratio) for a false positive and a false
   negative, and the metric that follows from them. The B1.1 baseline (majority class, and one simple rule) is
   scored on validation.
2. **One pipeline.** Every step that learns from data (imputation, encoding, scaling, feature selection, the
   model) is inside one `Pipeline`/`ColumnTransformer` that takes the raw dataframe. No statistic is computed on
   validation or test rows before scoring. `uv run python my-work/P2/train.py` reproduces every number from a
   fresh clone; all randomness uses `SEED = 0`.
3. **Splits.** A test set is held out at the start and scored **once**, at the end. Model choice and tuning use
   cross-validation on the remaining rows, with a splitter that matches the data (stratified for imbalance;
   grouped or time-ordered if rows share entities or time matters), and the README says why.
4. **Leakage audit** (`LEAKAGE_AUDIT.md`): a table with every input column, when its value is known relative to
   the prediction time, the leakage family it could belong to (if any), and the decision (kept, dropped,
   transformed). PII columns and why they were dropped or kept (`[RAI]`).
5. **Two models compared fairly.** Regularized logistic regression and gradient boosting, each tuned with the
   same budget and the same folds; mean and spread of the cross-validated metric reported for both.
6. **The test result.** The chosen model's test metric with a 95% bootstrap confidence interval (A6.2), next to
   the baselines. The decision threshold chosen from the M0 costs on validation folds (not on the test set), and
   the confusion matrix at that threshold.
7. **Explanation.** A SHAP summary plot for the gradient-boosted model and a plain-English paragraph on the top
   five features, including whether any of them looks like a leak or a proxy for the sensitive attribute.
8. **Fairness audit** (`[RAI]`). For the sensitive attribute: group sizes, base rates, and at the chosen threshold
   the selection rate, true-positive rate and false-positive rate per group, with the gaps between groups and
   bootstrap CIs for the smaller groups. One paragraph on what the gaps mean for the use you framed, and what you
   would do (or not) about them.
9. **Model card** (`MODEL_CARD.md`, `[RAI]`), with at least these sections: model details (type, version,
   date, owner), intended use and out-of-scope uses, data (source, time period, who is in it and who is not),
   metrics overall and by group with CIs, the decision threshold and why, known limitations, and ethical
   considerations. B4.6 (week 10) covers model cards in detail.
10. **The report** (`README.md`): the problem, the data, the pipeline, results table (baselines, both models,
    test score with CI), the threshold, the main findings of the leakage and fairness audits, and what would need
    to change before deployment. 1,000–2,000 words.

*Fast path* (if you are on the outline's fast track): one model (the gradient-boosted one, with the logistic
regression only as a baseline score), default hyperparameters plus one small random search; requirements 1–4 and
6–9 unchanged.

## Deliverables

All in `my-work/P2/`:

- `FRAMING.md` (M0)
- `download.py`, `pipeline.py`, `LEAKAGE_AUDIT.md` (M1)
- `figures/shap_summary.png`, an updated `LEAKAGE_AUDIT.md` (M2)
- `train.py` (cross-validation, tuning and the final test evaluation), `figures/*.png`, `MODEL_CARD.md`,
  `README.md` (due)

Ask for a review after each milestone (`Review my-work/P2/FRAMING.md`, and so on), and of the whole project at the
end (`Review my-work/P2/README.md`).

## How it is graded

The full rubric with its scoring guide is in `answers/projects/P2/REFERENCE.md` (you do not open it). Criteria and
weights, out of 100:

| Criterion | Weight |
|---|---|
| Framing: prediction time, decision, costs, metric, baselines (M0) | 10 |
| One reproducible pipeline, no contamination | 15 |
| Leakage audit, including PII | 10 |
| Model comparison: splitter, tuning, cross-validated spread | 15 |
| Test result with CI, threshold from costs | 15 |
| Explanation (SHAP) in plain English | 10 |
| Fairness audit | 10 |
| Model card and report | 15 |

**Pass ≥ 70%.** P2 ≥ 70% is part of checkpoint CP2.

## Stretch goals

- Calibrate the chosen model (`CalibratedClassifierCV`) and show a reliability diagram before and after.
- Compare SHAP importance with permutation importance on validation rows and explain any disagreement.
- Fit a model without the sensitive attribute and show whether the group gaps change (and why they may not:
  proxies).
- Write a `pytest` file that checks the pipeline on a few hand-made rows, including an unseen category and a
  missing value.

## Topics used

B1.1, B1.2, B1.3, B1.4, B1.5, B2.1, B2.2, B3.3, B3.4, B4.1, B4.3, B4.4, B4.6, A6.2
