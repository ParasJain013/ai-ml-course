# P2 reference notes and rubric

Brief: [`projects/P2/BRIEF.md`](../../../projects/P2/BRIEF.md). These notes are for the reviewer. The learner may
choose the dataset, so the review reruns the learner's scripts and checks the numbers against their own report;
the notes below add what to expect on the default dataset, UCI Adult.

## What a strong submission looks like

- **Framing that drives every later choice.** `FRAMING.md` names a real decision ("a lender's marketing team sends a
  premium-card offer to people predicted above 50K; a wasted offer costs 2 units, a missed customer 10"), a
  prediction time, and a cost ratio. The metric follows from it (precision at the campaign size, or expected cost
  at a threshold), not from habit. The majority-class baseline and one simple rule (for example "capital-gain > 0"
  or "education-num ≥ 13") are scored on validation. Adult has no time column, so the learner says the
  prediction time is "at the census" and that every feature is a snapshot; a strong learner notes this makes
  temporal leakage impossible to check and limits what the model can claim.
- **One pipeline from the raw dataframe.** `ColumnTransformer` with imputation (or `?` mapped to an explicit
  category), one-hot encoding with `handle_unknown="ignore"`, scaling for the linear model; the whole thing inside
  `Pipeline`, passed to `cross_val_score`/`GridSearchCV`. No `fit` or `fit_transform` on anything but training folds.
  The test split is made first, with a fixed seed, stratified on the label.
- **A leakage audit with reasoning, not a list.** Every column has "known when" and a decision. On Adult the
  interesting rows are: `fnlwgt` (a census sampling weight, not a property of the person: usually dropped, and
  the learner says why); `education` and `education-num` (the same information twice: keep one); `relationship`
  (contains "Husband"/"Wife", a proxy for sex and marital status; a fairness note, not a leak); `capital-gain` and
  `capital-loss` (reported for the same year as the income: arguably part of the outcome; a strong learner flags it
  as a judgement call tied to the framing). PII: no direct identifiers; `sex`, `race`, `native-country` are sensitive
  and the learner decides, with a reason, whether they are model inputs or audit-only columns.
- **A fair comparison.** Both models tuned on the same folds with comparable budgets; the table reports mean and
  standard deviation over folds; the winner is chosen on cross-validation, and the test set is scored once, for the
  winner (scoring both on test is acceptable if the choice was already made and the README says so).
- **A threshold from the costs.** Chosen on out-of-fold predictions by minimizing the expected cost, shown on a plot
  of cost against threshold; the confusion matrix at that threshold on test; not 0.5 by default.
- **SHAP explained in words a manager can follow.** "Being married (relationship = Husband) raises the predicted
  log-odds the most; this is partly household income reported by one spouse, and partly a proxy for sex." The
  learner checks the top features against the leakage audit.
- **A fairness audit with numbers and judgement.** Group sizes and base rates first (on Adult the base rate differs
  strongly between men and women), then selection rate, TPR and FPR per group at the chosen threshold, with
  bootstrap CIs for small groups (several `race` categories are small). The learner separates "the model
  reproduces a base-rate difference in the data" from "the model is worse for a group", and says what the framed
  use implies (an offer campaign versus a credit decision).
- **A model card someone could act on.** Intended use and explicit out-of-scope uses (not for credit or hiring
  decisions; data from one country and one year in the 1990s), metrics by group, known limitations.

## Reference approach and expected numbers

No reference run accompanies these notes: the OpenML and UCI hosts were unreachable from the environment that
generated them, so **no number below was measured**. At the first P2 review, the reviewer runs the learner's
pipeline (and, if time allows, a minimal reference: `LogisticRegression(C=1.0)` and `HistGradientBoostingClassifier()`
on the same folds) and checks the learner's numbers against that run, not against this page.

What can be said without a run:

- **Base rate.** Adult has roughly 24% positives (income > 50K) over the combined train and test files (VERIFY on
  the downloaded data, as of 2026-10). So the majority-class baseline scores about 0.76 accuracy, ROC-AUC 0.5,
  and a recall of 0 for the positive class. Any accuracy below roughly 0.80 is barely better than doing nothing.
- **Ordering.** Expect gradient boosting to beat regularized logistic regression on ROC-AUC by a small margin, and
  both to beat a one-rule baseline by a large one. If the logistic regression beats boosting by a wide margin, look
  for a bug (unscaled features do not explain it; a mis-specified boosting pipeline, a tiny `n_estimators`, or
  categorical codes fed as numbers might). If either model scores close to perfect (ROC-AUC above about 0.97),
  suspect a leak or label in the features.
- **Variability.** With about 48,000 rows and 5 folds, fold-to-fold standard deviations of ROC-AUC are small (in the
  third decimal); differences between the two models smaller than about twice that spread should not be called a
  win without a paired comparison (A6.3a).
- **Bootstrap CI.** With a test set of about 10,000 rows, a 95% bootstrap CI for ROC-AUC is narrow (roughly ±0.01;
  VERIFY on the learner's run); intervals for small `race` groups are several times wider.

## Rubric

| Criterion | Weight | Full marks | Partial | Zero |
|---|---|---|---|---|
| Framing (M0) | 10 | Decision, prediction time, costs (or cost ratio), metric that follows from them, two baselines scored on validation; committed before model code | Framing present but costs vague, metric chosen by habit, or one baseline only | No framing, or written after the models |
| Pipeline | 15 | One `Pipeline` from the raw dataframe; every fitted step inside; reproduces from a fresh clone with `SEED = 0`; data and models not committed | Runs, but one step fitted outside the folds with small effect (e.g. a scaler), or manual fixes needed | Contamination that changes results (selection or target encoding on all rows), does not run, or data committed |
| Leakage audit | 10 | Every column with "known when", family and decision; PII and sensitive columns handled with reasons; updated after SHAP | Most columns covered, reasons thin, or not updated in M2 | Missing, or a clear leak kept without comment |
| Model comparison | 15 | Splitter justified; both models tuned with comparable budgets on the same folds; mean and spread reported; choice made on CV | Tuning uneven, spread missing, or splitter not justified | Chosen on the test set, or one model untuned against a tuned one with conclusions drawn |
| Test result and threshold | 15 | Test scored once; 95% bootstrap CI; threshold chosen from costs on out-of-fold predictions; confusion matrix at it; baselines alongside | CI or cost-based threshold missing, or threshold tuned on test | Multiple test looks used for choices, or no test evaluation |
| Explanation | 10 | SHAP summary correct; top five features explained in plain English; proxies and possible leaks discussed | Plot correct but explanation generic or wrong in one feature | Missing or misread (e.g. SHAP sign read as causal effect) |
| Fairness audit | 10 | Group sizes and base rates; selection rate, TPR, FPR per group at the threshold with CIs; gaps interpreted against the framed use | Metrics computed but no CIs or no interpretation | Missing, or only overall metrics |
| Model card and report | 15 | Card with all required sections, out-of-scope uses and group metrics; report clear, numbers match the scripts, deployment changes named | Card or report incomplete or generic | Missing, or numbers that do not match the code |

Score each criterion as full (100% of its weight), partial (40–70%, reviewer's judgement, stated in the review) or
zero. **Pass ≥ 70.**

## Common mistakes and how to spot them in a review

- **`pd.get_dummies` or `StandardScaler().fit_transform` on the whole dataframe before splitting.** Search the code
  for `fit_transform` and `get_dummies` outside the pipeline. With one-hot it is usually harmless to the score but
  breaks at serving time when a new category arrives; with target encoding or selection it inflates the score.
- **`?` read as a category in some columns and as missing in others.** Check `df.isna().sum()` against
  `(df == "?").sum()`; note that OpenML's version may already convert `?` to missing (VERIFY).
- **Label encoded inconsistently.** Adult's train and test files historically spell the label differently (`>50K`
  and `>50K.`); a learner who merged them without normalizing gets four classes (VERIFY in the downloaded version).
- **Threshold 0.5 without a word.** Ask what the costs in `FRAMING.md` imply.
- **Accuracy as the headline metric.** At 24% positives, ask for the baseline comparison and the positive-class
  recall and precision.
- **Tuning on the test set,** or reporting the test score of every candidate. Check how many times the code calls
  `score` or `predict` on the test split.
- **SHAP read as causation** ("being married causes higher income"). Point to the sentence.
- **Fairness metrics without group sizes or CIs.** A gap of 0.10 in TPR on a group of 30 positives may be noise.
- **Dropping the sensitive attribute and declaring the model fair.** Ask about proxies (`relationship`,
  `marital-status`, `occupation`).
- **A model card that describes Adult as "US adults".** It is a 1994 census extract with filters; ask who is not in it.

## Production questions to ask in the review

- What would you log at prediction time so that a future version of this model can be trained on what the service
  actually saw (train/serve skew)?
- A new `occupation` value appears in production. What does your pipeline do, and how would you know it is
  happening?
- The marketing team changes the campaign size from 5,000 to 20,000. What changes in your threshold and your
  expected precision, and can you answer without retraining?
- Which group's metrics would you monitor in production, how would you get labels for them, and with what delay?
- If this model were proposed for a credit decision instead of a marketing offer, which parts of your fairness audit
  and model card would have to change, and would you sign off?
