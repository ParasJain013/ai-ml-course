# B4.6 Fairness: group metrics (demographic parity, equalized odds), auditing a model, a first model card

**Core** · `[RAI]` · Part B, week 4 (course week 10) · ~2 h · Lab:
[`labs/B/b4_6_fairness_model_cards.py`](../../labs/B/b4_6_fairness_model_cards.py)

## Why it matters

A model with good overall metrics can still fail one group of people much more often than another, and the people it fails rarely
see the validation report. Credit, hiring, insurance, healthcare and policing models have all been found doing this, and regulators increasingly
require an audit and documentation before deployment (the EU AI Act, US anti-discrimination law, sector rules; VERIFY the rules that apply to you,
as of 2026-10). A fairness audit is not a single number. You compute error rates per group, choose which gap matters for the use, accept that some
fairness definitions cannot all hold at once, and write down what you found. That write-up is a **model card**. This lesson computes the standard group
metrics from scratch on UCI Adult, checks them against `fairlearn`, shows the trade-off between definitions, and fills in a first model card. P2's fairness
audit and model card follow this lesson.

## Prerequisite check

- Can you compute selection rate, true positive rate and false positive rate from a confusion matrix? If not, revisit
  [B4.1](b4_1_classification_metrics.md).
- Can you choose a threshold from costs, and explain why a threshold is a policy decision? If not, revisit [B4.2](b4_2_regression_metrics.md).
- Can you compute a bootstrap confidence interval? If not, revisit [A6.2](../A/a6_2_confidence_intervals_bootstrap.md).
- Do you know what a proxy feature is, and why PII columns need review? If not, revisit [B1.2](b1_2_data_cleaning.md) and
  [B1.4](b1_4_data_leakage.md).

## Intuition

Pick a **sensitive attribute** (here `sex`; in other settings race, age band, disability) and split the evaluation rows by it. Then ask three different
questions, each a different idea of "fair":

- **Demographic parity:** does the model select (predict positive for) the same share of each group? It ignores the labels entirely.
- **Equal opportunity:** among people who truly are positive, does the model catch the same share in each group? Equal *true positive rates*.
- **Equalized odds:** equal true positive rates *and* equal false positive rates. People with the same true outcome are treated the same, whatever their group.

The groups in Adult have very different base rates: 31% of men and 11% of women in the data earn over $50K. A good model will then select more men than women
(demographic parity fails) even if it is equally accurate for both. Forcing equal selection rates means flagging women with lower scores and men with higher ones,
which unbalances the error rates. When base rates differ, **you cannot satisfy every definition at once**; the choice depends on which error harms whom. A missed
qualified applicant (a false negative) and a wrongly approved one (a false positive) have different costs, and these fall on different people.

Two common mistakes. "We removed the sensitive column, so the model is fair" (*fairness through unawareness*) fails because other columns encode it: here, `relationship = Husband`
is 100% male. And a gap computed on 13 people is noise. Small groups need confidence intervals before anyone acts on them.

> **Bridge:** a fairness audit is like slicing a latency SLO by region: the global p99 can be fine while one region breaches it every day, and only
> per-slice dashboards show it. **Where it breaks:** for latency every slice wants the same target. For fairness the "targets" (equal selection, equal TPR, equal FPR)
> contradict each other when base rates differ, so someone has to decide which one matters. That decision is a policy choice, made with stakeholders, not an
> engineering default.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{y}$, $\hat{\mathbf{y}}$ | true labels and thresholded predictions (0/1) | `(n,)` |
| $\mathbf{a}$ | sensitive attribute of each row; groups $a \in \mathcal{A}$ | `(n,)` |
| $\pi_a$ | base rate of group $a$: $P(Y = 1 \mid A = a)$ | scalar |
| $\text{SR}_a$ | selection rate $P(\hat{Y} = 1 \mid A = a)$ | scalar |
| $\text{TPR}_a$, $\text{FPR}_a$ | $P(\hat{Y} = 1 \mid Y = 1, A = a)$ and $P(\hat{Y} = 1 \mid Y = 0, A = a)$ | scalars |
| $t_a$ | decision threshold used for group $a$ | scalar |

**Step 1: per-group rates.** For group $a$, restrict the confusion matrix (B4.1) to its rows: $\text{SR}_a = (\text{TP}_a + \text{FP}_a)/n_a$, $\text{TPR}_a = \text{TP}_a/P_a$, $\text{FPR}_a = \text{FP}_a/N_a$.

**Step 2: the gap metrics** (as `fairlearn` defines them, maximum minus minimum over groups):

$$\Delta_{\text{DP}} = \max_a\text{SR}_a - \min_a\text{SR}_a, \quad \Delta_{\text{EOpp}} = \max_a\text{TPR}_a - \min_a\text{TPR}_a, \quad \Delta_{\text{EOdds}} = \max\big(\Delta_{\text{EOpp}},\; \max_a\text{FPR}_a - \min_a\text{FPR}_a\big).$$

A ratio form is also common: $\min_a\text{SR}_a / \max_a\text{SR}_a$. US employment guidance treats a selection-rate ratio below 0.8 as evidence of adverse impact (the "four-fifths rule"; VERIFY as of 2026-10).

**Step 3: why demographic parity and equalized odds conflict.** Split group $a$'s selected people by their true label:

$$\text{SR}_a = \text{TPR}_a\,\pi_a + \text{FPR}_a\,(1 - \pi_a).$$

(Selected people are true positives, a share $\text{TPR}_a$ of the $\pi_a$ positives, plus false positives, a share $\text{FPR}_a$ of the $1 - \pi_a$ negatives.) If equalized odds holds, every group has the same TPR $= T$ and FPR $= F$, so
$\text{SR}_a = F + (T - F)\,\pi_a$. For two groups, $\text{SR}_a - \text{SR}_b = (T - F)(\pi_a - \pi_b)$, so when $\pi_a \ne \pi_b$ the selection rates are equal only if $T = F$: a model that selects positives and negatives at the same rate, which carries no
information about the label. So, except for useless classifiers, **equalized odds and demographic parity cannot both hold when base rates differ**. A related result: with different base rates, a
score cannot be calibrated within every group *and* have equal false positive and false negative rates, unless it predicts perfectly (Kleinberg, Mullainathan and Raghavan, 2016; Chouldechova, 2017).

**Step 4: per-group thresholds (post-processing).** Keep the model's scores and choose a threshold $t_a$ per group so that one rate is equal across groups. For equal selection rates at an overall rate $r$, take $t_a$ = the $(1 - r)$ quantile of group $a$'s scores.
For equal TPR at a level $\tau$, take $t_a$ = the $(1 - \tau)$ quantile of group $a$'s *positive* rows' scores. Hardt, Price and Srebro (2016) showed that randomizing between two thresholds per group can achieve equalized
odds exactly. Per-group thresholds use the sensitive attribute at decision time, which some laws forbid. Whether they are allowed, and which rate to equalize, are questions for the stakeholders and for legal review, not for the data scientist alone.

**Step 5: uncertainty.** A group with $P_a$ positives estimates its TPR with standard error of about $\sqrt{\text{TPR}(1 - \text{TPR})/P_a}$: with 13 positives and TPR 0.5 that is ±0.14. Report bootstrap intervals per group (A6.2) and do not act on gaps inside them.

## Worked example by hand

Two groups of 10 people, with predictions already thresholded:

| Group | Labels $y$ | Predictions $\hat{y}$ |
|---|---|---|
| A | 1 1 1 1 1 0 0 0 0 0 | 1 1 1 1 0 1 0 0 0 0 |
| B | 1 1 0 0 0 0 0 0 0 0 | 1 0 1 0 0 0 0 0 0 0 |

- **Group A:** base rate $5/10 = 0.5$. Selected 5, so SR $= 0.5$. Of 5 positives, 4 selected: TPR $= 0.8$. Of 5 negatives, 1 selected: FPR $= 0.2$.
- **Group B:** base rate $2/10 = 0.2$. Selected 2: SR $= 0.2$. Of 2 positives, 1 selected: TPR $= 0.5$. Of 8 negatives, 1 selected: FPR $= 0.125$.
- **Gaps:** $\Delta_{\text{DP}} = 0.5 - 0.2 = 0.3$; $\Delta_{\text{EOpp}} = 0.8 - 0.5 = 0.3$; FPR gap $0.2 - 0.125 = 0.075$; $\Delta_{\text{EOdds}} = \max(0.3, 0.075) = 0.3$.
- **Step 3 check:** A: $0.8 \times 0.5 + 0.2 \times 0.5 = 0.5$ ✓. B: $0.5 \times 0.2 + 0.125 \times 0.8 = 0.1 + 0.1 = 0.2$ ✓.

```text
== worked example by hand ==
group A: base rate 0.50  selection rate 0.50  TPR 0.80  FPR 0.2000
group B: base rate 0.20  selection rate 0.20  TPR 0.50  FPR 0.1250
demographic parity difference 0.30   equal opportunity difference 0.30   equalized odds difference max(0.30, 0.0750) = 0.30
```

## Key plot

![Three panels of grouped bars, men and women, under three threshold policies. With a single 0.5 threshold, men have a higher selection rate (0.26 against 0.07), TPR (0.60 against 0.45) and FPR (0.10 against 0.02). Equal selection rates (0.19 each) push women's TPR to 0.81 and FPR to 0.12 above men's. Equal TPRs (0.58 each) leave a smaller selection gap (0.24 against 0.10) and a small FPR gap](img/b4_6_threshold_policies.png)

The same scores, three threshold policies. A single threshold favours men on every rate. Equalizing selection rates reverses the error gaps: women now have the
higher TPR *and* the higher FPR, and the TPR gap (0.32) is larger than before. Equalizing TPRs closes the opportunity gap and shrinks the others, but cannot close the selection gap,
because the base rates differ (Step 3).

## From scratch in NumPy

**The data and model.** UCI Adult with B4.1's split and logistic-regression pipeline, which uses `sex` as a feature. Validation rows: 6,500 men and 3,268 women.

```text
== data ==
train (29306, 12)  val (9768, 12)  (test 9768 rows kept aside)
Male  : 6500 validation rows (0.665)   base rate (>50K) 0.3066
Female: 3268 validation rows (0.335)   base rate (>50K) 0.1053

== the model ==
validation ROC-AUC 0.9033   accuracy at 0.5 0.8465
```

**Group rates and gaps.** Step 1 with a boolean mask per group, then Step 2:

```python
for g in np.unique(groups):
    m = groups == g
    yy, pp = y[m], pred[m]
    rows.append({"group": g, "n": int(m.sum()), "base_rate": yy.mean(), "selection_rate": pp.mean(),
                 "TPR": pp[yy == 1].mean(), "FPR": pp[yy == 0].mean()})
```

```text
== from scratch: group metrics ==
           n  base_rate  selection_rate     TPR     FPR
group                                                  
Female  3268     0.1053          0.0667  0.4535  0.0212
Male    6500     0.3066          0.2554  0.6031  0.1016
differences (max - min over groups): demographic parity 0.1887  equal opportunity 0.1496  equalized odds 0.1496
selection-rate ratio (female / male) 0.2612
```

- The model selects women at about a quarter of men's rate (selection-rate ratio 0.26), far below the four-fifths line. Part of that gap is the base rates (11% against 31%).
- The error rates differ too: a woman who does earn over $50K is caught 45% of the time, a man 60% (TPR gap 0.15), and men who do not
  are flagged five times as often (FPR 0.10 against 0.02). Part of this follows from the base rates alone (Step 3 and the Kleinberg–Chouldechova result: a well-calibrated score on groups with different base rates has unequal error rates), so these gaps are not, on their own, proof of extra bias. What they do show is who bears which error under this model.

**Small groups.** Bootstrap 95% intervals for the TPR of each race group (Step 5):

```text
== small groups need confidence intervals ==
White               n  8355  positives  2144  TPR 0.5882  95% CI [0.5679, 0.6063]  width 0.0385
Black               n   936  positives    99  TPR 0.5051  95% CI [0.4040, 0.5960]  width 0.1919
Asian-Pac-Islander  n   302  positives    73  TPR 0.5205  95% CI [0.4110, 0.6301]  width 0.2192
Amer-Indian-Eskimo  n   104  positives    13  TPR 0.5385  95% CI [0.2308, 0.7692]  width 0.5385
Other               n    71  positives     8  TPR 0.2500  95% CI [0.0000, 0.5000]  width 0.5000
```

The TPR of the "Other" group (0.25) looks alarming, but it rests on 8 people, and its interval runs from 0 to 0.5. The honest report is "too few positives to estimate", plus a plan to
collect more data, not a confident gap. The intervals for Black and Asian-Pacific-Islander people (widths about 0.2) are wide enough to hide a gap of 0.08 from the White group's rate.

**One threshold versus per-group thresholds.** Step 4: thresholds that equalize selection rates at the overall rate, and thresholds that equalize TPRs at the overall TPR.

```text
== one threshold versus per-group thresholds ==
single threshold 0.5       thresholds M 0.500 F 0.500   accuracy 0.8465   selection M 0.255 F 0.067   TPR M 0.603 F 0.453   FPR M 0.102 F 0.021
                           DP diff 0.1887   EO diff 0.1496   equalized odds diff 0.1496
equal selection rates      thresholds M 0.611 F 0.117   accuracy 0.8247   selection M 0.192 F 0.192   TPR M 0.489 F 0.808   FPR M 0.061 F 0.120
                           DP diff 0.0002   EO diff 0.3194   equalized odds diff 0.3194
equal true positive rates  thresholds M 0.523 F 0.308   accuracy 0.8429   selection M 0.243 F 0.103   TPR M 0.581 F 0.581   FPR M 0.093 F 0.046
                           DP diff 0.1404   EO diff 0.0004   equalized odds diff 0.0472
```

- Equal selection rates need a threshold of 0.117 for women against 0.611 for men; accuracy drops 2.2 points, and the equal-opportunity gap doubles (in the other direction).
- Equal TPRs cost only 0.4 points of accuracy and leave a 0.047 FPR gap, but the selection gap is still 0.14, as Step 3 says it must be.
- None of these is "the fair answer". Which policy fits depends on the decision, on who bears each error, and on what the law allows.

**Fairness through unawareness.** Retrain without `sex`:

```text
== fairness through unawareness: drop `sex` ==
without `sex`: DP diff 0.1813 (was 0.1887)   equalized odds diff 0.1233 (was 0.1496)
relationship = Husband: 3931 rows, share female 0.0000
relationship = Wife   : 466 rows, share female 1.0000
predicting sex from the other categorical columns: ROC-AUC 0.9274
```

The gaps shrink only a little. `relationship` encodes sex almost perfectly, and the other categorical columns predict it with ROC-AUC 0.93, so the model can still separate the
groups. Dropping the column also makes the audit harder, because you can no longer measure the gaps unless you keep the attribute for evaluation. Keep it for auditing even when the model does not use it.

## With fairlearn

```text
== with fairlearn ==
demographic_parity_difference, equal_opportunity_difference, equalized_odds_difference and MetricFrame match: OK
```

`MetricFrame(metrics={...}, y_true=y, y_pred=pred, sensitive_features=a)` computes any metric per group (`.by_group`), with `.difference()` and `.ratio()` for the gaps. `fairlearn.postprocessing.ThresholdOptimizer`
implements Step 4's randomized per-group thresholds for a chosen constraint, and `fairlearn.reductions` retrains a model under a fairness constraint. These are
fairlearn 0.15.0 APIs (VERIFY as of 2026-10). `equalized_odds_difference` takes the larger of the TPR and FPR gaps, as in Step 2.

**The model card** (template: `course/STYLE_GUIDE.md` §6.10), written by the lab from the numbers above to `outputs/b4_6/MODEL_CARD.md`:

```text
== model card ==
# Model card: Adult income classifier (B4.6 teaching model)

## Model details
Logistic regression (scikit-learn 1.9.1) on 5 standardized numeric and 7 one-hot categorical
features, threshold 0.5. Trained 2026-10-17 by the course lab; not versioned for deployment.

## Intended use
Teaching: auditing a classifier's error rates across groups. Predicts whether a person in the 1994 US census
extract reported income above $50,000.

## Out-of-scope uses
Any decision about a real person: credit, hiring, pay, benefits. The data is from one country and year (1994),
and the label is a proxy for nothing a decision-maker should act on today.

## Data
UCI Adult (Becker & Kohavi, 1996), 48,842 rows, stratified 60/20/20 split; metrics below on the 9,768
validation rows. `fnlwgt` (a sampling weight) and `education` (duplicates `education_num`) excluded; `?` kept
as a category. Men are 67% of rows; the base rate of >$50K is 30.7% for men and
10.5% for women.

## Metrics
Validation ROC-AUC 0.903, accuracy 0.847 (majority-class baseline 0.761).
No test-set result reported: this is a teaching model, and the test rows are kept for P2-style final evaluation.

## Decision threshold
0.5, the default: no costs were defined for this teaching model. A real use would set it from costs (B4.2).

## Metrics by group
| sex | n | base rate | selection rate | TPR | FPR |
|---|---|---|---|---|---|
| Male | 6500 | 0.307 | 0.255 | 0.603 | 0.102 |
| Female | 3268 | 0.105 | 0.067 | 0.453 | 0.021 |

Demographic parity difference 0.189; equal opportunity difference 0.150;
equalized odds difference 0.150. Women with incomes above $50K are recognized less often
(lower TPR), and men without them are flagged more often (higher FPR).

## Known limitations
Small race groups have wide intervals (TPR CI widths up to 0.54). Removing `sex` does not remove
the gaps: `relationship` (Husband/Wife) and other columns predict sex (ROC-AUC 0.93).

## Ethical considerations
The model uses `sex` directly. Per-group thresholds can equalize one rate but move the others, and may be
unlawful in some settings (VERIFY for the jurisdiction); the choice depends on which error harms whom.
```

A card for a real model (P2) also fills in an owner and a version, a test-set result with a CI, a threshold chosen from costs, CIs for small groups, and who was consulted about the trade-offs.

## In practice

- **When to use it:** audit every model that makes or informs decisions about people, before deployment and again in monitoring (J2.2). Decide the sensitive attributes,
  the decision, and which error is worse *before* computing anything.
  Then choose a definition:
  - when the decision gives people something they want (a loan, an interview), equal opportunity (equal TPR among the qualified) is often argued for;
  - when a false positive harms people (fraud flags, extra scrutiny), compare FPRs;
  - demographic parity when the label itself is believed to be biased (historical hiring decisions).
  - Write the reason in the model card.
- **Failure modes:**
  - *Overall metrics only.* Per-group rates are the audit.
  - *Fairness through unawareness.* Proxies remain; keep the attribute for evaluation.
  - *Reading noise as bias* in small groups. Bootstrap CIs; say "insufficient data" when it is true.
  - *Fixing one metric blindly.* Equalizing selection rates here doubled the TPR gap. Report all three gaps for every policy you consider.
  - *Biased labels.* If the historical label reflects past discrimination (who was promoted), every metric computed against it inherits that bias. No metric fixes a biased target.
  - *Thresholds by group without legal review.* Using the attribute at decision time is restricted in some domains.
- **Defaults that work:** `MetricFrame` with selection rate, TPR, FPR and group sizes, plus bootstrap CIs for small groups; report $\Delta_{\text{DP}}$, $\Delta_{\text{EOpp}}$ and $\Delta_{\text{EOdds}}$ at the chosen threshold; a model card from the §6.10 template
  with metrics by group; mitigation (thresholds, reweighting, constrained training) only after deciding, with stakeholders, which gap matters.

## Exercises

Save your work to `my-work/B/b4_6_fairness_model_cards.md` (or `.py`) and ask for a review.

1. **By hand:** A screening model is evaluated on two groups. Group X: 400 people, 120 positives, the model selects 140 of whom 90 are positives. Group Y: 100 people, 10 positives, the model
   selects 15 of whom 6 are positives. (a) Compute each group's base rate, selection rate, TPR and FPR, and the three gaps. (b) Check Step 3's identity for both groups. (c) Suppose a new model
   has TPR 0.7 and FPR 0.1 in *both* groups. What are the selection rates, and what is $\Delta_{\text{DP}}$? (d) Roughly how wide is a 95% interval for group Y's TPR, and what does that mean for your report?
2. **Code:** On the lab's model: (a) audit by `race` with `MetricFrame` (selection rate, TPR, FPR, group size) and add bootstrap 95% CIs for each group's *gap* to the White group's TPR; (b) implement from
   scratch per-group thresholds (by `sex`) that equalize FPR at the overall FPR, and report all three gaps and accuracy; (c) run `fairlearn.postprocessing.ThresholdOptimizer` with
   `constraints="equalized_odds"` on the same model (fit on training rows, evaluated on validation) and compare its gaps and accuracy with your thresholds.
3. **Think:** A bank's new credit model excludes sex, race and age. The product owner says this makes it fair and that no audit is needed. Write the reply you would send: what could still go wrong,
   which data you need for the audit, which gap you would look at first for a lending decision and why, and which sections of the model card would capture the findings.

## Takeaways

1. A fairness audit computes selection rate, TPR and FPR per group at the decision threshold, with group sizes and confidence intervals; overall metrics hide group failures.
2. Demographic parity (equal selection), equal opportunity (equal TPR) and equalized odds (equal TPR and FPR) are different definitions; `fairlearn` reports each as max minus min over groups.
3. $\text{SR}_a = \text{TPR}_a\pi_a + \text{FPR}_a(1 - \pi_a)$: when base rates differ, demographic parity and equalized odds cannot both hold for a useful model, so choosing a definition is a policy decision.
4. Per-group thresholds can equalize one rate, but they move the others and use the attribute at decision time; dropping the attribute does not remove the gaps, because of proxies.
5. A model card (§6.10) records intended and out-of-scope uses, data, metrics overall and by group, the threshold and why, limitations and ethical considerations.

**What builds on this:** [D2.5](../D/d2_5_vision_error_analysis.md) and [E1.4](../E/e1_4_static_embeddings_bias.md) (bias checks for vision and
embeddings); [J2.2](../J/j2_2_monitoring_drift.md) (monitoring group metrics in production); [J2.5](../J/j2_5_documentation_governance.md) (model cards
and datasheets in depth); [L2.1](../L/l2_1_deploy_monitor_document.md) (the capstone's model card); P2 (fairness audit and `MODEL_CARD.md`).

**References:** M. Hardt, E. Price, N. Srebro, "Equality of opportunity in supervised learning", *NeurIPS*, 2016. J. Kleinberg, S. Mullainathan, M. Raghavan,
"Inherent trade-offs in the fair determination of risk scores", arXiv 1609.05807, 2016. A. Chouldechova, "Fair prediction with disparate impact: a study of bias in
recidivism prediction instruments", *Big Data* 5(2), 2017. M. Mitchell et al., "Model cards for model reporting", *FAT\* (FAccT)*, 2019, arXiv 1810.03993 (VERIFY link:
https://arxiv.org/abs/1810.03993). S. Barocas, M. Hardt, A. Narayanan, *Fairness and Machine Learning* (VERIFY link: https://fairmlbook.org). fairlearn documentation
(VERIFY link: https://fairlearn.org).
