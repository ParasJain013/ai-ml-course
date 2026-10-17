# Answer key: B4.6 Fairness: group metrics (demographic parity, equalized odds), auditing a model, a first model card

Lesson: [`lessons/B/b4_6_fairness_model_cards.md`](../../../lessons/B/b4_6_fairness_model_cards.md)

## Exercise 1 (by hand)

**Solution (a).** Group X: $P = 120$, $N = 280$, TP $= 90$, FP $= 140 - 90 = 50$. Base rate $0.3$, SR $= 140/400 = 0.35$, TPR $= 90/120 = 0.75$, FPR $= 50/280 = 0.1786$.
Group Y: $P = 10$, $N = 90$, TP $= 6$, FP $= 9$. Base rate $0.1$, SR $= 0.15$, TPR $= 0.6$, FPR $= 9/90 = 0.1$.
Gaps: $\Delta_{\text{DP}} = 0.35 - 0.15 = 0.20$; $\Delta_{\text{EOpp}} = 0.75 - 0.6 = 0.15$; FPR gap $0.0786$; $\Delta_{\text{EOdds}} = \max(0.15, 0.0786) = 0.15$.

**(b)** X: $0.75 \times 0.3 + 0.1786 \times 0.7 = 0.225 + 0.125 = 0.35$ ✓. Y: $0.6 \times 0.1 + 0.1 \times 0.9 = 0.06 + 0.09 = 0.15$ ✓.

**(c)** $\text{SR}_X = 0.7 \times 0.3 + 0.1 \times 0.7 = 0.28$, $\text{SR}_Y = 0.7 \times 0.1 + 0.1 \times 0.9 = 0.16$, $\Delta_{\text{DP}} = 0.12$. Perfect equalized odds still leaves a selection gap, because the base rates differ (Step 3).

**(d)** With 10 positives and TPR 0.6, the standard error is $\sqrt{0.6 \times 0.4/10} = 0.155$, so a 95% interval is about ±0.30 (0.3 to 0.9). Group Y's TPR is essentially unknown. The report should say so,
give the interval, and plan for more data, rather than state a 0.15 opportunity gap as a finding.

```text
== exercise 1 check ==
(a) group X: base 0.3000  SR 0.3500  TPR 0.7500  FPR 0.1786   (b) TPR*base + FPR*(1-base) = 0.3500
(a) group Y: base 0.1000  SR 0.1500  TPR 0.6000  FPR 0.1000   (b) TPR*base + FPR*(1-base) = 0.1500
    gaps: DP 0.2000  equal opportunity 0.1500  FPR 0.0786  equalized odds 0.1500
(c) TPR 0.7, FPR 0.1 in both: SR X 0.28  SR Y 0.16  DP difference 0.12
(d) TPR 0.6 on 10 positives: standard error 0.155, 95% interval about +-0.30
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Eight rates correct (FP and N derived) and three gaps | 5 |
| (b) Identity checked for both groups | 2 |
| (c) Selection rates 0.28 and 0.16, gap 0.12, and the conclusion | 3 |
| (d) About ±0.30, and the reporting consequence | 4 |

**Likely mistakes:**

- Computing FPR as FP / (number selected) (that is 1 − precision) instead of FP / N.
- Equalized odds as the *sum* or average of the TPR and FPR gaps instead of the larger one (fairlearn's definition).
- Saying in (c) that equal TPR and FPR imply equal selection.
- Ignoring the tiny group size in (d).

## Exercise 2 (code)

**Solution:** [`b4_6_fairness_model_cards_ex2.py`](b4_6_fairness_model_cards_ex2.py). The equal-FPR thresholds use quantiles of each group's *negative* rows' scores:

```python
target_fpr = pred[y_va == 0].mean()
th = {g: np.quantile(s_va[(sex_va == g) & (y_va == 0)], 1 - target_fpr, method="higher") for g in ["Male", "Female"]}
```

```text
== data and model (as in the lab) ==
val rows 9768

== (a) audit by race with bootstrap CIs for the TPR gap to White ==
                          n  selection_rate     TPR     FPR
sensitive_feature_0                                        
Amer-Indian-Eskimo    104.0          0.0673  0.5385  0.0000
Asian-Pac-Islander    302.0          0.1921  0.5205  0.0873
Black                 936.0          0.0823  0.5051  0.0323
Other                  71.0          0.0423  0.2500  0.0159
White                8355.0          0.2074  0.5882  0.0760
TPR(Black) - TPR(White) = -0.0831   95% CI [-0.1782, +0.0174]
TPR(Asian-Pac-Islander) - TPR(White) = -0.0676   95% CI [-0.1817, +0.0501]
TPR(Amer-Indian-Eskimo) - TPR(White) = -0.0497   95% CI [-0.3015, +0.2459]
TPR(Other) - TPR(White) = -0.3382   95% CI [-0.5933, +0.0350]

== (b) per-group thresholds that equalize FPR ==
target FPR 0.0700   thresholds M 0.586  F 0.209
single threshold 0.5               accuracy 0.8465   DP 0.1887  EOpp 0.1496  FPR gap 0.0804  EOdds 0.1496   TPR M 0.603 F 0.453
equal FPR thresholds (from scratch) accuracy 0.8363   DP 0.0688  EOpp 0.1898  FPR gap 0.0000  EOdds 0.1898   TPR M 0.511 F 0.701

== (c) fairlearn ThresholdOptimizer, equalized odds ==
ThresholdOptimizer (equalized odds) accuracy 0.8277   DP 0.0899  EOpp 0.0358  FPR gap 0.0024  EOdds 0.0358   TPR M 0.507 F 0.471
equalized-odds difference matches fairlearn's: OK
```

**What a strong write-up says:**

- **(a)** Every non-White group's TPR point estimate is below the White group's, but every interval for the gap includes 0. The data are consistent with no gap *and* with gaps of up to 0.18 (and 0.59
  for "Other"). Report the intervals, and that the audit cannot rule out meaningful gaps for the smaller groups. Do not report "no bias found".
- **(b)** Equal FPRs reverse the opportunity gap (women's TPR 0.70 against men's 0.51; $\Delta_{\text{EOpp}} = 0.19$), at a cost of one point of accuracy. Equalizing one rate moves the others, as in the lab.
- **(c)** `ThresholdOptimizer` randomizes between thresholds per group to bring both TPR and FPR close together ($\Delta_{\text{EOdds}} = 0.036$), at a cost of 1.9 points of accuracy. Randomized decisions, where two
  identical applicants can get different answers, are themselves hard to justify in many settings. That is a stakeholder question, not a technical one.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| (a) MetricFrame with sizes and rates; bootstrap CIs for gaps (resampling each group's positives) | 4 |
| (a) Interpretation: wide intervals, no "no bias" claim | 2 |
| (b) Equal-FPR thresholds from negatives' score quantiles; three gaps and accuracy | 4 |
| (c) ThresholdOptimizer fitted on training rows, evaluated on validation; compared with (b) | 3 |
| Discussion of trade-offs (accuracy, reversed gaps, randomization) | 3 |

**Likely mistakes:**

- Bootstrapping all rows and recomputing the TPR on whatever positives happen to land in the resample (acceptable), but then forgetting the group's tiny positive count in the interpretation.
- Computing the FPR thresholds from all rows' scores, which equalizes selection rates instead.
- Fitting `ThresholdOptimizer` on the validation rows it is evaluated on.

## Exercise 3 (think)

**Strong answer:** Excluding sex, race and age does not make the model fair. Proxies remain: postcode for race, years of employment or account age for age, product choices and
job titles for sex. In the lab, the remaining columns predicted sex with ROC-AUC 0.93, and dropping `sex` barely moved the gaps. Historical labels can also carry past discrimination
(who was granted credit and so had the chance to repay). Without the attributes you cannot even measure the problem. Data needed: the sensitive attributes for the evaluation rows (collected
lawfully and stored separately, used only for auditing), outcomes, and the model's scores and decisions; ideally also applicants who were refused, which needs reject inference or an experiment.
First gap for lending: equal opportunity, the approval rate among applicants who would repay (TPR), because a qualified applicant refused credit is the harm most lending rules target. Then the FPR gap (approvals of
people who default, which harms them through debt). Report both with CIs, plus the selection-rate ratio, since regulators often look at it (four-fifths rule; VERIFY jurisdiction). Model card sections:
Data (who is missing, how attributes were obtained), Metrics by group (sizes, rates, gaps, CIs), Decision threshold (and whether it differs by group), Known limitations (proxies, small groups, refused applicants
unobserved), Ethical considerations (which error harms whom, mitigations considered, legal review), and Out-of-scope uses.
**Weak answer:** "Since sensitive attributes are excluded, the model can't discriminate. An audit would require collecting sensitive data, which is a privacy risk, so it's better not to."
**Why weak:** it repeats the fairness-through-unawareness fallacy, ignores proxies and label bias, and treats measurement as the risk rather than the safeguard.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Proxies (with an example) and label bias | 3 |
| What data is needed for the audit, and how to hold it responsibly | 2 |
| A justified first gap for lending (equal opportunity / TPR), plus FPR and selection ratio with CIs | 3 |
| Model card sections that capture the findings | 2 |

## Scoring

Total 40 pts. Pass ≥ 70% (28). Topic ids to revisit for each lost criterion: group rates and gaps (Ex 1a–c) → B4.6, B4.1; small-group uncertainty (Ex 1d, Ex 2a)
→ A6.2, B4.6; thresholds and post-processing (Ex 2b–c) → B4.6, B4.2; proxies, audits and documentation (Ex 3) → B4.6, B1.4, J2.5.
