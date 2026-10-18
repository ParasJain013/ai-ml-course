# Answer key: B4.4 Class imbalance: class weights, resampling, threshold tuning

Lesson: [`lessons/B/b4_4_class_imbalance.md`](../../../lessons/B/b4_4_class_imbalance.md)

## Exercise 1 (by hand)

**Solution (a).** Odds $= 0.6/0.4 = 1.5$; divide by $w = 49$: $0.0306$; $p = 0.0306/1.0306 = 0.0297$. The weighted model's "60%" is a real 3%.

**(b)** Thresholding the weighted model at 0.5 equals thresholding the corrected probability at $1/(1 + 49) = 0.02$, the prevalence. It assumes a missed positive costs
49 times a false alarm ($c_{FP}/(c_{FP} + c_{FN}) = 1/50$ when $c_{FN}/c_{FP} = 49$).

**(c)** Always negative: accuracy 0.98; precision undefined (no predicted positives; report it as 0 or "n/a"); recall 0; F1 0. Always positive: accuracy 0.02; precision 0.02;
recall 1; $F_1 = 2(0.02)(1)/(1.02) = 0.0392$.

**(d)** Expected positives per fold: $50/5 = 10$. Without stratification the count varies a lot from fold to fold (the run below gave 13, 12, 11, 7, 7, and across 200 shuffles
it ranged from 2 to 20). A fold with 2 positives gives a meaningless recall or AP, and the fold-to-fold spread then mixes model quality with luck in the split. Stratification gives exactly 10
per fold.

```text
== exercise 1 check ==
(a) corrected p = 0.0297
(b) equivalent threshold 1/(1+w) = 0.0200; assumed cost ratio c_FN/c_FP = 49
(c) always negative: accuracy 0.98, recall 0, F1 0;  always positive: accuracy 0.02, precision 0.02, recall 1, F1 0.0392
(d) positives per fold: shuffled KFold [13, 12, 11, 7, 7]   StratifiedKFold [10, 10, 10, 10, 10]   over 200 shuffles: min 2, max 20, mean 10.0
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) Odds divided by $w$, corrected $p = 0.0297$ | 3 |
| (b) Threshold 0.02 and cost ratio 49 | 3 |
| (c) All eight numbers (precision of "always negative" flagged as undefined) | 3 |
| (d) 10 expected, the variability without stratification, and why it matters | 3 |

**Likely mistakes:**

- Dividing the *probability* by $w$ instead of the odds ($0.6/49 = 0.012$, wrong).
- Writing the equivalent threshold as $w/(1 + w)$.
- Saying unstratified folds are "biased". They are unbiased on average, just highly variable.

## Exercise 2 (code)

**Solution:** [`b4_4_class_imbalance_ex2.py`](b4_4_class_imbalance_ex2.py).

```text
== data (as in the lab) ==
train positives 360   undersampled rows 720   beta 0.01010

== LightGBM: plain, weighted, undersampled ==
model                    AP ROC-AUC   Brier  mean p   after prior correction
plain                0.9514  0.9982  0.0018  0.0094
scale_pos_weight=99  0.9502  0.9980  0.0020  0.0101   Brier 0.0018  mean p 0.0087
undersampled 1:1     0.8586  0.9953  0.0297  0.0477   Brier 0.0091  mean p 0.0211
validation prevalence 0.0100

== cost-based threshold for the plain model ==
threshold minimizing out-of-fold cost (c_FN = 50 c_FP): 0.001   formula 1/(1+50) = 0.020
validation at t = 0.001: cost per row 0.0377   flagged 165   recall 0.9333
validation at t = 0.020: cost per row 0.0475   flagged 129   recall 0.9083
validation at t = 0.500: cost per row 0.0673   flagged 112   recall 0.8667
```

**What a strong answer says:**

- LightGBM fits this non-linear problem far better than logistic regression (AP 0.95 against 0.73). For the flexible model, `scale_pos_weight=99` barely changes the ranking (AP 0.950 against
  0.951), unlike the linear model's collapse in the lab: Step 5 of the lesson is about models that cannot fit the boundary.
- The weighted booster's probabilities moved much less than the factor of 99 that Step 1 predicts (mean $p$ 0.0101 against 0.0094). Step 1 describes the optimum of the weighted loss; 300 shrunk trees do not
  reach it, especially for rows the model already predicts near 0. The prior correction still lands the Brier score back on the plain model's 0.0018. A strong answer reports this
  honestly rather than forcing the theory onto the numbers.
- Undersampling hurts here: training on 720 of 36,000 rows throws away 98% of the information about negatives, and AP drops to 0.86. Its probabilities are inflated (mean 0.048), and the prior correction only
  partly repairs them (Brier 0.0091 against 0.0018).
- The cost-minimizing threshold for $c_{FN} = 50\,c_{FP}$ is very low (0.001 on out-of-fold predictions), lower than the formula's 0.02. Boosted trees push most negatives to near-zero probabilities, so
  lowering the threshold adds few false positives. On validation, the tuned threshold costs 0.038 per row, against 0.048 at 0.02 and 0.067 at 0.5.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Three LightGBM variants trained on the lab's split (resampling on training rows only) | 3 |
| AP, ROC-AUC, Brier and mean $p$ reported for each | 3 |
| Prior correction applied with the right factor ($1/w$ or $\beta$) | 2 |
| Cost-minimizing threshold chosen on out-of-fold predictions, validation cost reported | 4 |
| Interpretation: flexible vs linear model, undersampling's information loss, honest note on the weighted booster | 4 |

**Likely mistakes:**

- Undersampling the whole dataset before splitting, so validation has a 1:1 mix and every metric is meaningless for production.
- Choosing the threshold on the validation rows and reporting the cost there.
- Applying the prior correction with $w$ instead of $1/w$ (making the probabilities even larger).

## Exercise 3 (think)

**Strong answer:** "Never predicts positive" means the threshold is wrong, not the data. At 0.5% prevalence a calibrated model's probabilities rarely reach 0.5, even for patients whose risk is 20 times
the average. Instead: (1) agree with clinicians on what a missed case costs compared with an unnecessary follow-up test, or on the capacity for follow-ups per week; (2) train without rebalancing,
check calibration (B4.1), and choose the threshold from that cost ratio or capacity on out-of-fold predictions (B4.2), reporting recall and precision there with confidence intervals (few positives);
(3) if a weighted model ranks better on validation (higher AP or recall at the agreed precision), use it but recalibrate it before showing any probability. What clinicians see: with SMOTE or
weights and no correction, probabilities inflated many times over. A patient with a true 2% risk might show "60%", which misleads every clinical judgment and erodes trust when most "60%" patients
turn out healthy. With the recommended approach: a calibrated risk ("2.1%, about four times the average") plus a flag driven by the agreed threshold. Also stratify all splits and keep resampling, if any, inside the training folds.
**Weak answer:** "SMOTE is the standard way to fix imbalance, so use it, and show the probabilities because clinicians like numbers."
**Why weak:** it misdiagnoses the problem (threshold, not data), ignores calibration, and would show clinicians inflated risks.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Diagnoses the threshold (not the class ratio) as the problem | 2 |
| Threshold from costs or capacity on out-of-fold predictions, with CIs | 3 |
| Explains that SMOTE/weights inflate probabilities, and why that harms clinicians | 3 |
| Recalibration (or prior correction) if a rebalanced model is used; stratification and no leakage | 2 |

## Scoring

Total 38 pts. Pass ≥ 70% (27). Topic ids to revisit for each lost criterion: prior correction and weights as thresholds (Ex 1a–b, Ex 2) → B4.4;
trivial classifiers and metrics (Ex 1c) → B4.1; stratification (Ex 1d) → B4.3; cost-based thresholds (Ex 2, Ex 3) → B4.2; calibration (Ex 3) → B4.1.
