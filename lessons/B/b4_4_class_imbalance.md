# B4.4 Class imbalance: class weights, resampling, threshold tuning

**Core** · Part B, week 4 (course week 10) · ~1.5 h · Lab:
[`labs/B/b4_4_class_imbalance.py`](../../labs/B/b4_4_class_imbalance.py)

## Why it matters

The problems that pay for ML are often rare-event problems: fraud, defaults, churn in a given month, machine failures, disease. With 1%
positives, a model that never predicts the positive class is 99% accurate and worthless, and a reasonable model may never cross the 0.5
threshold. The usual reflex is to "balance the data" with class weights or resampling. That does change the predictions, but it also
breaks the probabilities, and on this lesson's data it makes the precision-oriented metrics much *worse*. This lesson compares
doing nothing, class weights, oversampling, undersampling and threshold tuning on one 1%-positive problem. It derives exactly what reweighting does to the
probabilities and how to undo it. Interviewers ask "how do you handle imbalanced data?" constantly; the strong answer starts with the metric and the threshold, not with resampling.

## Prerequisite check

- Can you compute precision, recall, F1 and average precision, and say why accuracy misleads with rare positives? If not, revisit
  [B4.1](b4_1_classification_metrics.md).
- Can you derive the cost-based threshold $c_{FP}/(c_{FP} + c_{FN})$? If not, revisit [B4.2](b4_2_regression_metrics.md).
- Do you know why stratified folds matter when one class is rare? If not, revisit [B4.3](b4_3_cross_validation_tuning.md).
- Can you convert between a probability $p$ and odds $p/(1 - p)$? If not, revisit [B2.1](b2_1_logistic_regression.md).

## Intuition

Imbalance by itself is not a bug in the data; it is a fact about the world. It causes three practical problems:

1. **Misleading metrics.** Accuracy rewards ignoring the rare class. Use precision, recall, F1 or average precision (B4.1).
2. **A wrong default threshold.** A calibrated model on 1%-positive data rarely says "above 0.5", even for cases that are 30% likely to be positive. If you would act on a 30% chance
   (because a miss is expensive), the threshold must be lower (B4.2).
3. **Few positives to learn from and to evaluate on.** 120 validation positives make every metric noisy. Stratify the splits, and use confidence intervals (A6.2).

The popular fixes act on the training data:

- **Class weights:** count each positive as $w$ negatives in the loss (`class_weight="balanced"` sets $w = n_{\text{neg}}/n_{\text{pos}}$, 99 here).
- **Oversampling:** duplicate positives (drawn with replacement) until the classes are equal.
- **Undersampling:** throw away negatives until the classes are equal.

All three tell the model that positives are far more common than they are, so its probabilities rise. With a 1:1 training mix, the model's "50%" corresponds to a
real 1%. That moves cases over the 0.5 threshold, which is often the whole reason people do it. But the same thing is achieved, without corrupting the
probabilities, by keeping the model as it is and *lowering the threshold*. And for a model that cannot fit the data exactly (any real model), reweighting
also changes *which* mistakes the fit cares about, so the ranking itself changes, sometimes for the worse.

> **Bridge:** class weights are like scaling a single rare alert type's severity by 99 so that it pages someone. **Where it breaks:** the severity number is now
> wrong for everyone downstream. A dashboard that sums severities, or a rule that pages at severity ≥ 50, behaves differently. Changing the threshold at which
> you page keeps the raw signal honest. In ML terms, rebalancing changes the probabilities everyone else reads; threshold tuning only changes your decision.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\pi$ | positive share (prevalence) of the population | scalar |
| $p$ | true probability that a given input is positive, $P(y = 1 \mid \mathbf{x})$ | scalar |
| $q$ | the probability a model trained on rebalanced data converges to for that input | scalar |
| $w$ | weight of a positive relative to a negative in the training loss | scalar |
| $\beta$ | share of negatives kept by undersampling | scalar |
| $\ell$ | log-loss of one example | scalar |

**Step 1: what a positive weight $w$ does to the best prediction.** For one input with true probability $p$, a model predicting $q$ pays, in expectation, the weighted log-loss
$$L(q) = -w\,p\log q - (1 - p)\log(1 - q).$$
(Positives occur with probability $p$ and cost $w\log\frac1q$ each; negatives occur with probability $1 - p$ and cost $\log\frac{1}{1 - q}$.) Set the derivative to zero:
$$L'(q) = -\frac{w\,p}{q} + \frac{1 - p}{1 - q} = 0.$$
Multiply both sides by $q(1 - q)$ to clear the denominators, then divide by $(1 - p)(1 - q)$:
$$w\,p\,(1 - q) = (1 - p)\,q \iff \frac{q}{1 - q} = w\cdot\frac{p}{1 - p}.$$
The weighted model's **odds are the true odds times $w$**. With $w = 99$, a true 1% (odds $1/99$) becomes odds 1: a predicted 50%.

**Step 2: resampling does the same.** Duplicating each positive $w$ times gives exactly the same loss as weight $w$ (the sum simply contains each positive $w$ times),
and random oversampling does so in expectation. Keeping each negative with probability $\beta$ multiplies the number of negatives by $\beta$, so the odds of a positive rise by a factor $1/\beta$. Undersampling to 1:1 with 1% positives means
$\beta = 1/99$, the same factor of 99.

**Step 3: the prior correction.** Invert Step 1: multiply the model's odds by $1/w$ (or by $\beta$ after undersampling):
$$p = \frac{\beta\,q/(1 - q)}{1 + \beta\,q/(1 - q)}.$$
This is exact only if the model would have reached the true $p$ without rebalancing. In practice it restores the overall level of the probabilities, but not differences caused by the model fitting the reweighted data differently.

**Step 4: weights are a threshold in disguise.** The rebalanced model predicts positive when $q > 0.5$, that is when $w\frac{p}{1 - p} > 1$, that is when $p > \frac{1}{1 + w}$. So training with weight $w$ and thresholding at 0.5 is, for a perfect model,
the same decision as training without weights and thresholding the true probability at $1/(1 + w)$. Compare B4.2: $t^* = c_{FP}/(c_{FP} + c_{FN})$ equals $1/(1 + w)$ when $w = c_{FN}/c_{FP}$. **Choosing $w$ is choosing a cost ratio**, and `"balanced"` silently
asserts that a missed positive costs $n_{\text{neg}}/n_{\text{pos}}$ times a false alarm, a number that comes from the data, not from the business.

**Step 5: why the ranking can change too.** Steps 1–4 assume the model can represent $p$ exactly. A logistic regression cannot when the classes' true boundary is not
linear. Then the fit is a compromise, and the weights decide which rows the compromise favours: with $w = 99$, it bends towards getting the bulk of positives above the boundary
at the cost of mixing them with many negatives. The ranking near the top, which decides precision, can get worse even when the overall ranking (ROC-AUC) gets better.

## Worked example by hand

**Prior correction.** True prevalence 1%; undersample negatives to 1:1, so $\beta = 1/99 = 0.0101$. A model trained on the balanced sample says $q = 0.5$: odds 1, corrected odds $0.0101$, corrected $p = 0.0101/1.0101 = 0.0100$.
For $q = 0.9$: odds 9, corrected $9/99 = 0.0909$, $p = 0.0909/1.0909 = 0.0833$. For $q = 0.99$: odds 99, corrected odds 1, $p = 0.5$. A balanced-sample model needs to say 0.99 before the case is truly a coin flip.

**Weights as a threshold.** $w = 99$ and a 0.5 cut flags everything with true $p > 1/(1 + 99) = 0.01$, the prevalence itself. Every case more likely than average to be positive
gets flagged. That is the right decision only if a miss truly costs 99 false alarms.

**The trivial model.** Always predicting negative: accuracy $0.99$, recall 0, and F1 0 (no true positives).

```text
== worked example by hand ==
balanced-sample prediction 0.5: odds 1.0000 -> true odds beta x odds = 0.0101 -> p = 0.0100
balanced-sample prediction 0.9: corrected p = 0.0833
balanced-sample prediction 0.99: corrected p = 0.5000
always predict negative: accuracy 0.99, recall 0, F1 0
```

## Key plot

![Reliability diagram on log scales with ten equal-count bins: the plain model's top bin lies on the diagonal (predicted 0.087, observed 0.084), while the class-weighted, oversampled and undersampled models' points lie far to the right of it, their top bin predicting 0.80 where 8.5% are positive](img/b4_4_rebalancing_calibration.png)

Ten bins of 1,200 validation rows each, sorted by predicted probability, on log scales. The plain logistic regression's top bin says 8.7% and is 8.4% positive.
The three rebalanced models put almost the same rows in their top bin but say 80%: ten times too high. In odds, $0.80/0.20 = 4$ against $0.087/0.913 = 0.095$, a factor of about 42: Step 1's
direction, but less than its factor of 99, because the rebalanced fit is a different model, not a rescaled one (Step 5). The scattered points in the lower bins are noise: each holds between 0 and 6 positives (zero rates are drawn at $10^{-4}$).

## From scratch in NumPy

**The data.** 60,000 synthetic rows, 10 features (5 informative), 1% positives, split 60/20/20 with stratification: 360 training positives and 120 validation positives.

```text
== data ==
train (36000, 10): 360 positives (0.0100)   val (12000, 10): 120 positives (0.0100)   (test 12000 rows kept aside)
```

**Resampling.** Oversampling draws positives with replacement; undersampling draws negatives without replacement:

```python
def oversample(X, y, gen):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    extra = gen.choice(pos, size=len(neg) - len(pos), replace=True)
    rows = gen.permutation(np.r_[neg, pos, extra])
    return X[rows], y[rows]

def correct_prior(p, beta):
    odds = beta * p / (1 - p)               # Step 3
    return odds / (1 + odds)
```

```text
== from scratch: resampling ==
oversampled: 71280 rows, positive share 0.5000 (360 distinct positives)
undersampled: 720 rows, positive share 0.5000, kept share of negatives beta = 0.01010
class_weight='balanced': a positive counts 99.00 times as much as a negative
threshold tuning: F1-maximizing threshold on out-of-fold training predictions = 0.2467
```

Oversampling shows the model the same 360 positives about 99 times each; undersampling trains on 720 rows and throws away 99% of the negatives. Threshold tuning keeps the plain
model and picks the F1-maximizing threshold from *out-of-fold* predictions on the training rows (`cross_val_predict` with stratified 5-fold), so the validation rows stay clean.

**The comparison.** Every strategy's validation scores, at threshold 0.5 (or the tuned threshold):

```text
== comparison on the validation set ==
strategy          thresh accuracy     F1 recall     AP ROC-AUC   Brier  mean p
do nothing         0.500   0.9949 0.6806 0.5417 0.7293  0.9115  0.0043  0.0105
class weights      0.500   0.8694 0.1211 0.9000 0.1740  0.9465  0.0967  0.1995
oversample         0.500   0.8689 0.1207 0.9000 0.1773  0.9465  0.0969  0.2005
undersample        0.500   0.8761 0.1268 0.9000 0.1593  0.9447  0.0949  0.1966
threshold tuning   0.247   0.9952 0.7212 0.6250 0.7293  0.9115  0.0043  0.0105
validation prevalence 0.0100
```

- **Class weights, oversampling and undersampling behave almost identically** (Step 2). At 0.5 they catch 90% of positives, but by flagging about 14% of all rows, so precision
  is about 6.5% and F1 0.12.
- **Their probabilities are inflated twentyfold:** mean predicted 0.20 against a true rate of 0.01, and the Brier score is 22 times worse.
- **Their ranking at the top got much worse:** average precision fell from 0.73 to 0.16–0.18, while ROC-AUC *rose* from 0.91 to 0.95 (Step 5). The reweighted linear model ranks more positives above the
  mass of negatives, but mixes them with many negatives near the top. If the business acts on the top of the list, that is the wrong trade. It is also another case of B4.1's lesson:
  ROC-AUC and AP can disagree, and with rare positives AP is the one that matches the work.
- **Threshold tuning wins on F1** (0.72), keeps the plain model's AP and its calibrated probabilities, and costs nothing but choosing a number on out-of-fold predictions.

**The prior correction** applied to the rebalanced models:

```text
== prior correction repairs the probabilities ==
class weights  Brier 0.0967 -> 0.0101   mean p 0.1995 -> 0.0120   ROC-AUC unchanged 0.9465
oversample     Brier 0.0969 -> 0.0100   mean p 0.2005 -> 0.0118   ROC-AUC unchanged 0.9465
undersample    Brier 0.0949 -> 0.0110   mean p 0.1966 -> 0.0135   ROC-AUC unchanged 0.9447
```

The correction brings the average probability back to about 1% and cuts the Brier score by nine-tenths. Because it is monotone, it does not change the ranking. It cannot reach the plain model's Brier (0.0043)
because the rebalanced fit is a different model (Step 5), not just a rescaled one.

## With scikit-learn

```text
== with scikit-learn: oversampling equals weighting ==
each positive duplicated 5x versus sample_weight 5: same coefficients (to solver tolerance): OK
from-scratch F1 matches sklearn.metrics.f1_score: OK
```

- `class_weight="balanced"` weights class $c$ by $n/(K\,n_c)$; for two classes the ratio is $n_{\text{neg}}/n_{\text{pos}}$. `class_weight={0: 1, 1: w}` sets an explicit ratio, and `fit(..., sample_weight=...)` per row.
  Gradient-boosting libraries use `scale_pos_weight` (XGBoost, LightGBM) for the same thing.
- Resampling must happen *inside* the training folds only, never before splitting: duplicated positives in a validation fold are leakage (B1.4). The imbalanced-learn
  library's pipeline handles this, and adds SMOTE, which creates synthetic positives by interpolating between neighbours. It is not part of this course's
  environment; on modern models it rarely beats weighting or threshold tuning (VERIFY as of 2026-10).
- `sklearn.model_selection.TunedThresholdClassifierCV` chooses the threshold by cross-validation for a metric or a cost function, the library version of the threshold-tuning row above.

## In practice

- **When to use it** (in this order): (1) pick a metric that reflects the rare class: AP, recall at a fixed precision, or expected cost; (2) stratify every split; (3) train a well-calibrated model
  without rebalancing; (4) choose the threshold from costs or capacity, on out-of-fold or validation predictions; (5) only then consider weights, if the model clearly underfits the
  positives, and treat $w$ as a cost ratio you can defend; (6) if you rebalance, apply the prior correction or recalibrate (B4.1) before anyone reads the probabilities.
- **Failure modes:**
  - *Accuracy or ROC-AUC alone* on rare positives (B4.1).
  - *Resampling before the split.* Copies of the same positive land in train and validation; scores inflate.
  - *Rebalanced probabilities used as probabilities* in pricing, expected-loss or alert-volume forecasts: off by the weight factor.
  - *"Balanced" as a default.* It encodes a cost ratio equal to the class ratio, which is rarely the business's.
  - *Too few positives to evaluate.* With 120 positives, a 0.02 difference in AP is within noise; bootstrap the validation set (A6.2) or use repeated stratified CV.
- **Defaults that work:** `StratifiedKFold(5, shuffle=True)`; average precision plus the confusion matrix at the chosen threshold; no rebalancing for the first model;
  threshold from `TunedThresholdClassifierCV` or a cost curve (B4.2); if a weighted model is better on your metric, recalibrate it with `CalibratedClassifierCV`
  (scikit-learn 1.9.1; VERIFY as of 2026-10).

## Exercises

Save your work to `my-work/B/b4_4_class_imbalance.md` (or `.py`) and ask for a review.

1. **By hand:** A population has 2% positives. (a) A model trained with positive weight $w = 49$ outputs $q = 0.6$ for a case. What is the corrected probability? (b) Which threshold on the
   corrected probability is equivalent to thresholding the weighted model at 0.5? What cost ratio does $w = 49$ assume? (c) For "always negative" and "always positive", give accuracy, precision, recall and F1. (d) With
   50 positives in 5,000 rows and *unstratified* 5-fold CV, what is the expected number of positives per validation fold, and why is stratification still worth it?
2. **Code:** On the lab's data, replace logistic regression by LightGBM (`n_estimators=300`, `learning_rate=0.05`, `n_jobs=1`). Train it plainly, with `scale_pos_weight=99`, and on the
   undersampled data. For each, report AP, ROC-AUC, Brier and mean predicted probability on validation; apply the prior correction to the two rebalanced models; and for the plain model,
   choose the threshold that minimizes expected cost with $c_{FN} = 50\,c_{FP}$ on out-of-fold training predictions and report the validation cost per row. Does rebalancing change the
   flexible model's ranking as much as the linear model's?
3. **Think:** A hospital screening model flags patients for a follow-up test; 0.5% of patients have the condition. A data scientist proposes SMOTE to get a balanced
   training set "because the model never predicts positive", and plans to show clinicians the model's probability for each patient. Explain what you would do instead, and what the
   clinicians would be shown under each approach.

## Takeaways

1. With rare positives, fix the metric and the threshold first: accuracy and a 0.5 cut are the real problems, not the class ratio.
2. A positive weight $w$ (or oversampling by $w$, or undersampling negatives to a share $1/w$) multiplies the model's odds by $w$; the prior correction multiplies them back by $1/w$.
3. Weighting is a threshold in disguise: weight $w$ with a 0.5 cut is threshold $1/(1 + w)$ on calibrated probabilities, an implicit cost ratio of $w$.
4. For a model that cannot fit the data exactly, reweighting also changes the ranking; here it raised ROC-AUC but collapsed average precision from 0.73 to 0.17.
5. Threshold tuning on out-of-fold predictions of an unweighted model gave the best F1 and kept calibrated probabilities; rebalance only for a measured gain, and recalibrate afterwards.

**What builds on this:** [B4.5](b4_5_bias_variance.md) (few positives mean high variance); [B4.6](b4_6_fairness_model_cards.md)
(rates per group, often with small groups); [B5.6](b5_6_anomaly_detection.md) (anomaly detection when positives are too rare to learn from); P2 (the threshold from costs).

**References:** C. Elkan, "The foundations of cost-sensitive learning", *IJCAI*, 2001. M. Saerens, P. Latinne, C. Decaestecker, "Adjusting the outputs of a classifier to new a
priori probabilities: a simple procedure", *Neural Computation* 14, 2002. N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, "SMOTE: synthetic minority over-sampling technique",
*JAIR* 16, 2002. scikit-learn User Guide, "Tuning the decision threshold for class prediction" (VERIFY link:
https://scikit-learn.org/stable/modules/classification_threshold.html).
