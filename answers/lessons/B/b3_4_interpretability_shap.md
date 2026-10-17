# Answer key: B3.4 Interpretability: permutation importance, partial dependence and SHAP values

Lesson: [`lessons/B/b3_4_interpretability_shap.md`](../../../lessons/B/b3_4_interpretability_shap.md)

## Exercise 1 (by hand)

**Solution (a).** $f(\mathbf{x}) = 3x_1 + 2x_2 + 6x_2x_3$ at $\mathbf{x} = (2, 1, 1)$, background $(0, 0, 0)$:

| $S$ | $\varnothing$ | $\{1\}$ | $\{2\}$ | $\{3\}$ | $\{1,2\}$ | $\{1,3\}$ | $\{2,3\}$ | $\{1,2,3\}$ |
|---|---|---|---|---|---|---|---|---|
| $v(S)$ | 0 | 6 | 2 | 0 | 8 | 6 | 8 | 14 |

$\phi_1 = \frac13(6 - 0) + \frac16(8 - 2) + \frac16(6 - 0) + \frac13(14 - 8) = 2 + 1 + 1 + 2 = 6$.
$\phi_2 = \frac13(2 - 0) + \frac16(8 - 6) + \frac16(8 - 0) + \frac13(14 - 6) = 0.667 + 0.333 + 1.333 + 2.667 = 5$.
$\phi_3 = \frac13(0 - 0) + \frac16(6 - 6) + \frac16(8 - 2) + \frac13(14 - 8) = 0 + 0 + 1 + 2 = 3$.
Efficiency: $6 + 5 + 3 = 14 = f(\mathbf{x}) - 0$. Feature 1 enters only linearly, so it gets exactly its term $3 \times 2 = 6$. The interaction $6x_2x_3 = 6$ is split
equally, 3 to feature 2 (on top of its linear $2$) and 3 to feature 3.

**(b)** ICE of row $i$: $f(u, x_{i2}) = u\,x_{i2}$, a line through the origin with slope $x_{i2}$: slopes 1, 3 and $-1$. $\text{PD}_1(u) = u \times \text{mean}(1, 3, -1) = u$: $0, 1, 2$ for $u = 0, 1, 2$.
The PD curve says "feature 1 increases the prediction with slope 1". It hides that for one row the effect is three times
as strong and for another it goes the *opposite* way: an interaction with $x_2$ that only the ICE curves reveal.

**(c)** Shuffling a column the model ignores leaves every prediction unchanged, so the permutation importance is exactly 0
(for any number of repeats). Its Shapley value is 0 by the **dummy** property: $v(S \cup \{2\}) = v(S)$ for every $S$, so every marginal
contribution is zero.

**(d)** Impurity importance adds up the training-set impurity decrease of every split on a feature, and a fully grown tree keeps
splitting until its leaves are pure. Deep in the tree, nodes hold a few rows, and a continuous noise column almost always
has *some* threshold that separates them by chance, so it wins splits and collects credit, with more distinct values meaning
more chances.

```text
== exercise 1 check ==
(a) v(S): {}: 0  {1}: 6  {2}: 2  {3}: 0  {1,2}: 8  {1,3}: 6  {2,3}: 8  {1,2,3}: 14
    phi [6.0, 5.0, 3.0]   sum 14 = f(x) - v(empty) = 14
(b) v = 0: ICE [0.0, 0.0, 0.0]   PD 0
(b) v = 1: ICE [1.0, 3.0, -1.0]   PD 1
(b) v = 2: ICE [2.0, 6.0, -2.0]   PD 2
```

**Rubric (13 pts):**

| Criterion | Points |
|---|---|
| (a) All 8 coalition values | 2 |
| (a) $\phi = (6, 5, 3)$ with the correct weights, efficiency checked, interaction split explained | 3 |
| (b) Three ICE lines, PD $= 0, 1, 2$, and "the average hides opposite slopes" | 3 |
| (c) Both are 0; names the dummy property | 2 |
| (d) Training data + many thresholds + deep small nodes | 3 |

**Likely mistakes:**

- Weighting every coalition equally ($\frac14$ each) instead of $\frac13, \frac16, \frac16, \frac13$. For $d = 3$ the sizes-0 and size-2 sets each represent two of the six orders.
- In (a), giving the whole interaction to one feature (for example $\phi_2 = 8$, $\phi_3 = 0$) because it "comes first" in the formula.
- In (b), computing $f$ at the *average* row ($v \times \bar{x}_2$) and calling it partial dependence. For this model it happens to give the same numbers, but PD is the average of
  predictions, not the prediction at the average; for non-linear models they differ.
- In (c), saying the Shapley value is "small" rather than exactly 0.

## Exercise 2 (code)

**Solution:** [`b3_4_interpretability_shap_ex2.py`](b3_4_interpretability_shap_ex2.py). The grouped importance is one change
to the lab's function: draw one permutation and apply it to all the group's columns, so each row keeps its own (latitude,
longitude) pair but moves to another district:

```python
idx = gen.permutation(len(X))
Xp[:, cols] = X[idx][:, cols]
```

```text
== data and the B3.3 model ==
8 features: 788 trees   val R^2 0.8335

== (a) a duplicated feature ==
9 features (MedInc twice): 1030 trees   val R^2 0.8316
                           perm. importance  mean |SHAP|
MedInc, 8-feature model              0.3452       0.3707
MedInc, 9-feature model              0.3341       0.3740
MedInc_copy                          0.0014       0.0238
both copies shuffled                 0.3636       0.3907

== (b) grouped permutation importance ==
Latitude alone 1.1814   Longitude alone 0.9791   sum 2.1604
Latitude + Longitude shuffled together: 0.5450 +- 0.0127
```

**(a) What happened.** A common expectation is that two identical columns split the credit evenly. XGBoost does not do
that here. When two columns give identical splits (equal gain), its split search keeps the column with the lower index, so the
original `MedInc` (column 0) wins every tie, and the copy is only used in trees where `colsample_bytree=0.8` happened to leave the original out. The copy ends up with almost no
importance (0.001 permutation, 0.024 mean |SHAP|), and shuffling the original alone (0.334) understates what the pair is worth (0.364), because the copy covers
for it a little. A random forest with `max_features` < 1 would split the credit more evenly. Either way, single-feature
importance of correlated features depends on the model's tie-breaking, not on the data. A strong answer says so and
recommends grouping known duplicates or near-duplicates before reporting.

**(b) What happened.** Shuffling latitude alone costs 1.18 of $R^2$, longitude alone 0.98, but both together only 0.55: far *less* than
either alone. Shuffled separately, each column produces impossible locations (a Bay Area latitude with a Los Angeles longitude, often in
the ocean), and the model's predictions there are wildly off. Shuffled together, each district gets another real location,
so the model sees plausible inputs and the damage reflects only the lost information. Individual permutation importances
of strongly dependent features measure extrapolation as much as reliance; the group number is the honest one.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| (a) Duplicate added, model retrained with the B3.3 recipe | 2 |
| (a) Permutation importance and mean \|SHAP\| for both copies, compared with the single-copy model | 3 |
| (a) Explains the lopsided split (tie-breaking + column sampling) and the "both shuffled" number | 3 |
| (b) Grouped permutation with one shared permutation per repeat | 3 |
| (b) Compares the group with the sum and explains impossible rows | 3 |
| Code quality: reuses functions, seeds, shapes in docstrings | 2 |

**Likely mistakes:**

- Shuffling the two group columns with *different* permutations, which is just two individual shuffles at once.
- Expecting, and reporting, an even split in (a) without looking at the numbers.
- Comparing importances from models trained with different random seeds or tree counts and reading the noise as signal.
- Computing permutation importance on the training rows.

## Exercise 3 (think)

**Strong answer:** *Sound:* per-applicant SHAP values from a TreeExplainer are exact for the model, add up to the score, and give a
consistent way to rank which inputs pushed this applicant's score down. That is a reasonable basis for adverse-action
reasons, which lenders usually must provide (check the local rules, for example the US ECOA/Regulation B requirements; VERIFY for your jurisdiction).
*Not sound:* "SHAP proves debt-to-income causes defaults." SHAP describes how the model uses the feature on this data; the
association may be confounded (income, employment type), and the model would give the feature credit even if it were only a
proxy. Causal claims need experiments or causal analysis. *Put in place before shipping:* (1) choose and document the base value and background,
for example "relative to approved applicants", since reasons change with it; (2) group correlated features (all
debt-related variables) so reasons are not split across near-duplicates or flipped by tie-breaking; (3) map features to
reasons a person can act on, and drop or merge reasons that are not actionable or are sensitive (age, or a postcode acting as a proxy for a protected
attribute, B4.6); (4) test stability: the same applicant scored by consecutive model versions should get mostly the same top
reasons, so measure the overlap of the top three between versions and on bootstrap refits; (5) check that the reasons
reproduce the decision (efficiency gives this for the score, but the decision is a threshold); (6) log the explanations
with the model version for audits, and include them in the model card (J2.5).
**Weak answer:** "SHAP values are the standard for explainability, so sending the top three to each applicant is fine; and
since debt-to-income has the largest SHAP values, it is the main cause of default."
**Why weak:** it treats a model explanation as causal truth, ignores background choice, correlated features, stability and
actionability, and puts no checks in place.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Separates model explanation from causation, with a reason (confounding or proxy) | 3 |
| Background / base value must be chosen and stated | 1 |
| Correlated features: group them | 2 |
| Stability across versions, with a way to measure it | 2 |
| Actionability, or sensitive/proxy features in reasons | 1 |
| Logging, documentation or regulatory check | 1 |

## Scoring

Total 39 pts. Pass ≥ 70% (28). Topic ids to revisit for each lost criterion: Shapley weights and properties (Ex 1a, 1c) →
B3.4; PD versus ICE (Ex 1b) → B3.4; impurity importance (Ex 1d) → B3.1, B3.4; correlated features and grouped importance (Ex 2)
→ B3.4; explanations versus causation and governance (Ex 3) → B3.4, B4.6, J2.5.
