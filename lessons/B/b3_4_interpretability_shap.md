# B3.4 Interpretability: permutation importance, partial dependence and SHAP values

**Core** · Part B, week 3 (course week 09) · ~2 h · Lab:
[`labs/B/b3_4_interpretability_shap.py`](../../labs/B/b3_4_interpretability_shap.py)

## Why it matters

A gradient-boosted model with 788 trees is accurate and unreadable. Before anyone ships it they will ask: which
inputs does it rely on, in which direction, and why did it give *this* customer *this* answer? You need those answers
to catch leaks (B1.4: a leaked column jumps to the top of every importance chart), to debug errors, to satisfy a
regulator or a customer who was refused, and to explain the model in a design review or an interview. This lesson builds the three
standard tools from scratch: **permutation importance** (global reliance), **partial dependence** (the average effect
of one feature) and **SHAP** values (one prediction split into per-feature contributions). It also shows why the
importance chart a tree library prints by default can rank pure noise above real features. The production mistake to avoid is
reading any of these as cause and effect: they describe the model, not the world.

## Prerequisite check

- Can you train the B3.3 XGBoost model with early stopping and say what its validation RMSE means? If not, revisit
  [B3.3](b3_3_gradient_boosting.md).
- Can you compute a split's impurity decrease (gain) in a tree? If not, revisit [B3.1](b3_1_decision_trees.md).
- Can you compute $R^2$ and say what $R^2 < 0$ means? If not, revisit [B1.5](b1_5_linear_regression.md).
- Can you explain why a score measured on training rows is optimistic? If not, revisit [B1.1](b1_1_ml_workflow.md).
- Do you know what an expectation over a sample is, $\frac1n\sum_i f(x_i)$? If not, revisit
  [A5.1](../A/a5_1_random_variables_expectation.md).

## Intuition

Three questions, three tools.

**"Which features does the model rely on?"** Take the validation set and destroy one feature's information by
shuffling that column, so every row gets some other row's value. The feature keeps its distribution, but its link
to the target is gone. If the score drops a lot, the model relied on that feature; if it does not move, the model
did not need it. That drop is the permutation importance. It treats the model as a black box: any model, any metric.

**"How does the prediction change when one feature changes?"** Take every validation row, set `MedInc` to 2, predict,
and average. Then set it to 3, then 4. The averages, plotted against the value, form the **partial dependence**
curve: the model's average response to that feature with everything else left as it was. The per-row curves
(before averaging) are called **ICE** curves (individual conditional expectation), and they show whether all rows respond the same way.

**"Why this prediction?"** Explain one prediction as a sum: *the average prediction, plus a contribution from each
feature*. SHAP gets the contributions from game theory. Treat the features as players who cooperate to move the
prediction away from the average, and split the payout fairly using **Shapley values**. "Fair" has a precise meaning:
average each feature's marginal contribution over every order in which the features could join.

And a warning. Tree libraries print a built-in importance: how much each feature reduced impurity across all splits
(**impurity-based importance**, MDI, mean decrease in impurity). It is free to compute, but it is measured on the training data and it rewards
features that offer many split points. Continuous noise offers thousands; a 0/1 flag offers one. The key plot shows the result.

> **Bridge:** permutation importance is chaos engineering for features. Kill one dependency (shuffle a column), run the
> traffic (validation set) and measure the error rate. **Where it breaks:** services fail independently, but features are
> correlated. Shuffle latitude alone and you create districts in the Pacific Ocean, rows the model never saw, so the
> measured damage partly reflects nonsense inputs, not reliance. And if two features carry the same information, killing one
> shows little damage because the other covers for it.

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $f$ | the trained model: features → prediction | — |
| $\mathbf{X}$, $\mathbf{y}$ | evaluation rows (validation) and targets | `(m, d)`, `(m,)` |
| $s(f, \mathbf{X}, \mathbf{y})$ | a score, here $R^2$ (higher is better) | scalar |
| $\mathbf{X}^{(\pi, j)}$ | $\mathbf{X}$ with column $j$ reordered by the permutation $\pi$ | `(m, d)` |
| $R$ | number of shuffles per feature | int |
| $\mathbf{x}_{-j}$ | a row with feature $j$ removed | `(d - 1,)` |
| $N = \{1, \dots, d\}$, $S \subseteq N$ | all features, and a subset of them (a **coalition**) | — |
| $v(S)$ | value of coalition $S$: the expected prediction when only the features in $S$ are known | scalar |
| $\phi_j$ | Shapley value of feature $j$ for one prediction | scalar |
| $\mathcal{B}$ | background rows used to fill in unknown features | `(n_bg, d)` |
| $n$, $m_{\text{node}}$ | training rows, and training rows reaching a tree node | int |
| $u$ | a grid value for one feature (partial dependence) | scalar |

**Step 1: permutation importance.** For feature $j$ and $R$ random permutations $\pi_1, \dots, \pi_R$ of the $m$ rows:

$$I_j = s(f, \mathbf{X}, \mathbf{y}) - \frac1R\sum_{r=1}^{R} s\big(f, \mathbf{X}^{(\pi_r, j)}, \mathbf{y}\big).$$

The model is never retrained; only predictions are recomputed, $d \times R$ times. Report the mean and the standard deviation across shuffles.
Compute it on *held-out* rows: on training rows it measures what the model memorized. An importance near zero, or slightly
negative, means "no reliance": a negative value is shuffling noise.

**Step 2: impurity importance (MDI).** For each tree, sum over every node that splits on feature $j$ the decrease in
impurity weighted by the share of training rows reaching the node, $\frac{m_{\text{node}}}{n}\,\text{gain}$ (B3.1 Step 2); average over trees; normalize so the importances
sum to 1. Two biases follow from the definition. It is computed on the training rows, so a split that fits noise
counts as much as one that fits signal. And a feature with more distinct values offers more candidate thresholds, so
by chance alone it finds bigger training gains, especially deep in the tree where nodes are small.

**Step 3: partial dependence and ICE.** Write $f(u, \mathbf{x}_{i,-j})$ for the prediction on row $i$ with feature $j$ replaced by $u$. For a grid of values $u$:

$$\text{ICE}_i(u) = f(u, \mathbf{x}_{i,-j}), \qquad \text{PD}_j(u) = \frac1m\sum_{i=1}^{m} f(u, \mathbf{x}_{i,-j}).$$

The PD curve is the average of the ICE curves. If the ICE curves are parallel, the feature acts the same way for every
row (no interactions with other features) and the PD curve tells the whole story. If they cross or fan out, the average hides it. The PD
curve also evaluates rows that may not exist: setting `MedInc` to 15 in a district of tiny, crowded homes creates a combination the model never saw.

**Step 4: the coalition value.** To ask "what does the model predict if it only knows the features in $S$?", fill in
the unknown features from background rows and average (the **interventional** definition). $f(\mathbf{x}_S, \mathbf{b}_{N \setminus S})$ is the prediction on a row that takes the features in $S$ from $\mathbf{x}$ and the rest from the background row $\mathbf{b}$:

$$v(S) = \frac{1}{|\mathcal{B}|}\sum_{\mathbf{b} \in \mathcal{B}} f(\mathbf{x}_S, \mathbf{b}_{N \setminus S}).$$

$v(N) = f(\mathbf{x})$ is the actual prediction, and $v(\varnothing)$ is the average prediction over the background, the **base value**.

**Step 5: Shapley values.** The contribution of feature $j$ is its marginal effect $v(S \cup \{j\}) - v(S)$, averaged over all orders in which the $d$ features could be
added. In an order, the features before $j$ form some set $S$. Of the $d!$ orders, $|S|!\,(d - |S| - 1)!$ have exactly the features of $S$ before $j$
(arrange $S$ before, the rest after). So

$$\phi_j = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|!\,(d - |S| - 1)!}{d!}\Big[v(S \cup \{j\}) - v(S)\Big].$$

For $d = 3$ the weights are $\frac13$ for $|S| = 0$, $\frac16$ for each of the two sets with $|S| = 1$, and $\frac13$ for $|S| = 2$. Shapley values are the only
attribution with these properties (Shapley, 1953):

- **Efficiency:** $\sum_j \phi_j = v(N) - v(\varnothing) = f(\mathbf{x}) - \text{base value}$. The contributions add up exactly to the prediction.
- **Symmetry:** two features that contribute identically in every coalition get equal values.
- **Dummy:** a feature that never changes $v$ gets $\phi_j = 0$.
- **Additivity:** for a sum of models, such as a boosted ensemble, the values add up over the models.

The sum has $2^{d-1}$ terms per feature, so exact enumeration only works for a handful of features. For tree ensembles,
TreeSHAP (Lundberg and colleagues, 2020) computes exact Shapley values in time polynomial in the tree depth and size, which is why
`shap.TreeExplainer` is fast. It offers two value functions: **interventional** (Step 4, with a background set) and
**tree-path-dependent** (the default when no background is given), which follows each tree's training-row counts instead of a background set, approximating a conditional
expectation. They usually agree on the big picture and differ in detail. SHAP (Lundberg and Lee, 2017) is the name
for using Shapley values this way.

## Worked example by hand

A model with three features, $f(\mathbf{x}) = 10x_1 + 5x_2 + 4x_1x_3$, explained at $\mathbf{x} = (1, 1, 1)$ with a single background row $(0, 0, 0)$, so $v(S)$ is $f$ with the features
outside $S$ set to 0. $f(\mathbf{x}) = 19$ and the base value $v(\varnothing) = 0$.

| $S$ | $\varnothing$ | $\{1\}$ | $\{2\}$ | $\{3\}$ | $\{1,2\}$ | $\{1,3\}$ | $\{2,3\}$ | $\{1,2,3\}$ |
|---|---|---|---|---|---|---|---|---|
| $v(S)$ | 0 | 10 | 5 | 0 | 15 | 14 | 5 | 19 |

**Feature 1.** The sets without feature 1 are $\varnothing$, $\{2\}$, $\{3\}$, $\{2, 3\}$:
$\phi_1 = \frac13(10 - 0) + \frac16(15 - 5) + \frac16(14 - 0) + \frac13(19 - 5) = 3.333 + 1.667 + 2.333 + 4.667 = 12$.

**Feature 2.** $\phi_2 = \frac13(5 - 0) + \frac16(15 - 10) + \frac16(5 - 0) + \frac13(19 - 14) = 1.667 + 0.833 + 0.833 + 1.667 = 5$.

**Feature 3, by orders.** List the $3! = 6$ orders and feature 3's marginal contribution in each: $(1, 2, 3)$: $19 - 15 = 4$; $(1, 3, 2)$: $14 - 10 = 4$; $(2, 1, 3)$: $19 - 15 = 4$;
$(2, 3, 1)$: $5 - 5 = 0$; $(3, 1, 2)$: $0 - 0 = 0$; $(3, 2, 1)$: $0$. The average is $12/6 = 2$.

**Check.** $12 + 5 + 2 = 19 = f(\mathbf{x}) - v(\varnothing)$ (efficiency). Read the result: the linear terms go entirely to their own features ($10x_1 \to 10$ for feature 1, $5x_2 \to 5$ for
feature 2), and the interaction $4x_1x_3 = 4$ is split equally between features 1 and 3 (2 each), because neither produces it alone.

```text
== worked example by hand ==
v(S) for every coalition: {}: 0  {1}: 10  {2}: 5  {3}: 0  {1,2}: 15  {1,3}: 14  {2,3}: 5  {1,2,3}: 19
Shapley values [12.0, 5.0, 2.0]   sum 19.0000 = f(x) - v(empty) = 19.0000
```

## Key plot

![Two bar charts over 14 features. Left, impurity importance from a random forest: MedInc first, a pure-noise column (red, hatched) 10th, above four real 0/1 ocean-proximity flags. Right, permutation importance on validation rows: the noise column is at zero, last, while three of the four flags are small but positive](img/b3_4_noise_feature_importance.png)

A random forest on the 8 numeric features, the five 0/1 indicators of the `ocean_proximity` category, and a column of
pure random noise. Impurity importance ranks the noise 10th of 14, above four real features. They are 0/1 flags with a
single possible threshold, while the noise offers thousands of thresholds to fit training rows with. Permutation
importance on validation rows ranks the noise last, at zero, and gives three of the four flags small but real credit (ISLAND, with 5 districts, gets none).

## From scratch in NumPy

**The model.** The B3.3 XGBoost model, retrained with the same settings (788 trees, validation $R^2$ 0.834, RMSE 0.472).

```text
== data ==
train (12384, 8)  val (4128, 8)  (test 4128 rows kept aside)

== the B3.3 model ==
XGBoost: 788 trees   val R^2 0.8335   val RMSE 0.4724
```

**Permutation importance.** Step 1, with 20 shuffles per feature:

```python
def perm_importance(predict, X, y, n_repeats, gen):
    baseline = r2(y, predict(X))
    drops = np.zeros((X.shape[1], n_repeats))
    for j in range(X.shape[1]):
        Xp = X.copy()
        for k in range(n_repeats):
            Xp[:, j] = X[gen.permutation(len(X)), j]    # break the link between column j and the target
            drops[j, k] = baseline - r2(y, predict(Xp))
    return drops.mean(axis=1), drops.std(axis=1)
```

```text
== from scratch: permutation importance ==
Latitude    R^2 drop 1.1886 +- 0.0200
Longitude   R^2 drop 0.9840 +- 0.0160
MedInc      R^2 drop 0.3475 +- 0.0061
AveOccup    R^2 drop 0.1452 +- 0.0058
AveRooms    R^2 drop 0.0879 +- 0.0028
HouseAge    R^2 drop 0.0495 +- 0.0025
AveBedrms   R^2 drop 0.0098 +- 0.0011
Population  R^2 drop 0.0069 +- 0.0013
```

Shuffling latitude drops $R^2$ by 1.19, from 0.83 to $-0.35$: worse than predicting the average price for every district. That
is the bridge's warning in numbers. The model uses latitude and longitude *together* to place a district on the map;
a shuffled latitude moves districts into the ocean or the desert, and the model's answers for those impossible rows are
very wrong. The ranking is still informative: location first, then income, then occupancy. The two weakest
features cost less than 0.01 of $R^2$.

**Partial dependence.** Step 3. Set the column to each grid value for every row, predict, and keep both the per-row
values (ICE) and their mean (PD):

```python
for k, v in enumerate(grid_values):
    Xg = X.copy()
    Xg[:, j] = v
    ice[:, k] = predict(Xg)          # (m,) one ICE value per row
pd_curve = ice.mean(axis=0)          # (g,)
```

```text
== from scratch: partial dependence ==
MedInc =  1.583 ($10,000s): average prediction 1.5842   ICE range 0.295..4.104
MedInc =  3.097 ($10,000s): average prediction 1.9335   ICE range 0.465..4.994
MedInc =  4.611 ($10,000s): average prediction 2.2013   ICE range 0.653..5.256
MedInc =  6.125 ($10,000s): average prediction 2.7335   ICE range 0.962..5.677
MedInc =  7.337 ($10,000s): average prediction 3.2251   ICE range 1.543..5.775
```

Raising median income from $16k to $73k raises the average predicted value from $158k to $323k, and the curve is steeper at
the top. The ICE range at each grid value is wide (about $380k to $470k between the lowest and highest districts) because location shifts
whole curves up and down. The lab saves the PD curve with the ICE curves behind it to `outputs/b3_4/pdp_ice_medinc.png`.

**Exact Shapley values for a 3-feature model.** A smaller XGBoost model on `MedInc`, `Latitude` and `Longitude`, with 200 random training rows as the
background. For each row to explain, `coalition_value` fills the unknown features from every background row (Step 4),
and `shapley_exact` enumerates all 8 coalitions with the Step 5 weights:

```python
def coalition_value(x, S):
    Z = background.copy()                 # (200, 3)
    Z[:, list(S)] = x[list(S)]            # features in S known, the rest from the background
    return float(model3.predict(Z).mean())
```

```text
== from scratch: exact Shapley values for a 3-feature model ==
x = [4.65, 33.82, -118.06]: phi [0.2246, 0.7272, -0.6979]   base 2.0317 + sum 0.2539 = 2.2856   prediction 2.2856
x = [4.82, 38.97, -121.05]: phi [0.1864, -1.3542, 0.9992]   base 2.0317 + sum -0.1686 = 1.8631   prediction 1.8631
x = [2.75, 37.64, -120.99]: phi [-0.2968, -1.2539, 0.7007]   base 2.0317 + sum -0.8500 = 1.1817   prediction 1.1817
x = [4.78, 33.94, -117.92]: phi [0.1342, 0.5281, -0.7768]   base 2.0317 + sum -0.1145 = 1.9172   prediction 1.9172
x = [4.71, 38.46, -122.68]: phi [0.2034, -1.6477, 1.8112]   base 2.0317 + sum 0.3669 = 2.3986   prediction 2.3986
```

Efficiency holds on every row: base value plus contributions equals the prediction. Latitude and longitude often get large values
of *opposite* sign: knowing only the latitude (northern California, say) moves the prediction one way, and
learning the longitude (on the coast, near San Francisco) pulls it back. That is the interaction between the two, split between them.

## With scikit-learn and shap

```text
== with scikit-learn and shap ==
permutation importance: same ranking True   largest gap 0.0048 = 1.4 standard errors: OK
grid ends: ours 1.5829..7.3366   sklearn 1.5825..7.3582
partial dependence matches PartialDependenceDisplay at all 20 of its grid points: OK
exact Shapley values match shap.TreeExplainer(interventional): OK   expected_value 2.0317
```

- `sklearn.inspection.permutation_importance` uses different random shuffles, so equality is the wrong test. The two agree within
  1.4 standard errors on every feature, with the same ranking.
- `PartialDependenceDisplay` builds its grid with slightly different percentile interpolation (its upper end is 7.358 rather than 7.337); evaluated on its grid,
  our function gives exactly its numbers.
- `shap.TreeExplainer(model3, data=masker, feature_perturbation="interventional")` reproduces our enumeration to $10^{-5}$ (XGBoost predicts in float32). By default
  shap silently subsamples the background to 100 rows; the lab passes `shap.maskers.Independent(background, max_samples=200)` to use all 200
  (VERIFY as of 2026-10).

**SHAP on the full B3.3 model.** `shap.TreeExplainer(model)` with the default tree-path-dependent algorithm explains all 4,128
validation rows in a few seconds:

```text
== SHAP on the B3.3 model ==
SHAP values (4128, 8); base value 2.0698 + row sum = prediction for every row: OK
mean |SHAP| by feature: Latitude 0.4820  Longitude 0.3954  MedInc 0.3707  AveOccup 0.1962  AveRooms 0.1429  HouseAge 0.0589  AveBedrms 0.0334  Population 0.0275
district 3496: prediction 5.4243 (actual 5.0000) = base 2.0698
   + 1.3342  MedInc = 8.4331
   + 0.6333  Latitude = 34.0200
   + 0.5571  AveOccup = 1.8589
   + 0.3350  Longitude = -118.6000
   + 0.2533  AveBedrms = 1.4325
   + 0.1199  HouseAge = 36.0000
   + 0.1186  AveRooms = 6.2669
   + 0.0031  Population = 606.0000
saved outputs/b3_4/shap_summary.png and outputs/b3_4/shap_waterfall.png
```

- **Global view:** the mean absolute SHAP value per feature gives the same ranking as permutation importance here. The
  summary (beeswarm) plot in `outputs/b3_4/shap_summary.png` shows one dot per district and feature, coloured by the feature's value. High income
  pushes predictions up by as much as +2.1 ($210k); high latitude (north) mostly pushes them down; low occupancy pushes them up.
- **Local view:** the most expensive predicted district ($542k; its true value is at the dataset's $500k cap, so really "at least $500k") gets
  +$133k from its median income of $84k, +$63k from its latitude (Los Angeles), +$56k from low occupancy (1.9 people per home), and so on.
  The waterfall plot in `outputs/b3_4/shap_waterfall.png` draws this as a staircase from the base value to the prediction.
- B3.3's XGBoost total-gain importance put `MedInc` first and `Latitude` second, while permutation and SHAP put `Latitude` first. Gain,
  permutation and SHAP importance answer different questions, so do not expect their rankings to agree.

**The noise feature** behind the key plot:

```text
== key plot: a noise feature ==
features 14: 8 numeric, 5 ocean_proximity indicators ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'], NOISE
feature     impurity (MDI)  perm. train  perm. val
MedInc              0.3111       0.5192     0.4241
INLAND              0.1466       0.2231     0.1697
AveOccup            0.1075       0.2093     0.1436
AveRooms            0.1037       0.1077     0.0628
Longitude           0.0885       0.1788     0.1435
Latitude            0.0786       0.1635     0.1345
HouseAge            0.0446       0.0741     0.0438
AveBedrms           0.0332       0.0308     0.0064
Population          0.0290       0.0274     0.0045
NOISE               0.0211       0.0169    -0.0009
<1H OCEAN           0.0202       0.0328     0.0193
NEAR OCEAN          0.0095       0.0108     0.0065
NEAR BAY            0.0062       0.0070     0.0049
ISLAND              0.0003       0.0004    -0.0000
NOISE rank out of 14: impurity 10   permutation on training rows 11   permutation on validation rows 14
real features that impurity importance ranks below NOISE: ['<1H OCEAN', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
saved lessons/B/img/b3_4_noise_feature_importance.png
```

The middle column is the second lesson: permutation importance computed on *training* rows also gives the noise credit (0.017),
because the deep forest memorized it. Only held-out rows show the truth: shuffling the noise changes nothing (−0.0009).

## In practice

- **When to use which:** permutation importance on held-out data for "what does this model rely on" (any model, any metric);
  PD with ICE curves for "how does the prediction move with this feature"; SHAP for per-prediction explanations
  (customer-facing reasons, debugging one error) and, through mean |SHAP|, a global ranking. Use impurity importance only as a quick
  first look, never in a report.
- **Failure modes:**
  - *Reading explanations as causes.* "The model uses latitude" does not mean moving a house north lowers its price. All three
    tools describe the model's behaviour on this data distribution. Causal claims need experiments or causal methods.
  - *Correlated features.* Permutation and PD evaluate impossible rows, and correlated features share or swap credit: duplicate a
    feature and each copy's importance drops (Exercise 2). Permute correlated features as a group, or report importance
    per group.
  - *Leaks.* A feature with far more importance than domain sense allows is a leak until proved otherwise (B1.4).
  - *Background choice.* Interventional SHAP values are relative to the background: explain against "all customers" or "approved
    customers" and you get different numbers. State the base value and background you used.
  - *Slow explainers.* `shap.KernelExplainer` (model-agnostic) needs many model calls per row and is approximate; use TreeExplainer for
    trees, and explain a sample, not millions of rows.
- **Defaults that work:** `permutation_importance(model, X_val, y_val, n_repeats=10, random_state=0)`;
  `PartialDependenceDisplay.from_estimator(model, X_val, [feature], kind="both")`; `shap.TreeExplainer(model)` then
  `shap.summary_plot` and `shap.plots.waterfall`, with shap 0.53.0 pinned (its API changes often: VERIFY as of 2026-10).

## Exercises

Save your work to `my-work/B/b3_4_interpretability_shap.md` (or `.py`) and ask for a review.

1. **By hand:** (a) For $f(\mathbf{x}) = 3x_1 + 2x_2 + 6x_2x_3$ at $\mathbf{x} = (2, 1, 1)$ with background row $(0, 0, 0)$, tabulate $v(S)$ for all 8 coalitions and compute $\phi_1, \phi_2, \phi_3$. Check efficiency and say how the
   interaction term was split. (b) The model $f(x_1, x_2) = x_1x_2$ is evaluated on three rows: $(1, 1)$, $(2, 3)$, $(3, -1)$. Compute the ICE curve of each row and the partial
   dependence $\text{PD}_1(u)$ for $u = 0, 1, 2$. What does the PD curve hide? (c) A model ignores feature 2 completely. What are its permutation importance and its Shapley value,
   and which Shapley property guarantees the second? (d) In two sentences, explain why a column of random noise gets positive impurity importance.
2. **Code:** Starting from the lab: (a) add an exact copy of `MedInc` as a ninth column, retrain the B3.3 XGBoost model, and compute
   permutation importance (validation) and mean |SHAP| for both copies. Compare them with the single-copy numbers and explain
   what happened. (b) Write `group_permutation_importance(predict, X, y, groups, n_repeats, gen)`, which shuffles all columns in a group with the *same*
   permutation, and report the importance of the group {Latitude, Longitude} against the sum of their individual importances. Explain the difference.
3. **Think:** A lender's product manager wants to send every rejected applicant "the top three reasons you were declined",
   taken from SHAP values of the credit model, and to tell the risk team that "SHAP proves debt-to-income causes defaults". What is sound
   in this plan, what is not, and what would you put in place before shipping it? Consider the background and base value,
   correlated features, stability between model versions, and what a rejected applicant can actually change.

## Takeaways

1. Permutation importance = score drop when one column is shuffled, measured on held-out rows. It works for any model,
   but correlated features make it misleading: impossible rows, shared credit.
2. Impurity importance (MDI) comes from training data and favours features with many split points; it can rank pure
   noise above real low-cardinality features. Do not report it.
3. Partial dependence averages the model's prediction as one feature varies; ICE curves show the per-row curves the
   average can hide.
4. Shapley values split one prediction into per-feature contributions that add up exactly to (prediction − base value),
   by averaging each feature's marginal contribution over all orders; interactions are shared.
5. TreeSHAP makes exact Shapley values cheap for tree ensembles. All these tools explain the model, not the world:
   never read them as causal effects.

**What builds on this:** [B4.6](b4_6_fairness_model_cards.md) (explanations in fairness audits and model cards);
[D2.5](../D/d2_5_vision_error_analysis.md) (Grad-CAM, the vision counterpart); [J2.5](../J/j2_5_documentation_governance.md)
(documenting explanations for governance); P2 M2 (a SHAP summary and the top five features in plain English) and the capstone.

**References:** L. S. Shapley, "A value for n-person games", in *Contributions to the Theory of Games II*, 1953.
L. Breiman, "Random forests", *Machine Learning* 45, 2001 (permutation importance). S. M. Lundberg, S.-I. Lee, "A
unified approach to interpreting model predictions", *NeurIPS*, 2017. S. M. Lundberg et al., "From local explanations
to global understanding with explainable AI for trees", *Nature Machine Intelligence* 2, 2020. A. Goldstein, A. Kapelner,
J. Bleich, E. Pitkin, "Peeking inside the black box: visualizing statistical learning with plots of individual conditional
expectation", *Journal of Computational and Graphical Statistics* 24, 2015. C. Strobl, A.-L. Boulesteix, A. Zeileis,
T. Hothorn, "Bias in random forest variable importance measures", *BMC Bioinformatics* 8, 2007. C. Molnar, *Interpretable
Machine Learning* (VERIFY link: https://christophm.github.io/interpretable-ml-book/).
