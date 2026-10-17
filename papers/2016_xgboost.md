# XGBoost: A Scalable Tree Boosting System (Tianqi Chen, Carlos Guestrin, 2016)

Read in week 09 with B3.3. Link: KDD 2016; arXiv 1603.02754, https://arxiv.org/abs/1603.02754 (VERIFY link). Time: ~1 h.

## Why read it

XGBoost is the library that made gradient-boosted trees the default model for tabular data, and this is its
paper. It is short and readable, and it splits cleanly into two halves. The first half is the machine-learning idea: a
regularized objective and second-order (Newton) leaf values that turn "fit a tree to the residuals" into a principled
optimization. You derived it in B3.3 Step 6; the paper is the source. The second half is systems engineering: how to
find splits fast on data that does not fit in cache or memory, and how to handle sparse inputs. Interviewers ask about the
first half ("what does XGBoost add to gradient boosting?"); the second half is a good model of how an ML system paper
argues for its design.

## Before you start

- [B3.1](../lessons/B/b3_1_decision_trees.md): split search on sorted features, impurity gain.
- [B3.3](../lessons/B/b3_3_gradient_boosting.md): the boosting loop, pseudo-residuals, the regularized objective and the gain formula (Step 6).
- [A4.1](../lessons/A/a4_1_derivatives.md): first and second derivatives; a second-order Taylor expansion $f(a + h) \approx f(a) + f'(a)h + \frac12 f''(a)h^2$.
- [B2.1](../lessons/B/b2_1_logistic_regression.md): the gradient $p - y$ and the curvature $p(1 - p)$ of log-loss.

## How to read it

1. **Read closely, Section 2 (Tree Boosting in a Nutshell).** All of it, with a pen. Rederive the leaf weight and the split
   gain in this course's notation before reading the paper's version (it uses $\Omega$, $g_i$, $h_i$, $T$, $w$, $\gamma$, $\lambda$ as B3.3 does). Section 2.3 (shrinkage and column
   subsampling) is half a page.
2. **Read for the idea, Sections 3.1–3.2 (exact greedy and approximate split finding).** The exact algorithm is B3.1's
   sorted scan using $G$ and $H$ instead of class counts. For the approximate one, understand what "global" and "local" proposals
   mean, and look at the experiment comparing them.
3. **Read for the idea, Section 3.4 (sparsity-aware split finding).** This is what the B3.3 lab relied on when it passed
   `NaN` bedroom counts straight to XGBoost.
4. **Skim, Section 3.3 (weighted quantile sketch).** Understand *why* $h_i$ is the weight (one paragraph); skip the appendix
   with the proofs.
5. **Skim, Section 4 (system design).** Get one sentence for each of column blocks, cache-aware access and
   out-of-core blocks. You will not implement these.
6. **Skim, Section 6 (evaluations).** Look at what is compared, on which datasets, and what is measured: speed or accuracy.
7. **Read last, Sections 1, 5 and 7.** The introduction's claims about adoption make more sense once you know what the
   system is.

## Key ideas

- **Regularized objective.** Each tree $f$ pays $\Omega(f) = \gamma T + \frac12\lambda\lVert\mathbf{w}\rVert^2$: a cost per leaf and an L2 penalty on leaf values. With $\Omega = 0$ the method is ordinary gradient tree
  boosting (Friedman, 2001).
- **Second-order approximation.** Expanding the loss to second order around the current predictions turns each tree's
  objective into a sum of independent parabolas, one per leaf. This gives the closed-form leaf value $w_j^* = -G_j/(H_j + \lambda)$ and a closed-form
  score for any tree structure. One code path then serves every twice-differentiable loss: only $g_i$ and $h_i$ change.
- **Split gain from the structure score.** The gain of a split is the structure score of the two children minus that of
  the parent, minus $\gamma$, so pruning is built into the criterion: a split whose gain does not exceed $\gamma$ is not worth its extra leaf.
- **Shrinkage and column subsampling** as extra regularizers, alongside the penalty.
- **Approximate split finding with a weighted quantile sketch.** Propose candidate thresholds at quantiles of each
  feature, weighted by $h_i$, instead of trying every value; this makes very large and distributed data tractable.
- **Sparsity-aware split finding.** Each split learns a default direction for missing (or zero, in sparse data)
  entries by trying both, and the scan only visits non-missing entries.

## The one equation or figure to understand

The split gain (Section 2.2), in B3.3's notation:

$$\text{gain} = \frac12\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma, \qquad G = \sum_{i} g_i,\; H = \sum_{i} h_i.$$

Make sure you can (1) derive it from $w^* = -G/(H + \lambda)$ and the per-leaf objective $-\frac12\frac{G^2}{H + \lambda}$; (2) say what it becomes for squared error ($g_i = F - y_i$, $h_i = 1$) and check that, with
$\lambda = \gamma = 0$, it is half the drop in summed squared error that B3.1's regression tree uses; (3) say what $\lambda$ does to a leaf holding very few rows
(small $H$: its weight is shrunk hard towards 0) and what $\gamma$ does to the depth of the tree.

## Questions while reading

Answer these in `my-work/papers/2016_xgboost.md`; the reference answers are in `answers/papers/2016_xgboost.md`.

1. (Section 2.1) Write $\Omega(f)$ and say what each term penalizes. What does the objective reduce to when $\Omega = 0$, and which
   regularization tools did plain gradient boosting already have without it?
2. (Section 2.2) Derive $w_j^*$ and the structure score $-\frac12\sum_j\frac{G_j^2}{H_j + \lambda} + \gamma T$ from the second-order approximation. Why is the second-order term worth having, compared with
   fitting trees to the gradient alone as in B3.3 Steps 2–3?
3. (Section 2.2) For binary log-loss with $F$ the logit, what are $g_i$ and $h_i$? Which rows get the largest $h_i$, and what does that mean for
   their influence on the leaf values?
4. (Section 2.3) What two extra techniques does this section add, and what reasons does the paper give for column
   subsampling?
5. (Sections 3.2–3.3) What is the difference between the global and the local variant of the approximate algorithm, and
   what does the experiment comparing them suggest about how many candidate splits each needs? Why are the quantiles
   weighted by $h_i$?
6. (Section 3.4) How does the sparsity-aware algorithm choose where missing values go, and why does its cost scale with
   the number of non-missing entries? How does this compare with imputing the training median, as the B3.3 lab did for
   scikit-learn?
7. (Section 4) In one sentence each: what problem do the column block, cache-aware access and out-of-core blocks solve?
8. (Sections 1 and 6) What evidence does the paper give for XGBoost's quality: what is measured and against what? What
   would you want to see before concluding that it is more *accurate* than other gradient-boosting implementations, rather than
   faster?

## What changed since

VERIFY (as of 2026-10): library features and benchmark conclusions move quickly.

- **Kept everywhere:** the second-order objective with L2-penalized leaf values, learned default directions for missing
  values, shrinkage and column subsampling. LightGBM and CatBoost both use gradients and Hessians the same way.
- **Replaced as the default way to find splits:** histogram-based split finding. LightGBM (Ke and colleagues, NeurIPS 2017)
  bins features once and grows trees leaf-wise, with further speedups (gradient-based one-side sampling, exclusive feature
  bundling). XGBoost added its own `hist` method, and in recent versions it is the default `tree_method`; the exact
  greedy algorithm is now mostly a teaching tool.
- **Added later:** native categorical-feature support, GPU training (`device="cuda"`), monotonic and interaction
  constraints, multi-output trees. CatBoost (Prokhorenkova and colleagues, NeurIPS 2018) added ordered boosting and
  target statistics designed to avoid a target leak when encoding categorical features.
- **Still holds broadly:** on medium-sized tabular datasets, tuned gradient-boosted trees remain hard to beat with deep
  learning (Grinsztajn, Oyallon and Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular
  data?", NeurIPS 2022 Datasets and Benchmarks). Pre-trained tabular models such as TabPFN (Hollmann and colleagues,
  *Nature*, 2025) are competitive on small datasets; check the current state before quoting either side.

## Connection to the course

- [B3.3](../lessons/B/b3_3_gradient_boosting.md): Step 6 is the paper's Section 2; the lab's XGBoost model uses
  shrinkage, row and column subsampling and native `NaN` handling.
- [B3.4](../lessons/B/b3_4_interpretability_shap.md): TreeSHAP explains XGBoost models; XGBoost's `total_gain` importance
  is the paper's split gain summed per feature (its default `gain` importance is the average per split).
- [B4.3](../lessons/B/b4_3_cross_validation_tuning.md): tuning `learning_rate`, `max_depth`, `min_child_weight`, `reg_lambda`, `gamma`
  with cross-validation; `min_child_weight` is a lower bound on $H$ in a leaf.
- [B4.4](../lessons/B/b4_4_class_imbalance.md) and [B6.4](../lessons/B/b6_4_arima_vs_boosting.md): boosting with class
  weights and on lag features.
- P2 M2 (the gradient-boosted challenger model) and P12.
