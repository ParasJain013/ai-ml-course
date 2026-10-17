# Part B. Classical ML — detailed syllabus

Course weeks 7–12. Weekly detail for Part B of [`COURSE_OUTLINE.md`](COURSE_OUTLINE.md), which stays
the source of truth: the topic ids, Core/Depth labels, projects P2 and P3, the XGBoost paper and
checkpoint CP2 below are taken from it unchanged. Written 2026-10-16.

**Goal:** frame, build, evaluate and explain classical models end to end, without leakage, and know
when a gradient-boosted tree beats a neural net.

## At a glance

| Week | Course wk | Theme | Topics | Hours | Project | GPU |
|---|---|---|---|---|---|---|
| B1 | 7 | Workflow & data | B1.1–B1.5 | 12.5 | P2 M0 | no |
| B2 | 8 | Linear models & simple classifiers | B2.1–B2.5 | 12.5 | P2 M1 | no |
| B3 | 9 | Trees & ensembles | B3.1–B3.4 + XGBoost paper | 13.5 | P2 M2 | no |
| B4 | 10 | Evaluation | B4.1–B4.6 | 15 | **P2 due** | no |
| B5 | 11 | Unsupervised learning | B5.1–B5.6 | 13.5 | P3 M0–M1 | no |
| B6 | 12 | Recommenders, time series | B6.1–B6.4 | 12.5 | **P3 due** · CP2 | no |

About 80 hours in total. Each week's hours include about 1 hour for the weekly quiz and cheat sheet
(1.5 in B6, which adds the CP2 self-check). Topic hours cover reading the lesson and doing its lab
and exercises.

**How the hours are counted.** *Reading* is working through the lessons, including the worked
examples by hand (and, in B3, the paper). *Coding* is the labs and the code exercises. *Project* is
P2 or P3.

**B4 is the heaviest week (15 h)** because it carries six topics and the P2 deadline, and the outline
does not let either move. Two ways to keep it inside your budget: finish the P2 M2 milestone early
in B3, and if B4 still runs long, do the P2 fairness audit (the last P2 step) before the B4.5 lab,
since B4.5 does not feed P2.

**If a week runs long,** drop its Depth topics first (B2.5, B5.2, B5.3, B6.2, B6.4) and come back
to them later. None of them feeds P2 or P3's required parts, and none is on the fast path.

**Libraries this part needs** (Step 2 pins the versions): `numpy`, `pandas`, `matplotlib`, `scipy`,
`scikit-learn`, `xgboost`, `lightgbm`, `shap`, `optuna` (Bayesian tuning in B4.3), `fairlearn`
(only to check your from-scratch fairness metrics in B4.6), `umap-learn` (B5.5) and `statsmodels`
(ARIMA and exponential smoothing in B6.4). Library APIs here change between versions: VERIFY (as of
2026-10) when Step 2 pins them.

**Datasets** (downloaded into `data/`, never committed; VERIFY availability): scikit-learn's bundled
sets (breast cancer, diabetes, wine, digits), California housing via `fetch_california_housing`,
UCI Adult (Census Income) via OpenML, the UCI SMS Spam Collection, MovieLens 100K from GroupLens, and
a small daily or monthly time series (for example a public electricity-demand or bike-share series).

**Hardware:** everything in Part B runs on the laptop CPU. The slowest labs (random forests, gradient
boosting with tuning, UMAP) take a minute or two at default settings on datasets up to ~50,000 rows,
well within 15 GB of RAM. No week needs a GPU. XGBoost and LightGBM have GPU modes, but on data this
size they are not faster in practice, so the course does not use them. If your laptop is busy, every
lab also runs unchanged on a free Colab or Kaggle CPU notebook.

---

## B1 · Workflow & data (course week 7)

**Learning objectives**

- I can turn a vague business request into an ML problem statement: target, unit of prediction, metric, and the baseline it must beat.
- I can split data into train, validation and test sets, and explain what each one is for and why the test set is touched once.
- I can clean a messy table (wrong types, missing values, outliers, duplicates) and justify every decision in a cleaning log.
- I can identify the PII columns in a dataset and decide which to drop, hash or coarsen before modelling. `[RAI]`
- I can build a scikit-learn `Pipeline` with a `ColumnTransformer` that scales numeric columns and one-hot encodes categorical ones, and explain why the preprocessing must live inside the pipeline.
- I can name and give an example of target leakage, train/test contamination and temporal leakage, and catch each in code.
- I can fit linear regression with the closed form and with gradient descent from scratch and match scikit-learn's coefficients.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| B1.1 | The ML workflow: framing, baselines, train/validation/test splits, what "generalize" means | Core | 2 |
| B1.2 | Data cleaning: types, missing values, outliers, duplicates; `[RAI]` PII handling and data minimization | Core | 2 |
| B1.3 | Feature engineering: scaling, categorical encoding, dates, interactions, Pipelines and ColumnTransformer | Core | 2.5 |
| B1.4 | Data leakage: target leakage, train/test contamination, temporal leakage, and how to catch each | Core | 2 |
| B1.5 | Linear regression from scratch (closed form and GD) versus scikit-learn | Core | 2 |

**Hours:** reading 4 · coding 6.5 · project 1 · quiz and cheat sheet 1 · **total 12.5**

**Labs**

- **B1.1:** a `train_val_test_split` function from scratch (shuffled, seeded, with a stratify
  option), checked against two calls of `train_test_split`. Three baselines on California housing
  and on breast cancer: predict the mean, predict the majority class, and a one-feature rule. Every
  later model in Part B is reported next to these numbers.
- **B1.2:** a cleaning function for a deliberately dirtied copy of a dataset (numbers stored as
  strings, sentinel values such as `-999`, near-duplicate rows, impossible values). Mean, median and
  "missing indicator" imputation from scratch, checked against `SimpleImputer`. IQR and z-score
  outlier flags from scratch. A PII pass that drops direct identifiers, hashes a join key and
  coarsens a birth date to an age band.
- **B1.3:** a standard scaler, min-max scaler and one-hot encoder from scratch, each with
  `fit`/`transform` methods, checked against scikit-learn's. Date features (day of week, month,
  "days since") and one interaction term. Then the same preprocessing as a `ColumnTransformer` inside
  a `Pipeline`, with an assertion that both give identical matrices.
- **B1.4:** three planted leaks on one dataset: a feature computed from the target, a scaler fitted
  on all the data before splitting, and a random split of time-ordered data. Measure the inflated
  score of each, fix it, and measure again. Write a `leakage_check` helper that flags features whose
  single-feature score is suspiciously close to perfect.
- **B1.5:** linear regression with the normal equations (reusing A2.4) and with batch gradient
  descent (reusing A4.4), on standardized features. Match `LinearRegression` coefficients to 4
  decimal places, and show what happens to GD without scaling.

**Project milestone: P2 M0, frame the problem (1 h).** Pick a tabular classification dataset with a
sensitive attribute you can audit later (UCI Adult is the default; VERIFY availability). Write the
problem statement: target, unit of prediction, who acts on the prediction, the metric and why, and
the B1.1 baseline score. List the columns you will exclude for leakage or PII reasons, with one line
each.

**Prerequisites:** A1.3 (pandas), A2.4 (normal equations), A4.4 (gradient descent), A6.1 (EDA).

**Resources** (optional)

- scikit-learn User Guide, "Common pitfalls and recommended practices": https://scikit-learn.org/stable/common_pitfalls.html (VERIFY link)
- scikit-learn User Guide, "Pipelines and composite estimators": https://scikit-learn.org/stable/modules/compose.html (VERIFY link)
- Martin Zinkevich, "Rules of Machine Learning: Best Practices for ML Engineering" (Google): https://developers.google.com/machine-learning/guides/rules-of-ml (VERIFY link)

**GPU:** no.

**Used later in:** every model in the course (B1.1, B1.3); P2 and every later project's leakage audit
(B1.4); time-series validation in B6.3 (B1.4); logistic regression in B2.1 (B1.5); privacy in I4.1
and J2.5 (B1.2).

---

## B2 · Linear models & simple classifiers (course week 8)

**Learning objectives**

- I can derive the gradient of log-loss for logistic regression, implement it with gradient descent from scratch, and match scikit-learn's predicted probabilities.
- I can explain L1, L2 and elastic net as MAP estimates with Laplace and Gaussian priors, and show on a plot why L1 produces exact zeros and L2 does not.
- I can implement k-nearest neighbours, explain why features must be scaled for it, and show the curse of dimensionality with a simulation.
- I can build a multinomial naive Bayes spam filter with Laplace smoothing from scratch, and explain what the "naive" assumption is and why the model still works.
- I can explain the maximum-margin idea, the hinge loss and the kernel trick, and say when an RBF-kernel SVM is a sensible choice. *(Depth)*

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| B2.1 | Logistic regression: sigmoid, log-loss, gradient descent from scratch | Core | 2.5 |
| B2.2 | Regularization: L1, L2 and elastic net, read as MAP priors from A5.5 | Core | 2 |
| B2.3 | k-nearest neighbours: distance metrics and the curse of dimensionality | Core | 1.5 |
| B2.4 | Naive Bayes: Gaussian and multinomial, with a spam-filter example | Core | 1.5 |
| B2.5 | Support vector machines: maximum margin, hinge loss and the kernel trick | Depth | 2 |

**Hours:** reading 3.5 · coding 6 · project 2 · quiz and cheat sheet 1 · **total 12.5**

**Labs**

- **B2.1:** a numerically stable sigmoid and log-loss (no `log(0)`), the vectorized gradient
  checked with your A4.2 gradient checker, and batch GD on breast cancer. Match
  `LogisticRegression(penalty=None)` probabilities to 3 decimal places (VERIFY the parameter
  spelling for the pinned version). Plot the decision boundary on two features. This is the CP2
  "30-minute" exercise; time yourself on a second attempt.
- **B2.2:** ridge in closed form and lasso by coordinate descent with soft-thresholding, both from
  scratch, checked against `Ridge` and `Lasso`. Plot the coefficient paths as the penalty grows, and
  the L1 diamond versus the L2 circle meeting the loss contours. Add L2 to your B2.1 logistic
  regression.
- **B2.3:** kNN classification and regression from scratch, reusing the A1.2 vectorized distance
  matrix, checked against `KNeighborsClassifier`. Accuracy with and without scaling. A simulation
  of the ratio between nearest and farthest neighbour distances as the dimension grows from 2 to 1,000.
- **B2.4:** a multinomial naive Bayes spam filter on the SMS Spam Collection: tokenizer, word
  counts, Laplace smoothing and log-probabilities from scratch, checked against `MultinomialNB` with
  `CountVectorizer`. Gaussian naive Bayes from scratch on the wine dataset, checked against `GaussianNB`.
- **B2.5 (Depth):** a linear SVM trained by sub-gradient descent on the hinge loss, compared with
  `LinearSVC`. An RBF-kernel `SVC` on a two-moons dataset, with the margin and support vectors
  plotted for several values of `C` and `gamma`.

**Project milestone: P2 M1, pipeline and baseline (2 h).** A single `Pipeline` that takes the raw
dataframe and produces predictions: cleaning, encoding, scaling and a regularized logistic regression.
Score it on a validation set against the M0 baseline. Write the first version of the leakage audit:
for every feature, when is its value known relative to the prediction time?

**Prerequisites:** B1.1, B1.3, B1.5; A4.2 (gradient checking), A5.3 (Bayes' theorem), A5.5 (MLE and
MAP), A6.4 (cross-entropy); A4.5 (Lagrange multipliers) for B2.5.

**Resources** (optional)

- James, Witten, Hastie, Tibshirani & Taylor, *An Introduction to Statistical Learning with Applications in Python* (free PDF), chapters 4, 6 and 9: https://www.statlearning.com/ (VERIFY link)
- scikit-learn User Guide, "Linear Models": https://scikit-learn.org/stable/modules/linear_model.html (VERIFY link)
- Stanford CS229 lecture notes (Andrew Ng), the parts on logistic regression and SVMs: https://cs229.stanford.edu/ (VERIFY link)

**GPU:** no.

**Used later in:** the classification head of every neural network in Parts C–F (B2.1); weight
decay in C3.5 and AdamW in C3.1 (B2.2); nearest-neighbour search in embeddings and RAG in E1.3 and
I2.1 (B2.3); the TF-IDF baseline in E1.2 (B2.4).

---

## B3 · Trees & ensembles (course week 9)

**Learning objectives**

- I can compute Gini impurity and entropy for a split by hand, and grow a decision tree from scratch that matches scikit-learn's on a small dataset.
- I can explain why bagging plus feature subsampling reduces variance, and use out-of-bag error as a free validation estimate.
- I can implement gradient boosting for squared error as "fit a small tree to the residuals, take a small step", and explain the roles of the learning rate, tree depth and number of trees.
- I can train XGBoost and LightGBM with early stopping, and explain two ideas the XGBoost paper adds to plain gradient boosting.
- I can explain a model with permutation importance, partial dependence and SHAP values, and say why impurity-based feature importance can mislead.
- I can name the situations where a gradient-boosted tree is the right first choice over a neural net, and the ones where it is not.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| B3.1 | Decision trees: Gini and entropy splits built from scratch | Core | 2.5 |
| B3.2 | Random forests: bagging, feature subsampling, out-of-bag error | Core | 2 |
| B3.3 | Gradient boosting: fitting residuals from scratch, then XGBoost / LightGBM in practice | Core | 3 |
| B3.4 | Interpretability: permutation importance, partial dependence and SHAP values | Core | 2 |
| — | Paper reading guide: *XGBoost: A Scalable Tree Boosting System* (Chen, Guestrin, 2016), with B3.3 | — | 1 |

**Hours:** reading 4.5 · coding 6 · project 2 · quiz and cheat sheet 1 · **total 13.5**

**Labs**

- **B3.1:** a CART classifier from scratch: Gini and entropy, exhaustive threshold search on numeric
  features, recursive splitting with `max_depth` and `min_samples_leaf`, and prediction. Compare its
  accuracy and the top split with `DecisionTreeClassifier`. Plot training versus validation accuracy
  as depth grows to see overfitting.
- **B3.2:** a random forest from scratch on top of your B3.1 tree: bootstrap samples, random
  feature subsets per split, majority vote, and out-of-bag error. Compare with
  `RandomForestClassifier(oob_score=True)`, and plot validation error against the number of trees
  for one tree, bagging and a full forest.
- **B3.3:** gradient boosting for regression from scratch (residuals, shallow trees from
  `DecisionTreeRegressor`, shrinkage), checked against `GradientBoostingRegressor` with the same
  settings. Then XGBoost and LightGBM on California housing with early stopping on a validation
  set, compared on score and training time. Show what a learning rate that is too high does to the
  validation curve.
- **B3.4:** permutation importance from scratch, checked against
  `sklearn.inspection.permutation_importance`. A one-feature partial dependence curve from scratch,
  checked against `PartialDependenceDisplay`. Exact Shapley values for a 3-feature model by
  enumerating every coalition, then `shap.TreeExplainer` on the B3.3 model: a summary plot and one
  single-prediction explanation. Add a random-noise feature and show that impurity-based importance
  ranks it higher than permutation importance does.

**Project milestone: P2 M2, the challenger model (2 h).** Add a gradient-boosted model (XGBoost or
LightGBM) to the same pipeline and compare it with the M1 logistic regression on the validation set.
Produce a SHAP summary plot and explain the top five features in plain English. Check whether any
top feature is a leak you missed in M1, and update the leakage audit. If you can, start M3's
cross-validation here, because B4 is the heaviest week.

**Prerequisites:** B1.1, B1.3, B1.4, B2.1; A6.2 (the bootstrap, for bagging), A6.4 (entropy), A4.4
(gradient descent, for boosting as gradient descent on the loss).

**Resources** (optional)

- Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System" (2016), arXiv 1603.02754: https://arxiv.org/abs/1603.02754 (VERIFY link)
- Christoph Molnar, *Interpretable Machine Learning* (free online), chapters on permutation importance, PDP and SHAP: https://christophm.github.io/interpretable-ml-book/ (VERIFY link)
- LightGBM documentation, "Parameters Tuning": https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html (VERIFY link)

**GPU:** no.

**Used later in:** P2 and P12 (B3.3); isolation forests in B5.6 (B3.1); gradient boosting on lag
features in B6.4 (B3.3); explaining models in the capstone and in D2.5's Grad-CAM comparison (B3.4).

---

## B4 · Evaluation (course week 10)

**Teaching order this week:** B4.1 → B4.2 → B4.3 → B4.4 first, because P2's final milestone uses
them. Then B4.6, which P2 also needs, then B4.5. This swaps the last two topics only in the order
you study them; the ids do not change.

**Learning objectives**

- I can compute precision, recall, F1, ROC-AUC and PR-AUC from a confusion matrix or from scores by hand, and explain when PR-AUC is more honest than ROC-AUC.
- I can draw a reliability diagram, compute the Brier score, and say whether a model's probabilities can be trusted as probabilities.
- I can choose a regression metric (MAE, RMSE, MAPE, R²) and a classification threshold from the real cost of each kind of error.
- I can pick the right cross-validation scheme (k-fold, stratified, group, time-series) for a dataset and explain what goes wrong with the wrong one.
- I can tune hyperparameters with grid, random and Bayesian search inside cross-validation without leaking the test set, and explain nested cross-validation.
- I can handle class imbalance with class weights, resampling or threshold tuning, and show which of them changes calibration.
- I can read a learning curve and a validation curve and say whether a model needs more data, more capacity or more regularization.
- I can compute demographic parity and equalized-odds differences for a model, explain the trade-off between them, and write a model card. `[RAI]`

**Topics** (listed in id order; see the teaching order above)

| Id | Topic | Level | Hours |
|---|---|---|---|
| B4.1 | Classification metrics: confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC, calibration | Core | 2 |
| B4.2 | Regression metrics, and choosing a metric that matches the real cost of errors | Core | 1 |
| B4.3 | Cross-validation (k-fold, stratified, group, time-series) and hyperparameter tuning (grid, random, Bayesian) | Core | 2 |
| B4.4 | Class imbalance: class weights, resampling, threshold tuning | Core | 1.5 |
| B4.5 | The bias–variance trade-off, learning curves and validation curves | Core | 1.5 |
| B4.6 | `[RAI]` Fairness: group metrics (demographic parity, equalized odds), auditing a model, a first model card | Core | 2 |

**Hours:** reading 4 · coding 6 · project 4 · quiz and cheat sheet 1 · **total 15**

**Labs**

- **B4.1:** the confusion matrix, precision, recall, F1, the ROC curve with AUC (trapezoid rule) and
  the PR curve with average precision, all from scratch from scores, checked against
  `sklearn.metrics`. A reliability diagram and Brier score from scratch, then `CalibratedClassifierCV`
  to fix an overconfident model.
- **B4.2:** MAE, RMSE, MAPE and R² from scratch. One dataset where the MAE and RMSE winners differ,
  and one where MAPE explodes because of near-zero targets. A cost-weighted threshold picker: given
  the cost of a false positive and a false negative, choose the threshold that minimizes expected cost.
- **B4.3:** k-fold, stratified k-fold, group k-fold and an expanding-window time-series split from
  scratch, checked against scikit-learn's splitters. Show a group leak (the same customer in train
  and validation) inflating the score. Grid search versus random search versus Optuna on the same
  budget for a LightGBM model, all inside cross-validation, with a final score on an untouched test
  set. A small nested cross-validation to show how optimistic the inner score is.
- **B4.4:** a 1%-positive dataset. Compare accuracy, F1 and PR-AUC for doing nothing, class weights,
  random oversampling and undersampling (from scratch), and threshold tuning. Plot the reliability
  diagram for each and show which ones break calibration.
- **B4.5:** a bias–variance decomposition by simulation: fit polynomials of increasing degree to
  many resampled training sets and measure bias², variance and noise. Learning curves and validation
  curves from scratch, checked against `learning_curve` and `validation_curve`.
- **B4.6:** demographic parity difference, equal-opportunity difference and equalized-odds
  difference from scratch on UCI Adult, checked against `fairlearn.metrics`. Show that a single
  threshold for everyone and per-group thresholds trade one metric against another. Fill in a model
  card for the model (the template comes from `STYLE_GUIDE.md`).

**Project milestone: P2 due, evaluate, audit and document (4 h).** Cross-validate and tune both
models with the right splitter, and report the test score once, with a bootstrap confidence interval
(A6.2). Justify the metric and the decision threshold from the costs you wrote in M0. Finish the
leakage audit, run the fairness audit on the sensitive attribute, and write the model card with
intended use, metrics by group and known limitations. P2 needs **≥ 70%** on its rubric to pass CP2.

**Prerequisites:** B1.1, B1.4, B2.1, B2.2, B3.3; A6.2 (bootstrap CIs), A6.3a (hypothesis tests, for
comparing models fairly).

**Resources** (optional)

- scikit-learn User Guide, "Cross-validation: evaluating estimator performance": https://scikit-learn.org/stable/modules/cross_validation.html (VERIFY link)
- Google Machine Learning Crash Course, "Classification" module (thresholds, ROC and AUC): https://developers.google.com/machine-learning/crash-course (VERIFY link)
- Mitchell et al., "Model Cards for Model Reporting" (2019), arXiv 1810.03993: https://arxiv.org/abs/1810.03993 (VERIFY link)

**GPU:** no.

**Used later in:** every project from P2 on (all of B4); debugging training runs in C4.3 (B4.5);
RAG and agent evals in I2.4 and I3.3 (B4.1, B4.3); drift monitoring in J2.2 (B4.1, B4.2); model cards
in J2.5 and L2.1 (B4.6); the fairness checks in D2.5 and E1.4 (B4.6).

---

## B5 · Unsupervised learning (course week 11)

**Learning objectives**

- I can implement k-means with k-means++ initialization from scratch, and choose k with the elbow method and silhouette scores while saying why neither is decisive.
- I can explain when k-means fails (non-round, unequal-size or noisy clusters) and pick hierarchical clustering or DBSCAN instead. *(Depth)*
- I can fit a Gaussian mixture with the EM algorithm from scratch and explain soft assignments. *(Depth)*
- I can implement PCA via the SVD, choose the number of components from explained variance, and reconstruct the data from them.
- I can make a t-SNE or UMAP plot and list three ways it misleads (cluster sizes, distances between clusters, sensitivity to hyperparameters).
- I can detect anomalies with a z-score rule, an isolation forest and a density-based score, and evaluate them when only a few labels exist.

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| B5.1 | k-means from scratch, and choosing k (elbow, silhouette) | Core | 2 |
| B5.2 | Hierarchical clustering and DBSCAN: when clusters aren't round | Depth | 1.5 |
| B5.3 | Gaussian mixture models and the EM algorithm | Depth | 2 |
| B5.4 | PCA from scratch via SVD (A3.2), explained variance | Core | 2 |
| B5.5 | t-SNE and UMAP: what they show and how they mislead | Core | 1.5 |
| B5.6 | Anomaly detection: statistical rules, isolation forest, density-based scores | Core | 1.5 |

**Hours:** reading 4 · coding 6.5 · project 2 · quiz and cheat sheet 1 · **total 13.5**

**Labs**

- **B5.1:** k-means (Lloyd's algorithm) with k-means++ initialization from scratch, checked against
  `KMeans` up to a relabelling of the clusters. Elbow plot and silhouette score from scratch,
  checked against `silhouette_score`. Show a run that lands in a bad local minimum, and how `n_init`
  fixes it.
- **B5.2 (Depth):** single, complete and Ward linkage on a small dataset with a dendrogram
  (`scipy.cluster.hierarchy`). DBSCAN from scratch (core points, region growing, noise), checked
  against `DBSCAN` on two-moons and concentric circles, where k-means fails.
- **B5.3 (Depth):** EM for a 1-D and then a 2-D Gaussian mixture from scratch: the E-step
  responsibilities, the M-step updates and the log-likelihood rising every iteration. Check against
  `GaussianMixture`, and plot the fitted ellipses (links to A3.3 and A5.2).
- **B5.4:** PCA from scratch with `np.linalg.svd` on centred data (reusing A3.2): components,
  explained-variance ratio and reconstruction. Check against `PCA` (allowing for sign flips). On the
  digits dataset, plot the cumulative explained variance and reconstructions at 2, 10 and 30 components.
- **B5.5:** t-SNE (`sklearn.manifold.TSNE`) and UMAP on digits, each at three perplexity or
  `n_neighbors` settings. Then three "it lies" demos: equal-size clusters drawn as different sizes,
  distances between clusters that mean nothing, and apparent clusters in pure noise.
- **B5.6:** z-score and IQR rules (reusing B1.2), an isolation forest from scratch on top of random
  splits, checked against `IsolationForest`, and a k-nearest-neighbour distance score. Evaluate all
  three with PR-AUC on a dataset with a few labelled anomalies.

**Project milestone: P3 M0–M1, data, views and first clusters (2 h).** Choose one dataset for both
segmentation and anomaly detection, such as customer transactions, retail orders or website sessions
(VERIFY availability). Engineer per-entity features, standardize them, and produce PCA and UMAP views.
Fit k-means for several k, record the elbow and silhouette results, and write down your first reading
of the segments.

**Prerequisites:** A3.2 (SVD), A3.3 (covariance), A5.2 (Gaussian and multivariate Gaussian), A5.5
(MLE, for EM); B1.2 (outlier rules), B1.3 (scaling), B2.3 (distances), B3.1 (trees, for isolation
forests), B4.1 (PR-AUC).

**Resources** (optional)

- scikit-learn User Guide, "Clustering": https://scikit-learn.org/stable/modules/clustering.html (VERIFY link)
- Wattenberg, Viégas & Johnson, "How to Use t-SNE Effectively", Distill (2016): https://distill.pub/2016/misread-tsne/ (VERIFY link)
- UMAP documentation, "How to Use UMAP" and "Understanding UMAP": https://umap-learn.readthedocs.io/ (VERIFY link)

**GPU:** no.

**Used later in:** P3 (all of B5); visualizing embeddings in E1.3 and I2.1 (B5.4, B5.5); the bottleneck
idea in autoencoders in G1.1 (B5.4); EM's ELBO view in VAEs in G1.2 (B5.3); drift and anomaly
alerts in J2.2 (B5.6); low-rank thinking in J1.3 (B5.4).

---

## B6 · Recommenders, time series (course week 12)

**Learning objectives**

- I can build a user–item matrix, implement user-based collaborative filtering and matrix factorization (by alternating least squares or SGD) from scratch, and beat a popularity baseline.
- I can explain the difference between explicit ratings and implicit feedback, and why "not clicked" does not mean "disliked".
- I can compute precision@k, recall@k and NDCG@k by hand, and explain the retrieve-then-rank pattern used by production recommenders and by RAG. *(Depth)*
- I can decompose a time series into trend and seasonality, build lag and rolling features without leaking the future, and beat a seasonal-naive baseline under walk-forward validation.
- I can fit ARIMA and exponential smoothing and say when they beat gradient boosting on lag features, and when they don't. *(Depth)*

**Topics**

| Id | Topic | Level | Hours |
|---|---|---|---|
| B6.1 | Recommender systems: collaborative filtering and matrix factorization from scratch, implicit feedback | Core | 2.5 |
| B6.2 | Ranking metrics (precision@k, NDCG) and the retrieve-then-rank pattern | Depth | 1.5 |
| B6.3 | Time-series forecasting: trend, seasonality, lag features, naive baselines, walk-forward validation | Core | 2.5 |
| B6.4 | ARIMA / exponential smoothing versus gradient boosting on lag features | Depth | 1.5 |

**Hours:** reading 3 · coding 5 · project 3 · quiz, cheat sheet and CP2 self-check 1.5 · **total 12.5**

**Labs**

- **B6.1:** on MovieLens 100K: a popularity baseline, user-based collaborative filtering with cosine
  similarity (reusing A1.5), and matrix factorization by SGD with biases and L2 regularization from
  scratch. Compare RMSE on a per-user time-ordered split, not a random one (B1.4). Then turn the
  ratings into implicit "watched" data and train with sampled negatives.
- **B6.2 (Depth):** precision@k, recall@k, MAP@k and NDCG@k from scratch, checked against
  `sklearn.metrics.ndcg_score`. A two-stage recommender: retrieve 100 candidates by dot product of
  the B6.1 factors, then rerank them with a LightGBM model on a few extra features, and show the
  NDCG@10 gain.
- **B6.3:** trend and seasonality by moving averages and seasonal decomposition, plotted. Naive,
  seasonal-naive and moving-average baselines. Lag and rolling-window features built with
  `shift()` so they never see the future, and a test that fails if they do. Walk-forward validation
  from scratch (reusing the B4.3 splitter), with MAE reported per fold.
- **B6.4 (Depth):** simple exponential smoothing from scratch, then Holt–Winters and ARIMA with
  `statsmodels`, against LightGBM on the B6.3 lag features under the same walk-forward folds. Report
  which wins and by how much, on a short series and on a long one.

**Project milestone: P3 due, stability, anomalies and the write-up (3 h).** Check cluster stability
(rerun with different seeds and on bootstrap resamples, and compare assignments with the adjusted
Rand index). Name and describe each segment with feature profiles. Run two anomaly detectors from
B5.6 and inspect the top 20 anomalies by hand. Write a short report a non-ML reader could act on:
what each segment is, what to do about it, which anomalies look real, and what the analysis cannot
tell you. P3 needs **≥ 70%** on its rubric to pass CP2.

**Prerequisites:** A1.5 (cosine similarity), A3.2 (SVD, for matrix factorization), A4.4 (SGD); B1.4
(temporal leakage), B2.2 (regularization), B3.3 (gradient boosting), B4.3 (time-series
cross-validation).

**Resources** (optional)

- Koren, Bell & Volinsky, "Matrix Factorization Techniques for Recommender Systems", *IEEE Computer* (2009) (VERIFY link: find via the authors' pages or a library; do not rely on an unofficial PDF)
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (free online, 3rd edition), chapters 2–3, 5, 8–9: https://otexts.com/fpp3/ (VERIFY link)
- GroupLens, MovieLens datasets page: https://grouplens.org/datasets/movielens/ (VERIFY link)

**GPU:** no.

**Used later in:** embeddings as learned factors in E1.3 (B6.1); retrieve-then-rank and reranking in
RAG in I2.3, and retrieval metrics in I2.4 (B6.2); drift over time in J2.2 (B6.3); a tabular or
forecasting capstone in P12 (B6.3, B6.4).

---

## You're ready for Part C when…

This is checkpoint **CP2**. Answer each item without notes. The answers belong in your quiz reviews,
not here.

1. You can implement logistic regression with gradient descent from scratch, from a blank file, in
   **30 minutes**, and match scikit-learn's probabilities.
2. You can explain target leakage, train/test contamination and temporal leakage, with one concrete
   example each and how you would catch it.
3. You can say why preprocessing has to sit inside a `Pipeline` that is cross-validated as a whole.
4. Given a fraud problem with 0.5% positives, you can pick and justify a metric and a threshold, and
   say why accuracy and ROC-AUC alone would mislead.
5. You can choose the cross-validation scheme for grouped data and for time-ordered data, and
   explain what nested cross-validation protects against.
6. You can explain L1 versus L2 regularization as priors, and say which one gives a sparse model.
7. You can explain why gradient boosting usually beats a random forest on tabular data, what the
   learning rate and number of trees trade off, and when you would still reach for a neural net.
8. You can explain a single prediction with SHAP and say why impurity importance can mislead.
9. You can read a learning curve and say whether the model needs more data, more capacity or more regularization.
10. You can compute a demographic-parity and an equalized-odds difference, and explain why a model
    usually cannot satisfy both at once.
11. You can explain PCA in one sentence in terms of the SVD, and name two ways a t-SNE or UMAP plot misleads.
12. You can beat a seasonal-naive baseline under walk-forward validation, and say why a random split
    of a time series is wrong.
13. **P2 and P3 each score ≥ 70%** on their rubrics, and your Part B weekly quizzes average **≥ 80%**.

If two or more items fail, take one catch-up week on the topic ids your reviews flagged before you
start Part C. That is the outline's checkpoint rule.
