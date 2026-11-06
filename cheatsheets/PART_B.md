# Part B cheat sheet: Classical ML

Two pages for the whole of Part B. Details, more code and more pitfalls are in the weekly sheets:
[B1 Workflow & data](w07_b1.md) · [B2 Linear models & simple classifiers](w08_b2.md) ·
[B3 Trees & ensembles](w09_b3.md) · [B4 Evaluation](w10_b4.md) ·
[B5 Unsupervised learning](w11_b5.md) · [B6 Recommenders, time series](w12_b6.md).

## Formulas
| Name | Formula | Shapes |
|---|---|---|
| Least squares | $\mathbf{A}^\top\mathbf{A}\boldsymbol\theta = \mathbf{A}^\top\mathbf{y}$ ($\mathbf{A} = [\mathbf{1}, \mathbf{X}]$); MSE gradient $\frac2n\mathbf{A}^\top(\mathbf{A}\boldsymbol\theta - \mathbf{y})$ | `(d+1,)` |
| Logistic regression | $p = \sigma(\mathbf{w}^\top\mathbf{x} + b)$; log-loss $\log(1 + e^z) - yz$; $\partial\ell/\partial z = p - y$ | `(n,)` |
| Ridge / lasso | $+\lambda\lVert\mathbf{w}\rVert_2^2$ (Gaussian prior, shrinks) / $+\lambda\lVert\mathbf{w}\rVert_1$ (Laplace prior, zeros); soft threshold $S(a, t)$ | `(d,)` |
| kNN distances | $\lVert\mathbf{q}\rVert^2 + \lVert\mathbf{x}\rVert^2 - 2\mathbf{Q}\mathbf{X}^\top$; cosine $= 1 - \frac{\mathbf{u}\cdot\mathbf{v}}{\lVert\mathbf{u}\rVert\lVert\mathbf{v}\rVert}$ | `(m, n)` |
| Naive Bayes | $\log\pi_c + \sum_w x_w\log\theta_{cw}$, $\theta_{cw} = \frac{N_{cw} + \alpha}{N_c + \alpha V}$ | `(K, V)` |
| SVM | $\frac{\lambda}{2}\lVert\mathbf{w}\rVert^2 + \frac1n\sum\max(0, 1 - y_if_i)$; margin $2/\lVert\mathbf{w}\rVert$; RBF $e^{-\gamma\lVert\mathbf{x} - \mathbf{z}\rVert^2}$ | |
| Tree splits | Gini $1 - \sum p_k^2$; entropy $-\sum p_k\log_2 p_k$; gain $= I(S) - \sum\frac{m_c}{m}I(S_c)$; leaf = mean or majority | |
| Bagging, forests | OOB share $1/e = 0.368$; $\operatorname{Var}[\bar T] = \rho\sigma^2 + (1 - \rho)\sigma^2/B$ | |
| Boosting | $F_M = F_0 + \eta\sum f_m$, $f_m$ fits $-\partial\ell/\partial F$; XGBoost leaf $-G/(H + \lambda)$ | `(n,)` |
| SHAP | $\sum_j\phi_j = f(\mathbf{x}) - \mathbb{E}f$; Shapley weights $\frac{\lvert S\rvert!(d - \lvert S\rvert - 1)!}{d!}$ | `(d,)` |
| Classification metrics | precision $\frac{TP}{TP + FP}$, recall $\frac{TP}{P}$, AUC $= P(s_+ > s_-)$, AP baseline = prevalence, Brier $\frac1n\sum(\hat p - y)^2$ | |
| Regression metrics | MAE (median), RMSE (mean), pinball loss (quantile $\tau = \frac{c_u}{c_u + c_o}$), $R^2 = 1 - \frac{SSE}{SST}$ | |
| Cost threshold | flag if $p > \frac{c_{FP}}{c_{FP} + c_{FN}}$ on calibrated $p$ | |
| Bias–variance | $\mathbb{E}(y - \hat f)^2 = \sigma^2 + \text{bias}^2 + \text{variance}$ | |
| Fairness | $\text{SR}_a = \text{TPR}_a\pi_a + \text{FPR}_a(1 - \pi_a)$; DP diff (SR), EOdds = max(TPR gap, FPR gap) | |
| k-means / ARI | $J = \sum\lVert\mathbf{x}_i - \mathbf{c}_{z_i}\rVert^2$; ARI $= \frac{\text{index} - \mathbb{E}}{\max - \mathbb{E}}$ over pairs | `(k, d)` |
| GMM / EM | $\gamma_{ik} \propto \pi_k\mathcal{N}(\mathbf{x}_i\mid\boldsymbol\mu_k, \boldsymbol\Sigma_k)$; M-step = weighted MLE; BIC $= -2\ell + p\log n$ | `(n, K)` |
| PCA | $\mathbf{X}_c = \mathbf{U}\boldsymbol\Sigma\mathbf{V}^\top$; $\mathbf{Z} = \mathbf{X}_c\mathbf{V}_k$; ratio $\sigma_j^2/\sum\sigma_i^2$ | `(n, k)` |
| Isolation forest | $s = 2^{-E[h]/c(\psi)}$, $c(m) = 2H(m - 1) - \frac{2(m - 1)}{m}$ | |
| Matrix factorization | $\hat r = \mu + b_u + b_i + \mathbf{p}_u\cdot\mathbf{q}_i$, fitted on observed cells only; $\mathbf{p}_u \mathrel{+}= \eta(e\mathbf{q}_i - \lambda\mathbf{p}_u)$ | `(k,)` |
| Ranking | P@k, R@k; AP@k; NDCG $= \sum\frac{\mathrm{rel}_j}{\log_2(j + 1)}$ ÷ ideal over **all** relevant items | |
| Forecasting | seasonal naive $y_{T+h-m}$; SES $\ell_t = \alpha y_t + (1 - \alpha)\ell_{t-1}$; skill $1 - \mathrm{MAE}/\mathrm{MAE}_{\text{snaive}}$ | |

## Code
```python
make_pipeline(ColumnTransformer([...]), model); cross_val_score(pipe, X, y, cv=cv)   # every fitted step inside the folds
StratifiedKFold(5, shuffle=True, random_state=0); GroupKFold(5); TimeSeriesSplit(5, test_size=h)
make_pipeline(StandardScaler(), LogisticRegression(C=1.0)); RidgeCV(alphas=np.logspace(-3, 3, 13))
lgb.LGBMRegressor(n_estimators=5000, learning_rate=0.05, verbose=-1).fit(X_fit, y_fit, eval_X=X_es, eval_y=y_es,
                                                                         callbacks=[lgb.early_stopping(100)])
permutation_importance(model, X_val, y_val, n_repeats=10, random_state=0); shap.TreeExplainer(model).shap_values(X_val)
average_precision_score(y, s); CalibratedClassifierCV(model, method="isotonic", cv=5); TunedThresholdClassifierCV(model)
KMeans(n_clusters=k, n_init=10, random_state=0); adjusted_rand_score(a, b); PCA(n_components=0.95)
IsolationForest(random_state=0).fit(X); ndcg_score(y_true_all, y_score, k=10); lgb.LGBMRanker(objective="lambdarank")
s.shift(12).rolling(12).mean(); ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12).fit()
```

## Defaults and rules of thumb
- Frame first (decision, prediction time, metric, baseline); split to match how the model will be used (time, group, stratified).
- Baselines always: mean/majority, prevalence, popularity, seasonal naive. A model that cannot beat them is not learning.
- Standardize for anything with distances, penalties, kernels or gradient descent; trees need no scaling.
- Tabular default: gradient boosting with early stopping on a slice that is not the reporting set; linear model as the readable baseline.
- Tune on log grids or with Optuna inside CV; nested CV or an untouched test set for the final number, with a bootstrap CI.
- Threshold from costs or capacity on out-of-fold predictions; calibrate before reading probabilities as risks.
- Unsupervised results are claims: check stability (ARI across seeds and bootstraps) and whether people can act on them.
- Forecasts: features at least as old as the horizon, walk-forward validation, every fold reported.
- `[RAI]` Model card for every project; metrics by group with sizes and CIs; PII minimized; exposure by group for recommenders.

## Pitfalls
- Leakage: preprocessing fitted on all rows, features recorded after the outcome, target encoding in-fold, random splits of time.
- Choosing on the test set; reporting the best of many scores; early stopping on the reporting rows.
- Accuracy or ROC-AUC on rare positives; MAPE with zeros; $R^2$ across datasets; RMSE for a ranking problem.
- Impurity importance and SHAP read as causes; correlated features split the credit.
- Trees cannot extrapolate (prices, trends): model differences.
- k-means on unscaled data; labels compared with `==`; t-SNE/UMAP sizes and gaps read as facts.
- Missing ratings treated as zero; IDCG from retrieved items only.

## Interview one-liners
- **Leakage:** information unavailable at prediction time, or from the evaluation rows, reaching the model.
- **Regularization:** a prior on the weights; L1 sparse, L2 shrunk.
- **Gradient boosting:** gradient descent in function space with shallow trees, shrunk by $\eta$.
- **Random forest:** bagging plus random features per split, less correlated trees, lower variance.
- **Calibration:** $P(y = 1 \mid \hat p = q) = q$; separate from ranking quality.
- **Nested CV:** the inner loop tunes, the outer loop scores the tuning procedure.
- **PCA:** SVD of centred data; maximum variance = minimum reconstruction error.
- **Collaborative filtering / MF:** learn from who interacted with what; factors fitted to observed cells only.
- **Walk-forward validation:** train on the past, forecast the next horizon, roll forward.
- **ARIMA vs boosting:** classical for short single series; boosting for many series with known drivers.
