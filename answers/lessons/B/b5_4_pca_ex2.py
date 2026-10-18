# %% [markdown]
# # B5.4 PCA — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b5_4_pca_ex2.py   (CPU, ~20 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

FloatArray = npt.NDArray[np.float64]
SEED = 0
T0 = time.perf_counter()


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
X = np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [4.0, 7.0]])  # (4, 2)
mean = X.mean(axis=0)
Xc = X - mean
G = Xc.T @ Xc
S = G / (len(X) - 1)
lam, Q = np.linalg.eigh(S)
lam, Q = lam[::-1], Q[:, ::-1]
Q = Q * np.sign(Q[np.argmax(np.abs(Q), axis=0), np.arange(2)])  # largest-|entry| positive
print(f"(a) mean {mean.tolist()}   Xc^T Xc {G.tolist()}   S {np.round(S, 4).tolist()}")
print(f"    eigenvalues of S {np.round(lam, 4).tolist()}   v1 {np.round(Q[:, 0], 4).tolist()}   v2 {np.round(Q[:, 1], 4).tolist()}")
print(f"(b) explained-variance ratio {np.round(lam / lam.sum(), 4).tolist()}")
z = Xc @ Q[:, 0]
X_hat = mean + np.outer(z, Q[:, 0])
print(f"(c) PC1 scores {np.round(z, 4).tolist()}   (= [-8, -2, 4, 6]/sqrt(5))")
print(f"    rank-1 reconstruction {np.round(X_hat, 4).tolist()}")
print(f"    squared error {np.sum((X - X_hat) ** 2):.4f}   sigma_2^2 = (n-1) lambda_2 = {3 * lam[1]:.4f}")
x_new = np.array([6.0, 6.0])
z_new = (x_new - mean) @ Q[:, 0]
x_new_hat = mean + z_new * Q[:, 0]
print(f"(d) new point score {z_new:.4f}   reconstruction {np.round(x_new_hat, 4).tolist()}   "
      f"squared error {np.sum((x_new - x_new_hat) ** 2):.4f}")

# %%
section("data")
X_all, y_all = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.25, random_state=SEED, stratify=y_all)
print(f"train {X_tr.shape}   test {X_te.shape}")

# %%
section("grid search over n_components")
pipe = Pipeline([("pca", PCA(svd_solver="full")), ("clf", LogisticRegression(max_iter=2000))])
grid = {"pca__n_components": [5, 10, 15, 20, 25, 30, 40, 50, 64]}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy").fit(X_tr, y_tr)
res = search.cv_results_
print(f"{'k':>3} {'CV accuracy':>12} {'std':>6}")
for k, m, sd in zip(grid["pca__n_components"], res["mean_test_score"], res["std_test_score"]):
    print(f"{k:>3} {m:>12.4f} {sd:>6.4f}")
print(f"~ mean fit time per fold: k = 5 {res['mean_fit_time'][0]:.3f} s, k = 64 {res['mean_fit_time'][-1]:.3f} s")
best_k = search.best_params_["pca__n_components"]
best = res["mean_test_score"].max()
best_std = res["std_test_score"][int(np.argmax(res["mean_test_score"]))]
within = [k for k, m in zip(grid["pca__n_components"], res["mean_test_score"]) if m >= best - best_std]
print(f"best k = {best_k} (CV accuracy {best:.4f});  smallest k within one std of the best: {min(within)}")

# %%
section("variance rules")
full = PCA(svd_solver="full").fit(X_tr)
cum = np.cumsum(full.explained_variance_ratio_)
for target in [0.90, 0.95]:
    k = int(np.searchsorted(cum, target) + 1)
    acc = res["mean_test_score"][grid["pca__n_components"].index(min(grid["pca__n_components"], key=lambda g: abs(g - k)))]
    print(f"{target:.0%} rule keeps k = {k}   (CV accuracy at the nearest grid value: {acc:.4f})")

# %%
section("test set, scored once")
chosen = min(within)
final = Pipeline([("pca", PCA(n_components=chosen, svd_solver="full")), ("clf", LogisticRegression(max_iter=2000))])
final.fit(X_tr, y_tr)
print(f"chosen k = {chosen}: test accuracy {final.score(X_te, y_te):.4f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
