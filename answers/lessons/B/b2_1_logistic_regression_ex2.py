# %% [markdown]
# # B2.1 Logistic regression — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b2_1_logistic_regression_ex2.py   (CPU, ~3 s)

# %%
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()
FEATURES = ["mean radius", "mean texture", "mean smoothness", "mean concavity", "mean symmetry"]
BATCH_SIZE = 32
EPOCH_CHECKPOINTS = [1, 5, 20, 100]


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


def sigmoid(z: FloatArray) -> FloatArray:
    """Stable sigmoid: exp of a non-positive number only. z (...) -> (...)."""
    ez = np.exp(-np.abs(z))  # in (0, 1]: never overflows
    return np.where(z >= 0, 1.0 / (1.0 + ez), ez / (1.0 + ez))


def mean_log_loss(z: FloatArray, y: FloatArray) -> float:
    """Mean log-loss from logits: log(1 + e^z) - y z. z (n,), y (n,) -> scalar."""
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def gradient(theta: FloatArray, A: FloatArray, y: FloatArray) -> FloatArray:
    """Gradient of the mean log-loss. theta (k,), A (n, k), y (n,) -> (k,)."""
    return A.T @ (sigmoid(A @ theta) - y) / len(y)


def numerical_gradient(f: Callable[[FloatArray], float], x: FloatArray, h: float = 1e-6) -> FloatArray:
    """Central differences, one coordinate at a time. x (k,) -> (k,)."""
    g = np.zeros_like(x)
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = h
        g[j] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def fit_logistic_gd(
    A: FloatArray, y: FloatArray, lr: float, max_iters: int = 100_000, tol: float = 1e-7
) -> tuple[FloatArray, int]:
    """Full-batch GD from zero until |grad| < tol. A (n, k), y (n,) -> (theta (k,), steps)."""
    theta = np.zeros(A.shape[1])
    for step in range(max_iters):
        g = gradient(theta, A, y)
        if np.linalg.norm(g) < tol:
            return theta, step
        theta -= lr * g
    return theta, max_iters


def run_epochs(
    A: FloatArray, y: FloatArray, lr: float, epochs: int, batch_size: int | None,
    gen: np.random.Generator,
) -> list[float]:
    """GD for a number of passes over the data; batch_size None = full batch. -> loss after each epoch."""
    n = len(y)
    theta = np.zeros(A.shape[1])
    losses = []
    for _ in range(epochs):
        if batch_size is None:
            theta -= lr * gradient(theta, A, y)
        else:
            order = gen.permutation(n)  # reshuffle every epoch
            for start in range(0, n, batch_size):
                rows = order[start:start + batch_size]
                theta -= lr * gradient(theta, A[rows], y[rows])
        losses.append(mean_log_loss(A @ theta, y))
    return losses


# %%
section("exercise 1 check")
x3, y3 = np.array([-1.0, 0.0, 2.0]), np.array([0.0, 0.0, 1.0])
w, b = 1.0, -0.5
z3 = w * x3 + b
p3 = sigmoid(z3)
per = np.logaddexp(0.0, z3) - y3 * z3
gw, gb = float(np.mean((p3 - y3) * x3)), float(np.mean(p3 - y3))
print(f"(a) z {show(z3)}  p {show(p3)}  per-example loss {show(per)}  mean {per.mean():.4f}")
print(f"(b) dL/dw {gw:.4f}  dL/db {gb:.4f}")
w2, b2 = w - 0.5 * gw, b - 0.5 * gb
z3b = w2 * x3 + b2
print(f"(c) w {w2:.4f}  b {b2:.4f}  p {show(sigmoid(z3b))}  mean loss {mean_log_loss(z3b, y3):.4f}")
print(f"(d) odds ratio e^1 = {np.e:.4f}")
A3 = np.column_stack([np.ones(3), x3])
for steps in [100, 1_000, 10_000]:
    th = np.array([b, w])
    for _ in range(steps):
        th -= 0.5 * gradient(th, A3, y3)
    print(f"(e) after {steps:>6} more steps: b {th[0]:.3f}  w {th[1]:.3f}  loss {mean_log_loss(A3 @ th, y3):.6f}")

# %%
section("data and standardization")
bc = load_breast_cancer()
names = list(bc.feature_names)
X = bc.data[:, [names.index(f) for f in FEATURES]].astype(np.float64)  # (569, 5)
y = (bc.target == 0).astype(np.float64)  # 1 = malignant
gen = np.random.default_rng(SEED)
parts: list[list[IntArray]] = [[], [], []]
for c in np.unique(y):  # the same stratified 60/20/20 split as the lab (B1.1's function)
    rows = gen.permutation(np.flatnonzero(y == c))
    n_te, n_va = round(0.2 * len(rows)), round(0.2 * len(rows))
    parts[2].append(rows[:n_te])
    parts[1].append(rows[n_te:n_te + n_va])
    parts[0].append(rows[n_te + n_va:])
i_tr, i_va, _ = (gen.permutation(np.concatenate(p)) for p in parts)
mu, sd = X[i_tr].mean(0), X[i_tr].std(0)
A_tr = np.column_stack([np.ones(len(i_tr)), (X[i_tr] - mu) / sd])  # (343, 6)
A_va = np.column_stack([np.ones(len(i_va)), (X[i_va] - mu) / sd])  # (113, 6)
y_tr, y_va = y[i_tr], y[i_va]
print(f"train {A_tr.shape}  val {A_va.shape}")

# %%
section("gradient check")
th0 = np.random.default_rng(1).normal(size=A_tr.shape[1])
g_a = gradient(th0, A_tr, y_tr)
g_n = numerical_gradient(lambda t: mean_log_loss(A_tr @ t, y_tr), th0)
rel = np.linalg.norm(g_a - g_n) / (np.linalg.norm(g_a) + np.linalg.norm(g_n))
assert rel < 1e-6, rel
print(f"relative error {rel:.1e}: OK")

# %%
section("full-batch GD versus scikit-learn")
L = np.linalg.eigvalsh(A_tr.T @ A_tr / len(y_tr))[-1] / 4  # curvature bound (lesson, step 6)
theta, steps = fit_logistic_gd(A_tr, y_tr, lr=1.0 / L)
sk = LogisticRegression(C=np.inf, max_iter=10_000, tol=1e-10).fit(A_tr[:, 1:], y_tr)
p_mine, p_sk = sigmoid(A_va @ theta), sk.predict_proba(A_va[:, 1:])[:, 1]
print(f"steps {steps}  theta {show(theta)}")
print(f"max |p_mine - p_sklearn| on val {np.max(np.abs(p_mine - p_sk)):.1e}")
np.testing.assert_allclose(p_mine, p_sk, atol=5e-4)  # "3 decimals" = within half a unit of the 3rd
print("match to 3 decimals: OK")

# %%
section("mini-batch versus full batch, same examples processed")
full = run_epochs(A_tr, y_tr, 1.0 / L, max(EPOCH_CHECKPOINTS), None, np.random.default_rng(SEED))
mini = run_epochs(A_tr, y_tr, 1.0 / L, max(EPOCH_CHECKPOINTS), BATCH_SIZE, np.random.default_rng(SEED))
updates_per_epoch = -(-len(y_tr) // BATCH_SIZE)  # ceil(343 / 32) = 11
print(f"optimum train loss {mean_log_loss(A_tr @ theta, y_tr):.4f}; mini-batch makes {updates_per_epoch} updates per epoch")
for e in EPOCH_CHECKPOINTS:
    print(f"epoch {e:>3} ({e * len(y_tr):>6} examples): full batch {full[e - 1]:.4f}   mini-batch {mini[e - 1]:.4f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
