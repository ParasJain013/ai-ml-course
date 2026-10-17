# %% [markdown]
# # B2.1 Logistic regression — lab
# Lesson: lessons/B/b2_1_logistic_regression.md
# Run: uv run python labs/B/b2_1_logistic_regression.py   (CPU, defaults: ~5 s)
# GPU: no

# %%
from __future__ import annotations

import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:  # plain `uv run`: draw to files, never open windows
    matplotlib.use("Agg")
    # matplotlib 3.11 warns when plt.show() runs under Agg; under `uv run` it is a harmless no-op
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()


def repo_root() -> Path:
    """Find the course folder (works under `uv run` and in VS Code cells)."""
    here = Path(globals().get("__file__", Path.cwd())).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("Run labs from inside the ai-ml-course folder.")


ROOT = repo_root()
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "b2_1"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE only lets the perfect-separation demo iterate for longer.
GPU_SCALE = False
MAX_ITERS = 100_000  # GD steps allowed on the 5-feature problem (it needs about 2,000)
GRAD_TOL = 1e-7  # stop when the gradient norm falls below this
SEP_CS = [1.0, 1e2, 1e4, 1e6]  # weaker and weaker L2 penalties in the perfect-separation demo
SEP_ITERS = 200_000 if GPU_SCALE else 20_000  # lbfgs iterations allowed there
VAL_FRAC, TEST_FRAC = 0.2, 0.2
FEATURES = ["mean radius", "mean texture", "mean smoothness", "mean concavity", "mean symmetry"]

# %%
# Data: the Wisconsin breast cancer set bundled with scikit-learn. 569 tumours, 30 measurements of
# the cell nuclei in a biopsy image. We make malignant the positive class (y = 1), the event we
# want to detect; scikit-learn's own coding is the reverse (1 = benign).
section("data")
bc = load_breast_cancer()
all_names = list(bc.feature_names)
X_full = bc.data.astype(np.float64)  # (569, 30)
y_all = (bc.target == 0).astype(np.float64)  # (569,) 1 = malignant
cols = [all_names.index(f) for f in FEATURES]
X_all = X_full[:, cols]  # (569, 5)


def train_val_test_split(
    n: int,
    val_frac: float,
    test_frac: float,
    gen: np.random.Generator,
    stratify: FloatArray | None = None,
) -> tuple[IntArray, IntArray, IntArray]:
    """Shuffle row indices and cut them three ways. n rows -> (train, val, test) index arrays."""
    # from labs/B/b1_1_ml_workflow.py
    if stratify is None:
        groups = [np.arange(n)]
    else:
        groups = [np.flatnonzero(stratify == c) for c in np.unique(stratify)]  # one group per class
    parts: list[list[IntArray]] = [[], [], []]
    for rows in groups:
        rows = gen.permutation(rows)
        n_test, n_val = round(test_frac * len(rows)), round(val_frac * len(rows))
        parts[2].append(rows[:n_test])
        parts[1].append(rows[n_test:n_test + n_val])
        parts[0].append(rows[n_test + n_val:])
    train, val, test = (gen.permutation(np.concatenate(p)) for p in parts)
    return train, val, test


i_tr, i_va, i_te = train_val_test_split(
    len(y_all), VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED), stratify=y_all
)
X_tr, y_tr, X_va, y_va = X_all[i_tr], y_all[i_tr], X_all[i_va], y_all[i_va]
n, d = X_tr.shape
print(f"all rows {X_full.shape}  malignant share {y_all.mean():.4f}")
print(f"features used ({d}): {FEATURES}")
print(f"train X {X_tr.shape}  val X {X_va.shape}  (test {len(i_te)} rows kept aside)")
print(f"malignant share: train {y_tr.mean():.4f}  val {y_va.mean():.4f}")
print("feature means (train):", dict(zip(FEATURES, np.round(X_tr.mean(0), 4).tolist())))

# %%
# Worked example by hand: one feature, four points, one gradient-descent step from zero.
section("worked example by hand")


def sigmoid(z: FloatArray) -> FloatArray:
    """Numerically stable logistic sigmoid, elementwise. z (...) -> (...)."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))  # exp of a non-positive number: never overflows
    ez = np.exp(z[~pos])  # z < 0, so exp(z) < 1: never overflows
    out[~pos] = ez / (1.0 + ez)
    return out


def log_loss_from_logits(z: FloatArray, y: FloatArray) -> float:
    """Mean log-loss computed from logits, never from probabilities. z (n,), y (n,) -> scalar."""
    # -[y log p + (1 - y) log(1 - p)] with p = sigmoid(z) simplifies to log(1 + e^z) - y z
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


x4 = np.array([0.0, 1.0, 2.0, 3.0])
y4 = np.array([0.0, 1.0, 0.0, 1.0])
w4, b4 = 0.0, 0.0
z4 = w4 * x4 + b4
p4 = sigmoid(z4)
print(f"x = {x4.tolist()}   y = {y4.tolist()}   start w = 0, b = 0")
print(f"z = {show(z4)}   p = {show(p4)}   loss = {log_loss_from_logits(z4, y4):.4f} (= log 2)")
err4 = p4 - y4
gw, gb = float(np.mean(err4 * x4)), float(np.mean(err4))
print(f"p - y = {show(err4)}   dL/dw = mean((p - y) x) = {gw:.4f}   dL/db = mean(p - y) = {gb:.4f}")
w4, b4 = w4 - 1.0 * gw, b4 - 1.0 * gb
z4 = w4 * x4 + b4
print(f"one step with lr = 1: w = {w4:.4f}  b = {b4:.4f}")
print(f"z = {show(z4)}   p = {show(sigmoid(z4))}   loss = {log_loss_from_logits(z4, y4):.4f}")
th4 = np.zeros(2)
A4 = np.column_stack([np.ones(4), x4])  # (4, 2)
for _ in range(10_000):
    th4 -= 1.0 * A4.T @ (sigmoid(A4 @ th4) - y4) / 4
print(f"after 10,000 steps: b = {th4[0]:.4f}  w = {th4[1]:.4f}   "
      f"p = {show(sigmoid(A4 @ th4))}   loss = {log_loss_from_logits(A4 @ th4, y4):.4f}")
print(f"at the optimum mean(p - y) = {np.mean(sigmoid(A4 @ th4) - y4):.1e}: "
      f"average predicted probability = observed positive rate")

# %%
# Numerical stability: the textbook formulas overflow or take log(0); the logit form does not.
section("from scratch: stable sigmoid and log-loss")
z_big = np.array([-800.0, -30.0, 0.0, 30.0, 800.0])
y_big = np.array([1.0, 1.0, 1.0, 0.0, 0.0])  # the extreme ones are confidently wrong
with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
    p_naive = 1.0 / (1.0 + np.exp(-z_big))
    loss_naive = -(y_big * np.log(p_naive) + (1 - y_big) * np.log(1 - p_naive))
print(f"z = {z_big.tolist()}   y = {y_big.tolist()}")
print(f"naive per-example loss:   {loss_naive.tolist()}")
print(f"stable per-example loss:  {(np.logaddexp(0.0, z_big) - y_big * z_big).tolist()}")
np.testing.assert_allclose(sigmoid(np.linspace(-50, 50, 1001)), expit(np.linspace(-50, 50, 1001)),
                           rtol=1e-12, atol=0)
print("stable sigmoid matches scipy.special.expit: OK")

# %%
# From scratch: the loss and gradient for a whole dataset, checked with the A4.2 gradient checker.
section("from scratch: gradient check")


def add_intercept(X: FloatArray) -> FloatArray:
    """Prepend a column of ones. X (n, d) -> (n, d + 1)."""
    return np.column_stack([np.ones(len(X)), X])


def loss_and_grad(theta: FloatArray, A: FloatArray, y: FloatArray) -> tuple[float, FloatArray]:
    """Mean log-loss and its gradient. theta (d + 1,), A (n, d + 1), y (n,) -> (scalar, (d + 1,))."""
    z = A @ theta  # (n,) logits
    grad = A.T @ (sigmoid(z) - y) / len(y)  # (d + 1, n) @ (n,) -> (d + 1,)
    return log_loss_from_logits(z, y), grad


H = 1e-6  # central-difference step (A4.1)


def numerical_gradient(f: Callable[[FloatArray], float], x: FloatArray, h: float = H) -> FloatArray:
    """Central-difference gradient of a scalar function of an array of any shape. x (...) -> (...)."""
    # from labs/A/a4_2_gradients_jacobians.py
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + h
        f_plus = f(x)
        x[idx] = old - h
        f_minus = f(x)
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * h)
    return grad


def rel_error(a: FloatArray, b: FloatArray) -> float:
    """Relative error |a - b| / (|a| + |b|), safe when both are zero."""
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12))


A_chk = add_intercept(rng.normal(size=(20, 3)))  # (20, 4) small random problem
y_chk = (rng.random(20) < 0.5).astype(np.float64)
th_chk = rng.normal(size=4)
_, g_analytic = loss_and_grad(th_chk, A_chk, y_chk)
g_numeric = numerical_gradient(lambda t: loss_and_grad(t, A_chk, y_chk)[0], th_chk.copy())
print(f"analytic gradient {show(g_analytic)}")
print(f"numeric gradient  {show(g_numeric)}")
err = rel_error(g_analytic, g_numeric)
assert err < 1e-6, err
print(f"relative error {err:.1e} < 1e-6: OK")

# %%
# From scratch: batch gradient descent on standardized breast-cancer features.
section("from scratch: gradient descent on breast cancer")


def fit_logistic_gd(
    A: FloatArray, y: FloatArray, lr: float, max_iters: int = MAX_ITERS, tol: float = GRAD_TOL
) -> tuple[FloatArray, list[float], int]:
    """Batch GD on mean log-loss from theta = 0. A (n, d + 1), y (n,) -> (theta (d + 1,), losses, steps)."""
    theta = np.zeros(A.shape[1])
    losses = []
    for step in range(1, max_iters + 1):
        loss, grad = loss_and_grad(theta, A, y)
        losses.append(loss)
        if np.linalg.norm(grad) < tol:
            return theta, losses, step
        theta -= lr * grad
    return theta, losses, max_iters


mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)  # (d,) training statistics only (B1.4)
A_tr = add_intercept((X_tr - mu) / sd)  # (n, d + 1)
A_va = add_intercept((X_va - mu) / sd)
# The log-loss Hessian is A^T S A / n with S = diag(p(1 - p)) and p(1 - p) <= 1/4, so its largest
# eigenvalue is at most L = lambda_max(A^T A / n) / 4. A fixed step of 1 / L can never overshoot.
L_smooth = float(np.linalg.eigvalsh(A_tr.T @ A_tr / n)[-1] / 4)
theta_gd, losses_gd, steps_gd = fit_logistic_gd(A_tr, y_tr, lr=1.0 / L_smooth)
p_tr_gd, p_va_gd = sigmoid(A_tr @ theta_gd), sigmoid(A_va @ theta_gd)
print(f"smoothness bound L = {L_smooth:.4f}   lr = 1/L = {1 / L_smooth:.4f}")
print(f"converged in {steps_gd} steps   train log-loss: start {losses_gd[0]:.4f} -> {losses_gd[-1]:.4f}")
print(f"intercept b = {theta_gd[0]:.4f}")
print("weights per 1 sd:", dict(zip(FEATURES, np.round(theta_gd[1:], 4).tolist())))
print(f"train: accuracy {np.mean((p_tr_gd >= 0.5) == y_tr):.4f}  "
      f"log-loss {log_loss_from_logits(A_tr @ theta_gd, y_tr):.4f}")
print(f"val:   accuracy {np.mean((p_va_gd >= 0.5) == y_va):.4f}  "
      f"log-loss {log_loss_from_logits(A_va @ theta_gd, y_va):.4f}")
print(f"mean predicted p on train {p_tr_gd.mean():.4f} = malignant share {y_tr.mean():.4f}")
print(f"odds ratio per 1 sd of mean radius: exp({theta_gd[1]:.4f}) = {np.exp(theta_gd[1]):.2f}")

# %%
# With scikit-learn. In 1.9.1 `penalty=None` is deprecated (VERIFY as of 2026-10): no penalty is
# C=np.inf. lbfgs is a quasi-Newton method, so it converges far faster than plain GD.
section("with scikit-learn")
sk = LogisticRegression(C=np.inf, max_iter=10_000, tol=1e-10)
sk.fit((X_tr - mu) / sd, y_tr)
p_va_sk = sk.predict_proba((X_va - mu) / sd)[:, 1]  # column 1 = P(y = 1)
print(f"LogisticRegression(C=inf) intercept_ {sk.intercept_[0]:.4f}   coef_ {show(sk.coef_[0])}")
print(f"max |p_mine - p_sklearn| on val: {np.max(np.abs(p_va_gd - p_va_sk)):.1e}")
# Both stop at a gradient tolerance, not at the exact optimum, so allow 1e-5 rather than 1e-6.
np.testing.assert_allclose(p_va_gd, p_va_sk, atol=1e-5)
print("probabilities match scikit-learn to 3 decimals (and to 1e-5): OK")
print(f"sklearn log_loss on val {log_loss(y_va, p_va_sk):.4f}   accuracy {accuracy_score(y_va, p_va_sk >= 0.5):.4f}")
pipe = make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=10_000, tol=1e-10))
pipe.fit(X_tr, y_tr)
np.testing.assert_allclose(pipe.predict_proba(X_va)[:, 1], p_va_sk, atol=1e-6)
print("Pipeline(StandardScaler, LogisticRegression) gives the same probabilities: OK")
default = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, y_tr)
p_def = default.predict_proba(X_va)[:, 1]
print(f"default C=1.0 (an L2 penalty, B2.2): coef_ {show(default[-1].coef_[0])}")
print(f"default C=1.0: val log-loss {log_loss(y_va, p_def):.4f}   accuracy {accuracy_score(y_va, p_def >= 0.5):.4f}")

# %%
# Perfect separation: with all 30 features a plane splits the training classes exactly, so the
# unpenalized loss has no minimum: any separating w can be doubled to lower the loss further.
# Weaker penalties (larger C) let the weights grow without limit; C=inf would never stop.
section("perfect separation (all 30 features)")
mu30, sd30 = X_full[i_tr].mean(axis=0), X_full[i_tr].std(axis=0)
Z30_tr = (X_full[i_tr] - mu30) / sd30  # (343, 30)
Z30_va = (X_full[i_va] - mu30) / sd30
for C in SEP_CS:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # large C: lbfgs chases a moving target
        m30 = LogisticRegression(C=C, max_iter=SEP_ITERS, tol=1e-10).fit(Z30_tr, y_tr)
    p30_tr, p30_va = m30.predict_proba(Z30_tr)[:, 1], m30.predict_proba(Z30_va)[:, 1]
    print(f"C = {C:>9.0e}: train accuracy {accuracy_score(y_tr, p30_tr >= 0.5):.4f}   "
          f"train log-loss {log_loss(y_tr, p30_tr):.6f}   |w| = {np.linalg.norm(m30.coef_):8.2f}   "
          f"val log-loss {log_loss(y_va, p30_va):.4f}")
print("train loss -> 0 and |w| grows with C: no finite unpenalized minimum; keep a penalty (B2.2)")

# %%
# Key plot: the decision boundary of a two-feature model, with probability contours.
section("key plot")
two = [0, 1]  # mean radius, mean texture
A2 = add_intercept((X_tr[:, two] - mu[two]) / sd[two])
theta2, _, steps2 = fit_logistic_gd(A2, y_tr, lr=1.0 / float(np.linalg.eigvalsh(A2.T @ A2 / n)[-1] / 4))
print(f"two-feature model (mean radius, mean texture): b = {theta2[0]:.4f}  w = {show(theta2[1:])}  "
      f"({steps2} steps)")
A2_va = add_intercept((X_va[:, two] - mu[two]) / sd[two])
print(f"two-feature val accuracy {np.mean((A2_va @ theta2 >= 0) == y_va):.4f}")
g0 = np.linspace(X_tr[:, 0].min() - 1, X_tr[:, 0].max() + 1, 300)
g1 = np.linspace(X_tr[:, 1].min() - 2, X_tr[:, 1].max() + 2, 300)
G0, G1 = np.meshgrid(g0, g1)
grid = np.column_stack([G0.ravel(), G1.ravel()])  # (90000, 2)
P_grid = sigmoid(add_intercept((grid - mu[two]) / sd[two]) @ theta2).reshape(G0.shape)
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
cs = ax.contour(G0, G1, P_grid, levels=[0.1, 0.5, 0.9], colors=["0.5", "k", "0.5"],
                linestyles=["--", "-", "--"])
ax.clabel(cs, fmt={0.1: "p = 0.1", 0.5: "p = 0.5", 0.9: "p = 0.9"}, fontsize=8)
ax.scatter(X_tr[y_tr == 0, 0], X_tr[y_tr == 0, 1], s=12, marker="o", alpha=0.6, label="benign (y = 0)")
ax.scatter(X_tr[y_tr == 1, 0], X_tr[y_tr == 1, 1], s=14, marker="^", alpha=0.6, label="malignant (y = 1)")
ax.set_xlabel("mean radius (image units)")
ax.set_ylabel("mean texture (grey-level sd)")
ax.set_title("Logistic regression draws a straight boundary at p = 0.5")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(IMG / "b2_1_decision_boundary.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved lessons/B/img/b2_1_decision_boundary.png")

# %%
# Try this (not graded):
# 1. Set lr = 4 / L_smooth and then 10 / L_smooth in fit_logistic_gd. When does it start to oscillate?
# 2. Skip standardization (A_tr = add_intercept(X_tr)) and count the steps GD needs (B1.5 again).
# 3. Replace FEATURES with all 30 names and watch sklearn's coef_ grow as you raise C from 1 to 1e6.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
