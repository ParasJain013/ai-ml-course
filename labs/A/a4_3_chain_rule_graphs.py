# %% [markdown]
# # A4.3 The chain rule as a computational graph — lab
# Lesson: lessons/A/a4_3_chain_rule_graphs.md
# Run: uv run python labs/A/a4_3_chain_rule_graphs.py   (CPU, defaults: ~3 s)
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
from scipy import optimize

FloatArray = npt.NDArray[np.float64]
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
OUT = ROOT / "outputs" / "a4_3"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "A" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; the knobs size the batch used to compare backprop with finite differences.
GPU_SCALE = False
N_ROWS = 5_000 if GPU_SCALE else 64  # examples in the batched neuron
N_FEATURES = 5  # features for the batched gradient check
D_COST = 20_000 if GPU_SCALE else 1_000  # features in the cost comparison
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


def numerical_jacobian(f: Callable[[FloatArray], FloatArray], x: FloatArray, h: float = H) -> FloatArray:
    """Central-difference Jacobian of f: R^n -> R^m. x (n,) -> (m, n)."""
    # from labs/A/a4_2_gradients_jacobians.py
    cols = []
    for j in range(x.size):
        e_j = np.zeros_like(x)
        e_j[j] = h
        cols.append((f(x + e_j) - f(x - e_j)) / (2 * h))
    return np.stack(cols, axis=1)


def rel_error(a: FloatArray, b: FloatArray) -> float:
    """Relative error |a - b| / (|a| + |b|), safe when both are zero."""
    # from labs/A/a4_2_gradients_jacobians.py
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12))


# %%
# Data: the inputs of the worked examples, and a small random classification batch.
section("data")
XYZ = np.array([2.0, 3.0, -4.0])  # (x, y, z) for f = (x y + z)^2
X_NEURON = np.array([1.0, 2.0])  # (2,) one example
W_NEURON = np.array([0.5, -0.25])  # (2,) weights
B_NEURON, Y_NEURON = 0.0, 1.0
X = rng.normal(size=(N_ROWS, N_FEATURES))  # (n, d)
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float64)  # (n,) labels 0/1
print("graph inputs (x, y, z) =", show(XYZ), "  neuron x =", show(X_NEURON), " w =", show(W_NEURON))
print("batch X", X.shape, " y", y.shape, " share of 1s =", round(float(y.mean()), 4))

# %%
# Worked example by hand, part 1: f = (x y + z)^2, forward then backward.
section("worked example: f = (x y + z)^2")


def graph_forward(x: float, y: float, z: float) -> tuple[float, dict[str, float]]:
    """Forward pass; returns f and the cache of every intermediate value the backward pass needs."""
    a = x * y  # multiply node
    s = a + z  # add node
    f = s * s  # square node
    return f, {"x": x, "y": y, "z": z, "a": a, "s": s}


def graph_backward(cache: dict[str, float]) -> dict[str, float]:
    """Backward pass: start from df/df = 1 and multiply by each node's local derivative."""
    df_ds = 2 * cache["s"]  # square: d(s^2)/ds = 2s
    df_da = df_ds * 1.0  # add: ds/da = 1
    df_dz = df_ds * 1.0  # add: ds/dz = 1
    df_dx = df_da * cache["y"]  # multiply: da/dx = y
    df_dy = df_da * cache["x"]  # multiply: da/dy = x
    return {"s": df_ds, "a": df_da, "x": df_dx, "y": df_dy, "z": df_dz}


f_val, cache = graph_forward(*XYZ)
grads = graph_backward(cache)
print(f"forward:  a = x*y = {cache['a']}   s = a + z = {cache['s']}   f = s^2 = {f_val}")
print("backward: " + "   ".join(f"df/d{k} = {v}" for k, v in grads.items()))
expanded = np.array([2 * (XYZ[0] * XYZ[1] + XYZ[2]) * XYZ[1], 2 * (XYZ[0] * XYZ[1] + XYZ[2]) * XYZ[0],
                     2 * (XYZ[0] * XYZ[1] + XYZ[2])])
print("check by expanding the formula: (df/dx, df/dy, df/dz) =", show(expanded))

# %%
# Worked example by hand, part 2: a variable used twice. Gradients from both paths add.
section("worked example: fan-out g = x * (x + y)")
x_f, y_f = 3.0, 1.0
a_f = x_f + y_f  # add node
g_f = x_f * a_f  # multiply node; x feeds both nodes
dg_da = x_f  # multiply: dg/da = x
dg_dx = a_f + dg_da * 1.0  # direct path (dg/dx holding a fixed = a) + path through a
dg_dy = dg_da * 1.0
print(f"forward: a = {a_f}  g = {g_f}   backward: dg/dx = {a_f} (direct) + {dg_da} (via a) = {dg_dx}   dg/dy = {dg_dy}")
print(f"check: g = x^2 + x y, dg/dx = 2x + y = {2 * x_f + y_f}   dg/dy = x = {x_f}")

# %%
# Worked example by hand, part 3: one neuron, sigmoid, squared error.
section("worked example: one neuron")


def sigmoid(z: FloatArray | float) -> FloatArray | float:
    """Logistic sigmoid 1 / (1 + e^-z), elementwise."""
    return 1.0 / (1.0 + np.exp(-z))


z_n = float(X_NEURON @ W_NEURON + B_NEURON)
p_n = float(sigmoid(z_n))
L_n = (p_n - Y_NEURON) ** 2
dL_dp = 2 * (p_n - Y_NEURON)  # square node
dp_dz = p_n * (1 - p_n)  # sigmoid node: sigma' = sigma (1 - sigma)
dL_dz = dL_dp * dp_dz
dL_dw = dL_dz * X_NEURON  # dz/dw_j = x_j
dL_db = dL_dz  # dz/db = 1
print(f"forward:  z = {z_n}   p = sigmoid(z) = {p_n}   L = (p - y)^2 = {L_n}")
print(f"backward: dL/dp = {dL_dp}   dp/dz = {dp_dz}   dL/dz = {dL_dz}")
print(f"          dL/dw = {show(dL_dw)}   dL/db = {dL_db}")


def neuron_loss(params: FloatArray) -> float:
    """Loss of the single neuron with params = (w1, w2, b) packed into one array. (3,) -> scalar."""
    p = sigmoid(X_NEURON @ params[:2] + params[2])
    return float((p - Y_NEURON) ** 2)


num = numerical_gradient(neuron_loss, np.array([*W_NEURON, B_NEURON]))
print("numerical (dL/dw1, dL/dw2, dL/db) =", show(num))

# %%
# From scratch: forward/backward with a cache, checked against finite differences at random points.
section("from scratch: graph checks at random points")
for trial in range(3):
    pt = rng.normal(size=3)
    _, c = graph_forward(*pt)
    g = graph_backward(c)
    analytic = np.array([g["x"], g["y"], g["z"]])
    numeric = numerical_gradient(lambda v: graph_forward(*v)[0], pt)
    print(f"point {trial}: rel error < 1e-8: {rel_error(analytic, numeric) < 1e-8}")

# %%
# From scratch: the neuron over a batch. Mean squared error over n examples.
section("from scratch: batched neuron")


def neuron_forward(w: FloatArray, b: float, X: FloatArray, y: FloatArray) -> tuple[float, dict[str, FloatArray]]:
    """Forward pass. X (n, d), w (d,), y (n,) -> (loss, cache)."""
    z = X @ w + b  # (n,)
    p = sigmoid(z)  # (n,)
    loss = float(np.mean((p - y) ** 2))
    return loss, {"X": X, "y": y, "p": p}


def neuron_backward(cache: dict[str, FloatArray]) -> tuple[FloatArray, float]:
    """Backward pass. -> (dL/dw (d,), dL/db scalar)."""
    X, y, p = cache["X"], cache["y"], cache["p"]
    n = len(y)
    dL_dp = 2.0 / n * (p - y)  # (n,): mean of squares
    dL_dz = dL_dp * p * (1 - p)  # (n,): elementwise, the diagonal Jacobian of sigmoid (A4.2)
    dL_dw = X.T @ dL_dz  # (d,): z = X w, pattern X^T G (A4.2)
    dL_db = float(dL_dz.sum())  # z = ... + b adds b to every row, so its gradient sums over rows
    return dL_dw, dL_db


w0 = rng.normal(size=N_FEATURES)
b0 = 0.1
loss0, cache0 = neuron_forward(w0, b0, X, y)
dw, db = neuron_backward(cache0)
assert dw.shape == w0.shape, dw.shape
num_w = numerical_gradient(lambda v: neuron_forward(v, b0, X, y)[0], w0.copy())
num_b = (neuron_forward(w0, b0 + H, X, y)[0] - neuron_forward(w0, b0 - H, X, y)[0]) / (2 * H)
print(f"loss = {loss0:.4f}   dL/dw shape {dw.shape}   rel error < 1e-8: {rel_error(dw, num_w) < 1e-8}"
      f"   dL/db = {db:.6f} vs numerical {num_b:.6f}")

# %%
# From scratch: the chain rule for vector functions is a product of Jacobians.
section("from scratch: Jacobians multiply")
A = rng.normal(size=(3, 4))  # (m, n): inner map u = A x
x_j = rng.normal(size=4)  # (n,)
J_inner = A  # Jacobian of x -> A x is A (A4.2), shape (3, 4)
J_outer = np.diag(1 - np.tanh(A @ x_j) ** 2)  # Jacobian of elementwise tanh at u = A x, shape (3, 3)
J_chain = J_outer @ J_inner  # (3, 3) @ (3, 4) -> (3, 4)
J_num = numerical_jacobian(lambda v: np.tanh(A @ v), x_j)
print("J_tanh(Ax) = J_tanh @ J_A:", np.allclose(J_chain, J_num, atol=1e-8), "  shape", J_chain.shape)
v = rng.normal(size=3)  # an upstream gradient, one entry per output
vjp_full = v @ J_chain  # builds the (3, 4) Jacobian first
vjp_cheap = (v * (1 - np.tanh(A @ x_j) ** 2)) @ A  # never builds a Jacobian: elementwise, then @ A
print("v^T J without forming J:", np.allclose(vjp_full, vjp_cheap), "  shape", vjp_cheap.shape)

# %%
# From scratch: why backward beats finite differences. Count forward passes for d parameters.
section("from scratch: cost of a gradient")
Xc = rng.normal(size=(N_ROWS, D_COST))  # (n, D_COST)
wc = rng.normal(size=D_COST) / np.sqrt(D_COST)
t = time.perf_counter()
_, cc = neuron_forward(wc, 0.0, Xc, y)
neuron_backward(cc)
t_back = time.perf_counter() - t
t = time.perf_counter()
numerical_gradient(lambda v_: neuron_forward(v_, 0.0, Xc, y)[0], wc.copy())
t_num = time.perf_counter() - t
print(f"d = {D_COST}: finite differences need {2 * D_COST} forward passes; backprop needs 1 forward + 1 backward")
print(f"~ time: backprop {t_back * 1e3:.2f} ms   finite differences {t_num * 1e3:.0f} ms")

# %%
# With SciPy: check_grad on the batched neuron (the "library" for checking; autograd comes in C1.3).
section("with scipy")
params0 = np.array([*w0, b0])


def batch_loss(params: FloatArray) -> float:
    """Batched neuron loss with params = (w, b). (d + 1,) -> scalar."""
    return neuron_forward(params[:-1], float(params[-1]), X, y)[0]


def batch_grad(params: FloatArray) -> FloatArray:
    """Backprop gradient packed the same way. (d + 1,) -> (d + 1,)."""
    _, c = neuron_forward(params[:-1], float(params[-1]), X, y)
    gw, gb = neuron_backward(c)
    return np.append(gw, gb)


print(f"scipy.optimize.check_grad on the batched neuron < 1e-6: {optimize.check_grad(batch_loss, batch_grad, params0) < 1e-6}")
np.testing.assert_allclose(optimize.approx_fprime(params0, batch_loss), batch_grad(params0), rtol=1e-4, atol=1e-7)
# rtol 1e-4: approx_fprime uses forward differences with step 1.5e-8 (A4.1)
print("scipy.optimize.approx_fprime matches backprop (rtol 1e-4): OK")

# %%
# Key plot: the graph of f = (x y + z)^2 with forward values and backward gradients.
section("key plot")
fig, ax = plt.subplots(figsize=(8, 3.6), dpi=120)
nodes = {  # name: (x, y, label, value, gradient)
    "x": (0.0, 2.0, "x", cache["x"], grads["x"]),
    "y": (0.0, 1.0, "y", cache["y"], grads["y"]),
    "z": (0.0, 0.0, "z", cache["z"], grads["z"]),
    "a": (2.0, 1.5, "a = x·y", cache["a"], grads["a"]),
    "s": (4.0, 0.9, "s = a + z", cache["s"], grads["s"]),
    "f": (6.0, 0.9, "f = s²", f_val, 1.0),
}
edges = [("x", "a", "∂a/∂x = y = 3", 0.15), ("y", "a", "∂a/∂y = x = 2", -0.3), ("a", "s", "∂s/∂a = 1", 0.12),
         ("z", "s", "∂s/∂z = 1", 0.12), ("s", "f", "∂f/∂s = 2s = 4", 0.12)]  # last item: label offset
for src, dst, local, dy in edges:
    (x1, y1), (x2, y2) = nodes[src][:2], nodes[dst][:2]
    ax.annotate("", xy=(x2 - 0.55, y2), xytext=(x1 + 0.45, y1),
                arrowprops=dict(arrowstyle="->", color="0.4", lw=1.2))
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, local, fontsize=7, ha="center", color="0.25")
for name, (nx, ny, label, val, grad) in nodes.items():
    ax.text(nx, ny, label, ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.3"))
    ax.text(nx, ny - 0.33, f"value {val:g}", ha="center", fontsize=8, color="tab:blue")
    ax.text(nx, ny - 0.53, f"∂f/∂{name} = {grad:g}", ha="center", fontsize=8, color="tab:red",
            style="italic")
ax.text(3.0, 2.45, "forward: values left → right (blue)    backward: gradients right → left (red)",
        ha="center", fontsize=8)
ax.set_xlim(-0.8, 6.8)
ax.set_ylim(-0.8, 2.7)
ax.axis("off")
ax.set_title("Backprop multiplies local derivatives from the output back to every input", fontsize=10)
fig.tight_layout()
path = IMG / "a4_3_graph_forward_backward.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Replace the squared error in the neuron with log loss -[y log p + (1 - y) log(1 - p)].
#    Print dL/dz for one example and compare it with p - y.
# 2. Set z_n to 10 (w = (10, 0)). How big is dp/dz now, and what does that do to dL/dw?
# 3. Change graph_backward to forget the "+ 1.0 *" path for z. Which check fails?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
