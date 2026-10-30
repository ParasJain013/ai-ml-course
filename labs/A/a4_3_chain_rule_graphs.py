# %% [markdown]
# # A4.3 The chain rule as a computational graph — lab
# Lesson: lessons/A/a4_3_chain_rule_graphs.md
# Run: uv run python labs/A/a4_3_chain_rule_graphs.py   (CPU, defaults: ~10 s, mostly importing PyTorch)
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
# Nothing here needs a GPU; the knobs size the batch of the vectorized neuron and the cost comparison.
GPU_SCALE = False
N_ROWS = 10_000 if GPU_SCALE else 8  # examples in the batched one-neuron model
N_FEATURES = 20 if GPU_SCALE else 3  # features d
COST_DIMS = (10, 100, 1_000, 10_000) if GPU_SCALE else (10, 100, 1_000)  # parameter counts to time


# from labs/A/a4_2_gradients_jacobians.py
def numerical_gradient(f: Callable[[FloatArray], float], x: FloatArray, h: float = 1e-5) -> FloatArray:
    """Central-difference gradient of a scalar f at x of any shape. Returns the same shape as x."""
    grad = np.zeros_like(x, dtype=np.float64)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + h
        f_plus = f(x)
        x[idx] = old - h
        f_minus = f(x)
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2.0 * h)
    return grad


# from labs/A/a4_2_gradients_jacobians.py
def rel_error(a: FloatArray, b: FloatArray) -> float:
    """Largest relative error max |a - b| / max(|a|, |b|, 1e-12), elementwise."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)))


def sigmoid(z: FloatArray | float) -> FloatArray | float:
    """Logistic sigmoid 1 / (1 + e^-z), elementwise."""
    return 1.0 / (1.0 + np.exp(-z))


# %%
# Data: the inputs of the three worked examples and a small batch for the vectorized neuron.
section("data")
X = rng.normal(size=(N_ROWS, N_FEATURES))  # (n, d)
y = (rng.uniform(size=N_ROWS) < 0.5).astype(np.float64)  # (n,) targets 0 or 1
print("graph 1: f = (x*y + z)^2 at x = 3, y = -2, z = 4")
print("graph 2: f = (x + y) * x at x = 3, y = 1   (x is used twice)")
print("graph 3: one neuron, loss = (sigmoid(w . x + b) - y)^2 at x = (1, 2), w = (0.5, -0.25), b = 0, y = 1")
print("batch for the vectorized neuron: X", X.shape, " y", y.shape)

# %%
# Worked example by hand, graph 1: forward, then backward node by node.
section("worked example: f = (x*y + z)^2")


def graph1_forward(x: float, y_: float, z: float) -> dict[str, float]:
    """Forward pass; returns every intermediate value (the cache the backward pass needs)."""
    a = x * y_  # multiply node
    b = a + z  # add node
    f = b**2  # square node
    return {"x": x, "y": y_, "z": z, "a": a, "b": b, "f": f}


def graph1_backward(c: dict[str, float]) -> dict[str, float]:
    """Backward pass: upstream gradient times local derivative, from the output back to the inputs."""
    df_df = 1.0  # seed: df/df = 1
    df_db = df_df * 2 * c["b"]  # square: local derivative 2b
    df_da = df_db * 1.0  # add: local derivative 1 for each input
    df_dz = df_db * 1.0
    df_dx = df_da * c["y"]  # multiply: local derivative is the OTHER input
    df_dy = df_da * c["x"]
    return {"f": df_df, "b": df_db, "a": df_da, "z": df_dz, "x": df_dx, "y": df_dy}


cache1 = graph1_forward(3.0, -2.0, 4.0)
grads1 = graph1_backward(cache1)
print(f"{'node':<5}{'value':>8}{'df/dnode':>10}")
for node in ("x", "y", "z", "a", "b", "f"):
    print(f"{node:<5}{cache1[node]:>8.1f}{grads1[node]:>10.1f}")
f1 = lambda v: (v[0] * v[1] + v[2]) ** 2  # noqa: E731
num1 = numerical_gradient(f1, np.array([3.0, -2.0, 4.0]))
print("numerical gradient (x, y, z) =", show(num1), f"  rel error {rel_error([grads1['x'], grads1['y'], grads1['z']], num1):.1e}")

# %%
# Worked example by hand, graph 2: a variable used twice. Gradients from both paths ADD.
section("worked example: fan-out")
x, y2 = 3.0, 1.0
q = x + y2  # add node
f2 = q * x  # multiply node; x feeds both the add and the multiply
df_dq = x  # multiply: the other input
df_dx_direct = q  # multiply: the other input (path x -> f)
df_dx_via_q = df_dq * 1.0  # add: 1 (path x -> q -> f)
df_dx = df_dx_direct + df_dx_via_q
df_dy = df_dq * 1.0
print(f"q = {q}  f = {f2}")
print(f"df/dx = direct path {df_dx_direct} + path through q {df_dx_via_q} = {df_dx}   df/dy = {df_dy}")
print("check with f = x^2 + x y: df/dx = 2x + y =", 2 * x + y2, "  df/dy = x =", x)
print("numerical:", show(numerical_gradient(lambda v: (v[0] + v[1]) * v[0], np.array([3.0, 1.0]))))

# %%
# Worked example by hand, graph 3: one neuron with a sigmoid and a squared-error loss.
section("worked example: one neuron")
xv = np.array([1.0, 2.0])  # (d,) one example
wv = np.array([0.5, -0.25])  # (d,)
bv, yv = 0.0, 1.0
z = wv @ xv + bv
p = sigmoid(z)
loss = (p - yv) ** 2
dl_dp = 2 * (p - yv)
dp_dz = p * (1 - p)  # sigmoid'(z) = sigmoid(z) (1 - sigmoid(z))
dl_dz = dl_dp * dp_dz
dl_dw = dl_dz * xv  # z = w . x + b: dz/dw_j = x_j
dl_db = dl_dz * 1.0
print(f"forward:  z = {z:.4f}  p = sigmoid(z) = {p:.4f}  loss = {loss:.4f}")
print(f"backward: dl/dp = {dl_dp:.4f}  dp/dz = {dp_dz:.4f}  dl/dz = {dl_dz:.4f}  dl/dw = {show(dl_dw)}  dl/db = {dl_db:.4f}")
theta = np.array([*wv, bv])  # pack (w1, w2, b) to check all three at once
neuron = lambda t: float((sigmoid(t[:2] @ xv + t[2]) - yv) ** 2)  # noqa: E731
num3 = numerical_gradient(neuron, theta.copy())
print("numerical (dw1, dw2, db) =", show(num3), f"  rel error {rel_error(np.array([*dl_dw, dl_db]), num3):.1e}")
zs = np.array([-3.0, 0.0, 2.0])
print("sigmoid'(z) = s(1 - s) at z = [-3, 0, 2]:", show(sigmoid(zs) * (1 - sigmoid(zs))),
      "  numerical:", show(np.array([(sigmoid(t + 1e-5) - sigmoid(t - 1e-5)) / 2e-5 for t in zs])))

# %%
# From scratch: the same neuron on a batch, with shapes. Backward = vector-Jacobian products.
section("from scratch: a batch of examples")


def neuron_forward(X_: FloatArray, w: FloatArray, b: float, y_: FloatArray) -> tuple[float, dict[str, FloatArray]]:
    """Mean squared error of sigmoid(X w + b). X (n, d), w (d,), y (n,) -> (loss, cache)."""
    z_ = X_ @ w + b  # (n,)
    p_ = sigmoid(z_)  # (n,)
    loss_ = float(np.mean((p_ - y_) ** 2))
    return loss_, {"X": X_, "p": p_, "y": y_}


def neuron_backward(cache: dict[str, FloatArray]) -> tuple[FloatArray, float]:
    """Gradients of the mean loss. Returns dw (d,) and db (scalar)."""
    X_, p_, y_ = cache["X"], cache["p"], cache["y"]
    n = len(y_)
    dp = 2.0 * (p_ - y_) / n  # (n,): d mean((p - y)^2) / dp_i
    dz = dp * p_ * (1.0 - p_)  # (n,): elementwise sigmoid, so its Jacobian is diagonal
    dw = X_.T @ dz  # (d,) = J^T dz, with J = X the Jacobian of z = X w with respect to w
    db = float(dz.sum())  # b is broadcast to every example, so its gradient sums
    return dw, db


w = rng.normal(size=N_FEATURES)  # (d,)
b = 0.1
loss_b, cache_b = neuron_forward(X, w, b, y)
dw, db = neuron_backward(cache_b)
assert dw.shape == w.shape, dw.shape
print(f"loss = {loss_b:.6f}   dw {dw.shape} = {show(dw)}   db = {db:.4f}")
dw_num = numerical_gradient(lambda w_: neuron_forward(X, w_, b, y)[0], w.copy())
db_num = numerical_gradient(lambda b_: neuron_forward(X, w, float(b_[0]), y)[0], np.array([b]))
print(f"check against numerical gradient: dw rel error {rel_error(dw, dw_num):.1e}   db rel error {rel_error(db, db_num[0]):.1e}")

# %%
# From scratch: why reverse mode. Count the loss evaluations each method needs.
section("from scratch: cost of a gradient")
for d in COST_DIMS:
    Xd = rng.normal(size=(64, d))
    yd = (rng.uniform(size=64) < 0.5).astype(np.float64)
    wd = rng.normal(size=d) / np.sqrt(d)
    t0 = time.perf_counter()
    dw_rev, _ = neuron_backward(neuron_forward(Xd, wd, 0.0, yd)[1])
    t_rev = time.perf_counter() - t0
    t0 = time.perf_counter()
    dw_fd = numerical_gradient(lambda w_: neuron_forward(Xd, w_, 0.0, yd)[0], wd.copy())
    t_fd = time.perf_counter() - t0
    print(f"d = {d:>5}: finite differences need {2 * d:>5} forward passes, reverse mode 1 forward + 1 backward;"
          f"  agree to {rel_error(dw_rev, dw_fd):.0e}")
    print(f"~ d = {d:>5}: finite differences {t_fd * 1e3:8.2f} ms   reverse mode {t_rev * 1e3:6.3f} ms")

# %%
# With PyTorch: autograd builds the same graph during the forward pass and walks it backwards.
section("with pytorch")
import torch  # noqa: E402

xt = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
yt = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
zt = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
ft = (xt * yt + zt) ** 2
print("graph recorded by autograd:", ft.grad_fn.name(), "<-", ft.grad_fn.next_functions[0][0].name(),
      "<-", ft.grad_fn.next_functions[0][0].next_functions[0][0].name())
ft.backward()
print(f"f = {ft.item():.1f}   df/dx = {xt.grad.item():.1f}   df/dy = {yt.grad.item():.1f}   df/dz = {zt.grad.item():.1f}")
assert (xt.grad.item(), yt.grad.item(), zt.grad.item()) == (grads1["x"], grads1["y"], grads1["z"])
print("matches the hand backward pass: OK")

xt2 = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
yt2 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
((xt2 + yt2) * xt2).backward()
print(f"fan-out: df/dx = {xt2.grad.item():.1f}  df/dy = {yt2.grad.item():.1f}   (autograd adds the two paths too)")

Xt, ytt = torch.tensor(X), torch.tensor(y)
wt = torch.tensor(w, requires_grad=True)
bt = torch.tensor(b, dtype=torch.float64, requires_grad=True)
torch.mean((torch.sigmoid(Xt @ wt + bt) - ytt) ** 2).backward()
np.testing.assert_allclose(wt.grad.numpy(), dw, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(bt.grad.item(), db, rtol=1e-6, atol=1e-8)
print("batched neuron: torch gradients match neuron_backward: OK")

# %%
# Key plot: the computational graph of f = (x*y + z)^2 with forward values and backward gradients.
section("key plot")
pos = {"x": (0.0, 2.6), "y": (0.0, 1.3), "z": (0.0, 0.0), "a": (1.9, 1.95), "b": (3.6, 0.95), "f": (5.5, 0.95)}
labels = {"a": "a = x·y", "b": "b = a + z", "f": "f = b²", "x": "x", "y": "y", "z": "z"}
local = {("x", "a"): ("∂a/∂x = y = −2", 0.2), ("y", "a"): ("∂a/∂y = x = 3", -0.22), ("a", "b"): ("∂b/∂a = 1", 0.2),
         ("z", "b"): ("∂b/∂z = 1", -0.2), ("b", "f"): ("∂f/∂b = 2b = −4", 0.15)}
R_NODE = 0.4
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=120)
for (u, v), (text, dy) in local.items():
    (x0, y0), (x1, y1) = pos[u], pos[v]
    vec = np.array([x1 - x0, y1 - y0]) / np.hypot(x1 - x0, y1 - y0)  # unit vector along the edge
    start, end = np.array([x0, y0]) + R_NODE * vec, np.array([x1, y1]) - R_NODE * vec
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="0.4", lw=1.3))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="center", fontsize=7.5, color="0.25")
for node, (px, py) in pos.items():
    ax.add_patch(plt.Circle((px, py), R_NODE, facecolor="white", edgecolor="0.2", lw=1.3))
    ax.text(px, py, labels[node], ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(px, py - 0.53, f"value {cache1[node]:g}", ha="center", fontsize=8, color="tab:blue")
    ax.text(px, py - 0.7, f"∂f/∂{node} = {grads1[node]:g}", ha="center", fontsize=8, color="tab:red")
ax.text(2.7, 3.55, "forward pass: values flow left to right (blue)", ha="center", fontsize=9, color="tab:blue")
ax.text(2.7, 3.32, "backward pass: ∂f/∂node = (∂f/∂child) × (local derivative on the edge), right to left (red)",
        ha="center", fontsize=9, color="tab:red")
ax.set_xlim(-0.6, 6.1)
ax.set_ylim(-0.95, 3.7)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Backpropagation multiplies local derivatives along each path back from the output")
fig.tight_layout()
path = IMG / "a4_3_graph_forward_backward.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Replace the squared-error loss of the neuron by -(y log p + (1 - y) log(1 - p)). Derive dl/dz by
#    hand and compare with numerical_gradient. (You will meet this loss, cross-entropy, in A6.4.)
# 2. Set z = 20 in the single neuron. What is dp/dz? What does that mean for learning?
# 3. In graph 2, forget to add the two paths (use only the direct one). Which check catches it?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
