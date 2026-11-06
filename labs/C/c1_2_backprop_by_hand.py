# %% [markdown]
# # C1.2 Backpropagation by hand on a tiny network — lab
# Lesson: lessons/C/c1_2_backprop_by_hand.md
# Run: uv run python labs/C/c1_2_backprop_by_hand.py   (CPU, defaults: ~5 s)
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
import torch

FloatArray = npt.NDArray[np.float64]
SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
OUT = ROOT / "outputs" / "c1_2"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "C" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; the network has 12 parameters.
GPU_SCALE = False
LR_STEP = 0.5  # learning rate of the single hand-computed step
LR_TRAIN = 5.0  # learning rate when training the same network on XOR
TRAIN_STEPS = 3000

# %%
# Data and starting weights. One example for the hand calculation, then a batch of 4 (XOR, with
# one-hot targets: column 0 = "XOR is 1", column 1 = "XOR is 0").
# Shapes follow the course convention: W (d_in, d_out), one example per row.
section("data")
x = np.array([[0.5, 1.0]])  # (n=1, 2) one example, written as a row
y = np.array([[1.0, 0.0]])  # (1, 2) its target
W1 = np.array([[0.1, 0.3], [-0.2, 0.4]])  # (2, 2) input -> hidden; column j = hidden unit j
b1 = np.array([0.1, -0.1])  # (2,)
W2 = np.array([[0.5, -0.4], [0.3, 0.2]])  # (2, 2) hidden -> output
b2 = np.array([0.0, 0.1])  # (2,)
X_batch = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])  # (4, 2)
Y_batch = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # (4, 2)
print(f"x {x.shape} = {show(x)}   y {y.shape} = {show(y)}")
print(f"W1 {W1.shape} = {show(W1)}   b1 = {show(b1)}")
print(f"W2 {W2.shape} = {show(W2)}   b2 = {show(b2)}")
print(f"batch X {X_batch.shape}  Y {Y_batch.shape}")

# %%
# Worked example by hand: forward pass, loss, every backward step, printed in the lesson's order.
section("worked example: forward pass")


def sigmoid(z: FloatArray) -> FloatArray:
    """Logistic sigmoid, elementwise, stable for large |z|."""
    ez = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + ez), ez / (1.0 + ez))


z1 = x @ W1 + b1  # (1, 2)
h = sigmoid(z1)  # (1, 2)
z2 = h @ W2 + b2  # (1, 2)
a2 = sigmoid(z2)  # (1, 2)
loss = 0.5 * float(np.sum((a2 - y) ** 2))
print(f"z1 = x W1 + b1   = {show(z1)}")
print(f"h  = sigmoid(z1) = {show(h)}")
print(f"z2 = h W2 + b2   = {show(z2)}")
print(f"a2 = sigmoid(z2) = {show(a2)}")
print(f"L  = 1/2 sum (a2 - y)^2 = {loss:.6f}")

section("worked example: backward pass")
dA2 = a2 - y  # (1, 2) dL/da2
dZ2 = dA2 * a2 * (1 - a2)  # (1, 2) dL/dz2 = delta2
dW2 = h.T @ dZ2  # (2, 1) @ (1, 2) -> (2, 2)
db2 = dZ2.sum(axis=0)  # (2,)
dH = dZ2 @ W2.T  # (1, 2) @ (2, 2) -> (1, 2)
dZ1 = dH * h * (1 - h)  # (1, 2) delta1
dW1 = x.T @ dZ1  # (2, 1) @ (1, 2) -> (2, 2)
db1 = dZ1.sum(axis=0)  # (2,)
dX = dZ1 @ W1.T  # (1, 2)
print(f"dL/da2 = a2 - y              = {show(dA2)}")
print(f"a2 (1 - a2)                  = {show(a2 * (1 - a2))}")
print(f"delta2 = dL/dz2              = {show(dZ2)}")
print(f"dL/dW2 = h^T delta2          = {show(dW2)}")
print(f"dL/db2 = delta2              = {show(db2)}")
print(f"dL/dh  = delta2 W2^T         = {show(dH)}")
print(f"h (1 - h)                    = {show(h * (1 - h))}")
print(f"delta1 = dL/dz1              = {show(dZ1)}")
print(f"dL/dW1 = x^T delta1          = {show(dW1)}")
print(f"dL/db1 = delta1              = {show(db1)}")
print(f"dL/dx  = delta1 W1^T         = {show(dX)}")

# %%
# Check every gradient against central differences (the A4.2 checker), then take one step.
section("gradient check and one gradient-descent step")


def numerical_gradient(f: Callable[[FloatArray], float], x: FloatArray, h: float = 1e-6) -> FloatArray:
    """Central-difference gradient of a scalar function of an array of any shape. x (...) -> (...)."""
    # from labs/A/a4_2_gradients_jacobians.py
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + h
        f_plus = f(x)
        x[idx] = old - h
        f_minus = f(x)
        x[idx] = old  # restore: the caller's array must come back unchanged
        grad[idx] = (f_plus - f_minus) / (2 * h)
    return grad


def rel_error(a: FloatArray, b: FloatArray) -> float:
    """Relative error |a - b| / (|a| + |b|), safe when both are zero."""
    # from labs/A/a4_2_gradients_jacobians.py
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12))


def forward_loss(X: FloatArray, Y: FloatArray, W1: FloatArray, b1: FloatArray, W2: FloatArray, b2: FloatArray) -> float:
    """Mean over examples of 1/2 ||a2 - y||^2 for the 2-layer sigmoid net. X (n, 2), Y (n, 2) -> scalar."""
    A2 = sigmoid(sigmoid(X @ W1 + b1) @ W2 + b2)  # (n, 2)
    return 0.5 * float(np.sum((A2 - Y) ** 2)) / len(X)


params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
analytic = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
for name, p in params.items():
    num = numerical_gradient(lambda _: forward_loss(x, y, W1, b1, W2, b2), p)  # p is perturbed in place
    err = rel_error(analytic[name], num)
    print(f"dL/d{name:<3} numeric {show(num)}  rel error {'< 1e-8' if err < 1e-8 else f'{err:.1e}'}  {'PASS' if err < 1e-6 else 'FAIL'}")
num_x = numerical_gradient(lambda v: forward_loss(v, y, W1, b1, W2, b2), x.copy())
print(f"dL/dx   numeric {show(num_x)}  rel error {'< 1e-8' if rel_error(dX, num_x) < 1e-8 else 'FAIL'}")

new = {k: params[k] - LR_STEP * analytic[k] for k in params}
loss_after = forward_loss(x, y, new["W1"], new["b1"], new["W2"], new["b2"])
print(f"one step with lr = {LR_STEP}: W2 becomes {show(new['W2'])}")
print(f"loss {loss:.6f} -> {loss_after:.6f}  (fell by {loss - loss_after:.6f})")

# %%
# From scratch: the same backward pass for a batch, in matrix form, checked three ways.
section("from scratch: batch backward pass in matrix form")


def forward_backward(
    X: FloatArray, Y: FloatArray, W1: FloatArray, b1: FloatArray, W2: FloatArray, b2: FloatArray
) -> tuple[float, dict[str, FloatArray]]:
    """Mean loss and all gradients of the 2-layer sigmoid net. X (n, d), Y (n, k) -> (loss, grads)."""
    n = X.shape[0]
    Z1 = X @ W1 + b1  # (n, h)
    H = sigmoid(Z1)  # (n, h)
    Z2 = H @ W2 + b2  # (n, k)
    A2 = sigmoid(Z2)  # (n, k)
    loss = 0.5 * float(np.sum((A2 - Y) ** 2)) / n
    D2 = (A2 - Y) * A2 * (1 - A2) / n  # (n, k) delta2, with the 1/n of the mean
    D1 = (D2 @ W2.T) * H * (1 - H)  # (n, k) @ (k, h) -> (n, h) delta1
    grads = {
        "W2": H.T @ D2,  # (h, n) @ (n, k) -> (h, k): sums over the batch
        "b2": D2.sum(axis=0),  # (k,)
        "W1": X.T @ D1,  # (d, n) @ (n, h) -> (d, h)
        "b1": D1.sum(axis=0),  # (h,)
        "X": D1 @ W1.T,  # (n, h) @ (h, d) -> (n, d)
    }
    return loss, grads


loss_b, g_b = forward_backward(X_batch, Y_batch, W1, b1, W2, b2)
for name in ["W1", "b1", "W2", "b2", "X"]:
    print(f"d{name:<2} shape {str(g_b[name].shape):<7} = {show(g_b[name])}")

# Check 1: the matrix form equals the average of the one-example gradients (a Python loop).
per_example = [forward_backward(X_batch[i : i + 1], Y_batch[i : i + 1], W1, b1, W2, b2)[1] for i in range(4)]
for name in ["W1", "b1", "W2", "b2"]:
    looped = sum(g[name] for g in per_example) / 4
    np.testing.assert_allclose(g_b[name], looped, rtol=1e-12, atol=1e-15)
print("matrix form = mean of per-example gradients: OK")
# Check 2: numerical gradients.
worst = max(
    rel_error(g_b[name], numerical_gradient(lambda _: forward_loss(X_batch, Y_batch, W1, b1, W2, b2), p))
    for name, p in params.items()
)
print(f"numerical check on the batch: worst rel error {'< 1e-8' if worst < 1e-8 else f'{worst:.1e}'}  {'PASS' if worst < 1e-6 else 'FAIL'}")

# %%
# From scratch: train the same network on XOR with these gradients.
section("from scratch: training the 2-2-2 network on XOR")
P = {k: v.copy() for k, v in params.items()}
losses = []
for t in range(TRAIN_STEPS + 1):
    l, g = forward_backward(X_batch, Y_batch, P["W1"], P["b1"], P["W2"], P["b2"])
    losses.append(l)
    if t in (0, 100, 500, 1000, 2000, TRAIN_STEPS):
        print(f"step {t:>4}: loss {l:.4f}")
    for k in P:
        P[k] -= LR_TRAIN * g[k]
A_final = sigmoid(sigmoid(X_batch @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"])
print(f"P(XOR = 1) for (0,0) (0,1) (1,0) (1,1): {show(A_final[:, 0])}")
print(f"learned W1 = {show(P['W1'])}  b1 = {show(P['b1'])}")

fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
ax.plot(losses)
ax.set(title="XOR: a long plateau, then the hidden units specialize", xlabel="gradient-descent step", ylabel="mean loss")
fig.tight_layout()
fig.savefig(OUT / "c1_2_xor_loss.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"saved {(OUT / 'c1_2_xor_loss.png').relative_to(ROOT).as_posix()}")

# %%
# With PyTorch: the same network and loss in float64; autograd's gradients equal ours.
section("with PyTorch")
tp = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in params.items()}
xt = torch.tensor(X_batch, dtype=torch.float64)
yt = torch.tensor(Y_batch, dtype=torch.float64)
a2t = torch.sigmoid(torch.sigmoid(xt @ tp["W1"] + tp["b1"]) @ tp["W2"] + tp["b2"])
loss_t = 0.5 * ((a2t - yt) ** 2).sum() / xt.shape[0]
loss_t.backward()
print(f"torch loss {loss_t.item():.6f}   ours {loss_b:.6f}")
for name in ["W1", "b1", "W2", "b2"]:
    diff = float(np.max(np.abs(tp[name].grad.numpy() - g_b[name])))
    print(f"d{name:<2} max |torch - ours| {'< 1e-12' if diff < 1e-12 else f'= {diff:.1e}'}")  # last digits vary by machine
    np.testing.assert_allclose(tp[name].grad.numpy(), g_b[name], rtol=1e-6, atol=1e-8)
print("matches PyTorch autograd: OK")

# %%
# Key plot: the network with its forward values and backward gradients for the single example.
section("key plot")
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=120)
pos = {
    ("x", 0): (0, 2.6), ("x", 1): (0, 0.6),
    ("h", 0): (4, 2.6), ("h", 1): (4, 0.6),
    ("a", 0): (8, 2.6), ("a", 1): (8, 0.6),
}  # fmt: skip
for (src, dst, W, dW) in [("x", "h", W1, dW1), ("h", "a", W2, dW2)]:
    for i in range(2):
        for j in range(2):
            (x0, y0), (x1, y1) = pos[(src, i)], pos[(dst, j)]
            ax.annotate("", xy=(x1 - 0.55, y1), xytext=(x0 + 0.55, y0), arrowprops=dict(arrowstyle="->", color="0.5"))
            frac = 0.5 if i == j else 0.25  # crossing edges: label near the source, away from the crossing
            tx, ty = x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)
            ax.text(tx, ty + 0.12, f"w={W[i, j]:+.2f}\ng={dW[i, j]:+.4f}", fontsize=7.5, ha="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))  # fmt: skip
labels = {
    ("x", 0): f"x1 = {x[0, 0]:.2f}\ndL/dx1 = {dX[0, 0]:+.4f}",
    ("x", 1): f"x2 = {x[0, 1]:.2f}\ndL/dx2 = {dX[0, 1]:+.4f}",
    ("h", 0): f"z = {z1[0, 0]:+.4f}\nh = {h[0, 0]:.4f}\nδ1 = {dZ1[0, 0]:+.4f}",
    ("h", 1): f"z = {z1[0, 1]:+.4f}\nh = {h[0, 1]:.4f}\nδ1 = {dZ1[0, 1]:+.4f}",
    ("a", 0): f"z = {z2[0, 0]:+.4f}\na = {a2[0, 0]:.4f} (y=1)\nδ2 = {dZ2[0, 0]:+.4f}",
    ("a", 1): f"z = {z2[0, 1]:+.4f}\na = {a2[0, 1]:.4f} (y=0)\nδ2 = {dZ2[0, 1]:+.4f}",
}
for key, (px, py) in pos.items():
    ax.add_patch(plt.Circle((px, py), 0.55, fc="tab:blue" if key[0] == "x" else "tab:orange", alpha=0.25, ec="k"))
    ax.text(px, py, labels[key], ha="center", va="center", fontsize=7.5)
ax.text(4, -0.45, f"L = 1/2 Σ (a - y)² = {loss:.4f}.  Edge labels: weight w and its gradient g = dL/dw.", ha="center", fontsize=9)
ax.set(xlim=(-1, 9), ylim=(-0.8, 3.4), title="Backprop on the 2-2-2 network: values forward, deltas backward")
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout()
key_path = IMG / "c1_2_network_gradients.png"
fig.savefig(key_path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"saved {key_path.relative_to(ROOT).as_posix()}")

# %%
# Try this (not graded):
# - Replace the squared error with binary cross-entropy on each output: delta2 becomes a2 - y, and the
#   a2 (1 - a2) factor disappears (C1.4 shows why). Does XOR train faster?
# - Start from all-zero weights. Both hidden units get identical gradients forever (symmetry), so the
#   network never solves XOR. Print dW1 to see it (C3.3 is about breaking this symmetry).
# - Set LR_TRAIN to 50: the loss jumps around and may never leave the plateau.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
