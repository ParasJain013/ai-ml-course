# %% [markdown]
# # C1.2 Backpropagation by hand — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/C/c1_2_backprop_by_hand_ex2.py   (CPU, ~5 s)

# %%
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
SEED = 0
T0 = time.perf_counter()
OUT = Path(__file__).resolve().parents[3] / "outputs" / "c1_2"
OUT.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray, dec: int = 4) -> str:
    """Format a small array on one line, rounded, without -0.0."""
    return str((np.round(M, dec) + 0.0).tolist())


def sigmoid(z: FloatArray) -> FloatArray:
    """Logistic sigmoid, elementwise, stable for large |z|."""
    ez = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + ez), ez / (1.0 + ez))


# %%
section("exercise 1 check: ReLU hidden layer")
W1 = np.array([[0.1, 0.3], [-0.2, 0.4]])
b1 = np.array([0.1, -0.1])
W2 = np.array([[0.5, -0.4], [0.3, 0.2]])
b2 = np.array([0.0, 0.1])
x = np.array([[1.0, 0.5]])
y = np.array([[0.0, 1.0]])
z1 = x @ W1 + b1
h = np.maximum(z1, 0)
z2 = h @ W2 + b2
a = sigmoid(z2)
loss = 0.5 * float(np.sum((a - y) ** 2))
print(f"(a) z1 {show(z1, 5)}  h {show(h, 5)}  z2 {show(z2, 5)}  a {show(a, 5)}  loss {loss:.5f}")
d2 = (a - y) * a * (1 - a)
gW2 = h.T @ d2
gh = d2 @ W2.T
d1 = gh * (z1 > 0)
gW1 = x.T @ d1
print(f"(b) delta2 {show(d2, 5)}  shape {d2.shape}")
print(f"(b) dW2 {show(gW2, 5)}  shape {gW2.shape}")
print(f"(b) dh {show(gh, 5)}  delta1 {show(d1, 5)}  shape {d1.shape}")
print(f"(b) dW1 {show(gW1, 5)}  shape {gW1.shape}")
W2n, b2n = W2 - 1.0 * gW2, b2 - 1.0 * d2[0]
a_new = sigmoid(h @ W2n + b2n)
print(f"(c) W2 {show(W2n, 5)}  b2 {show(b2n, 5)}  loss {loss:.5f} -> {0.5 * float(np.sum((a_new - y) ** 2)):.5f}")

# %%
section("general MLP forward/backward")


def init_mlp(widths: list[int], gen: np.random.Generator) -> list[dict[str, FloatArray]]:
    """He-scaled random weights for widths [d_0, ..., d_L]. -> list of {"W": (d_l, d_{l+1}), "b": (d_{l+1},)}."""
    return [
        {"W": gen.normal(0, np.sqrt(2 / a), size=(a, b)), "b": np.zeros(b)}
        for a, b in zip(widths[:-1], widths[1:])
    ]


def forward_backward(
    X: FloatArray, Y: FloatArray, layers: list[dict[str, FloatArray]]
) -> tuple[float, list[dict[str, FloatArray]]]:
    """MSE loss and gradients of a ReLU MLP with a linear output. X (n, d), Y (n, k) -> (loss, grads per layer)."""
    n = X.shape[0]
    Hs = [X]  # cache: the input to every layer
    Zs = []
    for l, p in enumerate(layers):
        Z = Hs[-1] @ p["W"] + p["b"]  # (n, d_{l+1})
        Zs.append(Z)
        if l < len(layers) - 1:
            Hs.append(np.maximum(Z, 0.0))
    out = Zs[-1]
    loss = float(np.mean(np.sum((out - Y) ** 2, axis=1)))  # mean over examples of the squared error
    D = 2 * (out - Y) / n  # (n, k) delta of the last layer
    grads: list[dict[str, FloatArray]] = [{} for _ in layers]
    for l in range(len(layers) - 1, -1, -1):
        grads[l]["W"] = Hs[l].T @ D  # (d_l, n) @ (n, d_{l+1})
        grads[l]["b"] = D.sum(axis=0)  # (d_{l+1},)
        if l > 0:
            D = (D @ layers[l]["W"].T) * (Zs[l - 1] > 0)  # (n, d_l): through W, then through the ReLU
    for p, g in zip(layers, grads):
        assert g["W"].shape == p["W"].shape and g["b"].shape == p["b"].shape
    return loss, grads


def numerical_gradient(f: Callable[[FloatArray], float], x: FloatArray, h: float = 1e-6) -> FloatArray:
    """Central differences on an array of any shape, restoring each entry. x (...) -> (...)."""
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + h
        fp = f(x)
        x[idx] = old - h
        fm = f(x)
        x[idx] = old
        grad[idx] = (fp - fm) / (2 * h)
    return grad


def rel_error(a: FloatArray, b: FloatArray) -> float:
    """|a - b| / (|a| + |b|), safe when both are zero."""
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12))


gen = np.random.default_rng(SEED)
layers = init_mlp([3, 4, 5, 2], gen)
for p in layers:
    p["b"] = gen.normal(size=p["b"].shape)  # non-zero biases so their gradients are tested too
Xc, Yc = gen.normal(size=(5, 3)), gen.normal(size=(5, 2))
_, g = forward_backward(Xc, Yc, layers)
worst = 0.0
for l, p in enumerate(layers):
    for name in ["W", "b"]:
        num = numerical_gradient(lambda _: forward_backward(Xc, Yc, layers)[0], p[name])
        err = rel_error(g[l][name], num)
        worst = max(worst, err)
        print(f"layer {l + 1} d{name} shape {str(p[name].shape):<7} rel error {'< 1e-8' if err < 1e-8 else f'{err:.1e}'}")
print(f"gradient check 3-4-5-2 on a batch of 5: {'PASS' if worst < 1e-6 else 'FAIL'}")

# %%
section("training 1-32-32-1 on sin(3x)")
X = np.sort(gen.uniform(-2, 2, size=(200, 1)), axis=0)
Y = np.sin(3 * X) + 0.1 * gen.normal(size=X.shape)
net = init_mlp([1, 32, 32, 1], np.random.default_rng(SEED))
LR, STEPS = 0.05, 5000
for t in range(STEPS + 1):
    loss, g = forward_backward(X, Y, net)
    if t in (0, 500, 1000, 2000, STEPS):
        print(f"step {t:>4}: train MSE {loss:.4f}")
    for p, gp in zip(net, g):
        p["W"] -= LR * gp["W"]
        p["b"] -= LR * gp["b"]
print(f"noise variance (the best possible MSE on average): {0.1**2:.4f}")
H = X
for l, p in enumerate(net):
    H = H @ p["W"] + p["b"]
    if l < len(net) - 1:
        H = np.maximum(H, 0)
fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
ax.scatter(X[:, 0], Y[:, 0], s=8, label="data")
ax.plot(X[:, 0], H[:, 0], "k-", label="1-32-32-1 ReLU MLP")
ax.set(title="A from-scratch MLP fits sin(3x)", xlabel="x", ylabel="y")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(OUT / "c1_2_ex2_sin_fit.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved outputs/c1_2/c1_2_ex2_sin_fit.png")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
