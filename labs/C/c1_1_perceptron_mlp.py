# %% [markdown]
# # C1.1 Perceptron and multilayer perceptron — lab
# Lesson: lessons/C/c1_1_perceptron_mlp.md
# Run: uv run python labs/C/c1_1_perceptron_mlp.py   (CPU, defaults: ~5 s)
# GPU: no

# %%
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:  # plain `uv run`: draw to files, never open windows
    matplotlib.use("Agg")
    # matplotlib 3.11 warns when plt.show() runs under Agg; under `uv run` it is a harmless no-op
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

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
OUT = ROOT / "outputs" / "c1_1"  # scratch figures, gitignored
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
# Nothing here needs a GPU; GPU_SCALE only averages the piece counts over more random networks.
GPU_SCALE = False
N_POINTS = 40  # separable 2-D points, half per class
EPOCHS = 100  # perceptron passes over the data
N_NETS = 200 if GPU_SCALE else 20  # random ReLU networks averaged per architecture
GRID = 200_001 if GPU_SCALE else 20_001  # points along the line used to count linear pieces
SAW_GRID = 64 * 313  # cells on [0, 1]; every kink of the sawtooth (a multiple of 1/64) is a cell edge

# %%
# Data: two Gaussian clouds that a straight line can separate, and the four XOR points.
# Labels are -1 and +1, the perceptron's convention (logistic regression used 0 and 1).
section("data")
centres = np.array([[1.0, 1.0], [-1.0, -1.0]])
X_sep = np.vstack([rng.normal(c, 0.7, size=(N_POINTS // 2, 2)) for c in centres])  # (40, 2)
y_sep = np.repeat([1.0, -1.0], N_POINTS // 2)  # (40,)
order = rng.permutation(N_POINTS)
X_sep, y_sep = X_sep[order], y_sep[order]
X_xor = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])  # (4, 2)
y_xor = np.array([-1.0, 1.0, 1.0, -1.0])  # +1 when exactly one input is 1
print(f"separable X {X_sep.shape}  y {y_sep.shape}  class counts {int((y_sep > 0).sum())} / {int((y_sep < 0).sum())}")
print("first 3 rows:", show(X_sep[:3]), "labels", show(y_sep[:3]))
print("XOR X", show(X_xor), " y", show(y_xor))

# %%
# Worked example by hand: the perceptron rule on three points, learning rate 1, from zero.
section("worked example: perceptron rule by hand")


def perceptron_epoch(
    X: FloatArray, y: FloatArray, w: FloatArray, b: float, lr: float = 1.0, verbose: bool = False
) -> tuple[FloatArray, float, int]:
    """One pass of the perceptron rule. X (n, d), y (n,) in {-1, +1}, w (d,) -> (w, b, mistakes)."""
    w = w.copy()
    mistakes = 0
    for i in range(len(y)):
        z = float(X[i] @ w + b)  # the score
        if y[i] * z <= 0:  # wrong side, or exactly on the boundary: update
            w += lr * y[i] * X[i]
            b += lr * y[i]
            mistakes += 1
            if verbose:
                print(f"  point {i + 1} x={show(X[i])} y={y[i]:+.0f} z={z:+.1f} -> update: w={show(w)} b={b:+.1f}")
        elif verbose:
            print(f"  point {i + 1} x={show(X[i])} y={y[i]:+.0f} z={z:+.1f} -> correct, no change")
    return w, b, mistakes


X3 = np.array([[2.0, 1.0], [1.0, 3.0], [-1.0, -1.0]])
y3 = np.array([1.0, -1.0, -1.0])
w3, b3 = np.zeros(2), 0.0
for epoch in range(1, 10):
    print(f"epoch {epoch}:")
    w3, b3, m3 = perceptron_epoch(X3, y3, w3, b3, verbose=True)
    if m3 == 0:
        print(f"no mistakes in epoch {epoch}: converged with w={show(w3)} b={b3:+.1f}")
        break

# %%
# From scratch: the perceptron on separable data (converges) and on XOR (never does).
section("from scratch: perceptron on separable data and on XOR")


def predict(X: FloatArray, w: FloatArray, b: float) -> FloatArray:
    """Perceptron prediction: +1 where the score is positive, else -1. X (n, d) -> (n,)."""
    return np.where(X @ w + b > 0, 1.0, -1.0)


def train_perceptron(X: FloatArray, y: FloatArray, epochs: int) -> tuple[FloatArray, float, list[int]]:
    """Run the perceptron rule for up to `epochs` passes. X (n, d), y (n,) -> (w (d,), b, mistakes per epoch)."""
    w, b = np.zeros(X.shape[1]), 0.0
    history: list[int] = []
    for _ in range(epochs):
        w, b, m = perceptron_epoch(X, y, w, b)
        history.append(m)
        if m == 0:
            break
    return w, b, history


w_sep, b_sep, hist_sep = train_perceptron(X_sep, y_sep, EPOCHS)
acc_sep = np.mean(predict(X_sep, w_sep, b_sep) == y_sep)
print(f"separable: mistakes per epoch, first 10: {hist_sep[:10]} ... last 3: {hist_sep[-3:]}")
print(f"separable: converged after {len(hist_sep)} epochs  w={show(w_sep)} b={b_sep:+.1f}  train accuracy {acc_sep:.4f}")

print("XOR, epoch 1 from w = 0, b = 0:")
w_x, b_x, _ = perceptron_epoch(X_xor, y_xor, np.zeros(2), 0.0, verbose=True)
w_x, b_x, hist_x = train_perceptron(X_xor, y_xor, EPOCHS)
acc_x = np.mean(predict(X_xor, w_x, b_x) == y_xor)
print(f"XOR: mistakes per epoch, first 8: {hist_x[:8]} ... epoch {len(hist_x)}: {hist_x[-1]}")
print(f"XOR: after {len(hist_x)} epochs  w={show(w_x)} b={b_x:+.1f}  train accuracy {acc_x:.4f}  (never 1.0)")

# %%
# From scratch: two layers with hand-set weights solve XOR.
# Hidden unit 1 is OR (fires if x1 + x2 > 0.5), hidden unit 2 is AND (fires if x1 + x2 > 1.5),
# and the output fires if OR and not AND, which is XOR.
section("from scratch: a two-layer network with hand-set weights solves XOR")


def step(z: FloatArray) -> FloatArray:
    """Heaviside step, elementwise: 1 where z > 0, else 0."""
    return (z > 0).astype(np.float64)


W1_xor = np.array([[1.0, 1.0], [1.0, 1.0]])  # (d_in=2, d_hidden=2): column j is hidden unit j's weights
b1_xor = np.array([-0.5, -1.5])  # (2,) OR threshold, AND threshold
W2_xor = np.array([[1.0], [-1.0]])  # (2, 1): +OR, -AND
b2_xor = np.array([-0.5])  # (1,)


def xor_net(X: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Hand-set 2-2-1 step network. X (n, 2) -> hidden (n, 2), output (n,)."""
    H = step(X @ W1_xor + b1_xor)  # (n, 2)
    out = step(H @ W2_xor + b2_xor)[:, 0]  # (n,)
    return H, out


H_xor, out_xor = xor_net(X_xor)
for x_row, h_row, o in zip(X_xor, H_xor, out_xor):
    print(f"x={show(x_row)}  hidden (OR, AND)={show(h_row)}  output={o:.0f}")
print("matches XOR:", "OK" if np.array_equal(2 * out_xor - 1, y_xor) else "FAIL")

# %%
# From scratch: activation functions and their derivatives.
section("from scratch: activations and derivatives")


def sigmoid(z: FloatArray) -> FloatArray:
    """Logistic sigmoid, elementwise, stable for large |z|."""
    ez = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + ez), ez / (1.0 + ez))


def relu(z: FloatArray) -> FloatArray:
    """max(0, z), elementwise."""
    return np.maximum(z, 0.0)


ACTS = {
    "sigmoid": (sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z))),
    "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    "relu": (relu, lambda z: (z > 0).astype(np.float64)),  # derivative at exactly 0 taken as 0
}
z_pts = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
print(f"z            {show(z_pts)}")
for name, (f, df) in ACTS.items():
    print(f"{name:<8} f  {show(f(z_pts))}")
    print(f"{name:<8} f' {show(df(z_pts))}")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=120)
zz = np.linspace(-5, 5, 401)
for (name, (f, df)), ls in zip(ACTS.items(), ["-", "--", ":"]):
    axes[0].plot(zz, f(zz), ls, label=name)
    axes[1].plot(zz, df(zz), ls, label=name)
axes[0].set(title="Activations: sigmoid and tanh saturate", xlabel="z", ylabel="f(z)")
axes[1].set(title="Derivatives: sigmoid's peaks at 0.25", xlabel="z", ylabel="f'(z)")
for ax in axes:
    ax.legend(loc="best")
fig.tight_layout()
fig.savefig(OUT / "c1_1_activations.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"saved {(OUT / 'c1_1_activations.png').relative_to(ROOT).as_posix()}")

# %%
# From scratch: stacking linear layers without an activation collapses to one linear layer.
section("from scratch: two linear layers equal one")
W1 = rng.normal(size=(3, 4))  # (d_in=3, 4)
W2 = rng.normal(size=(4, 2))  # (4, d_out=2)
X_lin = rng.normal(size=(5, 3))  # (n=5, 3)
two_layers = (X_lin @ W1) @ W2  # (5, 2)
one_layer = X_lin @ (W1 @ W2)  # (5, 2): a single (3, 2) matrix
print(f"W1 {W1.shape} @ W2 {W2.shape} = W {(W1 @ W2).shape}")
print(f"max |(X W1) W2 - X (W1 W2)| = {np.max(np.abs(two_layers - one_layer)):.1e}")
np.testing.assert_allclose(two_layers, one_layer, rtol=1e-12, atol=1e-12)
print("two linear layers equal one linear layer: OK")
with_relu = relu(X_lin @ W1) @ W2
print(f"with a ReLU in between, max |difference| = {np.max(np.abs(with_relu - one_layer)):.4f} (no longer one matrix)")

# %%
# From scratch: a ReLU network is piecewise linear. Count its linear pieces along a line.
# Along a line, the network is linear between the points where some ReLU switches on or off, so the
# number of pieces is 1 + the number of times the on/off pattern of all units changes.
section("from scratch: linear pieces of ReLU networks along a line")


def count_pieces(Ws: list[FloatArray], bs: list[FloatArray], X_line: FloatArray) -> int:
    """Count linear pieces of a ReLU MLP along a path of points. X_line (m, d) -> number of pieces."""
    H = X_line
    patterns = []
    for W, b in zip(Ws[:-1], bs[:-1]):  # hidden layers only; the last layer is linear
        Z = H @ W + b  # (m, width)
        patterns.append(Z > 0)
        H = relu(Z)
    P = np.concatenate(patterns, axis=1)  # (m, total hidden units)
    changes = np.any(P[1:] != P[:-1], axis=1)  # (m - 1,)
    return int(changes.sum()) + 1


def random_relu_net(
    widths: list[int], gen: np.random.Generator
) -> tuple[list[FloatArray], list[FloatArray]]:
    """Random MLP with He-scaled weights. widths [d_in, h1, ..., d_out] -> (Ws, bs)."""
    Ws = [gen.normal(0, np.sqrt(2 / a), size=(a, b)) for a, b in zip(widths[:-1], widths[1:])]
    bs = [gen.normal(0, 0.5, size=b) for b in widths[1:]]
    return Ws, bs


t = np.linspace(-3, 3, GRID)
direction = np.array([0.6, 0.8])
X_line = t[:, None] * direction  # (GRID, 2): points on a line through the origin
gen = np.random.default_rng(SEED)
results: dict[str, float] = {}
for label, widths in [
    ("1 hidden layer,  8 units", [2, 8, 1]),
    ("1 hidden layer, 16 units", [2, 16, 1]),
    ("1 hidden layer, 32 units", [2, 32, 1]),
    ("2 hidden layers, 8 + 8", [2, 8, 8, 1]),
    ("4 hidden layers, 8 x 4", [2, 8, 8, 8, 8, 1]),
]:
    counts = [count_pieces(*random_relu_net(widths, gen), X_line) for _ in range(N_NETS)]
    results[label] = float(np.mean(counts))
    print(f"random {label:<26} units {sum(widths[1:-1]):>2}: mean pieces {np.mean(counts):5.1f}  (max {max(counts)})")

# Depth can do far better than random weights suggest: the tent map, composed L times.
# One layer of 2 ReLUs: g(x) = 2 relu(x) - 4 relu(x - 0.5) folds [0, 1] onto itself (up, then down).
W_tent = np.array([[1.0, 1.0]])  # (1, 2)
b_tent = np.array([0.0, -0.5])  # (2,)
v_tent = np.array([[2.0], [-4.0]])  # (2, 1)


def sawtooth_net(L: int) -> tuple[list[FloatArray], list[FloatArray]]:
    """L tent layers of 2 ReLUs each, merged into a plain MLP. Input (m, 1) -> output (m, 1)."""
    Ws = [W_tent] + [v_tent @ W_tent] * (L - 1) + [v_tent]  # fold the output map into the next layer
    bs = [b_tent] * L + [np.zeros(1)]
    return Ws, bs


def mlp_forward(Ws: list[FloatArray], bs: list[FloatArray], X: FloatArray) -> FloatArray:
    """ReLU MLP forward pass with a linear last layer. X (m, d_in) -> (m, d_out)."""
    H = X
    for W, b in zip(Ws[:-1], bs[:-1]):
        H = relu(H @ W + b)
    return H @ Ws[-1] + bs[-1]


x01 = ((np.arange(SAW_GRID) + 0.5) / SAW_GRID)[:, None]  # (SAW_GRID, 1) cell midpoints: never exactly on a kink
for L in range(1, 7):
    pieces = count_pieces(*sawtooth_net(L), x01)
    print(f"sawtooth, {L} layer(s) of 2 ReLUs ({2 * L:>2} units): {pieces:>2} pieces; one hidden layer needs >= {pieces - 1} units")

fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
for L, ls in zip([1, 2, 4], ["-", "--", ":"]):
    ax.plot(x01[:, 0], mlp_forward(*sawtooth_net(L), x01)[:, 0], ls, label=f"{L} layer(s): {2**L} pieces")
ax.set(title="Each tent layer doubles the pieces: depth multiplies", xlabel="input x", ylabel="network output")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "c1_1_sawtooth.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"saved {(OUT / 'c1_1_sawtooth.png').relative_to(ROOT).as_posix()}")

# %%
# With scikit-learn: Perceptron runs the same rule; MLPClassifier learns XOR by itself.
section("with scikit-learn")
sk = Perceptron(eta0=1.0, shuffle=False, tol=None, max_iter=EPOCHS, random_state=SEED)
sk.fit(X_sep, y_sep)
print(f"sklearn Perceptron: coef_ {show(sk.coef_[0])}  intercept_ {sk.intercept_[0]:+.1f}")
np.testing.assert_allclose(sk.coef_[0], w_sep, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(sk.intercept_[0], b_sep, rtol=1e-12, atol=1e-12)
print("matches the from-scratch perceptron exactly: OK")
sk_x = Perceptron(eta0=1.0, shuffle=False, tol=None, max_iter=EPOCHS, random_state=SEED).fit(X_xor, y_xor)
print(f"sklearn Perceptron on XOR: train accuracy {sk_x.score(X_xor, y_xor):.4f}")
mlp = MLPClassifier(hidden_layer_sizes=(4,), activation="tanh", solver="lbfgs", max_iter=1000, random_state=SEED)
mlp.fit(X_xor, y_xor)
print(f"MLPClassifier(4 tanh units) on XOR: predictions {mlp.predict(X_xor).tolist()}  train accuracy {mlp.score(X_xor, y_xor):.4f}")
print(f"MLPClassifier probabilities P(y=+1): {show(mlp.predict_proba(X_xor)[:, 1])}")

# %%
# Key plot: the perceptron's line separates the clouds, fails on XOR; two layers carve XOR out.
section("key plot")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=120)


def grid_points(lo: float, hi: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """A 300 x 300 grid over [lo, hi]^2. -> (G1, G2, points (90000, 2))."""
    G1, G2 = np.meshgrid(np.linspace(lo, hi, 300), np.linspace(lo, hi, 300))
    return G1, G2, np.c_[G1.ravel(), G2.ravel()]


def draw(ax: plt.Axes, X: FloatArray, y: FloatArray, G1: FloatArray, G2: FloatArray, region: FloatArray, title: str) -> None:
    """Shade the predicted regions (+1 orange, -1 blue) and plot the points with class markers."""
    ax.contourf(G1, G2, region.reshape(G1.shape), levels=[-2, 0, 2], colors=["tab:blue", "tab:orange"], alpha=0.2)
    ax.scatter(*X[y > 0].T, marker="^", c="tab:orange", edgecolor="k", s=50, label="y = +1")
    ax.scatter(*X[y < 0].T, marker="o", c="tab:blue", edgecolor="k", s=50, label="y = -1")
    ax.set(title=title, xlabel="x1", ylabel="x2", xlim=(G1.min(), G1.max()), ylim=(G2.min(), G2.max()))


G1s, G2s, pts_s = grid_points(-3, 3)
draw(axes[0], X_sep, y_sep, G1s, G2s, predict(pts_s, w_sep, b_sep), f"Perceptron, separable: {acc_sep:.0%} after {len(hist_sep)} epochs")
gx1, gx2, grid_x = grid_points(-0.5, 1.5)
draw(axes[1], X_xor, y_xor, gx1, gx2, predict(grid_x, w_x, b_x), f"Perceptron on XOR: {acc_x:.0%} after {EPOCHS} epochs")
_, out_grid = xor_net(grid_x)
draw(axes[2], X_xor, y_xor, gx1, gx2, 2 * out_grid - 1, "Two layers (OR and not AND): 100%")
axes[0].legend(loc="lower right")
fig.tight_layout()
key_path = IMG / "c1_1_xor_boundaries.png"
fig.savefig(key_path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"saved {key_path.relative_to(ROOT).as_posix()}")

# %%
# Try this (not graded):
# - Move the two cloud centres closer (e.g. [0.5, 0.5] and [-0.5, -0.5]) so the classes overlap:
#   the perceptron never converges, and the last weights depend on where you stop.
# - Change the perceptron learning rate to 0.1 starting from w = 0: the weights scale by 0.1 but
#   the boundary and the number of mistakes do not change. Why?
# - Count pieces along a line for a random 2-64-1 network and a random 2-8-8-8-8-8-8-8-8-1 network
#   (64 units each). Which has more pieces with random weights?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
