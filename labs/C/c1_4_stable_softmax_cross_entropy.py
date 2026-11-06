# %% [markdown]
# # C1.4 Softmax and cross-entropy, written to be numerically stable — lab
# Lesson: lessons/C/c1_4_stable_softmax_cross_entropy.md
# Run: uv run python labs/C/c1_4_stable_softmax_cross_entropy.py   (CPU, defaults: ~5 s)
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
from scipy.special import log_softmax as scipy_log_softmax
from scipy.special import logsumexp as scipy_logsumexp
from scipy.special import softmax as scipy_softmax
from sklearn.metrics import log_loss

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
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
OUT = ROOT / "outputs" / "c1_4"  # scratch figures, gitignored
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
# Nothing here needs a GPU: every array is tiny.
GPU_SCALE = False
N_CHECK, K_CHECK = 5, 3  # batch and class count for the gradient checks

# %%
# Data: a few hand-picked logit vectors, and a random batch for the checks.
section("data")
z_small = np.array([2.0, 1.0, 0.0])  # the worked example; true class is class 1 (index 0)
z_big = np.array([1000.0, 1001.0, 1002.0])
z_neg = np.array([-1000.0, -1001.0, -1002.0])
Z_rand = rng.normal(0, 2, size=(N_CHECK, K_CHECK))  # (n, K) logits
y_rand = rng.integers(0, K_CHECK, size=N_CHECK)  # (n,) class indices
print(f"worked example logits {show(z_small)}  true class index 0")
print(f"large logits {show(z_big)}   very negative logits {show(z_neg)}")
print(f"random batch Z {Z_rand.shape}  y {y_rand.tolist()}")

# %%
# Worked example by hand: softmax, cross-entropy and its gradient for three classes.
section("worked example: softmax, cross-entropy and p - y")


def softmax_naive(z: FloatArray) -> FloatArray:
    """Softmax straight from the definition. z (..., K) -> (..., K). Overflows for large logits."""
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


e_small = np.exp(z_small)
p_small = softmax_naive(z_small)
y_onehot = np.array([1.0, 0.0, 0.0])
print(f"exp(z) = {show(e_small)}   sum = {e_small.sum():.4f}")
print(f"p = softmax(z) = {show(p_small)}   sum = {p_small.sum():.4f}")
print(f"cross-entropy = -log p[0] = {-np.log(p_small[0]):.4f}")
print(f"gradient dL/dz = p - y = {show(p_small - y_onehot)}   sum = {(p_small - y_onehot).sum() + 0.0:.4f}")

# %%
# The overflow: the naive softmax on [1000, 1001, 1002], and the max-subtraction fix.
section("from scratch: naive softmax overflows")
with np.errstate(over="ignore", invalid="ignore"):
    print(f"exp([1000, 1001, 1002]) = {np.exp(z_big).tolist()}")
    print(f"naive softmax(z_big) = {softmax_naive(z_big).tolist()}")
    print(f"naive softmax(z_neg) = {softmax_naive(z_neg).tolist()}  (exp underflows to 0: 0/0)")
print(f"largest float64 = {np.finfo(np.float64).max:.4e}; exp overflows above log of that = {np.log(np.finfo(np.float64).max):.2f}")
print(f"largest float32 = {np.finfo(np.float32).max:.4e}; exp overflows above {np.log(np.finfo(np.float32).max):.2f}")

section("from scratch: stable softmax, log-sum-exp, log-softmax")


def softmax(z: FloatArray) -> FloatArray:
    """Stable softmax over the last axis: subtract the max first. z (..., K) -> (..., K)."""
    shifted = z - z.max(axis=-1, keepdims=True)  # largest entry is now 0, so exp(...) <= 1
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def logsumexp(z: FloatArray) -> FloatArray:
    """log(sum(exp(z))) over the last axis, without overflow. z (..., K) -> (...)."""
    m = z.max(axis=-1)
    return m + np.log(np.exp(z - m[..., None]).sum(axis=-1))


def log_softmax(z: FloatArray) -> FloatArray:
    """log(softmax(z)) without ever taking log of a rounded-to-zero probability. z (..., K) -> (..., K)."""
    return z - logsumexp(z)[..., None]


for name, z in [("z_big", z_big), ("z_neg", z_neg), ("z_small", z_small)]:
    print(f"{name}: softmax {show(softmax(z))}  logsumexp {logsumexp(z):.4f}  log_softmax {show(log_softmax(z))}")
np.testing.assert_allclose(logsumexp(Z_rand), scipy_logsumexp(Z_rand, axis=-1), rtol=1e-12)
np.testing.assert_allclose(log_softmax(Z_rand), scipy_log_softmax(Z_rand, axis=-1), rtol=1e-12, atol=1e-14)
np.testing.assert_allclose(softmax(Z_rand), scipy_softmax(Z_rand, axis=-1), rtol=1e-12)
for z in (z_big, z_neg):
    np.testing.assert_allclose(logsumexp(z), scipy_logsumexp(z), rtol=1e-12)
    np.testing.assert_allclose(log_softmax(z), scipy_log_softmax(z), rtol=1e-12)
print("matches scipy.special logsumexp, log_softmax and softmax: OK")
print(f"softmax(z + 7) == softmax(z): {np.allclose(softmax(z_small + 7), softmax(z_small))}  (shift invariance)")

# %%
# Cross-entropy straight from logits, and why -log(softmax(z)) is not good enough.
section("from scratch: cross-entropy from logits")


def cross_entropy(Z: FloatArray, y: IntArray) -> float:
    """Mean cross-entropy from logits: logsumexp(z_i) - z_i[y_i]. Z (n, K), y (n,) class indices -> scalar."""
    n = Z.shape[0]
    return float(np.mean(logsumexp(Z) - Z[np.arange(n), y]))


def cross_entropy_naive(Z: FloatArray, y: IntArray) -> float:
    """Mean of -log(softmax(z)[y]) with the naive softmax. For comparison only."""
    n = Z.shape[0]
    return float(np.mean(-np.log(softmax_naive(Z)[np.arange(n), y])))


def cross_entropy_via_probs(Z: FloatArray, y: IntArray) -> float:
    """Mean of -log(stable softmax(z)[y]): no overflow, but the probability can still round to 0."""
    n = Z.shape[0]
    return float(np.mean(-np.log(softmax(Z)[np.arange(n), y])))


cases = {
    "moderate [2, 1, 0], y=0": (np.array([[2.0, 1.0, 0.0]]), np.array([0])),
    "large [1000, 1001, 1002], y=2": (z_big[None, :], np.array([2])),
    "confidently wrong [0, 800, 0], y=0": (np.array([[0.0, 800.0, 0.0]]), np.array([0])),
}
with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    for label, (Zc, yc) in cases.items():
        print(
            f"{label:<36} naive {cross_entropy_naive(Zc, yc):>9.4f}   "
            f"-log(stable softmax) {cross_entropy_via_probs(Zc, yc):>9.4f}   from logits {cross_entropy(Zc, yc):>9.4f}"
        )

# %%
# The gradient: dL/dz = p - y for one example, (P - Y)/n for a batch mean. Check numerically.
section("from scratch: the gradient p - y, checked")


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
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * h)
    return grad


def rel_error(a: FloatArray, b: FloatArray) -> float:
    """Relative error |a - b| / (|a| + |b|), safe when both are zero."""
    # from labs/A/a4_2_gradients_jacobians.py
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12))


def one_hot(y: IntArray, k: int) -> FloatArray:
    """Class indices to one-hot rows. y (n,) -> (n, k)."""
    Y = np.zeros((len(y), k))
    Y[np.arange(len(y)), y] = 1.0
    return Y


G_analytic = (softmax(Z_rand) - one_hot(y_rand, K_CHECK)) / N_CHECK  # (n, K)
G_numeric = numerical_gradient(lambda Z: cross_entropy(Z, y_rand), Z_rand.copy())
err = rel_error(G_analytic, G_numeric)
print(f"analytic row 0 {show(G_analytic[0])}   numeric row 0 {show(G_numeric[0])}")
print(f"relative error over the batch {'< 1e-8' if err < 1e-8 else f'{err:.1e}'}: {'PASS' if err < 1e-6 else 'FAIL'}")
print(f"every row of the gradient sums to 0: {np.allclose(G_analytic.sum(axis=1), 0)}")
J = np.diag(p_small) - np.outer(p_small, p_small)  # softmax Jacobian dp_k/dz_j
print(f"softmax Jacobian for z = [2, 1, 0]:\n{np.round(J, 4)}")
g_chain = J.T @ (-y_onehot / p_small)  # chain rule through the Jacobian: (K, K) @ (K,)
print(f"chain rule through the Jacobian: {show(g_chain)}   p - y: {show(p_small - y_onehot)}")

# %%
# The two-class case is logistic regression: logits [0, z] give softmax = [1 - sigmoid(z), sigmoid(z)].
section("from scratch: two classes = logistic regression")
z2 = rng.normal(0, 3, size=8)  # (n,) one logit per example
y2 = rng.integers(0, 2, size=8)  # (n,) labels 0 / 1
Z2 = np.stack([np.zeros_like(z2), z2], axis=1)  # (n, 2)
ce_two = cross_entropy(Z2, y2)
logloss_b21 = float(np.mean(np.logaddexp(0.0, z2) - y2 * z2))  # B2.1's loss from logits
sigmoid = 1.0 / (1.0 + np.exp(-z2))
print(f"softmax([0, z])[1] equals sigmoid(z): {np.allclose(softmax(Z2)[:, 1], sigmoid)}")
print(f"2-class cross-entropy {ce_two:.6f}   B2.1 log-loss {logloss_b21:.6f}   sklearn log_loss {log_loss(y2, sigmoid):.6f}")
g_two = (softmax(Z2) - one_hot(y2, 2)) / len(y2)
print(f"gradient wrt the logit z: (p1 - y) / n = {show(g_two[:3, 1])} ...   B2.1's (sigmoid - y) / n = {show(((sigmoid - y2) / len(y2))[:3])} ...")

# %%
# A fused cross-entropy operation for the C1.3 tensor engine.
section("from scratch: a fused cross-entropy op for the tensor engine")


def topological_order(root: Tensor) -> list[Tensor]:
    """Every node reachable from root, each after all of its inputs (iterative DFS post-order)."""
    # from labs/C/c1_3_autograd_engine.py
    order: list[Tensor] = []
    visited: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(root, False)]
    while stack:
        node, inputs_done = stack.pop()
        if inputs_done:
            order.append(node)
            continue
        if id(node) in visited:
            continue
        visited.add(id(node))
        stack.append((node, True))
        for child in node._prev:
            if id(child) not in visited:
                stack.append((child, False))
    return order


def unbroadcast(grad: FloatArray, shape: tuple[int, ...]) -> FloatArray:
    """Sum a gradient down to `shape`, undoing NumPy broadcasting."""
    # from labs/C/c1_3_autograd_engine.py
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """A NumPy array that records operations for reverse-mode autodiff (a subset of the C1.3 class)."""

    # from labs/C/c1_3_autograd_engine.py (add, mul, sum, exp, log, matmul), plus cross_entropy below

    def __init__(self, data: FloatArray | float, _prev: tuple[Tensor, ...] = (), op: str = "") -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._prev = _prev
        self._backward: Callable[[], None] = lambda: None
        self.op = op

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __add__(self, other: Tensor | float) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other: Tensor | float) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += unbroadcast(other.data * out.grad, self.shape)
            other.grad += unbroadcast(self.data * out.grad, other.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward() -> None:
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward() -> None:
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.shape)

        out._backward = _backward
        return out

    def exp(self) -> Tensor:
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def cross_entropy(self, y: IntArray) -> Tensor:
        """Mean cross-entropy of logits (n, K) against class indices y (n,), as ONE node. -> scalar."""
        n = self.data.shape[0]
        lse = logsumexp(self.data)  # (n,) stable
        out = Tensor(np.mean(lse - self.data[np.arange(n), y]), (self,), "cross_entropy")

        def _backward() -> None:
            P = np.exp(self.data - lse[:, None])  # (n, K) softmax, recomputed stably from the cache
            P[np.arange(n), y] -= 1.0  # P - Y without building Y
            self.grad += P / n * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Backpropagate from a scalar tensor."""
        if self.data.size != 1:
            raise ValueError(f"backward() needs a scalar output, got shape {self.shape}")
        self.grad = np.ones_like(self.data)
        for node in reversed(topological_order(self)):
            node._backward()


def unfused_cross_entropy(Z: Tensor, Y: FloatArray) -> Tensor:
    """The same loss built from exp, sum, log, mul: mean(log(sum(exp(Z))) - sum(Z * Y)). No max subtraction."""
    lse = Z.exp().sum(axis=1).log()  # (n,)
    picked = (Z * Tensor(Y)).sum(axis=1)  # (n,) z_i[y_i]
    return (lse + picked * -1.0).sum() * (1.0 / Z.shape[0])


Zt = Tensor(Z_rand.copy())
loss_fused = Zt.cross_entropy(y_rand)
loss_fused.backward()
Zu = Tensor(Z_rand.copy())
loss_unfused = unfused_cross_entropy(Zu, one_hot(y_rand, K_CHECK))
loss_unfused.backward()
print(f"fused loss {float(loss_fused.data):.6f}   unfused loss {float(loss_unfused.data):.6f}   numpy {cross_entropy(Z_rand, y_rand):.6f}")
err_f = rel_error(Zt.grad, G_numeric)
print(f"fused gradient vs numeric: rel error {'< 1e-8' if err_f < 1e-8 else f'{err_f:.1e}'}  {'PASS' if err_f < 1e-6 else 'FAIL'}")
print(f"graph nodes: fused {len(topological_order(loss_fused))}, unfused {len(topological_order(loss_unfused))}")

# A layer in front of it: logits = X W + b, so the fused op is a drop-in loss for an MLP.
Xl = Tensor(rng.normal(size=(N_CHECK, 4)))
Wl = Tensor(rng.normal(size=(4, K_CHECK)))
bl = Tensor(np.zeros(K_CHECK))
(Xl @ Wl + bl).cross_entropy(y_rand).backward()
num_W = numerical_gradient(lambda W: cross_entropy(Xl.data @ W + bl.data, y_rand), Wl.data.copy())
err_W = rel_error(Wl.grad, num_W)
print(f"dense layer + fused loss, dW vs numeric: rel error {'< 1e-8' if err_W < 1e-8 else f'{err_W:.1e}'}  {'PASS' if err_W < 1e-6 else 'FAIL'}")

with np.errstate(over="ignore", invalid="ignore"):
    Zbig_f, Zbig_u = Tensor(z_big[None, :].copy()), Tensor(z_big[None, :].copy())
    Lf = Zbig_f.cross_entropy(np.array([2]))
    Lf.backward()
    Lu = unfused_cross_entropy(Zbig_u, one_hot(np.array([2]), 3))
    Lu.backward()
print(f"logits [1000, 1001, 1002], y = 2: fused loss {float(Lf.data):.4f} grad {show(Zbig_f.grad)}")
print(f"                                 unfused loss {float(Lu.data)} grad {Zbig_u.grad.tolist()}")

# %%
# With PyTorch: F.cross_entropy takes logits and class indices, and is stable.
section("with PyTorch")
Zp = torch.tensor(Z_rand, dtype=torch.float64, requires_grad=True)
loss_p = torch.nn.functional.cross_entropy(Zp, torch.tensor(y_rand))
loss_p.backward()
print(f"torch cross_entropy {loss_p.item():.6f}   ours {cross_entropy(Z_rand, y_rand):.6f}")
diff = float(np.max(np.abs(Zp.grad.numpy() - Zt.grad)))
print(f"max |torch grad - fused grad| {'< 1e-12' if diff < 1e-12 else f'= {diff:.1e}'}")
np.testing.assert_allclose(Zp.grad.numpy(), Zt.grad, rtol=1e-6, atol=1e-10)
print("matches torch.nn.functional.cross_entropy: OK")
zb32 = torch.tensor([[1000.0, 1001.0, 1002.0]], dtype=torch.float32)
print(f"float32, logits [1000, 1001, 1002]: torch softmax {show(torch.softmax(zb32, dim=1).double().numpy())}  "
      f"cross_entropy(y=2) {torch.nn.functional.cross_entropy(zb32, torch.tensor([2])).item():.4f}")
print(f"torch.exp(torch.tensor(89.0)) in float32 = {torch.exp(torch.tensor(89.0)).item()}")

# %%
# Key plot: where the naive formula breaks, in float32 and float64, as logits grow.
section("key plot")
scales = np.logspace(0, 3.2, 400)  # logits s * [1, 0, -1] with the true class the smallest one


def ce_naive_dtype(s: float, dtype: type) -> float:
    """Naive -log(softmax(z)[2]) in a given float type for z = s [1, 0, -1]."""
    z = (np.array([1.0, 0.0, -1.0]) * s).astype(dtype)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        e = np.exp(z)
        return float(-np.log(e[2] / e.sum()))


stable = np.array([cross_entropy((np.array([[1.0, 0.0, -1.0]]) * s), np.array([2])) for s in scales])
naive64 = np.array([ce_naive_dtype(s, np.float64) for s in scales])
naive32 = np.array([ce_naive_dtype(s, np.float32) for s in scales])
first_bad32 = scales[~np.isfinite(naive32)][0]
first_bad64 = scales[~np.isfinite(naive64)][0]
print(f"naive cross-entropy first returns inf/nan at scale {first_bad32:.1f} (float32), {first_bad64:.1f} (float64)")
print(f"stable value there: {cross_entropy(np.array([[1.0, 0.0, -1.0]]) * first_bad32, np.array([2])):.2f} and "
      f"{cross_entropy(np.array([[1.0, 0.0, -1.0]]) * first_bad64, np.array([2])):.2f}")
fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
ax.plot(scales, stable, "k-", lw=3, alpha=0.4, label="from logits (log-sum-exp), any dtype")
ok32, ok64 = np.isfinite(naive32), np.isfinite(naive64)
ax.plot(scales[ok32], naive32[ok32], "--", label="naive, float32")
ax.plot(scales[ok64], naive64[ok64], ":", lw=2, label="naive, float64")
ax.axvline(first_bad32, color="tab:blue", lw=0.8)
ax.axvline(first_bad64, color="tab:orange", lw=0.8)
ax.text(first_bad32 * 1.05, 5, f"float32 breaks\nat s = {first_bad32:.0f}", fontsize=8)
ax.text(first_bad64 * 1.05, 5, f"float64 breaks\nat s = {first_bad64:.0f}", fontsize=8)
ax.set(xscale="log", yscale="log", xlabel="logit scale s (logits = s * [1, 0, -1]; log scale)",
       ylabel="cross-entropy of the true class (log scale)",
       title=f"Naive cross-entropy fails at s = {first_bad32:.0f} in float32: p rounds to 0")
ax.legend(loc="upper left")
fig.tight_layout()
key_path = IMG / "c1_4_naive_vs_stable.png"
fig.savefig(key_path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"saved {key_path.relative_to(ROOT).as_posix()}")

# %%
# Try this (not graded):
# - Compute softmax(z / T) for z = [2, 1, 0] and T = 0.1, 1, 10 (temperature, used in F2.4). What happens
#   to the probabilities, and to the max-subtraction trick, as T -> 0?
# - Label smoothing: replace the one-hot target with 0.9 on the true class and 0.05 elsewhere. Derive the
#   gradient (still p - y, with the smoothed y) and check it numerically.
# - Feed `cross_entropy` probabilities instead of logits by mistake: the loss is still finite. Why is that
#   worse than a crash?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
