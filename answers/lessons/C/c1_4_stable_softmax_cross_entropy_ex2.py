# %% [markdown]
# # C1.4 Stable softmax and cross-entropy — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/C/c1_4_stable_softmax_cross_entropy_ex2.py   (CPU, ~2 s)

# %%
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


def logsumexp(z: FloatArray) -> FloatArray:
    """Stable log(sum(exp(z))) over the last axis. z (..., K) -> (...)."""
    m = z.max(axis=-1)
    return m + np.log(np.exp(z - m[..., None]).sum(axis=-1))


def softmax(z: FloatArray) -> FloatArray:
    """Stable softmax over the last axis. z (..., K) -> (..., K)."""
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def numerical_gradient(f: Callable[[FloatArray], float], x: FloatArray, h: float = 1e-6) -> FloatArray:
    """Central differences on an array of any shape. x (...) -> (...)."""
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


# %%
section("exercise 1 check")
z = np.array([0.0, np.log(2), np.log(5)])
p = softmax(z)
print(f"(a) p {show(p)} = [1/8, 1/4, 5/8]  loss {-np.log(p[1]):.4f} = log 4  p - y {show(p - np.array([0, 1, 0]))}")
zb = np.array([500.0, 500.0, 498.0])
print(f"(b) LSE {logsumexp(zb):.4f}  softmax {show(softmax(zb))}")
P = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
Y = np.array([[1.0, 0, 0], [0, 1.0, 0]])
print(f"(c) G_Z = (P - Y) / 2 = {show((P - Y) / 2)}")
z1, z2 = 0.3, -1.1
print(f"(d) softmax([z1, z2])[1] {softmax(np.array([z1, z2]))[1]:.6f}  sigmoid(z2 - z1) {1 / (1 + np.exp(-(z2 - z1))):.6f}")

# %%
section("(a) cross-entropy with soft targets")


def cross_entropy_soft(Z: FloatArray, T: FloatArray) -> float:
    """Mean of -sum_k T_ik log p_ik, from logits. Z (n, K), T (n, K) rows summing to 1 -> scalar."""
    log_p = Z - logsumexp(Z)[:, None]  # (n, K) log-softmax
    return float(-np.mean(np.sum(T * log_p, axis=1)))


def cross_entropy_soft_grad(Z: FloatArray, T: FloatArray) -> FloatArray:
    """Gradient (P - T) / n, valid because each row of T sums to 1. -> (n, K)."""
    return (softmax(Z) - T) / Z.shape[0]


rng = np.random.default_rng(SEED)
n, K, eps = 4, 5, 0.1
Z = rng.normal(0, 2, size=(n, K))
y = rng.integers(0, K, size=n)
T = np.full((n, K), eps / (K - 1))
T[np.arange(n), y] = 1 - eps  # label smoothing
err = rel_error(cross_entropy_soft_grad(Z, T), numerical_gradient(lambda A: cross_entropy_soft(A, T), Z.copy()))
print(f"label-smoothed targets, rows sum to {show(T.sum(axis=1))}")
print(f"gradient (P - T)/n vs numeric: rel error {'< 1e-8' if err < 1e-8 else f'{err:.1e}'}  {'PASS' if err < 1e-6 else 'FAIL'}")
torch_soft = torch.nn.functional.cross_entropy(torch.tensor(Z), torch.tensor(T)).item()
print(f"loss {cross_entropy_soft(Z, T):.6f}   torch cross_entropy with probability targets {torch_soft:.6f}")
smoothed = torch.nn.functional.cross_entropy(torch.tensor(Z), torch.tensor(y), label_smoothing=eps).item()
T_torch = np.full((n, K), eps / K)
T_torch[np.arange(n), y] += 1 - eps
print(f"note: torch label_smoothing={eps} spreads eps over all K classes: {smoothed:.6f} = ours with that T {cross_entropy_soft(Z, T_torch):.6f}")

# %%
section("(b) binary cross-entropy with logits")


def bce_with_logits(z: FloatArray, y: FloatArray) -> FloatArray:
    """Per-example log(1 + e^z) - y z, stable: max(z, 0) - y z + log1p(e^-|z|). z, y (n,) -> (n,)."""
    return np.maximum(z, 0) - y * z + np.log1p(np.exp(-np.abs(z)))


zz = np.array([-1000.0, -5.0, 0.0, 5.0, 1000.0], dtype=np.float32)
yy = np.array([1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
ours = bce_with_logits(zz, yy)
theirs = torch.nn.functional.binary_cross_entropy_with_logits(
    torch.tensor(zz), torch.tensor(yy), reduction="none"
).numpy()
print(f"float32 z {zz.tolist()}  y {yy.tolist()}")
print(f"ours  {show(ours.astype(np.float64))}  dtype {ours.dtype}")
print(f"torch {show(theirs.astype(np.float64))}")
np.testing.assert_allclose(ours, theirs, rtol=1e-6, atol=1e-6)
with np.errstate(over="ignore", divide="ignore"):
    naive = -(yy * np.log(1 / (1 + np.exp(-zz))) + (1 - yy) * np.log(1 - 1 / (1 + np.exp(-zz))))
print(f"naive {naive.tolist()}")
z64 = rng.normal(size=6)
y64 = rng.integers(0, 2, size=6).astype(np.float64)
gb = (1 / (1 + np.exp(-z64)) - y64) / 6
err_b = rel_error(gb, numerical_gradient(lambda v: float(np.mean(bce_with_logits(v, y64))), z64.copy()))
print(f"gradient (sigmoid(z) - y)/n vs numeric: rel error {'< 1e-8' if err_b < 1e-8 else f'{err_b:.1e}'}  {'PASS' if err_b < 1e-6 else 'FAIL'}")

# %%
section("(c) softmax vector-Jacobian product")


def softmax_vjp(p: FloatArray, g: FloatArray) -> FloatArray:
    """g^T J for J = diag(p) - p p^T, without forming J: p * (g - g.p). p, g (K,) -> (K,)."""
    return p * (g - g @ p)


pv = softmax(rng.normal(size=6))
gv = rng.normal(size=6)
J = np.diag(pv) - np.outer(pv, pv)  # (K, K), symmetric
print(f"vjp     {show(softmax_vjp(pv, gv))}")
print(f"J.T @ g {show(J.T @ gv)}")
np.testing.assert_allclose(softmax_vjp(pv, gv), J.T @ gv, rtol=1e-12, atol=1e-15)
print("matches the explicit Jacobian: OK  (O(K) work and memory instead of O(K^2))")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
