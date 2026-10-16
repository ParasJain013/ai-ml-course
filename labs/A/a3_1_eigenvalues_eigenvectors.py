# %% [markdown]
# # A3.1 Eigenvalues and eigenvectors — lab
# Lesson: lessons/A/a3_1_eigenvalues_eigenvectors.md
# Run: uv run python labs/A/a3_1_eigenvalues_eigenvectors.py   (CPU, defaults: ~3 s)
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
OUT = ROOT / "outputs" / "a3_1"  # scratch figures, gitignored
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
# Nothing here needs a GPU; the knob sizes the random symmetric matrix used to test power iteration.
GPU_SCALE = False
D_RANDOM = 2_000 if GPU_SCALE else 6  # size of the random symmetric test matrix
TOL = 1e-12  # power iteration stops when 1 - |cos(angle between steps)| < TOL
MAX_ITERS = 10_000

# %%
# Data: the two 2x2 matrices of the worked example.
section("data")
A = np.array([[2.0, 1.0], [1.0, 2.0]])  # (2, 2), symmetric
B = np.array([[4.0, 1.0], [2.0, 3.0]])  # (2, 2), not symmetric
print("A =", show(A), "  symmetric:", np.allclose(A, A.T))
print("B =", show(B), "  symmetric:", np.allclose(B, B.T))

# %%
# Worked example by hand, part 1: eigenvalues from the characteristic polynomial.
section("worked example: characteristic polynomial")


def eig_2x2(M: FloatArray) -> FloatArray:
    """Real eigenvalues of a 2x2 matrix from lambda^2 - trace*lambda + det = 0. M (2, 2) -> (2,)."""
    tr = M[0, 0] + M[1, 1]
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    disc = tr**2 - 4 * det
    if disc < 0:
        raise ValueError(f"complex eigenvalues: discriminant {disc:.4f} < 0")
    root = np.sqrt(disc)
    return np.array([(tr + root) / 2, (tr - root) / 2])


for name, M in (("A", A), ("B", B)):
    tr, det = np.trace(M), np.linalg.det(M)
    print(f"{name}: trace = {tr:.4f}  det = {det:.4f}  eigenvalues = {show(eig_2x2(M))}")

v1, v2 = np.array([1.0, 1.0]), np.array([1.0, -1.0])
print("A @ (1, 1)  =", show(A @ v1), " = 3 * (1, 1)")
print("A @ (1, -1) =", show(A @ v2), " = 1 * (1, -1)")
print("A's eigenvectors are orthogonal: (1, 1) . (1, -1) =", v1 @ v2)
u1, u2 = np.array([1.0, 1.0]), np.array([1.0, -2.0])
print("B @ (1, 1)  =", show(B @ u1), " = 5 * (1, 1)")
print("B @ (1, -2) =", show(B @ u2), " = 2 * (1, -2)")
print("B's eigenvectors are not orthogonal: (1, 1) . (1, -2) =", u1 @ u2)
x = np.array([1.0, 0.0])
print("an ordinary vector turns: A @ (1, 0) =", show(A @ x), " (not a multiple of (1, 0))")

# %%
# Worked example by hand, part 2: four steps of power iteration on A from x0 = (1, 0).
section("worked example: power iteration by hand")
x = np.array([1.0, 0.0])
v_top = np.array([1.0, 1.0]) / np.sqrt(2.0)
for k in range(1, 5):
    x_raw = A @ x  # unnormalized, to match the hand computation
    rayleigh = (x_raw @ A @ x_raw) / (x_raw @ x_raw)
    angle = np.degrees(np.arccos(abs(x_raw @ v_top) / np.linalg.norm(x_raw)))
    print(f"A^{k} x0 = {show(x_raw):<14} Rayleigh quotient = {rayleigh:.4f}  angle to (1, 1) = {angle:.4f} deg")
    x = x_raw  # keep the raw integers; the code version normalizes every step

# %%
# From scratch: power iteration with a convergence test that tolerates sign flips.
section("from scratch: power iteration")


def power_iteration(
    M: FloatArray, tol: float = TOL, max_iters: int = MAX_ITERS, seed: int = SEED
) -> tuple[float, FloatArray, int]:
    """Dominant eigenpair of a square matrix. M (d, d) -> (eigenvalue, unit eigenvector (d,), iterations)."""
    d = M.shape[0]
    x = np.random.default_rng(seed).normal(size=d)  # (d,), random start: almost surely has a v1 component
    x /= np.linalg.norm(x)
    for k in range(1, max_iters + 1):
        y = M @ x  # (d,)
        y /= np.linalg.norm(y)
        # |cos| near 1 means the direction stopped changing; abs() allows a negative eigenvalue,
        # which flips the vector's sign every step without changing its line.
        if 1.0 - abs(y @ x) < tol:
            x = y
            break
        x = y
    else:
        raise RuntimeError(f"power iteration did not converge in {max_iters} iterations")
    lam = float(x @ M @ x)  # Rayleigh quotient (x has unit length)
    return lam, x, k


lam, v, iters = power_iteration(A)
print(f"A: lambda = {lam:.6f}  v = {show(v)}  iterations = {iters}")
lam_b, v_b, iters_b = power_iteration(B)
print(f"B: lambda = {lam_b:.6f}  v = {show(v_b)}  iterations = {iters_b}   (v is (1, 1)/sqrt(2) = 0.7071)")

G = rng.normal(size=(D_RANDOM, D_RANDOM))
S = (G + G.T) / 2  # (d, d) random symmetric matrix
lam_s, v_s, iters_s = power_iteration(S)
residual = np.linalg.norm(S @ v_s - lam_s * v_s)
print(f"random symmetric {S.shape}: lambda = {lam_s:.6f}  iterations = {iters_s}  |S v - lambda v| < 1e-5: {residual < 1e-5}")

# %%
# Convergence speed is set by |lambda_2 / lambda_1|.
section("from scratch: convergence speed")
for ratio in (0.1, 0.5, 0.9, 0.99):
    D = np.diag([1.0, ratio])  # eigenvalues 1 and ratio, eigenvectors e1 and e2
    _, _, it = power_iteration(D, tol=1e-10)
    print(f"|lambda2/lambda1| = {ratio:<5} iterations to converge = {it}")

# %%
# A Markov chain: the long-run share of visitors on each page is an eigenvector with eigenvalue 1.
section("from scratch: steady state of a Markov chain")
P = np.array(
    [
        [0.5, 0.4, 0.1],  # from Home: stay, go to Search, go to Checkout
        [0.3, 0.4, 0.3],  # from Search
        [0.2, 0.3, 0.5],  # from Checkout
    ]
)  # (3, 3), row i = probabilities of the next page given page i; rows sum to 1
share = np.array([1.0, 0.0, 0.0])  # everyone starts on Home
for _ in range(100):
    share = share @ P  # (3,) one step: new share = old share times the transition matrix
print("share after 100 steps (Home, Search, Checkout) =", show(share))
print("P^T @ share == share:", np.allclose(P.T @ share, share), "-> eigenvector of P^T with eigenvalue 1")

# %%
# With NumPy: eig (any square matrix) and eigh (symmetric), and how to read their outputs.
section("with numpy")
vals, vecs = np.linalg.eig(A)  # eigenvectors are the COLUMNS of vecs, order not guaranteed
print("np.linalg.eig(A) returns dtype", vals.dtype, "even though the eigenvalues are real")
assert np.allclose(vals.imag, 0) and np.allclose(vecs.imag, 0)
vals, vecs = vals.real, vecs.real  # safe only after checking the imaginary parts are zero
print("np.linalg.eig(A):  values =", show(vals), "  vectors (columns) =", show(vecs))
vals_h, vecs_h = np.linalg.eigh(A)  # symmetric only; values in ascending order
print("np.linalg.eigh(A): values =", show(vals_h), "  vectors (columns) =", show(vecs_h))
top = np.argmax(np.abs(vals_h))
v_np = vecs_h[:, top]
v_mine = v * np.sign(v @ v_np)  # eigenvectors are only defined up to sign
np.testing.assert_allclose(lam, vals_h[top], rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(v_mine, v_np, rtol=1e-6, atol=1e-8)
print("power iteration matches np.linalg.eigh on A: OK")

vals_s, vecs_s = np.linalg.eigh(S)
top_s = np.argmax(np.abs(vals_s))
np.testing.assert_allclose(lam_s, vals_s[top_s], rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(v_s * np.sign(v_s @ vecs_s[:, top_s]), vecs_s[:, top_s], rtol=1e-5, atol=1e-6)
# rtol 1e-5 on the vector: the stopping test bounds 1 - |cos|, so the angle error is about sqrt(2 * TOL)
print(f"power iteration matches np.linalg.eigh on the random {S.shape} matrix: OK")
print("eigenvalues of the random matrix (eigh, ascending) =", show(vals_s))

# Reconstruct both matrices from their eigendecompositions.
Lam = np.diag(vals_h)
print("A == Q diag(lambda) Q^T:", np.allclose(vecs_h @ Lam @ vecs_h.T, A), "  Q^T Q == I:", np.allclose(vecs_h.T @ vecs_h, np.eye(2)))
vals_b, vecs_b = np.linalg.eig(B)
vals_b, vecs_b = vals_b.real, vecs_b.real  # B's eigenvalues are real (5 and 2): checked by hand above
print("B: values =", show(vals_b), "  B == V diag(lambda) V^-1:", np.allclose(vecs_b @ np.diag(vals_b) @ np.linalg.inv(vecs_b), B))
print("trace = sum of eigenvalues:", np.isclose(np.trace(B), vals_b.sum()), "  det = product:", np.isclose(np.linalg.det(B), vals_b.prod()))

theta = np.pi / 2
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  # 90-degree rotation
print("a 90-degree rotation has no real eigenvector: eigenvalues =", np.round(np.linalg.eigvals(R), 4))

vals_p, vecs_p = np.linalg.eig(P.T)
k1 = np.argmin(np.abs(vals_p - 1.0))
pi = np.real(vecs_p[:, k1])
pi /= pi.sum()  # eig returns unit length; a probability vector must sum to 1
np.testing.assert_allclose(pi, share, rtol=1e-6, atol=1e-8)
print("Markov steady state from np.linalg.eig(P.T), rescaled to sum 1 =", show(pi), " matches: OK")

# %%
# Key plot: what each matrix does to the unit circle, with its eigenvectors.
section("key plot")
t = np.linspace(0.0, 2 * np.pi, 400)
circle = np.stack([np.cos(t), np.sin(t)], axis=1)  # (400, 2), points on the unit circle as rows
fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=120)
for ax, M, name in ((axes[0], A, "A = [[2, 1], [1, 2]] (symmetric)"), (axes[1], B, "B = [[4, 1], [2, 3]]")):
    image = circle @ M.T  # (400, 2): apply M to every point (rows convention, A2.1)
    ax.plot(circle[:, 0], circle[:, 1], "--", color="0.5", lw=1, label="unit circle")
    ax.plot(image[:, 0], image[:, 1], "-", color="tab:blue", lw=2, label="its image under the matrix")
    w, V = np.linalg.eig(M)
    w, V = w.real, V.real  # both matrices have real eigenvalues (worked example)
    for j, (style, color) in enumerate((("-", "tab:red"), ("-", "tab:green"))):
        vj = V[:, j] / np.linalg.norm(V[:, j])
        ax.annotate("", xy=vj, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=1.5, ls="--"))
        ax.annotate("", xy=w[j] * vj, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
        ax.plot([], [], style, color=color, lw=2.5, label=f"eigenvector, stretched by λ = {w[j]:.0f}")
    e1 = np.array([1.0, 0.0])
    ax.annotate("", xy=M @ e1, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="0.2", lw=1.2))
    ax.plot([], [], "-", color="0.2", lw=1.2, label=f"(1, 0) lands on {show(M @ e1)}: turned")
    lim = 5.5 if M is B else 3.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("x₁ (unitless)")
    ax.set_ylabel("x₂ (unitless)")
    ax.set_title(name, fontsize=10)
    ax.legend(loc="lower right", fontsize=7)
fig.suptitle("Eigenvectors keep their direction; every other vector is turned")
fig.tight_layout()
path = IMG / "a3_1_unit_circle_eigenvectors.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Set B = [[1, 1], [0, 1]] (a shear). How many independent eigenvectors does np.linalg.eig report?
# 2. Run power_iteration on np.diag([2.0, -2.0]). Why does it never converge, whatever TOL is?
# 3. Change P so that Checkout always returns to Home ([1, 0, 0]). How does the steady state move?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
