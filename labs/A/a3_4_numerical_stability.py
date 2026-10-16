# %% [markdown]
# # A3.4 Numerical stability — lab
# Lesson: lessons/A/a3_4_numerical_stability.md
# Run: uv run python labs/A/a3_4_numerical_stability.py   (CPU, defaults: ~3 s)
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
import scipy.linalg

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
OUT = ROOT / "outputs" / "a3_4"  # scratch figures, gitignored
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
# Nothing here needs a GPU; the knobs size the timing test and the range of Hilbert matrices.
GPU_SCALE = False
N_TIMING = 4_000 if GPU_SCALE else 800  # size of the random system used to time inv vs solve
HILBERT_SIZES = list(range(2, 14))  # Hilbert matrices from 2x2 (well conditioned) to 13x13 (hopeless)
EPS = np.finfo(np.float64).eps  # 2.22e-16: the gap between 1.0 and the next float64

# %%
# Data: Hilbert matrices H[i, j] = 1 / (i + j + 1) (0-based), a classic ill-conditioned family.
section("data")
print(f"float64 machine epsilon = {EPS:.4e}  (about {-np.log10(EPS):.1f} significant digits)")
print("1.0 + 1e-16 == 1.0:", 1.0 + 1e-16 == 1.0, "  0.1 + 0.2 == 0.3:", 0.1 + 0.2 == 0.3)
H4 = scipy.linalg.hilbert(4)  # (4, 4)
print("Hilbert(4) =", show(H4))

# %%
# Worked example by hand, part 1: a nearly singular 2x2 system and its condition number.
section("worked example: an ill-conditioned 2x2 system")
A = np.array([[1.0, 1.0], [1.0, 1.0001]])  # (2, 2): the columns are almost the same direction
b = np.array([2.0, 2.0001])
b_nudged = np.array([2.0, 2.0002])  # change one entry by 0.0001
x = np.linalg.solve(A, b)
x_nudged = np.linalg.solve(A, b_nudged)
s = np.linalg.svd(A, compute_uv=False)
print("x for b =", show(b), "->", show(x))
print("x for b' =", show(b_nudged), "->", show(x_nudged))
rel_in = np.linalg.norm(b_nudged - b) / np.linalg.norm(b)
rel_out = np.linalg.norm(x_nudged - x) / np.linalg.norm(x)
print(f"relative change in b: {rel_in:.3e}   relative change in x: {rel_out:.4f}   amplification: {rel_out / rel_in:.0f}")
print(f"singular values = {s[0]:.4f}, {s[1]:.4e}   condition number = {s[0] / s[1]:.0f}   np.linalg.cond = {np.linalg.cond(A):.0f}")

# %%
# Worked example by hand, part 2: QR by Gram-Schmidt on a 2x2 matrix.
section("worked example: Gram-Schmidt")
A2 = np.array([[3.0, 1.0], [4.0, 2.0]])  # columns a1 = (3, 4), a2 = (1, 2)
a1, a2 = A2[:, 0], A2[:, 1]
r11 = np.linalg.norm(a1)
q1 = a1 / r11
r12 = q1 @ a2
a2_perp = a2 - r12 * q1  # remove the part of a2 along q1 (projection, A2.4)
r22 = np.linalg.norm(a2_perp)
q2 = a2_perp / r22
print(f"r11 = |a1| = {r11:.4f}   q1 = {show(q1)}")
print(f"r12 = q1 . a2 = {r12:.4f}   a2 - r12 q1 = {show(a2_perp)}   r22 = {r22:.4f}   q2 = {show(q2)}")
Q2 = np.column_stack([q1, q2])
R2 = np.array([[r11, r12], [0.0, r22]])
print("Q =", show(Q2), "  R =", show(R2))
print("Q R == A:", np.allclose(Q2 @ R2, A2), "  Q^T Q == I:", np.allclose(Q2.T @ Q2, np.eye(2)))

# %%
# Worked example by hand, part 3: Cholesky of a 3x3 symmetric positive-definite matrix.
section("worked example: Cholesky")
C = np.array([[4.0, 2.0, 2.0], [2.0, 5.0, 3.0], [2.0, 3.0, 6.0]])  # (3, 3)
L_hand = np.array([[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [1.0, 1.0, 2.0]])
print("C =", show(C))
print("L (by hand) =", show(L_hand), "  L L^T == C:", np.allclose(L_hand @ L_hand.T, C))

# %%
# From scratch: Gram-Schmidt QR (classical and modified), Cholesky, and triangular solves.
section("from scratch: QR, Cholesky, triangular solves")


def qr_classical(A: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Thin QR by classical Gram-Schmidt. A (n, d), n >= d, full column rank -> Q (n, d), R (d, d)."""
    n, d = A.shape
    Q, R = np.zeros((n, d)), np.zeros((d, d))
    for j in range(d):
        R[:j, j] = Q[:, :j].T @ A[:, j]  # (j,): coefficients along the q's found so far, from the ORIGINAL column
        v = A[:, j] - Q[:, :j] @ R[:j, j]  # (n,): what is left after removing those parts
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R


def qr_modified(A: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Thin QR by modified Gram-Schmidt. A (n, d), n >= d, full column rank -> Q (n, d), R (d, d)."""
    n, d = A.shape
    V = A.copy()  # (n, d): columns are updated in place as each q is removed
    Q, R = np.zeros((n, d)), np.zeros((d, d))
    for j in range(d):
        R[j, j] = np.linalg.norm(V[:, j])
        Q[:, j] = V[:, j] / R[j, j]
        R[j, j + 1 :] = Q[:, j] @ V[:, j + 1 :]  # coefficients from the UPDATED columns
        V[:, j + 1 :] -= np.outer(Q[:, j], R[j, j + 1 :])  # remove q_j from every later column now
    return Q, R


def cholesky(C: FloatArray) -> FloatArray:
    """Lower-triangular L with L L^T = C. C (d, d) symmetric positive definite -> L (d, d)."""
    d = C.shape[0]
    L = np.zeros((d, d))
    for j in range(d):
        diag = C[j, j] - L[j, :j] @ L[j, :j]
        if diag <= 0:
            raise np.linalg.LinAlgError(f"not positive definite (pivot {j} is {diag:.3g})")
        L[j, j] = np.sqrt(diag)
        L[j + 1 :, j] = (C[j + 1 :, j] - L[j + 1 :, :j] @ L[j, :j]) / L[j, j]
    return L


def forward_substitution(L: FloatArray, b: FloatArray) -> FloatArray:
    """Solve L y = b for lower-triangular L, first equation down. L (d, d), b (d,) -> (d,)."""
    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


# from labs/A/a2_3_linear_systems_rank.py (rewritten with slices)
def back_substitution(U: FloatArray, y: FloatArray) -> FloatArray:
    """Solve U x = y for upper-triangular U, last equation up. U (d, d), y (d,) -> (d,)."""
    x = np.zeros_like(y)
    for i in range(len(y) - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]
    return x


Q_m, R_m = qr_modified(A2)
print("modified Gram-Schmidt on the worked example: Q =", show(Q_m), " R =", show(R_m))
L = cholesky(C)
print("cholesky(C) =", show(L))
b3 = np.array([8.0, 10.0, 11.0])
y3 = forward_substitution(L, b3)  # L y = b
x3 = back_substitution(L.T, y3)  # L^T x = y
print("solve C x = b via L y = b, then L^T x = y:  y =", show(y3), "  x =", show(x3), "  C x == b:", np.allclose(C @ x3, b3))
try:
    cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))
except np.linalg.LinAlgError as err:
    print("cholesky([[1, 2], [2, 1]]) ->", err, " (eigenvalues 3 and -1)")

# %%
# Losing orthogonality: classical vs modified Gram-Schmidt vs NumPy on an ill-conditioned matrix.
# The exact small numbers depend on the CPU and BLAS build, so these lines start with "~".
section("from scratch: loss of orthogonality")
H = scipy.linalg.hilbert(10)
print(f"Hilbert(10): condition number about 10^{np.log10(np.linalg.cond(H)):.0f}")
for name, qr in (("classical Gram-Schmidt", qr_classical), ("modified Gram-Schmidt", qr_modified), ("np.linalg.qr (Householder)", np.linalg.qr)):
    Q, R = qr(H)
    print(f"~ {name:<27} |Q^T Q - I| = {np.abs(Q.T @ Q - np.eye(10)).max():.1e}   |Q R - H| = {np.abs(Q @ R - H).max():.1e}")

# %%
# inv vs solve vs lstsq on Hilbert systems with a known answer x = (1, ..., 1).
section("solving ill-conditioned systems: inv vs solve vs lstsq")
rows = []
for n in HILBERT_SIZES:
    Hn = scipy.linalg.hilbert(n)
    x_true = np.ones(n)
    bn = Hn @ x_true
    cond = np.linalg.cond(Hn)
    answers = {
        "inv": np.linalg.inv(Hn) @ bn,
        "solve": np.linalg.solve(Hn, bn),
        "lstsq": np.linalg.lstsq(Hn, bn, rcond=None)[0],
    }
    err = {k: np.linalg.norm(v - x_true) / np.linalg.norm(x_true) for k, v in answers.items()}
    res = {k: np.linalg.norm(Hn @ v - bn) / np.linalg.norm(bn) for k, v in answers.items()}
    rows.append((n, cond, err, res))
print(f"~ {'n':>2} {'cond':>8}  {'error inv':>9} {'error solve':>11} {'error lstsq':>11}   {'resid inv':>9} {'resid solve':>11}")
for n, cond, err, res in rows:
    print(f"~ {n:>2} {cond:8.1e}  {err['inv']:9.1e} {err['solve']:11.1e} {err['lstsq']:11.1e}   {res['inv']:9.1e} {res['solve']:11.1e}")
n12 = [r for r in rows if r[0] == 12][0]
print("at n = 12, solve's error is within 100 * eps * cond:", bool(n12[2]["solve"] < 100 * EPS * n12[1]))
print("at n = 12, solve's residual is below inv's:", bool(n12[3]["solve"] < n12[3]["inv"]))

# %%
# Least squares three ways on an ill-conditioned polynomial fit: normal equations, QR, lstsq.
section("least squares: normal equations vs QR vs lstsq")
x_poly = np.linspace(0.0, 1.0, 50)
DEGREE = 9
X_poly = np.column_stack([x_poly**k for k in range(DEGREE + 1)])  # (50, 10): columns 1, x, ..., x^9
w_true = np.ones(DEGREE + 1)
y_poly = X_poly @ w_true  # noiseless, so the exact answer is w_true
cond_X = np.linalg.cond(X_poly)
print(f"X (50, 10): condition number about 10^{np.log10(cond_X):.0f};  X^T X: about 10^{np.log10(np.linalg.cond(X_poly.T @ X_poly)):.0f}")
w_normal = np.linalg.solve(X_poly.T @ X_poly, X_poly.T @ y_poly)
Q_p, R_p = qr_modified(X_poly)
w_qr = back_substitution(R_p, Q_p.T @ y_poly)  # R w = Q^T y
Q_np, R_np = np.linalg.qr(X_poly)
w_qr_np = scipy.linalg.solve_triangular(R_np, Q_np.T @ y_poly)
w_lstsq = np.linalg.lstsq(X_poly, y_poly, rcond=None)[0]
for name, w in (("normal equations", w_normal), ("QR (modified G-S)", w_qr), ("QR (np.linalg.qr)", w_qr_np), ("np.linalg.lstsq", w_lstsq)):
    print(f"~ {name:<18} relative error in w = {np.linalg.norm(w - w_true) / np.linalg.norm(w_true):.1e}")
print("QR error < normal-equations error:", bool(np.linalg.norm(w_qr_np - w_true) < np.linalg.norm(w_normal - w_true)))

# %%
# With NumPy and SciPy: the library versions agree with ours.
section("with numpy and scipy")
Q_np2, R_np2 = np.linalg.qr(A2)
signs = np.sign(np.diag(R_np2))  # LAPACK may return negative diagonal entries in R; flip to match
np.testing.assert_allclose(Q_np2 * signs, Q_m, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(R_np2 * signs[:, None], R_m, rtol=1e-6, atol=1e-8)
print("np.linalg.qr R (raw) =", show(R_np2), "-> after flipping signs, Q and R match ours: OK")
np.testing.assert_allclose(np.linalg.cholesky(C), L, rtol=1e-6, atol=1e-8)
print("np.linalg.cholesky(C) matches: OK")
c_and_lower = scipy.linalg.cho_factor(C)
np.testing.assert_allclose(scipy.linalg.cho_solve(c_and_lower, b3), x3, rtol=1e-6, atol=1e-8)
print("scipy.linalg.cho_solve gives the same x: OK")
try:
    np.linalg.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))
except np.linalg.LinAlgError as err:
    print("np.linalg.cholesky([[1, 2], [2, 1]]) -> LinAlgError:", err)

M = rng.normal(size=(N_TIMING, N_TIMING))
v = rng.normal(size=N_TIMING)
t = time.perf_counter()
x_inv = np.linalg.inv(M) @ v
t_inv = time.perf_counter() - t
t = time.perf_counter()
x_solve = np.linalg.solve(M, v)
t_solve = time.perf_counter() - t
print(f"~ random {N_TIMING}x{N_TIMING} system: inv @ b {t_inv * 1000:.0f} ms, solve {t_solve * 1000:.0f} ms")

# %%
# Key plot: relative error against the condition number for the Hilbert systems.
section("key plot")
conds = np.array([r[1] for r in rows])
fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=120)
ax.loglog(conds, [r[2]["inv"] for r in rows], "s--", ms=5, label="inv(A) @ b")
ax.loglog(conds, [r[2]["solve"] for r in rows], "o-", ms=5, label="solve(A, b)")
ax.loglog(conds, [r[2]["lstsq"] for r in rows], "^:", ms=5, label="lstsq(A, b)")
ax.loglog(conds, EPS * conds, "-", color="0.4", lw=1, label="ε × cond(A)")
for n, c, err, _ in rows[::2]:
    ax.annotate(f"n={n}", (c, err["solve"]), textcoords="offset points", xytext=(4, -10), fontsize=7)
ax.set_xlabel("condition number of the Hilbert matrix (log scale)")
ax.set_ylabel("relative error in x (log scale)")
ax.set_title("The error grows in step with cond(A); inv(A) @ b is the worst", fontsize=10)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
path = IMG / "a3_4_error_vs_condition.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Rerun the polynomial fit with DEGREE = 5 and with DEGREE = 12. When do the normal equations break?
# 2. Rescale x_poly to [-1, 1] instead of [0, 1]. What happens to cond(X)? (Scaling features is a cheap fix.)
# 3. Compute np.linalg.cond(H4.astype(np.float32)) vs float64, and solve a Hilbert(8) system in float32.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
