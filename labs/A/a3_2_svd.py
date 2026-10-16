# %% [markdown]
# # A3.2 Singular value decomposition — lab
# Lesson: lessons/A/a3_2_svd.md
# Run: uv run python labs/A/a3_2_svd.py   (CPU, defaults: ~10 s, mostly importing scikit-learn)
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
OUT = ROOT / "outputs" / "a3_2"  # scratch figures, gitignored
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
# Nothing here needs a GPU; the knobs size the random test matrix and the ranks tried on the image.
GPU_SCALE = False
N_ROWS, N_COLS = (2_000, 500) if GPU_SCALE else (6, 4)  # random matrix for the from-scratch SVD
RANKS = (5, 20, 50)  # ranks shown in the image-compression example
TOL = 1e-14  # power-iteration stopping tolerance (1 - |cos| between steps)

# %%
# Data: the worked-example matrix, a random rectangular matrix, and a grayscale photo.
section("data")
from sklearn.datasets import load_sample_image  # noqa: E402  (bundled with scikit-learn, no download)

A = np.array([[3.0, 0.0], [4.0, 5.0]])  # (2, 2)
M = rng.normal(size=(N_ROWS, N_COLS))  # (n, d)
photo = load_sample_image("china.jpg")  # (427, 640, 3) uint8, RGB
gray = photo.mean(axis=2) / 255.0  # (427, 640) floats in [0, 1]; a plain average of the 3 channels
print("A =", show(A))
print("random matrix M:", M.shape)
print("photo:", photo.shape, photo.dtype, "-> grayscale", gray.shape, f"values in [{gray.min():.3f}, {gray.max():.3f}]")

# %%
# Worked example by hand: the SVD of A from the eigendecomposition of A^T A.
section("worked example: SVD by hand")
AtA = A.T @ A  # (2, 2), symmetric
print("A^T A =", show(AtA))
evals, evecs = np.linalg.eigh(AtA)  # ascending
order = np.argsort(evals)[::-1]  # largest first
evals, V = evals[order], evecs[:, order]
V = V * np.sign(V[0])  # make the first entry of each column positive, to match the hand signs
sigma = np.sqrt(evals)
print("eigenvalues of A^T A =", show(evals), " singular values = sqrt =", show(sigma))
print("sigma_1 = 3 sqrt(5) =", round(3 * np.sqrt(5), 4), "  sigma_2 = sqrt(5) =", round(np.sqrt(5), 4))
print("V (columns v1, v2) =", show(V))
U = (A @ V) / sigma  # (2, 2): column j is A v_j / sigma_j
print("U = A V / sigma =", show(U), "  (1, 3)/sqrt(10) =", show(np.array([1, 3]) / np.sqrt(10)))
print("U^T U == I:", np.allclose(U.T @ U, np.eye(2)), "  V^T V == I:", np.allclose(V.T @ V, np.eye(2)))
print("U diag(sigma) V^T == A:", np.allclose(U @ np.diag(sigma) @ V.T, A))
print("sigma_1 * sigma_2 =", round(sigma.prod(), 4), " = |det A| =", round(abs(np.linalg.det(A)), 4))
A1 = sigma[0] * np.outer(U[:, 0], V[:, 0])  # best rank-1 approximation
err = A - A1
print("rank-1 approximation sigma_1 u1 v1^T =", show(A1))
print("A - A1 =", show(err), "  squared Frobenius error =", round((err**2).sum(), 4), " = sigma_2^2 =", round(sigma[1] ** 2, 4))

# %%
# From scratch: SVD by power iteration on A^T A, then deflation.
section("from scratch: SVD by power iteration and deflation")


# from labs/A/a3_1_eigenvalues_eigenvectors.py
def power_iteration(
    M: FloatArray, tol: float = TOL, max_iters: int = 100_000, seed: int = SEED
) -> tuple[float, FloatArray, int]:
    """Dominant eigenpair of a square matrix. M (d, d) -> (eigenvalue, unit eigenvector (d,), iterations)."""
    d = M.shape[0]
    x = np.random.default_rng(seed).normal(size=d)  # (d,), random start
    x /= np.linalg.norm(x)
    for k in range(1, max_iters + 1):
        y = M @ x  # (d,)
        y /= np.linalg.norm(y)
        if 1.0 - abs(y @ x) < tol:
            x = y
            break
        x = y
    else:
        raise RuntimeError(f"power iteration did not converge in {max_iters} iterations")
    lam = float(x @ M @ x)  # Rayleigh quotient (x has unit length)
    return lam, x, k


def svd_power(A: FloatArray, k: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Top-k SVD by power iteration and deflation. A (n, d) -> U (n, k), s (k,), V (d, k)."""
    n, d = A.shape
    R = A.copy()  # (n, d), what is left after removing the components found so far
    U, s, V = np.zeros((n, k)), np.zeros(k), np.zeros((d, k))
    for j in range(k):
        _, v, iters = power_iteration(R.T @ R)  # top right singular vector of what is left, (d,)
        Rv = R @ v  # (n,)
        s[j] = np.linalg.norm(Rv)  # sigma_j = |R v_j|
        U[:, j] = Rv / s[j]  # u_j = R v_j / sigma_j
        V[:, j] = v
        R = R - s[j] * np.outer(U[:, j], v)  # deflate: remove the rank-1 layer just found
        print(f"  component {j + 1}: sigma = {s[j]:.6f}  power iterations = {iters}")
    return U, s, V


k = min(M.shape)
U_mine, s_mine, V_mine = svd_power(M, k)
print("U", U_mine.shape, " s", s_mine.shape, " V", V_mine.shape)
recon = U_mine @ np.diag(s_mine) @ V_mine.T
print("U diag(s) V^T == M:", np.allclose(recon, M), "  max error < 1e-12:", bool(np.abs(recon - M).max() < 1e-12))

# %%
# With NumPy: np.linalg.svd, the shapes it returns, and the comparison (allowing for sign flips).
section("with numpy")
U_full, s_np, Vt_full = np.linalg.svd(M)  # default full_matrices=True
print("np.linalg.svd(M) default:             U", U_full.shape, " s", s_np.shape, " Vt", Vt_full.shape)
U_np, s_np, Vt_np = np.linalg.svd(M, full_matrices=False)  # thin SVD: what you almost always want
print("np.linalg.svd(M, full_matrices=False): U", U_np.shape, " s", s_np.shape, " Vt", Vt_np.shape)
print("numpy returns V^T (rows are the v_j), not V; singular values sorted descending:", show(s_np))
signs = np.sign(np.sum(U_mine * U_np, axis=0))  # +1 or -1 per component
np.testing.assert_allclose(s_mine, s_np, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(U_mine * signs, U_np, rtol=1e-6, atol=1e-6)
np.testing.assert_allclose(V_mine * signs, Vt_np.T, rtol=1e-6, atol=1e-6)
# atol 1e-6 on the vectors: power iteration stops at 1 - |cos| < 1e-14, an angle of about 1.4e-7,
# and deflation passes each component's error on to the next one.
print("matches np.linalg.svd (after aligning signs): OK")

# %%
# Rank from singular values, and least squares with the pseudo-inverse.
section("with numpy: rank and least squares via the SVD")
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
group = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])  # one-hot for a two-level category
X = np.column_stack([np.ones(6), x, group, 1.0 - group])  # (6, 4): ones = group + (1 - group), dependent
y = np.array([1.0, 2.1, 2.9, 5.2, 6.1, 6.8])  # (6,)
U_x, s_x, Vt_x = np.linalg.svd(X, full_matrices=False)
tol = s_x.max() * max(X.shape) * np.finfo(float).eps  # the cutoff np.linalg.matrix_rank uses
rank = int((s_x > tol).sum())
print("singular values of X =", show(s_x), f"  cutoff {tol:.1e} -> rank {rank} (matrix_rank says {np.linalg.matrix_rank(X)})")
print("cond(X^T X) > 1e15:", bool(np.linalg.cond(X.T @ X) > 1e15), "-> the normal equations are singular")
s_inv = np.where(s_x > tol, 1.0 / s_x, 0.0)  # invert only the non-zero singular values
w_svd = Vt_x.T @ (s_inv * (U_x.T @ y))  # w = V Sigma^+ U^T y, (4,)
w_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)
w_pinv = np.linalg.pinv(X) @ y
np.testing.assert_allclose(w_svd, w_lstsq, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(w_svd, w_pinv, rtol=1e-6, atol=1e-8)
print("w (SVD, from scratch) =", show(w_svd))
print("w (np.linalg.lstsq)   =", show(w_lstsq))
print("w (np.linalg.pinv)    =", show(w_pinv), "  all three agree: OK")
print("fitted values =", show(X @ w_svd))

# %%
# Low-rank approximation of the photo: error and storage against the rank.
section("image compression")
U_g, s_g, Vt_g = np.linalg.svd(gray, full_matrices=False)  # U (427, 427), s (427,), Vt (427, 640)
n_px, d_px = gray.shape
total = (s_g**2).sum()
print("singular values: first five =", show(s_g[:5]), f"  last = {s_g[-1]:.4f}")
for r in RANKS:
    approx = U_g[:, :r] @ np.diag(s_g[:r]) @ Vt_g[:r]  # (427, 640), rank r
    rel_err = np.linalg.norm(gray - approx) / np.linalg.norm(gray)  # Frobenius norms
    rel_err_formula = np.sqrt((s_g[r:] ** 2).sum() / total)
    storage = r * (n_px + d_px + 1) / (n_px * d_px)
    assert np.isclose(rel_err, rel_err_formula)
    print(f"rank {r:>2}: relative error {rel_err:.4f} (formula {rel_err_formula:.4f})  "
          f"energy kept {(s_g[:r] ** 2).sum() / total:.4f}  storage {storage:.1%} of the pixels")

# %%
# With scikit-learn: TruncatedSVD gives the same top singular values.
section("with scikit-learn")
from sklearn.decomposition import TruncatedSVD  # noqa: E402

tsvd = TruncatedSVD(n_components=50, algorithm="arpack", random_state=SEED).fit(gray)
np.testing.assert_allclose(tsvd.singular_values_, s_g[:50], rtol=1e-6, atol=1e-8)
print("TruncatedSVD(algorithm='arpack') top 50 singular values match np.linalg.svd: OK")
tsvd_r = TruncatedSVD(n_components=50, random_state=SEED).fit(gray)  # default algorithm: randomized
worst = np.max(np.abs(tsvd_r.singular_values_ - s_g[:50]) / s_g[:50])
print(f"TruncatedSVD default (randomized): largest relative difference over the top 50 = {worst:.1e}")
print("components_ shape:", tsvd.components_.shape, "= the top rows of V^T")

# %%
# Key plot: the photo at ranks 5, 20, 50 and the reconstruction error against the rank.
section("key plot")
fig = plt.figure(figsize=(7, 4.4), dpi=120)
grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.1])
panels = [("original (rank 427)", gray)] + [
    (f"rank {r}: {r * (n_px + d_px + 1) / (n_px * d_px):.0%} of the numbers", U_g[:, :r] @ np.diag(s_g[:r]) @ Vt_g[:r])
    for r in RANKS
]
for i, (title, img) in enumerate(panels):
    ax = fig.add_subplot(grid[0, i])
    ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
ax = fig.add_subplot(grid[1, :])
ranks_all = np.arange(1, 201)
rel_errs = np.sqrt(1 - np.cumsum(s_g**2)[ranks_all - 1] / total)
ax.plot(ranks_all, rel_errs, "-", color="tab:blue", lw=2, label="relative error ‖A − A_k‖ / ‖A‖")
ax.plot(RANKS, rel_errs[np.array(RANKS) - 1], "o", color="tab:red", ms=7, label="ranks shown above")
for r in RANKS:
    ax.annotate(f"{rel_errs[r - 1]:.3f}", (r, rel_errs[r - 1]), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.set_xlabel("rank k (number of singular values kept)")
ax.set_ylabel("relative error (Frobenius)")
ax.set_ylim(0, None)
ax.legend(loc="upper right", fontsize=8)
ax.set_title("A few singular values carry most of the image; the error falls fast, then slowly", fontsize=10)
fig.tight_layout()
path = IMG / "a3_2_image_rank_error.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Replace the photo by rng.random((427, 640)) (pure noise). How fast does the error fall now, and why?
# 2. Set N_ROWS, N_COLS = 4, 6 (wide). What shapes does svd(..., full_matrices=False) return?
# 3. In the least-squares cell, drop the column of ones. Is X still rank-deficient? Does w change?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
