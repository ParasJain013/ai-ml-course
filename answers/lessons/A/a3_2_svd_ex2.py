# %% [markdown]
# # A3.2 Singular value decomposition — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a3_2_svd_ex2.py   (CPU, ~3 s)

# %%
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()
OUT = Path(__file__).resolve().parents[3] / "outputs" / "a3_2_answers"
OUT.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
section("exercise 1 check")
A = np.array([[2.0, 2.0], [-1.0, 1.0]])
U, s, Vt = np.linalg.svd(A)
print("A^T A =", show(A.T @ A), " singular values =", show(s), " (2 sqrt 2 =", round(2 * np.sqrt(2), 4), ")")
signs = np.sign(Vt[:, 0])  # make v1, v2 start with a positive entry, as in the hand solution
U, Vt = U * signs, Vt * signs[:, None]
print("V (columns) =", show(Vt.T), " U (columns) =", show(U))
A1 = s[0] * np.outer(U[:, 0], Vt[0])
print("rank-1 approximation =", show(A1), " squared Frobenius error =", round(((A - A1) ** 2).sum(), 4))
print("condition number =", round(s[0] / s[1], 4), " |det A| =", round(abs(np.linalg.det(A)), 4), "= s1 * s2 =", round(s.prod(), 4))
print("storage of a rank-40 approximation of a 1000 x 800 matrix:", 40 * (1000 + 800 + 1), "numbers =",
      f"{40 * 1801 / 800_000:.1%}")

# %%
section("exercise 2: low-rank signal plus noise")
N_USERS, N_ITEMS, TRUE_RANK, NOISE_SD = 100, 80, 3, 1.0
P = rng.normal(size=(N_USERS, TRUE_RANK))  # (100, 3)
Q = rng.normal(size=(N_ITEMS, TRUE_RANK))  # (80, 3)
L = P @ Q.T  # (100, 80), exactly rank 3
Y = L + rng.normal(0.0, NOISE_SD, size=L.shape)  # (100, 80), what you observe
print("L", L.shape, "rank", np.linalg.matrix_rank(L), "  Y rank", np.linalg.matrix_rank(Y))
U_y, s_y, Vt_y = np.linalg.svd(Y, full_matrices=False)
print("top 6 singular values of Y =", show(s_y[:6]))


def truncate(U: FloatArray, s: FloatArray, Vt: FloatArray, k: int) -> FloatArray:
    """Rank-k reconstruction. U (n, r), s (r,), Vt (r, d) -> (n, d)."""
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k]


ks = np.arange(1, 21)
err_vs_clean = np.array([np.linalg.norm(truncate(U_y, s_y, Vt_y, k) - L) / np.linalg.norm(L) for k in ks])
print(f"relative error of the raw noisy Y against L:      {np.linalg.norm(Y - L) / np.linalg.norm(L):.4f}")
for k in (1, 2, 3, 4, 10, 20):
    print(f"relative error of the rank-{k:<2} truncation against L: {err_vs_clean[k - 1]:.4f}")
best_k = int(ks[np.argmin(err_vs_clean)])
energy = np.cumsum(s_y**2) / np.sum(s_y**2)
k_energy = int(np.searchsorted(energy, 0.90) + 1)
print("best k against the clean matrix:", best_k, f"  k chosen by the 90% energy rule: {k_energy} (energy at k=3: {energy[2]:.3f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
ax1.semilogy(np.arange(1, len(s_y) + 1), s_y, "o-", ms=3)
ax1.axvline(TRUE_RANK, color="0.5", ls="--", label=f"true rank {TRUE_RANK}")
ax1.set_xlabel("index j")
ax1.set_ylabel("singular value σ_j (log scale)")
ax1.set_title("Three values stand out; the rest are noise")
ax1.legend()
ax2.plot(ks, err_vs_clean, "s-", ms=4)
ax2.axhline(np.linalg.norm(Y - L) / np.linalg.norm(L), color="tab:red", ls=":", label="raw noisy Y")
ax2.set_xlabel("rank k kept")
ax2.set_ylabel("relative error against the clean L")
ax2.set_ylim(0, None)
ax2.set_title("Truncating at the true rank removes most of the noise")
ax2.legend()
fig.tight_layout()
fig.savefig(OUT / "ex2_denoise.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved outputs/a3_2_answers/ex2_denoise.png")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
