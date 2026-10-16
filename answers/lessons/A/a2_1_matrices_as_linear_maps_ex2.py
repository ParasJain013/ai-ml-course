# %% [markdown]
# # A2.1 Matrices as linear maps — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a2_1_matrices_as_linear_maps_ex2.py   (CPU, ~3 s)

# %%
from __future__ import annotations

import math
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
OUT = Path(__file__).resolve().parents[3] / "outputs" / "a2_1_answers"
OUT.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
section("exercise 1 check")
M = np.array([[0.0, 2.0], [1.0, 0.0]])
v = np.array([3.0, -1.0])
print("M e1 =", show(M[:, 0]), "  M e2 =", show(M[:, 1]))
print("row view:", [float(M[0] @ v), float(M[1] @ v)], "  column view:", show(3 * M[:, 0] + (-1) * M[:, 1]))
print("det(M) =", M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
for shapes in [[(10, 2), (2, 2)], [(2, 10), (2, 2)], [(2, 2), (2,)], [(4, 3), (3, 2), (4, 2)]]:
    try:
        out = np.ones(shapes[0])
        for s in shapes[1:]:
            out = out @ np.ones(s)
        print(shapes, "->", out.shape)
    except ValueError:
        print(shapes, "-> error")

# %%
section("exercise 2: reflection matrix")


def reflection(phi_deg: float) -> FloatArray:
    """Reflect across the line through the origin at angle phi to the x-axis. -> (2, 2).

    Reflection sends the direction at angle alpha to the direction at angle 2*phi - alpha.
    e1 is at alpha = 0, so it goes to angle 2*phi; e2 is at alpha = 90, so it goes to 2*phi - 90.
    """
    two_phi = math.radians(2.0 * phi_deg)
    col1 = [math.cos(two_phi), math.sin(two_phi)]  # image of e1
    col2 = [math.cos(two_phi - math.pi / 2), math.sin(two_phi - math.pi / 2)]  # = (sin 2phi, -cos 2phi)
    return np.array(col1 + col2).reshape(2, 2, order="F")  # columns are the images


for phi in [0.0, 30.0, 45.0, 90.0, 123.0]:
    F = reflection(phi)
    pts = rng.normal(size=(100, 2))  # (100, 2), points as rows
    np.testing.assert_allclose(pts @ F.T @ F.T, pts, rtol=1e-6, atol=1e-8)  # twice = identity
    np.testing.assert_allclose(np.linalg.norm(pts @ F.T, axis=1), np.linalg.norm(pts, axis=1), rtol=1e-6, atol=1e-8)
    on_line = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi))])
    np.testing.assert_allclose(F @ on_line, on_line, rtol=1e-6, atol=1e-8)  # points on the line stay
    det = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    assert math.isclose(det, -1.0, abs_tol=1e-12), det
    print(f"phi = {phi:5.1f}: F = {show(F)}  det = {det + 0.0:.4f}  all checks OK")

# %%
section("exercise 2: plot")
F_SHAPE = np.array(
    [[0, 0], [0.25, 0], [0.25, 0.5], [0.6, 0.5], [0.6, 0.7], [0.25, 0.7], [0.25, 0.8], [0.8, 0.8], [0.8, 1.0], [0, 1.0], [0, 0]]
)
F30 = reflection(30.0)
fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
ax.fill(F_SHAPE[:, 0], F_SHAPE[:, 1], color="0.8", label="original")
moved = F_SHAPE @ F30.T
ax.fill(moved[:, 0], moved[:, 1], color="tab:blue", alpha=0.6, hatch="//", label="reflected")
t = np.linspace(-1.5, 1.5, 2)
ax.plot(t * math.cos(math.radians(30)), t * math.sin(math.radians(30)), "k--", label="mirror line, 30°")
ax.set_aspect("equal")
ax.set_xlabel("x1 (unitless)")
ax.set_ylabel("x2 (unitless)")
ax.set_title("Reflection across the 30° line flips the F (det = -1)")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "reflection_30.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved outputs/a2_1_answers/reflection_30.png")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
