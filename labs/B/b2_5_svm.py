# %% [markdown]
# # B2.5 Support vector machines — lab
# Lesson: lessons/B/b2_5_svm.md
# Run: uv run python labs/B/b2_5_svm.py   (CPU, defaults: ~10 s)
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
from PIL import Image
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
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
OUT = ROOT / "outputs" / "b2_5"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE runs the sub-gradient solver longer and uses more moons.
GPU_SCALE = False
SG_STEPS = 400_000 if GPU_SCALE else 100_000  # sub-gradient descent steps on the hinge loss
SG_CHECKPOINTS = [100, 1_000, 10_000, SG_STEPS]
N_MOONS = 2_000 if GPU_SCALE else 300
C_GRID = [0.1, 1.0, 100.0]
GAMMA_GRID = [0.1, 1.0, 10.0]
FEATURES = ["mean radius", "mean texture", "mean smoothness", "mean concavity", "mean symmetry"]
VAL_FRAC, TEST_FRAC = 0.2, 0.2

# %%
# Data: the breast-cancer features and split from B2.1, with labels coded -1 / +1 (SVM convention).
section("data")
bc = load_breast_cancer()
all_names = list(bc.feature_names)
X_all = bc.data[:, [all_names.index(f) for f in FEATURES]].astype(np.float64)  # (569, 5)
y_all = np.where(bc.target == 0, 1.0, -1.0)  # +1 = malignant, -1 = benign


def train_val_test_split(
    n: int, val_frac: float, test_frac: float, gen: np.random.Generator, stratify: FloatArray | None = None
) -> tuple[IntArray, IntArray, IntArray]:
    """Shuffle row indices and cut them three ways. n rows -> (train, val, test) index arrays."""
    # from labs/B/b1_1_ml_workflow.py
    groups = [np.arange(n)] if stratify is None else [np.flatnonzero(stratify == c) for c in np.unique(stratify)]
    parts: list[list[IntArray]] = [[], [], []]
    for rows in groups:
        rows = gen.permutation(rows)
        n_test, n_val = round(test_frac * len(rows)), round(val_frac * len(rows))
        parts[2].append(rows[:n_test])
        parts[1].append(rows[n_test:n_test + n_val])
        parts[0].append(rows[n_test + n_val:])
    train, val, test = (gen.permutation(np.concatenate(p)) for p in parts)
    return train, val, test


i_tr, i_va, i_te = train_val_test_split(len(y_all), VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED), y_all)
mu, sd = X_all[i_tr].mean(axis=0), X_all[i_tr].std(axis=0)
Z_tr, Z_va = (X_all[i_tr] - mu) / sd, (X_all[i_va] - mu) / sd
y_tr, y_va = y_all[i_tr], y_all[i_va]
n, d = Z_tr.shape
print(f"train Z {Z_tr.shape}  val Z {Z_va.shape}  (test {len(i_te)} rows kept aside)   labels {sorted(set(y_tr.tolist()))}")

# %%
# Worked example by hand: four separable points, the maximum-margin line, hinge losses, a kernel.
section("worked example by hand")
P4 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 2.0], [3.0, 2.0]])
y4 = np.array([-1.0, -1.0, 1.0, 1.0])
w_hand, b_hand = np.array([0.4, 0.8]), -1.4  # derived in the lesson from the support vectors (1, 0) and (2, 2)
margins = y4 * (P4 @ w_hand + b_hand)
print(f"hand solution w = {w_hand.tolist()}  b = {b_hand}   y * f(x) per point = {show(margins)}")
print(f"margin width 2 / |w| = {2 / np.linalg.norm(w_hand):.4f}   distance between (1, 0) and (2, 2) = {np.sqrt(5):.4f}")
hard = SVC(kernel="linear", C=1e6).fit(P4, y4)  # huge C: (almost) no violations allowed = hard margin
print(f"SVC(kernel='linear', C=1e6): w = {show(hard.coef_[0])}  b = {hard.intercept_[0]:.4f}  "
      f"support vectors {hard.support_vectors_.tolist()}")
for x_new, y_new in [([1.5, 1.0], 1.0), ([2.0, 1.0], 1.0), ([2.5, 2.0], 1.0)]:
    m = y_new * (np.array(x_new) @ w_hand + b_hand)
    print(f"point {x_new} with y = {y_new:+.0f}: y f(x) = {m:.4f}   hinge loss max(0, 1 - y f) = {max(0.0, 1 - m):.4f}")
x_a, x_b = np.array([1.0, 2.0]), np.array([3.0, 1.0])


def phi(x: FloatArray) -> FloatArray:
    """Explicit degree-2 feature map whose dot product equals (x . z)^2. (2,) -> (3,)."""
    return np.array([x[0] ** 2, np.sqrt(2) * x[0] * x[1], x[1] ** 2])


print(f"phi(x) = {show(phi(x_a))}   phi(z) = {show(phi(x_b))}")
print(f"kernel trick: (x . z)^2 = ({x_a @ x_b})^2 = {(x_a @ x_b) ** 2}   phi(x) . phi(z) = {phi(x_a) @ phi(x_b):.4f}")
print(f"RBF kernel exp(-gamma |x - z|^2), |x - z|^2 = {np.sum((x_a - x_b) ** 2)}: gamma 0.1 -> "
      f"{np.exp(-0.1 * np.sum((x_a - x_b) ** 2)):.4f}   gamma 1 -> {np.exp(-1.0 * np.sum((x_a - x_b) ** 2)):.4f}")

# %%
# From scratch: a linear SVM by sub-gradient descent on (lam/2)|w|^2 + mean hinge loss.
section("from scratch: linear SVM by sub-gradient descent")


def svm_objective(w: FloatArray, b: float, Z: FloatArray, y: FloatArray, lam: float) -> float:
    """(lam/2)|w|^2 + (1/n) sum max(0, 1 - y (Z w + b)). Z (n, d), y (n,) in {-1, +1}."""
    return float(lam / 2 * w @ w + np.mean(np.maximum(0.0, 1.0 - y * (Z @ w + b))))


def fit_linear_svm_sg(
    Z: FloatArray, y: FloatArray, lam: float, steps: int, checkpoints: list[int]
) -> tuple[FloatArray, float, dict[int, tuple[FloatArray, float]]]:
    """Full-batch sub-gradient descent with step 1 / (lam t). -> (w (d,), b, snapshots at checkpoints)."""
    w, b = np.zeros(Z.shape[1]), 0.0
    snaps = {}
    for t in range(1, steps + 1):
        active = y * (Z @ w + b) < 1.0  # (n,) points inside the margin or misclassified
        grad_w = lam * w - (y[active, None] * Z[active]).sum(axis=0) / len(y)  # (d,)
        grad_b = -y[active].sum() / len(y)
        eta = 1.0 / (lam * t)  # decreasing step: needed because the hinge has kinks
        w, b = w - eta * grad_w, b - eta * grad_b
        if t in checkpoints:
            snaps[t] = (w.copy(), b)
    return w, b, snaps


C = 1.0
lam = 1.0 / (C * n)  # scikit-learn's (1/2)|w|^2 + C sum hinge, divided by C n
svc_lin = SVC(kernel="linear", C=C, tol=1e-8).fit(Z_tr, y_tr)
w_svc, b_svc = svc_lin.coef_[0], float(svc_lin.intercept_[0])
w_sg, b_sg, snaps = fit_linear_svm_sg(Z_tr, y_tr, lam, SG_STEPS, SG_CHECKPOINTS)
print(f"C = {C} -> lambda = 1/(C n) = {lam:.6f}")
for t, (w_t, b_t) in snaps.items():
    print(f"step {t:>6}: objective {svm_objective(w_t, b_t, Z_tr, y_tr, lam):.6f}   "
          f"max |w - w_svc| {np.max(np.abs(w_t - w_svc)):.1e}")
print(f"SVC optimum objective {svm_objective(w_svc, b_svc, Z_tr, y_tr, lam):.6f}")
print(f"from scratch: w {show(w_sg)}  b {b_sg:.4f}")
print(f"SVC linear:   w {show(w_svc)}  b {b_svc:.4f}")
# sub-gradient methods converge slowly (error ~ 1/t at best); 1e-3 on weights of size ~1 is the honest bar
np.testing.assert_allclose(w_sg, w_svc, atol=1e-3)
np.testing.assert_allclose(b_sg, b_svc, atol=1e-3)
print("from-scratch linear SVM matches SVC(kernel='linear') to 1e-3: OK")
margin_vals = y_tr * (Z_tr @ w_svc + b_svc)
print(f"training points with y f(x) <= 1 (on or inside the margin): {np.sum(margin_vals <= 1 + 1e-6)}   "
      f"SVC support vectors: {len(svc_lin.support_)}")
print(f"val accuracy: from scratch {np.mean(np.sign(Z_va @ w_sg + b_sg) == y_va):.4f}   "
      f"SVC {np.mean(svc_lin.predict(Z_va) == y_va):.4f}")

# %%
# With scikit-learn: LinearSVC's defaults solve a different problem (squared hinge, penalized intercept).
section("with scikit-learn: LinearSVC and logistic regression")
lsvc_hinge = LinearSVC(loss="hinge", C=C, max_iter=100_000, tol=1e-8).fit(Z_tr, y_tr)
lsvc_default = LinearSVC(C=C).fit(Z_tr, y_tr)
logreg = LogisticRegression(C=C).fit(Z_tr, y_tr)
for name, model in [("LinearSVC(loss='hinge')", lsvc_hinge), ("LinearSVC() default", lsvc_default),
                    ("LogisticRegression(C=1)", logreg)]:
    print(f"{name:<24} w {show(model.coef_[0])}  b {model.intercept_[0]:.4f}  "
          f"val acc {np.mean(model.predict(Z_va) == y_va):.4f}")
print(f"LinearSVC(loss='hinge') max |w - w_svc| {np.max(np.abs(lsvc_hinge.coef_[0] - w_svc)):.1e}")
scores = svc_lin.decision_function(Z_va[:3])
print(f"decision_function for 3 val tumours (signed score, not a probability): {show(scores)}")

# %%
# The kernel trick: an RBF SVC on two moons. Its decision function is a weighted sum of kernels
# centred on the support vectors; check that by recomputing it from dual_coef_.
section("kernel SVM on two moons")
Xm, ym_int = make_moons(n_samples=N_MOONS, noise=0.25, random_state=SEED)
ym = np.where(ym_int == 1, 1.0, -1.0)
im_tr, im_va, _ = train_val_test_split(len(ym), VAL_FRAC, TEST_FRAC, np.random.default_rng(SEED), ym)
Xm_tr, ym_tr, Xm_va, ym_va = Xm[im_tr], ym[im_tr], Xm[im_va], ym[im_va]
print(f"moons: train {Xm_tr.shape}  val {Xm_va.shape}  (features already on comparable scales)")
lin_moons = SVC(kernel="linear", C=1.0).fit(Xm_tr, ym_tr)
print(f"linear SVC on moons: val accuracy {np.mean(lin_moons.predict(Xm_va) == ym_va):.4f}")
fits = {}
for C_m in C_GRID:
    for g in GAMMA_GRID:
        model = SVC(kernel="rbf", C=C_m, gamma=g).fit(Xm_tr, ym_tr)
        fits[(C_m, g)] = model
        print(f"RBF C = {C_m:>5}  gamma = {g:>4}: train acc {np.mean(model.predict(Xm_tr) == ym_tr):.4f}  "
              f"val acc {np.mean(model.predict(Xm_va) == ym_va):.4f}  support vectors {len(model.support_):>3} "
              f"of {len(ym_tr)}")
m_ref = fits[(1.0, 1.0)]
sv = m_ref.support_vectors_  # (s, 2)
K_va = np.exp(-1.0 * ((Xm_va[:, None, :] - sv[None, :, :]) ** 2).sum(axis=-1))  # (n_val, s) RBF kernel
f_manual = K_va @ m_ref.dual_coef_[0] + m_ref.intercept_[0]  # sum_i alpha_i y_i k(x_i, x) + b
np.testing.assert_allclose(f_manual, m_ref.decision_function(Xm_va), rtol=1e-8, atol=1e-10)
print(f"f(x) = sum over {len(sv)} support vectors of alpha_i y_i k(x_i, x) + b reproduces decision_function: OK")
print(f"alpha_i y_i ranges over [{m_ref.dual_coef_.min():.4f}, {m_ref.dual_coef_.max():.4f}]  "
      f"(|alpha_i| <= C = 1; {np.sum(np.isclose(np.abs(m_ref.dual_coef_), 1.0))} at the bound C)")

# %%
# Key plot: decision boundary, margins and support vectors for a grid of C and gamma.
section("key plot")
g0 = np.linspace(-1.7, 2.7, 200)
g1 = np.linspace(-1.3, 1.8, 200)
G0, G1 = np.meshgrid(g0, g1)
grid = np.column_stack([G0.ravel(), G1.ravel()])
fig, axes = plt.subplots(len(C_GRID), len(GAMMA_GRID), figsize=(9, 7.5), dpi=100, sharex=True, sharey=True)
for r, C_m in enumerate(C_GRID):
    for c, g in enumerate(GAMMA_GRID):
        ax = axes[r, c]
        model = fits[(C_m, g)]
        F = model.decision_function(grid).reshape(G0.shape)
        ax.contourf(G0, G1, F > 0, levels=[-0.5, 0.5, 1.5], colors=["#dbe9f6", "#fde3c8"])
        ax.contour(G0, G1, F, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=[0.8, 1.4, 0.8])
        ax.scatter(Xm_tr[ym_tr < 0, 0], Xm_tr[ym_tr < 0, 1], s=8, marker="o", color="tab:blue")
        ax.scatter(Xm_tr[ym_tr > 0, 0], Xm_tr[ym_tr > 0, 1], s=10, marker="^", color="tab:orange")
        ax.scatter(*model.support_vectors_.T, s=40, facecolors="none", edgecolors="k", linewidths=0.6)
        acc = np.mean(model.predict(Xm_va) == ym_va)
        ax.set_title(f"C={C_m:g}, gamma={g:g}: val {acc:.2f}, {len(model.support_)} SV", fontsize=8)
        if r == len(C_GRID) - 1:
            ax.set_xlabel("feature 1")
        if c == 0:
            ax.set_ylabel("feature 2")
fig.suptitle("Larger gamma bends the boundary; larger C narrows the margin (circles = support vectors)")
fig.tight_layout()
key_png = IMG / "b2_5_rbf_c_gamma_grid.png"
fig.savefig(key_png, dpi=100, bbox_inches="tight")
# 9 panels of antialiased contours make a large PNG; a 64-colour palette keeps it under ~150 KB
Image.open(key_png).convert("RGB").quantize(colors=64).save(key_png, optimize=True)
plt.show()
plt.close(fig)
print("saved lessons/B/img/b2_5_rbf_c_gamma_grid.png")

# %%
# Try this (not graded):
# 1. Fit SVC(kernel="poly", degree=3) on the moons and compare it with the RBF kernel.
# 2. Remove the standardization of the breast-cancer features and refit SVC(kernel="rbf"). What happens?
# 3. Use SVC(probability=True) and compare predict_proba with the sign of decision_function on a few points.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
