# %% [markdown]
# # B2.5 Support vector machines — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b2_5_svm_ex2.py   (CPU, ~25 s)

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
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()
C_GRID = [0.01, 0.1, 1.0, 10.0]
SG_STEPS = 200_000
OUT = Path(__file__).resolve().parents[3] / "outputs" / "b2_5"
OUT.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
section("exercise 1 check")
P = np.array([[1.0, 1.0], [2.0, 0.0], [3.0, 3.0], [4.0, 2.0]])
y4 = np.array([-1.0, -1.0, 1.0, 1.0])
D = np.sqrt(((P[:2, None, :] - P[None, 2:, :]) ** 2).sum(-1))  # (2 neg, 2 pos) cross distances
print(f"(a) cross distances (rows (1,1),(2,0); cols (3,3),(4,2)): {show(D)}")
w_h, b_h = np.array([0.5, 0.5]), -2.0
print(f"    hard margin w {w_h.tolist()} b {b_h}: y f(x) = {show(y4 * (P @ w_h + b_h))}")
hard = SVC(kernel="linear", C=1e6).fit(P, y4)
print(f"    SVC(C=1e6): w {show(hard.coef_[0])}  b {hard.intercept_[0]:.4f}  support vectors {len(hard.support_)}")
print(f"(b) width 2/|w| = {2 / np.linalg.norm(w_h):.4f}")
w_c, b_c = np.array([0.5, 0.5]), -2.5
hinge = np.maximum(0.0, 1 - y4 * (P @ w_c + b_c))
print(f"(c) hinge losses {show(hinge)}   objective 0.5|w|^2 + C sum = {0.5 * w_c @ w_c:.4f} + {hinge.sum():.4f} "
      f"= {0.5 * w_c @ w_c + hinge.sum():.4f}")
x, z = np.array([1.0, 2.0]), np.array([3.0, 1.0])
phi = lambda v: np.array([1, np.sqrt(2) * v[0], np.sqrt(2) * v[1], v[0] ** 2, np.sqrt(2) * v[0] * v[1], v[1] ** 2])
print(f"(d) (1 + x.z)^2 = {(1 + x @ z) ** 2}   phi(x).phi(z) = {phi(x) @ phi(z):.4f}")

# %%
section("data: all 30 features")
bc = load_breast_cancer()
X_all, y_all = bc.data.astype(np.float64), np.where(bc.target == 0, 1.0, -1.0)
gen = np.random.default_rng(SEED)  # the lab's stratified 60/20/20 split
parts: list[list[IntArray]] = [[], [], []]
for c in np.unique(y_all):
    rows = gen.permutation(np.flatnonzero(y_all == c))
    n_te, n_va = round(0.2 * len(rows)), round(0.2 * len(rows))
    parts[2].append(rows[:n_te])
    parts[1].append(rows[n_te:n_te + n_va])
    parts[0].append(rows[n_te + n_va:])
i_tr, i_va, _ = (gen.permutation(np.concatenate(p)) for p in parts)
mu, sd = X_all[i_tr].mean(0), X_all[i_tr].std(0)
Z_tr, Z_va = (X_all[i_tr] - mu) / sd, (X_all[i_va] - mu) / sd
y_tr, y_va = y_all[i_tr], y_all[i_va]
n = len(y_tr)
print(f"train {Z_tr.shape}  val {Z_va.shape}")


def hinge_objective(w: FloatArray, b: float, lam: float) -> float:
    return float(lam / 2 * w @ w + np.mean(np.maximum(0.0, 1 - y_tr * (Z_tr @ w + b))))


def fit_svm_sg(lam: float, steps: int) -> tuple[FloatArray, float]:
    """Sub-gradient descent on (lam/2)|w|^2 + mean hinge, step 1/(lam t), averaged over the last half."""
    w, b = np.zeros(Z_tr.shape[1]), 0.0
    w_avg, b_avg = np.zeros_like(w), 0.0
    for t in range(1, steps + 1):
        act = y_tr * (Z_tr @ w + b) < 1
        gw = lam * w - (y_tr[act, None] * Z_tr[act]).sum(0) / n
        gb = -y_tr[act].sum() / n
        w, b = w - gw / (lam * t), b - gb / (lam * t)
        if t > steps // 2:  # averaging the iterates smooths out the zig-zag around the kinks
            w_avg += w
            b_avg += b
    k = steps - steps // 2
    return w_avg / k, b_avg / k


def fit_logistic_gd(lam: float, tol: float = 1e-8, max_iters: int = 500_000) -> tuple[FloatArray, float]:
    """GD on mean log-loss + (lam/2)|w|^2 with labels in {-1, +1}: loss log(1 + exp(-y f))."""
    A = np.column_stack([np.ones(n), Z_tr])
    theta = np.zeros(A.shape[1])
    pen = np.r_[0.0, np.ones(Z_tr.shape[1])]
    lr = 1.0 / (np.linalg.eigvalsh(A.T @ A / n)[-1] / 4 + lam)
    y01 = (y_tr + 1) / 2
    for _ in range(max_iters):
        z = A @ theta
        p = np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))
        g = A.T @ (p - y01) / n + lam * pen * theta
        if np.linalg.norm(g) < tol:
            break
        theta -= lr * g
    return theta[1:], float(theta[0])


# %%
section("hinge versus log-loss across C")
print(f"{'C':>6} | {'SVM: val acc':>12} {'yf<1':>5} {'|w|':>7} {'obj gap':>8} | {'LR: val acc':>11} "
      f"{'p(1-p)>.01':>10} {'|w|':>7} {'max dp':>8}")
for C in C_GRID:
    lam = 1.0 / (C * n)
    w_s, b_s = fit_svm_sg(lam, SG_STEPS)
    svc = SVC(kernel="linear", C=C, tol=1e-8).fit(Z_tr, y_tr)
    gap = hinge_objective(w_s, b_s, lam) - hinge_objective(svc.coef_[0], svc.intercept_[0], lam)
    active_s = int(np.sum(y_tr * (Z_tr @ w_s + b_s) < 1))
    w_l, b_l = fit_logistic_gd(lam)
    sk = LogisticRegression(C=C, max_iter=100_000, tol=1e-10).fit(Z_tr, y_tr)
    z_tr = Z_tr @ w_l + b_l
    p_tr = 1 / (1 + np.exp(-z_tr))
    p_va_mine = 1 / (1 + np.exp(-(Z_va @ w_l + b_l)))
    dp = np.max(np.abs(p_va_mine - sk.predict_proba(Z_va)[:, 1]))
    print(f"{C:>6} | {np.mean(np.sign(Z_va @ w_s + b_s) == y_va):>12.4f} {active_s:>5} {np.linalg.norm(w_s):>7.3f} "
          f"{gap:>8.1e} | {np.mean(np.sign(Z_va @ w_l + b_l) == y_va):>11.4f} {int(np.sum(p_tr * (1 - p_tr) > 0.01)):>10} "
          f"{np.linalg.norm(w_l):>7.3f} {dp:>8.1e}")
print("obj gap = our hinge objective minus SVC's (>= 0; small = converged); max dp = max |p_mine - p_sklearn| on val")

# %%
section("loss plot")
m = np.linspace(-3, 3, 601)
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
ax.plot(m, np.maximum(0, 1 - m), "-", label="hinge max(0, 1 - yf)")
ax.plot(m, np.log1p(np.exp(-m)), "--", label="log-loss log(1 + e^(-yf))")
ax.plot(m, (m < 0).astype(float), ":", label="0-1 loss (misclassified)")
ax.set_xlabel("y f(x) (positive = correct side)")
ax.set_ylabel("loss")
ax.set_title("The hinge is exactly 0 beyond y f = 1; log-loss never is")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "ex2_hinge_vs_logloss.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved outputs/b2_5/ex2_hinge_vs_logloss.png")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
