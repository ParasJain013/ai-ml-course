# %% [markdown]
# # B3.3 Gradient boosting — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b3_3_gradient_boosting_ex2.py   (CPU, ~5 s)

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
import xgboost as xgb
from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()
N_TREES, LR, DEPTH, MIN_LEAF = 100, 0.1, 3, 5
FIVE = ["mean radius", "mean texture", "mean smoothness", "mean concavity", "mean symmetry"]  # B2.1's features
OUT = Path(__file__).resolve().parents[3] / "outputs" / "b3_3"
OUT.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
x4 = np.array([[1.0], [2.0], [3.0], [4.0]])
y4 = np.array([2.0, 2.0, 6.0, 10.0])
F0 = y4.mean()
r = y4 - F0
sse = {k + 0.5: ((r[:k] - r[:k].mean()) ** 2).sum() + ((r[k:] - r[k:].mean()) ** 2).sum() for k in (1, 2, 3)}
print(f"(a) F0 {F0}   residuals {r.tolist()}   SSE by cut " + "  ".join(f"{c}: {v:.4f}" for c, v in sse.items()))
best = min(sse, key=sse.get)
k = int(best - 0.5)
leaf = np.where(np.arange(4) < k, r[:k].mean(), r[k:].mean())
print(f"    best cut {best}: leaves {r[:k].mean()} and {r[k:].mean()}   F1 = {np.round(F0 + 0.1 * leaf, 4).tolist()}")
g = F0 - y4
lam, gam = 2.0, 1.0
GL, HL, GR, HR = g[:k].sum(), float(k), g[k:].sum(), float(4 - k)
gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - (GL + GR) ** 2 / (HL + HR + lam)) - gam
print(f"(b) lambda 2: w_L = {-GL / (HL + lam)}  w_R = {-GR / (HR + lam)}   gain (gamma 1) = {gain:.4f}")
one = xgb.XGBRegressor(n_estimators=1, max_depth=1, learning_rate=1.0, reg_lambda=lam, gamma=gam, base_score=F0,
                       min_child_weight=0, tree_method="exact", n_jobs=1).fit(x4, y4)
print(f"    XGBoost leaf weights {np.round(one.predict(x4) - F0, 4).tolist()}")
p0 = 0.25
f0 = np.log(p0 / (1 - p0))
print(f"(c) F0 = log(0.25/0.75) = {f0:.4f}   pseudo-residuals: positive {1 - p0}, negative {-p0}   "
      f"Newton leaf ({1 - p0} + {-p0}) / (2 x {p0 * (1 - p0)}) = {(1 - p0 - p0) / (2 * p0 * (1 - p0)):.4f}")

# %%
section("data")
bc = load_breast_cancer()
X_all = bc.data.astype(np.float64)
y_all = (bc.target == 0).astype(np.int64)  # 1 = malignant, as in B2.1
gen = np.random.default_rng(SEED)  # B2.1's stratified 60/20/20 split
parts: list[list[IntArray]] = [[], [], []]
for c in np.unique(y_all):
    rows = gen.permutation(np.flatnonzero(y_all == c))
    n_te, n_va = round(0.2 * len(rows)), round(0.2 * len(rows))
    parts[2].append(rows[:n_te])
    parts[1].append(rows[n_te:n_te + n_va])
    parts[0].append(rows[n_te + n_va:])
i_tr, i_va, _ = (gen.permutation(np.concatenate(p)) for p in parts)
X_tr, y_tr, X_va, y_va = X_all[i_tr], y_all[i_tr], X_all[i_va], y_all[i_va]
print(f"train {X_tr.shape}  val {X_va.shape}  positive rate {y_tr.mean():.4f}")


# %%
def pseudo_residual(y: IntArray, F: FloatArray) -> FloatArray:
    """y - sigmoid(F), evaluated with scikit-learn's formula. y (n,), F (n,) -> (n,).

    Mathematically this is just y - expit(F). Splits that tie in exact arithmetic are separated by
    the last bits of these numbers, so matching scikit-learn's trees exactly needs the same formula.
    """
    e = np.exp(-np.maximum(F, -37.0))
    grad = np.where(F > -37, ((1 - y) - y * e) / (1 + e), np.exp(np.minimum(F, -37.0)) - y)  # expit(F) - y
    return -grad


def fit_gb_classifier(X: FloatArray, y: IntArray, n_trees: int, lr: float, max_depth: int, min_leaf: int
                      ) -> tuple[float, list[tuple[DecisionTreeRegressor, dict[int, float]]]]:
    """Binary gradient boosting on log-loss with Newton leaf values. X (n, d), y (n,) -> (F0, [(tree, leaf values)])."""
    f0 = float(np.log(y.mean() / (1 - y.mean())))  # log-odds of the positive rate
    F = np.full(len(y), f0)  # (n,) logits
    tie_breaker = np.random.RandomState(SEED)  # mirrors scikit-learn's tie-breaking only
    stages = []
    for _ in range(n_trees):
        residual = pseudo_residual(y, F)  # (n,) y - p = -gradient of log-loss w.r.t. the logit
        p = y - residual  # (n,) predicted probabilities, computed the way scikit-learn does
        tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_leaf,
                                     random_state=tie_breaker).fit(X, residual)
        leaf_of = tree.apply(X)  # (n,) leaf id of each row
        values = {}
        for leaf in np.unique(leaf_of):
            in_leaf = leaf_of == leaf
            hess = np.mean(p[in_leaf] * (1 - p[in_leaf]))
            values[int(leaf)] = float(np.mean(residual[in_leaf])) / float(hess) if hess > 1e-150 else 0.0  # Newton step
        F += lr * np.array([values[int(lf)] for lf in leaf_of])
        stages.append((tree, values))
    return f0, stages


def staged_logits(f0: float, stages: list[tuple[DecisionTreeRegressor, dict[int, float]]], lr: float,
                  X: FloatArray) -> FloatArray:
    """Logits after 1..M trees. X (m, d) -> (M, m)."""
    steps = np.stack([np.array([vals[int(lf)] for lf in tree.apply(X)]) for tree, vals in stages])
    return f0 + lr * np.cumsum(steps, axis=0)


def log_loss(y: IntArray, p: FloatArray) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


section("from scratch versus GradientBoostingClassifier (B2.1's five features)")
cols5 = [list(bc.feature_names).index(f) for f in FIVE]
f0_5, stages5 = fit_gb_classifier(X_tr[:, cols5], y_tr, N_TREES, LR, DEPTH, MIN_LEAF)
P5 = expit(staged_logits(f0_5, stages5, LR, X_va[:, cols5]))  # (M, m)
gbc = GradientBoostingClassifier(n_estimators=N_TREES, learning_rate=LR, max_depth=DEPTH, min_samples_leaf=MIN_LEAF,
                                 random_state=SEED).fit(X_tr[:, cols5], y_tr)
sk_stages = np.stack([p[:, 1] for p in gbc.staged_predict_proba(X_va[:, cols5])])
np.testing.assert_allclose(P5, sk_stages, rtol=1e-9, atol=1e-12)
print(f"F0 = log-odds of {y_tr.mean():.4f} = {f0_5:.4f}")
print(f"predict_proba matches GradientBoostingClassifier at all {N_TREES} stages: OK")

section("validation log-loss: five features and all 30")
P30 = expit(staged_logits(*fit_gb_classifier(X_tr, y_tr, N_TREES, LR, DEPTH, MIN_LEAF)[:2], LR, X_va))
curves = {"5 features": [log_loss(y_va, P5[m]) for m in range(N_TREES)],
          "30 features": [log_loss(y_va, P30[m]) for m in range(N_TREES)]}
for label, P in [("5 features", P5), ("30 features", P30)]:
    ll = curves[label]
    best_m = int(np.argmin(ll))
    print(f"{label:<11}: val log-loss after 10 / 50 / 100 trees {ll[9]:.4f} / {ll[49]:.4f} / {ll[99]:.4f}   "
          f"best {ll[best_m]:.4f} at {best_m + 1:>3}   val accuracy at 100 {np.mean((P[-1] > 0.5) == y_va):.4f}")
print("B2.1 logistic regression on the 5 features: val log-loss 0.0856, accuracy 0.9823")
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
for (label, ll), style in zip(curves.items(), ["--", "-"]):
    ax.plot(np.arange(1, N_TREES + 1), ll, style, label=f"boosting, {label}")
ax.axhline(0.0856, color="gray", linestyle=":", label="B2.1 logistic regression, 5 features")
ax.set_xlabel("boosting rounds")
ax.set_ylabel("validation log-loss (nats)")
ax.set_ylim(0, 0.7)
ax.set_title("On 5 features logistic regression wins; on 30 boosting does")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "ex2_classifier_logloss.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved outputs/b3_3/ex2_classifier_logloss.png")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
