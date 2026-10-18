# %% [markdown]
# # B4.4 Class imbalance — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b4_4_class_imbalance_ex2.py   (CPU, ~30 s)

# %%
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()
C_FP, C_FN = 1.0, 50.0


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
q, w = 0.6, 49.0
odds = (q / (1 - q)) / w
print(f"(a) corrected p = {odds / (1 + odds):.4f}")
print(f"(b) equivalent threshold 1/(1+w) = {1 / (1 + w):.4f}; assumed cost ratio c_FN/c_FP = {w:g}")
pi = 0.02
print(f"(c) always negative: accuracy {1 - pi:.2f}, recall 0, F1 0;  always positive: accuracy {pi:.2f}, precision {pi:.2f}, "
      f"recall 1, F1 {2 * pi / (1 + pi):.4f}")
y50 = np.r_[np.ones(50, dtype=int), np.zeros(4950, dtype=int)]
counts = [int(y50[te].sum()) for _, te in KFold(5, shuffle=True, random_state=SEED).split(y50)]
counts_s = [int(y50[te].sum()) for _, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(5000), y50)]
spread = [int(y50[te].sum()) for s in range(200) for _, te in KFold(5, shuffle=True, random_state=s).split(y50)]
print(f"(d) positives per fold: shuffled KFold {counts}   StratifiedKFold {counts_s}   "
      f"over 200 shuffles: min {min(spread)}, max {max(spread)}, mean {np.mean(spread):.1f}")

# %%
section("data (as in the lab)")
X_all, y_all = make_classification(n_samples=60_000, n_features=10, n_informative=5, n_redundant=2, weights=[0.99],
                                   flip_y=0.0, class_sep=1.0, random_state=SEED)
y_all = y_all.astype(np.int64)
gen = np.random.default_rng(SEED)  # the lab's stratified 60/20/20 split
parts: list[list[IntArray]] = [[], [], []]
for c in np.unique(y_all):
    rows = gen.permutation(np.flatnonzero(y_all == c))
    n_te, n_va = round(0.2 * len(rows)), round(0.2 * len(rows))
    parts[2].append(rows[:n_te])
    parts[1].append(rows[n_te:n_te + n_va])
    parts[0].append(rows[n_te + n_va:])
i_tr, i_va, _ = (gen.permutation(np.concatenate(p)) for p in parts)
X_tr, y_tr, X_va, y_va = X_all[i_tr], y_all[i_tr], X_all[i_va], y_all[i_va]
pos, neg = np.flatnonzero(y_tr == 1), np.flatnonzero(y_tr == 0)
kept = np.random.default_rng(SEED).choice(neg, size=len(pos), replace=False)
under = np.r_[pos, kept]
beta = len(pos) / len(neg)
print(f"train positives {len(pos)}   undersampled rows {len(under)}   beta {beta:.5f}")


def lgbm(**kw: float) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, n_jobs=1, random_state=SEED, deterministic=True,
                              force_row_wise=True, verbose=-1, **kw)


def correct(p: FloatArray, b: float) -> FloatArray:
    o = b * p / (1 - p)
    return o / (1 + o)


# %%
section("LightGBM: plain, weighted, undersampled")
runs = {
    "plain": (lgbm().fit(X_tr, y_tr), None),
    "scale_pos_weight=99": (lgbm(scale_pos_weight=len(neg) / len(pos)).fit(X_tr, y_tr), len(pos) / len(neg)),
    "undersampled 1:1": (lgbm().fit(X_tr[under], y_tr[under]), beta),
}
print(f"{'model':<20} {'AP':>6} {'ROC-AUC':>7} {'Brier':>7} {'mean p':>7}   {'after prior correction':<24}")
for name, (m, b) in runs.items():
    p = m.predict_proba(X_va)[:, 1]
    line = (f"{name:<20} {average_precision_score(y_va, p):>6.4f} {roc_auc_score(y_va, p):>7.4f} "
            f"{brier_score_loss(y_va, p):>7.4f} {p.mean():>7.4f}")
    if b is not None:
        pc = correct(p, b)
        line += f"   Brier {brier_score_loss(y_va, pc):.4f}  mean p {pc.mean():.4f}"
    print(line)
print(f"validation prevalence {y_va.mean():.4f}")

# %%
section("cost-based threshold for the plain model")
oof = cross_val_predict(lgbm(), X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                        method="predict_proba")[:, 1]
grid = np.linspace(0, 1, 1001)


def cost_per_row(y: IntArray, p: FloatArray, t: float) -> float:
    flag = p >= t
    return float((C_FP * np.sum(flag & (y == 0)) + C_FN * np.sum(~flag & (y == 1))) / len(y))


costs = [cost_per_row(y_tr, oof, t) for t in grid]
t_best = float(grid[int(np.argmin(costs))])
p_va = runs["plain"][0].predict_proba(X_va)[:, 1]
print(f"threshold minimizing out-of-fold cost (c_FN = {C_FN:g} c_FP): {t_best:.3f}   formula 1/(1+50) = {1 / 51:.3f}")
for t in [t_best, 1 / 51, 0.5]:
    flag = p_va >= t
    print(f"validation at t = {t:.3f}: cost per row {cost_per_row(y_va, p_va, t):.4f}   flagged {int(flag.sum())}   "
          f"recall {flag[y_va == 1].mean():.4f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
