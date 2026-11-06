# %% [markdown]
# # B6.1 Recommender systems — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b6_1_recommender_systems_ex2.py   (CPU, ~40 s)
# Needs data/ml-100k/ (run labs/B/b6_1_recommender_systems.py once first; it downloads the data).

# %%
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
SEED = 0
T0 = time.perf_counter()
N_HOLD = 5
N_FACTORS, EPOCHS_MF, LAM_MF, LR, BATCH = 20, 37, 0.1, 0.01, 256  # the settings the lab chose on validation


def section(name: str) -> None:
    print(f"\n== {name} ==")


def repo_root() -> Path:
    here = Path(globals().get("__file__", Path.cwd())).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("Run from inside the ai-ml-course folder.")


ROOT = repo_root()


# %%
# from labs/B/b6_1_recommender_systems.py
def cosine_rows(A: FloatArray) -> FloatArray:
    """Cosine similarity between all rows. A (m, d) -> (m, m); a zero row gets similarity 0."""
    norms = np.linalg.norm(A, axis=1)
    norms[norms == 0] = 1.0
    U = A / norms[:, None]
    return U @ U.T


# %%
section("exercise 1 check")
R_toy = np.array([[5, 4, np.nan, 1], [4, 5, 4, 2], [1, 2, 1, 5], [5, np.nan, 5, 1]])
M_toy = ~np.isnan(R_toy)
C_toy = np.where(M_toy, R_toy - np.nanmean(R_toy, axis=1)[:, None], 0.0)
S_toy = cosine_rows(C_toy)
print(f"(a) s(Ben, Dee) = {S_toy[1, 3]:.3f}")
print(f"(b) s(Dee, Ana/Ben/Cy) = {np.round(S_toy[3, :3], 3).tolist()}")
top2 = [0, 1]  # Ana and Ben: the two largest similarities among the raters of item 2
pred = np.nanmean(R_toy[3]) + S_toy[3, top2] @ C_toy[top2, 1] / np.abs(S_toy[3, top2]).sum()
print(f"    Dee, item 2: {pred:.3f}")
mu = 3.5
print(f"(c) b_A = {(20 - 4 * mu) / (25 + 4):.4f}, b_B = {(1700 - 400 * mu) / (25 + 400):.4f}")
bu, bi, p, q, r, eta, lam = 0.2, 0.3, np.array([0.5, 0.0]), np.array([0.4, 0.2]), 2.0, 0.1, 0.1
e = r - (mu + bu + bi + p @ q)
bu2, bi2 = bu + eta * (e - lam * bu), bi + eta * (e - lam * bi)
p2, q2 = p + eta * (e * q - lam * p), q + eta * (e * p - lam * q)
print(f"(d) r_hat {mu + bu + bi + p @ q:.2f}, e {e:.2f}; b_u {bu2:.4f}, b_i {bi2:.4f}, "
      f"p_u {np.round(p2, 4).tolist()}, q_i {np.round(q2, 4).tolist()}; new r_hat {mu + bu2 + bi2 + p2 @ q2:.4f}")

# %%
section("data and the lab's time split")
ratings = pd.read_csv(ROOT / "data" / "ml-100k" / "u.data", sep="\t", names=["user", "item", "rating", "ts"])
ratings[["user", "item"]] -= 1
n_users, n_items = ratings["user"].nunique(), ratings["item"].nunique()
df = ratings.sort_values(["user", "ts", "item"], kind="stable")
pos_from_end = df.groupby("user").cumcount(ascending=False)
test = df[pos_from_end < N_HOLD]
# train + validation, concatenated in the same order as the lab, so SGD sees the same mini-batches
trval = pd.concat([df[pos_from_end >= 2 * N_HOLD], df[(pos_from_end >= N_HOLD) & (pos_from_end < 2 * N_HOLD)]])
u_tv, i_tv, r_tv = (trval[c].to_numpy() for c in ["user", "item", "rating"])
u_te, i_te, r_te = (test[c].to_numpy() for c in ["user", "item", "rating"])
r_tv, r_te = r_tv.astype(float), r_te.astype(float)
print(f"train+validation {len(trval):,}, test {len(test):,}")


def rmse(y: FloatArray, y_hat: FloatArray) -> float:
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


# bias model, as in the lab
mu = float(r_tv.mean())
cnt_i = np.bincount(i_tv, minlength=n_items)
b_i = np.bincount(i_tv, weights=r_tv - mu, minlength=n_items) / (25.0 + cnt_i)
b_u = np.bincount(u_tv, weights=r_tv - mu - b_i[i_tv], minlength=n_users) / (10.0 + np.bincount(u_tv, minlength=n_users))
R = np.zeros((n_users, n_items))
R[u_tv, i_tv] = r_tv
M = np.zeros((n_users, n_items), dtype=bool)
M[u_tv, i_tv] = True
B = mu + b_u[:, None] + b_i[None, :]  # (n_users, n_items) bias-model prediction for every cell
pred_bias = np.clip(B[u_te, i_te], 1, 5)
print(f"bias model test RMSE {rmse(r_te, pred_bias):.4f}")


# %%
def neighbour_predict(C: FloatArray, M: BoolArray, S: FloatArray, base: FloatArray,
                      u: IntArray, i: IntArray, m: int) -> FloatArray:
    """User-based: base[u, i] + sum_v s_uv C[v, i] / sum |s_uv| over the m most similar raters of i."""
    out = np.empty(len(u))
    for row, (uu, ii) in enumerate(zip(u, i)):
        raters = np.flatnonzero(M[:, ii])
        if len(raters) == 0:
            out[row] = base[uu, ii]
            continue
        top = raters[np.argsort(-S[uu, raters], kind="stable")[:m]]
        s = S[uu, top]
        den = np.abs(s).sum()
        out[row] = base[uu, ii] + (s @ C[top, ii]) / den if den > 0 else base[uu, ii]
    return np.clip(out, 1, 5)


section("(a) user-based CF: centred on the user mean versus on the bias model")
user_mean = R.sum(1) / M.sum(1)
C_mean = np.where(M, R - user_mean[:, None], 0.0)
S_mean = cosine_rows(C_mean)
np.fill_diagonal(S_mean, 0.0)
base_mean = np.repeat(user_mean[:, None], n_items, axis=1)
C_bias = np.where(M, R - B, 0.0)  # residuals after the bias model, 0 where missing
S_bias = cosine_rows(C_bias)
np.fill_diagonal(S_bias, 0.0)
preds = {}
for m in [20, 40, 100]:
    p_mean = neighbour_predict(C_mean, M, S_mean, base_mean, u_te, i_te, m)
    p_bias = neighbour_predict(C_bias, M, S_bias, B, u_te, i_te, m)
    preds[("user", m)] = p_bias
    print(f"m={m:<3}  user-mean centred {rmse(r_te, p_mean):.4f}   bias-model centred {rmse(r_te, p_bias):.4f}")

section("(b) item-based CF on the bias-model residuals")
S_items = cosine_rows(C_bias.T)  # (n_items, n_items): similarity of item columns
np.fill_diagonal(S_items, 0.0)
# item-based is user-based on the transposed problem: swap the roles of u and i
for m in [20, 40, 100]:
    p_item = neighbour_predict(C_bias.T, M.T, S_items, B.T, i_te, u_te, m)
    preds[("item", m)] = p_item
    print(f"m={m:<3}  item-based {rmse(r_te, p_item):.4f}")

# %%
section("(c) bootstrap CI: best CF versus matrix factorization")
best_key = min(preds, key=lambda k: rmse(r_te, preds[k]))
print(f"best CF variant: {best_key[0]}-based, m={best_key[1]}, test RMSE {rmse(r_te, preds[best_key]):.4f}")

# matrix factorization exactly as in the lab (same seed, settings and training rows)
g = np.random.default_rng(SEED)
P = g.normal(0, 0.1, size=(n_users, N_FACTORS))
Q = g.normal(0, 0.1, size=(n_items, N_FACTORS))
bu_mf, bi_mf = np.zeros(n_users), np.zeros(n_items)
g = np.random.default_rng(SEED)
for _ in range(EPOCHS_MF):
    order = g.permutation(len(r_tv))
    for s in range(0, len(r_tv), BATCH):
        b = order[s:s + BATCH]
        u, i, r = u_tv[b], i_tv[b], r_tv[b]
        e = r - (mu + bu_mf[u] + bi_mf[i] + np.sum(P[u] * Q[i], 1))
        Pu, Qi = P[u], Q[i]
        np.add.at(bu_mf, u, LR * (e - LAM_MF * bu_mf[u]))
        np.add.at(bi_mf, i, LR * (e - LAM_MF * bi_mf[i]))
        np.add.at(P, u, LR * (e[:, None] * Qi - LAM_MF * Pu))
        np.add.at(Q, i, LR * (e[:, None] * Pu - LAM_MF * Qi))
pred_mf = np.clip(mu + bu_mf[u_te] + bi_mf[i_te] + np.sum(P[u_te] * Q[i_te], 1), 1, 5)
print(f"MF test RMSE {rmse(r_te, pred_mf):.4f} (the lab printed 1.0023)")

gb = np.random.default_rng(SEED)
n = len(r_te)
diffs = np.empty(1000)
for b in range(1000):
    ix = gb.integers(0, n, size=n)  # resample test ratings, the SAME rows for both models (paired)
    diffs[b] = rmse(r_te[ix], preds[best_key][ix]) - rmse(r_te[ix], pred_mf[ix])
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"RMSE(best CF) - RMSE(MF): {rmse(r_te, preds[best_key]) - rmse(r_te, pred_mf):.4f}, 95% CI [{lo:.4f}, {hi:.4f}]")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
