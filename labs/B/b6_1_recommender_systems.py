# %% [markdown]
# # B6.1 Recommender systems: collaborative filtering and matrix factorization — lab
# Lesson: lessons/B/b6_1_recommender_systems.md
# Run: uv run python labs/B/b6_1_recommender_systems.py   (CPU, defaults: ~30 s)
# GPU: no

# %%
from __future__ import annotations

import hashlib
import sys
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:  # plain `uv run`: draw to files, never open windows
    matplotlib.use("Agg")
    # matplotlib 3.11 warns when plt.show() runs under Agg; under `uv run` it is a harmless no-op
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

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
OUT = ROOT / "outputs" / "b6_1"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU (it is NumPy); GPU_SCALE only trains wider factor models for longer.
GPU_SCALE = False
N_FACTORS = 64 if GPU_SCALE else 20  # k, the length of each user and item vector
EPOCHS = 60 if GPU_SCALE else 40
BATCH = 256  # ratings per mini-batch of SGD
LR = 0.01  # eta
LAMBDAS = [0.05, 0.1, 0.2]  # L2 strengths tried on the validation ratings
N_HOLD = 5  # per user: the last 5 ratings are the test set, the 5 before them validation
K_NEIGHBOURS = 40  # user-based collaborative filtering
N_NEG = 4  # sampled negatives per positive in the implicit-feedback model
LR_IMPLICIT = 0.05  # log-loss gradients are at most 1 in size, so a larger step than LR is safe
IMPLICIT_EPOCHS = 30 if GPU_SCALE else 20
TOP_K = 10

ML_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML_ZIP_MD5 = "0e33842e24a9c977be4e0107933c0723"  # of the official zip
U_DATA_MD5 = "6e47046882bad158b0efbb84cd5cb987"  # of u.data inside it


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_movielens() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download MovieLens 100K once into data/ml-100k/ and return (ratings, items)."""
    folder = DATA / "ml-100k"
    if not (folder / "u.data").exists():
        DATA.mkdir(parents=True, exist_ok=True)
        zpath = DATA / "ml-100k.zip"
        print(f"downloading {ML_URL}")
        urllib.request.urlretrieve(ML_URL, zpath)
        if md5(zpath) != ML_ZIP_MD5:
            raise RuntimeError(f"{zpath} has the wrong checksum; delete it and rerun")
        with zipfile.ZipFile(zpath) as z:
            for name in ["u.data", "u.item", "u.user"]:
                z.extract(f"ml-100k/{name}", DATA)
    if md5(folder / "u.data") != U_DATA_MD5:
        raise RuntimeError(f"{folder / 'u.data'} is not the MovieLens 100K file; delete data/ml-100k and rerun")
    ratings = pd.read_csv(folder / "u.data", sep="\t", names=["user", "item", "rating", "ts"])
    items = pd.read_csv(folder / "u.item", sep="|", encoding="latin-1", header=None, usecols=[0, 1, 2],
                        names=["item", "title", "release"])
    print(f"MovieLens 100K from {folder.relative_to(ROOT).as_posix()}/ (source: GroupLens, {ML_URL})")
    return ratings, items


# %%
# Worked example by hand (lesson): 4 users x 4 items, '-' = not rated.
section("worked example by hand")
R_toy = np.array([
    [5, 4, np.nan, 1],  # Ana
    [4, 5, 4, 2],       # Ben
    [1, 2, 1, 5],       # Cy
    [5, np.nan, 5, 1],  # Dee
])
names = ["Ana", "Ben", "Cy", "Dee"]
M_toy = ~np.isnan(R_toy)
means = np.nanmean(R_toy, axis=1)
C_toy = np.where(M_toy, R_toy - means[:, None], 0.0)  # centred ratings, 0 where missing
print("user means:", np.round(means, 3).tolist())
print("centred ratings (0 = missing):")
for nm, row in zip(names, C_toy):
    print(f"  {nm:<4}", np.round(row, 3).tolist())


def cosine_rows(A: FloatArray) -> FloatArray:
    """Cosine similarity between all rows. A (m, d) -> (m, m); a zero row gets similarity 0."""
    norms = np.linalg.norm(A, axis=1)
    norms[norms == 0] = 1.0
    U = A / norms[:, None]
    return U @ U.T


S_toy = cosine_rows(C_toy)
print("cosine similarity of Ana with Ben, Cy, Dee:", np.round(S_toy[0, 1:], 3).tolist())
# predict Ana's rating of item 3 from the users who rated it (Ben, Cy, Dee)
nb = [1, 2, 3]
num = sum(S_toy[0, v] * C_toy[v, 2] for v in nb)
den = sum(abs(S_toy[0, v]) for v in nb)
print(f"Ana, item 3: {means[0]:.3f} + ({num:.3f}) / {den:.3f} = {means[0] + num / den:.3f}")

# one SGD step of matrix factorization with biases, k = 2
mu, bu, bi = 3.0, 0.5, 0.0
p, q = np.array([0.1, 0.2]), np.array([0.3, -0.1])
r, eta, lam = 5.0, 0.1, 0.1
r_hat = mu + bu + bi + p @ q
e = r - r_hat
bu_new, bi_new = bu + eta * (e - lam * bu), bi + eta * (e - lam * bi)
p_new, q_new = p + eta * (e * q - lam * p), q + eta * (e * p - lam * q)
print(f"MF step: r_hat = {mu} + {bu} + {bi} + {p @ q:.3f} = {r_hat:.3f}, error e = {e:.3f}")
print(f"  b_u {bu} -> {bu_new:.4f}, b_i {bi} -> {bi_new:.4f}")
print(f"  p_u {p.tolist()} -> {np.round(p_new, 4).tolist()}, q_i {q.tolist()} -> {np.round(q_new, 4).tolist()}")
print(f"  new prediction {mu + bu_new + bi_new + p_new @ q_new:.4f} (closer to {r})")

# %%
# Data: ratings, ids from 0, and the per-user time-ordered split.
section("data")
ratings, items = load_movielens()
ratings["user"] -= 1  # MovieLens ids start at 1; arrays start at 0
ratings["item"] -= 1
items["item"] -= 1
titles = items.set_index("item")["title"]
n_users, n_items = ratings["user"].nunique(), ratings["item"].nunique()
density = len(ratings) / (n_users * n_items)
print(f"{len(ratings):,} ratings, {n_users} users, {n_items} items, density {density:.4f} (94% of the matrix is missing)")
print("rating counts 1..5:", np.bincount(ratings["rating"], minlength=6)[1:].tolist())
per_user = ratings.groupby("user").size()
per_item = ratings.groupby("item").size()
print(f"ratings per user: min {per_user.min()}, median {per_user.median():.0f}, max {per_user.max()}")
print(f"ratings per item: min {per_item.min()}, median {per_item.median():.0f}, max {per_item.max()}"
      f"; {int((per_item < 5).sum())} items have fewer than 5")


def time_split(df: pd.DataFrame, n_hold: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per user, sorted by time: last n_hold rows -> test, the n_hold before -> validation, rest -> train."""
    df = df.sort_values(["user", "ts", "item"], kind="stable")
    pos_from_end = df.groupby("user").cumcount(ascending=False)  # 0 = the user's latest rating
    test = df[pos_from_end < n_hold]
    val = df[(pos_from_end >= n_hold) & (pos_from_end < 2 * n_hold)]
    train = df[pos_from_end >= 2 * n_hold]
    return train, val, test


def random_split(df: pd.DataFrame, n_hold: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Same sizes as time_split, but each user's held-out rows are chosen at random."""
    order = np.random.default_rng(seed).permutation(len(df))
    df = df.iloc[order]
    pos = df.groupby("user").cumcount()
    return df[pos >= 2 * n_hold], df[(pos >= n_hold) & (pos < 2 * n_hold)], df[pos < n_hold]


train, val, test = time_split(ratings, N_HOLD)
print(f"time split: train {len(train):,}, validation {len(val):,}, test {len(test):,} ratings")
last_train = train.groupby("user")["ts"].max()
first_test = test.groupby("user")["ts"].min()
assert (last_train <= first_test.loc[last_train.index]).all()  # every test rating is no earlier than the user's training
print("every user's test ratings come at or after all of their training ratings: OK")
cold = ~test["item"].isin(train["item"])
print(f"test ratings of items never seen in training (cold items): {int(cold.sum())}")


def arrays(df: pd.DataFrame) -> tuple[IntArray, IntArray, FloatArray]:
    return df["user"].to_numpy(np.int64), df["item"].to_numpy(np.int64), df["rating"].to_numpy(np.float64)


def rmse(y: FloatArray, y_hat: FloatArray) -> float:
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


# %%
# Baselines: global mean, item mean ("popularity" for ratings), and the bias model mu + b_u + b_i.
section("baselines")


def bias_baseline(u: IntArray, i: IntArray, r: FloatArray, lam_i: float, lam_u: float
                  ) -> tuple[float, FloatArray, FloatArray]:
    """Damped means: b_i = sum(r - mu) / (lam_i + n_i), then b_u = sum(r - mu - b_i) / (lam_u + n_u)."""
    mu = float(r.mean())
    b_i = np.bincount(i, weights=r - mu, minlength=n_items) / (lam_i + np.bincount(i, minlength=n_items))
    b_u = np.bincount(u, weights=r - mu - b_i[i], minlength=n_users) / (lam_u + np.bincount(u, minlength=n_users))
    return mu, b_u, b_i


def fit_baselines(tr: pd.DataFrame) -> dict[str, tuple]:
    u, i, r = arrays(tr)
    mu = float(r.mean())
    item_mean = np.bincount(i, weights=r, minlength=n_items) / np.maximum(np.bincount(i, minlength=n_items), 1)
    item_mean[np.bincount(i, minlength=n_items) == 0] = mu  # unseen items fall back to the global mean
    return {"mu": mu, "item_mean": item_mean, "bias": bias_baseline(u, i, r, lam_i=25.0, lam_u=10.0)}


def predict_baselines(fit: dict[str, tuple], u: IntArray, i: IntArray) -> dict[str, FloatArray]:
    mu_b, b_u, b_i = fit["bias"]
    return {"global mean": np.full(len(u), fit["mu"]), "item mean": fit["item_mean"][i],
            "bias model": np.clip(mu_b + b_u[u] + b_i[i], 1, 5)}


base = fit_baselines(train)
u_te, i_te, r_te = arrays(test)
for name, pred in predict_baselines(base, u_te, i_te).items():
    print(f"{name:<12} test RMSE {rmse(r_te, pred):.4f}")


# %%
# User-based collaborative filtering with cosine similarity on mean-centred ratings (A1.5).
section("from scratch: user-based collaborative filtering")


def rating_matrix(tr: pd.DataFrame) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Dense user-item matrix (n_users, n_items) with 0 where missing, and the mask of observed cells."""
    u, i, r = arrays(tr)
    R = np.zeros((n_users, n_items))
    R[u, i] = r
    M = np.zeros((n_users, n_items), dtype=bool)
    M[u, i] = True
    return R, M


def user_cf_predict(R: FloatArray, M: npt.NDArray[np.bool_], u: IntArray, i: IntArray, k: int) -> FloatArray:
    """Predict r_ui = mean_u + sum_v s_uv c_vi / sum_v |s_uv| over the k most similar users v who rated i."""
    user_mean = R.sum(axis=1) / np.maximum(M.sum(axis=1), 1)  # (n_users,)
    C = np.where(M, R - user_mean[:, None], 0.0)  # (n_users, n_items) centred, 0 = missing
    S = cosine_rows(C)  # (n_users, n_users)
    np.fill_diagonal(S, 0.0)
    out = np.empty(len(u))
    for row, (uu, ii) in enumerate(zip(u, i)):
        raters = np.flatnonzero(M[:, ii])  # users who rated item ii
        if len(raters) == 0:
            out[row] = user_mean[uu]
            continue
        sims = S[uu, raters]
        top = raters[np.argsort(-sims, kind="stable")[:k]]
        s = S[uu, top]
        den = np.abs(s).sum()
        out[row] = user_mean[uu] + (s @ C[top, ii]) / den if den > 0 else user_mean[uu]
    return np.clip(out, 1, 5)


R_tr, M_tr = rating_matrix(train)
C_tr = np.where(M_tr, R_tr - (R_tr.sum(1) / M_tr.sum(1))[:, None], 0.0)
S_tr = cosine_rows(C_tr)
print(f"similarity matrix {S_tr.shape}; mean off-diagonal similarity {S_tr[~np.eye(n_users, dtype=bool)].mean():.4f}")
pred_cf = user_cf_predict(R_tr, M_tr, u_te, i_te, K_NEIGHBOURS)
print(f"user-based CF (k={K_NEIGHBOURS}) test RMSE {rmse(r_te, pred_cf):.4f}")
for k_try in [5, 20, 100]:
    print(f"  with k={k_try:<3} test RMSE {rmse(r_te, user_cf_predict(R_tr, M_tr, u_te, i_te, k_try)):.4f}")


# %%
# Matrix factorization by mini-batch SGD with biases and L2, from scratch.
section("from scratch: matrix factorization by SGD")


def init_mf(k: int, seed: int) -> dict[str, FloatArray]:
    g = np.random.default_rng(seed)
    return {"b_u": np.zeros(n_users), "b_i": np.zeros(n_items),
            "P": g.normal(0, 0.1, size=(n_users, k)), "Q": g.normal(0, 0.1, size=(n_items, k))}


def mf_predict(th: dict[str, FloatArray], mu: float, u: IntArray, i: IntArray) -> FloatArray:
    """r_hat = mu + b_u + b_i + p_u . q_i for each (u, i) pair -> (m,)."""
    return mu + th["b_u"][u] + th["b_i"][i] + np.sum(th["P"][u] * th["Q"][i], axis=1)


def mf_objective(th: dict[str, FloatArray], mu: float, u: IntArray, i: IntArray, r: FloatArray, lam: float) -> float:
    """J = sum (r - r_hat)^2 + lam * sum over ratings of (b_u^2 + b_i^2 + |p_u|^2 + |q_i|^2)."""
    e = r - mf_predict(th, mu, u, i)
    reg = th["b_u"][u] ** 2 + th["b_i"][i] ** 2 + np.sum(th["P"][u] ** 2, 1) + np.sum(th["Q"][i] ** 2, 1)
    return float(np.sum(e**2) + lam * np.sum(reg))


def mf_sgd_step(th: dict[str, FloatArray], mu: float, u: IntArray, i: IntArray, r: FloatArray,
                lr: float, lam: float) -> None:
    """One mini-batch: every rating applies its own SGD update, computed at the current parameters."""
    e = r - mf_predict(th, mu, u, i)  # (B,)
    Pu, Qi = th["P"][u], th["Q"][i]  # (B, k) each
    np.add.at(th["b_u"], u, lr * (e - lam * th["b_u"][u]))
    np.add.at(th["b_i"], i, lr * (e - lam * th["b_i"][i]))
    np.add.at(th["P"], u, lr * (e[:, None] * Qi - lam * Pu))
    np.add.at(th["Q"], i, lr * (e[:, None] * Pu - lam * Qi))


def train_mf(tr: pd.DataFrame, va: pd.DataFrame | None, k: int, lam: float, epochs: int, seed: int
             ) -> tuple[dict[str, FloatArray], float, list[float], list[float]]:
    u, i, r = arrays(tr)
    mu = float(r.mean())
    th = init_mf(k, seed)
    g = np.random.default_rng(seed)
    tr_curve, va_curve = [], []
    for _ in range(epochs):
        order = g.permutation(len(r))
        for s in range(0, len(r), BATCH):
            b = order[s:s + BATCH]
            mf_sgd_step(th, mu, u[b], i[b], r[b], LR, lam)
        tr_curve.append(rmse(r, mf_predict(th, mu, u, i)))
        if va is not None:
            uv, iv, rv = arrays(va)
            va_curve.append(rmse(rv, np.clip(mf_predict(th, mu, uv, iv), 1, 5)))
    return th, mu, tr_curve, va_curve


# Gradient check (A4.1): the SGD update direction is minus the gradient of J (up to the factor 2).
u_s, i_s, r_s = arrays(train.iloc[:50])
th_chk = init_mf(3, SEED)
th_chk["b_u"] += 0.1
e_s = r_s - mf_predict(th_chk, 3.5, u_s, i_s)
grad_P = np.zeros_like(th_chk["P"])
np.add.at(grad_P, u_s, -2 * e_s[:, None] * th_chk["Q"][i_s] + 2 * 0.05 * th_chk["P"][u_s])
num_grad = np.zeros_like(th_chk["P"])
h = 1e-6
for uu in np.unique(u_s):
    for f in range(3):
        th_chk["P"][uu, f] += h
        jp = mf_objective(th_chk, 3.5, u_s, i_s, r_s, 0.05)
        th_chk["P"][uu, f] -= 2 * h
        jm = mf_objective(th_chk, 3.5, u_s, i_s, r_s, 0.05)
        th_chk["P"][uu, f] += h
        num_grad[uu, f] = (jp - jm) / (2 * h)
np.testing.assert_allclose(grad_P, num_grad, rtol=1e-5, atol=1e-6)  # central differences in float64
print("analytic gradient of J w.r.t. P matches central differences: OK")

results_val = {}
for lam in LAMBDAS:
    _, _, tr_c, va_c = train_mf(train, val, N_FACTORS, lam, EPOCHS, SEED)
    best_ep = int(np.argmin(va_c)) + 1
    results_val[lam] = (min(va_c), best_ep, tr_c, va_c)
    print(f"lambda={lam:<5} train RMSE after {EPOCHS} epochs {tr_c[-1]:.4f}; best validation RMSE {min(va_c):.4f} at epoch {best_ep}")
lam_best = min(results_val, key=lambda l: results_val[l][0])
ep_best = results_val[lam_best][1]
print(f"chosen on validation: lambda={lam_best}, epochs={ep_best}")

# refit on train + validation with the chosen settings, then score the untouched test set once
trval = pd.concat([train, val])
th, mu_tv, _, _ = train_mf(trval, None, N_FACTORS, lam_best, ep_best, SEED)
pred_mf = np.clip(mf_predict(th, mu_tv, u_te, i_te), 1, 5)
print(f"matrix factorization (k={N_FACTORS}) test RMSE {rmse(r_te, pred_mf):.4f}")

# %%
# With scikit-learn / SciPy: cosine similarity, and why a plain truncated SVD of the filled-in matrix is not MF.
section("with scikit-learn and SciPy")
np.testing.assert_allclose(S_tr, cosine_similarity(C_tr), rtol=1e-10, atol=1e-12)
print("from-scratch user similarity matches sklearn.metrics.pairwise.cosine_similarity: OK")

R_tv, M_tv = rating_matrix(trval)
base_tv = fit_baselines(trval)
mu_b, b_u_tv, b_i_tv = base_tv["bias"]
B_tv = mu_b + b_u_tv[:, None] + b_i_tv[None, :]  # (n_users, n_items) bias-model predictions
for fill_name, filled in [("missing = 0", np.where(M_tv, R_tv, 0.0)),
                          ("missing = bias model", np.where(M_tv, R_tv, B_tv))]:
    centre = 0.0 if fill_name == "missing = 0" else B_tv
    U_, s_, Vt_ = svds(filled - centre, k=N_FACTORS, random_state=SEED)  # best rank-k fit of ALL cells (A3.2)
    approx = U_ @ np.diag(s_) @ Vt_ + centre
    print(f"truncated SVD, {fill_name:<21} test RMSE {rmse(r_te, np.clip(approx[u_te, i_te], 1, 5)):.4f}")
print("(SVD fits every cell, including the 94% that were invented; MF fits only the observed ratings)")

# %%
# Random split versus time split: the same models, scored both ways.
section("random split versus time split")
table = {}
for split_name, (tr_s, va_s, te_s) in [("random", random_split(ratings, N_HOLD, SEED)),
                                       ("time", (train, val, test))]:
    u_s2, i_s2, r_s2 = arrays(te_s)
    fit_s = fit_baselines(pd.concat([tr_s, va_s]))
    scores = {n: rmse(r_s2, p) for n, p in predict_baselines(fit_s, u_s2, i_s2).items()}
    R_s, M_s = rating_matrix(pd.concat([tr_s, va_s]))
    scores["user CF"] = rmse(r_s2, user_cf_predict(R_s, M_s, u_s2, i_s2, K_NEIGHBOURS))
    th_s, mu_s, _, _ = train_mf(pd.concat([tr_s, va_s]), None, N_FACTORS, lam_best, ep_best, SEED)
    scores["MF"] = rmse(r_s2, np.clip(mf_predict(th_s, mu_s, u_s2, i_s2), 1, 5))
    table[split_name] = scores
print(f"{'model':<12} {'random':>8} {'time':>8}")
for name in table["time"]:
    print(f"{name:<12} {table['random'][name]:>8.4f} {table['time'][name]:>8.4f}")
print("(baselines and CF here use train + validation, like MF, so the columns are comparable)")

# %%
# Implicit feedback: "watched" = 1, sampled negatives, logistic loss; judged by top-10 recall, not RMSE.
section("implicit feedback: sampled negatives")


def train_implicit(pos_u: IntArray, pos_i: IntArray, seen: npt.NDArray[np.bool_], k: int, lam: float,
                   epochs: int, seed: int) -> dict[str, FloatArray]:
    """Logistic MF: P(watch) = sigmoid(b_i + p_u . q_i); each positive is paired with N_NEG random unseen items."""
    g = np.random.default_rng(seed)
    th = init_mf(k, seed)
    for _ in range(epochs):
        neg_u = np.repeat(pos_u, N_NEG)
        neg_i = g.integers(0, n_items, size=len(neg_u))
        keep = ~seen[neg_u, neg_i]  # drop "negatives" the user actually watched
        uu = np.r_[pos_u, neg_u[keep]]
        ii = np.r_[pos_i, neg_i[keep]]
        yy = np.r_[np.ones(len(pos_u)), np.zeros(int(keep.sum()))]
        order = g.permutation(len(yy))
        for s in range(0, len(yy), BATCH):
            b = order[s:s + BATCH]
            ub, ib, yb = uu[b], ii[b], yy[b]
            z = th["b_i"][ib] + np.sum(th["P"][ub] * th["Q"][ib], axis=1)
            gz = 1 / (1 + np.exp(-z)) - yb  # d(log loss)/dz = sigmoid(z) - y (B2.1)
            Pu, Qi = th["P"][ub], th["Q"][ib]
            np.add.at(th["b_i"], ib, -LR_IMPLICIT * gz)
            np.add.at(th["P"], ub, -LR_IMPLICIT * (gz[:, None] * Qi + lam * Pu))
            np.add.at(th["Q"], ib, -LR_IMPLICIT * (gz[:, None] * Pu + lam * Qi))
    return th


def recall_at_k(scores: FloatArray, seen: npt.NDArray[np.bool_], test_u: IntArray, test_i: IntArray, k: int) -> float:
    """Share of each user's held-out items that appear in their top k unseen items, averaged over users."""
    s = np.where(seen, -np.inf, scores)  # never recommend what the user already has
    top = np.argpartition(-s, k, axis=1)[:, :k]  # (n_users, k), unordered
    hit = np.zeros((n_users, n_items), dtype=bool)
    hit[np.arange(n_users)[:, None], top] = True
    hits = np.bincount(test_u, weights=hit[test_u, test_i], minlength=n_users)
    n_test = np.bincount(test_u, minlength=n_users)
    return float(np.mean(hits[n_test > 0] / n_test[n_test > 0]))


seen_tv = M_tv  # everything in train + validation counts as "already watched"
pop = M_tv.sum(axis=0).astype(float)  # how many users watched each item
scores_pop = np.tile(pop, (n_users, 1))
scores_explicit = th["b_i"][None, :] + th["b_u"][:, None] + th["P"] @ th["Q"].T + mu_tv
pos_u, pos_i, _ = arrays(trval)
th_imp = train_implicit(pos_u, pos_i, seen_tv, N_FACTORS, lam=0.01, epochs=IMPLICIT_EPOCHS, seed=SEED)
scores_implicit = th_imp["b_i"][None, :] + th_imp["P"] @ th_imp["Q"].T
for name, sc in [("popularity (most watched)", scores_pop), ("explicit MF (highest predicted rating)", scores_explicit),
                 ("implicit MF (sampled negatives)", scores_implicit)]:
    print(f"{name:<40} recall@{TOP_K} {recall_at_k(sc, seen_tv, u_te, i_te, TOP_K):.4f}")
u0 = int(u_te[0])
for name, sc in [("explicit MF", scores_explicit), ("implicit MF", scores_implicit)]:
    s0 = np.where(seen_tv[u0], -np.inf, sc[u0])
    top3 = np.argsort(-s0)[:3]
    print(f"user {u0 + 1}, top 3 by {name}: " + "; ".join(f"{titles[j]} ({int(pop[j])} watchers)" for j in top3))

# %%
# Key plot: test RMSE of every model under a random split and under the time split.
section("key plot: random versus time split")
fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
names_m = list(table["time"])
x = np.arange(len(names_m))
ax.bar(x - 0.2, [table["random"][m] for m in names_m], width=0.4, label="random split", hatch="//", edgecolor="black")
ax.bar(x + 0.2, [table["time"][m] for m in names_m], width=0.4, label="time split (last 5 per user)", edgecolor="black")
ax.set_xticks(x, names_m)
ax.set_ylabel("test RMSE (stars, lower is better)")
ax.set_title("A random split flatters every model; MF still wins on the time split")
ax.set_ylim(0, max(max(t.values()) for t in table.values()) * 1.15)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(IMG / "b6_1_random_vs_time_split.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved lessons/B/img/b6_1_random_vs_time_split.png")

# training curves for the three lambdas, a scratch figure
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
for lam, (_, _, tr_c, va_c) in results_val.items():
    ax.plot(range(1, EPOCHS + 1), tr_c, linestyle="--", label=f"train, lambda={lam}")
    ax.plot(range(1, EPOCHS + 1), va_c, label=f"validation, lambda={lam}")
ax.set_xlabel("epoch")
ax.set_ylabel("RMSE (stars)")
ax.set_title("Small lambda: train RMSE keeps falling, validation turns up")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "mf_curves.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved outputs/b6_1/mf_curves.png")

# %%
# Try this (not graded):
# 1. Set N_FACTORS = 2 and plot every item's q_i. Which films sit at the extremes of each axis?
# 2. Use a single global random 10% test split instead of 5 per user. Which users dominate the RMSE now?
# 3. In train_implicit, sample negatives in proportion to popularity instead of uniformly. What happens to recall@10?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
