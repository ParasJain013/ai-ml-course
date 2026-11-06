# %% [markdown]
# # B6.2 Ranking metrics — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b6_2_ranking_metrics_ex2.py   (CPU, ~30 s)
# Needs data/ml-100k/ (run labs/B/b6_1_recommender_systems.py once first; it downloads the data).

# %%
from __future__ import annotations

import contextlib
import hashlib
import io
import time
import urllib.request
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import ndcg_score

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
SEED = 0
T0 = time.perf_counter()


def repo_root() -> Path:
    here = Path(globals().get("__file__", Path.cwd())).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("Run from inside the ai-ml-course folder.")


ROOT = repo_root()
DATA = ROOT / "data"


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE only trains the retriever longer and retrieves more candidates.
GPU_SCALE = False
N_FACTORS = 64 if GPU_SCALE else 20
EPOCHS = 30 if GPU_SCALE else 20
N_CANDIDATES = 300 if GPU_SCALE else 100  # stage 1 keeps this many items per user
N_HOLD = 5  # per user: last 5 interactions = test, the 5 before = the period that labels the reranker
K = 10  # the length of the list a user sees
N_NEG, LR, BATCH, LAM = 4, 0.05, 256, 0.01  # implicit MF as in B6.1
N_THREADS = 1  # deterministic LightGBM results on every machine
N_BOOT = 1000

ML_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML_ZIP_MD5 = "0e33842e24a9c977be4e0107933c0723"
U_DATA_MD5 = "6e47046882bad158b0efbb84cd5cb987"


# %%
# from labs/B/b6_1_recommender_systems.py
def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_movielens() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download MovieLens 100K once into data/ml-100k/ and return (ratings, items with genre columns)."""
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
    genre_cols = [f"g{j}" for j in range(19)]
    items = pd.read_csv(folder / "u.item", sep="|", encoding="latin-1", header=None,
                        names=["item", "title", "release", "video", "url", *genre_cols])
    print(f"MovieLens 100K from {folder.relative_to(ROOT).as_posix()}/ (source: GroupLens, {ML_URL})")
    return ratings, items


# %%
# The metrics, from scratch. `ranked` is the list shown to one user (best first); `rel` maps item -> relevance.
def precision_at_k(ranked: list[int], rel: dict[int, float], k: int) -> float:
    """Share of the top k that is relevant (rel > 0). Divides by k even if the list is shorter."""
    return sum(rel.get(i, 0) > 0 for i in ranked[:k]) / k


def recall_at_k(ranked: list[int], rel: dict[int, float], k: int) -> float:
    """Share of the user's relevant items that appear in the top k."""
    n_rel = sum(v > 0 for v in rel.values())
    return sum(rel.get(i, 0) > 0 for i in ranked[:k]) / n_rel if n_rel else 0.0


def average_precision_at_k(ranked: list[int], rel: dict[int, float], k: int) -> float:
    """Mean of precision@j over the positions j <= k that hold a relevant item, divided by min(k, n_relevant)."""
    n_rel = sum(v > 0 for v in rel.values())
    hits, total = 0, 0.0
    for j, i in enumerate(ranked[:k], start=1):
        if rel.get(i, 0) > 0:
            hits += 1
            total += hits / j
    return total / min(k, n_rel) if n_rel else 0.0


def dcg_at_k(gains: list[float], k: int) -> float:
    """Discounted cumulative gain: sum of gain_j / log2(j + 1) over positions j = 1..k."""
    g = np.asarray(gains[:k], dtype=float)
    return float(np.sum(g / np.log2(np.arange(2, len(g) + 2))))


def ndcg_at_k(ranked: list[int], rel: dict[int, float], k: int) -> float:
    """DCG of the list divided by the DCG of the best possible list built from ALL the user's relevant items."""
    ideal = dcg_at_k(sorted(rel.values(), reverse=True), k)
    return dcg_at_k([rel.get(i, 0.0) for i in ranked], k) / ideal if ideal > 0 else 0.0



# %%
section("exercise 1 check")
rel1 = {"X": 1.0, "Y": 1.0, "Z": 1.0}
for name, lst in [("system 1", ["X", "a", "b", "Y", "c"]), ("system 2", ["a", "X", "Y", "b", "c"])]:
    print(f"{name}: P@5 {precision_at_k(lst, rel1, 5):.4f}  R@5 {recall_at_k(lst, rel1, 5):.4f}  "
          f"AP@5 {average_precision_at_k(lst, rel1, 5):.4f}  NDCG@5 {ndcg_at_k(lst, rel1, 5):.4f}")
rel_g = {"X": 3.0, "Y": 1.0, "Z": 1.0}
for name, lst in [("system 1", ["X", "a", "b", "Y", "c"]), ("system 2", ["a", "X", "Y", "b", "c"])]:
    print(f"{name}, graded (X=3, Y=Z=1): NDCG@5 {ndcg_at_k(lst, rel_g, 5):.4f}")

# %%
# The lab's pipeline, copied from labs/B/b6_2_ranking_metrics.py; its own printing is silenced.
_quiet = contextlib.redirect_stdout(io.StringIO())
_quiet.__enter__()
# %%
# Data: per-user time split into three blocks.
section("data and the three time blocks")
ratings, items = load_movielens()
ratings["user"] -= 1
ratings["item"] -= 1
items["item"] -= 1
n_users, n_items = ratings["user"].nunique(), ratings["item"].nunique()
genres = items[[f"g{j}" for j in range(19)]].to_numpy(float)  # (n_items, 19) one-hot genres
year = pd.to_datetime(items["release"], format="%d-%b-%Y", errors="coerce").dt.year.fillna(1990).to_numpy(float)
df = ratings.sort_values(["user", "ts", "item"], kind="stable")
pos_from_end = df.groupby("user").cumcount(ascending=False)
test = df[pos_from_end < N_HOLD]
label = df[(pos_from_end >= N_HOLD) & (pos_from_end < 2 * N_HOLD)]
base = df[pos_from_end >= 2 * N_HOLD]
print(f"base {len(base):,} | label block {len(label):,} | test {len(test):,} interactions (every rating = 'watched')")
print("stage 1 is trained on base; the reranker learns to predict the label block; everything is refitted on")
print("base + label block and judged on the test block, so no model ever sees the interactions it is scored on")


def mask_of(d: pd.DataFrame) -> BoolArray:
    M = np.zeros((n_users, n_items), dtype=bool)
    M[d["user"].to_numpy(), d["item"].to_numpy()] = True
    return M


def relevant_sets(d: pd.DataFrame) -> list[dict[int, float]]:
    out: list[dict[int, float]] = [{} for _ in range(n_users)]
    for u, i in zip(d["user"].to_numpy(), d["item"].to_numpy()):
        out[u][int(i)] = 1.0  # binary relevance: the user watched it
    return out


# %%
# Stage 1: the B6.1 implicit-feedback factor model, used as a retriever.
section("stage 1: retrieve candidates with the factor model")


def train_implicit(hist: pd.DataFrame, seed: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Logistic MF with sampled negatives (B6.1). Returns b_i (n_items,), P (n_users, k), Q (n_items, k)."""
    g = np.random.default_rng(seed)
    seen = mask_of(hist)
    pos_u, pos_i = hist["user"].to_numpy(), hist["item"].to_numpy()
    b_i, P, Q = np.zeros(n_items), g.normal(0, 0.1, (n_users, N_FACTORS)), g.normal(0, 0.1, (n_items, N_FACTORS))
    for _ in range(EPOCHS):
        neg_u = np.repeat(pos_u, N_NEG)
        neg_i = g.integers(0, n_items, size=len(neg_u))
        keep = ~seen[neg_u, neg_i]
        uu, ii = np.r_[pos_u, neg_u[keep]], np.r_[pos_i, neg_i[keep]]
        yy = np.r_[np.ones(len(pos_u)), np.zeros(int(keep.sum()))]
        order = g.permutation(len(yy))
        for s in range(0, len(yy), BATCH):
            b = order[s:s + BATCH]
            ub, ib, yb = uu[b], ii[b], yy[b]
            gz = 1 / (1 + np.exp(-(b_i[ib] + np.sum(P[ub] * Q[ib], 1)))) - yb
            Pu, Qi = P[ub], Q[ib]
            np.add.at(b_i, ib, -LR * gz)
            np.add.at(P, ub, -LR * (gz[:, None] * Qi + LAM * Pu))
            np.add.at(Q, ib, -LR * (gz[:, None] * Pu + LAM * Qi))
    return b_i, P, Q


def retrieve(scores: FloatArray, seen: BoolArray, n: int) -> IntArray:
    """Top-n unseen items per user, best first. scores (n_users, n_items) -> (n_users, n)."""
    s = np.where(seen, -np.inf, scores)
    top = np.argpartition(-s, n, axis=1)[:, :n]
    order = np.argsort(-np.take_along_axis(s, top, 1), axis=1, kind="stable")
    return np.take_along_axis(top, order, 1)


def mean_metric(lists: IntArray, rels: list[dict[int, float]], fn, k: int) -> float:
    vals = [fn(list(lists[u]), rels[u], k) for u in range(n_users) if rels[u]]
    return float(np.mean(vals))


b_i1, P1, Q1 = train_implicit(base, SEED)
seen_base = mask_of(base)
cand_label = retrieve(b_i1[None, :] + P1 @ Q1.T, seen_base, N_CANDIDATES)
rel_label = relevant_sets(label)
for k_r in [10, 50, N_CANDIDATES]:
    print(f"recall@{k_r:<3} of the retriever on the label block: {mean_metric(cand_label, rel_label, recall_at_k, k_r):.4f}")
print(f"(recall@{N_CANDIDATES} is the ceiling: the reranker can only reorder what stage 1 retrieved)")

# %%
# Stage 2: features for every (user, candidate) pair, computed only from the history before the block being predicted.
section("stage 2: features for the reranker")
FEATURES = ["mf_score", "mf_rank", "item_pop", "item_pop_recent", "item_mean_rating", "item_year",
            "user_activity", "genre_affinity", "year_gap"]


def pair_features(hist: pd.DataFrame, cands: IntArray, scores: FloatArray) -> FloatArray:
    """Feature matrix for all (u, c) pairs, rows ordered user by user. -> (n_users * n_cand, len(FEATURES))."""
    n_c = cands.shape[1]
    u_idx = np.repeat(np.arange(n_users), n_c)
    c_idx = cands.ravel()
    i_h, u_h, r_h = hist["item"].to_numpy(), hist["user"].to_numpy(), hist["rating"].to_numpy(float)
    pop = np.bincount(i_h, minlength=n_items).astype(float)
    recent = hist["ts"].to_numpy() >= hist["ts"].max() - 30 * 86400  # the last 30 days of the history
    pop_recent = np.bincount(i_h[recent], minlength=n_items).astype(float)
    mu = r_h.mean()
    mean_r = (np.bincount(i_h, weights=r_h, minlength=n_items) + 10 * mu) / (pop + 10)  # damped (B6.1)
    activity = np.bincount(u_h, minlength=n_users).astype(float)
    profile = np.zeros((n_users, genres.shape[1]))  # each user's genre mix
    np.add.at(profile, u_h, genres[i_h])
    profile /= np.maximum(np.linalg.norm(profile, axis=1, keepdims=True), 1e-12)
    g_norm = genres / np.maximum(np.linalg.norm(genres, axis=1, keepdims=True), 1e-12)
    affinity = np.sum(profile[u_idx] * g_norm[c_idx], axis=1)  # cosine of user genre mix and item genres
    user_year = np.bincount(u_h, weights=year[i_h], minlength=n_users) / np.maximum(activity, 1)
    return np.column_stack([
        scores[u_idx, c_idx], np.tile(np.arange(n_c), n_users).astype(float), pop[c_idx], pop_recent[c_idx],
        mean_r[c_idx], year[c_idx], activity[u_idx], affinity, year[c_idx] - user_year[u_idx]])


def recency_features(hist: pd.DataFrame, cands: IntArray, Q: FloatArray) -> FloatArray:
    """Two features MF cannot express: similarity to the user's 5 latest items, and item popularity in the
    30 days before the user's own last interaction. -> (n_users * n_cand, 2)."""
    n_c = cands.shape[1]
    u_idx, c_idx = np.repeat(np.arange(n_users), n_c), cands.ravel()
    h = hist.sort_values(["user", "ts", "item"], kind="stable")
    last5 = h.groupby("user").tail(5)
    recent_taste = np.zeros((n_users, Q.shape[1]))  # sum of the item vectors of each user's 5 latest items
    np.add.at(recent_taste, last5["user"].to_numpy(), Q[last5["item"].to_numpy()])
    recent_taste /= np.maximum(np.linalg.norm(recent_taste, axis=1, keepdims=True), 1e-12)
    Q_unit = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    sim_recent = np.sum(recent_taste[u_idx] * Q_unit[c_idx], axis=1)  # cosine (A1.5)
    t_user = h.groupby("user")["ts"].max().to_numpy()  # each user's "now"
    order = np.argsort(h["ts"].to_numpy(), kind="stable")
    ts_sorted, it_sorted = h["ts"].to_numpy()[order], h["item"].to_numpy()[order]
    pop_user_time = np.zeros((n_users, n_c))
    for u in range(n_users):  # interactions by anyone in the 30 days up to this user's last one
        lo = np.searchsorted(ts_sorted, t_user[u] - 30 * 86400)
        hi = np.searchsorted(ts_sorted, t_user[u], side="right")
        pop_user_time[u] = np.bincount(it_sorted[lo:hi], minlength=n_items)[cands[u]]
    return np.column_stack([sim_recent, pop_user_time.ravel()])


FEATURES_RECENT = ["sim_to_last5", "item_pop_at_user_time"]


def new_ranker() -> lgb.LGBMRanker:
    return lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=15,
                          min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          random_state=SEED, n_jobs=N_THREADS, deterministic=True, force_row_wise=True, verbose=-1)


def rerank(cands: IntArray, scores: FloatArray) -> IntArray:
    """Reorder each user's candidates by the reranker's scores, best first."""
    return np.take_along_axis(cands, np.argsort(-scores, axis=1, kind="stable"), 1)


X_static = pair_features(base, cand_label, b_i1[None, :] + P1 @ Q1.T)
X_all = np.column_stack([X_static, recency_features(base, cand_label, Q1)])
M_label = mask_of(label)
y_tr = M_label[np.repeat(np.arange(n_users), N_CANDIDATES), cand_label.ravel()].astype(int)
print(f"training pairs X {X_all.shape}, positives {y_tr.sum()} ({y_tr.mean():.2%}); one query group per user, {N_CANDIDATES} rows each")

# Choose the feature set on held-out USERS of the label block (20%), never on the test block.
val_users = np.sort(np.random.default_rng(SEED).permutation(n_users)[: n_users // 5])
is_val = np.isin(np.arange(n_users), val_users)
rows_tr, rows_va = np.repeat(~is_val, N_CANDIDATES), np.repeat(is_val, N_CANDIDATES)
rel_val = [rel_label[u] for u in val_users]


def mean_ndcg(lists: IntArray, rels: list[dict[int, float]]) -> float:
    return float(np.mean([ndcg_at_k(list(l), r, K) for l, r in zip(lists, rels) if r]))


print(f"choosing features on {len(val_users)} held-out users of the label block (NDCG@{K}):")
print(f"  retrieval order only               {mean_ndcg(cand_label[val_users], rel_val):.4f}")
val_score = {}
for name, cols in [("static features", len(FEATURES)), ("static + recency features", len(FEATURES) + 2)]:
    r_val = new_ranker().fit(X_all[rows_tr, :cols], y_tr[rows_tr], group=[N_CANDIDATES] * int((~is_val).sum()))
    s_val = r_val.predict(X_all[rows_va, :cols]).reshape(-1, N_CANDIDATES)
    val_score[cols] = mean_ndcg(rerank(cand_label[val_users], s_val), rel_val)
    print(f"  rerank, {name:<26} {val_score[cols]:.4f}")
n_cols = max(val_score, key=val_score.get)
names_used = (FEATURES + FEATURES_RECENT)[:n_cols]
print(f"chosen: {n_cols} features; refit the reranker on all {n_users} users of the label block")
ranker = new_ranker().fit(X_all[:, :n_cols], y_tr, group=[N_CANDIDATES] * n_users)
gain = ranker.booster_.feature_importance(importance_type="gain")
print("feature importance (share of total gain):")
for name, v in sorted(zip(names_used, gain / gain.sum()), key=lambda t: -t[1]):
    print(f"  {name:<22} {v:.3f}")

# %%
# The test: refit stage 1 on base + label block, retrieve, rerank, compare.
section("two-stage recommender on the test block")
hist_test = pd.concat([base, label])
b_i2, P2, Q2 = train_implicit(hist_test, SEED)
seen_test = mask_of(hist_test)
scores_test = b_i2[None, :] + P2 @ Q2.T
cand_test = retrieve(scores_test, seen_test, N_CANDIDATES)
X_te = np.column_stack([pair_features(hist_test, cand_test, scores_test), recency_features(hist_test, cand_test, Q2)])
s_rerank = ranker.predict(X_te[:, :n_cols]).reshape(n_users, N_CANDIDATES)
reranked = rerank(cand_test, s_rerank)
pop_scores = np.tile(np.bincount(hist_test["item"].to_numpy(), minlength=n_items).astype(float), (n_users, 1))
popular = retrieve(pop_scores, seen_test, K)
rel_test = relevant_sets(test)
systems = {"popularity": popular, "retrieval only (MF)": cand_test, "retrieve + rerank": reranked}
print(f"{'system':<22} {'P@10':>7} {'R@10':>7} {'MAP@10':>7} {'NDCG@10':>8}")
table = {}
for name, lists in systems.items():
    row = [mean_metric(lists, rel_test, fn, K) for fn in [precision_at_k, recall_at_k, average_precision_at_k, ndcg_at_k]]
    table[name] = row
    print(f"{name:<22} {row[0]:>7.4f} {row[1]:>7.4f} {row[2]:>7.4f} {row[3]:>8.4f}")
print(f"stage-1 recall@{N_CANDIDATES} on the test block (the ceiling): {mean_metric(cand_test, rel_test, recall_at_k, N_CANDIDATES):.4f}")

# paired bootstrap over users of the NDCG@10 gain (A6.2)
users = [u for u in range(n_users) if rel_test[u]]
nd_ret = np.array([ndcg_at_k(list(cand_test[u]), rel_test[u], K) for u in users])
nd_rr = np.array([ndcg_at_k(list(reranked[u]), rel_test[u], K) for u in users])
gb = np.random.default_rng(SEED)
boots = [np.mean((nd_rr - nd_ret)[gb.integers(0, len(users), len(users))]) for _ in range(N_BOOT)]
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"NDCG@10 gain from reranking: {np.mean(nd_rr - nd_ret):+.4f}, 95% bootstrap CI over users [{lo:+.4f}, {hi:+.4f}]")

# the same NDCG through sklearn, if you only hand it the 100 candidates: it cannot see relevant items stage 1 missed
y_true_c = np.array([[rel_test[u].get(int(i), 0.0) for i in cand_test[u]] for u in users])
sk = ndcg_score(y_true_c, s_rerank[users], k=K, ignore_ties=True)
print(f"sklearn ndcg_score on the candidate lists only: {sk:.4f} (higher: its ideal ignores relevant items not retrieved)")

_quiet.__exit__(None, None, None)

# %%
section("(a) MRR@10, catalogue coverage and popularity of what is recommended")
pop_hist = np.bincount(hist_test["item"].to_numpy(), minlength=n_items)


def mrr_at_k(ranked: list[int], rel: dict[int, float], k: int) -> float:
    """1 / position of the first relevant item in the top k, or 0 if there is none."""
    for j, i in enumerate(ranked[:k], start=1):
        if rel.get(i, 0) > 0:
            return 1.0 / j
    return 0.0


print(f"{'system':<22} {'MRR@10':>7} {'coverage':>9} {'mean watchers of recommended':>29}")
for name, lists in systems.items():
    top = lists[:, :K]
    mrr = mean_metric(lists, rel_test, mrr_at_k, K)
    coverage = len(np.unique(top)) / n_items
    print(f"{name:<22} {mrr:>7.4f} {coverage:>9.4f} {pop_hist[top].mean():>29.1f}")

# %%
section("(b) NDCG@10 by user activity")
activity = np.bincount(hist_test["user"].to_numpy(), minlength=n_users)
users_arr = np.array(users)
cuts = np.quantile(activity[users_arr], [1 / 3, 2 / 3])
group = np.digitize(activity[users_arr], cuts)  # 0 light, 1 medium, 2 heavy
for g_id, g_name in enumerate(["light", "medium", "heavy"]):
    sel = group == g_id
    a = activity[users_arr[sel]]
    print(f"{g_name:<7} users {sel.sum():>3} (history {a.min()}-{a.max()}): retrieval {nd_ret[sel].mean():.4f}  "
          f"rerank {nd_rr[sel].mean():.4f}  gain {(nd_rr - nd_ret)[sel].mean():+.4f}")

# %%
section("(c) ablation on the validation users: drop sim_to_last5")
keep_cols = [c for c in range(len(FEATURES) + 2) if c != len(FEATURES)]  # sim_to_last5 is column 9
for name, cols in [("all 11 features", list(range(len(FEATURES) + 2))), ("without sim_to_last5", keep_cols)]:
    r_val = new_ranker().fit(X_all[rows_tr][:, cols], y_tr[rows_tr], group=[N_CANDIDATES] * int((~is_val).sum()))
    s_val = r_val.predict(X_all[rows_va][:, cols]).reshape(-1, N_CANDIDATES)
    print(f"{name:<22} validation NDCG@10 {mean_ndcg(rerank(cand_label[val_users], s_val), rel_val):.4f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
