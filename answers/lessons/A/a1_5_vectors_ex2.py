# %% [markdown]
# # A1.5 Vectors — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a1_5_vectors_ex2.py   (CPU, ~3 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors

FloatArray = npt.NDArray[np.float64]
SEED = 0
T0 = time.perf_counter()


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
a, b, c = np.array([3.0, 4.0, 0.0]), np.array([4.0, 3.0, 0.0]), np.array([0.0, 0.0, 2.0])


def cos(u: FloatArray, v: FloatArray) -> float:
    """Cosine similarity of two (d,) vectors."""
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


for b_vec, label in [(b, "b"), (10 * b, "10b")]:
    print(
        f"{label:>3}: a.{label}={a @ b_vec:g}  |{label}|_2={np.linalg.norm(b_vec):g}  "
        f"|{label}|_1={np.linalg.norm(b_vec, ord=1):g}  cos={cos(a, b_vec):.4f}  "
        f"|a-{label}|={np.linalg.norm(a - b_vec):.4f}"
    )
print(f"a.c={a @ c:g}  |c|_2={np.linalg.norm(c):g}  cos(a,c)={cos(a, c):.4f}  |a-c|={np.linalg.norm(a - c):.4f}")
a_hat, b_hat = a / np.linalg.norm(a), b / np.linalg.norm(b)
print("|a_hat - b_hat|^2 =", round(float(np.sum((a_hat - b_hat) ** 2)), 4), " 2 - 2cos =", round(2 - 2 * cos(a, b), 4))

# %%
section("exercise 2: search")
FEATURES = ["price", "outdoor", "electronics", "kids", "kitchen"]
DOCS = [
    "budget tent", "luxury tent", "kids bike", "phone charger", "laptop", "toy robot",
    "chef knife", "blender", "camping stove", "tablet for kids", "garden hose", "rice cooker",
]
M = np.array(
    [
        [0.2, 0.9, 0.0, 0.1, 0.0],  # budget tent
        [0.9, 0.9, 0.0, 0.0, 0.0],  # luxury tent
        [0.3, 0.6, 0.0, 0.9, 0.0],  # kids bike
        [0.1, 0.0, 0.9, 0.0, 0.0],  # phone charger
        [0.9, 0.0, 0.9, 0.0, 0.0],  # laptop
        [0.2, 0.0, 0.5, 0.9, 0.0],  # toy robot
        [0.4, 0.0, 0.0, 0.0, 0.9],  # chef knife
        [0.4, 0.0, 0.6, 0.0, 0.9],  # blender
        [0.3, 0.9, 0.0, 0.0, 0.6],  # camping stove
        [0.5, 0.0, 0.9, 0.8, 0.0],  # tablet for kids
        [0.2, 0.9, 0.0, 0.0, 0.0],  # garden hose
        [0.3, 0.0, 0.5, 0.0, 0.9],  # rice cooker
    ]
)  # (12, 5)


def search(query: FloatArray, M: FloatArray, k: int) -> list[tuple[int, float]]:
    """Top-k rows of M by cosine similarity to query (d,), best first. M (m, d)."""
    q_hat = query / np.linalg.norm(query)  # (d,)
    M_hat = M / np.linalg.norm(M, axis=1, keepdims=True)  # (m, d)
    scores = M_hat @ q_hat  # (m,)
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in order]


query = np.array([0.1, 0.8, 0.0, 0.0, 0.5])  # "cheap outdoor cooking"
hits = search(query, M, k=3)
print("query 'cheap outdoor cooking':", [(DOCS[i], round(s, 4)) for i, s in hits])
nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(M)
dist, idx = nn.kneighbors(query[None, :])
assert [i for i, _ in hits] == idx[0].tolist()
np.testing.assert_allclose([s for _, s in hits], 1 - dist[0], rtol=1e-6, atol=1e-8)
print("matches NearestNeighbors(metric='cosine'): OK")

# %%
section("exercise 2: dot product vs cosine")
query2 = np.array([0.1, 0.9, 0.0, 0.0, 0.0])  # "cheap outdoor"
dots = M @ query2  # (12,)
coss = (M / np.linalg.norm(M, axis=1, keepdims=True)) @ (query2 / np.linalg.norm(query2))
top_dot, top_cos = int(np.argmax(dots)), int(np.argmax(coss))
print(f"raw dot top-1: {DOCS[top_dot]!r} dot={dots[top_dot]:.4f} |doc|={np.linalg.norm(M[top_dot]):.4f}")
print(f"cosine  top-1: {DOCS[top_cos]!r} cos={coss[top_cos]:.4f} |doc|={np.linalg.norm(M[top_cos]):.4f}")
print(f"luxury tent: dot={dots[1]:.4f} cos={coss[1]:.4f};  garden hose: dot={dots[10]:.4f} cos={coss[10]:.4f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
