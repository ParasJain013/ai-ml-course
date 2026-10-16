# %% [markdown]
# # A1.5 Vectors — lab
# Lesson: lessons/A/a1_5_vectors.md
# Run: uv run python labs/A/a1_5_vectors.py   (CPU, defaults: ~5 s)
# GPU: no

# %%
from __future__ import annotations

import math
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

FloatArray = npt.NDArray[np.float64]
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
OUT = ROOT / "outputs" / "a1_5"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "A" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; the knob sizes the random-vector agreement check.
GPU_SCALE = False
N_RANDOM = 100_000 if GPU_SCALE else 2_000  # random vectors in the library check
D_RANDOM = 384  # a typical sentence-embedding width (VERIFY (as of 2026-10) for current models)

# %%
# Data: hand-made 4-feature "word vectors". Real embeddings are learned (E1.3), not typed in.
section("data")
FEATURES = ["royal", "male", "female", "food"]
WORDS = ["king", "queen", "man", "woman", "prince", "apple", "banana", "bread"]
E = np.array(
    [
        [0.9, 0.8, 0.1, 0.0],  # king
        [0.9, 0.1, 0.8, 0.0],  # queen
        [0.1, 0.9, 0.1, 0.0],  # man
        [0.1, 0.1, 0.9, 0.0],  # woman
        [0.7, 0.7, 0.1, 0.0],  # prince
        [0.0, 0.0, 0.0, 0.9],  # apple
        [0.0, 0.0, 0.1, 0.8],  # banana
        [0.1, 0.0, 0.0, 0.7],  # bread
    ]
)  # (8, 4): one word per row
print("embedding matrix:", E.shape, " features:", FEATURES)

# %%
# Worked example by hand: a = (1, 2, 2), b = (2, 0, 1).
section("worked example")
a_list = [1.0, 2.0, 2.0]
b_list = [2.0, 0.0, 1.0]


def add(a: list[float], b: list[float]) -> list[float]:
    """Elementwise sum of two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(f"lengths differ: {len(a)} vs {len(b)}")
    return [x + y for x, y in zip(a, b, strict=True)]


def scale(c: float, a: list[float]) -> list[float]:
    """Multiply every entry of a by the scalar c."""
    return [c * x for x in a]


def dot(a: list[float], b: list[float]) -> float:
    """Dot product: sum of elementwise products."""
    if len(a) != len(b):
        raise ValueError(f"lengths differ: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b, strict=True))


def norm_l2(a: list[float]) -> float:
    """Euclidean length: sqrt of the sum of squares."""
    return math.sqrt(dot(a, a))


def norm_l1(a: list[float]) -> float:
    """L1 length: sum of absolute values."""
    return sum(abs(x) for x in a)


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity a.b / (|a| |b|); undefined for a zero vector."""
    na, nb = norm_l2(a), norm_l2(b)
    if na == 0.0 or nb == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return dot(a, b) / (na * nb)


print("a + b =", add(a_list, b_list), "  2a =", scale(2.0, a_list))
print("a . b =", dot(a_list, b_list))
print("|a|_2 =", norm_l2(a_list), "  |b|_2 =", round(norm_l2(b_list), 4))
print("|a|_1 =", norm_l1(a_list), "  |b|_1 =", norm_l1(b_list))
cos_ab = cosine(a_list, b_list)
print("cos(a, b) =", round(cos_ab, 4), "  angle =", round(math.degrees(math.acos(cos_ab)), 2), "degrees")

# %%
# From scratch: pure Python against NumPy, on the worked example and on random vectors.
section("from scratch vs numpy")
a, b = np.array(a_list), np.array(b_list)  # (3,), (3,)
checks = {
    "dot": (dot(a_list, b_list), float(a @ b)),
    "L2 norm": (norm_l2(a_list), float(np.linalg.norm(a))),
    "L1 norm": (norm_l1(a_list), float(np.linalg.norm(a, ord=1))),
    "cosine": (cos_ab, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))),
}
for name, (mine, theirs) in checks.items():
    np.testing.assert_allclose(mine, theirs, rtol=1e-6, atol=1e-8)
    print(f"{name:<8} pure Python {mine:.4f}  numpy {theirs:.4f}  OK")

# %%
# The unit-vector identity: for |a| = |b| = 1, |a - b|^2 = 2 - 2 cos(a, b).
section("unit vectors: distance and cosine agree")
a_hat, b_hat = a / np.linalg.norm(a), b / np.linalg.norm(b)
lhs = float(np.sum((a_hat - b_hat) ** 2))
rhs = 2.0 - 2.0 * float(a_hat @ b_hat)
print("|a_hat - b_hat|^2 =", round(lhs, 4), "  2 - 2 cos =", round(rhs, 4))
np.testing.assert_allclose(lhs, rhs, rtol=1e-6, atol=1e-8)
print("identity holds: OK")

# %%
# Brute-force nearest-neighbour search: what a vector database does, without the index.
section("nearest neighbours")


def cosine_matrix(Q: FloatArray, M: FloatArray) -> FloatArray:
    """Cosine similarity of every query row with every row of M. Q (q, d), M (m, d) -> (q, m)."""
    Q_hat = Q / np.linalg.norm(Q, axis=1, keepdims=True)  # (q, d), rows of length 1
    M_hat = M / np.linalg.norm(M, axis=1, keepdims=True)  # (m, d)
    return Q_hat @ M_hat.T  # (q, d) @ (d, m) -> (q, m): all the dot products


def top_k(query: FloatArray, M: FloatArray, k: int, exclude: set[int] | None = None) -> list[tuple[int, float]]:
    """Indices and cosine scores of the k rows of M most similar to query (d,)."""
    scores = cosine_matrix(query[None, :], M)[0]  # (m,)
    order = np.argsort(-scores, kind="stable")  # highest score first
    picked = [int(i) for i in order if exclude is None or int(i) not in exclude]
    return [(i, float(scores[i])) for i in picked[:k]]


S = cosine_matrix(E, E)  # (8, 8)
print("cosine(king, queen) =", round(S[0, 1], 4), "  cosine(king, apple) =", round(S[0, 5], 4))
for word in ["queen", "banana"]:
    i = WORDS.index(word)
    hits = top_k(E[i], E, k=3, exclude={i})
    print(f"top 3 for {word!r}:", [(WORDS[j], round(s, 4)) for j, s in hits])
analogy = E[0] - E[2] + E[3]  # king - man + woman, (4,)
print("king - man + woman =", (np.round(analogy, 4) + 0.0).tolist())
hits = top_k(analogy, E, k=1, exclude={0, 2, 3})
print("nearest (excluding the three inputs):", [(WORDS[j], round(s, 4)) for j, s in hits])

# %%
# With scikit-learn and SciPy: the same numbers from the libraries.
section("with scikit-learn and scipy")
from scipy.spatial.distance import cosine as scipy_cosine_distance  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

np.testing.assert_allclose(S, cosine_similarity(E), rtol=1e-6, atol=1e-8)
print("cosine_matrix matches sklearn cosine_similarity: OK")
print("scipy cosine *distance*(a, b) =", round(scipy_cosine_distance(a, b), 4), " = 1 - cos:", round(1 - cos_ab, 4))
R = rng.normal(size=(N_RANDOM, D_RANDOM))  # (N_RANDOM, 384)
q = rng.normal(size=(5, D_RANDOM))  # (5, 384)
np.testing.assert_allclose(cosine_matrix(q, R), cosine_similarity(q, R), rtol=1e-6, atol=1e-8)
print(f"random {N_RANDOM} x {D_RANDOM}: matches sklearn: OK")
nn = NearestNeighbors(n_neighbors=4, metric="cosine").fit(E)
dist, idx = nn.kneighbors(E[[1]])  # neighbours of queen (includes queen itself at distance 0)
print("sklearn neighbours of 'queen':", [(WORDS[j], round(float(1 - d_), 4)) for j, d_ in zip(idx[0], dist[0], strict=True)])

# %%
# Euclidean distance and cosine can disagree when lengths differ.
section("cosine vs euclidean")
q2 = np.array([1.0, 1.0])
u = np.array([5.0, 5.0])  # same direction, far away
v = np.array([1.0, 0.0])  # different direction, close by
for name, vec in [("u", u), ("v", v)]:
    print(
        f"{name}: euclidean {np.linalg.norm(q2 - vec):.4f}  cosine "
        f"{q2 @ vec / (np.linalg.norm(q2) * np.linalg.norm(vec)):.4f}"
    )

# %%
# Key plot: the two metrics pick different neighbours.
section("key plot")
fig, ax = plt.subplots(figsize=(5.5, 5), dpi=120)
for vec, name, style in [(q2, "query q (1, 1)", "-"), (u, "u (5, 5)", "--"), (v, "v (1, 0)", ":")]:
    ax.annotate("", xy=vec, xytext=(0, 0), arrowprops={"arrowstyle": "->", "linestyle": style, "lw": 1.6})
    ax.plot(*vec, marker="o", linestyle="none", label=name)
ax.plot([q2[0], v[0]], [q2[1], v[1]], color="tab:green", linewidth=3, alpha=0.5, label="Euclidean: v is closest (1.0)")
ax.text(1.6, 3.8, "Cosine: u is closest\n(angle 0°, cos = 1.0)", fontsize=9)
ax.text(1.25, 0.2, "angle 45°, cos = 0.707", fontsize=9)
ax.set_xlim(-0.3, 5.6)
ax.set_ylim(-0.3, 5.6)
ax.set_aspect("equal")
ax.set_xlabel("feature 1 (unitless)")
ax.set_ylabel("feature 2 (unitless)")
ax.set_title("Cosine compares direction; Euclidean compares position")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
path = IMG / "a1_5_cosine_vs_euclidean.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Add a word ("princess") to E and WORDS. Does prince - man + woman find it?
# 2. Scale every row of E by a random positive number. Which results change: cosine or Euclidean?
# 3. Time top_k on R (2,000 x 384). How would it scale to 10 million vectors? (Part I: indexes.)

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
