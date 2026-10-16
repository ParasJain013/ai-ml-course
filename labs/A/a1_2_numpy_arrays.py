# %% [markdown]
# # A1.2 NumPy arrays — lab
# Lesson: lessons/A/a1_2_numpy_arrays.md
# Run: uv run python labs/A/a1_2_numpy_arrays.py   (CPU, defaults: ~10 s)
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
OUT = ROOT / "outputs" / "a1_2"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "A" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; the larger sizes just make the timing plot's gap more dramatic.
GPU_SCALE = False
D_FEATURES = 16  # features per point in the timing runs
SIZES = [50, 100, 200, 400, 800, 1_600] if GPU_SCALE else [25, 50, 100, 200, 400]
LOOP_MAX = 400  # the pure-Python version is skipped above this size (it gets slow)

# %%
# Data: the hand-worked example and random data for the timing runs.
section("data")
A = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])  # (3, 2): 3 points, 2 features
B = np.array([[1.0, 1.0], [3.0, 0.0]])  # (2, 2): 2 points, 2 features
print("A shape:", A.shape, "B shape:", B.shape)
print("A dtype:", A.dtype, " ndim:", A.ndim, " size:", A.size)
X_big = rng.normal(size=(400, D_FEATURES))  # (400, 16)
print("X_big shape:", X_big.shape, "first row (rounded):", np.round(X_big[0, :4], 4).tolist())

# %%
# Indexing, views and copies.
section("indexing")
M = np.arange(12).reshape(3, 4)  # (3, 4): rows 0..2, columns 0..3
print(M)
print("M[1]        ->", M[1].tolist(), " shape", M[1].shape)
print("M[:, 1]     ->", M[:, 1].tolist(), " shape", M[:, 1].shape)
print("M[:, [1]]   ->", M[:, [1]].tolist(), " shape", M[:, [1]].shape)
print("M[M % 2 == 0] ->", M[M % 2 == 0].tolist())
view = M[0, :2]  # basic slice: a view onto M's memory
view[0] = 99
print("after view[0] = 99, M[0, 0] =", M[0, 0])
copy = M[[0], :2]  # fancy (list) index: always a copy
copy[0, 0] = -1
print("after copy[0, 0] = -1, M[0, 0] =", M[0, 0])

# %%
# Axes: which axis does a reduction remove?
section("axes")
M = np.arange(12).reshape(3, 4)  # fresh copy: the cell above wrote 99 into M[0, 0]
print("M.sum(axis=0) ->", M.sum(axis=0).tolist(), " shape", M.sum(axis=0).shape)
print("M.sum(axis=1) ->", M.sum(axis=1).tolist(), " shape", M.sum(axis=1).shape)
print("M.sum(axis=1, keepdims=True) shape", M.sum(axis=1, keepdims=True).shape)

# %%
# Broadcasting: predict the shape, then let NumPy confirm it.
section("broadcasting shape predictor")


def broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Predict the broadcast result shape of several array shapes, or raise ValueError."""
    ndim = max(len(s) for s in shapes)
    padded = [(1,) * (ndim - len(s)) + tuple(s) for s in shapes]  # left-pad with 1s
    out = []
    for axis in range(ndim):
        sizes = {p[axis] for p in padded} - {1}
        if len(sizes) > 1:
            raise ValueError(f"axis {axis}: sizes {sorted(sizes)} do not match and none is 1")
        out.append(sizes.pop() if sizes else 1)
    return tuple(out)


CASES = [((3, 1), (1, 2)), ((5, 1), (3,)), ((4, 3), (3,)), ((4, 3), (4,)), ((2, 1, 3), (4, 1))]
for s1, s2 in CASES:
    reason = ""
    try:
        mine = str(broadcast_shape(s1, s2))
    except ValueError as err:
        mine = "error"
        reason = str(err)
    try:
        theirs = str((np.zeros(s1) + np.zeros(s2)).shape)
    except ValueError:
        theirs = "error"
    agree = mine == theirs
    print(f"{str(s1):>10} + {str(s2):<6} -> mine {mine:<10} numpy {theirs:<10} agree {agree}")
    if reason:
        print(f"{'':>22}({reason})")
    assert agree

# %%
# Worked example: the 3 x 2 distance matrix by hand, three ways.
section("worked example: distances")


def pairwise_loops(A: FloatArray, B: FloatArray) -> FloatArray:
    """Euclidean distances with plain Python loops. A (n, d), B (m, d) -> (n, m)."""
    a_rows, b_rows = A.tolist(), B.tolist()
    out = [[0.0] * len(b_rows) for _ in a_rows]
    for i, a in enumerate(a_rows):
        for j, b in enumerate(b_rows):
            total = 0.0
            for k in range(len(a)):
                total += (a[k] - b[k]) ** 2
            out[i][j] = math.sqrt(total)
    return np.array(out)


def pairwise_broadcast(A: FloatArray, B: FloatArray) -> FloatArray:
    """Euclidean distances by broadcasting a difference cube. A (n, d), B (m, d) -> (n, m)."""
    diff = A[:, None, :] - B[None, :, :]  # (n, 1, d) - (1, m, d) -> (n, m, d)
    return np.sqrt((diff**2).sum(axis=-1))  # (n, m)


def pairwise_trick(A: FloatArray, B: FloatArray) -> FloatArray:
    """Euclidean distances via |a|^2 + |b|^2 - 2 a.b. A (n, d), B (m, d) -> (n, m)."""
    a_sq = (A**2).sum(axis=1)[:, None]  # (n, 1)
    b_sq = (B**2).sum(axis=1)[None, :]  # (1, m)
    cross = A @ B.T  # (n, d) @ (d, m) -> (n, m)
    d_sq = a_sq + b_sq - 2.0 * cross  # (n, m)
    return np.sqrt(np.maximum(d_sq, 0.0))  # clip tiny negatives from rounding


print("|a|^2:", (A**2).sum(axis=1).tolist(), " |b|^2:", (B**2).sum(axis=1).tolist())
print("A @ B.T:", (A @ B.T).tolist())
print("squared distances:", (pairwise_broadcast(A, B) ** 2).round(4).tolist())
for fn in (pairwise_loops, pairwise_broadcast, pairwise_trick):
    print(f"{fn.__name__:<18}", np.round(fn(A, B), 4).tolist())

# %%
# From scratch: the three versions agree on random data, and their speed differs.
section("from scratch: agreement")
P, Q = X_big[:200], X_big[200:]  # (200, 16) and (200, 16)
D_loop = pairwise_loops(P, Q)
D_bcast = pairwise_broadcast(P, Q)
D_trick = pairwise_trick(P, Q)
assert D_loop.shape == (200, 200), D_loop.shape
np.testing.assert_allclose(D_bcast, D_loop, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(D_trick, D_loop, rtol=1e-6, atol=1e-8)
print("broadcast == loops: OK")
print("trick == loops: OK")
print("D[0, :3] =", np.round(D_loop[0, :3], 4).tolist())


def best_time(fn, *args, repeats: int = 3) -> float:  # noqa: ANN001  (any function)
    """Return the fastest of `repeats` wall-clock timings of fn(*args), in seconds."""
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t)
    return min(times)


timings: dict[str, list[float]] = {"loops": [], "broadcast": [], "trick": []}
for n in SIZES:
    P_n = rng.normal(size=(n, D_FEATURES))
    Q_n = rng.normal(size=(n, D_FEATURES))
    loop_t = best_time(pairwise_loops, P_n, Q_n, repeats=1) if n <= LOOP_MAX else float("nan")
    timings["loops"].append(loop_t)
    timings["broadcast"].append(best_time(pairwise_broadcast, P_n, Q_n))
    timings["trick"].append(best_time(pairwise_trick, P_n, Q_n))
    print(
        f"~ n={n:>5}: loops {loop_t * 1e3:9.2f} ms  broadcast "
        f"{timings['broadcast'][-1] * 1e3:8.3f} ms  trick {timings['trick'][-1] * 1e3:8.3f} ms"
    )

# %%
# Standardization: subtract each column's mean, divide by its standard deviation.
section("from scratch: standardize")
S = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 60.0]])  # (3, 2)


def standardize(X: FloatArray) -> FloatArray:
    """Scale every column to mean 0 and (population) std 1. X (n, d) -> (n, d)."""
    mu = X.mean(axis=0)  # (d,) one mean per column
    sigma = X.std(axis=0)  # (d,) ddof=0: divide by n
    if np.any(sigma == 0):
        raise ValueError("a column is constant; its standard deviation is 0")
    return (X - mu) / sigma  # (n, d) - (d,) broadcasts over rows


print("column means:", S.mean(axis=0).tolist(), " column stds:", np.round(S.std(axis=0), 4).tolist())
Z = standardize(S)
print("standardized:\n", np.round(Z, 4))
print("new means:", (np.round(Z.mean(axis=0), 4) + 0.0).tolist(), " new stds:", np.round(Z.std(axis=0), 4).tolist())

# %%
# With SciPy and scikit-learn: the library versions, and the check that both agree.
section("with scipy and scikit-learn")
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

np.testing.assert_allclose(pairwise_broadcast(P, Q), cdist(P, Q), rtol=1e-6, atol=1e-8)
print("pairwise_broadcast matches scipy cdist: OK")
np.testing.assert_allclose(pairwise_trick(P, Q), cdist(P, Q), rtol=1e-6, atol=1e-8)
print("pairwise_trick matches scipy cdist: OK")
np.testing.assert_allclose(standardize(S), StandardScaler().fit_transform(S), rtol=1e-6, atol=1e-8)
print("standardize matches StandardScaler: OK")
print("np.std ddof=0:", np.round(S[:, 0].std(), 4), " ddof=1:", np.round(S[:, 0].std(ddof=1), 4))

# %%
# A failure mode of the trick: far from the origin, it loses precision.
section("the trick's precision trap")
far = np.array([[1e8, 1e8]])  # (1, 2)
near = far + np.array([[1e-3, 0.0]])  # (1, 2): true distance 0.001
print("true distance: 0.001")
print(f"broadcast: {pairwise_broadcast(far, near)[0, 0]:.10f}")
print(f"trick:     {pairwise_trick(far, near)[0, 0]:.10f}")
centred = np.vstack([far, near]) - np.vstack([far, near]).mean(axis=0)
print("trick after centring:", round(float(pairwise_trick(centred[:1], centred[1:])[0, 0]), 6))

# %%
# Key plot: time against size, log-log.
section("key plot")
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
ax.plot(SIZES, timings["loops"], marker="o", linestyle="-", label="Python loops")
ax.plot(SIZES, timings["broadcast"], marker="s", linestyle="--", label="broadcasting")
ax.plot(SIZES, timings["trick"], marker="^", linestyle=":", label="norm trick (matmul)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(SIZES, [str(n) for n in SIZES])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel("n points in each set (log scale)")
ax.set_ylabel("time for the n x n distance matrix, s (log scale)")
ax.set_title(f"Python loops are 10-100x slower than vectorized NumPy (d = {D_FEATURES})")
ax.legend(loc="best")
fig.tight_layout()
path = IMG / "a1_2_distance_timing.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Predict, then check: np.zeros((2, 3)) + np.zeros((3, 2)). Why does it fail?
# 2. Time pairwise_broadcast at n = 2000, d = 512. Watch the memory: the (n, m, d) cube is
#    2000 * 2000 * 512 * 8 bytes = 16 GB. The trick never builds it.
# 3. Replace X.std(axis=0) with X.std(axis=0, ddof=1) in standardize and see which check breaks.

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
