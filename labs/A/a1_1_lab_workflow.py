# %% [markdown]
# # A1.1 The lab workflow — lab
# Lesson: lessons/A/a1_1_lab_workflow.md
# Run: uv run python labs/A/a1_1_lab_workflow.py   (CPU, defaults: ~10 s, most of it importing torch)
# GPU: no

# %%
from __future__ import annotations

import platform
import random
import sys
import time
from pathlib import Path

import warnings

import matplotlib

if "ipykernel" not in sys.modules:  # plain `uv run`: draw to files, never open windows
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

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
OUT = ROOT / "outputs" / "a1_1"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "A" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing in this lab is big; the knob is here because every lab has one.
GPU_SCALE = False
N_DRAWS = 1_000_000 if GPU_SCALE else 1_000  # numbers drawn in the plot cell

# %%
# Health check: is the environment the one pyproject.toml pins?
section("health check")
import matplotlib as mpl  # noqa: E402  (already imported; re-imported for its version)
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402

print(f"python        {platform.python_version()}")
for name, module in [
    ("numpy", np),
    ("pandas", pd),
    ("matplotlib", mpl),
    ("scipy", scipy),
    ("scikit-learn", sklearn),
    ("torch", torch),
]:
    print(f"{name:<13} {module.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"~ machine: {platform.machine()} {platform.system()} {platform.release()}")

# %%
# Worked example by hand: a toy linear congruential generator (LCG).
# x_{k+1} = (a * x_k + c) mod m, with m = 16, a = 5, c = 3, seed x_0 = 7.
section("worked example: toy LCG")


def lcg(seed: int, a: int, c: int, m: int, n: int) -> IntArray:
    """Return n states of x_{k+1} = (a x_k + c) mod m, starting after the seed -> (n,)."""
    out = np.empty(n, dtype=np.int64)
    x = seed
    for k in range(n):
        x = (a * x + c) % m
        out[k] = x
    return out


M, A, C, X0 = 16, 5, 3, 7
states = lcg(X0, A, C, M, 6)
print("x_1..x_6:", states.tolist())
print("u_1..u_6:", (states / M).tolist())

# %%
# From scratch: the properties a seeded generator must have.
section("from scratch: properties")
full = lcg(X0, A, C, M, 16)
print("all 16 states:", full.tolist())
print("x_16 == seed:", bool(full[-1] == X0))
print("distinct states:", len(set(full.tolist())))

run_1 = lcg(X0, A, C, M, 5)
run_2 = lcg(X0, A, C, M, 5)
other = lcg(8, A, C, M, 5)
print("same seed, same numbers:", np.array_equal(run_1, run_2))
print("seed 8 gives:", other.tolist())

# %%
# With NumPy, Python and PyTorch: real generators, same idea.
section("with numpy: same seed, same numbers")
first = np.random.default_rng(SEED).random(3)
second = np.random.default_rng(SEED).random(3)
print("run 1:", np.round(first, 4).tolist())
print("run 2:", np.round(second, 4).tolist())
np.testing.assert_array_equal(first, second)
print("matches across runs: OK")

section("with numpy: the stream position matters")
gen = np.random.default_rng(SEED)
five_at_once = gen.random(5)
gen = np.random.default_rng(SEED)
three_then_two = np.concatenate([gen.random(3), gen.random(2)])
np.testing.assert_array_equal(five_at_once, three_then_two)
print("random(5) == random(3) + random(2): OK")
gen = np.random.default_rng(SEED)
_ = gen.random(1)  # one extra draw added "upstream"
shifted = gen.random(5)
print("before:", np.round(five_at_once, 4).tolist())
print("after an extra draw:", np.round(shifted, 4).tolist())

section("with numpy: independent streams")
data_rng, model_rng = np.random.default_rng(SEED).spawn(2)
print("data stream: ", np.round(data_rng.random(3), 4).tolist())
print("model stream:", np.round(model_rng.random(3), 4).tolist())

section("python and torch")
random.seed(SEED)
py_a = [random.random() for _ in range(3)]
random.seed(SEED)
py_b = [random.random() for _ in range(3)]
print("python random:", [round(v, 4) for v in py_a], "repeatable:", py_a == py_b)
torch.manual_seed(SEED)
t_a = torch.rand(3)
torch.manual_seed(SEED)
t_b = torch.rand(3)
print("torch.rand:   ", [round(v, 4) for v in t_a.tolist()], "repeatable:", bool(torch.equal(t_a, t_b)))

# %%
# Key plot: consecutive pairs (x_k, x_{k+1}). A toy LCG puts them on a few lines;
# NumPy's PCG64 fills the square. This is why you never write your own generator.
section("key plot")
toy = lcg(1, 5, 1, 256, 256) / 256  # m = 256, a = 5, c = 1: full period 256
pcg = rng.random(N_DRAWS)
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=120, sharey=True)
axes[0].scatter(toy[:-1], toy[1:], s=8, marker="o")
axes[0].set_title("Toy LCG (m=256, a=5, c=1)")
axes[1].scatter(pcg[:-1], pcg[1:], s=4, marker="x", color="tab:orange")
axes[1].set_title(f"NumPy default_rng (PCG64), {N_DRAWS:,} draws")
for ax in axes:
    ax.set_xlabel("$u_k$ (draw k, in [0, 1))")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
axes[0].set_ylabel("$u_{k+1}$ (next draw, in [0, 1))")
fig.suptitle("A toy generator's pairs fall on 5 lines; a real one fills the square")
fig.tight_layout()
path = IMG / "a1_1_lcg_vs_pcg64.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Change SEED to 1 at the top and re-run. Which printed numbers change, and which do not?
# 2. In the toy LCG, try a = 4 (so a - 1 = 3 is odd). How many distinct states do you get now?
# print(len(set(lcg(7, 4, 3, 16, 16).tolist())))
# 3. Run this file twice from the terminal and diff the output (lines starting "~" may differ).

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
