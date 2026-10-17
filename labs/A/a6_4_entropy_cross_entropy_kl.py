# %% [markdown]
# # A6.4 Entropy, cross-entropy and KL divergence — lab
# Lesson: lessons/A/a6_4_entropy_cross_entropy_kl.md
# Run: uv run python labs/A/a6_4_entropy_cross_entropy_kl.py   (CPU, defaults: ~3 s)
# GPU: no

# %%
from __future__ import annotations

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
from scipy import integrate, special, stats
from sklearn.metrics import log_loss, mutual_info_score

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
OUT = ROOT / "outputs" / "a6_4"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "A" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE only makes the grids finer.
GPU_SCALE = False
GRID_DX = 0.002 if GPU_SCALE else 0.01  # spacing of the x grid for continuous KL integrals
N_FIT = 121 if GPU_SCALE else 61  # grid points per parameter when fitting a Gaussian by KL
N_LABELS = 10_000  # labels in the "best constant prediction" demonstration

# %%
# Data: small distributions, hand-made classifier outputs, a joint table, a two-bump density.
section("data")
p3 = np.array([0.5, 0.25, 0.25])
q3 = np.array([1 / 3, 1 / 3, 1 / 3])
y_true = np.array([0, 1, 2])  # class index of each of 3 examples
probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]])  # (3, K) predicted probabilities
joint = np.array([[0.30, 0.10], [0.05, 0.55]])  # p(x, y): x = "has the feature", y = "label"
print("p =", show(p3), "  q =", show(q3))
print("classifier probabilities (rows = examples, columns = classes):", show(probs), "  true classes:", y_true.tolist())
print("joint table p(x, y):", show(joint), "  sums to", joint.sum())

# %%
# Worked example by hand.
section("worked example by hand")
h_fair = -2 * 0.5 * np.log(0.5)
h_bias = -(0.9 * np.log(0.9) + 0.1 * np.log(0.1))
print(f"fair coin: H = log 2 = {h_fair:.4f} nats = {h_fair / np.log(2):.4f} bits")
print(f"0.9/0.1 coin: -0.9 log 0.9 - 0.1 log 0.1 = {0.9 * -np.log(0.9):.4f} + {0.1 * -np.log(0.1):.4f} = "
      f"{h_bias:.4f} nats = {h_bias / np.log(2):.4f} bits")
h_p = -np.sum(p3 * np.log(p3))
ce_pq = -np.sum(p3 * np.log(q3))
kl_pq = np.sum(p3 * np.log(p3 / q3))
kl_qp = np.sum(q3 * np.log(q3 / p3))
print(f"H(p) = {h_p:.4f} nats ({h_p / np.log(2):.4f} bits)   H(p, q) = log 3 = {ce_pq:.4f}   "
      f"KL(p || q) = {kl_pq:.4f}   H(p) + KL = {h_p + kl_pq:.4f}")
print(f"KL(q || p) = (1/3)(log(2/3) + 2 log(4/3)) = {kl_qp:.4f}   (not equal to KL(p || q))")
losses = -np.log(probs[np.arange(3), y_true])
print(f"per-example loss -log q(true class) = {show(losses)}   mean = {losses.mean():.4f}")
print(f"a confident mistake, q(true) = 0.01: loss = {-np.log(0.01):.4f}")
h03 = -(0.3 * np.log(0.3) + 0.7 * np.log(0.7))
print(f"30% positives, predict 0.3 for everyone: loss = H(0.3) = {h03:.4f} nats")
kl_gauss = 0.5 * (0.25 + 1.0 - 1.0 - np.log(0.25))
print(f"KL(N(1, 0.5^2) || N(0, 1)) = 0.5 * (0.25 + 1 - 1 - log 0.25) = {kl_gauss:.4f}")

# %%
# From scratch: entropy, cross-entropy and KL with the 0 log 0 = 0 convention.
section("from scratch: entropy, cross-entropy, KL")


def entropy(p: FloatArray, base: float = np.e) -> float:
    """H(p) = -sum p log p, with 0 log 0 = 0. p (K,) -> scalar."""
    nz = p > 0
    return float(-np.sum(p[nz] * np.log(p[nz])) / np.log(base)) + 0.0  # + 0.0 turns -0.0 into 0.0


def cross_entropy(p: FloatArray, q: FloatArray, base: float = np.e) -> float:
    """H(p, q) = -sum p log q; infinite if q = 0 where p > 0. p, q (K,) -> scalar."""
    nz = p > 0
    if np.any(q[nz] == 0):
        return float("inf")
    return float(-np.sum(p[nz] * np.log(q[nz])) / np.log(base))


def kl_divergence(p: FloatArray, q: FloatArray, base: float = np.e) -> float:
    """KL(p || q) = sum p log(p / q). p, q (K,) -> scalar >= 0."""
    nz = p > 0
    if np.any(q[nz] == 0):
        return float("inf")
    return float(np.sum(p[nz] * np.log(p[nz] / q[nz])) / np.log(base))


for name, p, q in [("p3 vs uniform", p3, q3), ("random 5-class", rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5)))]:
    gap = cross_entropy(p, q) - (entropy(p) + kl_divergence(p, q))
    print(f"{name}: H(p) = {entropy(p):.4f}   H(p, q) = {cross_entropy(p, q):.4f}   KL(p||q) = {kl_divergence(p, q):.4f}   "
          f"KL(q||p) = {kl_divergence(q, p):.4f}   H(p, q) - H(p) - KL = {gap:.1e}")
print(f"entropy of K equally likely outcomes = log K: K = 2, 4, 8 -> {[round(entropy(np.ones(k) / k, 2), 4) for k in (2, 4, 8)]} bits")
print(f"a certain outcome [1, 0, 0]: H = {entropy(np.array([1.0, 0.0, 0.0])):.4f}")
p_mix = np.array([0.5, 0.5, 0.0])
q_miss = np.array([1.0, 0.0, 0.0])
print(f"q gives 0 to an outcome p allows: KL(p || q) = {kl_divergence(p_mix, q_miss)}   "
      f"KL(q || p) = {kl_divergence(q_miss, p_mix):.4f}")
smallest = min(kl_divergence(a, b) for a, b in (rng.dirichlet(np.ones(4), size=2) for _ in range(10_000)))
print(f"smallest KL over 10,000 random pairs of 4-class distributions: {smallest:.6f}   (never negative)")

# %%
# From scratch: cross-entropy is the classification loss, and its minimum is the entropy of the labels.
section("from scratch: cross-entropy as a loss")


def mean_ce_loss(y: npt.NDArray[np.int64], probs: FloatArray) -> float:
    """Average -log of the probability given to the true class. y (n,), probs (n, K) -> scalar."""
    return float(-np.mean(np.log(probs[np.arange(len(y)), y])))


onehot = np.eye(3)[y_true]  # (3, K) the "true distribution" of each example
per_example_ce = [cross_entropy(onehot[i], probs[i]) for i in range(3)]
print(f"mean -log q(true) = {mean_ce_loss(y_true, probs):.4f}   mean H(onehot, q) = {np.mean(per_example_ce):.4f}   "
      "(the same number)")
labels = (rng.random(N_LABELS) < 0.3).astype(np.int64)  # 30% positives
q_grid = np.linspace(0.05, 0.95, 19)
loss_grid = [mean_ce_loss(labels, np.column_stack([1 - np.full(N_LABELS, q), np.full(N_LABELS, q)])) for q in q_grid]
best = q_grid[int(np.argmin(loss_grid))]
rate = labels.mean()
print(f"labels: {rate:.4f} positive   best constant prediction on a 0.05 grid = {best:.2f}   "
      f"its loss = {min(loss_grid):.4f}   entropy of the labels = {entropy(np.array([1 - rate, rate])):.4f}")
print(f"predicting 0.5 for everyone costs log 2 = {np.log(2):.4f}; the extra over the entropy is "
      f"KL = {kl_divergence(np.array([1 - rate, rate]), np.array([0.5, 0.5])):.4f}")

# %%
# From scratch: mutual information, KL between Gaussians, forward vs reverse KL.
section("from scratch: mutual information and continuous KL")


def mutual_information(joint: FloatArray) -> float:
    """I(X; Y) = KL(p(x, y) || p(x) p(y)). joint (r, c) summing to 1 -> scalar."""
    px, py = joint.sum(axis=1), joint.sum(axis=0)  # marginals (r,), (c,)
    return kl_divergence(joint.ravel(), np.outer(px, py).ravel())


mi = mutual_information(joint)
py = joint.sum(axis=0)
h_y_given_x = sum(joint[i].sum() * entropy(joint[i] / joint[i].sum()) for i in range(2))
print(f"I(X; Y) = {mi:.4f} nats   H(Y) = {entropy(py):.4f}   H(Y | X) = {h_y_given_x:.4f}   "
      f"H(Y) - H(Y | X) = {entropy(py) - h_y_given_x:.4f}")
print(f"independent table (outer product of the marginals): I = {abs(mutual_information(np.outer(joint.sum(1), py))):.4f}")


def kl_gauss_1d(mu: float, sd: float, mu0: float = 0.0, sd0: float = 1.0) -> float:
    """KL(N(mu, sd^2) || N(mu0, sd0^2)) in closed form."""
    return float(np.log(sd0 / sd) + (sd**2 + (mu - mu0) ** 2) / (2 * sd0**2) - 0.5)


x = np.arange(-10, 10, GRID_DX)  # grid for numerical integrals


def kl_numeric(p: FloatArray, q: FloatArray) -> float:
    """KL between two densities sampled on the grid x, by a Riemann sum. p, q (len(x),) -> scalar."""
    mask = p > 1e-300
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(np.maximum(q[mask], 1e-300)))) * GRID_DX)


for mu, sd in [(1.0, 0.5), (0.0, 2.0), (3.0, 1.0)]:
    num = kl_numeric(stats.norm.pdf(x, mu, sd), stats.norm.pdf(x))
    print(f"KL(N({mu}, {sd}^2) || N(0, 1)): closed form {kl_gauss_1d(mu, sd):.4f}   numerical {num:.4f}")

target = 0.5 * stats.norm.pdf(x, -2.5, 0.7) + 0.5 * stats.norm.pdf(x, 2.5, 0.7)  # two bumps
mus, sds = np.linspace(-4, 4, N_FIT), np.linspace(0.3, 5, N_FIT)
fwd = np.full((N_FIT, N_FIT), np.inf)
rev = np.full((N_FIT, N_FIT), np.inf)
for i, mu in enumerate(mus):
    for j, sd in enumerate(sds):
        qd = stats.norm.pdf(x, mu, sd)
        fwd[i, j] = kl_numeric(target, qd)  # KL(p || q): penalizes q missing any mass of p
        rev[i, j] = kl_numeric(qd, target)  # KL(q || p): penalizes q putting mass where p has none
i_f, j_f = np.unravel_index(np.argmin(fwd), fwd.shape)
i_r, j_r = np.unravel_index(np.argmin(rev), rev.shape)
fit_fwd, fit_rev = (mus[i_f], sds[j_f]), (mus[i_r], sds[j_r])
print(f"target: two bumps at -2.5 and 2.5 (sd 0.7 each); mean 0, sd {np.sqrt(np.sum(x**2 * target) * GRID_DX):.4f}")
print(f"best Gaussian by forward KL(p || q): mu = {fit_fwd[0]:.2f}, sd = {fit_fwd[1]:.2f}   KL = {fwd[i_f, j_f]:.4f}")
print(f"best Gaussian by reverse KL(q || p): mu = {fit_rev[0]:.2f}, sd = {fit_rev[1]:.2f}   KL = {rev[i_r, j_r]:.4f}")

# %%
# With SciPy and scikit-learn.
section("with scipy and scikit-learn")
checks = {
    "stats.entropy(p) = H(p)": (stats.entropy(p3), entropy(p3)),
    "stats.entropy(p, base=2) = H(p) in bits": (stats.entropy(p3, base=2), entropy(p3, 2)),
    "stats.entropy(p, q) = KL(p || q), not H(p, q)": (stats.entropy(p3, q3), kl_divergence(p3, q3)),
    "special.rel_entr(p, q).sum() = KL": (special.rel_entr(p3, q3).sum(), kl_divergence(p3, q3)),
    "log_loss(y, probs) = mean CE": (log_loss(y_true, probs, labels=[0, 1, 2]), mean_ce_loss(y_true, probs)),
}
for name, (lib, mine) in checks.items():
    np.testing.assert_allclose(lib, mine, rtol=1e-12)
    print(f"{name:<46} library {lib:.4f}   mine {mine:.4f}   OK")
xs = np.repeat([0, 0, 1, 1], (joint * 1000).round().astype(int).ravel())  # 1,000 rows matching the table
ys = np.repeat([0, 1, 0, 1], (joint * 1000).round().astype(int).ravel())
np.testing.assert_allclose(mutual_info_score(xs, ys), mi, rtol=1e-10)
print(f"mutual_info_score on 1,000 rows built from the table: {mutual_info_score(xs, ys):.4f}   mine {mi:.4f}   OK")
val, _ = integrate.quad(lambda t: stats.norm.pdf(t, 1, 0.5) * (stats.norm.logpdf(t, 1, 0.5) - stats.norm.logpdf(t)), -10, 10)
np.testing.assert_allclose(val, kl_gauss_1d(1.0, 0.5), rtol=1e-8)
print(f"quad of KL(N(1, 0.25) || N(0, 1)): {val:.6f}   closed form {kl_gauss_1d(1.0, 0.5):.6f}   OK")

# %%
# Key plot: binary entropy, and forward versus reverse KL fits to a two-bump density.
section("key plot")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=120)
ax = axes[0]
pp = np.linspace(0.001, 0.999, 500)
hb = -(pp * np.log2(pp) + (1 - pp) * np.log2(1 - pp))
ax.plot(pp, hb, "-", lw=2)
for pt, offset, align in ((0.5, (0, -30), "center"), (0.9, (-12, 6), "right")):
    hv = -(pt * np.log2(pt) + (1 - pt) * np.log2(1 - pt))
    ax.plot(pt, hv, "o", color="tab:red")
    ax.annotate(f"P(heads) = {pt}: {hv:.3f} bits", (pt, hv), textcoords="offset points", xytext=offset, ha=align)
ax.set(xlabel="P(heads)", ylabel="entropy H (bits)", ylim=(0, 1.1),
       title="A coin is most uncertain at 50/50: 1 bit")
ax = axes[1]
ax.plot(x, target, "k-", lw=2, label="target p: two bumps")
ax.plot(x, stats.norm.pdf(x, *fit_fwd), "--", lw=2, label=f"min KL(p||q): covers both, sd {fit_fwd[1]:.2f}")
ax.plot(x, stats.norm.pdf(x, *fit_rev), ":", lw=2.5, label=f"min KL(q||p): picks one, sd {fit_rev[1]:.2f}")
ax.set(xlabel="x", ylabel="density", xlim=(-7, 7),
       title="Best single Gaussian depends on the KL direction")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
path_png = IMG / "a6_4_entropy_kl_direction.png"
fig.savefig(path_png, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path_png.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Compute the entropy of the letters in a paragraph of English text (in bits). How far is it from log2(26)?
# 2. Label smoothing: replace the one-hot targets by 0.9 / 0.05 / 0.05. What is the smallest possible loss now?
# 3. Move the two bumps closer (+-1). At what distance do the forward and reverse fits agree?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
