# %% [markdown]
# # A6.4 Entropy, cross-entropy and KL — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a6_4_entropy_cross_entropy_kl_ex2.py   (CPU, ~2 s)

# %%
from __future__ import annotations

import time
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import log_loss, mutual_info_score

FloatArray = npt.NDArray[np.float64]
SEED = 0
T0 = time.perf_counter()
N, K = 1_000, 5
N_MI = 20_000
N_BINS = 10


def section(name: str) -> None:
    print(f"\n== {name} ==")


def entropy(p: FloatArray, base: float = np.e) -> float:
    """H(p) with 0 log 0 = 0. p (K,) -> scalar."""
    nz = p > 0
    return float(-np.sum(p[nz] * np.log(p[nz])) / np.log(base))


def kl(p: FloatArray, q: FloatArray) -> float:
    """KL(p || q) in nats. p, q (K,) -> scalar."""
    nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] / q[nz])))


# %%
section("exercise 1 check")
for d in ([0.25] * 4, [0.5, 0.25, 0.125, 0.125], [0.97, 0.01, 0.01, 0.01]):
    print(f"(a) H({d}) = {entropy(np.array(d), 2):.4f} bits")
p, q = np.array([0.5, 0.5]), np.array([0.9, 0.1])
print(f"(b) H(p) = {entropy(p):.4f}   H(p, q) = {-np.sum(p * np.log(q)):.4f}   KL(p||q) = {kl(p, q):.4f}   "
      f"KL(q||p) = {kl(q, p):.4f}")
l1, l2 = -np.log(0.6), -np.log(0.25)
print(f"(c) losses {l1:.4f} and {l2:.4f}   mean {(l1 + l2) / 2:.4f}   uniform model: log 4 = {np.log(4):.4f}")
for mu, sd in ((0.0, 0.5), (2.0, 1.0)):
    print(f"(d) KL(N({mu}, {sd}^2) || N(0, 1)) = {0.5 * (sd**2 + mu**2 - 1 - np.log(sd**2)):.4f}")


# %%
def cross_entropy_from_logits(Z: FloatArray, y: npt.NDArray[np.int64]) -> float:
    """Mean cross-entropy from logits via a stable log-softmax. Z (n, K), y (n,) -> scalar."""
    Zs = Z - Z.max(axis=1, keepdims=True)  # (n, K) largest logit per row becomes 0: exp never overflows
    log_norm = np.log(np.exp(Zs).sum(axis=1))  # (n,) log sum_j exp(z_j - max) >= 0
    log_probs_true = Zs[np.arange(len(y)), y] - log_norm  # (n,) log softmax of the true class
    return float(-log_probs_true.mean())


def naive_ce(Z: FloatArray, y: npt.NDArray[np.int64]) -> float:
    """Mean cross-entropy via softmax then log: overflows for large logits. Z (n, K), y (n,) -> scalar."""
    P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


section("exercise 2: stable cross-entropy from logits")
rng = np.random.default_rng(SEED)
Z = rng.normal(0, 2, size=(N, K))
y = rng.integers(0, K, size=N)
P = np.exp(Z - Z.max(axis=1, keepdims=True))
P /= P.sum(axis=1, keepdims=True)
mine = cross_entropy_from_logits(Z, y)
np.testing.assert_allclose(mine, log_loss(y, P, labels=list(range(K))), rtol=1e-10)
print(f"random logits: mine {mine:.6f}   sklearn log_loss {log_loss(y, P, labels=list(range(K))):.6f}   OK")
Z_big = np.array([[1000.0, 0.0, -1000.0], [1000.0, 0.0, -1000.0]])
y_big = np.array([0, 1])
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)  # the naive version overflows on purpose
    print(f"logits (1000, 0, -1000), classes [0, 1]: naive {naive_ce(Z_big, y_big)}   "
          f"stable {cross_entropy_from_logits(Z_big, y_big):.4f}   (exact: (0 + 1000) / 2 = 500)")

section("exercise 2: mutual information versus correlation")
x = rng.normal(0, 1, size=(N_MI, 3))
label_lin = (x[:, 1] + rng.normal(0, 1, N_MI) > 0).astype(int)
label_abs = (np.abs(x[:, 2]) > 1).astype(int)
features = {
    "independent": (x[:, 0], label_lin),
    "linear": (x[:, 1], label_lin),
    "|x| > 1": (x[:, 2], label_abs),
}
for name, (feat, lab) in features.items():
    bins = pd.qcut(feat, N_BINS, labels=False)  # 10 equal-count bins
    mi = mutual_info_score(bins, lab)
    r = abs(np.corrcoef(feat, lab)[0, 1])
    print(f"{name:<12} mutual information = {mi:.4f} nats   |Pearson r| = {r:.4f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
