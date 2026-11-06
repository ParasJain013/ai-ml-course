# %% [markdown]
# # P4 reference run: a NumPy MLP on digits, matched against PyTorch
# For the reviewer: gives the expected numbers quoted in REFERENCE.md. Not the learner's solution (they
# must use their own autograd engine; this uses a hand-written backward pass to stay short).
# Run: uv run python answers/projects/P4/reference_run.py   (CPU, ~10 s)

# %%
from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

FloatArray = npt.NDArray[np.float64]
SEED = 0
T0 = time.perf_counter()
WIDTHS = [64, 128, 10]
LR, BATCH, EPOCHS = 0.1, 32, 30


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("data")
digits = load_digits()
X = digits.data / 16.0  # pixel values 0..16 -> 0..1
y = digits.target
X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)
print(f"train {X_tr.shape}  val {X_va.shape}  test {X_te.shape}")


# %%
def logsumexp(Z: FloatArray) -> FloatArray:
    m = Z.max(axis=1)
    return m + np.log(np.exp(Z - m[:, None]).sum(axis=1))


def forward_backward(params: list[FloatArray], Xb: FloatArray, yb: npt.NDArray[np.int64]) -> tuple[float, list[FloatArray]]:
    """Mean CE and gradients of the 64-128-10 ReLU MLP. params [W1, b1, W2, b2]."""
    W1, b1, W2, b2 = params
    n = len(yb)
    Z1 = Xb @ W1 + b1
    H = np.maximum(Z1, 0)
    Z2 = H @ W2 + b2
    lse = logsumexp(Z2)
    loss = float(np.mean(lse - Z2[np.arange(n), yb]))
    D2 = np.exp(Z2 - lse[:, None])
    D2[np.arange(n), yb] -= 1
    D2 /= n
    D1 = (D2 @ W2.T) * (Z1 > 0)
    return loss, [Xb.T @ D1, D1.sum(0), H.T @ D2, D2.sum(0)]


def accuracy(params: list[FloatArray], Xs: FloatArray, ys: npt.NDArray[np.int64]) -> float:
    W1, b1, W2, b2 = params
    return float(np.mean(np.argmax(np.maximum(Xs @ W1 + b1, 0) @ W2 + b2, axis=1) == ys))


gen = np.random.default_rng(SEED)
init = [
    gen.normal(0, np.sqrt(2 / 64), size=(64, 128)), np.zeros(128),
    gen.normal(0, np.sqrt(2 / 128), size=(128, 10)), np.zeros(10),
]  # fmt: skip
batch_order = [np.random.default_rng(SEED + e).permutation(len(y_tr)) for e in range(EPOCHS)]  # shared by both

# %%
section("numpy training, with PyTorch fed the same batches")
P_np = [p.copy() for p in init]
P_t = [torch.tensor(p.copy(), dtype=torch.float64, requires_grad=True) for p in init]
opt = torch.optim.SGD(P_t, lr=LR)
worst_grad = 0.0
checked = 0
for epoch in range(EPOCHS):
    losses_np, losses_t = [], []
    for start in range(0, len(y_tr), BATCH):
        idx = batch_order[epoch][start : start + BATCH]
        loss, grads = forward_backward(P_np, X_tr[idx], y_tr[idx])
        opt.zero_grad()
        Xb_t = torch.tensor(X_tr[idx])
        logits = torch.relu(Xb_t @ P_t[0] + P_t[1]) @ P_t[2] + P_t[3]
        loss_t = torch.nn.functional.cross_entropy(logits, torch.tensor(y_tr[idx]))
        loss_t.backward()
        if checked < 5:  # the required 5-batch check, before any divergence could accumulate
            worst_grad = max(worst_grad, max(float(np.max(np.abs(g - p.grad.numpy()))) for g, p in zip(grads, P_t)))
            checked += 1
        opt.step()
        for p, g in zip(P_np, grads):
            p -= LR * g
        losses_np.append(loss)
        losses_t.append(loss_t.item())
    if epoch in (0, 4, 9, 19, EPOCHS - 1):
        print(f"epoch {epoch + 1:>2}: train loss numpy {np.mean(losses_np):.4f}  torch {np.mean(losses_t):.4f}   "
              f"val acc numpy {accuracy(P_np, X_va, y_va):.4f}")
print(f"max |grad numpy - grad torch| over the first 5 batches: {'< 1e-12' if worst_grad < 1e-12 else f'{worst_grad:.1e}'}")
param_gap = max(float(np.max(np.abs(p - q.detach().numpy()))) for p, q in zip(P_np, P_t))
print(f"max |param numpy - param torch| after {EPOCHS} epochs: {'< 1e-10' if param_gap < 1e-10 else f'{param_gap:.1e}'}")
P_t_np = [q.detach().numpy() for q in P_t]
print(f"test accuracy: numpy {accuracy(P_np, X_te, y_te):.4f}   torch {accuracy(P_t_np, X_te, y_te):.4f}")

# %%
section("float32 versus float64")
init32 = [torch.tensor(p, dtype=torch.float32, requires_grad=True) for p in init]
idx = batch_order[0][:BATCH]
logits32 = torch.relu(torch.tensor(X_tr[idx], dtype=torch.float32) @ init32[0] + init32[1]) @ init32[2] + init32[3]
torch.nn.functional.cross_entropy(logits32, torch.tensor(y_tr[idx])).backward()
_, g64 = forward_backward(init, X_tr[idx], y_tr[idx])
gap32 = max(float(np.max(np.abs(g - p.grad.numpy().astype(np.float64)))) for g, p in zip(g64, init32))
print(f"first batch: max |float64 grad - float32 grad| = {gap32:.1e}")
opt32 = torch.optim.SGD(init32, lr=LR)
P_64 = [p.copy() for p in init]
gaps = []
for epoch in range(EPOCHS):
    for start in range(0, len(y_tr), BATCH):
        idx = batch_order[epoch][start : start + BATCH]
        _, g64 = forward_backward(P_64, X_tr[idx], y_tr[idx])
        opt32.zero_grad()
        lg = torch.relu(torch.tensor(X_tr[idx], dtype=torch.float32) @ init32[0] + init32[1]) @ init32[2] + init32[3]
        torch.nn.functional.cross_entropy(lg, torch.tensor(y_tr[idx])).backward()
        gaps.append(max(float(np.max(np.abs(g - q.grad.numpy().astype(np.float64)))) for g, q in zip(g64, init32)))
        opt32.step()
        for p, g in zip(P_64, g64):
            p -= LR * g
print(f"same batches for {EPOCHS} epochs, float32 torch vs float64 numpy: worst grad gap {max(gaps):.1e}, "
      f"batches over 1e-6: {sum(g > 1e-6 for g in gaps)} of {len(gaps)}")
print(f"test accuracy float32 {accuracy([q.detach().numpy().astype(np.float64) for q in init32], X_te, y_te):.4f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
