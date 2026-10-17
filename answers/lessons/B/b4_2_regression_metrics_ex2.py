# %% [markdown]
# # B4.2 Regression metrics — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b4_2_regression_metrics_ex2.py   (CPU, ~15 s; needs the B4.2 lab's download)

# %%
from __future__ import annotations

import hashlib
import io
import tarfile
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

FloatArray = npt.NDArray[np.float64]
SEED = 0
T0 = time.perf_counter()
QUANTILES = [0.1, 0.5, 0.9]
HOUSING_TGZ = Path(__file__).resolve().parents[3] / "data" / "california_housing" / "housing.tgz"
HOUSING_SHA256 = "d4cd501af90475f09b814c7447c7701f59bf28e8cf1180205ae5ace9737a0109"


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
y = np.array([2.0, 4.0, 6.0, 8.0, 100.0])
for name, c in [("median", np.median(y)), ("mean", y.mean())]:
    e = y - c
    print(f"(a) constant = {name} {c:g}: MAE {np.mean(np.abs(e)):.4f}  RMSE {np.sqrt(np.mean(e ** 2)):.4f}")
f = np.array([1.0, 5.0, 6.0, 9.0, 80.0])
e = y - f
print(f"(b) MAE {np.mean(np.abs(e)):.4f}  RMSE {np.sqrt(np.mean(e ** 2)):.4f}  MAPE {np.mean(np.abs(e) / y):.4f}  "
      f"WAPE {np.sum(np.abs(e)) / np.sum(y):.4f}")
c_fp, c_fn, p = 4.0, 60.0, 0.08
print(f"(c) threshold {c_fp / (c_fp + c_fn):.4f}; expected cost if reviewed {(1 - p) * c_fp:.2f}, if not {p * c_fn:.2f}")
print(f"(d) forecast 10: actual 5 -> APE {abs(5 - 10) / 5:.4f}; actual 15 -> APE {abs(15 - 10) / 15:.4f}; "
      f"under-forecast of 0 for any positive actual -> APE 1.0")

# %%
section("data")
if not HOUSING_TGZ.exists():
    raise FileNotFoundError("Run labs/B/b4_2_regression_metrics.py once first: it downloads the housing data.")
raw = HOUSING_TGZ.read_bytes()
if hashlib.sha256(raw).hexdigest() != HOUSING_SHA256:
    raise ValueError(f"{HOUSING_TGZ} has the wrong checksum; delete it and rerun the lab")
with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
    member = tar.extractfile("housing.csv")
    assert member is not None
    h = pd.read_csv(io.BytesIO(member.read()))
X = np.column_stack([h["median_income"], h["housing_median_age"], h["total_rooms"] / h["households"],
                     h["total_bedrooms"] / h["households"], h["population"], h["population"] / h["households"],
                     h["latitude"], h["longitude"]]).astype(np.float64)
y_all = (h["median_house_value"] / 100_000).to_numpy(dtype=np.float64)
gen = np.random.default_rng(SEED)  # the lab's (and B3.3's) 60/20/20 split
rows = gen.permutation(len(y_all))
n_te = n_va = round(0.2 * len(y_all))
i_tr = gen.permutation(rows[n_te + n_va:])
i_va = gen.permutation(rows[n_te:n_te + n_va])
y_va = y_all[i_va]
print(f"train {len(i_tr)}  val {len(i_va)}")


# %%
def pinball(y: FloatArray, y_hat: FloatArray, tau: float) -> float:
    """Mean pinball (quantile) loss at quantile tau. y, y_hat (n,) -> scalar."""
    e = y - y_hat
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))  # tau|e| above the prediction, (1-tau)|e| below


section("pinball loss check")
y_toy, f_toy = np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0])
for tau in QUANTILES:
    np.testing.assert_allclose(pinball(y_va, np.full_like(y_va, 2.0), tau), mean_pinball_loss(y_va, np.full_like(y_va, 2.0), alpha=tau))
print(f"toy: y {y_toy.tolist()}, prediction 2 -> pinball at 0.9 = {pinball(y_toy, f_toy, 0.9):.4f} "
      "(0.9 x 1 above, 0.1 x 1 below, averaged over 3 rows)")
print("matches sklearn.metrics.mean_pinball_loss: OK")

# %%
section("four models")
preds = {}
for name, kwargs in [("squared_error", {"loss": "squared_error"})] + [
        (f"quantile {q}", {"loss": "quantile", "quantile": q}) for q in QUANTILES]:
    m = HistGradientBoostingRegressor(max_iter=300, random_state=SEED, **kwargs).fit(X[i_tr], y_all[i_tr])
    preds[name] = m.predict(X[i_va])
print(f"{'model':<15} {'MAE':>7} {'RMSE':>7} " + " ".join(f"{'pin@' + str(q):>8}" for q in QUANTILES) + f" {'y < pred':>9}")
for name, p in preds.items():
    e = y_va - p
    print(f"{name:<15} {np.mean(np.abs(e)):>7.4f} {np.sqrt(np.mean(e ** 2)):>7.4f} "
          + " ".join(f"{pinball(y_va, p, q):>8.4f}" for q in QUANTILES) + f" {np.mean(y_va < p):>9.4f}")
inside = np.mean((y_va >= preds["quantile 0.1"]) & (y_va <= preds["quantile 0.9"]))
print(f"share of validation targets between the 0.1 and 0.9 predictions: {inside:.4f} (nominal 0.80)")
crossed = np.mean(preds["quantile 0.1"] > preds["quantile 0.9"])
print(f"rows where the 0.1 prediction is above the 0.9 prediction (quantile crossing): {crossed:.4f}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
