# %% [markdown]
# # B4.6 Fairness — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b4_6_fairness_model_cards_ex2.py   (CPU, ~20 s; needs the B4.1/B4.6 download)

# %%
from __future__ import annotations

import hashlib
import io
import time
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from fairlearn.metrics import MetricFrame, equalized_odds_difference, false_positive_rate, selection_rate, true_positive_rate
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 0
T0 = time.perf_counter()
N_BOOT = 1000
ADULT_FILE = Path(__file__).resolve().parents[3] / "data" / "adult" / "adult-all.csv"
ADULT_SHA256 = "21e0ea2f925a00338929a8c86c27354c72ac1c79819bcca81c7d91c3d64218c2"
ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
                 "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country",
                 "income"]
NUMERIC = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = ["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
section("exercise 1 check")
groups = {"X": dict(n=400, P=120, sel=140, TP=90), "Y": dict(n=100, P=10, sel=15, TP=6)}
r = {}
for g, d in groups.items():
    fp, N = d["sel"] - d["TP"], d["n"] - d["P"]
    r[g] = dict(base=d["P"] / d["n"], SR=d["sel"] / d["n"], TPR=d["TP"] / d["P"], FPR=fp / N)
    print(f"(a) group {g}: base {r[g]['base']:.4f}  SR {r[g]['SR']:.4f}  TPR {r[g]['TPR']:.4f}  FPR {r[g]['FPR']:.4f}   "
          f"(b) TPR*base + FPR*(1-base) = {r[g]['TPR'] * r[g]['base'] + r[g]['FPR'] * (1 - r[g]['base']):.4f}")
dp = abs(r["X"]["SR"] - r["Y"]["SR"])
eopp = abs(r["X"]["TPR"] - r["Y"]["TPR"])
fpr_gap = abs(r["X"]["FPR"] - r["Y"]["FPR"])
print(f"    gaps: DP {dp:.4f}  equal opportunity {eopp:.4f}  FPR {fpr_gap:.4f}  equalized odds {max(eopp, fpr_gap):.4f}")
srs = {g: 0.7 * r[g]["base"] + 0.1 * (1 - r[g]["base"]) for g in r}
print(f"(c) TPR 0.7, FPR 0.1 in both: SR X {srs['X']:.2f}  SR Y {srs['Y']:.2f}  DP difference {srs['X'] - srs['Y']:.2f}")
se = np.sqrt(0.6 * 0.4 / 10)
print(f"(d) TPR 0.6 on 10 positives: standard error {se:.3f}, 95% interval about +-{1.96 * se:.2f}")

# %%
section("data and model (as in the lab)")
if not ADULT_FILE.exists():
    raise FileNotFoundError("Run labs/B/b4_6_fairness_model_cards.py once first: it downloads UCI Adult.")
raw = ADULT_FILE.read_bytes()
if hashlib.sha256(raw).hexdigest() != ADULT_SHA256:
    raise ValueError(f"{ADULT_FILE} has the wrong checksum; delete it and rerun the lab")
adult = pd.read_csv(io.BytesIO(raw), header=None, names=ADULT_COLUMNS, skipinitialspace=True)
y_all = (adult["income"] == ">50K").to_numpy(dtype=np.int64)
gen = np.random.default_rng(SEED)  # the lab's stratified 60/20/20 split
parts: list[list[IntArray]] = [[], [], []]
for c in np.unique(y_all):
    rows = gen.permutation(np.flatnonzero(y_all == c))
    n_te, n_va = round(0.2 * len(rows)), round(0.2 * len(rows))
    parts[2].append(rows[:n_te])
    parts[1].append(rows[n_te:n_te + n_va])
    parts[0].append(rows[n_te + n_va:])
i_tr, i_va, _ = (gen.permutation(np.concatenate(p)) for p in parts)
X_tr, X_va = adult.iloc[i_tr][NUMERIC + CATEGORICAL], adult.iloc[i_va][NUMERIC + CATEGORICAL]
y_tr, y_va = y_all[i_tr], y_all[i_va]
sex_tr, sex_va = adult.iloc[i_tr]["sex"].to_numpy(), adult.iloc[i_va]["sex"].to_numpy()
race_va = adult.iloc[i_va]["race"].to_numpy()
pre = ColumnTransformer([("num", StandardScaler(), NUMERIC), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
model = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000))]).fit(X_tr, y_tr)
s_va = model.predict_proba(X_va)[:, 1]
pred = (s_va >= 0.5).astype(np.int64)
print(f"val rows {len(y_va)}")

# %%
section("(a) audit by race with bootstrap CIs for the TPR gap to White")
mf = MetricFrame(metrics={"n": lambda y, p: len(y), "selection_rate": selection_rate, "TPR": true_positive_rate,
                          "FPR": false_positive_rate}, y_true=y_va, y_pred=pred, sensitive_features=race_va)
print(mf.by_group.round(4).to_string())
boot = np.random.default_rng(SEED)
white_pos = np.flatnonzero((race_va == "White") & (y_va == 1))
for g in ["Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]:
    g_pos = np.flatnonzero((race_va == g) & (y_va == 1))
    diffs = [pred[boot.choice(g_pos, len(g_pos))].mean() - pred[boot.choice(white_pos, len(white_pos))].mean()
             for _ in range(N_BOOT)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"TPR({g}) - TPR(White) = {pred[g_pos].mean() - pred[white_pos].mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")

# %%
section("(b) per-group thresholds that equalize FPR")


def rates(p: IntArray) -> dict[str, float]:
    out = {}
    for g in ["Male", "Female"]:
        m = sex_va == g
        out[g] = (p[m].mean(), p[m & (y_va == 1)].mean(), p[m & (y_va == 0)].mean())
    return out


def report(name: str, p: IntArray) -> None:
    rr = rates(p)
    dp = abs(rr["Male"][0] - rr["Female"][0])
    eo = abs(rr["Male"][1] - rr["Female"][1])
    fg = abs(rr["Male"][2] - rr["Female"][2])
    print(f"{name:<34} accuracy {np.mean(p == y_va):.4f}   DP {dp:.4f}  EOpp {eo:.4f}  FPR gap {fg:.4f}  "
          f"EOdds {max(eo, fg):.4f}   TPR M {rr['Male'][1]:.3f} F {rr['Female'][1]:.3f}")


target_fpr = pred[y_va == 0].mean()
th = {g: float(np.quantile(s_va[(sex_va == g) & (y_va == 0)], 1 - target_fpr, method="higher")) for g in ["Male", "Female"]}
pred_fpr = (s_va >= np.where(sex_va == "Male", th["Male"], th["Female"])).astype(np.int64)
print(f"target FPR {target_fpr:.4f}   thresholds M {th['Male']:.3f}  F {th['Female']:.3f}")
report("single threshold 0.5", pred)
report("equal FPR thresholds (from scratch)", pred_fpr)

# %%
section("(c) fairlearn ThresholdOptimizer, equalized odds")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    to = ThresholdOptimizer(estimator=model, constraints="equalized_odds", prefit=True, predict_method="predict_proba",
                            objective="accuracy_score")
    to.fit(X_tr, y_tr, sensitive_features=sex_tr)
    pred_to = to.predict(X_va, sensitive_features=sex_va, random_state=SEED).astype(np.int64)
report("ThresholdOptimizer (equalized odds)", pred_to)
np.testing.assert_allclose(equalized_odds_difference(y_va, pred_to, sensitive_features=sex_va),
                           max(abs(rates(pred_to)["Male"][1] - rates(pred_to)["Female"][1]),
                               abs(rates(pred_to)["Male"][2] - rates(pred_to)["Female"][2])))
print("equalized-odds difference matches fairlearn's: OK")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
