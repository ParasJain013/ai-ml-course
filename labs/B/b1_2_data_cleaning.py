# %% [markdown]
# # B1.2 Data cleaning — lab
# Lesson: lessons/B/b1_2_data_cleaning.md
# Run: uv run python labs/B/b1_2_data_cleaning.py   (CPU, defaults: ~5 s)
# GPU: no

# %%
from __future__ import annotations

import hashlib
import hmac
import os
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
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer

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
OUT = ROOT / "outputs" / "b1_2"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "B" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def section(name: str) -> None:
    print(f"\n== {name} ==")


def show(M: FloatArray) -> str:
    """Format a small array on one line, rounded to 4 decimals, without -0.0."""
    return str((np.round(M, 4) + 0.0).tolist())


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# Nothing here needs a GPU; GPU_SCALE only makes the synthetic table bigger.
GPU_SCALE = False
N_CUSTOMERS = 50_000 if GPU_SCALE else 2_000  # clean rows before the dirt is added
AS_OF = pd.Timestamp("2026-01-01")  # the date the snapshot was taken
SENTINEL = -999  # what the source system writes for "unknown"
# The key for hashing join keys. In real work it comes from a secret store, never from the code.
HASH_KEY = os.environ.get("PII_HASH_KEY", "course-demo-key-not-secret").encode()

# %%
# Data: a synthetic customer table, generated clean and then made dirty in the ways real exports are.
section("data")
FIRST = np.array(["Ana", "Ben", "Chloe", "Dev", "Ema", "Femi", "Gus", "Hana", "Ivan", "Jo", "Kai", "Lena"])
LAST = np.array(["Ahmed", "Brown", "Chen", "Diaz", "Evans", "Fofana", "Garcia", "Hughes", "Ito", "Jones"])
DISTRICTS = np.array([f"{a}{b}" for a in ("N", "E", "SE", "SW", "W") for b in range(1, 9)])  # 40 districts


def make_customers(n: int, gen: np.random.Generator) -> pd.DataFrame:
    """Generate n clean customer rows with direct identifiers, quasi-identifiers and a churn label."""
    first, last = gen.choice(FIRST, n), gen.choice(LAST, n)
    birth = pd.Timestamp("1950-01-01") + pd.to_timedelta(gen.integers(0, 55 * 365, n), unit="D")
    signup = pd.Timestamp("2023-01-01") + pd.to_timedelta(gen.integers(0, 3 * 365, n), unit="D")
    inward = [f"{gen.integers(1, 10)}{chr(65 + gen.integers(26))}{chr(65 + gen.integers(26))}" for _ in range(n)]
    plan = gen.choice(["basic", "pro", "team"], n, p=[0.6, 0.3, 0.1])
    spend = np.round(gen.lognormal(np.log(30), 0.6, n) * np.where(plan == "team", 4, 1), 2)
    sessions = gen.poisson(np.where(plan == "basic", 8, 20))
    churn_logit = -0.8 - 0.08 * sessions + 0.01 * spend
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
        "name": [f"{f} {l}" for f, l in zip(first, last)],
        "email": [f"{f.lower()}.{l.lower()}{i}@example.com" for i, (f, l) in enumerate(zip(first, last))],
        "phone": [f"+44 7{gen.integers(100, 999)} {gen.integers(100000, 999999)}" for _ in range(n)],
        "birth_date": birth,
        "sex": gen.choice(["F", "M"], n),
        "postcode": [f"{d} {w}" for d, w in zip(gen.choice(DISTRICTS, n), inward)],
        "country": gen.choice(["GB", "IE", "FR"], n, p=[0.8, 0.12, 0.08]),
        "plan": plan,
        "signup_date": signup,
        "monthly_spend": spend,
        "sessions_30d": sessions,
        "churned": (gen.random(n) < 1 / (1 + np.exp(-churn_logit))).astype(int),
    })


def make_dirty(clean: pd.DataFrame, gen: np.random.Generator) -> pd.DataFrame:
    """Apply the damage real exports have: strings for numbers, sentinels, typos, duplicates."""
    d = clean.copy()
    n = len(d)
    d["birth_date"] = d["birth_date"].dt.strftime("%Y-%m-%d")  # dates exported as text
    d["signup_date"] = d["signup_date"].dt.strftime("%Y-%m-%d")
    spend = d["monthly_spend"].map(lambda v: f"{v:.2f}")
    fmt = gen.random(n)
    spend[fmt < 0.15] = "£" + d.loc[fmt < 0.15, "monthly_spend"].map(lambda v: f"{v:,.2f}")  # currency text
    spend[(fmt >= 0.15) & (fmt < 0.20)] = " " + spend[(fmt >= 0.15) & (fmt < 0.20)] + " "  # stray spaces
    miss = gen.random(n)
    spend[miss < 0.05] = "N/A"
    spend[(miss >= 0.05) & (miss < 0.08)] = ""
    spend[(miss >= 0.08) & (miss < 0.10)] = str(SENTINEL)
    spend[(miss >= 0.10) & (miss < 0.11)] = "-" + spend[(miss >= 0.10) & (miss < 0.11)].str.strip("£ ")  # sign typo
    d["monthly_spend"] = spend
    sess = d["sessions_30d"].to_numpy().copy()
    r = gen.random(n)
    p_lost = np.where(d["churned"] == 1, 0.15, 0.03)  # tracking stops when a leaving user uninstalls
    sess[r < p_lost] = SENTINEL
    sess[(r >= p_lost) & (r < p_lost + 0.005)] = 99_999  # a counter that overflowed
    d["sessions_30d"] = sess
    d.loc[gen.random(n) < 0.02, "birth_date"] = "1900-01-01"  # the source's "unknown" date
    d.loc[gen.random(n) < 0.005, "birth_date"] = "2031-05-04"  # impossible: in the future
    variants = {"GB": ["GB", "UK", "uk", " United Kingdom", "gb"], "IE": ["IE", "Ireland", "ireland "], "FR": ["FR", "France"]}
    d["country"] = [gen.choice(variants[c]) for c in d["country"]]
    d["plan"] = [p if u > 0.1 else p.upper() if u > 0.05 else f" {p.capitalize()} " for p, u in zip(d["plan"], gen.random(n))]
    d.loc[gen.random(n) < 0.03, "plan"] = None
    exact = d.sample(30, random_state=SEED)  # exported twice
    near = d.sample(20, random_state=SEED + 1).copy()  # same person, email typed differently
    near["email"] = "  " + near["email"].str.upper()
    return pd.concat([d, exact, near], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


clean_truth = make_customers(N_CUSTOMERS, rng)
raw = make_dirty(clean_truth, rng)
print(f"raw table: {raw.shape[0]} rows x {raw.shape[1]} columns")
print(raw.dtypes.astype(str).to_dict())
print(raw[["customer_id", "birth_date", "country", "plan", "monthly_spend", "sessions_30d"]].head(6).to_string())

# %%
# Worked example by hand: parse eight spend strings, impute, flag outliers.
section("worked example by hand")
hand = ["12.5", "£1,040.00", "N/A", "-999", " 15 ", "14", "16", "13"]


def parse_money(s: object) -> float:
    """Turn '£1,040.00', ' 15 ', 'N/A', '' or '-999' into a float (NaN when unknown)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    t = str(s).strip().replace("£", "").replace(",", "")
    if t in {"", "N/A", "NA", "null", "None"}:
        return np.nan
    v = float(t)  # raises ValueError on anything else: better than guessing
    return np.nan if v == SENTINEL else v


x_h = np.array([parse_money(s) for s in hand])
obs = x_h[~np.isnan(x_h)]
print(f"parsed: {show(x_h)}   observed n = {len(obs)}   missing = {int(np.isnan(x_h).sum())}")
print(f"mean = {obs.sum():.1f} / {len(obs)} = {obs.mean():.4f}   median = {np.median(obs):.4f}")
print(f"mean-imputed:   {show(np.where(np.isnan(x_h), obs.mean(), x_h))}")
print(f"median-imputed: {show(np.where(np.isnan(x_h), np.median(obs), x_h))}   indicator: "
      f"{np.isnan(x_h).astype(int).tolist()}")
q1, q3 = np.quantile(obs, [0.25, 0.75])
print(f"sorted observed {np.sort(obs).tolist()}   Q1 = {q1:.4f}   Q3 = {q3:.4f}   IQR = {q3 - q1:.4f}   "
      f"fences [{q1 - 1.5 * (q3 - q1):.4f}, {q3 + 1.5 * (q3 - q1):.4f}]")
z_h = (obs - obs.mean()) / obs.std()
print(f"z-scores (ddof=0): {show(z_h)}   largest possible |z| for n = 6: sqrt(n - 1) = "
      f"{np.sqrt(5):.4f}")

# %%
# From scratch: the cleaning function, with a log of every decision and the rows it touched.
section("from scratch: cleaning with a log")
COUNTRY_MAP = {"gb": "GB", "uk": "GB", "united kingdom": "GB", "ie": "IE", "ireland": "IE", "fr": "FR", "france": "FR"}
MAX_SESSIONS = 2_000  # 30 days x ~65 sessions a day: anything above is a broken counter
OLDEST = pd.Timestamp("1906-01-01")


def clean_customers(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fix types, sentinels, impossible values, labels and duplicates. -> (clean frame, log lines)."""
    log: list[str] = []
    d = df.copy()
    n0 = len(d)

    d = d.drop_duplicates()
    log.append(f"dropped {n0 - len(d)} exact duplicate rows")
    d["email"] = d["email"].str.strip().str.lower()
    n1 = len(d)
    d = d.drop_duplicates(subset=["email"], keep="first")
    log.append(f"dropped {n1 - len(d)} near-duplicates (same email after strip + lowercase)")

    spend = d["monthly_spend"].map(parse_money)
    log.append(f"monthly_spend: text -> float; {int(spend.isna().sum())} unknown ('', 'N/A', {SENTINEL})")
    neg = spend < 0
    spend[neg] = np.nan
    log.append(f"monthly_spend: {int(neg.sum())} negative values -> NaN (impossible for a subscription)")
    d["monthly_spend"] = spend

    sess = d["sessions_30d"].astype("float64")
    sent = sess == SENTINEL
    big = sess > MAX_SESSIONS
    sess[sent | big] = np.nan
    d["sessions_30d"] = sess
    log.append(f"sessions_30d: {int(sent.sum())} sentinel {SENTINEL} -> NaN; {int(big.sum())} > {MAX_SESSIONS} -> NaN")

    birth = pd.to_datetime(d["birth_date"], format="%Y-%m-%d")
    bad_birth = (birth < OLDEST) | (birth > AS_OF)
    d["birth_date"] = birth.where(~bad_birth)
    log.append(f"birth_date: text -> date; {int(bad_birth.sum())} placeholder or future dates -> NaT")
    d["signup_date"] = pd.to_datetime(d["signup_date"], format="%Y-%m-%d")

    before = d["country"].nunique()
    d["country"] = d["country"].str.strip().str.lower().map(COUNTRY_MAP)
    log.append(f"country: {before} spellings -> {d['country'].nunique()} codes; unmapped: {int(d['country'].isna().sum())}")
    before = d["plan"].nunique()
    d["plan"] = d["plan"].str.strip().str.lower()
    log.append(f"plan: {before} spellings -> {d['plan'].nunique()}; missing kept as NaN: {int(d['plan'].isna().sum())}")
    log.append(f"rows: {n0} -> {len(d)}")
    return d.reset_index(drop=True), log


df, cleaning_log = clean_customers(raw)
for line in cleaning_log:
    print(" -", line)
print(f"missing after cleaning (share): "
      f"{ {c: round(float(v), 4) for c, v in df.isna().mean().items() if v > 0} }")
print(f"recovered spend vs truth (same customers): max |difference| = "
      f"{np.nanmax(np.abs(df.set_index('customer_id')['monthly_spend'] - clean_truth.set_index('customer_id')['monthly_spend'])):.4f}")

# %%
# From scratch: imputation. Learn the fill values on the training rows only, then apply everywhere.
section("from scratch: imputation")
order = rng.permutation(len(df))
train_rows, test_rows = order[: int(0.8 * len(df))], order[int(0.8 * len(df)) :]
num_cols = ["monthly_spend", "sessions_30d"]
X_num = df[num_cols].to_numpy(dtype=np.float64)  # (n, 2)
X_tr, X_te = X_num[train_rows], X_num[test_rows]


def fit_impute(X: FloatArray, strategy: str) -> FloatArray:
    """Learn one fill value per column, ignoring NaN. X (n, d) -> (d,)."""
    if strategy == "mean":
        return np.nanmean(X, axis=0)
    if strategy == "median":
        return np.nanmedian(X, axis=0)
    raise ValueError(f"unknown strategy {strategy!r}")


def transform_impute(X: FloatArray, fill: FloatArray, add_indicator: bool = False) -> FloatArray:
    """Replace NaN in column j by fill[j]; optionally append 0/1 'was missing' columns. (n, d) -> (n, d[+d])."""
    miss = np.isnan(X)  # (n, d)
    out = np.where(miss, fill[None, :], X)  # (n, d) broadcast the (1, d) fill row
    return np.hstack([out, miss.astype(np.float64)]) if add_indicator else out


fill_mean, fill_median = fit_impute(X_tr, "mean"), fit_impute(X_tr, "median")
print(f"train fill values  mean: {show(fill_mean)}   median: {show(fill_median)}")
Xte_mean = transform_impute(X_te, fill_mean)
Xte_ind = transform_impute(X_te, fill_median, add_indicator=True)
print(f"test shape {X_te.shape} -> mean-imputed {Xte_mean.shape}   median + indicator {Xte_ind.shape}")
sp_obs = X_tr[:, 0][~np.isnan(X_tr[:, 0])]
sp_mean_imp = transform_impute(X_tr, fill_mean)[:, 0]
print(f"monthly_spend on train: observed sd {sp_obs.std(ddof=1):.4f}   after mean imputation sd "
      f"{sp_mean_imp.std(ddof=1):.4f}   share of rows equal to the fill value "
      f"{np.mean(sp_mean_imp == fill_mean[0]):.4f}")
churn_tr = df["churned"].to_numpy()[train_rows]
miss_sess = np.isnan(X_tr[:, 1])
print(f"churn rate when sessions_30d is missing {churn_tr[miss_sess].mean():.4f} (n = {int(miss_sess.sum())}) "
      f"vs observed {churn_tr[~miss_sess].mean():.4f} (n = {int((~miss_sess).sum())})")
print("-> here missingness is itself a signal (tracking stops when users leave); the indicator keeps it")

# %%
# From scratch: outlier flags. Flag, look, decide; never delete by rule alone.
section("from scratch: outlier flags")


def iqr_flags(x: FloatArray, k: float = 1.5) -> FloatArray:
    """True outside Tukey's fences [Q1 - k IQR, Q3 + k IQR]. x (n,) -> (n,) bool."""
    q1, q3 = np.quantile(x, [0.25, 0.75])
    return (x < q1 - k * (q3 - q1)) | (x > q3 + k * (q3 - q1))


def z_flags(x: FloatArray, thr: float = 3.0) -> FloatArray:
    """True where |x - mean| / sd > thr (sd with ddof=0). x (n,) -> (n,) bool."""
    return np.abs((x - x.mean()) / x.std()) > thr


sp = df["monthly_spend"].dropna().to_numpy()
print(f"monthly_spend (n = {len(sp)}): IQR flags {int(iqr_flags(sp).sum())}   z > 3 flags {int(z_flags(sp).sum())}")
print(f"log(monthly_spend):          IQR flags {int(iqr_flags(np.log(sp)).sum())}   z > 3 flags "
      f"{int(z_flags(np.log(sp)).sum())}")
team = df.loc[df["monthly_spend"].notna(), "plan"].to_numpy() == "team"
print(f"of the IQR-flagged raw spends, share on the 'team' plan: {team[iqr_flags(sp)].mean():.4f}   "
      f"(share of all customers on team: {team.mean():.4f})")
print("-> the 'outliers' are mostly real team customers: keep them, maybe model log(spend)")

# %%
# From scratch: the PII pass. Drop direct identifiers, hash the join key, coarsen quasi-identifiers.
section("from scratch: PII pass")
DIRECT_IDENTIFIERS = ["name", "email", "phone"]


def pseudonymize(values: pd.Series, key: bytes) -> pd.Series:
    """Keyed hash (HMAC-SHA256) of each id, first 16 hex characters. Same id + key -> same token."""
    return values.map(lambda v: hmac.new(key, str(v).encode(), hashlib.sha256).hexdigest()[:16])


def age_band(birth: pd.Series, as_of: pd.Timestamp) -> pd.Series:
    """Coarsen a birth date to a 10-year age band ('25-34'); NaT -> NaN."""
    age = (as_of - birth).dt.days // 365.25
    edges = [0, 18, 25, 35, 45, 55, 65, 200]
    labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    return pd.cut(age, bins=edges, right=False, labels=labels).astype("object")


def share_unique(frame: pd.DataFrame, cols: list[str]) -> tuple[float, int]:
    """Share of rows whose combination of cols is unique, and the smallest group size (k)."""
    sizes = frame.groupby(cols, dropna=False)[cols[0]].transform("size")
    return float(np.mean(sizes == 1)), int(sizes.min())


raw_qi = ["birth_date", "postcode", "sex"]
u_raw, k_raw = share_unique(df, raw_qi)
safe = df.drop(columns=DIRECT_IDENTIFIERS).copy()
safe["customer_key"] = pseudonymize(safe.pop("customer_id"), HASH_KEY)
safe["age_band"] = age_band(safe.pop("birth_date"), AS_OF)
safe["district"] = safe.pop("postcode").str.split(" ").str[0]
u_mid, k_mid = share_unique(safe.assign(birth=df["birth_date"].dt.year), ["birth", "district", "sex"])
u_safe, k_safe = share_unique(safe, ["age_band", "district", "sex"])
print(f"dropped {DIRECT_IDENTIFIERS}; customer_id -> keyed hash, e.g. C00001 -> "
      f"{pseudonymize(pd.Series(['C00001']), HASH_KEY).iloc[0]}")
print(f"columns kept: {list(safe.columns)}")
print(f"unique on (birth date, full postcode, sex):  {u_raw:.4f} of rows   smallest group k = {k_raw}")
print(f"unique on (birth year, district, sex):       {u_mid:.4f} of rows   smallest group k = {k_mid}")
print(f"unique on (age band, district, sex):         {u_safe:.4f} of rows   smallest group k = {k_safe}")
unsalted = hashlib.sha256(b"C00001").hexdigest()[:16]
print(f"why a key: plain sha256('C00001') = {unsalted} can be rebuilt by anyone who guesses the id format")

# %%
# With scikit-learn and SciPy: SimpleImputer and scipy.stats agree with the from-scratch versions.
section("with scikit-learn and scipy")
for strategy, mine in (("mean", fill_mean), ("median", fill_median)):
    imp = SimpleImputer(strategy=strategy).fit(X_tr)
    np.testing.assert_allclose(imp.statistics_, mine, rtol=1e-12)
    np.testing.assert_allclose(imp.transform(X_te), transform_impute(X_te, mine), rtol=1e-12)
    print(f"SimpleImputer(strategy={strategy!r}) statistics_ {show(imp.statistics_)}   matches mine: OK")
imp_ind = SimpleImputer(strategy="median", add_indicator=True).fit(X_tr)
np.testing.assert_allclose(imp_ind.transform(X_te), Xte_ind, rtol=1e-12)
print(f"SimpleImputer(add_indicator=True) -> {imp_ind.transform(X_te).shape}   matches mine: OK")
np.testing.assert_allclose(np.abs(stats.zscore(sp)) > 3, z_flags(sp))
print("scipy.stats.zscore flags match z_flags: OK")
q = np.percentile(sp, [25, 75])
np.testing.assert_allclose(stats.iqr(sp), q[1] - q[0], rtol=1e-12)
print(f"scipy.stats.iqr = {stats.iqr(sp):.4f}   matches np.quantile: OK")

# %%
# Key plot: what mean imputation does to a distribution, and how coarsening hides individuals.
section("key plot")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=120)
ax = axes[0]
bins = np.linspace(0, 150, 61)
ax.hist(sp_obs, bins=bins, alpha=0.6, label="observed values")
ax.hist(sp_mean_imp, bins=bins, histtype="step", lw=1.8, color="tab:red", label="after mean imputation")
ax.axvline(fill_mean[0], color="tab:red", ls="--", lw=1)
ax.axvline(fill_median[0], color="k", ls=":", lw=1.2, label=f"median {fill_median[0]:.1f}")
ax.set(xlabel="monthly spend (GBP, cut at 150)", ylabel="customers (train rows)",
       title=f"Mean imputation stacks {np.mean(sp_mean_imp == fill_mean[0]):.0%} of rows on {fill_mean[0]:.1f}")
ax.legend(loc="upper right", fontsize=8)
ax = axes[1]
names = ["birth date +\nfull postcode + sex", "birth year +\ndistrict + sex", "age band +\ndistrict + sex"]
vals = [u_raw, u_mid, u_safe]
bars = ax.bar(names, vals, color=["tab:red", "tab:orange", "tab:blue"], hatch="/")
for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center")
ax.set_ylim(0, 1.1)
ax.set(ylabel="share of customers unique on these columns",
       title=f"Coarsening quasi-identifiers: {u_raw:.0%} unique -> {u_safe:.0%}")
fig.tight_layout()
path_png = IMG / "b1_2_imputation_and_reidentification.png"
fig.savefig(path_png, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path_png.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Change MAX_SESSIONS to 50. How many real customers does the "impossible value" rule now delete?
# 2. Impute monthly_spend by plan (median per plan, learned on train). Does the spike in the plot shrink?
# 3. Add 'sex' and 'plan' to the quasi-identifiers in share_unique. How unique is everyone now?

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
