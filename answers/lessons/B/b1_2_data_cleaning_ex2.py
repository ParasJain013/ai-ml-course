# %% [markdown]
# # B1.2 Data cleaning — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/B/b1_2_data_cleaning_ex2.py   (CPU, ~2 s)

# %%
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]
SEED = 0
rng = np.random.default_rng(SEED)
T0 = time.perf_counter()
N_CUSTOMERS = 2_000
AS_OF = pd.Timestamp("2026-01-01")
SENTINEL = -999


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# Exercise 1 check.
section("exercise 1 check")
raw_h = ["7", "£12", "-999", "9", "", "8", "250", "10", "N/A", "11"]


def parse_money(s: object) -> float:
    """Turn '£1,040.00', ' 15 ', 'N/A', '' or '-999' into a float (NaN when unknown)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    t = str(s).strip().replace("£", "").replace(",", "")
    if t in {"", "N/A", "NA", "null", "None"}:
        return np.nan
    v = float(t)  # raises ValueError on anything else: better than guessing
    return np.nan if v == SENTINEL else v


x = np.array([parse_money(s) for s in raw_h])
obs = x[~np.isnan(x)]
pi = float(np.isnan(x).mean())
print(f"parsed {np.round(x, 4).tolist()}   observed {obs.tolist()}   n_obs {len(obs)}   missing share {pi:.2f}")
print(f"mean {obs.sum():.0f} / {len(obs)} = {obs.mean():.4f}   median {np.median(obs):.4f}")
print(f"mean-imputed   {np.round(np.where(np.isnan(x), obs.mean(), x), 2).tolist()}")
print(f"median-imputed {np.where(np.isnan(x), np.median(obs), x).tolist()}")
sd_obs = obs.std()
filled = np.where(np.isnan(x), obs.mean(), x)
print(f"sd observed (divide by n) {sd_obs:.4f}   predicted after mean imputation sqrt(1 - {pi:.1f}) * sd = "
      f"{np.sqrt(1 - pi) * sd_obs:.4f}   actual {filled.std():.4f}")
q1, q3 = np.quantile(obs, [0.25, 0.75])
lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
z = (obs - obs.mean()) / sd_obs
print(f"Q1 {q1}  Q3 {q3}  IQR {q3 - q1}  fences [{lo}, {hi}]  Tukey flags {obs[(obs < lo) | (obs > hi)].tolist()}")
print(f"z of 250 = {z[obs == 250][0]:.4f}   ceiling sqrt(n_obs - 1) = sqrt(6) = {np.sqrt(6):.4f}   "
      f"|z| > 3 flags {obs[np.abs(z) > 3].tolist()}")

# %%
# The lab's generator and cleaning function, copied.
# from labs/B/b1_2_data_cleaning.py
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


raw = make_dirty(make_customers(N_CUSTOMERS, rng), rng)
df, _ = clean_customers(raw)


# %%
# Exercise 2: audit_table reports problems; it never changes the data.
EMAIL_RE = re.compile(r"^\s*[\w.+-]+@[\w-]+\.[\w.]+\s*$")
PHONE_RE = re.compile(r"^\s*(\+|0)\d[\d\s]{7,}\s*$")  # +44 7123 456789 or 07123 456789; not dates
SENTINELS = {"-999", "-999.0", "N/A", "", "1900-01-01"}


@dataclass
class Rules:
    """What the clean table must look like. Every field is optional."""

    dtypes: dict[str, str] = field(default_factory=dict)  # column -> expected dtype kind ("f", "i", "M", "O")
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)  # numeric column -> [low, high]
    allowed: dict[str, set[str]] = field(default_factory=dict)  # categorical column -> allowed values
    key: str | None = None  # must be unique after strip + lowercase
    max_missing: float = 0.2  # largest acceptable missing share per column
    accepted_pii: set[str] = field(default_factory=set)  # columns knowingly kept (to be dropped in the PII pass)


def audit_table(df: pd.DataFrame, rules: Rules) -> list[str]:
    """Check df against rules. -> one readable line per problem (empty list: clean)."""
    problems: list[str] = []
    for col, kind in rules.dtypes.items():
        got = df[col].dtype.kind
        if got != kind and not (kind == "O" and got in "OUT"):
            problems.append(f"type: {col} is {df[col].dtype} (kind {got!r}), expected kind {kind!r}")
    for col in df.columns:
        as_text = df[col].astype("string").str.strip()
        hits = int(as_text.isin(SENTINELS).sum())
        if hits:
            problems.append(f"sentinel: {col} has {hits} values in {sorted(SENTINELS)}")
    for col, (low, high) in rules.ranges.items():
        v = pd.to_numeric(df[col], errors="coerce")  # coercion is fine here: we only count
        bad = int(((v < low) | (v > high)).sum())
        if bad:
            problems.append(f"range: {col} has {bad} values outside [{low}, {high}]")
    for col, ok in rules.allowed.items():
        vals = df[col].dropna()
        bad = sorted(set(vals.astype(str)) - ok)
        if bad:
            problems.append(f"category: {col} has {len(bad)} unexpected values, e.g. {bad[:4]}")
    if rules.key is not None:
        dup = int(df[rules.key].astype("string").str.strip().str.lower().duplicated().sum())
        if dup:
            problems.append(f"key: {dup} duplicate {rules.key} values after strip + lowercase")
    for col, share in df.isna().mean().items():
        if share > rules.max_missing:
            problems.append(f"missing: {col} is {share:.1%} missing (limit {rules.max_missing:.0%})")
    for col in df.columns:
        if df[col].dtype.kind not in "OUT" or col in rules.accepted_pii:
            continue
        sample = df[col].dropna().astype(str).head(200)
        for kind, rx in (("email addresses", EMAIL_RE), ("phone numbers", PHONE_RE)):
            if len(sample) and sample.str.match(rx).mean() > 0.5:
                problems.append(f"PII: {col} looks like {kind}")
    return problems


section("exercise 2: audit raw vs cleaned")
rules = Rules(
    dtypes={"monthly_spend": "f", "sessions_30d": "f", "birth_date": "M", "signup_date": "M"},
    ranges={"monthly_spend": (0, 10_000), "sessions_30d": (0, 2_000)},
    allowed={"country": {"GB", "IE", "FR"}, "plan": {"basic", "pro", "team"}},
    key="email",
    max_missing=0.2,
)
raw_problems = audit_table(raw, rules)
print(f"raw table: {len(raw_problems)} problems")
for p in raw_problems:
    print("  -", p)
clean_problems = audit_table(df, rules)
print(f"cleaned table: {len(clean_problems)} problems")
for p in clean_problems:
    print("  -", p)
rules.accepted_pii = {"email", "phone"}  # known; the PII pass (lesson) drops them before modelling
print(f"cleaned table, with email and phone accepted until the PII pass: {audit_table(df, rules)}")

# %%
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
