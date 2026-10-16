# %% [markdown]
# # A1.3 pandas — lab
# Lesson: lessons/A/a1_3_pandas.md
# Run: uv run python labs/A/a1_3_pandas.py   (CPU, defaults: ~5 s)
# GPU: no

# %%
from __future__ import annotations

import sqlite3
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
OUT = ROOT / "outputs" / "a1_3"  # scratch figures, gitignored
IMG = ROOT / "lessons" / "A" / "img"  # the lesson's key plot, committed
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 100)


def section(name: str) -> None:
    print(f"\n== {name} ==")


# %%
# GPU_SCALE: full-size settings for a free Colab/Kaggle GPU. Defaults fit the laptop CPU in < 5 min.
# The tables are hand-made and tiny so every number can be checked by hand; nothing scales.
GPU_SCALE = False
N_SYNTHETIC = 1_000_000 if GPU_SCALE else 100_000  # rows in the dict-vs-groupby timing

# %%
# Data: two small tables, built in Python, loaded into both SQLite and pandas.
section("data")
customers = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4, 5, 6],
        "name": ["Asha", "Ben", "Chen", "Dana", "Eli", "Farah"],
        "city": ["Pune", "Leeds", "Pune", "Austin", "Leeds", None],
        "signup_year": [2023, 2024, 2024, 2023, 2025, 2025],
    }
)
orders = pd.DataFrame(
    {
        "order_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "customer_id": [1, 1, 2, 3, 3, 4, 4, 4, 2, 9],
        "amount": [120.0, 80.0, 200.0, None, 50.0, 300.0, 100.0, 20.0, 60.0, 75.0],
        "status": ["paid", "paid", "paid", "paid", "refunded", "paid", "paid", "refunded", "paid", "paid"],
    }
)
print(customers)
print(orders)
print(orders.dtypes.to_string())
print("missing per column:", orders.isna().sum().to_dict())

con = sqlite3.connect(":memory:")
customers.to_sql("customers", con, index=False)
orders.to_sql("orders", con, index=False)


def sql(query: str) -> pd.DataFrame:
    """Run a query on the in-memory SQLite database and return a DataFrame."""
    return pd.read_sql_query(query, con)


def check(name: str, from_sql: pd.DataFrame, from_pandas: pd.DataFrame) -> None:
    """Assert two result tables hold the same rows in the same order, then print it."""
    left = from_sql.reset_index(drop=True)
    right = from_pandas.reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)
    print(f"{name}: SQL and pandas agree ({len(left)} rows)")


# %%
# Worked example: the numbers the lesson computes by hand.
section("worked example")
c3 = orders[orders["customer_id"] == 3]
print(c3)
print("COUNT(*):", len(c3), " COUNT(amount):", int(c3["amount"].count()))
print("SUM(amount):", c3["amount"].sum(), " AVG(amount):", c3["amount"].mean())

# %%
# Q1: filter (WHERE).
section("Q1 filter")
q1_sql = sql(
    """
    SELECT order_id, customer_id, amount
    FROM orders
    WHERE status = 'paid' AND amount >= 100
    ORDER BY order_id
    """
)
q1_pd = orders.loc[
    (orders["status"] == "paid") & (orders["amount"] >= 100),
    ["order_id", "customer_id", "amount"],
].sort_values("order_id")
print(q1_pd.to_string(index=False))
check("Q1", q1_sql, q1_pd)

# %%
# Q2: GROUP BY with aggregates.
section("Q2 group by")
q2_sql = sql(
    """
    SELECT customer_id,
           COUNT(*)      AS n_orders,
           COUNT(amount) AS n_with_amount,
           SUM(amount)   AS total,
           AVG(amount)   AS avg_amount
    FROM orders
    GROUP BY customer_id
    ORDER BY customer_id
    """
)
q2_pd = (
    orders.groupby("customer_id")
    .agg(
        n_orders=("order_id", "size"),
        n_with_amount=("amount", "count"),
        total=("amount", "sum"),
        avg_amount=("amount", "mean"),
    )
    .reset_index()
    .sort_values("customer_id")
)
print(q2_pd.to_string(index=False))
check("Q2", q2_sql, q2_pd)

# %%
# Q3: INNER JOIN.
section("Q3 inner join")
q3_sql = sql(
    """
    SELECT o.order_id, c.name, c.city, o.amount
    FROM orders AS o
    JOIN customers AS c ON o.customer_id = c.customer_id
    ORDER BY o.order_id
    """
)
q3_pd = orders.merge(customers, on="customer_id", how="inner")[
    ["order_id", "name", "city", "amount"]
].sort_values("order_id")
print(q3_pd.to_string(index=False))
print("orders:", len(orders), " rows after inner join:", len(q3_pd), " (order 110 has no customer)")
check("Q3", q3_sql, q3_pd)

# %%
# Q4: LEFT JOIN, and what happens to the rows with no match.
section("Q4 left join")
q4_sql = sql(
    """
    SELECT c.customer_id, c.name, o.order_id, o.amount
    FROM customers AS c
    LEFT JOIN orders AS o ON c.customer_id = o.customer_id
    ORDER BY c.customer_id, o.order_id
    """
)
q4_pd = customers.merge(orders, on="customer_id", how="left")[
    ["customer_id", "name", "order_id", "amount"]
].sort_values(["customer_id", "order_id"])
print(q4_pd.to_string(index=False))
print("order_id dtype after the left join:", q4_pd["order_id"].dtype, "(was", orders["order_id"].dtype, ")")
check("Q4", q4_sql, q4_pd)

# %%
# Q5: GROUP BY ... HAVING (filter on an aggregate).
section("Q5 having")
q5_sql = sql(
    """
    SELECT c.city, COUNT(*) AS n_orders, SUM(o.amount) AS total
    FROM orders AS o
    JOIN customers AS c ON o.customer_id = c.customer_id
    WHERE o.status = 'paid'
    GROUP BY c.city
    HAVING SUM(o.amount) > 250
    ORDER BY total DESC
    """
)
paid = orders[orders["status"] == "paid"].merge(customers, on="customer_id", how="inner")
by_city = (
    paid.groupby("city")
    .agg(n_orders=("order_id", "size"), total=("amount", "sum"))
    .reset_index()
)
print("before HAVING:\n" + by_city.to_string(index=False))
q5_pd = by_city[by_city["total"] > 250].sort_values("total", ascending=False)
check("Q5", q5_sql, q5_pd)

# %%
# From scratch: a group-by mean with plain dictionaries, against groupby().mean().
section("from scratch: group-by mean")


def groupby_mean(keys: list[int], values: list[float | None]) -> dict[int, float]:
    """Mean of the non-missing values per key, like SQL AVG or pandas mean (skipna)."""
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for key, value in zip(keys, values, strict=True):
        if value is None or value != value:  # skip None and NaN (NaN != NaN)
            continue
        sums[key] = sums.get(key, 0.0) + value
        counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sorted(sums)}


mine = groupby_mean(orders["customer_id"].tolist(), orders["amount"].tolist())
theirs = orders.groupby("customer_id")["amount"].mean().to_dict()
print("mine:  ", mine)
print("pandas:", theirs)
assert mine.keys() == theirs.keys()
np.testing.assert_allclose(list(mine.values()), list(theirs.values()), rtol=1e-6, atol=1e-8)
print("matches pandas groupby().mean(): OK")

big = pd.DataFrame(
    {"key": rng.integers(0, 100, N_SYNTHETIC), "value": rng.normal(size=N_SYNTHETIC)}
)
t = time.perf_counter()
mine_big = groupby_mean(big["key"].tolist(), big["value"].tolist())
t_dict = time.perf_counter() - t
t = time.perf_counter()
theirs_big = big.groupby("key")["value"].mean()
t_pd = time.perf_counter() - t
np.testing.assert_allclose(list(mine_big.values()), theirs_big.to_numpy(), rtol=1e-6, atol=1e-8)
print(f"{N_SYNTHETIC:,} rows: dict and pandas agree: OK")
print(f"~ dict {t_dict * 1e3:.1f} ms, pandas {t_pd * 1e3:.1f} ms")

# %%
# Missing values: where pandas and SQL treat NULL / NaN differently.
section("missing values")
print("SQL GROUP BY city:")
print(sql("SELECT city, COUNT(*) AS n FROM customers GROUP BY city ORDER BY city").to_string(index=False))
print("pandas groupby('city').size():", customers.groupby("city").size().to_dict())
print("pandas groupby('city', dropna=False).size():", customers.groupby("city", dropna=False).size().to_dict())
all_missing = pd.Series([None, None], dtype="float64")
print("SQL SUM of only NULLs:", sql("SELECT SUM(amount) AS s FROM orders WHERE order_id = 104")["s"].iloc[0])
print("pandas sum of only NaN:", all_missing.sum(), " with min_count=1:", all_missing.sum(min_count=1))
print("NaN == NaN:", np.nan == np.nan, " isna() finds it:", bool(pd.isna(np.nan)))
filled = orders["amount"].fillna(orders["amount"].median())
print("median used to fill:", orders["amount"].median(), " order 104 now:", filled.iloc[3])

# %%
# Copy-on-write: chained assignment does nothing in pandas 3.
section("chained assignment")
df = orders.copy()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    df[df["status"] == "refunded"]["amount"] = 0.0  # chained: writes into a temporary copy
print("warning raised:", [w.category.__name__ for w in caught])
print("refunded amounts after chained assignment:", df.loc[df["status"] == "refunded", "amount"].tolist())
df.loc[df["status"] == "refunded", "amount"] = 0.0  # one .loc call: writes into df
print("refunded amounts after .loc assignment:   ", df.loc[df["status"] == "refunded", "amount"].tolist())

# %%
# Key plot: paid revenue per customer, via LEFT JOIN so customers with no orders still appear.
section("key plot")
revenue = (
    customers.merge(orders[orders["status"] == "paid"], on="customer_id", how="left")
    .groupby(["customer_id", "name"], as_index=False)
    .agg(n_paid=("order_id", "count"), amount=("amount", "sum"))  # sum of only NaN -> 0.0
    .sort_values("customer_id")
)
print(revenue.to_string(index=False))
fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
bars = ax.bar(revenue["name"], revenue["amount"], color="tab:blue", edgecolor="black")
for bar, value, n_paid in zip(bars, revenue["amount"], revenue["n_paid"], strict=True):
    if n_paid == 0:
        label = "0\n(no orders)"
    elif value == 0:
        label = "0\n(amount\nmissing)"
    else:
        label = f"{value:.0f}"
    ax.annotate(label, (bar.get_x() + bar.get_width() / 2, value), ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, revenue["amount"].max() * 1.15)
ax.set_xlabel("customer")
ax.set_ylabel("paid revenue (currency units)")
ax.set_title("A LEFT JOIN keeps Eli and Farah; an inner join drops them")
fig.tight_layout()
path = IMG / "a1_3_left_join_revenue.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print("saved", path.relative_to(ROOT).as_posix())

# %%
# Try this (not graded):
# 1. Change Q3 to how="outer" and add indicator=True. Which rows say "left_only" / "right_only"?
# 2. In Q2, replace "size" with "count" for n_orders. Which customer's number changes, and why?
# 3. Write Q5 with .query("total > 250") instead of a boolean mask.

# %%
con.close()
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
