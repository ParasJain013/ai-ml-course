# %% [markdown]
# # A1.3 pandas — answer to exercise 2 (and a check of exercise 1)
# Run: uv run python answers/lessons/A/a1_3_pandas_ex2.py   (CPU, ~1 s)

# %%
from __future__ import annotations

import sqlite3
import time

import numpy as np
import pandas as pd

SEED = 0
T0 = time.perf_counter()
pd.set_option("display.width", 100)


def section(name: str) -> None:
    print(f"\n== {name} ==")


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
con = sqlite3.connect(":memory:")
customers.to_sql("customers", con, index=False)
orders.to_sql("orders", con, index=False)


def sql(query: str) -> pd.DataFrame:
    """Run a query on the in-memory database."""
    return pd.read_sql_query(query, con)


# %%
section("exercise 1 check")
print("(a) rows:", len(orders.merge(customers, on="customer_id", how="left")))
print("(b)")
print(
    sql(
        "SELECT status, COUNT(*) AS n, COUNT(amount) AS n_amount, AVG(amount) AS avg "
        "FROM orders GROUP BY status ORDER BY status"
    ).round(4).to_string(index=False)
)
print("(c)")
print(
    sql(
        "SELECT c.signup_year, COUNT(o.order_id) AS n_order_id, COUNT(*) AS n_rows "
        "FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id "
        "GROUP BY c.signup_year ORDER BY c.signup_year"
    ).to_string(index=False)
)

# %%
section("exercise 2: city summary")
city_sql = sql(
    """
    SELECT c.city,
           COUNT(DISTINCT c.customer_id) AS n_customers,
           SUM(CASE WHEN o.status = 'paid' THEN 1 ELSE 0 END) AS n_paid,
           AVG(CASE WHEN o.status = 'refunded' THEN 1.0
                    WHEN o.status IS NOT NULL THEN 0.0 END) AS refund_share
    FROM customers AS c
    LEFT JOIN orders AS o ON c.customer_id = o.customer_id
    GROUP BY c.city
    ORDER BY c.city
    """
)
joined = customers.merge(orders, on="customer_id", how="left", validate="one_to_many")
joined["is_paid"] = (joined["status"] == "paid").astype(int)
# 1.0 / 0.0 for real orders, NaN for the "no order" rows so the mean skips them (like SQL AVG)
joined["is_refund"] = (joined["status"] == "refunded").astype(float).where(joined["status"].notna())
city_pd = (
    joined.groupby("city", dropna=False)
    .agg(
        n_customers=("customer_id", "nunique"),
        n_paid=("is_paid", "sum"),
        refund_share=("is_refund", "mean"),
    )
    .reset_index()
    .sort_values("city", na_position="first")  # SQLite sorts NULL first
    .reset_index(drop=True)
)
print(city_pd.round(4).to_string(index=False))
pd.testing.assert_frame_equal(city_sql, city_pd, check_dtype=False)
print("SQL and pandas agree: OK")

# %%
section("exercise 2: the double-count trap")
payments = pd.DataFrame(
    {
        "order_id": [101, 102, 103, 104, 106, 106, 107, 109, 110],  # 106 paid in two parts
        "method": ["card", "card", "upi", "card", "card", "gift", "upi", "card", "card"],
    }
)
paid = orders[orders["status"] == "paid"]
true_revenue = paid["amount"].sum()
naive = paid.merge(payments, on="order_id", how="left")
print("paid orders:", len(paid), " rows after merge:", len(naive))
print("true revenue:", true_revenue, " naive revenue after merge:", naive["amount"].sum())
try:
    paid.merge(payments, on="order_id", how="left", validate="one_to_one")
except pd.errors.MergeError as err:
    print("validate='one_to_one' raised MergeError:", str(err).splitlines()[0])
dedup = payments.groupby("order_id", as_index=False).agg(methods=("method", lambda s: "+".join(sorted(s))))
fixed = paid.merge(dedup, on="order_id", how="left", validate="one_to_one")
print("after aggregating payments first:", fixed["amount"].sum(), " order 106 methods:", fixed.loc[fixed["order_id"] == 106, "methods"].iloc[0])
assert np.isclose(fixed["amount"].sum(), true_revenue)

# %%
con.close()
section("done")
print(f"~ elapsed: {time.perf_counter() - T0:.1f} s")
