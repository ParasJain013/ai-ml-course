# Answer key: A1.3 pandas

Lesson: [`lessons/A/a1_3_pandas.md`](../../../lessons/A/a1_3_pandas.md)

## Exercise 1 (by hand)

**Solution.**

(a) `orders LEFT JOIN customers`, orders on the left. Every order keeps its row. Orders for customers
1–4 match exactly one customer each ($n_R(k) = 1$), and order 110 (customer 9) matches none, so it is
kept once with missing customer columns. Rows $= 9 \cdot 1 + 1 = 10$, the same as `orders`. A left
join onto a table whose key is unique never changes the left table's row count; that is a useful
invariant to assert.

(b) Per `status`:

| status | rows | `COUNT(*)` | `COUNT(amount)` | `AVG(amount)` |
|---|---|---|---|---|
| paid | 101, 102, 103, 104, 106, 107, 109, 110 | 8 | 7 (104 missing) | $(120 + 80 + 200 + 300 + 100 + 60 + 75)/7 = 935/7 \approx 133.5714$ |
| refunded | 105, 108 | 2 | 2 | $(50 + 20)/2 = 35$ |

(c) `customers LEFT JOIN orders`, grouped by `signup_year`:

| signup_year | customers | joined rows | `COUNT(o.order_id)` | `COUNT(*)` |
|---|---|---|---|---|
| 2023 | Asha (2 orders), Dana (3) | 5 | 5 | 5 |
| 2024 | Ben (2), Chen (2) | 4 | 4 | 4 |
| 2025 | Eli (0), Farah (0) | 2 (one NULL row each) | **0** | **2** |

`COUNT(*)` counts rows, and the left join creates one row per unmatched customer, filled with NULLs.
`COUNT(o.order_id)` counts non-NULL order ids, and those two rows have none. So "number of orders per
year" must be `COUNT(o.order_id)`; `COUNT(*)` would claim the 2025 cohort placed 2 orders.

```text
== exercise 1 check ==
(a) rows: 10
(b)
  status  n  n_amount      avg
    paid  8         7 133.5714
refunded  2         2  35.0000
(c)
 signup_year  n_order_id  n_rows
        2023           5       5
        2024           4       4
        2025           0       2
```

**Rubric (12 pts):**

| Criterion | Points |
|---|---|
| (a) 10 rows, with the reason (unique right key; order 110 kept) | 2 |
| (b) counts correct for both statuses | 2 |
| (b) paid average uses 7 in the denominator, not 8 | 2 |
| (c) table correct | 3 |
| (c) explains `COUNT(*)` vs `COUNT(col)` on NULL-filled rows | 3 |

**Likely mistakes:**

- (a): answering 9 because order 110 "has no customer". That is the inner join; a left join keeps
  every left row.
- (b): dividing the paid total by 8, i.e. treating the missing amount as 0 ($935/8 \approx 116.9$).
- (c): writing 0 rows for 2025, forgetting that a left join keeps unmatched left rows.

## Exercise 2 (code)

**Solution:** [`a1_3_pandas_ex2.py`](a1_3_pandas_ex2.py). The SQL:

```sql
SELECT c.city,
       COUNT(DISTINCT c.customer_id) AS n_customers,
       SUM(CASE WHEN o.status = 'paid' THEN 1 ELSE 0 END) AS n_paid,
       AVG(CASE WHEN o.status = 'refunded' THEN 1.0
                WHEN o.status IS NOT NULL THEN 0.0 END) AS refund_share
FROM customers AS c
LEFT JOIN orders AS o ON c.customer_id = o.customer_id
GROUP BY c.city
ORDER BY c.city
```

The pandas version. The key move is making the refund indicator NaN (not 0) on the rows that the
left join invented for customers with no orders, so the mean skips them exactly as SQL's `AVG`
skips the `CASE` that returns NULL:

```python
joined = customers.merge(orders, on="customer_id", how="left", validate="one_to_many")
joined["is_paid"] = (joined["status"] == "paid").astype(int)
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
```

Real output, including the double-counting demonstration:

```text
== exercise 2: city summary ==
  city  n_customers  n_paid  refund_share
   NaN            1       0           NaN
Austin            1       2        0.3333
 Leeds            2       2        0.0000
  Pune            2       3        0.2500
SQL and pandas agree: OK

== exercise 2: the double-count trap ==
paid orders: 8  rows after merge: 9
true revenue: 935.0  naive revenue after merge: 1235.0
validate='one_to_one' raised MergeError: Merge keys are not unique in right dataset; not a one-to-one merge
after aggregating payments first: 935.0  order 106 methods: card+gift
```

Order 106 (300) has two payment rows, so the naive merge counts it twice: $935 + 300 = 1235$.
`validate="one_to_one"` raises instead of silently duplicating. The fix is to aggregate the
many-side to one row per key *before* joining.

**Rubric (20 pts):**

| Criterion | Points |
|---|---|
| SQL correct: left join, `COUNT(DISTINCT ...)`, refund share that ignores no-order rows | 4 |
| pandas correct: `how="left"`, `dropna=False`, `nunique` | 4 |
| Refund share is NaN (not 0) for the city with no orders, matching SQL | 2 |
| `assert_frame_equal` used (sorting / index handled), not eyeballing | 3 |
| Double-count shown with numbers (row count and revenue before/after) | 3 |
| `validate=` demonstrated raising, and a correct fix (aggregate first) | 3 |
| Readability: named aggregation, comments where the logic is subtle | 1 |

**Likely mistakes:**

- Using `count` for `n_customers`: counts joined rows, so Dana counts 3 times. Needs `nunique`.
- Computing the refund share as `refunded / n_paid` instead of `refunded / all orders`.
- Filling the indicator with 0 on no-order rows, which drags shares down for cities with
  order-less customers (Leeds would still be 0 here, but in general it is wrong) and turns Farah's
  group into 0.0 instead of NaN.
- Forgetting `dropna=False`, so Farah's group disappears and the SQL comparison fails on row count.
- Fixing the double count with `drop_duplicates("order_id")` on the merged table: it hides the
  second payment method instead of representing it.

## Exercise 3 (think)

**Strong answer:** The missingness is not random: it hits mobile users, for three months, because of
a bug. So any choice changes the feature *differently for mobile users*, and the model can learn
"mobile" through it. Options: (1) **Drop rows with missing amounts** before averaging (what `mean()`
does by default): the average is computed on fewer orders for affected mobile users; unbiased only if
the missing orders look like the present ones, which is plausible for a logging bug but should be
checked by comparing affected and unaffected periods. (2) **Fill with 0**: wrong; it says those
orders were free, so mobile users' averages drop and the model may learn "mobile → low spender → churn".
(3) **Fill with a global median**: pulls affected users toward the population, erasing real
differences, and if the median is computed on all data including the test period it leaks
(B1.4). (4) **Add an indicator** such as "fraction of this customer's orders with a missing amount",
alongside option 1: the model can learn whether the gap matters, and you can see its importance.
(5) **Recover the data**: the payments system may hold the true amounts; a join to the source of
truth beats any imputation. I would try (5) first; failing that, use (1) plus the indicator from
(4), fit any imputation statistics on training data only, and check model performance separately
for mobile and web users so a learned shortcut shows up.

**Weak answer:** "Fill the missing values with the mean so no rows are lost." Weak because it treats
the gap as random noise, ignores that it is concentrated in one platform and period, does not mention
leakage from computing the mean on all data, and does not consider recovering the true values or
measuring the effect on the affected group.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Recognizes the missingness is systematic (mobile, one period), not random | 3 |
| Evaluates at least three options with their effect on mobile users | 3 |
| Rejects fill-with-0 with the reason | 1 |
| Mentions leakage of imputation statistics or fitting on training data only | 1 |
| Recommends a concrete choice (recover / drop + indicator) and a check by segment | 2 |

## Scoring

Total 42 pts. Pass ≥ 70% (30 pts). Topic ids to revisit for each lost criterion: exercise 1 →
A1.3 "Missing values" and "How many rows does a join produce?"; exercise 2 → A1.3 "With pandas";
exercise 3 → A1.3 "Missing values", and later B1.2 (data cleaning) and B1.4 (leakage).
