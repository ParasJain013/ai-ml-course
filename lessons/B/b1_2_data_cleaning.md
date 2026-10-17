# B1.2 Data cleaning

**Core** · `[RAI]` · Part B, week 1 (course week 07) · ~2 h · Lab:
[`labs/B/b1_2_data_cleaning.py`](../../labs/B/b1_2_data_cleaning.py)

## Why it matters

Real tables arrive broken: prices stored as text with currency symbols, `-999` meaning "unknown",
the same country spelled five ways, customers exported twice, ages of 126. A model trained on that
learns from the breakage. Worse, most of the damage is silent: a `-999` averaged into a column
does not raise an error, it just moves every number downstream. **Data cleaning** is the step that
makes the table mean what you think it means, and it is where most of a real project's hours go.
The same pass is where you decide which personal data the model actually needs (`[RAI]`): a
training table full of names, emails and exact birth dates is a liability that no accuracy gain
justifies.

## Prerequisite check

- Can you select, filter and group a DataFrame (`df.loc`, `.str` methods, `groupby`)? If not, revisit
  [A1.3](../A/a1_3_pandas.md).
- Can you compute quartiles, the IQR and Tukey's fences by hand? If not, revisit
  [A6.1](../A/a6_1_descriptive_statistics_eda.md).
- Can you say why a z-score uses the mean and standard deviation, and why those are not robust? If not,
  revisit [A6.1](../A/a6_1_descriptive_statistics_eda.md).
- Can you explain why anything learned from data (even a fill value) must come from the training rows
  only? If not, revisit [B1.1](b1_1_ml_workflow.md).

## Intuition

### Cleaning is a pipeline of logged decisions

Treat cleaning like a database migration: a script, not a series of clicks in a notebook, that turns
the raw export into a table with known types and known meaning, and that you can rerun on next
month's export. Every step is a decision ("negative spend is impossible, set it to missing"), and every
decision gets a line in a **cleaning log** with the number of rows it touched. The log is how a reviewer
(or you, in three months) can tell a careful choice from an accident. "Dropped 211 rows" is not a log
line; "monthly_spend: 211 unknown ('', 'N/A', -999) set to missing, kept the rows" is.

> **Bridge:** a cleaning script is a migration plus a set of database constraints: it converts types
> (`ALTER COLUMN ... TYPE numeric USING ...`), normalizes values, and enforces rules such as `CHECK
> (monthly_spend >= 0)` and `UNIQUE (email)`. **Where it breaks:** a database rejects a row that breaks a
> constraint, and the write fails loudly. A cleaning script must decide what to do instead (drop the
> row, blank the value, fix it), and each choice changes what the model learns, so it has to be
> justified, not just enforced.

### The five usual problems

1. **Wrong types.** Numbers stored as text (`"£1,040.00"`, `" 15 "`), dates as strings, booleans as
   `"Y"`/`"N"`. Parse them explicitly, and make the parser fail on anything unexpected rather than
   guess: a value that does not parse is a finding, not something to coerce to `NaN` silently.
2. **Missing values**, including disguised ones. A **sentinel value** such as `-999`, `0`, `""`,
   `"N/A"` or the date `1900-01-01` is a placeholder the source system writes for "unknown". It looks
   like data, so it is worse than an empty cell. Convert every sentinel to a real missing value first.
3. **Impossible values.** Values that break a rule of the domain: negative spend on a subscription, a
   birth date in the future, 99,999 sessions in 30 days (a counter that overflowed). These are errors;
   treat them as missing, with a log line.
4. **Inconsistent categories.** `"UK"`, `"uk"`, `" United Kingdom"`, `"GB"` are one country. Normalize
   (strip, lowercase) and map to a fixed set of codes; anything that does not map is reported, not
   guessed.
5. **Duplicates.** Exact duplicates (the export ran twice) and near-duplicates (the same person with
   their email typed in capitals). Decide what identifies a real entity (here, the normalized email) and
   keep one row per entity.

### Missing values: why they are missing matters

There are three situations, and they call for different handling:

- **Missing completely at random:** a sensor dropped a reading for no reason related to anything. Any
  sensible fill value works; you only lose some precision.
- **Missing for a reason you can see:** spend is missing more often for one plan, because that plan is
  billed by a different system. Fill using the related column (median per plan).
- **Missing for a reason tied to the value or the target:** session tracking stops when a user uninstalls
  the app, so "sessions unknown" is most common among people about to leave. Here the *fact* that the value
  is missing predicts churn. In the lab, customers with missing sessions churned at 57%, against 20% for the
  rest. Filling the gap with the median erases that signal; a **missing indicator** column (1 if the value
  was missing) keeps it.

The options are: drop the rows (only if few and random, otherwise you bias the sample), drop the column
(if mostly empty and not needed), **impute** (fill with a statistic such as the median, learned on the
training rows), and add a missing indicator. Imputing with the mean is the common default and the most
misleading one: it stacks every missing row on one value, shrinks the spread of the column, and is pulled
by outliers. The median is safer for skewed columns.

### Outliers: flag, look, then decide

An **outlier** is a value far from the bulk. It is either an error (a typo, a unit mix-up) or a real,
rare case (a big customer). Deleting outliers by a rule removes exactly the cases a model most needs to
see if they are real. So: flag them with a rule, look at who they are, and decide per cause. In the lab,
76% of the spend values flagged by Tukey's rule belong to the "team" plan, which is billed at about four
times the price: real customers, not errors. The right response is to keep them and perhaps model
log(spend), where they are not extreme.

Two flag rules, both from A6.1. **Tukey's fences** flag values beyond 1.5 IQR outside the quartiles; they
use robust statistics, so a few extreme values do not hide each other. **z-scores** flag values more than 3
standard deviations from the mean; the extremes themselves inflate the mean and standard deviation, so
they can hide (mask) each other, and in a small sample a z-score cannot even reach 3 (the math shows why).

### `[RAI]` Personal data: minimize before you model

Personal data in a training table creates risk (breaches, misuse, legal exposure) and rarely improves the
model. **Data minimization** means keeping only what the task needs. Sort the columns into three kinds:

- **Direct identifiers:** name, email, phone, national id, exact address. They identify a person on their
  own. A churn model does not need them: drop them.
- **Join keys:** a customer id you need to link tables or trace a prediction back. Replace it with a
  **pseudonym**: a keyed hash (HMAC) of the id, with the key held in a secret store. The same id always
  maps to the same token, so joins still work, but the token cannot be reversed without the key. A plain,
  unkeyed hash is not enough: ids like `C00001` come from a small, guessable space, so anyone can hash all of
  them and match.
- **Quasi-identifiers:** columns that are harmless alone but identifying together, such as birth date,
  postcode and sex. Sweeney (2000) estimated that 87% of the US population is unique on 5-digit ZIP code,
  sex and full date of birth. **Coarsen** them to what the model needs: an age band instead of a birth date,
  a postal district instead of a full postcode.

**PII** (personally identifiable information) covers all three kinds. Pseudonymized data is still personal
data under laws such as the EU GDPR, because whoever holds the key can re-identify it (VERIFY as of 2026-10;
not legal advice). The goal of this pass is to reduce risk, not to declare the table "anonymous".

## The math

**Symbols**

| Symbol | Meaning | Shape |
|---|---|---|
| $\mathbf{x}$ | one numeric column, with some entries missing | `(n,)` |
| $n$, $n_{\text{obs}}$ | number of rows, and of rows where $\mathbf{x}$ is observed | ints |
| $\pi$ | missing share, $1 - n_{\text{obs}}/n$ | scalar |
| $m_i$ | missing indicator: 1 if $x_i$ is missing, else 0 | scalar |
| $\bar{x}_{\text{obs}}$, $\operatorname{med}_{\text{obs}}$ | mean and median of the observed values (training rows) | scalars |
| $\hat{\sigma}^2_{\text{obs}}$ | variance of the observed values, dividing by $n_{\text{obs}}$ | scalar |
| $Q_1, Q_3$ | first and third quartiles; $\mathrm{IQR} = Q_3 - Q_1$ | scalars |
| $z_i$ | z-score $(x_i - \bar{x})/\hat{\sigma}$, with $\hat{\sigma}$ dividing by $n$ | scalar |
| $k_{\text{anon}}$ | size of the smallest group of rows sharing the same quasi-identifier values | int |

**Imputation is a learned transformation.** Imputing with the mean means: on the training rows, compute
$\bar{x}_{\text{obs}}$ from the observed values; on any row (training, validation, test or production), replace a
missing $x_i$ by that number. Written with the indicator,

$$\tilde{x}_i = (1 - m_i)\,x_i + m_i\,\bar{x}_{\text{obs}}.$$

The fill value is a parameter learned from data, exactly like a model weight, so it is learned on the
training rows only (B1.4 shows what goes wrong otherwise).

**Mean imputation shrinks the variance.** Every filled value sits exactly at the mean, so it adds $0$ to the
sum of squared deviations but adds 1 to the count. With the divide-by-$n$ variance,

$$\operatorname{Var}[\tilde{\mathbf{x}}] = \frac{1}{n}\sum_{i\,\text{observed}}(x_i - \bar{x}_{\text{obs}})^2 = \frac{n_{\text{obs}}}{n}\,\hat{\sigma}^2_{\text{obs}} = (1 - \pi)\,\hat{\sigma}^2_{\text{obs}}.$$

With 12% missing, the standard deviation shrinks by $\sqrt{0.88} = 0.94$: the column looks more certain than
it is, and its correlation with every other column is weakened. The median fill has the same "pile on one
value" problem, at a less extreme spot for skewed data.

**Tukey's fences** (A6.1): flag $x_i$ if $x_i < Q_1 - 1.5\,\mathrm{IQR}$ or $x_i > Q_3 + 1.5\,\mathrm{IQR}$.

**A z-score cannot exceed $\sqrt{n - 1}$.** Let $d_i = x_i - \bar{x}$ be the deviations; they sum to zero, so the
others sum to $-d_j$. The smallest the other $n - 1$ squared deviations can add up to, given that their sum is
$-d_j$, is when they are all equal, $-d_j/(n - 1)$ each (any spread around their common average only adds to the sum of
squares: the same fact as "a variance is never negative"), giving $(n - 1)\cdot d_j^2/(n - 1)^2 = d_j^2/(n - 1)$. So

$$n\,\hat{\sigma}^2 = \sum_i d_i^2 \;\ge\; d_j^2 + \frac{d_j^2}{n - 1} = d_j^2\,\frac{n}{n - 1} \quad\Longrightarrow\quad z_j^2 = \frac{d_j^2}{\hat{\sigma}^2} \le n - 1.$$

For $n = 6$ no z-score can exceed $\sqrt{5} = 2.24$, however wild the value: a "$\lvert z\rvert > 3$" rule cannot flag anything
in fewer than 11 values. The single extreme value inflates $\hat{\sigma}$ enough to hide itself. Tukey's fences use
quartiles, which the extreme value barely moves.

**Keyed hashing.** A pseudonym is $\operatorname{token}(\text{id}) = \operatorname{HMAC\text{-}SHA256}(\text{key}, \text{id})$, truncated to 16 hex characters (64 bits).
It is deterministic (joins work), and without the key it cannot be recomputed. With a plain
$\operatorname{SHA256}(\text{id})$ an attacker hashes all 100,000 possible ids `C00000`–`C99999` in well under a second and
reverses every token.

**Uniqueness and $k$-anonymity.** Group the rows by their quasi-identifier values. The share of rows in a group
of size 1 is the share of people singled out by those columns. A table is **$k$-anonymous** (Sweeney, 2002) if
every group has at least $k$ rows, i.e. $k_{\text{anon}} \ge k$. Coarsening merges groups and raises $k_{\text{anon}}$.

## Worked example by hand

Eight spend values arrive as text: `"12.5"`, `"£1,040.00"`, `"N/A"`, `"-999"`, `" 15 "`, `"14"`, `"16"`, `"13"`.

1. **Parse.** Strip spaces, remove `£` and the thousands comma, map `"N/A"` and the sentinel `-999` to missing:
   $12.5, 1040, \text{NaN}, \text{NaN}, 15, 14, 16, 13$. Six observed values, two missing.
2. **Fill values.** Mean $= (12.5 + 1040 + 15 + 14 + 16 + 13)/6 = 1110.5/6 = 185.08$. Sorted: $12.5, 13, 14, 15, 16, 1040$, so the
   median is $(14 + 15)/2 = 14.5$. Mean imputation would claim the two unknown customers spend £185 a month,
   more than twelve times what five of the six known customers spend, because one value dragged the mean.
   The indicator column is $0, 0, 1, 1, 0, 0, 0, 0$.
3. **Tukey's fences.** With NumPy's linear rule (A6.1), $Q_1$ is at position $(6 - 1) \times 0.25 = 1.25$ of the sorted
   values (0-based): $13 + 0.25 \times (14 - 13) = 13.25$. $Q_3$ is at position $3.75$: $15 + 0.75 \times (16 - 15) = 15.75$. IQR $= 2.5$, fences
   $[13.25 - 3.75,\ 15.75 + 3.75] = [9.5, 19.5]$: 1040 is flagged.
4. **z-scores.** The z-score of 1040 is $2.24$, which is exactly the ceiling $\sqrt{6 - 1}$: with six values the
   $\lvert z\rvert > 3$ rule never fires.

What to do with 1040? Look at the customer. If it is a team account billed at £1,040, keep it. If the source
had a known bug writing pence instead of pounds, divide by 100 and log it. Either way, the decision is
written down.

```text
== worked example by hand ==
parsed: [12.5, 1040.0, nan, nan, 15.0, 14.0, 16.0, 13.0]   observed n = 6   missing = 2
mean = 1110.5 / 6 = 185.0833   median = 14.5000
mean-imputed:   [12.5, 1040.0, 185.0833, 185.0833, 15.0, 14.0, 16.0, 13.0]
median-imputed: [12.5, 1040.0, 14.5, 14.5, 15.0, 14.0, 16.0, 13.0]   indicator: [0, 0, 1, 1, 0, 0, 0, 0]
sorted observed [12.5, 13.0, 14.0, 15.0, 16.0, 1040.0]   Q1 = 13.2500   Q3 = 15.7500   IQR = 2.5000   fences [9.5000, 19.5000]
z-scores (ddof=0): [-0.4514, 2.2361, -0.4449, -0.4475, -0.4422, -0.4501]   largest possible |z| for n = 6: sqrt(n - 1) = 2.2361
```

## Key plot

![Left: histogram of monthly spend for training customers, a right-skewed hump peaking near 20 GBP; the mean-imputed version adds a spike of about 230 customers at 48.1, well right of the median 33.2. Right: bar chart of the share of customers unique on their quasi-identifiers: 100% for birth date, full postcode and sex; 64% for birth year, district and sex; 3% for age band, district and sex](img/b1_2_imputation_and_reidentification.png)

Left: mean imputation puts 12% of the training customers on one value, which is not even typical (it sits
right of the median because the column is skewed). Right: every customer is unique on (birth date, full
postcode, sex), so those three "harmless" columns identify everyone; coarsening to (age band, district,
sex) brings it to 3%, which is still not zero.

## From scratch in NumPy

**The data.** The lab generates 2,000 clean customers and damages them the way real exports are damaged:
spend as text with `£`, spaces, `"N/A"`, `""`, `-999` and sign typos; a session count with `-999` (more often for
customers who later left) and an overflowed counter; placeholder and future birth dates; ten spellings of
three countries; plan names in three casings; 30 exact and 20 near-duplicate rows.

**Cleaning with a log.** One function, one decision per step, one log line per decision:

```python
def clean_customers(df):
    log, d, n0 = [], df.copy(), len(df)
    d = d.drop_duplicates()
    log.append(f"dropped {n0 - len(d)} exact duplicate rows")
    d["email"] = d["email"].str.strip().str.lower()               # the identity of a customer
    d = d.drop_duplicates(subset=["email"], keep="first")
    spend = d["monthly_spend"].map(parse_money)                    # text -> float, sentinels -> NaN
    spend[spend < 0] = np.nan                                      # impossible for a subscription
    ...
    d["country"] = d["country"].str.strip().str.lower().map(COUNTRY_MAP)   # unmapped -> NaN, reported
    return d, log
```

```text
== from scratch: cleaning with a log ==
 - dropped 30 exact duplicate rows
 - dropped 20 near-duplicates (same email after strip + lowercase)
 - monthly_spend: text -> float; 211 unknown ('', 'N/A', -999)
 - monthly_spend: 19 negative values -> NaN (impossible for a subscription)
 - sessions_30d: 119 sentinel -999 -> NaN; 9 > 2000 -> NaN
 - birth_date: text -> date; 34 placeholder or future dates -> NaT
 - country: 10 spellings -> 3 codes; unmapped: 0
 - plan: 9 spellings -> 3; missing kept as NaN: 58
 - rows: 2050 -> 2000
missing after cleaning (share): {'birth_date': 0.017, 'plan': 0.029, 'monthly_spend': 0.115, 'sessions_30d': 0.064}
recovered spend vs truth (same customers): max |difference| = 0.0000
```

The last line compares the cleaned spend with the generator's true values for the same customers: every
value that survived parsing is exactly right. `parse_money` raises an error on any string it does not
recognize; if a new format appears next month, the script stops instead of silently producing `NaN`.

**Imputation.** Fit the fill values on the training rows (an 80/20 split here), apply them to any rows.
`transform_impute` broadcasts a `(1, d)` row of fill values over the `(n, d)` array and can append the `d`
indicator columns:

```python
def transform_impute(X, fill, add_indicator=False):
    miss = np.isnan(X)                               # (n, d)
    out = np.where(miss, fill[None, :], X)           # (n, d)
    return np.hstack([out, miss.astype(np.float64)]) if add_indicator else out   # (n, 2d)
```

```text
== from scratch: imputation ==
train fill values  mean: [48.125, 13.169]   median: [33.25, 11.0]
test shape (400, 2) -> mean-imputed (400, 2)   median + indicator (400, 4)
monthly_spend on train: observed sd 50.0206   after mean imputation sd 46.9048   share of rows equal to the fill value 0.1206
churn rate when sessions_30d is missing 0.5670 (n = 97) vs observed 0.1956 (n = 1503)
-> here missingness is itself a signal (tracking stops when users leave); the indicator keeps it
```

The mean fill for spend is £48.1 while the median is £33.25: the mean is pulled up by the team plan's
large bills. Mean imputation cut the standard deviation from 50.0 to 46.9, as the formula predicts:
$\sqrt{1 - 0.1206} \times 50.02 = 46.91$. And the sessions column shows informative missingness: 57% churn when it is
missing against 20% when it is present, a signal the indicator keeps and a plain fill destroys.

**Outlier flags.**

```text
== from scratch: outlier flags ==
monthly_spend (n = 1770): IQR flags 160   z > 3 flags 40
log(monthly_spend):          IQR flags 41   z > 3 flags 9
of the IQR-flagged raw spends, share on the 'team' plan: 0.7625   (share of all customers on team: 0.1028)
-> the 'outliers' are mostly real team customers: keep them, maybe model log(spend)
```

Tukey flags 160 spend values (9%); the z rule only 40, because those 160 large values inflate the standard
deviation and partly hide each other. Three quarters of the flagged values are team customers, who make up
only 10% of the table: a different population, not an error. On the log scale the distribution is much closer
to symmetric, and only 41 values are flagged.

**The PII pass.** Drop the direct identifiers, replace the id with a keyed hash, coarsen the birth date to
an age band and the postcode to its district (the part before the space):

```python
safe = df.drop(columns=["name", "email", "phone"]).copy()
safe["customer_key"] = pseudonymize(safe.pop("customer_id"), HASH_KEY)   # HMAC-SHA256, key from env
safe["age_band"] = age_band(safe.pop("birth_date"), AS_OF)               # '25-34', ...
safe["district"] = safe.pop("postcode").str.split(" ").str[0]            # 'SW3 4AB' -> 'SW3'
```

```text
== from scratch: PII pass ==
dropped ['name', 'email', 'phone']; customer_id -> keyed hash, e.g. C00001 -> dfff21d1c200e2bf
columns kept: ['sex', 'country', 'plan', 'signup_date', 'monthly_spend', 'sessions_30d', 'churned', 'customer_key', 'age_band', 'district']
unique on (birth date, full postcode, sex):  1.0000 of rows   smallest group k = 1
unique on (birth year, district, sex):       0.6405 of rows   smallest group k = 1
unique on (age band, district, sex):         0.0295 of rows   smallest group k = 1
why a key: plain sha256('C00001') = d7ca7d5dc67b6e13 can be rebuilt by anyone who guesses the id format
```

Every one of the 2,000 customers is unique on (birth date, full postcode, sex). Birth year and district still
single out 64%. Age band and district bring it down to 3%, but the smallest group still has one person in it
($k_{\text{anon}} = 1$): that customer is identifiable by anyone who knows their age band, district and sex. Reaching $k_{\text{anon}} \ge 5$
would take coarser bands, fewer districts, or suppressing the rare combinations. The model probably needs
none of these columns at full precision; check before keeping them.

## With scikit-learn and SciPy

`SimpleImputer` is the fitted-fill-value transformer: `fit` learns `statistics_` on the training rows, and
`transform` fills any rows with them; `add_indicator=True` appends the missing-indicator columns (only for
columns that had missing values during `fit`). SciPy has `stats.zscore` (divide-by-$n$ standard deviation by
default) and `stats.iqr`:

```text
== with scikit-learn and scipy ==
SimpleImputer(strategy='mean') statistics_ [48.125, 13.169]   matches mine: OK
SimpleImputer(strategy='median') statistics_ [33.25, 11.0]   matches mine: OK
SimpleImputer(add_indicator=True) -> (400, 4)   matches mine: OK
scipy.stats.zscore flags match z_flags: OK
scipy.stats.iqr = 31.4825   matches np.quantile: OK
```

In B1.3 the imputer goes inside a `ColumnTransformer`, so that one `fit` call learns every fill value on the
training rows and the same object applies them in production.

## In practice

- **When to use it:** on every new dataset, and again on every refresh of it. Write the cleaning as a
  script with a log, keep the raw file untouched, and make the script rerunnable (same input, same
  output).
- **Failure modes:**
  - *Sentinels left in.* A `-999` averaged into a column. You notice a minimum or a spike at a round
    negative number in `describe()` or a histogram. Search every numeric column for suspicious repeated
    values before anything else.
  - *Coercing instead of failing.* `pd.to_numeric(errors="coerce")` turns every unexpected format into
    `NaN` without a word. You notice when the missing rate jumps after a source change. Count what the parser
    rejected and stop when it is new.
  - *Dropping every row with a missing value.* Often removes a non-random slice (people who did not answer,
    users who left). You notice in the log's row counts; compare the dropped rows with the kept ones.
  - *Fill values learned on all rows.* A mild leak (B1.4); in production you will not have the future rows
    to average. Fit on training rows, inside a pipeline.
  - *Deleting outliers by rule.* Removes your biggest customers. Flag, inspect, then decide by cause.
  - *Near-duplicates.* Same entity, different spelling. You notice when a supposed key is not unique after
    normalizing case and whitespace.
  - `[RAI]` *Personal data kept "in case".* Names and exact birth dates in a training table that is copied to
    laptops and notebooks. You notice when you list the columns and ask "which feature uses this?". Drop,
    pseudonymize or coarsen before the data leaves the source system.
- **Defaults that work:** median imputation plus a missing indicator for numeric columns; an explicit
  "missing" category for categorical columns; Tukey flags with $c = 1.5$ for review, never automatic deletion;
  HMAC-SHA256 pseudonyms with the key outside the code; age bands of 10 years and postal districts unless
  the task needs more. `SimpleImputer` defaults to `strategy="mean"` (VERIFY as of 2026-10, scikit-learn
  1.9.1): change it to `"median"` deliberately.

## Exercises

Save your work to `my-work/B/b1_2_data_cleaning.md` (or `.py`) and ask for a review.

1. **By hand:** Ten values of "minutes on hold" arrive as text: `"7"`, `"£12"`, `"-999"`, `"9"`, `""`, `"8"`, `"250"`,
   `"10"`, `"N/A"`, `"11"` (yes, someone typed a currency symbol). (a) Parse them, listing the observed values and the
   missing share $\pi$. (b) Give the mean, the median, and the mean- and median-imputed columns. (c) Compute the
   standard deviation (divide by $n$) of the observed values and use the formula to predict the standard
   deviation after mean imputation, without imputing. (d) Compute Tukey's fences and the largest possible
   z-score for this many observed values. Which values does each rule flag? (e) What would you check before
   deciding what to do with 250?
2. **Code:** Write `audit_table(df, rules)` that returns a list of human-readable problems, one per rule
   violation, without changing the data: columns whose type differs from the expected one, sentinel values
   (`-999`, `"N/A"`, `""`, `1900-01-01`), values outside an allowed range or set, duplicate keys after
   normalizing case and whitespace, missing rates above a threshold, and string columns that look like
   emails or phone numbers (a simple regex). Run it on the raw and on the cleaned customer table from the
   lab (copy the generator), and show that the cleaned table passes every rule except the ones you decided
   to accept.
3. **Think:** `[RAI]` A teammate wants to keep `name`, `email`, full `birth_date` and full `postcode` in the
   churn training table "in case they're useful later", and says hashing the email with SHA-256 makes the
   table anonymous. Respond: what would you keep, drop or transform for a churn model and why, what is wrong
   with the hashing claim, and what risk remains after your changes.

## Takeaways

1. Clean with a rerunnable script that logs every decision and the rows it touched; parsers fail loudly
   on unknown formats instead of coercing to missing.
2. Convert sentinels (`-999`, `""`, `1900-01-01`) and impossible values to missing first; then decide per column:
   drop, impute (median by default, fitted on training rows) and add a missing indicator when missingness
   may carry signal.
3. Mean imputation piles rows on one value and shrinks the variance by a factor $1 - \pi$; it is a default, not
   a neutral choice.
4. Flag outliers with robust rules (Tukey), then inspect them: real extremes stay, errors are fixed or
   blanked. A z-score cannot exceed $\sqrt{n - 1}$, so z rules miss outliers in small samples.
5. `[RAI]` Minimize personal data before modelling: drop direct identifiers, pseudonymize join keys with a
   keyed hash, coarsen quasi-identifiers, and measure how unique people still are.

**What builds on this:** [B1.3](b1_3_feature_engineering.md) (the imputer inside a pipeline);
[B1.4](b1_4_data_leakage.md) (fill values learned on all rows); [B5.6](b5_6_anomaly_detection.md)
(outliers as the target); [I4.1](../I/i4_1_guardrails.md) (PII redaction for LLM inputs);
[J2.5](../J/j2_5_documentation_governance.md) (retention and governance); P2 (cleaning log and the
columns you exclude for PII reasons, milestone M0).

**References:** L. Sweeney, "Simple Demographics Often Identify People Uniquely", Carnegie Mellon University,
Data Privacy Working Paper 3, 2000. L. Sweeney, "k-anonymity: a model for protecting privacy",
*International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems* 10(5), 2002. R. J. A. Little
and D. B. Rubin, *Statistical Analysis with Missing Data*, 3rd ed., Wiley, 2019. Regulation (EU) 2016/679
(GDPR), Article 5(1)(c) (data minimization) and Recital 26 (pseudonymized data) (VERIFY as of 2026-10).
