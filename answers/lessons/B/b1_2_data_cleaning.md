# Answer key: B1.2 Data cleaning

Lesson: [`lessons/B/b1_2_data_cleaning.md`](../../../lessons/B/b1_2_data_cleaning.md)

## Exercise 1 (by hand)

**Solution (a).** Strip spaces and `£`, map `""`, `"N/A"` and `-999` to missing:
$7, 12, \text{NaN}, 9, \text{NaN}, 8, 250, 10, \text{NaN}, 11$. Seven observed values, three missing: $\pi = 3/10 = 0.3$.

**(b)** Mean $= (7 + 12 + 9 + 8 + 250 + 10 + 11)/7 = 307/7 = 43.86$. Sorted: $7, 8, 9, 10, 11, 12, 250$; median $= 10$ (the 4th of 7).
Mean-imputed: $7, 12, 43.86, 9, 43.86, 8, 250, 10, 43.86, 11$. Median-imputed: $7, 12, 10, 9, 10, 8, 250, 10, 10, 11$. The mean fill
is four times every typical value, because of the 250.

**(c)** Deviations from 43.86: $-36.86, -31.86, -34.86, -35.86, 206.14, -33.86, -32.86$. Their squares sum to about 49,595, so
$\hat{\sigma}^2_{\text{obs}} = 49{,}594/7 = 7{,}085$ and $\hat{\sigma}_{\text{obs}} = 84.17$. After mean imputation, $\hat{\sigma} = \sqrt{1 - 0.3} \times 84.17 = 70.42$: the three filled
values add nothing to the sum of squares but count as rows.

**(d)** $Q_1$ at position $(7 - 1) \times 0.25 = 1.5$: halfway between 8 and 9, so 8.5. $Q_3$ at position 4.5: halfway between 11 and 12,
so 11.5. IQR $= 3$; fences $[8.5 - 4.5,\ 11.5 + 4.5] = [4, 16]$: Tukey flags 250. The largest possible z-score with 7 values
is $\sqrt{6} = 2.449$, and 250's z-score is 2.449 (it sits at the ceiling), so "$\lvert z\rvert > 3$" flags nothing.

**(e)** Before deciding: is it in the same unit (seconds instead of minutes would make it 4.2 minutes)? Is it
a call that really stayed open for four hours (a stuck line, a system that never closed the record)?
How many such values are there in the full data, and are they from one system or agent? Is the record
otherwise valid (start and end timestamps)? Then: fix a unit error, blank a system error with a log line,
or keep a real extreme (and consider a robust metric or log scale).

```text
== exercise 1 check ==
parsed [7.0, 12.0, nan, 9.0, nan, 8.0, 250.0, 10.0, nan, 11.0]   observed [7.0, 12.0, 9.0, 8.0, 250.0, 10.0, 11.0]   n_obs 7   missing share 0.30
mean 307 / 7 = 43.8571   median 10.0000
mean-imputed   [7.0, 12.0, 43.86, 9.0, 43.86, 8.0, 250.0, 10.0, 43.86, 11.0]
median-imputed [7.0, 12.0, 10.0, 9.0, 10.0, 8.0, 250.0, 10.0, 10.0, 11.0]
sd observed (divide by n) 84.1723   predicted after mean imputation sqrt(1 - 0.3) * sd = 70.4236   actual 70.4236
Q1 8.5  Q3 11.5  IQR 3.0  fences [4.0, 16.0]  Tukey flags [250.0]
z of 250 = 2.4491   ceiling sqrt(n_obs - 1) = sqrt(6) = 2.4495   |z| > 3 flags []
```

**Rubric (14 pts):**

| Criterion | Points |
|---|---|
| (a) Correct parsing, including `£12` and the sentinel; $\pi = 0.3$ | 2 |
| (b) Mean 43.86 and median 10; both imputed columns | 3 |
| (c) $\hat{\sigma}_{\text{obs}} = 84.17$ (divide by $n$) and the prediction $\sqrt{0.7} \times 84.17 = 70.42$ from the formula | 3 |
| (d) Quartiles with the linear rule, fences [4, 16], 250 flagged; ceiling $\sqrt{6}$ and why z flags nothing | 4 |
| (e) At least two concrete checks (unit, source, record validity) before deciding | 2 |

**Likely mistakes:**

- Treating `-999` as a value (the mean becomes $(307 - 999)/8 = -86.5$) or `£12` as unparseable.
- Dividing by $n_{\text{obs}} - 1$ in (c) and then wondering why the prediction is off by a few percent; the formula
  is exact for the divide-by-$n$ variance.
- Computing quartiles with a different rule (for example the median of each half: $Q_1 = 8$, $Q_3 = 12$). Accept
  it if the rule is stated; the conclusion is the same.
- Concluding from "no $\lvert z\rvert > 3$" that there is no outlier.

## Exercise 2 (code)

**Solution:** [`b1_2_data_cleaning_ex2.py`](b1_2_data_cleaning_ex2.py). `audit_table` takes a `Rules` object
(expected dtypes, ranges, allowed category sets, a key, a missing-rate limit, and PII columns knowingly kept)
and returns one line per problem without modifying the frame.

```text
== exercise 2: audit raw vs cleaned ==
raw table: 14 problems
  - type: monthly_spend is str (kind 'O'), expected kind 'f'
  - type: sessions_30d is int64 (kind 'i'), expected kind 'f'
  - type: birth_date is str (kind 'O'), expected kind 'M'
  - type: signup_date is str (kind 'O'), expected kind 'M'
  - sentinel: birth_date has 26 values in ['', '-999', '-999.0', '1900-01-01', 'N/A']
  - sentinel: monthly_spend has 214 values in ['', '-999', '-999.0', '1900-01-01', 'N/A']
  - sentinel: sessions_30d has 122 values in ['', '-999', '-999.0', '1900-01-01', 'N/A']
  - range: monthly_spend has 72 values outside [0, 10000]
  - range: sessions_30d has 132 values outside [0, 2000]
  - category: country has 7 unexpected values, e.g. [' United Kingdom', 'France', 'Ireland', 'UK']
  - category: plan has 6 unexpected values, e.g. [' Basic ', ' Pro ', ' Team ', 'BASIC']
  - key: 50 duplicate email values after strip + lowercase
  - PII: email looks like email addresses
  - PII: phone looks like phone numbers
cleaned table: 2 problems
  - PII: email looks like email addresses
  - PII: phone looks like phone numbers
cleaned table, with email and phone accepted until the PII pass: []
```

The raw table fails on every kind of rule; the cleaned table passes everything except the PII check,
which is correct: cleaning does not remove personal data, the PII pass does. Accepting `email` and `phone`
explicitly (rather than turning the check off) documents that they are known and scheduled for removal.
Numbers depend on the learner's thresholds; what is graded is that each check exists, reports instead of
fixing, and catches the planted problems.

**Rubric (16 pts):**

| Criterion | Points |
|---|---|
| Type, sentinel, range, category, duplicate-key and missing-rate checks, each producing a readable line | 7 |
| PII heuristic for emails and phones that does not fire on dates or ids | 2 |
| The function never modifies the frame (no `inplace`, no assignment to `df`) | 2 |
| Raw table flagged on the planted problems; cleaned table passes (or remaining items explained) | 3 |
| Readability: rules as data, not hard-coded inside the function; type hints, docstrings | 2 |

**Likely mistakes:**

- A phone regex that matches dates like `1958-12-25` (digits and dashes): in the reference it is restricted to
  numbers starting with `+` or `0`.
- Checking duplicates on the raw key without normalizing case and whitespace: misses the 20 near-duplicates.
- Using `pd.to_numeric(errors="coerce")` and then reporting the coerced `NaN`s as "missing", hiding the format
  problem. Coercion is fine for *counting* range violations, not for deciding what a value means.
- Returning booleans instead of messages: an audit nobody can read is not used.

## Exercise 3 (think)

**Strong answer:** For a churn model, none of `name`, `email` or `phone` is a feature; drop them from the training
table. If records must be joined to other tables or traced back for a retention call, replace the id (or
email) with a keyed pseudonym (HMAC-SHA256 with a key in a secret store, not in the repository), and keep the
mapping only in the production system that needs it. Coarsen `birth_date` to an age band (or drop it if age
does not help) and `postcode` to a district or region; check how unique people are on the remaining
quasi-identifiers and coarsen further if small groups remain. "In case it's useful later" is exactly what
data minimization rules out (and GDPR's minimization and purpose-limitation principles require a purpose
now; VERIFY as of 2026-10, not legal advice). On hashing: a plain SHA-256 of an email is not anonymous. It is
deterministic, so anyone with a list of emails (a marketing list, a breach dump) can hash them and match; and
under GDPR, pseudonymized data is still personal data. Remaining risk after the changes: re-identification
through combinations of quasi-identifiers and behaviour (a unique spend pattern in a small district), access to
the key, copies of the raw export, and model outputs that reveal individual information; mitigate with access
control, retention limits, and measuring $k$-anonymity on the released table.

**Weak answer:** "Hash the email with SHA-256 so it's anonymous, keep the other columns because more data
gives a better model." Weak because unkeyed hashes of guessable identifiers are reversible, pseudonymized
data is still personal data, and keeping identifying columns without a use creates risk with no benefit.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Drops direct identifiers, with the reason (not features for this task) | 2 |
| Pseudonymizes join keys with a *keyed* hash, key outside the data and code | 2 |
| Coarsens quasi-identifiers and mentions measuring uniqueness / $k$-anonymity | 2 |
| Explains why plain hashing is not anonymization (dictionary attack; still personal data) | 2 |
| Names a remaining risk and a mitigation | 2 |

## Scoring

Total 40 pts. Pass ≥ 70% (28 pts). Topic ids to revisit for each lost criterion: exercise 1 → B1.2 "The math"
(variance shrinkage, z ceiling) and A6.1 (quartiles); exercise 2 → B1.2 "Intuition" (the five problems) and
"In practice"; exercise 3 → B1.2 "Personal data", then I4.1 and J2.5.
