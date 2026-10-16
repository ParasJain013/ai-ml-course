# Answer key: A1.1 The lab workflow

Lesson: [`lessons/A/a1_1_lab_workflow.md`](../../../lessons/A/a1_1_lab_workflow.md)

## Exercise 1 (by hand)

**Solution:** $m = 8$, $a = 5$, $c = 1$, $x_0 = 2$. Each step is $x_{k+1} = (5x_k + 1) \bmod 8$.

| $k$ | $5x_{k-1} + 1$ | $x_k$ |
|---|---|---|
| 1 | $5 \cdot 2 + 1 = 11$ | $11 - 8 = 3$ |
| 2 | $5 \cdot 3 + 1 = 16$ | $16 - 2 \cdot 8 = 0$ |
| 3 | $5 \cdot 0 + 1 = 1$ | $1$ |
| 4 | $5 \cdot 1 + 1 = 6$ | $6$ |
| 5 | $5 \cdot 6 + 1 = 31$ | $31 - 3 \cdot 8 = 7$ |
| 6 | $5 \cdot 7 + 1 = 36$ | $36 - 4 \cdot 8 = 4$ |
| 7 | $5 \cdot 4 + 1 = 21$ | $21 - 2 \cdot 8 = 5$ |
| 8 | $5 \cdot 5 + 1 = 26$ | $26 - 3 \cdot 8 = 2$ |

$x_8 = 2 = x_0$ and all 8 states 0–7 appeared once, so the **period is 8** (the maximum, $m$).

Hull–Dobell check for $m = 8 = 2^3$ (only prime factor 2):

1. $c = 1$ and $m = 8$ share no factor other than 1. ✓
2. $a - 1 = 4$ is divisible by 2. ✓
3. $8$ is divisible by 4, and $a - 1 = 4$ is divisible by 4. ✓

All three hold, so the theorem predicts the full period 8, which matches.

With $c = 2$: $x_1 = 12 \bmod 8 = 4$, $x_2 = 22 \bmod 8 = 6$, $x_3 = 32 \bmod 8 = 0$,
$x_4 = 2 \bmod 8 = 2 = x_0$. **Period 4.** Condition 1 fails: $c = 2$ and $m = 8$ share the
factor 2. Intuitively, starting from an even number, $5x + 2$ is always even, so the odd states can
never be reached. Code check (from `a1_1_lab_workflow_ex2.py`):

```text
== exercise 1 check ==
c=1 states: [3, 0, 1, 6, 7, 4, 5, 2] period: 8
c=2 states: [4, 6, 0, 2, 4, 6, 0, 2] period: 4
```

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| States $x_1 \dots x_8$ correct for $c = 1$ | 3 |
| Period 8 stated, with the reason (the seed returns at step 8) | 1 |
| All three conditions checked correctly, with the right numbers | 3 |
| $c = 2$: states and period 4 correct | 2 |
| Names condition 1 as the one that fails, with a reason | 1 |

**Likely mistakes:**

- Computing $x_1$ from $x_1$ instead of from $x_0$ (an off-by-one in the table), so the whole
  sequence is shifted by one.
- Treating "mod" as division: $11 \bmod 8$ is the remainder 3, not 1.375.
- Counting the period as 9 by including both $x_0$ and $x_8$.
- Checking condition 3 against $a$ instead of $a - 1$.
- Saying "$c = 2$ fails because 2 is even": the condition is about a shared factor with $m$. With
  $m = 9$, $c = 2$ would be fine.

## Exercise 2 (code)

**Solution:** [`a1_1_lab_workflow_ex2.py`](a1_1_lab_workflow_ex2.py). The core function:

```python
def shuffle_split(
    n: int, test_frac: float, rng: np.random.Generator
) -> tuple[IntArray, IntArray]:
    """Return (train_idx, test_idx) for a random split of rows 0..n-1. -> (n - n_test,), (n_test,)."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    n_test = int(round(n * test_frac))
    perm = rng.permutation(n)  # (n,) every index exactly once, in random order
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])
```

A permutation guarantees (b) by construction: every index appears exactly once, so slicing it in two
gives disjoint sets that cover everything. Part (c) creates two spawned children from the same seed
twice, sends the extra draw to the *other* child, and shows the split child is unaffected. Real
output:

```text
== exercise 2: shuffle_split ==
(a) first 5 test idx, run 1: [4, 5, 8, 9, 10]
(a) first 5 test idx, run 2: [4, 5, 8, 9, 10]
(a) same seed, same split: OK
(b) disjoint and covers 0..n-1: OK
(c) with an extra draw first: [5, 8, 9, 10, 11] changed: True
(c) spawned stream, no extra draw:        [7, 8, 9, 14, 15]
(c) spawned stream, extra draw elsewhere: [7, 8, 9, 14, 15]
(c) spawned split unaffected: OK

== why n = 10 fooled part (c) ==
permutation(10):               [4, 6, 2, 7, 3, 5, 9, 0, 8, 1]
permutation(10) after a draw:  [4, 6, 2, 7, 3, 5, 9, 0, 8, 1]
```

The last block is a real trap found while writing this key. With $n = 10$, NumPy 2.5.4 happens to
return the *same* permutation after an extra `random()` call, so a test of part (c) at $n = 10$
"shows" that an extra draw changes nothing. With $n = 100$ the split does change. The cause is inside
NumPy's bounded-integer sampling and was not traced further; the point is that one tiny example is
not evidence of a property. Test with sizes that make a coincidence unlikely, and assert on the whole
array, not on a printed prefix.

**Rubric (15 pts):**

| Criterion | Points |
|---|---|
| `shuffle_split` correct: right sizes, takes a `Generator`, no global state | 4 |
| Type hints and a docstring with shapes; validates `test_frac` | 2 |
| (a) demonstrated with two generators made from the same seed, asserted | 2 |
| (b) asserted (no overlap, covers `0..n-1`), not just printed | 3 |
| (c) shows the extra draw changes the split *and* the spawned stream does not | 3 |
| Follows the lab conventions (`SEED`, `section()`, runs with `uv run`) | 1 |

**Likely mistakes:**

- Using `np.random.seed(SEED)` and `np.random.permutation`: the legacy global generator the lesson
  says not to use. Works, loses 4 points for "no global state".
- Drawing the test indices with `rng.choice(n, n_test)` without `replace=False`: duplicates appear
  and (b) fails, sometimes, which is worse than always.
- Reusing *one* generator object for both "run 1" and "run 2" of part (a): the second call continues
  the stream, so the splits differ and the learner concludes seeding does not work.
- Testing part (c) at a small $n$ and getting the coincidence above.
- Calling `spawn` once and then reusing the same child objects for both comparisons in (c), so the
  second comparison starts from an advanced stream.

## Exercise 3 (think)

**Strong answer:** 0.005 accuracy may be noise. Before believing it I would ask: (1) Was everything
except the feature identical: same data snapshot, same split, same library versions? "Different
days" suggests the data or the environment may have changed, so I want both runs' version lines and
data hashes. (2) How big is the validation set? On 1,000 rows, 0.005 is 5 examples, well inside
run-to-run variation. (3) What is the spread across seeds? I would rerun both versions over 5–10
seeds (or 5-fold cross-validation, B4.3) and compare the mean and spread; if the intervals overlap
heavily, there is no evidence of an improvement. (4) Does `np.random.seed(42)` actually control all
the randomness? It does not seed Python's `random`, PyTorch, or any library with its own generator,
and any extra draw added with the new feature shifts the split, so the two runs may not even share a
validation set. Changes to the script: one `SEED` constant feeding `np.random.default_rng` (spawned
per purpose), `random_state=` on every estimator and splitter, a fixed split saved to disk or
derived from a dedicated stream, versions and a data hash printed at the start, and results reported
as mean ± spread over several seeds.

**Weak answer:** "The seed was fixed with `np.random.seed(42)`, so the runs are comparable and
0.917 > 0.912 means the feature helps." This is weak because it trusts a seed that only covers
NumPy's legacy global generator, ignores that the new feature's code may shift the random stream
(and therefore the split), ignores environment and data changes between days, and treats a single
pair of numbers as evidence without any measure of noise.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Says the difference may be noise and asks for a measure of variation (seeds or CV) | 3 |
| Identifies that the global `np.random.seed` does not cover other libraries or stream shifts | 2 |
| Asks about environment and data changes between the two days | 2 |
| Considers the validation-set size (how many examples 0.005 represents) | 1 |
| Concrete script changes (seeded generators, `random_state`, versions, fixed split) | 2 |

## Scoring

Total 35 pts. Pass ≥ 70% (25 pts). Topic ids to revisit for each lost criterion: exercise 1 →
A1.1 "The math" section; exercise 2 → A1.1 "With NumPy" (streams and `spawn`) and A1.2 (array
indexing); exercise 3 → A1.1 "In practice", and later B4.3 (cross-validation).
