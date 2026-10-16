# Answer key: A1.4 matplotlib

Lesson: [`lessons/A/a1_4_matplotlib.md`](../../../lessons/A/a1_4_matplotlib.md)

## Exercise 1 (by hand)

**Solution.** $\mathbf{x} = (0.5, 1.5, 1.5, 2.0, 3.9, 4.0, 6.5, 8.0)$, $n = 8$, $k = 4$,
$[\ell, u] = [0.5, 8.0]$.

- Width: $h = (8.0 - 0.5)/4 = 7.5/4 = 1.875$.
- Edges: $e_j = 0.5 + 1.875j$: $0.5,\ 2.375,\ 4.25,\ 6.125,\ 8.0$.
- Bin indices $\min(\lfloor (x - 0.5)/1.875 \rfloor, 3)$:

| $x$ | $(x - 0.5)/1.875$ | floor | min with 3 |
|---|---|---|---|
| 0.5 | 0 | 0 | 0 |
| 1.5 | 0.5333 | 0 | 0 |
| 1.5 | 0.5333 | 0 | 0 |
| 2.0 | 0.8 | 0 | 0 |
| 3.9 | 1.8133 | 1 | 1 |
| 4.0 | 1.8667 | 1 | 1 |
| 6.5 | 3.2 | 3 | 3 |
| 8.0 | 4.0 | 4 | **3** (the maximum goes in the last bin) |

- Counts: $(4, 2, 0, 2)$.
- Densities $c_j/(n h) = c_j / 15$: $4/15 \approx 0.2667$, $2/15 \approx 0.1333$, $0$, $0.1333$.
- Areas: $(4 + 2 + 0 + 2)/15 \times 1.875 = 8 \times 0.125 = 1$. ✓
- Sturges: $n = 8$: $\lceil \log_2 8 \rceil + 1 = 3 + 1 = 4$. $n = 100{,}000$: $\log_2 100000 \approx
  16.61$, so $\lceil 16.61 \rceil + 1 = 18$.

```text
== exercise 1 check ==
h: 1.875  edges: [0.5, 2.375, 4.25, 6.125, 8.0]
raw (x - lo) / h: [0.0, 0.5333, 0.5333, 0.8, 1.8133, 1.8667, 3.2, 4.0]
counts: [4, 2, 0, 2]  density: [0.2667, 0.1333, 0.0, 0.1333]
Sturges n=8: 4  n=100000: 18
```

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| $h$ and the five edges correct | 2 |
| Bin indices correct, including 8.0 → bin 3 via the min | 3 |
| Counts correct | 1 |
| Densities correct and areas shown to sum to 1 | 2 |
| Sturges for both $n$ correct (ceiling applied) | 2 |

**Likely mistakes:**

- Putting 8.0 in a fifth bin (index 4), or dropping it.
- Dividing by $n$ only for the density (giving fractions that sum to 1 as *heights*, not areas).
- Rounding $\log_2 100000$ down to 16, giving 17.
- Using 5 edges as "5 bins".

## Exercise 2 (code)

**Solution:** [`a1_4_matplotlib_ex2.py`](a1_4_matplotlib_ex2.py). The core is the lie-factor
function; the plotting functions are thin wrappers that save the figure and return it:

```python
def lie_factor(first: float, last: float, y_start: float) -> float:
    """Tufte's lie factor for two bars drawn from y_start: graphic effect / data effect."""
    data_effect = (last - first) / first
    drawn_effect = (last - y_start) / (first - y_start) - 1.0
    return drawn_effect / data_effect
```

```text
== exercise 2: lie factor ==
misleading (axis from 87): lie factor 436.0
honest (axis from 0):      lie factor 1.0

== exercise 2: bin counts ==
  5 bins: max count 797, empty bins   0
 30 bins: max count 218, empty bins   7
300 bins: max count  34, empty bins 159
```

By hand: accuracy 87.2 → 88.0 is a $0.8/87.2 \approx 0.92\%$ change; drawn from 87, the bars are 0.2
and 1.0 long, a 400% change; $4 / 0.00917 \approx 436$. The closer the axis starts to the smaller
value, the larger the lie factor, without limit.

On bins: 5 bins put 797 of 1,000 values in the first bar and hide the shape of the peak; 300 bins
leave 159 bins empty and show mostly noise; 30 bins (or `"auto"`) show a smooth right-skewed decay.
A good sentence: "I would publish 30 bins (or a log-x histogram), because it shows the peak near 20–100
ms and the long tail without empty-bin noise."

**Rubric (15 pts):**

| Criterion | Points |
|---|---|
| Lie factor computed correctly (graphic effect / data effect, both relative) | 4 |
| Honest version returns ~1.0 | 1 |
| Charts follow the rules: labelled axes with units, titles, values annotated | 3 |
| Object-oriented API, `plt.close(fig)` after saving | 2 |
| Three histograms as subplots with `sharex=True` | 2 |
| A justified choice of bin count | 2 |
| Type hints, docstrings | 1 |

**Likely mistakes:**

- Using absolute differences for the effects ($88.0 - 87.2$ vs $1.0 - 0.2$), which gives a ratio of
  1 and hides the distortion.
- Swapping numerator and denominator (lie factor $\approx 0.0023$).
- Calling `plt.bar` and `plt.ylim` (pyplot state API) instead of methods on `ax`.
- Not closing figures inside a helper that is called repeatedly.

## Exercise 3 (think)

**Strong answer:** The chart has three problems and one missing comparison. (1) **Range:** why 6
weeks, and why starting at launch? I would ask for at least 6 months before launch so I can see the
normal level and seasonality; ticket volume often drops after holidays or with fewer users, and a
6-week window can catch a natural dip. (2) **Axis:** 1,150–1,250 is a zoomed line chart; that is
acceptable for a line, but I want the actual size of the drop in tickets and as a percentage (e.g.
"−40 tickets/week, −3%") so nobody reads the slope as a collapse. (3) **Noise:** how much does weekly
volume vary normally? If weekly counts swing by ±60, a 40-ticket drop is within noise. (4)
**Comparison:** what else changed: a UI release, a pricing change, fewer active users? Tickets per
active user is a better metric than raw tickets. Ideally the model was rolled out to part of the
traffic (an A/B test, A6.3), and I would ask for the chart of tickets per user for the treatment and
control groups over the same weeks. The chart I would ask for: weekly tickets per 1,000 active users
for 6 months before and 6 weeks after, with the launch date marked, a band showing the normal
week-to-week range, and, if available, control versus treatment as two labelled lines.

**Weak answer:** "The y-axis should start at 0." Weak because for a line chart zooming is acceptable,
and it misses the real issues: the cherry-picked window with no baseline, no measure of normal
variation, raw counts instead of a rate, and no control group to separate the model's effect from
everything else that happened.

**Rubric (10 pts):**

| Criterion | Points |
|---|---|
| Asks for a longer range / baseline before launch (cherry-picking, seasonality) | 3 |
| Asks for the size of normal variation (noise) and the effect size | 2 |
| Asks for a rate (per active user) instead of raw counts, or about other changes | 2 |
| Mentions a control group / A/B test | 1 |
| Describes a concrete better chart | 2 |

## Scoring

Total 35 pts. Pass ≥ 70% (25 pts). Topic ids to revisit for each lost criterion: exercise 1 →
A1.4 "Equal-width binning" and "Density"; exercise 2 → A1.4 "The lie factor" and "Figure, Axes";
exercise 3 → A1.4 "Plots that don't mislead", and later A6.3 (A/B tests).
