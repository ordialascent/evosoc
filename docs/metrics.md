# What we measure, and why

GA papers are easy to get wrong in the comparison rather than the algorithm.
This setup follows established practice from evolutionary-computation
benchmarking: COCO (Hansen et al. 2021), IOHprofiler (Doerr et al. 2018),
and Arcuri & Briand (2011) on statistics for randomised algorithms.

## Ground rules for a fair comparison

| Rule | Why | Where it is enforced |
|---|---|---|
| **Same budget in fitness evaluations**, and evaluations on the x-axis, never generations | Variants may use different population sizes or produce different numbers of offspring per generation. Only evaluations are comparable. | `Context.evaluate` raises `BudgetExhausted` at the budget |
| **Many independent runs** (≥ 20–30 seeds for claims, 10 for exploring) | A GA run is a random sample. One run shows nothing. | `seeds` in the config |
| **Common random numbers** | Seed *k* gives every variant the same initial population, so runs can be compared in pairs, which gives more statistical power. | `Context.initial_population` |
| **Measuring never changes the run** | Diversity sampling uses its own random generator. | `experiment.run_one` |
| **Equal tuning effort** | A variant tuned harder than the baseline wins for the wrong reason. | discipline (write it down in the config) |
| **Report the median and IQR**, not the mean ± std | GA outcomes are skewed and bounded below by the optimum. | `analysis` |

## Online metrics (recorded every generation)

Stored in `generations.parquet`, one row per `ctx.report(...)`:

| Column | Meaning | What it tells you |
|---|---|---|
| `best_so_far` | Best objective value found so far in the run | Solution quality over time: the main curve |
| `pop_best`, `pop_mean`, `pop_median`, `pop_worst`, `pop_std` | Fitness distribution of the *current* population | Selection pressure. The mean catching up with the best means the population is converging. |
| `diversity` | Mean pairwise genotypic distance (for TSP: the fraction of edges two tours don't share), estimated from sampled pairs | Exploration vs. exploitation. Diversity collapsing early is the usual sign of **premature convergence**. |
| `unique_fraction` | Distinct solutions ÷ population size | Duplicates crowding out the population |
| `x_*` | Anything a variant reports via `ctx.report(pop, name=value)` | Variant-specific internals, measured with the same machinery |

Separately, `improvements.parquet` logs *every* best-so-far improvement with
its exact evaluation number (as IOHprofiler does). All quality-over-time
metrics are computed from it, so they don't depend on how a variant groups its
evaluations into generations. (Sampling best-so-far once per generation
biases anytime metrics and hitting times in favour of variants with large
generations.)

## Run-level metrics (computed afterwards)

| Metric | Definition | Question it answers |
|---|---|---|
| **Gap** | `(best − optimum) / optimum`. For instances with an unknown optimum, the best value any run found is used instead. | How good is the final answer? The standard TSP quality measure. |
| **Anytime gap** | Mean gap over a log-spaced evaluation grid, i.e. the area under the convergence curve | Who is better *throughout* the run, not only at the arbitrary budget cut-off? |
| **Hitting time** | Evaluations until the gap first falls to ≤ target (20 %, 10 %, 5 %, 2 %, 1 %, 0 %) | How fast is a given quality reached? |
| **Success rate** and **ERT** | Share of runs that hit the target, and the expected running time: evaluations spent over all runs ÷ number of successes (COCO) | Speed and reliability in one number, with failed runs counted fairly |
| **Runtime ECDF** | Share of (run, target) pairs reached by *x* evaluations, over all problems | The single summary figure: a higher curve that rises earlier is better |

## Statistics

* **Mann–Whitney U** test (unpaired) and **Wilcoxon signed-rank** test (paired
  by seed, possible because of common random numbers). Both are
  non-parametric: GA results are not normally distributed.
* **Vargha–Delaney A12** effect size: the probability that a random variant
  run beats a random baseline run. 0.5 means no difference; 0.56, 0.64 and
  0.71 are the thresholds for small, medium and large effects. A p-value
  alone says nothing about *how much* better something is.
* **Holm correction** across all variant × problem comparisons.

## Figures

| File | Content |
|---|---|
| `convergence` | Rows: best-so-far gap, population mean gap, diversity, distinct genomes. Columns: problems. Each variant's median line with an IQR band, over evaluations on a log axis. |
| `final_gap`, `anytime_gap` | Every run as a dot over a box plot, per variant and problem |
| `runtime_ecdf` | The COCO/IOHprofiler data profile |
| `best_tours` | The best tour each variant found: the explainable picture |
| `social` | The social analysis (below) |

Colours follow the variant, not its rank: the baseline is always blue, and
other variants follow in config order. The palette was checked for
colour-vision deficiencies.

## Social analysis

How the population *as a whole* fares, not only its best member. Figure
`report/figures/social.png`, scores in `summary.md` and
`report/tables/social_*.csv`.

Two scales, both going up:

* **Progress** (fitness): `(random − length) / (random − optimum)`. 0 % = no
  better than a typical random tour (its expected length, computed exactly),
  100 % = the optimal tour (or the best value any run found, if the optimum
  is unknown). Below 0 % = worse than a typical random tour.
* **Optimal edges** (closeness to the optimum): the share of the optimal
  tour's connections that a tour has. A random tour has about 2/n of them.
  A tour can be short without being structurally close to the optimum, which
  is why this is tracked separately. Needs a known optimal tour (all bundled
  TSPLIB instances); empty for random instances.

| Row in the figure | Score | Question |
|---|---|---|
| Best so far (progress) | `best_progress`: at the end | How good is the best tour found? |
| Population average (progress) | `average_progress`: at the end | How is the average individual doing? |
| Closest to optimum so far (optimal edges) | `closest_to_optimum`: the highest ever | How close has anyone come to the optimal tour? |
| Population average (optimal edges) | `average_closeness`: at the end | How close is the average individual? |
| Wealth concentration (Gini) | `gini`: averaged over the run | Is fitness held by a few "high earners", or shared? |
| Best–worst gap (progress points) | `spread`: averaged over the run | How far apart are the best and worst individual? |

**Wealth and the Gini coefficient.** An individual's wealth is how much
shorter its tour is than a typical random tour (0 if it is not shorter). The
Gini coefficient of the population's wealth is 0 when everyone is equally
wealthy and approaches 1 when one individual holds it all. Note that it
depends on where "zero wealth" is: a population of random tours scores
about 0.7, because about half of them are worse than average and so own
nothing.

For `gini` and `spread` neither direction is better, so their verdicts say
*higher* or *lower* than the baseline instead of *better* or *worse*.

"Closest so far" and the other population metrics are sampled once per
reported generation, so an individual born and gone within one generation is
not seen. Best so far is exact (every evaluation).

## Possible later additions

* Export to the IOHprofiler format, to use the interactive
  [IOHanalyzer](https://iohanalyzer.liacs.nl) on the same data.
* Fitness-distance correlation, edge-frequency entropy and selection
  intensity, if the questions call for them.
