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

Colours follow the variant, not its rank: the baseline is always blue, and
other variants follow in config order. The palette was checked for
colour-vision deficiencies.

## Possible later additions

* Export to the IOHprofiler format, to use the interactive
  [IOHanalyzer](https://iohanalyzer.liacs.nl) on the same data.
* Fitness-distance correlation, edge-frequency entropy and selection
  intensity, if the questions call for them.
