# evosoc

Controlled experiments with genetic algorithms. A "normal", biologically
plausible GA solves the **travelling salesman problem** as the baseline, and
variants of it are run under identical conditions and compared, in graphs and
statistically.

TSP was chosen because it is easy to explain: *find the shortest round trip
through all cities*. It is also a classic GA benchmark with published optimal
solutions (TSPLIB), so results read as "x % longer than the best possible
tour".

## Quick start

```bash
uv venv && uv pip install -e '.[dev]'     # or: python -m venv .venv && pip install -e '.[dev]'
.venv/bin/evosoc run configs/smoke.toml   # runs every variant × problem × seed, then writes the report
.venv/bin/pytest
```

The output goes to `results/<experiment>/` (git-ignored):

```
results/smoke/
  config.toml, manifest.json           exact config + git commit + versions
  runs/<variant>/<problem>/seed<k>/    raw data per run (parquet + summary.json)
  report/figures/*.png|svg             convergence, final quality, runtime ECDF, best tours
  report/tables/*.csv                  per-run results, ERT, comparisons vs baseline
  report/summary.md                    medians, A12 effect sizes, Holm-corrected p-values
```

Runs are resumable: finished runs are skipped. `evosoc analyze results/<name>`
regenerates only the report.

## An experiment is a grid

```toml
name = "baseline-vs-x"
budget = 200_000          # fitness evaluations per run: identical for every variant
seeds = 30
baseline = "baseline"

[[problems]]              # as many as you like
type = "tsplib"           # bundled: eil51 berlin52 st70 eil76 kroA100 ch150 a280
name = "berlin52"

[[problems]]
type = "tsp_uniform"      # seeded random instance
n = 200
seed = 1

[[variants]]              # as many as you like; each one is compared to `baseline`
name = "baseline"
algorithm = "..."         # name registered with @register(...)
params = { }
```

This shape covers both directions the project may take: **many variants
against one baseline**, or **one baseline + one variant across many problems**.

## Layout

| Path | Role |
|---|---|
| `src/evosoc/problems/` | `Problem` protocol + TSP (TSPLIB loader, random instances). New problem types plug in here. |
| `src/evosoc/algorithms/` | `Algorithm` base class, `Context` (evaluation budget and bookkeeping), registry. `steady_state_ga` is **the baseline**; `random_search` is a sanity floor. |
| `src/evosoc/operators/` | Crossover and mutation operators (Numba-compiled) |
| `src/evosoc/metrics.py` | Per-generation population metrics |
| `src/evosoc/experiment.py` | Config, runner (parallel, resumable, common random numbers), loading results |
| `src/evosoc/analysis/` | Run-level metrics, statistics, figures, report |
| `configs/` | Experiment definitions |
| `docs/metrics.md` | **What is tracked and why**, with the benchmarking practice it follows |
| `docs/baseline-ga.md` | **Why the baseline GA looks the way it does**: every component chosen to be as nature-like as works |

## Writing an algorithm

```python
from dataclasses import dataclass

from evosoc.algorithms import Algorithm, Context, Population, register


@register("my_ga")
@dataclass
class MyGA(Algorithm):
    pop_size: int = 200

    def run(self, ctx: Context) -> None:
        pop = ctx.initial_population(self.pop_size)  # same individuals for every variant per seed
        while True:  # ends when the budget runs out
            ...  # use ctx.rng and ctx.evaluate(genome)
            ctx.report(pop)  # once per generation; extra metrics: ctx.report(pop, foo=1.0)
```

The algorithm never counts evaluations, times itself or saves anything, so
every variant is measured in exactly the same way.
