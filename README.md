# evosoc

Controlled experiments with genetic algorithms. A plain, textbook GA solves
the **travelling salesman problem** as the baseline, and
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
| `src/evosoc/operators/` | Crossover (order crossover) and mutation (inversion) for tours |
| `src/evosoc/metrics.py` | Per-generation population metrics |
| `src/evosoc/experiment.py` | Config, runner (parallel, resumable, common random numbers), loading results |
| `src/evosoc/analysis/` | Run-level metrics, statistics, figures, report |
| `configs/` | Experiment definitions |
| `docs/metrics.md` | **What is tracked and why**, with the benchmarking practice it follows |
| `docs/baseline-ga.md` | **Why the baseline GA looks the way it does**: a classroom GA; wherever the textbook offers options, the one closest to nature |

## Trying a variant

1. **Copy the baseline**:
   `cp src/evosoc/algorithms/steady_state.py src/evosoc/algorithms/my_variant.py`.
   Any file in `src/evosoc/algorithms/` is picked up automatically.
2. **Rename it**: `@register("my_variant")` and the class name. Then change
   whatever you like inside `run`. Leave `steady_state.py` alone: it is the
   reference, and `tests/test_baseline.py` fails if its behaviour changes.
3. **Write a config**: copy `configs/baseline.toml` to `configs/my-variant.toml`,
   change `name`, and add the variant next to the baseline:

   ```toml
   [[variants]]
   name = "my-variant"
   algorithm = "my_variant"
   params = { pop_size = 200 }
   ```
4. **Run**: `.venv/bin/evosoc run configs/my-variant.toml`. For a quick look,
   use `seeds = 3` and a smaller budget (a minute or two); the verdicts need
   10+ seeds, and claims 20+.
5. **Read** `results/my-variant/report/summary.md`: per problem, whether the
   variant is **better**, **worse** or shows **no clear difference** from the
   baseline, plus the figures in `report/figures/`.

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

What a variant must do: evaluate every genome through `ctx.evaluate` (or
`ctx.evaluate_all`), start from `ctx.initial_population`, use `ctx.rng` for all
randomness, and call `ctx.report` once per generation. Everything else is free.

The algorithm never counts evaluations, times itself or saves anything, so
every variant is measured in exactly the same way.
