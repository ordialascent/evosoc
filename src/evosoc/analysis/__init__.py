"""Turn an experiment's raw runs into figures, tables and a summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..experiment import ExperimentConfig, finished_runs, load_runs  # noqa: E402
from ..problems import make_problem  # noqa: E402
from . import plots, stats  # noqa: E402

#: Target gaps for hitting times, ERT and the runtime ECDF.
TARGETS = (0.20, 0.10, 0.05, 0.02, 0.01, 0.0)


def analyze(exp_dir: str | Path) -> Path:
    exp_dir = Path(exp_dir)
    cfg = ExperimentConfig.from_toml(exp_dir / "config.toml")
    out = exp_dir / "report"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    improvements = load_runs(exp_dir, "improvements")
    reference = stats.references(improvements)
    improvements = stats.add_gap(improvements, reference)
    generations = stats.add_gap(load_runs(exp_dir, "generations"), reference)
    order = plots.variant_order(improvements, cfg.baseline)

    final = stats.final_rows(generations)
    final = final.merge(stats.anytime_gap(improvements, cfg.budget), on=stats.RUN_KEYS)
    hits = stats.hitting_times(improvements, TARGETS, cfg.budget)
    ert = stats.ert(hits)

    summary = (
        final.groupby(["problem", "variant"])[["gap", "anytime_gap", "diversity"]]
        .median()
        .add_prefix("median_")
    )
    comparisons = {
        m: stats.compare_to_baseline(final, cfg.baseline, m) for m in ("gap", "anytime_gap")
    }

    final.to_csv(out / "tables" / "final_runs.csv", index=False)
    summary.to_csv(out / "tables" / "summary.csv")
    ert.to_csv(out / "tables" / "ert.csv")
    for m, table in comparisons.items():
        table.to_csv(out / "tables" / f"vs_baseline_{m}.csv", index=False)

    tables = {"improvements": improvements, "generations": generations}
    _save(plots.convergence(tables, order, cfg.budget), out / "figures" / "convergence")
    _save(plots.final_distribution(final, order, "gap"), out / "figures" / "final_gap")
    _save(plots.final_distribution(final, order, "anytime_gap"), out / "figures" / "anytime_gap")
    _save(plots.runtime_ecdf(hits, order, cfg.budget), out / "figures" / "runtime_ecdf")
    tours = _best_tours(exp_dir)
    coords = {}
    for spec in cfg.problems:
        problem = make_problem(spec)
        if hasattr(problem, "coords"):
            coords[problem.name] = problem.coords
    if coords and not tours.empty:
        _save(plots.best_tours(coords, tours, order), out / "figures" / "best_tours")

    _write_markdown(out / "summary.md", cfg, summary, comparisons)
    return out


def _best_tours(exp_dir: Path) -> pd.DataFrame:
    rows = [json.loads((r / "summary.json").read_text()) for r in finished_runs(exp_dir)]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["problem"] = df["problem_name"]
    return df.sort_values("best").groupby(["problem", "variant"], as_index=False).first()


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def _write_markdown(path: Path, cfg: ExperimentConfig, summary: pd.DataFrame, comparisons) -> None:
    lines = [
        f"# {cfg.name}",
        "",
        f"Budget {cfg.budget:,} evaluations per run, {len(cfg.seeds)} seeds, "
        f"baseline `{cfg.baseline}`.",
        "",
        "## Medians per variant (lower is better)",
        "",
        summary.to_markdown(floatfmt=".4f"),
        "",
    ]
    for metric, table in comparisons.items():
        if table.empty:
            continue
        cols = ["problem", "variant", "median", "baseline_median", "a12", "p_holm"]
        lines += [
            f"## `{metric}` vs baseline",
            "",
            "A12 > 0.5: the variant tends to beat the baseline (0.56 small, 0.64 medium, "
            "0.71 large). p: Mann-Whitney U, Holm-corrected.",
            "",
            table[cols].to_markdown(index=False, floatfmt=".4g"),
            "",
        ]
    path.write_text("\n".join(lines))
