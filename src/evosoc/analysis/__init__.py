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
from . import plots, social, stats  # noqa: E402

#: Target gaps for hitting times, ERT and the runtime ECDF.
TARGETS = (0.20, 0.10, 0.05, 0.02, 0.01, 0.0)


def analyze(exp_dir: str | Path) -> Path:
    exp_dir = Path(exp_dir)
    cfg = ExperimentConfig.from_toml(exp_dir / "config.toml", check_algorithms=False)
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

    social_section: list[str] = []
    if "gini" in generations:  # results from before the social metrics lack it
        generations = social.add_columns(generations)
        tables = {"improvements": improvements, "generations": generations}
        _save(
            plots.convergence(tables, order, cfg.budget, social.CURVES),
            out / "figures" / "social",
        )
        if "quarter1_mean" in generations:
            _save(plots.quarters(generations, order, cfg.budget), out / "figures" / "quarters")
        runs = social.scores(generations, cfg.budget)
        social_comparison = social.compare(runs, cfg.baseline)
        runs.to_csv(out / "tables" / "social_runs.csv", index=False)
        social_comparison.to_csv(out / "tables" / "social_vs_baseline.csv", index=False)
        social_section = _social_markdown(runs, social_comparison)

    _write_markdown(out / "summary.md", cfg, summary, comparisons, social_section)
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


def _social_markdown(runs: pd.DataFrame, comparison: pd.DataFrame) -> list[str]:
    scores = [c for c, _, _ in social.SCORES]
    medians = runs.groupby(["problem", "variant"])[scores].median().reset_index()
    lines = [
        "## Social analysis",
        "",
        "How the population as a whole fares (figure: `figures/social.png`). Tour lengths "
        "in the problem's own distance units. Medians over runs:",
        "",
        *[f"* `{c}`: {d}" for c, d, _ in social.SCORES],
        "",
        medians.to_markdown(index=False, floatfmt=".3f"),
        "",
    ]
    if not comparison.empty:
        cols = ["problem", "variant", "score", "verdict", "median", "baseline_median", "a12"]
        lines += [
            "### Social scores vs baseline",
            "",
            "Verdict as above. For `gini` and `spread` neither direction is better, so the "
            "verdict says higher / lower.",
            "",
            comparison[cols].to_markdown(index=False, floatfmt=".3g"),
            "",
        ]
    return lines


def _write_markdown(
    path: Path,
    cfg: ExperimentConfig,
    summary: pd.DataFrame,
    comparisons,
    extra: list[str],
) -> None:
    lines = [
        f"# {cfg.name}",
        "",
        f"Budget {cfg.budget:,} evaluations per run, {len(cfg.seeds)} seeds, "
        f"baseline `{cfg.baseline}`.",
        "",
    ]
    if len(cfg.seeds) < 10:
        lines += [
            f"> **Only {len(cfg.seeds)} seeds: too few for the verdicts below to mean much.** "
            "Fine for a quick look at the figures; use 10 or more (20+ for claims).",
            "",
        ]
    lines += [
        "## Medians per variant (lower is better)",
        "",
        summary.reset_index().to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    for metric, table in comparisons.items():
        if table.empty:
            continue
        cols = ["problem", "variant", "verdict", "median", "baseline_median", "a12", "p_holm"]
        lines += [
            f"## `{metric}` vs baseline",
            "",
            "Verdict: better / worse than the baseline at p < 0.05 (Mann-Whitney U, "
            "Holm-corrected), else no clear difference. A12: probability that a variant run "
            "beats a baseline run (0.5 = even; 0.56 / 0.64 / 0.71 = small / medium / large "
            "effect).",
            "",
            table[cols].to_markdown(index=False, floatfmt=".4g"),
            "",
        ]
    path.write_text("\n".join(lines + extra))
