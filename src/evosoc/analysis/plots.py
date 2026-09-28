"""Figures comparing variants. Every function returns a matplotlib Figure.

Colour follows the variant, never its rank: the baseline always takes the
first slot of a fixed, colour-vision-deficiency-checked categorical palette,
and other variants follow in config order.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from .stats import eval_grid, on_grid

#: Fixed categorical order (validated for adjacent-pair CVD separation).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2,
    "font.family": "sans-serif",
}


class Curve(NamedTuple):
    """One row of a :func:`convergence` figure."""

    source: str  # "improvements" or "generations"
    column: str
    label: str
    ylim: tuple[float, float] | None = None  # None = automatic
    percent: bool = False  # show values as percentages
    #: Column holding a per-problem value to mark with a dashed line (e.g. "optimum").
    reference: str | None = None


#: Convergence panels. Gaps are relative to the optimum, or to the best-known
#: value if it is unknown.
CURVES = [
    Curve("improvements", "gap", "Best-so-far gap", percent=True),
    Curve("generations", "pop_mean_gap", "Population mean gap", percent=True),
    Curve("generations", "diversity", "Diversity\n(mean pairwise distance)", (0, 1.02), True),
    Curve("generations", "unique_fraction", "Distinct genomes", (0, 1.02), True),
]


def colors(variants: Sequence[str]) -> dict[str, str]:
    if len(variants) > len(PALETTE):
        raise ValueError(
            f"{len(variants)} variants but only {len(PALETTE)} distinguishable colours: "
            "split the comparison into several figures"
        )
    return dict(zip(variants, PALETTE, strict=False))


def variant_order(df: pd.DataFrame, baseline: str) -> list[str]:
    others = [v for v in pd.unique(df["variant"]) if v != baseline]
    return [baseline, *others] if baseline in set(df["variant"]) else others


def convergence(
    tables: dict[str, pd.DataFrame],
    order: Sequence[str],
    budget: int,
    metrics: Sequence[Curve] = CURVES,
    log_x: bool = True,
) -> Figure:
    """Median (line) and interquartile range (band) over runs, per variant.

    Rows are metrics, columns are problems. ``tables`` maps the source names
    used in ``metrics`` ("improvements", "generations") to run tables.
    """
    problems = list(pd.unique(tables["improvements"]["problem"]))
    grid = eval_grid(budget)
    pal = colors(order)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            len(metrics),
            len(problems),
            figsize=(3.6 * len(problems) + 1, 2.6 * len(metrics)),
            squeeze=False,
            sharex=True,
            layout="constrained",
        )
        for r, curve in enumerate(metrics):
            table = tables[curve.source]
            g = on_grid(table, curve.column, grid)
            q = (
                g.groupby(["problem", "variant", "evaluations"])["value"]
                .quantile([0.25, 0.5, 0.75])
                .unstack()
            )
            for c, problem in enumerate(problems):
                ax = axes[r, c]
                for variant in order:
                    if (problem, variant) not in q.index.droplevel(2):
                        continue
                    s = q.loc[(problem, variant)]
                    ax.fill_between(
                        s.index, s[0.25], s[0.75], color=pal[variant], alpha=0.18, linewidth=0
                    )
                    ax.plot(s.index, s[0.5], color=pal[variant], label=variant)
                if curve.reference is not None:
                    ref = table.loc[table["problem"] == problem, curve.reference].dropna()
                    if not ref.empty:
                        ax.axhline(ref.iloc[0], color=INK_2, linestyle="--", linewidth=1)
                        ax.annotate(
                            f"{curve.reference} {ref.iloc[0]:g}",
                            (0, ref.iloc[0]),
                            xycoords=("axes fraction", "data"),
                            xytext=(3, 3),
                            textcoords="offset points",
                            ha="left",
                            va="bottom",
                            fontsize=7,
                            color=INK_2,
                        )
                if log_x:
                    ax.set_xscale("log")
                # Shared x-axis: numbers once per column, above the first and below
                # the last panel; inner panels get no tick marks, so none look cut off.
                if r == 0:
                    ax.tick_params(which="both", top=True, labeltop=True)
                if r < len(metrics) - 1:
                    ax.tick_params(which="both", bottom=False)
                if curve.percent:
                    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
                if curve.ylim is not None:
                    ax.set_ylim(*curve.ylim)
                if r == 0:
                    ax.set_title(problem)
                if c == 0:
                    ax.set_ylabel(curve.label)
                if r == len(metrics) - 1:
                    ax.set_xlabel("Fitness evaluations")
        _legend(fig, axes[0, 0])
    return fig


QUARTERS = ["best 25 %", "25–50 %", "50–75 %", "worst 25 %"]


def quarters(generations: pd.DataFrame, order: Sequence[str], budget: int) -> Figure:
    """Mean fitness of each quarter of the population over time (median over
    runs). Rows are variants, columns problems; darkest line = best quarter.
    Lines that fan out show a population splitting apart; lines that merge
    show it converging."""
    problems = list(pd.unique(generations["problem"]))
    variants = [v for v in order if v in set(generations["variant"])]
    pal = colors(order)
    grid = eval_grid(budget)
    alphas = [1.0, 0.7, 0.45, 0.25]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            len(variants),
            len(problems),
            figsize=(3.6 * len(problems) + 1, 2.8 * len(variants)),
            squeeze=False,
            sharex=True,
            layout="constrained",
        )
        for k, name in enumerate(QUARTERS, 1):
            g = on_grid(generations, f"quarter{k}_mean", grid)
            med = g.groupby(["problem", "variant", "evaluations"])["value"].median()
            for r, variant in enumerate(variants):
                for c, problem in enumerate(problems):
                    if (problem, variant) not in med.index.droplevel(2):
                        continue
                    s = med.loc[(problem, variant)]
                    axes[r, c].plot(
                        s.index, s.to_numpy(), color=pal[variant], alpha=alphas[k - 1], label=name
                    )
        for r, variant in enumerate(variants):
            for c, problem in enumerate(problems):
                ax = axes[r, c]
                ref = generations.loc[generations["problem"] == problem, "optimum"].dropna()
                if not ref.empty:
                    ax.axhline(ref.iloc[0], color=INK_2, linestyle="--", linewidth=1)
                ax.set_xscale("log")
                if r == 0:
                    ax.set_title(problem)
                    ax.tick_params(which="both", top=True, labeltop=True)
                if r < len(variants) - 1:
                    ax.tick_params(which="both", bottom=False)
                else:
                    ax.set_xlabel("Fitness evaluations")
                if c == 0:
                    ax.set_ylabel(f"{variant}\n(mean tour length\nper quarter)")
            axes[r, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    return fig


def final_distribution(final: pd.DataFrame, order: Sequence[str], metric: str = "gap") -> Figure:
    """Every run's end result (dots) over a box plot, per variant and problem."""
    problems = list(pd.unique(final["problem"]))
    pal = colors(order)
    rng = np.random.default_rng(0)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1,
            len(problems),
            figsize=(1.2 * len(order) * len(problems) + 1.5, 3.2),
            squeeze=False,
            layout="constrained",
        )
        for c, problem in enumerate(problems):
            ax = axes[0, c]
            p = final[final["problem"] == problem]
            data = [p.loc[p["variant"] == v, metric].to_numpy() for v in order]
            ax.boxplot(
                data,
                widths=0.5,
                showfliers=False,
                medianprops={"color": INK, "linewidth": 1.5},
                boxprops={"color": INK_2},
                whiskerprops={"color": INK_2},
                capprops={"color": INK_2},
            )
            for i, (v, values) in enumerate(zip(order, data, strict=True), 1):
                x = i + rng.uniform(-0.12, 0.12, len(values))
                ax.scatter(
                    x, values, s=18, color=pal[v], edgecolor=SURFACE, linewidth=0.8, zorder=3
                )
            ax.set_xticks(range(1, len(order) + 1), order, rotation=20, ha="right")
            ax.set_title(problem)
            ax.grid(axis="x", visible=False)
            if metric.endswith("gap"):
                ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
        label = {"gap": "Final gap", "anytime_gap": "Anytime gap (whole run)"}.get(metric, metric)
        axes[0, 0].set_ylabel(label)
    return fig


def runtime_ecdf(hits: pd.DataFrame, order: Sequence[str], budget: int) -> Figure:
    """Fraction of (run, target) pairs solved by a given number of evaluations.

    The COCO/IOHprofiler "data profile": aggregates all problems and targets
    into one anytime curve per variant. Higher and further left is better.
    """
    pal = colors(order)
    grid = eval_grid(budget)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
        for variant in order:
            h = hits.loc[hits["variant"] == variant, "hit_evaluations"].to_numpy()
            if h.size == 0:
                continue
            solved = (h[None, :] <= grid[:, None]).sum(axis=1) / h.size  # NaN never <=
            ax.step(grid, solved, where="post", color=pal[variant], label=variant)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Fitness evaluations")
        targets = ", ".join(f"{t:.0%}" for t in sorted(hits["target"].unique(), reverse=True))
        ax.set_title("Runtime ECDF, all problems")
        ax.set_ylabel(f"(run, target) pairs reached\ntarget gaps: {targets}")
        _legend(fig, ax)
    return fig


def best_tours(coords: dict[str, np.ndarray], tours: pd.DataFrame, order: Sequence[str]) -> Figure:
    """The best tour each variant found, per problem (TSP only).

    ``tours`` has columns problem, variant, best, best_genome.
    """
    problems = list(coords)
    pal = colors(order)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            len(problems),
            len(order),
            figsize=(2.8 * len(order), 2.8 * len(problems)),
            squeeze=False,
            layout="constrained",
        )
        for r, problem in enumerate(problems):
            xy = coords[problem]
            for c, variant in enumerate(order):
                ax = axes[r, c]
                ax.set_aspect("equal")
                ax.axis("off")
                row = tours[(tours["problem"] == problem) & (tours["variant"] == variant)]
                if row.empty:
                    continue
                t = np.asarray(row.iloc[0]["best_genome"])
                loop = xy[np.append(t, t[0])]
                ax.plot(loop[:, 0], loop[:, 1], color=pal[variant], linewidth=1.2)
                ax.scatter(xy[:, 0], xy[:, 1], s=6, color=INK, zorder=3)
                ax.set_title(f"{variant}\n{problem}: {row.iloc[0]['best']:.0f}", fontsize=9)
    return fig


def _legend(fig: Figure, ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) >= 2:
        fig.legend(handles, labels, loc="outside upper center", ncols=min(len(labels), 8))
