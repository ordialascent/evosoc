"""Social analysis: how the population as a whole fares, not just its best member.

Every measure keeps its natural unit: tour length for fitness, a plain 0-1
number for the Gini coefficient. See ``docs/metrics.md``, "Social analysis".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats
from .plots import Curve

#: Figure rows. The dashed line in the length rows is the optimum.
CURVES = [
    Curve("improvements", "best_so_far", "Best so far\n(tour length)", reference="optimum"),
    Curve("generations", "pop_mean", "Population average\n(tour length)", reference="optimum"),
    Curve("generations", "gini", "Wealth concentration\n(Gini)", ylim=(0, 1.02)),
    Curve("generations", "spread", "Best–worst gap\n(tour length)"),
    Curve("generations", "pop_worst", "Poorest individual\n(tour length)", reference="optimum"),
]

#: Per-run scores: (column, description, which direction is better; None = neither).
SCORES = [
    ("best_length", "best tour length found", "lower"),
    ("average_length", "population average tour length at the end", "lower"),
    ("gini", "wealth concentration (Gini, 0-1), averaged over the run", None),
    ("spread", "best–worst gap in tour length, averaged over the run", None),
    ("worst_length", "poorest individual's tour length at the end", "lower"),
]


def add_columns(generations: pd.DataFrame) -> pd.DataFrame:
    """Add the derived column: the best-worst gap."""
    df = generations.copy()
    df["spread"] = df["pop_worst"] - df["pop_best"]
    return df


def scores(generations: pd.DataFrame, budget: int) -> pd.DataFrame:
    """One row per run with the ``SCORES`` columns."""
    final = stats.final_rows(generations)
    out = final[stats.RUN_KEYS].copy()
    out["best_length"] = final["best_so_far"].to_numpy()
    out["average_length"] = final["pop_mean"].to_numpy()
    out["worst_length"] = final["pop_worst"].to_numpy()
    grid = np.linspace(budget / 100, budget, 100).round().astype(int)
    for column in ("gini", "spread"):
        g = stats.on_grid(generations, column, grid)
        mean = g.groupby(stats.RUN_KEYS, as_index=False)["value"].mean()
        out = out.merge(mean.rename(columns={"value": column}), on=stats.RUN_KEYS)
    return out


def compare(runs: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Every score of every variant against the baseline, with a verdict."""
    tables = []
    for column, _, better in SCORES:
        t = stats.compare_to_baseline(runs, baseline, column, higher=better or "neutral")
        t.insert(2, "score", column)
        tables.append(t)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
