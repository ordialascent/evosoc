"""Social analysis: how the population as a whole fares, not just its best member.

Everything is on a *progress* scale that goes up: 0 % = no better than a
random tour, 100 % = the optimum (or the best value any run found, if the
optimum is unknown). See ``docs/metrics.md``, "Social analysis".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats

#: Figure rows: (source table, column, y label, y limits or None for auto).
CURVES = [
    ("improvements", "progress", "Best so far\n(progress)", None),
    ("generations", "progress_mean", "Population average\n(progress)", None),
    ("generations", "closeness_so_far", "Closest to optimum so far\n(optimal edges)", (0, 1.02)),
    ("generations", "closeness_mean", "Population average\n(optimal edges)", (0, 1.02)),
    ("generations", "gini", "Wealth concentration\n(Gini)", (0, 1.02)),
    ("generations", "progress_spread", "Best–worst gap\n(progress points)", None),
]

#: Per-run scores: (column, description, which direction is better; None = neither).
SCORES = [
    ("best_progress", "best tour found, progress at the end", "higher"),
    ("average_progress", "population average progress at the end", "higher"),
    ("closest_to_optimum", "most optimal edges any individual ever had", "higher"),
    ("average_closeness", "population average optimal edges at the end", "higher"),
    ("gini", "wealth concentration (Gini), averaged over the run", None),
    ("spread", "best–worst gap in progress points, averaged over the run", None),
]


def add_progress(df: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    """Add the progress columns: (random - value) / (random - reference)."""
    df = df.copy()
    ref = df["problem"].map(reference)
    scale = df["random_value"] - ref

    def progress(value: pd.Series) -> pd.Series:
        return (df["random_value"] - value) / scale

    df["progress"] = progress(df["best_so_far"])
    if "pop_mean" in df:
        df["progress_mean"] = progress(df["pop_mean"])
        df["progress_spread"] = (df["pop_worst"] - df["pop_best"]) / scale
        runs = df.sort_values("evaluations").groupby(stats.RUN_KEYS)["closeness_best"]
        df["closeness_so_far"] = runs.cummax()
    return df


def scores(generations: pd.DataFrame, budget: int) -> pd.DataFrame:
    """One row per run with the ``SCORES`` columns."""
    final = stats.final_rows(generations)
    out = final[stats.RUN_KEYS].copy()
    out["best_progress"] = final["progress"].to_numpy()
    out["average_progress"] = final["progress_mean"].to_numpy()
    out["closest_to_optimum"] = final["closeness_so_far"].to_numpy()
    out["average_closeness"] = final["closeness_mean"].to_numpy()
    grid = np.linspace(budget / 100, budget, 100).round().astype(int)
    for column, name in (("gini", "gini"), ("progress_spread", "spread")):
        g = stats.on_grid(generations, column, grid)
        mean = g.groupby(stats.RUN_KEYS, as_index=False)["value"].mean()
        out = out.merge(mean.rename(columns={"value": name}), on=stats.RUN_KEYS)
    return out


def compare(runs: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Every score of every variant against the baseline, with a verdict."""
    tables = []
    for column, _, better in SCORES:
        if runs[column].isna().all():  # e.g. no known optimal tour
            continue
        t = stats.compare_to_baseline(runs, baseline, column, higher=better or "neutral")
        t.insert(2, "score", column)
        tables.append(t)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
