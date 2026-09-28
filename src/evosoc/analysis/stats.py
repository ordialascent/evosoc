"""Run-level metrics and statistical comparison against the baseline.

Conventions (following common practice in EC benchmarking, e.g. COCO and
IOHprofiler, and Arcuri & Briand 2011 on comparing randomised algorithms):

* Every run gets the same evaluation budget; the x-axis is evaluations,
  never generations, because variants may use evaluations differently.
* Quality is reported as the relative gap to the optimum (or to the best
  value any run found, when the optimum is unknown).
* Distributions are summarised by median and IQR, not mean +- std: GA
  outcomes are usually skewed and bounded.
* Differences are tested non-parametrically (Mann-Whitney U; Wilcoxon
  signed-rank when runs are paired by seed), with the Vargha-Delaney A12
  effect size and Holm correction for the number of comparisons.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

RUN_KEYS = ["variant", "problem", "seed"]


def references(improvements: pd.DataFrame) -> pd.Series:
    """Per problem: the known optimum, or else the best value any run found."""
    by_problem = improvements.groupby("problem")
    return by_problem["optimum"].first().fillna(by_problem["best_so_far"].min())


def add_gap(df: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    """Add relative gaps to the reference: ``gap`` for best-so-far, and
    ``pop_mean_gap`` when the table has population metrics."""
    df = df.copy()
    ref = df["problem"].map(reference)
    df["gap"] = (df["best_so_far"] - ref) / ref
    if "pop_mean" in df:
        df["pop_mean_gap"] = (df["pop_mean"] - ref) / ref
    return df


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The last recorded row of every run."""
    return df.sort_values("evaluations").groupby(RUN_KEYS, as_index=False).last()


def on_grid(df: pd.DataFrame, column: str, grid: np.ndarray) -> pd.DataFrame:
    """Resample ``column`` of every run onto common evaluation counts.

    Uses the last value recorded at or before each grid point (NaN before a
    run's first report), so runs with different generation sizes line up.
    """
    parts = []
    for key, run in df.sort_values("evaluations").groupby(RUN_KEYS):
        idx = np.searchsorted(run["evaluations"].to_numpy(), grid, side="right") - 1
        values = np.where(idx >= 0, run[column].to_numpy()[np.maximum(idx, 0)], np.nan)
        parts.append(
            pd.DataFrame(dict(zip(RUN_KEYS, key, strict=True), evaluations=grid, value=values))
        )
    return pd.concat(parts, ignore_index=True)


def eval_grid(budget: int, points: int = 200) -> np.ndarray:
    return np.unique(np.geomspace(1, budget, points).round().astype(int))


def hitting_times(
    improvements: pd.DataFrame, targets: Sequence[float], budget: int
) -> pd.DataFrame:
    """Evaluations each run needed to reach each target gap (NaN = never).

    Uses the improvement trace, so times are exact to the evaluation.
    """
    rows = []
    for key, run in improvements.sort_values("evaluations").groupby(RUN_KEYS):
        gap = run["gap"].to_numpy()
        evals = run["evaluations"].to_numpy()
        for t in targets:
            hit = np.flatnonzero(gap <= t)
            rows.append(
                {
                    **dict(zip(RUN_KEYS, key, strict=True)),
                    "target": t,
                    "hit_evaluations": evals[hit[0]] if len(hit) else np.nan,
                    "evaluations_used": budget,
                }
            )
    return pd.DataFrame(rows)


def ert(hits: pd.DataFrame) -> pd.DataFrame:
    """Expected running time per target (Hansen et al., COCO).

    Total evaluations spent over all runs (up to the hit, or the whole
    budget for failed runs) divided by the number of successful runs.
    """

    def one(g: pd.DataFrame) -> pd.Series:
        ok = g["hit_evaluations"].notna()
        spent = g["hit_evaluations"].where(ok, g["evaluations_used"]).sum()
        return pd.Series(
            {
                "success_rate": ok.mean(),
                "ert": spent / ok.sum() if ok.any() else np.inf,
            }
        )

    return hits.groupby(["variant", "problem", "target"]).apply(one, include_groups=False)


def anytime_gap(improvements: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Mean best-so-far gap over a log-spaced evaluation grid, per run.

    A single number for the whole convergence curve (area under it, on a
    log-evaluation axis): rewards being good early *and* late. Lower is better.
    Computed from the improvement trace, whose first entry is evaluation 1,
    so every run is defined on the whole grid whatever its generation size.
    """
    g = on_grid(improvements, "gap", eval_grid(budget))
    return (
        g.groupby(RUN_KEYS, as_index=False)["value"].mean().rename(columns={"value": "anytime_gap"})
    )


def a12(x: np.ndarray, y: np.ndarray) -> float:
    """Vargha-Delaney A12 for minimisation: P(x < y) + 0.5 P(x == y).

    0.5 = no difference; > 0.5 = ``x`` tends to be better (lower) than ``y``.
    Conventional thresholds on |A12 - 0.5|: 0.06 small, 0.14 medium, 0.21 large.
    """
    x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
    return float(((x < y).sum() + 0.5 * (x == y).sum()) / (x.size * y.size))


def compare_to_baseline(runs: pd.DataFrame, baseline: str, metric: str) -> pd.DataFrame:
    """One row per (problem, variant != baseline): medians, tests, effect size.

    ``runs`` holds one row per run (``RUN_KEYS`` + ``metric``); lower is better.
    """
    rows = []
    for problem, p in runs.groupby("problem"):
        base = p[p["variant"] == baseline].set_index("seed")[metric]
        for variant, v in p[p["variant"] != baseline].groupby("variant"):
            v = v.set_index("seed")[metric]
            row = {
                "problem": problem,
                "variant": variant,
                "n": len(v),
                "median": v.median(),
                "iqr": v.quantile(0.75) - v.quantile(0.25),
                "baseline_median": base.median(),
                "baseline_iqr": base.quantile(0.75) - base.quantile(0.25),
                "a12": a12(v.to_numpy(), base.to_numpy()),
                "p_mannwhitney": _safe_p(stats.mannwhitneyu, v, base),
            }
            common = v.index.intersection(base.index)
            if len(common) >= 5:
                row["p_wilcoxon_paired"] = _safe_p(stats.wilcoxon, v[common], base[common])
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p_mannwhitney"].to_numpy())
    return out


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def _safe_p(test, x, y) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # degenerate samples
            p = float(test(x, y).pvalue)
    except ValueError:  # e.g. all differences zero
        return 1.0
    return 1.0 if np.isnan(p) else p
