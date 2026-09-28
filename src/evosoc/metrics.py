"""Per-generation measurements of a population.

These are the *online* metrics, recorded while a run is in progress. Metrics
over whole runs (gap to optimum, hitting times, ERT, anytime AUC) are derived
afterwards in :mod:`evosoc.analysis`. See ``docs/metrics.md`` for why each
one is tracked.
"""

from __future__ import annotations

import numpy as np

from .algorithms.base import Population
from .problems import Problem


def snapshot(
    problem: Problem,
    pop: Population,
    rng: np.random.Generator,
    max_pairs: int = 200,
) -> dict[str, float]:
    f = pop.fitness
    return {
        "pop_best": float(f.min()),
        "pop_mean": float(f.mean()),
        "pop_median": float(np.median(f)),
        "pop_worst": float(f.max()),
        "pop_std": float(f.std()),
        "pop_size": len(pop),
        "diversity": mean_pairwise_distance(problem, pop, rng, max_pairs),
        "unique_fraction": len({problem.canonical(g) for g in pop.genomes}) / len(pop),
    }


def mean_pairwise_distance(
    problem: Problem, pop: Population, rng: np.random.Generator, max_pairs: int
) -> float:
    """Mean genotypic distance between individuals, estimated from sampled pairs.

    ``rng`` must not be the algorithm's generator, so that measuring a run
    never changes its outcome.
    """
    n = len(pop)
    if n < 2:
        return 0.0
    i = rng.integers(0, n, size=max_pairs)
    j = (i + rng.integers(1, n, size=max_pairs)) % n  # j != i
    g = pop.genomes
    return float(np.mean([problem.distance(g[a], g[b]) for a, b in zip(i, j, strict=True)]))
