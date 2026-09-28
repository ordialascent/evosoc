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
        # Social metrics (docs/metrics.md, "Social analysis").
        "gini": gini(np.maximum(problem.random_value - f, 0.0)),
        **quarter_means(f),
    }


def quarter_means(fitness: np.ndarray) -> dict[str, float]:
    """Mean fitness of each quarter of the population, from the best quarter
    (``quarter1_mean``) to the worst (``quarter4_mean``)."""
    if fitness.size < 4:
        return {f"quarter{k}_mean": np.nan for k in range(1, 5)}
    quarters = np.array_split(np.sort(fitness), 4)
    return {f"quarter{k}_mean": float(q.mean()) for k, q in enumerate(quarters, 1)}


def gini(wealth: np.ndarray) -> float:
    """Gini coefficient of non-negative amounts: 0 = shared equally,
    towards 1 = held by a single individual."""
    total = wealth.sum()
    if total <= 0:
        return 0.0
    x = np.sort(wealth)
    n = x.size
    return float(2 * np.dot(np.arange(1, n + 1), x) / (n * total) - (n + 1) / n)


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
    return float(np.mean(problem.distances([g[a] for a in i], [g[b] for b in j])))
