"""The baseline: a steady-state GA with overlapping generations.

Design rationale in ``docs/baseline-ga.md``. There are no discrete
generations: individuals are born and die one at a time, as in most animal
populations. Each birth:

1. **Mate choice**: two different parents, each the fitter of two individuals
   that happen to meet (binary tournament).
2. **Reproduction**: with probability ``crossover_rate``, order crossover (OX);
   otherwise a clone of the first parent.
3. **Mutation**: with probability ``mutation_rate``, one inversion.
4. **Death**: the newborn joins the population, and the less fit of two
   individuals that happen to meet dies (reverse binary tournament), keeping
   the population at its carrying capacity ``pop_size``.

A side effect of (4) is that the current best individual can never die,
since it can never be the less fit of two different individuals.

Requires permutation genomes (TSP tours). One ``report`` covers
``pop_size`` births, a "generation equivalent", for the per-generation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..operators.permutation import inversion, order_crossover
from . import register
from .base import Algorithm, Context, Population


@register("steady_state_ga")
@dataclass
class SteadyStateGA(Algorithm):
    pop_size: int = 200
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2

    def __post_init__(self) -> None:
        if self.pop_size < 2:
            raise ValueError("pop_size must be at least 2 (reproduction needs two parents)")
        for name in ("crossover_rate", "mutation_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def run(self, ctx: Context) -> None:
        rng = ctx.rng
        if ctx.budget < 2 * self.pop_size:
            raise ValueError(
                f"budget {ctx.budget} is too small for pop_size {self.pop_size}: it must cover "
                "the initial population and at least one generation's worth of births"
            )
        pop = ctx.initial_population(self.pop_size)
        genomes = np.array(pop.genomes, dtype=np.int64)  # (pop_size, n)
        fitness = pop.fitness.copy()
        n = genomes.shape[1]
        if n < 4:
            raise ValueError(f"steady_state_ga needs tours of at least 4 cities, got {n}")

        while True:
            ctx.report(Population(list(genomes), fitness.copy()))
            for _ in range(self.pop_size):
                mother = self._contest(fitness, rng, winner=True)
                father = mother
                while father == mother:
                    father = self._contest(fitness, rng, winner=True)

                if rng.random() < self.crossover_rate:
                    i, j = np.sort(rng.choice(n + 1, size=2, replace=False))
                    child = order_crossover(genomes[mother], genomes[father], i, j)
                else:
                    child = genomes[mother].copy()
                if rng.random() < self.mutation_rate:
                    inversion(child, rng)

                f = ctx.evaluate(child)
                dead = self._contest(fitness, rng, winner=False)
                genomes[dead] = child
                fitness[dead] = f

    @staticmethod
    def _contest(fitness: np.ndarray, rng: np.random.Generator, winner: bool) -> int:
        """Two different individuals meet at random: index of the fitter
        (``winner``) or the less fit one."""
        a = rng.integers(len(fitness))
        b = rng.integers(len(fitness) - 1)
        b += b >= a  # distinct from a
        better = a if fitness[a] <= fitness[b] else b
        return int(better if winner else a + b - better)
