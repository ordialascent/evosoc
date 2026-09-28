"""Pure random search.

Not a contender: a floor every real algorithm must clear by a wide margin,
and a way to exercise the harness end to end before the GAs exist.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import register
from .base import Algorithm, Context, Population


@register("random_search")
@dataclass
class RandomSearch(Algorithm):
    #: Genomes sampled per "generation" (one report per batch).
    batch: int = 100

    def run(self, ctx: Context) -> None:
        pop = ctx.initial_population(self.batch)
        while True:
            ctx.report(pop)
            genomes = [ctx.problem.random_genome(ctx.rng) for _ in range(self.batch)]
            pop = Population(genomes, ctx.evaluate_all(genomes))
