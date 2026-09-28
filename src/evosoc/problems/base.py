"""The interface every optimisation problem implements.

The harness (runner, metrics, plots) only talks to problems through this
protocol, so swapping TSP for another problem later needs no harness changes.
All problems are *minimisation* problems.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Problem(Protocol):
    name: str
    #: Known optimal objective value, or None if unknown.
    optimum: float | None

    def evaluate(self, genome: Any) -> float:
        """Objective value of a genome (lower is better)."""
        ...

    def random_genome(self, rng: np.random.Generator) -> Any:
        """A uniformly random valid genome."""
        ...

    def distance(self, a: Any, b: Any) -> float:
        """Genotypic distance in [0, 1], used for population-diversity metrics."""
        ...

    def canonical(self, genome: Any) -> Hashable:
        """A key that is equal for genomes that encode the same solution."""
        ...
