"""What an algorithm sees of the experiment, and what it must provide.

An algorithm never counts evaluations, times itself or writes results. It
evaluates genomes through :class:`Context` and calls :meth:`Context.report`
once per generation. When the evaluation budget is spent,
``Context.evaluate`` raises :class:`BudgetExhausted`, which ends the run.

Keeping all bookkeeping here guarantees every variant is measured the same
way and gets exactly the same budget, whatever its internal structure.
"""

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..problems import Problem


class BudgetExhausted(Exception):
    """Raised by :meth:`Context.evaluate` once the evaluation budget is spent."""


@dataclass
class Population:
    genomes: list[Any]
    #: Objective values, aligned with ``genomes`` (lower is better).
    fitness: np.ndarray

    def __len__(self) -> int:
        return len(self.genomes)


class Context:
    def __init__(
        self,
        problem: Problem,
        budget: int,
        rng: np.random.Generator,
        init_rng: np.random.Generator,
        on_report: Callable[[Context, Population, dict[str, float]], None],
    ) -> None:
        self.problem = problem
        self.budget = budget
        #: The algorithm's own source of randomness.
        self.rng = rng
        self._init_rng = init_rng
        self._on_report = on_report
        self.evaluations = 0
        self.generation = 0
        self.best_so_far = np.inf
        self.best_genome: Any = None
        #: (evaluation number, new best value) at every improvement: the exact
        #: best-so-far curve, independent of how evaluations are grouped.
        self.improvements: list[tuple[int, float]] = []
        self.start_time = time.perf_counter()

    def evaluate(self, genome: Any) -> float:
        if self.evaluations >= self.budget:
            raise BudgetExhausted
        self.evaluations += 1
        value = self.problem.evaluate(genome)
        if value < self.best_so_far:
            self.best_so_far = value
            self.best_genome = copy.deepcopy(genome)  # algorithms may mutate in place
            self.improvements.append((self.evaluations, value))
        return value

    def evaluate_all(self, genomes: Sequence[Any]) -> np.ndarray:
        return np.array([self.evaluate(g) for g in genomes])

    def initial_population(self, size: int) -> Population:
        """Random initial population.

        Drawn from a generator that depends only on (problem, seed), so every
        variant starts seed ``k`` from the same individuals (common random
        numbers), which makes paired comparisons between variants possible.
        """
        genomes = [self.problem.random_genome(self._init_rng) for _ in range(size)]
        return Population(genomes, self.evaluate_all(genomes))

    def report(self, population: Population, **extra: float) -> None:
        """Record one snapshot. Call once per generation.

        ``extra`` holds variant-specific measurements; they are stored as
        ``x_<name>`` columns next to the standard metrics.
        """
        self.generation += 1
        self._on_report(self, population, extra)


class Algorithm(ABC):
    #: Registry name, set by :func:`evosoc.algorithms.register`.
    name: str = ""

    @abstractmethod
    def run(self, ctx: Context) -> None:
        """Evolve until ``ctx.evaluate`` raises :class:`BudgetExhausted`."""
