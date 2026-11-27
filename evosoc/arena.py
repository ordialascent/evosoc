"""The arena: a hidden map, a measuring device and a budget.

A contestant never sees the cities or the distances between them. The only
way to learn anything about the map is `measure` (one round trip) or
`measure_many` (several round trips), and every measured trip costs one unit
of budget, whoever asks and for whatever reason. When the budget is spent,
the next measurement raises `BudgetSpent` and the run is over. The result of
a run is the shortest trip measured.

Season 1 (Open Books): every measurement is public, so the arena also tells
anyone the best trip measured so far.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .operators import inversion_mutation, order_crossover
from .tsp import distance_matrix, make_cities


class BudgetSpent(Exception):
    """Raised when a measurement is asked for after the budget is used up."""


@dataclass(frozen=True)
class Toolbox:
    """The ways to make and change a trip, the same for every contestant (rules §1).

    Frozen with Baseline v1 for the season. Contestants decide who uses
    these tools, when and how often; they may not add new ones.
    """

    population: int
    crossover: Callable[[np.random.Generator, np.ndarray, np.ndarray], np.ndarray]
    mutation: Callable[[np.random.Generator, np.ndarray], np.ndarray]

    def random_tour(self, rng: np.random.Generator, cities: int) -> np.ndarray:
        return rng.permutation(cities)

    def describe(self) -> dict:
        return {"population": self.population, "crossover": self.crossover.__name__,
                "mutation": self.mutation.__name__}


# Provisional until Baseline v1 is tuned and frozen; then this is the
# season-1 toolbox and never changes during the season.
TOOLBOX = Toolbox(population=100, crossover=order_crossover, mutation=inversion_mutation)

CURVE_POINTS = 20


def curve_marks(budget: int, points: int = CURVE_POINTS) -> np.ndarray:
    """Budget levels (measurements used) at which the best trip so far is recorded."""
    marks = np.unique(np.round(np.geomspace(min(100, budget), budget, points)).astype(int))
    return marks


def _meter(dist: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """A device that measures round trips on a map it keeps to itself."""

    def lengths(tours: np.ndarray) -> np.ndarray:
        return dist[tours, np.roll(tours, -1, axis=1)].sum(axis=1)

    return lengths


class Arena:
    """One run: one hidden map, one budget, one seed."""

    def __init__(self, cities: int, budget: int, seed: int, toolbox: Toolbox = TOOLBOX):
        self.cities = cities
        self.budget = budget
        self.seed = seed
        self.toolbox = toolbox
        # The contestant's own randomness: fixed by the seed, separate from the map's.
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.used = 0
        self.best_length = float("inf")
        self.best_tour: np.ndarray | None = None
        self._lengths = _meter(distance_matrix(make_cities(cities, seed)))
        self._marks = curve_marks(budget)
        self._curve: list[float] = []

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    def measure(self, tour) -> float:
        """Length of one round trip. Costs 1 unit of budget."""
        return float(self.measure_many(np.asarray(tour)[None, :])[0])

    def measure_many(self, tours) -> np.ndarray:
        """Lengths of several round trips (shape (count, cities)). Costs 1 unit each.

        If the budget runs out part way, the trips that still fit are
        measured (and count), then `BudgetSpent` is raised.
        """
        tours = np.asarray(tours)
        if tours.ndim != 2 or tours.shape[1] != self.cities:
            raise ValueError(f"tours must have shape (count, {self.cities})")
        if not (np.sort(tours, axis=1) == np.arange(self.cities)).all():
            raise ValueError("every tour must visit each city exactly once")
        if self.remaining <= 0:
            raise BudgetSpent
        fits = tours[: self.remaining]
        lengths = self._lengths(fits)
        self._record(fits, lengths)
        if len(fits) < len(tours):
            raise BudgetSpent
        return lengths

    def _record(self, tours: np.ndarray, lengths: np.ndarray) -> None:
        start = self.used
        self.used += len(tours)
        # Best so far after each measurement, to fill in the curve marks passed.
        running = np.minimum.accumulate(np.r_[self.best_length, lengths])[1:]
        k = int(np.argmin(lengths))
        if lengths[k] < self.best_length:
            self.best_length = float(lengths[k])
            self.best_tour = tours[k].copy()
        while len(self._curve) < len(self._marks) and self._marks[len(self._curve)] <= self.used:
            self._curve.append(float(running[self._marks[len(self._curve)] - start - 1]))

    def curve(self) -> list[list[float]]:
        """[[measurements used, best trip so far], ...] at the curve marks reached."""
        return [[int(m), v] for m, v in zip(self._marks, self._curve)]
