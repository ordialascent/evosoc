"""The baseline: a standard, textbook genetic algorithm.

Each generation:
  1. Keep the `elite` best trips unchanged.
  2. Fill the rest of the next generation with children. For each child,
     two parents are chosen by tournament, combined by order crossover
     (with probability `crossover_rate`, otherwise the child copies the
     first parent) and then mutated by reversing a stretch of the trip
     (with probability `mutation_rate`).

Nobody inside the population decides anything and nothing is owned: the
rules are fixed before the run and reward is current trip length only.
Later variants change exactly these social rules and nothing else.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .operators import inversion_mutation, order_crossover, tournament
from .tsp import tour_lengths


@dataclass(frozen=True)
class Config:
    population: int = 100
    generations: int = 1000
    tournament_size: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.3
    elite: int = 2  # members kept unchanged each generation (the "rulers")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class History:
    best: np.ndarray  # shortest trip in each generation
    mean: np.ndarray  # average trip in each generation
    worst: np.ndarray  # longest trip in each generation
    diversity: np.ndarray  # how different the average member is from the best one
    measured: np.ndarray  # trips measured so far, at the end of each generation
    best_tour: np.ndarray  # shortest trip found in the final generation
    evaluations: int  # trips measured in total


def generations_for_budget(population: int, elite: int, budget: int) -> int:
    """Generations that measure about `budget` trips in total.

    Kept members are not measured again, so a population that keeps more
    members makes fewer new trips per generation and gets more generations.
    """
    return max(1, round((budget - population) / (population - elite)))


def difference_from_best(pop: np.ndarray, best: np.ndarray) -> float:
    """Share of an average member's links between cities that the best trip does not use.

    0 means everyone travels the same route as the best member; values near
    1 mean their routes have almost nothing in common.
    """
    n = len(best)
    uses = np.zeros((n, n), dtype=bool)
    nxt = np.roll(best, -1)
    uses[best, nxt] = uses[nxt, best] = True
    shared = uses[pop, np.roll(pop, -1, axis=1)].sum(axis=1)
    return float(1 - shared.mean() / n)


def run(dist: np.ndarray, config: Config, seed: int) -> History:
    """Evolve round trips over the cities in `dist`. Same seed, same result."""
    rng = np.random.default_rng(seed)
    n = len(dist)
    pop = np.array([rng.permutation(n) for _ in range(config.population)])
    lengths = tour_lengths(dist, pop)
    evaluations = len(pop)

    best, mean, worst, diversity, measured = [], [], [], [], []

    def record() -> None:
        best.append(lengths.min())
        mean.append(lengths.mean())
        worst.append(lengths.max())
        diversity.append(difference_from_best(pop, pop[int(np.argmin(lengths))]))
        measured.append(evaluations)

    record()
    for _ in range(config.generations):
        order = np.argsort(lengths, kind="stable")
        children = [pop[k].copy() for k in order[: config.elite]]
        while len(children) < config.population:
            mother = pop[tournament(rng, lengths, config.tournament_size)]
            father = pop[tournament(rng, lengths, config.tournament_size)]
            if rng.random() < config.crossover_rate:
                child = order_crossover(rng, mother, father)
            else:
                child = mother.copy()
            if rng.random() < config.mutation_rate:
                child = inversion_mutation(rng, child)
            children.append(child)
        pop = np.array(children)
        lengths = tour_lengths(dist, pop)
        evaluations += config.population - config.elite
        record()

    return History(
        best=np.array(best),
        mean=np.array(mean),
        worst=np.array(worst),
        diversity=np.array(diversity),
        measured=np.array(measured),
        best_tour=pop[int(np.argmin(lengths))].copy(),
        evaluations=evaluations,
    )
