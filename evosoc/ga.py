"""The baseline: a standard, textbook genetic algorithm.

Each generation:
  1. Keep the `elite` best trips unchanged.
  2. Fill the rest of the next generation with children. For each child,
     two parents are chosen by tournament, combined by crossover (order
     crossover unless `crossover` says otherwise; with probability
     `crossover_rate`, otherwise the child copies the first parent) and
     then mutated by reversing a stretch of the trip
     (with probability `mutation_rate`).

Nobody inside the population decides anything and nothing is owned: the
rules are fixed before the run and reward is current trip length only.
Later variants change exactly these social rules and nothing else.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .operators import edge_crossover, inversion_mutation, order_crossover, tournament
from .tsp import tour_lengths


@dataclass(frozen=True)
class Config:
    population: int = 100
    generations: int = 1000
    tournament_size: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.3
    elite: int = 2  # members kept unchanged each generation (the "rulers")
    # How the rulers are chosen each generation:
    #   "merit":      the fittest `elite` members (the textbook rule)
    #   "lottery":    `elite` members drawn at random, whatever their trip
    #   "hereditary": each ruler's seat passes to one of its own children,
    #                 whatever that child's trip; the first rulers are the
    #                 fittest of the random starting population
    rulers_chosen: str = "merit"
    # How two parents are combined into a child:
    #   "order": a stretch of one parent, the rest in the other's order (textbook)
    #   "edge":  links between cities taken from either parent (gentler)
    crossover: str = "order"

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
    population: np.ndarray  # every member's trip in the final generation


CROSSOVERS = {"order": order_crossover, "edge": edge_crossover}

RULER_CHOICES = ("merit", "lottery", "hereditary")


def kept_per_generation(config: Config) -> int:
    """Members carried over unchanged, and so not measured again, each generation."""
    return 0 if config.rulers_chosen == "hereditary" else config.elite


def generations_for_budget(population: int, kept: int, budget: int) -> int:
    """Generations that measure about `budget` trips in total.

    Kept members are not measured again, so a population that keeps more
    members makes fewer new trips per generation and gets more generations.
    """
    return max(1, round((budget - population) / (population - kept)))


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
    if config.rulers_chosen not in RULER_CHOICES:
        raise ValueError(f"rulers_chosen must be one of {RULER_CHOICES}")
    if config.crossover not in CROSSOVERS:
        raise ValueError(f"crossover must be one of {tuple(CROSSOVERS)}")
    crossover = CROSSOVERS[config.crossover]
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

    def make_child(mother: np.ndarray) -> np.ndarray:
        father = pop[tournament(rng, lengths, config.tournament_size)]
        if rng.random() < config.crossover_rate:
            child = crossover(rng, mother, father)
        else:
            child = mother.copy()
        if rng.random() < config.mutation_rate:
            child = inversion_mutation(rng, child)
        return child

    kept = kept_per_generation(config)
    # Hereditary rulers sit in the first `elite` places of the population.
    if config.rulers_chosen == "hereditary":
        first = np.argsort(lengths, kind="stable")[: config.elite]
        rest = np.setdiff1d(np.arange(config.population), first)
        pop, lengths = pop[np.r_[first, rest]], lengths[np.r_[first, rest]]

    record()
    for _ in range(config.generations):
        if config.rulers_chosen == "merit":
            order = np.argsort(lengths, kind="stable")
            children = [pop[k].copy() for k in order[: config.elite]]
        elif config.rulers_chosen == "lottery":
            seats = rng.choice(config.population, size=config.elite, replace=False)
            children = [pop[k].copy() for k in seats]
        else:  # hereditary: each heir is a new child of a ruler, and takes the seat
            children = [make_child(pop[k]) for k in range(config.elite)]
        while len(children) < config.population:
            children.append(make_child(pop[tournament(rng, lengths, config.tournament_size)]))
        pop = np.array(children)
        lengths = tour_lengths(dist, pop)
        evaluations += config.population - kept
        record()

    return History(
        best=np.array(best),
        mean=np.array(mean),
        worst=np.array(worst),
        diversity=np.array(diversity),
        measured=np.array(measured),
        best_tour=pop[int(np.argmin(lengths))].copy(),
        evaluations=evaluations,
        population=pop.copy(),
    )
