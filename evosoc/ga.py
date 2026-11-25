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

Ownership (`mothers="wealth"`): each member earns from its trip every
generation (by rank: the best earns twice the average, the worst nothing).
A share `shared` of all earnings goes into a common pool that is split
evenly. Wealth buys children: every child's mother is drawn in proportion
to wealth. When a member is replaced, a share `inherited` of its wealth is
split evenly between its heirs (its children, plus itself if it is kept);
the rest is used up. Partners (fathers) are still chosen by tournament.
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
    # Members kept unchanged each generation: the protected class (called
    # "rulers" in older names). They decide nothing for anyone else; they
    # only survive, and so can be picked as parents again.
    elite: int = 2
    # How the protected members are chosen each generation:
    #   "merit":      the fittest `elite` members (the textbook rule)
    #   "lottery":    `elite` members drawn at random, whatever their trip
    #   "hereditary": each protected seat passes to one of the holder's own
    #                 children, whatever that child's trip; the first holders are the
    #                 fittest of the random starting population
    rulers_chosen: str = "merit"
    # How two parents are combined into a child:
    #   "order": a stretch of one parent, the rest in the other's order (textbook)
    #   "edge":  links between cities taken from either parent (gentler)
    crossover: str = "order"
    # Who gets to have children (as mother):
    #   "tournament": the winner of a small contest (the textbook rule)
    #   "wealth":     drawn in proportion to wealth; see `shared` and `inherited`
    mothers: str = "tournament"
    shared: float = 0.0  # share of all earnings pooled and split evenly
    inherited: float = 0.0  # share of a member's wealth passed on to its heirs

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
    # One entry per new generation: how unequally the children were spread
    # over the members of the generation before (Gini: 0 = everyone had the
    # same number of children, near 1 = one member had them all).
    parent_gini: np.ndarray
    # One entry per new generation: how unequal wealth was when the mothers
    # were drawn (Gini; all 0 without ownership).
    wealth_gini: np.ndarray


CROSSOVERS = {"order": order_crossover, "edge": edge_crossover}

RULER_CHOICES = ("merit", "lottery", "hereditary")

MOTHER_CHOICES = ("tournament", "wealth")


def gini(values) -> float:
    """Inequality of non-negative amounts: 0 if everyone has the same, near 1 if one has all."""
    v = np.sort(np.asarray(values, dtype=float))
    total = v.sum()
    if total <= 0:
        return 0.0
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1) @ v / (n * total))


def earnings(lengths: np.ndarray) -> np.ndarray:
    """What each member earns from its trip this generation.

    Earnings fall in a straight line with rank: the shortest trip earns 2,
    the longest earns 0, the average member earns 1. Ties are broken by
    position, so the total is always the population size.
    """
    n = len(lengths)
    rank = np.empty(n)
    rank[np.argsort(lengths, kind="stable")] = np.arange(n)
    return 2 * (n - 1 - rank) / max(n - 1, 1)


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
    if config.mothers not in MOTHER_CHOICES:
        raise ValueError(f"mothers must be one of {MOTHER_CHOICES}")
    if not (0 <= config.shared <= 1 and 0 <= config.inherited <= 1):
        raise ValueError("shared and inherited must be between 0 and 1")
    by_wealth = config.mothers == "wealth"
    crossover = CROSSOVERS[config.crossover]
    rng = np.random.default_rng(seed)
    n = len(dist)
    pop = np.array([rng.permutation(n) for _ in range(config.population)])
    lengths = tour_lengths(dist, pop)
    evaluations = len(pop)

    best, mean, worst, diversity, measured = [], [], [], [], []
    parent_gini, wealth_gini = [], []
    # Wealth each member was born with; the first generation starts with nothing.
    estate = np.zeros(config.population)

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
    # Hereditary protected seats are the first `elite` places of the population.
    if config.rulers_chosen == "hereditary":
        first = np.argsort(lengths, kind="stable")[: config.elite]
        rest = np.setdiff1d(np.arange(config.population), first)
        pop, lengths = pop[np.r_[first, rest]], lengths[np.r_[first, rest]]

    record()
    for _ in range(config.generations):
        if by_wealth:
            wealth = estate + (1 - config.shared) * earnings(lengths) + config.shared
        # `source[j]` is the member of this generation that new member j comes
        # from: its mother, or itself if it is kept unchanged.
        if config.rulers_chosen == "merit":
            source = np.argsort(lengths, kind="stable")[: config.elite].tolist()
            children = [pop[k].copy() for k in source]
        elif config.rulers_chosen == "lottery":
            source = rng.choice(config.population, size=config.elite, replace=False).tolist()
            children = [pop[k].copy() for k in source]
        else:  # hereditary: each heir is a new child of a seat holder, and takes the seat
            source = list(range(config.elite))
            children = [make_child(pop[k]) for k in source]
        copies = len(children) if config.rulers_chosen != "hereditary" else 0
        if by_wealth:
            mothers = rng.choice(config.population, size=config.population - len(children),
                                 p=wealth / wealth.sum())
            for k in mothers.tolist():
                children.append(make_child(pop[k]))
                source.append(k)
        while len(children) < config.population:
            k = tournament(rng, lengths, config.tournament_size)
            children.append(make_child(pop[k]))
            source.append(k)
        source = np.array(source)
        parent_gini.append(gini(np.bincount(source[copies:], minlength=config.population)))
        if by_wealth:
            wealth_gini.append(gini(wealth))
            heirs = np.bincount(source, minlength=config.population)
            estate = config.inherited * wealth[source] / heirs[source]
        else:
            wealth_gini.append(0.0)
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
        parent_gini=np.array(parent_gini),
        wealth_gini=np.array(wealth_gini),
    )
