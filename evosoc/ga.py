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

Central planning (`parents="planned"`): members no longer win their own
small contests. A planning board looks at the whole population at once and
decides who has children and with whom: it ranks everyone and hands out
children by rank, with the same shares a contest of 3 gives on average but
without the contests' luck, and pairs the parents (at random, or each
mother with the most different of 3 candidates, to keep variety). It also picks the
protected members. The board's strength is that it sees everyone; its
weakness is that it does not see true trip lengths, only lengths measured
on its own simplified map (`planners_view`: the cities snapped to a grid of
that many cells per side; 0 = the true map). Members still know their own
true trip, which is what the contests in the other societies use.

Capital (`returns`, `heir`): inherited wealth can earn a return, growing by
the share `returns` each generation before it is used, and the estate can go
to a single heir instead of being split: the child with the shortest trip
(`heir="fittest"`) or the first child born (`heir="first"`; a kept member
counts as its own first child).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

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
    # Who decides who has children:
    #   "tournament": each child's parents win small contests of true trip length
    #   "planned":    a planning board ranks everyone on its own map and
    #                 decides; see `planned_pairs` and `planners_view`
    parents: str = "tournament"
    # How the board pairs parents: "random", or "different" (each mother gets
    # the most different of 3 candidates drawn by the board's shares).
    planned_pairs: str = "random"
    # The board's map: cities snapped to a grid of this many cells per side.
    # 0 = the board sees the true map. Coarser grids hide more local detail.
    planners_view: int = 0
    returns: float = 0.0  # growth of inherited wealth per generation (0.5 = +50%)
    # Who inherits: "split" evenly between all heirs, or everything to one
    # heir: the "fittest" child (shortest trip) or the "first" child born.
    heir: str = "split"

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
    # One entry per new generation: share of the children whose mother's trip
    # was in the better half of the generation before (0.5 = no link between
    # trip and children; 0.875 for a contest of 3).
    mothers_better_half: np.ndarray
    # One entry per new generation: share of all wealth owned by the richest
    # tenth when the mothers were drawn (0.1 without ownership: all equal).
    top_tenth_share: np.ndarray
    # One entry per new generation under central planning: the share of the
    # parents the board chose that truly were in the better half (empty
    # otherwise). Contests of 3 on the true map pick 87.5% from the better
    # half on average; a board that is misled by its map picks fewer.
    planners_right: np.ndarray = field(default_factory=lambda: np.empty(0))


CROSSOVERS = {"order": order_crossover, "edge": edge_crossover}

RULER_CHOICES = ("merit", "lottery", "hereditary")

MOTHER_CHOICES = ("tournament", "wealth")

HEIR_CHOICES = ("split", "fittest", "first")

PARENT_CHOICES = ("tournament", "planned")

PAIRING_CHOICES = ("random", "different")


def planners_map(cities: np.ndarray, grid: int) -> np.ndarray:
    """The planning board's distance table: every city moved to the centre of
    its cell in a grid of `grid` x `grid` cells (0 = the true map)."""
    from .tsp import distance_matrix

    if grid <= 0:
        return distance_matrix(cities)
    return distance_matrix((np.floor(cities * grid).clip(0, grid - 1) + 0.5) / grid)


def gini(values) -> float:
    """Inequality of non-negative amounts: 0 if everyone has the same, near 1 if one has all."""
    v = np.sort(np.asarray(values, dtype=float))
    total = v.sum()
    if total <= 0:
        return 0.0
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1) @ v / (n * total))


def top_share(values, top: float = 0.1) -> float:
    """Share of the total owned by the richest `top` part (at least one member)."""
    v = np.sort(np.asarray(values, dtype=float))[::-1]
    total = v.sum()
    if total <= 0:
        return top
    return float(v[: max(1, round(top * len(v)))].sum() / total)


def estates(passed: np.ndarray, source: np.ndarray, lengths: np.ndarray, heir: str) -> np.ndarray:
    """What each new member inherits.

    `passed[j]` is what the member that new member j comes from leaves
    behind, `source[j]` is that member, and `lengths` are the new members'
    trips. "split" divides each estate evenly between all heirs; "fittest"
    and "first" give it all to the heir with the shortest trip or the
    lowest place, and nothing to the others.
    """
    if heir == "split":
        heirs = np.bincount(source)
        return passed / heirs[source]
    order = (np.argsort(lengths, kind="stable") if heir == "fittest"
             else np.arange(len(source)))
    _, first = np.unique(source[order], return_index=True)
    chosen = order[first]
    out = np.zeros(len(source))
    out[chosen] = passed[chosen]
    return out


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


def run(dist: np.ndarray, config: Config, seed: int, view: np.ndarray | None = None) -> History:
    """Evolve round trips over the cities in `dist`. Same seed, same result.

    `view` is the planning board's distance table (see `planners_map`); it is
    only used when `config.parents == "planned"`, and defaults to the true map.
    """
    if config.rulers_chosen not in RULER_CHOICES:
        raise ValueError(f"rulers_chosen must be one of {RULER_CHOICES}")
    if config.crossover not in CROSSOVERS:
        raise ValueError(f"crossover must be one of {tuple(CROSSOVERS)}")
    if config.mothers not in MOTHER_CHOICES:
        raise ValueError(f"mothers must be one of {MOTHER_CHOICES}")
    if not (0 <= config.shared <= 1 and 0 <= config.inherited <= 1):
        raise ValueError("shared and inherited must be between 0 and 1")
    if config.parents not in PARENT_CHOICES:
        raise ValueError(f"parents must be one of {PARENT_CHOICES}")
    if config.planned_pairs not in PAIRING_CHOICES:
        raise ValueError(f"planned_pairs must be one of {PAIRING_CHOICES}")
    planned = config.parents == "planned"
    if planned and (config.mothers != "tournament" or config.rulers_chosen != "merit"):
        raise ValueError("central planning needs mothers='tournament' and rulers_chosen='merit'")
    if view is None:
        view = dist
    if not 0 <= config.returns <= 1:
        raise ValueError("returns must be between 0 and 1")
    if config.heir not in HEIR_CHOICES:
        raise ValueError(f"heir must be one of {HEIR_CHOICES}")
    by_wealth = config.mothers == "wealth"
    crossover = CROSSOVERS[config.crossover]
    rng = np.random.default_rng(seed)
    n = len(dist)
    pop = np.array([rng.permutation(n) for _ in range(config.population)])
    lengths = tour_lengths(dist, pop)
    evaluations = len(pop)

    best, mean, worst, diversity, measured = [], [], [], [], []
    parent_gini, wealth_gini, mothers_better_half, top_tenth_share = [], [], [], []
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

    planners_right = []

    # The board hands out children by rank with the same shares a contest of
    # `tournament_size` gives on average, so a board with a perfect map is as
    # demanding as the contests; it only removes their luck.
    ranks = np.arange(config.population)
    t = config.tournament_size
    shares = ((config.population - ranks) ** t - (config.population - ranks - 1) ** t) / config.population**t

    def by_plan(count: int) -> np.ndarray:
        """`count` rank positions, spread as evenly over `shares` as whole children allow."""
        marks = (rng.random() + np.arange(count)) / count
        picks = np.searchsorted(np.cumsum(shares), marks, side="right").clip(0, config.population - 1)
        return rng.permutation(picks)

    def plan_generation() -> tuple[list, list]:
        """The board's decisions for one generation: who is kept, and each
        child's two parents, all judged on the board's own map."""
        order = np.argsort(tour_lengths(view, pop), kind="stable")
        true_rank = np.empty(config.population, dtype=int)
        true_rank[np.argsort(lengths, kind="stable")] = ranks
        source = order[: config.elite].tolist()
        children = [pop[k].copy() for k in source]
        count = config.population - len(children)
        mothers = order[by_plan(count)]
        if config.planned_pairs == "random":
            fathers = order[by_plan(count)]
        else:
            fathers = []
            for m in mothers.tolist():
                options = order[rng.choice(config.population, size=3, p=shares)]
                apart = [difference_from_best(pop[[c]], pop[m]) for c in options.tolist()]
                fathers.append(int(options[int(np.argmax(apart))]))
            fathers = np.array(fathers)
        chosen = np.r_[mothers, fathers]
        planners_right.append(float((true_rank[chosen] < config.population // 2).mean()))
        for m, f in zip(mothers.tolist(), fathers.tolist()):
            if rng.random() < config.crossover_rate:
                child = crossover(rng, pop[m], pop[f])
            else:
                child = pop[m].copy()
            if rng.random() < config.mutation_rate:
                child = inversion_mutation(rng, child)
            children.append(child)
            source.append(m)
        return source, children

    record()
    for _ in range(config.generations):
        if planned:
            source, children = plan_generation()
            source = np.array(source)
            parent_gini.append(gini(np.bincount(source[config.elite:], minlength=config.population)))
            rank = np.empty(config.population, dtype=int)
            rank[np.argsort(lengths, kind="stable")] = np.arange(config.population)
            mothers_better_half.append(float(np.mean(rank[source[config.elite:]] < config.population / 2)))
            wealth_gini.append(0.0)
            top_tenth_share.append(0.1)
            pop = np.array(children)
            lengths = tour_lengths(dist, pop)
            evaluations += config.population - kept
            record()
            continue
        if by_wealth:
            wealth = ((1 + config.returns) * estate
                      + (1 - config.shared) * earnings(lengths) + config.shared)
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
        rank = np.empty(config.population, dtype=int)
        rank[np.argsort(lengths, kind="stable")] = np.arange(config.population)
        mothers_better_half.append(float(np.mean(rank[source[copies:]] < config.population / 2)))
        if by_wealth:
            wealth_gini.append(gini(wealth))
            top_tenth_share.append(top_share(wealth))
        else:
            wealth_gini.append(0.0)
            top_tenth_share.append(0.1)
        pop = np.array(children)
        lengths = tour_lengths(dist, pop)
        if by_wealth:
            estate = estates(config.inherited * wealth[source], source, lengths, config.heir)
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
        planners_right=np.array(planners_right),
        mothers_better_half=np.array(mothers_better_half),
        top_tenth_share=np.array(top_tenth_share),
    )
