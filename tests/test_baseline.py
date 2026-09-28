"""Guarantees of the baseline GA (see docs/baseline-ga.md, "Guarantees")."""

import numpy as np
import pytest

from evosoc.algorithms import BudgetExhausted, Context, make_algorithm
from evosoc.algorithms.steady_state import SteadyStateGA
from evosoc.problems import TSP

#: Reference run: berlin52, seed 0, 20k evaluations, default parameters.
#: Changing the baseline's behaviour in any way changes this number. Update it
#: only when that change is intended, and say so in the commit message.
#: (best tour length, number of improvements, evaluation of the last improvement)
PINNED = (8799.0, 82, 19823)


def run(problem, budget, seed=0, **params):
    """Run the GA in memory; returns the context and every reported population."""
    reports = []
    ss_init, ss_algo = np.random.SeedSequence(seed).spawn(2)
    ctx = Context(
        problem,
        budget,
        rng=np.random.default_rng(ss_algo),
        init_rng=np.random.default_rng(ss_init),
        on_report=lambda c, pop, extra: reports.append(
            (c.evaluations, c.best_so_far, [g.copy() for g in pop.genomes], pop.fitness.copy())
        ),
    )
    with pytest.raises(BudgetExhausted):
        make_algorithm("steady_state_ga", params).run(ctx)
    return ctx, reports


@pytest.fixture(scope="module")
def berlin():
    return TSP.tsplib("berlin52")


def test_pinned_reference_run(berlin):
    ctx, _ = run(berlin, 20_000)
    assert (ctx.best_so_far, len(ctx.improvements), ctx.improvements[-1][0]) == PINNED


@pytest.mark.parametrize("crossover", ["erx", "ox"])
def test_reproducible(berlin, crossover):
    a, _ = run(berlin, 5_000, seed=3, crossover=crossover)
    b, _ = run(berlin, 5_000, seed=3, crossover=crossover)
    assert a.improvements == b.improvements
    np.testing.assert_array_equal(a.best_genome, b.best_genome)


@pytest.mark.parametrize("crossover", ["erx", "ox"])
def test_invariants(berlin, crossover):
    ctx, reports = run(berlin, 10_000, seed=1, pop_size=50, crossover=crossover)
    assert ctx.evaluations == 10_000
    assert len(reports) == 10_000 // 50  # one report per 50 births, the first after init
    for evaluations, best_so_far, genomes, fitness in reports:
        assert evaluations % 50 == 0
        assert len(genomes) == 50
        for g, f in zip(genomes, fitness, strict=True):
            assert sorted(g) == list(range(berlin.n))
            assert f == berlin.evaluate(g)
        # The best individual never dies.
        assert fitness.min() == best_so_far
    assert berlin.evaluate(ctx.best_genome) == ctx.best_so_far


def test_beats_random_search(berlin):
    ctx, _ = run(berlin, 20_000)
    rng = np.random.default_rng(0)
    random_best = min(berlin.evaluate(rng.permutation(berlin.n)) for _ in range(20_000))
    assert ctx.best_so_far < 0.5 * random_best


@pytest.mark.parametrize(
    "params",
    [
        {"pop_size": 1},
        {"tournament_size": 1},
        {"pop_size": 10, "tournament_size": 11},
        {"crossover_rate": 1.5},
        {"mutation_rate": -0.1},
        {"crossover": "pmx"},
    ],
)
def test_rejects_invalid_parameters(params):
    with pytest.raises(ValueError):
        SteadyStateGA(**params)


def test_rejects_tiny_instances():
    tiny = TSP("tiny", np.random.default_rng(0).uniform(size=(4, 2)))
    with pytest.raises(ValueError, match="at least 5 cities"):
        run(tiny, 1_000)


def test_rejects_budget_smaller_than_population(berlin):
    with pytest.raises(ValueError, match="too small"):
        run(berlin, 300, pop_size=200)


def test_works_on_smallest_allowed_instance():
    small = TSP("small", np.random.default_rng(0).uniform(size=(5, 2)))
    ctx, _ = run(small, 2_000, pop_size=10)
    assert ctx.evaluations == 2_000
