import numpy as np

from evosoc import ga
from evosoc.operators import inversion_mutation, order_crossover
from evosoc.tsp import distance_matrix, make_cities, reference_tour, tour_length

SMALL = ga.Config(population=30, generations=40)


def _dist(n=15, seed=3):
    return distance_matrix(make_cities(n, seed))


def test_same_seed_same_result():
    dist = _dist()
    a = ga.run(dist, SMALL, seed=7)
    b = ga.run(dist, SMALL, seed=7)
    assert np.array_equal(a.best, b.best)
    assert np.array_equal(a.best_tour, b.best_tour)


def test_different_seeds_differ():
    dist = _dist()
    assert not np.array_equal(ga.run(dist, SMALL, 1).mean, ga.run(dist, SMALL, 2).mean)


def test_operators_keep_valid_trips():
    rng = np.random.default_rng(0)
    n = 12
    for _ in range(200):
        m, f = rng.permutation(n), rng.permutation(n)
        assert sorted(order_crossover(rng, m, f)) == list(range(n))
        assert sorted(inversion_mutation(rng, m)) == list(range(n))


def test_elitism_never_loses_the_best():
    h = ga.run(_dist(), SMALL, seed=0)
    assert np.all(np.diff(h.best) <= 1e-12)


def test_evolution_beats_random_and_reaches_yardstick_area():
    dist = _dist(20, 5)
    h = ga.run(dist, ga.Config(population=60, generations=200), seed=0)
    ref = tour_length(dist, reference_tour(dist))
    assert h.best[-1] < 0.5 * h.mean[0]
    assert h.best[-1] < 1.15 * ref


def test_evaluation_budget():
    h = ga.run(_dist(), SMALL, seed=0)
    assert h.evaluations == SMALL.population + SMALL.generations * (SMALL.population - SMALL.elite)
