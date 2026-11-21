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


def test_experiment_is_repeatable(tmp_path):
    from evosoc.experiment import run_baseline

    a = run_baseline(tmp_path / "a", cities=12, seeds=3, config=SMALL, checkpoints=(10,))
    b = run_baseline(tmp_path / "b", cities=12, seeds=3, config=SMALL, checkpoints=(10,))
    assert a == b
    assert (tmp_path / "a" / "baseline.csv").read_text() == (tmp_path / "b" / "baseline.csv").read_text()
    assert set(a["gap_to_reference_pct_by_generation"]) == {"10", str(SMALL.generations)}


def test_difference_from_best():
    best = np.arange(10)
    assert ga.difference_from_best(np.array([best, best[::-1], np.roll(best, 3)]), best) == 0
    other = np.array([[0, 2, 4, 6, 8, 1, 3, 5, 7, 9]])
    assert 0.5 < ga.difference_from_best(other, best) <= 1


def test_records_one_entry_per_generation():
    h = ga.run(_dist(), SMALL, seed=0)
    assert len(h.diversity) == len(h.measured) == SMALL.generations + 1
    assert h.measured[-1] == h.evaluations
    assert np.all((h.diversity >= 0) & (h.diversity <= 1))


def test_same_budget_for_every_ruling_class():
    budget = 100 + 1000 * 98
    for k in (1, 2, 5, 20, 50):
        gens = ga.generations_for_budget(100, k, budget)
        used = 100 + gens * (100 - k)
        assert abs(used - budget) <= (100 - k) / 2
    assert ga.generations_for_budget(100, 2, budget) == 1000


def test_rulers_experiment_is_repeatable(tmp_path):
    from evosoc.experiment import run_rulers

    kw = dict(rulers=(1, 2, 10), cities=12, seeds=3, population=30, budget=1000,
              checkpoints=(300, 1000))
    a = run_rulers(tmp_path / "a", **kw)
    b = run_rulers(tmp_path / "b", **kw)
    assert a == b
    assert (tmp_path / "a" / "rulers.csv").read_text() == (tmp_path / "b" / "rulers.csv").read_text()
    assert set(a["societies"]) == {"1", "2", "10"}
