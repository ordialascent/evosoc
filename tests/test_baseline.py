import numpy as np

from evosoc import ga
from evosoc.operators import edge_crossover, inversion_mutation, order_crossover
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
        assert sorted(edge_crossover(rng, m, f)) == list(range(n))


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


def test_every_way_of_choosing_rulers_keeps_valid_trips_and_is_repeatable():
    dist = _dist()
    for way in ga.RULER_CHOICES:
        config = ga.Config(population=30, generations=40, elite=5, rulers_chosen=way)
        a, b = ga.run(dist, config, 4), ga.run(dist, config, 4)
        assert np.array_equal(a.best, b.best)
        assert sorted(a.best_tour) == list(range(len(dist)))
        assert a.evaluations == 30 + 40 * (30 - ga.kept_per_generation(config))


def test_merit_rulers_never_lose_the_best_but_others_can():
    dist = _dist()
    merit = ga.run(dist, ga.Config(population=30, generations=60, elite=3), 0)
    assert np.all(np.diff(merit.best) <= 1e-12)
    lost = [np.any(np.diff(ga.run(dist, ga.Config(population=30, generations=60, elite=3,
                                                   rulers_chosen=way), s).best) > 0)
            for way in ("lottery", "hereditary") for s in range(3)]
    assert any(lost)


def test_unknown_way_of_choosing_rulers_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        ga.run(_dist(), ga.Config(rulers_chosen="divine right"), 0)


def test_choosing_experiment_is_repeatable(tmp_path):
    from evosoc.experiment import run_choosing

    kw = dict(rulers=(3,), cities=12, seeds=3, population=20, budget=600, checkpoints=(200, 600))
    a = run_choosing(tmp_path / "a", **kw)
    b = run_choosing(tmp_path / "b", **kw)
    assert a == b
    assert (tmp_path / "a" / "choosing.csv").read_text() == (tmp_path / "b" / "choosing.csv").read_text()
    assert set(a["societies"]) == {"none", "baseline", "merit-3", "lottery-3", "hereditary-3"}
    assert (tmp_path / "a" / "choosing.png").exists()


def _links(tour):
    return {frozenset(p) for p in zip(tour.tolist(), np.roll(tour, -1).tolist())}


def test_edge_crossover_of_identical_parents_copies_them():
    rng = np.random.default_rng(1)
    for _ in range(50):
        m = rng.permutation(20)
        assert _links(edge_crossover(rng, m, m.copy())) == _links(m)


def test_edge_crossover_mostly_reuses_parent_links():
    rng = np.random.default_rng(2)
    new = []
    for _ in range(200):
        m, f = rng.permutation(30), rng.permutation(30)
        new.append(len(_links(edge_crossover(rng, m, f)) - _links(m) - _links(f)))
    assert np.mean(new) < 3  # out of 30 links


def test_crossover_setting_is_used_and_checked():
    dist = _dist()
    a = ga.run(dist, SMALL, seed=4)
    b = ga.run(dist, ga.Config(population=30, generations=40, crossover="edge"), seed=4)
    assert not np.array_equal(a.best, b.best)
    assert len(b.population) == 30
    import pytest

    with pytest.raises(ValueError):
        ga.run(dist, ga.Config(crossover="nope"), seed=0)


def test_measure_damage_reversal_of_a_perfect_trip_is_never_better():
    from evosoc.experiment import measure_damage

    dist = _dist(12, 4)
    best = reference_tour(dist)
    pop = np.array([best] * 10)
    d = measure_damage(dist, pop, seed=0, children=50)
    # Copies of one trip: crossover gives the same trip back, reversals rarely help.
    assert d["order crossover"][0] == 0 and d["edge crossover"][0] == 0
    assert d["reversal"][1] >= 0


def test_reproduction_experiment_is_repeatable(tmp_path):
    from evosoc.experiment import run_reproduction

    kw = dict(cities=12, seeds=2, damage_seeds=2, stages=(0, 5), budget=1500)
    a = run_reproduction(tmp_path / "a", **kw)
    b = run_reproduction(tmp_path / "b", **kw)
    assert a == b
    assert (tmp_path / "a" / "reproduction.csv").read_text() == \
        (tmp_path / "b" / "reproduction.csv").read_text()
    assert set(a["societies"]) == {"textbook", "fewer reversals", "edge crossover"}


def test_pressure_experiment_is_repeatable_and_reads_budgets_off_one_run(tmp_path):
    from evosoc.experiment import best_at, run_pressure, trips_for

    kw = dict(cities=(10, 14), budgets=(3, 10), rulers=(0, 2, 5), seeds=3, population=20)
    a = run_pressure(tmp_path / "a", **kw)
    b = run_pressure(tmp_path / "b", **kw)
    assert a == b
    assert (tmp_path / "a" / "pressure.png").exists()
    assert a["budgets"] == {"3": trips_for(3, 20), "10": trips_for(10, 20)}
    cell = a["results"]["14"]["edge crossover"]["3"]
    assert cell["winner"] in (0, 2, 5) and cell["runner_up"] != cell["winner"]

    # Reading a long run at the short budget equals running only the short budget.
    dist = _dist(14, 1)
    long = ga.run(dist, ga.Config(population=20, elite=5, generations=40), 2)
    short = ga.run(dist, ga.Config(population=20, elite=5, generations=4), 2)
    assert best_at(long, short.measured[-1]) == short.best[-1]
    assert np.array_equal(long.best[:5], short.best)
