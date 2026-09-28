import numpy as np
import pytest

from evosoc.problems import TSP, make_problem
from evosoc.problems.tsp import TSPLIB_OPTIMA


def square() -> TSP:
    return TSP("square", np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float))


def test_tour_length():
    p = square()
    assert p.evaluate(np.array([0, 1, 2, 3])) == pytest.approx(4.0)
    assert p.evaluate(np.array([0, 2, 1, 3])) == pytest.approx(2 + 2 * np.sqrt(2))


def test_canonical_ignores_start_and_direction():
    p = square()
    a = np.array([0, 1, 2, 3])
    assert p.canonical(a) == p.canonical(np.array([2, 3, 0, 1]))
    assert p.canonical(a) == p.canonical(a[::-1])
    assert p.distance(a, a[::-1]) == 0.0
    assert p.distance(a, np.array([0, 2, 1, 3])) == pytest.approx(0.5)


@pytest.mark.parametrize("name", sorted(TSPLIB_OPTIMA))
def test_bundled_tsplib(name):
    p = make_problem({"type": "tsplib", "name": name})
    assert p.optimum == TSPLIB_OPTIMA[name]
    rng = np.random.default_rng(0)
    assert p.evaluate(p.random_genome(rng)) > p.optimum


@pytest.mark.parametrize("name", sorted(TSPLIB_OPTIMA))
def test_published_optimal_tour_scores_optimum(name):
    """Checks the TSPLIB distance convention (EUC_2D rounding) end to end."""
    p = TSP.tsplib(name)
    assert sorted(p.optimal_genome) == list(range(p.n))
    assert p.evaluate(p.optimal_genome) == TSPLIB_OPTIMA[name]


def test_random_value_is_the_mean_random_tour_length():
    p = TSP.uniform(30, seed=0)
    rng = np.random.default_rng(0)
    lengths = [p.evaluate(p.random_genome(rng)) for _ in range(20_000)]
    assert np.mean(lengths) == pytest.approx(p.random_value, rel=0.01)
    assert p.optimal_genome is None


def test_uniform_is_deterministic():
    a, b = TSP.uniform(30, seed=3), TSP.uniform(30, seed=3)
    np.testing.assert_array_equal(a.coords, b.coords)


def test_gini():
    from evosoc.metrics import gini

    assert gini(np.ones(10)) == 0.0
    assert gini(np.array([0.0, 0.0, 0.0, 1.0])) == pytest.approx(0.75)
    assert gini(np.zeros(5)) == 0.0
