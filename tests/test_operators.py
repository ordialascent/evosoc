import numpy as np
import pytest

from evosoc.operators.permutation import inversion, order_crossover
from evosoc.problems import TSP


def edges(tour: np.ndarray) -> set[frozenset[int]]:
    return {frozenset(e) for e in zip(tour, np.roll(tour, -1), strict=True)}


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_ox_keeps_segment_and_is_a_tour(rng):
    a, b = rng.permutation(20), rng.permutation(20)
    child = order_crossover(a, b, 5, 12)
    assert sorted(child) == list(range(20))
    np.testing.assert_array_equal(child[5:12], a[5:12])


def test_inversion_changes_exactly_two_edges(rng):
    p = TSP.uniform(40, seed=0)
    for _ in range(100):
        tour = rng.permutation(40)
        child = tour.copy()
        inversion(child, rng)
        assert sorted(child) == list(range(40))
        assert len(edges(tour) - edges(child)) == 2
        assert p.distance(tour, child) == pytest.approx(2 / 40)


def test_ox_of_identical_parents_is_the_parent(rng):
    a = rng.permutation(30)
    np.testing.assert_array_equal(order_crossover(a, a.copy(), 7, 19), a)
