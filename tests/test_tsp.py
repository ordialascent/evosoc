from pathlib import Path

import numpy as np
import pytest

from evosoc.problems import TSP, make_problem
from evosoc.problems.tsp import TSPLIB_OPTIMA

DATA = Path(__file__).parent / "data"


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
    lines = (DATA / f"{name}.opt.tour").read_text().split("TOUR_SECTION")[1].split()
    tour = np.array([int(x) - 1 for x in lines[: lines.index("-1")]])
    assert TSP.tsplib(name).evaluate(tour) == TSPLIB_OPTIMA[name]


def test_uniform_is_deterministic():
    a, b = TSP.uniform(30, seed=3), TSP.uniform(30, seed=3)
    np.testing.assert_array_equal(a.coords, b.coords)
