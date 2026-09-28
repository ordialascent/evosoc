"""Variation operators for permutation genomes (TSP tours).

They run once per birth, hundreds of thousands of times per run, so the
crossover is compiled with Numba. All randomness comes from the run's
generator, so runs stay reproducible.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def order_crossover(p1: np.ndarray, p2: np.ndarray, i: int, j: int) -> np.ndarray:
    """Order crossover, OX (Davis 1985): keep ``p1[i:j]`` in place, fill the
    remaining positions with the other cities in the order they follow in ``p2``."""
    n = p1.size
    child = np.empty(n, np.int64)
    taken = np.zeros(n, np.bool_)
    for k in range(i, j):
        child[k] = p1[k]
        taken[p1[k]] = True
    pos = j % n
    for k in range(n):
        c = p2[(j + k) % n]
        if not taken[c]:
            child[pos] = c
            pos = (pos + 1) % n
    return child


def inversion(tour: np.ndarray, rng: np.random.Generator) -> None:
    """Reverse a random segment in place: a chromosomal inversion.

    For a tour this replaces exactly two edges (a 2-opt move). Segment lengths
    are 2 .. n-2, since reversing 1 or n-1 cities leaves the tour unchanged.
    """
    n = tour.size
    length = rng.integers(2, n - 1)
    i = rng.integers(0, n - length + 1)
    tour[i : i + length] = tour[i : i + length][::-1].copy()
