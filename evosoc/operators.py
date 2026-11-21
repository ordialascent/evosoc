"""Textbook GA operators for round trips (permutations of cities)."""

from __future__ import annotations

import numpy as np


def tournament(rng: np.random.Generator, lengths: np.ndarray, size: int) -> int:
    """Pick `size` members at random; the one with the shortest trip wins."""
    picks = rng.integers(0, len(lengths), size=size)
    return int(picks[np.argmin(lengths[picks])])


def order_crossover(rng: np.random.Generator, mother: np.ndarray, father: np.ndarray) -> np.ndarray:
    """Order crossover (OX1).

    The child copies one stretch of the mother's trip as is, then fills the
    remaining cities in the order they appear in the father's trip.
    """
    n = len(mother)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    child = np.full(n, -1)
    child[i : j + 1] = mother[i : j + 1]
    taken = np.zeros(n, dtype=bool)
    taken[mother[i : j + 1]] = True
    rest = [c for c in np.roll(father, -(j + 1)) if not taken[c]]
    positions = [(j + 1 + k) % n for k in range(n - (j - i + 1))]
    child[positions] = rest
    return child


def inversion_mutation(rng: np.random.Generator, tour: np.ndarray) -> np.ndarray:
    """Reverse a random stretch of the trip."""
    n = len(tour)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    out = tour.copy()
    out[i : j + 1] = out[i : j + 1][::-1]
    return out
