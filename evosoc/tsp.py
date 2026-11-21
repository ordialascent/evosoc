"""The problem every population solves: the travelling salesman.

A map of cities is placed at random in a 1 x 1 square. A solution is a
round trip that visits every city once. Shorter trips are better.
"""

from __future__ import annotations

import numpy as np


def make_cities(n: int, seed: int) -> np.ndarray:
    """Return an (n, 2) array of city coordinates, fixed by `seed`."""
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))


def distance_matrix(cities: np.ndarray) -> np.ndarray:
    diff = cities[:, None, :] - cities[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def tour_length(dist: np.ndarray, tour: np.ndarray) -> float:
    """Length of the round trip that visits cities in the order of `tour`."""
    return float(dist[tour, np.roll(tour, -1)].sum())


def tour_lengths(dist: np.ndarray, tours: np.ndarray) -> np.ndarray:
    """Lengths of many tours at once; `tours` has shape (count, n)."""
    return dist[tours, np.roll(tours, -1, axis=1)].sum(axis=1)


def reference_tour(dist: np.ndarray) -> np.ndarray:
    """A good tour found without any evolution, used as a yardstick.

    Starts from nearest-neighbour and improves it with 2-opt (reverse any
    stretch of the trip that makes it shorter) until nothing helps.
    Deterministic: no randomness involved.
    """
    n = len(dist)
    tour = [0]
    unvisited = set(range(1, n))
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda c: (dist[last, c], c))
        tour.append(nxt)
        unvisited.remove(nxt)
    tour = np.array(tour)

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                if delta < -1e-12:
                    tour[i : j + 1] = tour[i : j + 1][::-1]
                    improved = True
    return tour
