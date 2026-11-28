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


def edge_crossover(rng: np.random.Generator, mother: np.ndarray, father: np.ndarray) -> np.ndarray:
    """Edge recombination (ERX).

    A trip is really a set of links between cities, so the child is built
    almost entirely from links that at least one parent already uses. Start at
    the mother's first city. From the current city, go to a neighbour (in
    either parent) that has the fewest onward options left, so no city gets
    stranded; ties are broken at random. Only when every neighbour is already
    visited does the child jump to a random unvisited city, which creates a
    new link.
    """
    n = len(mother)
    # Each city's neighbours in either parent, in ascending order, and how
    # many distinct ones are still unvisited. A repeated neighbour becomes the
    # stand-in city n, which counts as visited.
    city = np.empty((n, 4), dtype=np.int64)
    for col, parent in enumerate((mother, father)):
        city[parent[:-1], 2 * col] = parent[1:]
        city[parent[-1], 2 * col] = parent[0]
        city[parent[1:], 2 * col + 1] = parent[:-1]
        city[parent[0], 2 * col + 1] = parent[-1]
    city.sort(axis=1)
    repeat = city[:, 1:] == city[:, :-1]
    city[:, 1:][repeat] = n
    adjacent = city.tolist()
    left = (4 - repeat.sum(axis=1)).tolist()
    visited = [False] * n + [True]
    order = [0] * n
    integers = rng.integers
    current = int(mother[0])
    for k in range(n - 1):
        order[k] = current
        visited[current] = True
        # Unvisited neighbours lose one onward option; keep those with the fewest.
        fewest, options = 5, None
        for c in adjacent[current]:
            if not visited[c]:
                c_left = left[c] - 1
                left[c] = c_left
                if c_left < fewest:
                    fewest, options = c_left, [c]
                elif c_left == fewest:
                    options.append(c)
        if options is None:
            options = [c for c in range(n) if not visited[c]]
        # A draw from one option uses no randomness (numpy returns 0 without
        # touching the generator), so it is skipped; the stream is unchanged.
        current = options[int(integers(len(options)))] if len(options) > 1 else options[0]
    order[n - 1] = current
    return np.array(order, dtype=mother.dtype)
