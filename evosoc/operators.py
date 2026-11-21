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
    links = [set() for _ in range(n)]
    for parent in (mother, father):
        nxt, prv = np.roll(parent, -1), np.roll(parent, 1)
        for c, a, b in zip(parent.tolist(), nxt.tolist(), prv.tolist()):
            links[c].update((a, b))
    child = np.empty(n, dtype=mother.dtype)
    visited = np.zeros(n, dtype=bool)
    current = int(mother[0])
    for k in range(n):
        child[k] = current
        visited[current] = True
        for c in links[current]:
            links[c].discard(current)
        if k == n - 1:
            break
        options = sorted(links[current])
        if options:
            fewest = min(len(links[c]) for c in options)
            options = [c for c in options if len(links[c]) == fewest]
        else:
            options = np.flatnonzero(~visited).tolist()
        current = options[int(rng.integers(len(options)))]
    return child
