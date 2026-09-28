"""Variation operators for permutation genomes (TSP tours).

Compiled with Numba: these run once per birth, hundreds of thousands of times
per run. They draw no random numbers themselves; the caller passes uniform
draws from the run's generator, so runs stay reproducible.
"""

from __future__ import annotations

import numpy as np
from numba import njit


def erx_uniforms(n: int, rng: np.random.Generator) -> np.ndarray:
    """The random draws :func:`edge_recombination` needs for tours of ``n`` cities."""
    return rng.random(6 * n + 1)


@njit(cache=True)
def edge_recombination(p1: np.ndarray, p2: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Edge recombination crossover (Whitley et al. 1989), with shared edges
    preferred (Starkweather et al. 1991).

    The child is built by walking from city to city, moving along an edge
    that one of the parents has whenever possible:

    1. An edge **both** parents have is always taken first (inherited from both).
    2. Otherwise the neighbour with the fewest remaining options is taken, so
       that no city gets stranded without an inherited edge left.
    3. Only when every inherited edge leads to an already visited city is a
       random unvisited city chosen: a new edge that neither parent had.

    ``u`` holds uniform draws in [0, 1), see :func:`erx_uniforms`.
    """
    n = p1.size
    adj = np.full((n, 4), -1, np.int64)
    shared = np.zeros((n, 4), np.bool_)
    deg = np.zeros(n, np.int64)
    for p in (p1, p2):
        for i in range(n):
            a = p[i]
            for b in (p[i - 1], p[(i + 1) % n]):
                known = False
                for k in range(deg[a]):
                    if adj[a, k] == b:
                        shared[a, k] = True
                        known = True
                if not known:
                    adj[a, deg[a]] = b
                    deg[a] += 1

    child = np.empty(n, np.int64)
    visited = np.zeros(n, np.bool_)
    cur = p1[0] if u[0] < 0.5 else p2[0]
    r = 1  # next unused random draw
    for pos in range(n):
        child[pos] = cur
        visited[cur] = True
        if pos == n - 1:
            break
        best, best_score, ties = -1, 1 << 30, 0
        for k in range(deg[cur]):
            c = adj[cur, k]
            if visited[c]:
                continue
            if shared[cur, k]:
                score = -1
            else:
                score = 0
                for m in range(deg[c]):
                    if not visited[adj[c, m]]:
                        score += 1
            if score < best_score:
                best, best_score, ties = c, score, 1
            elif score == best_score:
                ties += 1  # uniform choice among ties (reservoir sampling)
                if u[r] * ties < 1.0:
                    best = c
                r += 1
        if best == -1:
            skip = int(u[r] * (n - pos - 1))
            r += 1
            for c in range(n):
                if not visited[c]:
                    if skip == 0:
                        best = c
                        break
                    skip -= 1
        cur = best
    return child


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
