"""The fast edge recombination must give exactly the outputs of the original one."""

import numpy as np

from evosoc.operators import edge_crossover


def edge_crossover_reference(rng, mother, father):
    """Edge recombination as first written for Round 0 (sets). Kept as the test oracle."""
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


def test_edge_crossover_matches_reference():
    source = np.random.default_rng(78)
    for trial in range(600):
        n = int(source.choice([3, 4, 5, 8, 13, 50, 100, 200]))
        mother = source.permutation(n)
        kind = trial % 4
        if kind == 0:
            father = source.permutation(n)
        elif kind == 1:
            father = mother.copy()  # clone pairing
        elif kind == 2:
            father = mother.copy()  # near clone: one reversed stretch
            i, j = sorted(source.choice(n, 2, replace=False))
            father[i:j + 1] = father[i:j + 1][::-1]
        else:
            father = mother[::-1].copy()  # same links, other direction
        seed = int(source.integers(2**32))
        a, b = np.random.default_rng(seed), np.random.default_rng(seed)
        fast, slow = edge_crossover(a, mother, father), edge_crossover_reference(b, mother, father)
        assert fast.dtype == slow.dtype
        assert (fast == slow).all(), (trial, n)
        # The generator is left in the same state, so everything after matches too.
        assert a.integers(2**62) == b.integers(2**62)
