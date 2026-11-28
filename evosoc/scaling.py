"""Scaling probe: how the textbook GA behaves as the society grows (training seeds only).

Baseline v1's rules (keep the 2 best, contests of 7, every child crossed and
mutated) play societies of `POPULATIONS` members on maps of `CITIES` cities.
The budget is a number of generations times the population, so a bigger
society gets proportionally more measurements and is not handicapped by its
own slowness. Along the way the probe records, at a few generations:

- `gap`: best trip so far against the yardstick, %;
- `repeats`: share of this generation's children that are trips already in
  the society (or made twice this generation): measurements that teach nothing;
- `distinct`: share of members whose trip no other member has;
- `shared`: share of the best trip's links that the average member uses
  (1.0: everybody follows the leader).

The probe only watches; it measures nothing extra. All maps are training maps.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np

from .arena import TOOLBOX, Arena, BudgetSpent
from .contestants.baseline.baseline_v1 import BaselineV1
from .match import gap, yardstick
from .rules import FIRST_MATCH_SEED

POPULATIONS = (30, 100, 300, 1_000, 3_000, 10_000)
CITIES = (50, 100, 200)
GENERATIONS = (100, 1_000)
MAPS = 2
FIRST_SEED = 500_000  # training seeds reserved for this probe


def probe_seed(cities: int, map_index: int) -> int:
    seed = FIRST_SEED + 1_000 * CITIES.index(cities) + map_index
    assert seed < FIRST_MATCH_SEED
    return seed


def canonical_hash(tours: np.ndarray) -> np.ndarray:
    """One number per round trip, the same whatever city it starts at or direction it runs."""
    n = tours.shape[1]
    start = np.argmax(tours == 0, axis=1)
    rows = tours[np.arange(len(tours))[:, None], (start[:, None] + np.arange(n)) % n]
    flip = rows[:, 1] > rows[:, -1]
    rows[flip, 1:] = rows[flip, 1:][:, ::-1]
    weights = np.random.default_rng(0).integers(1, 2**62, size=n, dtype=np.int64)
    with np.errstate(over="ignore"):
        return (rows.astype(np.int64) * weights).sum(axis=1)


def shared_links(best: np.ndarray, tours: np.ndarray) -> float:
    """Mean share of the best trip's links found in each trip."""
    n = len(best)
    after, before = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    after[best], before[best] = np.roll(best, -1), np.roll(best, 1)
    nxt = np.roll(tours, -1, axis=1)
    hit = (after[tours] == nxt) | (before[tours] == nxt)
    return float(hit.mean())


class Watched(BaselineV1):
    """Baseline v1 with a notebook: same moves, same measurements, plus observations."""

    def __init__(self, stick: float, marks: set[int]):
        self.stick, self.marks, self.notes, self.generation = stick, marks, [], 0
        self.pop_hashes = None

    def measure_children(self, arena, children):
        self.generation += 1
        if self.generation in self.marks:
            hashes = canonical_hash(children)
            known = np.isin(hashes, self.pop_hashes) if self.pop_hashes is not None else 0
            _, first = np.unique(hashes, return_index=True)
            twice = np.ones(len(hashes), dtype=bool)
            twice[first] = False
            repeats = float(np.mean(known | twice))
        lengths = arena.measure_many(children)
        if self.generation in self.marks:
            self.notes.append({"generation": self.generation, "repeats": repeats,
                               "children": len(children)})
        return lengths

    def observe(self, pop):
        """Called with the society after each generation (see `_run`)."""
        hashes = canonical_hash(pop)
        self.pop_hashes = hashes
        if self.notes and self.notes[-1]["generation"] == self.generation and "gap" not in self.notes[-1]:
            _, counts = np.unique(hashes, return_counts=True)
            self.notes[-1].update({
                "gap": gap(self.arena.best_length, self.stick),
                "distinct": float((counts == 1).sum() / len(pop)),
                "shared": shared_links(self.arena.best_tour, pop),
            })


def _run(job) -> dict:
    population, cities, generations, seed = job
    stick = yardstick(cities, seed)
    marks = set(np.unique(np.round(np.geomspace(1, generations, 12)).astype(int)).tolist())
    toolbox = type(TOOLBOX)(population=population, crossover=TOOLBOX.crossover,
                            mutation=TOOLBOX.mutation)
    budget = population * (generations + 1)
    arena = Arena(cities, budget, seed, toolbox)
    ga = Watched(stick, marks)
    start = time.perf_counter()
    # Baseline v0's loop, with the society shown to the notebook after every
    # generation. Moves and measurements are exactly Baseline v1's.
    rng = arena.rng
    ga.arena = arena
    pop = np.array([toolbox.random_tour(rng, cities) for _ in range(population)])
    lengths = arena.measure_many(pop)
    ga.observe(pop)
    try:
        while True:
            kept = ga.survivors(rng, pop, lengths)
            mothers = ga.mothers(rng, pop, lengths, population - len(kept))
            fathers = ga.fathers(rng, pop, lengths, mothers)
            children = np.array([ga.child(rng, toolbox, pop[m], pop[f])
                                 for m, f in zip(mothers.tolist(), fathers.tolist())])
            child_lengths = ga.measure_children(arena, children)
            pop = np.concatenate([pop[kept], children])
            lengths = np.concatenate([lengths[kept], child_lengths])
            ga.observe(pop)
    except BudgetSpent:
        pass
    return {"population": population, "cities": cities, "generations": generations, "seed": seed,
            "budget": budget, "yardstick": stick, "result": arena.best_length,
            "gap": gap(arena.best_length, stick), "seconds": time.perf_counter() - start,
            "notes": [n for n in ga.notes if "gap" in n]}


def jobs() -> list[tuple]:
    return [(p, c, g, probe_seed(c, i)) for c in CITIES for g in GENERATIONS
            for p in POPULATIONS for i in range(MAPS)]


def probe(store: Path, workers: int | None = None, log: Callable[[str], None] | None = None) -> list:
    """Run every probe job once, keeping each finished one in `store` (resumable)."""
    store.mkdir(parents=True, exist_ok=True)
    path = lambda j: store / ("probe-" + "-".join(map(str, j)) + ".json")  # noqa: E731
    todo = [j for j in jobs() if not path(j).exists()]
    todo.sort(key=lambda j: -j[0] * j[1] * j[2])  # longest first
    start, done = time.perf_counter(), 0
    with ProcessPoolExecutor(workers or os.cpu_count() or 1) as pool:
        futures = {pool.submit(_run, j): j for j in todo}
        for future in as_completed(futures):
            j = futures[future]
            path(j).write_text(json.dumps(future.result()) + "\n")
            done += 1
            if log:
                eta = (time.perf_counter() - start) / done * (len(todo) - done)
                log(f"{done}/{len(todo)} probe runs done ({j}), about {eta / 60:.0f} min to go")
    return [json.loads(path(j).read_text()) for j in jobs()]
