"""Baseline v0: the textbook genetic algorithm, untuned.

The engine's placeholder before Round 0; it never played a match. Each
generation keeps the 2 shortest trips; every other place goes to a child
whose two parents each win a contest of 3 random members; the child is a
crossover of its parents (90% of children, otherwise a copy of the mother)
and has a stretch reversed (30% of children). Every child is measured once.
It plays with the arena's toolbox, which since Round 0 is the frozen
season-1 toolbox; its untuned settings (100 members, order crossover) are
kept in Round 0's record (evosoc/tuning.py).

Each matter is one method, so a later version can hand exactly one matter
to a Board or to the Members by overriding it.
"""

from __future__ import annotations

import numpy as np

from ...arena import Arena
from ...contestant import Contestant


class Baseline(Contestant):
    name = "Baseline"
    version = 0
    team = "neutral"
    parent = None
    idea = "A textbook genetic algorithm: fixed rules, nobody decides anything."
    battle_cry = ""
    declaration = {
        "survival": ("law", "the 2 shortest trips are kept unchanged"),
        "reproduction": ("law", "each child's mother wins a contest of 3 random members"),
        "pairing": ("law", "each child's father wins a separate contest of 3 random members"),
        "variation": ("law", "the toolbox's crossover for 90% of children, a reversed stretch for 30%"),
        "resources": ("law", "every child is measured once; nothing else is measured"),
        "contact": ("law", "everyone can meet everyone"),
    }

    elite = 2
    contest_size = 3
    crossover_rate = 0.9
    mutation_rate = 0.3

    # Survival
    def survivors(self, rng, pop, lengths) -> np.ndarray:
        return np.argsort(lengths, kind="stable")[: self.elite]

    # Reproduction
    def mothers(self, rng, pop, lengths, count: int) -> np.ndarray:
        return np.array([self.contest(rng, lengths) for _ in range(count)], dtype=int)

    # Pairing
    def fathers(self, rng, pop, lengths, mothers: np.ndarray) -> np.ndarray:
        return np.array([self.contest(rng, lengths) for _ in mothers], dtype=int)

    # Contact: who can be drawn into a contest
    def contest(self, rng, lengths) -> int:
        picks = rng.integers(0, len(lengths), size=self.contest_size)
        return int(picks[np.argmin(lengths[picks])])

    # Variation
    def child(self, rng, toolbox, mother: np.ndarray, father: np.ndarray) -> np.ndarray:
        if rng.random() < self.crossover_rate:
            out = toolbox.crossover(rng, mother, father)
        else:
            out = mother.copy()
        if rng.random() < self.mutation_rate:
            out = toolbox.mutation(rng, out)
        return out

    # Resources: every child is measured once, nothing more.
    def measure_children(self, arena: Arena, children: np.ndarray) -> np.ndarray:
        return arena.measure_many(children)

    def play(self, arena: Arena) -> None:
        rng, toolbox = arena.rng, arena.toolbox
        size = toolbox.population
        pop = np.array([toolbox.random_tour(rng, arena.cities) for _ in range(size)])
        lengths = arena.measure_many(pop)
        while True:
            kept = self.survivors(rng, pop, lengths)
            mothers = self.mothers(rng, pop, lengths, size - len(kept))
            fathers = self.fathers(rng, pop, lengths, mothers)
            children = np.array([self.child(rng, toolbox, pop[m], pop[f])
                                 for m, f in zip(mothers.tolist(), fathers.tolist())])
            child_lengths = self.measure_children(arena, children)
            pop = np.concatenate([pop[kept], children])
            lengths = np.concatenate([lengths[kept], child_lengths])


contestant = Baseline()
