"""Baseline v1: the textbook genetic algorithm, tuned for the arena and frozen.

Round 0's result (data/tuning/baseline_v1.json, docs/round0.md). Same rules
as v0, with the settings that won a seeded search on training maps: each
generation keeps the 2 shortest trips; every other place goes to a child
whose two parents each win a contest of 7 random members; every child is an
edge recombination of its parents with one stretch reversed. Every child is
measured once. With it the season-1 toolbox is frozen: 30 members, edge
recombination, stretch reversal (arena.TOOLBOX).

The yardstick of every match and the common ancestor of both teams' first
contestants. Never edited.
"""

from __future__ import annotations

from .baseline_v0 import Baseline


class BaselineV1(Baseline):
    version = 1
    parent = "baseline_v0"
    idea = "The textbook genetic algorithm with the settings that won Round 0's tuning search."
    declaration = {
        "survival": ("law", "the 2 shortest trips are kept unchanged"),
        "reproduction": ("law", "each child's mother wins a contest of 7 random members"),
        "pairing": ("law", "each child's father wins a separate contest of 7 random members"),
        "variation": ("law", "every child: edge recombination of its parents, then one reversed stretch"),
        "resources": ("law", "every child is measured once; nothing else is measured"),
        "contact": ("law", "everyone can meet everyone"),
    }

    elite = 2
    contest_size = 7
    crossover_rate = 1.0
    mutation_rate = 1.0


contestant = BaselineV1()
