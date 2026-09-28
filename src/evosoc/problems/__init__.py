"""Problem registry: build problems from experiment-config entries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import Problem
from .tsp import TSP

#: Maps a config ``type`` to a constructor taking the remaining keys as kwargs.
PROBLEMS: dict[str, Callable[..., Problem]] = {
    "tsplib": TSP.tsplib,
    "tsp_uniform": TSP.uniform,
}


def make_problem(spec: dict[str, Any]) -> Problem:
    """``{"type": "tsplib", "name": "berlin52"}`` -> a ``Problem``."""
    spec = dict(spec)
    kind = spec.pop("type")
    try:
        factory = PROBLEMS[kind]
    except KeyError:
        raise ValueError(f"unknown problem type {kind!r}; known: {sorted(PROBLEMS)}") from None
    return factory(**spec)


__all__ = ["PROBLEMS", "TSP", "Problem", "make_problem"]
