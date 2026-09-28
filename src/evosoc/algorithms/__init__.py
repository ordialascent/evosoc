"""Algorithm registry: variants are referenced by name from experiment configs."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any

from .base import Algorithm, BudgetExhausted, Context, Population

ALGORITHMS: dict[str, type[Algorithm]] = {}


def register(name: str) -> Callable[[type[Algorithm]], type[Algorithm]]:
    def deco(cls: type[Algorithm]) -> type[Algorithm]:
        if name in ALGORITHMS:
            raise ValueError(f"algorithm {name!r} registered twice")
        cls.name = name
        ALGORITHMS[name] = cls
        return cls

    return deco


def make_algorithm(name: str, params: dict[str, Any]) -> Algorithm:
    try:
        cls = ALGORITHMS[name]
    except KeyError:
        raise ValueError(f"unknown algorithm {name!r}; known: {sorted(ALGORITHMS)}") from None
    return cls(**params)


# Import every module in this package, so a new file with an @register(...)
# class is picked up without being listed anywhere.
for _module in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_module.name}")

__all__ = [
    "ALGORITHMS",
    "Algorithm",
    "BudgetExhausted",
    "Context",
    "Population",
    "make_algorithm",
    "register",
]
