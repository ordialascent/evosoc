"""What a contestant is, and how to find one by its id.

Every contestant version lives in its own module under
`evosoc/contestants/<team>/`, named `<name>_v<version>.py`, and defines one
module-level `contestant`. Its id is the module name (e.g. `baseline_v0`).
A version is never edited once it has played, so every match can be rerun.
"""

from __future__ import annotations

import importlib
import pkgutil

from . import contestants as _package
from .arena import Arena
from .rules import MATTERS


class Contestant:
    """A society of trips. Subclasses set the declaration and implement `play`."""

    name: str = ""
    version: int = 0
    team: str = "neutral"  # "centralizers", "decentralizers" or "neutral"
    parent: str | None = None  # id of the version this one changes
    idea: str = ""  # the one idea of this version, in one sentence
    battle_cry: str = ""
    # Who decides each of the six matters: "law", "board" or "member", with
    # a short note on how. {matter: (decider, how)}
    declaration: dict[str, tuple[str, str]] = {}

    @property
    def id(self) -> str:
        return f"{self.name.lower().replace(' ', '_')}_v{self.version}"

    @property
    def charter(self) -> dict[str, str]:
        return {m: d for m, (d, _) in self.declaration.items()}

    def play(self, arena: Arena) -> None:
        """Search for a short round trip until the arena's budget is spent."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "id": self.id, "name": self.name, "version": self.version, "team": self.team,
            "parent": self.parent, "idea": self.idea, "battle_cry": self.battle_cry,
            "declaration": {m: {"decider": self.declaration[m][0], "how": self.declaration[m][1]}
                            for m in MATTERS if m in self.declaration},
            "module": type(self).__module__,
        }


def all_contestants() -> dict[str, Contestant]:
    """Every contestant version in the repository, by id."""
    found = {}
    for info in pkgutil.walk_packages(_package.__path__, _package.__name__ + "."):
        if info.ispkg:
            continue
        module = importlib.import_module(info.name)
        c = getattr(module, "contestant", None)
        if isinstance(c, Contestant):
            if module.__name__.rsplit(".", 1)[1] != c.id:
                raise ValueError(f"{module.__name__}: module must be named {c.id}.py")
            found[c.id] = c
    return dict(sorted(found.items()))


def load(contestant_id: str) -> Contestant:
    found = all_contestants()
    if contestant_id not in found:
        raise KeyError(f"no contestant {contestant_id!r}; known: {', '.join(found)}")
    return found[contestant_id]
