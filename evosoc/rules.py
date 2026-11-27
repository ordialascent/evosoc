"""The rules of the game as data: events, seeds, charters (see docs/rules.md)."""

from __future__ import annotations

from dataclasses import dataclass

SEASON = 1
SEASON_NAME = "Open Books"
RULES_VERSION = "1.1"


@dataclass(frozen=True)
class Event:
    key: str
    name: str
    cities: int
    budget: int
    tests: str


EVENTS = (
    Event("sprint", "Sprint", 50, 10_000, "who gets good fast"),
    Event("marathon", "Marathon", 50, 100_000, "who gets furthest with time"),
    Event("big", "Big Country", 100, 100_000, "who copes with a harder problem than they have time for"),
)
EVENT_BY_KEY = {e.key: e for e in EVENTS}

MAPS_PER_EVENT = 30

# Seeds below this are for teams (development and training); seeds at or
# above it belong to the referee.
FIRST_MATCH_SEED = 1_000_000


def match_seed(match: int, event: int, map_index: int) -> int:
    """Seed of map `map_index` (0-29) of event `event` (0-2) in match `match` (rules §5)."""
    if not (match >= 1 and 0 <= event < len(EVENTS) and 0 <= map_index < 100):
        raise ValueError("match >= 1, event 0-2, map 0-99")
    return FIRST_MATCH_SEED + 1_000 * match + 100 * event + map_index


def training_seed(event: int, map_index: int) -> int:
    """Seed of training map `map_index` for event `event`: below the match seeds."""
    seed = 100_000 * event + map_index
    if not 0 <= seed < FIRST_MATCH_SEED:
        raise ValueError("training seeds must stay below the match seeds")
    return seed


MATTERS = ("survival", "reproduction", "pairing", "variation", "resources", "contact")
DECIDERS = ("law", "board", "member")
TEAMS = ("centralizers", "decentralizers", "neutral")


def charter_problems(team: str, charter: dict) -> list[str]:
    """Everything wrong with a contestant's charter under rules §6 (empty list: legal)."""
    problems = []
    if team not in TEAMS:
        problems.append(f"unknown team {team!r}")
    if set(charter) != set(MATTERS):
        problems.append(f"charter must name exactly the six matters {MATTERS}")
    bad = {m: d for m, d in charter.items() if d not in DECIDERS}
    if bad:
        problems.append(f"unknown deciders {bad}")
    deciders = set(charter.values())
    if team == "centralizers":
        if "board" not in deciders:
            problems.append("a centralizer gives at least one matter to the Board")
        if "member" in deciders:
            problems.append("a centralizer gives no matter to Member")
    elif team == "decentralizers":
        if "member" not in deciders:
            problems.append("a decentralizer gives at least one matter to Member")
        if "board" in deciders:
            problems.append("a decentralizer gives no matter to the Board")
    elif team == "neutral" and deciders - {"law"}:
        problems.append("a neutral contestant leaves every matter to Law")
    return problems
