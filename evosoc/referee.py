"""The referee: fair-play checks (rules §7) and scoring (rules §4)."""

from __future__ import annotations

import inspect
import re
import time
from math import comb

from .arena import Arena
from .contestant import Contestant
from .rules import EVENTS, charter_problems, training_seed

SIGNIFICANCE = 0.05
TIE = 1e-9
MAX_SLOWDOWN = 5.0

# Names a contestant's code may not use: they would read the hidden map, or
# reach around the arena to do so.
FORBIDDEN = ("_lengths", "_meter", "distance_matrix", "make_cities", "tsp", "__closure__",
             "__dict__", "vars(", "gc", "inspect", "ctypes", "globals(", "getattr(", "setattr(")


def _token(word: str) -> str:
    """Regex for `word` as a whole name (not as part of a longer name)."""
    end = r"(?!\w)" if word[-1].isalnum() or word[-1] == "_" else ""
    return r"(?<!\w)" + re.escape(word) + end


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test: how likely a split this uneven is for two equals."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(a: float, b: float) -> int:
    """+1 if `a` is the shorter trip, -1 if `b` is, 0 for a tied map."""
    if abs(a - b) <= TIE * max(abs(a), abs(b)):
        return 0
    return 1 if a < b else -1


def score_event(a_results, b_results) -> dict:
    """Maps won, sign test and verdict for one event (`a_results`, `b_results`: per map)."""
    outcomes = [compare(a, b) for a, b in zip(a_results, b_results, strict=True)]
    won_a, won_b = outcomes.count(1), outcomes.count(-1)
    p = sign_test(won_a, won_b)
    winner = None
    if p < SIGNIFICANCE:
        winner = "a" if won_a > won_b else "b"
    return {"maps_won": {"a": won_a, "b": won_b, "tied": outcomes.count(0)},
            "p_value": p, "winner": winner}


def score_match(event_winners: list) -> dict:
    a, b = event_winners.count("a"), event_winners.count("b")
    return {"events_won": {"a": a, "b": b}, "score": f"{a}–{b}",
            "winner": "a" if a > b else "b" if b > a else None}


def source_of(contestant: Contestant) -> str:
    """Source of every class in the contestant's own lineage of contestant modules."""
    parts = []
    for cls in type(contestant).__mro__:
        if cls.__module__.startswith("evosoc.contestants."):
            parts.append(inspect.getsource(cls))
    parts.append(inspect.getsource(inspect.getmodule(type(contestant))))
    return "\n".join(parts)


def run_once(contestant: Contestant, cities: int, budget: int, seed: int) -> dict:
    """Play one map. Returns the result, the budget used, the curve and the time taken."""
    from .arena import BudgetSpent

    arena = Arena(cities, budget, seed)
    start = time.perf_counter()
    try:
        contestant.play(arena)
    except BudgetSpent:
        pass
    seconds = time.perf_counter() - start
    if arena.best_tour is None:
        raise RuntimeError(f"{contestant.id} measured no trip")
    return {"result": arena.best_length, "used": arena.used, "seconds": seconds,
            "curve": arena.curve()}


def check(contestant: Contestant) -> list[dict]:
    """The pre-match checks of rules §7 that can be run before the match.

    Check 6 (running time) needs the match's timings; see `check_time`.
    Check 1 (declaration matches the code) needs a human read; this only
    checks that a complete declaration is there.
    """
    out = []
    d = contestant.declaration
    complete = (contestant.name and contestant.idea and len(d) == 6
                and all(isinstance(v, tuple) and len(v) == 2 and v[1] for v in d.values()))
    out.append({"check": "declaration", "passed": bool(complete),
                "note": "complete; code reviewed by the referee" if complete
                else "declaration incomplete: name, idea and six (decider, how) entries needed"})
    problems = charter_problems(contestant.team, contestant.charter)
    out.append({"check": "charter", "passed": not problems, "note": "; ".join(problems) or "legal"})

    event = EVENTS[0]
    seed = training_seed(0, 0)
    budget = 2_000
    first = run_once(contestant, event.cities, budget, seed)
    out.append({"check": "budget", "passed": first["used"] <= budget,
                "note": f"used {first['used']} of {budget}; results are measured trips by construction"})

    source = source_of(contestant)
    hits = sorted({w for w in FORBIDDEN if re.search(_token(w), source)})
    out.append({"check": "hidden map", "passed": not hits,
                "note": "no access to the map except measurement" if not hits
                else f"forbidden names in code: {', '.join(hits)}"})

    second = run_once(contestant, event.cities, budget, seed)
    same = first["result"] == second["result"] and first["used"] == second["used"]
    out.append({"check": "determinism", "passed": same,
                "note": "same seed, same result" if same
                else f"{first['result']} vs {second['result']} on the same seed"})
    return out


def check_time(seconds: float, baseline_seconds: float) -> dict:
    """Check 6: total running time at most 5 times Baseline v1 on the same maps."""
    ratio = seconds / baseline_seconds if baseline_seconds > 0 else 0.0
    return {"check": "running time", "passed": ratio <= MAX_SLOWDOWN,
            "note": f"{ratio:.2f} times the baseline (limit {MAX_SLOWDOWN:g})"}
