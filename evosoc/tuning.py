"""Round 0: tuning the textbook genetic algorithm for the arena (training seeds only).

A seeded random search in two stages, so the outcome can be rerun exactly:

1. **Screen.** The untuned settings (Baseline v0) plus `SCREEN` settings drawn
   at random from `SPACE` each play `SCREEN_MAPS` training maps per event.
2. **Final.** The best `FINALISTS` of the screen and Baseline v0 play
   `FINAL_MAPS` fresh training maps per event. The best of the final is
   frozen as Baseline v1.

A setting's score is its mean gap to the yardstick, averaged over the three
events with equal weight (lower is better). All maps are training maps
(seeds below 1,000,000); no match seed is ever used.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from statistics import mean, median

import numpy as np

from . import operators
from .arena import Arena, BudgetSpent, Toolbox
from .contestants.baseline.baseline_v0 import Baseline
from .match import gap, yardstick
from .referee import score_event
from .rules import EVENTS, training_seed

SEARCH_SEED = 2026
SCREEN = 31
SCREEN_MAPS = 6
FINALISTS = 4
FINAL_MAPS = 20
FINAL_FIRST_MAP = 100

SPACE = {
    "population": (30, 50, 100, 200, 400),
    "crossover": ("order_crossover", "edge_crossover"),
    "elite": (0, 1, 2, 5, 10),
    "contest_size": (2, 3, 5, 7),
    "crossover_rate": (0.5, 0.7, 0.9, 1.0),
    "mutation_rate": (0.1, 0.3, 0.6, 1.0),
}
UNTUNED = {"population": 100, "crossover": "order_crossover", "elite": 2, "contest_size": 3,
           "crossover_rate": 0.9, "mutation_rate": 0.3}


def settings_to_try() -> list[dict]:
    """Baseline v0's settings first, then `SCREEN` distinct random draws from `SPACE`."""
    rng = np.random.default_rng(SEARCH_SEED)
    out, seen = [UNTUNED], {tuple(UNTUNED.values())}
    while len(out) < 1 + SCREEN:
        s = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        if s["elite"] >= s["population"]:
            continue
        if tuple(s.values()) not in seen:
            seen.add(tuple(s.values()))
            out.append(s)
    return out


def label(s: dict) -> str:
    x = "OX" if s["crossover"] == "order_crossover" else "ERX"
    return (f"pop {s['population']}, {x}, elite {s['elite']}, contest {s['contest_size']}, "
            f"cross {s['crossover_rate']:g}, mutate {s['mutation_rate']:g}")


def run_setting(s: dict, cities: int, budget: int, seed: int) -> float:
    """Play one training map with the textbook GA under settings `s`; returns the result."""
    toolbox = Toolbox(population=s["population"], crossover=getattr(operators, s["crossover"]),
                      mutation=operators.inversion_mutation)
    ga = Baseline()
    ga.elite, ga.contest_size = s["elite"], s["contest_size"]
    ga.crossover_rate, ga.mutation_rate = s["crossover_rate"], s["mutation_rate"]
    arena = Arena(cities, budget, seed, toolbox)
    try:
        ga.play(arena)
    except BudgetSpent:
        pass
    return arena.best_length


def _job(job):
    if job[0] == "yardstick":
        return yardstick(*job[1:])
    _, key, cities, budget, seed = job
    return run_setting(dict(key), cities, budget, seed)


def evaluate(settings: list[dict], maps: range, workers: int | None = None) -> list[dict]:
    """Every setting on the same training maps. Returns per-setting, per-event results."""
    jobs = []
    for e, ev in enumerate(EVENTS):
        for i in maps:
            seed = training_seed(e, i)
            jobs.append(("yardstick", ev.cities, seed))
            for s in settings:
                jobs.append(("run", tuple(s.items()), ev.cities, ev.budget, seed))
    with ProcessPoolExecutor(workers or os.cpu_count() or 1) as pool:
        done = dict(zip(jobs, pool.map(_job, jobs, chunksize=1)))
    out = []
    for s in settings:
        events = {}
        for e, ev in enumerate(EVENTS):
            results, gaps = [], []
            for i in maps:
                seed = training_seed(e, i)
                r = done[("run", tuple(s.items()), ev.cities, ev.budget, seed)]
                results.append(r)
                gaps.append(gap(r, done[("yardstick", ev.cities, seed)]))
            events[ev.key] = {"results": results, "mean_gap": mean(gaps),
                              "median_gap": median(gaps)}
        out.append({"settings": s, "label": label(s), "events": events,
                    "score": mean(v["mean_gap"] for v in events.values())})
    return out


def _summary(rows: list[dict]) -> list[dict]:
    """Rows without the per-map results, for the record."""
    return [{**r, "events": {e: {k: x for k, x in d.items() if k != "results"}
                             for e, d in r["events"].items()}} for r in rows]


def tune(workers: int | None = None, log=print) -> dict:
    """Run both stages and return the full tuning record."""
    started = time.time()
    candidates = settings_to_try()
    screen_maps = range(SCREEN_MAPS)
    log(f"screen: {len(candidates)} settings x {SCREEN_MAPS} maps x {len(EVENTS)} events")
    screen = evaluate(candidates, screen_maps, workers)
    ranked = sorted(screen, key=lambda r: r["score"])
    for r in ranked:
        log(f"  {r['score']:+7.2f}%  {r['label']}")

    finalists = [r["settings"] for r in ranked[:FINALISTS]]
    if UNTUNED not in finalists:
        finalists.append(UNTUNED)
    final_maps = range(FINAL_FIRST_MAP, FINAL_FIRST_MAP + FINAL_MAPS)
    log(f"final: {len(finalists)} settings x {FINAL_MAPS} fresh maps x {len(EVENTS)} events")
    final = sorted(evaluate(finalists, final_maps, workers), key=lambda r: r["score"])
    for r in final:
        log(f"  {r['score']:+7.2f}%  {r['label']}")

    winner = final[0]
    untuned = next(r for r in final if r["settings"] == UNTUNED)
    versus = {}
    for ev in EVENTS:
        versus[ev.key] = score_event(winner["events"][ev.key]["results"],
                                     untuned["events"][ev.key]["results"])
    return {
        "method": {"search_seed": SEARCH_SEED, "space": {k: list(v) for k, v in SPACE.items()},
                   "screen_settings": len(candidates), "screen_maps": list(screen_maps),
                   "finalists": len(finalists), "final_maps": [final_maps.start, final_maps.stop - 1],
                   "score": "mean gap to the yardstick (%), averaged over the three events"},
        "screen": _summary(ranked),
        "final": _summary(final),
        "winner": winner["settings"],
        "winner_vs_untuned": versus,
        "minutes": round((time.time() - started) / 60, 1),
    }

