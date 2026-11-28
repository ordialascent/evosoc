"""Running matches and training sessions, in parallel over maps."""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median
from typing import Callable

from . import referee
from .arena import TOOLBOX
from .contestant import load
from .rules import (EVENTS, MAPS_PER_EVENT, RULES_VERSION, SEASON, SEASON_NAME, match_seed,
                    training_seed)
from .tsp import distance_matrix, make_cities, reference_tour, tour_length

BASELINE = "baseline_v1"


def yardstick(cities: int, seed: int) -> float:
    """The referee's yardstick for a map: greedy trip, untangled by 2-opt. Never shown to contestants."""
    dist = distance_matrix(make_cities(cities, seed))
    return tour_length(dist, reference_tour(dist))


def _job(job):
    kind, *args = job
    if kind == "yardstick":
        return yardstick(*args)
    contestant_id, cities, budget, seed = args
    return referee.run_once(load(contestant_id), cities, budget, seed)


def _stored(store: Path, job) -> Path:
    """Where a finished job is kept: named by the job and the arena it ran in."""
    key = json.dumps([list(job), RULES_VERSION, TOOLBOX.describe()])
    return store / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"


def _run_all(jobs: list, workers: int | None, store: Path | None = None,
             log: Callable[[str], None] | None = None, run: Callable = _job) -> dict:
    """Run each distinct job once; returns {job: result}.

    With a `store`, every finished job is written to it at once, and jobs
    already there are read back instead of run again. Every job is seeded
    and independent, so a run that was stopped and resumed gives exactly the
    results of one that never stopped. `run` does one job (a module-level
    function, so worker processes can call it); a job is a tuple whose first
    item is its kind ("run" jobs: kind, ..., cities, budget, seed).
    """
    unique = list(dict.fromkeys(jobs))
    results = {}
    if store is not None:
        store.mkdir(parents=True, exist_ok=True)
        for job in unique:
            path = _stored(store, job)
            if path.exists():
                results[job] = json.loads(path.read_text())["result"]
    todo = [j for j in unique if j not in results]
    start, total, resumed = time.perf_counter(), len(unique), len(results)

    def finished(job, result):
        results[job] = result
        if store is not None:
            path = _stored(store, job)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"job": list(job), "result": result}) + "\n")
            tmp.replace(path)
        if log:
            done = len(results) - resumed
            eta = (time.perf_counter() - start) / done * (len(todo) - done)
            log(f"{len(results)}/{total} runs done, about {eta / 60:.0f} min to go")

    workers = workers or os.cpu_count() or 1
    if workers == 1:
        for job in todo:
            finished(job, run(job))
    else:
        with ProcessPoolExecutor(workers) as pool:
            # Longest jobs first, so the last minutes are not one long run alone.
            order = sorted(todo, key=lambda j: -(j[-3] * j[-2] if j[0] == "run" else 0))
            futures = {pool.submit(run, job): job for job in order}
            for future in as_completed(futures):
                finished(futures[future], future.result())
    return {j: results[j] for j in unique}


def gap(result: float, stick: float) -> float:
    """Percent longer than the yardstick (negative: shorter)."""
    return 100 * (result / stick - 1)


def play(a: str, b: str, seeds_for, events=EVENTS, maps: int = MAPS_PER_EVENT,
         baseline: str = BASELINE, workers: int | None = None, store: Path | None = None,
         log: Callable[[str], None] | None = None) -> dict:
    """Play `a` against `b` (and `baseline` alongside) on the maps `seeds_for(event_index, map)`."""
    roles = {"a": a, "b": b, "baseline": baseline}
    jobs = []
    for e, event in ((EVENTS.index(ev), ev) for ev in events):
        for i in range(maps):
            seed = seeds_for(e, i)
            jobs.append(("yardstick", event.cities, seed))
            for cid in roles.values():
                jobs.append(("run", cid, event.cities, event.budget, seed))
    results = _run_all(jobs, workers, store, log)

    out_events, seconds = [], {r: 0.0 for r in roles}
    for e, event in ((EVENTS.index(ev), ev) for ev in events):
        rows = []
        for i in range(maps):
            seed = seeds_for(e, i)
            stick = results[("yardstick", event.cities, seed)]
            row = {"map": i, "seed": seed, "yardstick": stick}
            for role, cid in roles.items():
                r = results[("run", cid, event.cities, event.budget, seed)]
                row[role] = {**r, "gap": gap(r["result"], stick)}
                seconds[role] += r["seconds"]
            rows.append(row)
        scored = referee.score_event([r["a"]["result"] for r in rows], [r["b"]["result"] for r in rows])
        out_events.append({
            "key": event.key, "name": event.name, "cities": event.cities, "budget": event.budget,
            **scored,
            "median_gap": {role: median(r[role]["gap"] for r in rows) for role in roles},
            "maps": rows,
        })
    return {"contestants": roles, "events": out_events, "seconds": seconds}


def run_match(number: int, a: str, b: str, maps: int = MAPS_PER_EVENT, events=EVENTS,
              workers: int | None = None, baseline: str = BASELINE, store: Path | None = None,
              log: Callable[[str], None] | None = None) -> dict:
    """Referee a full match (rules §4, §5, §7) and return its record."""
    if number < 1:
        raise ValueError("matches are numbered from 1")
    checks = {"a": referee.check(load(a)), "b": referee.check(load(b))}
    record = play(a, b, lambda e, i: match_seed(number, e, i), events, maps, baseline, workers,
                  store, log)
    for role in ("a", "b"):
        checks[role].append(referee.check_time(record["seconds"][role], record["seconds"]["baseline"]))
    failed = {role: [c["check"] for c in checks[role] if not c["passed"]] for role in ("a", "b")}
    verdict = referee.score_match([ev["winner"] for ev in record["events"]])
    forfeit = None
    if failed["a"] or failed["b"]:
        forfeit = {role: failed[role] for role in ("a", "b") if failed[role]}
        if failed["a"] and failed["b"]:
            verdict = {**verdict, "winner": None, "score": "forfeit–forfeit"}
        else:
            loser = "a" if failed["a"] else "b"
            winner = "b" if loser == "a" else "a"
            verdict = {**verdict, "winner": winner, "score": f"forfeit by {loser}"}
    return {
        "match": number, "season": SEASON, "season_name": SEASON_NAME, "rules_version": RULES_VERSION,
        "toolbox": TOOLBOX.describe(), "maps_per_event": maps,
        "contestants": record["contestants"], **verdict, "forfeit": forfeit, "checks": checks,
        "seconds": record["seconds"], "events": record["events"],
    }


def train(contestant: str, against: str = BASELINE, events=EVENTS, maps: int = 10,
          first_map: int = 0, workers: int | None = None, store: Path | None = None,
          log: Callable[[str], None] | None = None) -> dict:
    """Play `contestant` against `against` on training maps (seeds below the match seeds)."""
    return play(contestant, against, lambda e, i: training_seed(e, first_map + i), events, maps,
                baseline=BASELINE, workers=workers, store=store, log=log)


def save(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
