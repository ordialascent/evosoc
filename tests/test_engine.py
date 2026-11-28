import json
from pathlib import Path

import numpy as np
import pytest

from evosoc import referee
from evosoc.__main__ import main
from evosoc.arena import Arena, BudgetSpent, curve_marks
from evosoc.contestant import Contestant, all_contestants, load
from evosoc.match import run_match, train
from evosoc.rules import (EVENT_BY_KEY, FIRST_MATCH_SEED, MATTERS, charter_problems,
                          match_seed, training_seed)
from evosoc.tsp import distance_matrix, make_cities, tour_length


# --- Arena and budget -------------------------------------------------------

def test_measure_is_true_length_and_costs_one():
    arena = Arena(20, 10, seed=3)
    tour = np.arange(20)
    dist = distance_matrix(make_cities(20, 3))
    assert arena.measure(tour) == pytest.approx(tour_length(dist, tour))
    assert arena.used == 1 and arena.remaining == 9


def test_budget_runs_out_exactly():
    arena = Arena(10, 25, seed=0)
    rng = np.random.default_rng(0)
    arena.measure_many([rng.permutation(10) for _ in range(20)])
    with pytest.raises(BudgetSpent):
        arena.measure_many([rng.permutation(10) for _ in range(20)])
    assert arena.used == 25
    with pytest.raises(BudgetSpent):
        arena.measure(np.arange(10))
    assert arena.used == 25


def test_best_is_a_measured_trip():
    arena = Arena(15, 50, seed=1)
    rng = np.random.default_rng(1)
    tours = np.array([rng.permutation(15) for _ in range(50)])
    lengths = arena.measure_many(tours)
    assert arena.best_length == lengths.min()
    assert (arena.best_tour == tours[np.argmin(lengths)]).all()


def test_invalid_tours_are_refused_and_free():
    arena = Arena(5, 10, seed=0)
    for bad in ([0, 1, 2, 3, 3], [0, 1, 2, 3], [[0, 1, 2, 3, 5]]):
        with pytest.raises(ValueError):
            arena.measure_many(np.atleast_2d(bad))
    assert arena.used == 0


def test_curve_tracks_best_so_far():
    arena = Arena(10, 1000, seed=2)
    rng = np.random.default_rng(2)
    for _ in range(10):
        arena.measure_many([rng.permutation(10) for _ in range(100)])
    curve = arena.curve()
    assert [m for m, _ in curve] == curve_marks(1000).tolist()
    values = [v for _, v in curve]
    assert values == sorted(values, reverse=True)
    assert values[-1] == arena.best_length


# --- Rules ------------------------------------------------------------------

def test_seeds():
    assert match_seed(1, 0, 0) == 1_001_000
    assert match_seed(3, 2, 29) == 1_003_229
    assert training_seed(2, 99_999) < FIRST_MATCH_SEED
    with pytest.raises(ValueError):
        match_seed(0, 0, 0)


def test_charters():
    law = {m: "law" for m in MATTERS}
    assert charter_problems("neutral", law) == []
    assert charter_problems("centralizers", law)  # needs a Board
    assert charter_problems("centralizers", {**law, "pairing": "board"}) == []
    assert charter_problems("centralizers", {**law, "pairing": "board", "contact": "member"})
    assert charter_problems("decentralizers", {**law, "contact": "member"}) == []
    assert charter_problems("decentralizers", {**law, "contact": "member", "survival": "board"})
    assert charter_problems("neutral", {**law, "contact": "member"})
    assert charter_problems("centralizers", {"survival": "board"})  # matters missing


# --- Scoring ----------------------------------------------------------------

def test_sign_test_threshold_for_30_maps():
    assert referee.sign_test(21, 9) < 0.05
    assert referee.sign_test(20, 10) > 0.05
    assert referee.sign_test(0, 0) == 1.0


def test_event_and_match_scoring():
    a = [1.0] * 22 + [2.0] * 8
    b = [2.0] * 22 + [1.0] * 8
    ev = referee.score_event(a, b)
    assert ev["maps_won"] == {"a": 22, "b": 8, "tied": 0} and ev["winner"] == "a"
    tied = referee.score_event([1.0] * 30, [1.0 + 1e-12] * 30)
    assert tied["maps_won"]["tied"] == 30 and tied["winner"] is None
    assert referee.score_match(["a", None, "b"])["winner"] is None
    assert referee.score_match(["a", None, None]) == {
        "events_won": {"a": 1, "b": 0}, "score": "1–0", "winner": "a"}


# --- Contestants and fair play -------------------------------------------------

def test_every_contestant_declares_a_legal_charter():
    found = all_contestants()
    assert "baseline_v0" in found
    for cid, c in found.items():
        assert c.id == cid
        assert set(c.declaration) == set(MATTERS), cid
        assert c.idea, cid
        assert charter_problems(c.team, c.charter) == [], cid


def test_baselines_pass_the_checks():
    for cid in ("baseline_v0", "baseline_v1"):
        assert all(row["passed"] for row in referee.check(load(cid))), cid


def test_round0_froze_the_toolbox_and_the_baseline():
    from evosoc import arena, match, tuning
    assert arena.TOOLBOX.describe() == {"population": 30, "crossover": "edge_crossover",
                                        "mutation": "inversion_mutation"}
    assert match.BASELINE == "baseline_v1"
    v1 = load("baseline_v1")
    frozen = {"population": arena.TOOLBOX.population, "crossover": arena.TOOLBOX.crossover.__name__,
              "elite": v1.elite, "contest_size": v1.contest_size,
              "crossover_rate": v1.crossover_rate, "mutation_rate": v1.mutation_rate}
    record = json.loads((Path(__file__).resolve().parents[1] / "data/tuning/baseline_v1.json").read_text())
    assert record["winner"] == frozen
    tried = tuning.settings_to_try()
    assert tried == tuning.settings_to_try() and tried[0] == tuning.UNTUNED
    assert frozen in tried
    one = tuning.run_setting(frozen, 30, 1500, seed=5)
    assert one == tuning.run_setting(frozen, 30, 1500, seed=5)
    assert one == referee.run_once(v1, 30, 1500, seed=5)["result"]


def test_runs_are_deterministic_and_use_the_whole_budget():
    b = load("baseline_v0")
    one = referee.run_once(b, 30, 1500, seed=5)
    two = referee.run_once(b, 30, 1500, seed=5)
    other = referee.run_once(b, 30, 1500, seed=6)
    assert one["result"] == two["result"] and one["curve"] == two["curve"]
    assert one["result"] != other["result"]
    assert one["used"] == 1500


class Peeker(Contestant):
    name, version, team, idea = "Peeker", 1, "neutral", "reads the map"
    declaration = {m: ("law", "-") for m in MATTERS}

    def play(self, arena):
        arena.measure(np.arange(arena.cities))
        return arena._lengths


class Wobbly(Contestant):
    name, version, team, idea = "Wobbly", 1, "neutral", "not seeded"
    declaration = {m: ("law", "-") for m in MATTERS}
    calls = 0

    def play(self, arena):
        Wobbly.calls += 1
        tour = np.arange(arena.cities)
        tour[1:] = np.random.default_rng(Wobbly.calls).permutation(tour[1:])
        arena.measure(tour)


def test_referee_catches_map_reading_and_randomness(monkeypatch):
    monkeypatch.setattr(referee, "source_of", lambda c: __import__("inspect").getsource(type(c)))
    peek = {r["check"]: r["passed"] for r in referee.check(Peeker())}
    assert peek["hidden map"] is False
    wobble = {r["check"]: r["passed"] for r in referee.check(Wobbly())}
    assert wobble["determinism"] is False


# --- Matches ---------------------------------------------------------------

def test_dummy_match_end_to_end(tmp_path):
    out = tmp_path / "001.json"
    main(["match", "1", "baseline_v0", "baseline_v0", "--maps", "3", "--event", "sprint",
          "--workers", "1", "--out", str(out)])
    record = json.loads(out.read_text())
    assert record["contestants"] == {"a": "baseline_v0", "b": "baseline_v0", "baseline": "baseline_v1"}
    ev = record["events"][0]
    assert ev["maps_won"] == {"a": 0, "b": 0, "tied": 3}
    assert record["winner"] is None and record["score"] == "0–0" and record["forfeit"] is None
    assert [m["seed"] for m in ev["maps"]] == [1_001_000, 1_001_001, 1_001_002]
    assert all(m["a"]["used"] == EVENT_BY_KEY["sprint"].budget for m in ev["maps"])
    assert {c["check"] for c in record["checks"]["a"]} == {
        "declaration", "charter", "budget", "hidden map", "determinism", "running time"}


def test_match_records_are_repeatable():
    def results(rec):
        return [[m[r]["result"] for r in ("a", "b", "baseline")] + [m["yardstick"]]
                for ev in rec["events"] for m in ev["maps"]]
    events = (EVENT_BY_KEY["sprint"],)
    one = run_match(2, "baseline_v0", "baseline_v0", maps=2, events=events, workers=1)
    two = run_match(2, "baseline_v0", "baseline_v0", maps=2, events=events, workers=2)
    assert results(one) == results(two)


def test_training_stays_below_match_seeds():
    rec = train("baseline_v0", maps=2, events=(EVENT_BY_KEY["sprint"],), workers=1)
    assert all(m["seed"] < FIRST_MATCH_SEED for ev in rec["events"] for m in ev["maps"])


def test_stopped_and_resumed_runs_give_the_same_record(tmp_path):
    from evosoc.rules import EVENTS

    def results(record):
        return [[(row["yardstick"], row["a"]["result"], row["a"]["curve"], row["b"]["result"])
                 for row in ev["maps"]] for ev in record["events"]]

    events = EVENTS[:1]
    whole = train("baseline_v0", "baseline_v1", events, maps=3, workers=1)
    store = tmp_path / "runs"
    first = train("baseline_v0", "baseline_v1", events, maps=3, workers=2, store=store)
    kept = sorted(store.glob("*.json"))
    assert len(kept) == 3 * 3  # a yardstick and two distinct contestants per map
    for path in kept[::2]:
        path.unlink()  # as if the run had been stopped part way
    lines = []
    resumed = train("baseline_v0", "baseline_v1", events, maps=3, workers=2, store=store,
                    log=lines.append)
    assert results(whole) == results(first) == results(resumed)
    assert len(lines) == len(kept[::2]) and lines[-1].startswith("9/9 runs done")
