"""Command line.

    python -m evosoc contestants              list contestants, write data/contestants/*.json
    python -m evosoc check <id>               the referee's pre-match checks for one contestant
    python -m evosoc train <id> [--against <id>] [--event sprint] [--maps 10]
    python -m evosoc match <N> <a> <b>        referee match N, write data/matches/NNN.json
    python -m evosoc tune                     Round 0: tune the textbook GA, write data/tuning/baseline_v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contestant import all_contestants, load
from .match import BASELINE, run_match, save, train
from .referee import check
from .rules import EVENT_BY_KEY, EVENTS, MAPS_PER_EVENT

DATA = Path("data")


def _events(keys):
    return tuple(EVENT_BY_KEY[k] for k in keys) if keys else EVENTS


def _summary(record: dict) -> str:
    names = record["contestants"]
    lines = [f"a = {names['a']}, b = {names['b']}, alongside: {names['baseline']}"]
    for ev in record["events"]:
        won = ev["maps_won"]
        g = ev["median_gap"]
        verdict = names[ev["winner"]] if ev["winner"] else "draw"
        lines.append(
            f"{ev['name']:<12} maps {won['a']}-{won['b']} (tied {won['tied']}), p = {ev['p_value']:.3f}: "
            f"{verdict:<16} median gap a {g['a']:+.2f}%, b {g['b']:+.2f}%, baseline {g['baseline']:+.2f}%")
    if "score" in record:
        w = record["winner"]
        lines.append(f"Match {record['match']}: {record['score']}, "
                     f"{'winner ' + names[w] if w else 'draw'}")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="evosoc", description="Centralizers vs Decentralizers.")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("contestants", help="list contestants and write their declarations as JSON")
    c.add_argument("--out", type=Path, default=DATA / "contestants")

    k = sub.add_parser("check", help="run the referee's pre-match checks on a contestant")
    k.add_argument("contestant")

    t = sub.add_parser("train", help="play on training maps (seeds below 1,000,000)")
    t.add_argument("contestant")
    t.add_argument("--against", default=BASELINE)
    t.add_argument("--event", nargs="+", choices=list(EVENT_BY_KEY), help="default: all three")
    t.add_argument("--maps", type=int, default=10, help="maps per event")
    t.add_argument("--first-map", type=int, default=0, help="first training map")
    t.add_argument("--workers", type=int, default=None)
    t.add_argument("--out", type=Path, help="also write the full record as JSON")

    m = sub.add_parser("match", help="referee a match on fresh match seeds and record it")
    m.add_argument("number", type=int)
    m.add_argument("a")
    m.add_argument("b")
    m.add_argument("--maps", type=int, default=MAPS_PER_EVENT,
                   help="maps per event (fewer only for trials; official matches use 30)")
    m.add_argument("--event", nargs="+", choices=list(EVENT_BY_KEY), help="default: all three")
    m.add_argument("--workers", type=int, default=None)
    m.add_argument("--out", type=Path, help="default: data/matches/NNN.json")

    u = sub.add_parser("tune", help="Round 0: seeded search for the baseline's settings (training maps)")
    u.add_argument("--workers", type=int, default=None)
    u.add_argument("--out", type=Path, default=DATA / "tuning" / "baseline_v1.json")

    args = parser.parse_args(argv)
    if args.command == "contestants":
        args.out.mkdir(parents=True, exist_ok=True)
        for cid, contestant in all_contestants().items():
            (args.out / f"{cid}.json").write_text(json.dumps(contestant.describe(), indent=1) + "\n")
            print(f"{cid:<28} {contestant.team:<15} {contestant.idea}")
    elif args.command == "check":
        ok = True
        for row in check(load(args.contestant)):
            ok &= row["passed"]
            print(f"{'PASS' if row['passed'] else 'FAIL'}  {row['check']:<12} {row['note']}")
        raise SystemExit(0 if ok else 1)
    elif args.command == "train":
        record = train(args.contestant, args.against, _events(args.event), args.maps,
                       args.first_map, args.workers)
        print(_summary(record))
        if args.out:
            save(record, args.out)
    elif args.command == "match":
        record = run_match(args.number, args.a, args.b, args.maps, _events(args.event), args.workers)
        out = args.out or DATA / "matches" / f"{args.number:03d}.json"
        save(record, out)
        print(_summary(record))
        print(f"written to {out}")

    elif args.command == "tune":
        from .tuning import tune
        record = tune(args.workers, log=lambda line: print(line, flush=True))
        save(record, args.out)
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
