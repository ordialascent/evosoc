"""Command line: python -m evosoc {baseline,rulers,choosing,reproduction,ownership,pressure} [options]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import ga
from .experiment import (BASELINE_BUDGET, PRESSURE_BUDGETS, PRESSURE_CITIES,
                         PRESSURE_REPRODUCTION, PRESSURE_RULERS, REPRODUCTION, run_baseline,
                         run_choosing, run_ownership, run_pressure, run_reproduction, run_rulers)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="evosoc")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("baseline", help="run the textbook GA over several seeds")
    b.add_argument("--cities", type=int, default=50)
    b.add_argument("--map-seed", type=int, default=1, help="which map of cities")
    b.add_argument("--seeds", type=int, default=30, help="how many independent runs")
    b.add_argument("--generations", type=int, default=ga.Config.generations)
    b.add_argument("--population", type=int, default=ga.Config.population)
    b.add_argument("--out", type=Path, default=Path("results"))

    r = sub.add_parser("rulers", help="experiment 1: the k fittest rule and always survive")
    r.add_argument("--rulers", type=int, nargs="+", default=[1, 2, 5, 20, 50],
                   help="sizes of the ruling class to compare (2 is the baseline)")
    r.add_argument("--cities", type=int, default=50)
    r.add_argument("--map-seed", type=int, default=1, help="which map of cities")
    r.add_argument("--seeds", type=int, default=30, help="how many independent runs")
    r.add_argument("--population", type=int, default=ga.Config.population)
    r.add_argument("--budget", type=int, default=BASELINE_BUDGET,
                   help="trips each society may measure in total")
    r.add_argument("--out", type=Path, default=Path("results"))

    c = sub.add_parser("choosing", help="experiment 2: rulers chosen by merit, birth or lottery")
    c.add_argument("--rulers", type=int, nargs="+", default=[5, 20],
                   help="sizes of the ruling class to compare")
    c.add_argument("--ways", nargs="+", choices=ga.RULER_CHOICES, default=list(ga.RULER_CHOICES),
                   help="how the rulers are chosen")
    c.add_argument("--cities", type=int, default=50)
    c.add_argument("--map-seed", type=int, default=1, help="which map of cities")
    c.add_argument("--seeds", type=int, default=30, help="how many independent runs")
    c.add_argument("--population", type=int, default=ga.Config.population)
    c.add_argument("--budget", type=int, default=BASELINE_BUDGET,
                   help="trips each society may measure in total")
    c.add_argument("--out", type=Path, default=Path("results"))

    p = sub.add_parser("reproduction",
                       help="experiment 3: how much damage making a child does, and its effect")
    p.add_argument("--cities", type=int, default=50)
    p.add_argument("--map-seed", type=int, default=1, help="which map of cities")
    p.add_argument("--seeds", type=int, default=30, help="how many independent runs")
    p.add_argument("--budget", type=int, default=BASELINE_BUDGET,
                   help="trips each society may measure in total")
    p.add_argument("--out", type=Path, default=Path("results"))

    o = sub.add_parser("ownership",
                       help="experiment 4: earnings pooled, kept or inherited")
    o.add_argument("--cities", type=int, default=50)
    o.add_argument("--map-seed", type=int, default=1, help="which map of cities")
    o.add_argument("--seeds", type=int, default=30, help="how many independent runs")
    o.add_argument("--budget", type=int, default=BASELINE_BUDGET,
                   help="trips each society may measure in total")
    o.add_argument("--out", type=Path, default=Path("results"))

    t = sub.add_parser("pressure",
                       help="experiment 5: does concentrated power win when time is short?")
    t.add_argument("--cities", type=int, nargs="+", default=list(PRESSURE_CITIES),
                   help="map sizes to compare")
    t.add_argument("--budgets", type=int, nargs="+", default=list(PRESSURE_BUDGETS),
                   help="budgets, in baseline generations")
    t.add_argument("--rulers", type=int, nargs="+", default=list(PRESSURE_RULERS),
                   help="sizes of the ruling class to compare (0 = no rulers)")
    t.add_argument("--ways", nargs="+", choices=list(REPRODUCTION),
                   default=list(PRESSURE_REPRODUCTION), help="ways of making children")
    t.add_argument("--map-seed", type=int, default=1, help="which map of cities")
    t.add_argument("--seeds", type=int, default=30, help="how many independent runs")
    t.add_argument("--out", type=Path, default=Path("results"))
    t.add_argument("--name", default="pressure", help="file name of the results (no extension)")
    t.add_argument("--title", help="title of the chart")
    args = parser.parse_args(argv)

    if args.command == "baseline":
        config = ga.Config(population=args.population, generations=args.generations)
        summary = run_baseline(args.out, args.cities, args.map_seed, args.seeds, config)
        print(json.dumps(summary, indent=2))
    elif args.command == "rulers":
        if any(not 1 <= k < args.population for k in args.rulers):
            parser.error("every ruler count must be at least 1 and below the population size")
        summary = run_rulers(args.out, tuple(sorted(set(args.rulers))), args.cities,
                             args.map_seed, args.seeds, args.population, args.budget)
        print(json.dumps(summary, indent=2))
    elif args.command == "choosing":
        if any(not 1 <= k < args.population for k in args.rulers):
            parser.error("every ruler count must be at least 1 and below the population size")
        summary = run_choosing(args.out, tuple(sorted(set(args.rulers))),
                               tuple(dict.fromkeys(args.ways)), args.cities, args.map_seed,
                               args.seeds, args.population, args.budget)
        print(json.dumps(summary, indent=2))
    elif args.command == "reproduction":
        summary = run_reproduction(args.out, args.cities, args.map_seed, args.seeds,
                                   budget=args.budget)
        print(json.dumps(summary, indent=2))
    elif args.command == "ownership":
        summary = run_ownership(args.out, args.cities, args.map_seed, args.seeds,
                                budget=args.budget)
        print(json.dumps(summary, indent=2))
    elif args.command == "pressure":
        rulers = tuple(sorted(set(args.rulers)))
        if len(rulers) < 2 or any(not 0 <= k < ga.Config.population for k in rulers):
            parser.error("give at least two ruler counts, each from 0 to below the population size")
        summary = run_pressure(args.out, tuple(sorted(set(args.cities))),
                               tuple(sorted(set(args.budgets))), rulers,
                               ways=tuple(dict.fromkeys(args.ways)), map_seed=args.map_seed,
                               seeds=args.seeds, name=args.name, title=args.title)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
