"""Command line: python -m evosoc {baseline,rulers,choosing} [options]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import ga
from .experiment import BASELINE_BUDGET, run_baseline, run_choosing, run_rulers


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


if __name__ == "__main__":
    main()
