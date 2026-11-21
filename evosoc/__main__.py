"""Command line: python -m evosoc baseline [options]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import ga
from .experiment import run_baseline


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="evosoc")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("baseline", help="run the textbook GA over several seeds")
    b.add_argument("--cities", type=int, default=50)
    b.add_argument("--map-seed", type=int, default=1, help="which map of cities")
    b.add_argument("--seeds", type=int, default=10, help="how many independent runs")
    b.add_argument("--generations", type=int, default=ga.Config.generations)
    b.add_argument("--population", type=int, default=ga.Config.population)
    b.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    if args.command == "baseline":
        config = ga.Config(population=args.population, generations=args.generations)
        summary = run_baseline(args.out, args.cities, args.map_seed, args.seeds, config)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
