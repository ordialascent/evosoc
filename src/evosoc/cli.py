"""Command line: ``evosoc run <config.toml>`` and ``evosoc analyze <results/name>``."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="evosoc")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an experiment (resumes if partly done)")
    run.add_argument("config", type=Path)
    run.add_argument("--results", type=Path, default=Path("results"))
    run.add_argument(
        "-j", "--jobs", type=int, default=None, help="parallel runs (default: all CPUs)"
    )
    run.add_argument("--no-analyze", action="store_true", help="skip the report afterwards")

    analyze = sub.add_parser("analyze", help="write figures and tables for an experiment")
    analyze.add_argument("experiment_dir", type=Path)

    args = parser.parse_args(argv)
    from .analysis import analyze as analyze_experiment
    from .experiment import ExperimentConfig, run_experiment

    if args.command == "run":
        exp_dir = run_experiment(ExperimentConfig.from_toml(args.config), args.results, args.jobs)
        if not args.no_analyze:
            print(f"report: {analyze_experiment(exp_dir)}")
    else:
        print(f"report: {analyze_experiment(args.experiment_dir)}")


if __name__ == "__main__":
    main()
