"""Experiment configs and the runner.

An experiment is a full grid: every variant x every problem x every seed,
each run given the same evaluation budget. That one shape covers both
directions the project may take:

* many variants compared against one baseline on a few problems, and
* a baseline and one variant compared across many problems.

Output layout (one tidy row per generation, resumable)::

    results/<experiment>/
        config.toml            copy of the config that produced the results
        manifest.json          git commit, package versions, timestamps
        runs/<variant>/<problem>/seed<k>/
            generations.parquet    population metrics, one row per generation
            improvements.parquet   (evaluation, value) at every best-so-far improvement
            summary.json           final best genome etc.; marks the run as finished
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
import tomllib
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__, metrics
from .algorithms import BudgetExhausted, Context, Population, make_algorithm
from .problems import make_problem


@dataclass(frozen=True)
class Variant:
    name: str
    algorithm: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    #: Fitness evaluations per run: the unit of cost shared by all variants.
    budget: int
    seeds: list[int]
    problems: list[dict[str, Any]]
    variants: list[Variant]
    #: Variant every other variant is compared against in the analysis.
    baseline: str
    #: Upper bound on the number of pairs sampled for the diversity metric.
    diversity_pairs: int = 200
    path: Path | None = None

    @classmethod
    def from_toml(cls, path: str | Path, check_algorithms: bool = True) -> ExperimentConfig:
        """Load a config. ``check_algorithms=False`` skips building the variants,
        so finished results can be analysed after a variant's code is gone."""
        path = Path(path)
        raw = tomllib.loads(path.read_text())
        seeds = raw["seeds"]
        variants = [Variant(**v) for v in raw["variants"]]
        cfg = cls(
            name=raw["name"],
            budget=int(raw["budget"]),
            seeds=list(range(seeds)) if isinstance(seeds, int) else list(seeds),
            problems=raw["problems"],
            variants=variants,
            baseline=raw.get("baseline", variants[0].name),
            diversity_pairs=raw.get("diversity_pairs", 200),
            path=path,
        )
        names = [v.name for v in variants]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate variant names in {path}")
        if cfg.baseline not in names:
            raise ValueError(f"baseline {cfg.baseline!r} is not one of the variants {names}")
        # Fail now, not hours into a run: every variant and problem must build.
        for v in variants if check_algorithms else []:
            try:
                make_algorithm(v.algorithm, v.params)
            except (TypeError, ValueError) as e:
                raise ValueError(f"{path}: variant {v.name!r}: {e}") from None
        for p in cfg.problems:
            make_problem(p)
        return cfg


@dataclass(frozen=True)
class RunSpec:
    variant: Variant
    problem: dict[str, Any]
    problem_name: str
    seed: int
    budget: int
    diversity_pairs: int
    out: Path  # run directory


def expand(cfg: ExperimentConfig, results_root: Path) -> list[RunSpec]:
    runs_dir = results_root / cfg.name / "runs"
    specs = []
    for problem in cfg.problems:
        pname = make_problem(problem).name
        for variant in cfg.variants:
            for seed in cfg.seeds:
                specs.append(
                    RunSpec(
                        variant=variant,
                        problem=problem,
                        problem_name=pname,
                        seed=seed,
                        budget=cfg.budget,
                        diversity_pairs=cfg.diversity_pairs,
                        out=runs_dir / variant.name / pname / f"seed{seed}",
                    )
                )
    return specs


def run_one(spec: RunSpec) -> Path:
    problem = make_problem(spec.problem)
    algorithm = make_algorithm(spec.variant.algorithm, spec.variant.params)

    # Seeds depend on (seed, problem) only, never on the variant: common
    # random numbers. zlib.crc32 because hash() of a str differs per process.
    root = np.random.SeedSequence([spec.seed, zlib.crc32(spec.problem_name.encode())])
    init_ss, algo_ss, metric_ss = root.spawn(3)
    metric_rng = np.random.default_rng(metric_ss)

    rows: list[dict[str, Any]] = []
    last: list[Population] = []

    def on_report(ctx: Context, pop: Population, extra: dict[str, float]) -> None:
        last[:] = [pop]
        rows.append(
            {
                "evaluations": ctx.evaluations,
                "generation": ctx.generation,
                "time_s": time.perf_counter() - ctx.start_time,
                "best_so_far": ctx.best_so_far,
                **metrics.snapshot(problem, pop, metric_rng, spec.diversity_pairs),
                **{f"x_{k}": v for k, v in extra.items()},
            }
        )

    ctx = Context(
        problem,
        spec.budget,
        rng=np.random.default_rng(algo_ss),
        init_rng=np.random.default_rng(init_ss),
        on_report=on_report,
    )
    try:
        algorithm.run(ctx)
        raise RuntimeError(f"{spec.variant.algorithm} returned before exhausting its budget")
    except BudgetExhausted:
        pass
    # Close the curve at the full budget with the last reported population.
    if last and rows[-1]["evaluations"] < ctx.evaluations:
        ctx.report(last[0])

    def table(data: pd.DataFrame) -> pd.DataFrame:
        data.insert(0, "variant", spec.variant.name)
        data.insert(1, "problem", spec.problem_name)
        data.insert(2, "seed", spec.seed)
        data["optimum"] = problem.optimum if problem.optimum is not None else np.nan
        return data

    spec.out.mkdir(parents=True, exist_ok=True)
    table(pd.DataFrame(rows)).to_parquet(spec.out / "generations.parquet", index=False)
    improvements = pd.DataFrame(ctx.improvements, columns=["evaluations", "best_so_far"])
    table(improvements).to_parquet(spec.out / "improvements.parquet", index=False)
    summary = {
        "variant": spec.variant.name,
        "algorithm": spec.variant.algorithm,
        "params": spec.variant.params,
        "problem": spec.problem,
        "problem_name": spec.problem_name,
        "seed": spec.seed,
        "budget": spec.budget,
        "evaluations": ctx.evaluations,
        "best": ctx.best_so_far,
        "optimum": problem.optimum,
        "best_genome": _jsonable(ctx.best_genome),
        "time_s": time.perf_counter() - ctx.start_time,
    }
    # summary.json is written last (atomically), so its presence marks a finished run.
    tmp = spec.out / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=1))
    tmp.replace(spec.out / "summary.json")
    return spec.out


def run_experiment(
    cfg: ExperimentConfig, results_root: Path = Path("results"), jobs: int | None = None
) -> Path:
    exp_dir = results_root / cfg.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    if cfg.path is not None:
        shutil.copyfile(cfg.path, exp_dir / "config.toml")
    _write_manifest(exp_dir)

    specs = [s for s in expand(cfg, results_root) if not (s.out / "summary.json").exists()]
    total = len(cfg.problems) * len(cfg.variants) * len(cfg.seeds)
    print(f"{cfg.name}: {total - len(specs)}/{total} runs already done, {len(specs)} to go")
    jobs = jobs or os.cpu_count() or 1
    if jobs == 1:
        for i, spec in enumerate(specs, 1):
            run_one(spec)
            print(f"  [{i}/{len(specs)}] {spec.out.relative_to(exp_dir)}")
    else:
        with ProcessPoolExecutor(jobs) as pool:
            futures = {pool.submit(run_one, s): s for s in specs}
            for i, fut in enumerate(as_completed(futures), 1):
                out = fut.result()
                print(f"  [{i}/{len(specs)}] {out.relative_to(exp_dir)}")
    return exp_dir


def finished_runs(exp_dir: str | Path) -> list[Path]:
    return sorted(p.parent for p in Path(exp_dir).glob("runs/*/*/seed*/summary.json"))


def load_runs(exp_dir: str | Path, table: str = "generations") -> pd.DataFrame:
    """One table of every finished run, concatenated into a long DataFrame.

    ``table`` is ``"generations"`` (one row per reported generation) or
    ``"improvements"`` (one row per best-so-far improvement).
    """
    runs = finished_runs(exp_dir)
    if not runs:
        raise FileNotFoundError(f"no finished runs under {exp_dir}")
    return pd.concat([pd.read_parquet(r / f"{table}.parquet") for r in runs], ignore_index=True)


def _write_manifest(exp_dir: Path) -> None:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    manifest = {
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "evosoc": __version__,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))


def _jsonable(genome: Any) -> Any:
    return genome.tolist() if isinstance(genome, np.ndarray) else genome
