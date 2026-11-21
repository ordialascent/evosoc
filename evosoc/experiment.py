"""Run the baseline over many seeds and save the numbers and a figure."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from . import ga
from .tsp import distance_matrix, make_cities, reference_tour, tour_length


def run_baseline(
    out_dir: Path,
    cities: int = 50,
    map_seed: int = 1,
    seeds: int = 30,
    config: ga.Config = ga.Config(),
    checkpoints: tuple[int, ...] = (100, 300),
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    coords = make_cities(cities, map_seed)
    dist = distance_matrix(coords)
    ref = reference_tour(dist)
    ref_len = tour_length(dist, ref)

    # Runs are independent and each one is fixed by its seed, so running them
    # in parallel gives exactly the same numbers as running them one by one.
    with ProcessPoolExecutor() as pool:
        histories = list(pool.map(partial(ga.run, dist, config), range(seeds)))

    with open(out_dir / "baseline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "generation", "best", "mean", "worst"])
        for seed, h in enumerate(histories):
            for g in range(len(h.best)):
                w.writerow([seed, g, f"{h.best[g]:.6f}", f"{h.mean[g]:.6f}", f"{h.worst[g]:.6f}"])

    best = np.array([h.best for h in histories])
    finals = best[:, -1]

    def gap(lengths: np.ndarray) -> float:
        return round(float((np.median(lengths) / ref_len - 1) * 100), 2)

    gens = sorted({g for g in checkpoints if g < config.generations} | {config.generations})
    summary = {
        "cities": cities,
        "map_seed": map_seed,
        "seeds": seeds,
        "config": config.as_dict(),
        "evaluations_per_run": histories[0].evaluations,
        "reference_length": round(ref_len, 6),
        "final_best": {
            "median": round(float(np.median(finals)), 6),
            "min": round(float(finals.min()), 6),
            "max": round(float(finals.max()), 6),
        },
        # How much longer the median run's best trip is than the yardstick,
        # at an early, a middle and the final generation.
        "gap_to_reference_pct_by_generation": {str(g): gap(best[:, g]) for g in gens},
    }
    (out_dir / "baseline.json").write_text(json.dumps(summary, indent=2) + "\n")

    from .plots import plot_baseline

    plot_baseline(out_dir / "baseline.png", coords, histories, ref, ref_len, gens)
    return summary


# The baseline's budget: 100 random trips, then 1,000 generations of 98 children.
BASELINE_BUDGET = 100 + 1000 * 98
# The points where results are compared: what the baseline has measured
# after 100, 300 and 1,000 generations.
BUDGET_CHECKPOINTS = (100 + 100 * 98, 100 + 300 * 98, BASELINE_BUDGET)


def best_at(history: ga.History, measured: int) -> float:
    """Shortest trip found once `measured` trips had been measured."""
    g = int(np.searchsorted(history.measured, measured, side="right")) - 1
    return float(history.best[max(g, 0)])


def run_rulers(
    out_dir: Path,
    rulers: tuple[int, ...] = (1, 2, 5, 20, 50),
    cities: int = 50,
    map_seed: int = 1,
    seeds: int = 30,
    population: int = 100,
    budget: int = BASELINE_BUDGET,
    checkpoints: tuple[int, ...] = BUDGET_CHECKPOINTS,
) -> dict:
    """Experiment 1: a ruling class of the k fittest, who are guaranteed to survive.

    Everything else is the baseline. Every society gets the same number of
    trip measurements, so a society that keeps more rulers (and therefore
    makes fewer children per generation) runs for more generations.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    coords = make_cities(cities, map_seed)
    dist = distance_matrix(coords)
    ref = reference_tour(dist)
    ref_len = tour_length(dist, ref)
    checkpoints = tuple(c for c in checkpoints if c <= budget) or (budget,)

    societies = {}
    with ProcessPoolExecutor() as pool:
        for k in rulers:
            gens = ga.generations_for_budget(population, k, budget)
            config = ga.Config(population=population, generations=gens, elite=k)
            societies[k] = (config, list(pool.map(partial(ga.run, dist, config), range(seeds))))

    with open(out_dir / "rulers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rulers", "seed", "generation", "measured", "best", "mean", "diversity"])
        for k, (_, histories) in societies.items():
            for seed, h in enumerate(histories):
                for g in range(len(h.best)):
                    w.writerow([k, seed, g, h.measured[g], f"{h.best[g]:.6f}",
                                f"{h.mean[g]:.6f}", f"{h.diversity[g]:.6f}"])

    def pct(x) -> float:
        return round(float((x / ref_len - 1) * 100), 2)

    summary = {
        "cities": cities,
        "map_seed": map_seed,
        "seeds": seeds,
        "population": population,
        "budget": budget,
        "reference_length": round(ref_len, 6),
        "societies": {},
    }
    for k, (config, histories) in societies.items():
        entry = {
            "generations": config.generations,
            "evaluations_per_run": histories[0].evaluations,
            # Median gap to the yardstick, and the range of the middle half of runs.
            "gap_to_reference_pct_by_measured": {},
            "difference_from_best_by_measured": {},
        }
        for c in checkpoints:
            gaps = np.array([best_at(h, c) for h in histories])
            g_idx = [int(np.searchsorted(h.measured, c, side="right")) - 1 for h in histories]
            div = np.array([h.diversity[g] for h, g in zip(histories, g_idx)])
            entry["gap_to_reference_pct_by_measured"][str(c)] = {
                "median": pct(np.median(gaps)),
                "middle_half": [pct(np.percentile(gaps, 25)), pct(np.percentile(gaps, 75))],
            }
            entry["difference_from_best_by_measured"][str(c)] = round(float(np.median(div)), 3)
        summary["societies"][str(k)] = entry
    (out_dir / "rulers.json").write_text(json.dumps(summary, indent=2) + "\n")

    from .plots import plot_rulers

    plot_rulers(out_dir / "rulers.png", {k: h for k, (_, h) in societies.items()},
                ref_len, checkpoints)
    return summary


def shuffle_test(a, b, rounds: int = 10_000, seed: int = 0) -> float:
    """How often a gap between the medians of `a` and `b` at least this large
    appears when the runs are shuffled between the two groups at random.

    Small values mean the difference is unlikely to be luck.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    both = np.concatenate([a, b])
    observed = abs(np.median(a) - np.median(b))
    hits = 0
    for _ in range(rounds):
        rng.shuffle(both)
        hits += abs(np.median(both[: len(a)]) - np.median(both[len(a):])) >= observed - 1e-12
    return round((hits + 1) / (rounds + 1), 4)


def run_choosing(
    out_dir: Path,
    rulers: tuple[int, ...] = (5, 20),
    ways: tuple[str, ...] = ga.RULER_CHOICES,
    cities: int = 50,
    map_seed: int = 1,
    seeds: int = 30,
    population: int = 100,
    budget: int = BASELINE_BUDGET,
    checkpoints: tuple[int, ...] = BUDGET_CHECKPOINTS,
) -> dict:
    """Experiment 2: does it matter how the rulers are chosen?

    For each size of ruling class, compare rulers chosen by merit (the
    fittest), by birth (a ruler's child inherits the seat) and by lottery
    (drawn at random). Two references run alongside: the baseline (the 2
    fittest are kept) and a society with no rulers at all. Every society gets
    the same number of trip measurements.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    coords = make_cities(cities, map_seed)
    dist = distance_matrix(coords)
    ref_len = tour_length(dist, reference_tour(dist))
    checkpoints = tuple(c for c in checkpoints if c <= budget) or (budget,)

    plan = {"none": ga.Config(population=population, elite=0),
            "baseline": ga.Config(population=population, elite=2)}
    for k in rulers:
        for way in ways:
            plan[f"{way}-{k}"] = ga.Config(population=population, elite=k, rulers_chosen=way)

    societies = {}
    with ProcessPoolExecutor() as pool:
        for name, config in plan.items():
            gens = ga.generations_for_budget(population, ga.kept_per_generation(config), budget)
            config = ga.Config(**{**config.as_dict(), "generations": gens})
            societies[name] = (config, list(pool.map(partial(ga.run, dist, config), range(seeds))))

    with open(out_dir / "choosing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["society", "seed", "generation", "measured", "best", "mean", "diversity"])
        for name, (_, histories) in societies.items():
            for seed, h in enumerate(histories):
                for g in range(len(h.best)):
                    w.writerow([name, seed, g, h.measured[g], f"{h.best[g]:.6f}",
                                f"{h.mean[g]:.6f}", f"{h.diversity[g]:.6f}"])

    def pct(x) -> float:
        return round(float((x / ref_len - 1) * 100), 2)

    def finals(name: str) -> list[float]:
        return [best_at(h, budget) for h in societies[name][1]]

    summary = {
        "cities": cities,
        "map_seed": map_seed,
        "seeds": seeds,
        "population": population,
        "budget": budget,
        "reference_length": round(ref_len, 6),
        "societies": {},
    }
    for name, (config, histories) in societies.items():
        entry = {
            "rulers": config.elite,
            "rulers_chosen": config.rulers_chosen if config.elite else None,
            "generations": config.generations,
            "evaluations_per_run": histories[0].evaluations,
            "gap_to_reference_pct_by_measured": {},
            "difference_from_best_by_measured": {},
        }
        for c in checkpoints:
            gaps = np.array([best_at(h, c) for h in histories])
            g_idx = [int(np.searchsorted(h.measured, c, side="right")) - 1 for h in histories]
            div = np.array([h.diversity[g] for h, g in zip(histories, g_idx)])
            entry["gap_to_reference_pct_by_measured"][str(c)] = {
                "median": pct(np.median(gaps)),
                "middle_half": [pct(np.percentile(gaps, 25)), pct(np.percentile(gaps, 75))],
            }
            entry["difference_from_best_by_measured"][str(c)] = round(float(np.median(div)), 3)
        if config.elite and config.rulers_chosen != "merit" and f"merit-{config.elite}" in plan:
            # Chance that the end result differs this much from merit rulers by luck alone.
            entry["shuffle_test_vs_merit_at_end"] = shuffle_test(finals(name),
                                                                 finals(f"merit-{config.elite}"))
        if name != "baseline":
            entry["shuffle_test_vs_baseline_at_end"] = shuffle_test(finals(name), finals("baseline"))
        summary["societies"][name] = entry
    (out_dir / "choosing.json").write_text(json.dumps(summary, indent=2) + "\n")

    from .plots import plot_choosing

    plot_choosing(out_dir / "choosing.png", {n: h for n, (_, h) in societies.items()},
                  rulers, ways, ref_len, checkpoints)
    return summary
