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
