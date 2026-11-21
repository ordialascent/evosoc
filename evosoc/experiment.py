"""Run the baseline over many seeds and save the numbers and a figure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import ga
from .tsp import distance_matrix, make_cities, reference_tour, tour_length


def run_baseline(
    out_dir: Path,
    cities: int = 50,
    map_seed: int = 1,
    seeds: int = 10,
    config: ga.Config = ga.Config(),
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    coords = make_cities(cities, map_seed)
    dist = distance_matrix(coords)
    ref = reference_tour(dist)
    ref_len = tour_length(dist, ref)

    histories = [ga.run(dist, config, seed) for seed in range(seeds)]

    with open(out_dir / "baseline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "generation", "best", "mean", "worst"])
        for seed, h in enumerate(histories):
            for g in range(len(h.best)):
                w.writerow([seed, g, f"{h.best[g]:.6f}", f"{h.mean[g]:.6f}", f"{h.worst[g]:.6f}"])

    finals = np.array([h.best[-1] for h in histories])
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
        "final_gap_to_reference_pct": {
            "median": round(float((np.median(finals) / ref_len - 1) * 100), 2),
        },
    }
    (out_dir / "baseline.json").write_text(json.dumps(summary, indent=2) + "\n")

    from .plots import plot_baseline

    plot_baseline(out_dir / "baseline.png", coords, histories, ref, ref_len)
    return summary
