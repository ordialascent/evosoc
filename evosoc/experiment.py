"""Run the baseline over many seeds and save the numbers and a figure."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from . import ga
from .tsp import distance_matrix, make_cities, reference_tour, tour_length, tour_lengths


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
    """Experiment 1: a protected class of the k fittest, who are guaranteed to survive.

    Everything else is the baseline. Every society gets the same number of
    trip measurements, so a society that protects more members (and therefore
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
    """Experiment 2: does it matter who is protected?

    For each size of protected class, compare members protected by merit
    (the fittest), by birth (a protected member's child inherits the seat)
    and by lottery (drawn at random). Two references run alongside: the
    baseline (the 2 fittest are kept) and a society where nobody is protected. Every society gets
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
            # Chance that the end result differs this much from merit protection by luck alone.
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


# Ways of making a child, compared in Experiment 3. Each is a complete
# recipe: how two parents are combined, and how often a stretch of the
# child's trip is reversed afterwards.
REPRODUCTION = {
    "textbook": {"crossover": "order", "mutation_rate": 0.3},
    "fewer reversals": {"crossover": "order", "mutation_rate": 0.05},
    "edge crossover": {"crossover": "edge", "mutation_rate": 0.3},
}
# Societies compared under each way of making children. Keys match the stored
# results: "no rulers" means nobody is protected.
SOCIETIES = {
    "no rulers": {"elite": 0},
    "lottery-5": {"elite": 5, "rulers_chosen": "lottery"},
    "baseline": {"elite": 2},
    "merit-5": {"elite": 5},
}
# Single steps whose damage is measured: each takes parents from a population
# and returns (child, the better parent's trip length).
STEPS = ("order crossover", "edge crossover", "reversal")


def _one_step(rng, dist, pop, lengths, step: str, tournament_size: int = 3):
    from .operators import edge_crossover, inversion_mutation, order_crossover, tournament

    mother = pop[tournament(rng, lengths, tournament_size)]
    if step == "reversal":
        return inversion_mutation(rng, mother), tour_length(dist, mother)
    father = pop[tournament(rng, lengths, tournament_size)]
    combine = order_crossover if step == "order crossover" else edge_crossover
    better = min(tour_length(dist, mother), tour_length(dist, father))
    return combine(rng, mother, father), better


def measure_damage(dist: np.ndarray, pop: np.ndarray, seed: int, children: int = 300) -> dict:
    """For each single step, how often a child is worse than its better parent,
    and by how much (median change in trip length, in %)."""
    lengths = tour_lengths(dist, pop)
    out = {}
    for s, step in enumerate(STEPS):
        rng = np.random.default_rng([seed, s])
        worse, change = [], []
        for _ in range(children):
            child, better = _one_step(rng, dist, pop, lengths, step)
            length = tour_length(dist, child)
            worse.append(length > better + 1e-9)
            change.append(length / better - 1)
        out[step] = (float(np.mean(worse)), float(np.median(change)))
    return out


def _population_at(dist, generations: int, seed: int) -> np.ndarray:
    return ga.run(dist, ga.Config(generations=generations), seed).population


def run_reproduction(
    out_dir: Path,
    cities: int = 50,
    map_seed: int = 1,
    seeds: int = 30,
    damage_seeds: int = 10,
    stages: tuple[int, ...] = (0, 100, 300, 1000),
    budget: int = BASELINE_BUDGET,
    checkpoints: tuple[int, ...] = BUDGET_CHECKPOINTS,
) -> dict:
    """Experiment 3: how much damage does making a child do, and does it decide
    which society wins?

    Part 1 measures, for populations taken from baseline runs at several
    stages, how often a single step of making a child (order crossover, edge
    crossover or a random reversal) produces a trip worse than the better
    parent. Part 2 runs four societies (nobody protected, 5 protected by
    lottery, the baseline, 5 protected by merit) under three ways of making children, on the
    same budget, map and seeds.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    coords = make_cities(cities, map_seed)
    dist = distance_matrix(coords)
    ref_len = tour_length(dist, reference_tour(dist))
    checkpoints = tuple(c for c in checkpoints if c <= budget) or (budget,)

    societies = {}
    with ProcessPoolExecutor() as pool:
        # Part 1: damage done by each step, at each stage of a baseline run.
        damage = {}
        for g in stages:
            pops = list(pool.map(partial(_population_at, dist, g), range(damage_seeds)))
            per_seed = [measure_damage(dist, p, seed) for seed, p in enumerate(pops)]
            damage[g] = {step: (float(np.mean([d[step][0] for d in per_seed])),
                                float(np.median([d[step][1] for d in per_seed])))
                         for step in STEPS}

        # Part 2: societies under each way of making children.
        for way, recipe in REPRODUCTION.items():
            for name, rules in SOCIETIES.items():
                config = ga.Config(**rules, **recipe)
                gens = ga.generations_for_budget(config.population,
                                                 ga.kept_per_generation(config), budget)
                config = ga.Config(**{**config.as_dict(), "generations": gens})
                histories = list(pool.map(partial(ga.run, dist, config), range(seeds)))
                societies[(way, name)] = (config, histories)

    # Damage inside a society where nobody is protected, once it has stopped improving.
    stuck = [measure_damage(dist, h.population, seed)
             for seed, h in enumerate(societies[("textbook", "no rulers")][1][:damage_seeds])]
    damage["no rulers, end"] = {step: (float(np.mean([d[step][0] for d in stuck])),
                                       float(np.median([d[step][1] for d in stuck])))
                                for step in STEPS}

    with open(out_dir / "reproduction.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reproduction", "society", "seed", "generation", "measured", "best", "mean",
                    "diversity"])
        for (way, name), (_, histories) in societies.items():
            for seed, h in enumerate(histories):
                for g in range(len(h.best)):
                    w.writerow([way, name, seed, g, h.measured[g], f"{h.best[g]:.6f}",
                                f"{h.mean[g]:.6f}", f"{h.diversity[g]:.6f}"])

    def pct(x) -> float:
        return round(float((x / ref_len - 1) * 100), 2)

    def finals(key) -> list[float]:
        return [best_at(h, budget) for h in societies[key][1]]

    summary = {
        "cities": cities,
        "map_seed": map_seed,
        "seeds": seeds,
        "budget": budget,
        "reference_length": round(ref_len, 6),
        "reproduction": REPRODUCTION,
        # Share of children worse than their better parent, and the median
        # change in trip length (%), by stage of a baseline run (generation).
        "damage": {str(stage): {step: {"worse_share": round(w, 3), "median_change_pct": round(c * 100, 2)}
                                for step, (w, c) in by_step.items()}
                   for stage, by_step in damage.items()},
        "societies": {},
    }
    for (way, name), (config, histories) in societies.items():
        entry = {
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
        if name != "merit-5":
            # Chance that the end result differs this much from 5 protected by merit by luck alone.
            entry["shuffle_test_vs_merit_5_at_end"] = shuffle_test(finals((way, name)),
                                                                   finals((way, "merit-5")))
        summary["societies"].setdefault(way, {})[name] = entry
    (out_dir / "reproduction.json").write_text(json.dumps(summary, indent=2) + "\n")

    from .plots import plot_reproduction

    plot_reproduction(out_dir / "reproduction.png", damage, STEPS,
                      {k: h for k, (_, h) in societies.items()}, tuple(REPRODUCTION),
                      tuple(SOCIETIES), ref_len, budget)
    return summary


# Societies compared in Experiment 4. The first two are references from the
# earlier experiments (mothers chosen by tournament); in the others nobody
# is protected, and every child's mother is drawn in proportion to wealth.
OWNERSHIP = {
    "no rulers": {"elite": 0},
    "baseline": {"elite": 2},
    "pooled": {"elite": 0, "mothers": "wealth", "shared": 1.0, "inherited": 0.0},
    "half pooled": {"elite": 0, "mothers": "wealth", "shared": 0.5, "inherited": 0.0},
    "kept": {"elite": 0, "mothers": "wealth", "shared": 0.0, "inherited": 0.0},
    "kept, half inherited": {"elite": 0, "mothers": "wealth", "shared": 0.0, "inherited": 0.5},
    "kept, mostly inherited": {"elite": 0, "mothers": "wealth", "shared": 0.0, "inherited": 0.9},
}
# Ways of making children each society is run under (see Experiment 3).
OWNERSHIP_REPRODUCTION = ("textbook", "edge crossover")


def run_ownership(
    out_dir: Path,
    cities: int = 50,
    map_seed: int = 1,
    seeds: int = 30,
    budget: int = BASELINE_BUDGET,
    checkpoints: tuple[int, ...] = BUDGET_CHECKPOINTS,
    societies_run: dict = OWNERSHIP,
    reproduction: tuple[str, ...] = OWNERSHIP_REPRODUCTION,
) -> dict:
    """Experiment 4: who owns what a member produces?

    Each generation every member earns from its trip; wealth buys children.
    Societies differ in how much of what is earned is pooled and split
    evenly, and how much of a member's wealth its children inherit. Every
    society runs under the textbook and the edge-crossover way of making
    children, on the same budget, map and seeds.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    coords = make_cities(cities, map_seed)
    dist = distance_matrix(coords)
    ref_len = tour_length(dist, reference_tour(dist))
    checkpoints = tuple(c for c in checkpoints if c <= budget) or (budget,)

    societies = {}
    with ProcessPoolExecutor() as pool:
        for way in reproduction:
            for name, rules in societies_run.items():
                config = ga.Config(**rules, **REPRODUCTION[way])
                gens = ga.generations_for_budget(config.population,
                                                 ga.kept_per_generation(config), budget)
                config = ga.Config(**{**config.as_dict(), "generations": gens})
                histories = list(pool.map(partial(ga.run, dist, config), range(seeds)))
                societies[(way, name)] = (config, histories)

    with open(out_dir / "ownership.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reproduction", "society", "seed", "generation", "measured", "best", "mean",
                    "diversity", "parent_gini", "wealth_gini"])
        for (way, name), (_, histories) in societies.items():
            for seed, h in enumerate(histories):
                for g in range(len(h.best)):
                    pg = f"{h.parent_gini[g - 1]:.4f}" if g else ""
                    wg = f"{h.wealth_gini[g - 1]:.4f}" if g else ""
                    w.writerow([way, name, seed, g, h.measured[g], f"{h.best[g]:.6f}",
                                f"{h.mean[g]:.6f}", f"{h.diversity[g]:.6f}", pg, wg])

    def pct(x) -> float:
        return round(float((x / ref_len - 1) * 100), 2)

    def finals(key) -> list[float]:
        return [best_at(h, budget) for h in societies[key][1]]

    summary = {
        "cities": cities,
        "map_seed": map_seed,
        "seeds": seeds,
        "budget": budget,
        "reference_length": round(ref_len, 6),
        "reproduction": {way: REPRODUCTION[way] for way in reproduction},
        "societies": {},
    }
    for (way, name), (config, histories) in societies.items():
        entry = {
            "rules": societies_run[name],
            "generations": config.generations,
            "evaluations_per_run": histories[0].evaluations,
            "gap_to_reference_pct_by_measured": {},
            "difference_from_best_by_measured": {},
            # Median over runs of the average over the second half of the run.
            "wealth_gini_late": round(float(np.median(
                [h.wealth_gini[len(h.wealth_gini) // 2:].mean() for h in histories])), 3),
            "parent_gini_late": round(float(np.median(
                [h.parent_gini[len(h.parent_gini) // 2:].mean() for h in histories])), 3),
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
        for other in ("pooled", "no rulers"):
            if name != other and (way, other) in societies:
                # Chance that the end result differs this much by luck alone.
                entry[f"shuffle_test_vs_{other.replace(' ', '_')}_at_end"] = shuffle_test(
                    finals((way, name)), finals((way, other)))
        summary["societies"].setdefault(way, {})[name] = entry
    (out_dir / "ownership.json").write_text(json.dumps(summary, indent=2) + "\n")

    from .plots import plot_ownership

    plot_ownership(out_dir / "ownership.png", {k: h for k, (_, h) in societies.items()},
                   reproduction, tuple(societies_run), ref_len, budget)
    return summary


# Societies compared in Experiment 8. Nobody is protected; every child's
# mother is drawn in proportion to wealth, nothing is pooled. The first two
# are references from Experiment 4; in the others children inherit half of
# a member's wealth, split evenly or left to one heir, and inherited wealth
# earns a return of 0, 50% or 100% per generation.
CAPITAL = {
    "no rulers": {"elite": 0},
    "kept": {"elite": 0, "mothers": "wealth"},
    "split": {"elite": 0, "mothers": "wealth", "inherited": 0.5},
    "fittest heir": {"elite": 0, "mothers": "wealth", "inherited": 0.5, "heir": "fittest"},
    "first-born": {"elite": 0, "mothers": "wealth", "inherited": 0.5, "heir": "first"},
    "split, +50%": {"elite": 0, "mothers": "wealth", "inherited": 0.5, "returns": 0.5},
    "first-born, +50%": {"elite": 0, "mothers": "wealth", "inherited": 0.5, "returns": 0.5,
                         "heir": "first"},
    "split, +100%": {"elite": 0, "mothers": "wealth", "inherited": 0.5, "returns": 1.0},
    "fittest heir, +100%": {"elite": 0, "mothers": "wealth", "inherited": 0.5, "returns": 1.0,
                            "heir": "fittest"},
    "first-born, +100%": {"elite": 0, "mothers": "wealth", "inherited": 0.5, "returns": 1.0,
                          "heir": "first"},
}
# Edge crossover first: under the textbook way every wealth society fails
# (Experiment 4), so that row is only for completeness.
CAPITAL_REPRODUCTION = ("edge crossover", "textbook")


def run_capital(
    out_dir: Path,
    cities: int = 50,
    map_seed: int = 1,
    seeds: int = 30,
    budget: int = BASELINE_BUDGET,
    checkpoints: tuple[int, ...] = BUDGET_CHECKPOINTS,
    societies_run: dict = CAPITAL,
    reproduction: tuple[str, ...] = CAPITAL_REPRODUCTION,
) -> dict:
    """Experiment 8: capital that earns a return, and a single heir.

    Like Experiment 4, wealth buys children. Here inherited wealth can grow
    each generation, and an estate can go to one heir instead of being split.
    Besides the best trip, it records how unequal wealth is, how much of it
    the richest tenth own, and how often children still go to members whose
    own trip is in the better half.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    coords = make_cities(cities, map_seed)
    dist = distance_matrix(coords)
    ref_len = tour_length(dist, reference_tour(dist))
    checkpoints = tuple(c for c in checkpoints if c <= budget) or (budget,)

    societies = {}
    with ProcessPoolExecutor() as pool:
        for way in reproduction:
            for name, rules in societies_run.items():
                config = ga.Config(**rules, **REPRODUCTION[way])
                gens = ga.generations_for_budget(config.population,
                                                 ga.kept_per_generation(config), budget)
                config = ga.Config(**{**config.as_dict(), "generations": gens})
                histories = list(pool.map(partial(ga.run, dist, config), range(seeds)))
                societies[(way, name)] = (config, histories)

    with open(out_dir / "capital.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reproduction", "society", "seed", "generation", "measured", "best", "mean",
                    "diversity", "wealth_gini", "top_tenth_share", "mothers_better_half"])
        for (way, name), (_, histories) in societies.items():
            for seed, h in enumerate(histories):
                for g in range(len(h.best)):
                    late = [f"{h.wealth_gini[g - 1]:.4f}", f"{h.top_tenth_share[g - 1]:.4f}",
                            f"{h.mothers_better_half[g - 1]:.4f}"] if g else ["", "", ""]
                    w.writerow([way, name, seed, g, h.measured[g], f"{h.best[g]:.6f}",
                                f"{h.mean[g]:.6f}", f"{h.diversity[g]:.6f}", *late])

    def pct(x) -> float:
        return round(float((x / ref_len - 1) * 100), 2)

    def finals(key) -> list[float]:
        return [best_at(h, budget) for h in societies[key][1]]

    def late(histories, field: str) -> float:
        # Median over runs of the average over the second half of the run.
        return round(float(np.median([getattr(h, field)[len(getattr(h, field)) // 2:].mean()
                                      for h in histories])), 3)

    summary = {
        "cities": cities,
        "map_seed": map_seed,
        "seeds": seeds,
        "budget": budget,
        "reference_length": round(ref_len, 6),
        "reproduction": {way: REPRODUCTION[way] for way in reproduction},
        "societies": {},
    }
    for (way, name), (config, histories) in societies.items():
        entry = {
            "rules": societies_run[name],
            "generations": config.generations,
            "evaluations_per_run": histories[0].evaluations,
            "gap_to_reference_pct_by_measured": {},
            "wealth_gini_late": late(histories, "wealth_gini"),
            "top_tenth_share_late": late(histories, "top_tenth_share"),
            "mothers_better_half_late": late(histories, "mothers_better_half"),
        }
        for c in checkpoints:
            gaps = np.array([best_at(h, c) for h in histories])
            entry["gap_to_reference_pct_by_measured"][str(c)] = {
                "median": pct(np.median(gaps)),
                "middle_half": [pct(np.percentile(gaps, 25)), pct(np.percentile(gaps, 75))],
            }
        for other in ("kept", "split"):
            if name != other and (way, other) in societies:
                # Chance that the end result differs this much by luck alone.
                entry[f"shuffle_test_vs_{other}_at_end"] = shuffle_test(
                    finals((way, name)), finals((way, other)))
        summary["societies"].setdefault(way, {})[name] = entry
    (out_dir / "capital.json").write_text(json.dumps(summary, indent=2) + "\n")

    from .plots import plot_capital

    plot_capital(out_dir / "capital.png", {k: h for k, (_, h) in societies.items()},
                 reproduction, tuple(societies_run), ref_len, budget)
    return summary


# Experiment 5: maps of different size, and the points where the budget is
# read off, in baseline generations (100 trips, then 98 new trips each).
PRESSURE_CITIES = (30, 50, 100)
PRESSURE_BUDGETS = (100, 300, 1000)
PRESSURE_RULERS = (0, 2, 10, 50)
PRESSURE_REPRODUCTION = ("textbook", "edge crossover")


def trips_for(generations: int, population: int = 100, kept: int = 2) -> int:
    """Trips the baseline has measured after `generations` generations."""
    return population + generations * (population - kept)


def run_pressure(
    out_dir: Path,
    cities: tuple[int, ...] = PRESSURE_CITIES,
    budgets: tuple[int, ...] = PRESSURE_BUDGETS,
    rulers: tuple[int, ...] = PRESSURE_RULERS,
    ways: tuple[str, ...] = PRESSURE_REPRODUCTION,
    map_seed: int = 1,
    seeds: int = 30,
    population: int = 100,
    name: str = "pressure",
    title: str | None = None,
) -> dict:
    """Experiment 5: does protecting the top win when time is short?

    Societies that protect their 0, 2, 10 or 50 fittest members solve maps of 30, 50 and 100
    cities, under the textbook and the edge-crossover way of making children.
    Each is read off after a short, a medium and a long budget (what the
    baseline measures in 100, 300 and 1,000 generations). A run does not
    know its budget, so reading a long run at an earlier point gives exactly
    the result of a shorter run with the same seed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    marks = tuple(trips_for(g, population) for g in budgets)
    longest = max(marks)

    maps, societies = {}, {}
    with ProcessPoolExecutor() as pool:
        for n in cities:
            dist = distance_matrix(make_cities(n, map_seed))
            maps[n] = tour_length(dist, reference_tour(dist))
            for way in ways:
                for k in rulers:
                    config = ga.Config(population=population, elite=k, **REPRODUCTION[way])
                    gens = ga.generations_for_budget(population, k, longest)
                    config = ga.Config(**{**config.as_dict(), "generations": gens})
                    societies[(n, way, k)] = list(pool.map(partial(ga.run, dist, config),
                                                           range(seeds)))

    with open(out_dir / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cities", "reproduction", "rulers", "seed", "budget_generations", "measured",
                    "best"])
        for (n, way, k), histories in societies.items():
            for seed, h in enumerate(histories):
                for g, m in zip(budgets, marks):
                    w.writerow([n, way, k, seed, g, m, f"{best_at(h, m):.6f}"])

    summary = {
        "map_seed": map_seed,
        "seeds": seeds,
        "population": population,
        "rulers": list(rulers),
        # Budget names (baseline generations) and the trips each one allows.
        "budgets": {str(g): m for g, m in zip(budgets, marks)},
        "reference_length": {str(n): round(r, 6) for n, r in maps.items()},
        "results": {},
    }
    for n in cities:
        for way in ways:
            for g, m in zip(budgets, marks):
                gaps = {k: (np.array([best_at(h, m) for h in societies[(n, way, k)]])
                            / maps[n] - 1) * 100 for k in rulers}
                ranked = sorted(rulers, key=lambda k: (np.median(gaps[k]), k))
                cell = {
                    "gap_to_reference_pct": {
                        str(k): {"median": round(float(np.median(v)), 2),
                                 "middle_half": [round(float(np.percentile(v, 25)), 2),
                                                 round(float(np.percentile(v, 75)), 2)]}
                        for k, v in gaps.items()},
                    "winner": ranked[0],
                    "runner_up": ranked[1],
                    # Chance that the winner's lead over the runner-up is luck alone.
                    "shuffle_test_winner_vs_runner_up": shuffle_test(gaps[ranked[0]],
                                                                     gaps[ranked[1]]),
                }
                summary["results"].setdefault(str(n), {}).setdefault(way, {})[str(g)] = cell
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2) + "\n")

    from .plots import plot_pressure

    plot_pressure(out_dir / f"{name}.png", societies, maps, summary, cities, ways, rulers,
                  budgets, marks, title=title)
    return summary


# Experiment 7: a planning board decides who has children and with whom.
# Two references (the contests of Experiments 1-6), then the board with an
# ever coarser map of the cities (grid cells per side; 0 = the true map),
# pairing parents at random or each mother with the most different candidate.
PLANNING_VIEWS = (0, 20, 10, 5)
PLANNING = {
    "nobody protected": {"elite": 0},
    "baseline": {"elite": 2},
    **{f"board, {'true map' if g == 0 else f'grid {g}'}": {"elite": 2, "parents": "planned",
                                                           "planners_view": g}
       for g in PLANNING_VIEWS},
    **{f"board, {'true map' if g == 0 else f'grid {g}'}, different pairs": {
        "elite": 2, "parents": "planned", "planners_view": g, "planned_pairs": "different"}
       for g in (0, 10)},
}
PLANNING_REPRODUCTION = ("textbook", "edge crossover")


def run_planning(
    out_dir: Path,
    cities: int = 50,
    map_seed: int = 1,
    seeds: int = 30,
    budget: int = BASELINE_BUDGET,
    checkpoints: tuple[int, ...] = BUDGET_CHECKPOINTS,
    societies_run: dict = PLANNING,
    reproduction: tuple[str, ...] = PLANNING_REPRODUCTION,
) -> dict:
    """Experiment 7: what if a few really decide for everyone?

    A planning board ranks the whole population and decides who has
    children and with whom, instead of each child's parents winning small
    contests. The board sees everyone at once, but judges trips on its own
    simplified map. Compared with the contests (nobody protected, and the
    baseline), under both ways of making children, on the same budget, map
    and seeds.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    coords = make_cities(cities, map_seed)
    dist = distance_matrix(coords)
    ref_len = tour_length(dist, reference_tour(dist))
    checkpoints = tuple(c for c in checkpoints if c <= budget) or (budget,)

    societies = {}
    with ProcessPoolExecutor() as pool:
        for way in reproduction:
            for name, rules in societies_run.items():
                config = ga.Config(**rules, **REPRODUCTION[way])
                gens = ga.generations_for_budget(config.population,
                                                 ga.kept_per_generation(config), budget)
                config = ga.Config(**{**config.as_dict(), "generations": gens})
                view = ga.planners_map(coords, config.planners_view)
                histories = list(pool.map(partial(ga.run, dist, config, view=view), range(seeds)))
                societies[(way, name)] = (config, histories)

    with open(out_dir / "planning.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reproduction", "society", "seed", "generation", "measured", "best", "mean",
                    "diversity", "planners_right"])
        for (way, name), (_, histories) in societies.items():
            for seed, h in enumerate(histories):
                for g in range(len(h.best)):
                    right = f"{h.planners_right[g - 1]:.4f}" if g and len(h.planners_right) else ""
                    w.writerow([way, name, seed, g, h.measured[g], f"{h.best[g]:.6f}",
                                f"{h.mean[g]:.6f}", f"{h.diversity[g]:.6f}", right])

    def pct(x) -> float:
        return round(float((x / ref_len - 1) * 100), 2)

    def finals(key) -> list[float]:
        return [best_at(h, budget) for h in societies[key][1]]

    summary = {
        "cities": cities,
        "map_seed": map_seed,
        "seeds": seeds,
        "budget": budget,
        "reference_length": round(ref_len, 6),
        "reproduction": {way: REPRODUCTION[way] for way in reproduction},
        "societies": {},
    }
    for (way, name), (config, histories) in societies.items():
        entry = {
            "rules": societies_run[name],
            "generations": config.generations,
            "evaluations_per_run": histories[0].evaluations,
            "gap_to_reference_pct_by_measured": {},
            "difference_from_best_by_measured": {},
        }
        if len(histories[0].planners_right):
            # Share of the board's chosen parents that truly were in the better
            # half: median over runs of the average over the first and the
            # second half of the run.
            entry["planners_right"] = {
                part: round(float(np.median([np.array_split(h.planners_right, 2)[i].mean()
                                             for h in histories])), 3)
                for i, part in enumerate(("first_half", "second_half"))}
        for c in checkpoints:
            gaps = np.array([best_at(h, c) for h in histories])
            g_idx = [int(np.searchsorted(h.measured, c, side="right")) - 1 for h in histories]
            div = np.array([h.diversity[g] for h, g in zip(histories, g_idx)])
            entry["gap_to_reference_pct_by_measured"][str(c)] = {
                "median": pct(np.median(gaps)),
                "middle_half": [pct(np.percentile(gaps, 25)), pct(np.percentile(gaps, 75))],
            }
            entry["difference_from_best_by_measured"][str(c)] = round(float(np.median(div)), 3)
        for other in ("baseline", "board, true map"):
            if name != other and (way, other) in societies:
                # Chance that the end result differs this much by luck alone.
                entry[f"shuffle_test_vs_{other.replace(', ', '_').replace(' ', '_')}_at_end"] = (
                    shuffle_test(finals((way, name)), finals((way, other))))
        summary["societies"].setdefault(way, {})[name] = entry
    (out_dir / "planning.json").write_text(json.dumps(summary, indent=2) + "\n")

    from .plots import plot_planning

    plot_planning(out_dir / "planning.png", coords,
                  {k: h for k, (_, h) in societies.items()}, reproduction, tuple(societies_run),
                  ref_len, budget)
    return summary
