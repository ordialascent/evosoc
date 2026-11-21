"""Figures meant to be read without knowing any formulas."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .tsp import distance_matrix, tour_length  # noqa: E402

BLUE = "#2a6fdb"
GREY = "#8a8f98"
REF = "#d1495b"


def _draw_tour(ax, coords: np.ndarray, tour: np.ndarray, color: str, title: str) -> None:
    loop = np.append(tour, tour[0])
    ax.plot(coords[loop, 0], coords[loop, 1], "-", color=color, lw=1.6, zorder=1)
    ax.scatter(coords[:, 0], coords[:, 1], s=22, color="black", zorder=2)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_baseline(path: Path, coords, histories, ref, ref_len: float) -> None:
    dist = distance_matrix(coords)
    best = np.array([h.best for h in histories])
    mean = np.array([h.mean for h in histories])
    gens = np.arange(best.shape[1])

    finals = best[:, -1]
    median_run = int(np.argsort(finals)[len(finals) // 2])
    first = histories[median_run]

    fig = plt.figure(figsize=(12, 4.2), layout="constrained")
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 2.2])

    rng = np.random.default_rng(median_run)
    random_tour = rng.permutation(len(coords))
    _draw_tour(
        fig.add_subplot(grid[0]), coords, random_tour, GREY,
        f"A random trip\nlength {tour_length(dist, random_tour):.2f}",
    )
    _draw_tour(
        fig.add_subplot(grid[1]), coords, first.best_tour, BLUE,
        f"Best evolved trip (typical run)\nlength {first.best[-1]:.2f}",
    )
    _draw_tour(
        fig.add_subplot(grid[2]), coords, ref, REF,
        f"Yardstick (no evolution)\nlength {ref_len:.2f}",
    )

    ax = fig.add_subplot(grid[3])
    ax.fill_between(gens, best.min(axis=0), best.max(axis=0), color=BLUE, alpha=0.18,
                    label=f"best trip, range over {len(histories)} runs")
    ax.plot(gens, np.median(best, axis=0), color=BLUE, lw=2, label="best trip, median run")
    ax.plot(gens, np.median(mean, axis=0), color=GREY, lw=1.5, ls="--",
            label="average member, median run")
    ax.axhline(ref_len, color=REF, lw=1.2, ls=":", label="yardstick")
    ax.set_xlabel("generation")
    ax.set_ylabel("trip length (shorter is better)")
    ax.set_title("How the population improves", fontsize=10)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    fig.suptitle("Baseline: a textbook genetic algorithm on the travelling salesman", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)
