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


def plot_baseline(path: Path, coords, histories, ref, ref_len: float, checkpoints=()) -> None:
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
    for g in checkpoints:
        med = float(np.median(best[:, g]))
        ax.axvline(g, color=GREY, lw=0.8, alpha=0.5)
        last = g == gens[-1]
        ax.annotate(f"{(med / ref_len - 1) * 100:+.0f}%", (g, med), xytext=(-4 if last else 4, 8),
                    textcoords="offset points", fontsize=8, color=BLUE,
                    ha="right" if last else "left")
    ax.set_xlabel("generation")
    ax.set_ylabel("trip length (shorter is better)")
    ax.set_title("How the population improves", fontsize=10)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    fig.suptitle("Baseline: a textbook genetic algorithm on the travelling salesman", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _ruler_colors(ks) -> dict:
    """Darker = more rulers. The baseline (2 rulers) is drawn in black."""
    cmap = plt.get_cmap("plasma")
    others = [k for k in ks if k != 2]
    colors = {k: cmap(0.85 - 0.7 * i / max(len(others) - 1, 1)) for i, k in enumerate(others)}
    colors[2] = "black"
    return colors


def _ruler_label(k: int) -> str:
    name = "1 ruler" if k == 1 else f"{k} rulers"
    return name + " (baseline)" if k == 2 else name


def plot_rulers(path: Path, societies: dict, ref_len: float, checkpoints=()) -> None:
    """Compare societies that differ only in how many of the fittest are guaranteed to survive."""
    ks = sorted(societies)
    colors = _ruler_colors(ks)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained",
                             gridspec_kw={"width_ratios": [1.4, 1.4, 1]})
    ax_gap, ax_div, ax_end = axes

    for k in ks:
        hs = societies[k]
        x = hs[0].measured / 1000
        gap = (np.median([h.best for h in hs], axis=0) / ref_len - 1) * 100
        div = np.median([h.diversity for h in hs], axis=0)
        ls = "--" if k == 2 else "-"
        ax_gap.plot(x, gap, color=colors[k], lw=1.8, ls=ls, label=_ruler_label(k))
        ax_div.plot(x, div, color=colors[k], lw=1.8, ls=ls, label=_ruler_label(k))

    for ax in (ax_gap, ax_div):
        for c in checkpoints:
            ax.axvline(c / 1000, color=GREY, lw=0.8, alpha=0.5)
        ax.set_xlabel("trips measured (thousands): the same budget for everyone")
        ax.grid(alpha=0.3)

    ax_gap.axhline(0, color=REF, lw=1.2, ls=":", label="yardstick")
    ax_gap.set_ylim(-10, 100)
    ax_gap.set_ylabel("best trip, % longer than the yardstick")
    ax_gap.set_title("How fast the best trip improves (median of runs)", fontsize=10)
    ax_gap.legend(fontsize=8, frameon=False)

    ax_div.set_ylim(0, 1)
    ax_div.set_ylabel("difference from the best member")
    ax_div.set_title("How varied the population stays (median of runs)", fontsize=10)

    rng = np.random.default_rng(0)  # only spreads the dots sideways so they do not overlap
    for i, k in enumerate(ks):
        finals = (np.array([h.best[-1] for h in societies[k]]) / ref_len - 1) * 100
        ax_end.scatter(i + rng.uniform(-0.18, 0.18, len(finals)), finals, s=12,
                       color=colors[k], alpha=0.7)
        ax_end.hlines(np.median(finals), i - 0.3, i + 0.3, color=colors[k], lw=2.5)
    ax_end.axhline(0, color=REF, lw=1.2, ls=":")
    ax_end.set_xticks(range(len(ks)), [str(k) + ("\nbaseline" if k == 2 else "") for k in ks])
    ax_end.set_xlabel("number of rulers")
    ax_end.set_ylabel("best trip at the end, % longer than the yardstick")
    ax_end.set_title(f"Where each run ends ({len(societies[ks[0]])} runs, bar = median)",
                     fontsize=10)
    ax_end.grid(alpha=0.3, axis="y")

    fig.suptitle("Experiment 1: the k fittest rule and are guaranteed to survive", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)
