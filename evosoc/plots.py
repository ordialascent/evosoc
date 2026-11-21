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


WAY_COLORS = {"merit": BLUE, "hereditary": "#9b5de5", "lottery": "#f18f01"}
WAY_LABELS = {"merit": "by merit (the fittest)", "hereditary": "by birth (a ruler's child)",
              "lottery": "by lottery (at random)"}


def plot_choosing(path: Path, societies: dict, rulers, ways, ref_len: float,
                  checkpoints=()) -> None:
    """One row per size of ruling class: the same rulers, chosen in different ways."""
    fig, axes = plt.subplots(len(rulers), 3, figsize=(14, 4.2 * len(rulers)),
                             layout="constrained", squeeze=False,
                             gridspec_kw={"width_ratios": [1.4, 1.4, 1]})

    def curves(hs):
        x = hs[0].measured / 1000
        gap = (np.median([h.best for h in hs], axis=0) / ref_len - 1) * 100
        div = np.median([h.diversity for h in hs], axis=0)
        return x, gap, div

    references = [("baseline", "baseline (2 fittest kept)", "black", "--"),
                  ("none", "no rulers", GREY, ":")]
    rng = np.random.default_rng(0)  # only spreads the dots sideways so they do not overlap

    for row, k in zip(axes, rulers):
        ax_gap, ax_div, ax_end = row
        for name, label, color, ls in references:
            x, gap, div = curves(societies[name])
            ax_gap.plot(x, gap, color=color, lw=1.4, ls=ls, label=label)
            ax_div.plot(x, div, color=color, lw=1.4, ls=ls)
        for way in ways:
            x, gap, div = curves(societies[f"{way}-{k}"])
            ax_gap.plot(x, gap, color=WAY_COLORS[way], lw=1.9, label=f"{k} rulers {WAY_LABELS[way]}")
            ax_div.plot(x, div, color=WAY_COLORS[way], lw=1.9)

        for ax in (ax_gap, ax_div):
            for c in checkpoints:
                ax.axvline(c / 1000, color=GREY, lw=0.8, alpha=0.5)
            ax.set_xlabel("trips measured (thousands): the same budget for everyone")
            ax.grid(alpha=0.3)
        ax_gap.axhline(0, color=REF, lw=1.2, ls=":", label="yardstick")
        ax_gap.set_ylim(-10, 200)
        ax_gap.set_ylabel("best trip, % longer than the yardstick")
        ax_gap.set_title(f"{k} rulers: how fast the best trip improves (median of runs)",
                         fontsize=10)
        ax_gap.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1, 0.06))
        ax_div.set_ylim(0, 1)
        ax_div.set_ylabel("difference from the best member")
        ax_div.set_title(f"{k} rulers: how varied the population stays", fontsize=10)

        columns = [(f"{w}-{k}", WAY_COLORS[w], w if w != "hereditary" else "birth")
                   for w in ways] + [("baseline", "black", "baseline"), ("none", GREY, "no\nrulers")]
        for i, (name, color, label) in enumerate(columns):
            finals = (np.array([h.best[-1] for h in societies[name]]) / ref_len - 1) * 100
            ax_end.scatter(i + rng.uniform(-0.18, 0.18, len(finals)), finals, s=12,
                           color=color, alpha=0.7)
            ax_end.hlines(np.median(finals), i - 0.3, i + 0.3, color=color, lw=2.5)
        ax_end.axhline(0, color=REF, lw=1.2, ls=":")
        ax_end.set_xticks(range(len(columns)), [c[2] for c in columns])
        ax_end.set_ylabel("best trip at the end, % longer than the yardstick")
        ax_end.set_title("Where each run ends (bar = median)", fontsize=10)
        ax_end.grid(alpha=0.3, axis="y")

    fig.suptitle("Experiment 2: does it matter how the rulers are chosen?", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)


STEP_COLORS = {"order crossover": "#f18f01", "edge crossover": BLUE, "reversal": "#9b5de5"}
STEP_LABELS = {"order crossover": "order crossover (textbook)",
               "edge crossover": "edge crossover (keeps the parents' links)",
               "reversal": "reversing a random stretch"}
SOCIETY_STYLES = {"no rulers": (GREY, ":"), "lottery-5": ("#f18f01", "-"),
                  "baseline": ("black", "--"), "merit-5": (BLUE, "-")}
SOCIETY_LABELS = {"no rulers": "no rulers", "lottery-5": "5 rulers by lottery",
                  "baseline": "baseline (2 fittest kept)", "merit-5": "5 rulers by merit"}


def plot_reproduction(path: Path, damage: dict, steps, societies: dict, ways, names,
                      ref_len: float, budget: int) -> None:
    """Top: how often one step of making a child spoils the trip. Bottom: the
    same four societies under three ways of making children."""
    fig = plt.figure(figsize=(14, 8.6), layout="constrained")
    grid = fig.add_gridspec(2, len(ways))
    ax_dmg = fig.add_subplot(grid[0, :])

    stages = list(damage)
    labels = ["random\nstart" if s == 0 else f"baseline after\n{s:,} generations"
              if isinstance(s, int) else "no rulers (textbook),\nend of run" for s in stages]
    width = 0.8 / len(steps)
    for j, step in enumerate(steps):
        shares = [damage[s][step][0] * 100 for s in stages]
        xs = np.arange(len(stages)) + (j - (len(steps) - 1) / 2) * width
        ax_dmg.bar(xs, shares, width * 0.92, color=STEP_COLORS[step], label=STEP_LABELS[step])
        for x, v in zip(xs, shares):
            ax_dmg.text(x, v + 1.5, f"{v:.0f}%", ha="center", fontsize=7.5)
    ax_dmg.set_xticks(range(len(stages)), labels, fontsize=9)
    ax_dmg.set_ylim(0, 110)
    ax_dmg.set_ylabel("children worse than\ntheir better parent (%)")
    ax_dmg.set_title("How often a single step of making a child spoils the trip", fontsize=10)
    ax_dmg.legend(fontsize=8, frameon=False, loc="upper left")
    ax_dmg.grid(alpha=0.3, axis="y")

    for col, way in enumerate(ways):
        ax = fig.add_subplot(grid[1, col])
        for name in names:
            hs = societies[(way, name)]
            x = hs[0].measured / 1000
            gap = (np.median([h.best for h in hs], axis=0) / ref_len - 1) * 100
            color, ls = SOCIETY_STYLES[name]
            # Best trip once the budget is spent (the last generation may go slightly over it).
            at_budget = [h.best[int(np.searchsorted(h.measured, budget, side="right")) - 1]
                         for h in hs]
            end = (np.median(at_budget) / ref_len - 1) * 100
            ax.plot(x, gap, color=color, ls=ls, lw=1.8,
                    label=f"{SOCIETY_LABELS[name]}: {end:+.1f}% at the end")
        ax.axhline(0, color=REF, lw=1.2, ls=":")
        ax.set_ylim(-10, 200)
        ax.set_xlim(0, budget / 1000)
        ax.set_xlabel("trips measured (thousands)")
        if col == 0:
            ax.set_ylabel("best trip, % longer than the yardstick\n(median of runs)")
        ax.set_title(f"Making children: {way}", fontsize=10)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle("Experiment 3: how much damage does making a child do?", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)
