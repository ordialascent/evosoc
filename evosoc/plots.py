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


NOBODY = "nobody protected"  # no member is kept: everyone is replaced by a child


def _ruler_colors(ks) -> dict:
    """Darker = a larger protected class. The baseline (2 protected) is drawn in black."""
    cmap = plt.get_cmap("plasma")
    others = [k for k in ks if k != 2]
    colors = {k: cmap(0.85 - 0.7 * i / max(len(others) - 1, 1)) for i, k in enumerate(others)}
    colors[2] = "black"
    return colors


def _ruler_label(k: int) -> str:
    """Plain name for a society whose k fittest are protected from replacement."""
    if k == 0:
        return NOBODY
    name = f"{k} protected"
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
    ax_end.set_xlabel("size of the protected class")
    ax_end.set_ylabel("best trip at the end, % longer than the yardstick")
    ax_end.set_title(f"Where each run ends ({len(societies[ks[0]])} runs, bar = median)",
                     fontsize=10)
    ax_end.grid(alpha=0.3, axis="y")

    fig.suptitle("Experiment 1: the k fittest are protected: they always survive, unchanged",
                 fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)


WAY_COLORS = {"merit": BLUE, "hereditary": "#9b5de5", "lottery": "#f18f01"}
WAY_LABELS = {"merit": "by merit (the fittest)", "hereditary": "by birth (a protected member's child)",
              "lottery": "by lottery (at random)"}


def plot_choosing(path: Path, societies: dict, rulers, ways, ref_len: float,
                  checkpoints=()) -> None:
    """One row per size of protected class: the same number protected, chosen in different ways."""
    fig, axes = plt.subplots(len(rulers), 3, figsize=(14, 4.2 * len(rulers)),
                             layout="constrained", squeeze=False,
                             gridspec_kw={"width_ratios": [1.4, 1.4, 1]})

    def curves(hs):
        x = hs[0].measured / 1000
        gap = (np.median([h.best for h in hs], axis=0) / ref_len - 1) * 100
        div = np.median([h.diversity for h in hs], axis=0)
        return x, gap, div

    references = [("baseline", "baseline (2 fittest kept)", "black", "--"),
                  ("none", NOBODY, GREY, ":")]
    rng = np.random.default_rng(0)  # only spreads the dots sideways so they do not overlap

    for row, k in zip(axes, rulers):
        ax_gap, ax_div, ax_end = row
        for name, label, color, ls in references:
            x, gap, div = curves(societies[name])
            ax_gap.plot(x, gap, color=color, lw=1.4, ls=ls, label=label)
            ax_div.plot(x, div, color=color, lw=1.4, ls=ls)
        for way in ways:
            x, gap, div = curves(societies[f"{way}-{k}"])
            ax_gap.plot(x, gap, color=WAY_COLORS[way], lw=1.9, label=f"{k} protected {WAY_LABELS[way]}")
            ax_div.plot(x, div, color=WAY_COLORS[way], lw=1.9)

        for ax in (ax_gap, ax_div):
            for c in checkpoints:
                ax.axvline(c / 1000, color=GREY, lw=0.8, alpha=0.5)
            ax.set_xlabel("trips measured (thousands): the same budget for everyone")
            ax.grid(alpha=0.3)
        ax_gap.axhline(0, color=REF, lw=1.2, ls=":", label="yardstick")
        ax_gap.set_ylim(-10, 200)
        ax_gap.set_ylabel("best trip, % longer than the yardstick")
        ax_gap.set_title(f"{k} protected: how fast the best trip improves (median of runs)",
                         fontsize=10)
        ax_gap.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1, 0.06))
        ax_div.set_ylim(0, 1)
        ax_div.set_ylabel("difference from the best member")
        ax_div.set_title(f"{k} protected: how varied the population stays", fontsize=10)

        columns = [(f"{w}-{k}", WAY_COLORS[w], w if w != "hereditary" else "birth")
                   for w in ways] + [("baseline", "black", "baseline"), ("none", GREY, "nobody\nprotected")]
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

    fig.suptitle("Experiment 2: does it matter who is protected?", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)


STEP_COLORS = {"order crossover": "#f18f01", "edge crossover": BLUE, "reversal": "#9b5de5"}
STEP_LABELS = {"order crossover": "order crossover (textbook)",
               "edge crossover": "edge crossover (keeps the parents' links)",
               "reversal": "reversing a random stretch"}
SOCIETY_STYLES = {"no rulers": (GREY, ":"), "lottery-5": ("#f18f01", "-"),
                  "baseline": ("black", "--"), "merit-5": (BLUE, "-")}
SOCIETY_LABELS = {"no rulers": NOBODY, "lottery-5": "5 protected by lottery",
                  "baseline": "baseline (2 fittest kept)", "merit-5": "5 protected by merit"}


def plot_reproduction(path: Path, damage: dict, steps, societies: dict, ways, names,
                      ref_len: float, budget: int) -> None:
    """Top: how often one step of making a child spoils the trip. Bottom: the
    same four societies under three ways of making children."""
    fig = plt.figure(figsize=(14, 8.6), layout="constrained")
    grid = fig.add_gridspec(2, len(ways))
    ax_dmg = fig.add_subplot(grid[0, :])

    stages = list(damage)
    labels = ["random\nstart" if s == 0 else f"baseline after\n{s:,} generations"
              if isinstance(s, int) else "nobody protected (textbook),\nend of run" for s in stages]
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


OWNERSHIP_STYLES = {
    "no rulers": (GREY, ":"),
    "baseline": ("black", "--"),
    "pooled": ("#d1495b", "-"),
    "half pooled": ("#f18f01", "-"),
    "kept": ("#2a6fdb", "-"),
    "kept, half inherited": ("#6a4c93", "-"),
    "kept, mostly inherited": ("#2d6a4f", "-"),
}
OWNERSHIP_LABELS = {
    "no rulers": "nobody protected, no wealth",
    "baseline": "baseline (2 fittest kept, no wealth)",
    "pooled": "all earnings pooled",
    "half pooled": "half pooled, half kept",
    "kept": "all earnings kept",
    "kept, half inherited": "kept, children inherit half",
    "kept, mostly inherited": "kept, children inherit 90%",
}
OWNERSHIP_TICKS = {
    "no rulers": "nobody\nprotected", "baseline": "base-\nline", "pooled": "pooled",
    "half pooled": "half\npooled", "kept": "kept", "kept, half inherited": "kept,\ninherit\nhalf",
    "kept, mostly inherited": "kept,\ninherit\n90%",
}


def plot_ownership(path: Path, societies: dict, ways, names, ref_len: float, budget: int) -> None:
    """One row per way of making children: progress, wealth inequality, end results."""
    fig, axes = plt.subplots(len(ways), 3, figsize=(15, 4.4 * len(ways) + 0.8), layout="constrained",
                             squeeze=False, gridspec_kw={"width_ratios": [1.4, 1.1, 1.1]})
    rng = np.random.default_rng(0)  # only spreads the dots sideways so they do not overlap
    for row, way in zip(axes, ways):
        ax_gap, ax_gini, ax_end = row
        for name in names:
            hs = societies[(way, name)]
            color, ls = OWNERSHIP_STYLES.get(name, (GREY, "-"))
            label = OWNERSHIP_LABELS.get(name, name)
            x = hs[0].measured / 1000
            gap = (np.median([h.best for h in hs], axis=0) / ref_len - 1) * 100
            ax_gap.plot(x, gap, color=color, ls=ls, lw=1.8, label=label)
            if name not in ("no rulers", "baseline"):  # the references have no wealth
                ax_gini.plot(x[1:], np.median([h.wealth_gini for h in hs], axis=0),
                             color=color, ls=ls, lw=1.8, label=label)
        ax_gap.axhline(0, color=REF, lw=1.2, ls=":", label="yardstick")
        ax_gap.set_ylim(-10, 270 if way == "textbook" else 60)
        ax_gap.set_xlim(0, budget / 1000)
        ax_gap.set_xlabel("trips measured (thousands): the same budget for everyone")
        ax_gap.set_ylabel("best trip, % longer than the yardstick\n(median of runs)")
        ax_gap.set_title(f"Making children: {way}. How fast the best trip improves", fontsize=10)
        ax_gap.grid(alpha=0.3)

        ax_gini.set_ylim(-0.02, 0.6)
        ax_gini.set_xlim(0, budget / 1000)
        ax_gini.set_xlabel("trips measured (thousands)")
        ax_gini.set_ylabel("wealth inequality (0 = all equal)")
        ax_gini.set_title("How unequal wealth is (median of runs)", fontsize=10)
        ax_gini.grid(alpha=0.3)

        for i, name in enumerate(names):
            hs = societies[(way, name)]
            at_budget = [h.best[int(np.searchsorted(h.measured, budget, side="right")) - 1]
                         for h in hs]
            ends = (np.array(at_budget) / ref_len - 1) * 100
            color, _ = OWNERSHIP_STYLES.get(name, (GREY, "-"))
            ax_end.scatter(i + rng.uniform(-0.18, 0.18, len(ends)), ends, s=12, color=color,
                           alpha=0.7)
            ax_end.hlines(np.median(ends), i - 0.3, i + 0.3, color=color, lw=2.5)
        ax_end.axhline(0, color=REF, lw=1.2, ls=":")
        ax_end.set_xticks(range(len(names)), [OWNERSHIP_TICKS.get(n, n) for n in names],
                          fontsize=8)
        ax_end.set_ylabel("best trip at the end, % longer than the yardstick")
        ax_end.set_title("Where each run ends (bar = median)", fontsize=10)
        ax_end.grid(alpha=0.3, axis="y")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("Experiment 4: who owns what a member produces?", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)


CAPITAL_STYLES = {
    "no rulers": (GREY, ":"),
    "kept": ("#2a6fdb", "--"),
    "split": ("#6a4c93", "-"),
    "fittest heir": ("#2a9d8f", "-"),
    "first-born": ("#d1495b", "-"),
    "split, +50%": ("#9d79bc", "-."),
    "first-born, +50%": ("#f18f01", "-."),
    "split, +100%": ("#b8a1d9", ":"),
    "fittest heir, +100%": ("#7fc8bf", ":"),
    "first-born, +100%": ("#8c2f39", ":"),
}
CAPITAL_LABELS = {
    "no rulers": "no wealth (contest for mothers)",
    "kept": "earnings kept, nothing inherited",
    "split": "half inherited, split between children",
    "fittest heir": "half inherited, all to the fittest child",
    "first-born": "half inherited, all to the first-born",
    "split, +50%": "split, inherited wealth grows 50% a generation",
    "first-born, +50%": "first-born, inherited wealth grows 50%",
    "split, +100%": "split, inherited wealth doubles each generation",
    "fittest heir, +100%": "fittest child, inherited wealth doubles",
    "first-born, +100%": "first-born, inherited wealth doubles",
}
CAPITAL_TICKS = {
    "no rulers": "no\nwealth", "kept": "kept", "split": "split", "fittest heir": "fittest\nheir",
    "first-born": "first-\nborn", "split, +50%": "split\n+50%", "first-born, +50%": "first-\nborn\n+50%",
    "split, +100%": "split\n+100%", "fittest heir, +100%": "fittest\n+100%",
    "first-born, +100%": "first-\nborn\n+100%",
}


def plot_capital(path: Path, societies: dict, ways, names, ref_len: float, budget: int) -> None:
    """One row per way of making children: progress, how much the richest
    tenth own, whether children still go to good members, end results."""
    fig, axes = plt.subplots(len(ways), 4, figsize=(19, 4.4 * len(ways) + 1.0),
                             layout="constrained", squeeze=False,
                             gridspec_kw={"width_ratios": [1.3, 1, 1, 1.15]})
    rng = np.random.default_rng(0)  # only spreads the dots sideways so they do not overlap
    for row, way in zip(axes, ways):
        ax_gap, ax_top, ax_merit, ax_end = row
        for name in names:
            hs = societies[(way, name)]
            color, ls = CAPITAL_STYLES.get(name, (GREY, "-"))
            label = CAPITAL_LABELS.get(name, name)
            x = hs[0].measured / 1000
            gap = (np.median([h.best for h in hs], axis=0) / ref_len - 1) * 100
            ax_gap.plot(x, gap, color=color, ls=ls, lw=1.8, label=label)
            if name != "no rulers":  # the reference has no wealth
                ax_top.plot(x[1:], np.median([h.top_tenth_share for h in hs], axis=0) * 100,
                            color=color, ls=ls, lw=1.8)
            ax_merit.plot(x[1:], np.median([h.mothers_better_half for h in hs], axis=0) * 100,
                          color=color, ls=ls, lw=1.8)
        ax_gap.axhline(0, color=REF, lw=1.2, ls=":", label="yardstick")
        ax_gap.set_ylim(-10, 270 if way == "textbook" else 80)
        ax_gap.set_xlim(0, budget / 1000)
        ax_gap.set_xlabel("trips measured (thousands): the same budget for everyone")
        ax_gap.set_ylabel("best trip, % longer than the yardstick\n(median of runs)")
        ax_gap.set_title(f"Making children: {way}. How fast the best trip improves", fontsize=10)
        ax_gap.grid(alpha=0.3)

        ax_top.axhline(10, color=GREY, lw=1, ls=":")
        ax_top.set_ylim(0, 100)
        ax_top.set_xlim(0, budget / 1000)
        ax_top.set_xlabel("trips measured (thousands)")
        ax_top.set_ylabel("% of all wealth owned by the richest tenth")
        ax_top.set_title("How much the richest tenth own\n(dotted: 10% = all equal)", fontsize=10)
        ax_top.grid(alpha=0.3)

        ax_merit.axhline(50, color=GREY, lw=1, ls=":")
        ax_merit.set_ylim(40, 100)
        ax_merit.set_xlim(0, budget / 1000)
        ax_merit.set_xlabel("trips measured (thousands)")
        ax_merit.set_ylabel("% of children whose mother's trip\nwas in the better half")
        ax_merit.set_title("Do children still go to good members?\n(dotted: 50% = no link at all)",
                           fontsize=10)
        ax_merit.grid(alpha=0.3)

        for i, name in enumerate(names):
            hs = societies[(way, name)]
            at_budget = [h.best[int(np.searchsorted(h.measured, budget, side="right")) - 1]
                         for h in hs]
            ends = (np.array(at_budget) / ref_len - 1) * 100
            color, _ = CAPITAL_STYLES.get(name, (GREY, "-"))
            ax_end.scatter(i + rng.uniform(-0.18, 0.18, len(ends)), ends, s=12, color=color,
                           alpha=0.7)
            ax_end.hlines(np.median(ends), i - 0.3, i + 0.3, color=color, lw=2.5)
        ax_end.axhline(0, color=REF, lw=1.2, ls=":")
        ax_end.set_xticks(range(len(names)), [CAPITAL_TICKS.get(n, n) for n in names],
                          fontsize=7.5)
        ax_end.set_ylabel("best trip at the end, % longer than the yardstick")
        ax_end.set_title("Where each run ends (bar = median)", fontsize=10)
        ax_end.grid(alpha=0.3, axis="y")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("Experiment 8: capital that earns a return, and a single heir", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pressure(path: Path, societies: dict, maps: dict, summary: dict, cities, ways, rulers,
                  budgets, marks, title: str | None = None) -> None:
    """One row per way of making children. Left: one chart per map size, the
    best trip as the budget is used up. Right: which society is ahead at each
    budget on each map."""
    colors = _ruler_colors(rulers)
    if 0 in colors:
        colors[0] = "#2a9d8f"

    def name(k: int) -> str:
        return _ruler_label(k)

    fig = plt.figure(figsize=(17, 4.6 * len(ways)), layout="constrained")
    grid = fig.add_gridspec(len(ways), len(cities) + 1, width_ratios=[1] * len(cities) + [1.15])
    ticks = [0, 5, 10, 25, 50, 100, 200, 400]
    for row, way in enumerate(ways):
        for col, n in enumerate(cities):
            ax = fig.add_subplot(grid[row, col])
            for k in rulers:
                hs = societies[(n, way, k)]
                x = hs[0].measured / 1000
                ratio = np.median([h.best for h in hs], axis=0) / maps[n]
                keep = hs[0].measured <= marks[-1]
                ax.plot(x[keep], ratio[keep], color=colors[k], lw=1.8,
                        ls="--" if k == 2 else "-", label=name(k))
            for g, m in zip(budgets, marks):
                ax.axvline(m / 1000, color=GREY, lw=0.8, alpha=0.6)
            ax.axhline(1, color=REF, lw=1.2, ls=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1, marks[-1] / 1000 * 1.05)
            ax.set_ylim(0.95, 5)
            ax.set_yticks([1 + t / 100 for t in ticks], [f"+{t}%" for t in ticks])
            ax.minorticks_off()
            xt = [t for t in (1, 3, 10, 30, 100, 300, 1000) if t <= marks[-1] / 1000 * 1.05]
            ax.set_xticks(xt, [f"{t:,}" for t in xt])
            ax.grid(alpha=0.3)
            ax.set_title(f"{n} cities, making children: {way}", fontsize=10)
            if col == 0:
                ax.set_ylabel("best trip, % longer than the yardstick\n(median of runs)")
            if row == len(ways) - 1:
                ax.set_xlabel("trips measured (thousands, log scale)")
            if row == 0 and col == 0:
                ax.legend(fontsize=8, frameon=False, loc="upper right")

        ax = fig.add_subplot(grid[row, len(cities)])
        for i, n in enumerate(cities):
            for j, g in enumerate(budgets):
                cell = summary["results"][str(n)][way][str(g)]
                k = cell["winner"]
                clear = cell["shuffle_test_winner_vs_runner_up"] < 0.05
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=colors[k],
                                           alpha=0.85 if clear else 0.35))
                gap = cell["gap_to_reference_pct"][str(k)]["median"]
                short = {r: name(r).replace(" (baseline)", "") for r in (k, cell["runner_up"])}
                note = "clear lead" if clear else f"close to {short[cell['runner_up']]}"
                ax.text(j + 0.5, i + 0.5, f"{short[k]}\n{gap:+.1f}%\n{note}",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if clear and k != 0 else "black")
        ax.set_xlim(0, len(budgets))
        ax.set_ylim(len(cities), 0)
        ax.set_xticks(np.arange(len(budgets)) + 0.5, [f"{g:,}" for g in budgets])
        ax.set_yticks(np.arange(len(cities)) + 0.5, [f"{n} cities" for n in cities])
        ax.set_xlabel("budget (in baseline generations)")
        ax.set_title(f"Who is ahead ({way})", fontsize=10)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle(title or "Experiment 5: does protecting the top win when time is short?\n"
                 "Grey lines: the short, medium and long budget", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)


PLANNING_STYLES = {
    "nobody protected": (GREY, ":"),
    "baseline": ("black", "--"),
    "board, true map": ("#2a6fdb", "-"),
    "board, grid 20": ("#2d6a4f", "-"),
    "board, grid 10": ("#f18f01", "-"),
    "board, grid 5": ("#d1495b", "-"),
    "board, true map, different pairs": ("#2a6fdb", "-."),
    "board, grid 10, different pairs": ("#f18f01", "-."),
}
PLANNING_LABELS = {
    "nobody protected": "contests, nobody protected",
    "baseline": "contests (baseline)",
    "board, true map": "board, true map",
    "board, grid 20": "board, fine grid (20 x 20)",
    "board, grid 10": "board, medium grid (10 x 10)",
    "board, grid 5": "board, coarse grid (5 x 5)",
    "board, true map, different pairs": "board, true map, pairs different parents",
    "board, grid 10, different pairs": "board, medium grid, pairs different parents",
}
PLANNING_TICKS = {
    "nobody protected": "nobody\nprotected", "baseline": "base-\nline",
    "board, true map": "board\ntrue\nmap", "board, grid 20": "board\n20x20",
    "board, grid 10": "board\n10x10", "board, grid 5": "board\n5x5",
    "board, true map, different pairs": "board\ntrue map\ndiff.\npairs",
    "board, grid 10, different pairs": "board\n10x10\ndiff.\npairs",
}


def _draw_board_map(ax, coords: np.ndarray, grid: int, title: str) -> None:
    """The cities (dots) and where the board thinks they are (squares)."""
    ax.scatter(coords[:, 0], coords[:, 1], s=10, color="black", zorder=2)
    if grid:
        for v in np.linspace(0, 1, grid + 1):
            ax.axhline(v, color=GREY, lw=0.4, alpha=0.6)
            ax.axvline(v, color=GREY, lw=0.4, alpha=0.6)
        seen = (np.floor(coords * grid).clip(0, grid - 1) + 0.5) / grid
        for (x0, y0), (x1, y1) in zip(coords, seen):
            ax.plot([x0, x1], [y0, y1], color=REF, lw=0.6, alpha=0.7, zorder=1)
        ax.scatter(seen[:, 0], seen[:, 1], s=26, marker="s", facecolors="none",
                   edgecolors=REF, lw=1.0, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_planning(path: Path, coords, societies: dict, ways, names, ref_len: float,
                  budget: int) -> None:
    """Top: the board's maps and how often its picks are right. Then one row
    per way of making children: progress and where each run ends."""
    grids = sorted({int(n.split("grid ")[1].split(",")[0]) for n in names if "grid " in n},
                   reverse=True)
    fig = plt.figure(figsize=(15, 3.6 + 4.4 * len(ways)), layout="constrained")
    outer = fig.add_gridspec(1 + len(ways), 1, height_ratios=[3.2] + [4.4] * len(ways))
    top = outer[0].subgridspec(1, len(grids) + 3, width_ratios=[1] * (len(grids) + 1) + [0.15, 2.4])
    _draw_board_map(fig.add_subplot(top[0]), coords, 0, "The true map\n(what members know)")
    for i, g in enumerate(grids):
        _draw_board_map(fig.add_subplot(top[i + 1]), coords, g,
                        f"Board's map, {g} x {g} grid\n(squares: where it thinks cities are)")
    ax_right = fig.add_subplot(top[-1])
    for way, ls in zip(ways, ("-", "--")):
        for name in names:
            hs = societies[(way, name)]
            if not len(hs[0].planners_right) or "different" in name:
                continue
            color, _ = PLANNING_STYLES.get(name, (GREY, "-"))
            x = hs[0].measured[1:] / 1000
            right = np.median([h.planners_right for h in hs], axis=0)
            # Smooth over 25 generations so the trend is readable.
            w = max(1, min(25, len(right)))
            smooth = np.convolve(right, np.ones(w) / w, mode="valid")
            ax_right.plot(x[(w - 1) // 2:(w - 1) // 2 + len(smooth)], smooth * 100, color=color, ls=ls, lw=1.5,
                          label=f"{PLANNING_LABELS.get(name, name)} ({way})")
    ax_right.axhline(87.5, color="black", lw=1, ls="--")
    ax_right.text(budget / 1000, 88.2, "contests on the true map", ha="right", fontsize=8)
    ax_right.axhline(50, color=GREY, lw=1, ls=":")
    ax_right.text(budget / 1000, 50.7, "coin toss", ha="right", fontsize=8, color=GREY)
    ax_right.set_xlim(0, budget / 1000)
    ax_right.set_ylim(45, 95)
    ax_right.set_xlabel("trips measured (thousands)")
    ax_right.set_ylabel("% of chosen parents that\ntruly were in the better half")
    ax_right.set_title("How often the board picks the right parents\n(median of runs; solid: "
                       f"{ways[0]}, dashed: {ways[-1]})", fontsize=9)
    ax_right.grid(alpha=0.3)

    rng = np.random.default_rng(0)  # only spreads the dots sideways so they do not overlap
    first_ax = None
    for r, way in enumerate(ways):
        row = outer[r + 1].subgridspec(1, 2, width_ratios=[1.5, 1.1])
        ax_gap, ax_end = fig.add_subplot(row[0]), fig.add_subplot(row[1])
        first_ax = first_ax or ax_gap
        for name in names:
            hs = societies[(way, name)]
            color, ls = PLANNING_STYLES.get(name, (GREY, "-"))
            x = hs[0].measured / 1000
            gap = (np.median([h.best for h in hs], axis=0) / ref_len - 1) * 100
            ax_gap.plot(x, gap, color=color, ls=ls, lw=1.8, label=PLANNING_LABELS.get(name, name))
        ax_gap.axhline(0, color=REF, lw=1.2, ls=":", label="yardstick")
        ax_gap.set_ylim(-10, 200 if way == "textbook" else 60)
        ax_gap.set_xlim(0, budget / 1000)
        ax_gap.set_xlabel("trips measured (thousands): the same budget for everyone")
        ax_gap.set_ylabel("best trip, % longer than the yardstick\n(median of runs)")
        ax_gap.set_title(f"Making children: {way}. How fast the best trip improves", fontsize=10)
        ax_gap.grid(alpha=0.3)

        for i, name in enumerate(names):
            hs = societies[(way, name)]
            ends = (np.array([best_at_budget(h, budget) for h in hs]) / ref_len - 1) * 100
            color, _ = PLANNING_STYLES.get(name, (GREY, "-"))
            ax_end.scatter(i + rng.uniform(-0.18, 0.18, len(ends)), ends, s=12, color=color,
                           alpha=0.7)
            ax_end.hlines(np.median(ends), i - 0.3, i + 0.3, color=color, lw=2.5)
        ax_end.axhline(0, color=REF, lw=1.2, ls=":")
        ax_end.set_xticks(range(len(names)), [PLANNING_TICKS.get(n, n) for n in names], fontsize=7.5)
        ax_end.set_ylabel("best trip at the end, % longer than the yardstick")
        ax_end.set_title("Where each run ends (bar = median)", fontsize=10)
        ax_end.grid(alpha=0.3, axis="y")

    handles, labels = first_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle("Experiment 7: a planning board decides who has children, judging trips on "
                 "its own map", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def best_at_budget(history, budget: int) -> float:
    return float(history.best[int(np.searchsorted(history.measured, budget, side="right")) - 1])
