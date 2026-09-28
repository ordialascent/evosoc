# The "normal" baseline GA: design notes

> Status: **decided and implemented** as `steady_state_ga`
> (`src/evosoc/algorithms/steady_state.py`). The final spec is at the bottom.

## The brief

A standard GA that is known to solve TSP reasonably well, and that, among
the well-established options, stays **as close as possible to what happens in
biological evolution**. It should be a credible baseline, not a straw man.

Guiding test for every component: *could this plausibly happen in a natural
population without anyone having global knowledge of the population?* When
the most natural option is known not to work, pick the most natural option
that does, and write down why.

Explicitly out of scope for the baseline, because it is engineering rather
than evolution: local search/2-opt hill-climbing on offspring (Lamarckian
memetic algorithms), heuristic seeding (e.g. nearest-neighbour tours),
restarts, and adaptive parameter control.

## Component by component

### Representation
Permutation of cities (path representation), the standard choice.
Biological note: the heritable unit that matters for TSP is the **edge**
(which city follows which). This is why edge-preserving operators are the
more faithful analogue of genes being passed on.

### Population
Fixed size *N* (a carrying capacity), random initialisation, **panmictic**
(any individual can mate with any other). Typical *N*: 100–500.

### Parent selection: the "naturalness" ladder

| Scheme | How natural? | Works on TSP? | Notes |
|---|---|---|---|
| Truncation / threshold (top *k* %) | ✗ Hard cut-off. Needs a global ranking. | Yes, but converges fast | The "threshold" method: animal breeding, not nature |
| Linear ranking | ✗ Needs a global ranking | Yes | |
| Fitness-proportionate (roulette, SUS) | ✓ Classic Holland GA. Expected offspring ∝ fitness mirrors **fecundity selection**. | Poorly without fitness scaling: tour lengths are all similar, so pressure is weak, or one super-individual takes over | Needs an arbitrary fitness transform (1/length, windowing, sigma scaling), which is itself engineering |
| **Tournament (small *k*, e.g. 2)** | ✓ **Local contests** between a few individuals that happen to meet, with no global knowledge. Resembles mate competition and territorial contests. | Yes: robust, standard, invariant to fitness scaling | Pressure is set by *k*; *k* = 2 is mild |

**Decision:** binary tournament. It is the most natural scheme that is also
robust. Fitness-proportionate selection is the "textbook natural" choice, but
it only works with a fitness transform that nature doesn't have. It could be
an alternative baseline if we want to be purist.

### Survival / replacement

| Scheme | Natural analogue | Notes |
|---|---|---|
| Generational (all parents die) | Annual plants, mayflies: non-overlapping generations | The classic "simple GA". Loses the best individual without elitism. |
| Generational + elitism | None: an immortal champion is unnatural | Common in practice. Not needed for reporting, since best-so-far is tracked by the harness anyway. |
| **Steady-state** (one birth, one death at a time) | Most animals: **overlapping generations**, continuous births and deaths | Death chosen by a *reverse* tournament (the less fit of two dies) is local and natural, and implicitly keeps good individuals |

**Decision:** steady-state with reverse-tournament death (overlapping
generations, continuous births and deaths).

### Recombination (sexual reproduction, two parents)

| Operator | Heredity | Notes |
|---|---|---|
| PMX (Goldberg & Lingle 1985) | Positions | Weak for TSP: position is not what makes a tour good |
| **OX, order crossover** (Davis 1985) | Relative order | Simple, widely used, works decently |
| **ERX, edge recombination** (Whitley et al. 1989) | **Edges (adjacencies)**. The child is built almost entirely from its parents' edges. | The most faithful "genes are inherited" analogue for TSP, and clearly stronger than OX |
| EAX, edge assembly (Nagata) | Edges | State of the art, but so engineered that it is effectively a local optimiser. Excluded. |

**Decision:** ERX, including the preference for edges both parents share
(Starkweather et al. 1991): a trait carried by both parents is always passed
on. Chosen because it is the more faithful model of heredity. The data
(below) shows it is also much faster, though not better at the end. OX stays
available as `crossover = "ox"`. Crossover rate high
(0.8–1.0): sexually reproducing species always recombine.

### Mutation

| Operator | Natural analogue | Notes |
|---|---|---|
| Swap two cities | Point-like change | Weak; breaks 4 edges |
| **Inversion** (reverse a segment) | A real biological mutation: **chromosomal inversion** | Changes exactly 2 edges (a 2-opt move). The best-performing classical TSP mutation. |
| Displacement / insertion | Transposition ("jumping genes") | Plausible second choice |

**Decision:** inversion, applied to each offspring with a small probability
(e.g. 0.1–0.3), with a uniformly random segment.

## The baseline, as implemented

```
population   N = 200, random permutations, panmictic, fixed size
per birth:
  mate choice  two parents, each the winner of a binary tournament (distinct individuals)
  reproduce    ERX crossover with p = 0.9, otherwise a clone of the first parent
  mutate       one inversion with p = 0.2 (segment length 2 .. n-2)
  death        newborn joins; the loser of a reverse binary tournament dies
               (option: death = "random")
reporting    every N births ("generation equivalent")
```

Consequence of the death rule: the current best individual can never die (it
is never the less fit of two), so no explicit elitism is needed.

## What the baseline does (measured)

`configs/baseline.toml`: 200k evaluations, 20 seeds. Median final gap to the optimum:

| | eil51 | berlin52 | kroA100 |
|---|---|---|---|
| baseline (ERX) | 6.5 % | 8.9 % | 10.4 % |
| same GA with OX | 4.8 % | 6.7 % | 10.8 % |
| random search | 190 % | 197 % | 533 % |

* It works: it is far better than random search on every instance.
* **ERX vs OX:** ERX is far ahead for most of the run (anytime gap better in
  every run, A12 = 0 for OX). At the end they are level, except on berlin52,
  where OX is slightly better (A12 = 0.75, p = 0.02).
* **Premature convergence, the classic panmictic-GA failure:** genotypic
  diversity falls from about 95 % to about 1 % within roughly 8 000
  evaluations (40 generation-equivalents), and about 90 % of the population
  becomes clones. Most of the budget is then spent refining a single family.
  This is a known property of a well-mixed population under selection, not
  a bug. It is exactly what the diversity panels are there to show.

## Parameter check

`configs/baseline-check.toml`: one change at a time from the default, 10 seeds.

| change | berlin52 final gap | kroA100 final gap | verdict |
|---|---|---|---|
| default (N = 200, mutation 0.2, tournament death) | 8.0 % | 10.3 % | |
| mutation on every birth | 4.7 % | 12.2 % | mixed; keeps diversity at ~15–19 % |
| random death | 8.1 % | 25.9 % | worse (significant on kroA100) |
| N = 1000 | **1.3 %** | 9.0 % | better at the end, slower until then: a budget trade-off |
| OX crossover | 4.8 % | 10.8 % | level at the end, slower |

No change wins clearly, so the conventional defaults stay. This keeps the
baseline untuned, and any variant should get the same (zero) tuning effort.

## Not adopted (for now)

* **A second, purist reference with fitness-proportionate selection.** Could
  be added later as one more variant if the question comes up.
* **Generational replacement.** Same: one more variant if it is ever needed.

## Still open

1. **Budget and seeds.** The baseline levels off at around 20–50k
   evaluations, so 200k is comfortably past that. Use at least 20 seeds for
   claims.
2. **Population size.** N = 1000 gives a clearly better end result on
   berlin52. Worth revisiting if the experiments focus on final quality
   rather than speed.
