# The "normal" baseline GA: design notes

> Status: **notes for the next step**. Nothing here is implemented yet.

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

**Leaning:** binary tournament. It is the most natural scheme that is also
robust. Fitness-proportionate selection is the "textbook natural" choice, but
it only works with a fitness transform that nature doesn't have. It could be
an alternative baseline if we want to be purist.

### Survival / replacement

| Scheme | Natural analogue | Notes |
|---|---|---|
| Generational (all parents die) | Annual plants, mayflies: non-overlapping generations | The classic "simple GA". Loses the best individual without elitism. |
| Generational + elitism | None: an immortal champion is unnatural | Common in practice. Not needed for reporting, since best-so-far is tracked by the harness anyway. |
| **Steady-state** (one birth, one death at a time) | Most animals: **overlapping generations**, continuous births and deaths | Death chosen by a *reverse* tournament (the less fit of two dies) is local and natural, and implicitly keeps good individuals |

**Leaning:** steady-state with reverse-tournament death, or a plain
generational GA without elitism if we want the most textbook version.
**To decide.**

### Recombination (sexual reproduction, two parents)

| Operator | Heredity | Notes |
|---|---|---|
| PMX (Goldberg & Lingle 1985) | Positions | Weak for TSP: position is not what makes a tour good |
| **OX, order crossover** (Davis 1985) | Relative order | Simple, widely used, works decently |
| **ERX, edge recombination** (Whitley et al. 1989) | **Edges (adjacencies)**. The child is built almost entirely from its parents' edges. | The most faithful "genes are inherited" analogue for TSP, and clearly stronger than OX |
| EAX, edge assembly (Nagata) | Edges | State of the art, but so engineered that it is effectively a local optimiser. Excluded. |

**Leaning:** ERX, with OX as the simpler fallback. Crossover rate high
(0.8–1.0): sexually reproducing species always recombine.

### Mutation

| Operator | Natural analogue | Notes |
|---|---|---|
| Swap two cities | Point-like change | Weak; breaks 4 edges |
| **Inversion** (reverse a segment) | A real biological mutation: **chromosomal inversion** | Changes exactly 2 edges (a 2-opt move). The best-performing classical TSP mutation. |
| Displacement / insertion | Transposition ("jumping genes") | Plausible second choice |

**Leaning:** inversion, applied to each offspring with a small probability
(e.g. 0.1–0.3), with a uniformly random segment.

## Candidate baseline (to confirm next step)

```
population   N = 200, random permutations, panmictic
mate choice  binary tournament (×2 → two parents)
reproduce    ERX crossover (p = 0.9), else copy a parent
mutate       inversion (p = 0.2 per offspring)
survival     steady-state: offspring replaces the loser of a reverse binary tournament
             (alternative: generational, no elitism)
budget       e.g. 100·n² evaluations? decide per instance set
```

## Open questions

1. Steady-state (overlapping generations) or generational?
2. ERX (more faithful, slower) or OX (simpler, faster)?
3. Is a purist fitness-proportionate variant worth keeping as a second reference?
4. Budget and number of seeds: once the baseline runs, measure where its
   convergence plateaus and set the budget past it.
5. Performance: pure Python/NumPy handles 50–150 cities comfortably. If it
   gets slow, the operators can be JIT-compiled with Numba without touching
   the harness.
