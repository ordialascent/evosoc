# First draft: one-dimensional table

> Earlier draft, kept for reference. The [two-dial grid](README.md) supersedes it.

Each row is one lever of a genetic algorithm. The baseline is in the middle; the outer columns push the lever to a socialist or an anarcho-capitalist extreme. Rows marked † are existing techniques from the GA literature.

| # | Lever | Socialist (collective, planned) | **Baseline (textbook GA)** | Anarcho-capitalist (individual, owned) |
|---|---|---|---|---|
| 1 | **Who reproduces** | Everyone gets an equal or planned share of offspring | Tournament selection: the fitter of a few random picks reproduces | Reproduction is bought with wealth you have built up; no limit |
| 2 | **What you're rewarded for** | The group's overall output, shared out | Your own current quality, with no memory of the past | Your own quality, kept as property that builds up over time |
| 3 | **Inheritance** | None: everything returns to a common pool each generation | Nothing to inherit, because the baseline has no resources | Children inherit their parents' wealth (dynasties) |
| 4 | **Who mates with whom** | The centre assigns pairs, for example to keep diversity | Random pairing among those selected | Mate choice: you pick or bid for the best partners† |
| 5 | **Mutation rate** | One rate for everyone, which the centre adjusts from feedback (1/5 success rule†) | Fixed rate, set beforehand | Each individual carries its own rate in its genes and it evolves (self-adaptation†) |
| 6 | **Variation method** | The centre picks the method that's working best and everyone uses it (adaptive operator selection†) | One fixed crossover and one fixed mutation | Each individual owns and uses its own method |
| 7 | **How new solutions are made** | The centre studies the best solutions and designs the next generation from that picture (EDA†) | Parents combine (crossover) and mutate | Same as baseline, but who mates is a private deal |
| 8 | **Survival** | Fixed, guaranteed lifespan for all | The whole population is replaced each generation, except a few of the best | You survive as long as you can pay upkeep; the bankrupt die |
| 9 | **Elitism** | None, or the best solution is public property | The 1–2 best are copied unchanged into the next generation | The wealthy persist without limit |
| 10 | **Knowledge sharing** | The best solution is published to everyone | Genes spread only through mating | Genes are property: copying them means paying the owner |
| 11 | **Extra improvement effort** | Every individual gets the same extra improvement effort (local search) | Each child is evaluated once, with no extra effort | The wealthy buy extra improvement effort (capital investment) |
| 12 | **Group structure** | One organised collective, by design | One mixed population | Groups form and break up voluntarily (firms, contracts) |
| 13 | **Who sets the rules and parameters** | A ruling class of k members | The designer, fixed before the run | Nobody; behaviour emerges from individual choices |

Held fixed across all variants: the problem, the population size, the total number of evaluations, and the code that implements each operator.
