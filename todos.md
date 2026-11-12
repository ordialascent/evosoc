# evosoc

All to-dos for the evosoc experiment: controlled experiments with genetic algorithms (TSP) as an abstract metaphor for society. One fixed, textbook baseline GA; variants change the social mechanics and are compared against it under identical conditions.

## Variants

### Variant 1: Autonomous governors

**Idea:** Each governor acts autonomously on the population. A governor goes out, takes two members of the population, and does something with them. The population's fitness depends on what the governors do, and governors are ranked on the fitness of the population.

**Fixed so far:** governors take seats from the 200 and do no route optimization themselves. No extra fitness evaluations. Everything else as in the baseline.

**To flesh out later:** what exactly a governor does with the two, how a governor's rank is derived from population fitness, and what follows from the ranking.

### Variant 2: Collective council

**Idea:** Each governor contributes to one collective decision on how the council interacts with the population. Governors don't act on the population individually; the council acts as one.

**Fixed so far:** governors take seats from the 200 and do no route optimization themselves. No extra fitness evaluations. Everything else as in the baseline.

**To flesh out later:** what each governor contributes, how contributions become one decision, and whether/how governors are ranked or replaced.

## Central driver candidates

### 1/5 success rule

**Candidate central driver (textbook): Rechenberg's 1/5 success rule** (evolution strategies, 1970s).

**How it works:** the centre counts how many mutations lead to an improvement. If more than one fifth succeed, it makes mutations bigger; if fewer, smaller.

**Why it fits:** the purest central planner. One statistic, one rule, one lever for everyone.

**Place in the taxonomy:** adaptive parameter control (Eiben, Hinterding & Michalewicz, 1999). Baseline = parameter tuning; this = the centre adjusts based on feedback.

**To think about:** what "mutation size" means for inversion on a tour, and which lever the rule would drive.

### Adaptive operator selection

**Candidate central driver (textbook): adaptive operator selection.**

**How it works:** the centre has several tools (e.g. different crossovers or mutations), tracks which one produced good children recently, and uses the successful ones more often.

**Why it fits:** a ministry allocating budget to whichever department delivers.

**Place in the taxonomy:** adaptive parameter control (Eiben, Hinterding & Michalewicz, 1999).

**To think about:** which tools the centre chooses between without breaking the "baseline methods stay identical" rule.

### Estimation of distribution algorithm (EDA)

**Candidate central driver (textbook): estimation of distribution algorithms (EDAs).**

**How it works:** individuals don't mate. The centre looks at the best part of the population, builds a statistical picture of what good solutions look like, and produces the next generation from that picture.

**Why it fits:** central planning in the literal sense. The centre designs the next generation instead of letting it arise from pairings. The most radical of the central drivers.

**To think about:** it replaces crossover and mutation entirely, so it breaks "baseline methods stay identical." Is that acceptable for the centralization story, or too big a change to compare fairly?

### Meta-GA

**Candidate central driver (textbook): meta-GA** (Grefenstette, 1986).

**How it works:** an outer GA evolves the settings of an inner GA. Each outer individual is a set of settings; it is judged by how well the inner GA does with them.

**Why it fits:** the textbook ancestor of "governors as chromosomes." The governors are the outer population, the governed are the inner one.

**To think about:** classic meta-GAs run many inner GAs, which costs evaluations. Here it would have to run within one population and one budget, so governors are judged on the shared population, not separate runs.

## Decentralized candidates

### Self-adaptive parameters (anarcho-capitalist)

**Candidate variant (textbook): self-adaptive parameters.** The anarcho-capitalist end of the spectrum.

**How it works:** the rules move from the programmer into the individuals. Each route carries its own settings (e.g. mutation rate, mating preferences, maybe crossover rate) and passes them on to its children. Settings that work spread with their carriers; settings that don't die out.

**Why it fits:** no central authority and no fixed law beyond physics. Each agent carries the full weight of its own decisions.

**Place in the taxonomy:** self-adaptive parameter control (Eiben, Hinterding & Michalewicz, 1999); standard in evolution strategies.

**Spectrum:** council (the centre sets rules as it goes) → baseline (rules fixed from outside, night-watchman state) → self-adaptive (each individual carries its own rules).

**To think about:**

- It breaks the baseline rule of "no adaptive parameters," so it is a genuinely different system, not the baseline with a twist.
- Which parameters count as laws of physics (not decentralizable: fitness function, budget, probably carrying capacity of 200) and which count as laws of the state. "The worse of two dies" could be either.
