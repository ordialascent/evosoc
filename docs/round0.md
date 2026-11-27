# Round 0: the Baseline

Baseline v1 (`evosoc/contestants/baseline/baseline_v1.py`) is the textbook genetic algorithm with the settings that won a seeded search on training maps. It is frozen, belongs to no team, runs alongside every match as the yardstick, and is the common ancestor of both teams' first contestants. With it the season-1 toolbox is frozen (rules §1).

Rerun: `python -m evosoc tune` (code `evosoc/tuning.py`, record `data/tuning/baseline_v1.json`, 36 min on 4 cores).

## Frozen settings

| Setting | v0 (untuned) | **v1** |
|---|---|---|
| Population (toolbox) | 100 | **30** |
| Crossover (toolbox) | order crossover | **edge recombination** |
| Mutation (toolbox) | stretch reversal | stretch reversal |
| Kept unchanged per generation | 2 | **2** |
| Contest size (mother, father) | 3 | **7** |
| Children made by crossover | 90% | **100%** |
| Children with a reversed stretch | 30% | **100%** |

## Method

- Space: population {30, 50, 100, 200, 400} × crossover {order, edge} × kept {0, 1, 2, 5, 10} × contest {2, 3, 5, 7} × crossover rate {0.5, 0.7, 0.9, 1.0} × mutation rate {0.1, 0.3, 0.6, 1.0}.
- Screen: v0's settings plus 31 distinct random draws (search seed 2026), each on training maps 0–5 of every event (18 runs per setting).
- Final: the best 4 of the screen plus v0 on fresh training maps 100–119 of every event (60 runs per setting).
- Score: mean gap to the yardstick (%), averaged over the three events with equal weight. Lower is better. No match seed was used.

## Final (training maps 100–119)

Mean gap to the yardstick, %.

| Setting | Sprint | Marathon | Big Country | Score |
|---|---|---|---|---|
| **pop 30, edge, kept 2, contest 7, cross 1.0, mutate 1.0** | +5.24 | **−2.20** | **+1.38** | **+1.47** |
| pop 30, edge, kept 10, contest 5, cross 0.7, mutate 0.6 | **+3.93** | +1.34 | +2.66 | +2.64 |
| pop 50, edge, kept 5, contest 7, cross 0.5, mutate 0.6 | +6.47 | +1.97 | +2.61 | +3.68 |
| pop 30, order, kept 2, contest 5, cross 0.7, mutate 0.3 | +11.36 | +3.04 | +3.13 | +5.84 |
| v0: pop 100, order, kept 2, contest 3, cross 0.9, mutate 0.3 | +58.40 | −0.08 | +6.90 | +21.74 |

v1 against v0, maps won (sign test): Sprint 20–0 (p < 0.001), Marathon 16–4 (p = 0.012), Big Country 18–2 (p < 0.001). The runner-up is better in the Sprint (+3.93% vs +5.24%); v1 wins on the combined score through Marathon and Big Country.

## Screen (training maps 0–5)

Median score of the screened settings that share a value (count in brackets):

| Factor | Values |
|---|---|
| Kept unchanged | 0: +113.7 (9) · 1: +20.8 (7) · 2: +12.2 (6) · 5: +20.0 (6) · 10: +13.6 (4) |
| Contest size | 2: +121.1 (5) · 3: +20.9 (12) · 5: +13.4 (10) · 7: +73.8 (5) |
| Population | 30: +14.8 (10) · 50: +69.0 (6) · 100: +17.4 (6) · 200: +36.2 (4) · 400: +94.2 (6) |
| Crossover | order: +20.8 (19) · edge: +21.0 (13) |

Factors interact strongly; single-factor medians mix good and bad partners. The top three of the screen all use edge recombination; edge needs constant mutation to keep variety, order crossover does not.

## Findings

1. **Keeping the best is essential.** All 9 settings that kept nobody scored +18% or worse, 6 of them +97% or worse. Without it, good trips are lost again in the next generation and the search drifts.
2. **Small and harsh wins.** 30 members, parents picked from contests of 7, every child crossed and mutated: many generations, strong selection, no wasted measurements. A child that is a plain copy of its mother costs a measurement and teaches nothing; v1 never makes one.
3. **Budget sets the trade-off.** v0 (100 members) is as good as v1 in the Marathon (−0.08% vs −2.20%) and 11 times worse in the Sprint: a large population needs a long run.

## Notes for the team charters

- **Board size.** With 30 members, a board drawn from the members (≤10%) has at most 3.
- **Keeping the best is a global act.** Keeping "the 2 shortest" needs a ranking of everyone. Under Open Books that ranking is public, so a Member rule can read it as a public signal, but deciding *who* survives from it is a Board's or the Law's job.
- **A contest of 7 fits a Member's view.** A member meets at most 10 others per decision (rules §6), so the Law's contest of 7 can be run by a member itself.
- **Everyone meets everyone.** The Law's contact rule draws contest entrants from the whole population; any neighbourhood structure is a team's change.
- **Running time.** Edge recombination is slower than order crossover: Baseline v1 takes about 2.5 s (Sprint), 23 s (Marathon) and 39 s (Big Country) per map on one core. A full match now takes about 25 minutes on 4 cores.
