# Rules of the game

Season 1: **Open Books**. Version 1.1.

Two teams of genetic algorithms solve the same travelling-salesman problems under the same budget. Team Centralizers: a board decides for everyone. Team Decentralizers: each member decides for itself. A match is three events; whoever wins more events wins the match. The loser adapts its contestant and challenges again.

## 1. The world

| Rule | Value |
|---|---|
| Problem | Round trip through all cities, shortest wins. Cities uniform at random in the unit square. |
| Map access | The map is hidden. Nobody can read city positions or the distance between two cities. The only way to learn anything about a map is to **measure a whole round trip**. |
| Measurement | Returns the exact length of one round trip. Costs **1 unit** of budget, whoever asks and for whatever reason (newborns, re-checks, trial moves, scouting). |
| Budget | Fixed per event (§3). The run stops when it is spent. |
| Result of a run | The shortest trip the society has *measured* by the time the budget is spent. |
| Population | Fixed at the Baseline v1 size: **30 members**. Both teams have the same number of members at all times. |
| Toolbox | The ways to make and change a trip (crossover and mutation operators) are those of Baseline v1: **edge recombination** (a child built from links its parents use) and **stretch reversal** (reverse one stretch of the trip). Contestants decide who uses them, when and how often; they may not add new ones. |
| Randomness | All randomness comes from the run's seed. Same seed, same result. |

## 2. Information: Open Books (season 1)

Every measured trip length is public. Any actor may read any measurement that has been made, at no cost.

Season 1 is the world in which central planning has its best case: the board can see everything that anyone knows. The question it asks: **with equal and perfect information, does central coordination beat spontaneous order?** A later season removes this (see §8).

## 3. Events

| Event | Cities | Budget (measurements) | Tests |
|---|---|---|---|
| Sprint | 50 | 10,000 | who gets good fast |
| Marathon | 50 | 100,000 | who gets furthest with time |
| Big Country | 100 | 100,000 | who copes with a harder problem than they have time for |

Each event is played on **30 maps**. Every map is new (generated from its seed), and both contestants play the same 30 maps with the same seeds.

## 4. Scoring

1. **Map:** the contestant with the shorter result wins the map. Equal results (relative difference below 1e-9) are a tied map.
2. **Event:** tied maps are left out. A contestant wins the event if it wins so many of the remaining maps that a fair coin would do this rarely: two-sided exact sign test, p < 0.05 (for 30 maps without ties: 21 or more). Otherwise the event is a draw.
3. **Match:** the contestant with more event wins wins the match. Equal event wins is a draw. Score format: events won, e.g. `2–1`, `1–0` (two draws).
4. **Reported for every event:** maps won by each side, median gap of each side to the yardstick (greedy trip improved by untangling crossings, computed by the referee, invisible to contestants), and the same for Baseline v1.
5. **Baseline v1** runs alongside in every match as a reference. It is not scored.

## 5. Seeds

| Use | Seeds |
|---|---|
| Teams: development and training | 0 – 999,999 |
| Match *N*, event *e* (0 Sprint, 1 Marathon, 2 Big Country), map *i* (0–29) | 1,000,000 + 1,000·*N* + 100·*e* + *i* |

Match seeds are new for every match, so no contestant can have been tuned on them. Teams never run on seeds ≥ 1,000,000.

## 6. Team charters

Every contestant declares, for each of six **matters**, who decides it:

| Matter | Question |
|---|---|
| Survival | Who stays into the next generation? |
| Reproduction | Who has children, and how many? |
| Pairing | Who has children with whom? |
| Variation | How much is changed when a child is made (crossover and mutation use)? |
| Resources | Who gets budget or wealth, e.g. to measure extra trials? |
| Contact | Who meets whom: groups, neighbourhoods, migration. |

Possible deciders:

- **Law:** the fixed rule of Baseline v1 for that matter. Neutral; belongs to no team. Nobody may change a Law, only replace it with a decider.
- **Board:** an outside planner, or a group of members (at most 10% of the population), deciding for everyone. It may use all public information. Members must follow its decisions.
- **Member:** each member decides for itself only. It may use what it knows about itself, about at most 10 other members it meets in that decision, and public signals that nobody controls (for example a price that emerges from everyone's actions; the mechanism must be declared and must not itself decide anything).

A **legal centralizer** gives at least one matter to the Board and no matter to Member.
A **legal decentralizer** gives at least one matter to Member and no matter to the Board.

Each new version changes **one idea** relative to its parent version, describable in one sentence, and states it in its declaration.

## 7. Fair play, disqualification and protest

The referee checks before every match:

1. The declaration (six matters, deciders, the one idea) matches the code.
2. The charter of §6 holds.
3. Budget: the run never measures more than the event budget, and every result is a measured trip.
4. The hidden map is never read except through measurement.
5. Determinism: two runs with the same seed give the same result.
6. Running time: at most 5 times Baseline v1 on the same event.

A contestant that fails a check **forfeits** the match (counts as a loss; the record says why) and must be fixed in its team's next version.

**Protest.** The loser of a match may file **one protest** against that match: that the winner cheated or is not true to its team's principle. **The House** (the show's impartial referee, reviewing afresh, not the one who refereed the match) decides, and its decision is final. If the protest is upheld, the result is overturned (the winner forfeits) and the record says why. When a decision would apply to other cases, the House writes it into these rules as a general rule, so the same question is not argued twice.

The House also decides any question these rules do not answer.

## 8. Seasons

A season is a fixed world (§1–§3). Season 1 ends after 6 matches, or earlier if Chris decides. Then the referee proposes the next season's world, and Chris approves it before it applies. Contestants carry over.

Planned next: **Season 2, Fog.** Knowledge becomes local: a measurement is known only to whoever made it or saw it made, so a board must pay to verify what it decides on, while members know their own trips. This is the knowledge problem the prequel showed matters most (a board deciding on a coarse view lost badly). Details are settled when season 1 ends.

## Glossary

- **Gap to the yardstick:** how much longer a trip is than the referee's yardstick trip for that map, in percent. Negative is better than the yardstick.
- **Sign test:** counts maps won by each side; asks how likely such a split would be if both were equally good.
