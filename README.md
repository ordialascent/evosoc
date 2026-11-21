# evosoc

A genetic algorithm experiment comparing social structures. Does a population evolve better when power and resources are concentrated (rulers decide, output is pooled) or when they are spread out (each member decides and keeps what it earns)? When does which work better, and why?

Every society solves the same problem under the same budget. Only the social rules change.

## The problem: the travelling salesman

50 cities are scattered on a map. A solution is a round trip that visits every city once. Shorter is better. Anyone can check a solution at a glance, and the problem is hard enough that a population has to improve step by step.

## Step 1: the neutral baseline

Before comparing societies, we need a reference point that has no social structure at all: a standard, textbook genetic algorithm.

- **Population:** 100 round trips, each starting as a random order of cities.
- **Each generation:**
  1. The 2 shortest trips are kept unchanged.
  2. Every other place goes to a child. Each parent is chosen by a small contest: pick 3 members at random, the shortest trip wins.
  3. The child takes a stretch of one parent's trip and fills in the remaining cities in the order the other parent visits them (order crossover).
  4. Sometimes (30% of children), a random stretch of the child's trip is reversed (mutation).
- **Budget:** 1,000 generations, about 98,000 trips measured per run. That is long enough for runs to settle, so the result shows where a population gets stuck, not only how fast it starts.

Nobody inside the population decides anything, and nothing is owned: the rules are fixed before the run, and a member is rewarded only for the length of its current trip.

![Baseline results](results/baseline.png)

**How to read it.** Left to right: a random trip, the best trip the population evolved in a typical run, and a yardstick found without evolution (a greedy trip, then repeatedly untangled where it crosses itself). On the right, the solid line is the best trip in each generation for the typical run; the shaded band shows the best and worst of 30 independent runs. The dashed line is the average member. The percentages mark how much longer the typical best trip is than the yardstick after 100, 300 and 1,000 generations.

**What it shows.** Evolution turns a tangle into a sensible route. It is far off early (+64% at generation 100), close by generation 300 (+7%) and level with the yardstick at the end (+1%). Single runs end anywhere from 6% shorter to 7% longer than the yardstick: luck matters, which is why every comparison uses 30 seeds. Societies will be compared at all three points, since one may start fast and stall while another starts slow and keeps going.

## Experiment 1: a ruling class that always survives

The first step away from the baseline turns one dial only: **who decides**. A ruling class of the *k* fittest members is guaranteed to survive into the next generation, unchanged. Rulers are chosen by merit (the shortest trips), and what they gain is survival. Everyone else is replaced by children, exactly as in the baseline. The baseline itself is the case of 2 rulers.

- **Compared:** 1, 2 (baseline), 5, 20 and 50 rulers in a population of 100.
- **Fair budget:** every society may measure the same number of trips (98,100). Rulers are not measured again, so a society with more rulers makes fewer children per generation and gets more generations (50 rulers: 1,960 generations of 50 children).
- **Same map, same 30 seeds** as the baseline.
- **Variety:** how different the average member's route is from the best member's route (the share of its city-to-city links that the best trip does not use). 0 means everyone copies the leader.

![Experiment 1 results](results/rulers.png)

**How to read it.** Left: how far the best trip is from the yardstick as the budget is used up (median of 30 runs). Middle: how varied the population stays. Right: where each of the 30 runs ends; the bar is the median.

| Rulers | After 9,900 trips | After 29,500 trips | At the end (98,100) |
|---|---|---|---|
| 1 | +81% | +13% | +0.9% |
| 2 (baseline) | +64% | +7% | +0.6% |
| 5 | +44% | +3% | +0.3% |
| 20 | +36% | +3% | +1.1% |
| 50 | +27% | +4% | +3.0% |

*Median gap to the yardstick; the full numbers, including the spread, are in `results/rulers.json`.*

**What it shows.**

- **A larger ruling class gets results faster.** Keeping the best members safe means good routes are never lost and get copied again and again. With 50 rulers the population is more than twice as close to the yardstick early on as the baseline.
- **It also makes everyone alike sooner.** The more rulers, the faster variety collapses (middle chart). Once everyone travels nearly the same route, there is little left to recombine.
- **In the long run, too many rulers hurts.** With 50 rulers, runs stop improving around 30,000 trips and end clearly worse than the baseline (+3.0% vs +0.6%; a shuffling test says a gap this large would arise by chance about 2% of the time). Between 1 and 20 rulers the end results differ by less than the luck between runs.
- **So the answer depends on the time available.** A big ruling class is the better choice when the budget is short, and the worse one when there is time for the population to get stuck. This is tested directly in a later experiment that varies the budget and the difficulty of the map.

## Run it yourself

```
pip install -e ".[test]"
python -m evosoc baseline          # writes results/baseline.{png,json,csv}
python -m evosoc rulers            # writes results/rulers.{png,json,csv}
pytest
```

Every run is seeded: the same options give exactly the same numbers. `--map-seed` picks the map, `--seeds` sets how many independent runs; see `python -m evosoc baseline --help` and `python -m evosoc rulers --help`.
