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

## Run it yourself

```
pip install -e ".[test]"
python -m evosoc baseline          # writes results/baseline.{png,json,csv}
pytest
```

Every run is seeded: the same options give exactly the same numbers. `--map-seed` picks the map, `--seeds` sets how many independent runs; see `python -m evosoc baseline --help`.
