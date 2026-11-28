# How big is a nation? The scaling probe

Before Round 1 the arena grows from a tribe of 30 members to a nation. This probe picks the size. It runs Baseline v1's rules (keep the 2 best, contests of 7, every child crossed and mutated) with 30 to 10,000 members.

The budget is counted in **generations**: members × generations measurements. A bigger society gets proportionally more measurements, so it is not handicapped by its own slowness.

Rerun: `python -m evosoc scale` (code `evosoc/scaling.py`, record `data/tuning/scaling.json`, resumable from `data/runs/scaling/`, about 55 min on 4 cores). It uses training maps 500,000+ only, 2 maps per cell, so single cells are noisy and only the trends count.

## Gap to the yardstick at the end, %

Lower is better.

| Cities | Generations | 30 | 100 | 300 | **1,000** | 3,000 | 10,000 |
|---|---|---|---|---|---|---|---|
| 50 | 100 | +38.9 | +19.1 | +3.3 | **+0.5** | +8.9 | +13.7 |
| 50 | 1,000 | −3.8 | +1.7 | −3.2 | **−5.0** | −3.1 | −4.9 |
| 100 | 100 | +126.6 | +85.6 | +59.2 | **+56.5** | +53.0 | +47.7 |
| 100 | 1,000 | +12.1 | +4.9 | +1.6 | **−4.4** | +3.2 | +8.8 |
| 200 | 100 | +329.9 | +261.7 | +239.8 | **+178.5** | +143.2 | +122.6 |
| 200 | 1,000 | +66.4 | +47.9 | +28.3 | **+25.3** | +113.2 | +100.6 |

## How much the society follows its leader at the end

The share of the best trip's links that the average member uses. 1.0 means everybody copies the leader.

| Cities | Generations | 30 | 100 | 300 | **1,000** | 3,000 | 10,000 |
|---|---|---|---|---|---|---|---|
| 50 | 100 | 0.90 | 0.83 | 0.76 | **0.69** | 0.50 | 0.48 |
| 50 | 1,000 | 0.92 | 0.89 | 0.86 | **0.81** | 0.74 | 0.74 |
| 100 | 100 | 0.88 | 0.81 | 0.77 | **0.39** | 0.32 | 0.29 |
| 100 | 1,000 | 0.97 | 0.93 | 0.93 | **0.88** | 0.58 | 0.50 |
| 200 | 100 | 0.95 | 0.88 | 0.58 | **0.32** | 0.22 | 0.17 |
| 200 | 1,000 | 0.97 | 0.96 | 0.93 | **0.89** | 0.31 | 0.23 |

At every size, practically every member is a distinct trip, and no child repeats a trip that is still in the society.

## Findings

1. **More members help, up to a point.** With the same number of generations, a society of 1,000 beats one of 30 everywhere: on 100 cities over 1,000 generations, −4.4% against +12.1%. More members carry more variety into each generation.
2. **Past about 1,000, a society is too slow to make up its mind.** At 3,000 and 10,000 members the society has not converged after 1,000 generations: members share only 23–58% of the leader's links, against 88–89% at 1,000. The result gets worse again (200 cities: +25% at 1,000 members, +113% at 3,000). The picture does change at that size, but only because the run is too short for the society. A fair test of 10,000 members would need several times more generations, which is days per match.
3. **50 cities are too easy for a nation.** A society of 1,000 ends at −5% of the yardstick, close to the best possible trip, so matches would end on ties. A nation needs 100 cities or more.

## Decision

**1,000 members.** It is the largest society that still converges within a run. A member's view of 10 is 1% of everyone, and a Board may have up to 100 members: a real gap between local and central.

The events keep their meaning, with budgets in generations of 1,000 members:

| Event | Cities | Generations | Budget | Tests |
|---|---|---|---|---|
| Sprint | 100 | 100 | 100,000 | who gets good fast (still about +55% from the yardstick) |
| Marathon | 100 | 1,000 | 1,000,000 | who gets furthest with time (near the yardstick) |
| Big Country | 200 | 1,000 | 1,000,000 | a problem bigger than the time available (about +25%) |
