"""Symmetric Euclidean travelling salesman problem.

A genome is a tour: a permutation of city indices (``np.ndarray`` of ints).
Its objective is the closed tour length.

Instances come from two sources:

* TSPLIB (Reinelt 1991), the standard benchmark library with proven optima,
  so results can be reported as a gap to the optimum.
* Seeded uniform-random instances, for scaling studies (optimum unknown).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

#: Proven optimal tour lengths for the bundled TSPLIB instances.
TSPLIB_OPTIMA: dict[str, float] = {
    "eil51": 426,
    "berlin52": 7542,
    "st70": 675,
    "eil76": 538,
    "kroA100": 21282,
    "ch150": 6528,
    "a280": 2579,
}


@dataclass(frozen=True, eq=False)
class TSP:
    name: str
    coords: np.ndarray
    optimum: float | None = None
    #: TSPLIB EUC_2D rounds each edge to the nearest integer; random instances don't.
    round_distances: bool = False
    #: A known optimal tour, or None if unknown.
    optimal_genome: np.ndarray | None = field(default=None, repr=False)
    dist: np.ndarray = field(init=False, repr=False)
    #: Expected length of a uniformly random tour: n times the mean distance
    #: between two different cities.
    random_value: float = field(init=False)

    def __post_init__(self) -> None:
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        d = np.sqrt((diff**2).sum(axis=-1))
        if self.round_distances:
            d = np.floor(d + 0.5)
        object.__setattr__(self, "dist", d)
        n = len(self.coords)
        object.__setattr__(self, "random_value", float(d.sum() / (n - 1)))

    @property
    def n(self) -> int:
        return len(self.coords)

    # --- Problem protocol -------------------------------------------------

    def evaluate(self, tour: np.ndarray) -> float:
        return float(self.dist[tour, np.roll(tour, -1)].sum())

    def random_genome(self, rng: np.random.Generator) -> np.ndarray:
        return rng.permutation(self.n)

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Fraction of edges of ``a`` that ``b`` does not share (0 = same tour)."""
        succ, pred = np.empty_like(b), np.empty_like(b)
        succ[b] = np.roll(b, -1)
        pred[b] = np.roll(b, 1)
        nxt = np.roll(a, -1)
        shared = np.count_nonzero((succ[a] == nxt) | (pred[a] == nxt))
        return 1.0 - shared / self.n

    def canonical(self, tour: np.ndarray) -> bytes:
        # A tour is its undirected edge set: invariant to start city and direction.
        return self.edge_keys(tour).tobytes()

    # --- helpers ----------------------------------------------------------

    def edge_keys(self, tour: np.ndarray) -> np.ndarray:
        """Sorted integer keys of the tour's undirected edges."""
        a, b = tour, np.roll(tour, -1)
        return np.sort(np.minimum(a, b) * self.n + np.maximum(a, b))

    # --- constructors -----------------------------------------------------

    @classmethod
    def uniform(cls, n: int, seed: int, scale: float = 1000.0) -> TSP:
        rng = np.random.default_rng(seed)
        return cls(name=f"uniform{n}-s{seed}", coords=rng.uniform(0, scale, size=(n, 2)))

    @classmethod
    def from_tsplib(cls, path: str | Path, optimum: float | None = None) -> TSP:
        """Load a ``.tsp`` file, and the optimal tour from ``<name>.opt.tour``
        next to it if there is one."""
        path = Path(path)
        header, coords = _parse_tsplib(path.read_text())
        kind = header.get("EDGE_WEIGHT_TYPE")
        if kind != "EUC_2D":
            raise ValueError(f"{path}: only EUC_2D instances are supported, got {kind}")
        name = header.get("NAME", Path(path).stem)
        if optimum is None:
            optimum = TSPLIB_OPTIMA.get(name)
        tour_file = path.with_name(f"{path.stem}.opt.tour")
        optimal = _parse_tour(tour_file.read_text()) if tour_file.exists() else None
        return cls(
            name=name,
            coords=coords,
            optimum=optimum,
            round_distances=True,
            optimal_genome=optimal,
        )

    @classmethod
    def tsplib(cls, name: str) -> TSP:
        """Load one of the bundled TSPLIB instances by name, e.g. ``berlin52``."""
        folder = resources.files("evosoc.problems") / "data" / "tsplib"
        with resources.as_file(folder) as f:
            return cls.from_tsplib(f / f"{name}.tsp")


def _parse_tsplib(text: str) -> tuple[dict[str, str], np.ndarray]:
    header: dict[str, str] = {}
    coords: list[tuple[float, float]] = []
    in_coords = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "EOF":
            continue
        if line == "NODE_COORD_SECTION":
            in_coords = True
        elif in_coords:
            _, x, y = line.split()[:3]
            coords.append((float(x), float(y)))
        elif ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    return header, np.array(coords)


def _parse_tour(text: str) -> np.ndarray:
    """A TSPLIB ``.tour`` file (1-based city numbers ending in -1) as a 0-based tour."""
    numbers = text.split("TOUR_SECTION")[1].split()
    return np.array([int(x) - 1 for x in numbers[: numbers.index("-1")]])
