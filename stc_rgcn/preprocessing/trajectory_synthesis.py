"""Stage 3: constrained random-walk trajectory synthesis.

A walker starts at home in a grid cell and, hour by hour, picks its next
(cell, purpose) pair by Bayesian scoring under three groups of behavioural
constraints:

=======  ======================================================================
``PPC``  trip purpose constraints: per-purpose POI condition, minimum stay,
         valid time windows and per-period factors, plus the forbidden purpose
         transitions that make the day a coherent chain
``DCC``  destination choice: purpose to POI-category affinity, and a
         per-purpose distance decay
``TMC``  trip mode choice: mode to POI-category affinity, and a per-mode
         maximum distance and service time window
=======  ======================================================================

Each group is switched independently through :class:`ConstraintConfig`, whose
:meth:`~ConstraintConfig.ppc_only`, :meth:`~ConstraintConfig.ppc_dcc` and
:meth:`~ConstraintConfig.full` constructors build the three progressive
variants the paper compares.

The constraint tables under ``data/constraints/`` are keyed by the original
Chinese purpose and POI-category names; :data:`stc_rgcn.utils.schema.PURPOSE_LABELS`
maps them to the purpose codes used elsewhere.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd

from ..config import DEFAULT_CONSTRAINT_DIR, DEFAULT_INTERIM_DIR

#: Purpose a walker starts and ends its day in. Matches the ``hj`` purpose code.
HOME_PURPOSE = "回家"

#: POI category a cell must contain to emit trajectories, i.e. be residential.
RESIDENTIAL_POI = "商务住宅"

#: Transport modes assumed when no modal table is loaded.
FALLBACK_MODES = ("subway", "bus", "driving", "non_vehicle")

#: One synthesized trajectory per this many residents.
RESIDENTS_PER_TRAJECTORY = 100

#: Probability mass reserved for staying in the current cell.
STAY_PROBABILITY = 0.05

#: Earliest and latest hour (exclusive) a walker may leave home.
DEPARTURE_HOURS = (5, 11)

#: Hour from which the walker is forced to head home.
RETURN_HOME_HOUR = 23

#: Distance decay applied when DCC provides no per-purpose rate.
DEFAULT_DECAY_RATE = 0.1


def _constraint_path(name):
    return os.path.join(DEFAULT_CONSTRAINT_DIR, name)


@dataclass(frozen=True)
class ConstraintConfig:
    """Which constraint groups are active, and where their tables live."""

    ppc_generation_path: str = field(
        default_factory=lambda: _constraint_path("ppc_generation.csv"))
    ppc_transitions_path: str = field(
        default_factory=lambda: _constraint_path("ppc_transitions.csv"))
    dcc_poi_weights_path: str = field(
        default_factory=lambda: _constraint_path("dcc_poi_weights.csv"))
    dcc_distance_decay_path: str = field(
        default_factory=lambda: _constraint_path("dcc_distance_decay.csv"))
    tmc_poi_weights_path: str = field(
        default_factory=lambda: _constraint_path("tmc_poi_weights.csv"))
    tmc_mode_availability_path: str = field(
        default_factory=lambda: _constraint_path("tmc_mode_availability.csv")
    )

    enable_ppc: bool = True
    enable_dcc: bool = True
    enable_tmc: bool = True

    #: Purpose vocabulary. ``None`` reads it from the PPC generation table,
    #: which is also the vocabulary the other tables are keyed by.
    purposes: Optional[tuple[str, ...]] = None
    home_purpose: str = HOME_PURPOSE
    residential_poi: str = RESIDENTIAL_POI

    @classmethod
    def ppc_only(cls, **overrides):
        """Purpose priors only (the paper's ``GCN_fun-PPC``)."""
        return replace(cls(enable_dcc=False, enable_tmc=False), **overrides)

    @classmethod
    def ppc_dcc(cls, **overrides):
        """Purpose priors plus destination choice (``GCN_fun-PPC-DCC``)."""
        return replace(cls(enable_tmc=False), **overrides)

    @classmethod
    def full(cls, **overrides):
        """The full behavioural decision chain (``STC-RGCN w/o Rel``)."""
        return replace(cls(), **overrides)


# ── Constraint-table parsing ───────────────────────────────────────────────
def parse_time_windows(value):
    """Parse ``'8-10;12-14'`` into ``[(8, 10), (12, 14)]``."""
    if pd.isnull(value) or not str(value).strip():
        return []
    return [tuple(int(x) for x in window.split("-")) for window in str(value).split(";")]


def parse_time_factors(value):
    """Parse ``'8-10:1.2;12-14:1.5'`` into ``[((8, 10), 1.2), ((12, 14), 1.5)]``."""
    if pd.isnull(value) or not str(value).strip():
        return []
    factors = []
    for item in str(value).split(";"):
        window, factor = item.split(":")
        start, end = (int(x) for x in window.split("-"))
        factors.append(((start, end), float(factor)))
    return factors


def parse_poi_condition(value):
    """Compile a POI condition string into a ``grid -> bool`` predicate.

    Supported forms are ``exist <category>`` and ``<category><op><threshold>``
    for ``>``, ``<`` and ``=``.  An unparseable or empty condition becomes a
    predicate that always holds, so a malformed row relaxes rather than
    silently removes a purpose.
    """
    always_true = lambda grid: True  # noqa: E731 - kept inline for picklability
    if pd.isnull(value) or not str(value).strip():
        return always_true

    text = str(value).strip()
    try:
        if text.startswith("exist"):
            category = text.split(" ", 1)[1].strip()
            return lambda grid, c=category: grid["poi_density"].get(c, 0) > 0
        for operator, compare in (
            (">", lambda density, t: density > t),
            ("<", lambda density, t: density < t),
            ("=", lambda density, t: density == t),
        ):
            if operator in text:
                raw_category, raw_threshold = text.split(operator)
                category = raw_category.strip()
                threshold = float(raw_threshold)
                # Bound as defaults so each predicate captures its own values.
                return (
                    lambda grid, c=category, t=threshold, f=compare:
                    f(grid["poi_density"].get(c, 0), t)
                )
    except (IndexError, ValueError):
        print("warning: unparseable POI condition {!r}, treating it as always true".format(text))
    return always_true


@dataclass
class PurposeRule:
    """Generation constraints attached to one trip purpose."""

    min_stay: int = 1
    condition: callable = field(default_factory=lambda: (lambda grid: True))
    time_windows: list[tuple[int, int]] = field(default_factory=list)
    time_factors: list[tuple[tuple[int, int], float]] = field(default_factory=list)


class TrajectorySynthesizer:
    """Generate synthetic daily trajectories under a :class:`ConstraintConfig`.

    Parameters
    ----------
    grid_data_path
        CSV of grid cells with ``start_ID``, ``neighbors``, ``poi_density`` and
        ``pop`` columns.  Not shipped with the repository.
    config
        Which constraint groups apply; defaults to all three.
    """

    def __init__(self, grid_data_path, config=None, verbose=True):
        self.grid_data_path = grid_data_path
        self.config = config or ConstraintConfig()
        self.grids = self._load_grid_data(grid_data_path, verbose=verbose)

        self.purpose_rules = self._load_purpose_rules()
        self.forbidden_transitions = self._load_forbidden_transitions()
        self.purpose_poi_weights = self._load_weight_table(
            self.config.dcc_poi_weights_path, "purpose", self.config.enable_dcc
        )
        self.mode_poi_weights = self._load_weight_table(
            self.config.tmc_poi_weights_path, "method", self.config.enable_tmc
        )
        self.mode_availability = self._load_mode_availability()
        self.distance_decay = self._load_distance_decay()

        if self.config.home_purpose not in self.purpose_rules:
            raise ValueError(
                "home purpose {!r} is not in the purpose vocabulary {}".format(
                    self.config.home_purpose, sorted(self.purpose_rules)
                )
            )

    # ── Loading ────────────────────────────────────────────────────────────
    def _purpose_vocabulary(self):
        if self.config.purposes:
            return list(self.config.purposes)
        table = pd.read_csv(self.config.ppc_generation_path)
        return table["purpose"].tolist()

    def _load_purpose_rules(self):
        """Build one :class:`PurposeRule` per purpose.

        The vocabulary always comes from the generation table; its constraints
        are only applied when PPC is enabled.
        """
        rules = {purpose: PurposeRule() for purpose in self._purpose_vocabulary()}
        if not self.config.enable_ppc:
            return rules

        table = pd.read_csv(self.config.ppc_generation_path)
        for row in table.itertuples():
            rules[row.purpose] = PurposeRule(
                min_stay=int(row.min_stay),
                condition=parse_poi_condition(getattr(row, "poi_condition", "")),
                time_windows=parse_time_windows(getattr(row, "time_windows", "")),
                time_factors=parse_time_factors(getattr(row, "time_factors", "")),
            )
        return rules

    def _load_forbidden_transitions(self):
        if not self.config.enable_ppc:
            return {}
        table = pd.read_csv(self.config.ppc_transitions_path)
        return {
            row.current_purpose: [p.strip() for p in str(row.forbidden_purposes).split(",")]
            for row in table.itertuples()
        }

    @staticmethod
    def _load_weight_table(path, index_column, enabled):
        if not enabled:
            return {}
        return pd.read_csv(path, index_col=index_column).to_dict(orient="index")

    def _load_mode_availability(self):
        if not self.config.enable_tmc:
            return {}
        table = pd.read_csv(self.config.tmc_mode_availability_path)
        return {
            row.method: {
                "max_distance": float(row.max_distance),
                "time_windows": parse_time_windows(row.time_window),
            }
            for row in table.itertuples()
        }

    def _load_distance_decay(self):
        if not self.config.enable_dcc:
            return {}
        table = pd.read_csv(self.config.dcc_distance_decay_path)
        return table.set_index("purpose")["decay_rate"].to_dict()

    @staticmethod
    def _load_grid_data(path, verbose=True):
        """Load grid cells keyed by ``start_ID``.

        ``neighbors`` entries look like ``<id>(<mode>:<something>:<distance>)``
        separated by ``;``; ``poi_density`` entries look like
        ``<category>:<density>`` separated by ``;``.
        """
        frame = pd.read_csv(path)
        frame["start_ID"] = frame["start_ID"].astype(str)

        grids = {}
        for row in frame.itertuples():
            neighbors = []
            if pd.notnull(row.neighbors):
                for entry in str(row.neighbors).split(";"):
                    if "(" not in entry or ")" not in entry:
                        continue
                    grid_part, detail = entry.split("(", 1)
                    mode, _, distance = detail.rstrip(")").split(":")
                    neighbors.append(
                        {
                            "id": grid_part.strip(),
                            "mode": mode.strip(),
                            "distance": float(distance),
                        }
                    )

            poi_density = {}
            if pd.notnull(row.poi_density):
                for pair in str(row.poi_density).split(";"):
                    if ":" not in pair:
                        continue
                    category, density = pair.split(":")
                    poi_density[category.strip()] = float(density.strip())

            grids[row.start_ID] = {
                "id": row.start_ID,
                "neighbors": neighbors,
                "poi_density": poi_density,
                "population": int(row.pop),
            }

        if verbose:
            print("loaded {} grid cells from {}".format(len(grids), path))
        return grids

    # ── Constraint checks ──────────────────────────────────────────────────
    def _is_forbidden(self, current_purpose, next_purpose):
        if not self.config.enable_ppc:
            return False
        return next_purpose in self.forbidden_transitions.get(current_purpose, [])

    def _min_stay(self, purpose):
        if not self.config.enable_ppc:
            return 1
        return self.purpose_rules[purpose].min_stay

    def _allows_purpose(self, grid, purpose):
        if not self.config.enable_ppc:
            return True
        return self.purpose_rules[purpose].condition(grid)

    def _is_mode_available(self, mode, hour):
        if not self.config.enable_tmc:
            return True
        windows = self.mode_availability.get(mode, {}).get("time_windows", [])
        return any(start <= hour < end for start, end in windows)

    def _fits_time_window(self, purpose, hour, min_stay):
        """Whether an activity starting at ``hour`` fits one of its windows."""
        windows = self.purpose_rules[purpose].time_windows
        if not windows:
            return True  # no declared window means no restriction
        end_hour = hour + min_stay  # activities do not wrap past midnight
        return any(start <= hour and end_hour <= end for start, end in windows)

    # ── Generation ─────────────────────────────────────────────────────────
    def residential_grids(self):
        """Grid cells that can emit trajectories: populated and residential."""
        return [
            grid_id
            for grid_id, grid in self.grids.items()
            if grid["population"] > 0
            and grid["poi_density"].get(self.config.residential_poi)
        ]

    def synthesize_all(self, max_workers=None):
        """Synthesize trajectories for every eligible cell, in parallel.

        Returns
        -------
        pandas.DataFrame
            One row per trajectory, with ``grid_id``, ``trajectory_id``,
            ``population``, ``trajectory`` and ``num_points``.
        """
        grid_ids = self.residential_grids()
        workers = max_workers or multiprocessing.cpu_count()

        records = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_synthesize_for_grid, grid_id, self.grid_data_path, self.config)
                for grid_id in grid_ids
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    records.extend(future.result())
                except Exception as exc:  # one bad cell must not kill the run
                    print("error: a grid cell failed: {}".format(exc))

        return pd.DataFrame(records)

    def synthesize_for_grid(self, grid_id):
        """Synthesize every trajectory for one grid cell."""
        grid = self.grids.get(grid_id)
        if not grid or grid["population"] <= 0:
            return []
        if not grid["poi_density"].get(self.config.residential_poi):
            return []

        count = max(1, grid["population"] // RESIDENTS_PER_TRAJECTORY)
        records = []
        for index in range(1, count + 1):
            points = self._synthesize_one(grid_id)
            records.append(
                {
                    "grid_id": grid_id,
                    "trajectory_id": "{}_{}".format(grid_id, index),
                    "population": grid["population"],
                    "trajectory": " -> ".join(points),
                    "num_points": len(points),
                }
            )
        return records

    def _synthesize_one(self, start_grid):
        """Walk one 24-hour day starting from ``start_grid``."""
        departure_hour = np.random.randint(*DEPARTURE_HOURS)
        current_grid = start_grid
        current_purpose = self.config.home_purpose
        stay_counter = self._min_stay(current_purpose)
        heading_home = False

        trajectory = ["{}({} {:02d}:00)".format(start_grid, current_purpose, departure_hour)]

        for offset in range(24 - departure_hour):
            hour = (departure_hour + offset) % 24 + 1

            if hour >= RETURN_HOME_HOUR and not heading_home:
                heading_home = True
                current_purpose = self.config.home_purpose
                stay_counter = 0

            stay_counter += 1
            if stay_counter < self._min_stay(current_purpose):
                trajectory.append("{}({} {:02d}:00)".format(current_grid, current_purpose, hour))
                continue

            next_grid, next_purpose = self._sample_next_action(current_grid, current_purpose, hour)
            if next_purpose != current_purpose:
                current_purpose = next_purpose
                stay_counter = 0
            current_grid = next_grid
            trajectory.append("{}({} {:02d}:00)".format(current_grid, current_purpose, hour))

        return self._merge_consecutive_stays(trajectory)

    @staticmethod
    def _merge_consecutive_stays(trajectory):
        """Collapse consecutive points that share a cell and a purpose."""
        merged, current = [], None

        def render(entry):
            if entry["start"] == entry["end"]:
                return "{}({} {})".format(entry["grid"], entry["purpose"], entry["start"])
            return "{}({} {}-{})".format(
                entry["grid"], entry["purpose"], entry["start"], entry["end"]
            )

        for point in trajectory:
            grid, detail = point.split("(", 1)
            purpose, start = detail.rstrip(")").split(" ")
            if current and current["grid"] == grid and current["purpose"] == purpose:
                current["end"] = start
                continue
            if current:
                merged.append(render(current))
            current = {"grid": grid, "purpose": purpose, "start": start, "end": start}

        if current:
            merged.append(render(current))
        return merged

    def _sample_next_action(self, current_grid, current_purpose, hour):
        """Pick the next ``(grid, purpose)`` by joint Bayesian scoring.

        ``Pr(A | B1, B2) ~ Pr(B2 | B1, A) * Pr(B1 | A) * Pr(A)`` where ``A`` is
        the (destination, purpose) action, ``B1`` its POI evidence and ``B2``
        the transport-mode evidence.
        """
        actions, probabilities = [], []

        stay_allowed = True
        if self.config.enable_ppc:
            stay_allowed = self._allows_purpose(
                self.grids[current_grid], current_purpose
            ) and self._fits_time_window(current_purpose, hour, self._min_stay(current_purpose))
        if stay_allowed:
            actions.append((current_grid, current_purpose))
            probabilities.append(STAY_PROBABILITY)

        scored = self._score_neighbor_actions(current_grid, current_purpose, hour)
        if scored:
            total_mass = 1.0 - STAY_PROBABILITY if stay_allowed else 1.0
            total_score = sum(scored.values())
            if total_score > 0:
                for action, score in scored.items():
                    actions.append(action)
                    probabilities.append(score / total_score * total_mass)

        if not actions:
            return current_grid, current_purpose

        total = sum(probabilities)
        if total <= 0:
            return current_grid, current_purpose
        probabilities = [p / total for p in probabilities]
        return actions[np.random.choice(len(actions), p=probabilities)]

    def _score_neighbor_actions(self, current_grid, current_purpose, hour):
        """Joint score of every reachable ``(neighbour, purpose)`` action."""
        terms: dict[tuple[str, str], dict[str, float]] = {}

        for neighbor in self.grids[current_grid]["neighbors"]:
            mode = neighbor["mode"]
            distance = neighbor["distance"]

            if self.config.enable_tmc:
                if not self._is_mode_available(mode, hour):
                    continue
                limit = self.mode_availability.get(mode, {}).get("max_distance")
                if limit is not None and distance > limit:
                    continue

            neighbor_grid = self.grids.get(neighbor["id"])
            if not neighbor_grid:
                continue

            for purpose in self._candidate_purposes(neighbor_grid, current_purpose, hour):
                key = (neighbor["id"], purpose)
                if key not in terms:
                    terms[key] = {
                        "prior": self._prior(purpose, distance, hour),
                        "poi_likelihood": self._poi_likelihood(neighbor_grid, purpose),
                        "mode_numerator": 0.0,
                    }
                terms[key]["mode_numerator"] += self._mode_numerator(
                    neighbor_grid, purpose, mode
                )

        scores = {}
        for (grid_id, purpose), term in terms.items():
            denominator = self._mode_denominator(self.grids[grid_id], purpose)
            mode_likelihood = term["mode_numerator"] / denominator if denominator > 0 else 0.0
            scores[(grid_id, purpose)] = (
                mode_likelihood * term["poi_likelihood"] * term["prior"]
            )
        return scores

    def _candidate_purposes(self, neighbor_grid, current_purpose, hour):
        for purpose in self.purpose_rules:
            if self.config.enable_ppc:
                if self._is_forbidden(current_purpose, purpose):
                    continue
                if not self._allows_purpose(neighbor_grid, purpose):
                    continue
                if not self._fits_time_window(purpose, hour, self._min_stay(purpose)):
                    continue
            yield purpose

    def _prior(self, purpose, distance, hour):
        """``Pr(A)``: time-of-day preference times distance decay."""
        factor = 1.0
        if self.config.enable_ppc:
            for (start, end), value in self.purpose_rules[purpose].time_factors:
                if start <= hour < end:
                    factor *= value
        decay = 0.0
        if self.config.enable_dcc:
            decay = self.distance_decay.get(purpose, DEFAULT_DECAY_RATE)
        return factor / (1.0 + decay * distance**2)

    def _poi_likelihood(self, grid, purpose):
        """``Pr(B1 | A)``: POI density weighted by purpose affinity."""
        weights = self.purpose_poi_weights.get(purpose, {}) if self.config.enable_dcc else {}
        return sum(
            weights.get(category, 1.0) * density
            for category, density in grid["poi_density"].items()
        )

    def _mode_numerator(self, grid, purpose, mode):
        purpose_weights = {}
        if self.config.enable_dcc:
            purpose_weights = self.purpose_poi_weights.get(purpose, {})
        mode_weights = {}
        if self.config.enable_tmc:
            mode_weights = self.mode_poi_weights.get(mode, {})
        return sum(
            purpose_weights.get(category, 1.0) * mode_weights.get(category, 1.0) * density
            for category, density in grid["poi_density"].items()
        )

    def _mode_denominator(self, grid, purpose):
        """``Pr(B2 | B1, A)`` denominator: the same sum over every mode."""
        modes = list(self.mode_poi_weights) if self.mode_poi_weights else list(FALLBACK_MODES)
        return sum(self._mode_numerator(grid, purpose, mode) for mode in modes)


def _synthesize_for_grid(grid_id, grid_data_path, config):
    """Worker entry point: rebuild the synthesizer in the child process.

    The synthesizer holds compiled lambdas, which do not pickle, so each worker
    reconstructs it from the path and config instead of receiving an instance.
    """
    synthesizer = TrajectorySynthesizer(grid_data_path, config, verbose=False)
    return synthesizer.synthesize_for_grid(grid_id)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="stc-rgcn-synthesize",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--grid-data", required=True,
                        help="Grid CSV with start_ID, neighbors, poi_density and pop columns")
    parser.add_argument("--output", default=os.path.join(DEFAULT_INTERIM_DIR, "trajectories.csv"),
                        help="Output trajectory CSV")
    parser.add_argument("--constraints", choices=["ppc", "ppc_dcc", "full"], default="full",
                        help="Which constraint groups to enable")
    parser.add_argument("--constraint-dir", default=DEFAULT_CONSTRAINT_DIR,
                        help="Directory holding the six constraint tables")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Worker processes (default: one per CPU)")
    parser.add_argument("--seed", type=int, default=None, help="Numpy random seed")
    return parser


def _config_from_args(args):
    paths = {
        "ppc_generation_path": os.path.join(args.constraint_dir, "ppc_generation.csv"),
        "ppc_transitions_path": os.path.join(args.constraint_dir, "ppc_transitions.csv"),
        "dcc_poi_weights_path": os.path.join(args.constraint_dir, "dcc_poi_weights.csv"),
        "dcc_distance_decay_path": os.path.join(args.constraint_dir, "dcc_distance_decay.csv"),
        "tmc_poi_weights_path": os.path.join(args.constraint_dir, "tmc_poi_weights.csv"),
        "tmc_mode_availability_path": os.path.join(
            args.constraint_dir, "tmc_mode_availability.csv"
        ),
    }
    factory = {
        "ppc": ConstraintConfig.ppc_only,
        "ppc_dcc": ConstraintConfig.ppc_dcc,
        "full": ConstraintConfig.full,
    }[args.constraints]
    return factory(**paths)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.seed is not None:
        np.random.seed(args.seed)

    synthesizer = TrajectorySynthesizer(args.grid_data, _config_from_args(args))
    trajectories = synthesizer.synthesize_all(max_workers=args.max_workers)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    trajectories.to_csv(args.output, index=False)
    print("wrote {} trajectories to {}".format(len(trajectories), args.output))


if __name__ == "__main__":
    main()
