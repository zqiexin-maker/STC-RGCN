"""Derived quantities: zone masses, OD features, distances and spatial interaction.

Two groups:

* **Masses** (:class:`NodeMass`, :class:`PurposeMass`, :func:`build_purpose_masses`)
  -- how much flow each zone emits and attracts, the descriptor the classical
  baselines use. One vectorized pass produces all 15 purposes at once, since
  fitting one model per purpose needs them repeatedly.
* **Distances** (:func:`resolve_distance_matrix`, :func:`intervening_opportunities`)
  -- OD distance in km, recovered from ``grid_distance.csv`` or from the
  coordinates embedded in the entity names, plus the radiation model's
  intervening-opportunities term.

The four near-identical copies of the mass logic that used to live in the
baseline files differed only in column order, which made them easy to mix up;
named arrays replace them.
"""

import re
from typing import NamedTuple

import numpy as np

from ...utils.schema import NUM_PURPOSES

# Entity names carry their grid coordinates, e.g. ``HYID8133500|17797000``.
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# Coordinates above this magnitude are projected metres rather than degrees.
_DEGREE_UPPER_BOUND = 1000.0
METERS_PER_KM = 1000.0


# ───────────────────────────────────────────────────────────────────────────
# Zone masses
# ───────────────────────────────────────────────────────────────────────────

class NodeMass(NamedTuple):
    """Per-zone flow masses, indexed by entity id.

    Attributes
    ----------
    out_flow : (E,) float64
        Total flow emitted by each zone.
    in_flow : (E,) float64
        Total flow attracted by each zone.
    mass : (E,) float64
        ``out_flow + in_flow``, the gravity-model mass term.
    """

    out_flow: np.ndarray
    in_flow: np.ndarray
    mass: np.ndarray


class PurposeMass(NamedTuple):
    """Per-zone masses broken down by trip purpose.

    Attributes
    ----------
    out_flow, in_flow : (E, K) float64
        ``out_flow[i, k]`` is the flow emitted by zone *i* for purpose *k*.
    """

    out_flow: np.ndarray
    in_flow: np.ndarray

    def for_purpose(self, purpose_idx):
        """:class:`NodeMass` restricted to one purpose."""
        out_flow = self.out_flow[:, purpose_idx]
        in_flow = self.in_flow[:, purpose_idx]
        return NodeMass(out_flow, in_flow, out_flow + in_flow)

    def total(self):
        """:class:`NodeMass` over the total flow (all purposes summed)."""
        out_flow = self.out_flow.sum(axis=1)
        in_flow = self.in_flow.sum(axis=1)
        return NodeMass(out_flow, in_flow, out_flow + in_flow)


def build_purpose_masses(od_ids, purpose_flows, num_entities, floor=0.0):
    """Vectorized per-purpose masses from one pass over the OD records.

    Fitting one model per purpose means the mass statistics are needed 15 times;
    accumulating them in a single ``(E, K)`` buffer avoids rereading the split
    files once per purpose.

    Parameters
    ----------
    od_ids : (N, 2) int
    purpose_flows : (N, K) float
    num_entities : int
    floor : float
        Lower bound applied to both flows. Defaults to ``0.0`` (raw totals, as
        the published baselines use); log-space models that must handle empty
        zones can pass ``1.0`` instead.
    """
    od_ids = np.asarray(od_ids, dtype=np.int64)
    flows = np.asarray(purpose_flows, dtype=np.float64).reshape(-1, NUM_PURPOSES)
    if len(od_ids) != len(flows):
        raise ValueError("{} OD pairs but {} flow rows".format(len(od_ids), len(flows)))

    out_flow = np.zeros((num_entities, NUM_PURPOSES), dtype=np.float64)
    in_flow = np.zeros((num_entities, NUM_PURPOSES), dtype=np.float64)
    np.add.at(out_flow, od_ids[:, 0], flows)
    np.add.at(in_flow, od_ids[:, 1], flows)

    if floor:
        out_flow = np.maximum(out_flow, floor)
        in_flow = np.maximum(in_flow, floor)
    return PurposeMass(out_flow, in_flow)


def od_mass_features(od_ids, mass, distances=None):
    """Feature matrix of an OD pair set.

    Columns: ``[origin_out, origin_in, dest_out, dest_in]`` plus the distance
    when ``distances`` is given. This is the input used by the tree ensembles.
    """
    od_ids = np.asarray(od_ids, dtype=np.int64)
    origin, dest = od_ids[:, 0], od_ids[:, 1]
    columns = [mass.out_flow[origin], mass.in_flow[origin],
               mass.out_flow[dest], mass.in_flow[dest]]
    if distances is not None:
        columns.append(np.asarray(distances, dtype=np.float64).reshape(-1))
    return np.column_stack(columns)


def flows_to_probs(purpose_flows):
    """Row-normalize ``(N, K)`` purpose flows into probability distributions.

    Rows that sum to zero (the model predicted no flow at all) fall back to the
    uniform distribution, which is the neutral choice for a distribution metric.
    """
    flows = np.asarray(purpose_flows, dtype=np.float64).reshape(-1, NUM_PURPOSES)
    row_sum = flows.sum(axis=1, keepdims=True)
    return np.where(row_sum > 0, flows / np.where(row_sum > 0, row_sum, 1.0),
                    1.0 / NUM_PURPOSES)


# ───────────────────────────────────────────────────────────────────────────
# Distances
# ───────────────────────────────────────────────────────────────────────────

def parse_entity_coordinates(id2entity):
    """Extract ``(E, 2)`` coordinates from the entity names.

    Entity strings look like ``HYID8133500|17797000``: a grid label embedding
    two numeric coordinates. ``None`` is returned when the names do not carry at
    least two numbers, in which case callers fall back to constant distances.

    On the real 500-zone study area this recovers a 500 m grid spanning roughly
    85 x 102 km, with the expected distance decay (log-flow vs log-distance
    correlation about -0.49).
    """
    if not id2entity:
        return None
    coords = np.full((max(id2entity) + 1, 2), np.nan)

    for eid, name in id2entity.items():
        numbers = _NUMBER.findall(str(name))
        if len(numbers) < 2:
            return None
        coords[eid] = (float(numbers[-2]), float(numbers[-1]))

    return None if np.isnan(coords).any() else coords


def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance (km) between lon/lat points."""
    radius_km = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (np.sin(dphi / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2)
    return 2 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def coords_to_distance_matrix(coords, floor=0.1):
    """Pairwise distances (km) from an ``(E, 2)`` coordinate array.

    Handles both projected coordinates (treated as metres, converted with a
    planar Euclidean distance) and lon/lat degrees (great-circle distance).
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.max(initial=0.0) > _DEGREE_UPPER_BOUND:
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1)) / METERS_PER_KM
    else:
        dist = haversine_distance(coords[:, 0][:, None], coords[:, 1][:, None],
                                  coords[:, 0][None, :], coords[:, 1][None, :])
    np.fill_diagonal(dist, 0.0)
    return np.maximum(dist, floor)


def build_distance_matrix(id2entity, dist_dict, num_entities, missing=np.inf):
    """Materialize ``dist_dict`` as an ``(E, E)`` matrix (km).

    Unknown pairs become ``missing`` (``inf`` by default, so they are excluded
    from radius queries instead of being treated as adjacent).
    """
    dist = np.full((num_entities, num_entities), missing, dtype=np.float64)
    index = {name: eid for eid, name in id2entity.items() if name}
    for (src, dst), value in dist_dict.items():
        i, j = index.get(src), index.get(dst)
        if i is not None and j is not None:
            dist[i, j] = value
    return dist


def resolve_distance_matrix(id2entity, dist_dict, num_entities,
                            default=0.1, missing=np.inf):
    """Best available ``(E, E)`` OD distance matrix (km), in priority order.

    1. ``grid_distance.csv`` -- real OD distances, when supplied;
    2. coordinates parsed from the entity names;
    3. a constant ``default`` everywhere (degenerate but never crashes).

    The constant case makes any distance-dependent model meaningless, so callers
    should report which source was used.
    """
    if dist_dict:
        return build_distance_matrix(id2entity, dist_dict, num_entities,
                                     missing=missing)
    coords = parse_entity_coordinates(id2entity)
    if coords is not None:
        return coords_to_distance_matrix(coords, floor=default)
    return np.full((num_entities, num_entities), default, dtype=np.float64)


def intervening_opportunities(od_ids, dist_matrix, mass, floor=0.1):
    """Radiation-model ``s_ij``: mass inside the circle centred at *i*, radius
    ``d(i, j)``, excluding *i* and *j* themselves.

    Parameters
    ----------
    od_ids : (N, 2) int
    dist_matrix : (E, E) float
        Pairwise distances; ``inf`` entries are never within any radius.
    mass : (E,) float
        Attractiveness of each zone (the radiation model uses the summed mass).
    floor : float
        Lower bound on the radius, so a zero-distance pair still has a circle.

    Returns
    -------
    (N,) float64
    """
    od_ids = np.asarray(od_ids, dtype=np.int64)
    origin, dest = od_ids[:, 0], od_ids[:, 1]

    radius = np.maximum(dist_matrix[origin, dest], floor)      # (N,)
    within = dist_matrix[origin] <= radius[:, None]            # (N, E)

    # exclude the two endpoints of each pair
    rows = np.arange(len(od_ids))
    within[rows, origin] = False
    within[rows, dest] = False

    return within @ np.asarray(mass, dtype=np.float64)
