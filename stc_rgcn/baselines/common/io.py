"""Dataset reading and result writing for every baseline.

Three layers, from raw to model-ready:

1. **Sidecar loaders** -- :func:`load_entity_dict`, :func:`load_relation_dict`,
   :func:`load_grid_distance`, :func:`load_node_features`.
2. **Row readers** -- :func:`iter_od_records` (streaming) and the array-returning
   :func:`read_od_split` / :func:`read_triples` built on top of it.
3. **Bundled loaders** -- :func:`load_baseline_dataset` (all splits + dictionaries).

:func:`write_predictions` writes the standard 34-column result file; every baseline
uses it so any result folder can be scored with ``stc-rgcn-eval score``.

Every reader parses the 40-column layout documented in
``stc_rgcn.utils.schema``; no baseline should re-implement it.
"""

import csv
import os

import numpy as np

from ...utils.schema import (
    COL_DESTINATION,
    COL_ORIGIN,
    DEFAULT_FEATURE_FILE,
    DISTANCE_COLUMNS,
    ENTITY_DICT,
    GRID_DISTANCE_FILE,
    NUM_COLUMNS,
    NUM_FEATURE_METADATA_COLUMNS,
    NUM_PURPOSES,
    PREDICTION_COLUMNS,
    PURPOSE_CODES,
    PURPOSE_DICT,
    PURPOSE_FLOW_SLICE,
    PURPOSE_PROB_SLICE,
)
from ...utils.schema import resolve_split_file as _resolve_split_file

# ───────────────────────────────────────────────────────────────────────────
# Sidecar files
# ───────────────────────────────────────────────────────────────────────────

def load_entity_dict(data_path):
    """Load ``entities.dict`` (``<id>\\t<entity string>``).

    Returns ``(entity2id, id2entity)``.
    """
    entity2id, id2entity = {}, {}
    with open(os.path.join(data_path, ENTITY_DICT), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            eid, entity = line.split("\t", 1)
            entity2id[entity] = int(eid)
            id2entity[int(eid)] = entity
    return entity2id, id2entity


def load_relation_dict(data_path, filename=PURPOSE_DICT):
    """Load the trip-purpose dictionary.

    Ids are normalized to start at 0: some exports are already 0-indexed, others
    start at 1, and guessing wrong silently shifts every purpose label by one.
    Falls back to ``relations.dict`` and finally to the built-in purpose labels
    when neither file exists. Returns ``(relation2id, id2relation)``.
    """
    fpath = os.path.join(data_path, filename)
    if not os.path.exists(fpath):
        legacy = os.path.join(data_path, "relations.dict")
        if not os.path.exists(legacy):
            return ({n: i for i, n in enumerate(PURPOSE_CODES)},
                    {i: n for i, n in enumerate(PURPOSE_CODES)})
        fpath = legacy

    pairs = []
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rid, name = line.split("\t", 1)
            pairs.append((int(rid), name))
    if not pairs:
        return ({n: i for i, n in enumerate(PURPOSE_CODES)},
                {i: n for i, n in enumerate(PURPOSE_CODES)})

    offset = min(rid for rid, _ in pairs)      # 0 or 1 depending on the export
    return ({name: rid - offset for rid, name in pairs},
            {rid - offset: name for rid, name in pairs})


def _pick_column(fieldnames, aliases, path):
    """Return the first of ``aliases`` present in ``fieldnames``."""
    for alias in aliases:
        if alias in fieldnames:
            return alias
    raise KeyError(
        "{}: none of the columns {} are present; found {}".format(
            path, list(aliases), list(fieldnames)))


def load_grid_distance(data_path):
    """Load ``grid_distance.csv`` into ``{(origin, destination): distance_km}``.

    Both the original Chinese headers (``起点网格ID,终点网格ID,distance``) and
    English aliases are accepted -- see :data:`stc_rgcn.utils.schema.DISTANCE_COLUMNS`.
    Returns an empty mapping when the file is absent, in which case callers
    fall back to the coordinates embedded in the entity names.
    """
    fpath = os.path.join(data_path, GRID_DISTANCE_FILE)
    distances = {}
    if not os.path.exists(fpath):
        return distances

    with open(fpath, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        origin_col = _pick_column(fields, DISTANCE_COLUMNS["origin"], fpath)
        dest_col = _pick_column(fields, DISTANCE_COLUMNS["destination"], fpath)
        distance_col = _pick_column(fields, DISTANCE_COLUMNS["distance"], fpath)
        for row in reader:
            key = (row[origin_col].strip(), row[dest_col].strip())
            distances[key] = float(row[distance_col])
    return distances


def load_node_features(data_path, num_entities,
                       filename=DEFAULT_FEATURE_FILE, skip_cols=NUM_FEATURE_METADATA_COLUMNS):
    """Load the per-node attribute matrix (one row per entity, row *i* = node *i*).

    The first ``skip_cols`` columns are non-feature columns (index / coordinates)
    and are dropped. Returns an ``(num_entities, feat_dim)`` float32 array, or
    ``None`` when the file is missing.
    """
    fpath = os.path.join(data_path, filename)
    if not os.path.exists(fpath):
        return None

    rows = []
    with open(fpath, encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            columns = line.rstrip("\n").split("\t")
            if len(columns) <= skip_cols:
                raise ValueError(
                    "{}: line {} has {} columns, expected more than {}".format(
                        fpath, line_idx + 1, len(columns), skip_cols))
            try:
                rows.append([float(v) for v in columns[skip_cols:]])
            except ValueError as exc:
                raise ValueError("{}: line {}: {}".format(fpath, line_idx + 1, exc))

    if len(rows) != num_entities:
        raise ValueError(
            "node count mismatch: {} entities in entities.dict but {} "
            "rows in {} -- make sure they describe the same zones".format(
                num_entities, len(rows), fpath))

    return np.array(rows, dtype=np.float32)


def resolve_split_file(data_dir, split, fallback=("valid", "train"), warn=True):
    """Path of ``split`` inside ``data_dir``, degrading gracefully.

    Thin wrapper over :func:`stc_rgcn.utils.schema.resolve_split_file` that supplies
    the baseline default: a dataset shipped without a test split evaluates on
    validation (then train) rather than crashing. Pass ``fallback=()`` for
    strict behaviour.
    """
    return _resolve_split_file(data_dir, split, fallback=fallback, warn=warn)


# ───────────────────────────────────────────────────────────────────────────
# Row readers
# ───────────────────────────────────────────────────────────────────────────

def iter_od_records(path, entity2id, verbose=False):
    """Stream the OD records of one split file.

    Yields ``(origin_id, dest_id, purpose_probs, purpose_flows)`` where the
    flows are a 15-vector of floats and the probabilities are a 15-vector that
    either sums to 1 or is all-zero. Malformed rows and unknown entities are
    skipped (counted when ``verbose``).
    """
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < NUM_COLUMNS:
                skipped += 1
                continue
            try:
                origin = entity2id[parts[COL_ORIGIN]]
                dest = entity2id[parts[COL_DESTINATION]]
                probs = [float(x) for x in parts[PURPOSE_PROB_SLICE]]
                flows = [float(x) for x in parts[PURPOSE_FLOW_SLICE]]
            except (KeyError, ValueError):
                skipped += 1
                continue
            if len(probs) != NUM_PURPOSES or len(flows) != NUM_PURPOSES:
                skipped += 1
                continue
            yield origin, dest, probs, flows

    if verbose and skipped:
        print("  {}: {} malformed row(s) skipped".format(
            os.path.basename(path), skipped))


def read_od_split(path, entity2id, verbose=True):
    """Read one split file into arrays.

    Returns ``(od_ids, total_flows, purpose_flows, purpose_probs)`` with shapes
    ``(N, 2)`` int64, ``(N,)`` float64, ``(N, 15)`` float64, ``(N, 15)`` float64.
    """
    origins, dests, totals, pflows, probs = [], [], [], [], []
    for origin, dest, prob, flow in iter_od_records(path, entity2id):
        origins.append(origin)
        dests.append(dest)
        totals.append(sum(flow))
        pflows.append(flow)
        probs.append(prob)

    if verbose:
        print("  {}: {} rows".format(os.path.basename(path), len(origins)))
    return (
        np.column_stack((origins, dests)).astype(np.int64) if origins
        else np.zeros((0, 2), dtype=np.int64),
        np.array(totals, dtype=np.float64),
        np.array(pflows, dtype=np.float64).reshape(-1, NUM_PURPOSES),
        np.array(probs, dtype=np.float64).reshape(-1, NUM_PURPOSES),
    )


def read_triples(file_path, entity2id):
    """Read a split file in the R-GCN quadruple layout used by :func:`load_baseline_dataset`.

    Returns ``(quadruples, purpose_probs, total_flows, purpose_flows)`` with
    ``quadruples`` of shape ``(N, 3)`` = ``(head, 0, tail)``.
    """
    quadruples, purpose_probs, total_flows, purpose_flows = [], [], [], []

    for origin, dest, probs, flows in iter_od_records(file_path, entity2id):
        prob_sum = sum(probs)
        # all-zero rows (no purpose distribution for this OD) are kept as-is;
        # the rest must already be normalized
        if prob_sum > 1e-8 and abs(prob_sum - 1.0) > 1e-3:
            continue
        total = sum(flows)
        if total < 0:
            continue
        quadruples.append((origin, 0, dest))
        purpose_probs.append(probs)
        total_flows.append(total)
        purpose_flows.append(flows)

    return (
        np.array(quadruples, dtype=np.int64).reshape(-1, 3),
        np.array(purpose_probs, dtype=np.float32).reshape(-1, NUM_PURPOSES),
        np.array(total_flows, dtype=np.float32),
        np.array(purpose_flows, dtype=np.float32).reshape(-1, NUM_PURPOSES),
    )


def load_baseline_dataset(data_path, verbose=True):
    """Load every split plus the sidecar dictionaries.

    Returns a dict with ``entity2id``, ``id2entity``, ``relation2id``,
    ``id2relation``, ``dist_dict`` and one entry per split holding
    ``(ids_matrix[N, 2], purpose_probs[N, 15], total_flows[N], purpose_flows[N, 15])``.
    """
    if verbose:
        print("Loading dataset from {} ...".format(data_path))

    entity2id, id2entity = load_entity_dict(data_path)
    relation2id, id2relation = load_relation_dict(data_path)
    dist_dict = load_grid_distance(data_path)

    result = {
        "entity2id": entity2id,
        "id2entity": id2entity,
        "relation2id": relation2id,
        "id2relation": id2relation,
        "dist_dict": dist_dict,
    }
    for split in ("train", "valid", "test"):
        quad, prob, flow, pflow = read_triples(
            resolve_split_file(data_path, split, fallback=()), entity2id)
        result[split] = (quad[:, [0, 2]], prob, flow, pflow)   # keep (head, tail)

    if verbose:
        print("\n" + "=" * 50)
        print("  entities       : {}".format(len(entity2id)))
        print("  trip purposes  : {}".format(len(relation2id)))
        print("  distance pairs : {}".format(len(dist_dict)))
        for split in ("train", "valid", "test"):
            print("  {:<12s}: {} rows".format(split, len(result[split][0])))
        print("=" * 50 + "\n")
    return result


# ───────────────────────────────────────────────────────────────────────────
# Result writing
# ───────────────────────────────────────────────────────────────────────────

def write_predictions(od_ids, id2entity, pred_probs, true_probs,
                 pred_flows, true_flows, path):
    """Write predictions to ``path`` in the standard 34-column layout.

    Parameters
    ----------
    od_ids : (N, 2) array-like of int
        Integer (origin, destination) entity ids.
    id2entity : dict[int, str]
        Entity-id to entity-string map (a missing id falls back to the number).
    pred_probs, true_probs : (N, 15) array-like
        Predicted / ground-truth trip-purpose distributions.
    pred_flows, true_flows : (N,) array-like
        Predicted / ground-truth total flows.
    path : str
        Output file path; parent directories are created as needed.
    """
    od_ids = np.asarray(od_ids)
    pred_probs = np.asarray(pred_probs, dtype=np.float64).reshape(-1, NUM_PURPOSES)
    true_probs = np.asarray(true_probs, dtype=np.float64).reshape(-1, NUM_PURPOSES)
    pred_flows = np.asarray(pred_flows, dtype=np.float64).reshape(-1)
    true_flows = np.asarray(true_flows, dtype=np.float64).reshape(-1)

    if not (len(od_ids) == len(pred_probs) == len(true_probs)
            == len(pred_flows) == len(true_flows)):
        raise ValueError(
            "length mismatch: od_ids={}, pred_probs={}, true_probs={}, "
            "pred_flows={}, true_flows={}".format(
                len(od_ids), len(pred_probs), len(true_probs),
                len(pred_flows), len(true_flows)))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(PREDICTION_COLUMNS) + "\n")
        for i in range(len(od_ids)):
            origin = id2entity.get(int(od_ids[i, 0]), str(od_ids[i, 0]))
            dest = id2entity.get(int(od_ids[i, 1]), str(od_ids[i, 1]))
            pred_p = "\t".join("{:.6e}".format(x) for x in pred_probs[i])
            true_p = "\t".join("{:.6e}".format(x) for x in true_probs[i])
            f.write("\t".join((origin, dest, pred_p, true_p,
                               "{:.6f}".format(pred_flows[i]),
                               "{:.6f}".format(true_flows[i]))) + "\n")
    print("[OK] results saved: {}".format(path))
