"""Stage 5: concatenate POI and trajectory embeddings into ``features.txt``.

Row *i* of the output describes entity *i* of ``entities.dict``, which is the
alignment the dataset loader relies on.  Entities without a trajectory
embedding are zero-filled, and the count of such entities is reported: a large
number means the two sources are keyed differently and the join is wrong.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from ..config import DEFAULT_DATA_DIR, DEFAULT_INTERIM_DIR
from ..utils.schema import ENTITY_DICT

#: Fraction of zero-filled rows above which the join is treated as broken.
MAX_MISSING_FRACTION = 0.5


def load_poi_features(path):
    """Load the tab-separated POI/landuse matrix, one row per entity."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append([float(x) for x in line.split("\t")])
            except ValueError as exc:
                raise ValueError(
                    "{}: line {} is not numeric: {}".format(path, line_no, exc)
                ) from exc

    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError("{}: rows have inconsistent widths {}".format(path, sorted(widths)))
    return np.asarray(rows, dtype=np.float64)


def load_trajectory_embeddings(path):
    """Load stage 4 output into ``{grid_id: vector}``.

    Accepts both the wide layout written since 1.0 (``grid_id`` plus one column
    per dimension) and the earlier layout, whose ``embedding`` column held a
    stringified list.
    """
    frame = pd.read_csv(path)
    if "grid_id" not in frame.columns:
        raise KeyError("{}: needs a 'grid_id' column".format(path))

    if "embedding" in frame.columns:
        vectors = {
            int(row["grid_id"]): [float(x) for x in str(row["embedding"]).strip("[]").split(",")]
            for _, row in frame.iterrows()
        }
    else:
        value_columns = [c for c in frame.columns if c != "grid_id"]
        vectors = {
            int(row.grid_id): [float(getattr(row, c)) for c in value_columns]
            for row in frame.itertuples()
        }

    widths = {len(v) for v in vectors.values()}
    if len(widths) != 1:
        raise ValueError("{}: embeddings have inconsistent widths {}".format(path, sorted(widths)))
    return vectors, widths.pop()


def count_entities(entity_dict_path):
    with open(entity_dict_path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def fuse_features(entity_dict_path, trajectory_embedding_path, poi_feature_path=None):
    """Build the fused feature matrix.

    Trajectory embeddings are looked up by *entity index*, so grid ids must be
    the 0-based entity numbering produced by stage 2.

    Returns
    -------
    tuple
        ``(matrix, num_missing)`` -- the fused matrix and how many entities had
        no trajectory embedding.
    """
    num_entities = count_entities(entity_dict_path)
    trajectory_vectors, trajectory_dim = load_trajectory_embeddings(trajectory_embedding_path)

    poi_matrix = None
    if poi_feature_path:
        poi_matrix = load_poi_features(poi_feature_path)
        if len(poi_matrix) != num_entities:
            raise ValueError(
                "{}: has {} rows but {} lists {} entities".format(
                    poi_feature_path, len(poi_matrix), entity_dict_path, num_entities
                )
            )

    zero = [0.0] * trajectory_dim
    rows, num_missing = [], 0
    for index in range(num_entities):
        vector = trajectory_vectors.get(index)
        if vector is None:
            vector, num_missing = zero, num_missing + 1
        rows.append(list(poi_matrix[index]) + vector if poi_matrix is not None else list(vector))

    if num_missing > num_entities * MAX_MISSING_FRACTION:
        raise ValueError(
            "{} of {} entities have no trajectory embedding. Trajectory grid ids "
            "must be the 0-based entity indices from stage 2; they look keyed "
            "differently here.".format(num_missing, num_entities)
        )
    return np.asarray(rows, dtype=np.float64), num_missing


def build_parser():
    parser = argparse.ArgumentParser(
        prog="stc-rgcn-fuse-features",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--entities", default=os.path.join(DEFAULT_DATA_DIR, ENTITY_DICT),
                        help="Entity dictionary that fixes the row order")
    parser.add_argument("--trajectory-embeddings",
                        default=os.path.join(DEFAULT_INTERIM_DIR, "trajectory_embeddings.csv"),
                        help="Stage 4 output")
    parser.add_argument("--poi-features", default=None,
                        help="POI/landuse matrix; omit for trajectory-only features")
    parser.add_argument("--output", default=os.path.join(DEFAULT_DATA_DIR, "features.txt"),
                        help="Fused feature file")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    matrix, num_missing = fuse_features(
        entity_dict_path=args.entities,
        trajectory_embedding_path=args.trajectory_embeddings,
        poi_feature_path=args.poi_features,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in matrix:
            handle.write("\t".join("{:.6f}".format(value) for value in row) + "\n")

    print("entities            : {}".format(len(matrix)))
    print("feature dimension   : {}".format(matrix.shape[1]))
    print("zero-filled entities: {}".format(num_missing))
    print("written to          : {}".format(args.output))


if __name__ == "__main__":
    main()
