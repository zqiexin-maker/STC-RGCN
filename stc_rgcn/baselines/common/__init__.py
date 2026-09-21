"""Shared building blocks for the baselines.

============  ============================================================
``io``        OD readers, sidecar dictionaries, node features, result writer
``features``  zone masses, OD feature matrices, distances, radiation s_ij
``metrics``   flow and distribution metrics, plus ``evaluate_all``
``cli``       shared argparse helpers
============  ============================================================

The 40-column split layout and the 34-column prediction layout are defined
once in :mod:`stc_rgcn.utils.schema` and re-exported here, so a baseline and the
model can never disagree about a column index.

No optional third-party library is imported at module level: scikit-learn is
used when installed and falls back to numpy, and dgl / xgboost are imported
lazily by the models that need them.
"""

from ...config import DEFAULT_BASELINE_DIR, DEFAULT_DATA_DIR
from ...utils.schema import (
    COL_DESTINATION,
    COL_ORIGIN,
    NUM_COLUMNS,
    NUM_PURPOSES,
    PREDICTION_COLUMNS,
    PURPOSE_CODES,
    PURPOSE_FLOW_SLICE,
    PURPOSE_LABELS,
    PURPOSE_PROB_SLICE,
    SPLIT_FILES,
)
from . import cli, features, io, metrics
from .cli import add_common_args, banner, resolve_output, section
from .features import (
    METERS_PER_KM,
    NodeMass,
    PurposeMass,
    build_distance_matrix,
    build_purpose_masses,
    coords_to_distance_matrix,
    flows_to_probs,
    haversine_distance,
    intervening_opportunities,
    od_mass_features,
    parse_entity_coordinates,
    resolve_distance_matrix,
)
from .io import (
    iter_od_records,
    load_baseline_dataset,
    load_entity_dict,
    load_grid_distance,
    load_node_features,
    load_relation_dict,
    read_od_split,
    read_triples,
    resolve_split_file,
    write_predictions,
)
from .metrics import (
    cosine_similarity,
    cpc,
    cpl,
    evaluate_all,
    evaluate_distribution,
    js_divergence,
    jsd,
    kl_divergence,
    mae,
    mape,
    per_purpose_mae,
    r2,
    rmse,
    row_cpc,
)

__all__ = [
    # schema / config
    "COL_DESTINATION", "COL_ORIGIN", "DEFAULT_BASELINE_DIR", "DEFAULT_DATA_DIR",
    "NUM_COLUMNS", "NUM_PURPOSES", "PREDICTION_COLUMNS", "PURPOSE_CODES",
    "PURPOSE_FLOW_SLICE", "PURPOSE_LABELS", "PURPOSE_PROB_SLICE", "SPLIT_FILES",
    # cli
    "add_common_args", "banner", "resolve_output", "section",
    # io
    "iter_od_records", "load_baseline_dataset", "load_entity_dict",
    "load_grid_distance", "load_node_features", "load_relation_dict",
    "read_od_split", "read_triples", "resolve_split_file", "write_predictions",
    # features
    "METERS_PER_KM", "NodeMass", "PurposeMass", "build_distance_matrix",
    "build_purpose_masses", "coords_to_distance_matrix", "flows_to_probs",
    "haversine_distance", "intervening_opportunities", "od_mass_features",
    "parse_entity_coordinates", "resolve_distance_matrix",
    # metrics
    "cosine_similarity", "cpc", "cpl", "evaluate_all", "evaluate_distribution",
    "js_divergence", "jsd", "kl_divergence", "mae", "mape", "per_purpose_mae",
    "r2", "rmse", "row_cpc",
    # submodules
    "cli", "features", "io", "metrics",
]
