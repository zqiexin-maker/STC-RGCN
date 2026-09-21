"""On-disk dataset layout, shared by the model, the baselines and the scorer.

Split files
-----------
``train.txt`` / ``valid.txt`` / ``test.txt`` are tab-separated, header-less and
have 40 columns per OD pair (0-indexed)::

    [0]      origin entity string, e.g. ``HYID8114000|17795500``
    [1..4]   4 transport-mode shares, continuous, used as edge relation weights
    [5]      destination entity string
    [6..20]  15 trip-purpose probabilities (row sums to 1)
    [21..24] 4 transport-mode flows      (unused: the shares above are used)
    [25..39] 15 trip-purpose flows       (total flow = row sum)

Sidecar files in the same directory::

    entities.dict           <id>\\t<entity string>, ids 0..N-1
    relations_mode.dict     <id>\\t<transport mode>, 4 rows
    relations_purpose.dict  <id>\\t<purpose code>,   15 rows
    features.txt            one row per entity; the first 2 columns are metadata
                            and are dropped by the loader
    grid_distance.csv       optional real OD distances in km

Older releases of this dataset used ``{train,valid,test}_with_flows.txt`` and
``relations_method.dict``.  :func:`resolve_split_file` and
:func:`resolve_sidecar_file` accept both spellings, so an existing private
dataset does not need renaming.
"""

from __future__ import annotations

import os

# ── Row layout ─────────────────────────────────────────────────────────────
NUM_COLUMNS = 40
COL_ORIGIN = 0
COL_DESTINATION = 5
MODE_SHARE_SLICE = slice(1, 5)
PURPOSE_PROB_SLICE = slice(6, 21)
PURPOSE_FLOW_SLICE = slice(25, 40)

#: Number of node metadata columns in ``features.txt`` that are not features.
NUM_FEATURE_METADATA_COLUMNS = 2

# ── Transport modes ────────────────────────────────────────────────────────
#: Order that :func:`stc_rgcn.data.read_split` reorders mode shares into.
MODE_NAMES = ("bus", "driving", "subway", "non_vehicle")
NUM_MODES = len(MODE_NAMES)

# ── Trip purposes ──────────────────────────────────────────────────────────
#: Purpose codes in ``relations_purpose.dict`` order (column order of the
#: 15-dimensional distribution).  The codes are pinyin abbreviations of the
#: Chinese survey categories; :data:`PURPOSE_LABELS` spells them out.
PURPOSE_CODES = (
    "sx", "dxsk", "jypx", "gw", "shfw", "xxyy", "ms",
    "sw", "gz", "hj", "yl", "tqfy", "tccx", "kccx", "ly",
)
NUM_PURPOSES = len(PURPOSE_CODES)

#: ``code -> (original Chinese category, English gloss)``.  The Chinese strings
#: are the keys used by the behavioural constraint tables under
#: ``data/constraints/``; the English glosses are documentation only.
PURPOSE_LABELS = {
    "sx": ("上学", "school"),
    "dxsk": ("大学上课", "university class"),
    "jypx": ("教育培训", "education and training"),
    "gw": ("购物", "shopping"),
    "shfw": ("生活服务", "daily services"),
    "xxyy": ("休闲娱乐", "leisure and entertainment"),
    "ms": ("美食", "dining"),
    "sw": ("商务", "business"),
    "gz": ("工作", "work"),
    "hj": ("回家", "going home"),
    "yl": ("医疗", "healthcare"),
    "tqfy": ("探亲访友", "visiting friends and relatives"),
    "tccx": ("同城出行", "intra-city travel"),
    "kccx": ("跨城出行", "inter-city travel"),
    "ly": ("旅游", "tourism"),
}

# ── File names ─────────────────────────────────────────────────────────────
#: ``split -> (preferred name, legacy name)``.
SPLIT_FILES = {
    "train": ("train.txt", "train_with_flows.txt"),
    "valid": ("valid.txt", "valid_with_flows.txt"),
    "test": ("test.txt", "test_with_flows.txt"),
}

ENTITY_DICT = "entities.dict"
PURPOSE_DICT = "relations_purpose.dict"
#: ``relations_method.dict`` was the pre-1.0 spelling of ``relations_mode.dict``.
MODE_DICT = ("relations_mode.dict", "relations_method.dict")
DEFAULT_FEATURE_FILE = "features.txt"
GRID_DISTANCE_FILE = "grid_distance.csv"

#: ``grid_distance.csv`` column names, in the order origin, destination,
#: distance.  Both the original Chinese headers and English aliases are read.
DISTANCE_COLUMNS = {
    "origin": ("起点网格ID", "origin", "origin_id"),
    "destination": ("终点网格ID", "destination", "destination_id"),
    "distance": ("distance", "distance_km"),
}

# ── Prediction-file schema (34 columns) ────────────────────────────────────
#: Every model and baseline writes this layout, so ``stc-rgcn-eval score`` can
#: score a whole directory without per-model special cases.
PREDICTION_COLUMNS = (
    ("origin", "destination")
    + tuple("pred_prob_{}".format(code) for code in PURPOSE_CODES)
    + tuple("true_prob_{}".format(code) for code in PURPOSE_CODES)
    + ("pred_flow", "true_flow")
)


def resolve_split_file(data_dir, split, fallback=(), warn=True):
    """Return the path of ``split`` inside ``data_dir``.

    Accepts both the current and the legacy file name.  When the split is
    absent, the names in ``fallback`` are tried in order, which lets a dataset
    without a test split still be scored on ``valid``.

    Raises
    ------
    FileNotFoundError
        If neither ``split`` nor any ``fallback`` exists.
    """
    for candidate in (split, *fallback):
        for name in SPLIT_FILES[candidate]:
            path = os.path.join(data_dir, name)
            if os.path.exists(path):
                if candidate != split and warn:
                    print(
                        "warning: split {!r} not found in {}, falling back to {!r}".format(
                            split, data_dir, candidate
                        )
                    )
                return path
    tried = ", ".join(n for c in (split, *fallback) for n in SPLIT_FILES[c])
    raise FileNotFoundError("none of [{}] found in {}".format(tried, data_dir))


def resolve_sidecar_file(data_dir, names):
    """Return the first of ``names`` that exists in ``data_dir``.

    ``names`` may be a single file name or a tuple of accepted spellings.
    """
    if isinstance(names, str):
        names = (names,)
    for name in names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "none of [{}] found in {}".format(", ".join(names), data_dir)
    )
