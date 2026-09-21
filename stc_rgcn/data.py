"""Dataset loading.

:func:`load_dataset` returns an :class:`OdDataset`.  Which fields it populates
depends on the :class:`~stc_rgcn.config.Variant`: the purpose-only variant
carries no flow targets, the flow-only variant carries no purpose
distributions, and the two ablations without mode weights carry discrete
relation ids instead of transport-mode shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .config import Variant
from .utils.schema import (
    COL_DESTINATION,
    COL_ORIGIN,
    DEFAULT_FEATURE_FILE,
    ENTITY_DICT,
    MODE_DICT,
    MODE_NAMES,
    MODE_SHARE_SLICE,
    NUM_COLUMNS,
    NUM_FEATURE_METADATA_COLUMNS,
    NUM_MODES,
    NUM_PURPOSES,
    PURPOSE_DICT,
    PURPOSE_FLOW_SLICE,
    PURPOSE_PROB_SLICE,
    resolve_sidecar_file,
    resolve_split_file,
)

#: Tolerance for the "purpose probabilities sum to 1" check on each row.
#:
#: Split files store probabilities rounded to 3 decimals, so 15 columns can
#: drift by up to ``15 * 5e-4 = 7.5e-3`` from 1 through rounding alone.  The
#: pre-1.0 tolerance of ``1e-4`` was tighter than the data's own precision and
#: rejected two thirds of the shipped sample rows.  Rows within this tolerance
#: are renormalized by the training loop; anything beyond it is a real error.
PROB_SUM_TOLERANCE = 1e-2

#: Dummy relation id used by the variants that collapse all relations into one.
SINGLE_RELATION_ID = 0


@dataclass
class Split:
    """One train/valid/test split.

    Attributes
    ----------
    od_pairs
        ``(N, 2)`` origin/destination entity ids, or ``(N, 3)`` triples of
        ``(origin, relation_id, destination)`` for the variants that use
        discrete relations.
    mode_shares
        ``(N, 4)`` continuous transport-mode shares in :data:`MODE_NAMES`
        order, or ``None`` when the variant ignores them.
    purpose_probs
        ``(N, 15)`` trip-purpose distributions, or ``None``.
    flows
        ``(N,)`` total flow per OD pair, or ``None``.
    """

    od_pairs: np.ndarray
    mode_shares: Optional[np.ndarray] = None
    purpose_probs: Optional[np.ndarray] = None
    flows: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.od_pairs)


@dataclass
class OdDataset:
    """Everything one training run reads from disk."""

    variant: Variant
    entity_to_id: dict[str, int]
    mode_to_id: dict[str, int]
    purpose_to_id: dict[str, int]
    train: Split
    valid: Split
    test: Split
    node_features: Optional[torch.Tensor] = None

    @property
    def num_entities(self) -> int:
        return len(self.entity_to_id)

    @property
    def num_relations(self) -> int:
        """Relation channels the convolution is built for.

        One for the variants that collapse relations to a dummy id, otherwise
        one per transport mode.
        """
        return len(self.mode_to_id)

    @property
    def feature_dim(self) -> Optional[int]:
        return None if self.node_features is None else self.node_features.shape[1]

    def describe(self) -> str:
        """Render the dataset statistics printed at the start of a run."""
        lines = [
            "Dataset ({}):".format(self.variant),
            "  entities            : {}".format(self.num_entities),
            "  relation channels   : {}".format(self.num_relations),
            "  purposes            : {}".format(len(self.purpose_to_id)),
        ]
        if self.node_features is not None:
            lines.append("  node feature dim    : {}".format(self.feature_dim))
        for name in ("train", "valid", "test"):
            lines.append("  {:<5} OD pairs      : {}".format(name, len(getattr(self, name))))
        return "\n".join(lines)


def _read_id_dict(path):
    """Read a ``<id>\\t<name>`` dictionary file into ``{name: id}``."""
    mapping = {}
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_id, name = line.split("\t")
            except ValueError as exc:
                raise ValueError(
                    "{}: line {} is not '<id>\\t<name>': {!r}".format(path, line_no, line)
                ) from exc
            mapping[name] = int(raw_id)
    if not mapping:
        raise ValueError("{} is empty".format(path))
    return mapping


def load_node_features(path, num_entities):
    """Load ``features.txt`` as a ``(num_entities, feature_dim)`` float tensor.

    The first :data:`NUM_FEATURE_METADATA_COLUMNS` columns of each row are
    metadata and are dropped, so the shipped 146-column file yields 144
    features.  Row *i* must describe entity *i*.
    """
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            columns = line.split("\t")
            try:
                rows.append([float(x) for x in columns[NUM_FEATURE_METADATA_COLUMNS:]])
            except ValueError as exc:
                raise ValueError(
                    "{}: line {} is malformed: {}".format(path, line_no, exc)
                ) from exc

    if len(rows) != num_entities:
        raise ValueError(
            "{}: has {} rows but the entity dictionary has {} entries".format(
                path, len(rows), num_entities
            )
        )
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError("{}: rows have inconsistent widths {}".format(path, sorted(widths)))
    return torch.tensor(rows, dtype=torch.float32)


def read_split(path, entity_to_id, mode_to_id, variant):
    """Parse one 40-column split file into a :class:`Split`.

    Only the columns the ``variant`` consumes are parsed and validated; rows
    whose origin or destination is not in ``entity_to_id`` are skipped and
    reported as a count.
    """
    wants_modes = variant.uses_mode_weights
    wants_probs = variant.has_purpose_head
    wants_flows = variant.has_flow_head

    if wants_modes:
        missing = set(MODE_NAMES) - set(mode_to_id)
        if missing:
            raise KeyError("mode dictionary is missing {}".format(sorted(missing)))
        # Output position k holds the raw column that the dictionary assigns to
        # MODE_NAMES[k], so the model always sees the modes in a fixed order.
        mode_order = [mode_to_id[name] for name in MODE_NAMES]

    od_pairs = []
    mode_shares = []
    purpose_probs = []
    flows = []
    skipped = 0

    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < NUM_COLUMNS:
                raise ValueError(
                    "{}: line {} has {} columns, expected at least {}".format(
                        path, line_no, len(parts), NUM_COLUMNS
                    )
                )

            origin = entity_to_id.get(parts[COL_ORIGIN])
            destination = entity_to_id.get(parts[COL_DESTINATION])
            if origin is None or destination is None:
                skipped += 1
                continue

            if variant.uses_discrete_relations:
                od_pairs.append((origin, SINGLE_RELATION_ID, destination))
            else:
                od_pairs.append((origin, destination))

            if wants_modes:
                raw = _parse_floats(parts, MODE_SHARE_SLICE, NUM_MODES, path, line_no, "mode share")
                if any(value < 0 for value in raw):
                    raise ValueError("{}: line {} has a negative mode share".format(path, line_no))
                mode_shares.append([raw[index] for index in mode_order])

            if wants_probs:
                probs = _parse_floats(
                    parts, PURPOSE_PROB_SLICE, NUM_PURPOSES, path, line_no, "purpose probability"
                )
                if abs(sum(probs) - 1.0) > PROB_SUM_TOLERANCE:
                    raise ValueError(
                        "{}: line {} purpose probabilities sum to {:.6f}, expected 1".format(
                            path, line_no, sum(probs)
                        )
                    )
                purpose_probs.append(probs)

            if wants_flows:
                purpose_flows = _parse_floats(
                    parts, PURPOSE_FLOW_SLICE, NUM_PURPOSES, path, line_no, "purpose flow"
                )
                total = sum(purpose_flows)
                if total < 0:
                    raise ValueError("{}: line {} has a negative total flow".format(path, line_no))
                flows.append(total)

    if not od_pairs:
        raise ValueError("{}: no usable rows (all {} were skipped)".format(path, skipped))
    if skipped:
        print("warning: {}: skipped {} row(s) with unknown entities".format(path, skipped))

    return Split(
        od_pairs=np.array(od_pairs, dtype=np.int64),
        mode_shares=np.array(mode_shares, dtype=np.float32) if wants_modes else None,
        purpose_probs=np.array(purpose_probs, dtype=np.float32) if wants_probs else None,
        flows=np.array(flows, dtype=np.float32) if wants_flows else None,
    )


def _parse_floats(parts, column_slice, expected, path, line_no, what):
    """Slice ``parts``, convert to float and check the resulting width."""
    try:
        values = [float(x) for x in parts[column_slice]]
    except ValueError as exc:
        raise ValueError(
            "{}: line {} has a non-numeric {}: {}".format(path, line_no, what, exc)
        ) from exc
    if len(values) != expected:
        raise ValueError(
            "{}: line {} has {} {} columns, expected {}".format(
                path, line_no, len(values), what, expected
            )
        )
    return values


def load_dataset(data_dir, variant, feature_file=DEFAULT_FEATURE_FILE, verbose=True):
    """Load the dictionaries, node features and all three splits.

    Parameters
    ----------
    data_dir
        Directory holding ``entities.dict`` and the three split files.
    variant
        :class:`~stc_rgcn.config.Variant` deciding which columns are read.
    feature_file
        Node feature file inside ``data_dir``; ignored by the variants that do
        not consume node features.
    """
    variant = Variant(variant)

    entity_to_id = _read_id_dict(resolve_sidecar_file(data_dir, ENTITY_DICT))
    purpose_to_id = _read_id_dict(resolve_sidecar_file(data_dir, PURPOSE_DICT))
    mode_to_id = _read_id_dict(resolve_sidecar_file(data_dir, MODE_DICT))

    if variant.uses_discrete_relations:
        # A single dummy relation replaces the four transport modes.
        mode_to_id = {"all": SINGLE_RELATION_ID}

    node_features = None
    if variant.uses_node_features:
        node_features = load_node_features(
            resolve_sidecar_file(data_dir, feature_file), len(entity_to_id)
        )

    splits = {
        name: read_split(
            resolve_split_file(data_dir, name), entity_to_id, mode_to_id, variant
        )
        for name in ("train", "valid", "test")
    }

    dataset = OdDataset(
        variant=variant,
        entity_to_id=entity_to_id,
        mode_to_id=mode_to_id,
        purpose_to_id=purpose_to_id,
        node_features=node_features,
        **splits,
    )
    if verbose:
        print(dataset.describe())
    return dataset
