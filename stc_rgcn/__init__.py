"""STC-RGCN: Spatio-Temporal Constrained Relational Graph Convolutional Network.

A multi-task relational GCN that predicts, for every origin-destination pair,
both the 15-dimensional trip-purpose distribution and the total flow volume.
Relation weights are continuous transport-mode shares rather than discrete
relation types.

Typical use::

    from stc_rgcn import Variant, load_dataset, StcRgcn

    dataset = load_dataset("data/sample", Variant.FULL)
    model = StcRgcn(Variant.FULL, dataset.num_entities, dataset.num_relations,
                    feature_dim=dataset.feature_dim)

The package root holds what defines the method — :mod:`~stc_rgcn.config` (variant
semantics), :mod:`~stc_rgcn.data`, :mod:`~stc_rgcn.model`, :mod:`~stc_rgcn.training`
and the :mod:`~stc_rgcn.run` entry point. :mod:`~stc_rgcn.utils` holds the pieces they
are built out of: graph construction, layers, losses, GradNorm, metrics, the flow
scaler, early stopping and the dataset schema.

Submodules that pull in optional dependencies (``preprocessing`` needs gensim
and geopandas, ``baselines`` needs xgboost / dgl) are not imported here.
"""

from .config import Task, Variant
from .data import OdDataset, Split, load_dataset
from .model import StcRgcn
from .utils.early_stopping import EarlyStopper
from .utils.graph import build_full_graph, sample_subgraph
from .utils.transforms import LogMinMaxScaler

__version__ = "1.0.0"

__all__ = [
    "EarlyStopper",
    "LogMinMaxScaler",
    "OdDataset",
    "Split",
    "StcRgcn",
    "Task",
    "Variant",
    "build_full_graph",
    "load_dataset",
    "sample_subgraph",
    "__version__",
]
