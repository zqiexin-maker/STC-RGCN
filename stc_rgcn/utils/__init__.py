"""Supporting building blocks.

The package root keeps what defines STC-RGCN end to end — the variant semantics
(:mod:`~stc_rgcn.config`), the dataset (:mod:`~stc_rgcn.data`), the network
(:mod:`~stc_rgcn.model`), the training steps (:mod:`~stc_rgcn.training`) and the
entry point (:mod:`~stc_rgcn.run`). Everything they are built *out of* lives here.

==================  ======================================================
``schema``          on-disk dataset layout: column indices, purpose codes,
                    file names and the readers that resolve them
``graph``           full-graph construction and training subgraph sampling
``layers``          ``MeanAggregationConv``, ``RelationWeightedConv``
``losses``          KL, L1 and Tweedie losses, Tweedie power estimation
``gradnorm``        ``GradNormBalancer`` for multi-task loss balancing
``metrics``         flow and trip-purpose metrics, shared by training, the
                    standalone scorer and the baselines
``transforms``      ``LogMinMaxScaler`` for the heavy-tailed flow target
``early_stopping``  ``EarlyStopper`` and the per-variant metric rules
==================  ======================================================
"""

from . import (
    early_stopping,
    gradnorm,
    graph,
    layers,
    losses,
    metrics,
    schema,
    transforms,
)
from .early_stopping import EarlyStopper, monitored_metrics
from .gradnorm import GradNormBalancer
from .graph import build_full_graph, sample_subgraph
from .layers import MeanAggregationConv, RelationWeightedConv
from .losses import estimate_tweedie_power, flow_l1_loss, purpose_kl_loss, tweedie_loss
from .metrics import (
    common_part_of_commuters,
    flow_metrics,
    flow_metrics_by_magnitude,
    purpose_distribution_metrics,
)
from .schema import (
    NUM_PURPOSES,
    PREDICTION_COLUMNS,
    PURPOSE_CODES,
    PURPOSE_LABELS,
    resolve_sidecar_file,
    resolve_split_file,
)
from .transforms import LogMinMaxScaler

__all__ = [
    # classes
    "EarlyStopper",
    "GradNormBalancer",
    "LogMinMaxScaler",
    "MeanAggregationConv",
    "RelationWeightedConv",
    # schema
    "NUM_PURPOSES",
    "PREDICTION_COLUMNS",
    "PURPOSE_CODES",
    "PURPOSE_LABELS",
    "resolve_sidecar_file",
    "resolve_split_file",
    # functions
    "build_full_graph",
    "common_part_of_commuters",
    "estimate_tweedie_power",
    "flow_l1_loss",
    "flow_metrics",
    "flow_metrics_by_magnitude",
    "monitored_metrics",
    "purpose_distribution_metrics",
    "purpose_kl_loss",
    "sample_subgraph",
    "tweedie_loss",
    # submodules
    "early_stopping",
    "gradnorm",
    "graph",
    "layers",
    "losses",
    "metrics",
    "schema",
    "transforms",
]
