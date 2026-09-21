"""Scoring of saved predictions and ablation contribution analysis.

``scoring`` reads the shared 34-column prediction layout that every model and
baseline writes, so a directory of results from different models can be scored
in one pass.  ``contribution`` turns a table of ablation results into
per-module contributions.
"""

from .contribution import CONTRIBUTION_METRICS, module_contributions
from .scoring import (
    aggregate_group_stats,
    group_by_model,
    read_prediction_file,
    score_file,
    score_folder,
)

__all__ = [
    "CONTRIBUTION_METRICS",
    "aggregate_group_stats",
    "group_by_model",
    "module_contributions",
    "read_prediction_file",
    "score_file",
    "score_folder",
]
