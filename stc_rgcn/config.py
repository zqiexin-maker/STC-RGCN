"""Model variants, task selection and default paths.

The two enums here replace the string-membership tests that used to be spread
across the training loop, the model and the data loader (``if model_type in
['base', 'feature', ...]``).  Every capability a variant implies is exposed as a
property, so adding a variant means editing one file.
"""

from __future__ import annotations

import os
from enum import Enum

# ── Repository layout ──────────────────────────────────────────────────────
# The defaults below assume the package is used from a checkout (``pip install
# -e .`` keeps the files in place).  Installed as a plain wheel there is no
# ``data/`` beside the package, so pass ``--data-dir`` explicitly; the CLI says
# so when the default is missing.
# The package directory is ``src/`` itself, so the repository root is one level up.
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_DIR)

DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data", "sample")
DEFAULT_CONSTRAINT_DIR = os.path.join(REPO_ROOT, "data", "constraints")
#: Intermediate products of the preprocessing pipeline (git-ignored).
DEFAULT_INTERIM_DIR = os.path.join(REPO_ROOT, "data", "interim")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "runs")
DEFAULT_BASELINE_DIR = os.path.join(REPO_ROOT, "runs", "baselines")


class Variant(str, Enum):
    """Model variant, i.e. which inputs and which prediction heads are enabled.

    ``BASE`` .. ``RELATION`` are the ablations; ``FULL`` is the model reported in
    the paper and ``FULL_TWEEDIE`` swaps its L1 flow loss for a Tweedie loss.
    """

    BASE = "base"
    FEATURE = "feature"
    RELATION = "relation"
    PURPOSE = "purpose"
    FLOW = "flow"
    FULL = "full"
    FULL_TWEEDIE = "full_tweedie"

    def __str__(self) -> str:  # so argparse choices and f-strings read nicely
        return self.value

    # ── Inputs ─────────────────────────────────────────────────────────────
    @property
    def uses_node_features(self) -> bool:
        """Whether the fused POI + trajectory node features are fed to the model."""
        return self in {
            Variant.FEATURE,
            Variant.PURPOSE,
            Variant.FLOW,
            Variant.FULL,
            Variant.FULL_TWEEDIE,
        }

    @property
    def uses_mode_weights(self) -> bool:
        """Whether the continuous 4-mode transport shares drive the convolutions.

        ``False`` selects the plain mean-aggregation convolution instead, which
        is the ablation the paper calls "no relation".
        """
        return self in {
            Variant.RELATION,
            Variant.PURPOSE,
            Variant.FLOW,
            Variant.FULL,
            Variant.FULL_TWEEDIE,
        }

    @property
    def uses_discrete_relations(self) -> bool:
        """Whether split rows are read as ``(origin, relation_id, destination)``.

        The two variants that do not consume mode shares collapse all relations
        to a single dummy id and keep the classic triple layout; everything else
        works on plain origin/destination pairs.
        """
        return self in {Variant.BASE, Variant.FEATURE}

    # ── Heads ──────────────────────────────────────────────────────────────
    @property
    def has_purpose_head(self) -> bool:
        """Whether the 15-dimensional trip-purpose distribution is predicted."""
        return self is not Variant.FLOW

    @property
    def has_flow_head(self) -> bool:
        """Whether the scalar total flow is predicted."""
        return self is not Variant.PURPOSE

    @property
    def uses_tweedie_loss(self) -> bool:
        """Whether the flow head is trained with a Tweedie instead of an L1 loss."""
        return self is Variant.FULL_TWEEDIE

    @property
    def is_multi_task(self) -> bool:
        """Whether both heads are active, so the two losses need balancing."""
        return self.has_purpose_head and self.has_flow_head


class Task(str, Enum):
    """Which loss drives training and early stopping on a multi-head variant.

    Ignored by the single-head variants (:attr:`Variant.PURPOSE` and
    :attr:`Variant.FLOW`), which can only optimize the head they have.
    """

    FLOW = "flow"
    PURPOSE = "purpose"
    MULTI = "multi"

    def __str__(self) -> str:
        return self.value
