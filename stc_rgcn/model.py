"""The STC-RGCN network.

A single class covers every :class:`~stc_rgcn.config.Variant`: the variant
decides whether node features are encoded, whether the convolutions are
relation-weighted, and which of the two prediction heads exist.  Losses live in
:mod:`stc_rgcn.utils.losses` and metrics in :mod:`stc_rgcn.utils.metrics`, so
this module only describes the architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Variant
from .utils.layers import MeanAggregationConv, RelationWeightedConv
from .utils.schema import NUM_PURPOSES

#: Default widths. The head input is ``2 * NODE_DIM`` (an origin and a
#: destination representation), plus ``num_bases`` when relation weights are on.
EMBEDDING_DIM = 128
FEATURE_EMBEDDING_DIM = 72
HIDDEN_DIM = 256
NODE_DIM = 100

#: Hidden widths of the two prediction heads, from the pair representation
#: down to their output. The purpose head uses all four, the flow head the
#: first three.
PURPOSE_HEAD_DIMS = (256, 128, 64, 32)
FLOW_HEAD_DIMS = (256, 128, 64)

#: Negative slope shared by every LeakyReLU in the encoder and the heads.
LEAKY_SLOPE = 0.1

#: Dropout inside the feature encoder, independent of the ``dropout`` argument
#: that applies to the convolution stack and the heads.
FEATURE_DROPOUT = 0.2


def _mlp(dims, dropout, dropout_after=1, final_activation=None):
    """Build ``Linear -> LayerNorm -> LeakyReLU`` blocks over ``dims``.

    The last pair in ``dims`` becomes a bare ``Linear`` unless
    ``final_activation`` is given.  ``dropout_after`` is the index of the block
    that dropout follows, matching the original hand-written stacks.
    """
    layers = []
    for index, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
        is_last = index == len(dims) - 2
        layers.append(nn.Linear(in_dim, out_dim))
        if not is_last:
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.LeakyReLU(LEAKY_SLOPE))
            if index == dropout_after:
                layers.append(nn.Dropout(dropout))
        elif final_activation is not None:
            layers.append(final_activation)
    return nn.Sequential(*layers)


def _init_linear_(module, nonlinearity):
    for layer in module:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)


class StcRgcn(nn.Module):
    """Spatio-Temporal Constrained Relational Graph Convolutional Network.

    Parameters
    ----------
    variant
        Which inputs and heads to build; see :class:`stc_rgcn.config.Variant`.
    num_entities
        Number of zones, i.e. rows of the entity embedding table.
    num_relations
        Number of relation channels: 4 transport modes, or 1 for the variants
        that collapse relations.
    feature_dim
        Width of the node feature matrix; required when the variant consumes
        node features.
    num_bases
        Basis decomposition rank of :class:`~stc_rgcn.utils.layers.RelationWeightedConv`.
    dropout
        Dropout applied between the convolutions and inside the heads.
    """

    def __init__(
        self,
        variant,
        num_entities,
        num_relations,
        feature_dim=None,
        num_bases=4,
        dropout=0.2,
        num_purposes=NUM_PURPOSES,
    ):
        super().__init__()
        self.variant = Variant(variant)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.dropout = dropout
        self.num_purposes = num_purposes

        if self.variant.uses_node_features and feature_dim is None:
            raise ValueError("variant {} needs feature_dim".format(self.variant))

        self.entity_embedding = nn.Embedding(num_entities, EMBEDDING_DIM)
        nn.init.xavier_normal_(self.entity_embedding.weight)

        encoder_dim = EMBEDDING_DIM
        if self.variant.uses_node_features:
            self.feature_encoder = nn.Sequential(
                nn.Linear(feature_dim, HIDDEN_DIM),
                nn.LayerNorm(HIDDEN_DIM),
                nn.LeakyReLU(LEAKY_SLOPE),
                nn.Linear(HIDDEN_DIM, EMBEDDING_DIM),
                nn.LayerNorm(EMBEDDING_DIM),
                nn.LeakyReLU(LEAKY_SLOPE),
                nn.Dropout(FEATURE_DROPOUT),
                nn.Linear(EMBEDDING_DIM, FEATURE_EMBEDDING_DIM),
                nn.LayerNorm(FEATURE_EMBEDDING_DIM),
                nn.LeakyReLU(LEAKY_SLOPE),
            )
            encoder_dim += FEATURE_EMBEDDING_DIM

        if self.variant.uses_mode_weights:
            # Shared by both convolutions and by the heads, so the mapping from
            # mode shares to basis coefficients is learnt once.
            self.mode_projection = nn.Linear(num_relations, num_bases, bias=False)
            nn.init.xavier_uniform_(self.mode_projection.weight)
            self.conv1 = RelationWeightedConv(
                encoder_dim, HIDDEN_DIM, num_relations, num_bases, self.mode_projection
            )
            self.conv2 = RelationWeightedConv(
                HIDDEN_DIM, NODE_DIM, num_relations, num_bases, self.mode_projection
            )
        else:
            self.mode_projection = None
            self.conv1 = MeanAggregationConv(encoder_dim, HIDDEN_DIM)
            self.conv2 = MeanAggregationConv(HIDDEN_DIM, NODE_DIM)

        head_dim = 2 * NODE_DIM + (num_bases if self.variant.uses_mode_weights else 0)

        if self.variant.has_purpose_head:
            if self.variant.uses_mode_weights:
                self.purpose_head = _mlp(
                    [head_dim, *PURPOSE_HEAD_DIMS, num_purposes], dropout
                )
            else:
                # Without relation features a linear read-out is enough.
                self.purpose_head = nn.Sequential(nn.Linear(head_dim, num_purposes))
            _init_linear_(self.purpose_head, "relu")
        else:
            self.purpose_head = None

        if self.variant.has_flow_head:
            # Softplus keeps flows non-negative; the Tweedie loss clamps
            # instead, so its head stays linear.
            final = None if self.variant.uses_tweedie_loss else nn.Softplus()
            if self.variant.uses_mode_weights or self.variant is Variant.FEATURE:
                self.flow_head = _mlp(
                    [head_dim, *FLOW_HEAD_DIMS, 1], dropout, final_activation=final
                )
            else:
                self.flow_head = nn.Sequential(
                    nn.Linear(head_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus()
                )
            _init_linear_(self.flow_head, "leaky_relu")
        else:
            self.flow_head = None

    # ── Encoder ────────────────────────────────────────────────────────────
    def forward(self, data):
        """Encode every node of ``data`` into a ``[num_nodes, NODE_DIM]`` matrix."""
        x = self.entity_embedding(data.entity.long())

        if self.variant.uses_node_features:
            x = torch.cat([x, self.feature_encoder(data.x)], dim=1)

        if self.variant.uses_mode_weights:
            x = F.relu(self.conv1(x, data.edge_index, data.edge_type, data.edge_norm))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, data.edge_index, data.edge_type, data.edge_norm)
        else:
            x = F.relu(self.conv1(x, data.edge_index.long()))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, data.edge_index.long())

        return x

    # ── Heads ──────────────────────────────────────────────────────────────
    def _pair_features(self, node_embedding, origin_idx, destination_idx, mode_shares):
        origin = node_embedding[origin_idx]
        destination = node_embedding[destination_idx]
        if self.variant.uses_mode_weights and mode_shares is not None:
            relation = mode_shares @ self.mode_projection.weight.T
            return torch.cat([origin, relation, destination], dim=1)
        return torch.cat([origin, destination], dim=1)

    def predict_purpose_probs(self, node_embedding, origin_idx, destination_idx, mode_shares=None):
        """Predict the trip-purpose distribution for each OD pair."""
        if self.purpose_head is None:
            raise RuntimeError("variant {} has no purpose head".format(self.variant))
        pair = self._pair_features(node_embedding, origin_idx, destination_idx, mode_shares)
        return torch.softmax(self.purpose_head(pair), dim=-1)

    def predict_flow(self, node_embedding, origin_idx, destination_idx, mode_shares=None):
        """Predict the total flow for each OD pair."""
        if self.flow_head is None:
            raise RuntimeError("variant {} has no flow head".format(self.variant))
        pair = self._pair_features(node_embedding, origin_idx, destination_idx, mode_shares)
        return self.flow_head(pair).squeeze(1)

    # ── Regularization ─────────────────────────────────────────────────────
    def parameter_penalty(self):
        """Mean squared magnitude of the embeddings, convolutions and heads.

        Applied on top of the task losses with weight ``--reg-weight``.  It is
        a mean-of-squares per tensor rather than a global L2 norm, so tensors
        contribute regardless of their size.
        """
        penalty = torch.mean(self.entity_embedding.weight.pow(2))

        for conv in (self.conv1, self.conv2):
            if isinstance(conv, RelationWeightedConv):
                penalty = penalty + torch.mean(conv.basis.pow(2))
                if conv.root is not None:
                    penalty = penalty + torch.mean(conv.root.pow(2))
            else:
                penalty = penalty + torch.mean(conv.linear.weight.pow(2))

        for head in (self.purpose_head, self.flow_head):
            if head is None:
                continue
            for layer in head:
                weight = getattr(layer, "weight", None)
                if weight is not None:
                    penalty = penalty + torch.mean(weight.pow(2))

        return penalty

    def extra_repr(self):
        return "variant={}, num_entities={}, num_relations={}, num_bases={}".format(
            self.variant, self.num_entities, self.num_relations, self.num_bases
        )
