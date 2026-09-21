"""Graph convolution layers.

:class:`RelationWeightedConv` is the layer the paper contributes: instead of
looking a discrete relation type up in a weight table, it reads a *continuous*
vector of transport-mode shares per edge and mixes the basis matrices with it.
:class:`MeanAggregationConv` is the relation-free ablation counterpart.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing


def uniform_fan_in_(tensor, fan_in):
    """Fill ``tensor`` in place with ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``."""
    if tensor is None:
        return tensor
    bound = 1.0 / math.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


class MeanAggregationConv(MessagePassing):
    """Plain graph convolution: transform each neighbour, then average.

    Named to avoid confusion with :class:`torch_geometric.nn.GCNConv`, which
    applies symmetric degree normalization this layer deliberately omits.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="mean")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return self.linear(x_j)

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


class RelationWeightedConv(MessagePassing):
    """R-GCN convolution driven by continuous relation weights.

    For each edge the shared ``mode_projection`` maps its transport-mode share
    vector onto ``num_bases`` coefficients, which mix the basis matrices into a
    per-edge transform.  The message is that transform applied to the
    *concatenated* source and target features, so an edge sees both of its
    endpoints rather than only the neighbour.

    Parameters
    ----------
    in_channels
        Width of a single node representation.  The per-edge transform
        consumes ``2 * in_channels`` because endpoints are concatenated.
    mode_projection
        ``nn.Linear(num_modes, num_bases, bias=False)`` shared with the model
        and the prediction heads, so relation weights are learnt once.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_relations,
        num_bases,
        mode_projection,
        root_weight=True,
        bias=True,
        **kwargs,
    ):
        super().__init__(aggr="mean", **kwargs)

        self.in_channels = in_channels
        self.message_channels = in_channels * 2
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.mode_projection = mode_projection

        self.basis = nn.Parameter(torch.empty(num_bases, self.message_channels, out_channels))

        if root_weight:
            # Self-loop / residual transform applied to a node's own features.
            self.root = nn.Parameter(torch.empty(in_channels, out_channels))
        else:
            self.register_parameter("root", None)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.num_bases * self.message_channels
        nn.init.xavier_uniform_(self.basis)
        uniform_fan_in_(self.root, fan_in)
        uniform_fan_in_(self.bias, fan_in)

    def forward(self, x, edge_index, mode_shares, edge_norm=None):
        """Propagate ``x`` over ``edge_index`` weighted by ``mode_shares``.

        ``mode_shares`` is ``[num_edges, num_relations]`` and continuous, not a
        relation-id vector.
        """
        basis_coefficients = self.mode_projection(mode_shares)  # [E, num_bases]
        message_weight = torch.einsum(
            "eb,bio->eio", basis_coefficients, self.basis
        )  # [E, 2*in_channels, out_channels]
        return self.propagate(
            edge_index, x=x, message_weight=message_weight, edge_norm=edge_norm
        )

    def message(self, x_j, x_i, message_weight, edge_norm):
        # [E, 2*in] batch-multiplied by [E, 2*in, out] -> [E, out]
        endpoints = torch.cat([x_j, x_i], dim=1)
        message = torch.bmm(endpoints.unsqueeze(1), message_weight).squeeze(1)
        if edge_norm is not None:
            message = message * edge_norm.view(-1, 1)
        return message

    def update(self, aggr_out, x):
        """Add the residual root transform and the bias."""
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {}, num_relations={}, num_bases={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.num_relations,
            self.num_bases,
        )
