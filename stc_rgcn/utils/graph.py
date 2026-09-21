"""Graph construction and subgraph sampling.

Two graphs are built from a :class:`~stc_rgcn.data.Split`:

``build_full_graph``
    the whole training split, used as the encoder input when scoring the
    validation and test splits;
``sample_subgraph``
    a fresh random subgraph per training step.  Only a fraction of the sampled
    edges carries the message passing; those same edges are the supervision
    targets.

Both attach an explicit ``target_index``.  Supervision never reads
``edge_index`` directly, because the discrete-relation variants make their
message-passing graph bidirectional while OD targets stay directional.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from ..config import Variant

#: Guards the division when a node has no incoming relation weight at all.
DEGREE_EPSILON = 1e-6


def _scatter_add(source, index, num_rows):
    """Sum ``source`` rows into ``num_rows`` buckets given by ``index``.

    Equivalent to ``torch_scatter.scatter_add(..., dim=0, dim_size=num_rows)``,
    written with ``index_add_`` so the package does not need ``torch_scatter``
    (whose wheels must match the exact torch/CUDA build).
    """
    shape = (num_rows,) + tuple(source.shape[1:])
    out = torch.zeros(shape, dtype=source.dtype, device=source.device)
    return out.index_add_(0, index.to(torch.long), source)


def discrete_edge_norm(edge_type, edge_index, num_nodes, num_relations):
    """Per-edge ``1 / deg(source, relation)`` for discrete relation ids.

    This is the normalization from the original R-GCN paper: each edge is
    scaled by how many edges of the same relation leave its source node.
    """
    edge_type = edge_type.to(torch.long)
    one_hot = torch.nn.functional.one_hot(edge_type, num_classes=num_relations).to(torch.float)
    degree = _scatter_add(one_hot, edge_index[0], num_nodes)
    edge_offsets = torch.arange(edge_index.shape[1], device=edge_type.device) * num_relations
    flat_index = edge_type + edge_offsets
    return 1.0 / degree[edge_index[0]].view(-1)[flat_index]


def weighted_edge_norm(mode_shares, edge_index, num_nodes):
    """Per-edge inverse weighted in-degree for continuous relation weights.

    Each edge contributes the sum of its 4 transport-mode shares to its
    destination; every edge is then scaled by the inverse of that total.
    """
    edge_weight = mode_shares.sum(dim=1)
    destination = edge_index[1]
    degree = _scatter_add(edge_weight, destination, num_nodes)
    return 1.0 / (degree[destination] + DEGREE_EPSILON)


def _as_tensor(array, dtype):
    return None if array is None else torch.as_tensor(array, dtype=dtype)


def build_full_graph(split, variant, num_entities, num_relations, node_features=None):
    """Build the encoder input covering every edge of ``split``.

    The result carries structure only: prediction targets come from the split
    being scored, not from this graph.
    """
    variant = Variant(variant)
    od_pairs = split.od_pairs

    if variant.uses_discrete_relations:
        origin, relation, destination = od_pairs.transpose()
        edge_index = torch.tensor(np.stack([origin, destination]), dtype=torch.long)
        edge_type = torch.tensor(relation, dtype=torch.long)
        edge_norm = discrete_edge_norm(edge_type, edge_index, num_entities, num_relations)
    else:
        origin, destination = od_pairs.transpose()
        edge_index = torch.tensor(np.stack([origin, destination]), dtype=torch.long)
        edge_type = torch.as_tensor(split.mode_shares, dtype=torch.float32)
        edge_norm = weighted_edge_norm(edge_type, edge_index, num_entities)

    data = Data(edge_index=edge_index, num_nodes=num_entities)
    data.entity = torch.arange(num_entities, dtype=torch.long)
    data.edge_type = edge_type
    data.edge_norm = edge_norm
    data.target_index = edge_index
    if variant.uses_node_features:
        data.x = node_features
    return data


def sample_subgraph(
    split,
    variant,
    num_entities,
    num_relations,
    sample_size,
    graph_split_ratio,
    node_features=None,
    generator=None,
):
    """Draw one training subgraph together with its supervision targets.

    ``sample_size`` edges are drawn from ``split``; a ``graph_split_ratio``
    fraction of them becomes the subgraph, and that same fraction is what the
    step is supervised on.

    Parameters
    ----------
    generator
        Optional :class:`numpy.random.Generator` for reproducible sampling.
    """
    variant = Variant(variant)
    rng = generator if generator is not None else np.random
    num_edges = len(split)
    if num_edges == 0:
        raise ValueError("cannot sample from an empty split")

    replace = num_edges < sample_size
    sampled = rng.choice(num_edges, sample_size, replace=replace)

    kept = max(1, int(sample_size * graph_split_ratio))
    sampled = sampled[rng.choice(sample_size, size=kept, replace=False)]

    od_pairs = split.od_pairs[sampled]
    if variant.uses_discrete_relations:
        origin, relation, destination = od_pairs.transpose()
    else:
        origin, destination = od_pairs.transpose()
        relation = None

    # Relabel the sampled entities into a compact 0..M-1 range so the subgraph
    # only allocates embeddings for the nodes it actually touches.
    unique_entities, relabelled = np.unique(
        np.concatenate([origin, destination]), return_inverse=True
    )
    local_origin, local_destination = np.reshape(relabelled, (2, -1))
    num_local = len(unique_entities)

    target_index = torch.tensor(
        np.stack([local_origin, local_destination]), dtype=torch.long
    )

    if variant.uses_discrete_relations:
        # Message passing runs on the undirected graph, but the targets stay
        # directional, so edge_index and target_index deliberately differ.
        edge_index = torch.tensor(
            np.stack(
                [
                    np.concatenate([local_origin, local_destination]),
                    np.concatenate([local_destination, local_origin]),
                ]
            ),
            dtype=torch.long,
        )
        edge_type = torch.tensor(np.concatenate([relation, relation]), dtype=torch.long)
        edge_norm = discrete_edge_norm(edge_type, edge_index, num_local, num_relations)
    else:
        edge_index = target_index
        edge_type = torch.as_tensor(split.mode_shares[sampled], dtype=torch.float32)
        edge_norm = weighted_edge_norm(edge_type, edge_index, num_local)

    data = Data(edge_index=edge_index, num_nodes=num_local)
    data.entity = torch.as_tensor(unique_entities, dtype=torch.long)
    data.edge_type = edge_type
    data.edge_norm = edge_norm
    data.target_index = target_index

    if variant.uses_node_features:
        if node_features is None:
            raise ValueError("variant {} needs node features, got None".format(variant))
        if node_features.shape[0] < num_entities:
            raise ValueError(
                "node features have {} rows, fewer than the {} entities".format(
                    node_features.shape[0], num_entities
                )
            )
        data.x = node_features[unique_entities]

    if split.purpose_probs is not None:
        data.target_purpose_probs = _as_tensor(split.purpose_probs[sampled], torch.float32)
    if split.flows is not None:
        data.target_flows = _as_tensor(split.flows[sampled], torch.float32)

    return data
