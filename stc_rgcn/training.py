"""Training step, validation and test-time prediction.

Keeping these out of the CLI module makes a run scriptable: build a model, load
a :class:`~stc_rgcn.data.OdDataset` and call :func:`train_step` in your own
loop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .config import Variant
from .utils.graph import sample_subgraph
from .utils.losses import flow_l1_loss, purpose_kl_loss, tweedie_loss
from .utils.schema import PREDICTION_COLUMNS, PURPOSE_PROB_SLICE


@dataclass
class StepLosses:
    """The losses of one training step, before any GradNorm reweighting."""

    total: torch.Tensor
    penalty: torch.Tensor
    purpose: Optional[torch.Tensor] = None
    flow: Optional[torch.Tensor] = None

    def task_losses(self, device):
        """The two task losses, with a zero stand-in for a missing head."""
        zero = torch.zeros((), device=device)
        return [
            self.purpose if self.purpose is not None else zero,
            self.flow if self.flow is not None else zero,
        ]


def split_endpoints(split, variant, device=None):
    """Return ``(origin_idx, destination_idx)`` tensors for a split.

    Discrete-relation variants store ``(origin, relation, destination)``
    triples, so the destination is the third column rather than the second.
    """
    od_pairs = torch.as_tensor(split.od_pairs, dtype=torch.long)
    destination_column = 2 if Variant(variant).uses_discrete_relations else 1
    origin = od_pairs[:, 0]
    destination = od_pairs[:, destination_column]
    if device is not None:
        origin, destination = origin.to(device), destination.to(device)
    return origin, destination


def train_step(
    model,
    split,
    num_entities,
    num_relations,
    sample_size,
    graph_split_ratio,
    reg_weight,
    node_features=None,
    tweedie_power=1.5,
    device=None,
    generator=None,
):
    """Sample a subgraph, run both heads and sum the enabled losses."""
    variant = model.variant
    device = device or next(model.parameters()).device

    batch = sample_subgraph(
        split=split,
        variant=variant,
        num_entities=num_entities,
        num_relations=num_relations,
        sample_size=sample_size,
        graph_split_ratio=graph_split_ratio,
        node_features=node_features,
        generator=generator,
    ).to(device)

    node_embedding = model(batch)
    origin_idx, destination_idx = batch.target_index
    mode_shares = batch.edge_type if variant.uses_mode_weights else None

    purpose_loss = None
    flow_loss = None

    if variant.has_purpose_head and getattr(batch, "target_purpose_probs", None) is not None:
        pred_probs = model.predict_purpose_probs(
            node_embedding, origin_idx, destination_idx, mode_shares
        )
        true_probs = _normalized_targets(batch.target_purpose_probs)
        purpose_loss = purpose_kl_loss(pred_probs, true_probs)

    if variant.has_flow_head and getattr(batch, "target_flows", None) is not None:
        pred_flows = model.predict_flow(
            node_embedding, origin_idx, destination_idx, mode_shares
        )
        true_flows = batch.target_flows
        flow_loss = (
            tweedie_loss(pred_flows, true_flows, tweedie_power)
            if variant.uses_tweedie_loss
            else flow_l1_loss(pred_flows, true_flows)
        )

    penalty = model.parameter_penalty()
    total = reg_weight * penalty
    if purpose_loss is not None:
        total = total + purpose_loss
    if flow_loss is not None:
        total = total + flow_loss

    return StepLosses(total=total, penalty=penalty, purpose=purpose_loss, flow=flow_loss)


def _normalized_targets(probs, epsilon=1e-10):
    """Renormalize target distributions so the KL divergence is well defined.

    Split files store probabilities rounded to 3 decimals, so rows routinely
    miss 1 by a few thousandths.  The loader has already rejected anything
    beyond :data:`stc_rgcn.data.PROB_SUM_TOLERANCE`, so rescaling here is
    always a rounding correction and is done silently.
    """
    probs = probs + epsilon
    return probs / probs.sum(dim=1, keepdim=True)


@torch.no_grad()
def predict(model, graph, split, device=None):
    """Run both heads over ``split`` using ``graph`` as the encoder input.

    Returns
    -------
    tuple
        ``(pred_probs, pred_flows)``, each ``None`` when the variant has no
        such head.
    """
    variant = model.variant
    device = device or next(model.parameters()).device

    node_embedding = model(graph.to(device))
    origin_idx, destination_idx = split_endpoints(split, variant, device)
    mode_shares = (
        torch.as_tensor(split.mode_shares, dtype=torch.float32, device=device)
        if variant.uses_mode_weights and split.mode_shares is not None
        else None
    )

    pred_probs = (
        model.predict_purpose_probs(node_embedding, origin_idx, destination_idx, mode_shares)
        if variant.has_purpose_head
        else None
    )
    pred_flows = (
        model.predict_flow(node_embedding, origin_idx, destination_idx, mode_shares)
        if variant.has_flow_head
        else None
    )
    return pred_probs, pred_flows


@torch.no_grad()
def evaluate(model, graph, split, tweedie_power=1.5, device=None):
    """Validation losses for ``split``: KL, MAE and, for Tweedie runs, its loss."""
    device = device or next(model.parameters()).device
    pred_probs, pred_flows = predict(model, graph, split, device)
    results = {}

    if pred_probs is not None and split.purpose_probs is not None:
        true_probs = torch.as_tensor(split.purpose_probs, dtype=torch.float32, device=device)
        results["kl_loss"] = purpose_kl_loss(pred_probs, true_probs).item()

    if pred_flows is not None and split.flows is not None:
        true_flows = torch.as_tensor(split.flows, dtype=torch.float32, device=device)
        results["mae"] = flow_l1_loss(pred_flows, true_flows).item()
        if model.variant.uses_tweedie_loss:
            results["tweedie_loss"] = tweedie_loss(pred_flows, true_flows, tweedie_power).item()

    return results


def write_predictions(path, split, variant, pred_probs=None, pred_flows=None, id_to_entity=None):
    """Write predictions in the shared 34-column layout.

    Every model and baseline writes these columns, which is what lets
    ``stc-rgcn-eval score`` read a whole directory without knowing which model
    produced what.  Columns belonging to a head the variant does not have are
    left empty.
    """
    variant = Variant(variant)
    origin_idx, destination_idx = split_endpoints(split, variant)
    origin_idx = origin_idx.numpy()
    destination_idx = destination_idx.numpy()

    def name(entity_id):
        return id_to_entity[entity_id] if id_to_entity else str(entity_id)

    pred_probs = _to_numpy(pred_probs)
    pred_flows = _to_numpy(pred_flows)
    true_probs = split.purpose_probs
    true_flows = split.flows

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\t".join(PREDICTION_COLUMNS) + "\n")
        num_purposes = PURPOSE_PROB_SLICE.stop - PURPOSE_PROB_SLICE.start
        blanks = [""] * num_purposes

        for row in range(len(origin_idx)):
            fields = [name(origin_idx[row]), name(destination_idx[row])]
            fields += _format_row(pred_probs, row, blanks)
            fields += _format_row(true_probs, row, blanks)
            fields.append(_format_scalar(pred_flows, row))
            fields.append(_format_scalar(true_flows, row))
            handle.write("\t".join(fields) + "\n")

    return path


def _to_numpy(values):
    if values is None:
        return None
    return values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)


def _format_row(matrix, row, blanks):
    if matrix is None:
        return list(blanks)
    return ["{:.6e}".format(value) for value in matrix[row]]


def _format_scalar(values, row):
    if values is None:
        return ""
    return "{:.6f}".format(float(values[row]))
