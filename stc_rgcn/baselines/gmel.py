"""
GMEL baseline: multi-task OD flow volume + trip-purpose distribution prediction.

Reference:
    Liu et al., "Learning Geo-Contextual Embeddings for Commuting Flow
    Prediction", AAAI 2020.
    Upstream implementation: https://github.com/jackmiemie/GMEL

This module consolidates the project's adapted multi-file version (previously
baselines/gmel/code/) into a single self-contained script. Compared to
upstream GMEL, the original NYC data pipeline was replaced with the project
adapters in baselines/utils/data.py, and the GBRT fine-tuning stage was
removed: the trained GAT predicts flow directly, and a single softmax FFN head
on the frozen GAT embeddings predicts the 15-dim purpose distribution.

Shared concerns live in baselines/utils: dataset reading, the standard
34-column result layout, and the distribution metrics (js_divergence,
cosine_similarity, per_purpose_mae, evaluate_distribution) and row
normalization (flows_to_probs) that used to be duplicated here.

Pipeline (run by main()):
    1. train the two-branch GAT on total flow (multitask: edge volume +
       in-flow + out-flow),
    2. predict test-set total flow directly with the trained GAT,
    3. predict the 15-dim purpose distribution with the softmax FFN head,
    4. merge both into the standard 34-column result file.

Intermediates (checkpoint, npz embeddings, logs) are written under
<output-dir>/gmel_work/; the merged result goes to <output-dir>/gmel_result.txt.
"""

import argparse
import csv
import logging
import os
import sys

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

from ..config import DEFAULT_BASELINE_DIR, DEFAULT_DATA_DIR
from .common import (
    NUM_PURPOSES,
    PREDICTION_COLUMNS,
    PURPOSE_CODES,
    evaluate_distribution,
    flows_to_probs,
    load_baseline_dataset,
    load_entity_dict,
    load_node_features,
    resolve_distance_matrix,
)

# Tensorboard is optional: fall back to a no-op shim when unavailable.
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        """No-op replacement for torch.utils.tensorboard.SummaryWriter."""

        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self, *args, **kwargs):
            pass


# Standard 34-column result layout (identical to baselines.common.PREDICTION_COLUMNS).
OUT_COLS = list(PREDICTION_COLUMNS)

# Fixed hyperparameters (not exposed on the CLI).
MULTITASK_WEIGHTS = (0.5, 0.25, 0.25)
REG_PARAM = 0
GRAD_NORM = 1.0
EVALUATE_EVERY = 5
FFN_HIDDEN_DIM = 128   # purpose head hidden width
FFN_DROPOUT = 0.1
FFN_EPOCHS = 100
FFN_LR = 1e-3
FFN_BATCH_SIZE = 512

_DEFAULT_DATADIR = DEFAULT_DATA_DIR
_DEFAULT_OUTPUT_DIR = DEFAULT_BASELINE_DIR


# ---------------------------------------------------------------------------
# Data loading (project adapter, replaces upstream utils.load_dataset)
# ---------------------------------------------------------------------------

def _build_adj_matrix_from_od(train_ids, train_flows, num_nodes):
    """Build a sparse symmetric adjacency from training OD flows.

    Two zones are treated as geographic neighbors when a flow exists between
    them; edge weight = max(flow(i->j), flow(j->i)), min-max scaled to [0, 1].
    The diagonal holds explicit zero-weight entries, which become zero-weight
    self-loop edges in the DGL graph built from this matrix.
    """
    o = train_ids[:, 0].astype(int)
    d = train_ids[:, 1].astype(int)
    f = train_flows.astype(np.float32)

    sparse_mat = sparse.coo_matrix((f, (o, d)),
                                   shape=(num_nodes, num_nodes), dtype=np.float32)
    # Symmetrize by taking the max of both directions.
    sparse_mat = sparse_mat.maximum(sparse_mat.T)
    sparse_mat.setdiag(0)

    sparse_coo = sparse_mat.tocoo()
    src_nodes = sparse_coo.row
    dst_nodes = sparse_coo.col
    edge_weights = sparse_coo.data
    if len(edge_weights) > 0:
        max_weight = edge_weights.max()
        if max_weight > 0:
            edge_weights = edge_weights / max_weight

    return src_nodes, dst_nodes, edge_weights


def _build_flow_arrays(train_ids, train_flows, num_nodes):
    """Per-node in/out flow totals from the training OD matrix, shape (num_nodes, 1)."""
    inflow = np.zeros(num_nodes, dtype=np.float32)
    outflow = np.zeros(num_nodes, dtype=np.float32)

    o = train_ids[:, 0].astype(int)
    d = train_ids[:, 1].astype(int)
    f = train_flows.astype(np.float32)

    np.add.at(outflow, o, f)
    np.add.at(inflow, d, f)

    return inflow.reshape(-1, 1), outflow.reshape(-1, 1)


def _zscore(arr):
    """Column-wise z-score; near-zero std is treated as 1."""
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (arr - mean) / std


def load_od_dataset(data_path):
    """Load project data in the dict format expected by the GMEL training code.

    Returns a dict with keys:
        train/valid/test        : (N, 3) float32 [src, dst, total_flow]
        train_inflow/outflow    : (num_nodes, 1) float32
        node_feats              : (num_nodes, feat_dim) float32, z-scored
        ct_adjacency_sparse     : dict with src_nodes/dst_nodes/edge_weights
        ct_adjacency_withweight : None (dense matrix intentionally not built)
        num_nodes               : int
        distm                   : (num_nodes, num_nodes) float32 or None
        purpose_flows           : {'train'|'valid'|'test': (N, 15) float32}
    """
    raw = load_baseline_dataset(data_path)

    entity2id = raw["entity2id"]
    dist_dict = raw["dist_dict"]
    num_nodes = len(entity2id)

    def to_triplets(split_tuple):
        ids, _prob, flows, _pflows = split_tuple
        # Preallocated float32 array so OD ids and flow share one dtype.
        triplets = np.empty((ids.shape[0], 3), dtype=np.float32)
        triplets[:, 0] = ids[:, 0]
        triplets[:, 1] = ids[:, 1]
        triplets[:, 2] = flows.astype(np.float32)
        return triplets

    train_data = to_triplets(raw["train"])
    valid_data = to_triplets(raw["valid"])
    test_data = to_triplets(raw["test"])

    train_ids = raw["train"][0]       # (N, 2) int
    train_flows = raw["train"][2]     # (N,) float32
    train_inflow, train_outflow = _build_flow_arrays(train_ids, train_flows, num_nodes)

    node_feats_raw = load_node_features(data_path, num_nodes)
    if node_feats_raw is not None:
        node_feats = _zscore(node_feats_raw.astype(np.float32))
        print(f"[my_data_loader] node features: {node_feats.shape} (from features.txt)")
    else:
        # Fallback: 3-dim OD statistics features when the feature file is absent.
        out_f = train_outflow.ravel()
        in_f = train_inflow.ravel()
        deg = np.zeros(num_nodes, dtype=np.float32)
        np.add.at(deg, train_ids[:, 0].astype(int), 1.0)
        feat_raw = np.stack([np.log1p(out_f), np.log1p(in_f), np.log1p(deg)], axis=1)
        node_feats = _zscore(feat_raw)
        print(f"[my_data_loader] node features: {node_feats.shape} (OD-statistics fallback: feature file not found)")

    ct_src_nodes, ct_dst_nodes, ct_edge_weights = _build_adj_matrix_from_od(
        train_ids, train_flows, num_nodes)
    print(f"[my_data_loader] adjacency edges: {len(ct_src_nodes)}, "
          f"sparsity: {len(ct_src_nodes)/(num_nodes*num_nodes)*100:.6f}%")

    # Distances come from grid_distance.csv when present, otherwise from the
    # coordinates embedded in the entity names (see utils.spatial).
    distm = resolve_distance_matrix(raw["id2entity"], dist_dict,
                                    num_nodes).astype(np.float32)
    print(f"[my_data_loader] distance matrix: {distm.shape}, "
          f"median OD distance {np.median(distm):.2f} km")

    data = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data,
        "train_inflow": train_inflow,
        "train_outflow": train_outflow,
        "node_feats": node_feats,
        "ct_adjacency_sparse": {
            "src_nodes": ct_src_nodes,
            "dst_nodes": ct_dst_nodes,
            "edge_weights": ct_edge_weights,
        },
        "ct_adjacency_withweight": None,
        "num_nodes": num_nodes,
        "distm": distm,
        # Raw 15-dim purpose flows per split, used by the purpose head.
        "purpose_flows": {
            "train": raw["train"][3],
            "valid": raw["valid"][3],
            "test": raw["test"][3],
        },
    }

    print("\n[my_data_loader] dataset loaded:")
    print(f"  num_nodes : {num_nodes}")
    print(f"  train     : {train_data.shape}")
    print(f"  valid     : {valid_data.shape}")
    print(f"  test      : {test_data.shape}")
    print(f"  node_feats: {node_feats.shape}")

    return data


# ---------------------------------------------------------------------------
# Graph construction and training utilities
# ---------------------------------------------------------------------------

def build_graph_from_matrix(adjm, node_feats, device='cpu'):
    """Build a DGL graph from a dense adjacency or a sparse edge dict.

    adjm is either a dense (num_nodes, num_nodes) matrix or a dict with
    keys src_nodes/dst_nodes/edge_weights.
    """
    if isinstance(adjm, dict) and "src_nodes" in adjm and "dst_nodes" in adjm:
        src_nodes = adjm["src_nodes"]
        dst_nodes = adjm["dst_nodes"]
        edge_weights = adjm.get("edge_weights", np.ones_like(src_nodes, dtype=np.float32))
        num_nodes = node_feats.shape[0]

        print(f"[build_graph_from_matrix] sparse graph: {len(src_nodes)} edges, {num_nodes} nodes")

        g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
        g.edata['d'] = torch.tensor(edge_weights).float().view(-1, 1)
    else:
        dst, src = adjm.nonzero()
        d = adjm[adjm.nonzero()]
        num_nodes = adjm.shape[0]

        print(f"[build_graph_from_matrix] dense matrix: {len(src)} edges, {num_nodes} nodes")

        g = dgl.graph((src, dst), num_nodes=num_nodes)
        g.edata['d'] = torch.tensor(d).float().view(-1, 1)

    # Node attributes are the geographic features of each zone.
    g = g.to(device)
    g.ndata['attr'] = torch.from_numpy(node_feats).to(device)
    norm = comp_deg_norm(g)
    g.ndata['norm'] = torch.tensor(norm).float().view(-1, 1).to(device)
    return g


def comp_deg_norm(g):
    """Degree normalization factor 1/in_degree (0 for isolated nodes)."""
    # Move to CPU before numpy conversion; in_degrees may be a GPU tensor.
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().cpu().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm


def mini_batch_gen(train_data, mini_batch_size):
    """Yield shuffled mini-batches of trip samples."""
    samples = train_data[torch.randperm(train_data.shape[0])]
    for i in range(0, samples.shape[0], mini_batch_size):
        yield samples[i:i + mini_batch_size]


def evaluate(model, g, trip_od, trip_volume):
    """RMSE/MAE/MAPE/CPC/CPL on the given OD pairs (predictions scaled back)."""
    with torch.no_grad():
        src_embedding = model(g)
        dst_embedding = model.forward2(g)
        scaled_prediction = model.predict_edge(src_embedding, dst_embedding, trip_od)
        prediction = inverse_sqrt_scale(scaled_prediction)
        y = trip_volume.float().view(-1, 1)
        rmse = rmse_fn(prediction, y)
        mae = mae_fn(prediction, y)
        mape = mape_fn(prediction, y)
        cpc = cpc_fn(prediction, y)
        cpl = cpl_fn(prediction, y)
    return rmse.item(), mae.item(), mape.item(), cpc.item(), cpl.item()


def sqrt_scale(y):
    """Square-root scaling of the flow target, which compresses its heavy tail."""
    return torch.sqrt(y)


def inverse_sqrt_scale(scaled_y):
    """Invert sqrt_scale() (square)."""
    return scaled_y ** 2


def rmse_fn(y_hat, y):
    """Root Mean Square Error metric."""
    return torch.sqrt(torch.mean((y_hat - y) ** 2))


def mae_fn(y_hat, y):
    """Mean Absolute Error metric."""
    return torch.mean(torch.abs(y_hat - y))


def mape_fn(y_hat, y):
    """Mean Absolute Percentage Error metric."""
    return torch.mean(torch.abs(y_hat - y) / y)


def cpc_fn(y_hat, y):
    """Common Part of Commuters metric."""
    return 2 * torch.sum(torch.min(y_hat, y)) / (torch.sum(y_hat) + torch.sum(y))


def cpl_fn(y_hat, y):
    """Common Part of Links metric (topology overlap)."""
    yy_hat = y_hat > 0
    yy = y > 0
    return 2 * torch.sum(yy_hat * yy) / (torch.sum(yy_hat) + torch.sum(yy))


# ---------------------------------------------------------------------------
# GAT layers (dgl)
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """GAT hidden layer with edge features and self/neighbor convex combination."""

    def __init__(self, g, in_ndim, out_ndim, in_edim=1, out_edim=1):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc0 = nn.Linear(in_edim, out_edim, bias=False)
        self.fc1 = nn.Linear(in_ndim, out_ndim, bias=False)
        self.fc2 = nn.Linear(in_ndim, out_ndim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_ndim + out_edim, 1, bias=False)
        # equation (4)
        self.activation = F.relu
        # convex combination weights between self and neighbor aggregation
        self.weights = nn.Parameter(torch.Tensor(2, 1))
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

    def edge_feat_func(self, edges):
        """Transform edge features."""
        return {'t': self.fc0(edges.data['d'])}

    def edge_attention(self, edges):
        """Edge UDF for equation (2)."""
        z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['t']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        """Message UDF for equations (3) & (4)."""
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        """Reduce UDF for equations (3) & (4); core node update."""
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        z_neighbor = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        z_i = nodes.data['z_i']
        lambda_ = F.softmax(self.weights, dim=0)
        h = self.activation(z_i + z_neighbor)
        return {'h': h}

    def forward(self, h):
        self.g.apply_edges(self.edge_feat_func)
        z = self.fc1(h)
        self.g.ndata['z'] = z   # message passed to the others
        z_i = self.fc2(h)
        self.g.ndata['z_i'] = z_i   # message passed to self
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class GATInputLayer(nn.Module):
    """Input GAT layer: same attention scheme, operating on raw node features."""

    def __init__(self, g, in_ndim, out_ndim, in_edim=1, out_edim=1):
        super().__init__()
        self.g = g
        self.fc0 = nn.Linear(in_edim, out_edim, bias=False)
        self.fc1 = nn.Linear(in_ndim, out_ndim, bias=False)
        self.fc2 = nn.Linear(in_ndim, out_ndim, bias=False)
        self.attn_fc = nn.Linear(2 * out_ndim + out_edim, 1, bias=False)
        self.activation = F.relu
        self.weights = nn.Parameter(torch.Tensor(2, 1))
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

    def edge_feat_func(self, edges):
        """Transform edge features."""
        return {'t': self.fc0(edges.data['d'])}

    def edge_attention(self, edges):
        """Edge UDF for equation (2)."""
        z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['t']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        """Message UDF for equations (3) & (4)."""
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        """Reduce UDF for equations (3) & (4); core node update."""
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        z_neighbor = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        z_i = nodes.data['z_i']
        lambda_ = F.softmax(self.weights, dim=0)
        h = self.activation(z_i + z_neighbor)
        return {'h': h}

    def forward(self, attr):
        self.g.apply_edges(self.edge_feat_func)
        z = self.fc1(attr)
        self.g.ndata['z'] = z
        z_i = self.fc2(attr)
        self.g.ndata['z_i'] = z_i
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GmelNet(nn.Module):
    """Two-branch GAT with multitask heads (edge volume, in-flow, out-flow).

    Use forward()/forward2() for origin/destination embeddings and get_loss()
    to obtain the overall training loss.
    """

    def __init__(self, g, num_nodes, in_dim, h_dim, num_hidden_layers=1, dropout=0, device='cpu', reg_param=0):
        super().__init__()
        self.reg_param = reg_param
        self.gat = GAT(g, num_nodes, in_dim, h_dim, h_dim, num_hidden_layers, dropout, device)   # GAT for origin nodes
        self.gat2 = GAT(g, num_nodes, in_dim, h_dim, h_dim, num_hidden_layers, dropout, device)  # GAT for destination nodes
        self.edge_regressor = nn.Bilinear(h_dim, h_dim, 1)
        self.in_regressor = nn.Linear(h_dim, 1)
        self.out_regressor = nn.Linear(h_dim, 1)

    def forward(self, g):
        """Propagate the graph to get embeddings for origin nodes."""
        return self.gat.forward(g)

    def forward2(self, g):
        """Propagate the graph to get embeddings for destination nodes."""
        return self.gat2.forward(g)

    def get_loss(self, trip_od, scaled_trip_volume, in_flows, out_flows, g, multitask_weights=[0.5, 0.25, 0.25]):
        """Overall multitask loss on a batch of OD trips."""
        trip_volume = inverse_sqrt_scale(scaled_trip_volume)
        # nodes appearing in this batch
        out_nodes, _out_flows_idx = torch.unique(trip_od[:, 0], return_inverse=True)
        in_nodes, _in_flows_idx = torch.unique(trip_od[:, 1], return_inverse=True)
        scaled_out_flows = sqrt_scale(out_flows[out_nodes])
        scaled_in_flows = sqrt_scale(in_flows[in_nodes])
        # embeddings and predictions
        src_embedding = self.forward(g)
        dst_embedding = self.forward2(g)
        edge_prediction = self.predict_edge(src_embedding, dst_embedding, trip_od)
        in_flow_prediction = self.predict_inflow(dst_embedding, in_nodes)
        out_flow_prediction = self.predict_outflow(src_embedding, out_nodes)
        # task losses
        edge_predict_loss = mse(edge_prediction, scaled_trip_volume)
        in_predict_loss = mse(in_flow_prediction, scaled_in_flows)
        out_predict_loss = mse(out_flow_prediction, scaled_out_flows)
        reg_loss = 0.5 * (self.regularization_loss(src_embedding) + self.regularization_loss(dst_embedding))
        return multitask_weights[0] * edge_predict_loss + multitask_weights[1] * in_predict_loss \
            + multitask_weights[2] * out_predict_loss + self.reg_param * reg_loss

    def predict_edge(self, src_embedding, dst_embedding, trip_od):
        """Predict trip volume for the given OD pairs from node embeddings."""
        src_emb = src_embedding[trip_od[:, 0]]
        dst_emb = dst_embedding[trip_od[:, 1]]
        return self.edge_regressor(src_emb, dst_emb)

    def predict_inflow(self, embedding, in_nodes_idx):
        return self.in_regressor(embedding[in_nodes_idx])

    def predict_outflow(self, embedding, out_nodes_idx):
        return self.out_regressor(embedding[out_nodes_idx])

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2))


class GAT(nn.Module):
    """Stack of GAT input/hidden layers producing node embeddings."""

    def __init__(self, g, num_nodes, in_dim, h_dim, out_dim, num_hidden_layers=1, dropout=0, device='cpu'):
        super().__init__()
        self.g = g
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.device = device
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return GATInputLayer(self.g, self.in_dim, self.h_dim)

    def build_hidden_layer(self, idx):
        return GATLayer(self.g, self.h_dim, self.h_dim)

    def build_output_layer(self):
        return None

    def forward(self, g):
        h = g.ndata['attr']
        for layer in self.layers:
            h = layer(h)
        return h


def mse(y_hat, y):
    """Chunked mean squared error to bound memory for large batches."""
    limit = 20000
    if y_hat.shape[0] < limit:
        return torch.mean((y_hat - y) ** 2)
    else:
        acc_sqe_sum = 0  # accumulative squared error sum
        for i in range(0, y_hat.shape[0], limit):
            acc_sqe_sum += torch.sum((y_hat[i: i + limit] - y[i: i + limit]) ** 2)
        return acc_sqe_sum / y_hat.shape[0]


# ---------------------------------------------------------------------------
# Purpose-distribution softmax head
# ---------------------------------------------------------------------------

def build_features(triplets, src_emb, dst_emb, distm):
    """Concatenate src_emb + dst_emb + scaled_dist as FFN input features."""
    idx_src = triplets[:, 0].astype(np.int64)
    idx_dst = triplets[:, 1].astype(np.int64)
    feat_src = src_emb[idx_src]
    feat_dst = dst_emb[idx_dst]
    feat_dist = distm[idx_src, idx_dst].reshape(-1, 1)
    return np.concatenate([feat_src, feat_dst, feat_dist], axis=1).astype(np.float32)


class PurposeSoftmaxNet(nn.Module):
    """FFN with softmax output over the 15 purpose probabilities.

    Input dimension: 2 * embedding_size + 1.
    """

    def __init__(self, in_dim, hidden_dim=128, num_purposes=15, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_purposes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)   # (batch, 15), rows sum to 1


def _log_metrics(logger, split, metrics):
    logger.info(
        f"  [{split}] JSD={metrics['JSD_mean']:.4f} | "
        f"Cosine={metrics['Cosine_mean']:.4f} | "
        f"overall_MAE={metrics['overall_MAE']:.4f}"
    )
    mae_str = " | ".join(
        f"{PURPOSE_CODES[i]}:{v:.4f}"
        for i, v in enumerate(metrics["per_purpose_MAE"])
    )
    logger.info(f"  [{split}] per_purpose_MAE: {mae_str}")


def _write_metrics_csv(path, metrics_test, metrics_valid):
    keys = ["JSD_mean", "JSD_median", "Cosine_mean", "overall_MAE"]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["metric", "test", "valid"])
        for k in keys:
            w.writerow([k,
                        f"{metrics_test[k]:.6f}",
                        f"{metrics_valid[k]:.6f}"])


def _write_purpose_intermediate(test_ids, pred_probs, true_probs, filename):
    """Save the purpose-head predictions in GMEL's intermediate layout.

    Merged with the flow predictions into the shared 34-column result file by
    :func:`merge_flow_purpose`; not the final output.
    """
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("origin\tdest\tpred_probs\ttrue_probs\n")
        for i in range(len(test_ids)):
            pred_p = "\t".join([f"{x:.6e}" for x in pred_probs[i]])
            true_p = "\t".join([f"{x:.6e}" for x in true_probs[i]])
            f.write(f"{test_ids[i,0]}\t{test_ids[i,1]}\t{pred_p}\t{true_p}\n")
    print(f"results saved: {filename}")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def train_gat(config, data, work_dir):
    """Train the two-branch GAT on total flow; returns the best checkpoint path."""
    models_dir = os.path.join(work_dir, 'models')
    embeddings_dir = os.path.join(work_dir, 'embeddings')
    log_dir = os.path.join(work_dir, 'log')
    device = torch.device(config['device'])

    writer = SummaryWriter(
        log_dir=os.path.join(work_dir, 'tensorboard'),
        comment='#layers{}_emb{}_multitask{}'.format(config['num_hidden_layers'],
                                                     config['embedding_size'],
                                                     config['multitask_weights']))
    logger = logging.getLogger('#layers{}_emb{}_multitask{}'.format(config['num_hidden_layers'],
                                                                    config['embedding_size'],
                                                                    config['multitask_weights']))  # experiment name
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training_log.log'), encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    # random seed for reproducibility
    torch.manual_seed(2019)
    np.random.seed(2019)

    train_data = data['train']
    valid_data = data['valid']
    train_inflow = data['train_inflow']
    train_outflow = data['train_outflow']
    node_feats = data['node_feats']
    num_nodes = data['num_nodes']

    # census tract adjacency - prefer the sparse format
    if 'ct_adjacency_sparse' in data and data['ct_adjacency_sparse'] is not None:
        ct_adj = data['ct_adjacency_sparse']
        print(f"[train] sparse adjacency: {len(ct_adj['src_nodes'])} edges")
    elif 'ct_adjacency_withweight' in data and data['ct_adjacency_withweight'] is not None:
        ct_adj = data['ct_adjacency_withweight']
        print(f"[train] dense adjacency: {ct_adj.shape}")
    else:
        raise KeyError("no adjacency data: neither ct_adjacency_sparse nor "
                       "ct_adjacency_withweight is available")

    print(f"[train] nodes: {num_nodes}")

    train_data = torch.from_numpy(train_data)
    trip_od_train = train_data[:, :2].long().to(device)
    trip_volume_train = train_data[:, -1].float().to(device)
    trip_od_valid = torch.from_numpy(valid_data[:, :2]).long().to(device)
    trip_volume_valid = torch.from_numpy(valid_data[:, -1]).float().to(device)
    train_inflow = torch.from_numpy(train_inflow).view(-1, 1).float().to(device)
    train_outflow = torch.from_numpy(train_outflow).view(-1, 1).float().to(device)
    g = build_graph_from_matrix(ct_adj, node_feats.astype(np.float32), device)
    g.to(device)

    model = GmelNet(g, num_nodes, in_dim=node_feats.shape[1], h_dim=config['embedding_size'],
                    num_hidden_layers=config['num_hidden_layers'], dropout=0, device=device,
                    reg_param=config['reg_param'])
    model.to(device)

    model_state_file = os.path.join(models_dir, 'model_state_layers{}_emb{}_multitask{}.pth'.format(
        config['num_hidden_layers'], config['embedding_size'], config['multitask_weights']))
    best_rmse = 1e6

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)

    for epoch in range(config['max_epochs']):
        model.train()
        batch_gen = mini_batch_gen(train_data, mini_batch_size=int(config['mini_batch_size']))

        for mini_batch in batch_gen:
            optimizer.zero_grad()
            trip_od = mini_batch[:, :2].long().to(device)
            scaled_trip_volume = sqrt_scale(mini_batch[:, -1].float()).to(device)
            loss = model.get_loss(trip_od, scaled_trip_volume, train_inflow, train_outflow, g,
                                  multitask_weights=config['multitask_weights'])
            writer.add_scalar('mini_loss', loss.item(), global_step=epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm'])
            optimizer.step()
        scheduler.step()

        # train-set report (debug level)
        if logger.level == logging.DEBUG:
            model.eval()
            with torch.no_grad():
                loss = model.get_loss(trip_od_train, sqrt_scale(trip_volume_train), train_inflow, train_outflow, g)
            rmse, mae, mape, cpc, cpl = evaluate(model, g, trip_od_train, trip_volume_train)
            logger.debug("Evaluation on train dataset:")
            logger.debug("Epoch {:04d} | Loss = {:.4f}".format(epoch, loss))
            logger.debug(
                "RMSE {:.4f} | MAE {:.4f} | MAPE {:.4f} | CPC {:.4f} | CPL {:.4f} |".format(rmse, mae, mape, cpc, cpl))
            writer.add_scalar('overall-loss', loss.item(), epoch)
            writer.add_scalar('RMSE', rmse, epoch)
            writer.add_scalar('MAE', mae, epoch)
            writer.add_scalar('MAPE', mape, epoch)
            writer.add_scalar('CPC', cpc, epoch)
            writer.add_scalar('CPL', cpl, epoch)

        # validation
        if epoch % config['evaluate_every'] == 0:
            model.eval()
            with torch.no_grad():
                loss = model.get_loss(trip_od_valid, sqrt_scale(trip_volume_valid), train_inflow, train_outflow, g)
            rmse, mae, mape, cpc, cpl = evaluate(model, g, trip_od_valid, trip_volume_valid)
            logger.info("-----------------------------------------")
            logger.info("Evaluation on Validation:")
            logger.info("Epoch {:04d} | Loss = {:.4f}".format(epoch, loss))
            logger.info(
                "RMSE {:.4f} | MAE {:.4f} | MAPE {:.4f} | CPC {:.4f} | CPL {:.4f} |".format(rmse, mae, mape, cpc, cpl))
            # keep the best model and its embeddings
            if rmse < best_rmse:
                best_rmse = rmse
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'rmse': rmse, 'mae': mae, 'mape': mape,
                            'cpc': cpc, 'cpl': cpl}, model_state_file)
                src_embedding = model(g).detach().cpu().numpy()
                dst_embedding = model.forward2(g).detach().cpu().numpy()
                emb_fp = os.path.join(embeddings_dir,
                                      'censustract_embeddings_total_layers{}_emb{}_multitask{}.npz'.format(
                                          config['num_hidden_layers'], config['embedding_size'],
                                          config['multitask_weights']))
                np.savez(emb_fp, src_embedding, dst_embedding)
                logger.info('Best RMSE found on epoch {}'.format(epoch))
            logger.info("-----------------------------------------")

    return model_state_file


def gat_direct_predict(model_path, config, data, outputs_dir):
    """Predict test-set total flow with the trained GAT (no GBRT fine-tuning)."""
    device = torch.device(config['device'])

    if 'ct_adjacency_sparse' in data and data['ct_adjacency_sparse'] is not None:
        ct_adj = data['ct_adjacency_sparse']
    elif 'ct_adjacency_withweight' in data and data['ct_adjacency_withweight'] is not None:
        ct_adj = data['ct_adjacency_withweight']
    else:
        raise KeyError("no adjacency matrix data found")

    num_nodes = data['num_nodes']
    node_feats = data['node_feats'].astype(np.float32)
    g = build_graph_from_matrix(ct_adj, node_feats, device)
    g.to(device)

    model = GmelNet(
        g, num_nodes,
        in_dim=node_feats.shape[1],
        h_dim=config['embedding_size'],
        num_hidden_layers=config['num_hidden_layers'],
        dropout=0,
        device=device,
        reg_param=config['reg_param']
    )
    model.to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"[GAT Direct Predict] model loaded: {model_path}")
    print(f"[GAT Direct Predict] best RMSE: {checkpoint.get('rmse', 'N/A')}")

    test_data = data['test']
    trip_od_test = torch.from_numpy(test_data[:, :2]).long().to(device)
    trip_volume_test = torch.from_numpy(test_data[:, -1]).float().to(device)

    with torch.no_grad():
        src_embedding = model(g)
        dst_embedding = model.forward2(g)
        scaled_prediction = model.predict_edge(src_embedding, dst_embedding, trip_od_test)
        prediction = inverse_sqrt_scale(scaled_prediction)
        prediction = prediction.cpu().numpy().flatten()

    y_true = trip_volume_test.cpu().numpy()
    rmse = float(np.sqrt(np.mean((prediction - y_true) ** 2)))
    mae = float(np.mean(np.abs(prediction - y_true)))
    mape = float(np.mean(np.abs((prediction - y_true) / (y_true + 1e-8))) * 100)
    cpc = float(2 * np.sum(np.minimum(prediction, y_true)) / (np.sum(prediction) + np.sum(y_true) + 1e-8))
    pred_pos = prediction > 0
    true_pos = y_true > 0
    cpl = float(2 * np.sum(pred_pos & true_pos) / (np.sum(pred_pos) + np.sum(true_pos) + 1e-8))
    print("\n[GAT Direct] test metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MAPE: {mape:.4f}")
    print(f"  CPC:  {cpc:.4f}")
    print(f"  CPL:  {cpl:.4f}")

    # same file layout as the original GBRT-stage output
    test_ids = test_data[:, :2].astype(int)
    os.makedirs(outputs_dir, exist_ok=True)
    output_file = os.path.join(outputs_dir, 'total_flow_results.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("origin\tdest\tpred_flow\ttrue_flow\n")
        for i in range(len(test_ids)):
            f.write(f"{test_ids[i,0]}\t{test_ids[i,1]}\t{prediction[i]:.6f}\t{y_true[i]:.6f}\n")
    print(f"[OK] predictions saved: {output_file}")

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'CPC': cpc, 'CPL': cpl}


def run_purpose_softmax(config, data, work_dir):
    """Train the softmax FFN purpose head on frozen GAT embeddings and evaluate."""
    models_dir = os.path.join(work_dir, 'models')
    embeddings_dir = os.path.join(work_dir, 'embeddings')
    out_dir = os.path.join(work_dir, 'outputs')
    log_dir = os.path.join(work_dir, 'log')
    for d in (models_dir, embeddings_dir, out_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    logger = logging.getLogger("purpose_softmax")
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(os.path.join(log_dir, 'purpose_softmax.log'), encoding='utf-8')
        fh.setFormatter(formatter)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)

    device = torch.device(config['device'])
    logger.info(f"device: {device}")

    distm = data.get('distm')

    # true purpose flows -> true probability distributions (row-normalized)
    pflows = data['purpose_flows']
    true_probs_train = flows_to_probs(pflows['train'])    # (N_tr, 15)
    true_probs_valid = flows_to_probs(pflows['valid'])    # (N_va, 15)
    true_probs_test = flows_to_probs(pflows['test'])      # (N_te, 15)

    test_ids = data['test'][:, :2].astype(int)
    train_data = data['train']
    valid_data = data['valid']
    test_data = data['test']

    logger.info(f"train: {len(train_data)}, valid: {len(valid_data)}, test: {len(test_data)}")

    # frozen embeddings from the total-flow GAT stage
    logger.info("== loading pretrained GAT embeddings ==")
    emb_fp = os.path.join(embeddings_dir,
                          'censustract_embeddings_total_layers{}_emb{}_multitask{}.npz'.format(
                              config['num_hidden_layers'], config['embedding_size'],
                              config['multitask_weights']))
    if not os.path.exists(emb_fp):
        raise FileNotFoundError(
            f"embedding file not found: {emb_fp}\n"
            f"run step 1 (GAT pretraining) to generate embeddings first."
        )

    embs = np.load(emb_fp)
    src_emb = embs['arr_0'].astype(np.float32)   # (num_nodes, emb_dim)
    dst_emb = embs['arr_1'].astype(np.float32)
    emb_dim = src_emb.shape[1]
    logger.info(f"embedding dim: {emb_dim}")

    logger.info("== building FFN input features ==")
    if distm is not None:
        # scale distances to the embedding magnitude so features share a range
        max_dist = distm.max()
        max_emb = max(src_emb.max(), dst_emb.max())
        scaled_distm = distm / (max_dist + 1e-12) * max_emb
    else:
        n = src_emb.shape[0]
        scaled_distm = np.zeros((n, n), dtype=np.float32)
        logger.warning("distm is None, using an all-zero distance matrix")

    X_train = build_features(train_data, src_emb, dst_emb, scaled_distm)
    X_valid = build_features(valid_data, src_emb, dst_emb, scaled_distm)
    X_test = build_features(test_data, src_emb, dst_emb, scaled_distm)

    feat_dim = X_train.shape[1]
    logger.info(f"feature dim: {feat_dim} (= 2*{emb_dim} + 1)")

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(true_probs_train).to(device)
    X_valid_t = torch.from_numpy(X_valid).to(device)
    y_valid_t = torch.from_numpy(true_probs_valid).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(true_probs_test).to(device)

    logger.info("== building FFN softmax model ==")
    model = PurposeSoftmaxNet(
        in_dim=feat_dim,
        hidden_dim=FFN_HIDDEN_DIM,
        num_purposes=NUM_PURPOSES,
        dropout=FFN_DROPOUT,
    ).to(device)

    logger.info(f"FFN architecture: {feat_dim} -> {FFN_HIDDEN_DIM} -> {FFN_HIDDEN_DIM//2} -> {NUM_PURPOSES}")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"FFN parameters: {total_params:,}")

    # only the FFN is optimized; the GAT embeddings stay frozen
    optimizer = torch.optim.Adam(model.parameters(), lr=FFN_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    # MSE on the probability distributions (simple and effective)
    criterion = nn.MSELoss()

    logger.info("== training FFN softmax classifier ==")
    best_jsd = float('inf')
    best_state = None

    for epoch in range(FFN_EPOCHS):
        model.train()
        perm = torch.randperm(X_train_t.shape[0])
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, X_train_t.shape[0], FFN_BATCH_SIZE):
            idx = perm[i:i + FFN_BATCH_SIZE]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            pred = model(xb)            # (batch, 15) softmax output
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == FFN_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                pred_valid = model(X_valid_t).cpu().numpy()
                metrics_v = evaluate_distribution(pred_valid, true_probs_valid)

            logger.info(
                f"Epoch {epoch:03d}/{FFN_EPOCHS-1} | "
                f"train_loss={epoch_loss/n_batches:.6f} | "
                f"valid_JSD={metrics_v['JSD_mean']:.4f} | "
                f"valid_Cos={metrics_v['Cosine_mean']:.4f}"
            )
            if metrics_v['JSD_mean'] < best_jsd:
                best_jsd = metrics_v['JSD_mean']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    logger.info("== test-set evaluation (best model) ==")
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_test = model(X_test_t).cpu().numpy()
        pred_valid_final = model(X_valid_t).cpu().numpy()

    metrics_test = evaluate_distribution(pred_test, true_probs_test)
    metrics_valid = evaluate_distribution(pred_valid_final, true_probs_valid)

    _log_metrics(logger, "Test", metrics_test)
    _log_metrics(logger, "Valid", metrics_valid)

    logger.info("== saving results ==")

    # per-purpose MAE
    pp_csv = os.path.join(out_dir, "purpose_per_category_mae.csv")
    with open(pp_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["split"] + PURPOSE_CODES)
        w.writerow(["test"] + [f"{v:.6f}" for v in metrics_test["per_purpose_MAE"]])
        w.writerow(["valid"] + [f"{v:.6f}" for v in metrics_valid["per_purpose_MAE"]])
    logger.info(f"per-purpose MAE saved: {pp_csv}")

    # overall metrics
    summary_csv = os.path.join(out_dir, "purpose_overall_metrics.csv")
    _write_metrics_csv(summary_csv, metrics_test, metrics_valid)
    logger.info(f"overall metrics saved: {summary_csv}")

    # probability matrices
    np.save(os.path.join(out_dir, "pred_probs_test.npy"), pred_test)
    np.save(os.path.join(out_dir, "pred_probs_valid.npy"), pred_valid_final)
    np.save(os.path.join(out_dir, "true_probs_test.npy"), true_probs_test)
    logger.info(f"probability matrix saved to {out_dir}")

    # detailed per-OD results
    _write_purpose_intermediate(
        test_ids=test_ids,
        pred_probs=pred_test,
        true_probs=true_probs_test,
        filename=os.path.join(out_dir, "purpose_prediction_results.txt")
    )

    ffn_fp = os.path.join(models_dir, "purpose_softmax_ffn.pth")
    torch.save(best_state, ffn_fp)
    logger.info(f"FFN model saved: {ffn_fp}")

    logger.info("== all done ==")
    return metrics_test


# ---------------------------------------------------------------------------
# Merge flow and purpose results into the standard result file
# ---------------------------------------------------------------------------

def load_flow_results(filepath):
    """Read total_flow_results.txt into {(origin, dest): (pred_flow, true_flow)}."""
    flow_map = {}
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (int(row["origin"]), int(row["dest"]))
            flow_map[key] = (float(row["pred_flow"]), float(row["true_flow"]))
    return flow_map


def load_purpose_results(filepath):
    """Read purpose_prediction_results.txt into (origin, dest, pred[15], true[15]) rows.

    The file has a 4-column header; data rows may be tab- or space-separated
    (32 numeric fields when fully split), so both layouts are handled.
    """
    rows = []
    with open(filepath, encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        assert len(header) == 4, f"expected 4 columns, got {len(header)}, header={header}"

        for line in f:
            if not line.strip():
                continue
            fields = line.strip().split("\t")
            if len(fields) != 4:
                # mixed tab/space separators: re-split on whitespace
                all_vals = line.strip().split()
                if len(all_vals) != 32:
                    print(f"warning: skipping malformed row (fields={len(all_vals)})")
                    continue
                origin = int(all_vals[0])
                dest = int(all_vals[1])
                pred_vals = list(map(float, all_vals[2:17]))    # indices 2..16
                true_vals = list(map(float, all_vals[17:32]))   # indices 17..31
            else:
                origin = int(fields[0])
                dest = int(fields[1])
                pred_vals = list(map(float, fields[2].split()))
                true_vals = list(map(float, fields[3].split()))
            if len(pred_vals) != 15 or len(true_vals) != 15:
                print(f"warning: skipping malformed row origin={origin}, dest={dest}")
                continue
            rows.append((origin, dest, pred_vals, true_vals))
    return rows


def merge_flow_purpose(data_path, flow_fp, purpose_fp, out_fp, logger):
    """Merge flow and purpose predictions into the standard 34-column file."""
    _, id2entity = load_entity_dict(data_path)
    logger.info(f"[merge] entity dictionary: {len(id2entity)} entries")

    flow_map = load_flow_results(flow_fp)
    logger.info(f"[merge] flow results: {len(flow_map)} rows")

    purpose_rows = load_purpose_results(purpose_fp)
    logger.info(f"[merge] purpose results: {len(purpose_rows)} rows")

    os.makedirs(os.path.dirname(out_fp) or ".", exist_ok=True)
    matched = 0
    missing_flow = 0

    with open(out_fp, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(OUT_COLS)
        for origin, dest, pred_vals, true_vals in purpose_rows:
            key = (origin, dest)
            if key in flow_map:
                pred_flow, true_flow = flow_map[key]
                matched += 1
            else:
                pred_flow, true_flow = "", ""
                missing_flow += 1
            hyid_origin = id2entity.get(origin, str(origin))
            hyid_dest = id2entity.get(dest, str(dest))
            writer.writerow(
                [hyid_origin, hyid_dest]
                + [f"{v:.10e}" for v in pred_vals]
                + [f"{v:.10e}" for v in true_vals]
                + ([f"{pred_flow:.6f}", f"{true_flow:.6f}"] if pred_flow != "" else ["", ""])
            )

    logger.info(f"[merge] merged {len(purpose_rows)} rows")
    logger.info(f"  matched flow records: {matched}")
    logger.info(f"  missing flow records: {missing_flow}")
    logger.info(f"  output file: {out_fp}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def resolve_device(name):
    """Resolve the compute device, falling back to CPU when CUDA is unavailable."""
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if name.startswith('cuda') and not torch.cuda.is_available():
        print(f"[device] CUDA is unavailable, falling back to CPU (requested: {name})")
        return 'cpu'
    return name


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="stc-rgcn-baseline-gmel",
        description='GMEL baseline: GAT flow prediction + softmax purpose head, '
                    'merged into the standard 34-column result file.')
    parser.add_argument('--datadir', default=_DEFAULT_DATADIR,
                        help='Dataset directory (split files, entities.dict, node features)')
    parser.add_argument('--device', default='auto',
                        help="Device: 'auto', 'cpu', 'cuda' or 'cuda:N' "
                             "(falls back to CPU when CUDA is unavailable)")
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of GAT training epochs')
    parser.add_argument('--embedding-size', type=int, default=16,
                        help='GAT node embedding dimension')
    parser.add_argument('--num-hidden-layers', type=int, default=1,
                        help='Number of GAT hidden layers')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='GAT learning rate')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='GAT mini-batch size')
    parser.add_argument('--output-dir', default=_DEFAULT_OUTPUT_DIR,
                        help='Directory receiving gmel_result.txt; intermediates '
                             'are written to <output-dir>/gmel_work/')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.device = resolve_device(args.device)

    output_dir = os.path.abspath(args.output_dir)
    work_dir = os.path.join(output_dir, 'gmel_work')
    models_dir = os.path.join(work_dir, 'models')
    embeddings_dir = os.path.join(work_dir, 'embeddings')
    outputs_dir = os.path.join(work_dir, 'outputs')
    log_dir = os.path.join(work_dir, 'log')
    for d in (output_dir, work_dir, models_dir, embeddings_dir, outputs_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'gmel_run.log'), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    config = {
        'data_path': os.path.abspath(args.datadir),
        'device': args.device,
        'embedding_size': args.embedding_size,
        'num_hidden_layers': args.num_hidden_layers,
        'multitask_weights': MULTITASK_WEIGHTS,
        'reg_param': REG_PARAM,
        'max_epochs': args.epochs,
        'mini_batch_size': args.batch_size,
        'lr': args.lr,
        'grad_norm': GRAD_NORM,
        'evaluate_every': EVALUATE_EVERY,
        'num_purposes': NUM_PURPOSES,
        'merge_output_file': 'gmel_result.txt',
    }
    final_fp = os.path.join(output_dir, config['merge_output_file'])

    logger.info("=" * 60)
    logger.info("one-shot run: GAT direct prediction + purpose distribution + merge")
    logger.info("=" * 60)
    logger.info(f"device: {config['device']}")
    logger.info(f"GAT config: layers={config['num_hidden_layers']}, emb={config['embedding_size']}")
    logger.info(f"merged output: {final_fp}")

    # loaded once and shared by all stages
    data = load_od_dataset(config['data_path'])

    # GAT training on total flow
    logger.info("")
    logger.info("=" * 40)
    logger.info("step 1: GAT training (total flow)")
    logger.info("=" * 40)
    model_path = train_gat(config, data, work_dir)

    # GAT direct prediction on the test set
    logger.info("")
    logger.info("=" * 40)
    logger.info("step 2: GAT direct prediction (total flow)")
    logger.info("=" * 40)
    gat_direct_predict(model_path, config, data, outputs_dir)

    # purpose distribution prediction (single softmax model)
    logger.info("")
    logger.info("=" * 40)
    logger.info("step 3: purpose distribution (single softmax model)")
    logger.info("=" * 40)
    run_purpose_softmax(config, data, work_dir)

    # merge flow and purpose results
    logger.info("")
    logger.info("=" * 40)
    logger.info("step 4: merge flow and purpose results")
    logger.info("=" * 40)
    merge_flow_purpose(
        config['data_path'],
        os.path.join(outputs_dir, 'total_flow_results.txt'),
        os.path.join(outputs_dir, 'purpose_prediction_results.txt'),
        final_fp,
        logger)

    logger.info("")
    logger.info("=" * 60)
    logger.info("all done!")
    logger.info("=" * 60)
    logger.info("output files:")
    logger.info(f"  - {os.path.join(outputs_dir, 'total_flow_results.txt')} (total-flow predictions, GAT direct)")
    logger.info(f"  - {os.path.join(outputs_dir, 'purpose_prediction_results.txt')} (purpose distributions)")
    logger.info(f"  - {os.path.join(outputs_dir, 'purpose_overall_metrics.csv')} (distribution metrics)")
    logger.info(f"  - {os.path.join(outputs_dir, 'purpose_per_category_mae.csv')} (per-purpose MAE)")
    logger.info(f"  - {final_fp} (merged result)")


if __name__ == '__main__':
    main()


