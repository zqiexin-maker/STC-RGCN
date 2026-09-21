"""
SI-GCN baseline: single-file PyTorch re-implementation for OD total-flow prediction.

The original SI-GCN is implemented in TensorFlow 1.x; this file re-implements the
architecture in PyTorch. (An earlier revision of this repo vendored a fragment of
the upstream code, which shipped only the data adapter and the model-builder
vocabulary -- BasisGcn, RelationEmbedding, BilinearDiag/DistDecoder -- without the
encoders/decoders/extras packages they reference, hence this re-implementation.)

References
----------
Yao et al., "Spatial Origin-Destination Flow Imputation Using Graph Convolutional
Neural Networks", IEEE Transactions on Intelligent Transportation Systems, 2020
(SI-GCN). Upstream code: https://github.com/susurrant/flow-imputation (TF1.x).
Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional
Networks", ESWC 2017 (R-GCN, basis decomposition). Upstream code:
https://github.com/MichSchli/RelationPrediction (TF1.x), which SI-GCN forked.

Architecture (this file)
------------------------
Encoder: relational GCN with basis decomposition (Schlichtkrull et al. 2017):
    H' = act( sum_{edges o->d} W_r h_o / in_degree(d) + H W_self + b ),
with W_r = sum_b a_rb V_b over learnable bases V_b and coefficients a_rb. The
graph has ONE relation: nodes are zones (entities), directed edges are the
observed training OD pairs, and messages (origin -> destination) are aggregated
with in-degree (mean) normalization. Two GCN layers (the last is linear, as
upstream), ReLU activation, optional dropout between layers. Encoder input is a
learnable node-embedding table (upstream RandomEmbedding analogue); pass
--use-features to instead use a linear projection of features.txt
(upstream AffineTransform analogue).

Decoder: distance decoder (DistDecoder analogue). Each OD pair is scored by
d = ||h_origin - h_dest||_2 in the final embedding space and mapped to a
non-negative flow estimate by a small monotone-decreasing MLP built from
softplus-positive weights:
    f(d) = softplus( b - sum_k softplus(w2_k) * softplus(w1_k * d + c_k) )
f is provably non-increasing in d and bounded above by softplus(b). This is the
chosen "small learnable monotone mapping": it encodes the gravity-style prior
that embedding-close zones exchange more flow while staying fully learnable.

Training: full-batch Adam minimizing MSE on log1p(total flow) (--no-log trains
on raw flows). Early stopping monitors MAE in raw-flow space on the validation
split (--patience); the best checkpoint is restored for evaluation. The
message-passing graph contains training edges only; validation/evaluation pairs
are decoded without entering the graph.

Purpose distribution: SI-GCN is a flow-only model, so the predicted 15-dim
purpose distribution is the constant training-set flow-weighted average
distribution (sum of purpose flows / sum of total flows over training rows),
assigned to every evaluation OD pair and printed at runtime.

Data conventions follow the adapter that accompanied the vendored fragment
(upstream my_data_loader.py): 40-column whitespace-separated split files where
column 0 is the origin entity string, column 5 the destination entity string,
and columns 25-39 the 15 purpose flows (total flow = their sum); entity ids come
from entities.dict.

Output follows the standard 34-column result layout
(runs/baselines/sigcn_result.txt).
"""

import argparse
import copy
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    NUM_PURPOSES,
    PURPOSE_CODES,
    add_common_args,
    banner,
    evaluate_all,
    load_entity_dict,
    load_node_features,
    read_od_split,
    resolve_split_file,
    section,
    write_predictions,
)


def set_seed(seed):
    """Seed python/numpy/torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _inv_softplus(x):
    """Exact inverse of softplus for x > 0 (guarded near zero)."""
    x = max(float(x), 1e-3)
    return x if x > 20.0 else math.log(math.expm1(x))


class BasisRGCNLayer(nn.Module):
    """Relational GCN layer with basis decomposition (Schlichtkrull et al. 2017).

    W_r = sum_b a_rb V_b; messages flow along directed edges (src -> dst) and
    are averaged over each destination's in-degree, plus a self-transform W_self.
    """

    def __init__(self, in_dim, out_dim, num_relations=1, num_bases=4,
                 use_activation=True):
        super().__init__()
        self.num_relations = num_relations
        self.num_bases = max(1, num_bases)
        self.out_dim = out_dim
        self.use_activation = use_activation

        self.bases = nn.Parameter(torch.empty(self.num_bases, in_dim, out_dim))
        self.coefs = nn.Parameter(torch.empty(num_relations, self.num_bases))
        self.w_self = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.bases)
        nn.init.constant_(self.coefs, 1.0 / self.num_bases)
        nn.init.xavier_uniform_(self.w_self)

    def forward(self, h, edge_index):
        src, dst = edge_index[0], edge_index[1]
        # Basis decomposition: (R, in, out)
        w_rel = torch.einsum("rb,bio->rio", self.coefs, self.bases)
        if self.num_relations == 1:
            msg = h[src] @ w_rel[0].t()
        else:
            msg = torch.bmm(h[src].unsqueeze(1), w_rel[src.new_zeros(src.shape)]
                            .transpose(1, 2)).squeeze(1)

        n = h.size(0)
        agg = h.new_zeros((n, self.out_dim))
        agg.index_add_(0, dst, msg)
        deg = h.new_zeros(n)
        deg.index_add_(0, dst, torch.ones_like(src, dtype=h.dtype))
        agg = agg / deg.clamp(min=1.0).unsqueeze(1)

        out = agg + h @ self.w_self.t() + self.bias
        return F.relu(out) if self.use_activation else out


class MonotoneDistDecoder(nn.Module):
    """Distance decoder (DistDecoder analogue) with a learnable monotone map.

    f(d) = softplus( b - sum_k softplus(w2_k) * softplus(w1_k * d + c_k) ) is
    non-increasing in the embedding distance d = ||h_o - h_d||_2 (softplus-
    positive weights) and bounded above by softplus(b). It predicts log1p(flow)
    by default, or the raw flow with --no-log; either way the output is
    non-negative.
    """

    def __init__(self, hidden_units=8, init_output=1.0):
        super().__init__()
        self.raw_w1 = nn.Parameter(torch.full((hidden_units,), _inv_softplus(0.05)))
        self.raw_w2 = nn.Parameter(torch.full((hidden_units,), _inv_softplus(0.01)))
        self.c = nn.Parameter(torch.zeros(hidden_units))
        self.raw_b = nn.Parameter(torch.tensor(_inv_softplus(init_output)))

    def forward(self, h_origin, h_dest):
        d = torch.norm(h_origin - h_dest, dim=-1)
        z = F.softplus(F.softplus(self.raw_w1) * d.unsqueeze(1) + self.c)   # (N, H)
        s = self.raw_b - torch.sum(F.softplus(self.raw_w2) * z, dim=1)
        return F.softplus(s)


class SIGCN(nn.Module):
    """Encoder (input embedding + stacked basis-decomposition R-GCN layers)."""

    def __init__(self, num_entities, embedding_size=32, num_bases=4, num_layers=2,
                 dropout=0.0, feature_dim=None, decoder_hidden=8,
                 decoder_init_output=1.0):
        super().__init__()
        num_layers = max(1, num_layers)
        if feature_dim is None:
            self.input_embed = nn.Embedding(num_entities, embedding_size)
            nn.init.normal_(self.input_embed.weight, mean=0.0, std=0.1)
            self.input_proj = None
        else:
            self.input_proj = nn.Linear(feature_dim, embedding_size)
            self.input_embed = None
        self.gcn_layers = nn.ModuleList([
            BasisRGCNLayer(embedding_size, embedding_size,
                           num_relations=1, num_bases=num_bases,
                           use_activation=(i < num_layers - 1))
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.decoder = MonotoneDistDecoder(hidden_units=decoder_hidden,
                                           init_output=decoder_init_output)

    def encode(self, edge_index, node_features=None):
        """Full-graph forward pass; returns final node embeddings."""
        if self.input_embed is not None:
            h = self.input_embed.weight
        else:
            h = self.input_proj(node_features)
        last = len(self.gcn_layers) - 1
        for i, layer in enumerate(self.gcn_layers):
            h = layer(h, edge_index)
            if i != last:
                h = self.dropout(h)
        return h

    def decode(self, h, origins, dests):
        return self.decoder(h[origins], h[dests])


def to_flow_space(pred, use_log):
    """Map decoder outputs back to raw flow counts (clip guards expm1 overflow)."""
    if use_log:
        return np.expm1(np.clip(pred, 0.0, 30.0))
    return np.maximum(pred, 0.0)


def train_model(model, edge_index, train_o, train_d, y_train,
                valid_o, valid_d, valid_true, device, args, feats=None):
    """Full-batch training with early stopping on validation MAE (raw-flow space)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    use_log = not args.no_log
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    valid_true = valid_true.astype(np.float64)

    best_mae, best_epoch, best_state, bad_rounds = float("inf"), -1, None, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        h = model.encode(edge_index, feats)
        loss = F.mse_loss(model.decode(h, train_o, train_d), y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            h_val = model.encode(edge_index, feats)
            pred_val = model.decode(h_val, valid_o, valid_d).cpu().numpy()
        valid_mae = float(np.mean(np.abs(to_flow_space(pred_val, use_log) - valid_true)))

        if valid_mae < best_mae - 1e-9:
            best_mae, best_epoch = valid_mae, epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_rounds = 0
        else:
            bad_rounds += 1

        if epoch == 1 or epoch % 20 == 0:
            print("  epoch {:4d}  train_mse={:.4f}  valid_MAE={:.4f}".format(
                epoch, loss.item(), valid_mae))
        if bad_rounds >= args.patience:
            print("  early stopping at epoch {} (no valid_MAE improvement for "
                  "{} epochs)".format(epoch, args.patience))
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print("  restored best checkpoint: epoch {}, valid_MAE={:.4f}".format(
        best_epoch, best_mae))
    return model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="stc-rgcn-baseline-sigcn",
        description="SI-GCN baseline (PyTorch re-implementation); see module docstring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser, with_seed=True, dest="outdir")
    parser.add_argument("--split", default="test", choices=("test", "valid", "train"),
                        help="Split to evaluate (falls back to 'valid' when the "
                             "test split is not shipped with the dataset)")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                        help="Compute device (auto = cuda if available)")
    parser.add_argument("--embedding-size", type=int, default=32,
                        help="Node embedding / hidden dimension of the R-GCN encoder")
    parser.add_argument("--num-bases", type=int, default=4,
                        help="Number of basis matrices in the basis decomposition")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of stacked R-GCN layers")
    parser.add_argument("--decoder-hidden", type=int, default=8,
                        help="Hidden units of the monotone distance-decoder MLP")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout applied between R-GCN layers")
    parser.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Adam weight decay")
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early-stopping patience on validation MAE")
    parser.add_argument("--no-log", action="store_true",
                        help="Train on raw flows instead of log1p(flows)")
    parser.add_argument("--use-features", action="store_true",
                        help="Use features.txt (linear projection) as "
                             "encoder input instead of learnable embeddings")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    use_log = not args.no_log

    banner("SI-GCN baseline (PyTorch re-implementation of the TF1.x upstream)")
    print("  device={}  seed={}  target_space={}".format(
        device, args.seed, "raw flow" if args.no_log else "log1p(flow)"))
    print("  embedding_size={}  num_bases={}  num_layers={}  dropout={}".format(
        args.embedding_size, args.num_bases, args.num_layers, args.dropout))

    section("Loading entity dictionary and OD splits", 1, 5)
    entity2id, id2entity = load_entity_dict(args.datadir)
    num_entities = max(entity2id.values()) + 1
    print("  entities: {}".format(len(entity2id)))

    train_ids, train_flow, train_pflow, _ = read_od_split(
        resolve_split_file(args.datadir, "train"), entity2id)
    valid_ids, valid_flow, _, _ = read_od_split(
        resolve_split_file(args.datadir, "valid"), entity2id)
    eval_path = resolve_split_file(args.datadir, args.split)
    eval_ids, eval_flow, eval_pflow, _ = read_od_split(eval_path, entity2id)
    if os.path.abspath(eval_path) == os.path.abspath(
            resolve_split_file(args.datadir, "valid")):
        print("  [warn] no test split: early stopping and evaluation both use "
              "the validation split")

    # Training graph edges / regression targets: keep positive-flow rows only,
    # matching the vendored adapter's total_flow mode.
    positive = train_flow > 0
    tr_h, tr_t = train_ids[positive, 0], train_ids[positive, 1]
    tr_flow, tr_pflow = train_flow[positive], train_pflow[positive]
    print("  training edges (flow > 0): {}".format(len(tr_h)))
    y_train = np.log1p(tr_flow) if use_log else tr_flow.astype(np.float32)

    section("Building tensors", 2, 5)
    to_dev = lambda arr: torch.from_numpy(np.asarray(arr)).to(device)   # noqa: E731
    edge_index = torch.stack([to_dev(tr_h.astype(np.int64)),
                              to_dev(tr_t.astype(np.int64))])
    train_o, train_d = to_dev(tr_h.astype(np.int64)), to_dev(tr_t.astype(np.int64))
    valid_o = to_dev(valid_ids[:, 0].astype(np.int64))
    valid_d = to_dev(valid_ids[:, 1].astype(np.int64))
    eval_o = to_dev(eval_ids[:, 0].astype(np.int64))
    eval_d = to_dev(eval_ids[:, 1].astype(np.int64))

    feature_dim, feats = None, None
    if args.use_features:
        feats = load_node_features(args.datadir, num_entities)
        if feats is None:
            print("  features.txt not found; "
                  "falling back to learnable embeddings")
        else:
            feature_dim = feats.shape[1]
            feats = torch.from_numpy(feats).to(device)
            print("  node features: dim {}".format(feature_dim))

    section("Training SI-GCN", 3, 5)
    model = SIGCN(num_entities,
                  embedding_size=args.embedding_size,
                  num_bases=args.num_bases,
                  num_layers=args.num_layers,
                  dropout=args.dropout,
                  feature_dim=feature_dim,
                  decoder_hidden=args.decoder_hidden,
                  decoder_init_output=float(np.mean(y_train))).to(device)
    print("  model parameters: {}".format(
        sum(p.numel() for p in model.parameters())))
    train_model(model, edge_index, train_o, train_d, y_train,
                valid_o, valid_d, valid_flow, device, args, feats=feats)

    section("Predicting evaluation split", 4, 5)
    model.eval()
    with torch.no_grad():
        h = model.encode(edge_index, feats)
        pred_eval = model.decode(h, eval_o, eval_d).cpu().numpy()
    pred_flow_eval = to_flow_space(pred_eval, use_log)

    # Purpose allocation: constant training flow-weighted average distribution.
    dist_sum = tr_pflow.sum(axis=0)
    global_dist = dist_sum / dist_sum.sum()
    print("  purpose distribution = training flow-weighted average "
          "(constant for every OD pair):")
    print("    " + "  ".join("{}={:.4f}".format(n, p)
                             for n, p in zip(PURPOSE_CODES, global_dist)))
    pred_probs = np.tile(global_dist, (len(eval_ids), 1))

    # True probabilities from purpose flows (zero rows stay zero).
    true_probs = np.zeros((len(eval_ids), NUM_PURPOSES), dtype=np.float64)
    nonzero = eval_flow > 0
    true_probs[nonzero] = eval_pflow[nonzero] / eval_flow[nonzero, None]

    section("Saving results and evaluating", 5, 5)
    output = os.path.join(args.outdir, "sigcn_result.txt")
    write_predictions(eval_ids, id2entity, pred_probs, true_probs,
                 pred_flow_eval, eval_flow, output)
    evaluate_all(eval_flow, pred_flow_eval, true_probs, pred_probs,
                 model_name="SI-GCN (PyTorch re-implementation)")
    banner("SI-GCN prediction completed")


if __name__ == "__main__":
    main()
