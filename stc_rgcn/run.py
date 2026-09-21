"""Training entry point: train and evaluate one model variant.

Run it either way::

    python -m stc_rgcn.run --variant full --task multi --patience 20
    stc-rgcn-train    --variant full --task multi --patience 20

Each run writes a self-contained directory under ``--output-dir``::

    runs/<run-name>/
        config.json       the resolved arguments
        best_model.pt     best-validation checkpoint
        predictions.txt   test predictions in the shared 34-column layout
        metrics.json      final test metrics

Run names default to ``<variant>-<task>``, so two variants never overwrite each
other's checkpoints the way the pre-1.0 fixed file names did.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import replace

import numpy as np
import torch

from .config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, Task, Variant
from .data import load_dataset
from .model import StcRgcn
from .training import evaluate, predict, train_step, write_predictions
from .utils.early_stopping import EarlyStopper, monitored_metrics
from .utils.gradnorm import GradNormBalancer
from .utils.graph import build_full_graph
from .utils.losses import estimate_tweedie_power
from .utils.metrics import flow_metrics, flow_metrics_by_magnitude, purpose_distribution_metrics
from .utils.schema import DEFAULT_FEATURE_FILE
from .utils.transforms import LogMinMaxScaler

CHECKPOINT_NAME = "best_model.pt"
PREDICTIONS_NAME = "predictions.txt"
CONFIG_NAME = "config.json"
METRICS_NAME = "metrics.json"


def build_parser():
    # prog is left to argparse so it reports whichever entry point was used:
    # "run.py" under `python -m`, "stc-rgcn-train" from the installed script.
    parser = argparse.ArgumentParser(
        description="Train STC-RGCN on an OD dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                      help="Directory holding entities.dict and the three splits")
    data.add_argument("--feature-file", default=DEFAULT_FEATURE_FILE,
                      help="Node feature file inside --data-dir")
    data.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                      help="Parent directory for run directories")
    data.add_argument("--run-name", default=None,
                      help="Run directory name (default: <variant>-<task>)")

    setup = parser.add_argument_group("model")
    setup.add_argument("--variant", type=Variant, choices=list(Variant), default=Variant.FULL,
                       help="Which inputs and prediction heads to enable")
    setup.add_argument("--task", type=Task, choices=list(Task), default=Task.MULTI,
                       help="Which loss drives training and early stopping")
    setup.add_argument("--num-bases", type=int, default=4,
                       help="Basis decomposition rank of the relation-weighted convolution")
    setup.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout in the convolution stack and the heads")
    setup.add_argument("--tweedie-power", type=float, default=None,
                       help="Tweedie power for the full_tweedie variant "
                            "(default: estimated from the training flows)")

    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=3000, help="Maximum number of epochs")
    train.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate")
    train.add_argument("--sample-size", type=int, default=3000,
                       help="Edges sampled per training step")
    train.add_argument("--graph-split-ratio", type=float, default=0.5,
                       help="Fraction of sampled edges used for message passing and supervision")
    train.add_argument("--reg-weight", type=float, default=0.005,
                       help="Weight of the parameter-norm penalty; 0 disables it")
    train.add_argument("--clip-grad-norm", type=float, default=1.0,
                       help="Max gradient norm for clipping (not the GradNorm algorithm)")
    train.add_argument("--log-transform", action="store_true",
                       help="Train on log(flow) rescaled to [0, 1] using training statistics")
    train.add_argument("--gradnorm", action="store_true",
                       help="Balance the two task losses with GradNorm")

    stop = parser.add_argument_group("evaluation")
    stop.add_argument("--eval-every", type=int, default=2, help="Validate every N epochs")
    stop.add_argument("--patience", type=int, default=20,
                      help="Validations without improvement before stopping")
    stop.add_argument("--delta", type=float, default=0.01,
                      help="Minimum improvement that counts")
    stop.add_argument("--eval-on-gpu", action="store_true",
                      help="Keep validation on the training device; by default the "
                           "full graph is scored on CPU, which fits larger datasets")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default="auto",
                         help="'auto', 'cpu', 'cuda' or 'cuda:<index>'")
    runtime.add_argument("--seed", type=int, default=None,
                         help="Seed for python, numpy and torch")
    return parser


def resolve_device(name):
    """Turn a ``--device`` string into a :class:`torch.device`, falling back to CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("warning: CUDA is unavailable, falling back to CPU")
        return torch.device("cpu")
    return device


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _apply_flow_scaling(dataset, enabled):
    """Fit the flow scaler on the training split and apply it to all three."""
    if not enabled or dataset.train.flows is None:
        return LogMinMaxScaler.identity()

    scaler = LogMinMaxScaler.fit(dataset.train.flows)
    for name in ("train", "valid", "test"):
        split = getattr(dataset, name)
        setattr(dataset, name, replace(split, flows=scaler.transform(split.flows)))
    print(
        "log + min-max flow normalization: log-min {:.4f}, log-max {:.4f}".format(
            scaler.log_min, scaler.log_max
        )
    )
    return scaler


def _training_loss(losses, task, reg_weight, penalty):
    """Select which task losses back-propagate for a non-GradNorm run.

    Unlike pre-1.0 releases, the parameter penalty is applied for single-task
    runs too; pass ``--reg-weight 0`` to reproduce the old behaviour.
    """
    if task is Task.MULTI:
        return losses.total
    selected = losses.flow if task is Task.FLOW else losses.purpose
    if selected is None:
        raise RuntimeError(
            "--task {} needs a head the variant does not have".format(task)
        )
    return selected + reg_weight * penalty


def _check_data_dir(path):
    """Fail early, and helpfully, when the dataset directory is missing."""
    if os.path.isdir(path):
        return
    hint = ""
    if os.path.abspath(path) == os.path.abspath(DEFAULT_DATA_DIR):
        hint = (
            " This is the built-in default, which only exists in a source checkout; "
            "pass --data-dir to point at your own dataset."
        )
    raise FileNotFoundError("dataset directory not found: {}.{}".format(path, hint))


def run(args):
    _check_data_dir(args.data_dir)
    device = resolve_device(args.device)
    set_seed(args.seed)
    generator = np.random.default_rng(args.seed)
    print("device: {}".format(device))

    run_name = args.run_name or "{}-{}".format(args.variant, args.task)
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, CONFIG_NAME), "w", encoding="utf-8") as handle:
        json.dump({k: str(v) for k, v in vars(args).items()}, handle, indent=2)

    dataset = load_dataset(args.data_dir, args.variant, feature_file=args.feature_file)
    scaler = _apply_flow_scaling(dataset, args.log_transform)

    tweedie_power = args.tweedie_power
    if args.variant.uses_tweedie_loss and tweedie_power is None:
        tweedie_power = estimate_tweedie_power(dataset.train.flows)
    tweedie_power = tweedie_power if tweedie_power is not None else 1.5

    graph = build_full_graph(
        dataset.train,
        args.variant,
        dataset.num_entities,
        dataset.num_relations,
        node_features=dataset.node_features,
    )

    model = StcRgcn(
        variant=args.variant,
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations,
        feature_dim=dataset.feature_dim,
        num_bases=args.num_bases,
        dropout=args.dropout,
    ).to(device)
    print(model)

    balancer = None
    if args.gradnorm:
        if args.variant.is_multi_task:
            balancer = GradNormBalancer(num_tasks=2, device=device)
        else:
            print("warning: --gradnorm needs two heads; ignoring it for {}".format(args.variant))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(
        monitored_metrics(args.variant, args.task), patience=args.patience, delta=args.delta
    )
    checkpoint_path = os.path.join(run_dir, CHECKPOINT_NAME)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        losses = train_step(
            model=model,
            split=dataset.train,
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            sample_size=args.sample_size,
            graph_split_ratio=args.graph_split_ratio,
            reg_weight=args.reg_weight,
            node_features=dataset.node_features,
            tweedie_power=tweedie_power,
            device=device,
            generator=generator,
        )

        if balancer is not None:
            task_losses = losses.task_losses(device)
            weights = balancer.compute_weights(task_losses, model)
            loss = sum(weight * task for weight, task in zip(weights, task_losses))
            loss = loss + args.reg_weight * losses.penalty
            if epoch % 10 == 0:
                print(
                    "Epoch {}: purpose weight {:.4f}, flow weight {:.4f}".format(
                        epoch, weights[0].item(), weights[1].item()
                    )
                )
        else:
            loss = _training_loss(losses, args.task, args.reg_weight, losses.penalty)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()

        if epoch % args.eval_every:
            continue

        model.eval()
        # Scoring the full graph on CPU keeps large datasets off the GPU.
        eval_device = device if args.eval_on_gpu else torch.device("cpu")
        model.to(eval_device)
        results = evaluate(model, graph, dataset.valid, tweedie_power, eval_device)

        if stopper.update(results):
            torch.save(model.state_dict(), checkpoint_path)
        print(stopper.describe(epoch, loss.item()))

        model.to(device)
        if stopper.should_stop:
            break

    if not os.path.exists(checkpoint_path):
        print("warning: validation never improved, scoring the final weights instead")
        torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    id_to_entity = {index: name for name, index in dataset.entity_to_id.items()}
    pred_probs, pred_flows = predict(model, graph, dataset.test, device)
    write_predictions(
        os.path.join(run_dir, PREDICTIONS_NAME),
        dataset.test,
        args.variant,
        pred_probs=pred_probs,
        pred_flows=pred_flows,
        id_to_entity=id_to_entity,
    )

    report = {}
    if pred_probs is not None and dataset.test.purpose_probs is not None:
        report["purpose"] = purpose_distribution_metrics(pred_probs, dataset.test.purpose_probs)
    if pred_flows is not None and dataset.test.flows is not None:
        report["flow"] = flow_metrics(pred_flows, dataset.test.flows, scaler)
        report["flow_by_magnitude"] = flow_metrics_by_magnitude(
            pred_flows, dataset.test.flows, scaler
        )

    with open(os.path.join(run_dir, METRICS_NAME), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    _print_report(report)
    print("\nrun directory: {}".format(run_dir))
    return report


def _print_report(report):
    for section, values in report.items():
        print("\n{}:".format(section.replace("_", " ")))
        for key, value in values.items():
            if isinstance(value, list):
                continue
            print("  {:<22}: {:.6f}".format(key, value))


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
