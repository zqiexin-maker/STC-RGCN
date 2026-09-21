"""``stc-rgcn-eval``: score prediction files or analyse an ablation table.

Two subcommands::

    stc-rgcn-eval score --folder runs/baselines
    stc-rgcn-eval contribution --csv ablation_results.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from ..config import DEFAULT_BASELINE_DIR
from ..utils.transforms import LogMinMaxScaler
from .contribution import CONTRIBUTION_METRICS, module_contributions
from .scoring import score_folder


def build_parser():
    parser = argparse.ArgumentParser(
        prog="stc-rgcn-eval",
        description="Score STC-RGCN and baseline predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    score = subcommands.add_parser("score", help="Score a folder of prediction files")
    score.add_argument("--folder", default=DEFAULT_BASELINE_DIR,
                       help="Directory of prediction files")
    score.add_argument("--pattern", default="*.txt", help="Glob pattern inside --folder")
    score.add_argument("--log-transform", action="store_true",
                       help="Predictions are log + min-max normalized; also pass "
                            "--log-min/--log-max to invert the transform")
    score.add_argument("--log-min", type=float, default=0.0,
                       help="log_min used at training time")
    score.add_argument("--log-max", type=float, default=1.0,
                       help="log_max used at training time")
    score.add_argument("--magnitude-threshold", type=float, default=30.0,
                       help="Raw flow separating the large and small groups")
    score.add_argument("--no-group", action="store_true",
                       help="Do not group repeats of the same model")
    score.add_argument("--per-file", action="store_true",
                       help="Also print metrics for each individual file")
    score.set_defaults(handler=_run_score)

    contribution = subcommands.add_parser(
        "contribution", help="Module contributions from an ablation table"
    )
    contribution.add_argument("--csv", required=True,
                              help="Ablation results CSV with model/task columns")
    contribution.add_argument("--baseline", default="base",
                              help="Model whose 'single' row is the reference")
    contribution.add_argument("--metrics", nargs="+",
                              default=["kl", "cosine_sim", "raw_mae", "cpc"],
                              choices=sorted(CONTRIBUTION_METRICS),
                              help="Metrics to average into the contribution")
    contribution.add_argument("--out", default=None,
                              help="Write the table to this CSV as well as printing it")
    contribution.set_defaults(handler=_run_contribution)

    return parser


def _run_score(args):
    scaler = (
        LogMinMaxScaler(log_min=args.log_min, log_max=args.log_max)
        if args.log_transform
        else LogMinMaxScaler.identity()
    )
    score_folder(
        args.folder,
        pattern=args.pattern,
        scaler=scaler,
        group=not args.no_group,
        per_file=args.per_file,
    )


def _run_contribution(args):
    table = module_contributions(args.csv, baseline_model=args.baseline, metrics=args.metrics)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("module contributions (baseline: {}(single))\n".format(args.baseline))
    print(table.to_string(index=False))

    if args.out:
        table.to_csv(args.out, index=False, encoding="utf-8-sig")
        print("\nwritten to {}".format(args.out))


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
