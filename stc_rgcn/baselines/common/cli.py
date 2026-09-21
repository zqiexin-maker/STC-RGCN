"""Shared command-line plumbing for the baseline entry points.

Each baseline is runnable through its console script
(``stc-rgcn-baseline-physics``) or as a module
(``python -m stc_rgcn.baselines.physics``).  Because the package is installed
rather than run from the repository root, no ``sys.path`` manipulation is
needed -- pre-1.0 releases injected the repo root at the top of every entry
point.

:func:`add_common_args` supplies the arguments every baseline shares
(``--datadir``, ``--outdir``/``--output``, ``--seed``) with consistent defaults.
"""

import os

from ...config import DEFAULT_BASELINE_DIR, DEFAULT_DATA_DIR


def add_common_args(parser, with_seed=False, dest="output"):
    """Add the arguments shared by all baselines to ``parser``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
    with_seed : bool
        Also add ``--seed`` (only meaningful for stochastic models).
    dest : {"output", "outdir"}
        ``"output"``   -- add ``--output``, a *file* path (single-file baselines)
        ``"outdir"``   -- add ``--outdir``, a *directory* (baselines that emit
                          several files, e.g. GMEL's intermediates)
    """
    parser.add_argument(
        "--datadir", default=DEFAULT_DATA_DIR,
        help="Dataset directory holding entities.dict and "
             "{train,valid,test}_with_flows.txt "
             "(default: %(default)s)")
    if dest == "outdir":
        parser.add_argument(
            "--outdir", default=DEFAULT_BASELINE_DIR,
            help="Directory for result files (default: %(default)s)")
    elif dest == "output":
        parser.add_argument(
            "--output", default=None,
            help="Result file path (default: <datadir's sibling "
                 "runs/baselines>/<model>_result.txt)")
    else:
        raise ValueError("dest must be 'output' or 'outdir', got {!r}".format(dest))

    if with_seed:
        parser.add_argument("--seed", type=int, default=42,
                            help="Random seed (default: %(default)s)")
    return parser


def resolve_output(args, model_name, dest="output"):
    """Return the output path implied by ``args`` for ``model_name``.

    Fills in ``<result_dir>/<model_name>_result.txt`` when ``--output`` was not
    given, so baselines do not each hard-code their own default filename.
    """
    if dest == "outdir":
        return args.outdir
    if getattr(args, "output", None):
        return args.output
    return os.path.join(DEFAULT_BASELINE_DIR, "{}_result.txt".format(model_name))


def section(title, step=None, total=None):
    """Print a ``[1/5] Loading ...`` style progress header."""
    prefix = "[{}/{}] ".format(step, total) if step and total else ""
    print("\n{}{}".format(prefix, title))


def banner(title, width=70):
    """Print a framed banner around ``title``."""
    print("=" * width)
    print(title)
    print("=" * width)
