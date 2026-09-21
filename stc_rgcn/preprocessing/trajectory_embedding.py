"""Stage 4: synthesized trajectories -> one Word2Vec vector per grid cell.

Each trajectory is read as a sentence of grid ids, so cells that recur in
similar movement contexts end up close together.  Skip-gram is used because
the corpus is small relative to its vocabulary.
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd

from ..config import DEFAULT_INTERIM_DIR

#: Trajectory entries look like ``12345(回家 08:00-09:00)``; the grid id is the
#: leading run of digits before the opening parenthesis.
_GRID_PATTERN = re.compile(r"(\d+)\(")

DEFAULT_VECTOR_SIZE = 72


def extract_grid_sequence(trajectory):
    """Pull the ordered grid ids out of one trajectory string."""
    return _GRID_PATTERN.findall(str(trajectory))


def build_trajectory_embeddings(
    trajectories,
    vector_size=DEFAULT_VECTOR_SIZE,
    window=10,
    min_count=1,
    workers=4,
    epochs=5,
    seed=42,
):
    """Train skip-gram Word2Vec over grid sequences.

    Parameters
    ----------
    trajectories
        DataFrame with a ``trajectory`` column, as written by stage 3.

    Returns
    -------
    pandas.DataFrame
        ``grid_id`` plus one column per embedding dimension.
    """
    from gensim.models import Word2Vec

    if "trajectory" not in trajectories.columns:
        raise KeyError("the trajectory table needs a 'trajectory' column")

    sentences = [extract_grid_sequence(t) for t in trajectories["trajectory"]]
    sentences = [s for s in sentences if s]
    if not sentences:
        raise ValueError("no grid ids could be parsed from the trajectories")

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,  # skip-gram
        workers=workers,
        epochs=epochs,
        seed=seed,
    )

    grid_ids = sorted({grid for sentence in sentences for grid in sentence}, key=int)
    columns = ["grid_id"] + ["traj_{}".format(i) for i in range(vector_size)]
    rows = [[int(grid)] + model.wv[grid].tolist() for grid in grid_ids]
    return pd.DataFrame(rows, columns=columns)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="stc-rgcn-traj-embedding",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trajectories",
                        default=os.path.join(DEFAULT_INTERIM_DIR, "trajectories.csv"),
                        help="Trajectory CSV written by stage 3")
    parser.add_argument("--output",
                        default=os.path.join(DEFAULT_INTERIM_DIR, "trajectory_embeddings.csv"),
                        help="Output CSV of grid embeddings")
    parser.add_argument("--vector-size", type=int, default=DEFAULT_VECTOR_SIZE,
                        help="Embedding dimensionality")
    parser.add_argument("--window", type=int, default=10, help="Word2Vec context window")
    parser.add_argument("--epochs", type=int, default=5, help="Word2Vec training epochs")
    parser.add_argument("--workers", type=int, default=4, help="Worker threads")
    parser.add_argument("--seed", type=int, default=42, help="Word2Vec random seed")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    embeddings = build_trajectory_embeddings(
        pd.read_csv(args.trajectories),
        vector_size=args.vector_size,
        window=args.window,
        workers=args.workers,
        epochs=args.epochs,
        seed=args.seed,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    embeddings.to_csv(args.output, index=False, float_format="%.6f")
    print("wrote {} grid embeddings to {}".format(len(embeddings), args.output))


if __name__ == "__main__":
    main()
