"""Stage 1: POI shapefile -> one Doc2Vec vector per grid cell.

Each grid cell is treated as a document whose "words" are the POI category
codes it contains, so cells with a similar POI composition end up close
together.  Cells with too few POIs are dropped and reported.

The POI shapefile is not part of this repository; point ``--poi-shapefile`` at
your own.  It must carry the columns named by ``--grid-column`` and by the
three category columns that are concatenated into one label.
"""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from ..config import DEFAULT_INTERIM_DIR

LOGGER = logging.getLogger(__name__)

#: Columns concatenated into the POI category label that Doc2Vec consumes.
DEFAULT_CATEGORY_COLUMNS = ("gridcode", "level_1", "level_2")

#: Dimensionality of the POI embedding, matched by the trajectory embedding so
#: the two halves of ``features.txt`` are the same width.
DEFAULT_VECTOR_SIZE = 72


def build_poi_embeddings(
    shapefile,
    grid_column="start_ID",
    category_columns=DEFAULT_CATEGORY_COLUMNS,
    vector_size=DEFAULT_VECTOR_SIZE,
    min_pois=2,
    epochs=100,
    window=5,
    seed=42,
):
    """Train Doc2Vec over per-grid POI compositions.

    Returns
    -------
    tuple
        ``(embeddings, report)`` -- a DataFrame of ``grid_id`` plus
        ``vector_size`` columns, and a per-grid quality DataFrame recording
        which cells were kept and why.
    """
    import geopandas as gpd
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from sklearn.preprocessing import LabelEncoder

    pois = gpd.read_file(shapefile)
    missing = [c for c in (grid_column, *category_columns) if c not in pois.columns]
    if missing:
        raise KeyError("{}: missing column(s) {}".format(shapefile, missing))
    LOGGER.info("grids in the shapefile: %d", pois[grid_column].nunique())

    # One label per (gridcode, level_1, level_2) combination.
    label = pois[category_columns[0]].astype(str)
    for column in category_columns[1:]:
        label = label + ";" + pois[column].astype(str)
    pois["poi_category"] = LabelEncoder().fit_transform(label).astype(str)

    report = []
    for grid_id, group in pois.groupby(grid_column):
        reasons = []
        if len(group) < min_pois:
            reasons.append("fewer than {} POIs".format(min_pois))
        if group["poi_category"].isnull().any():
            reasons.append("null category code")
        report.append(
            {
                "grid_id": grid_id,
                "poi_count": len(group),
                "is_valid": not reasons,
                "reasons": "; ".join(reasons),
            }
        )
    report = pd.DataFrame(report)

    valid_ids = report.loc[report["is_valid"], "grid_id"].tolist()
    if not valid_ids:
        raise ValueError("{}: no grid cell passed the quality checks".format(shapefile))
    LOGGER.info("kept %d grids, dropped %d", len(valid_ids), len(report) - len(valid_ids))

    categories_by_grid = pois.groupby(grid_column)["poi_category"].apply(list)
    corpus = [TaggedDocument(categories_by_grid[grid_id], [grid_id]) for grid_id in valid_ids]

    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=1,
        dm=1,           # PV-DM
        dm_mean=1,      # average, rather than sum, the context vectors
        epochs=epochs,
        seed=seed,
    )
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

    rows, missing_vectors = [], []
    for grid_id in valid_ids:
        try:
            rows.append([grid_id] + model.dv[grid_id].tolist())
        except KeyError:
            missing_vectors.append(grid_id)
    if missing_vectors:
        LOGGER.warning("%d valid grids received no vector", len(missing_vectors))

    columns = ["grid_id"] + ["poi_{}".format(i) for i in range(vector_size)]
    return pd.DataFrame(rows, columns=columns), report


def build_parser():
    parser = argparse.ArgumentParser(
        prog="stc-rgcn-poi-embedding",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--poi-shapefile", required=True,
                        help="Input POI shapefile (not shipped with the repository)")
    parser.add_argument("--output", default=os.path.join(DEFAULT_INTERIM_DIR, "poi_embeddings.txt"),
                        help="Tab-separated output file")
    parser.add_argument("--report", default=None,
                        help="Optional CSV recording which grids were dropped and why")
    parser.add_argument("--grid-column", default="start_ID", help="Grid id column")
    parser.add_argument("--vector-size", type=int, default=DEFAULT_VECTOR_SIZE,
                        help="Embedding dimensionality")
    parser.add_argument("--min-pois", type=int, default=2,
                        help="Minimum POIs for a grid cell to be embedded")
    parser.add_argument("--epochs", type=int, default=100, help="Doc2Vec training epochs")
    parser.add_argument("--window", type=int, default=5, help="Doc2Vec context window")
    parser.add_argument("--seed", type=int, default=42, help="Doc2Vec random seed")
    return parser


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    embeddings, report = build_poi_embeddings(
        shapefile=args.poi_shapefile,
        grid_column=args.grid_column,
        vector_size=args.vector_size,
        min_pois=args.min_pois,
        epochs=args.epochs,
        window=args.window,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    embeddings.to_csv(args.output, sep="\t", index=False, float_format="%.6f")
    print("wrote {} grid embeddings to {}".format(len(embeddings), args.output))

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        report.to_csv(args.report, index=False, encoding="utf-8-sig")
        print("wrote the quality report to {}".format(args.report))


if __name__ == "__main__":
    main()
