"""Stage 2: join OD records, entities and POI features into a model dataset.

Keeps the entities that appear in both the OD records and the POI feature
file, renumbers them from 0, writes the matching feature file, and splits the
surviving OD rows into train/valid/test.

The entity dictionary and the feature file are written in the same row order,
because the loader matches them by position.
"""

from __future__ import annotations

import argparse
import os

from ..utils.schema import COL_DESTINATION, COL_ORIGIN, ENTITY_DICT, SPLIT_FILES

DEFAULT_SPLIT_RATIOS = (0.6, 0.2, 0.2)
DEFAULT_SEED = 42


def _read_od_endpoints(path):
    """Collect every entity string that appears as an origin or a destination."""
    endpoints = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > COL_DESTINATION:
                endpoints.update([parts[COL_ORIGIN], parts[COL_DESTINATION]])
    return endpoints


def _read_keyed_rows(path):
    """Read ``<key>\\t<rest>`` rows into ``{key: rest}``."""
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                rows[parts[0]] = "\t".join(parts[1:])
    return rows


def build_dataset(
    od_path,
    entity_path,
    feature_path,
    poi_feature_path,
    output_dir,
    split_ratios=DEFAULT_SPLIT_RATIOS,
    seed=DEFAULT_SEED,
):
    """Build ``entities.dict``, ``features.txt`` and the three split files.

    Parameters
    ----------
    od_path
        OD records in the 40-column layout.
    entity_path, feature_path
        Parallel files: row *i* of ``feature_path`` describes the entity on row
        *i* of ``entity_path``.
    poi_feature_path
        ``<entity>\\t<features...>`` POI features, joined onto each entity.
    split_ratios
        Train/valid/test fractions; must sum to 1.
    """
    from sklearn.model_selection import train_test_split

    if abs(sum(split_ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1, got {}".format(split_ratios))
    os.makedirs(output_dir, exist_ok=True)

    print("[1/4] reading OD endpoints")
    od_entities = _read_od_endpoints(od_path)

    print("[2/4] joining POI features onto the entity list")
    poi_features = _read_keyed_rows(poi_feature_path)

    kept_entities, kept_features = [], []
    with open(entity_path, encoding="utf-8") as entities, \
            open(feature_path, encoding="utf-8") as features:
        for entity_line, feature_line in zip(entities, features):
            parts = entity_line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            entity = parts[1]
            # An entity survives only with both OD records and POI features.
            if entity in od_entities and entity in poi_features:
                kept_entities.append(entity)
                kept_features.append(
                    "{}\t{}".format(feature_line.rstrip("\n"), poi_features[entity])
                )

    if not kept_entities:
        raise ValueError("no entity appears in both the OD records and the POI features")

    with open(os.path.join(output_dir, ENTITY_DICT), "w", encoding="utf-8") as handle:
        handle.write("\n".join("{}\t{}".format(i, e) for i, e in enumerate(kept_entities)))
    with open(os.path.join(output_dir, "features.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(kept_features))

    print("[3/4] filtering OD rows to the surviving entities")
    valid = set(kept_entities)
    rows = []
    with open(od_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if (
                len(parts) > COL_DESTINATION
                and parts[COL_ORIGIN] in valid
                and parts[COL_DESTINATION] in valid
            ):
                rows.append(line)
    if not rows:
        raise ValueError("no OD row survived entity filtering")

    print("[4/4] splitting {} rows {}".format(len(rows), split_ratios))
    train_ratio, valid_ratio, test_ratio = split_ratios
    train, remainder = train_test_split(
        rows, test_size=valid_ratio + test_ratio, random_state=seed
    )
    valid_rows, test_rows = train_test_split(
        remainder, test_size=test_ratio / (valid_ratio + test_ratio), random_state=seed
    )

    for name, split_rows in (("train", train), ("valid", valid_rows), ("test", test_rows)):
        path = os.path.join(output_dir, SPLIT_FILES[name][0])
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(split_rows))

    print(
        "wrote {} entities and {}/{}/{} train/valid/test rows to {}".format(
            len(kept_entities), len(train), len(valid_rows), len(test_rows), output_dir
        )
    )
    return {
        "entities": len(kept_entities),
        "train": len(train),
        "valid": len(valid_rows),
        "test": len(test_rows),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        prog="stc-rgcn-build-dataset",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--od-records", required=True,
                        help="OD records in the 40-column layout")
    parser.add_argument("--entities", required=True, help="<id>\\t<entity> dictionary")
    parser.add_argument("--features", required=True,
                        help="Landuse features, one row per entity, aligned with --entities")
    parser.add_argument("--poi-features", required=True,
                        help="<entity>\\t<features...> POI features from stage 1")
    parser.add_argument("--output-dir", required=True, help="Directory to write the dataset into")
    parser.add_argument("--split-ratios", type=float, nargs=3, default=list(DEFAULT_SPLIT_RATIOS),
                        metavar=("TRAIN", "VALID", "TEST"), help="Split fractions, summing to 1")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Split random seed")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    build_dataset(
        od_path=args.od_records,
        entity_path=args.entities,
        feature_path=args.features,
        poi_feature_path=args.poi_features,
        output_dir=args.output_dir,
        split_ratios=tuple(args.split_ratios),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
