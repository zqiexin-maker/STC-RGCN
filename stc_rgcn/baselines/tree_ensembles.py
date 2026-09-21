"""Tree-ensemble baselines: XGBoost, LightGBM and Random Forest.

Each OD pair is described by five features -- the out/in flow mass of the origin,
the out/in flow mass of the destination and the OD distance -- and one regressor
is fitted per trip purpose plus one for the total flow. The 15 purpose
predictions are then normalized into the predicted purpose distribution.

Distances come from ``grid_distance.csv`` when available and from the entity-name
coordinates otherwise; a constant is used only when neither is available.

Usage::

    stc-rgcn-baseline-trees --model xgboost
    stc-rgcn-baseline-trees --model lightgbm --n-estimators 400
"""

import argparse
import os

import numpy as np

from .common import (
    NUM_PURPOSES,
    add_common_args,
    banner,
    build_purpose_masses,
    evaluate_all,
    flows_to_probs,
    iter_od_records,
    load_entity_dict,
    load_grid_distance,
    od_mass_features,
    resolve_distance_matrix,
    resolve_output,
    resolve_split_file,
    section,
    write_predictions,
)

MODEL_CHOICES = ("xgboost", "lightgbm", "random_forest")


# ───────────────────────────────────────────────────────────────────────────
# Regressors
# ───────────────────────────────────────────────────────────────────────────

def build_regressor(name, params=None):
    """Instantiate one of the supported regressors.

    The heavy libraries are imported lazily so that a missing optional
    dependency only breaks the model that needs it.
    """
    params = params or {}
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_alpha=params.get("reg_alpha", 0),
            reg_lambda=params.get("reg_lambda", 1),
            random_state=42,
            n_jobs=-1,
        )
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", -1),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            num_leaves=params.get("num_leaves", 31),
            min_child_samples=params.get("min_child_samples", 20),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError("unknown model {!r}; choose from {}".format(name, MODEL_CHOICES))


def fit_predict(name, X_train, y_train, X_eval, params=None):
    """Fit ``name`` and return non-negative predictions for ``X_eval``."""
    model = build_regressor(name, params)
    model.fit(X_train, y_train)
    return np.maximum(model.predict(X_eval), 0.0)


# ───────────────────────────────────────────────────────────────────────────
# Pipeline
# ───────────────────────────────────────────────────────────────────────────

def _read_calibration(data_dir, entity2id, eval_path):
    """Read train + evaluation splits once; returns ``(od_ids, purpose_flows)``."""
    ids, flows = [], []
    for path in (resolve_split_file(data_dir, "train"), eval_path):
        if not os.path.exists(path):
            continue
        for origin, dest, _prob, pflows in iter_od_records(path, entity2id):
            ids.append((origin, dest))
            flows.append(pflows)
    return (np.array(ids, dtype=np.int64).reshape(-1, 2),
            np.array(flows, dtype=np.float64).reshape(-1, NUM_PURPOSES))


def _read_split(path, entity2id):
    """Read a split into ``(od_ids, total_flows, purpose_flows)``."""
    ids, flows = [], []
    for origin, dest, _prob, pflows in iter_od_records(path, entity2id):
        ids.append((origin, dest))
        flows.append(pflows)
    flows = np.array(flows, dtype=np.float64).reshape(-1, NUM_PURPOSES)
    return np.array(ids, dtype=np.int64).reshape(-1, 2), flows.sum(axis=1), flows


def _distances(od_ids, dist_matrix):
    od_ids = np.asarray(od_ids, dtype=np.int64)
    raw = dist_matrix[od_ids[:, 0], od_ids[:, 1]]
    return np.where(np.isfinite(raw), raw, 0.1)


def run_ml_model(name, data_dir, output, eval_split="test", params=None,
                 verbose=True):
    """Fit the ensemble for the total flow and each purpose, then score and save."""
    if verbose:
        banner("{} baseline".format(name.upper()))

    section("Loading entity dictionary and distances", 1, 5)
    entity2id, id2entity = load_entity_dict(data_dir)
    dist_dict = load_grid_distance(data_dir)
    dist_matrix = resolve_distance_matrix(id2entity, dist_dict, len(entity2id))
    if verbose:
        print("  entities: {}".format(len(entity2id)))

    section("Reading calibration and evaluation splits", 2, 5)
    eval_path = resolve_split_file(data_dir, eval_split)
    calib_ids, calib_pflows = _read_calibration(data_dir, entity2id, eval_path)
    eval_ids, true_total, true_pflow = _read_split(eval_path, entity2id)
    if verbose:
        print("  calibration rows: {}   evaluation rows: {}".format(
            len(calib_ids), len(eval_ids)))

    section("Building OD features", 3, 5)
    masses = build_purpose_masses(calib_ids, calib_pflows, len(entity2id))
    mass = masses.total()
    X_train = od_mass_features(calib_ids, mass,
                               _distances(calib_ids, dist_matrix))
    X_eval = od_mass_features(eval_ids, mass, _distances(eval_ids, dist_matrix))
    if verbose:
        print("  feature matrix: {}".format(X_train.shape))

    section("Fitting total flow and the {} purposes".format(NUM_PURPOSES), 4, 5)
    pred_total = fit_predict(name, X_train, calib_pflows.sum(axis=1), X_eval, params)
    pred_pflow = np.zeros((len(eval_ids), NUM_PURPOSES), dtype=np.float64)
    for p in range(NUM_PURPOSES):
        flows_p = calib_pflows[:, p]
        keep = flows_p > 0
        if not keep.any():
            print("  purpose {}/{}: no calibration rows, skipped".format(
                p + 1, NUM_PURPOSES))
            continue
        pred_pflow[:, p] = fit_predict(name, X_train[keep], flows_p[keep],
                                       X_eval, params)
        if verbose:
            print("  purpose {}/{} fitted".format(p + 1, NUM_PURPOSES))

    section("Saving results and evaluating", 5, 5)
    pred_probs = flows_to_probs(pred_pflow)
    true_probs = flows_to_probs(true_pflow)
    write_predictions(eval_ids, id2entity, pred_probs, true_probs,
                 pred_total, true_total, output)
    evaluate_all(true_total, pred_total, true_probs, pred_probs,
                 model_name="{} (ML baseline)".format(name.upper()))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="stc-rgcn-baseline-trees",
        description="XGBoost / LightGBM / Random Forest OD baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default="xgboost", choices=MODEL_CHOICES)
    parser.add_argument("--split", default="test", choices=("test", "valid", "train"),
                        help="Split to evaluate (falls back to 'valid' when the "
                             "test split is not shipped with the dataset)")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Omit to use the model-specific default "
                             "(XGBoost 6, LightGBM -1, RandomForest unlimited)")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="XGBoost / LightGBM only")
    parser.add_argument("--subsample", type=float, default=0.8,
                        help="XGBoost / LightGBM only")
    add_common_args(parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    params = {"n_estimators": args.n_estimators,
              "learning_rate": args.learning_rate,
              "subsample": args.subsample}
    if args.max_depth is not None:
        params["max_depth"] = args.max_depth

    output = resolve_output(args, "{}_results".format(args.model))
    run_ml_model(args.model, args.datadir, output,
                 eval_split=args.split, params=params)
    banner("{} prediction completed".format(args.model.upper()))


if __name__ == "__main__":
    main()
