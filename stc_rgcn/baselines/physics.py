"""Physics-inspired OD baselines: gravity models and the radiation model.

Four closed-form spatial-interaction models, all fitted per trip purpose and for
the total flow:

===================  =========================================================
``gm-o``  Osgood     ``T = K * m_i * m_j / d^beta``
``gm-p``  power law  ``T = K * m_i^b0 * m_j^b1 * d^b2``
``gm-e``  exponential ``T = K * m_i^b0 * m_j^b1 * exp(b2 * d)``
``radiation``        ``T = scale * m_i * n_j / ((m_i + s_ij) * (m_i + n_j + s_ij))``
===================  =========================================================

``m_i``/``m_j`` are the out/in flow masses of the origin and destination
(:class:`stc_rgcn.baselines.common.features.NodeMass`), ``d`` the OD distance in km, and
``s_ij`` the intervening opportunities inside ``d(i, j)``
(:func:`stc_rgcn.baselines.common.features.intervening_opportunities`).

All four share one pipeline (:func:`run_model`): fit on the calibration splits
(train + test, as in the published setup), predict the test split, convert the
15 purpose flows into a distribution, then score and save the standard
34-column result file.

Usage::

    stc-rgcn-baseline-physics --model gm-o
    stc-rgcn-baseline-physics --model radiation --output rad.txt
"""

import os

import numpy as np

from .common import (
    NUM_PURPOSES,
    add_common_args,
    banner,
    build_purpose_masses,
    evaluate_all,
    flows_to_probs,
    intervening_opportunities,
    iter_od_records,
    load_entity_dict,
    load_grid_distance,
    read_od_split,
    resolve_distance_matrix,
    resolve_output,
    resolve_split_file,
    section,
    write_predictions,
)

DEFAULT_DISTANCE = 0.1      # km, used when an OD pair is missing from grid_distance.csv


# ───────────────────────────────────────────────────────────────────────────
# Fitting context
# ───────────────────────────────────────────────────────────────────────────

class ODContext:
    """Everything a fitted model needs about the study area.

    Holds the entity dictionaries, the OD distance matrix and the zone masses
    (which change between the total-flow run and each per-purpose run).

    Distances come from ``grid_distance.csv`` when present, otherwise from the
    coordinates embedded in the entity names, otherwise from a constant -- see
    :func:`stc_rgcn.baselines.common.features.resolve_distance_matrix`.
    """

    def __init__(self, data_dir, entity2id, id2entity, dist_dict):
        self.data_dir = data_dir
        self.entity2id = entity2id
        self.id2entity = id2entity
        self.dist_dict = dist_dict
        self.num_entities = len(entity2id)
        self.mass = None
        self._dist_matrix = None

    @property
    def dist_matrix(self):
        """Lazily materialized ``(E, E)`` distance matrix (km)."""
        if self._dist_matrix is None:
            self._dist_matrix = resolve_distance_matrix(
                self.id2entity, self.dist_dict, self.num_entities)
        return self._dist_matrix

    def distances(self, od_ids):
        """OD distances (km) for an ``(N, 2)`` id matrix.

        Unknown pairs collapse to :data:`DEFAULT_DISTANCE`; the gravity models
        fit in log space and cannot handle ``inf`` or zero distances.
        """
        od_ids = np.asarray(od_ids, dtype=np.int64)
        raw = self.dist_matrix[od_ids[:, 0], od_ids[:, 1]]
        return np.clip(np.where(np.isfinite(raw), raw, DEFAULT_DISTANCE),
                       DEFAULT_DISTANCE, None)


# ───────────────────────────────────────────────────────────────────────────
# Shared helpers
# ───────────────────────────────────────────────────────────────────────────

def _ols(X, y):
    """Least squares with intercept; returns ``(coef, intercept)``.

    Uses ``sklearn.linear_model.LinearRegression`` when available and falls back
    to ``numpy.linalg.lstsq`` so the baselines run without scikit-learn.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    try:
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(X, y)
        return np.asarray(reg.coef_), float(reg.intercept_)
    except ImportError:
        design = np.column_stack([X, np.ones(len(X))])
        solution, *_ = np.linalg.lstsq(design, y, rcond=None)
        return solution[:-1], float(solution[-1])


def _gravity_design(od_ids, flows, ctx, min_mass=1e-12):
    """Rows usable by a log-space gravity fit.

    Keeps only pairs with positive flow, positive distance and positive masses
    at both endpoints -- log-space fits cannot use anything else.

    Returns ``(log_m_i, log_m_j, distance, flow)`` arrays.
    """
    od_ids = np.asarray(od_ids, dtype=np.int64)
    flows = np.asarray(flows, dtype=np.float64)
    origin, dest = od_ids[:, 0], od_ids[:, 1]
    distance = ctx.distances(od_ids)
    m_i, m_j = ctx.mass.out_flow[origin], ctx.mass.in_flow[dest]

    keep = ((flows > 0) & (distance > 0)
            & (m_i > min_mass) & (m_j > min_mass))
    return np.log(m_i[keep]), np.log(m_j[keep]), distance[keep], flows[keep]


# ───────────────────────────────────────────────────────────────────────────
# gm-o : T = K * m_i * m_j / d^beta
# ───────────────────────────────────────────────────────────────────────────

def fit_gm_o(od_ids, flows, ctx):
    """Fit ``beta`` and ``K``; returns ``(beta, K)``."""
    log_m_i, log_m_j, distance, flow = _gravity_design(od_ids, flows, ctx)
    if len(flow) == 0:
        raise ValueError("gm-o: no usable training rows")
    # log(m_i * m_j / T) = beta * log(d) - log(K)
    y = (log_m_i + log_m_j) - np.log(flow)
    x = np.log(distance)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(np.exp(-intercept))


def predict_gm_o(od_ids, ctx, params):
    beta, K = params
    od_ids = np.asarray(od_ids, dtype=np.int64)
    distance = ctx.distances(od_ids)
    m_i = ctx.mass.out_flow[od_ids[:, 0]]
    m_j = ctx.mass.in_flow[od_ids[:, 1]]
    return K * m_i * m_j / np.power(distance, beta)


# ───────────────────────────────────────────────────────────────────────────
# gm-p : T = K * m_i^b0 * m_j^b1 * d^b2
# ───────────────────────────────────────────────────────────────────────────

def fit_gm_p(od_ids, flows, ctx):
    """Fit the power-law form; returns ``(coefs, K)``."""
    log_m_i, log_m_j, distance, flow = _gravity_design(od_ids, flows, ctx)
    if len(flow) == 0:
        raise ValueError("gm-p: no usable training rows")
    X = np.column_stack([log_m_i, log_m_j, np.log(distance)])
    coefs, intercept = _ols(X, np.log(flow))
    return coefs, float(np.exp(intercept))


def predict_gm_p(od_ids, ctx, params):
    coefs, K = params
    od_ids = np.asarray(od_ids, dtype=np.int64)
    distance = ctx.distances(od_ids)
    m_i = ctx.mass.out_flow[od_ids[:, 0]]
    m_j = ctx.mass.in_flow[od_ids[:, 1]]
    return K * np.power(m_i, coefs[0]) * np.power(m_j, coefs[1]) * np.power(distance, coefs[2])


# ───────────────────────────────────────────────────────────────────────────
# gm-e : T = K * m_i^b0 * m_j^b1 * exp(b2 * d)
# ───────────────────────────────────────────────────────────────────────────

def fit_gm_e(od_ids, flows, ctx):
    """Fit the exponential-decay form; returns ``(coefs, K)``.

    ``coefs[2]`` is expected to come out negative: flow decays with distance.
    """
    log_m_i, log_m_j, distance, flow = _gravity_design(od_ids, flows, ctx)
    if len(flow) == 0:
        raise ValueError("gm-e: no usable training rows")
    X = np.column_stack([log_m_i, log_m_j, distance])     # distance enters raw
    coefs, intercept = _ols(X, np.log(flow))
    return coefs, float(np.exp(intercept))


def predict_gm_e(od_ids, ctx, params):
    coefs, K = params
    od_ids = np.asarray(od_ids, dtype=np.int64)
    distance = ctx.distances(od_ids)
    m_i = ctx.mass.out_flow[od_ids[:, 0]]
    m_j = ctx.mass.in_flow[od_ids[:, 1]]
    return K * np.power(m_i, coefs[0]) * np.power(m_j, coefs[1]) * np.exp(coefs[2] * distance)


# ───────────────────────────────────────────────────────────────────────────
# radiation : T = scale * m_i * n_j / ((m_i + s_ij) * (m_i + n_j + s_ij))
# ───────────────────────────────────────────────────────────────────────────

def _radiation_raw(od_ids, ctx):
    """Unscaled radiation probabilities for each OD pair."""
    od_ids = np.asarray(od_ids, dtype=np.int64)
    origin, dest = od_ids[:, 0], od_ids[:, 1]

    # The circle radii are the true OD distances when known, otherwise the
    # synthetic-grid fallback; unknown pairs are never "inside" the circle.
    dist_matrix = ctx.dist_matrix
    radius = dist_matrix[origin, dest]
    known = np.isfinite(radius)
    radius = np.where(known, radius, DEFAULT_DISTANCE)

    s_ij = intervening_opportunities(od_ids, dist_matrix, ctx.mass.mass)
    m_i = ctx.mass.out_flow[origin]
    n_j = ctx.mass.in_flow[dest]

    denom1 = m_i + s_ij
    denom2 = m_i + n_j + s_ij
    safe = (denom1 > 0) & (denom2 > 0)
    return np.where(safe, m_i * n_j / np.where(safe, denom1 * denom2, 1.0), 0.0)


def fit_radiation(od_ids, flows, ctx):
    """Calibrate the radiation model; returns ``(scale,)``.

    The formula yields tiny unscaled values, so one multiplicative constant is
    fitted on the calibration splits. The scale is the least-squares solution
    ``sum(raw * flow) / sum(raw^2)`` rather than a mass-preserving ratio: flows
    are heavy-tailed, and a ratio fitted to the total mass lets a handful of
    high-``raw`` pairs dominate the error.
    """
    raw = _radiation_raw(od_ids, ctx)
    flows = np.asarray(flows, dtype=np.float64)
    denominator = float(np.sum(raw ** 2))
    if denominator <= 1e-12:
        return (1.0,)
    return (float(np.sum(raw * flows) / denominator),)


def predict_radiation(od_ids, ctx, params):
    (scale,) = params
    return _radiation_raw(od_ids, ctx) * scale


MODEL_REGISTRY = {
    "gm-o": (fit_gm_o, predict_gm_o),
    "gm-p": (fit_gm_p, predict_gm_p),
    "gm-e": (fit_gm_e, predict_gm_e),
    "radiation": (fit_radiation, predict_radiation),
}


# ───────────────────────────────────────────────────────────────────────────
# Pipeline
# ───────────────────────────────────────────────────────────────────────────

def _read_calibration(data_dir, entity2id, eval_path):
    """Read the calibration splits once; returns ``(od_ids, purpose_flows)``.

    Following the published setup the classical models are calibrated on the
    training split *plus* the evaluation split, so ``eval_path`` is read twice
    (once here, once for scoring).
    """
    ids, flows = [], []
    for path in (resolve_split_file(data_dir, "train"), eval_path):
        if not os.path.exists(path):
            continue
        for origin, dest, _prob, pflows in iter_od_records(path, entity2id):
            ids.append((origin, dest))
            flows.append(pflows)
    return (np.array(ids, dtype=np.int64).reshape(-1, 2),
            np.array(flows, dtype=np.float64).reshape(-1, NUM_PURPOSES))


def run_model(name, data_dir, output, eval_split="test", verbose=True):
    """Fit ``name``, predict the evaluation split, then score and save.

    One model is fitted for the total flow and one for each of the 15 trip
    purposes; the per-purpose predictions are normalized into the predicted
    purpose distribution.
    """
    fit_fn, predict_fn = MODEL_REGISTRY[name]

    if verbose:
        banner("{} baseline".format(name.upper()))

    section("Loading entity dictionary and distances", 1, 5)
    entity2id, id2entity = load_entity_dict(data_dir)
    dist_dict = load_grid_distance(data_dir)
    ctx = ODContext(data_dir, entity2id, id2entity, dist_dict)
    if verbose:
        source = ("grid_distance.csv" if dist_dict
                  else "entity-name coordinates" if ctx.dist_matrix.std() > 0
                  else "constant")
        print("  entities: {}   distance source: {}".format(len(entity2id), source))

    section("Reading calibration and evaluation splits", 2, 5)
    eval_path = resolve_split_file(data_dir, eval_split)
    calib_ids, calib_pflows = _read_calibration(data_dir, entity2id, eval_path)
    eval_ids, true_total, true_pflow, _ = read_od_split(eval_path, entity2id)
    n_eval = len(eval_ids)
    if verbose:
        print("  calibration rows: {}   evaluation rows: {}".format(
            len(calib_ids), n_eval))

    section("Building per-purpose zone masses", 3, 5)
    masses = build_purpose_masses(calib_ids, calib_pflows, len(entity2id))

    section("Fitting total flow and the {} purposes".format(NUM_PURPOSES), 4, 5)
    ctx.mass = masses.total()
    params = fit_fn(calib_ids, calib_pflows.sum(axis=1), ctx)
    pred_total = np.maximum(predict_fn(eval_ids, ctx, params), 0.0)
    if verbose:
        print("  total-flow params: {}".format(params))

    pred_pflow = np.zeros((n_eval, NUM_PURPOSES), dtype=np.float64)
    for p in range(NUM_PURPOSES):
        flows_p = calib_pflows[:, p]
        # only pairs carrying this purpose can be used to fit it
        keep = flows_p > 0
        if not keep.any():
            print("  purpose {}/{}: no calibration rows, skipped".format(
                p + 1, NUM_PURPOSES))
            continue
        ctx.mass = masses.for_purpose(p)
        params_p = fit_fn(calib_ids[keep], flows_p[keep], ctx)
        pred_pflow[:, p] = np.maximum(predict_fn(eval_ids, ctx, params_p), 0.0)
        if verbose:
            print("  purpose {}/{} fitted".format(p + 1, NUM_PURPOSES))

    section("Saving results and evaluating", 5, 5)
    pred_probs = flows_to_probs(pred_pflow)
    true_probs = flows_to_probs(true_pflow)
    write_predictions(eval_ids, id2entity, pred_probs, true_probs,
                 pred_total, true_total, output)
    evaluate_all(true_total, pred_total, true_probs, pred_probs,
                 model_name="{} (physics baseline)".format(name.upper()))


def parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="stc-rgcn-baseline-physics",
        description="Gravity (GM-O / GM-P / GM-E) and radiation OD baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default="gm-o", choices=sorted(MODEL_REGISTRY),
                        help="Which spatial-interaction model to fit")
    parser.add_argument("--split", default="test", choices=("test", "valid", "train"),
                        help="Split to evaluate (falls back to 'valid' when the "
                             "test split is not shipped with the dataset)")
    add_common_args(parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = resolve_output(args, "physics_{}".format(args.model.replace("-", "_")))
    run_model(args.model, args.datadir, output, eval_split=args.split)
    banner("{} prediction completed".format(args.model.upper()))


if __name__ == "__main__":
    main()
