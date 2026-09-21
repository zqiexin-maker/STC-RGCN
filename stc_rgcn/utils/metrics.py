"""Prediction metrics for both tasks.

One numpy implementation serves the training loop, the standalone scorer and
the baselines, so a number reported by ``stc-rgcn-train`` and the same number
reported by ``stc-rgcn-eval score`` cannot drift apart.

Degenerate inputs (a constant target, an empty group) yield ``nan`` for the
affected metric rather than a sentinel value of a different type.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .transforms import LogMinMaxScaler

#: Relative-error bands reported as "share of predictions within X%".
ACCURACY_BANDS = (0.0, 0.10, 0.20)

#: Guards divisions by a zero target when forming relative errors.
RELATIVE_ERROR_EPSILON = 1e-6


def as_array(values):
    """Convert torch tensors, lists or arrays to a flat float64 numpy array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64).ravel()


def as_matrix(values):
    """Convert to a 2-D float64 numpy array, keeping the row/column layout."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.atleast_2d(np.asarray(values, dtype=np.float64))


# ── Scalar helpers ─────────────────────────────────────────────────────────
def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def mse(pred, true):
    return float(np.mean((pred - true) ** 2))


def rmse(pred, true):
    return float(np.sqrt(mse(pred, true)))


def r_squared(pred, true):
    """Coefficient of determination; ``nan`` when the target has no variance."""
    ss_total = float(np.sum((true - np.mean(true)) ** 2))
    if ss_total <= 0:
        return float("nan")
    return float(1.0 - np.sum((true - pred) ** 2) / ss_total)


def pearson_r2(pred, true):
    """Squared Pearson correlation; ``nan`` when either side is constant."""
    pred_centered = pred - np.mean(pred)
    true_centered = true - np.mean(true)
    denominator = np.sqrt(np.sum(pred_centered**2)) * np.sqrt(np.sum(true_centered**2))
    if denominator < 1e-8:
        return float("nan")
    return float((np.sum(pred_centered * true_centered) / denominator) ** 2)


def mape(pred, true):
    """Mean absolute percentage error over the non-zero targets."""
    nonzero = true != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((pred[nonzero] - true[nonzero]) / true[nonzero])) * 100)


def common_part_of_commuters(pred, true):
    """CPC = ``2 * sum(min(pred, true)) / sum(pred + true)``, in ``[0, 1]``."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    total = float(np.sum(pred) + np.sum(true))
    if total <= 0:
        return float("nan")
    return float(2.0 * np.sum(np.minimum(pred, true)) / total)


def _spearman(pred, true):
    if len(pred) < 2:
        return float("nan"), float("nan")
    correlation, p_value = stats.spearmanr(pred, true)
    return float(correlation), float(p_value)


def _pearson(pred, true):
    if len(pred) < 2:
        return float("nan"), float("nan")
    correlation, p_value = stats.pearsonr(pred, true)
    return float(correlation), float(p_value)


# ── Task metrics ───────────────────────────────────────────────────────────
def purpose_distribution_metrics(pred_probs, true_probs, epsilon=1e-10):
    """Metrics for the 15-dimensional trip-purpose distribution.

    Returns cosine similarity, MSE, MAE, RMSE, batch-mean KL divergence and
    the per-purpose :math:`R^2` list.
    """
    pred = np.clip(as_matrix(pred_probs), epsilon, None)
    true = np.clip(as_matrix(true_probs), epsilon, None)

    norms = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)
    cosine = np.divide(
        np.sum(pred * true, axis=1), norms, out=np.full(len(pred), np.nan), where=norms > 0
    )

    return {
        "cosine_sim": float(np.nanmean(cosine)),
        "mse": mse(pred, true),
        "mae": mae(pred, true),
        "rmse": rmse(pred, true),
        "kl": float(np.mean(np.sum(true * np.log(true / pred), axis=1))),
        "r2_per_purpose": [
            r_squared(pred[:, i], true[:, i]) for i in range(true.shape[1])
        ],
    }


def flow_metrics(pred_flows, true_flows, scaler=None):
    """Metrics for the total-flow target, on both scales.

    When ``scaler`` is a fitted :class:`~stc_rgcn.utils.transforms.LogMinMaxScaler`,
    the inputs are treated as normalized and are inverted back to raw flows;
    the ``norm_*`` keys then describe the training objective and the ``raw_*``
    keys the real-world scale.  Without a scaler both scales coincide.
    """
    scaler = scaler or LogMinMaxScaler.identity()
    pred_norm = as_array(pred_flows)
    true_norm = as_array(true_flows)
    pred_raw = scaler.inverse_transform(pred_norm)
    true_raw = scaler.inverse_transform(true_norm)

    spearman, spearman_p = _spearman(pred_norm, true_norm)
    pearson, pearson_p = _pearson(pred_norm, true_norm)

    relative_error = np.abs(pred_raw - true_raw) / (np.abs(true_raw) + RELATIVE_ERROR_EPSILON)
    accuracy = {
        "accuracy_{:.0f}%".format(band * 100): float(np.mean(relative_error <= band) * 100)
        for band in ACCURACY_BANDS
    }

    return {
        # Normalized scale: what the loss actually optimizes.
        "norm_mae": mae(pred_norm, true_norm),
        "norm_mse": mse(pred_norm, true_norm),
        "norm_rmse": rmse(pred_norm, true_norm),
        "norm_r2": pearson_r2(pred_norm, true_norm),
        # Named for what it is: the pre-1.0 key "norm_cpc" held a Pearson r,
        # not a common part of commuters. The real CPC is the "cpc" key below.
        "norm_scc": spearman,
        "norm_scc_pvalue": spearman_p,
        "norm_pearson": pearson,
        "norm_pearson_pvalue": pearson_p,
        # Raw scale: comparable across runs with different normalizations.
        "raw_mae": mae(pred_raw, true_raw),
        "raw_mse": mse(pred_raw, true_raw),
        "raw_rmse": rmse(pred_raw, true_raw),
        "raw_mape": mape(pred_raw, true_raw),
        "raw_r2": pearson_r2(pred_raw, true_raw),
        "cpc": common_part_of_commuters(pred_raw, true_raw),
        **accuracy,
    }


def flow_metrics_by_magnitude(pred_flows, true_flows, scaler=None, threshold=30.0):
    """Split the flow metrics into large and small OD pairs.

    Reported separately because a model can score well overall while getting
    every high-volume pair wrong, which is what ``threshold`` (raw flow units)
    separates.
    """
    scaler = scaler or LogMinMaxScaler.identity()
    pred_raw = scaler.inverse_transform(as_array(pred_flows))
    true_raw = scaler.inverse_transform(as_array(true_flows))

    result = {}
    for label, mask in (
        ("large", true_raw > threshold),
        ("small", true_raw <= threshold),
    ):
        pred_group, true_group = pred_raw[mask], true_raw[mask]
        result["{}_count".format(label)] = int(mask.sum())
        correlation, _ = _spearman(pred_group, true_group)
        result["{}_scc".format(label)] = correlation
        result["{}_cpc".format(label)] = (
            common_part_of_commuters(pred_group, true_group) if mask.sum() >= 2 else float("nan")
        )

    result["mean_cpc_large_small"] = float(
        np.nanmean([result["large_cpc"], result["small_cpc"]])
    )
    return result
