"""Evaluation metrics shared by every baseline.

Flow metrics (``mae``, ``rmse``, ``mape``, ``r2``, ``cpc``, ``cpl``) and
distribution metrics (``kl_divergence``, ``jsd``, ``js_divergence``,
``cosine_similarity``, ``per_purpose_mae``) live here so that the baselines,
GMEL and the reporting scripts all score predictions the same way. Everything
is numpy and scikit-learn is optional -- torch-free, so importing this module
never pulls in a deep-learning framework.

``evaluate_all`` is the one-call summary printer; torch-side training losses stay
inside the models that need them.
"""

import numpy as np

try:
    from sklearn.metrics import mean_absolute_error, r2_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ───────────────────────────────────────────────────────────────────────────
# Flow prediction
# ───────────────────────────────────────────────────────────────────────────

def mae(y_true, y_pred):
    """Mean absolute error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if _SKLEARN_AVAILABLE:
        return float(mean_absolute_error(y_true, y_pred))
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Root mean squared error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps=1e-8):
    """Mean absolute percentage error (percent). Zero true values are skipped."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.abs(y_true) > eps
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2(y_true, y_pred):
    """Coefficient of determination."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if _SKLEARN_AVAILABLE:
        return float(r2_score(y_true, y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


def cpc(y_pred, y_true):
    """Common Part of Commuters over the whole flow vector (0..1, higher better)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    total = np.sum(y_true) + np.sum(y_pred)
    if total <= 0:
        return float("nan")
    return float(2 * np.sum(np.minimum(y_pred, y_true)) / total)


def cpl(y_pred, y_true):
    """Common Part of Links: overlap of the non-zero link sets (0..1)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    pred_pos, true_pos = y_pred > 0, y_true > 0
    total = np.sum(pred_pos) + np.sum(true_pos)
    if total <= 0:
        return float("nan")
    return float(2 * np.sum(pred_pos & true_pos) / total)


# ───────────────────────────────────────────────────────────────────────────
# Purpose-distribution prediction
# ───────────────────────────────────────────────────────────────────────────

def _as_probs(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return p.reshape(-1, p.shape[-1]), q.reshape(-1, q.shape[-1])


def kl_divergence(p, q, eps=1e-10):
    """Mean ``KL(p || q)`` over rows of two ``(N, K)`` distributions."""
    p, q = _as_probs(p, q)
    p, q = np.clip(p, eps, 1), np.clip(q, eps, 1)
    return float(np.mean(np.sum(p * np.log(p / q), axis=1)))


def jsd(p, q, eps=1e-10):
    """Mean Jensen-Shannon divergence (symmetric, bounded)."""
    p, q = _as_probs(p, q)
    p, q = np.clip(p, eps, 1), np.clip(q, eps, 1)
    m = 0.5 * (p + q)
    return float(np.mean(
        0.5 * np.sum(p * np.log(p / m), axis=1)
        + 0.5 * np.sum(q * np.log(q / m), axis=1)))


def js_divergence(p, q, eps=1e-12):
    """Per-row Jensen-Shannon divergence, shape ``(N,)``.

    Row-wise variant used by GMEL, which reports the mean *and* median.
    """
    p, q = _as_probs(p, q)
    m = 0.5 * (p + q + eps)
    kl_pm = np.sum(p * np.log((p + eps) / m), axis=1)
    kl_qm = np.sum(q * np.log((q + eps) / m), axis=1)
    return 0.5 * (kl_pm + kl_qm)


def cosine_similarity(p, q):
    """Per-row cosine similarity, shape ``(N,)`` (higher is better)."""
    p, q = _as_probs(p, q)
    num = np.sum(p * q, axis=1)
    den = np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1) + 1e-12
    return num / den


def row_cpc(p, q):
    """Per-row Common Part of Commuters, in [0, 1].

    ``CPC = 2 * sum(min(p, q)) / (sum(p) + sum(q))``. The denominator matters:
    dropping it (as an earlier revision did) returns 2 for identical rows
    instead of 1, i.e. a value outside the metric's range. For row-normalized
    inputs the denominator is 2 and CPC reduces to ``sum(min(p, q))``.
    """
    p, q = _as_probs(p, q)
    total = p.sum(axis=1) + q.sum(axis=1)
    return np.where(total > 0,
                    2.0 * np.sum(np.minimum(p, q), axis=1) / np.where(total > 0, total, 1.0),
                    0.0)


def per_purpose_mae(pred_probs, true_probs):
    """MAE per purpose category, shape ``(K,)``."""
    pred_probs, true_probs = _as_probs(pred_probs, true_probs)
    return np.mean(np.abs(pred_probs - true_probs), axis=0)


def evaluate_distribution(pred_probs, true_probs):
    """Summary of distribution-prediction quality for ``(N, K)`` matrices.

    Returns a dict with ``JSD_mean``, ``JSD_median``, ``Cosine_mean``,
    ``overall_MAE`` and ``per_purpose_MAE``.
    """
    jsd_rows = js_divergence(true_probs, pred_probs)
    cos = cosine_similarity(pred_probs, true_probs)
    pp_mae = per_purpose_mae(pred_probs, true_probs)
    return {
        "JSD_mean": float(np.mean(jsd_rows)),
        "JSD_median": float(np.median(jsd_rows)),
        "Cosine_mean": float(np.mean(cos)),
        "overall_MAE": float(np.mean(pp_mae)),
        "per_purpose_MAE": pp_mae.tolist(),
    }


# ───────────────────────────────────────────────────────────────────────────
# Combined report
# ───────────────────────────────────────────────────────────────────────────

def evaluate_all(y_flow_true, y_flow_pred, y_prob_true, y_prob_pred,
                 model_name="Model"):
    """Print and return the standard flow + distribution metric bundle."""
    metrics = {
        "MAE": mae(y_flow_true, y_flow_pred),
        "RMSE": rmse(y_flow_true, y_flow_pred),
        "MAPE": mape(y_flow_true, y_flow_pred),
        "R2": r2(y_flow_true, y_flow_pred),
        "CPC": cpc(y_flow_pred, y_flow_true),
        "KL": kl_divergence(y_prob_true, y_prob_pred),
        "JSD": jsd(y_prob_true, y_prob_pred),
        "RowCPC": float(np.mean(row_cpc(y_prob_true, y_prob_pred))),
    }

    print("\n" + "=" * 52)
    print("  {}".format(model_name))
    print("=" * 52)
    print("  [flow]")
    print("    MAE  : {:.4f}".format(metrics["MAE"]))
    print("    RMSE : {:.4f}".format(metrics["RMSE"]))
    print("    MAPE : {:.2f}%".format(metrics["MAPE"]))
    print("    R2   : {:.4f}".format(metrics["R2"]))
    print("    CPC  : {:.4f}".format(metrics["CPC"]))
    print("  [purpose distribution]")
    print("    KL   : {:.4f}".format(metrics["KL"]))
    print("    JSD  : {:.4f}".format(metrics["JSD"]))
    print("    CPC  : {:.4f}".format(metrics["RowCPC"]))
    print("=" * 52 + "\n")
    return metrics
