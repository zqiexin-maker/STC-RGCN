"""Module-contribution analysis over a table of ablation results.

Each ablated model is compared against one baseline row along four dimensions
-- purpose distribution and flow, for single-task and multi-task runs -- and the
relative improvement per metric is averaged into a single contribution score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: ``metric -> (higher_is_better, section)``.  ``section`` decides which of the
#: four comparison dimensions a metric belongs to.
CONTRIBUTION_METRICS = {
    "kl": (False, "purpose"),
    "cosine_sim": (True, "purpose"),
    "mae": (False, "purpose"),
    "raw_mae": (False, "flow"),
    "raw_mse": (False, "flow"),
    "raw_rmse": (False, "flow"),
    "cpc": (True, "flow"),
}

#: Contributions are clipped to this range so one unstable metric cannot
#: dominate the average.
CONTRIBUTION_RANGE = (-1.0, 2.0)

#: Index columns expected in the ablation table.
INDEX_COLUMNS = ("model", "task")


def metric_contribution(ablation_value, baseline_value, higher_is_better):
    """Relative improvement of ``ablation_value`` over ``baseline_value``.

    Returns ``nan`` when either value is missing or the baseline is ~0.
    """
    if pd.isna(baseline_value) or pd.isna(ablation_value) or abs(baseline_value) < 1e-8:
        return np.nan
    delta = (
        ablation_value - baseline_value
        if higher_is_better
        else baseline_value - ablation_value
    )
    return float(np.clip(delta / abs(baseline_value), *CONTRIBUTION_RANGE))


def load_ablation_table(path):
    """Read the ablation CSV, drop ``*_std`` columns and index by model/task."""
    frame = pd.read_csv(path)
    missing = set(INDEX_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            "{}: missing required column(s) {}".format(path, sorted(missing))
        )
    value_columns = [c for c in frame.columns if not c.endswith("_std")]
    return frame[value_columns].set_index(list(INDEX_COLUMNS))


def _row(table, model, task):
    """Fetch one row as ``{column: float}``; empty when it is absent."""
    try:
        row = table.loc[(model, task)]
    except KeyError:
        return {}
    return {key: (float(value) if pd.notna(value) else np.nan) for key, value in row.items()}


def module_contributions(path, baseline_model="base", metrics=None):
    """Compute per-module contributions from an ablation results table.

    Parameters
    ----------
    path
        CSV with ``model``, ``task`` and one column per metric.
    baseline_model
        Model whose ``single`` row is the reference for every comparison.
    metrics
        Restrict the analysis to these metrics; ``None`` uses all of
        :data:`CONTRIBUTION_METRICS`.
    """
    table = load_ablation_table(path)
    metrics = list(metrics or CONTRIBUTION_METRICS)
    unknown = [m for m in metrics if m not in CONTRIBUTION_METRICS]
    if unknown:
        raise ValueError(
            "unknown metric(s) {}; available: {}".format(
                unknown, sorted(CONTRIBUTION_METRICS)
            )
        )

    baseline = _row(table, baseline_model, "single")
    if not baseline:
        raise ValueError(
            "{}: no row for the baseline ({!r}, 'single')".format(path, baseline_model)
        )

    models = [m for m in table.index.get_level_values("model").unique() if m != baseline_model]

    rows = []
    for model in models:
        by_task = {"single": _row(table, model, "single"), "multi": _row(table, model, "multi")}
        for task, section in (
            ("single", "purpose"),
            ("single", "flow"),
            ("multi", "purpose"),
            ("multi", "flow"),
        ):
            ablation = by_task[task]
            scores = {
                metric: metric_contribution(
                    ablation.get(metric),
                    baseline.get(metric),
                    CONTRIBUTION_METRICS[metric][0],
                )
                for metric in metrics
                if CONTRIBUTION_METRICS[metric][1] == section
            }
            valid = [v for v in scores.values() if not np.isnan(v)]
            if not valid:
                continue
            rows.append(
                {
                    "module": model,
                    "dimension": "{}-task-{}".format(task, section),
                    "mean_contribution": float(np.mean(valid)),
                    **{metric: scores.get(metric, np.nan) for metric in metrics},
                }
            )

    return pd.DataFrame(rows)
