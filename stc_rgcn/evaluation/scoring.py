"""Score saved prediction files.

Every model and baseline writes the 34-column layout described by
:data:`stc_rgcn.utils.schema.PREDICTION_COLUMNS`, with empty cells for a head the
model does not have.  :func:`score_folder` therefore reads a whole results
directory without being told which model produced which file, and reports
mean +/- std per model group -- files that differ only by a trailing run number
are treated as repeats of one model.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..utils.metrics import flow_metrics, flow_metrics_by_magnitude, purpose_distribution_metrics
from ..utils.schema import NUM_PURPOSES, PREDICTION_COLUMNS

#: Trailing run markers stripped when grouping repeats of the same model:
#: ``_run_3``, ``_run3``, ``-3`` or ``_3``.
_RUN_SUFFIX = re.compile(r"(_run_?\d+|[-_]\d+)$", re.IGNORECASE)

#: Counts are aggregated by summing, not by averaging like every other metric.
_COUNT_KEYS = ("large_count", "small_count")


@dataclass
class Predictions:
    """Parsed contents of one prediction file."""

    pred_probs: Optional[np.ndarray] = None
    true_probs: Optional[np.ndarray] = None
    pred_flows: Optional[np.ndarray] = None
    true_flows: Optional[np.ndarray] = None

    @property
    def has_probs(self):
        return self.pred_probs is not None and self.true_probs is not None

    @property
    def has_flows(self):
        return self.pred_flows is not None and self.true_flows is not None


def read_prediction_file(path, num_purposes=NUM_PURPOSES):
    """Parse one prediction file into a :class:`Predictions`.

    The layout is taken from the header when it names the standard columns;
    otherwise it is inferred from the column count, which keeps files written
    by pre-1.0 releases readable.
    """
    with open(path, encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]
    if not lines:
        return Predictions()

    header = lines[0].split("\t")
    rows = [line.split("\t") for line in lines[1:]]
    if not rows:
        return Predictions()

    layout = _resolve_layout(header, max(len(row) for row in rows), num_purposes)
    if layout is None:
        raise ValueError(
            "{}: cannot recognise the prediction layout (header has {} columns, "
            "rows up to {})".format(path, len(header), max(len(row) for row in rows))
        )
    prob_start, flow_start = layout

    pred_probs, true_probs, pred_flows, true_flows = [], [], [], []
    for line_no, row in enumerate(rows, start=2):
        if prob_start is not None:
            probs = _floats(row, prob_start, 2 * num_purposes, path, line_no)
            if probs is not None:
                pred_probs.append(probs[:num_purposes])
                true_probs.append(probs[num_purposes:])
        if flow_start is not None:
            flows = _floats(row, flow_start, 2, path, line_no)
            if flows is not None:
                pred_flows.append(flows[0])
                true_flows.append(flows[1])

    return Predictions(
        pred_probs=np.array(pred_probs) if pred_probs else None,
        true_probs=np.array(true_probs) if true_probs else None,
        pred_flows=np.array(pred_flows) if pred_flows else None,
        true_flows=np.array(true_flows) if true_flows else None,
    )


def _resolve_layout(header, width, num_purposes):
    """Return ``(prob_start, flow_start)`` column offsets, or ``None``."""
    if tuple(header) == PREDICTION_COLUMNS:
        return 2, 2 + 2 * num_purposes
    full_width = 2 + 2 * num_purposes + 2
    if width >= full_width:
        return 2, 2 + 2 * num_purposes
    if width >= 2 + 2 * num_purposes:  # purpose-only
        return 2, None
    if width >= 4:  # flow-only
        return None, 2
    return None


def _floats(row, start, count, path, line_no):
    """Read ``count`` floats from ``row``; ``None`` when they are blank."""
    cells = row[start : start + count]
    if len(cells) < count or any(cell == "" for cell in cells):
        return None
    try:
        return [float(cell) for cell in cells]
    except ValueError as exc:
        raise ValueError("{}: line {} is not numeric: {}".format(path, line_no, exc)) from exc


def score_file(path, scaler=None, magnitude_threshold=30.0):
    """Compute every applicable metric for one prediction file."""
    predictions = read_prediction_file(path)
    result = {"file": path, "scored": False}

    if predictions.has_probs:
        result["purpose"] = purpose_distribution_metrics(
            predictions.pred_probs, predictions.true_probs
        )
        result["scored"] = True

    if predictions.has_flows:
        result["flow"] = flow_metrics(predictions.pred_flows, predictions.true_flows, scaler)
        result["flow"].update(
            flow_metrics_by_magnitude(
                predictions.pred_flows,
                predictions.true_flows,
                scaler,
                threshold=magnitude_threshold,
            )
        )
        result["scored"] = True

    return result


def model_group_key(path):
    """Strip a trailing run number so repeats of one model group together.

    ``predictions_full_run_2.txt`` and ``predictions_full_run_3.txt`` both map
    to ``predictions_full``; ``gravity_gm_e.txt`` keeps its own name.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    match = _RUN_SUFFIX.search(stem)
    return stem[: match.start()] if match else stem


def group_by_model(paths):
    """Group file paths by :func:`model_group_key`."""
    groups = {}
    for path in paths:
        groups.setdefault(model_group_key(path), []).append(path)
    return groups


def aggregate_group_stats(results):
    """Mean and std of each metric across a group of scored files.

    Returns ``{section: {metric: {"mean": ..., "std": ..., "n": ...}}}``.
    List-valued metrics such as the per-purpose R^2 are skipped; counts are
    summed rather than averaged.
    """
    sections = {}
    for result in results:
        for section in ("purpose", "flow"):
            if section not in result:
                continue
            for key, value in result[section].items():
                if isinstance(value, (list, tuple)):
                    continue
                sections.setdefault(section, {}).setdefault(key, []).append(value)

    summary = {}
    for section, metrics in sections.items():
        summary[section] = {}
        for key, values in metrics.items():
            array = np.asarray(values, dtype=np.float64)
            if key in _COUNT_KEYS:
                summary[section][key] = {"mean": float(np.nansum(array)), "std": 0.0,
                                         "n": len(array)}
            else:
                summary[section][key] = {
                    "mean": float(np.nanmean(array)),
                    "std": float(np.nanstd(array)),
                    "n": int(np.count_nonzero(~np.isnan(array))),
                }
    return summary


def score_folder(folder, pattern="*.txt", scaler=None, group=True, per_file=False):
    """Score every matching file in ``folder`` and print a grouped summary.

    Returns the list of per-file results so callers can post-process them.
    """
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        print("warning: no files matching {!r} in {}".format(pattern, folder))
        return []

    print("scoring {} file(s) in {}\n".format(len(paths), folder))

    results = []
    for path in paths:
        try:
            result = score_file(path, scaler=scaler)
        except (OSError, ValueError) as exc:
            print("error: skipping {}: {}".format(path, exc))
            continue
        results.append(result)
        if per_file and result["scored"]:
            _print_result(os.path.basename(path), result)

    if group:
        _print_groups(results)
    return results


def _print_result(title, result):
    print("--- {}".format(title))
    for section in ("purpose", "flow"):
        if section not in result:
            continue
        print("  {}:".format(section))
        for key, value in result[section].items():
            if not isinstance(value, (list, tuple)):
                print("    {:<24}: {:.4f}".format(key, value))


def _print_groups(results):
    by_key = group_by_model([result["file"] for result in results])
    print("\n" + "=" * 70)
    print("grouped statistics")
    print("=" * 70)

    for model_name, paths in sorted(by_key.items()):
        group_results = [r for r in results if r["file"] in paths]
        summary = aggregate_group_stats(group_results)
        print("\n{}\nmodel : {}\nfiles : {}".format("-" * 60, model_name, len(paths)))
        if not summary:
            print("  (nothing scored; check the file layout)")
            continue
        for section, metrics in summary.items():
            print("\n  {} metrics (mean +/- std):".format(section))
            for key, stats in metrics.items():
                print("    {:<24}: {:>10.4f} +/- {:>8.4f}".format(key, stats["mean"], stats["std"]))
