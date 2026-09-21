"""Early stopping over one or two validation metrics.

Which metrics are monitored follows from the variant and the selected task: a
single-head variant can only watch its own metric, a multi-task run watches
both and only accepts a checkpoint that improves one of them without
regressing the other.
"""

from __future__ import annotations

from ..config import Task, Variant

#: Validation keys, all of them lower-is-better.
KL_LOSS = "kl_loss"
MAE = "mae"
TWEEDIE_LOSS = "tweedie_loss"


def monitored_metrics(variant, task):
    """Return the validation keys early stopping watches.

    A task explicitly restricted to one head wins over the variant's own
    capabilities, so ``--variant full --task flow`` stops on flow MAE alone.
    """
    variant, task = Variant(variant), Task(task)

    if variant is Variant.PURPOSE or task is Task.PURPOSE:
        return (KL_LOSS,)
    if variant is Variant.FLOW or task is Task.FLOW:
        return (MAE,)
    if variant.uses_tweedie_loss:
        return (KL_LOSS, TWEEDIE_LOSS)
    return (KL_LOSS, MAE)


class EarlyStopper:
    """Track the best validation metrics and decide when to stop.

    Parameters
    ----------
    metrics
        Validation keys to watch, from :func:`monitored_metrics`.
    patience
        Evaluations without an accepted improvement before stopping.
    delta
        Margin a metric must beat to count as improved, and may not exceed to
        count as "not worse".

    A checkpoint is accepted when *every* watched metric is no worse than its
    best and *at least one* strictly improves.  With a single metric that
    reduces to plain "improved".
    """

    def __init__(self, metrics, patience=20, delta=1e-4):
        if not metrics:
            raise ValueError("early stopping needs at least one metric to watch")
        self.metrics = tuple(metrics)
        self.patience = patience
        self.delta = delta
        self.best = {name: float("inf") for name in self.metrics}
        self.num_stale = 0
        self.should_stop = False

    def update(self, validation):
        """Feed one validation result.

        Returns
        -------
        bool
            Whether the caller should save this checkpoint.
        """
        current = {name: validation.get(name, float("inf")) for name in self.metrics}

        none_worse = all(current[n] <= self.best[n] + self.delta for n in self.metrics)
        any_better = any(current[n] < self.best[n] - self.delta for n in self.metrics)
        improved = none_worse and any_better

        if improved:
            self.best.update(current)
            self.num_stale = 0
        else:
            self.num_stale += 1
            if self.num_stale >= self.patience:
                self.should_stop = True

        self._last = (current, improved)
        return improved

    def describe(self, epoch, train_loss):
        """Render the progress line for the most recent :meth:`update`."""
        current, improved = getattr(self, "_last", ({}, False))
        parts = ["Epoch {}: train loss {:.4f}".format(epoch, train_loss)]
        parts += ["val {} {:.4f}".format(name, current.get(name, float("nan")))
                  for name in self.metrics]
        line = ", ".join(parts)

        if improved:
            line += " [saved]"
        else:
            line += " [no improvement {}/{}]".format(self.num_stale, self.patience)
        if self.should_stop:
            line += " [early stopping]"
        return line
