"""Flow target normalization.

Total flow is heavy-tailed, so ``--log-transform`` trains on
``log(flow)`` rescaled to ``[0, 1]``.  Reported metrics have to undo that
exactly, which is why the transform lives in one class instead of being
open-coded in the trainer, the model and the scorer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Added inside the log so zero-flow edges stay finite, and to the min-max
#: range so a constant training split does not divide by zero.
EPSILON = 1e-6


@dataclass(frozen=True)
class LogMinMaxScaler:
    """``flow -> (log(flow + eps) - log_min) / (log_max - log_min + eps)``.

    ``identity()`` returns a pass-through scaler, so downstream code can always
    hold a scaler rather than branching on whether normalization is enabled.
    """

    log_min: float = 0.0
    log_max: float = 1.0
    enabled: bool = True

    @classmethod
    def identity(cls):
        return cls(enabled=False)

    @classmethod
    def fit(cls, flows):
        """Fit on the *training* flows only, so valid/test cannot leak in."""
        log_flows = np.log(np.asarray(flows, dtype=np.float64) + EPSILON)
        return cls(log_min=float(log_flows.min()), log_max=float(log_flows.max()))

    @property
    def log_range(self):
        return self.log_max - self.log_min + EPSILON

    def transform(self, flows):
        if not self.enabled:
            return np.asarray(flows, dtype=np.float32)
        log_flows = np.log(np.asarray(flows, dtype=np.float64) + EPSILON)
        return ((log_flows - self.log_min) / self.log_range).astype(np.float32)

    def inverse_transform(self, values):
        """Map normalized values back to the raw flow scale."""
        if not self.enabled:
            return np.asarray(values, dtype=np.float64)
        log_flows = np.asarray(values, dtype=np.float64) * self.log_range + self.log_min
        return np.exp(log_flows) - EPSILON
