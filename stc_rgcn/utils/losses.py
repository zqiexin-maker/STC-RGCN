"""Loss functions for the two prediction heads.

The purpose head is trained with a KL divergence against the observed
distribution; the flow head with an L1 loss, or a Tweedie loss for the
:attr:`~stc_rgcn.config.Variant.FULL_TWEEDIE` variant, whose compound
Poisson-Gamma form suits a non-negative target with a point mass at zero.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

#: Floor applied to predictions and targets before the Tweedie log/power terms.
TWEEDIE_EPSILON = 1e-10

#: Valid range of the Tweedie power parameter for compound Poisson-Gamma data.
TWEEDIE_POWER_RANGE = (1.0, 2.0)


def purpose_kl_loss(pred_probs, true_probs):
    """Batch-mean KL divergence between two purpose distributions.

    Both arguments are probabilities; the log is taken here because
    :class:`torch.nn.KLDivLoss` expects log-probabilities on the input side.
    """
    return F.kl_div(pred_probs.log(), true_probs, reduction="batchmean")


def flow_l1_loss(pred_flows, true_flows):
    """Mean absolute error on the total-flow target."""
    return F.l1_loss(pred_flows, true_flows)


def tweedie_loss(pred_flows, true_flows, power, epsilon=TWEEDIE_EPSILON):
    """Negative Tweedie log-likelihood, up to terms constant in ``pred_flows``.

    ``power`` selects the member of the family: 1 is Poisson, 2 is Gamma, and
    anything in between is compound Poisson-Gamma, which is the regime
    :func:`estimate_tweedie_power` clips to.
    """
    pred_flows = torch.clamp(pred_flows, min=epsilon)
    true_flows = torch.clamp(true_flows, min=epsilon)

    if abs(power - 1.0) < epsilon:
        loss = F.poisson_nll_loss(pred_flows, true_flows, log_input=False, full=False)
    elif abs(power - 2.0) < 1e-5:
        ratio = true_flows / (pred_flows + epsilon)
        loss = torch.clamp(ratio - torch.log(ratio + epsilon) - 1, max=1e6)
    else:
        loss = (
            -torch.pow(true_flows, 2 - power) / ((1 - power) * (2 - power))
            + true_flows * torch.pow(pred_flows, 1 - power) / (1 - power)
            - torch.pow(pred_flows, 2 - power) / (2 - power)
        )

    return torch.mean(loss)


def estimate_tweedie_power(flows, verbose=True):
    """Estimate the Tweedie power from the variance-to-mean power law.

    Tweedie data satisfy ``Var(Y) = c * E[Y]**power``.  With a single sample
    there is nothing to regress, so this takes the crude ratio
    ``log(Var) / log(Mean)`` and clips it to
    :data:`TWEEDIE_POWER_RANGE`.  It is a starting value, not a fit -- pass
    ``--tweedie-power`` to override it.
    """
    values = flows.detach().cpu().numpy() if isinstance(flows, torch.Tensor) else np.asarray(flows)
    values = values.flatten()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot estimate a Tweedie power from an empty flow array")

    # Shift off zero so both logs stay finite.
    values = values + TWEEDIE_EPSILON
    mean = float(np.mean(values))
    variance = float(np.var(values))

    if mean <= 0 or variance <= 0 or abs(np.log(mean)) < TWEEDIE_EPSILON:
        power = float(np.mean(TWEEDIE_POWER_RANGE))
    else:
        power = float(np.clip(np.log(variance) / np.log(mean), *TWEEDIE_POWER_RANGE))

    if verbose:
        print("Estimated Tweedie power: {:.4f}".format(power))
    return power
