"""GradNorm loss balancing for the two-task variants.

Implements *GradNorm: Gradient Normalization for Adaptive Loss Balancing in
Deep Multitask Networks* (Chen et al., ICML 2018), adapted from
``LibMTL.weighting.GradNorm``.

The balancer owns one learnable scale per task.  Each step it measures how fast
each task is training relative to the others and nudges the scales so a task
that is falling behind gets a larger gradient.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class GradNormBalancer:
    """Adaptive weights for a fixed number of task losses.

    Parameters
    ----------
    num_tasks
        How many losses are balanced; 2 here (purpose and flow).
    alpha
        Restoring-force strength.  Larger values pull the tasks harder towards
        a common training rate.
    lr
        Learning rate of the Adam optimizer over the task scales.
    device
        Device the scales and gradients live on.

    Notes
    -----
    Relative training rates are measured against the *first* observed loss per
    task, as in the paper.  Releases before 1.0 compared against the previous
    step instead, which made the weights track step-to-step noise.
    """

    def __init__(self, num_tasks=2, alpha=1.5, lr=0.025, device=None):
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loss_scale = torch.ones(num_tasks, device=self.device, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.loss_scale], lr=lr)

        self.initial_losses = None
        self.step_count = 0

    def _task_gradients(self, losses, model):
        """Flat gradient of each task loss w.r.t. every model parameter.

        ``retain_graph`` keeps the graph alive so the caller can still run its
        own backward pass over the weighted loss afterwards.
        """
        gradients = []
        for loss in losses:
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            flat = [
                parameter.grad.reshape(-1).detach().clone().to(self.device)
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            gradients.append(torch.cat(flat) if flat else torch.zeros(1, device=self.device))
        model.zero_grad(set_to_none=True)
        return gradients

    def compute_weights(self, losses, model):
        """Update the task scales and return the weights for this step.

        The caller is responsible for forming ``sum(weight_i * loss_i)`` and
        running its own backward pass; this method only touches the scales.

        Returns
        -------
        torch.Tensor
            Detached, ``num_tasks``-long weights summing to ``num_tasks``.
        """
        self.step_count += 1
        losses = [loss.to(self.device) for loss in losses]

        if self.initial_losses is None:
            # Nothing to compare against yet: start from equal weights.
            self.initial_losses = torch.tensor(
                [loss.item() for loss in losses], device=self.device
            ).clamp_min(1e-12)
            return torch.ones(self.num_tasks, device=self.device)

        weights = self.num_tasks * F.softmax(self.loss_scale, dim=-1)

        gradient_norms = torch.stack(
            [
                torch.norm(weights[index] * gradient, p=2)
                for index, gradient in enumerate(self._task_gradients(losses, model))
            ]
        )
        mean_gradient_norm = gradient_norms.mean()

        current = torch.tensor([loss.item() for loss in losses], device=self.device)
        inverse_rates = current / self.initial_losses
        relative_rates = inverse_rates / inverse_rates.mean()

        target = (mean_gradient_norm * relative_rates.pow(self.alpha)).detach()
        gradient_loss = (gradient_norms - target).abs().sum()

        self.optimizer.zero_grad()
        gradient_loss.backward()
        self.optimizer.step()

        return weights.detach()
