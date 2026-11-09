"""
FairFake Gradient Reversal Layer (GRL) & Dynamic Scheduler (Feature 2.1)

Implements the Gradient Reversal Layer (Ganin & Lempitsky, 2015) in PyTorch.
In the forward pass, it acts as an identity operator:
    y = x
In the backward pass, it reverses and scales gradients passing through it:
    d(Loss) / dx = -lambda * d(Loss) / dy
"""

import math
import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional


class GradientReversalFunction(Function):
    """
    Custom autograd function reversing gradients during backward pass.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Gradients are negated and scaled by lambda
        grad_input = grad_output.neg() * ctx.lambda_
        # Return gradient for x and None for the lambda scalar parameter
        return grad_input, None


class GradientReversalLayer(nn.Module):
    """
    PyTorch Module wrapper for GradientReversalFunction.
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def set_lambda(self, lambda_: float):
        """Dynamically updates the gradient scaling factor lambda."""
        self.lambda_ = float(max(0.0, lambda_))

    def get_lambda(self) -> float:
        return self.lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lambda_ == 0.0 or not self.training:
            return x
        return GradientReversalFunction.apply(x, self.lambda_)

    def extra_repr(self) -> str:
        return f"lambda={self.lambda_:.4f}"


class GRLScheduler:
    """
    Schedules adversarial reversal weight lambda as training progresses:
        lambda(p) = 2 / (1 + exp(-gamma * p)) - 1
    where p in [0, 1] represents the training progress.
    """

    def __init__(
        self,
        grl_layer: GradientReversalLayer,
        gamma: float = 10.0,
        max_lambda: float = 1.0
    ):
        self.grl_layer = grl_layer
        self.gamma = float(gamma)
        self.max_lambda = float(max_lambda)

    def compute_lambda(self, progress: float) -> float:
        """
        Calculates lambda for progress in [0, 1].
        """
        p = float(max(0.0, min(1.0, progress)))
        val = 2.0 / (1.0 + math.exp(-self.gamma * p)) - 1.0
        return float(self.max_lambda * val)

    def step(self, current_step: int, total_steps: int) -> float:
        """
        Computes progress, updates GRL layer's lambda, and returns the new value.
        """
        if total_steps <= 0:
            progress = 1.0
        else:
            progress = current_step / float(total_steps)

        new_lambda = self.compute_lambda(progress)
        self.grl_layer.set_lambda(new_lambda)
        return new_lambda


if __name__ == "__main__":
    print("=" * 70)
    print("FAIRFAKE GRADIENT REVERSAL LAYER (GRL) VERIFICATION")
    print("=" * 70)

    grl = GradientReversalLayer(lambda_=1.0)
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # Forward pass (identity)
    y = grl(x)
    print("Forward pass identity check:")
    print("  Input x: ", x.data.numpy())
    print("  Output y:", y.data.numpy())

    # Backward pass
    loss = (y * 2.0).sum()
    loss.backward()
    print("Backward pass reversed gradient check (expected [-2, -2, -2]):")
    print("  x.grad:  ", x.grad.numpy())

    # Scheduler test
    scheduler = GRLScheduler(grl, gamma=10.0)
    print("\nDynamic Lambda Schedule Ramp-Up:")
    for step_pct in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        lamb = scheduler.compute_lambda(step_pct)
        print(f"  Progress: {step_pct:>5.1%} -> Reversal Lambda: {lamb:.4f}")
    print("=" * 70)
