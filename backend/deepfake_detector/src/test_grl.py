"""
Unit tests for FairFake Feature 2.1 (Gradient Reversal Layer & Scheduler)
"""

import unittest
import torch
from backend.deepfake_detector.src.grl import (
    GradientReversalFunction,
    GradientReversalLayer,
    GRLScheduler
)


class TestGRL(unittest.TestCase):
    def test_forward_pass_identity(self):
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 16)
        out = grl(x)
        self.assertTrue(torch.allclose(x, out))

    def test_backward_gradient_reversal(self):
        lambda_val = 0.75
        grl = GradientReversalLayer(lambda_=lambda_val)
        x = torch.tensor([[2.0, -3.0], [4.0, 1.0]], requires_grad=True)
        y = grl(x)

        # Standard linear loss: dLoss/dy = [[1, 2], [3, 4]]
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        loss = (y * weights).sum()
        loss.backward()

        expected_grad = -lambda_val * weights
        self.assertTrue(torch.allclose(x.grad, expected_grad))

    def test_lambda_schedule_bounds(self):
        grl = GradientReversalLayer(lambda_=0.0)
        scheduler = GRLScheduler(grl, gamma=10.0, max_lambda=1.0)

        # Step 0
        l0 = scheduler.step(0, 100)
        self.assertAlmostEqual(l0, 0.0, places=4)
        self.assertAlmostEqual(grl.get_lambda(), 0.0, places=4)

        # Step 50
        l50 = scheduler.step(50, 100)
        self.assertGreater(l50, 0.90)

        # Step 100
        l100 = scheduler.step(100, 100)
        self.assertAlmostEqual(l100, 1.0, places=3)
        self.assertAlmostEqual(grl.get_lambda(), 1.0, places=3)

    def test_eval_mode_bypass(self):
        # In eval mode (not training), GRL must act purely as identity with no overhead
        grl = GradientReversalLayer(lambda_=1.0)
        grl.eval()
        x = torch.tensor([5.0], requires_grad=True)
        y = grl(x)
        y.backward()
        # In eval mode without backward function application, gradient is direct (+1)
        self.assertEqual(x.grad.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
