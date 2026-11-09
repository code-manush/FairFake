"""
Unit tests for FairFake Feature 2.3 (Orthogonal Disentanglement Loss)
"""

import unittest
import torch
from backend.deepfake_detector.src.disentangle import (
    OrthogonalSubspaceSplitter,
    OrthogonalDisentanglementLoss,
    DisentangledForensicObjective
)


class TestDisentanglement(unittest.TestCase):
    def test_subspace_splitter_shapes(self):
        splitter = OrthogonalSubspaceSplitter(in_features=2048, forensic_dim=1024)
        z = torch.randn(4, 2048)
        z_f, z_b = splitter(z)

        self.assertEqual(z_f.shape, (4, 1024))
        self.assertEqual(z_b.shape, (4, 1024))
        # Direct concatenation should reconstruct original z
        reconstructed = torch.cat([z_f, z_b], dim=1)
        self.assertTrue(torch.allclose(z, reconstructed))

    def test_learned_projections_option(self):
        splitter = OrthogonalSubspaceSplitter(in_features=512, forensic_dim=256, use_projections=True)
        z = torch.randn(4, 512)
        z_f, z_b = splitter(z)
        self.assertEqual(z_f.shape, (4, 256))
        self.assertEqual(z_b.shape, (4, 256))

    def test_cosine_orthogonality(self):
        loss_fn = OrthogonalDisentanglementLoss(mode="cosine")

        # Perfectly orthogonal vectors
        u = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        v = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        res = loss_fn(u, v)
        self.assertAlmostEqual(res["cosine_penalty"].item(), 0.0, places=5)

        # Collinear vectors
        u_col = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        v_col = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        res_col = loss_fn(u_col, v_col)
        self.assertAlmostEqual(res_col["cosine_penalty"].item(), 1.0, places=4)

    def test_disentangled_objective_gradient_flow(self):
        objective = DisentangledForensicObjective(lambda_adversary=0.5, beta_orthogonality=0.2)

        # Forensics
        forensic_logits = torch.randn(4, requires_grad=True)
        forensic_targets = torch.tensor([1, 0, 1, 0])

        # Attributes
        attr_logits = {
            "Dark_Skin": torch.randn(4, requires_grad=True),
            "Heavy_Makeup": torch.randn(4, requires_grad=True)
        }
        attr_targets = {
            "Dark_Skin": torch.tensor([1, 1, 0, 0]),
            "Heavy_Makeup": torch.tensor([0, 1, 0, 1])
        }

        # Subspaces
        z_f = torch.randn(4, 64, requires_grad=True)
        z_b = torch.randn(4, 64, requires_grad=True)

        loss_dict = objective(
            forensic_logits=forensic_logits,
            forensic_targets=forensic_targets,
            attribute_logits=attr_logits,
            attribute_targets=attr_targets,
            z_forensic=z_f,
            z_biometric=z_b
        )

        self.assertIn("total_loss", loss_dict)
        self.assertIn("loss_forensic", loss_dict)
        self.assertIn("loss_adversary", loss_dict)
        self.assertIn("loss_orthogonality", loss_dict)

        # Verify backward gradient flow into all branches
        loss_dict["total_loss"].backward()
        self.assertIsNotNone(forensic_logits.grad)
        self.assertIsNotNone(attr_logits["Dark_Skin"].grad)
        self.assertIsNotNone(z_f.grad)
        self.assertIsNotNone(z_b.grad)


if __name__ == "__main__":
    unittest.main()
