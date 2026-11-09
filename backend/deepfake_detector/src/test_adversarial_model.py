"""
Unit tests for FairFake Feature 2.2 (Multi-Branch Adversarial Detector)
"""

import unittest
import torch
from backend.deepfake_detector.src.adversarial_model import (
    AdversarialXceptionDetector,
    AdversarialEfficientNetDetector,
    TARGET_ATTRIBUTES
)


class TestAdversarialModel(unittest.TestCase):
    def test_forward_pass_train_and_eval(self):
        model = AdversarialXceptionDetector(pretrained=False, initial_lambda=0.5)
        x = torch.randn(2, 3, 224, 224)

        # 1. Training mode
        model.train()
        train_out = model(x, return_dict=True)
        self.assertIn("forensic_logits", train_out)
        self.assertIn("attribute_logits", train_out)
        self.assertIn("latent_features", train_out)
        self.assertEqual(train_out["forensic_logits"].shape, (2,))
        self.assertEqual(train_out["latent_features"].shape, (2, 2048))

        for attr in TARGET_ATTRIBUTES:
            self.assertIn(attr, train_out["attribute_logits"])
            self.assertEqual(train_out["attribute_logits"][attr].shape, (2,))

        # 2. Evaluation mode (drop-in compatibility)
        model.eval()
        with torch.no_grad():
            eval_out = model(x)
        self.assertEqual(eval_out.shape, (2,))

    def test_freeze_and_unfreeze(self):
        model = AdversarialXceptionDetector(pretrained=False, freeze_backbone=True)
        for p in model.backbone.parameters():
            self.assertFalse(p.requires_grad)

        # Forensic classifier must still require grad
        for p in model.forensic_classifier.parameters():
            self.assertTrue(p.requires_grad)

        model.unfreeze_backbone()
        for p in model.backbone.parameters():
            self.assertTrue(p.requires_grad)

    def test_adversarial_gradient_flow(self):
        model = AdversarialXceptionDetector(pretrained=False, initial_lambda=1.0)
        model.train()
        x = torch.randn(2, 3, 224, 224)

        # Forward pass
        out = model(x, return_dict=True)
        attr_loss = out["attribute_logits"]["Dark_Skin"].sum()
        attr_loss.backward()

        # Backbone conv1 must have received gradients through the GRL branch
        backbone_param = next(model.backbone.parameters())
        self.assertIsNotNone(backbone_param.grad)
        self.assertFalse(torch.all(backbone_param.grad == 0))

    def test_efficientnet_adversarial_variant(self):
        eff_model = AdversarialEfficientNetDetector(pretrained=False, initial_lambda=0.5)
        eff_model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = eff_model(x)
        self.assertEqual(out.shape, (2,))


if __name__ == "__main__":
    unittest.main()
