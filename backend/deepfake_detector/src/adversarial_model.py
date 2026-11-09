"""
FairFake Multi-Branch Adversarial Detector Architecture (Feature 2.2)

Combines convolutional backbones (Xception / EfficientNet) with:
  1. A primary forensic classifier head (predicts genuine vs manipulated face).
  2. Multi-branch adversary attribute heads connected through a Gradient Reversal Layer (GRL).

During training, backpropagation simultaneously minimizes manipulation detection error
while penalizing the backbone if demographic/appearance attributes are predictable from
latent representations.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import timm

# Local imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from grl import GradientReversalLayer, GRLScheduler

# Top demographic and cosmetic attributes exhibiting high error disparity
TARGET_ATTRIBUTES = ["Dark_Skin", "Heavy_Makeup", "Eyeglasses", "Senior"]


class AdversarialHead(nn.Module):
    """
    Adversary prediction head mapping latent representations to attribute logits.
    """

    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AdversarialXceptionDetector(nn.Module):
    """
    Debiased Xception architecture with Gradient Reversal adversary branches.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        target_attributes: Optional[List[str]] = None,
        initial_lambda: float = 0.0
    ):
        super().__init__()
        self.target_attributes = target_attributes or TARGET_ATTRIBUTES

        # 1. Shared Feature Extractor Backbone
        self.backbone = timm.create_model("legacy_xception", pretrained=pretrained, num_classes=0)
        self.feat_dim = self.backbone.num_features

        # 2. Primary Forensic Classification Head
        self.forensic_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        # 3. Gradient Reversal Layer for Adversary Branch
        self.grl = GradientReversalLayer(lambda_=initial_lambda)

        # 4. Multi-Task Attribute Adversary Heads
        self.adversary_heads = nn.ModuleDict({
            attr: AdversarialHead(self.feat_dim, hidden_dim=128)
            for attr in self.target_attributes
        })

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freezes backbone weights during initial head warmup."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes backbone weights for end-to-end adversarial fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def set_adversarial_lambda(self, lambda_val: float):
        """Updates reversal scaling coefficient lambda."""
        self.grl.set_lambda(lambda_val)

    def get_adversarial_lambda(self) -> float:
        return self.grl.get_lambda()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts latent features from backbone."""
        return self.backbone(x)

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass.
        In eval mode (default), returns forensic logits for drop-in compatibility.
        In train mode with return_dict=True, returns forensic and adversarial logits.
        """
        feats = self.extract_features(x)
        forensic_logits = self.forensic_classifier(feats).squeeze(-1)

        # During standard inference / evaluation, return only the forensic logits
        if not self.training and not return_dict:
            return forensic_logits

        # Adversary branches through Gradient Reversal Layer
        reversed_feats = self.grl(feats)
        attribute_logits = {
            attr: head(reversed_feats)
            for attr, head in self.adversary_heads.items()
        }

        if return_dict:
            return {
                "forensic_logits": forensic_logits,
                "attribute_logits": attribute_logits,
                "latent_features": feats
            }

        return forensic_logits, attribute_logits


class AdversarialEfficientNetDetector(nn.Module):
    """
    Debiased EfficientNet-B0 architecture with Gradient Reversal adversary branches.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        target_attributes: Optional[List[str]] = None,
        initial_lambda: float = 0.0
    ):
        super().__init__()
        self.target_attributes = target_attributes or TARGET_ATTRIBUTES

        self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)
        self.feat_dim = self.backbone.num_features

        self.forensic_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        self.grl = GradientReversalLayer(lambda_=initial_lambda)
        self.adversary_heads = nn.ModuleDict({
            attr: AdversarialHead(self.feat_dim, hidden_dim=128)
            for attr in self.target_attributes
        })

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def set_adversarial_lambda(self, lambda_val: float):
        self.grl.set_lambda(lambda_val)

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        feats = self.backbone(x)
        forensic_logits = self.forensic_classifier(feats).squeeze(-1)

        if not self.training and not return_dict:
            return forensic_logits

        reversed_feats = self.grl(feats)
        attribute_logits = {
            attr: head(reversed_feats)
            for attr, head in self.adversary_heads.items()
        }

        if return_dict:
            return {
                "forensic_logits": forensic_logits,
                "attribute_logits": attribute_logits,
                "latent_features": feats
            }

        return forensic_logits, attribute_logits


if __name__ == "__main__":
    print("=" * 70)
    print("FAIRFAKE ADVERSARIAL BACKBONE ARCHITECTURE CHECK")
    print("=" * 70)

    device = torch.device("cpu")
    model = AdversarialXceptionDetector(pretrained=False, initial_lambda=1.0).to(device)

    dummy_input = torch.randn(4, 3, 224, 224)
    print(f"Input batch shape: {dummy_input.shape}")

    # 1. Training mode forward pass
    model.train()
    outputs = model(dummy_input, return_dict=True)
    print("\nTraining Mode Outputs:")
    print(f"  Forensic logits shape:  {outputs['forensic_logits'].shape}")
    print(f"  Latent features shape:  {outputs['latent_features'].shape}")
    print(f"  Active adversary heads: {list(outputs['attribute_logits'].keys())}")
    for k, v in outputs['attribute_logits'].items():
        print(f"    - {k:<15} logits shape: {v.shape}")

    # 2. Inference mode forward pass (drop-in compatibility)
    model.eval()
    with torch.no_grad():
        eval_out = model(dummy_input)
    print("\nInference Mode Output (Drop-in compatibility):")
    print(f"  Output shape: {eval_out.shape} (Expected: [4])")
    print("=" * 70)
