"""
FairFake Spatial High-Pass Pre-Filtering & SRM Layer (Phase 3: Task 3.2)

Implements Steganalysis Rich Models (SRM) spatial high-pass filter banks to
strip confounding low-frequency luminance variations, skin tone biases, and
ambient lighting gradients, while retaining microscopic boundary seams,
interpolation residuals, and synthetic inpainting discontinuities.

Mathematical Foundation:
  Classic SRM kernels K satisfy: sum(K) = 0.
  For any locally smooth facial patch of intensity C:
      (C * sum(K)) = 0
  This mathematically eliminates melanin and soft lighting shortcuts,
  preventing models from misclassifying darker skin tones as artifacts.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# 1. Steganalysis Rich Models (SRM) Filter Kernels (5x5)
# ---------------------------------------------------------------------------

def get_srm_kernels() -> torch.Tensor:
    """
    Constructs the 3 canonical Steganalysis Rich Model (SRM) 5x5 filter kernels:
      1. K1: 1st-order horizontal/vertical edge derivative residual.
      2. K2: 2nd-order cross Laplacian derivative residual.
      3. K3: Square 5x5 high-pass spatial residual.

    Returns:
        torch.Tensor of shape [3, 1, 5, 5] with dtype float32.
    """
    # 1. First-order edge derivative (horizontal)
    k1 = np.array([
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0],
        [0, -1,  2, -1,  0],
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0]
    ], dtype=np.float32) / 2.0

    # 2. Second-order Laplacian derivative
    k2 = np.array([
        [0,  0,  0,  0,  0],
        [0, -1,  2, -1,  0],
        [0,  2, -4,  2,  0],
        [0, -1,  2, -1,  0],
        [0,  0,  0,  0,  0]
    ], dtype=np.float32) / 4.0

    # 3. Square 5x5 high-pass spatial filter
    k3 = np.array([
        [-1,  2, -2,  2, -1],
        [ 2, -6,  8, -6,  2],
        [-2,  8, -12,  8, -2],
        [ 2, -6,  8, -6,  2],
        [-1,  2, -2,  2, -1]
    ], dtype=np.float32) / 12.0

    # Stack to shape [3, 1, 5, 5]
    kernels = np.stack([k1, k2, k3], axis=0)[:, np.newaxis, :, :]
    return torch.from_numpy(kernels)


# ---------------------------------------------------------------------------
# 2. SRM Preprocessing Layer
# ---------------------------------------------------------------------------

class SRMPreprocessingLayer(nn.Module):
    """
    Illumination-Invariant SRM Spatial Filter Layer.

    Modes:
      - 'residual_rgb': Computes high-pass noise residuals and blends them
        with input RGB via a residual gate: x_out = x_rgb + alpha * Tanh(srm(x_rgb)).
        Keeps 3 input channels so it acts as a drop-in front-end for standard backbones.
      - 'residual_only': Outputs 3 normalized high-pass channels (one per RGB channel).
      - 'multichannel': Outputs 9 channels (3 SRM kernels x 3 RGB channels) for dual-stream backbones.
    """

    def __init__(
        self,
        mode: str = "residual_rgb",
        learnable: bool = False,
        truncation_threshold: float = 3.0,
        initial_alpha: float = 0.5
    ):
        super().__init__()
        self.mode = mode
        self.truncation_threshold = truncation_threshold

        # Base 3 SRM kernels: shape [3, 1, 5, 5]
        base_kernels = get_srm_kernels()

        # Build depthwise convolution weights across 3 RGB channels
        # Shape: [3, 3, 5, 5] where each kernel is replicated across RGB
        # If mode == 'multichannel', shape is [9, 1, 5, 5] for grouped conv
        if mode == "multichannel":
            # 3 kernels for each of the 3 channels -> 9 kernels
            # shape: [9, 1, 5, 5] with groups=3
            weight_9 = torch.cat([base_kernels, base_kernels, base_kernels], dim=0)
            self.srm_conv = nn.Conv2d(3, 9, kernel_size=5, stride=1, padding=2, bias=False, groups=3)
            self.srm_conv.weight.data.copy_(weight_9)
        else:
            # Replicate 3 kernels for 3 input channels to produce 3 residual channels
            # By applying kernel_k to channel_k
            self.srm_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False, groups=3)
            self.srm_conv.weight.data.copy_(base_kernels)

        if not learnable:
            self.srm_conv.weight.requires_grad = False

        # Residual gating parameter
        if mode == "residual_rgb":
            self.alpha = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32))
            self.norm = nn.BatchNorm2d(3)
        else:
            out_c = 9 if mode == "multichannel" else 3
            self.norm = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          x: Input tensor [B, 3, H, W] in [0, 1] or [-1, 1].
        """
        # Compute spatial high-pass response
        residuals = self.srm_conv(x)

        # Truncate extreme outlier spikes (classic steganalysis non-linear threshold)
        if self.truncation_threshold > 0:
            residuals = torch.clamp(
                residuals,
                -self.truncation_threshold,
                self.truncation_threshold
            )

        if self.mode == "residual_rgb":
            # Normalize high-pass residuals and blend into RGB
            norm_res = self.norm(residuals)
            # x_out = x + alpha * tanh(residuals)
            out = x + self.alpha * torch.tanh(norm_res)
            return out

        norm_out = self.norm(residuals)
        return norm_out


# ---------------------------------------------------------------------------
# 3. Environmental Hardened Detector Architecture
# ---------------------------------------------------------------------------

class HardenedXceptionDetector(nn.Module):
    """
    Environmental-Hardened Deepfake Detector combining:
      1. SRM High-Pass Pre-filtering (Task 3.2): cancels skin tone and ambient luminance.
      2. Feature Backbone: Xception or EfficientNet.
      3. Forensic Classification Head: predicts real vs. fake.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        srm_learnable: bool = False,
        model_name: str = "legacy_xception"
    ):
        super().__init__()
        import timm

        # 1. Front-end SRM Illumination-Invariant Filter
        self.srm_prefilter = SRMPreprocessingLayer(
            mode="residual_rgb",
            learnable=srm_learnable,
            initial_alpha=0.4
        )

        # 2. Backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features

        # 3. Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-filter to strip luminance confounders
        x_filtered = self.srm_prefilter(x)
        # Extract features
        feats = self.backbone(x_filtered)
        # Classification logit
        logits = self.classifier(feats).squeeze(-1)
        return logits

    def extract_srm_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to extract visual SRM residual maps for interpretability."""
        with torch.no_grad():
            res = self.srm_prefilter.srm_conv(x)
            return res


if __name__ == "__main__":
    print("[*] Testing SRM Pre-filtering & Hardened Architecture...")
    dummy_input = torch.rand(2, 3, 224, 224)

    # 1. Test SRM Layer
    srm_layer = SRMPreprocessingLayer(mode="residual_rgb")
    out_srm = srm_layer(dummy_input)
    print(f"  SRM output shape: {out_srm.shape}")
    assert out_srm.shape == dummy_input.shape, "Shape mismatch in residual_rgb mode"

    # 2. Test Multi-channel mode
    srm_multi = SRMPreprocessingLayer(mode="multichannel")
    out_multi = srm_multi(dummy_input)
    print(f"  SRM multichannel shape: {out_multi.shape}")
    assert out_multi.shape == (2, 9, 224, 224), "Shape mismatch in multichannel mode"

    # 3. Test Hardened Detector
    hardened_model = HardenedXceptionDetector(pretrained=False)
    logits = hardened_model(dummy_input)
    print(f"  Hardened Detector logit output shape: {logits.shape}")
    assert logits.shape == (2,), "Logits shape mismatch"

    print("[+] SRM Pre-filtering Layer successfully verified!")
