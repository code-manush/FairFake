"""
Unit tests for SRM Pre-filtering & Hardened Detector (Phase 3: Task 3.2)
"""

import unittest
import sys
from pathlib import Path
import torch

src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from srm_filter import (
    get_srm_kernels,
    SRMPreprocessingLayer,
    HardenedXceptionDetector
)


class TestSRMFilter(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.input_tensor = torch.rand(self.batch_size, 3, 128, 128, dtype=torch.float32)

    def test_srm_kernel_zero_sum(self):
        kernels = get_srm_kernels()  # [3, 1, 5, 5]
        self.assertEqual(kernels.shape, (3, 1, 5, 5))
        for i in range(3):
            k = kernels[i, 0]
            k_sum = torch.sum(k).item()
            # SRM kernels must sum to ~0 to eliminate constant luminance
            self.assertAlmostEqual(k_sum, 0.0, places=4, msg=f"Kernel {i} sum is not zero: {k_sum}")

    def test_constant_input_annihilation(self):
        """A solid color image has zero high-frequency response."""
        srm_layer = SRMPreprocessingLayer(mode="residual_only", truncation_threshold=0.0)
        solid_img = torch.full((1, 3, 64, 64), 0.7, dtype=torch.float32)
        with torch.no_grad():
            res = srm_layer.srm_conv(solid_img)
            # Center of the convolution (away from borders) should be strictly 0
            center_patch = res[:, :, 10:54, 10:54]
            max_abs_diff = torch.max(torch.abs(center_patch)).item()
            self.assertAlmostEqual(max_abs_diff, 0.0, places=5)

    def test_srm_preprocessing_residual_rgb_mode(self):
        srm_layer = SRMPreprocessingLayer(mode="residual_rgb")
        out = srm_layer(self.input_tensor)
        self.assertEqual(out.shape, self.input_tensor.shape)
        # Should retain grad for alpha parameter
        self.assertTrue(srm_layer.alpha.requires_grad)

    def test_srm_preprocessing_multichannel_mode(self):
        srm_layer = SRMPreprocessingLayer(mode="multichannel")
        out = srm_layer(self.input_tensor)
        self.assertEqual(out.shape, (self.batch_size, 9, 128, 128))

    def test_hardened_xception_forward(self):
        model = HardenedXceptionDetector(pretrained=False, model_name="legacy_xception")
        model.eval()
        with torch.no_grad():
            logits = model(self.input_tensor)
            self.assertEqual(logits.shape, (self.batch_size,))
            
            # Check visual residual extraction
            res = model.extract_srm_residual(self.input_tensor)
            self.assertEqual(res.shape[0], self.batch_size)


if __name__ == "__main__":
    unittest.main()
