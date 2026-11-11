"""
Unit tests for Environmental Perturbation Augmentation Suite (Phase 3: Task 3.1)
"""

import unittest
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch

src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from perturbation_augment import (
    GammaCorrection,
    ShadowIllumination,
    ExposurePerturbation,
    JPEGCompression,
    WebPCompression,
    BlockQuantization,
    GaussianBlurPerturbation,
    MotionBlur,
    DefocusBlur,
    RobustPerturbationTransform,
    DemographicHardeningAugmentor,
    get_standard_stress_conditions,
    to_pil_image,
    from_pil_image
)


class TestPerturbationAugment(unittest.TestCase):

    def setUp(self):
        # Create synthetic test patterns
        self.pil_img = Image.new("RGB", (128, 128), color=(120, 80, 50))
        self.np_img = np.full((128, 128, 3), 128, dtype=np.uint8)
        self.tensor_img = torch.full((3, 128, 128), 0.5, dtype=torch.float32)

    def test_gamma_correction_pil_and_tensor(self):
        gc = GammaCorrection(gamma_range=(0.5, 2.0))
        out_pil = gc(self.pil_img, gamma=1.8)
        self.assertIsInstance(out_pil, Image.Image)
        self.assertEqual(out_pil.size, (128, 128))

        out_tensor = gc(self.tensor_img, gamma=0.5)
        self.assertIsInstance(out_tensor, torch.Tensor)
        self.assertEqual(out_tensor.shape, (3, 128, 128))
        self.assertTrue(out_tensor.max() <= 1.0)

    def test_shadow_illumination(self):
        shadow = ShadowIllumination()
        out = shadow(self.pil_img, intensity=0.5, angle=90)
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.size, (128, 128))

    def test_jpeg_and_webp_compression(self):
        jpeg = JPEGCompression()
        out_jpeg = jpeg(self.pil_img, quality=25)
        self.assertIsInstance(out_jpeg, Image.Image)

        webp = WebPCompression()
        out_webp = webp(self.pil_img, quality=30)
        self.assertIsInstance(out_webp, Image.Image)

    def test_block_quantization(self):
        bq = BlockQuantization(block_size=8)
        out_bq = bq(self.np_img, q_scale=5.0)
        self.assertIsInstance(out_bq, np.ndarray)
        self.assertEqual(out_bq.shape, (128, 128, 3))

    def test_optical_blurs(self):
        gb = GaussianBlurPerturbation()
        out_gb = gb(self.pil_img, sigma=2.0)
        self.assertEqual(out_gb.size, (128, 128))

        mb = MotionBlur()
        out_mb = mb(self.pil_img, kernel_size=9, angle=45)
        self.assertEqual(out_mb.size, (128, 128))

        db = DefocusBlur()
        out_db = db(self.pil_img, radius=3)
        self.assertEqual(out_db.size, (128, 128))

    def test_robust_composite_transform(self):
        transform = RobustPerturbationTransform()
        out_tensor = transform(self.tensor_img)
        self.assertEqual(out_tensor.shape, (3, 128, 128))

    def test_demographic_hardening_augmentor(self):
        augmentor = DemographicHardeningAugmentor()
        attrs = {"Dark_Skin": 1, "Heavy_Makeup": 1, "Eyeglasses": 1}
        out = augmentor(self.pil_img, attributes=attrs)
        self.assertIsInstance(out, Image.Image)

    def test_standard_stress_conditions_dict(self):
        conditions = get_standard_stress_conditions()
        self.assertIn("Clean", conditions)
        self.assertIn("Underexposure_Gamma_1_8", conditions)
        self.assertIn("JPEG_Quality_30", conditions)
        self.assertIn("Combined_Harsh_CCTV", conditions)

        for name, fn in conditions.items():
            res = fn(self.pil_img)
            self.assertIsNotNone(res, f"Condition {name} returned None")


if __name__ == "__main__":
    unittest.main()
