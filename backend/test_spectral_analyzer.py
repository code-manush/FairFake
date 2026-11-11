"""
Unit tests for Frequency-Domain Spectral Auditing Engine (Phase 4: Task 4.2)
"""

import unittest
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from spectral_analyzer import (
    to_grayscale_array,
    compute_2d_fft,
    compute_azimuthal_average,
    compute_checkerboard_score,
    compute_spectral_metrics
)


class TestSpectralAnalyzer(unittest.TestCase):

    def setUp(self):
        # 1. Natural synthetic smooth image (1/f decay)
        self.natural_img = np.zeros((128, 128), dtype=np.float32)
        for y in range(128):
            for x in range(128):
                self.natural_img[y, x] = 0.5 + 0.3 * np.sin(x / 8.0) * np.cos(y / 8.0)

        # 2. Synthetic periodic upsampling checkerboard image
        self.synth_img = self.natural_img.copy()
        self.synth_img[::4, :] += 0.35
        self.synth_img[:, ::4] += 0.35
        self.synth_img = np.clip(self.synth_img, 0.0, 1.0)

    def test_to_grayscale_array_types(self):
        # Test PIL Image
        pil_img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        arr_pil = to_grayscale_array(pil_img)
        self.assertEqual(arr_pil.shape, (64, 64))
        self.assertTrue(0.0 <= arr_pil.min() and arr_pil.max() <= 1.0)

        # Test PyTorch Tensor
        t_img = torch.rand(3, 64, 64)
        arr_t = to_grayscale_array(t_img)
        self.assertEqual(arr_t.shape, (64, 64))

    def test_compute_2d_fft(self):
        fshift, norm_spectrum = compute_2d_fft(self.natural_img)
        self.assertEqual(norm_spectrum.shape, (128, 128))
        self.assertTrue(0.0 <= norm_spectrum.min() and norm_spectrum.max() <= 1.0)

    def test_azimuthal_average(self):
        _, norm_spectrum = compute_2d_fft(self.natural_img)
        radii, profile = compute_azimuthal_average(norm_spectrum, num_bins=32)
        self.assertEqual(len(radii), 32)
        self.assertEqual(len(profile), 32)
        # First bin (DC component) should be higher than highest frequency tail
        self.assertGreater(profile[0], profile[-1])

    def test_checkerboard_detection_sensitivity(self):
        _, spec_nat = compute_2d_fft(self.natural_img)
        _, spec_syn = compute_2d_fft(self.synth_img)

        score_nat = compute_checkerboard_score(spec_nat)
        score_syn = compute_checkerboard_score(spec_syn)

        self.assertGreater(score_syn, score_nat)

    def test_compute_spectral_metrics_pipeline(self):
        metrics = compute_spectral_metrics(self.natural_img)
        self.assertIn("checkerboard_score", metrics)
        self.assertIn("high_freq_energy_ratio", metrics)
        self.assertIn("spectral_verdict", metrics)
        self.assertIn("radial_profile", metrics)
        self.assertIn("spectral_heatmap_b64", metrics)
        self.assertTrue(metrics["spectral_heatmap_b64"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
