"""
Unit tests for Audio-Visual Lip-Sync Mismatch Detection Engine (Phase 4: Task 4.3)
"""

import unittest
import sys
from pathlib import Path
import numpy as np

backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from multimodal_sync import (
    extract_mouth_aperture,
    extract_mouth_motion_energy,
    compute_audio_energy_envelope,
    AudioVisualSyncAuditor
)


class TestMultimodalSync(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Create synthetic face crop
        self.face = np.full((100, 100, 3), 130, dtype=np.uint8)
        # Add mouth opening in lower section
        self.face[65:85, 35:65, :] = 20  # dark oral cavity

    def test_mouth_aperture_extraction(self):
        v_ap, h_ap = extract_mouth_aperture(self.face)
        self.assertTrue(0.0 <= v_ap <= 1.0)
        self.assertTrue(0.0 <= h_ap <= 1.0)
        self.assertGreater(v_ap, 0.05)

    def test_audio_energy_envelope(self):
        audio = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)
        env = compute_audio_energy_envelope(audio, sample_rate=16000, target_fps=30.0)
        self.assertEqual(len(env), 30)
        self.assertTrue(0.0 <= env.min() and env.max() <= 1.0)

    def test_synchronous_audit_detection(self):
        # 30 frame perfectly in-sync speech
        t = np.linspace(0, 4 * np.pi, 30)
        vis_signal = list(0.3 + 0.2 * np.sin(t))
        aud_signal = np.array(0.3 + 0.2 * np.sin(t), dtype=np.float32)

        auditor = AudioVisualSyncAuditor(fps=30.0)
        res = auditor.audit_sync(vis_signal, aud_signal)
        self.assertEqual(res["multimodal_verdict"], "SYNCHRONOUS_AUDIO_VISUAL")
        self.assertGreater(res["sync_correlation"], 0.70)
        self.assertAlmostEqual(res["temporal_offset_ms"], 0.0, delta=35.0)

    def test_desynchronized_audit_detection(self):
        # Vis signal periodic, audio signal random white noise
        vis_signal = list(0.3 + 0.2 * np.sin(np.linspace(0, 4 * np.pi, 30)))
        aud_signal = np.random.uniform(0.1, 0.9, 30).astype(np.float32)

        auditor = AudioVisualSyncAuditor(fps=30.0)
        res = auditor.audit_sync(vis_signal, aud_signal)
        self.assertIn("multimodal_verdict", res)
        self.assertIn("sync_confidence", res)
        self.assertIn("temporal_offset_ms", res)


if __name__ == "__main__":
    unittest.main()
