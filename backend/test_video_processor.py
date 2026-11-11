"""
Unit tests for Spatio-Temporal Video Forensics Engine (Phase 4: Task 4.1)
"""

import unittest
import sys
from pathlib import Path
import numpy as np

backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from video_processor import (
    compute_optical_flow_magnitude,
    compute_flickering_index,
    estimate_eye_aspect_ratio_simulated,
    audit_blink_rate,
    extract_rppg_signals,
    VideoForensicsProcessor
)


class TestVideoProcessor(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Create 12 sequential frames
        self.frames = []
        for i in range(12):
            frame = np.full((96, 96, 3), 110, dtype=np.uint8)
            # Add subtle motion
            cv2_shift = int(i * 2)
            frame[20+cv2_shift:50+cv2_shift, 20:60, :] = 180
            self.frames.append(frame)

        self.probs = [0.15 + (i * 0.04) for i in range(12)]

    def test_optical_flow(self):
        gray1 = self.frames[0][..., 0]
        gray2 = self.frames[1][..., 0]
        mag, var = compute_optical_flow_magnitude(gray1, gray2)
        self.assertGreater(mag, 0.0)
        self.assertIsInstance(var, float)

    def test_flickering_index(self):
        gray_seq = [f[..., 0] for f in self.frames]
        flicker = compute_flickering_index(gray_seq)
        self.assertTrue(0.0 <= flicker <= 1.0)

    def test_blink_rate_audit(self):
        # Timeline with periodic blinks (EAR dips below 0.20)
        ear_timeline = [0.30, 0.30, 0.15, 0.30, 0.30, 0.30, 0.16, 0.30] * 5
        metrics = audit_blink_rate(ear_timeline, fps=10.0)
        self.assertIn("blink_count", metrics)
        self.assertIn("blinks_per_minute", metrics)
        self.assertIn("blink_status", metrics)
        self.assertGreater(metrics["blink_count"], 0)

    def test_rppg_pulse_extraction(self):
        rppg = extract_rppg_signals(self.frames)
        self.assertIn("bvp_correlation", rppg)
        self.assertIn("estimated_heart_rate_bpm", rppg)
        self.assertIn("rppg_verdict", rppg)
        self.assertIn("pulse_synchrony_score", rppg)
        self.assertTrue(45.0 <= rppg["estimated_heart_rate_bpm"] <= 150.0)

    def test_full_video_forensics_orchestrator(self):
        processor = VideoForensicsProcessor()
        res = processor.process_frames_sequence(self.frames, self.probs, fps=6.0)
        self.assertIn("duration_seconds", res)
        self.assertIn("temporal_consistency", res)
        self.assertIn("biological_signals", res)
        self.assertIn("per_second_timeline", res)
        self.assertEqual(res["overall_verdict"], "FAKE" if max(self.probs) >= 0.5 else "REAL")


if __name__ == "__main__":
    unittest.main()
