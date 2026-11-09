"""
Integration tests for FairFake Feature 1.5 (FastAPI Calibration Endpoints)
"""

import sys
import os
import unittest
from fastapi.testclient import TestClient

# Ensure backend directory is in sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from main import app


class TestCalibrationAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_get_thresholds(self):
        resp = self.client.get("/api/calibration/thresholds")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("thresholds", data)
        self.assertIn("metadata", data)
        self.assertIn("Dark_Skin", data["thresholds"])
        self.assertGreaterEqual(data["thresholds"]["Dark_Skin"], 0.50)

    def test_get_temperatures(self):
        resp = self.client.get("/api/calibration/temperatures")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("temperatures", data)
        self.assertIn("metadata", data)
        self.assertIn("Heavy_Makeup", data["temperatures"])
        self.assertGreaterEqual(data["temperatures"]["Heavy_Makeup"], 1.5)

    def test_post_evaluate_calibration_flip(self):
        # Score of 0.56 with Dark_Skin and Heavy_Makeup should flip from FAKE to REAL
        payload = {
            "raw_probability": 0.56,
            "attributes": ["Dark_Skin", "Heavy_Makeup"]
        }
        resp = self.client.post("/api/calibration/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertEqual(res["raw"]["verdict"], "FAKE")
        self.assertEqual(res["calibrated"]["verdict"], "REAL")
        self.assertTrue(res["comparison"]["is_verdict_flipped"])
        self.assertEqual(res["comparison"]["shift_type"], "FALSE_ALARM_PREVENTED")
        self.assertGreater(res["calibrated"]["effective_threshold"], 0.60)
        self.assertIn("Dark_Skin", res["explanation"])

    def test_post_evaluate_with_dict(self):
        payload = {
            "raw_probability": 0.58,
            "attributes_dict": {
                "skinTone": "Dark_Skin",
                "heavyMakeup": True,
                "glasses": True
            }
        }
        resp = self.client.post("/api/calibration/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertIn("Dark_Skin", res["biometric_context"]["detected_attributes"])
        self.assertIn("Eyeglasses", res["biometric_context"]["detected_attributes"])
        self.assertIn("Heavy_Makeup", res["biometric_context"]["detected_attributes"])
        self.assertTrue(res["comparison"]["is_verdict_flipped"])

    def test_post_evaluate_clear_fake(self):
        payload = {
            "raw_probability": 0.92,
            "attributes": ["Heavy_Makeup"]
        }
        resp = self.client.post("/api/calibration/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertEqual(res["raw"]["verdict"], "FAKE")
        self.assertEqual(res["calibrated"]["verdict"], "FAKE")
        self.assertFalse(res["comparison"]["is_verdict_flipped"])


if __name__ == "__main__":
    unittest.main()
