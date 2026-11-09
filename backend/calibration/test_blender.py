"""
Unit tests for FairFake Feature 1.4 (Multi-Attribute Threshold Blender)
"""

import os
import unittest
from backend.calibration.blender import (
    MultiAttributeBlender,
    normalize_attribute_inputs,
    evaluate_fair_verdict
)

class TestMultiAttributeBlender(unittest.TestCase):
    def test_normalize_attribute_inputs(self):
        # List of strings
        res1 = normalize_attribute_inputs(["dark_skin", "glasses", "makeup"])
        self.assertIn("Dark_Skin", res1)
        self.assertIn("Eyeglasses", res1)
        self.assertIn("Heavy_Makeup", res1)

        # DeepFace format dict
        res2 = normalize_attribute_inputs({
            "gender": "Woman",
            "skinTone": "Dark_Skin",
            "glasses": True,
            "heavyMakeup": False
        })
        self.assertIn("Dark_Skin", res2)
        self.assertIn("Eyeglasses", res2)
        self.assertNotIn("Heavy_Makeup", res2)

    def test_threshold_blending_bounds(self):
        blender = MultiAttributeBlender()
        # Even with multiple high-shift attributes, effective threshold must stay within bounds [0.35, 0.72]
        tau_eff, contribs = blender.blend_thresholds(["Dark_Skin", "Heavy_Makeup", "Eyeglasses", "Wearing_Hat"])
        self.assertLessEqual(tau_eff, 0.72)
        self.assertGreaterEqual(tau_eff, 0.35)
        self.assertGreater(tau_eff, 0.50)
        self.assertEqual(len(contribs), 4)

    def test_verdict_flipping_logic(self):
        # Score of 0.54 would normally be FAKE at tau=0.50
        # But with Dark_Skin and Heavy_Makeup, it should flip to REAL
        eval_res = evaluate_fair_verdict(0.54, ["Dark_Skin", "Heavy_Makeup"])
        self.assertEqual(eval_res["raw"]["verdict"], "FAKE")
        self.assertEqual(eval_res["calibrated"]["verdict"], "REAL")
        self.assertTrue(eval_res["comparison"]["is_verdict_flipped"])
        self.assertEqual(eval_res["comparison"]["shift_type"], "FALSE_ALARM_PREVENTED")
        self.assertIn("Dark_Skin", eval_res["explanation"])

    def test_unadorned_consistency(self):
        # Unadorned face with clear fake score 0.88 should remain FAKE
        eval_res = evaluate_fair_verdict(0.88, [])
        self.assertEqual(eval_res["raw"]["verdict"], "FAKE")
        self.assertEqual(eval_res["calibrated"]["verdict"], "FAKE")
        self.assertFalse(eval_res["comparison"]["is_verdict_flipped"])
        self.assertEqual(eval_res["calibrated"]["effective_threshold"], 0.50)

if __name__ == "__main__":
    unittest.main()
