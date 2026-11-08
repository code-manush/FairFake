"""
Unit tests for FairFake Feature 1.2 (Fairness Threshold Solver)
"""

import os
import unittest
from backend.calibration.fair_solver import (
    load_calibrated_thresholds,
    solve_all_thresholds,
    solve_cohort_threshold,
    get_cohort_threshold,
    DEFAULT_THRESHOLDS_PATH
)
from backend.calibration.metric_sweep import load_sweep_results, DEFAULT_CURVES_PATH

class TestFairSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.curves_data = load_sweep_results(DEFAULT_CURVES_PATH)

    def test_solve_cohort_threshold(self):
        dark_skin_pts = self.curves_data["cohort_curves"]["Dark_Skin"]["points"]
        res = solve_cohort_threshold(
            points=dark_skin_pts,
            target_tpr=0.72,
            target_fpr=0.33,
            criterion="fpr_parity"
        )
        self.assertIn("calibrated_threshold", res)
        self.assertIn("uncalibrated_threshold", res)
        self.assertIn("calibrated_metrics", res)
        self.assertIn("improvements", res)
        # For Dark Skin under FPR parity, threshold should shift upward to protect real faces
        self.assertGreater(res["calibrated_threshold"], 0.50)
        # Calibrated FPR should be lower than uncalibrated FPR
        self.assertLess(res["calibrated_metrics"]["fpr"], res["uncalibrated_metrics"]["fpr"])

    def test_solve_all_thresholds(self):
        solved = solve_all_thresholds(self.curves_data, criterion="fpr_parity")
        self.assertIn("metadata", solved)
        self.assertIn("thresholds", solved)
        self.assertIn("calibrations", solved)
        
        # Verify calibrated keys
        for attr in ["Dark_Skin", "Heavy_Makeup", "Eyeglasses", "Senior"]:
            self.assertIn(attr, solved["thresholds"])
            tau = solved["thresholds"][attr]
            self.assertTrue(0.38 <= tau <= 0.65)

    def test_json_persistence_and_lookup(self):
        self.assertTrue(os.path.exists(DEFAULT_THRESHOLDS_PATH))
        data = load_calibrated_thresholds(DEFAULT_THRESHOLDS_PATH)
        self.assertIn("thresholds", data)
        
        tau_dark = get_cohort_threshold("Dark_Skin", data)
        self.assertIsInstance(tau_dark, float)
        self.assertGreater(tau_dark, 0.50)
        
        # Unknown attribute fallback
        tau_unknown = get_cohort_threshold("NonExistentTrait", data, default_tau=0.50)
        self.assertEqual(tau_unknown, 0.50)

if __name__ == "__main__":
    unittest.main()
