"""
Unit tests for FairFake Feature 1.1 (Metric Sweep Engine)
"""

import os
import unittest
import numpy as np
import pandas as pd
from backend.calibration.metric_sweep import (
    generate_benchmark_validation_df,
    sweep_cohort_metrics,
    run_threshold_sweep,
    load_sweep_results,
    DEFAULT_CURVES_PATH
)

class TestMetricSweep(unittest.TestCase):
    def test_benchmark_generation(self):
        df = generate_benchmark_validation_df(n_samples=500, random_seed=42)
        self.assertEqual(len(df), 500)
        self.assertIn("predicted_prob", df.columns)
        self.assertIn("true_label", df.columns)
        self.assertIn("Dark_Skin", df.columns)
        self.assertTrue((df["predicted_prob"] >= 0.0).all() and (df["predicted_prob"] <= 1.0).all())

    def test_sweep_bounds_and_structure(self):
        df = generate_benchmark_validation_df(n_samples=200, random_seed=123)
        thresholds = np.array([0.3, 0.5, 0.7])
        points = sweep_cohort_metrics(df, thresholds)
        self.assertEqual(len(points), 3)
        for pt in points:
            self.assertTrue(0.0 <= pt["tpr"] <= 1.0)
            self.assertTrue(0.0 <= pt["fpr"] <= 1.0)
            self.assertTrue(0.0 <= pt["accuracy"] <= 1.0)

    def test_saved_json_curves(self):
        self.assertTrue(os.path.exists(DEFAULT_CURVES_PATH), "Curves JSON must exist on disk")
        data = load_sweep_results(DEFAULT_CURVES_PATH)
        self.assertIn("metadata", data)
        self.assertIn("global_curves", data)
        self.assertIn("cohort_curves", data)
        self.assertIn("snapshot_at_05", data)
        
        # Verify key attributes exist
        for key in ["Dark_Skin", "Heavy_Makeup", "Eyeglasses", "Senior"]:
            self.assertIn(key, data["cohort_curves"])
            points = data["cohort_curves"][key]["points"]
            self.assertEqual(len(points), 99)

if __name__ == "__main__":
    unittest.main()
