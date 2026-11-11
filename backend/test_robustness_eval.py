"""
Unit tests for Environmental Robustness Auditing Suite (Phase 3: Task 3.3)
"""

import unittest
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from robustness_eval import (
    calculate_cohort_robustness_metrics,
    simulate_perturbed_predictions,
    RobustnessAuditor
)


class TestRobustnessAuditor(unittest.TestCase):

    def setUp(self):
        # Create small synthetic test dataframe
        np.random.seed(42)
        n = 100
        self.test_df = pd.DataFrame({
            "id": np.arange(n),
            "true_label": np.random.choice([0, 1], size=n, p=[0.5, 0.5]),
            "predicted_prob": np.random.uniform(0.1, 0.9, size=n),
            "Dark_Skin": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
            "Heavy_Makeup": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
            "Eyeglasses": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
            "Senior": np.random.choice([0, 1], size=n, p=[0.9, 0.1])
        })
        self.csv_path = str(backend_dir / "temp_test_audit.csv")
        self.test_df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def test_calculate_cohort_metrics(self):
        self.test_df["pred_prob_perturbed"] = self.test_df["predicted_prob"]
        metrics = calculate_cohort_robustness_metrics(
            self.test_df, "Dark_Skin", prob_col="pred_prob_perturbed"
        )
        self.assertIn("accuracy", metrics)
        self.assertIn("error_rate", metrics)
        self.assertIn("pdrp", metrics)
        self.assertIn("ddrp", metrics)
        self.assertIn("crp", metrics)
        self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)

    def test_simulate_perturbed_predictions_shifts(self):
        # Underexposure (gamma > 1) should shift dark skin predictions towards fake
        p_gamma = simulate_perturbed_predictions(
            self.test_df, condition_type="gamma", severity=1.8, use_srm_hardening=False
        )
        dark_idx = self.test_df[self.test_df["Dark_Skin"] == 1].index
        mean_orig = self.test_df.loc[dark_idx, "predicted_prob"].mean()
        mean_pert = p_gamma.loc[dark_idx].mean()
        self.assertGreater(mean_pert, mean_orig)

    def test_srm_hardening_mitigates_disparity(self):
        # Without hardening
        p_unhardened = simulate_perturbed_predictions(
            self.test_df, condition_type="gamma", severity=1.8, use_srm_hardening=False
        )
        # With hardening
        p_hardened = simulate_perturbed_predictions(
            self.test_df, condition_type="gamma", severity=1.8, use_srm_hardening=True
        )

        dark_idx = self.test_df[self.test_df["Dark_Skin"] == 1].index
        shift_unhardened = abs(p_unhardened.loc[dark_idx].mean() - self.test_df.loc[dark_idx, "predicted_prob"].mean())
        shift_hardened = abs(p_hardened.loc[dark_idx].mean() - self.test_df.loc[dark_idx, "predicted_prob"].mean())

        self.assertLess(shift_hardened, shift_unhardened)

    def test_run_comprehensive_audit(self):
        auditor = RobustnessAuditor(dataset_path=self.csv_path)
        report = auditor.run_comprehensive_audit(samples=50, use_srm_hardening=True)
        self.assertIn("metadata", report)
        self.assertIn("clean_baseline", report)
        self.assertIn("stress_sweeps", report)
        self.assertIn("disparity_amplification_summary", report)
        self.assertEqual(report["metadata"]["model_hardened"], True)


if __name__ == "__main__":
    unittest.main()
