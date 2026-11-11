"""
Unit tests for FairFake Standalone SDK & Certification (Phase 5: Tasks 5.1 & 5.2)
"""

import unittest
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sdk_dir = project_root / "fairfake_sdk"
if str(sdk_dir) not in sys.path:
    sys.path.insert(0, str(sdk_dir))

import fairfake as ff
from fairfake.certification import (
    determine_grade,
    generate_scorecard,
    generate_compliance_badge
)


class TestFairFakeSDK(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 80
        self.df = pd.DataFrame({
            "id": np.arange(n),
            "true_label": np.random.choice([0, 1], size=n, p=[0.5, 0.5]),
            "predicted_prob": np.random.uniform(0.1, 0.9, size=n),
            "Dark_Skin": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
            "Heavy_Makeup": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
            "Eyeglasses": np.random.choice([0, 1], size=n, p=[0.85, 0.15])
        })
        self.tmp_csv = str(project_root / "temp_sdk_test.csv")
        self.df.to_csv(self.tmp_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.tmp_csv):
            os.remove(self.tmp_csv)

    def test_auditor_with_dataframe(self):
        auditor = ff.Auditor(dataset=self.df)
        report = auditor.run_audit(threshold=0.5)
        d = report.to_dict()
        self.assertIn("metadata", d)
        self.assertIn("summary", d)
        self.assertIn("cohort_metrics", d)
        self.assertIn("Dark_Skin", d["cohort_metrics"])

    def test_auditor_with_csv(self):
        auditor = ff.Auditor(dataset=self.tmp_csv)
        report = auditor.run_audit(threshold=0.5)
        self.assertEqual(report.result.total_samples, 80)
        self.assertTrue(0.0 <= report.result.global_accuracy <= 1.0)

    def test_determine_grade(self):
        # Grade A+ test
        grade_aplus = determine_grade(max_abs_crp=0.05, max_pdrp=1.1, max_ddrp=1.1)
        self.assertEqual(grade_aplus["grade"], "A+")

        # Grade B test
        grade_b = determine_grade(max_abs_crp=0.18, max_pdrp=1.4, max_ddrp=1.4)
        self.assertEqual(grade_b["grade"], "B")

        # Grade F test
        grade_f = determine_grade(max_abs_crp=0.45, max_pdrp=2.5, max_ddrp=1.5)
        self.assertEqual(grade_f["grade"], "F")

    def test_generate_scorecard_and_badge(self):
        auditor = ff.Auditor(dataset=self.df)
        report = auditor.run_audit()
        card = generate_scorecard(report.to_dict(), model_name="TestModel-v1")
        self.assertIn("certification", card)
        self.assertIn("fairness_grade", card["certification"])
        self.assertIn("metrics_summary", card)

        badge_svg = generate_compliance_badge(card["certification"]["fairness_grade"])
        self.assertTrue(badge_svg.startswith("<svg"))
        self.assertIn(f"Grade {card['certification']['fairness_grade']}", badge_svg)


if __name__ == "__main__":
    unittest.main()
