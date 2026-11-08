"""
Unit tests for FairFake Feature 1.3 (Demographic Temperature Scaler)
"""

import os
import unittest
import numpy as np
from backend.calibration.temperature_scaler import (
    prob_to_logit,
    logit_to_prob,
    scale_probabilities,
    compute_ece,
    compute_nll,
    fit_temperature,
    load_temperature_weights,
    calibrate_probability,
    DEFAULT_WEIGHTS_PATH
)

class TestTemperatureScaler(unittest.TestCase):
    def test_logit_prob_invertibility(self):
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        z = prob_to_logit(probs)
        reconstructed = logit_to_prob(z)
        np.testing.assert_allclose(probs, reconstructed, atol=1e-5)

    def test_temperature_softening(self):
        # High probability with T > 1 should be softened towards 0.5
        raw_prob = np.array([0.90])
        softened = scale_probabilities(raw_prob, temperature=2.0)
        self.assertLess(softened[0], 0.90)
        self.assertGreater(softened[0], 0.50)

        # Low probability with T > 1 should increase towards 0.5
        raw_low = np.array([0.10])
        softened_low = scale_probabilities(raw_low, temperature=2.0)
        self.assertGreater(softened_low[0], 0.10)
        self.assertLess(softened_low[0], 0.50)

    def test_fit_temperature_reduces_ece(self):
        np.random.seed(42)
        # Synthetic overconfident probabilities
        probs = np.clip(np.random.beta(5, 1, size=200), 0.05, 0.95)
        # Random true labels with 50% accuracy
        y_true = np.random.binomial(1, 0.5, size=200).astype(float)

        t_opt, uncal, cal = fit_temperature(probs, y_true)
        self.assertGreater(t_opt, 1.0, "Overconfident models should have T > 1")
        self.assertLess(cal["ece"], uncal["ece"], "Calibration must reduce ECE")

    def test_json_weights_persistence_and_inference(self):
        self.assertTrue(os.path.exists(DEFAULT_WEIGHTS_PATH))
        weights = load_temperature_weights(DEFAULT_WEIGHTS_PATH)
        self.assertIn("temperatures", weights)
        self.assertIn("metadata", weights)

        # Test inference function
        res = calibrate_probability(0.85, ["Dark_Skin", "Heavy_Makeup"], weights)
        self.assertIn("calibrated_probability", res)
        self.assertIn("effective_temperature", res)
        self.assertGreater(res["effective_temperature"], 1.5)
        self.assertLess(res["calibrated_probability"], 0.85)

if __name__ == "__main__":
    unittest.main()
