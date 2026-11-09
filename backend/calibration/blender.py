"""
FairFake Multi-Attribute Threshold Blending Engine (Feature 1.4)

Combines cohort-specific decision thresholds and temperature scales when a face
exhibits multiple overlapping demographic/appearance attributes.
Computes a unified effective threshold, calibrated probability, flipped-verdict status,
and human-readable forensic justification.
"""

import os
import sys
import math
import json
from typing import Dict, List, Any, Optional, Union, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
for p in [root_dir, parent_dir, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend.calibration.fair_solver import (
        load_calibrated_thresholds,
        DEFAULT_THRESHOLDS_PATH
    )
    from backend.calibration.temperature_scaler import (
        load_temperature_weights,
        calibrate_probability,
        DEFAULT_WEIGHTS_PATH
    )
except ImportError:
    from fair_solver import (
        load_calibrated_thresholds,
        DEFAULT_THRESHOLDS_PATH
    )
    from temperature_scaler import (
        load_temperature_weights,
        calibrate_probability,
        DEFAULT_WEIGHTS_PATH
    )

# Standard attribute alias mapper to normalize dictionary keys from DeepFace or UI
ATTRIBUTE_ALIAS_MAP = {
    # Skin tone
    "dark_skin": "Dark_Skin",
    "medium_skin": "Medium_Skin",
    "light_skin": "Light_Skin",
    "black": "Black",
    "white": "White",
    "asian": "Asian",
    # Age
    "senior": "Senior",
    "young": "Young",
    "middle-aged": "Young",  # fallback
    # Accessories
    "glasses": "Eyeglasses",
    "eyeglasses": "Eyeglasses",
    "heavymakeup": "Heavy_Makeup",
    "heavy_makeup": "Heavy_Makeup",
    "makeup": "Heavy_Makeup",
    "wearing_hat": "Wearing_Hat",
    "hat": "Wearing_Hat",
    "wearing_lipstick": "Wearing_Lipstick",
    "lipstick": "Wearing_Lipstick",
    # Facial hair / features
    "bald": "Bald",
    "mustache": "Mustache",
    "goatee": "Goatee",
    "smiling": "Smiling"
}


def normalize_attribute_inputs(
    attributes: Union[List[str], Dict[str, Any], None]
) -> List[str]:
    """
    Normalizes various attribute formats (list of strings, DeepFace dict, bool flags)
    into a standardized list of recognized attribute names.
    """
    if not attributes:
        return []

    recognized = set()

    if isinstance(attributes, list):
        for item in attributes:
            if not isinstance(item, str):
                continue
            clean = item.strip()
            lookup = clean.lower()
            if lookup in ATTRIBUTE_ALIAS_MAP:
                recognized.add(ATTRIBUTE_ALIAS_MAP[lookup])
            else:
                recognized.add(clean)

    elif isinstance(attributes, dict):
        for k, v in attributes.items():
            k_lower = k.lower()
            if isinstance(v, bool) and v:
                if k_lower in ATTRIBUTE_ALIAS_MAP:
                    recognized.add(ATTRIBUTE_ALIAS_MAP[k_lower])
                else:
                    recognized.add(k)
            elif isinstance(v, (int, float)) and v == 1:
                if k_lower in ATTRIBUTE_ALIAS_MAP:
                    recognized.add(ATTRIBUTE_ALIAS_MAP[k_lower])
                else:
                    recognized.add(k)
            elif isinstance(v, str):
                v_lower = v.lower()
                if v_lower in ATTRIBUTE_ALIAS_MAP:
                    recognized.add(ATTRIBUTE_ALIAS_MAP[v_lower])
                elif k_lower in ATTRIBUTE_ALIAS_MAP:
                    recognized.add(ATTRIBUTE_ALIAS_MAP[k_lower])

    return sorted(list(recognized))


class MultiAttributeBlender:
    """
    Blends individual cohort calibrations into a unified decision boundary.
    """

    def __init__(
        self,
        thresholds_path: str = DEFAULT_THRESHOLDS_PATH,
        weights_path: str = DEFAULT_WEIGHTS_PATH,
        min_threshold_bound: float = 0.35,
        max_threshold_bound: float = 0.72
    ):
        self.min_bound = min_threshold_bound
        self.max_bound = max_threshold_bound
        
        # Load precomputed calibration artifacts
        try:
            self.threshold_data = load_calibrated_thresholds(thresholds_path)
        except Exception:
            self.threshold_data = {"thresholds": {}, "metadata": {"global_default_threshold": 0.50}}
            
        try:
            self.weights_data = load_temperature_weights(weights_path)
        except Exception:
            self.weights_data = {"temperatures": {}, "metadata": {"global_temperature": 1.0}}
            
        self.global_threshold = float(
            self.threshold_data.get("metadata", {}).get("global_default_threshold", 0.50)
        )
        self.cohort_thresholds = self.threshold_data.get("thresholds", {})

    def blend_thresholds(
        self,
        detected_attributes: List[str],
        strategy: str = "weighted_residual"
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Combines individual attribute thresholds into an effective threshold tau_eff.
        Returns (tau_eff, list_of_contributions).
        """
        tau_0 = self.global_threshold
        contributions = []

        matching_deltas = []
        for attr in detected_attributes:
            if attr in self.cohort_thresholds:
                tau_a = float(self.cohort_thresholds[attr])
                delta_a = tau_a - tau_0
                contributions.append({
                    "attribute": attr,
                    "cohort_threshold": round(tau_a, 2),
                    "delta": round(delta_a, 2)
                })
                matching_deltas.append(delta_a)

        if not matching_deltas:
            return round(tau_0, 2), contributions

        if strategy == "max_margin":
            # Strategy: take the delta with the largest absolute deviation
            max_delta = max(matching_deltas, key=lambda d: abs(d))
            raw_eff = tau_0 + max_delta

        elif strategy == "mean":
            raw_eff = tau_0 + float(sum(matching_deltas) / len(matching_deltas))

        else: # "weighted_residual" (Default)
            # Combines deltas with sublinear damping to avoid excessive shift
            # sum_delta / sqrt(N) provides mathematically stable scaling
            total_shift = sum(matching_deltas)
            damped_shift = total_shift / math.sqrt(len(matching_deltas))
            raw_eff = tau_0 + damped_shift

        # Apply strict operational bounding
        tau_eff = max(self.min_bound, min(self.max_bound, raw_eff))
        return round(float(tau_eff), 2), contributions

    def evaluate(
        self,
        raw_prob: float,
        attributes: Union[List[str], Dict[str, Any], None] = None,
        strategy: str = "weighted_residual"
    ) -> Dict[str, Any]:
        """
        Full fairness-calibrated decision pipeline:
          1. Normalizes detected attributes.
          2. Blends cohort thresholds -> tau_effective.
          3. Applies demographic temperature scaling -> calibrated_prob.
          4. Compares standard decision (tau=0.50) vs fair calibrated decision.
          5. Generates human-readable forensic explanation.
        """
        detected_attrs = normalize_attribute_inputs(attributes)

        # 1. Standard Uncalibrated Verdict (tau = 0.50)
        raw_verdict = "FAKE" if raw_prob >= self.global_threshold else "REAL"

        # 2. Dynamic Threshold Blending
        tau_eff, contributions = self.blend_thresholds(detected_attrs, strategy=strategy)

        # 3. Demographic Temperature Probability Scaling
        temp_calib = calibrate_probability(
            raw_prob=raw_prob,
            detected_attributes=detected_attrs,
            weights_data=self.weights_data
        )
        calibrated_prob = temp_calib["calibrated_probability"]
        effective_temp = temp_calib["effective_temperature"]

        # 4. Calibrated Verdict
        # Using calibrated_prob evaluated against effective threshold
        calibrated_verdict = "FAKE" if calibrated_prob >= tau_eff else "REAL"

        # 5. Shift Analysis
        is_flipped = (raw_verdict != calibrated_verdict)
        if is_flipped:
            if raw_verdict == "FAKE" and calibrated_verdict == "REAL":
                shift_type = "FALSE_ALARM_PREVENTED"
                headline = "Prevented False Alarm via Fairness Calibration"
            else:
                shift_type = "EVASION_PREVENTED"
                headline = "Caught Evasive Manipulation via Cohort Sensitivity"
        else:
            shift_type = "UNCHANGED"
            headline = "Verdict Consistent Across Calibration"

        # 6. Detailed Explanation
        if contributions:
            attr_names = [c["attribute"] for c in contributions]
            attr_str = ", ".join(attr_names)
            explanation = (
                f"Operating threshold dynamically adjusted from {self.global_threshold:.2f} to {tau_eff:.2f} "
                f"({tau_eff - self.global_threshold:+.2f}) based on detected biometric attributes: [{attr_str}]. "
                f"Probability calibrated from {raw_prob:.1%} to {calibrated_prob:.1%} (T={effective_temp:.2f})."
            )
        else:
            explanation = (
                f"No high-bias attributes detected. Evaluated against global standard threshold {self.global_threshold:.2f}."
            )

        # Confidence metric (distance from decision boundary)
        conf_dist = abs(calibrated_prob - tau_eff)
        # Scale to 0-100% confidence
        calibrated_confidence = round(min(99.9, max(50.0, 50.0 + (conf_dist * 100.0))), 1)

        return {
            "raw": {
                "probability": round(raw_prob, 4),
                "threshold": round(self.global_threshold, 2),
                "verdict": raw_verdict
            },
            "calibrated": {
                "probability": round(calibrated_prob, 4),
                "effective_threshold": round(tau_eff, 2),
                "verdict": calibrated_verdict,
                "confidence": calibrated_confidence,
                "effective_temperature": effective_temp
            },
            "comparison": {
                "is_verdict_flipped": is_flipped,
                "shift_type": shift_type,
                "headline": headline,
                "threshold_delta": round(tau_eff - self.global_threshold, 2),
                "probability_delta": round(calibrated_prob - raw_prob, 4)
            },
            "biometric_context": {
                "detected_attributes": detected_attrs,
                "contributing_cohorts": contributions
            },
            "explanation": explanation
        }


# Global singleton instance for easy import
blender_instance = MultiAttributeBlender()


def evaluate_fair_verdict(
    raw_prob: float,
    attributes: Union[List[str], Dict[str, Any], None] = None
) -> Dict[str, Any]:
    """Convenience function calling global blender instance."""
    return blender_instance.evaluate(raw_prob, attributes)


if __name__ == "__main__":
    print("=" * 80)
    print("FAIRFAKE MULTI-ATTRIBUTE THRESHOLD BLENDING TEST")
    print("=" * 80)

    # Test Case 1: Unadorned face
    case1 = evaluate_fair_verdict(0.53, [])
    print("\nCase 1: Unadorned Face (prob = 0.53)")
    print(f"Verdict: {case1['raw']['verdict']} -> {case1['calibrated']['verdict']} (Flipped: {case1['comparison']['is_verdict_flipped']})")
    print(f"Explanation: {case1['explanation']}")

    # Test Case 2: Genuine Face with Dark Skin and Heavy Makeup
    case2 = evaluate_fair_verdict(0.56, ["Dark_Skin", "Heavy_Makeup"])
    print("\nCase 2: Dark Skin + Heavy Makeup (prob = 0.56)")
    print(f"Verdict: {case2['raw']['verdict']} -> {case2['calibrated']['verdict']} (Flipped: {case2['comparison']['is_verdict_flipped']})")
    print(f"Shift Type: {case2['comparison']['shift_type']}")
    print(f"Explanation: {case2['explanation']}")

    # Test Case 3: Deepface dictionary input
    case3 = evaluate_fair_verdict(0.58, {
        "gender": "Female",
        "skinTone": "Dark_Skin",
        "glasses": True,
        "heavyMakeup": True
    })
    print("\nCase 3: DeepFace Dictionary Input (prob = 0.58)")
    print(f"Detected: {case3['biometric_context']['detected_attributes']}")
    print(f"Effective Threshold: {case3['calibrated']['effective_threshold']:.2f}")
    print(f"Verdict: {case3['raw']['verdict']} -> {case3['calibrated']['verdict']} (Flipped: {case3['comparison']['is_verdict_flipped']})")
    print(f"Explanation: {case3['explanation']}")
    print("=" * 80)
