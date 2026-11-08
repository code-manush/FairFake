"""
FairFake Demographic Temperature Scaler (Feature 1.3)

Fits cohort-specific temperature parameters T_a > 0 to calibrate posterior probabilities:
    p_cal = sigma(logit(p) / T_a)
Minimizes Negative Log-Likelihood (NLL) and Expected Calibration Error (ECE)
to prevent overconfident false alarms on adorned and underrepresented face profiles.
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from typing import Dict, List, Any, Optional, Tuple

ATTRIBUTES = [
    "Male", "Female", "Young", "Senior", "Asian", "White", "Black",
    "Dark_Skin", "Shiny_Skin", "Bald", "Wavy_Hair", "Receding_Hairline", "Bangs", 
    "Black_Hair", "Blond_Hair", "Brown_Hair", "No_Beard", "Mustache", 
    "Goatee", "Oval_Face", "Square_Face", "Double_Chin", "Chubby", 
    "Obstructed_Forehead", "Fully_Visible_Forehead", "Mouth_Closed", 
    "Smiling", "Big_Lips", "Big_Nose", "Pointy_Nose", "Heavy_Makeup", 
    "Wearing_Hat", "Wearing_Lipstick", "Eyeglasses", "Attractive"
]

DEFAULT_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "temperature_weights.json")
DEFAULT_CSV_PATH = "data/audit_dataset.csv"


def prob_to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Converts probabilities to log-odds (logits) safely."""
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


def logit_to_prob(z: np.ndarray) -> np.ndarray:
    """Converts logits back to probabilities via sigmoid."""
    return 1.0 / (1.0 + np.exp(-z))


def scale_probabilities(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Scales probabilities using temperature T > 0."""
    t = max(0.1, float(temperature))
    z = prob_to_logit(probs)
    return logit_to_prob(z / t)


def compute_nll(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-6) -> float:
    """Computes binary cross-entropy (Negative Log-Likelihood)."""
    p_clipped = np.clip(probs, eps, 1.0 - eps)
    nll = -np.mean(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped))
    return float(nll)


def compute_brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Computes mean squared error between probabilities and binary labels."""
    return float(np.mean((probs - y_true) ** 2))


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) using equal-width probability bins.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(probs)
    
    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Select samples within this bin
        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_conf = np.mean(probs[in_bin])
            bin_acc = np.mean(y_true[in_bin])
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)
            
    return float(ece)


def fit_temperature(
    probs: np.ndarray,
    y_true: np.ndarray,
    bounds: Tuple[float, float] = (0.5, 3.5)
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Finds optimal temperature T minimizing NLL on the given samples.
    Returns (optimal_T, uncalibrated_metrics, calibrated_metrics).
    """
    z = prob_to_logit(probs)

    def objective(t: float) -> float:
        scaled_p = logit_to_prob(z / t)
        return compute_nll(scaled_p, y_true)

    res = minimize_scalar(objective, bounds=bounds, method="bounded")
    best_t = float(round(res.x, 3))

    # Evaluate uncalibrated vs calibrated
    calibrated_probs = scale_probabilities(probs, best_t)

    uncalib_metrics = {
        "nll": round(compute_nll(probs, y_true), 4),
        "brier": round(compute_brier_score(probs, y_true), 4),
        "ece": round(compute_ece(probs, y_true), 4)
    }

    calib_metrics = {
        "nll": round(compute_nll(calibrated_probs, y_true), 4),
        "brier": round(compute_brier_score(calibrated_probs, y_true), 4),
        "ece": round(compute_ece(calibrated_probs, y_true), 4)
    }

    return best_t, uncalib_metrics, calib_metrics


def train_demographic_temperature_scaler(
    df: pd.DataFrame,
    min_samples_per_cohort: int = 40
) -> Dict[str, Any]:
    """
    Fits global and attribute-specific temperature parameters across all available cohorts.
    """
    y_true = (df["true_label"] == "FAKE").astype(float).values
    probs = df["predicted_prob"].values

    print(f"Fitting global temperature on {len(df)} samples...")
    global_t, global_uncal, global_cal = fit_temperature(probs, y_true)
    print(f"Global Temperature: T = {global_t:.3f} (ECE: {global_uncal['ece']:.3f} -> {global_cal['ece']:.3f})")

    cohort_temperatures = {}
    cohort_evaluations = {}

    available_attrs = [c for c in ATTRIBUTES if c in df.columns]

    for attr in available_attrs:
        mask = (df[attr] == 1).values
        count = int(np.sum(mask))

        if count < min_samples_per_cohort:
            cohort_temperatures[attr] = global_t
            continue

        c_probs = probs[mask]
        c_y_true = y_true[mask]

        t_opt, uncal, cal = fit_temperature(c_probs, c_y_true)
        cohort_temperatures[attr] = t_opt
        cohort_evaluations[attr] = {
            "attribute": attr,
            "sample_count": count,
            "temperature": t_opt,
            "uncalibrated": uncal,
            "calibrated": cal,
            "ece_reduction": round(uncal["ece"] - cal["ece"], 4)
        }

    results = {
        "metadata": {
            "version": "1.0.0",
            "global_temperature": global_t,
            "global_evaluation": {
                "uncalibrated": global_uncal,
                "calibrated": global_cal,
                "ece_reduction": round(global_uncal["ece"] - global_cal["ece"], 4)
            },
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_cohorts_fitted": len(cohort_temperatures)
        },
        "temperatures": cohort_temperatures,
        "cohort_evaluations": cohort_evaluations
    }

    return results


def save_temperature_weights(results: Dict[str, Any], filepath: str = DEFAULT_WEIGHTS_PATH):
    """Saves temperature scaling weights to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Feature 1.3] Demographic temperature weights saved to: {filepath}")


def load_temperature_weights(filepath: str = DEFAULT_WEIGHTS_PATH) -> Dict[str, Any]:
    """Loads temperature weights from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Temperature weights file not found at {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def calibrate_probability(
    raw_prob: float,
    detected_attributes: Optional[List[str]] = None,
    weights_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Applies demographic temperature scaling to a single inference probability score.
    If multiple attributes are present, computes an effective temperature.
    """
    if weights_data is None:
        try:
            weights_data = load_temperature_weights()
        except Exception:
            return {
                "raw_probability": raw_prob,
                "calibrated_probability": raw_prob,
                "effective_temperature": 1.0,
                "applied_attributes": [],
                "delta": 0.0
            }

    global_t = weights_data.get("metadata", {}).get("global_temperature", 1.0)
    temp_map = weights_data.get("temperatures", {})

    active_temps = []
    active_attrs = []

    if detected_attributes:
        for attr in detected_attributes:
            if attr in temp_map:
                active_temps.append(temp_map[attr])
                active_attrs.append(attr)

    if active_temps:
        # Use maximum temperature to provide maximum softening against overconfident false alarms
        effective_t = float(max(active_temps))
    else:
        effective_t = float(global_t)

    calibrated_prob = float(scale_probabilities(np.array([raw_prob]), effective_t)[0])
    calibrated_prob = round(float(np.clip(calibrated_prob, 0.0001, 0.9999)), 4)
    delta = round(calibrated_prob - raw_prob, 4)

    return {
        "raw_probability": round(raw_prob, 4),
        "calibrated_probability": calibrated_prob,
        "effective_temperature": round(effective_t, 3),
        "applied_attributes": active_attrs,
        "delta": delta
    }


def main():
    parser = argparse.ArgumentParser(description="FairFake Demographic Temperature Scaler (Feature 1.3)")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to audit dataset CSV")
    parser.add_argument("--out", type=str, default=DEFAULT_WEIGHTS_PATH, help="Output temperature JSON path")
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        candidates = ["backend/data/audit_dataset.csv", "../data/audit_dataset.csv"]
        csv_path = next((p for p in candidates if os.path.exists(p)), None)

    if not csv_path or not os.path.exists(csv_path):
        print(f"Error: Audit dataset CSV not found at {args.csv}. Please run Feature 1.1 first.")
        sys.exit(1)

    print(f"Loading evaluation dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    results = train_demographic_temperature_scaler(df)
    save_temperature_weights(results, args.out)

    print("\n" + "=" * 80)
    print("DEMOGRAPHIC TEMPERATURE SCALING EVALUATION SUMMARY:")
    print("=" * 80)
    print(f"{'Attribute':<20} | {'Temp (T)':<9} | {'Uncal ECE':<11} | {'Calib ECE':<11} | {'ECE Reduction':<14}")
    print("-" * 80)

    # Sort cohorts by temperature
    sorted_cohorts = sorted(
        results["cohort_evaluations"].values(),
        key=lambda x: x["temperature"],
        reverse=True
    )

    for row in sorted_cohorts[:10]:
        uncal_ece = row["uncalibrated"]["ece"]
        cal_ece = row["calibrated"]["ece"]
        ece_red = row["ece_reduction"]
        print(f"{row['attribute']:<20} | {row['temperature']:<9.3f} | {uncal_ece:<11.2%} | {cal_ece:<11.2%} | {ece_red:<+14.2%}")
    print("=" * 80)

    # Sample demo inference
    demo = calibrate_probability(0.85, ["Dark_Skin", "Heavy_Makeup"], results)
    print("\n[Demo Calibration Check]")
    print(f"Raw Input Probability:       {demo['raw_probability']:.2%}")
    print(f"Active Attributes:           {demo['applied_attributes']}")
    print(f"Effective Temperature:       T = {demo['effective_temperature']:.3f}")
    print(f"Fair-Calibrated Probability: {demo['calibrated_probability']:.2%} (Delta: {demo['delta']:+.2%})")


if __name__ == "__main__":
    main()
