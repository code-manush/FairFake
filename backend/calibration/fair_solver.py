"""
FairFake Multi-Attribute Fairness Threshold Solver (Feature 1.2)

Solves for optimal demographic-specific decision thresholds tau_a* per attribute cohort
to enforce fairness criteria (Equalized Odds, Equal Opportunity, FPR Parity, Disparate Mistreatment)
using the cohort metric curves computed in Feature 1.1.
"""

import os
import sys
import json
import argparse
import datetime
from typing import Dict, List, Any, Optional

DEFAULT_CURVES_PATH = os.path.join(os.path.dirname(__file__), "cohort_metric_curves.json")
DEFAULT_THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "calibrated_thresholds.json")


def solve_cohort_threshold(
    points: List[Dict[str, Any]],
    target_tpr: float,
    target_fpr: float,
    criterion: str = "equalized_odds",
    alpha: float = 0.5,
    min_thresh: float = 0.38,
    max_thresh: float = 0.65,
    max_accuracy_drop: float = 0.04,
    lambda_reg: float = 0.35
) -> Dict[str, Any]:
    """
    Finds the optimal threshold tau* in points matching the specified fairness criterion,
    with a L2 regularization penalty toward the nominal baseline threshold (tau = 0.50).
    """
    # Filter points within operational bounds
    valid_points = [p for p in points if min_thresh <= p["threshold"] <= max_thresh]
    if not valid_points:
        valid_points = points
        
    # Find baseline point at tau = 0.50
    pt_05 = next((p for p in points if abs(p["threshold"] - 0.50) < 1e-4), points[len(points)//2])
    baseline_acc = pt_05.get("accuracy", 0.70)
    
    best_pt = pt_05
    best_loss = float("inf")
    
    for pt in valid_points:
        # Prevent accuracy degradation beyond acceptable tolerance
        if pt["accuracy"] < (baseline_acc - max_accuracy_drop):
            continue
            
        tpr = pt["tpr"]
        fpr = pt["fpr"]
        pdrp = pt.get("pdrp", 0.0)
        ddrp = pt.get("ddrp", 0.0)
        
        if criterion == "equal_opportunity":
            # Match True Positive Rate (fake detection sensitivity)
            fairness_loss = abs(tpr - target_tpr)
        elif criterion == "fpr_parity":
            # Match False Positive Rate (false alarms on pristine faces)
            fairness_loss = abs(fpr - target_fpr)
        elif criterion == "disparate_mistreatment":
            # Penalize elevated PDRP and DDRP directly
            fairness_loss = (max(0.0, pdrp) ** 2) + (max(0.0, ddrp) ** 2)
        else: # "equalized_odds" (Default)
            # Joint optimization over TPR and FPR parity
            tpr_gap = abs(tpr - target_tpr)
            fpr_gap = abs(fpr - target_fpr)
            fairness_loss = alpha * tpr_gap + (1.0 - alpha) * fpr_gap
            
        # Add regularization penalty toward nominal 0.50 threshold to avoid excessive drift
        reg_penalty = lambda_reg * ((pt["threshold"] - 0.50) ** 2)
        total_loss = fairness_loss + reg_penalty
            
        if total_loss < best_loss:
            best_loss = total_loss
            best_pt = pt
            
    # Calculate improvements
    delta_pdrp = round(best_pt.get("pdrp", 0.0) - pt_05.get("pdrp", 0.0), 4)
    delta_ddrp = round(best_pt.get("ddrp", 0.0) - pt_05.get("ddrp", 0.0), 4)
    delta_acc = round(best_pt.get("accuracy", 0.0) - pt_05.get("accuracy", 0.0), 4)
    
    return {
        "uncalibrated_threshold": 0.50,
        "calibrated_threshold": round(best_pt["threshold"], 2),
        "uncalibrated_metrics": {
            "tpr": pt_05["tpr"],
            "fpr": pt_05["fpr"],
            "fnr": pt_05["fnr"],
            "accuracy": pt_05["accuracy"],
            "pdrp": pt_05.get("pdrp", 0.0),
            "ddrp": pt_05.get("ddrp", 0.0)
        },
        "calibrated_metrics": {
            "tpr": best_pt["tpr"],
            "fpr": best_pt["fpr"],
            "fnr": best_pt["fnr"],
            "accuracy": best_pt["accuracy"],
            "pdrp": best_pt.get("pdrp", 0.0),
            "ddrp": best_pt.get("ddrp", 0.0)
        },
        "improvements": {
            "delta_pdrp": delta_pdrp,
            "delta_ddrp": delta_ddrp,
            "delta_accuracy": delta_acc
        },
        "loss": round(best_loss, 4)
    }


def solve_all_thresholds(
    curves_data: Dict[str, Any],
    criterion: str = "equalized_odds",
    alpha: float = 0.5
) -> Dict[str, Any]:
    """
    Computes calibrated thresholds for all evaluated appearance attributes.
    """
    meta_baseline = curves_data.get("metadata", {}).get("default_baseline_05", {})
    target_tpr = meta_baseline.get("tpr", 0.706)
    target_fpr = meta_baseline.get("fpr", 0.320)
    
    cohort_curves = curves_data.get("cohort_curves", {})
    
    threshold_map = {}
    detailed_calibrations = {}
    
    for attr, info in cohort_curves.items():
        points = info.get("points", [])
        if not points:
            continue
            
        calib = solve_cohort_threshold(
            points=points,
            target_tpr=target_tpr,
            target_fpr=target_fpr,
            criterion=criterion,
            alpha=alpha
        )
        
        calibrated_tau = calib["calibrated_threshold"]
        threshold_map[attr] = calibrated_tau
        detailed_calibrations[attr] = {
            "attribute": attr,
            "sample_count_with": info.get("sample_count_with", 0),
            **calib
        }
        
    result = {
        "metadata": {
            "version": "1.0.0",
            "criterion": criterion,
            "alpha": alpha,
            "global_default_threshold": 0.50,
            "target_baseline": {
                "tpr": target_tpr,
                "fpr": target_fpr
            },
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_attributes_calibrated": len(threshold_map)
        },
        "thresholds": threshold_map,
        "calibrations": detailed_calibrations
    }
    
    return result


def save_calibrated_thresholds(results: Dict[str, Any], filepath: str = DEFAULT_THRESHOLDS_PATH):
    """Saves threshold results to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Feature 1.2] Calibrated fairness thresholds saved to: {filepath}")


def load_calibrated_thresholds(filepath: str = DEFAULT_THRESHOLDS_PATH) -> Dict[str, Any]:
    """Loads calibrated thresholds from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Calibrated thresholds file not found at {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_cohort_threshold(
    attr_name: str,
    threshold_data: Optional[Dict[str, Any]] = None,
    default_tau: float = 0.50
) -> float:
    """Helper function to look up a calibrated threshold for an individual attribute."""
    if threshold_data is None:
        try:
            threshold_data = load_calibrated_thresholds()
        except Exception:
            return default_tau
            
    thresholds = threshold_data.get("thresholds", {})
    return float(thresholds.get(attr_name, default_tau))


def main():
    parser = argparse.ArgumentParser(description="FairFake Multi-Attribute Threshold Solver (Feature 1.2)")
    parser.add_argument("--curves", type=str, default=DEFAULT_CURVES_PATH, help="Input cohort curves JSON")
    parser.add_argument("--out", type=str, default=DEFAULT_THRESHOLDS_PATH, help="Output thresholds JSON")
    parser.add_argument("--criterion", type=str, default="equalized_odds",
                        choices=["equalized_odds", "equal_opportunity", "fpr_parity", "disparate_mistreatment"],
                        help="Fairness optimization criterion")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight between TPR and FPR in equalized odds")
    args = parser.parse_args()

    if not os.path.exists(args.curves):
        print(f"Error: Curves file not found at {args.curves}. Please run Feature 1.1 (metric_sweep.py) first.")
        sys.exit(1)

    print(f"Loading curves from {args.curves}...")
    with open(args.curves, "r", encoding="utf-8") as f:
        curves_data = json.load(f)

    print(f"Optimizing decision thresholds using criterion '{args.criterion}' (alpha={args.alpha})...")
    results = solve_all_thresholds(curves_data, criterion=args.criterion, alpha=args.alpha)
    save_calibrated_thresholds(results, args.out)

    print("\n" + "=" * 80)
    print(f"CALIBRATED DEMOGRAPHIC THRESHOLDS SUMMARY ({args.criterion.upper()}):")
    print("=" * 80)
    print(f"{'Attribute':<20} | {'Base Tau':<9} | {'Fair Tau':<9} | {'Base FPR':<9} | {'Fair FPR':<9} | {'Delta PDRP':<10}")
    print("-" * 80)
    
    # Sort attributes by magnitude of threshold adjustment
    sorted_calibs = sorted(
        results["calibrations"].values(),
        key=lambda x: abs(x["calibrated_threshold"] - 0.50),
        reverse=True
    )
    
    for row in sorted_calibs[:10]:
        base_fpr = row["uncalibrated_metrics"]["fpr"]
        fair_fpr = row["calibrated_metrics"]["fpr"]
        delta_pdrp = row["improvements"]["delta_pdrp"]
        print(f"{row['attribute']:<20} | {row['uncalibrated_threshold']:<9.2f} | {row['calibrated_threshold']:<9.2f} | {base_fpr:<9.2%} | {fair_fpr:<9.2%} | {delta_pdrp:<+10.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
