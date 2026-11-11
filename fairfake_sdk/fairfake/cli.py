"""
FairFake Command Line Interface (CLI Tool) (Task 5.1)
"""

import sys
import os
import argparse
import json
from .auditor import Auditor
from .certification import generate_scorecard, generate_compliance_badge


def main():
    parser = argparse.ArgumentParser(
        prog="fairfake",
        description="FairFake CLI: Demographic Fairness Auditing & Compliance Certifier"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Command: audit
    p_audit = subparsers.add_parser("audit", help="Run fairness audit on dataset predictions")
    p_audit.add_argument("--dataset", required=True, help="Path to predictions CSV")
    p_audit.add_argument("--threshold", type=float, default=0.5, help="Classification decision threshold")
    p_audit.add_argument("--output", default="fairness_audit.json", help="Output file path (.json or .html)")

    # Command: certify
    p_cert = subparsers.add_parser("certify", help="Issue formal fairness certification and compliance badge")
    p_cert.add_argument("--dataset", required=True, help="Path to predictions CSV")
    p_cert.add_argument("--model-name", default="Production-Model", help="Model identification name")
    p_cert.add_argument("--badge-out", default="fairness_badge.svg", help="Output SVG badge path")
    p_cert.add_argument("--scorecard-out", default="compliance_scorecard.json", help="Output scorecard JSON path")

    # Command: check-gate (CI/CD Quality Gate)
    p_gate = subparsers.add_parser("check-gate", help="Evaluate CI/CD pull-request quality gate")
    p_gate.add_argument("--dataset", required=True, help="Path to candidate model predictions CSV")
    p_gate.add_argument("--calibration-file", default=None, help="Path to calibrated thresholds JSON (Phase 1 post-hoc calibration)")
    p_gate.add_argument("--max-crp", type=float, default=0.20, help="Maximum allowed |CRP| disparity")
    p_gate.add_argument("--max-pdrp", type=float, default=1.60, help="Maximum allowed PDRP false alarm ratio")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    def _resolve_calibration_map(calib_path):
        if not calib_path:
            return None
        if not os.path.exists(calib_path):
            print(f"[!] Warning: Calibration file not found at {calib_path}. Auditing without calibration.")
            return None
        with open(calib_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("thresholds", data)

    if args.command == "audit":
        auditor = Auditor(dataset=args.dataset)
        report = auditor.run_audit(threshold=args.threshold)
        report.print_summary()
        if args.output.endswith(".html"):
            report.export_html(args.output)
        else:
            report.export_json(args.output)

    elif args.command == "certify":
        auditor = Auditor(dataset=args.dataset)
        report = auditor.run_audit()
        card = generate_scorecard(report.to_dict(), model_name=args.model_name)
        
        with open(args.scorecard_out, "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2)
        print(f"[+] Compliance Scorecard saved to: {args.scorecard_out}")
        print(f"    Certified Grade: {card['certification']['fairness_grade']} ({card['certification']['status']})")

        badge_svg = generate_compliance_badge(card["certification"]["fairness_grade"])
        with open(args.badge_out, "w", encoding="utf-8") as f:
            f.write(badge_svg)
        print(f"[+] Compliance Badge SVG saved to: {args.badge_out}")

    elif args.command == "check-gate":
        auditor = Auditor(dataset=args.dataset)
        calib_map = _resolve_calibration_map(args.calibration_file)
        if calib_map:
            print(f"[*] Applying dynamic post-hoc calibration matrix from: {args.calibration_file}")
        report = auditor.run_audit(calibration_map=calib_map)
        res = report.result
        
        print(f"\n[*] Evaluating CI/CD Fairness Gate...")
        print(f"    Max |CRP|: {res.max_abs_crp:.3f} (Allowed: <= {args.max_crp:.3f})")
        print(f"    Max PDRP:  {res.max_pdrp:.2f}x (Allowed: <= {args.max_pdrp:.2f}x)")

        violations = []
        if res.max_abs_crp > args.max_crp:
            violations.append(f"Max |CRP| {res.max_abs_crp:.3f} exceeds tolerance {args.max_crp:.3f}")
        if res.max_pdrp > args.max_pdrp:
            violations.append(f"Max PDRP {res.max_pdrp:.2f}x exceeds tolerance {args.max_pdrp:.2f}x")

        if violations:
            print("\n❌ CI/CD QUALITY GATE FAILED:")
            for v in violations:
                print(f"   - {v}")
            sys.exit(1)
        else:
            print("\n✅ CI/CD QUALITY GATE PASSED: Model satisfies demographic equity criteria.")
            sys.exit(0)


if __name__ == "__main__":
    main()
