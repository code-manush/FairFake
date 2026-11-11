"""
FairFake Fairness Report & Exporters (Task 5.1)
"""

import json
import datetime
from typing import Dict, Any, Optional
from .auditor import AuditResult


class FairnessReport:
    """
    Structured outcome of a fairness audit.
    Provides JSON, HTML, and terminal export formats.
    """

    def __init__(self, result: AuditResult):
        self.result = result
        self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "generated_at": self.timestamp,
                "total_samples": self.result.total_samples,
                "dataset_source": self.result.dataset_source,
                "global_accuracy": self.result.global_accuracy
            },
            "summary": {
                "max_abs_crp": self.result.max_abs_crp,
                "max_pdrp": self.result.max_pdrp,
                "max_ddrp": self.result.max_ddrp,
                "high_bias_cohorts": self.result.high_bias_cohorts,
                "total_cohorts_evaluated": len(self.result.cohort_metrics)
            },
            "cohort_metrics": {
                attr: {
                    "sample_count": rec.sample_count,
                    "accuracy": rec.accuracy,
                    "error_rate": rec.error_rate,
                    "pdrp": rec.pdrp,
                    "ddrp": rec.ddrp,
                    "rp": rec.rp,
                    "crp": rec.crp
                }
                for attr, rec in self.result.cohort_metrics.items()
            }
        }

    def export_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[+] Fairness report saved to JSON: {path}")

    def export_html(self, path: str):
        data = self.to_dict()
        rows = []
        for attr, m in sorted(data["cohort_metrics"].items(), key=lambda x: abs(x[1]["crp"]), reverse=True):
            is_high = abs(m["crp"]) > 0.15 or m["pdrp"] > 1.4 or m["ddrp"] > 1.4
            badge_class = "badge-high" if is_high else "badge-normal"
            rows.append(f"""
            <tr class="{'row-flagged' if is_high else ''}">
              <td class="attr-name"><strong>{attr}</strong></td>
              <td>{m['sample_count']}</td>
              <td>{(m['accuracy'] * 100):.1f}%</td>
              <td class="{badge_class}">{m['crp']:+.3f}</td>
              <td>{m['pdrp']:.2f}x</td>
              <td>{m['ddrp']:.2f}x</td>
            </tr>
            """)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>FairFake Demographic Fairness Audit Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; padding: 30px; margin: 0; }}
    .container {{ max-width: 1000px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 28px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border: 1px solid #334155; }}
    h1 {{ margin-top: 0; color: #38bdf8; font-size: 24px; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 24px 0; }}
    .stat-card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 16px; text-align: center; }}
    .stat-card .val {{ font-size: 28px; font-weight: bold; color: #f8fafc; margin-top: 4px; }}
    .stat-card .label {{ font-size: 11px; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }}
    th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid #334155; }}
    th {{ background: #0f172a; color: #94a3b8; font-size: 12px; text-transform: uppercase; }}
    .badge-high {{ color: #f87171; font-weight: bold; }}
    .badge-normal {{ color: #34d399; font-weight: bold; }}
    .row-flagged {{ background: rgba(239, 68, 68, 0.06); }}
  </style>
</head>
<body>
  <div class="container">
    <h1>FairFake Model Fairness & Bias Audit</h1>
    <p style="color: #94a3b8; font-size: 13px;">Generated at {data['metadata']['generated_at']} | Source: {data['metadata']['dataset_source']}</p>
    
    <div class="stats-grid">
      <div class="stat-card">
        <div class="label">Global Accuracy</div>
        <div class="val">{(data['metadata']['global_accuracy'] * 100):.1f}%</div>
      </div>
      <div class="stat-card">
        <div class="label">Max |CRP| Disparity</div>
        <div class="val" style="color: {'#f87171' if data['summary']['max_abs_crp'] > 0.15 else '#34d399'}">{data['summary']['max_abs_crp']:.3f}</div>
      </div>
      <div class="stat-card">
        <div class="label">Max PDRP (False Alarm Ratio)</div>
        <div class="val">{data['summary']['max_pdrp']:.2f}x</div>
      </div>
      <div class="stat-card">
        <div class="label">High Bias Cohorts</div>
        <div class="val">{len(data['summary']['high_bias_cohorts'])}</div>
      </div>
    </div>

    <table>
      <thead>
        <tr>
          <th>Cohort / Appearance Attribute</th>
          <th>Samples</th>
          <th>Accuracy</th>
          <th>CRP Disparity</th>
          <th>PDRP</th>
          <th>DDRP</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[+] Fairness report saved to HTML: {path}")

    def print_summary(self):
        d = self.to_dict()
        print("\n=== FairFake Demographic Audit Summary ===")
        print(f"Global Accuracy: {d['metadata']['global_accuracy']:.2%}")
        print(f"Total Samples Evaluated: {d['metadata']['total_samples']}")
        print(f"Max |CRP| Disparity: {d['summary']['max_abs_crp']:.3f}")
        print(f"Max PDRP: {d['summary']['max_pdrp']:.2f}x | Max DDRP: {d['summary']['max_ddrp']:.2f}x")
        print(f"Flagged High Bias Cohorts: {', '.join(d['summary']['high_bias_cohorts']) or 'None'}")
