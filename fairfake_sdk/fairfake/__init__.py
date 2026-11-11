"""
FairFake: Enterprise Deepfake Fairness & Demographic Auditing SDK (Phase 5: Task 5.1)

Core API providing:
  - `Auditor`: Audits any PyTorch, ONNX, or CSV predictions across 31 demographic attributes.
  - `FairnessReport`: Comprehensive report with CRP, PDRP, DDRP, and ECE metrics.
  - `Scorecard`: Issues compliance certifications (A+ to F) and SVG badges.
"""

from .auditor import Auditor, AuditResult
from .report import FairnessReport
from .certification import generate_scorecard, generate_compliance_badge

__version__ = "1.0.0"
__all__ = [
    "Auditor",
    "AuditResult",
    "FairnessReport",
    "generate_scorecard",
    "generate_compliance_badge"
]
