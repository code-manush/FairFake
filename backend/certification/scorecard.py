"""
Backend Certification Scorecard Module (Phase 5: Task 5.2)
"""

import os
import sys
from pathlib import Path

# Link to SDK implementation
sdk_path = Path(__file__).parent.parent.parent / "fairfake_sdk"
if str(sdk_path) not in sys.path:
    sys.path.insert(0, str(sdk_path))

from fairfake.certification import (
    determine_grade,
    generate_scorecard,
    generate_compliance_badge,
    GRADE_THRESHOLDS
)

__all__ = [
    "determine_grade",
    "generate_scorecard",
    "generate_compliance_badge",
    "GRADE_THRESHOLDS"
]
