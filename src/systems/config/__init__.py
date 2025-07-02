"""
Medical Configuration Module
============================

This module contains clinical guidelines and medical decision parameters
for VIGIA Medical AI system.

Medical Standards:
- NPUAP/EPUAP/PPPIA 2019 International Clinical Practice Guidelines
- Evidence-based medical decision protocols
- MINSAL (Chilean Ministry of Health) compliance parameters
"""

import os
import json
from typing import Dict, Any
from pathlib import Path

def load_clinical_guidelines() -> Dict[str, Any]:
    """Load NPUAP/EPUAP clinical guidelines configuration."""
    config_path = Path(__file__).parent / "clinical_guidelines.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_minsal_info() -> Dict[str, Any]:
    """Load MINSAL (Chilean Ministry of Health) medical information."""
    config_path = Path(__file__).parent / "minsal_extracted_info.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Medical configuration constants
MEDICAL_CONFIG = {
    "compliance_standards": ["NPUAP", "EPUAP", "PPPIA_2019", "MINSAL"],
    "evidence_levels": ["A", "B", "C"],
    "critical_grades": [3, 4],
    "emergency_response_time_minutes": 15
}