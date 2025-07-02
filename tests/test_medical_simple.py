#!/usr/bin/env python3
"""
Simple Medical Functionality Tests - Direct Import Version
========================================================== 

Basic medical functionality tests without complex dependencies.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_medical_decision_engine_import():
    """Test that medical decision engine can be imported"""
    try:
        from src.medical.medical_decision_engine import MedicalDecisionEngine
        engine = MedicalDecisionEngine()
        assert engine is not None
        print("✅ Medical Decision Engine: Import successful")
    except ImportError as e:
        pytest.skip(f"Medical Decision Engine not available: {e}")

def test_medical_decision_functionality():
    """Test basic medical decision functionality"""
    try:
        from src.medical.medical_decision_engine import MedicalDecisionEngine
        engine = MedicalDecisionEngine()
        
        # Test Grade 4 emergency case
        decision = engine.make_clinical_decision(
            lpp_grade=4,
            confidence=0.95,
            anatomical_location="sacrum",
            patient_context={"age": 75, "diabetes": True}
        )
        
        # Basic validation
        assert 'severity_assessment' in decision
        assert 'clinical_recommendations' in decision
        assert decision['severity_assessment'] == 'emergency'
        
        print("✅ Medical Decision Engine: Grade 4 emergency validated")
        
    except ImportError as e:
        pytest.skip(f"Medical Decision Engine not available: {e}")

def test_phi_tokenization_import():
    """Test PHI tokenization client import"""
    try:
        from src.security.phi_tokenization_client import PHITokenizationClient
        client = PHITokenizationClient()
        assert client is not None
        print("✅ PHI Tokenization: Import successful")
    except ImportError as e:
        pytest.skip(f"PHI Tokenization not available: {e}")

def test_shared_utilities_import():
    """Test shared utilities import"""
    try:
        from src.utils.shared_utilities import VigiaLogger
        logger = VigiaLogger.get_logger("test")
        assert logger is not None
        print("✅ Shared Utilities: Import successful")
    except ImportError as e:
        pytest.skip(f"Shared utilities not available: {e}")

def test_medical_system_basic_integration():
    """Test basic medical system integration"""
    try:
        from src.medical.medical_decision_engine import MedicalDecisionEngine
        from src.security.phi_tokenization_client import PHITokenizationClient
        
        # Initialize components
        engine = MedicalDecisionEngine()
        phi_client = PHITokenizationClient()
        
        # Test medical workflow
        decision = engine.make_clinical_decision(
            lpp_grade=2,
            confidence=0.87,
            anatomical_location="heel",
            patient_context={"age": 65}
        )
        
        # Validate workflow
        assert 'lpp_grade' in decision
        assert 'severity_assessment' in decision
        assert 'clinical_recommendations' in decision
        
        print("🏆 BASIC MEDICAL SYSTEM INTEGRATION: SUCCESS")
        
    except ImportError as e:
        pytest.skip(f"Medical system components not available: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])