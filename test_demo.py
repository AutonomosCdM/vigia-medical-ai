#!/usr/bin/env python3
"""
Quick test script for VIGIA demo functionality
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, "/Users/autonomos_dev/Projects/vigia")

def test_imports():
    """Test all imports and show what's available"""
    print("🧪 Testing VIGIA Demo Components...")
    
    # Test demo components
    try:
        from medical.medical_decision_engine import MedicalDecisionEngine
        print("✅ Demo medical engine imported")
        
        engine = MedicalDecisionEngine()
        print("✅ Demo medical engine initialized")
        
        # Test a simple decision
        decision = engine.make_clinical_decision(
            lpp_grade=2,
            confidence=0.85,
            anatomical_location="sacrum",
            patient_context={"age": 70, "diabetes": True}
        )
        print(f"✅ Demo medical decision generated: Grade {decision['lpp_grade']}")
        
    except Exception as e:
        print(f"❌ Demo medical engine error: {e}")
    
    # Test real VIGIA components
    real_components = 0
    
    try:
        from vigia_detect.core.async_pipeline import async_pipeline
        print("✅ Real async pipeline available")
        real_components += 1
    except Exception as e:
        print(f"❌ Async pipeline: {e}")
    
    try:
        from vigia_detect.core.phi_tokenization_client import PHITokenizationClient
        print("✅ Real PHI tokenization available")
        real_components += 1
    except Exception as e:
        print(f"❌ PHI tokenization: {e}")
    
    try:
        from vigia_detect.messaging.slack_notifier_refactored import SlackNotifier
        print("✅ Real Slack notifier available")
        real_components += 1
    except Exception as e:
        print(f"❌ Slack notifier: {e}")
    
    print(f"\n🎯 Component Summary:")
    print(f"Real VIGIA components: {real_components}/3")
    
    if real_components >= 2:
        print("🔬 Status: REAL SYSTEM READY")
    elif real_components > 0:
        print("🔄 Status: HYBRID SYSTEM READY")
    else:
        print("🎭 Status: DEMO MODE ONLY")
    
    return real_components

def test_gradio_import():
    """Test Gradio availability"""
    try:
        import gradio as gr
        print("✅ Gradio available")
        return True
    except ImportError as e:
        print(f"❌ Gradio not available: {e}")
        return False

if __name__ == "__main__":
    print("🩺 VIGIA Demo Test Suite")
    print("=" * 50)
    
    components = test_imports()
    gradio_ok = test_gradio_import()
    
    print("\n" + "=" * 50)
    
    if gradio_ok:
        print("🚀 Demo ready to launch!")
        print("Run: python demo/launch_medical_demo.py")
    else:
        print("⚠️  Install Gradio: pip install gradio")
    
    print(f"System will run in {'REAL' if components >= 2 else 'HYBRID' if components > 0 else 'DEMO'} mode")