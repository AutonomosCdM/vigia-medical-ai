#!/usr/bin/env python3
"""
Comprehensive validation script for Vigia Medical System
Validates real medical functionality after installation
"""

import asyncio
import sys
import json
import time
import redis
import requests
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vigia_detect.systems.medical_decision_engine import MedicalDecisionEngine
from vigia_detect.cv_pipeline.medical_detector_factory import create_medical_detector
from vigia_detect.core.phi_tokenization_client import PHITokenizationClient

class VigiaValidator:
    def __init__(self):
        self.results = {}
        self.total_tests = 8
        self.passed_tests = 0
        
    def log_test(self, test_name, status, details=""):
        """Log test result"""
        self.results[test_name] = {
            "status": status,
            "details": details,
            "timestamp": time.time()
        }
        
        status_symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_symbol} {test_name}: {status}")
        if details:
            print(f"   {details}")
        
        if status == "PASS":
            self.passed_tests += 1
    
    def test_medical_decision_engine(self):
        """Test real medical decision engine with NPUAP guidelines"""
        try:
            engine = MedicalDecisionEngine()
            
            # Test Grade 4 emergency case
            decision = engine.make_clinical_decision(
                lpp_grade=4,
                confidence=0.95,
                anatomical_location="sacrum",
                patient_context={"age": 75, "diabetes": True}
            )
            
            # Validate real medical response structure
            has_severity = 'severity_assessment' in decision
            has_recommendations = ('clinical_recommendations' in decision and 
                                 len(decision['clinical_recommendations']) > 0)
            has_evidence = 'evidence_documentation' in decision
            
            if has_severity and has_recommendations and has_evidence:
                severity = decision['severity_assessment']
                timeline = decision.get('intervention_timeline', 'N/A')
                
                self.log_test(
                    "Medical Decision Engine",
                    "PASS",
                    f"Real NPUAP Grade 4 → {severity.title()}: {timeline}"
                )
                return True
            else:
                missing = []
                if not has_severity: missing.append('severity_assessment')
                if not has_recommendations: missing.append('clinical_recommendations')  
                if not has_evidence: missing.append('evidence_documentation')
                
                self.log_test(
                    "Medical Decision Engine", 
                    "FAIL", 
                    f"Missing: {missing}. Got: {list(decision.keys())}"
                )
                return False
                
        except Exception as e:
            self.log_test("Medical Decision Engine", "FAIL", f"Error: {str(e)}")
            return False
    
    def test_cv_pipeline_framework(self):
        """Test CV pipeline framework (without actual model)"""
        try:
            detector = create_medical_detector()
            
            # Validate framework is ready
            if hasattr(detector, 'detect_medical_condition'):
                self.log_test(
                    "CV Pipeline Framework",
                    "PASS",
                    f"AdaptiveMedicalDetector ready: {type(detector).__name__}"
                )
                return True
            else:
                self.log_test("CV Pipeline Framework", "FAIL", "Missing detect_medical_condition method")
                return False
                
        except Exception as e:
            self.log_test("CV Pipeline Framework", "FAIL", f"Error: {str(e)}")
            return False
    
    def test_phi_tokenization_client(self):
        """Test PHI tokenization client configuration"""
        try:
            client = PHITokenizationClient()
            
            # Validate client configuration
            if (hasattr(client, 'config') and 
                client.config.tokenization_service_url and
                hasattr(client, 'cache')):
                
                self.log_test(
                    "PHI Tokenization Client",
                    "PASS",
                    f"HIPAA client ready: {client.config.tokenization_service_url}"
                )
                return True
            else:
                self.log_test("PHI Tokenization Client", "FAIL", "Invalid client configuration")
                return False
                
        except Exception as e:
            self.log_test("PHI Tokenization Client", "FAIL", f"Error: {str(e)}")
            return False
    
    def test_redis_connection(self):
        """Test Redis database connection"""
        try:
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            
            # Test medical data storage
            r.set('vigia:test:medical', 'Grade_4_LPP_Emergency')
            test_data = r.get('vigia:test:medical')
            
            if test_data and test_data.decode() == 'Grade_4_LPP_Emergency':
                self.log_test(
                    "Redis Medical Database",
                    "PASS",
                    "Connection and medical data storage working"
                )
                return True
            else:
                self.log_test("Redis Medical Database", "FAIL", "Data storage test failed")
                return False
                
        except Exception as e:
            self.log_test("Redis Medical Database", "FAIL", f"Error: {str(e)}")
            return False
    
    def test_ollama_service(self):
        """Test Ollama AI service"""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            
            if response.status_code == 200:
                models = response.json()
                model_names = [model.get('name', '') for model in models.get('models', [])]
                
                if any('medgemma' in name.lower() for name in model_names):
                    self.log_test(
                        "Ollama + MedGemma AI",
                        "PASS",
                        "MedGemma medical AI model loaded"
                    )
                else:
                    self.log_test(
                        "Ollama + MedGemma AI",
                        "WARN",
                        "Ollama running but MedGemma not loaded"
                    )
                return True
            else:
                self.log_test("Ollama + MedGemma AI", "FAIL", f"Service error: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Ollama + MedGemma AI", "FAIL", f"Error: {str(e)}")
            return False
    
    def test_environment_config(self):
        """Test environment configuration"""
        try:
            env_file = Path('.env')
            
            if env_file.exists():
                env_content = env_file.read_text()
                
                required_vars = [
                    'VIGIA_ENV',
                    'REDIS_URL',
                    'OLLAMA_URL',
                    'HOSPITAL_ID',
                    'STAFF_ID'
                ]
                
                missing_vars = [var for var in required_vars if var not in env_content]
                
                if not missing_vars:
                    self.log_test(
                        "Environment Configuration",
                        "PASS",
                        "All required medical environment variables configured"
                    )
                    return True
                else:
                    self.log_test(
                        "Environment Configuration",
                        "FAIL",
                        f"Missing variables: {missing_vars}"
                    )
                    return False
            else:
                self.log_test("Environment Configuration", "FAIL", ".env file not found")
                return False
                
        except Exception as e:
            self.log_test("Environment Configuration", "FAIL", f"Error: {str(e)}")
            return False
    
    def test_medical_audit_trail(self):
        """Test medical audit trail functionality"""
        try:
            engine = MedicalDecisionEngine()
            
            # Make a decision and check audit trail
            decision = engine.make_clinical_decision(
                lpp_grade=3,
                confidence=0.88,
                anatomical_location="heel",
                patient_context={"age": 60}
            )
            
            # Validate audit trail exists
            audit_trail = decision.get('audit_trail', {})
            quality_metrics = decision.get('quality_metrics', {})
            
            if (quality_metrics.get('assessment_id') or audit_trail.get('assessment_id')):
                
                assessment_id = (quality_metrics.get('assessment_id') or 
                               audit_trail.get('assessment_id'))
                self.log_test(
                    "Medical Audit Trail",
                    "PASS",
                    f"Complete audit trail: {assessment_id}"
                )
                return True
            else:
                self.log_test("Medical Audit Trail", "FAIL", "Incomplete audit trail")
                return False
                
        except Exception as e:
            self.log_test("Medical Audit Trail", "FAIL", f"Error: {str(e)}")
            return False
    
    def test_demo_readiness(self):
        """Test demo interface readiness"""
        try:
            demo_file = Path('launch_demo.py')
            
            if demo_file.exists():
                # Check if gradio is available
                try:
                    import gradio as gr
                    self.log_test(
                        "Demo Interface",
                        "PASS",
                        "Gradio demo ready for launch"
                    )
                    return True
                except ImportError:
                    self.log_test("Demo Interface", "WARN", "Gradio not installed")
                    return True  # Not critical
            else:
                self.log_test("Demo Interface", "FAIL", "Demo file not found")
                return False
                
        except Exception as e:
            self.log_test("Demo Interface", "FAIL", f"Error: {str(e)}")
            return False
    
    async def run_all_tests(self):
        """Run all validation tests"""
        print("🩺 VIGIA MEDICAL SYSTEM VALIDATION")
        print("=" * 50)
        print()
        
        # Run all tests
        tests = [
            self.test_medical_decision_engine,
            self.test_cv_pipeline_framework,  
            self.test_phi_tokenization_client,
            self.test_redis_connection,
            self.test_ollama_service,
            self.test_environment_config,
            self.test_medical_audit_trail,
            self.test_demo_readiness
        ]
        
        for test in tests:
            test()
            print()
        
        # Summary
        print("=" * 50)
        print(f"🏥 MEDICAL SYSTEM VALIDATION COMPLETE")
        print(f"📊 Results: {self.passed_tests}/{self.total_tests} tests passed")
        
        if self.passed_tests >= 6:  # Allow for optional services
            print("✅ SYSTEM READY FOR HACKATHON")
            print("🚀 Run: python launch_demo.py")
        elif self.passed_tests >= 4:
            print("⚠️  System partially ready - check failed tests")
        else:
            print("❌ System not ready - critical failures detected")
        
        print()
        print("📋 Detailed Results:")
        for test_name, result in self.results.items():
            status_symbol = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⚠️"
            print(f"   {status_symbol} {test_name}: {result['status']}")
            if result["details"]:
                print(f"      {result['details']}")
        
        return self.passed_tests >= 6

async def main():
    """Main validation function"""
    validator = VigiaValidator()
    success = await validator.run_all_tests()
    
    # Generate validation report
    report = {
        "validation_timestamp": time.time(),
        "system_ready": success,
        "tests_passed": validator.passed_tests,
        "total_tests": validator.total_tests,
        "results": validator.results
    }
    
    with open('validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Validation report saved: validation_report.json")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)