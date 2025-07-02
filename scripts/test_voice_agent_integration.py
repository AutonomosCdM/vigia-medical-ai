#!/usr/bin/env python3
"""
Test Voice Analysis Agent Integration with 9-Agent System
========================================================

Test the complete integration of VoiceAnalysisAgent with MasterMedicalOrchestrator.
"""

import asyncio
import sys
import tempfile
import wave
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.voice_analysis_agent import VoiceAnalysisAgent, create_voice_analysis_agent
from src.agents.base_agent import AgentMessage
from src.core.phi_tokenization_client import PHITokenizationClient

async def generate_test_audio() -> bytes:
    """Generate simple test audio"""
    # Generate 3 seconds of sine wave
    sample_rate = 16000
    duration = 3.0
    frequency = 440.0
    
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, False)
    signal = np.sin(2 * np.pi * frequency * t) * 0.8
    
    # Convert to 16-bit PCM
    audio_data = (signal * 32767).astype(np.int16)
    
    # Create WAV file in memory
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        with wave.open(temp_wav.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        # Read back as bytes
        with open(temp_wav.name, 'rb') as f:
            audio_bytes = f.read()
        
        # Cleanup
        import os
        os.unlink(temp_wav.name)
        
        return audio_bytes

async def test_voice_analysis_agent():
    """Test VoiceAnalysisAgent directly"""
    print("🎤 Testing VoiceAnalysisAgent directly...")
    
    try:
        # Create agent
        agent = create_voice_analysis_agent()
        await agent.initialize()
        print("✅ VoiceAnalysisAgent initialized")
        
        # Create Batman token
        phi_client = PHITokenizationClient()
        batman_token = await phi_client.create_token_async(
            hospital_mrn="TEST_VOICE_001",
            patient_data={"test": "voice_analysis"}
        )
        print(f"✅ Batman token created: {batman_token}")
        
        # Generate test audio
        audio_data = await generate_test_audio()
        print(f"✅ Test audio generated: {len(audio_data)} bytes")
        
        # Test direct analysis
        assessment = await agent.analyze_medical_voice_async(
            audio_data=audio_data,
            batman_token=batman_token,
            patient_context={
                "medical_condition": "post_surgical_monitoring",
                "anatomical_location": "sacrum"
            },
            medical_history={
                "chronic_pain": False,
                "previous_lpp": True
            }
        )
        
        print("✅ Voice analysis completed")
        print(f"   Analysis ID: {assessment.analysis_id}")
        print(f"   Pain Score: {assessment.pain_assessment['pain_score']:.3f}")
        print(f"   Urgency Level: {assessment.urgency_level}")
        print(f"   Confidence: {assessment.confidence_score:.3f}")
        print(f"   Recommendations: {len(assessment.medical_recommendations)}")
        
        return True, assessment
        
    except Exception as e:
        print(f"❌ Direct agent test failed: {e}")
        return False, None

async def test_a2a_message_processing():
    """Test A2A message processing"""
    print("\\n📡 Testing A2A message processing...")
    
    try:
        # Create agent
        agent = create_voice_analysis_agent()
        await agent.initialize()
        
        # Create Batman token
        phi_client = PHITokenizationClient()
        batman_token = await phi_client.create_token_async(
            hospital_mrn="TEST_A2A_001",
            patient_data={"test": "a2a_communication"}
        )
        
        # Generate test audio
        audio_data = await generate_test_audio()
        
        # Create A2A message
        message = AgentMessage(
            session_id="test_session_001",
            sender_id="master_medical_orchestrator",
            content={
                "audio_data": audio_data,
                "batman_token": batman_token,
                "patient_context": {
                    "medical_condition": "pressure_injury_monitoring",
                    "has_lpp_detected": True,
                    "lpp_grade": 2,
                    "anatomical_location": "heel"
                },
                "medical_history": {
                    "diabetes": True,
                    "mobility_limited": True
                }
            },
            message_type="processing_request",
            timestamp=datetime.now()
        )
        
        # Process message
        response = await agent.process_message(message)
        
        print("✅ A2A message processed")
        print(f"   Success: {response.success}")
        print(f"   Message: {response.message}")
        print(f"   Requires Human Review: {response.requires_human_review}")
        print(f"   Next Actions: {response.next_actions}")
        
        if response.success:
            assessment_data = response.content
            print(f"   Batman Token: {assessment_data['batman_token']}")
            print(f"   Pain Level: {assessment_data['pain_assessment']['pain_level']}")
            print(f"   Urgency: {assessment_data['urgency_level']}")
        
        return response.success, response
        
    except Exception as e:
        print(f"❌ A2A message test failed: {e}")
        return False, None

async def test_integration_with_orchestrator():
    """Test integration with MasterMedicalOrchestrator"""
    print("\\n🎭 Testing integration with MasterMedicalOrchestrator...")
    
    try:
        # Import and create orchestrator
        from src.agents.master_medical_orchestrator import MasterMedicalOrchestrator
        
        orchestrator = MasterMedicalOrchestrator()
        
        # Wait a bit for agent initialization
        await asyncio.sleep(2)
        
        # Check if voice analysis agent is registered
        if orchestrator.registered_agents.get('voice_analysis'):
            print("✅ VoiceAnalysisAgent registered in orchestrator")
            
            # Create test case data with voice
            phi_client = PHITokenizationClient()
            batman_token = await phi_client.create_token_async(
                hospital_mrn="TEST_ORCHESTRATOR_001",
                patient_data={"test": "orchestrator_integration"}
            )
            
            audio_data = await generate_test_audio()
            
            case_data = {
                "token_id": batman_token,
                "audio_data": audio_data,
                "patient_context": {
                    "medical_condition": "pressure_injury_grade_3",
                    "anatomical_location": "sacrum",
                    "post_surgical": True
                },
                "medical_history": {
                    "chronic_pain": True,
                    "diabetes": False
                }
            }
            
            # Mock image result for context
            image_result = {
                "success": True,
                "image_analysis": {
                    "lpp_detected": True,
                    "lpp_grade": 3,
                    "confidence": 0.85,
                    "anatomical_location": "sacrum"
                }
            }
            
            # Test voice analysis coordination
            voice_result = await orchestrator._coordinate_voice_analysis(case_data, image_result)
            
            print("✅ Voice analysis coordination completed")
            print(f"   Success: {voice_result.get('success')}")
            print(f"   Agent: {voice_result.get('agent')}")
            print(f"   Method: {voice_result.get('method')}")
            
            if voice_result.get('success'):
                data = voice_result.get('data', {})
                print(f"   Urgency Level: {data.get('urgency_level')}")
                print(f"   Confidence: {voice_result.get('confidence', 0):.3f}")
            
            return voice_result.get('success', False), voice_result
            
        else:
            print("❌ VoiceAnalysisAgent not registered in orchestrator")
            return False, None
            
    except Exception as e:
        print(f"❌ Orchestrator integration test failed: {e}")
        return False, None

async def main():
    """Run comprehensive voice agent integration tests"""
    print("🩺 VIGIA Medical AI - Voice Agent Integration Testing")
    print("=" * 65)
    
    tests = [
        ("Direct VoiceAnalysisAgent", test_voice_analysis_agent),
        ("A2A Message Processing", test_a2a_message_processing),
        ("Orchestrator Integration", test_integration_with_orchestrator)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\\n🔬 Running: {test_name}")
        try:
            success, result = await test_func()
            if success:
                passed_tests += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\\n" + "=" * 65)
    print(f"🏆 FINAL RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Voice analysis integration is working correctly.")
        success_rate = 100
    else:
        success_rate = (passed_tests / total_tests) * 100
        print(f"⚠️  {success_rate:.1f}% success rate - some components need attention")
    
    # Generate integration test report
    test_report = {
        "test_suite": "voice_agent_integration",
        "timestamp": datetime.now().isoformat(),
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "success_rate": f"{success_rate:.1f}%",
        "integration_status": "fully_integrated" if passed_tests == total_tests else "partial_integration",
        "hume_ai_available": False,  # Currently using mock
        "9_agent_system_ready": passed_tests >= 2
    }
    
    # Save test report
    report_path = project_root / "voice_integration_test_report.json"
    with open(report_path, 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print(f"📄 Integration test report saved to: {report_path}")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)