"""
Integration Validation Tests for Vigia Medical System
====================================================

Comprehensive validation of all external integrations and
agent implementations to ensure production readiness.
"""

import pytest
import asyncio
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Test imports
from src.agents.agent_factory import VigiaAgentFactory, create_complete_vigia_system
from src.messaging.whatsapp_processor import WhatsAppProcessor, MessageType
from src.messaging.whatsapp.isolated_bot import IsolatedWhatsAppBot
from src.communication.twilio_client import TwilioClient
from src.cv_pipeline.medical_detector_factory import create_medical_detector
from src.cv_pipeline.adaptive_medical_detector import AdaptiveMedicalDetector
from src.core.phi_tokenization_client import PHITokenizationClient

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAgentFactory:
    """Test agent factory and orchestration system"""
    
    @pytest.fixture
    def agent_factory(self):
        """Create agent factory for testing"""
        return VigiaAgentFactory()
    
    @pytest.mark.asyncio
    async def test_create_single_agent(self, agent_factory):
        """Test creating a single agent"""
        # Test creating risk assessment agent
        agent = await agent_factory.create_agent(
            agent_type="risk_assessment",
            agent_id="test_risk_agent"
        )
        
        assert agent is not None
        assert agent.agent_type == "risk_assessment"
        assert agent.agent_id == "test_risk_agent"
        assert len(agent.capabilities) > 0
        
        # Cleanup
        await agent_factory.shutdown_all_agents()
    
    @pytest.mark.asyncio
    async def test_agent_factory_statistics(self, agent_factory):
        """Test agent factory statistics tracking"""
        initial_stats = agent_factory.stats.copy()
        
        # Create an agent
        agent = await agent_factory.create_agent(
            agent_type="voice_analysis",
            agent_id="test_voice_agent"
        )
        
        # Check statistics updated
        assert agent_factory.stats['agents_created'] == initial_stats['agents_created'] + 1
        assert agent_factory.stats['agents_active'] == initial_stats['agents_active'] + 1
        
        # Cleanup
        await agent_factory.shutdown_all_agents()
    
    @pytest.mark.asyncio
    async def test_health_check_all_agents(self, agent_factory):
        """Test health checking all agents"""
        # Create a few agents
        await agent_factory.create_agent("diagnostic", "test_diagnostic")
        await agent_factory.create_agent("monai_review", "test_monai")
        
        # Perform health check
        health_results = await agent_factory.health_check_all_agents()
        
        assert 'total_agents' in health_results
        assert 'healthy_agents' in health_results
        assert health_results['total_agents'] == 2
        assert health_results['healthy_agents'] <= health_results['total_agents']
        
        # Cleanup
        await agent_factory.shutdown_all_agents()


class TestWhatsAppIntegration:
    """Test WhatsApp production integration"""
    
    @pytest.fixture
    def mock_twilio_client(self):
        """Mock Twilio client for testing"""
        client = Mock(spec=TwilioClient)
        client.is_available.return_value = True
        client.send_message.return_value = True
        client.send_media_message.return_value = True
        return client
    
    @pytest.fixture
    def whatsapp_processor(self, mock_twilio_client):
        """Create WhatsApp processor with mock client"""
        return WhatsAppProcessor(twilio_client=mock_twilio_client)
    
    @pytest.mark.asyncio
    async def test_send_message(self, whatsapp_processor):
        """Test sending WhatsApp message"""
        message_payload = {
            'to': '+56912345678',
            'type': MessageType.TEXT.value,
            'text': {'body': 'Test message'},
            'token_id': 'test_token'
        }
        
        response = await whatsapp_processor.send_message(message_payload)
        
        assert response.success is True
        assert response.message_id is not None
        assert response.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_send_medical_result(self, whatsapp_processor):
        """Test sending medical result via WhatsApp"""
        analysis_result = {
            'lpp_grade': 2,
            'confidence': 0.85,
            'anatomical_location': 'Sacro',
            'medical_recommendations': [
                'Implementar medidas de alivio de presión',
                'Protocolo de cuidado de heridas mejorado'
            ],
            'session_id': 'test_session'
        }
        
        response = await whatsapp_processor.send_medical_result(
            recipient='+56912345678',
            analysis_result=analysis_result,
            token_id='test_batman_token'
        )
        
        assert response.success is True
    
    @pytest.mark.asyncio
    async def test_webhook_processing(self, whatsapp_processor):
        """Test webhook processing"""
        webhook_data = {
            'From': 'whatsapp:+56912345678',
            'To': 'whatsapp:+56987654321',
            'Body': 'Hola, necesito ayuda médica',
            'MessageSid': 'test_message_id',
            'MediaUrl0': None
        }
        
        result = await whatsapp_processor.process_incoming_webhook(webhook_data)
        
        assert result['success'] is True
        assert result['from'] == '+56912345678'
        assert result['message_type'] == MessageType.TEXT.value
    
    def test_processor_statistics(self, whatsapp_processor):
        """Test processor statistics"""
        stats = whatsapp_processor.get_statistics()
        
        assert 'whatsapp_available' in stats
        assert 'twilio_configured' in stats
        assert 'statistics' in stats
        assert 'supported_types' in stats
        assert 'medical_templates' in stats
    
    def test_health_status(self, whatsapp_processor):
        """Test processor health status"""
        health = whatsapp_processor.get_health_status()
        
        assert 'processor_status' in health
        assert 'twilio_available' in health
        assert 'error_rate' in health
        assert 'last_health_check' in health


class TestWhatsAppBot:
    """Test isolated WhatsApp bot"""
    
    @pytest.fixture
    def mock_whatsapp_processor(self):
        """Mock WhatsApp processor"""
        processor = Mock(spec=WhatsAppProcessor)
        processor.send_message = AsyncMock(return_value=Mock(success=True))
        processor.send_welcome_message = AsyncMock(return_value=Mock(success=True))
        processor.send_analysis_request_confirmation = AsyncMock(return_value=Mock(success=True))
        processor.send_medical_result = AsyncMock(return_value=Mock(success=True))
        processor.send_help_message = AsyncMock(return_value=Mock(success=True))
        processor.is_available.return_value = True
        return processor
    
    @pytest.fixture
    def whatsapp_bot(self, mock_whatsapp_processor):
        """Create WhatsApp bot with mock processor"""
        return IsolatedWhatsAppBot(
            whatsapp_processor=mock_whatsapp_processor,
            session_timeout_minutes=30
        )
    
    @pytest.mark.asyncio
    async def test_new_session_creation(self, whatsapp_bot):
        """Test creating new WhatsApp session"""
        phone_number = '+56912345678'
        
        result = await whatsapp_bot.process_incoming_message(
            phone_number=phone_number,
            message_content='Hola',
            message_type=MessageType.TEXT.value
        )
        
        assert result['success'] is True
        assert len(whatsapp_bot.sessions) == 1
        
        # Check session was created properly
        session = list(whatsapp_bot.sessions.values())[0]
        assert session.phone_number == phone_number
        assert session.batman_token is not None
    
    @pytest.mark.asyncio
    async def test_image_message_handling(self, whatsapp_bot):
        """Test handling image messages"""
        phone_number = '+56912345678'
        
        # Set up medical analysis callback
        async def mock_medical_analysis(**kwargs):
            return {
                'lpp_grade': 1,
                'confidence': 0.75,
                'anatomical_location': 'Talón',
                'medical_recommendations': ['Monitoreo regular']
            }
        
        whatsapp_bot.set_medical_analysis_callback(mock_medical_analysis)
        
        # First create session
        await whatsapp_bot.process_incoming_message(
            phone_number=phone_number,
            message_content='Hola'
        )
        
        # Then send image
        result = await whatsapp_bot.process_incoming_message(
            phone_number=phone_number,
            message_content='Imagen de lesión',
            message_type=MessageType.IMAGE.value,
            media_url='https://example.com/image.jpg'
        )
        
        assert result['success'] is True
        assert result['processing_started'] is True
        assert result['image_count'] == 1
    
    @pytest.mark.asyncio
    async def test_session_timeout_handling(self, whatsapp_bot):
        """Test session timeout handling"""
        # Create session with short timeout
        bot_short_timeout = IsolatedWhatsAppBot(
            whatsapp_processor=whatsapp_bot.whatsapp_processor,
            session_timeout_minutes=0.01  # Very short timeout for testing
        )
        
        phone_number = '+56912345678'
        
        # Create session
        await bot_short_timeout.process_incoming_message(
            phone_number=phone_number,
            message_content='Hola'
        )
        
        # Get the session
        session = list(bot_short_timeout.sessions.values())[0]
        
        # Wait for timeout
        await asyncio.sleep(1)
        
        # Check session is expired
        assert session.is_expired(0.01) is True
    
    def test_bot_statistics(self, whatsapp_bot):
        """Test bot statistics"""
        stats = whatsapp_bot.get_session_statistics()
        
        assert 'bot_statistics' in stats
        assert 'active_sessions' in stats
        assert 'session_timeout_minutes' in stats
        assert 'whatsapp_available' in stats
    
    def test_bot_health_status(self, whatsapp_bot):
        """Test bot health status"""
        health = whatsapp_bot.get_health_status()
        
        assert 'bot_status' in health
        assert 'whatsapp_processor_available' in health
        assert 'active_sessions' in health
        assert 'medical_callback_configured' in health


class TestMedicalDetection:
    """Test medical detection systems"""
    
    def test_medical_detector_factory(self):
        """Test medical detector factory"""
        detector = create_medical_detector()
        
        assert detector is not None
        assert isinstance(detector, AdaptiveMedicalDetector)
    
    def test_adaptive_medical_detector_creation(self):
        """Test creating adaptive medical detector"""
        detector = AdaptiveMedicalDetector(
            monai_timeout=5.0,
            confidence_threshold_monai=0.7,
            confidence_threshold_yolo=0.6
        )
        
        assert detector is not None
        assert detector.monai_timeout == 5.0
        assert detector.confidence_threshold_monai == 0.7
        assert detector.confidence_threshold_yolo == 0.6
    
    def test_detector_statistics(self):
        """Test detector statistics"""
        detector = AdaptiveMedicalDetector()
        stats = detector.get_engine_statistics()
        
        assert 'monai_available' in stats
        assert 'yolo_available' in stats
        assert 'engine_stats' in stats
        assert 'configuration' in stats


class TestPHITokenization:
    """Test PHI tokenization system"""
    
    @pytest.fixture
    def phi_tokenizer(self):
        """Create PHI tokenizer"""
        return PHITokenizationClient()
    
    @pytest.mark.asyncio
    async def test_create_batman_token(self, phi_tokenizer):
        """Test creating Batman token"""
        hospital_mrn = "TEST-001"
        patient_data = {
            'name': 'Test Patient',
            'age': 65,
            'condition': 'pressure_injury_risk'
        }
        
        token = await phi_tokenizer.create_token_async(hospital_mrn, patient_data)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    @pytest.mark.asyncio
    async def test_resolve_batman_token(self, phi_tokenizer):
        """Test resolving Batman token"""
        # Create token first
        hospital_mrn = "TEST-002"
        patient_data = {'test': 'data'}
        
        token = await phi_tokenizer.create_token_async(hospital_mrn, patient_data)
        
        # Resolve token
        resolved_data = await phi_tokenizer.resolve_token_async(token)
        
        assert resolved_data is not None
        assert resolved_data['original_mrn'] == hospital_mrn
    
    def test_tokenizer_health(self, phi_tokenizer):
        """Test tokenizer health status"""
        health = phi_tokenizer.get_health_status()
        
        assert 'tokenizer_status' in health
        assert 'batman_mode_active' in health
        assert 'tokens_created' in health


class TestIntegrationEnd2End:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_create_minimal_medical_system(self):
        """Test creating minimal medical system"""
        factory = VigiaAgentFactory()
        
        # Create just a few key agents for testing
        agents = {}
        
        try:
            # Create core agents
            agents['risk'] = await factory.create_agent('risk_assessment')
            agents['diagnostic'] = await factory.create_agent('diagnostic')
            agents['monai'] = await factory.create_agent('monai_review')
            
            # Verify all created successfully
            assert len(agents) == 3
            assert all(agent.status == 'healthy' for agent in agents.values())
            
            # Test health check
            health_results = await factory.health_check_all_agents()
            assert health_results['total_agents'] == 3
            
        finally:
            # Cleanup
            await factory.shutdown_all_agents()
    
    @pytest.mark.asyncio
    async def test_whatsapp_to_medical_analysis_flow(self):
        """Test complete flow from WhatsApp to medical analysis"""
        # Mock components
        mock_twilio = Mock(spec=TwilioClient)
        mock_twilio.is_available.return_value = True
        mock_twilio.send_message.return_value = True
        
        whatsapp_processor = WhatsAppProcessor(twilio_client=mock_twilio)
        whatsapp_bot = IsolatedWhatsAppBot(whatsapp_processor=whatsapp_processor)
        
        # Set up medical analysis callback
        analysis_results = []
        
        async def mock_medical_analysis(**kwargs):
            result = {
                'lpp_grade': 2,
                'confidence': 0.82,
                'anatomical_location': 'Cóccix',
                'medical_recommendations': ['Protocolo de cuidado especializado'],
                'session_id': kwargs.get('session_id'),
                'batman_token': kwargs.get('batman_token')
            }
            analysis_results.append(result)
            return result
        
        whatsapp_bot.set_medical_analysis_callback(mock_medical_analysis)
        
        phone_number = '+56912345678'
        
        # Step 1: Patient initiates conversation
        result1 = await whatsapp_bot.process_incoming_message(
            phone_number=phone_number,
            message_content='Hola, necesito ayuda'
        )
        assert result1['success'] is True
        
        # Step 2: Patient sends image
        result2 = await whatsapp_bot.process_incoming_message(
            phone_number=phone_number,
            message_content='Imagen de lesión',
            message_type=MessageType.IMAGE.value,
            media_url='https://example.com/test_image.jpg'
        )
        assert result2['success'] is True
        assert result2['processing_started'] is True
        
        # Step 3: Verify medical analysis was triggered
        assert len(analysis_results) == 1
        analysis = analysis_results[0]
        assert analysis['lpp_grade'] == 2
        assert analysis['batman_token'] is not None
        
        # Step 4: Verify session updated with results
        session = list(whatsapp_bot.sessions.values())[0]
        assert len(session.analysis_results) == 1
        assert session.state == 'completed'


def test_import_validations():
    """Test that all critical imports work"""
    
    # Test agent imports
    from src.agents.risk_assessment_agent import RiskAssessmentAgent
    from src.agents.monai_review_agent import MonaiReviewAgent
    from src.agents.diagnostic_agent import DiagnosticAgent
    from src.agents.adk.voice_analysis import VoiceAnalysisAgent
    from src.agents.master_medical_orchestrator import MasterMedicalOrchestrator
    
    # Test integration imports
    from src.messaging.whatsapp_processor import WhatsAppProcessor
    from src.messaging.whatsapp.isolated_bot import IsolatedWhatsAppBot
    from src.communication.twilio_client import TwilioClient
    
    # Test medical detection imports
    from src.cv_pipeline.adaptive_medical_detector import AdaptiveMedicalDetector
    from src.cv_pipeline.medical_detector_factory import create_medical_detector
    
    # Test factory imports
    from src.agents.agent_factory import VigiaAgentFactory, create_complete_vigia_system
    
    assert True  # If we get here, all imports worked


if __name__ == '__main__':
    # Run basic validation
    test_import_validations()
    print("✅ All critical imports validated successfully")
    
    # Run async test example
    async def run_basic_test():
        factory = VigiaAgentFactory()
        agent = await factory.create_agent('risk_assessment')
        health = await factory.health_check_all_agents()
        await factory.shutdown_all_agents()
        print(f"✅ Basic agent creation test: {health['total_agents']} agents created")
    
    asyncio.run(run_basic_test())
    print("✅ Basic integration validation complete")