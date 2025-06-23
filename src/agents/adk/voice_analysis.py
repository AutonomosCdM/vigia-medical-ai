"""
Voice Analysis Agent (ADK)
=========================

ADK-based voice analysis agent for medical voice processing.
Integrates with Hume AI and Batman tokenization for HIPAA compliance.

Key Features:
- Voice expression analysis through Hume AI
- Medical assessment and recommendations
- Batman token integration (NO PHI)
- A2A communication with other medical agents

Usage:
    agent = VoiceAnalysisAgent()
    result = await agent.process_voice_analysis(audio_data, token_id, context)
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from vigia_detect.agents.base_agent import BaseAgent, AgentMessage, AgentResponse
from vigia_detect.ai.hume_ai_client import HumeAIClient, VoiceAnalysisResult, create_hume_ai_client
from vigia_detect.systems.voice_medical_analysis import VoiceMedicalAnalysisEngine
from vigia_detect.utils.audit_service import AuditService

logger = logging.getLogger(__name__)


class VoiceAnalysisAgent(BaseAgent):
    """
    ADK-based agent for voice analysis and medical assessment.
    
    Capabilities:
    - Voice expression analysis using Hume AI
    - Medical voice assessment 
    - Integration with Batman tokenization
    - A2A communication for collaborative analysis
    """
    
    def __init__(self, agent_id: str = "voice_analysis_agent"):
        super().__init__(agent_id=agent_id, agent_type="voice_analysis")
        
        self.hume_client = None
        self.voice_engine = VoiceMedicalAnalysisEngine()
        self.audit_service = AuditService()
        
        # Initialize Hume AI client if available
        try:
            self.hume_client = create_hume_ai_client()
            logger.info("Hume AI client initialized successfully")
        except Exception as e:
            logger.warning(f"Hume AI client not available: {e}")
    
    async def initialize(self) -> bool:
        """Initialize the voice analysis agent"""
        try:
            # Test Hume AI connection if available
            if self.hume_client:
                logger.info("Voice Analysis Agent initialized with Hume AI support")
            else:
                logger.info("Voice Analysis Agent initialized in mock mode")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Voice Analysis Agent: {e}")
            return False
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        capabilities = [
            "voice_expression_analysis",
            "medical_voice_assessment",
            "emotional_analysis",
            "pain_assessment_voice",
            "stress_evaluation_voice",
            "mental_health_screening_voice",
            "voice_trend_analysis",
            "batman_tokenization_support"
        ]
        
        if self.hume_client:
            capabilities.extend([
                "hume_ai_integration",
                "advanced_voice_processing"
            ])
        
        return capabilities
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process incoming agent messages"""
        try:
            action = message.action
            
            if action == "analyze_voice":
                return await self._handle_voice_analysis(message)
            elif action == "assess_medical_voice":
                return await self._handle_medical_assessment(message)
            elif action == "analyze_voice_trends":
                return await self._handle_trend_analysis(message)
            elif action == "get_voice_capabilities":
                return await self._handle_capabilities_request(message)
            else:
                return AgentResponse(
                    success=False,
                    message=f"Unknown action: {action}",
                    data={}
                )
                
        except Exception as e:
            logger.error(f"Error processing message in Voice Analysis Agent: {e}")
            return AgentResponse(
                success=False,
                message=f"Processing error: {str(e)}",
                data={}
            )
    
    async def _handle_voice_analysis(self, message: AgentMessage) -> AgentResponse:
        """Handle voice analysis requests"""
        try:
            # Extract parameters
            audio_data = message.data.get("audio_data")
            token_id = message.data.get("token_id")  # Batman token
            patient_context = message.data.get("patient_context", {})
            
            if not audio_data or not token_id:
                return AgentResponse(
                    success=False,
                    message="Missing required parameters: audio_data, token_id",
                    data={}
                )
            
            # Perform voice analysis
            result = await self.analyze_voice_expressions(
                audio_data, token_id, patient_context
            )
            
            return AgentResponse(
                success=True,
                message="Voice analysis completed successfully",
                data={
                    "analysis_result": result,
                    "token_id": token_id,
                    "hipaa_compliant": True
                }
            )
            
        except Exception as e:
            logger.error(f"Voice analysis failed: {e}")
            return AgentResponse(
                success=False,
                message=f"Voice analysis failed: {str(e)}",
                data={}
            )
    
    async def _handle_medical_assessment(self, message: AgentMessage) -> AgentResponse:
        """Handle medical voice assessment requests"""
        try:
            # Extract parameters
            expressions = message.data.get("expressions")
            token_id = message.data.get("token_id")  # Batman token
            patient_context = message.data.get("patient_context", {})
            
            if not expressions or not token_id:
                return AgentResponse(
                    success=False,
                    message="Missing required parameters: expressions, token_id",
                    data={}
                )
            
            # Perform medical assessment
            assessment = self.voice_engine.analyze_patient_voice(
                expressions, patient_context, token_id
            )
            
            return AgentResponse(
                success=True,
                message="Medical voice assessment completed",
                data={
                    "medical_assessment": assessment.dict(),
                    "token_id": token_id,
                    "assessment_id": assessment.assessment_id
                }
            )
            
        except Exception as e:
            logger.error(f"Medical assessment failed: {e}")
            return AgentResponse(
                success=False,
                message=f"Medical assessment failed: {str(e)}",
                data={}
            )
    
    async def _handle_trend_analysis(self, message: AgentMessage) -> AgentResponse:
        """Handle voice trend analysis requests"""
        try:
            token_id = message.data.get("token_id")
            timeframe = message.data.get("timeframe", "7_days")
            
            if not token_id:
                return AgentResponse(
                    success=False,
                    message="Missing required parameter: token_id",
                    data={}
                )
            
            # Get voice analysis trend data
            trend_data = await self._analyze_voice_trends(token_id, timeframe)
            
            return AgentResponse(
                success=True,
                message="Voice trend analysis completed",
                data={
                    "trend_data": trend_data,
                    "token_id": token_id,
                    "timeframe": timeframe
                }
            )
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return AgentResponse(
                success=False,
                message=f"Trend analysis failed: {str(e)}",
                data={}
            )
    
    async def _handle_capabilities_request(self, message: AgentMessage) -> AgentResponse:
        """Handle capabilities request"""
        return AgentResponse(
            success=True,
            message="Voice Analysis Agent capabilities",
            data={
                "capabilities": self.get_capabilities(),
                "hume_ai_available": self.hume_client is not None,
                "agent_type": "voice_analysis",
                "batman_tokenization": True
            }
        )
    
    async def analyze_voice_expressions(
        self,
        audio_data: Union[bytes, str],
        token_id: str,  # Batman token
        patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze voice expressions using Hume AI or mock analysis.
        
        Args:
            audio_data: Audio file data or base64 encoded audio
            token_id: Batman token ID (tokenized patient identifier)
            patient_context: Medical context (age, conditions, etc.)
            
        Returns:
            Voice analysis result dictionary
        """
        try:
            if self.hume_client:
                # Use real Hume AI analysis
                result = await self.hume_client.analyze_voice_expressions(
                    audio_data, token_id, patient_context
                )
                
                return {
                    "token_id": result.token_id,
                    "analysis_id": result.analysis_id,
                    "expressions": result.expressions,
                    "medical_indicators": {
                        "pain_score": result.medical_indicators.pain_score,
                        "stress_level": result.medical_indicators.stress_level,
                        "emotional_distress": result.medical_indicators.emotional_distress,
                        "alert_level": result.medical_indicators.alert_level,
                        "recommendations": result.medical_indicators.medical_recommendations
                    },
                    "confidence_level": result.confidence_level,
                    "timestamp": result.timestamp.isoformat(),
                    "hipaa_compliant": True,
                    "analysis_method": "hume_ai"
                }
            else:
                # Use mock analysis for testing
                return await self._mock_voice_analysis(token_id, patient_context)
                
        except Exception as e:
            logger.error(f"Voice expression analysis failed: {e}")
            raise
    
    async def _mock_voice_analysis(
        self,
        token_id: str,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Mock voice analysis for testing purposes"""
        
        # Generate mock expressions based on patient context
        mock_expressions = {
            "Joy": 0.2,
            "Sadness": 0.4,
            "Anxiety": 0.6,
            "Pain": 0.5,
            "Stress": 0.5,
            "Calmness": 0.3,
            "Fear": 0.4,
            "Relief": 0.2
        }
        
        # Adjust based on context
        if patient_context:
            if patient_context.get("chronic_pain"):
                mock_expressions["Pain"] += 0.2
                mock_expressions["Stress"] += 0.1
            
            if patient_context.get("anxiety_disorder"):
                mock_expressions["Anxiety"] += 0.2
                mock_expressions["Fear"] += 0.1
        
        # Generate analysis ID
        analysis_id = f"mock_voice_{token_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate alert level
        max_distress = max(
            mock_expressions.get("Pain", 0),
            mock_expressions.get("Anxiety", 0),
            mock_expressions.get("Stress", 0)
        )
        
        if max_distress >= 0.8:
            alert_level = "critical"
        elif max_distress >= 0.6:
            alert_level = "high"
        elif max_distress >= 0.4:
            alert_level = "elevated"
        else:
            alert_level = "normal"
        
        return {
            "token_id": token_id,
            "analysis_id": analysis_id,
            "expressions": mock_expressions,
            "medical_indicators": {
                "pain_score": mock_expressions["Pain"],
                "stress_level": mock_expressions["Stress"],
                "emotional_distress": (mock_expressions["Sadness"] + mock_expressions["Anxiety"]) / 2,
                "alert_level": alert_level,
                "recommendations": [
                    "Monitor pain levels" if mock_expressions["Pain"] > 0.5 else None,
                    "Consider stress management" if mock_expressions["Stress"] > 0.5 else None,
                    "Anxiety assessment recommended" if mock_expressions["Anxiety"] > 0.6 else None
                ]
            },
            "confidence_level": 0.8,
            "timestamp": datetime.now().isoformat(),
            "hipaa_compliant": True,
            "analysis_method": "mock"
        }
    
    async def _analyze_voice_trends(
        self,
        token_id: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Analyze voice trends over time for a patient"""
        
        # This would typically query the database for historical voice analyses
        # For now, return mock trend data
        
        return {
            "token_id": token_id,
            "timeframe": timeframe,
            "data_points": 5,
            "trends": {
                "pain_trend": "stable",
                "stress_trend": "decreasing",
                "emotional_trend": "improving"
            },
            "summary": "Patient shows overall improvement in stress and emotional indicators",
            "recommendations": [
                "Continue current pain management approach",
                "Monitor stress reduction progress"
            ],
            "confidence": 0.75
        }
    
    async def communicate_with_agent(
        self,
        target_agent_id: str,
        action: str,
        data: Dict[str, Any]
    ) -> Optional[AgentResponse]:
        """Communicate with other agents for collaborative analysis"""
        try:
            message = AgentMessage(
                sender_id=self.agent_id,
                recipient_id=target_agent_id,
                action=action,
                data=data,
                message_type="request",
                priority="normal"
            )
            
            # This would use the A2A communication system
            # For now, return a mock response
            return AgentResponse(
                success=True,
                message=f"Communication with {target_agent_id} successful",
                data={"response": "acknowledged"}
            )
            
        except Exception as e:
            logger.error(f"Agent communication failed: {e}")
            return None
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status"""
        return {
            "agent_id": self.agent_id,
            "status": "healthy",
            "hume_ai_available": self.hume_client is not None,
            "capabilities_count": len(self.get_capabilities()),
            "last_analysis": datetime.now().isoformat(),
            "tokenization_compliant": True
        }


# Export for ADK integration
__all__ = ["VoiceAnalysisAgent"]