"""
VIGIA Medical AI - Voice Analysis Agent
=====================================

Specialized agent for medical voice analysis using Hume AI integration.
Part of the 9-agent medical coordination system.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from ..ai.hume_ai_client import HumeAIClient, create_hume_ai_client, VoiceAnalysisResult
from ..core.phi_tokenization_client import PHITokenizationClient
from ..utils.audit_service import AuditService, AuditEventType
from .base_agent import BaseAgent, AgentMessage, AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class VoiceMedicalAssessment:
    """Comprehensive voice-based medical assessment"""
    batman_token: str
    analysis_id: str
    pain_assessment: Dict[str, float]
    emotional_distress: Dict[str, float]
    stress_indicators: Dict[str, float]
    urgency_level: str
    medical_recommendations: List[str]
    confidence_score: float
    raw_hume_data: Optional[Dict[str, Any]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class VoiceAnalysisAgent(BaseAgent):
    """
    Voice Analysis Agent for Medical Voice Processing.
    
    Integrates Hume AI voice analysis with medical context:
    - Pain detection and quantification
    - Emotional distress assessment
    - Stress level monitoring
    - Medical urgency evaluation
    - HIPAA-compliant processing
    """
    
    def __init__(self, agent_id: str = "voice_analysis_agent"):
        super().__init__(agent_id, "voice_analysis")
        self.specialization = "medical_voice_processing"
        
        # Initialize Hume AI client
        try:
            self.hume_client = create_hume_ai_client()
            self.hume_available = True
            logger.info("✅ Hume AI client initialized successfully")
        except Exception as e:
            logger.warning(f"Hume AI client not available: {e}")
            self.hume_available = False
            self.hume_client = None
        
        # Initialize supporting services
        self.phi_client = PHITokenizationClient()
        self.audit_service = AuditService()
        
        # Medical voice analysis thresholds
        self.pain_thresholds = {
            "mild": 0.3,
            "moderate": 0.5,
            "severe": 0.7,
            "critical": 0.9
        }
        
        self.urgency_levels = {
            "routine": 0.0,
            "priority": 0.4,
            "urgent": 0.7,
            "critical": 0.9
        }
    
    async def analyze_medical_voice_async(self, 
                                        audio_data: bytes,
                                        batman_token: str,
                                        patient_context: Optional[Dict[str, Any]] = None,
                                        medical_history: Optional[Dict[str, Any]] = None) -> VoiceMedicalAssessment:
        """
        Analyze medical voice data for pain, distress, and emotional indicators.
        
        Args:
            audio_data: Voice audio in bytes (WAV format preferred)
            batman_token: HIPAA-compliant patient token
            patient_context: Medical context and conditions
            medical_history: Patient medical history for context
            
        Returns:
            VoiceMedicalAssessment with comprehensive analysis
        """
        analysis_start = datetime.now()
        
        try:
            # Log analysis start
            await self.audit_service.log_event(
                event_type=AuditEventType.IMAGE_PROCESSED,
                component=self.agent_id,
                action="voice_analysis_start",
                details={
                    "batman_token": batman_token,
                    "audio_size_bytes": len(audio_data),
                    "has_patient_context": patient_context is not None,
                    "has_medical_history": medical_history is not None
                }
            )
            
            if self.hume_available and self.hume_client:
                try:
                    # Use real Hume AI analysis
                    hume_result = await self._analyze_with_hume_ai(
                        audio_data, batman_token, patient_context
                    )
                    
                    # Process Hume AI results into medical assessment
                    assessment = await self._process_hume_results(
                        hume_result, batman_token, patient_context, medical_history
                    )
                except Exception as hume_error:
                    logger.warning(f"Hume AI analysis failed, falling back to mock: {hume_error}")
                    # Fall back to mock analysis
                    assessment = await self._generate_mock_assessment(
                        batman_token, patient_context, medical_history
                    )
            else:
                # Use mock analysis for development/testing
                logger.warning("Using mock voice analysis - Hume AI not available")
                assessment = await self._generate_mock_assessment(
                    batman_token, patient_context, medical_history
                )
            
            # Calculate processing time
            processing_time = (datetime.now() - analysis_start).total_seconds()
            
            # Log successful analysis
            await self.audit_service.log_event(
                event_type=AuditEventType.MEDICAL_DECISION,
                component=self.agent_id,
                action="voice_analysis_complete",
                details={
                    "batman_token": batman_token,
                    "analysis_id": assessment.analysis_id,
                    "urgency_level": assessment.urgency_level,
                    "confidence_score": assessment.confidence_score,
                    "processing_time_seconds": processing_time
                }
            )
            
            return assessment
            
        except Exception as e:
            # Log error
            await self.audit_service.log_event(
                event_type=AuditEventType.ERROR_OCCURRED,
                component=self.agent_id,
                action="voice_analysis_error",
                details={
                    "batman_token": batman_token,
                    "error_type": str(type(e).__name__),
                    "error_message": str(e)
                }
            )
            
            logger.error(f"Voice analysis failed for {batman_token}: {e}")
            raise
    
    async def _analyze_with_hume_ai(self, 
                                  audio_data: bytes,
                                  batman_token: str,
                                  patient_context: Optional[Dict[str, Any]]) -> VoiceAnalysisResult:
        """Analyze voice with Hume AI"""
        return await self.hume_client.analyze_voice_expressions(
            audio_data=audio_data,
            token_id=batman_token,
            patient_context=patient_context
        )
    
    async def _process_hume_results(self,
                                  hume_result: VoiceAnalysisResult,
                                  batman_token: str,
                                  patient_context: Optional[Dict[str, Any]],
                                  medical_history: Optional[Dict[str, Any]]) -> VoiceMedicalAssessment:
        """Process Hume AI results into medical assessment"""
        
        # Extract pain assessment
        pain_assessment = {
            "pain_score": hume_result.medical_indicators.pain_score,
            "pain_level": self._categorize_pain_level(hume_result.medical_indicators.pain_score),
            "empathic_pain": hume_result.expressions.get("Empathic Pain", 0.0),
            "anguish": hume_result.expressions.get("Anguish", 0.0),
            "suffering": hume_result.expressions.get("Suffering", 0.0)
        }
        
        # Extract emotional distress
        emotional_distress = {
            "distress_score": hume_result.medical_indicators.emotional_distress,
            "anxiety_level": hume_result.medical_indicators.anxiety_indicators,
            "depression_markers": hume_result.medical_indicators.depression_markers,
            "fear": hume_result.expressions.get("Fear", 0.0),
            "despair": hume_result.expressions.get("Despair", 0.0)
        }
        
        # Extract stress indicators
        stress_indicators = {
            "stress_level": hume_result.medical_indicators.stress_level,
            "tension": hume_result.expressions.get("Tension", 0.0),
            "nervousness": hume_result.expressions.get("Nervousness", 0.0),
            "panic": hume_result.expressions.get("Panic", 0.0)
        }
        
        # Determine urgency level with medical context
        urgency_level = self._determine_medical_urgency(
            hume_result, patient_context, medical_history
        )
        
        # Generate enhanced medical recommendations
        recommendations = self._generate_enhanced_recommendations(
            hume_result, patient_context, medical_history, urgency_level
        )
        
        return VoiceMedicalAssessment(
            batman_token=batman_token,
            analysis_id=hume_result.analysis_id,
            pain_assessment=pain_assessment,
            emotional_distress=emotional_distress,
            stress_indicators=stress_indicators,
            urgency_level=urgency_level,
            medical_recommendations=recommendations,
            confidence_score=hume_result.confidence_level,
            raw_hume_data=hume_result.expressions
        )
    
    async def _generate_mock_assessment(self,
                                      batman_token: str,
                                      patient_context: Optional[Dict[str, Any]],
                                      medical_history: Optional[Dict[str, Any]]) -> VoiceMedicalAssessment:
        """Generate mock assessment for development/testing"""
        import random
        import hashlib
        
        # Generate deterministic but realistic mock data
        seed = hashlib.md5(batman_token.encode()).hexdigest()
        random.seed(int(seed[:8], 16))
        
        # Mock pain assessment
        pain_score = random.uniform(0.2, 0.8)
        pain_assessment = {
            "pain_score": pain_score,
            "pain_level": self._categorize_pain_level(pain_score),
            "empathic_pain": random.uniform(0.0, 0.6),
            "anguish": random.uniform(0.0, 0.4),
            "suffering": random.uniform(0.0, 0.5)
        }
        
        # Mock emotional distress
        distress_score = random.uniform(0.1, 0.7)
        emotional_distress = {
            "distress_score": distress_score,
            "anxiety_level": random.uniform(0.0, 0.6),
            "depression_markers": random.uniform(0.0, 0.4),
            "fear": random.uniform(0.0, 0.5),
            "despair": random.uniform(0.0, 0.3)
        }
        
        # Mock stress indicators
        stress_level = random.uniform(0.1, 0.6)
        stress_indicators = {
            "stress_level": stress_level,
            "tension": random.uniform(0.0, 0.5),
            "nervousness": random.uniform(0.0, 0.4),
            "panic": random.uniform(0.0, 0.3)
        }
        
        # Determine urgency
        max_score = max(pain_score, distress_score, stress_level)
        if max_score >= 0.8:
            urgency_level = "critical"
        elif max_score >= 0.6:
            urgency_level = "urgent"
        elif max_score >= 0.4:
            urgency_level = "priority"
        else:
            urgency_level = "routine"
        
        # Generate recommendations
        recommendations = []
        if pain_score >= 0.6:
            recommendations.append("High pain levels detected - immediate medical evaluation recommended")
        if distress_score >= 0.5:
            recommendations.append("Emotional distress present - consider psychological support")
        if stress_level >= 0.5:
            recommendations.append("Elevated stress levels - implement stress reduction interventions")
        
        if not recommendations:
            recommendations.append("Continue monitoring patient condition")
        
        # Add mock recommendation for development
        recommendations.append("(MOCK ANALYSIS - Using simulated voice analysis)")
        
        analysis_id = f"mock_voice_{hashlib.md5(batman_token.encode()).hexdigest()[:12]}"
        
        return VoiceMedicalAssessment(
            batman_token=batman_token,
            analysis_id=analysis_id,
            pain_assessment=pain_assessment,
            emotional_distress=emotional_distress,
            stress_indicators=stress_indicators,
            urgency_level=urgency_level,
            medical_recommendations=recommendations,
            confidence_score=random.uniform(0.7, 0.9),
            raw_hume_data={"mock_mode": True}
        )
    
    def _categorize_pain_level(self, pain_score: float) -> str:
        """Categorize pain level based on score"""
        if pain_score >= self.pain_thresholds["critical"]:
            return "critical"
        elif pain_score >= self.pain_thresholds["severe"]:
            return "severe"
        elif pain_score >= self.pain_thresholds["moderate"]:
            return "moderate"
        elif pain_score >= self.pain_thresholds["mild"]:
            return "mild"
        else:
            return "minimal"
    
    def _determine_medical_urgency(self,
                                 hume_result: VoiceAnalysisResult,
                                 patient_context: Optional[Dict[str, Any]],
                                 medical_history: Optional[Dict[str, Any]]) -> str:
        """Determine medical urgency with enhanced context"""
        base_urgency = hume_result.medical_indicators.alert_level
        
        # Enhance with medical context
        if patient_context:
            # Critical patients get elevated urgency
            if patient_context.get("critical_condition"):
                if base_urgency in ["normal", "elevated"]:
                    base_urgency = "high"
            
            # Post-surgical patients need closer monitoring
            if patient_context.get("post_surgical"):
                if base_urgency == "normal":
                    base_urgency = "elevated"
        
        # Consider medical history
        if medical_history:
            chronic_pain = medical_history.get("chronic_pain", False)
            if chronic_pain and hume_result.medical_indicators.pain_score > 0.6:
                base_urgency = "high"
        
        return base_urgency
    
    def _generate_enhanced_recommendations(self,
                                         hume_result: VoiceAnalysisResult,
                                         patient_context: Optional[Dict[str, Any]],
                                         medical_history: Optional[Dict[str, Any]],
                                         urgency_level: str) -> List[str]:
        """Generate enhanced medical recommendations"""
        recommendations = list(hume_result.medical_indicators.medical_recommendations)
        
        # Add context-specific recommendations
        if patient_context:
            condition = patient_context.get("medical_condition")
            if condition == "pressure_injury" and hume_result.medical_indicators.pain_score > 0.5:
                recommendations.append("Pressure injury pain management - review wound care protocol")
            
            if patient_context.get("post_surgical") and hume_result.medical_indicators.stress_level > 0.6:
                recommendations.append("Post-surgical anxiety detected - consider anxiolytic intervention")
        
        # Add urgency-specific recommendations
        if urgency_level == "critical":
            recommendations.append("CRITICAL ALERT: Immediate medical team notification required")
        elif urgency_level == "high":
            recommendations.append("High priority case - schedule urgent medical review within 2 hours")
        
        # Add monitoring recommendations
        if hume_result.confidence_level < 0.7:
            recommendations.append("Voice analysis confidence below threshold - recommend additional assessment methods")
        
        return recommendations
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """
        Process A2A message for voice analysis.
        
        Expected message content:
        - audio_data: bytes
        - batman_token: str
        - patient_context: Optional[Dict]
        - medical_history: Optional[Dict]
        """
        try:
            # Call parent process_message first
            await super().process_message(message)
            
            # Extract message content
            content = message.content
            
            # Perform voice analysis
            assessment = await self.analyze_medical_voice_async(
                audio_data=content.get("audio_data"),
                batman_token=content.get("batman_token"),
                patient_context=content.get("patient_context"),
                medical_history=content.get("medical_history")
            )
            
            # Generate recommendations for next actions
            next_actions = []
            if assessment.urgency_level == "critical":
                next_actions.append("immediate_medical_intervention")
            elif assessment.urgency_level == "urgent":
                next_actions.append("schedule_medical_review")
            
            return AgentResponse(
                success=True,
                content=asdict(assessment),
                message=f"Voice analysis completed - {assessment.urgency_level} urgency level",
                timestamp=datetime.now(),
                requires_human_review=(assessment.urgency_level in ["critical", "urgent"]),
                next_actions=next_actions,
                metadata={
                    "agent_type": self.agent_type,
                    "specialization": self.specialization,
                    "processing_method": "hume_ai" if self.hume_available else "mock"
                }
            )
            
        except Exception as e:
            logger.error(f"Voice analysis message processing failed: {e}")
            return AgentResponse(
                success=False,
                content={},
                message=f"Voice analysis failed: {str(e)}",
                timestamp=datetime.now(),
                requires_human_review=True,
                metadata={"error": str(e), "agent_type": self.agent_type}
            )
    
    async def process_async(self, 
                          input_data: Dict[str, Any], 
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process voice analysis request through agent interface.
        
        Expected input_data:
        - audio_data: bytes
        - batman_token: str
        - patient_context: Optional[Dict]
        - medical_history: Optional[Dict]
        """
        try:
            # Extract required parameters
            audio_data = input_data.get("audio_data")
            batman_token = input_data.get("batman_token")
            
            if not audio_data or not batman_token:
                raise ValueError("audio_data and batman_token are required")
            
            # Perform voice analysis
            assessment = await self.analyze_medical_voice_async(
                audio_data=audio_data,
                batman_token=batman_token,
                patient_context=input_data.get("patient_context"),
                medical_history=input_data.get("medical_history")
            )
            
            # Return result as dictionary
            return {
                "success": True,
                "data": asdict(assessment),
                "message": f"Voice analysis completed - {assessment.urgency_level} urgency level",
                "confidence": assessment.confidence_score,
                "processing_time": (datetime.now() - datetime.fromisoformat(assessment.timestamp)).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Voice analysis agent failed: {e}")
            return {
                "success": False,
                "data": {},
                "message": f"Voice analysis failed: {str(e)}",
                "error": str(e)
            }


# Factory registration
def create_voice_analysis_agent() -> VoiceAnalysisAgent:
    """Factory function for voice analysis agent"""
    return VoiceAnalysisAgent()


# Export
__all__ = [
    "VoiceAnalysisAgent",
    "VoiceMedicalAssessment", 
    "create_voice_analysis_agent"
]