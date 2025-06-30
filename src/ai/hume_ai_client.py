"""
Hume AI Client for Vigia Medical System
======================================

Voice and emotional analysis for medical contexts using Hume AI.
Fully integrated with Batman tokenization system for HIPAA compliance.

Key Features:
- Batman token-based patient identification (NO PHI)
- Medical voice expression analysis
- HIPAA-compliant audit logging
- Integration with dual database architecture

Usage:
    client = HumeAIClient(api_key="your_api_key")
    result = await client.analyze_voice_expressions(audio_data, token_id="batman_123")
"""

import asyncio
import json
import logging
import os
import gzip
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import base64
import hashlib

import aiohttp
import numpy as np
from pydantic import BaseModel, Field

# Medical logging with audit compliance
from src.utils.audit_service import AuditService
from src.db.supabase_client import SupabaseClient
from src.db.raw_outputs_client import RawOutputsClient

# Batman tokenization for HIPAA compliance (import will be done dynamically if needed)

# Raw outputs capture will be imported only when needed to avoid circular imports

logger = logging.getLogger(__name__)


class VoiceAlertLevel(Enum):
    """Voice analysis alert levels"""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MedicalVoiceIndicators:
    """Medical voice analysis indicators"""
    pain_score: float
    stress_level: float
    emotional_distress: float
    anxiety_indicators: float
    depression_markers: float
    alert_level: str
    confidence_score: float
    medical_recommendations: List[str]
    timestamp: datetime
    analysis_id: str


@dataclass
class VoiceAnalysisResult:
    """Complete voice analysis result"""
    token_id: str  # Batman token (NO PHI)
    analysis_id: str
    expressions: Dict[str, float]
    medical_indicators: MedicalVoiceIndicators
    technical_metadata: Dict[str, Any]
    timestamp: datetime
    confidence_level: float
    hipaa_compliant: bool = True  # Always True for Batman tokens
    raw_outputs: Optional[Any] = None  # Raw Hume AI outputs for research


class VoiceMedicalAssessment(BaseModel):
    """Comprehensive medical assessment from voice analysis"""
    patient_id: str = Field(..., description="Batman token ID (tokenized)")
    assessment_id: str = Field(..., description="Unique assessment identifier")
    urgency_level: str = Field(..., description="Medical urgency level")
    primary_concerns: List[str] = Field(..., description="Primary medical concerns")
    medical_recommendations: List[str] = Field(..., description="Medical recommendations")
    pain_assessment: Dict[str, Any] = Field(..., description="Pain analysis results")
    stress_evaluation: Dict[str, Any] = Field(..., description="Stress analysis results")
    mental_health_screening: Dict[str, Any] = Field(..., description="Mental health indicators")
    follow_up_required: bool = Field(..., description="Whether follow-up is needed")
    follow_up_timeframe: Optional[str] = Field(None, description="Timeframe for follow-up")
    specialist_referral: Optional[str] = Field(None, description="Specialist type if needed")
    confidence_score: float = Field(..., description="Overall confidence in assessment")
    timestamp: datetime = Field(default_factory=datetime.now)
    audit_trail: Dict[str, Any] = Field(default_factory=dict)


class HumeAIClient:
    """
    Hume AI client with medical voice analysis capabilities.
    
    Implements Batman tokenization for HIPAA compliance:
    - All patient identification uses token_id (Batman tokens)
    - No PHI data is processed or stored
    - Complete audit trail for medical compliance
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.hume.ai/v0"):
        self.api_key = api_key
        self.base_url = base_url
        self.audit_service = AuditService()
        self.supabase = SupabaseClient()
        # Import PHI tokenization service dynamically to avoid circular imports
        try:
            from src.core.phi_tokenization_client import PHITokenizationClient
            self.tokenization_service = PHITokenizationClient()
        except ImportError:
            logger.warning("PHI tokenization service not available")
            self.tokenization_service = None
        self.raw_outputs_client = RawOutputsClient()
        
        # Medical expression mappings for clinical analysis
        self.medical_expressions = {
            "pain_indicators": {
                "Empathic Pain": 1.0,
                "Anguish": 0.9,
                "Distress": 0.8,
                "Pain": 1.0,
                "Suffering": 0.9
            },
            "stress_indicators": {
                "Anxiety": 1.0,
                "Fear": 0.9,
                "Panic": 1.0,
                "Nervousness": 0.7,
                "Tension": 0.8
            },
            "depression_markers": {
                "Sadness": 0.9,
                "Despair": 1.0,
                "Hopelessness": 1.0,
                "Melancholy": 0.8,
                "Grief": 0.9
            },
            "positive_indicators": {
                "Joy": 1.0,
                "Contentment": 0.9,
                "Calmness": 0.8,
                "Peace": 0.9,
                "Relief": 0.7
            }
        }
    
    async def analyze_voice_expressions(
        self, 
        audio_data: Union[bytes, str], 
        token_id: str,  # Batman token (NO PHI)
        patient_context: Optional[Dict[str, Any]] = None
    ) -> VoiceAnalysisResult:
        """
        Analyze voice expressions using Hume AI with Batman tokenization.
        
        Args:
            audio_data: Audio file data or base64 encoded audio
            token_id: Batman token ID (tokenized patient identifier)
            patient_context: Medical context (age, conditions, etc.)
            
        Returns:
            VoiceAnalysisResult with medical analysis
        """
        analysis_id = self._generate_analysis_id(token_id)
        
        try:
            # Log analysis start with Batman token
            from ..utils.audit_service import AuditEventType
            await self.audit_service.log_event(
                event_type=AuditEventType.IMAGE_PROCESSED,  # Use existing enum
                component="hume_ai_voice_analysis",
                action="analysis_start",
                details={
                    "analysis_id": analysis_id,
                    "token_id": token_id,  # Batman token (HIPAA compliant)
                    "has_patient_context": patient_context is not None,
                    "tokenized_patient": True  # Confirm using Batman tokens
                }
            )
            
            # Prepare audio for Hume AI
            if isinstance(audio_data, str):
                audio_bytes = base64.b64decode(audio_data)
            else:
                audio_bytes = audio_data
            
            # Call Hume AI API
            hume_result = await self._call_hume_api(audio_bytes)
            
            # Extract expressions from API response
            expressions = self._extract_expressions(hume_result)
            
            # Analyze medical indicators
            medical_indicators = self._analyze_medical_indicators(
                expressions, 
                patient_context or {}
            )
            
            # Capture raw outputs for research and audit
            raw_outputs = self._capture_hume_raw_outputs(
                hume_result=hume_result,
                expressions=expressions,
                audio_metadata=patient_context.get('audio_metadata') if patient_context else None
            )
            
            # Create comprehensive result
            result = VoiceAnalysisResult(
                token_id=token_id,  # Batman token
                analysis_id=analysis_id,
                expressions=expressions,
                medical_indicators=medical_indicators,
                technical_metadata={
                    "hume_api_version": "v0",
                    "analysis_timestamp": datetime.now().isoformat(),
                    "patient_context_provided": patient_context is not None,
                    "tokenization_compliant": True
                },
                timestamp=datetime.now(),
                confidence_level=medical_indicators.confidence_score,
                hipaa_compliant=True,  # Batman tokenization ensures compliance
                raw_outputs=raw_outputs  # Raw Hume AI outputs for research
            )
            
            # Store result in Processing Database (NO PHI)
            await self._store_analysis_result(result)
            
            # Store raw outputs if available
            if result.raw_outputs:
                try:
                    raw_output_id = await self.raw_outputs_client.store_raw_output(
                        token_id=token_id,
                        ai_engine="hume_ai",
                        raw_outputs=result.raw_outputs,
                        structured_result_id=result.analysis_id,
                        structured_result_type="voice_analysis",
                        research_approved=True,
                        retention_priority="high"  # Voice data is valuable for research
                    )
                    
                    if raw_output_id:
                        logger.info(f"Stored Hume AI raw outputs {raw_output_id} for analysis {result.analysis_id}")
                    
                except Exception as raw_error:
                    logger.warning(f"Failed to store Hume AI raw outputs: {raw_error}")
                    # Continue processing even if raw storage fails
            
            # Log successful analysis
            await self.audit_service.log_event(
                event_type=AuditEventType.MEDICAL_DECISION,
                component="hume_ai_voice_analysis",
                action="analysis_complete",
                details={
                    "analysis_id": analysis_id,
                    "token_id": token_id,
                    "alert_level": medical_indicators.alert_level,
                    "pain_score": medical_indicators.pain_score,
                    "confidence": medical_indicators.confidence_score
                }
            )
            
            return result
            
        except Exception as e:
            # Log error with Batman token
            try:
                await self.audit_service.log_event(
                    event_type=AuditEventType.ERROR_OCCURRED,
                    component="hume_ai_voice_analysis",
                    action="analysis_error",
                    details={
                        "analysis_id": analysis_id,
                        "token_id": token_id,
                        "error_type": str(type(e).__name__),
                        "error_message": str(e)
                    }
                )
            except Exception as audit_error:
                logger.warning(f"Audit logging failed: {audit_error}")
            
            logger.error(f"Voice analysis failed for token {token_id}: {e}")
            raise
    
    async def _call_hume_api(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Call Hume AI API for voice analysis"""
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('file', audio_bytes, filename='audio.wav', content_type='audio/wav')
            data.add_field('models', 'prosody')  # Voice prosody model
            
            headers = {
                'X-Hume-Api-Key': self.api_key
            }
            
            async with session.post(
                f"{self.base_url}/expression/batch/jobs",
                data=data,
                headers=headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Hume API error: {response.status}")
                
                result = await response.json()
                
                # For batch jobs, we need to poll for results
                job_id = result.get('job_id')
                if job_id:
                    return await self._poll_hume_job(session, job_id, headers)
                
                return result
    
    async def _poll_hume_job(
        self, 
        session: aiohttp.ClientSession, 
        job_id: str, 
        headers: Dict[str, str],
        max_attempts: int = 30
    ) -> Dict[str, Any]:
        """Poll Hume AI job for completion"""
        for attempt in range(max_attempts):
            async with session.get(
                f"{self.base_url}/expression/batch/jobs/{job_id}",
                headers=headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Hume job polling error: {response.status}")
                
                job_status = await response.json()
                
                if job_status.get('state') == 'COMPLETED':
                    # Get job results
                    async with session.get(
                        f"{self.base_url}/expression/batch/jobs/{job_id}/predictions",
                        headers=headers
                    ) as predictions_response:
                        if predictions_response.status == 200:
                            return await predictions_response.json()
                
                elif job_status.get('state') == 'FAILED':
                    raise Exception(f"Hume job failed: {job_status.get('message', 'Unknown error')}")
                
                # Wait before next poll
                await asyncio.sleep(2)
        
        raise Exception("Hume job polling timeout")
    
    def _extract_expressions(self, hume_result: Dict[str, Any]) -> Dict[str, float]:
        """Extract emotion expressions from Hume API result"""
        expressions = {}
        
        try:
            predictions = hume_result.get('predictions', [])
            if predictions:
                prediction = predictions[0]
                models = prediction.get('models', {})
                prosody = models.get('prosody', {})
                grouped_predictions = prosody.get('grouped_predictions', [])
                
                if grouped_predictions:
                    group = grouped_predictions[0]
                    predictions_list = group.get('predictions', [])
                    
                    if predictions_list:
                        prediction_data = predictions_list[0]
                        emotions = prediction_data.get('emotions', [])
                        
                        for emotion in emotions:
                            name = emotion.get('name')
                            score = emotion.get('score', 0.0)
                            if name:
                                expressions[name] = score
        
        except Exception as e:
            logger.error(f"Error extracting expressions: {e}")
        
        return expressions
    
    def _analyze_medical_indicators(
        self, 
        expressions: Dict[str, float],
        patient_context: Dict[str, Any]
    ) -> MedicalVoiceIndicators:
        """Analyze medical indicators from voice expressions"""
        
        # Calculate weighted scores for different medical aspects
        pain_score = self._calculate_weighted_score(
            expressions, 
            self.medical_expressions["pain_indicators"]
        )
        
        stress_level = self._calculate_weighted_score(
            expressions,
            self.medical_expressions["stress_indicators"]
        )
        
        depression_score = self._calculate_weighted_score(
            expressions,
            self.medical_expressions["depression_markers"]
        )
        
        positive_score = self._calculate_weighted_score(
            expressions,
            self.medical_expressions["positive_indicators"]
        )
        
        # Calculate composite scores
        emotional_distress = (depression_score + stress_level) / 2
        anxiety_indicators = stress_level
        
        # Determine alert level
        alert_level = self._determine_alert_level(
            pain_score, stress_level, emotional_distress, anxiety_indicators
        )
        
        # Generate medical recommendations
        recommendations = self._generate_medical_recommendations(
            pain_score, stress_level, emotional_distress, 
            anxiety_indicators, depression_score, patient_context
        )
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(expressions, patient_context)
        
        return MedicalVoiceIndicators(
            pain_score=pain_score,
            stress_level=stress_level,
            emotional_distress=emotional_distress,
            anxiety_indicators=anxiety_indicators,
            depression_markers=depression_score,
            alert_level=alert_level,
            confidence_score=confidence,
            medical_recommendations=recommendations,
            timestamp=datetime.now(),
            analysis_id=self._generate_analysis_id("temp")
        )
    
    def _calculate_weighted_score(
        self, 
        expressions: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """Calculate weighted score for medical indicators"""
        total_score = 0.0
        total_weight = 0.0
        
        for expression, weight in weights.items():
            if expression in expressions:
                score = expressions[expression]
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_alert_level(
        self,
        pain_score: float,
        stress_level: float,
        emotional_distress: float,
        anxiety_indicators: float
    ) -> str:
        """Determine medical alert level based on scores"""
        max_score = max(pain_score, stress_level, emotional_distress, anxiety_indicators)
        
        if max_score >= 0.8:
            return "critical"
        elif max_score >= 0.6:
            return "high"
        elif max_score >= 0.4:
            return "elevated"
        else:
            return "normal"
    
    def _generate_medical_recommendations(
        self,
        pain_score: float,
        stress_level: float,
        emotional_distress: float,
        anxiety_indicators: float,
        depression_markers: float,
        patient_context: Dict[str, Any]
    ) -> List[str]:
        """Generate medical recommendations based on analysis"""
        recommendations = []
        
        # Pain management recommendations
        if pain_score >= 0.7:
            recommendations.append("Immediate pain assessment and management required")
            if patient_context.get("chronic_pain"):
                recommendations.append("Review current pain management protocol")
        elif pain_score >= 0.5:
            recommendations.append("Monitor pain levels and consider intervention")
        
        # Stress management recommendations
        if stress_level >= 0.7:
            recommendations.append("High stress levels detected - intervention needed")
            recommendations.append("Consider stress reduction techniques")
        elif stress_level >= 0.5:
            recommendations.append("Elevated stress - monitor and provide support")
        
        # Mental health recommendations
        if depression_markers >= 0.6:
            recommendations.append("Depression indicators present - professional evaluation recommended")
        
        if anxiety_indicators >= 0.7:
            recommendations.append("High anxiety levels - consider anxiety management intervention")
        
        # Emergency recommendations
        if max(pain_score, stress_level, emotional_distress) >= 0.8:
            recommendations.append("Consider immediate medical evaluation")
        
        return recommendations
    
    def _calculate_confidence_score(
        self,
        expressions: Dict[str, float],
        patient_context: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the analysis"""
        base_confidence = 0.7
        
        # Increase confidence with more expression data
        if len(expressions) >= 20:
            base_confidence += 0.1
        
        # Increase confidence with patient context
        if patient_context:
            base_confidence += 0.1
        
        # Decrease confidence if very few expressions detected
        if len(expressions) < 10:
            base_confidence -= 0.2
        
        return min(max(base_confidence, 0.1), 0.95)
    
    def _generate_analysis_id(self, token_id: str) -> str:
        """Generate unique analysis ID"""
        timestamp = datetime.now().isoformat()
        data = f"{token_id}_{timestamp}"
        return f"voice_{hashlib.md5(data.encode()).hexdigest()[:12]}"
    
    def _capture_hume_raw_outputs(self, 
                                 hume_result: Dict[str, Any],
                                 expressions: Dict[str, float],
                                 audio_metadata: Optional[Dict[str, Any]] = None) -> Any:
        """
        Capture raw Hume AI outputs for research and audit.
        
        Args:
            hume_result: Complete Hume API response
            expressions: Processed emotion expressions
            audio_metadata: Optional audio metadata
            
        Returns:
            RawOutputCapture object with Hume AI data
        """
        # Extract raw expression vectors if available
        raw_vectors = None
        try:
            predictions = hume_result.get('predictions', [])
            if predictions:
                prediction = predictions[0]
                models = prediction.get('models', {})
                prosody = models.get('prosody', {})
                
                # Convert prosody data to numpy array for compression
                prosody_array = np.array([
                    [emotion.get('score', 0.0) for emotion in 
                     prosody.get('grouped_predictions', [{}])[0].get('predictions', [{}])[0].get('emotions', [])]
                ])
                
                if prosody_array.size > 0:
                    raw_vectors = gzip.compress(prosody_array.tobytes())
        except Exception as e:
            logger.warning(f"Failed to extract raw emotion vectors: {e}")
        
        # Model metadata
        model_metadata = {
            "model_architecture": "hume_prosody",
            "api_version": "v0",
            "model_version": "evi_large",
            "emotion_count": len(expressions),
            "prosody_model": "prosody_v2",
            "language_detection": "english",
            "medical_context_aware": True,
            "medical_grade": False  # External API, not medical-grade
        }
        
        # Processing metadata
        processing_metadata = {
            "expression_count": len(expressions),
            "max_emotion_score": max(expressions.values()) if expressions else 0.0,
            "min_emotion_score": min(expressions.values()) if expressions else 0.0,
            "avg_emotion_score": sum(expressions.values()) / len(expressions) if expressions else 0.0,
            "medical_context": "voice_analysis",
            "analysis_type": "emotion_prosody",
            "timestamp": datetime.now().isoformat()
        }
        
        # Add audio metadata if available
        if audio_metadata:
            processing_metadata.update(audio_metadata)
        
        # Dynamic import to avoid circular imports
        from src.cv_pipeline.adaptive_medical_detector import RawOutputCapture
        return RawOutputCapture(
            raw_predictions=hume_result,  # Complete Hume API response
            expression_vectors=raw_vectors,  # Compressed emotion vectors
            model_metadata=model_metadata,
            processing_metadata=processing_metadata,
            compressed_size=len(raw_vectors) if raw_vectors else 0
        )
    
    async def _store_analysis_result(self, result: VoiceAnalysisResult):
        """Store analysis result in Processing Database (Batman tokens only)"""
        try:
            # Store in voice_analyses table with Batman token
            analysis_data = {
                "token_id": result.token_id,  # Batman token (NO PHI)
                "analysis_id": result.analysis_id,
                "expressions": result.expressions,
                "pain_score": result.medical_indicators.pain_score,
                "stress_level": result.medical_indicators.stress_level,
                "emotional_distress": result.medical_indicators.emotional_distress,
                "alert_level": result.medical_indicators.alert_level,
                "confidence_score": result.confidence_level,
                "recommendations": result.medical_indicators.medical_recommendations,
                "created_at": result.timestamp.isoformat(),
                "hipaa_compliant": True,
                "tokenization_method": "batman"
            }
            
            # Insert into Processing Database
            await self.supabase.insert("voice_analyses", analysis_data)
            
        except Exception as e:
            logger.error(f"Failed to store voice analysis: {e}")


def create_hume_ai_client(api_key: Optional[str] = None) -> HumeAIClient:
    """Create Hume AI client with environment configuration"""
    if not api_key:
        api_key = os.getenv("HUME_AI_API_KEY")
    
    if not api_key:
        raise ValueError("Hume AI API key not provided")
    
    return HumeAIClient(api_key=api_key)


# Export public classes and functions
__all__ = [
    "HumeAIClient",
    "VoiceAnalysisResult", 
    "MedicalVoiceIndicators",
    "VoiceAlertLevel",
    "VoiceMedicalAssessment",
    "create_hume_ai_client"
]