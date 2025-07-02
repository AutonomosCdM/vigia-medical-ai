"""
VIGIA Medical AI - Fallback Handler
==================================

Safe fallback mechanisms for edge cases with pre-approved medical responses,
graceful degradation, and seamless escalation to human experts.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import json
import random

from ..ai.medical_guardrails import ResponseSafetyLevel, SafetyViolationType
from ..core.medical_decision_engine import MedicalDecisionEngine
from ..utils.secure_logger import SecureLogger
from ..utils.audit_service import AuditService

logger = SecureLogger(__name__)

class FallbackTrigger(Enum):
    """Triggers for fallback activation"""
    SAFETY_VIOLATION = "safety_violation"
    LOW_CONFIDENCE = "low_confidence"
    MODEL_ERROR = "model_error"
    TIMEOUT = "timeout"
    EMERGENCY_DETECTED = "emergency_detected"
    UNKNOWN_CASE = "unknown_case"
    SYSTEM_OVERLOAD = "system_overload"
    COMPLIANCE_VIOLATION = "compliance_violation"

class FallbackStrategy(Enum):
    """Available fallback strategies"""
    SAFE_RESPONSE = "safe_response"
    HUMAN_ESCALATION = "human_escalation"
    ALTERNATIVE_MODEL = "alternative_model"
    SIMPLIFIED_RESPONSE = "simplified_response"
    EMERGENCY_PROTOCOL = "emergency_protocol"

@dataclass
class FallbackResponse:
    """Structured fallback response"""
    response_text: str
    strategy_used: FallbackStrategy
    confidence_level: float
    human_review_required: bool
    escalation_urgency: str  # low, medium, high, critical
    additional_actions: List[str]
    fallback_reason: str
    timestamp: datetime
    batman_token: Optional[str] = None

class MedicalFallbackHandler:
    """Medical-grade fallback handler for AI responses"""
    
    def __init__(self):
        self.audit_service = AuditService()
        self.medical_decision_engine = MedicalDecisionEngine()
        
        # Pre-approved safe responses by category
        self.safe_responses = {
            'general_consultation': [
                """
                I recommend consulting with a healthcare professional for a comprehensive 
                evaluation of your pressure injury concerns. They can provide personalized 
                assessment and evidence-based treatment recommendations following current 
                NPUAP guidelines.
                """,
                """
                For pressure injury assessment and management, it's important to work with 
                qualified healthcare providers who can examine the affected area and develop 
                an appropriate treatment plan based on your specific situation.
                """,
                """
                Pressure injury care requires professional medical evaluation. Please 
                schedule an appointment with your healthcare provider or wound care specialist 
                for proper assessment and treatment planning.
                """
            ],
            
            'emergency_response': [
                """
                The symptoms you've described may require immediate medical attention. 
                Please contact your healthcare provider, call your emergency services, 
                or visit the nearest emergency department for prompt evaluation and care.
                """,
                """
                This situation appears to need urgent medical evaluation. I strongly 
                recommend seeking immediate medical attention by contacting emergency 
                services or visiting your nearest emergency department.
                """,
                """
                For immediate medical concerns, please contact emergency services or 
                your healthcare provider right away. Do not delay seeking professional 
                medical care for potentially serious conditions.
                """
            ],
            
            'low_confidence': [
                """
                Based on the information provided, I cannot provide a confident assessment. 
                I recommend consulting with a healthcare professional who can perform a 
                proper clinical examination and provide accurate guidance.
                """,
                """
                The details provided require professional medical evaluation for accurate 
                assessment. Please consult with your healthcare provider or a wound care 
                specialist for personalized care recommendations.
                """,
                """
                This case would benefit from professional medical assessment. Healthcare 
                providers can offer more comprehensive evaluation and appropriate treatment 
                recommendations based on clinical examination.
                """
            ],
            
            'prevention_education': [
                """
                Pressure injury prevention follows evidence-based NPUAP guidelines including 
                regular repositioning, appropriate support surfaces, skin assessment, nutrition 
                optimization, and moisture management. Consult your healthcare team for 
                personalized prevention strategies.
                """,
                """
                Key pressure injury prevention strategies include frequent position changes, 
                proper nutrition, skin care, and risk assessment. Work with your healthcare 
                provider to develop a prevention plan appropriate for your specific risk factors.
                """,
                """
                Effective pressure injury prevention involves multiple evidence-based strategies. 
                Your healthcare team can help implement appropriate prevention measures based 
                on your individual risk assessment and care needs.
                """
            ],
            
            'system_limitation': [
                """
                I'm currently unable to provide a complete assessment for your inquiry. 
                For the most accurate and appropriate care guidance, please consult with 
                a healthcare professional who can provide comprehensive evaluation.
                """,
                """
                Due to system limitations, I cannot provide a detailed response to your 
                inquiry at this time. Please seek guidance from a qualified healthcare 
                provider for proper medical assessment and care recommendations.
                """,
                """
                I'm experiencing difficulties processing your request completely. For reliable 
                medical guidance regarding pressure injuries, please consult with your 
                healthcare provider or wound care specialist.
                """
            ]
        }
        
        # Emergency keywords for urgent response
        self.emergency_indicators = {
            'infection_signs': [
                'fever', 'chills', 'red streaks', 'pus', 'foul odor',
                'increasing pain', 'warm to touch', 'swelling'
            ],
            'severe_symptoms': [
                'severe pain', 'bleeding', 'large wound', 'deep wound',
                'bone visible', 'severe tissue damage'
            ],
            'systemic_concerns': [
                'sepsis', 'confusion', 'rapid heart rate', 'low blood pressure',
                'difficulty breathing', 'unconscious'
            ]
        }
        
        # Escalation criteria
        self.escalation_criteria = {
            'immediate': {
                'triggers': [FallbackTrigger.EMERGENCY_DETECTED],
                'max_response_time': 5,  # minutes
                'channels': ['slack', 'phone', 'email']
            },
            'urgent': {
                'triggers': [FallbackTrigger.SAFETY_VIOLATION, FallbackTrigger.COMPLIANCE_VIOLATION],
                'max_response_time': 30,  # minutes
                'channels': ['slack', 'email']
            },
            'standard': {
                'triggers': [FallbackTrigger.LOW_CONFIDENCE, FallbackTrigger.MODEL_ERROR],
                'max_response_time': 120,  # minutes
                'channels': ['email']
            }
        }
        
        # Fallback statistics
        self.fallback_stats = {
            'total_fallbacks': 0,
            'by_trigger': {trigger: 0 for trigger in FallbackTrigger},
            'by_strategy': {strategy: 0 for strategy in FallbackStrategy},
            'escalation_rate': 0.0,
            'average_confidence': 0.0
        }
        
        logger.info("MedicalFallbackHandler initialized")
    
    async def handle_fallback(self,
                             trigger: FallbackTrigger,
                             original_prompt: str,
                             failed_response: str = None,
                             error_details: Dict[str, Any] = None,
                             batman_token: str = None) -> FallbackResponse:
        """Handle fallback scenario with appropriate strategy"""
        
        try:
            self.fallback_stats['total_fallbacks'] += 1
            self.fallback_stats['by_trigger'][trigger] += 1
            
            # Determine fallback strategy
            strategy = await self._determine_fallback_strategy(trigger, original_prompt, error_details)
            
            # Generate fallback response
            fallback_response = await self._generate_fallback_response(
                strategy, trigger, original_prompt, batman_token
            )
            
            # Update statistics
            self.fallback_stats['by_strategy'][strategy] += 1
            
            # Log fallback event
            await self._log_fallback_event(trigger, strategy, fallback_response, batman_token)
            
            logger.info(f"Fallback handled: {trigger.value} -> {strategy.value}")
            
            return fallback_response
            
        except Exception as e:
            logger.error(f"Fallback handler error: {e}")
            
            # Ultimate fallback - always safe response
            return await self._create_emergency_fallback(trigger, batman_token)
    
    async def _determine_fallback_strategy(self,
                                          trigger: FallbackTrigger,
                                          original_prompt: str,
                                          error_details: Dict[str, Any] = None) -> FallbackStrategy:
        """Determine the most appropriate fallback strategy"""
        
        # Emergency situations always trigger emergency protocol
        if trigger == FallbackTrigger.EMERGENCY_DETECTED:
            return FallbackStrategy.EMERGENCY_PROTOCOL
        
        # Check for emergency indicators in prompt
        prompt_lower = original_prompt.lower()
        has_emergency_indicators = any(
            any(indicator in prompt_lower for indicator in indicators)
            for indicators in self.emergency_indicators.values()
        )
        
        if has_emergency_indicators:
            return FallbackStrategy.EMERGENCY_PROTOCOL
        
        # Safety violations require human escalation
        if trigger == FallbackTrigger.SAFETY_VIOLATION:
            return FallbackStrategy.HUMAN_ESCALATION
        
        # Compliance violations need safe response + escalation
        if trigger == FallbackTrigger.COMPLIANCE_VIOLATION:
            return FallbackStrategy.HUMAN_ESCALATION
        
        # System overload uses simplified response
        if trigger == FallbackTrigger.SYSTEM_OVERLOAD:
            return FallbackStrategy.SIMPLIFIED_RESPONSE
        
        # Model errors might benefit from alternative model
        if trigger == FallbackTrigger.MODEL_ERROR:
            # Check if alternative model is available
            if await self._alternative_model_available():
                return FallbackStrategy.ALTERNATIVE_MODEL
            else:
                return FallbackStrategy.SAFE_RESPONSE
        
        # Default to safe response for other cases
        return FallbackStrategy.SAFE_RESPONSE
    
    async def _generate_fallback_response(self,
                                         strategy: FallbackStrategy,
                                         trigger: FallbackTrigger,
                                         original_prompt: str,
                                         batman_token: str = None) -> FallbackResponse:
        """Generate appropriate fallback response based on strategy"""
        
        if strategy == FallbackStrategy.EMERGENCY_PROTOCOL:
            return await self._create_emergency_response(trigger, original_prompt, batman_token)
        
        elif strategy == FallbackStrategy.HUMAN_ESCALATION:
            return await self._create_escalation_response(trigger, original_prompt, batman_token)
        
        elif strategy == FallbackStrategy.ALTERNATIVE_MODEL:
            return await self._create_alternative_model_response(trigger, original_prompt, batman_token)
        
        elif strategy == FallbackStrategy.SIMPLIFIED_RESPONSE:
            return await self._create_simplified_response(trigger, original_prompt, batman_token)
        
        else:  # SAFE_RESPONSE
            return await self._create_safe_response(trigger, original_prompt, batman_token)
    
    async def _create_emergency_response(self,
                                        trigger: FallbackTrigger,
                                        original_prompt: str,
                                        batman_token: str = None) -> FallbackResponse:
        """Create emergency protocol response"""
        
        # Select appropriate emergency response
        emergency_text = random.choice(self.safe_responses['emergency_response'])
        
        # Add specific guidance based on detected indicators
        prompt_lower = original_prompt.lower()
        additional_guidance = []
        
        if any(indicator in prompt_lower for indicator in self.emergency_indicators['infection_signs']):
            additional_guidance.append("Signs of infection require immediate medical evaluation.")
        
        if any(indicator in prompt_lower for indicator in self.emergency_indicators['severe_symptoms']):
            additional_guidance.append("Severe symptoms need urgent professional assessment.")
        
        if additional_guidance:
            emergency_text += "\n\n" + " ".join(additional_guidance)
        
        return FallbackResponse(
            response_text=emergency_text.strip(),
            strategy_used=FallbackStrategy.EMERGENCY_PROTOCOL,
            confidence_level=0.95,  # High confidence in emergency protocol
            human_review_required=True,
            escalation_urgency="critical",
            additional_actions=[
                "immediate_medical_contact",
                "emergency_services_alert",
                "clinical_team_notification"
            ],
            fallback_reason=f"Emergency indicators detected: {trigger.value}",
            timestamp=datetime.now(),
            batman_token=batman_token
        )
    
    async def _create_escalation_response(self,
                                         trigger: FallbackTrigger,
                                         original_prompt: str,
                                         batman_token: str = None) -> FallbackResponse:
        """Create response that requires human escalation"""
        
        # Determine appropriate response category
        if 'prevent' in original_prompt.lower():
            response_text = random.choice(self.safe_responses['prevention_education'])
        else:
            response_text = random.choice(self.safe_responses['general_consultation'])
        
        # Add escalation notice
        escalation_notice = """
        
        Note: This inquiry has been flagged for additional review by our medical team 
        to ensure the most appropriate and safe guidance is provided.
        """
        
        response_text += escalation_notice
        
        return FallbackResponse(
            response_text=response_text.strip(),
            strategy_used=FallbackStrategy.HUMAN_ESCALATION,
            confidence_level=0.8,
            human_review_required=True,
            escalation_urgency="high" if trigger == FallbackTrigger.SAFETY_VIOLATION else "medium",
            additional_actions=[
                "medical_team_review",
                "safety_assessment",
                "response_audit"
            ],
            fallback_reason=f"Human escalation required: {trigger.value}",
            timestamp=datetime.now(),
            batman_token=batman_token
        )
    
    async def _create_safe_response(self,
                                   trigger: FallbackTrigger,
                                   original_prompt: str,
                                   batman_token: str = None) -> FallbackResponse:
        """Create safe fallback response"""
        
        # Determine most appropriate safe response category
        if trigger == FallbackTrigger.LOW_CONFIDENCE:
            response_text = random.choice(self.safe_responses['low_confidence'])
        elif trigger in [FallbackTrigger.MODEL_ERROR, FallbackTrigger.TIMEOUT, FallbackTrigger.SYSTEM_OVERLOAD]:
            response_text = random.choice(self.safe_responses['system_limitation'])
        else:
            response_text = random.choice(self.safe_responses['general_consultation'])
        
        # Add standard medical disclaimer
        disclaimer = """
        
        This response is for educational purposes only and does not constitute 
        medical advice. Always consult with qualified healthcare professionals 
        for medical concerns.
        """
        
        response_text += disclaimer
        
        return FallbackResponse(
            response_text=response_text.strip(),
            strategy_used=FallbackStrategy.SAFE_RESPONSE,
            confidence_level=0.7,
            human_review_required=False,
            escalation_urgency="low",
            additional_actions=["standard_monitoring"],
            fallback_reason=f"Safe fallback for: {trigger.value}",
            timestamp=datetime.now(),
            batman_token=batman_token
        )
    
    async def _create_simplified_response(self,
                                         trigger: FallbackTrigger,
                                         original_prompt: str,
                                         batman_token: str = None) -> FallbackResponse:
        """Create simplified response for system overload"""
        
        simplified_text = """
        For pressure injury concerns, the most important steps are:
        
        1. Consult with a healthcare professional for proper assessment
        2. Follow evidence-based prevention strategies if appropriate
        3. Seek immediate care for any concerning symptoms
        
        Please contact your healthcare provider for personalized guidance.
        """
        
        return FallbackResponse(
            response_text=simplified_text.strip(),
            strategy_used=FallbackStrategy.SIMPLIFIED_RESPONSE,
            confidence_level=0.6,
            human_review_required=False,
            escalation_urgency="low",
            additional_actions=["system_load_monitoring"],
            fallback_reason=f"Simplified response due to: {trigger.value}",
            timestamp=datetime.now(),
            batman_token=batman_token
        )
    
    async def _create_alternative_model_response(self,
                                                trigger: FallbackTrigger,
                                                original_prompt: str,
                                                batman_token: str = None) -> FallbackResponse:
        """Create response using alternative model"""
        
        # This would integrate with backup model
        # For now, using safe response with note about alternative processing
        
        response_text = random.choice(self.safe_responses['general_consultation'])
        
        alternative_note = """
        
        Note: This response was generated using our backup medical guidance system 
        to ensure continued service availability.
        """
        
        response_text += alternative_note
        
        return FallbackResponse(
            response_text=response_text.strip(),
            strategy_used=FallbackStrategy.ALTERNATIVE_MODEL,
            confidence_level=0.75,
            human_review_required=False,
            escalation_urgency="medium",
            additional_actions=["alternative_model_monitoring", "primary_model_restoration"],
            fallback_reason=f"Alternative model used: {trigger.value}",
            timestamp=datetime.now(),
            batman_token=batman_token
        )
    
    async def _create_emergency_fallback(self,
                                        trigger: FallbackTrigger,
                                        batman_token: str = None) -> FallbackResponse:
        """Create ultimate emergency fallback when all else fails"""
        
        emergency_fallback_text = """
        I'm currently unable to provide specific medical guidance. For any health 
        concerns, especially those related to pressure injuries or wounds, please 
        consult with a healthcare professional immediately.
        
        If you have urgent medical needs, contact emergency services or your 
        healthcare provider right away.
        
        This is an automated safety response to ensure your wellbeing.
        """
        
        return FallbackResponse(
            response_text=emergency_fallback_text.strip(),
            strategy_used=FallbackStrategy.SAFE_RESPONSE,
            confidence_level=0.5,
            human_review_required=True,
            escalation_urgency="critical",
            additional_actions=[
                "system_failure_alert",
                "immediate_technical_review",
                "manual_response_required"
            ],
            fallback_reason=f"Emergency fallback - system error: {trigger.value}",
            timestamp=datetime.now(),
            batman_token=batman_token
        )
    
    async def _alternative_model_available(self) -> bool:
        """Check if alternative model is available"""
        # This would check actual model availability
        # For now, returning False to use safe responses
        return False
    
    async def _log_fallback_event(self,
                                 trigger: FallbackTrigger,
                                 strategy: FallbackStrategy,
                                 response: FallbackResponse,
                                 batman_token: str = None) -> None:
        """Log fallback event for audit and analysis"""
        
        await self.audit_service.log_activity(
            activity_type="fallback_response",
            batman_token=batman_token,
            details={
                'trigger': trigger.value,
                'strategy': strategy.value,
                'confidence_level': response.confidence_level,
                'escalation_urgency': response.escalation_urgency,
                'human_review_required': response.human_review_required,
                'additional_actions': response.additional_actions,
                'fallback_reason': response.fallback_reason
            }
        )
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get fallback handler statistics"""
        
        total_fallbacks = self.fallback_stats['total_fallbacks']
        
        return {
            'total_fallbacks': total_fallbacks,
            'trigger_distribution': {
                trigger.value: count for trigger, count in self.fallback_stats['by_trigger'].items()
            },
            'strategy_distribution': {
                strategy.value: count for strategy, count in self.fallback_stats['by_strategy'].items()
            },
            'top_triggers': sorted(
                [(trigger.value, count) for trigger, count in self.fallback_stats['by_trigger'].items()],
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'escalation_rate': (
                self.fallback_stats['by_strategy'][FallbackStrategy.HUMAN_ESCALATION] / 
                max(total_fallbacks, 1)
            ),
            'emergency_rate': (
                self.fallback_stats['by_strategy'][FallbackStrategy.EMERGENCY_PROTOCOL] / 
                max(total_fallbacks, 1)
            )
        }
    
    async def test_fallback_scenarios(self) -> Dict[str, Any]:
        """Test various fallback scenarios for validation"""
        
        test_results = {}
        
        test_scenarios = [
            (FallbackTrigger.EMERGENCY_DETECTED, "Patient has severe bleeding and high fever"),
            (FallbackTrigger.SAFETY_VIOLATION, "Regular medical question"),
            (FallbackTrigger.LOW_CONFIDENCE, "Unclear symptoms described"),
            (FallbackTrigger.MODEL_ERROR, "Standard pressure injury question"),
            (FallbackTrigger.SYSTEM_OVERLOAD, "Prevention guidance request")
        ]
        
        for trigger, test_prompt in test_scenarios:
            try:
                response = await self.handle_fallback(
                    trigger=trigger,
                    original_prompt=test_prompt,
                    batman_token="TEST_TOKEN"
                )
                
                test_results[trigger.value] = {
                    'success': True,
                    'strategy': response.strategy_used.value,
                    'confidence': response.confidence_level,
                    'escalation_required': response.human_review_required,
                    'response_length': len(response.response_text)
                }
                
            except Exception as e:
                test_results[trigger.value] = {
                    'success': False,
                    'error': str(e)
                }
        
        return test_results


# Singleton instance
fallback_handler = MedicalFallbackHandler()

__all__ = [
    'MedicalFallbackHandler',
    'FallbackTrigger',
    'FallbackStrategy',
    'FallbackResponse',
    'fallback_handler'
]