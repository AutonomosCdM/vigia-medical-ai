"""
VIGIA Medical AI - Medical Guardrails
====================================

Medical-grade safety controls for LLM responses with NPUAP compliance,
content safety validation, and automatic escalation mechanisms.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from dataclasses import dataclass
import json

from ..utils.secure_logger import SecureLogger
from ..utils.audit_service import AuditService
from ..core.medical_decision_engine import MedicalDecisionEngine

logger = SecureLogger(__name__)

class SafetyViolationType(Enum):
    """Types of safety violations"""
    HARMFUL_MEDICAL_ADVICE = "harmful_medical_advice"
    MISINFORMATION = "misinformation"
    NPUAP_NONCOMPLIANCE = "npuap_noncompliance"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    LOW_CONFIDENCE = "low_confidence"
    MISSING_DISCLAIMER = "missing_disclaimer"
    EMERGENCY_MISHANDLING = "emergency_mishandling"
    MEDICATION_ERROR = "medication_error"
    DIAGNOSIS_OVERREACH = "diagnosis_overreach"

class ResponseSafetyLevel(Enum):
    """Safety levels for medical responses"""
    SAFE = "safe"
    NEEDS_REVIEW = "needs_review"
    UNSAFE = "unsafe"
    BLOCKED = "blocked"

@dataclass
class SafetyViolation:
    """Individual safety violation"""
    violation_type: SafetyViolationType
    severity: str  # low, medium, high, critical
    description: str
    suggested_fix: str
    confidence: float
    location: str  # where in text the violation occurs

@dataclass
class GuardrailResult:
    """Result of guardrail evaluation"""
    safety_level: ResponseSafetyLevel
    confidence_score: float
    violations: List[SafetyViolation]
    medical_accuracy_score: float
    npuap_compliance_score: float
    escalation_required: bool
    safe_alternative: Optional[str]
    processing_time: float
    timestamp: datetime

class MedicalGuardrails:
    """Medical guardrails for LLM safety validation"""
    
    def __init__(self):
        self.audit_service = AuditService()
        self.medical_decision_engine = MedicalDecisionEngine()
        
        # NPUAP Guidelines for validation
        self.npuap_guidelines = {
            'stage_definitions': {
                '0': 'No visible pressure injury',
                '1': 'Non-blanchable erythema of intact skin',
                '2': 'Partial thickness skin loss with exposed dermis',
                '3': 'Full thickness skin loss',
                '4': 'Full thickness skin and tissue loss',
                'U': 'Unstageable/unclassified',
                'DTI': 'Deep tissue pressure injury'
            },
            'risk_factors': [
                'immobility', 'limited mobility', 'sensory perception',
                'moisture', 'friction', 'shear', 'nutrition',
                'tissue perfusion', 'age'
            ],
            'prevention_strategies': [
                'frequent repositioning', 'pressure redistribution surfaces',
                'skin assessment', 'nutrition optimization',
                'moisture management', 'friction reduction'
            ]
        }
        
        # Harmful content patterns
        self.harmful_patterns = {
            'dangerous_advice': [
                r'ignore\s+medical\s+advice',
                r'stop\s+taking\s+medication',
                r'avoid\s+doctors?',
                r'self-treat\s+with',
                r'don\'t\s+seek\s+help'
            ],
            'misinformation': [
                r'pressure\s+ulcers?\s+heal\s+on\s+their\s+own',
                r'stage\s+4\s+is\s+not\s+serious',
                r'antibiotics?\s+are\s+never\s+needed',
                r'pain\s+means\s+healing'
            ],
            'inappropriate_diagnosis': [
                r'you\s+definitely\s+have',
                r'this\s+is\s+certainly',
                r'100%\s+sure\s+it\'s',
                r'no\s+doubt\s+about'
            ]
        }
        
        # Required disclaimers
        self.required_disclaimers = [
            'professional medical advice',
            'consult healthcare provider',
            'seek medical attention',
            'qualified medical professional'
        ]
        
        # Emergency keywords that require special handling
        self.emergency_keywords = [
            'bleeding', 'severe pain', 'infection signs',
            'fever', 'sepsis', 'emergency', 'urgent',
            'critical', 'life-threatening'
        ]
        
        # Safe fallback responses
        self.safe_responses = {
            'low_confidence': """
            Based on the information provided, I recommend consulting with a healthcare professional 
            for a proper assessment of this pressure injury concern. They can provide personalized 
            evaluation and treatment recommendations based on clinical examination.
            """,
            'emergency': """
            The symptoms you've described may require immediate medical attention. Please contact 
            your healthcare provider or emergency services promptly for proper evaluation and care.
            """,
            'general_safety': """
            For pressure injury concerns, it's important to follow evidence-based NPUAP guidelines 
            and work with qualified healthcare professionals. Please consult with your medical team 
            for personalized assessment and treatment planning.
            """
        }
        
        logger.info("MedicalGuardrails initialized with NPUAP compliance")
    
    async def validate_response(self,
                               response_text: str,
                               original_prompt: str,
                               confidence_score: float = None,
                               batman_token: str = None) -> GuardrailResult:
        """Comprehensive validation of medical response"""
        
        start_time = datetime.now()
        violations = []
        
        try:
            # 1. Content Safety Validation
            content_violations = await self._validate_content_safety(response_text)
            violations.extend(content_violations)
            
            # 2. Medical Accuracy Assessment
            medical_accuracy = await self._assess_medical_accuracy(response_text, original_prompt)
            
            # 3. NPUAP Compliance Check
            npuap_compliance = await self._check_npuap_compliance(response_text)
            
            # 4. Confidence Analysis
            confidence_violations = await self._analyze_confidence(response_text, confidence_score)
            violations.extend(confidence_violations)
            
            # 5. Disclaimer Requirements
            disclaimer_violations = await self._check_disclaimers(response_text)
            violations.extend(disclaimer_violations)
            
            # 6. Emergency Handling
            emergency_violations = await self._check_emergency_handling(response_text, original_prompt)
            violations.extend(emergency_violations)
            
            # Determine overall safety level
            safety_level = self._determine_safety_level(violations, medical_accuracy, npuap_compliance)
            
            # Check if escalation is required
            escalation_required = self._needs_escalation(violations, safety_level)
            
            # Generate safe alternative if needed
            safe_alternative = None
            if safety_level in [ResponseSafetyLevel.UNSAFE, ResponseSafetyLevel.BLOCKED]:
                safe_alternative = await self._generate_safe_alternative(violations, original_prompt)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = GuardrailResult(
                safety_level=safety_level,
                confidence_score=confidence_score or 0.0,
                violations=violations,
                medical_accuracy_score=medical_accuracy,
                npuap_compliance_score=npuap_compliance,
                escalation_required=escalation_required,
                safe_alternative=safe_alternative,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            # Audit log
            await self.audit_service.log_activity(
                activity_type="guardrails_validation",
                batman_token=batman_token,
                details={
                    'safety_level': safety_level.value,
                    'violations_count': len(violations),
                    'medical_accuracy': medical_accuracy,
                    'npuap_compliance': npuap_compliance,
                    'escalation_required': escalation_required
                }
            )
            
            logger.info(f"Guardrails validation completed: {safety_level.value} ({len(violations)} violations)")
            
            return result
            
        except Exception as e:
            logger.error(f"Guardrails validation failed: {e}")
            
            # Return safe default in case of error
            return GuardrailResult(
                safety_level=ResponseSafetyLevel.BLOCKED,
                confidence_score=0.0,
                violations=[SafetyViolation(
                    violation_type=SafetyViolationType.HARMFUL_MEDICAL_ADVICE,
                    severity="critical",
                    description="Validation system error",
                    suggested_fix="Use manual review",
                    confidence=1.0,
                    location="system"
                )],
                medical_accuracy_score=0.0,
                npuap_compliance_score=0.0,
                escalation_required=True,
                safe_alternative=self.safe_responses['general_safety'],
                processing_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now()
            )
    
    async def _validate_content_safety(self, response_text: str) -> List[SafetyViolation]:
        """Validate content safety against harmful patterns"""
        
        violations = []
        text_lower = response_text.lower()
        
        # Check for dangerous medical advice
        for pattern in self.harmful_patterns['dangerous_advice']:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                violations.append(SafetyViolation(
                    violation_type=SafetyViolationType.HARMFUL_MEDICAL_ADVICE,
                    severity="critical",
                    description=f"Contains potentially dangerous advice: '{match.group()}'",
                    suggested_fix="Remove dangerous advice and recommend professional consultation",
                    confidence=0.9,
                    location=f"Position {match.start()}-{match.end()}"
                ))
        
        # Check for medical misinformation
        for pattern in self.harmful_patterns['misinformation']:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                violations.append(SafetyViolation(
                    violation_type=SafetyViolationType.MISINFORMATION,
                    severity="high",
                    description=f"Contains medical misinformation: '{match.group()}'",
                    suggested_fix="Correct with evidence-based information",
                    confidence=0.85,
                    location=f"Position {match.start()}-{match.end()}"
                ))
        
        # Check for inappropriate diagnostic claims
        for pattern in self.harmful_patterns['inappropriate_diagnosis']:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                violations.append(SafetyViolation(
                    violation_type=SafetyViolationType.DIAGNOSIS_OVERREACH,
                    severity="high",
                    description=f"Inappropriate diagnostic certainty: '{match.group()}'",
                    suggested_fix="Use cautious language and recommend professional diagnosis",
                    confidence=0.8,
                    location=f"Position {match.start()}-{match.end()}"
                ))
        
        return violations
    
    async def _assess_medical_accuracy(self, response_text: str, original_prompt: str) -> float:
        """Assess medical accuracy using evidence-based validation"""
        
        accuracy_score = 0.8  # Default score
        text_lower = response_text.lower()
        
        # Check NPUAP stage definitions accuracy
        stage_mentions = re.findall(r'stage\s+(\d|u|dti)', text_lower)
        accurate_definitions = 0
        total_definitions = 0
        
        for stage in stage_mentions:
            total_definitions += 1
            if stage in self.npuap_guidelines['stage_definitions']:
                # Check if the description matches guidelines
                definition = self.npuap_guidelines['stage_definitions'][stage].lower()
                if any(keyword in text_lower for keyword in definition.split()[:3]):
                    accurate_definitions += 1
        
        if total_definitions > 0:
            definition_accuracy = accurate_definitions / total_definitions
            accuracy_score = (accuracy_score + definition_accuracy) / 2
        
        # Check for evidence-based risk factors
        mentioned_risk_factors = 0
        for risk_factor in self.npuap_guidelines['risk_factors']:
            if risk_factor in text_lower:
                mentioned_risk_factors += 1
        
        if 'risk' in text_lower or 'factor' in text_lower:
            risk_factor_score = min(1.0, mentioned_risk_factors / 5)  # Normalize to max 1.0
            accuracy_score = (accuracy_score + risk_factor_score) / 2
        
        # Check for evidence-based prevention strategies
        mentioned_strategies = 0
        for strategy in self.npuap_guidelines['prevention_strategies']:
            if any(word in text_lower for word in strategy.split()):
                mentioned_strategies += 1
        
        if 'prevent' in text_lower or 'treatment' in text_lower:
            strategy_score = min(1.0, mentioned_strategies / 4)
            accuracy_score = (accuracy_score + strategy_score) / 2
        
        return round(accuracy_score, 3)
    
    async def _check_npuap_compliance(self, response_text: str) -> float:
        """Check compliance with NPUAP guidelines"""
        
        compliance_score = 0.7  # Base score
        text_lower = response_text.lower()
        
        # Check for appropriate staging language
        if 'stage' in text_lower or 'grade' in text_lower:
            compliance_score += 0.1
        
        # Check for multidisciplinary approach mention
        team_keywords = ['team', 'multidisciplinary', 'healthcare provider', 'physician', 'nurse']
        if any(keyword in text_lower for keyword in team_keywords):
            compliance_score += 0.1
        
        # Check for prevention focus
        prevention_keywords = ['prevention', 'prevent', 'avoid', 'reduce risk']
        if any(keyword in text_lower for keyword in prevention_keywords):
            compliance_score += 0.1
        
        # Check for patient-centered language
        patient_keywords = ['patient', 'individual', 'person', 'client']
        if any(keyword in text_lower for keyword in patient_keywords):
            compliance_score += 0.1
        
        return min(1.0, compliance_score)
    
    async def _analyze_confidence(self, response_text: str, confidence_score: float) -> List[SafetyViolation]:
        """Analyze confidence levels and uncertainty indicators"""
        
        violations = []
        text_lower = response_text.lower()
        
        # Check for uncertainty indicators
        uncertainty_phrases = [
            'not sure', 'uncertain', 'maybe', 'possibly', 'might be',
            'could be', 'perhaps', 'seems like', 'appears to be'
        ]
        
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in text_lower)
        
        if uncertainty_count > 2 or (confidence_score and confidence_score < 0.6):
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.LOW_CONFIDENCE,
                severity="medium",
                description="Response shows high uncertainty or low confidence",
                suggested_fix="Recommend professional consultation for uncertain cases",
                confidence=0.8,
                location="throughout text"
            ))
        
        return violations
    
    async def _check_disclaimers(self, response_text: str) -> List[SafetyViolation]:
        """Check for required medical disclaimers"""
        
        violations = []
        text_lower = response_text.lower()
        
        # Check if any required disclaimer is present
        has_disclaimer = any(disclaimer in text_lower for disclaimer in self.required_disclaimers)
        
        if not has_disclaimer and len(response_text) > 100:  # Only for substantial responses
            violations.append(SafetyViolation(
                violation_type=SafetyViolationType.MISSING_DISCLAIMER,
                severity="medium",
                description="Missing required medical disclaimer",
                suggested_fix="Add disclaimer about consulting healthcare professionals",
                confidence=0.9,
                location="end of response"
            ))
        
        return violations
    
    async def _check_emergency_handling(self, response_text: str, original_prompt: str) -> List[SafetyViolation]:
        """Check proper handling of emergency situations"""
        
        violations = []
        text_lower = response_text.lower()
        prompt_lower = original_prompt.lower()
        
        # Check if prompt contains emergency keywords
        prompt_has_emergency = any(keyword in prompt_lower for keyword in self.emergency_keywords)
        
        if prompt_has_emergency:
            # Emergency response should emphasize urgency
            urgency_phrases = [
                'immediate', 'urgent', 'emergency', 'right away',
                'seek medical attention', 'contact doctor', 'call doctor'
            ]
            
            has_urgency = any(phrase in text_lower for phrase in urgency_phrases)
            
            if not has_urgency:
                violations.append(SafetyViolation(
                    violation_type=SafetyViolationType.EMERGENCY_MISHANDLING,
                    severity="critical",
                    description="Emergency situation not handled with appropriate urgency",
                    suggested_fix="Emphasize immediate medical attention requirement",
                    confidence=0.95,
                    location="overall response"
                ))
        
        return violations
    
    def _determine_safety_level(self,
                               violations: List[SafetyViolation],
                               medical_accuracy: float,
                               npuap_compliance: float) -> ResponseSafetyLevel:
        """Determine overall safety level based on violations and scores"""
        
        # Check for critical violations
        critical_violations = [v for v in violations if v.severity == "critical"]
        if critical_violations:
            return ResponseSafetyLevel.BLOCKED
        
        # Check for high severity violations
        high_violations = [v for v in violations if v.severity == "high"]
        if len(high_violations) >= 2:
            return ResponseSafetyLevel.UNSAFE
        
        # Check overall scores
        if medical_accuracy < 0.6 or npuap_compliance < 0.5:
            return ResponseSafetyLevel.UNSAFE
        
        # Check for medium severity violations
        medium_violations = [v for v in violations if v.severity == "medium"]
        if len(medium_violations) >= 3:
            return ResponseSafetyLevel.NEEDS_REVIEW
        
        # Check for any violations or low scores
        if violations or medical_accuracy < 0.8 or npuap_compliance < 0.7:
            return ResponseSafetyLevel.NEEDS_REVIEW
        
        return ResponseSafetyLevel.SAFE
    
    def _needs_escalation(self,
                         violations: List[SafetyViolation],
                         safety_level: ResponseSafetyLevel) -> bool:
        """Determine if human escalation is required"""
        
        # Always escalate critical violations
        if any(v.severity == "critical" for v in violations):
            return True
        
        # Escalate unsafe responses
        if safety_level == ResponseSafetyLevel.UNSAFE:
            return True
        
        # Escalate emergency mishandling
        if any(v.violation_type == SafetyViolationType.EMERGENCY_MISHANDLING for v in violations):
            return True
        
        return False
    
    async def _generate_safe_alternative(self,
                                        violations: List[SafetyViolation],
                                        original_prompt: str) -> str:
        """Generate safe alternative response"""
        
        # Determine the most appropriate safe response
        violation_types = {v.violation_type for v in violations}
        
        if SafetyViolationType.EMERGENCY_MISHANDLING in violation_types:
            return self.safe_responses['emergency']
        
        if SafetyViolationType.LOW_CONFIDENCE in violation_types:
            return self.safe_responses['low_confidence']
        
        return self.safe_responses['general_safety']
    
    async def get_guardrail_statistics(self) -> Dict[str, Any]:
        """Get guardrail performance statistics"""
        
        # This would query audit logs for statistics
        # For now, returning mock data
        
        return {
            'total_validations': 1250,
            'safe_responses': 1100,
            'needs_review': 120,
            'unsafe_responses': 25,
            'blocked_responses': 5,
            'escalations': 30,
            'common_violations': [
                {'type': 'missing_disclaimer', 'count': 45},
                {'type': 'low_confidence', 'count': 38},
                {'type': 'npuap_noncompliance', 'count': 22}
            ],
            'average_processing_time': 0.15,
            'accuracy_improvement': 0.23
        }


# Singleton instance
medical_guardrails = MedicalGuardrails()

__all__ = [
    'MedicalGuardrails',
    'SafetyViolationType',
    'ResponseSafetyLevel',
    'SafetyViolation',
    'GuardrailResult',
    'medical_guardrails'
]