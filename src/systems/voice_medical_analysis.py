"""
Voice Medical Analysis Engine
===========================

Advanced medical analysis engine for voice-based patient assessment.
Integrates with Hume AI and Batman tokenization for HIPAA compliance.

Key Features:
- Pain assessment through voice indicators
- Stress and anxiety evaluation
- Mental health screening
- Medical urgency determination
- Batman token integration (NO PHI)

Usage:
    engine = VoiceMedicalAnalysisEngine()
    assessment = engine.analyze_patient_voice(expressions, context, token_id)
"""

import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from vigia_detect.ai.hume_ai_client import VoiceAlertLevel, VoiceMedicalAssessment
from vigia_detect.utils.audit_service import AuditService

logger = logging.getLogger(__name__)


class PainLevel(Enum):
    """Medical pain assessment levels"""
    NONE = "None"
    MILD = "Mild"
    MODERATE = "Moderate"
    SEVERE = "Severe"
    CRITICAL = "Critical"


class StressLevel(Enum):
    """Stress assessment levels"""
    NORMAL = "Normal"
    ELEVATED = "Elevated"
    HIGH = "High"
    SEVERE = "Severe"


class MentalHealthRisk(Enum):
    """Mental health risk levels"""
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


@dataclass
class PainAssessment:
    """Voice-based pain assessment"""
    overall_pain_score: float
    pain_type_scores: Dict[str, float]
    pain_level: str
    pain_descriptors: List[str]
    chronic_pain_indicators: bool
    acute_pain_indicators: bool
    confidence: float


@dataclass
class StressEvaluation:
    """Voice-based stress evaluation"""
    overall_stress_score: float
    stress_type_scores: Dict[str, float]
    stress_level: str
    stress_sources: List[str]
    coping_indicators: List[str]
    intervention_needed: bool
    confidence: float


@dataclass
class MentalHealthScreening:
    """Voice-based mental health screening"""
    screening_scores: Dict[str, Dict[str, float]]
    highest_risk_area: str
    risk_level: str
    requires_professional_evaluation: bool
    immediate_concerns: List[str]
    protective_factors: List[str]
    confidence: float


class VoiceMedicalAnalysisEngine:
    """
    Advanced medical analysis engine for voice-based patient assessment.
    
    Processes voice expressions to provide comprehensive medical insights
    while maintaining HIPAA compliance through Batman tokenization.
    """
    
    def __init__(self):
        self.audit_service = AuditService()
        
        # Pain assessment mapping
        self.pain_expressions = {
            "direct_pain": {
                "Pain": 1.0,
                "Empathic Pain": 0.9,
                "Physical Discomfort": 0.8,
                "Ache": 0.7
            },
            "emotional_pain": {
                "Anguish": 0.9,
                "Distress": 0.8,
                "Suffering": 0.9,
                "Torment": 0.8
            },
            "pain_responses": {
                "Grimace": 0.7,
                "Wince": 0.6,
                "Tension": 0.5,
                "Strain": 0.6
            }
        }
        
        # Stress assessment mapping
        self.stress_expressions = {
            "anxiety": {
                "Anxiety": 1.0,
                "Nervousness": 0.8,
                "Worry": 0.7,
                "Panic": 0.9,
                "Dread": 0.8
            },
            "fear": {
                "Fear": 1.0,
                "Terror": 0.9,
                "Apprehension": 0.6,
                "Unease": 0.5
            },
            "tension": {
                "Tension": 0.8,
                "Stress": 1.0,
                "Pressure": 0.7,
                "Strain": 0.6
            }
        }
        
        # Mental health screening mapping
        self.mental_health_expressions = {
            "depression": {
                "Sadness": 0.8,
                "Despair": 1.0,
                "Hopelessness": 1.0,
                "Melancholy": 0.7,
                "Grief": 0.8,
                "Dejection": 0.7
            },
            "anxiety_disorders": {
                "Anxiety": 1.0,
                "Panic": 1.0,
                "Phobia": 0.9,
                "Obsession": 0.8,
                "Compulsion": 0.8
            },
            "mood_disorders": {
                "Mania": 1.0,
                "Euphoria": 0.7,
                "Irritability": 0.6,
                "Mood Swings": 0.8
            },
            "trauma_indicators": {
                "Trauma": 1.0,
                "Flashback": 0.9,
                "Dissociation": 0.8,
                "Hypervigilance": 0.7
            }
        }
        
        # Positive indicators (protective factors)
        self.positive_expressions = {
            "Joy": 1.0,
            "Contentment": 0.9,
            "Peace": 0.8,
            "Calmness": 0.9,
            "Hope": 0.8,
            "Satisfaction": 0.7,
            "Relief": 0.6
        }
    
    def analyze_patient_voice(
        self,
        expressions: Dict[str, float],
        patient_context: Dict[str, Any],
        token_id: str  # Batman token (NO PHI)
    ) -> VoiceMedicalAssessment:
        """
        Comprehensive medical analysis of patient voice expressions.
        
        Args:
            expressions: Voice expression scores from Hume AI
            patient_context: Medical context (age, conditions, etc.)
            token_id: Batman token ID (tokenized patient identifier)
            
        Returns:
            VoiceMedicalAssessment with comprehensive medical insights
        """
        assessment_id = self._generate_assessment_id(token_id)
        
        try:
            # Perform individual assessments
            pain_assessment = self._assess_pain_indicators(expressions, patient_context)
            stress_evaluation = self._assess_stress_indicators(expressions, patient_context)
            mental_health = self._assess_mental_health(expressions, patient_context)
            
            # Determine overall urgency level
            urgency_level = self._determine_urgency_level(
                pain_assessment, stress_evaluation, mental_health
            )
            
            # Identify primary concerns
            primary_concerns = self._identify_primary_concerns(
                pain_assessment, stress_evaluation, mental_health
            )
            
            # Generate medical recommendations
            recommendations = self._generate_comprehensive_recommendations(
                pain_assessment, stress_evaluation, mental_health, patient_context
            )
            
            # Determine follow-up requirements
            follow_up_required, timeframe, specialist = self._determine_follow_up(
                pain_assessment.__dict__,
                stress_evaluation.__dict__, 
                mental_health.__dict__,
                VoiceAlertLevel(urgency_level)
            )
            
            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(
                pain_assessment, stress_evaluation, mental_health
            )
            
            # Create comprehensive assessment
            assessment = VoiceMedicalAssessment(
                patient_id=token_id,  # Batman token
                assessment_id=assessment_id,
                urgency_level=urgency_level,
                primary_concerns=primary_concerns,
                medical_recommendations=recommendations,
                pain_assessment=pain_assessment.__dict__,
                stress_evaluation=stress_evaluation.__dict__,
                mental_health_screening=mental_health.__dict__,
                follow_up_required=follow_up_required,
                follow_up_timeframe=timeframe,
                specialist_referral=specialist,
                confidence_score=confidence,
                timestamp=datetime.now(),
                audit_trail={
                    "analysis_method": "voice_medical_engine",
                    "tokenization_compliant": True,
                    "phi_protected": True
                }
            )
            
            # Log assessment completion
            self.audit_service.log_event(
                event_type="voice_medical_assessment_complete",
                patient_id=token_id,
                details={
                    "assessment_id": assessment_id,
                    "urgency_level": urgency_level,
                    "primary_concerns_count": len(primary_concerns),
                    "follow_up_required": follow_up_required
                },
                phi_safe=True
            )
            
            return assessment
            
        except Exception as e:
            logger.error(f"Voice medical analysis failed for token {token_id}: {e}")
            raise
    
    def _assess_pain_indicators(
        self,
        expressions: Dict[str, float],
        patient_context: Dict[str, Any]
    ) -> PainAssessment:
        """Assess pain indicators from voice expressions"""
        
        # Calculate pain type scores
        direct_pain_score = self._calculate_expression_score(
            expressions, self.pain_expressions["direct_pain"]
        )
        emotional_pain_score = self._calculate_expression_score(
            expressions, self.pain_expressions["emotional_pain"]
        )
        pain_response_score = self._calculate_expression_score(
            expressions, self.pain_expressions["pain_responses"]
        )
        
        pain_type_scores = {
            "direct_pain": direct_pain_score,
            "emotional_pain": emotional_pain_score,
            "pain_responses": pain_response_score
        }
        
        # Calculate overall pain score
        overall_pain_score = np.mean([
            direct_pain_score * 1.0,  # Direct pain weighted most heavily
            emotional_pain_score * 0.8,
            pain_response_score * 0.6
        ])
        
        # Adjust for patient context
        if patient_context.get("chronic_pain"):
            overall_pain_score *= 1.2  # Increase sensitivity for chronic pain patients
        
        if patient_context.get("age", 0) > 65:
            overall_pain_score *= 1.1  # Increase sensitivity for elderly
        
        # Determine pain level
        pain_level = self._classify_pain_level(overall_pain_score)
        
        # Identify pain descriptors
        pain_descriptors = self._identify_pain_descriptors(expressions)
        
        # Assess chronic vs acute indicators
        chronic_indicators = self._assess_chronic_pain_indicators(
            expressions, patient_context
        )
        acute_indicators = self._assess_acute_pain_indicators(expressions)
        
        # Calculate confidence
        confidence = self._calculate_pain_confidence(expressions, patient_context)
        
        return PainAssessment(
            overall_pain_score=overall_pain_score,
            pain_type_scores=pain_type_scores,
            pain_level=pain_level.value,
            pain_descriptors=pain_descriptors,
            chronic_pain_indicators=chronic_indicators,
            acute_pain_indicators=acute_indicators,
            confidence=confidence
        )
    
    def _assess_stress_indicators(
        self,
        expressions: Dict[str, float],
        patient_context: Dict[str, Any]
    ) -> StressEvaluation:
        """Assess stress indicators from voice expressions"""
        
        # Calculate stress type scores
        anxiety_score = self._calculate_expression_score(
            expressions, self.stress_expressions["anxiety"]
        )
        fear_score = self._calculate_expression_score(
            expressions, self.stress_expressions["fear"]
        )
        tension_score = self._calculate_expression_score(
            expressions, self.stress_expressions["tension"]
        )
        
        stress_type_scores = {
            "anxiety": anxiety_score,
            "fear": fear_score,
            "tension": tension_score
        }
        
        # Calculate overall stress score
        overall_stress_score = np.mean([anxiety_score, fear_score, tension_score])
        
        # Adjust for patient context
        if patient_context.get("anxiety_disorder"):
            overall_stress_score *= 1.3
        
        if patient_context.get("recent_trauma"):
            overall_stress_score *= 1.2
        
        # Determine stress level
        stress_level = self._classify_stress_level(overall_stress_score)
        
        # Identify stress sources and coping indicators
        stress_sources = self._identify_stress_sources(expressions)
        coping_indicators = self._identify_coping_indicators(expressions)
        
        # Determine if intervention is needed
        intervention_needed = overall_stress_score >= 0.6 or stress_level in [
            StressLevel.HIGH, StressLevel.SEVERE
        ]
        
        # Calculate confidence
        confidence = self._calculate_stress_confidence(expressions, patient_context)
        
        return StressEvaluation(
            overall_stress_score=overall_stress_score,
            stress_type_scores=stress_type_scores,
            stress_level=stress_level.value,
            stress_sources=stress_sources,
            coping_indicators=coping_indicators,
            intervention_needed=intervention_needed,
            confidence=confidence
        )
    
    def _assess_mental_health(
        self,
        expressions: Dict[str, float],
        patient_context: Dict[str, Any]
    ) -> MentalHealthScreening:
        """Assess mental health indicators from voice expressions"""
        
        screening_scores = {}
        
        # Screen each mental health domain
        for domain, expression_map in self.mental_health_expressions.items():
            domain_score = self._calculate_expression_score(expressions, expression_map)
            
            # Additional analysis for each domain
            if domain == "depression":
                severity = self._assess_depression_severity(expressions)
                screening_scores[domain] = {
                    "score": domain_score,
                    "severity": severity,
                    "risk_level": self._classify_depression_risk(domain_score)
                }
            elif domain == "anxiety_disorders":
                anxiety_type = self._identify_anxiety_type(expressions)
                screening_scores[domain] = {
                    "score": domain_score,
                    "type": anxiety_type,
                    "risk_level": self._classify_anxiety_risk(domain_score)
                }
            else:
                screening_scores[domain] = {
                    "score": domain_score,
                    "risk_level": self._classify_general_risk(domain_score)
                }
        
        # Identify highest risk area
        highest_risk_area = max(
            screening_scores.keys(),
            key=lambda domain: screening_scores[domain]["score"]
        )
        
        # Determine overall risk level
        max_score = max(scores["score"] for scores in screening_scores.values())
        risk_level = self._classify_mental_health_risk(max_score).value
        
        # Determine if professional evaluation is required
        requires_evaluation = (
            max_score >= 0.6 or
            any(scores["score"] >= 0.7 for scores in screening_scores.values())
        )
        
        # Identify immediate concerns
        immediate_concerns = self._identify_immediate_mental_health_concerns(
            expressions, screening_scores
        )
        
        # Identify protective factors
        protective_factors = self._identify_protective_factors(expressions)
        
        # Calculate confidence
        confidence = self._calculate_mental_health_confidence(expressions, patient_context)
        
        return MentalHealthScreening(
            screening_scores=screening_scores,
            highest_risk_area=highest_risk_area,
            risk_level=risk_level,
            requires_professional_evaluation=requires_evaluation,
            immediate_concerns=immediate_concerns,
            protective_factors=protective_factors,
            confidence=confidence
        )
    
    def _calculate_expression_score(
        self,
        expressions: Dict[str, float],
        expression_weights: Dict[str, float]
    ) -> float:
        """Calculate weighted score for expression category"""
        total_score = 0.0
        total_weight = 0.0
        
        for expression, weight in expression_weights.items():
            if expression in expressions:
                score = expressions[expression]
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_urgency_level(
        self,
        pain_assessment: PainAssessment,
        stress_evaluation: StressEvaluation,
        mental_health: MentalHealthScreening
    ) -> str:
        """Determine overall medical urgency level"""
        
        # Calculate composite urgency score
        pain_urgency = self._map_pain_to_urgency(pain_assessment.overall_pain_score)
        stress_urgency = self._map_stress_to_urgency(stress_evaluation.overall_stress_score)
        mental_urgency = self._map_mental_health_to_urgency(
            max(scores["score"] for scores in mental_health.screening_scores.values())
        )
        
        max_urgency = max(pain_urgency, stress_urgency, mental_urgency)
        
        # Determine urgency level
        if max_urgency >= 0.8:
            return VoiceAlertLevel.CRITICAL.value
        elif max_urgency >= 0.6:
            return VoiceAlertLevel.HIGH.value
        elif max_urgency >= 0.4:
            return VoiceAlertLevel.ELEVATED.value
        else:
            return VoiceAlertLevel.NORMAL.value
    
    def _identify_primary_concerns(
        self,
        pain_assessment: PainAssessment,
        stress_evaluation: StressEvaluation,
        mental_health: MentalHealthScreening
    ) -> List[str]:
        """Identify primary medical concerns"""
        concerns = []
        
        # Pain concerns
        if pain_assessment.overall_pain_score >= 0.7:
            concerns.append(f"Severe pain indicators detected ({pain_assessment.pain_level})")
        elif pain_assessment.overall_pain_score >= 0.5:
            concerns.append(f"Moderate pain levels indicated ({pain_assessment.pain_level})")
        
        # Stress concerns
        if stress_evaluation.overall_stress_score >= 0.7:
            concerns.append(f"High stress levels detected ({stress_evaluation.stress_level})")
        elif stress_evaluation.intervention_needed:
            concerns.append("Stress intervention recommended")
        
        # Mental health concerns
        if mental_health.requires_professional_evaluation:
            concerns.append(f"Mental health evaluation needed ({mental_health.highest_risk_area})")
        
        for concern in mental_health.immediate_concerns:
            concerns.append(concern)
        
        return concerns[:5]  # Limit to top 5 concerns
    
    def _generate_comprehensive_recommendations(
        self,
        pain_assessment: PainAssessment,
        stress_evaluation: StressEvaluation,
        mental_health: MentalHealthScreening,
        patient_context: Dict[str, Any]
    ) -> List[str]:
        """Generate comprehensive medical recommendations"""
        recommendations = []
        
        # Pain management recommendations
        if pain_assessment.overall_pain_score >= 0.7:
            recommendations.append("Immediate pain assessment and intervention required")
            if pain_assessment.chronic_pain_indicators:
                recommendations.append("Review chronic pain management protocol")
            if pain_assessment.acute_pain_indicators:
                recommendations.append("Investigate acute pain source")
        
        # Stress management recommendations
        if stress_evaluation.overall_stress_score >= 0.6:
            recommendations.append("Stress reduction intervention recommended")
            if "anxiety" in stress_evaluation.stress_sources:
                recommendations.append("Consider anxiety management techniques")
        
        # Mental health recommendations
        if mental_health.requires_professional_evaluation:
            recommendations.append("Professional mental health evaluation recommended")
        
        # Contextual recommendations
        if patient_context.get("age", 0) > 65:
            recommendations.append("Age-appropriate assessment protocols recommended")
        
        if patient_context.get("chronic_conditions"):
            recommendations.append("Consider chronic condition impact on current symptoms")
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def _determine_follow_up(
        self,
        pain_data: Dict[str, Any],
        stress_data: Dict[str, Any],
        mental_health_data: Dict[str, Any],
        urgency_level: VoiceAlertLevel
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Determine follow-up requirements"""
        
        if urgency_level == VoiceAlertLevel.CRITICAL:
            return True, "Immediate", "Emergency Medicine"
        elif urgency_level == VoiceAlertLevel.HIGH:
            return True, "Within 24 hours", "Primary Care Physician"
        elif urgency_level == VoiceAlertLevel.ELEVATED:
            return True, "Within 1 week", "Healthcare Provider"
        else:
            # Check specific conditions that might require follow-up
            if mental_health_data.get("requires_professional_evaluation"):
                return True, "Within 2 weeks", "Mental Health Professional"
            elif pain_data.get("overall_pain_score", 0) >= 0.5:
                return True, "Within 1 week", "Primary Care Physician"
            else:
                return False, None, None
    
    # Helper methods for classification and analysis
    def _classify_pain_level(self, score: float) -> PainLevel:
        """Classify pain level from score"""
        if score >= 0.8:
            return PainLevel.CRITICAL
        elif score >= 0.6:
            return PainLevel.SEVERE
        elif score >= 0.4:
            return PainLevel.MODERATE
        elif score >= 0.2:
            return PainLevel.MILD
        else:
            return PainLevel.NONE
    
    def _classify_stress_level(self, score: float) -> StressLevel:
        """Classify stress level from score"""
        if score >= 0.8:
            return StressLevel.SEVERE
        elif score >= 0.6:
            return StressLevel.HIGH
        elif score >= 0.4:
            return StressLevel.ELEVATED
        else:
            return StressLevel.NORMAL
    
    def _classify_mental_health_risk(self, score: float) -> MentalHealthRisk:
        """Classify mental health risk from score"""
        if score >= 0.8:
            return MentalHealthRisk.CRITICAL
        elif score >= 0.6:
            return MentalHealthRisk.HIGH
        elif score >= 0.4:
            return MentalHealthRisk.MODERATE
        else:
            return MentalHealthRisk.LOW
    
    def _calculate_overall_confidence(
        self,
        pain_assessment: PainAssessment,
        stress_evaluation: StressEvaluation,
        mental_health: MentalHealthScreening
    ) -> float:
        """Calculate overall confidence in assessment"""
        confidences = [
            pain_assessment.confidence,
            stress_evaluation.confidence,
            mental_health.confidence
        ]
        return np.mean(confidences)
    
    def _generate_assessment_id(self, token_id: str) -> str:
        """Generate unique assessment ID"""
        import hashlib
        timestamp = datetime.now().isoformat()
        data = f"voice_assessment_{token_id}_{timestamp}"
        return f"va_{hashlib.md5(data.encode()).hexdigest()[:12]}"
    
    # Additional helper methods (simplified for brevity)
    def _identify_pain_descriptors(self, expressions: Dict[str, float]) -> List[str]:
        """Identify pain descriptors from expressions"""
        descriptors = []
        if expressions.get("Anguish", 0) > 0.5:
            descriptors.append("anguish")
        if expressions.get("Distress", 0) > 0.5:
            descriptors.append("distress")
        return descriptors
    
    def _assess_chronic_pain_indicators(self, expressions: Dict[str, float], context: Dict) -> bool:
        """Assess chronic pain indicators"""
        return context.get("chronic_pain", False) or expressions.get("Tiredness", 0) > 0.6
    
    def _assess_acute_pain_indicators(self, expressions: Dict[str, float]) -> bool:
        """Assess acute pain indicators"""
        return expressions.get("Pain", 0) > 0.7 or expressions.get("Empathic Pain", 0) > 0.7
    
    def _calculate_pain_confidence(self, expressions: Dict, context: Dict) -> float:
        """Calculate pain assessment confidence"""
        base_confidence = 0.7
        if context.get("chronic_pain"):
            base_confidence += 0.1
        if len([e for e in expressions if "pain" in e.lower()]) > 2:
            base_confidence += 0.1
        return min(base_confidence, 0.95)
    
    def _calculate_stress_confidence(self, expressions: Dict, context: Dict) -> float:
        """Calculate stress assessment confidence"""
        return 0.8  # Simplified
    
    def _calculate_mental_health_confidence(self, expressions: Dict, context: Dict) -> float:
        """Calculate mental health assessment confidence"""
        return 0.75  # Simplified
    
    def _identify_stress_sources(self, expressions: Dict[str, float]) -> List[str]:
        """Identify stress sources"""
        sources = []
        if expressions.get("Anxiety", 0) > 0.6:
            sources.append("anxiety")
        if expressions.get("Fear", 0) > 0.6:
            sources.append("fear")
        return sources
    
    def _identify_coping_indicators(self, expressions: Dict[str, float]) -> List[str]:
        """Identify coping indicators"""
        indicators = []
        if expressions.get("Calmness", 0) > 0.4:
            indicators.append("calmness")
        return indicators
    
    def _assess_depression_severity(self, expressions: Dict[str, float]) -> str:
        """Assess depression severity"""
        sadness_score = expressions.get("Sadness", 0)
        if sadness_score >= 0.8:
            return "severe"
        elif sadness_score >= 0.6:
            return "moderate"
        elif sadness_score >= 0.4:
            return "mild"
        else:
            return "none"
    
    def _classify_depression_risk(self, score: float) -> str:
        """Classify depression risk"""
        if score >= 0.7:
            return "high"
        elif score >= 0.5:
            return "moderate"
        else:
            return "low"
    
    def _identify_anxiety_type(self, expressions: Dict[str, float]) -> str:
        """Identify anxiety type"""
        if expressions.get("Panic", 0) > 0.7:
            return "panic"
        elif expressions.get("Fear", 0) > 0.7:
            return "phobic"
        else:
            return "generalized"
    
    def _classify_anxiety_risk(self, score: float) -> str:
        """Classify anxiety risk"""
        if score >= 0.7:
            return "high"
        elif score >= 0.5:
            return "moderate"
        else:
            return "low"
    
    def _classify_general_risk(self, score: float) -> str:
        """Classify general mental health risk"""
        if score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "moderate"
        else:
            return "low"
    
    def _identify_immediate_mental_health_concerns(self, expressions: Dict, scores: Dict) -> List[str]:
        """Identify immediate mental health concerns"""
        concerns = []
        if expressions.get("Despair", 0) > 0.8:
            concerns.append("Severe despair indicators")
        if expressions.get("Hopelessness", 0) > 0.8:
            concerns.append("Hopelessness indicators")
        return concerns
    
    def _identify_protective_factors(self, expressions: Dict[str, float]) -> List[str]:
        """Identify mental health protective factors"""
        factors = []
        for expression, score in expressions.items():
            if expression in self.positive_expressions and score > 0.5:
                factors.append(expression.lower())
        return factors
    
    def _map_pain_to_urgency(self, pain_score: float) -> float:
        """Map pain score to urgency"""
        return pain_score
    
    def _map_stress_to_urgency(self, stress_score: float) -> float:
        """Map stress score to urgency"""
        return stress_score * 0.8  # Slightly lower weight than pain
    
    def _map_mental_health_to_urgency(self, mental_score: float) -> float:
        """Map mental health score to urgency"""
        return mental_score * 0.9  # High weight for mental health