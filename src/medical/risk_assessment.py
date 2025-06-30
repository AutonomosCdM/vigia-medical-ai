"""
Medical Risk Assessment Module - VIGIA Medical AI
================================================

Consolidated risk assessment logic implementing medical-grade scales:
- Braden Scale for pressure injury risk (international standard)
- Norton Scale for pressure sore risk assessment
- Medical risk factor analysis with NPUAP guidelines

This module centralizes all risk assessment logic previously scattered
across multiple agents and systems.
"""

from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Medical risk levels for pressure injury"""
    VERY_LOW = "very_low"      # 19-23 Braden
    LOW = "low"               # 15-18 Braden  
    MODERATE = "moderate"     # 13-14 Braden
    HIGH = "high"            # 10-12 Braden
    VERY_HIGH = "very_high"  # 6-9 Braden


class RiskScale(Enum):
    """Medical risk assessment scales"""
    BRADEN = "braden"
    NORTON = "norton"
    COMBINED = "combined"


@dataclass
class RiskAssessmentResult:
    """Medical risk assessment result with clinical interpretation"""
    braden_score: int
    norton_score: int
    risk_level: RiskLevel
    risk_percentage: float
    risk_factors: List[str]
    clinical_recommendations: List[str]
    escalation_required: bool
    assessment_confidence: float
    scale_used: RiskScale


class MedicalRiskAssessment:
    """
    Comprehensive medical risk assessment using validated clinical scales.
    
    Implements:
    - Braden Scale (gold standard, 6-23 points)
    - Norton Scale (classic assessment, 5-20 points)
    - Combined clinical interpretation
    - NPUAP-aligned risk factor analysis
    """
    
    def __init__(self) -> None:
        """Initialize medical risk assessment system."""
        self.logger: logging.Logger = logging.getLogger(__name__)
    
    def assess_pressure_injury_risk(self, patient_context: Dict[str, Any]) -> RiskAssessmentResult:
        """
        Comprehensive pressure injury risk assessment.
        
        Args:
            patient_context: Patient clinical data including demographics and risk factors
            
        Returns:
            Complete risk assessment with clinical recommendations
        """
        # Calculate both validated scales
        braden_score = self._calculate_braden_score(patient_context)
        norton_score = self._calculate_norton_score(patient_context)
        
        # Determine risk level (primarily based on Braden as international standard)
        risk_level = self._determine_risk_level(braden_score)
        
        # Calculate risk percentage and factors
        risk_percentage = self._calculate_risk_percentage(braden_score, norton_score)
        risk_factors = self._identify_key_risk_factors(patient_context)
        
        # Generate clinical recommendations
        recommendations = self._generate_clinical_recommendations(
            braden_score, norton_score, risk_factors, patient_context
        )
        
        # Determine if escalation required
        escalation_required = self._requires_escalation(braden_score, risk_factors)
        
        # Calculate assessment confidence
        confidence = self._calculate_assessment_confidence(patient_context, risk_factors)
        
        return RiskAssessmentResult(
            braden_score=braden_score,
            norton_score=norton_score,
            risk_level=risk_level,
            risk_percentage=risk_percentage,
            risk_factors=risk_factors,
            clinical_recommendations=recommendations,
            escalation_required=escalation_required,
            assessment_confidence=confidence,
            scale_used=RiskScale.COMBINED
        )
    
    def _calculate_braden_score(self, patient_context: Dict[str, Any]) -> int:
        """
        Calculate Braden Scale score for pressure injury risk (6-23 scale).
        
        Braden Scale components:
        1. Sensory perception (1-4)
        2. Moisture (1-4) 
        3. Activity (1-4)
        4. Mobility (1-4)
        5. Nutrition (1-4)
        6. Friction and shear (1-3)
        
        Total: 6-23 (lower scores = higher risk)
        """
        score = 23  # Start with maximum (lowest risk)
        
        risk_factors = patient_context.get("risk_factors", {})
        age_range = patient_context.get("age_range", "40-50")
        
        # Extract age for calculation
        age = self._extract_age(age_range)
        
        # 1. Sensory perception (1-4)
        if risk_factors.get("neurological_impairment") or risk_factors.get("altered_mental_state"):
            score -= 2  # Severely limited or completely limited
        elif risk_factors.get("diabetes") and patient_context.get("glucose_control") == "poor":
            score -= 1  # Slightly limited
        elif age > 80:
            score -= 1  # Age-related sensory decline
        
        # 2. Moisture (1-4)
        if risk_factors.get("incontinence"):
            score -= 3  # Constantly moist
        elif risk_factors.get("moisture_exposure") or risk_factors.get("diaphoresis"):
            score -= 2  # Often moist
        elif risk_factors.get("occasional_moisture"):
            score -= 1  # Occasionally moist
        
        # 3. Activity (1-4)
        if risk_factors.get("bedbound") or risk_factors.get("immobility"):
            score -= 3  # Bedfast
        elif risk_factors.get("limited_mobility") or risk_factors.get("wheelchair_bound"):
            score -= 2  # Chairfast
        elif age > 75 or risk_factors.get("reduced_activity"):
            score -= 1  # Walks occasionally
        
        # 4. Mobility (1-4)
        if risk_factors.get("complete_immobility") or risk_factors.get("paralysis"):
            score -= 3  # Completely immobile
        elif risk_factors.get("very_limited_mobility"):
            score -= 2  # Very limited
        elif risk_factors.get("slightly_limited_mobility") or age > 70:
            score -= 1  # Slightly limited
        
        # 5. Nutrition (1-4)
        if risk_factors.get("severe_malnutrition") or patient_context.get("albumin", 3.5) < 2.5:
            score -= 3  # Very poor nutrition
        elif risk_factors.get("malnutrition") or risk_factors.get("poor_nutrition"):
            score -= 2  # Probably inadequate
        elif risk_factors.get("diabetes") or patient_context.get("albumin", 3.5) < 3.0:
            score -= 1  # Probably adequate
        
        # 6. Friction and shear (1-3)
        if risk_factors.get("friction_shear") or risk_factors.get("sliding_in_bed"):
            score -= 2  # Problem
        elif risk_factors.get("occasional_sliding") or risk_factors.get("assistance_required"):
            score -= 1  # Potential problem
        
        return max(6, min(23, score))  # Constrain to valid Braden range
    
    def _calculate_norton_score(self, patient_context: Dict[str, Any]) -> int:
        """
        Calculate Norton Scale score (5-20 scale).
        
        Norton Scale components:
        1. Physical condition (1-4)
        2. Mental state (1-4)
        3. Activity (1-4)
        4. Mobility (1-4)
        5. Incontinence (1-4)
        
        Total: 5-20 (lower scores = higher risk)
        """
        score = 20  # Start with maximum (lowest risk)
        
        risk_factors = patient_context.get("risk_factors", {})
        age = self._extract_age(patient_context.get("age_range", "40-50"))
        
        # 1. Physical condition (1-4)
        if risk_factors.get("severe_illness") or risk_factors.get("terminal_illness"):
            score -= 3  # Very poor
        elif risk_factors.get("chronic_illness") or risk_factors.get("malnutrition"):
            score -= 2  # Poor
        elif age > 75 or risk_factors.get("mild_illness"):
            score -= 1  # Fair
        
        # 2. Mental state (1-4)
        if risk_factors.get("comatose") or risk_factors.get("severe_dementia"):
            score -= 3  # Stuporous
        elif risk_factors.get("confusion") or risk_factors.get("altered_mental_state"):
            score -= 2  # Confused
        elif risk_factors.get("mild_confusion") or age > 80:
            score -= 1  # Apathetic
        
        # 3. Activity (1-4)
        if risk_factors.get("bedbound"):
            score -= 3  # Bedfast
        elif risk_factors.get("wheelchair_bound"):
            score -= 2  # Chairfast
        elif risk_factors.get("limited_activity"):
            score -= 1  # Walks with help
        
        # 4. Mobility (1-4)
        if risk_factors.get("immobile") or risk_factors.get("paralysis"):
            score -= 3  # Immobile
        elif risk_factors.get("very_limited_mobility"):
            score -= 2  # Very limited
        elif risk_factors.get("limited_mobility"):
            score -= 1  # Slightly limited
        
        # 5. Incontinence (1-4)
        if risk_factors.get("double_incontinence"):
            score -= 3  # Doubly incontinent
        elif risk_factors.get("incontinence"):
            score -= 2  # Usually incontinent (urine or feces)
        elif risk_factors.get("occasional_incontinence"):
            score -= 1  # Occasionally incontinent
        
        return max(5, min(20, score))  # Constrain to valid Norton range
    
    def _determine_risk_level(self, braden_score: int) -> RiskLevel:
        """Determine risk level based on Braden score (international standard)."""
        if braden_score >= 19:
            return RiskLevel.VERY_LOW
        elif braden_score >= 15:
            return RiskLevel.LOW
        elif braden_score >= 13:
            return RiskLevel.MODERATE
        elif braden_score >= 10:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    def _calculate_risk_percentage(self, braden_score: int, norton_score: int) -> float:
        """Calculate risk percentage based on combined scales."""
        # Convert Braden score to percentage (inverse relationship)
        braden_risk = ((23 - braden_score) / 17) * 100
        
        # Convert Norton score to percentage (inverse relationship)
        norton_risk = ((20 - norton_score) / 15) * 100
        
        # Weight Braden more heavily as it's the international standard
        combined_risk = (braden_risk * 0.7) + (norton_risk * 0.3)
        
        return min(100.0, max(0.0, combined_risk))
    
    def _identify_key_risk_factors(self, patient_context: Dict[str, Any]) -> List[str]:
        """Identify key medical risk factors present."""
        risk_factors = patient_context.get("risk_factors", {})
        identified_factors = []
        
        # High-priority medical risk factors
        high_priority_factors = [
            ("immobility", "Patient immobility/bedbound status"),
            ("incontinence", "Urinary or fecal incontinence"),
            ("malnutrition", "Nutritional deficiency"),
            ("diabetes", "Diabetes mellitus with poor glycemic control"),
            ("neurological_impairment", "Neurological impairment affecting sensation"),
            ("friction_shear", "Friction and shear forces"),
            ("altered_mental_state", "Altered mental status/confusion"),
            ("chronic_illness", "Chronic medical conditions")
        ]
        
        for factor_key, factor_description in high_priority_factors:
            if risk_factors.get(factor_key):
                identified_factors.append(factor_description)
        
        # Age-related factors
        age = self._extract_age(patient_context.get("age_range", "40-50"))
        if age > 75:
            identified_factors.append("Advanced age (>75 years)")
        
        return identified_factors
    
    def _generate_clinical_recommendations(self, braden_score: int, norton_score: int, 
                                         risk_factors: List[str], 
                                         patient_context: Dict[str, Any]) -> List[str]:
        """Generate evidence-based clinical recommendations."""
        recommendations = []
        
        # Risk-level specific recommendations
        if braden_score <= 12:  # High/Very High Risk
            recommendations.extend([
                "Implement pressure redistribution surfaces (specialty mattress/cushion)",
                "Reposition every 1-2 hours with positioning schedule",
                "Skin assessment every 8 hours with documentation",
                "Nutritional consultation for protein/calorie optimization",
                "Consider pressure mapping assessment"
            ])
        elif braden_score <= 18:  # Moderate Risk
            recommendations.extend([
                "Reposition every 2-4 hours as tolerated",
                "Daily comprehensive skin assessment",
                "Optimize nutrition and hydration status",
                "Use pressure-reducing surfaces as appropriate"
            ])
        else:  # Low Risk
            recommendations.extend([
                "Standard repositioning protocol",
                "Daily skin assessment during care",
                "Maintain optimal nutrition and mobility"
            ])
        
        # Factor-specific recommendations
        if "incontinence" in str(risk_factors).lower():
            recommendations.append("Implement incontinence management protocol with skin protection")
        
        if "nutrition" in str(risk_factors).lower():
            recommendations.append("Dietitian consultation for nutritional optimization")
        
        if "diabetes" in str(risk_factors).lower():
            recommendations.append("Optimize glycemic control and monitor for diabetic complications")
        
        return recommendations
    
    def _requires_escalation(self, braden_score: int, risk_factors: List[str]) -> bool:
        """Determine if immediate medical escalation is required."""
        # Very high risk requires immediate attention
        if braden_score <= 9:
            return True
        
        # Multiple high-priority risk factors
        high_priority_count = sum(1 for factor in risk_factors if any(
            priority in factor.lower() for priority in 
            ["immobility", "incontinence", "malnutrition", "neurological"]
        ))
        
        if high_priority_count >= 3:
            return True
        
        return False
    
    def _calculate_assessment_confidence(self, patient_context: Dict[str, Any], 
                                       risk_factors: List[str]) -> float:
        """Calculate confidence in risk assessment based on data completeness."""
        confidence = 100.0
        
        # Reduce confidence for missing data
        required_fields = ["age_range", "risk_factors"]
        for field in required_fields:
            if not patient_context.get(field):
                confidence -= 20.0
        
        # Reduce confidence if very few risk factors identified
        if len(risk_factors) < 2:
            confidence -= 15.0
        
        # Reduce confidence if patient context is minimal
        if len(patient_context) < 3:
            confidence -= 10.0
        
        return max(0.0, min(100.0, confidence))
    
    def _extract_age(self, age_range: str) -> int:
        """Extract age from age range string for calculations."""
        try:
            if "-" in age_range:
                return int(age_range.split("-")[0])
            return int(age_range)
        except (ValueError, AttributeError):
            return 65  # Default middle-age assumption


# Utility functions for backward compatibility and easy access
def assess_patient_risk(patient_context: Dict[str, Any]) -> RiskAssessmentResult:
    """Quick access function for patient risk assessment."""
    assessor = MedicalRiskAssessment()
    return assessor.assess_pressure_injury_risk(patient_context)


def calculate_braden_score(patient_context: Dict[str, Any]) -> int:
    """Quick access function for Braden score calculation."""
    assessor = MedicalRiskAssessment()
    return assessor._calculate_braden_score(patient_context)


def calculate_norton_score(patient_context: Dict[str, Any]) -> int:
    """Quick access function for Norton score calculation."""
    assessor = MedicalRiskAssessment()
    return assessor._calculate_norton_score(patient_context)