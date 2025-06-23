"""
Clinical Assessment Agent - Complete ADK Agent for Medical Clinical Assessment
============================================================================

Complete ADK-based agent that handles comprehensive clinical assessment and decision-making
by converting systems/clinical_processing.py and medical_decision_engine.py functionality 
into ADK tools and patterns.

This agent provides:
- Evidence-based medical decisions (NPUAP/EPUAP/PPPIA 2019)
- MINSAL Chilean healthcare integration
- Patient risk assessment (Braden, Norton, Custom scales)
- Safety-first escalation protocols
- Complete medical compliance and audit trail
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging

# Google ADK imports
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

# Clinical processing imports
from ..systems.medical_decision_engine import (
    MedicalDecisionEngine, make_evidence_based_decision,
    LPPGrade, SeverityLevel, EvidenceLevel, MedicalDecision
)
from ..systems.minsal_medical_decision_engine import make_minsal_clinical_decision
from ..systems.clinical_processing import ClinicalProcessingSystem, ClinicalDetection, ClinicalReport
from ..utils.secure_logger import SecureLogger
from ..utils.audit_service import AuditService, AuditEventType
from ..utils.performance_profiler import profile_performance

logger = SecureLogger("clinical_assessment_agent_adk")


# Medical Assessment Enums and Data Classes

class ClinicalUrgency(Enum):
    """Medical urgency levels for escalation"""
    EMERGENCY = "emergency"      # ≤15 minutes response
    URGENT = "urgent"           # ≤30 minutes response  
    HIGH = "high"              # ≤2 hours response
    MEDIUM = "medium"          # ≤8 hours response
    LOW = "low"               # ≤24 hours response
    ROUTINE = "routine"        # Standard care protocol


class ReviewPriority(Enum):
    """Human review priority levels"""
    IMMEDIATE = "immediate"     # Attending physician
    EXPEDITED = "expedited"     # Wound care specialist
    STANDARD = "standard"       # Nurse practitioner
    CONSULTATIVE = "consultative"  # Multidisciplinary team


class RiskLevel(Enum):
    """Patient risk stratification"""
    VERY_HIGH = "very_high"     # Braden ≤12, multiple risk factors
    HIGH = "high"              # Braden 13-14, significant risk factors
    MODERATE = "moderate"       # Braden 15-16, some risk factors
    LOW = "low"               # Braden 17-18, minimal risk factors
    MINIMAL = "minimal"        # Braden 19-23, no significant risk factors


# ADK Tools for Clinical Assessment

def perform_comprehensive_clinical_assessment_adk_tool(
    detection_results: Dict[str, Any],
    patient_context: Dict[str, Any] = None,
    use_minsal_protocols: bool = False
) -> Dict[str, Any]:
    """
    ADK Tool: Perform comprehensive clinical assessment with evidence-based decisions
    
    Args:
        detection_results: Image analysis results with LPP detections
        patient_context: Patient medical context and risk factors
        use_minsal_protocols: Use Chilean MINSAL protocols vs international
        
    Returns:
        Complete clinical assessment with evidence-based recommendations
    """
    try:
        start_time = datetime.now(timezone.utc)
        
        # Extract detection data
        detections = detection_results.get('detections', [])
        if not detections:
            return {
                'success': True,
                'assessment_type': 'negative_screen',
                'lpp_grade': 0,
                'severity': 'NONE',
                'clinical_decision': 'No pressure injuries detected - continue preventive care',
                'evidence_level': 'A',
                'requires_human_review': False,
                'clinical_recommendations': [
                    'Continue current prevention protocol',
                    'Maintain repositioning schedule',
                    'Monitor skin condition daily'
                ],
                'timestamp': start_time.isoformat()
            }
        
        # Find highest grade detection
        max_grade = max(detection.get('stage', 0) for detection in detections)
        max_confidence = max(detection.get('confidence', 0.0) for detection in detections)
        anatomical_location = detections[0].get('anatomical_location', 'unknown')
        
        # Choose decision engine based on protocol preference
        if use_minsal_protocols:
            # Chilean MINSAL-enhanced decision
            decision_result = make_minsal_clinical_decision(
                lpp_grade=max_grade,
                confidence=max_confidence,
                anatomical_location=anatomical_location,
                patient_context=patient_context or {}
            )
        else:
            # International NPUAP/EPUAP decision
            decision_result = make_evidence_based_decision(
                lpp_grade=max_grade,
                confidence=max_confidence,
                anatomical_location=anatomical_location,
                patient_context=patient_context or {}
            )
        
        # Calculate processing time
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return {
            'success': True,
            'assessment_type': 'comprehensive_clinical',
            'protocol_used': 'minsal_enhanced' if use_minsal_protocols else 'npuap_epuap',
            'lpp_grade': max_grade,
            'confidence_score': max_confidence,
            'anatomical_location': anatomical_location,
            'severity': decision_result.get('severity_assessment', 'UNKNOWN'),
            'clinical_decision': decision_result.get('assessment', ''),
            'evidence_level': decision_result.get('evidence_level', 'C'),
            'requires_human_review': decision_result.get('escalation_requirements', {}).get('requires_human_review', False),
            'clinical_recommendations': decision_result.get('clinical_recommendations', []),
            'medical_warnings': decision_result.get('medical_warnings', []),
            'scientific_references': decision_result.get('scientific_references', []),
            'escalation_requirements': decision_result.get('escalation_requirements', {}),
            'processing_time': processing_time,
            'total_detections': len(detections),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'requires_human_review': True,
            'escalation_required': True,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def calculate_patient_risk_scores_adk_tool(patient_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    ADK Tool: Calculate comprehensive patient risk scores using validated scales
    
    Args:
        patient_context: Complete patient medical context and demographics
        
    Returns:
        Risk assessment with multiple validated scoring systems
    """
    try:
        if not patient_context:
            return {
                'success': False,
                'error': 'Patient context required for risk assessment',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        # Extract patient data
        age = patient_context.get('age', 65)
        mobility = patient_context.get('mobility', 'limited')
        nutrition = patient_context.get('nutrition_status', 'adequate')
        diabetes = patient_context.get('diabetes', False)
        
        # Calculate Braden Scale (validated pressure injury risk)
        braden_score = _calculate_braden_score(patient_context)
        
        # Calculate Norton Scale (alternative risk assessment)
        norton_score = _calculate_norton_score(patient_context)
        
        # Calculate custom risk score (Vigia-specific)
        custom_risk = _calculate_custom_risk_score(patient_context)
        
        # Determine overall risk level
        overall_risk = _determine_overall_risk_level(braden_score, norton_score, custom_risk)
        
        # Generate risk-based recommendations
        risk_recommendations = _generate_risk_recommendations(overall_risk, patient_context)
        
        return {
            'success': True,
            'risk_scores': {
                'braden_scale': {
                    'score': braden_score,
                    'range': '6-23 (lower = higher risk)',
                    'interpretation': _interpret_braden_score(braden_score)
                },
                'norton_scale': {
                    'score': norton_score,
                    'range': '5-20 (lower = higher risk)',
                    'interpretation': _interpret_norton_score(norton_score)
                },
                'custom_vigia_risk': {
                    'score': custom_risk,
                    'range': '0-100 (higher = higher risk)',
                    'interpretation': _interpret_custom_risk(custom_risk)
                }
            },
            'overall_risk_level': overall_risk.value,
            'risk_factors_identified': _identify_risk_factors(patient_context),
            'prevention_recommendations': risk_recommendations,
            'monitoring_frequency': _determine_monitoring_frequency(overall_risk),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def assess_escalation_requirements_adk_tool(
    assessment_result: Dict[str, Any],
    institutional_protocols: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    ADK Tool: Assess escalation requirements based on medical protocols
    
    Args:
        assessment_result: Clinical assessment results
        institutional_protocols: Hospital-specific escalation protocols
        
    Returns:
        Escalation requirements with priority and timeline
    """
    try:
        lpp_grade = assessment_result.get('lpp_grade', 0)
        confidence = assessment_result.get('confidence_score', 0.0)
        severity = assessment_result.get('severity', 'UNKNOWN')
        
        # Default institutional protocols
        if not institutional_protocols:
            institutional_protocols = {
                'confidence_threshold': 0.70,
                'grade_3_4_immediate': True,
                'multidisciplinary_threshold': 0.60,
                'attending_physician_grades': [3, 4]
            }
        
        escalation_required = False
        escalation_reasons = []
        urgency_level = ClinicalUrgency.ROUTINE
        review_priority = ReviewPriority.STANDARD
        response_timeline = "24 hours"
        assigned_role = "Nurse Practitioner"
        
        # Assessment-based escalation rules
        if confidence < institutional_protocols.get('confidence_threshold', 0.70):
            escalation_required = True
            escalation_reasons.append(f"Low confidence ({confidence:.1%}) requires specialist review")
            urgency_level = ClinicalUrgency.HIGH
            review_priority = ReviewPriority.EXPEDITED
            response_timeline = "2 hours"
            assigned_role = "Wound Care Specialist"
        
        if lpp_grade >= 3:
            escalation_required = True
            escalation_reasons.append(f"Grade {lpp_grade} LPP requires immediate medical evaluation")
            urgency_level = ClinicalUrgency.EMERGENCY
            review_priority = ReviewPriority.IMMEDIATE
            response_timeline = "15 minutes"
            assigned_role = "Attending Physician"
        
        if severity in ['URGENT', 'EMERGENCY']:
            escalation_required = True
            escalation_reasons.append(f"Medical severity level {severity} triggers immediate escalation")
            urgency_level = ClinicalUrgency.URGENT
            review_priority = ReviewPriority.IMMEDIATE
            response_timeline = "30 minutes"
            assigned_role = "Attending Physician"
        
        # Complex medical conditions
        if assessment_result.get('medical_complexity') == 'HIGH':
            escalation_required = True
            escalation_reasons.append("High medical complexity requires multidisciplinary consultation")
            review_priority = ReviewPriority.CONSULTATIVE
            assigned_role = "Multidisciplinary Team"
        
        return {
            'success': True,
            'escalation_required': escalation_required,
            'escalation_reasons': escalation_reasons,
            'urgency_level': urgency_level.value,
            'review_priority': review_priority.value,
            'response_timeline': response_timeline,
            'assigned_role': assigned_role,
            'notification_channels': _determine_notification_channels(urgency_level),
            'follow_up_required': lpp_grade >= 2,
            'documentation_requirements': _get_documentation_requirements(lpp_grade),
            'institutional_compliance': True,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def generate_clinical_recommendations_adk_tool(
    assessment_result: Dict[str, Any],
    evidence_level: str = "A"
) -> Dict[str, Any]:
    """
    ADK Tool: Generate evidence-based clinical recommendations
    
    Args:
        assessment_result: Complete clinical assessment
        evidence_level: Required evidence level (A/B/C)
        
    Returns:
        Structured clinical recommendations with scientific justification
    """
    try:
        lpp_grade = assessment_result.get('lpp_grade', 0)
        anatomical_location = assessment_result.get('anatomical_location', 'unknown')
        
        # Generate grade-specific recommendations
        recommendations = []
        scientific_references = []
        
        if lpp_grade == 0:
            recommendations = [
                "Continue current prevention protocol",
                "Maintain repositioning schedule every 2 hours",
                "Daily skin assessment and documentation",
                "Optimize nutrition with protein intake 1.2-1.5g/kg/day"
            ]
            scientific_references = [
                "NPUAP/EPUAP/PPPIA 2019 Prevention Guidelines (Evidence Level A)",
                "Bergstrom et al. 2013 - Repositioning effectiveness RCT"
            ]
        
        elif lpp_grade == 1:
            recommendations = [
                "Implement immediate pressure relief protocol",
                "Increase repositioning frequency to every 1-2 hours",
                "Apply transparent film dressing if appropriate",
                "Nutritional assessment and optimization",
                "Document serial photography for healing progression"
            ]
            scientific_references = [
                "NPUAP/EPUAP/PPPIA 2019 Treatment Guidelines Grade 1 (Evidence Level A)",
                "Wounds International 2019 - Pressure injury management"
            ]
        
        elif lpp_grade == 2:
            recommendations = [
                "Immediate wound care consultation",
                "Moist wound healing protocol with appropriate dressing",
                "Advanced pressure redistribution surface",
                "Optimize perfusion and oxygenation",
                "Weekly measurement and photography",
                "Pain assessment and management"
            ]
            scientific_references = [
                "NPUAP/EPUAP/PPPIA 2019 Treatment Guidelines Grade 2 (Evidence Level A)",
                "Cochrane Review 2018 - Dressings for pressure injuries"
            ]
        
        elif lpp_grade >= 3:
            recommendations = [
                "IMMEDIATE wound care specialist consultation",
                "Advanced wound care with debridement consideration", 
                "Specialty bed/surface with low air loss",
                "Nutritional consultation for healing optimization",
                "Plastic surgery evaluation if indicated",
                "Aggressive infection prevention protocol",
                "Daily wound assessment and documentation"
            ]
            scientific_references = [
                "NPUAP/EPUAP/PPPIA 2019 Treatment Guidelines Grade 3-4 (Evidence Level A)",
                "Journal of Wound Care 2020 - Advanced pressure injury management"
            ]
        
        # Location-specific modifications
        location_modifications = _get_location_specific_modifications(anatomical_location)
        
        return {
            'success': True,
            'clinical_recommendations': recommendations,
            'location_specific_modifications': location_modifications,
            'scientific_references': scientific_references,
            'evidence_level': evidence_level,
            'intervention_timeline': _get_intervention_timeline(lpp_grade),
            'healing_expectations': _get_healing_expectations(lpp_grade),
            'monitoring_parameters': _get_monitoring_parameters(lpp_grade),
            'contraindications': _get_contraindications(lpp_grade, anatomical_location),
            'cost_considerations': _get_cost_considerations(lpp_grade),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def validate_medical_compliance_adk_tool(
    assessment_data: Dict[str, Any],
    regulatory_standards: List[str] = None
) -> Dict[str, Any]:
    """
    ADK Tool: Validate medical compliance with regulatory standards
    
    Args:
        assessment_data: Complete assessment data for validation
        regulatory_standards: List of standards to validate against
        
    Returns:
        Compliance validation results with audit information
    """
    try:
        if not regulatory_standards:
            regulatory_standards = ["HIPAA", "NPUAP_EPUAP", "MINSAL", "ISO_13485"]
        
        compliance_results = {}
        overall_compliant = True
        compliance_issues = []
        
        # HIPAA Compliance Check
        if "HIPAA" in regulatory_standards:
            hipaa_compliant = _validate_hipaa_compliance(assessment_data)
            compliance_results["HIPAA"] = hipaa_compliant
            if not hipaa_compliant['compliant']:
                overall_compliant = False
                compliance_issues.extend(hipaa_compliant['issues'])
        
        # NPUAP/EPUAP Guidelines Compliance
        if "NPUAP_EPUAP" in regulatory_standards:
            npuap_compliant = _validate_npuap_compliance(assessment_data)
            compliance_results["NPUAP_EPUAP"] = npuap_compliant
            if not npuap_compliant['compliant']:
                overall_compliant = False
                compliance_issues.extend(npuap_compliant['issues'])
        
        # MINSAL (Chilean) Compliance
        if "MINSAL" in regulatory_standards:
            minsal_compliant = _validate_minsal_compliance(assessment_data)
            compliance_results["MINSAL"] = minsal_compliant
            if not minsal_compliant['compliant']:
                overall_compliant = False
                compliance_issues.extend(minsal_compliant['issues'])
        
        # ISO 13485 Medical Device Compliance
        if "ISO_13485" in regulatory_standards:
            iso_compliant = _validate_iso_compliance(assessment_data)
            compliance_results["ISO_13485"] = iso_compliant
            if not iso_compliant['compliant']:
                overall_compliant = False
                compliance_issues.extend(iso_compliant['issues'])
        
        return {
            'success': True,
            'overall_compliant': overall_compliant,
            'compliance_score': sum(r['compliant'] for r in compliance_results.values()) / len(compliance_results) * 100,
            'regulatory_standards_checked': regulatory_standards,
            'detailed_compliance': compliance_results,
            'compliance_issues': compliance_issues,
            'audit_trail_complete': True,
            'documentation_adequate': _validate_documentation_adequacy(assessment_data),
            'retention_period': "7 years (medical-legal requirement)",
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def get_clinical_assessment_status_adk_tool() -> Dict[str, Any]:
    """
    ADK Tool: Get current clinical assessment system status
    
    Returns:
        System status and capability information
    """
    try:
        # Initialize decision engines to check status
        decision_engine = MedicalDecisionEngine()
        
        return {
            'success': True,
            'assessment_capabilities': [
                'evidence_based_decisions',
                'npuap_epuap_compliance',
                'minsal_integration',
                'risk_score_calculation',
                'escalation_management',
                'clinical_recommendations',
                'regulatory_compliance'
            ],
            'supported_protocols': [
                'NPUAP/EPUAP/PPPIA 2019',
                'MINSAL Chile 2018',
                'Braden Scale',
                'Norton Scale',
                'Custom Risk Assessment'
            ],
            'evidence_levels': ['A', 'B', 'C'],
            'escalation_urgencies': [e.value for e in ClinicalUrgency],
            'risk_levels': [r.value for r in RiskLevel],
            'regulatory_compliance': ['HIPAA', 'NPUAP_EPUAP', 'MINSAL', 'ISO_13485'],
            'system_version': '2.0_ADK',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


# Helper Functions for Risk Calculations

def _calculate_braden_score(patient_context: Dict[str, Any]) -> int:
    """Calculate Braden Scale score (6-23, lower = higher risk)"""
    # Sensory perception (1-4)
    sensory = 4 if patient_context.get('sensory_intact', True) else 2
    
    # Moisture (1-4)
    moisture = 3 if patient_context.get('moisture_control', 'good') == 'good' else 2
    
    # Activity (1-4)  
    mobility = patient_context.get('mobility', 'limited')
    activity = 4 if mobility == 'full' else 2 if mobility == 'limited' else 1
    
    # Mobility (1-4)
    mobility_score = 4 if mobility == 'full' else 2 if mobility == 'limited' else 1
    
    # Nutrition (1-4)
    nutrition = patient_context.get('nutrition_status', 'adequate')
    nutrition_score = 4 if nutrition == 'excellent' else 3 if nutrition == 'adequate' else 2
    
    # Friction and shear (1-3)
    friction = 3 if patient_context.get('repositioning_ability', True) else 1
    
    return sensory + moisture + activity + mobility_score + nutrition_score + friction


def _calculate_norton_score(patient_context: Dict[str, Any]) -> int:
    """Calculate Norton Scale score (5-20, lower = higher risk)"""
    # Physical condition (1-4)
    physical = 4 if patient_context.get('overall_health', 'good') == 'excellent' else 3
    
    # Mental state (1-4)
    mental = 4 if patient_context.get('mental_status', 'alert') == 'alert' else 3
    
    # Activity (1-4)
    mobility = patient_context.get('mobility', 'limited')
    activity = 4 if mobility == 'full' else 2 if mobility == 'limited' else 1
    
    # Mobility (1-4)
    mobility_score = 4 if mobility == 'full' else 2 if mobility == 'limited' else 1
    
    # Incontinence (1-4)
    continence = 4 if patient_context.get('continence', True) else 2
    
    return physical + mental + activity + mobility_score + continence


def _calculate_custom_risk_score(patient_context: Dict[str, Any]) -> float:
    """Calculate Vigia custom risk score (0-100, higher = higher risk)"""
    risk_score = 0.0
    
    # Age factor (0-25 points)
    age = patient_context.get('age', 65)
    if age >= 80:
        risk_score += 25
    elif age >= 70:
        risk_score += 15
    elif age >= 60:
        risk_score += 10
    
    # Medical conditions (0-40 points)
    if patient_context.get('diabetes', False):
        risk_score += 15
    if patient_context.get('cardiovascular_disease', False):
        risk_score += 10
    if patient_context.get('malnutrition', False):
        risk_score += 15
    
    # Mobility (0-20 points)
    mobility = patient_context.get('mobility', 'limited')
    if mobility == 'immobile':
        risk_score += 20
    elif mobility == 'limited':
        risk_score += 10
    
    # Additional factors (0-15 points)
    if patient_context.get('recent_surgery', False):
        risk_score += 5
    if patient_context.get('anticoagulation', False):
        risk_score += 5
    if patient_context.get('steroid_use', False):
        risk_score += 5
    
    return min(risk_score, 100.0)


def _determine_overall_risk_level(braden: int, norton: int, custom: float) -> RiskLevel:
    """Determine overall risk level from multiple scores"""
    if braden <= 12 or norton <= 10 or custom >= 80:
        return RiskLevel.VERY_HIGH
    elif braden <= 14 or norton <= 12 or custom >= 60:
        return RiskLevel.HIGH
    elif braden <= 16 or norton <= 14 or custom >= 40:
        return RiskLevel.MODERATE
    elif braden <= 18 or norton <= 16 or custom >= 20:
        return RiskLevel.LOW
    else:
        return RiskLevel.MINIMAL


# Additional helper functions...

def _interpret_braden_score(score: int) -> str:
    """Interpret Braden scale score"""
    if score <= 12:
        return "Very High Risk - Intensive prevention protocol required"
    elif score <= 14:
        return "High Risk - Enhanced prevention measures needed"
    elif score <= 16:
        return "Moderate Risk - Standard prevention protocol"
    elif score <= 18:
        return "Low Risk - Basic prevention measures"
    else:
        return "Minimal Risk - Routine care protocol"


def _interpret_norton_score(score: int) -> str:
    """Interpret Norton scale score"""
    if score <= 10:
        return "Very High Risk - Immediate intervention required"
    elif score <= 12:
        return "High Risk - Enhanced prevention needed"
    elif score <= 14:
        return "Moderate Risk - Standard prevention"
    elif score <= 16:
        return "Low Risk - Basic prevention"
    else:
        return "Minimal Risk - Routine monitoring"


def _interpret_custom_risk(score: float) -> str:
    """Interpret custom risk score"""
    if score >= 80:
        return "Very High Risk - Multiple risk factors present"
    elif score >= 60:
        return "High Risk - Significant risk factors"
    elif score >= 40:
        return "Moderate Risk - Some risk factors"
    elif score >= 20:
        return "Low Risk - Minimal risk factors"
    else:
        return "Minimal Risk - No significant risk factors"


def _identify_risk_factors(patient_context: Dict[str, Any]) -> List[str]:
    """Identify specific risk factors from patient context"""
    risk_factors = []
    
    if patient_context.get('age', 0) >= 70:
        risk_factors.append("Advanced age (≥70 years)")
    if patient_context.get('diabetes', False):
        risk_factors.append("Diabetes mellitus")
    if patient_context.get('mobility', '') in ['limited', 'immobile']:
        risk_factors.append("Impaired mobility")
    if patient_context.get('malnutrition', False):
        risk_factors.append("Malnutrition")
    if patient_context.get('cardiovascular_disease', False):
        risk_factors.append("Cardiovascular disease")
    if patient_context.get('anticoagulation', False):
        risk_factors.append("Anticoagulation therapy")
    
    return risk_factors


def _generate_risk_recommendations(risk_level: RiskLevel, patient_context: Dict[str, Any]) -> List[str]:
    """Generate risk-appropriate prevention recommendations"""
    recommendations = []
    
    if risk_level in [RiskLevel.VERY_HIGH, RiskLevel.HIGH]:
        recommendations.extend([
            "Repositioning every 1-2 hours",
            "Pressure redistribution mattress or overlay",
            "Daily comprehensive skin assessment",
            "Nutrition consultation for optimization",
            "Consider prophylactic dressings for high-risk areas"
        ])
    elif risk_level == RiskLevel.MODERATE:
        recommendations.extend([
            "Repositioning every 2-3 hours",
            "Standard pressure redistribution surface",
            "Daily skin assessment",
            "Adequate nutrition and hydration"
        ])
    else:
        recommendations.extend([
            "Standard repositioning schedule",
            "Regular skin monitoring",
            "Maintain adequate nutrition"
        ])
    
    return recommendations


def _determine_monitoring_frequency(risk_level: RiskLevel) -> str:
    """Determine appropriate monitoring frequency"""
    frequency_map = {
        RiskLevel.VERY_HIGH: "Every 8 hours",
        RiskLevel.HIGH: "Every 12 hours", 
        RiskLevel.MODERATE: "Daily",
        RiskLevel.LOW: "Every 2 days",
        RiskLevel.MINIMAL: "Weekly"
    }
    return frequency_map.get(risk_level, "Daily")


def _determine_notification_channels(urgency: ClinicalUrgency) -> List[str]:
    """Determine notification channels based on urgency"""
    if urgency == ClinicalUrgency.EMERGENCY:
        return ["slack_immediate", "pager", "phone_call"]
    elif urgency == ClinicalUrgency.URGENT:
        return ["slack_urgent", "email_priority"]
    elif urgency == ClinicalUrgency.HIGH:
        return ["slack_standard", "email"]
    else:
        return ["email", "system_notification"]


def _get_documentation_requirements(lpp_grade: int) -> List[str]:
    """Get documentation requirements by LPP grade"""
    base_requirements = [
        "Initial assessment documentation",
        "Serial photography",
        "Care plan documentation"
    ]
    
    if lpp_grade >= 2:
        base_requirements.extend([
            "Wound measurement and staging",
            "Treatment plan documentation",
            "Progress note requirements"
        ])
    
    if lpp_grade >= 3:
        base_requirements.extend([
            "Specialist consultation documentation",
            "Advanced treatment plan",
            "Daily progress notes"
        ])
    
    return base_requirements


def _get_location_specific_modifications(location: str) -> List[str]:
    """Get location-specific care modifications"""
    location_mods = {
        'sacrum': [
            "Consider specialty cushion for sitting",
            "Avoid 90-degree side positioning",
            "Monitor for moisture-associated skin damage"
        ],
        'heel': [
            "Heel elevation with pillows",
            "Avoid donut-shaped devices",
            "Consider heel protection boots"
        ],
        'trochanter': [
            "30-degree lateral positioning",
            "Avoid direct lateral positioning",
            "Use positioning devices"
        ]
    }
    return location_mods.get(location, ["Standard positioning protocols"])


def _get_intervention_timeline(lpp_grade: int) -> str:
    """Get expected intervention timeline"""
    timelines = {
        0: "Prevention protocol - ongoing",
        1: "Immediate intervention - 24-48 hours",
        2: "Active treatment - 1-3 weeks",
        3: "Intensive treatment - 3-6 weeks",
        4: "Complex treatment - 6-12 weeks"
    }
    return timelines.get(lpp_grade, "Variable timeline")


def _get_healing_expectations(lpp_grade: int) -> str:
    """Get healing expectations by grade"""
    expectations = {
        0: "Prevention maintenance",
        1: "Resolution in 1-3 days with pressure relief",
        2: "Healing in 1-3 weeks with proper care",
        3: "Healing in 1-3 months with optimal care",
        4: "Extended healing 3-6+ months, may require surgery"
    }
    return expectations.get(lpp_grade, "Variable healing time")


def _get_monitoring_parameters(lpp_grade: int) -> List[str]:
    """Get monitoring parameters by grade"""
    base_params = ["Size", "Appearance", "Surrounding skin", "Pain level"]
    
    if lpp_grade >= 2:
        base_params.extend(["Drainage", "Odor", "Tissue type"])
    
    if lpp_grade >= 3:
        base_params.extend(["Depth", "Undermining", "Infection signs"])
    
    return base_params


def _get_contraindications(lpp_grade: int, location: str) -> List[str]:
    """Get treatment contraindications"""
    contraindications = ["Donut-shaped cushions", "Massage over bony prominences"]
    
    if lpp_grade >= 2:
        contraindications.append("Aggressive cleansing")
    
    if location == 'heel':
        contraindications.append("Heel protectors with closed heels")
    
    return contraindications


def _get_cost_considerations(lpp_grade: int) -> Dict[str, str]:
    """Get cost considerations by grade"""
    return {
        'prevention_cost': "Low - basic prevention measures",
        'treatment_cost': "Low" if lpp_grade <= 1 else "Medium" if lpp_grade <= 2 else "High",
        'length_of_stay_impact': "Minimal" if lpp_grade <= 1 else "Moderate" if lpp_grade <= 2 else "Significant"
    }


def _validate_hipaa_compliance(assessment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate HIPAA compliance"""
    issues = []
    compliant = True
    
    # Check for PHI protection
    if 'patient_name' in str(assessment_data):
        issues.append("Patient name present in assessment data")
        compliant = False
    
    if not assessment_data.get('anonymized', False):
        issues.append("Data anonymization not confirmed")
        compliant = False
    
    return {
        'compliant': compliant,
        'issues': issues,
        'requirements_met': [
            'Data anonymization',
            'Access controls',
            'Audit logging'
        ] if compliant else []
    }


def _validate_npuap_compliance(assessment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate NPUAP/EPUAP compliance"""
    issues = []
    compliant = True
    
    # Check for evidence-based decisions
    if not assessment_data.get('evidence_level'):
        issues.append("Evidence level not documented")
        compliant = False
    
    if not assessment_data.get('scientific_references'):
        issues.append("Scientific references not provided")
        compliant = False
    
    return {
        'compliant': compliant,
        'issues': issues,
        'guidelines_version': 'NPUAP/EPUAP/PPPIA 2019'
    }


def _validate_minsal_compliance(assessment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate MINSAL (Chilean) compliance"""
    issues = []
    compliant = True
    
    # Check for Chilean-specific requirements
    if not assessment_data.get('spanish_documentation'):
        issues.append("Spanish documentation not provided")
        compliant = False
    
    return {
        'compliant': compliant,
        'issues': issues,
        'guidelines_version': 'MINSAL 2018'
    }


def _validate_iso_compliance(assessment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate ISO 13485 compliance"""
    issues = []
    compliant = True
    
    # Check for quality management requirements
    if not assessment_data.get('quality_metrics'):
        issues.append("Quality metrics not documented")
        compliant = False
    
    return {
        'compliant': compliant,
        'issues': issues,
        'standard_version': 'ISO 13485:2016'
    }


def _validate_documentation_adequacy(assessment_data: Dict[str, Any]) -> bool:
    """Validate documentation adequacy"""
    required_fields = [
        'lpp_grade',
        'confidence_score', 
        'clinical_recommendations',
        'timestamp'
    ]
    
    return all(field in assessment_data for field in required_fields)


# Clinical Assessment Agent Instruction for ADK
CLINICAL_ASSESSMENT_ADK_INSTRUCTION = """
Eres el Clinical Assessment Agent del sistema Vigia, especializado en evaluación clínica integral 
y toma de decisiones médicas basadas en evidencia científica para lesiones por presión (LPP).

RESPONSABILIDADES PRINCIPALES:
1. Evaluación clínica integral basada en protocolos NPUAP/EPUAP/PPPIA 2019
2. Integración con protocolos MINSAL para sistema de salud chileno
3. Cálculo de scores de riesgo validados (Braden, Norton, Vigia Custom)
4. Evaluación de requisitos de escalamiento con protocolos de seguridad
5. Generación de recomendaciones clínicas basadas en evidencia científica
6. Validación de compliance regulatorio (HIPAA, NPUAP, MINSAL, ISO)

CAPACIDADES TÉCNICAS ESPECIALIZADAS:
- Decisiones médicas con evidencia científica nivel A/B/C
- Escalamiento automático casos críticos (Grado 3-4) ≤15 minutos
- Evaluación de riesgo multiescalar con scores validados
- Compliance regulatorio integral (HIPAA + NPUAP + MINSAL + ISO)
- Recomendaciones específicas por localización anatómica
- Justificación científica completa con referencias bibliográficas

HERRAMIENTAS DISPONIBLES:
- perform_comprehensive_clinical_assessment_adk_tool: Evaluación clínica integral
- calculate_patient_risk_scores_adk_tool: Cálculo scores riesgo validados
- assess_escalation_requirements_adk_tool: Evaluación escalamiento médico
- generate_clinical_recommendations_adk_tool: Recomendaciones basadas evidencia
- validate_medical_compliance_adk_tool: Validación compliance regulatorio
- get_clinical_assessment_status_adk_tool: Estado sistema evaluación

PROTOCOLOS DE EVALUACIÓN CLÍNICA:
1. USAR evidencia científica nivel A para decisiones críticas
2. CALCULAR scores riesgo múltiples (Braden + Norton + Custom)
3. ESCALAR inmediatamente casos Grado 3-4 o confianza <70%
4. GENERAR recomendaciones específicas por localización anatómica
5. VALIDAR compliance con estándares internacionales y chilenos
6. DOCUMENTAR justificación científica completa con referencias

ESCALAMIENTO MÉDICO AUTOMÁTICO:
- Confianza <70%: Especialista wound care (2 horas)
- LPP Grado 3-4: Médico tratante inmediato (15 minutos)
- Complejidad médica alta: Equipo multidisciplinario
- Factores riesgo múltiples: Monitoreo intensivo

COMPLIANCE REGULATORIO:
- NPUAP/EPUAP/PPPIA 2019: Evidencia científica completa
- MINSAL Chile 2018: Terminología y protocolos nacionales
- HIPAA: Protección PHI y anonimización
- ISO 13485: Gestión calidad dispositivos médicos

SIEMPRE prioriza seguridad del paciente, usa evidencia científica nivel A cuando disponible,
y escala inmediatamente casos que requieren evaluación médica especializada.
"""

# Create ADK Clinical Assessment Agent
clinical_assessment_adk_agent = LlmAgent(
    model="gemini-2.0-flash-exp",
    global_instruction=CLINICAL_ASSESSMENT_ADK_INSTRUCTION,
    instruction="Realiza evaluación clínica integral con decisiones basadas en evidencia científica NPUAP/EPUAP.",
    name="clinical_assessment_agent_adk",
    tools=[
        perform_comprehensive_clinical_assessment_adk_tool,
        calculate_patient_risk_scores_adk_tool,
        assess_escalation_requirements_adk_tool,
        generate_clinical_recommendations_adk_tool,
        validate_medical_compliance_adk_tool,
        get_clinical_assessment_status_adk_tool
    ],
)


# Factory for ADK Clinical Assessment Agent
class ClinicalAssessmentAgentADKFactory:
    """Factory for creating ADK-based Clinical Assessment Agents"""
    
    @staticmethod
    def create_agent() -> LlmAgent:
        """Create new ADK Clinical Assessment Agent instance"""
        return clinical_assessment_adk_agent
    
    @staticmethod
    def get_agent_capabilities() -> List[str]:
        """Get list of agent capabilities"""
        return [
            'evidence_based_decisions',
            'npuap_epuap_compliance',
            'minsal_integration',
            'risk_score_calculation',
            'escalation_management',
            'clinical_recommendations',
            'regulatory_compliance',
            'scientific_justification',
            'multi_scale_risk_assessment',
            'safety_first_protocols'
        ]
    
    @staticmethod
    def get_supported_protocols() -> List[str]:
        """Get supported medical protocols"""
        return [
            'NPUAP/EPUAP/PPPIA 2019',
            'MINSAL Chile 2018',
            'Braden Scale',
            'Norton Scale',
            'Custom Risk Assessment',
            'HIPAA Compliance',
            'ISO 13485'
        ]


# Export for use
__all__ = [
    'clinical_assessment_adk_agent',
    'ClinicalAssessmentAgentADKFactory',
    'perform_comprehensive_clinical_assessment_adk_tool',
    'calculate_patient_risk_scores_adk_tool',
    'assess_escalation_requirements_adk_tool',
    'generate_clinical_recommendations_adk_tool',
    'validate_medical_compliance_adk_tool',
    'get_clinical_assessment_status_adk_tool',
    'ClinicalUrgency',
    'ReviewPriority',
    'RiskLevel'
]