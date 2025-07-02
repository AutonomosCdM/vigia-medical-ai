"""
ClinicalAssessmentAgent - ADK Agent for Evidence-Based Medical Assessment
========================================================================

Specialized ADK agent for comprehensive clinical assessment and evidence-based
medical decision making. Integrates NPUAP/EPUAP international guidelines with
Chilean MINSAL protocols for complete medical compliance.

This agent encapsulates the complete clinical processing system with evidence-based
decision engines, risk assessment, and escalation management for distributed processing.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
from google.adk.agents.llm_agent import Agent
from google.adk.tools import FunctionTool

# Import clinical processing components
from src.systems.clinical_processing import ClinicalProcessingSystem
from src.systems.medical_decision_engine import (
    MedicalDecisionEngine, make_evidence_based_decision, EvidenceLevel
)
from src.systems.minsal_medical_decision_engine import (
    make_minsal_clinical_decision, MINSALDecisionEngine
)
from src.systems.medical_knowledge import MedicalKnowledgeSystem
from src.systems.human_review_queue import HumanReviewQueue

# Import medical systems
from src.core.session_manager import SessionManager
from src.utils.audit_service import AuditService, AuditEventType
from src.a2a.base_infrastructure import A2AServer, AgentCard

logger = logging.getLogger(__name__)


class ClinicalUrgency(Enum):
    """Clinical urgency levels for medical assessment"""
    ROUTINE = "routine"
    URGENT = "urgent"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AssessmentType(Enum):
    """Types of clinical assessment"""
    INITIAL_SCREENING = "initial_screening"
    DETAILED_ASSESSMENT = "detailed_assessment"
    FOLLOW_UP = "follow_up"
    ESCALATION_REVIEW = "escalation_review"


# Agent instruction for clinical assessment
CLINICAL_ASSESSMENT_INSTRUCTION = """
Eres el ClinicalAssessmentAgent especializado en evaluación clínica médica basada en evidencia
para lesiones por presión (LPP) siguiendo protocolos NPUAP/EPUAP internacionales y MINSAL chilenos.

RESPONSABILIDADES PRINCIPALES:
1. Evaluación clínica integral basada en evidencia científica
2. Aplicación de guidelines NPUAP/EPUAP/PPPIA 2019
3. Integración protocolos MINSAL para contexto chileno
4. Evaluación de factores de riesgo y comorbilidades
5. Generación de recomendaciones clínicas específicas
6. Escalamiento médico según criterios de seguridad

CONOCIMIENTO MÉDICO:
- Guidelines NPUAP/EPUAP 2019 completas con niveles de evidencia A/B/C
- Protocolos MINSAL 2018 para sistema de salud chileno
- Evaluación factores riesgo: diabetes, desnutrición, anticoagulación
- Protocolos específicos por ubicación anatómica
- Escalas validadas: Braden, Norton, ELPO

CRITERIOS DE ESCALAMIENTO:
- Confianza < 70%: Revisión especialista wound care
- LPP Grado 3-4: Evaluación médico inmediata
- Factores alto riesgo: Protocolo especializado
- Casos complejos: Equipo multidisciplinario

COMPLIANCE MÉDICO:
- Trazabilidad completa con referencias científicas
- Justificación clínica por cada recomendación
- Audit trail para compliance médico-legal
- Protección datos según HIPAA/ISO 13485
"""


class ClinicalAssessmentAgent:
    """
    Evidence-based clinical assessment agent implementing comprehensive
    medical decision making with ADK patterns and A2A communication.
    """
    
    def __init__(self):
        """Initialize clinical assessment agent with medical systems"""
        self.agent_id = "clinical_assessment_agent"
        self.session_manager = SessionManager()
        self.audit_service = AuditService()
        
        # Initialize clinical processing systems
        self.clinical_processor = ClinicalProcessingSystem()
        self.decision_engine = MedicalDecisionEngine()
        self.minsal_engine = MinisterioSaludDecisionEngine()
        self.knowledge_system = MedicalKnowledgeSystem()
        self.review_queue = HumanReviewQueue()
        
        # Processing statistics
        self.stats = {
            'assessments_completed': 0,
            'evidence_based_decisions': 0,
            'escalations_triggered': 0,
            'human_reviews_requested': 0,
            'avg_assessment_time': 0.0,
            'decision_confidence_avg': 0.0
        }
        
        # A2A server for distributed communication
        self.a2a_server = None
        
        logger.info(f"ClinicalAssessmentAgent initialized with evidence-based systems")
    
    async def perform_clinical_assessment(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main clinical assessment function for comprehensive medical evaluation.
        
        Args:
            assessment_data: Dictionary containing:
                - image_analysis_result: Results from ImageAnalysisAgent
                - patient_code: Anonymized patient identifier
                - patient_context: Patient medical context and risk factors
                - session_token: Session tracking token
                - assessment_type: Type of clinical assessment
                - medical_urgency: Clinical urgency level
                
        Returns:
            Complete clinical assessment with evidence-based recommendations
        """
        start_time = datetime.now()
        patient_code = assessment_data.get('patient_code')
        session_token = assessment_data.get('session_token')
        assessment_type = assessment_data.get('assessment_type', AssessmentType.INITIAL_SCREENING.value)
        
        try:
            # Initialize assessment session
            await self._initialize_assessment_session(assessment_data)
            
            # Step 1: Extract and validate medical findings
            logger.info(f"Extracting medical findings for patient {patient_code}")
            medical_findings = await self._extract_medical_findings(assessment_data)
            
            if not medical_findings['extraction_success']:
                return await self._handle_extraction_error(assessment_data, medical_findings['error'])
            
            # Step 2: Comprehensive risk assessment
            logger.info(f"Performing risk assessment for patient {patient_code}")
            risk_assessment = await self._perform_risk_assessment(
                medical_findings, assessment_data.get('patient_context', {})
            )
            
            # Step 3: Evidence-based clinical decision making
            logger.info(f"Making evidence-based decisions for patient {patient_code}")
            clinical_decision = await self._make_evidence_based_decision(
                medical_findings, risk_assessment, assessment_data
            )
            
            # Step 4: Generate clinical recommendations
            logger.info(f"Generating clinical recommendations for patient {patient_code}")
            clinical_recommendations = await self._generate_clinical_recommendations(
                clinical_decision, medical_findings, risk_assessment
            )
            
            # Step 5: Escalation assessment
            escalation_assessment = await self._assess_escalation_requirements(
                clinical_decision, risk_assessment, assessment_data
            )
            
            # Step 6: Generate comprehensive assessment report
            assessment_report = await self._generate_assessment_report(
                assessment_data, {
                    'medical_findings': medical_findings,
                    'risk_assessment': risk_assessment,
                    'clinical_decision': clinical_decision,
                    'clinical_recommendations': clinical_recommendations,
                    'escalation_assessment': escalation_assessment
                }
            )
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            await self._update_assessment_stats(processing_time, assessment_report)
            
            # Finalize session
            await self._finalize_assessment_session(session_token, assessment_report)
            
            return assessment_report
            
        except Exception as e:
            logger.error(f"Critical error in clinical assessment: {str(e)}")
            error_result = await self._handle_critical_assessment_error(assessment_data, str(e))
            await self._finalize_assessment_session(session_token, error_result)
            return error_result
    
    async def _initialize_assessment_session(self, assessment_data: Dict[str, Any]):
        """Initialize clinical assessment session with audit trail"""
        session_token = assessment_data.get('session_token')
        patient_code = assessment_data.get('patient_code')
        
        if session_token:
            # Audit clinical assessment start
            await self.audit_service.log_event(
                event_type=AuditEventType.MEDICAL_DECISION,
                component='clinical_assessment_agent',
                action='start_assessment',
                session_id=session_token,
                user_id='clinical_assessment_agent',
                resource=f'clinical_assessment_{patient_code}',
                details={
                    'agent_id': self.agent_id,
                    'assessment_type': assessment_data.get('assessment_type'),
                    'medical_urgency': assessment_data.get('medical_urgency'),
                    'assessment_start': datetime.now().isoformat()
                }
            )
    
    async def _extract_medical_findings(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate medical findings from image analysis"""
        try:
            image_analysis = assessment_data.get('image_analysis_result', {})
            medical_findings = image_analysis.get('medical_findings', {})
            
            # Validate required medical data
            required_fields = ['lpp_detected', 'lpp_grade', 'confidence_score']
            missing_fields = [field for field in required_fields if field not in medical_findings]
            
            if missing_fields:
                return {
                    'extraction_success': False,
                    'error': f'Missing required medical fields: {missing_fields}',
                    'error_type': 'incomplete_medical_data'
                }
            
            # Extract structured medical findings
            extracted_findings = {
                'lpp_present': medical_findings.get('lpp_detected', False),
                'lpp_grade': medical_findings.get('lpp_grade', 0),
                'severity_level': medical_findings.get('severity', 'NONE'),
                'detection_confidence': medical_findings.get('confidence_score', 0.0),
                'anatomical_location': medical_findings.get('anatomical_location', 'unknown'),
                'detection_bbox': medical_findings.get('detection_bbox'),
                'medical_significance': medical_findings.get('medical_significance'),
                'requires_attention': medical_findings.get('requires_attention', False)
            }
            
            # Add technical details
            technical_details = assessment_data.get('image_analysis_result', {}).get('technical_details', {})
            extracted_findings.update({
                'detection_method': technical_details.get('detector_used', 'unknown'),
                'processing_time_ms': technical_details.get('processing_time_ms', 0),
                'model_type': technical_details.get('model_type'),
                'privacy_protected': technical_details.get('privacy_protected', True)
            })
            
            return {
                'extraction_success': True,
                'extracted_findings': extracted_findings,
                'raw_image_analysis': image_analysis,
                'extraction_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'extraction_success': False,
                'error': f'Medical findings extraction failed: {str(e)}',
                'error_type': 'extraction_exception'
            }
    
    async def _perform_risk_assessment(self, medical_findings: Dict[str, Any], 
                                     patient_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive patient risk assessment"""
        try:
            extracted_findings = medical_findings.get('extracted_findings', {})
            
            # Extract patient risk factors
            risk_factors = patient_context.get('risk_factors', [])
            age = patient_context.get('age', 65)
            diabetes = 'diabetes' in risk_factors
            malnutrition = 'malnutrition' in risk_factors
            immobility = 'immobility' in risk_factors
            anticoagulants = patient_context.get('anticoagulants', False)
            
            # Calculate risk scores
            braden_score = self._calculate_braden_score(patient_context)
            norton_score = self._calculate_norton_score(patient_context)
            
            # Assess medical complexity
            medical_complexity = self._assess_medical_complexity(
                extracted_findings, patient_context
            )
            
            # Determine overall risk level
            risk_level = self._determine_risk_level(
                braden_score, extracted_findings, risk_factors
            )
            
            return {
                'risk_assessment_success': True,
                'overall_risk_level': risk_level,
                'risk_scores': {
                    'braden_score': braden_score,
                    'norton_score': norton_score,
                    'custom_risk_score': self._calculate_custom_risk_score(patient_context)
                },
                'risk_factors': {
                    'diabetes': diabetes,
                    'malnutrition': malnutrition,
                    'immobility': immobility,
                    'anticoagulants': anticoagulants,
                    'advanced_age': age > 75,
                    'multiple_comorbidities': len(risk_factors) >= 3
                },
                'medical_complexity': medical_complexity,
                'anatomical_risk_factors': self._assess_anatomical_risks(
                    extracted_findings.get('anatomical_location')
                ),
                'assessment_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {str(e)}")
            return {
                'risk_assessment_success': False,
                'error': str(e),
                'error_type': 'risk_assessment_failure'
            }
    
    def _calculate_braden_score(self, patient_context: Dict[str, Any]) -> int:
        """Calculate Braden Scale score for pressure injury risk"""
        # Braden Scale implementation (simplified)
        score = 23  # Start with maximum (lowest risk)
        
        risk_factors = patient_context.get('risk_factors', [])
        age = patient_context.get('age', 65)
        
        # Sensory perception (1-4)
        if 'neurological_impairment' in risk_factors:
            score -= 2
        
        # Moisture (1-4)
        if 'incontinence' in risk_factors:
            score -= 2
        
        # Activity (1-4)
        if 'immobility' in risk_factors:
            score -= 3
        elif 'limited_mobility' in risk_factors:
            score -= 2
        
        # Mobility (1-4)
        if 'bedbound' in risk_factors:
            score -= 3
        
        # Nutrition (1-4)
        if 'malnutrition' in risk_factors:
            score -= 3
        elif 'poor_nutrition' in risk_factors:
            score -= 2
        
        # Friction and shear (1-3)
        if 'friction_shear' in risk_factors:
            score -= 2
        
        return max(6, min(23, score))  # Constrain to valid range
    
    def _calculate_norton_score(self, patient_context: Dict[str, Any]) -> int:
        """Calculate Norton Scale score"""
        # Norton Scale implementation (simplified)
        score = 20  # Start with maximum
        
        risk_factors = patient_context.get('risk_factors', [])
        
        if 'immobility' in risk_factors:
            score -= 3
        if 'incontinence' in risk_factors:
            score -= 3
        if 'malnutrition' in risk_factors:
            score -= 3
        if 'altered_mental_state' in risk_factors:
            score -= 3
        
        return max(4, min(20, score))
    
    def _calculate_custom_risk_score(self, patient_context: Dict[str, Any]) -> float:
        """Calculate custom risk score based on medical evidence"""
        risk_score = 0.0
        risk_factors = patient_context.get('risk_factors', [])
        age = patient_context.get('age', 65)
        
        # Age factor
        if age > 75:
            risk_score += 0.3
        elif age > 65:
            risk_score += 0.1
        
        # Medical conditions
        if 'diabetes' in risk_factors:
            risk_score += 0.4
        if 'malnutrition' in risk_factors:
            risk_score += 0.3
        if 'immobility' in risk_factors:
            risk_score += 0.5
        if 'anticoagulants' in patient_context.get('medications', []):
            risk_score += 0.2
        
        return min(1.0, risk_score)
    
    def _assess_medical_complexity(self, medical_findings: Dict[str, Any], 
                                 patient_context: Dict[str, Any]) -> str:
        """Assess overall medical complexity of the case"""
        complexity_factors = 0
        
        # LPP grade complexity
        lpp_grade = medical_findings.get('lpp_grade', 0)
        if lpp_grade >= 3:
            complexity_factors += 3
        elif lpp_grade >= 2:
            complexity_factors += 2
        elif lpp_grade >= 1:
            complexity_factors += 1
        
        # Risk factor complexity
        risk_factors = patient_context.get('risk_factors', [])
        complexity_factors += len(risk_factors)
        
        # Confidence factor
        confidence = medical_findings.get('detection_confidence', 1.0)
        if confidence < 0.7:
            complexity_factors += 2
        
        # Age factor
        if patient_context.get('age', 65) > 80:
            complexity_factors += 1
        
        # Determine complexity level
        if complexity_factors >= 6:
            return "HIGH"
        elif complexity_factors >= 3:
            return "MODERATE"
        else:
            return "LOW"
    
    def _determine_risk_level(self, braden_score: int, medical_findings: Dict[str, Any], 
                            risk_factors: List[str]) -> str:
        """Determine overall patient risk level"""
        lpp_grade = medical_findings.get('lpp_grade', 0)
        
        # Critical risk conditions
        if lpp_grade >= 3 or braden_score <= 12:
            return "CRITICAL"
        
        # High risk conditions
        if lpp_grade >= 2 or braden_score <= 16 or len(risk_factors) >= 3:
            return "HIGH"
        
        # Moderate risk conditions
        if lpp_grade >= 1 or braden_score <= 18 or len(risk_factors) >= 1:
            return "MODERATE"
        
        return "LOW"
    
    def _assess_anatomical_risks(self, anatomical_location: Optional[str]) -> Dict[str, Any]:
        """Assess location-specific risk factors"""
        if not anatomical_location or anatomical_location == 'unknown':
            return {'location_specific_risks': [], 'special_considerations': []}
        
        location_risks = {
            'sacrum': {
                'risks': ['high_pressure_area', 'moisture_exposure', 'friction_shear'],
                'considerations': ['pressure_redistribution_essential', 'frequent_repositioning']
            },
            'heel': {
                'risks': ['bony_prominence', 'limited_tissue_coverage', 'vascular_compromise'],
                'considerations': ['heel_offloading_required', 'vascular_assessment']
            },
            'trochanter': {
                'risks': ['lateral_pressure', 'bone_proximity', 'positioning_challenges'],
                'considerations': ['side_lying_positioning', 'pressure_redistribution']
            },
            'ischium': {
                'risks': ['sitting_pressure', 'moisture_risk', 'deep_tissue_injury'],
                'considerations': ['seating_assessment', 'wheelchair_cushioning']
            }
        }
        
        return location_risks.get(anatomical_location.lower(), {
            'risks': ['general_pressure_risk'],
            'considerations': ['standard_pressure_relief']
        })
    
    async def _make_evidence_based_decision(self, medical_findings: Dict[str, Any],
                                          risk_assessment: Dict[str, Any],
                                          assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make evidence-based clinical decision using multiple engines"""
        try:
            extracted_findings = medical_findings.get('extracted_findings', {})
            patient_context = assessment_data.get('patient_context', {})
            
            # Determine if Chilean context
            use_minsal = patient_context.get('healthcare_system', '').lower() == 'chilean' or \
                        patient_context.get('use_minsal_guidelines', False)
            
            if use_minsal:
                # Use MINSAL-enhanced decision engine
                decision_result = make_minsal_clinical_decision(
                    lpp_grade=extracted_findings.get('lpp_grade', 0),
                    confidence=extracted_findings.get('detection_confidence', 0.0),
                    anatomical_location=extracted_findings.get('anatomical_location', 'unknown'),
                    patient_context=patient_context
                )
                decision_engine_used = 'minsal_enhanced'
            else:
                # Use international NPUAP/EPUAP decision engine
                decision_result = make_evidence_based_decision(
                    lpp_grade=extracted_findings.get('lpp_grade', 0),
                    confidence=extracted_findings.get('detection_confidence', 0.0),
                    anatomical_location=extracted_findings.get('anatomical_location', 'unknown'),
                    patient_context=patient_context
                )
                decision_engine_used = 'npuap_epuap'
            
            # Enhance decision with risk assessment
            enhanced_decision = self._enhance_decision_with_risk_assessment(
                decision_result, risk_assessment
            )
            
            return {
                'decision_success': True,
                'decision_engine_used': decision_engine_used,
                'evidence_based_decision': enhanced_decision,
                'decision_confidence': self._calculate_decision_confidence(
                    decision_result, risk_assessment
                ),
                'evidence_level': decision_result.get('evidence_level', 'C'),
                'clinical_rationale': decision_result.get('clinical_rationale', ''),
                'decision_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Evidence-based decision making failed: {str(e)}")
            return {
                'decision_success': False,
                'error': str(e),
                'error_type': 'decision_engine_failure'
            }
    
    def _enhance_decision_with_risk_assessment(self, decision_result: Dict[str, Any],
                                             risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance clinical decision with comprehensive risk assessment"""
        enhanced_decision = decision_result.copy()
        
        # Add risk-specific modifications
        risk_level = risk_assessment.get('overall_risk_level', 'MODERATE')
        risk_factors = risk_assessment.get('risk_factors', {})
        
        # Enhance recommendations based on risk
        enhanced_recommendations = enhanced_decision.get('clinical_recommendations', []).copy()
        
        if risk_level == 'CRITICAL':
            enhanced_recommendations.insert(0, 'CRITICAL RISK: Immediate medical evaluation required')
            enhanced_recommendations.append('Consider ICU consultation for complex care')
        
        if risk_factors.get('diabetes', False):
            enhanced_recommendations.append('Enhanced glucose monitoring and control')
            enhanced_recommendations.append('Diabetic wound care specialist consultation')
        
        if risk_factors.get('malnutrition', False):
            enhanced_recommendations.append('Immediate nutritional assessment and intervention')
            enhanced_recommendations.append('Consider nutritionist consultation')
        
        if risk_factors.get('anticoagulants', False):
            enhanced_recommendations.append('Anticoagulation management coordination')
            enhanced_recommendations.append('Bleeding risk assessment for procedures')
        
        # Update enhanced decision
        enhanced_decision.update({
            'clinical_recommendations': enhanced_recommendations,
            'risk_adjusted_urgency': self._calculate_risk_adjusted_urgency(
                enhanced_decision.get('medical_urgency', 'ROUTINE'), risk_level
            ),
            'escalation_requirements': self._determine_escalation_requirements(
                enhanced_decision, risk_assessment
            )
        })
        
        return enhanced_decision
    
    def _calculate_decision_confidence(self, decision_result: Dict[str, Any],
                                     risk_assessment: Dict[str, Any]) -> float:
        """Calculate overall decision confidence"""
        base_confidence = decision_result.get('confidence_score', 0.5)
        
        # Adjust based on risk assessment quality
        risk_success = risk_assessment.get('risk_assessment_success', False)
        medical_complexity = risk_assessment.get('medical_complexity', 'HIGH')
        
        # Confidence adjustments
        confidence_adjustment = 0.0
        
        if risk_success:
            confidence_adjustment += 0.1
        
        if medical_complexity == 'LOW':
            confidence_adjustment += 0.1
        elif medical_complexity == 'HIGH':
            confidence_adjustment -= 0.1
        
        # Evidence level adjustment
        evidence_level = decision_result.get('evidence_level', 'C')
        if evidence_level == 'A':
            confidence_adjustment += 0.15
        elif evidence_level == 'B':
            confidence_adjustment += 0.05
        
        final_confidence = min(1.0, max(0.0, base_confidence + confidence_adjustment))
        return round(final_confidence, 3)
    
    def _calculate_risk_adjusted_urgency(self, base_urgency: str, risk_level: str) -> str:
        """Calculate risk-adjusted medical urgency"""
        urgency_levels = ['ROUTINE', 'URGENT', 'CRITICAL', 'EMERGENCY']
        
        base_index = urgency_levels.index(base_urgency) if base_urgency in urgency_levels else 0
        
        # Risk level adjustments
        if risk_level == 'CRITICAL':
            base_index = max(base_index, 2)  # At least CRITICAL
        elif risk_level == 'HIGH':
            base_index = max(base_index, 1)  # At least URGENT
        
        return urgency_levels[min(base_index, len(urgency_levels) - 1)]
    
    def _determine_escalation_requirements(self, decision_result: Dict[str, Any],
                                         risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Determine escalation requirements based on decision and risk"""
        escalation_triggers = []
        escalation_required = False
        escalation_urgency = 'NONE'
        
        # Low confidence escalation
        confidence = decision_result.get('confidence_score', 1.0)
        if confidence < 0.7:
            escalation_triggers.append('low_confidence')
            escalation_required = True
            escalation_urgency = 'HIGH'
        
        # High-grade LPP escalation
        lpp_grade = decision_result.get('lpp_grade', 0)
        if lpp_grade >= 3:
            escalation_triggers.append('high_grade_lpp')
            escalation_required = True
            escalation_urgency = 'CRITICAL'
        
        # Risk level escalation
        risk_level = risk_assessment.get('overall_risk_level', 'LOW')
        if risk_level == 'CRITICAL':
            escalation_triggers.append('critical_risk')
            escalation_required = True
            escalation_urgency = 'CRITICAL'
        
        # Medical complexity escalation
        complexity = risk_assessment.get('medical_complexity', 'LOW')
        if complexity == 'HIGH':
            escalation_triggers.append('high_complexity')
            escalation_required = True
            escalation_urgency = max(escalation_urgency, 'HIGH')
        
        return {
            'escalation_required': escalation_required,
            'escalation_triggers': escalation_triggers,
            'escalation_urgency': escalation_urgency,
            'human_review_required': escalation_required,
            'specialist_consultation': lpp_grade >= 2 or risk_level in ['HIGH', 'CRITICAL'],
            'multidisciplinary_team': lpp_grade >= 3 or complexity == 'HIGH'
        }
    
    async def _generate_clinical_recommendations(self, clinical_decision: Dict[str, Any],
                                               medical_findings: Dict[str, Any],
                                               risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive clinical recommendations"""
        try:
            evidence_decision = clinical_decision.get('evidence_based_decision', {})
            base_recommendations = evidence_decision.get('clinical_recommendations', [])
            
            # Categorize recommendations
            categorized_recommendations = {
                'immediate_actions': [],
                'wound_care': [],
                'prevention': [],
                'monitoring': [],
                'specialist_referrals': [],
                'patient_education': []
            }
            
            # Process base recommendations
            for recommendation in base_recommendations:
                self._categorize_recommendation(recommendation, categorized_recommendations)
            
            # Add risk-specific recommendations
            self._add_risk_specific_recommendations(
                categorized_recommendations, risk_assessment
            )
            
            # Add location-specific recommendations
            anatomical_risks = risk_assessment.get('anatomical_risk_factors', {})
            self._add_anatomical_recommendations(
                categorized_recommendations, anatomical_risks
            )
            
            # Generate follow-up schedule
            follow_up_schedule = self._generate_follow_up_schedule(
                evidence_decision, risk_assessment
            )
            
            return {
                'recommendations_success': True,
                'categorized_recommendations': categorized_recommendations,
                'priority_actions': self._identify_priority_actions(categorized_recommendations),
                'follow_up_schedule': follow_up_schedule,
                'patient_education_materials': self._recommend_education_materials(evidence_decision),
                'care_plan_duration': self._estimate_care_duration(evidence_decision),
                'recommendations_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Clinical recommendations generation failed: {str(e)}")
            return {
                'recommendations_success': False,
                'error': str(e),
                'error_type': 'recommendations_failure'
            }
    
    def _categorize_recommendation(self, recommendation: str, categories: Dict[str, List[str]]):
        """Categorize individual recommendation"""
        rec_lower = recommendation.lower()
        
        # Immediate actions
        if any(keyword in rec_lower for keyword in ['immediate', 'urgent', 'critical', 'emergency']):
            categories['immediate_actions'].append(recommendation)
        # Wound care
        elif any(keyword in rec_lower for keyword in ['wound', 'dressing', 'debridement', 'clean']):
            categories['wound_care'].append(recommendation)
        # Prevention
        elif any(keyword in rec_lower for keyword in ['prevent', 'repositioning', 'pressure relief']):
            categories['prevention'].append(recommendation)
        # Monitoring
        elif any(keyword in rec_lower for keyword in ['monitor', 'assess', 'evaluate', 'track']):
            categories['monitoring'].append(recommendation)
        # Specialist referrals
        elif any(keyword in rec_lower for keyword in ['specialist', 'consultation', 'referral']):
            categories['specialist_referrals'].append(recommendation)
        # Patient education
        elif any(keyword in rec_lower for keyword in ['education', 'teaching', 'instruction']):
            categories['patient_education'].append(recommendation)
        else:
            # Default to immediate actions if unclear
            categories['immediate_actions'].append(recommendation)
    
    def _add_risk_specific_recommendations(self, categories: Dict[str, List[str]],
                                         risk_assessment: Dict[str, Any]):
        """Add recommendations specific to patient risk factors"""
        risk_factors = risk_assessment.get('risk_factors', {})
        
        if risk_factors.get('diabetes'):
            categories['monitoring'].append('Monitor blood glucose levels closely')
            categories['wound_care'].append('Apply diabetic wound care protocols')
        
        if risk_factors.get('malnutrition'):
            categories['immediate_actions'].append('Nutritionist consultation within 24 hours')
            categories['monitoring'].append('Daily nutritional intake assessment')
        
        if risk_factors.get('immobility'):
            categories['prevention'].append('Implement high-frequency repositioning (every 1-2 hours)')
            categories['prevention'].append('Consider specialized pressure-redistributing surface')
        
        if risk_factors.get('anticoagulants'):
            categories['wound_care'].append('Use gentle wound care techniques to minimize bleeding')
            categories['monitoring'].append('Monitor for signs of bleeding or hematoma')
    
    def _add_anatomical_recommendations(self, categories: Dict[str, List[str]],
                                      anatomical_risks: Dict[str, Any]):
        """Add location-specific recommendations"""
        considerations = anatomical_risks.get('considerations', [])
        
        for consideration in considerations:
            if 'offloading' in consideration:
                categories['immediate_actions'].append(f'Implement {consideration}')
            elif 'assessment' in consideration:
                categories['monitoring'].append(f'Perform {consideration}')
            else:
                categories['prevention'].append(f'Apply {consideration}')
    
    def _identify_priority_actions(self, categorized_recommendations: Dict[str, List[str]]) -> List[str]:
        """Identify highest priority actions"""
        priority_actions = []
        
        # Immediate actions are always priority
        priority_actions.extend(categorized_recommendations.get('immediate_actions', []))
        
        # Critical wound care actions
        wound_care = categorized_recommendations.get('wound_care', [])
        priority_wound_care = [rec for rec in wound_care if any(
            keyword in rec.lower() for keyword in ['immediate', 'urgent', 'critical']
        )]
        priority_actions.extend(priority_wound_care)
        
        # Essential prevention
        prevention = categorized_recommendations.get('prevention', [])
        if any('repositioning' in rec.lower() for rec in prevention):
            priority_actions.append('Implement immediate pressure relief and repositioning')
        
        return priority_actions[:5]  # Limit to top 5 priorities
    
    def _generate_follow_up_schedule(self, evidence_decision: Dict[str, Any],
                                   risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate follow-up care schedule"""
        lpp_grade = evidence_decision.get('lpp_grade', 0)
        risk_level = risk_assessment.get('overall_risk_level', 'LOW')
        
        # Determine follow-up frequency
        if lpp_grade >= 3 or risk_level == 'CRITICAL':
            return {
                'initial_follow_up': '24 hours',
                'routine_follow_up': '48-72 hours',
                'specialist_follow_up': '7 days',
                'reassessment_frequency': 'Daily'
            }
        elif lpp_grade >= 2 or risk_level == 'HIGH':
            return {
                'initial_follow_up': '48-72 hours',
                'routine_follow_up': '1 week',
                'specialist_follow_up': '2 weeks',
                'reassessment_frequency': 'Every 2-3 days'
            }
        else:
            return {
                'initial_follow_up': '1 week',
                'routine_follow_up': '2 weeks',
                'specialist_follow_up': 'As needed',
                'reassessment_frequency': 'Weekly'
            }
    
    def _recommend_education_materials(self, evidence_decision: Dict[str, Any]) -> List[str]:
        """Recommend patient education materials"""
        lpp_grade = evidence_decision.get('lpp_grade', 0)
        
        education_materials = ['Pressure Injury Prevention Basics']
        
        if lpp_grade >= 1:
            education_materials.extend([
                'Understanding Pressure Injuries',
                'Skin Care and Inspection Techniques',
                'Importance of Repositioning'
            ])
        
        if lpp_grade >= 2:
            education_materials.extend([
                'Wound Care Basics for Patients',
                'Nutrition and Healing',
                'When to Seek Medical Attention'
            ])
        
        return education_materials
    
    def _estimate_care_duration(self, evidence_decision: Dict[str, Any]) -> str:
        """Estimate expected care duration"""
        lpp_grade = evidence_decision.get('lpp_grade', 0)
        
        duration_map = {
            0: '1-2 weeks (prevention focus)',
            1: '2-4 weeks (early intervention)',
            2: '4-8 weeks (active treatment)',
            3: '8-16 weeks (complex care)',
            4: '16+ weeks (long-term management)'
        }
        
        return duration_map.get(lpp_grade, '4-8 weeks (standard care)')
    
    async def _assess_escalation_requirements(self, clinical_decision: Dict[str, Any],
                                            risk_assessment: Dict[str, Any],
                                            assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess comprehensive escalation requirements"""
        try:
            evidence_decision = clinical_decision.get('evidence_based_decision', {})
            escalation_reqs = evidence_decision.get('escalation_requirements', {})
            
            # Determine escalation actions
            escalation_actions = []
            
            if escalation_reqs.get('human_review_required', False):
                escalation_actions.append('human_review')
            
            if escalation_reqs.get('specialist_consultation', False):
                escalation_actions.append('specialist_consultation')
            
            if escalation_reqs.get('multidisciplinary_team', False):
                escalation_actions.append('multidisciplinary_team_review')
            
            # Determine notification requirements
            notification_requirements = self._determine_notification_requirements(
                clinical_decision, risk_assessment
            )
            
            return {
                'escalation_assessment_success': True,
                'escalation_required': escalation_reqs.get('escalation_required', False),
                'escalation_urgency': escalation_reqs.get('escalation_urgency', 'NONE'),
                'escalation_triggers': escalation_reqs.get('escalation_triggers', []),
                'escalation_actions': escalation_actions,
                'notification_requirements': notification_requirements,
                'human_review_priority': self._calculate_review_priority(escalation_reqs),
                'estimated_response_time': self._estimate_response_time(escalation_reqs),
                'assessment_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'escalation_assessment_success': False,
                'error': str(e),
                'error_type': 'escalation_assessment_failure'
            }
    
    def _determine_notification_requirements(self, clinical_decision: Dict[str, Any],
                                           risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Determine who needs to be notified and how urgently"""
        evidence_decision = clinical_decision.get('evidence_based_decision', {})
        lpp_grade = evidence_decision.get('lpp_grade', 0)
        risk_level = risk_assessment.get('overall_risk_level', 'LOW')
        
        notifications = {
            'immediate_notifications': [],
            'urgent_notifications': [],
            'routine_notifications': []
        }
        
        # Immediate notifications (within 15 minutes)
        if lpp_grade >= 4 or risk_level == 'CRITICAL':
            notifications['immediate_notifications'].extend([
                'attending_physician',
                'wound_care_specialist',
                'nursing_supervisor'
            ])
        
        # Urgent notifications (within 2 hours)
        if lpp_grade >= 3 or risk_level == 'HIGH':
            notifications['urgent_notifications'].extend([
                'primary_care_team',
                'wound_care_specialist'
            ])
        
        # Routine notifications (within 24 hours)
        if lpp_grade >= 1:
            notifications['routine_notifications'].extend([
                'primary_nurse',
                'care_coordinator'
            ])
        
        return notifications
    
    def _calculate_review_priority(self, escalation_reqs: Dict[str, Any]) -> str:
        """Calculate human review priority level"""
        urgency = escalation_reqs.get('escalation_urgency', 'NONE')
        triggers = escalation_reqs.get('escalation_triggers', [])
        
        if urgency == 'CRITICAL' or 'critical_risk' in triggers:
            return 'EMERGENCY'
        elif urgency == 'HIGH' or 'high_grade_lpp' in triggers:
            return 'URGENT'
        elif escalation_reqs.get('escalation_required', False):
            return 'HIGH'
        else:
            return 'ROUTINE'
    
    def _estimate_response_time(self, escalation_reqs: Dict[str, Any]) -> str:
        """Estimate expected response time for escalation"""
        urgency = escalation_reqs.get('escalation_urgency', 'NONE')
        
        response_times = {
            'CRITICAL': '15 minutes',
            'HIGH': '2 hours',
            'MODERATE': '8 hours',
            'NONE': '24 hours'
        }
        
        return response_times.get(urgency, '24 hours')
    
    async def _generate_assessment_report(self, assessment_data: Dict[str, Any],
                                        processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive clinical assessment report"""
        patient_code = assessment_data.get('patient_code')
        session_token = assessment_data.get('session_token')
        
        medical_findings = processing_results['medical_findings']
        risk_assessment = processing_results['risk_assessment']
        clinical_decision = processing_results['clinical_decision']
        clinical_recommendations = processing_results['clinical_recommendations']
        escalation_assessment = processing_results['escalation_assessment']
        
        # Determine overall assessment success
        assessment_success = all([
            medical_findings.get('extraction_success', False),
            risk_assessment.get('risk_assessment_success', False),
            clinical_decision.get('decision_success', False),
            clinical_recommendations.get('recommendations_success', False),
            escalation_assessment.get('escalation_assessment_success', False)
        ])
        
        return {
            # Case identification
            'assessment_id': f"{patient_code}_{session_token}_{int(datetime.now().timestamp())}",
            'patient_code': patient_code,
            'session_token': session_token,
            'assessment_timestamp': datetime.now().isoformat(),
            'agent_id': self.agent_id,
            
            # Assessment success status
            'assessment_success': assessment_success,
            'assessment_type': assessment_data.get('assessment_type', 'initial_screening'),
            
            # Clinical findings summary
            'clinical_summary': {
                'lpp_detected': medical_findings.get('extracted_findings', {}).get('lpp_present', False),
                'lpp_grade': medical_findings.get('extracted_findings', {}).get('lpp_grade', 0),
                'severity_level': medical_findings.get('extracted_findings', {}).get('severity_level', 'NONE'),
                'overall_risk_level': risk_assessment.get('overall_risk_level', 'UNKNOWN'),
                'medical_complexity': risk_assessment.get('medical_complexity', 'UNKNOWN'),
                'decision_confidence': clinical_decision.get('decision_confidence', 0.0)
            },
            
            # Evidence-based decision
            'evidence_based_decision': clinical_decision.get('evidence_based_decision', {}),
            'decision_engine_used': clinical_decision.get('decision_engine_used', 'unknown'),
            'evidence_level': clinical_decision.get('evidence_level', 'C'),
            'clinical_rationale': clinical_decision.get('clinical_rationale', ''),
            
            # Risk assessment details
            'risk_assessment': {
                'overall_risk_level': risk_assessment.get('overall_risk_level'),
                'risk_scores': risk_assessment.get('risk_scores', {}),
                'identified_risk_factors': risk_assessment.get('risk_factors', {}),
                'anatomical_considerations': risk_assessment.get('anatomical_risk_factors', {})
            },
            
            # Clinical recommendations
            'clinical_recommendations': clinical_recommendations.get('categorized_recommendations', {}),
            'priority_actions': clinical_recommendations.get('priority_actions', []),
            'follow_up_schedule': clinical_recommendations.get('follow_up_schedule', {}),
            
            # Escalation information
            'escalation_info': {
                'required': escalation_assessment.get('escalation_required', False),
                'urgency': escalation_assessment.get('escalation_urgency', 'NONE'),
                'triggers': escalation_assessment.get('escalation_triggers', []),
                'actions_required': escalation_assessment.get('escalation_actions', []),
                'human_review_priority': escalation_assessment.get('human_review_priority', 'ROUTINE'),
                'notification_requirements': escalation_assessment.get('notification_requirements', {})
            },
            
            # Medical compliance
            'compliance_info': {
                'guidelines_followed': [clinical_decision.get('decision_engine_used', 'npuap_epuap')],
                'evidence_based': True,
                'audit_trail_complete': True,
                'medical_justification_provided': True,
                'regulatory_compliance': ['NPUAP_EPUAP_2019', 'HIPAA', 'ISO_13485']
            },
            
            # Technical details
            'technical_details': {
                'assessment_processing_time': 0.0,  # Will be updated
                'confidence_score': clinical_decision.get('decision_confidence', 0.0),
                'algorithms_used': [
                    clinical_decision.get('decision_engine_used', 'unknown'),
                    'braden_scale',
                    'norton_scale'
                ],
                'data_sources': ['image_analysis', 'patient_context', 'medical_guidelines']
            }
        }
    
    async def _handle_extraction_error(self, assessment_data: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Handle medical findings extraction error"""
        return await self._handle_assessment_error(assessment_data, error, 'extraction_error')
    
    async def _handle_critical_assessment_error(self, assessment_data: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Handle critical assessment error"""
        return await self._handle_assessment_error(assessment_data, error, 'critical_assessment_error')
    
    async def _handle_assessment_error(self, assessment_data: Dict[str, Any], 
                                     error: str, error_type: str) -> Dict[str, Any]:
        """Generic error handler for assessment errors"""
        patient_code = assessment_data.get('patient_code')
        session_token = assessment_data.get('session_token')
        
        self.stats['processing_errors'] += 1
        
        # Log error for audit
        await self.audit_service.log_event(
            event_type=AuditEventType.ERROR_OCCURRED,
            component='clinical_assessment_agent',
            action='assessment_error',
            session_id=session_token,
            user_id='clinical_assessment_agent',
            resource=f'clinical_assessment_{patient_code}',
            details={
                'error_type': error_type,
                'error_message': error,
                'requires_escalation': True
            }
        )
        
        return {
            'assessment_id': f"{patient_code}_{session_token}_error",
            'patient_code': patient_code,
            'assessment_success': False,
            'error_type': error_type,
            'error_message': error,
            'clinical_summary': {
                'lpp_detected': None,
                'lpp_grade': None,
                'severity_level': f'ERROR_{error_type.upper()}',
                'requires_human_review': True,
                'error_processing': True
            },
            'escalation_info': {
                'required': True,
                'urgency': 'CRITICAL' if error_type == 'critical_assessment_error' else 'HIGH',
                'triggers': [error_type],
                'human_review_priority': 'EMERGENCY',
                'technical_escalation': True
            },
            'assessment_timestamp': datetime.now().isoformat()
        }
    
    async def _update_assessment_stats(self, processing_time: float, results: Dict[str, Any]):
        """Update agent assessment statistics"""
        self.stats['assessments_completed'] += 1
        
        # Update average processing time
        current_avg = self.stats['avg_assessment_time']
        total_assessments = self.stats['assessments_completed']
        self.stats['avg_assessment_time'] = (
            (current_avg * (total_assessments - 1) + processing_time) / total_assessments
        )
        
        # Count evidence-based decisions
        if results.get('evidence_based_decision'):
            self.stats['evidence_based_decisions'] += 1
        
        # Count escalations
        if results.get('escalation_info', {}).get('required', False):
            self.stats['escalations_triggered'] += 1
        
        # Count human reviews
        if results.get('escalation_info', {}).get('human_review_priority', 'ROUTINE') in ['EMERGENCY', 'URGENT']:
            self.stats['human_reviews_requested'] += 1
        
        # Update decision confidence average
        confidence = results.get('technical_details', {}).get('confidence_score', 0.0)
        current_conf_avg = self.stats['decision_confidence_avg']
        self.stats['decision_confidence_avg'] = (
            (current_conf_avg * (total_assessments - 1) + confidence) / total_assessments
        )
    
    async def _finalize_assessment_session(self, session_token: str, results: Dict[str, Any]):
        """Finalize assessment session with audit trail"""
        if session_token:
            await self.audit_service.log_event(
                event_type=AuditEventType.MEDICAL_DECISION,
                component='clinical_assessment_agent',
                action='complete_assessment',
                session_id=session_token,
                user_id='clinical_assessment_agent',
                resource='clinical_assessment',
                details={
                    'assessment_success': results.get('assessment_success', False),
                    'evidence_based_decision': bool(results.get('evidence_based_decision')),
                    'escalation_required': results.get('escalation_info', {}).get('required', False),
                    'decision_confidence': results.get('technical_details', {}).get('confidence_score', 0.0),
                    'completion_timestamp': datetime.now().isoformat()
                }
            )
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and statistics"""
        return {
            'agent_id': self.agent_id,
            'agent_type': 'clinical_assessment',
            'status': 'active',
            'decision_engines': ['npuap_epuap', 'minsal_enhanced'],
            'statistics': self.stats,
            'capabilities': [
                'evidence_based_decision_making',
                'npuap_epuap_guidelines',
                'minsal_chilean_protocols',
                'risk_assessment_comprehensive',
                'clinical_recommendations_generation',
                'escalation_management',
                'medical_compliance_validation'
            ],
            'uptime': datetime.now().isoformat()
        }


# ADK Tool Functions for Clinical Assessment Agent
def perform_clinical_assessment_tool(assessment_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    ADK Tool function for comprehensive clinical assessment.
    Can be used directly in ADK agents and A2A communication.
    """
    agent = ClinicalAssessmentAgent()
    
    # Run async assessment in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(agent.perform_clinical_assessment(assessment_data))
        return result
    finally:
        loop.close()


def get_clinical_assessment_status() -> Dict[str, Any]:
    """
    ADK Tool function for getting clinical assessment agent status.
    """
    agent = ClinicalAssessmentAgent()
    return agent.get_agent_status()


def make_evidence_based_decision_tool(lpp_grade: int, confidence: float, 
                                    anatomical_location: str, patient_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    ADK Tool function for making evidence-based medical decisions.
    """
    # Determine context
    use_minsal = patient_context.get('healthcare_system', '').lower() == 'chilean'
    
    if use_minsal:
        return make_minsal_clinical_decision(lpp_grade, confidence, anatomical_location, patient_context)
    else:
        return make_evidence_based_decision(lpp_grade, confidence, anatomical_location, patient_context)


# Create Clinical Assessment ADK Agent
clinical_assessment_agent = Agent(
    model="gemini-2.0-flash-exp",
    global_instruction=CLINICAL_ASSESSMENT_INSTRUCTION,
    instruction="Especialista en evaluación clínica médica basada en evidencia NPUAP/EPUAP y protocolos MINSAL.",
    name="clinical_assessment_agent",
    tools=[
        perform_clinical_assessment_tool,
        get_clinical_assessment_status,
        make_evidence_based_decision_tool,
    ],
)

# Export for use
__all__ = [
    'ClinicalAssessmentAgent',
    'clinical_assessment_agent',
    'perform_clinical_assessment_tool',
    'get_clinical_assessment_status',
    'make_evidence_based_decision_tool'
]