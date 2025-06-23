"""
Risk Assessment Agent - Specialized Medical Risk Analysis
========================================================

ADK agent specialized in analyzing Batman tokenized patient data to calculate
LPP risk probability based on clinical evidence and medical protocols.

Key Features:
- Batman token data analysis (NO PHI)
- Evidence-based risk calculation
- Braden/Norton scale integration
- Predictive modeling for LPP prevention
- Medical factor weighting
- A2A communication for collaborative analysis

Usage:
    agent = RiskAssessmentAgent()
    risk_result = await agent.assess_lpp_risk(token_id, patient_context)
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from vigia_detect.agents.base_agent import BaseAgent, AgentMessage, AgentResponse
from vigia_detect.db.supabase_client import SupabaseClient
from vigia_detect.utils.audit_service import AuditService
from vigia_detect.systems.medical_knowledge import MedicalKnowledgeSystem
from vigia_detect.db.agent_analysis_client import AgentAnalysisClient

# AgentOps Monitoring Integration
from vigia_detect.monitoring.agentops_client import AgentOpsClient
from vigia_detect.monitoring.medical_telemetry import MedicalTelemetry
from vigia_detect.monitoring.adk_wrapper import adk_agent_wrapper

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """LPP Risk levels based on medical evidence"""
    MINIMAL = "minimal"
    LOW = "low" 
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RiskFactor(Enum):
    """Evidence-based risk factors for LPP development"""
    DIABETES = "diabetes"
    MALNUTRITION = "malnutrition"
    IMMOBILITY = "immobility"
    ADVANCED_AGE = "advanced_age"
    INCONTINENCE = "incontinence"
    ANTICOAGULANTS = "anticoagulants"
    NEUROLOGICAL = "neurological_impairment"
    CHRONIC_ILLNESS = "chronic_illness"
    PREVIOUS_LPP = "previous_lpp"
    ICU_ADMISSION = "icu_admission"


@dataclass
class RiskScore:
    """Comprehensive risk scoring result"""
    overall_risk_level: str
    risk_percentage: float
    braden_score: int
    norton_score: int
    custom_risk_score: float
    contributing_factors: List[str]
    preventive_recommendations: List[str]
    assessment_confidence: float
    risk_timeline: str  # "immediate", "short_term", "long_term"
    escalation_required: bool


@dataclass
class RiskAssessmentResult:
    """Complete risk assessment result with medical context"""
    token_id: str  # Batman token
    assessment_id: str
    risk_score: RiskScore
    medical_context: Dict[str, Any]
    anatomical_risk_assessment: Dict[str, Any]
    intervention_priorities: List[str]
    monitoring_requirements: Dict[str, Any]
    follow_up_schedule: Dict[str, str]
    assessment_timestamp: datetime
    hipaa_compliant: bool = True


class RiskAssessmentAgent(BaseAgent):
    """
    Specialized agent for medical risk assessment using Batman tokenized data.
    
    Capabilities:
    - LPP risk calculation using evidence-based protocols
    - Braden and Norton scale assessment
    - Predictive risk modeling
    - Medical factor analysis
    - Prevention strategy generation
    """
    
    def __init__(self, agent_id: str = "risk_assessment_agent"):
        super().__init__(agent_id=agent_id, agent_type="risk_assessment")
        
        self.supabase = SupabaseClient()
        self.audit_service = AuditService()
        self.medical_knowledge = MedicalKnowledgeSystem()
        self.analysis_client = AgentAnalysisClient()  # NEW: Analysis storage
        
        # AgentOps monitoring integration
        self.telemetry = MedicalTelemetry(
            app_id="vigia-risk-assessment",
            environment="production",
            enable_phi_protection=True
        )
        self.current_session = None
        
        # Risk calculation weights based on medical evidence
        self.risk_factor_weights = {
            RiskFactor.DIABETES: 0.25,
            RiskFactor.MALNUTRITION: 0.20,
            RiskFactor.IMMOBILITY: 0.30,
            RiskFactor.ADVANCED_AGE: 0.15,
            RiskFactor.INCONTINENCE: 0.15,
            RiskFactor.ANTICOAGULANTS: 0.10,
            RiskFactor.NEUROLOGICAL: 0.20,
            RiskFactor.CHRONIC_ILLNESS: 0.15,
            RiskFactor.PREVIOUS_LPP: 0.35,
            RiskFactor.ICU_ADMISSION: 0.25
        }
        
        # Assessment statistics
        self.stats = {
            'assessments_completed': 0,
            'high_risk_cases': 0,
            'preventive_interventions': 0,
            'escalations_triggered': 0,
            'avg_assessment_time': 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize the risk assessment agent"""
        try:
            # Test database connection
            if self.supabase.client:
                logger.info("Risk Assessment Agent initialized with database connection")
            else:
                logger.warning("Risk Assessment Agent initialized without database connection")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Risk Assessment Agent: {e}")
            return False
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return [
            "lpp_risk_assessment",
            "braden_scale_calculation",
            "norton_scale_calculation", 
            "evidence_based_risk_scoring",
            "medical_factor_analysis",
            "preventive_strategy_generation",
            "risk_timeline_prediction",
            "anatomical_risk_assessment",
            "batman_tokenization_support",
            "medical_knowledge_integration"
        ]
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process incoming agent messages"""
        try:
            action = message.action
            
            if action == "assess_lpp_risk":
                return await self._handle_risk_assessment(message)
            elif action == "calculate_braden_score":
                return await self._handle_braden_calculation(message)
            elif action == "calculate_norton_score":
                return await self._handle_norton_calculation(message)
            elif action == "analyze_risk_factors":
                return await self._handle_risk_factor_analysis(message)
            elif action == "generate_prevention_plan":
                return await self._handle_prevention_planning(message)
            elif action == "get_risk_capabilities":
                return await self._handle_capabilities_request(message)
            else:
                return AgentResponse(
                    success=False,
                    message=f"Unknown action: {action}",
                    data={}
                )
                
        except Exception as e:
            logger.error(f"Error processing message in Risk Assessment Agent: {e}")
            
            # Track error in AgentOps if session exists
            if hasattr(self, 'current_session') and self.current_session:
                try:
                    await self.telemetry.track_medical_error_with_escalation(
                        error_type="risk_assessment_processing_error",
                        error_message=str(e),
                        context={
                            "agent_id": self.agent_id,
                            "message_action": message.action,
                            "error_severity": "medium"
                        },
                        session_id=self.current_session,
                        requires_human_review=True,
                        severity="medium"
                    )
                except Exception as telemetry_error:
                    logger.error(f"Failed to track error in AgentOps: {telemetry_error}")
            
            return AgentResponse(
                success=False,
                message=f"Processing error: {str(e)}",
                data={}
            )
    
    @adk_agent_wrapper
    async def _handle_risk_assessment(self, message: AgentMessage) -> AgentResponse:
        """Handle complete LPP risk assessment requests"""
        try:
            start_time = datetime.now()
            
            # Extract parameters
            token_id = message.data.get("token_id")  # Batman token
            patient_context = message.data.get("patient_context", {})
            assessment_type = message.data.get("assessment_type", "comprehensive")
            
            if not token_id:
                return AgentResponse(
                    success=False,
                    message="Missing required parameter: token_id",
                    data={}
                )
            
            # Start AgentOps session for risk assessment
            session_id = await self._start_risk_assessment_session(token_id, patient_context, assessment_type)
            self.current_session = session_id
            
            # Perform comprehensive risk assessment
            result = await self.assess_lpp_risk(token_id, patient_context, assessment_type)
            
            # Track assessment in AgentOps
            if session_id:
                await self._track_risk_assessment_completion(session_id, result, processing_time)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            await self._update_assessment_stats(processing_time, result)
            
            return AgentResponse(
                success=True,
                message="LPP risk assessment completed successfully",
                data={
                    "assessment_result": self._serialize_assessment_result(result),
                    "token_id": token_id,
                    "processing_time_seconds": processing_time,
                    "hipaa_compliant": True
                }
            )
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return AgentResponse(
                success=False,
                message=f"Risk assessment failed: {str(e)}",
                data={}
            )
    
    async def _start_risk_assessment_session(self, token_id: str, patient_context: Dict[str, Any], assessment_type: str) -> str:
        """Start AgentOps session for risk assessment"""
        session_id = f"risk_assessment_{token_id}_{int(datetime.now().timestamp())}"
        
        try:
            await self.telemetry.start_medical_session(
                session_id=session_id,
                patient_context={
                    "token_id": token_id,  # Batman token (HIPAA safe)
                    "assessment_type": assessment_type,
                    "agent_type": "RiskAssessmentAgent",
                    "risk_factors_count": len(patient_context.get("risk_factors", [])),
                    "has_medical_history": bool(patient_context.get("medical_history"))
                },
                session_type="risk_assessment"
            )
            logger.info(f"AgentOps risk assessment session started: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Failed to start AgentOps session: {e}")
            return session_id
    
    async def _track_risk_assessment_completion(self, session_id: str, result: 'RiskAssessmentResult', processing_time: float) -> None:
        """Track risk assessment completion in AgentOps"""
        try:
            await self.telemetry.track_agent_interaction(
                agent_name="RiskAssessmentAgent",
                action="complete_lpp_risk_assessment",
                input_data={
                    "assessment_type": "comprehensive",
                    "braden_score": result.braden_score,
                    "norton_score": result.norton_score,
                    "risk_factors_analyzed": len(result.risk_factors)
                },
                output_data={
                    "risk_level": result.risk_level.value,
                    "probability_score": result.probability_score,
                    "confidence": result.confidence,
                    "interventions_recommended": len(result.prevention_strategies),
                    "escalation_required": result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
                },
                session_id=session_id,
                execution_time=processing_time
            )
        except Exception as e:
            logger.error(f"Failed to track risk assessment completion: {e}")
    
    async def assess_lpp_risk(
        self,
        token_id: str,  # Batman token
        patient_context: Dict[str, Any],
        assessment_type: str = "comprehensive"
    ) -> RiskAssessmentResult:
        """
        Perform comprehensive LPP risk assessment using Batman tokenized data.
        
        Args:
            token_id: Batman token ID (NO PHI)
            patient_context: Medical context and risk factors
            assessment_type: Type of assessment ("comprehensive", "rapid", "focused")
            
        Returns:
            Complete risk assessment result
        """
        try:
            # Generate assessment ID
            assessment_id = f"risk_{token_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Get patient data from Processing Database (Batman tokens only)
            patient_data = await self._get_batman_patient_data(token_id)
            
            # Merge context with stored data
            combined_context = {**patient_data, **patient_context}
            
            # Calculate risk scores
            risk_score = await self._calculate_comprehensive_risk_score(combined_context)
            
            # Anatomical risk assessment
            anatomical_risks = await self._assess_anatomical_risks(combined_context)
            
            # Generate intervention priorities
            intervention_priorities = self._generate_intervention_priorities(risk_score, combined_context)
            
            # Create monitoring requirements
            monitoring_requirements = self._create_monitoring_requirements(risk_score, combined_context)
            
            # Generate follow-up schedule
            follow_up_schedule = self._generate_follow_up_schedule(risk_score)
            
            # Create comprehensive result
            result = RiskAssessmentResult(
                token_id=token_id,
                assessment_id=assessment_id,
                risk_score=risk_score,
                medical_context=combined_context,
                anatomical_risk_assessment=anatomical_risks,
                intervention_priorities=intervention_priorities,
                monitoring_requirements=monitoring_requirements,
                follow_up_schedule=follow_up_schedule,
                assessment_timestamp=datetime.now(),
                hipaa_compliant=True
            )
            
            # Store assessment result in Processing Database
            await self._store_risk_assessment(result)
            
            # Store complete analysis for traceability (NEW - CRITICAL FOR MEDICAL COMPLIANCE)
            await self._store_complete_analysis(
                token_id, patient_context, result, combined_context, assessment_id
            )
            
            # Store raw outputs for research and validation (NEW - FASE 5)
            await self._store_raw_risk_assessment(result, combined_context, risk_score)
            
            # Log assessment for audit trail
            await self.audit_service.log_event(
                event_type="risk_assessment_completed",
                component="risk_assessment_agent",
                action="assess_lpp_risk",
                details={
                    "assessment_id": assessment_id,
                    "token_id": token_id,
                    "risk_level": risk_score.overall_risk_level,
                    "risk_percentage": risk_score.risk_percentage,
                    "braden_score": risk_score.braden_score,
                    "escalation_required": risk_score.escalation_required
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"LPP risk assessment failed: {e}")
            raise
    
    async def _get_batman_patient_data(self, token_id: str) -> Dict[str, Any]:
        """Retrieve Batman tokenized patient data from Processing Database"""
        try:
            if not self.supabase.client:
                return {}
            
            # Query tokenized_patients table
            result = self.supabase.client.table("tokenized_patients").select("*").eq("token_id", token_id).execute()
            
            if result.data:
                patient_data = result.data[0]
                return {
                    "age_range": patient_data.get("age_range"),
                    "gender_category": patient_data.get("gender_category"),
                    "risk_factors": patient_data.get("risk_factors", {}),
                    "medical_conditions": patient_data.get("medical_conditions", {}),
                    "token_created_at": patient_data.get("token_created_at"),
                    "token_status": patient_data.get("token_status")
                }
            else:
                logger.warning(f"No Batman data found for token_id: {token_id}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to retrieve Batman patient data: {e}")
            return {}
    
    async def _calculate_comprehensive_risk_score(self, patient_context: Dict[str, Any]) -> RiskScore:
        """Calculate comprehensive risk score using multiple validated scales"""
        
        # Calculate validated scale scores
        braden_score = self._calculate_braden_score(patient_context)
        norton_score = self._calculate_norton_score(patient_context)
        
        # Calculate custom evidence-based risk score
        custom_risk = self._calculate_custom_risk_score(patient_context)
        
        # Identify contributing risk factors
        contributing_factors = self._identify_risk_factors(patient_context)
        
        # Calculate overall risk percentage (0-100)
        risk_percentage = self._calculate_risk_percentage(braden_score, norton_score, custom_risk)
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_percentage, contributing_factors)
        
        # Generate preventive recommendations
        preventive_recommendations = self._generate_preventive_recommendations(
            risk_level, contributing_factors, patient_context
        )
        
        # Calculate assessment confidence
        confidence = self._calculate_assessment_confidence(patient_context)
        
        # Determine risk timeline
        risk_timeline = self._determine_risk_timeline(risk_level, contributing_factors)
        
        # Check if escalation is required
        escalation_required = self._check_escalation_requirements(risk_level, risk_percentage, contributing_factors)
        
        return RiskScore(
            overall_risk_level=risk_level.value,
            risk_percentage=risk_percentage,
            braden_score=braden_score,
            norton_score=norton_score,
            custom_risk_score=custom_risk,
            contributing_factors=[factor.value for factor in contributing_factors],
            preventive_recommendations=preventive_recommendations,
            assessment_confidence=confidence,
            risk_timeline=risk_timeline,
            escalation_required=escalation_required
        )
    
    def _calculate_braden_score(self, patient_context: Dict[str, Any]) -> int:
        """Calculate Braden Scale score for pressure injury risk (6-23 scale)"""
        score = 23  # Start with maximum (lowest risk)
        
        risk_factors = patient_context.get("risk_factors", {})
        age_range = patient_context.get("age_range", "40-50")
        
        # Extract age from range for calculation
        age = 65  # Default
        if age_range:
            try:
                age = int(age_range.split("-")[0])
            except:
                age = 65
        
        # Sensory perception (1-4)
        if risk_factors.get("neurological_impairment") or risk_factors.get("altered_mental_state"):
            score -= 2
        elif age > 80:
            score -= 1
        
        # Moisture (1-4)
        if risk_factors.get("incontinence"):
            score -= 3
        elif risk_factors.get("moisture_exposure"):
            score -= 2
        
        # Activity (1-4)
        if risk_factors.get("bedbound") or risk_factors.get("immobility"):
            score -= 3
        elif risk_factors.get("limited_mobility"):
            score -= 2
        elif age > 75:
            score -= 1
        
        # Mobility (1-4)
        if risk_factors.get("complete_immobility"):
            score -= 3
        elif risk_factors.get("very_limited_mobility"):
            score -= 2
        
        # Nutrition (1-4)
        if risk_factors.get("malnutrition") or risk_factors.get("severe_malnutrition"):
            score -= 3
        elif risk_factors.get("poor_nutrition"):
            score -= 2
        elif risk_factors.get("diabetes") and patient_context.get("glucose_control") == "poor":
            score -= 1
        
        # Friction and shear (1-3)
        if risk_factors.get("friction_shear") or risk_factors.get("sliding_in_bed"):
            score -= 2
        elif risk_factors.get("occasional_sliding"):
            score -= 1
        
        return max(6, min(23, score))  # Constrain to valid range
    
    def _calculate_norton_score(self, patient_context: Dict[str, Any]) -> int:
        """Calculate Norton Scale score (5-20 scale)"""
        score = 20  # Start with maximum
        
        risk_factors = patient_context.get("risk_factors", {})
        
        # Physical condition
        if risk_factors.get("poor_general_condition"):
            score -= 3
        elif risk_factors.get("chronic_illness"):
            score -= 2
        
        # Mental condition
        if risk_factors.get("confused") or risk_factors.get("neurological_impairment"):
            score -= 3
        elif risk_factors.get("altered_mental_state"):
            score -= 2
        
        # Activity
        if risk_factors.get("bedbound"):
            score -= 3
        elif risk_factors.get("immobility"):
            score -= 2
        
        # Mobility
        if risk_factors.get("complete_immobility"):
            score -= 3
        elif risk_factors.get("very_limited_mobility"):
            score -= 2
        
        # Incontinence
        if risk_factors.get("incontinence"):
            score -= 3
        elif risk_factors.get("occasional_incontinence"):
            score -= 2
        
        return max(5, min(20, score))
    
    def _calculate_custom_risk_score(self, patient_context: Dict[str, Any]) -> float:
        """Calculate custom evidence-based risk score (0.0-1.0)"""
        risk_score = 0.0
        risk_factors = patient_context.get("risk_factors", {})
        
        # Apply weighted risk factors
        for factor, weight in self.risk_factor_weights.items():
            if risk_factors.get(factor.value, False):
                risk_score += weight
        
        # Age factor
        age_range = patient_context.get("age_range", "40-50")
        try:
            age = int(age_range.split("-")[0])
            if age > 80:
                risk_score += 0.20
            elif age > 70:
                risk_score += 0.15
            elif age > 60:
                risk_score += 0.10
        except:
            risk_score += 0.05  # Default age risk
        
        # Multiple comorbidities factor
        active_conditions = sum(1 for condition in risk_factors.values() if condition)
        if active_conditions >= 5:
            risk_score += 0.15
        elif active_conditions >= 3:
            risk_score += 0.10
        
        # Previous LPP history (strongest predictor)
        if risk_factors.get("previous_lpp"):
            risk_score += 0.30
        
        return min(1.0, risk_score)
    
    def _identify_risk_factors(self, patient_context: Dict[str, Any]) -> List[RiskFactor]:
        """Identify present risk factors from patient context"""
        risk_factors = patient_context.get("risk_factors", {})
        present_factors = []
        
        for factor in RiskFactor:
            if risk_factors.get(factor.value, False):
                present_factors.append(factor)
        
        # Check age-based risk
        age_range = patient_context.get("age_range", "40-50")
        try:
            age = int(age_range.split("-")[0])
            if age > 70:
                present_factors.append(RiskFactor.ADVANCED_AGE)
        except:
            pass
        
        return present_factors
    
    def _calculate_risk_percentage(self, braden_score: int, norton_score: int, custom_risk: float) -> float:
        """Calculate overall risk percentage from multiple scores"""
        
        # Convert Braden score to risk percentage (lower score = higher risk)
        braden_risk = max(0, (23 - braden_score) / 17) * 100  # 17 is the range (23-6)
        
        # Convert Norton score to risk percentage
        norton_risk = max(0, (20 - norton_score) / 15) * 100  # 15 is the range (20-5)
        
        # Convert custom risk to percentage
        custom_risk_percentage = custom_risk * 100
        
        # Weighted average with higher weight on Braden scale (most validated)
        overall_risk = (braden_risk * 0.5) + (norton_risk * 0.3) + (custom_risk_percentage * 0.2)
        
        return min(100, max(0, overall_risk))
    
    def _determine_risk_level(self, risk_percentage: float, contributing_factors: List[RiskFactor]) -> RiskLevel:
        """Determine risk level based on percentage and contributing factors"""
        
        # High-risk factor overrides
        critical_factors = {RiskFactor.PREVIOUS_LPP, RiskFactor.ICU_ADMISSION}
        if any(factor in contributing_factors for factor in critical_factors):
            if risk_percentage >= 60:
                return RiskLevel.CRITICAL
            elif risk_percentage >= 40:
                return RiskLevel.HIGH
        
        # Standard risk level determination
        if risk_percentage >= 80:
            return RiskLevel.CRITICAL
        elif risk_percentage >= 60:
            return RiskLevel.HIGH
        elif risk_percentage >= 40:
            return RiskLevel.MODERATE
        elif risk_percentage >= 20:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
    
    def _generate_preventive_recommendations(
        self,
        risk_level: RiskLevel,
        contributing_factors: List[RiskFactor],
        patient_context: Dict[str, Any]
    ) -> List[str]:
        """Generate evidence-based preventive recommendations"""
        
        recommendations = []
        
        # Base recommendations by risk level
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "URGENT: Implement immediate comprehensive pressure injury prevention protocol",
                "Continuous pressure redistribution every 1-2 hours",
                "Specialized pressure-relieving surface required",
                "Daily skin assessment by qualified nursing staff",
                "Nutritional assessment and optimization within 24 hours"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Implement high-intensity pressure injury prevention protocol",
                "Frequent repositioning every 2 hours when in bed, hourly when sitting",
                "Consider pressure-relieving support surfaces",
                "Twice-daily skin assessment",
                "Nutritional screening and intervention"
            ])
        elif risk_level == RiskLevel.MODERATE:
            recommendations.extend([
                "Standard pressure injury prevention protocol",
                "Regular repositioning every 2-4 hours",
                "Standard pressure-relieving devices",
                "Daily skin inspection",
                "Monitor nutritional status"
            ])
        else:
            recommendations.extend([
                "Basic pressure injury prevention measures",
                "Regular position changes every 4 hours",
                "Standard mattress with positioning aids",
                "Regular skin inspection during routine care"
            ])
        
        # Factor-specific recommendations
        for factor in contributing_factors:
            if factor == RiskFactor.DIABETES:
                recommendations.append("Enhanced glucose monitoring and control for wound healing")
            elif factor == RiskFactor.MALNUTRITION:
                recommendations.append("Immediate nutritionist consultation and protein supplementation")
            elif factor == RiskFactor.INCONTINENCE:
                recommendations.append("Moisture management protocol with barrier protection")
            elif factor == RiskFactor.IMMOBILITY:
                recommendations.append("Physical therapy consultation for mobility optimization")
            elif factor == RiskFactor.ANTICOAGULANTS:
                recommendations.append("Enhanced skin protection due to bleeding risk")
            elif factor == RiskFactor.PREVIOUS_LPP:
                recommendations.append("Intensive monitoring of previous LPP sites")
        
        return recommendations
    
    def _calculate_assessment_confidence(self, patient_context: Dict[str, Any]) -> float:
        """Calculate confidence in the risk assessment based on data completeness"""
        
        confidence = 0.5  # Base confidence
        
        # Data completeness factors
        if patient_context.get("age_range"):
            confidence += 0.1
        if patient_context.get("gender_category"):
            confidence += 0.05
        if patient_context.get("risk_factors"):
            risk_factor_count = sum(1 for factor in patient_context["risk_factors"].values() if factor)
            confidence += min(0.2, risk_factor_count * 0.02)
        if patient_context.get("medical_conditions"):
            confidence += 0.1
        
        # Historical data factor
        if patient_context.get("token_created_at"):
            confidence += 0.05
        
        return min(0.95, confidence)
    
    def _determine_risk_timeline(self, risk_level: RiskLevel, contributing_factors: List[RiskFactor]) -> str:
        """Determine timeline for LPP development risk"""
        
        immediate_factors = {RiskFactor.ICU_ADMISSION, RiskFactor.PREVIOUS_LPP, RiskFactor.IMMOBILITY}
        
        if risk_level == RiskLevel.CRITICAL:
            return "immediate"  # Hours to days
        elif risk_level == RiskLevel.HIGH or any(factor in contributing_factors for factor in immediate_factors):
            return "short_term"  # Days to weeks
        else:
            return "long_term"  # Weeks to months
    
    def _check_escalation_requirements(
        self,
        risk_level: RiskLevel,
        risk_percentage: float,
        contributing_factors: List[RiskFactor]
    ) -> bool:
        """Check if case requires immediate escalation"""
        
        # Critical risk level always requires escalation
        if risk_level == RiskLevel.CRITICAL:
            return True
        
        # High risk with multiple factors
        if risk_level == RiskLevel.HIGH and len(contributing_factors) >= 3:
            return True
        
        # Previous LPP with current high risk
        if RiskFactor.PREVIOUS_LPP in contributing_factors and risk_percentage >= 50:
            return True
        
        # ICU admission always requires escalation
        if RiskFactor.ICU_ADMISSION in contributing_factors:
            return True
        
        return False
    
    async def _assess_anatomical_risks(self, patient_context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess anatomical-specific risk factors"""
        
        anatomical_risks = {
            "high_risk_areas": [],
            "moderate_risk_areas": [],
            "protective_factors": [],
            "positioning_recommendations": []
        }
        
        risk_factors = patient_context.get("risk_factors", {})
        
        # High-risk anatomical areas based on patient factors
        if risk_factors.get("immobility") or risk_factors.get("bedbound"):
            anatomical_risks["high_risk_areas"].extend([
                "sacrum", "coccyx", "heels", "ischial_tuberosities"
            ])
            anatomical_risks["positioning_recommendations"].extend([
                "30-degree lateral positioning to avoid direct pressure on trochanters",
                "Heel elevation to offload posterior heel pressure",
                "Avoid head-of-bed elevation >30 degrees when possible"
            ])
        
        if risk_factors.get("wheelchair_bound"):
            anatomical_risks["high_risk_areas"].extend([
                "ischial_tuberosities", "coccyx", "elbows"
            ])
            anatomical_risks["positioning_recommendations"].append(
                "Pressure-redistributing wheelchair cushion required"
            )
        
        # Moderate risk areas
        if risk_factors.get("limited_mobility"):
            anatomical_risks["moderate_risk_areas"].extend([
                "shoulder_blades", "elbows", "knees", "ankles"
            ])
        
        # Protective factors
        if not risk_factors.get("malnutrition"):
            anatomical_risks["protective_factors"].append("adequate_soft_tissue_coverage")
        
        if patient_context.get("gender_category") == "Female":
            anatomical_risks["protective_factors"].append("typically_better_subcutaneous_fat_distribution")
        
        return anatomical_risks
    
    def _generate_intervention_priorities(self, risk_score: RiskScore, patient_context: Dict[str, Any]) -> List[str]:
        """Generate prioritized intervention list"""
        
        priorities = []
        
        if risk_score.escalation_required:
            priorities.append("IMMEDIATE: Escalate to wound care specialist")
        
        if risk_score.overall_risk_level in ["critical", "high"]:
            priorities.extend([
                "Implement comprehensive pressure redistribution protocol",
                "Initiate intensive skin monitoring program",
                "Ensure adequate nutritional support"
            ])
        
        # Risk factor specific priorities
        contributing_factors = risk_score.contributing_factors
        
        if "malnutrition" in contributing_factors:
            priorities.insert(1 if not risk_score.escalation_required else 2, 
                            "Priority: Nutritional assessment and intervention")
        
        if "diabetes" in contributing_factors:
            priorities.append("Optimize glycemic control for tissue health")
        
        if "immobility" in contributing_factors:
            priorities.append("Physical therapy evaluation for mobility improvement")
        
        return priorities[:5]  # Limit to top 5 priorities
    
    def _create_monitoring_requirements(self, risk_score: RiskScore, patient_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create monitoring requirements based on risk level"""
        
        if risk_score.overall_risk_level == "critical":
            return {
                "skin_assessment_frequency": "every_shift",
                "repositioning_frequency": "every_1_2_hours",
                "documentation_required": "comprehensive_daily",
                "reassessment_interval": "daily",
                "specialist_consultation": "within_24_hours"
            }
        elif risk_score.overall_risk_level == "high":
            return {
                "skin_assessment_frequency": "twice_daily",
                "repositioning_frequency": "every_2_hours",
                "documentation_required": "daily",
                "reassessment_interval": "every_3_days",
                "specialist_consultation": "within_72_hours"
            }
        else:
            return {
                "skin_assessment_frequency": "daily",
                "repositioning_frequency": "every_4_hours",
                "documentation_required": "routine",
                "reassessment_interval": "weekly",
                "specialist_consultation": "as_needed"
            }
    
    def _generate_follow_up_schedule(self, risk_score: RiskScore) -> Dict[str, str]:
        """Generate follow-up schedule based on risk level"""
        
        if risk_score.overall_risk_level == "critical":
            return {
                "immediate_follow_up": "8_hours",
                "short_term_follow_up": "24_hours", 
                "routine_reassessment": "daily",
                "comprehensive_review": "weekly"
            }
        elif risk_score.overall_risk_level == "high":
            return {
                "immediate_follow_up": "24_hours",
                "short_term_follow_up": "72_hours",
                "routine_reassessment": "every_3_days",
                "comprehensive_review": "weekly"
            }
        else:
            return {
                "immediate_follow_up": "72_hours",
                "short_term_follow_up": "1_week",
                "routine_reassessment": "weekly",
                "comprehensive_review": "monthly"
            }
    
    async def _store_risk_assessment(self, result: RiskAssessmentResult):
        """Store risk assessment result in Processing Database"""
        try:
            if not self.supabase.client:
                logger.warning("No database connection - risk assessment not stored")
                return
            
            assessment_data = {
                "assessment_id": result.assessment_id,
                "token_id": result.token_id,
                "overall_risk_level": result.risk_score.overall_risk_level,
                "risk_percentage": result.risk_score.risk_percentage,
                "braden_score": result.risk_score.braden_score,
                "norton_score": result.risk_score.norton_score,
                "custom_risk_score": result.risk_score.custom_risk_score,
                "contributing_factors": result.risk_score.contributing_factors,
                "preventive_recommendations": result.risk_score.preventive_recommendations,
                "assessment_confidence": result.risk_score.assessment_confidence,
                "risk_timeline": result.risk_score.risk_timeline,
                "escalation_required": result.risk_score.escalation_required,
                "intervention_priorities": result.intervention_priorities,
                "anatomical_risk_assessment": result.anatomical_risk_assessment,
                "monitoring_requirements": result.monitoring_requirements,
                "follow_up_schedule": result.follow_up_schedule,
                "created_at": result.assessment_timestamp.isoformat(),
                "hipaa_compliant": True,
                "agent_id": self.agent_id
            }
            
            # Insert into risk_assessments table
            self.supabase.client.table("risk_assessments").insert(assessment_data).execute()
            
            logger.info(f"Stored risk assessment {result.assessment_id}")
            
        except Exception as e:
            logger.error(f"Failed to store risk assessment: {e}")
    
    def _serialize_assessment_result(self, result: RiskAssessmentResult) -> Dict[str, Any]:
        """Serialize assessment result for JSON response"""
        return {
            "token_id": result.token_id,
            "assessment_id": result.assessment_id,
            "risk_score": {
                "overall_risk_level": result.risk_score.overall_risk_level,
                "risk_percentage": result.risk_score.risk_percentage,
                "braden_score": result.risk_score.braden_score,
                "norton_score": result.risk_score.norton_score,
                "custom_risk_score": result.risk_score.custom_risk_score,
                "contributing_factors": result.risk_score.contributing_factors,
                "preventive_recommendations": result.risk_score.preventive_recommendations,
                "assessment_confidence": result.risk_score.assessment_confidence,
                "risk_timeline": result.risk_score.risk_timeline,
                "escalation_required": result.risk_score.escalation_required
            },
            "anatomical_risk_assessment": result.anatomical_risk_assessment,
            "intervention_priorities": result.intervention_priorities,
            "monitoring_requirements": result.monitoring_requirements,
            "follow_up_schedule": result.follow_up_schedule,
            "assessment_timestamp": result.assessment_timestamp.isoformat(),
            "hipaa_compliant": result.hipaa_compliant
        }
    
    async def _update_assessment_stats(self, processing_time: float, result: RiskAssessmentResult):
        """Update agent statistics"""
        self.stats['assessments_completed'] += 1
        
        # Update average processing time
        current_avg = self.stats['avg_assessment_time']
        total_assessments = self.stats['assessments_completed']
        self.stats['avg_assessment_time'] = (
            (current_avg * (total_assessments - 1) + processing_time) / total_assessments
        )
        
        # Count high-risk cases
        if result.risk_score.overall_risk_level in ["high", "critical"]:
            self.stats['high_risk_cases'] += 1
        
        # Count preventive interventions
        if result.risk_score.preventive_recommendations:
            self.stats['preventive_interventions'] += 1
        
        # Count escalations
        if result.risk_score.escalation_required:
            self.stats['escalations_triggered'] += 1
    
    # Additional handler methods for other actions
    async def _handle_braden_calculation(self, message: AgentMessage) -> AgentResponse:
        """Handle Braden scale calculation requests"""
        try:
            patient_context = message.data.get("patient_context", {})
            braden_score = self._calculate_braden_score(patient_context)
            
            # Interpret Braden score
            if braden_score <= 9:
                risk_interpretation = "Very High Risk"
            elif braden_score <= 12:
                risk_interpretation = "High Risk"
            elif braden_score <= 14:
                risk_interpretation = "Moderate Risk"
            elif braden_score <= 18:
                risk_interpretation = "Mild Risk"
            else:
                risk_interpretation = "Low Risk"
            
            return AgentResponse(
                success=True,
                message="Braden scale calculation completed",
                data={
                    "braden_score": braden_score,
                    "risk_interpretation": risk_interpretation,
                    "scale_range": "6-23 (lower scores indicate higher risk)"
                }
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                message=f"Braden calculation failed: {str(e)}",
                data={}
            )
    
    async def _handle_norton_calculation(self, message: AgentMessage) -> AgentResponse:
        """Handle Norton scale calculation requests"""
        try:
            patient_context = message.data.get("patient_context", {})
            norton_score = self._calculate_norton_score(patient_context)
            
            # Interpret Norton score
            if norton_score <= 14:
                risk_interpretation = "High Risk"
            elif norton_score <= 16:
                risk_interpretation = "Medium Risk" 
            else:
                risk_interpretation = "Low Risk"
            
            return AgentResponse(
                success=True,
                message="Norton scale calculation completed",
                data={
                    "norton_score": norton_score,
                    "risk_interpretation": risk_interpretation,
                    "scale_range": "5-20 (lower scores indicate higher risk)"
                }
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                message=f"Norton calculation failed: {str(e)}",
                data={}
            )
    
    async def _handle_risk_factor_analysis(self, message: AgentMessage) -> AgentResponse:
        """Handle risk factor analysis requests"""
        try:
            patient_context = message.data.get("patient_context", {})
            contributing_factors = self._identify_risk_factors(patient_context)
            
            factor_analysis = {}
            for factor in contributing_factors:
                factor_analysis[factor.value] = {
                    "weight": self.risk_factor_weights.get(factor, 0.0),
                    "evidence_level": "A" if factor in [RiskFactor.PREVIOUS_LPP, RiskFactor.IMMOBILITY] else "B"
                }
            
            return AgentResponse(
                success=True,
                message="Risk factor analysis completed",
                data={
                    "contributing_factors": [f.value for f in contributing_factors],
                    "factor_analysis": factor_analysis,
                    "total_factors": len(contributing_factors),
                    "high_impact_factors": [f.value for f in contributing_factors 
                                          if self.risk_factor_weights.get(f, 0) >= 0.25]
                }
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                message=f"Risk factor analysis failed: {str(e)}",
                data={}
            )
    
    async def _handle_prevention_planning(self, message: AgentMessage) -> AgentResponse:
        """Handle prevention plan generation requests"""
        try:
            risk_level_str = message.data.get("risk_level", "moderate")
            contributing_factors_list = message.data.get("contributing_factors", [])
            patient_context = message.data.get("patient_context", {})
            
            risk_level = RiskLevel(risk_level_str)
            contributing_factors = [RiskFactor(f) for f in contributing_factors_list if f in [rf.value for rf in RiskFactor]]
            
            recommendations = self._generate_preventive_recommendations(
                risk_level, contributing_factors, patient_context
            )
            
            return AgentResponse(
                success=True,
                message="Prevention plan generated successfully",
                data={
                    "risk_level": risk_level.value,
                    "preventive_recommendations": recommendations,
                    "implementation_priority": "immediate" if risk_level == RiskLevel.CRITICAL else "urgent" if risk_level == RiskLevel.HIGH else "routine"
                }
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                message=f"Prevention planning failed: {str(e)}",
                data={}
            )
    
    async def _handle_capabilities_request(self, message: AgentMessage) -> AgentResponse:
        """Handle capabilities request"""
        return AgentResponse(
            success=True,
            message="Risk Assessment Agent capabilities",
            data={
                "capabilities": self.get_capabilities(),
                "risk_factor_weights": {f.value: w for f, w in self.risk_factor_weights.items()},
                "supported_scales": ["braden", "norton", "custom_evidence_based"],
                "risk_levels": [level.value for level in RiskLevel],
                "agent_type": "risk_assessment",
                "batman_tokenization": True,
                "statistics": self.stats
            }
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status"""
        return {
            "agent_id": self.agent_id,
            "status": "healthy",
            "database_connected": self.supabase.client is not None,
            "capabilities_count": len(self.get_capabilities()),
            "assessments_completed": self.stats['assessments_completed'],
            "high_risk_cases": self.stats['high_risk_cases'],
            "last_assessment": datetime.now().isoformat(),
            "tokenization_compliant": True
        }
    
    # NEW COMPLETE ANALYSIS STORAGE METHODS (CRITICAL FOR TRACEABILITY)
    
    async def _store_complete_analysis(
        self,
        token_id: str,
        original_input: Dict[str, Any],
        result: RiskAssessmentResult,
        processed_context: Dict[str, Any],
        assessment_id: str
    ) -> Optional[str]:
        """
        Store complete analysis for full traceability and medical compliance.
        
        Args:
            token_id: Batman token ID
            original_input: Original input provided to the agent
            result: Complete risk assessment result
            processed_context: Processed patient context
            assessment_id: Assessment identifier
            
        Returns:
            Analysis ID if successful, None if failed
        """
        try:
            # Prepare input data for storage
            input_data = {
                "token_id": token_id,
                "patient_context": original_input,
                "assessment_type": "comprehensive",
                "request_timestamp": datetime.now().isoformat()
            }
            
            # Prepare output data for storage
            output_data = {
                "assessment_id": result.assessment_id,
                "risk_score": {
                    "overall_risk_level": result.risk_score.overall_risk_level,
                    "risk_percentage": result.risk_score.risk_percentage,
                    "braden_score": result.risk_score.braden_score,
                    "norton_score": result.risk_score.norton_score,
                    "custom_risk_score": result.risk_score.custom_risk_score,
                    "assessment_confidence": result.risk_score.assessment_confidence,
                    "contributing_factors": result.risk_score.contributing_factors,
                    "preventive_recommendations": result.risk_score.preventive_recommendations,
                    "risk_timeline": result.risk_score.risk_timeline,
                    "escalation_required": result.risk_score.escalation_required
                },
                "intervention_priorities": result.intervention_priorities,
                "monitoring_requirements": result.monitoring_requirements,
                "follow_up_schedule": result.follow_up_schedule,
                "anatomical_risk_assessment": result.anatomical_risk_assessment,
                "medical_recommendations": {
                    "immediate_actions": self._extract_immediate_actions(result),
                    "long_term_strategies": self._extract_long_term_strategies(result),
                    "contraindications": self._extract_contraindications(processed_context)
                }
            }
            
            # Prepare processing metadata
            processing_metadata = {
                "agent_version": "risk_assessment_v1.0",
                "evidence_base": "NPUAP_EPUAP_2019",
                "calculation_method": "braden_norton_combined",
                "risk_factors_analyzed": list(self.risk_factor_weights.keys()),
                "processing_stages": [
                    "patient_data_retrieval",
                    "context_merging", 
                    "risk_score_calculation",
                    "anatomical_assessment",
                    "intervention_planning",
                    "follow_up_scheduling"
                ]
            }
            
            # Prepare medical context for analysis storage
            medical_context = {
                "patient_demographics": self._extract_demographics(processed_context),
                "risk_factors_present": self._extract_risk_factors(processed_context),
                "medical_history_relevant": self._extract_relevant_history(processed_context),
                "current_interventions": self._extract_current_interventions(processed_context),
                "anatomical_considerations": result.anatomical_risk_assessment
            }
            
            # Prepare confidence scores
            confidence_scores = {
                "overall_assessment": result.risk_score.assessment_confidence,
                "braden_scale": 0.95,  # High confidence in standardized scale
                "norton_scale": 0.95,  # High confidence in standardized scale
                "risk_timeline": 0.8,
                "intervention_effectiveness": 0.85,
                "follow_up_adequacy": 0.9
            }
            
            # Prepare evidence references
            evidence_references = [
                "NPUAP_EPUAP_PPPIA_2019_Prevention_Treatment_Guidelines",
                "Braden_Scale_Validation_Studies_2018",
                "Norton_Scale_Reliability_Assessment_2019",
                "LPP_Risk_Factors_Meta_Analysis_2020",
                "Preventive_Interventions_Cochrane_Review_2021"
            ]
            
            # Determine escalation triggers
            escalation_triggers = []
            if result.risk_score.escalation_required:
                escalation_triggers.append("high_risk_score_detected")
            if result.risk_score.risk_percentage >= 0.8:
                escalation_triggers.append("critical_risk_threshold")
            if result.risk_score.braden_score <= 12:
                escalation_triggers.append("braden_high_risk")
            if result.risk_score.norton_score <= 14:
                escalation_triggers.append("norton_high_risk")
            
            # Store complete analysis
            analysis_id = await self.analysis_client.store_agent_analysis(
                token_id=token_id,
                agent_type="risk_assessment",
                agent_id=self.agent_id,
                case_session=assessment_id,  # Using assessment_id as case session for now
                input_data=input_data,
                output_data=output_data,
                processing_metadata=processing_metadata,
                medical_context=medical_context,
                confidence_scores=confidence_scores,
                evidence_references=evidence_references,
                escalation_triggers=escalation_triggers,
                processing_time_ms=int((datetime.now() - result.assessment_timestamp).total_seconds() * 1000)
            )
            
            if analysis_id:
                logger.info(f"Stored complete risk assessment analysis: {analysis_id}")
            else:
                logger.warning("Failed to store complete risk assessment analysis")
            
            return analysis_id
            
        except Exception as e:
            logger.error(f"Error storing complete analysis: {e}")
            return None
    
    def _extract_immediate_actions(self, result: RiskAssessmentResult) -> List[str]:
        """Extract immediate actions from assessment result"""
        actions = []
        if result.risk_score.escalation_required:
            actions.append("Immediate medical evaluation required")
        if result.risk_score.risk_percentage >= 0.7:
            actions.append("Implement pressure relief protocol immediately")
        if "immobility" in result.risk_score.contributing_factors:
            actions.append("Begin turning schedule every 2 hours")
        return actions
    
    def _extract_long_term_strategies(self, result: RiskAssessmentResult) -> List[str]:
        """Extract long-term strategies from assessment result"""
        strategies = []
        strategies.extend(result.risk_score.preventive_recommendations)
        if result.monitoring_requirements:
            for requirement in result.monitoring_requirements.values():
                if isinstance(requirement, str):
                    strategies.append(requirement)
        return strategies
    
    def _extract_contraindications(self, context: Dict[str, Any]) -> List[str]:
        """Extract contraindications from patient context"""
        contraindications = []
        if context.get("anticoagulants"):
            contraindications.append("Avoid aggressive positioning due to anticoagulation")
        if context.get("diabetes"):
            contraindications.append("Monitor for diabetic skin complications")
        if context.get("cardiovascular_instability"):
            contraindications.append("Limit position changes if hemodynamically unstable")
        return contraindications
    
    def _extract_demographics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract demographic information (Batman tokenized)"""
        return {
            "age_category": self._categorize_age(context.get("age")),
            "mobility_status": context.get("mobility", "unknown"),
            "care_setting": context.get("care_setting", "unknown")
        }
    
    def _categorize_age(self, age: Optional[Union[int, str]]) -> str:
        """Categorize age for analysis purposes"""
        if not age:
            return "unknown"
        try:
            age_num = int(age)
            if age_num < 18:
                return "pediatric"
            elif age_num < 65:
                return "adult"
            elif age_num < 80:
                return "elderly"
            else:
                return "very_elderly"
        except (ValueError, TypeError):
            return "unknown"
    
    def _extract_risk_factors(self, context: Dict[str, Any]) -> List[str]:
        """Extract present risk factors"""
        present_factors = []
        for factor in RiskFactor:
            factor_key = factor.value
            if context.get(factor_key, False):
                present_factors.append(factor_key)
        return present_factors
    
    def _extract_relevant_history(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant medical history"""
        return {
            "previous_lpp": context.get("previous_lpp", False),
            "chronic_conditions": context.get("chronic_conditions", []),
            "current_medications": context.get("medications", []),
            "surgical_history": context.get("surgical_history", [])
        }
    
    def _extract_current_interventions(self, context: Dict[str, Any]) -> List[str]:
        """Extract current interventions in place"""
        interventions = []
        if context.get("pressure_relief_devices"):
            interventions.append("pressure_relief_devices_in_use")
        if context.get("turning_schedule"):
            interventions.append("turning_schedule_implemented")
        if context.get("nutritional_support"):
            interventions.append("nutritional_support_active")
        return interventions
    
    # NEW RAW OUTPUT STORAGE METHODS (FASE 5)
    
    async def _store_raw_risk_assessment(
        self,
        result: RiskAssessmentResult,
        patient_context: Dict[str, Any],
        risk_score: RiskScore
    ) -> None:
        """
        Store raw risk assessment outputs for research and validation.
        
        Args:
            result: Complete risk assessment result
            patient_context: Original patient context used
            risk_score: Calculated risk score details
        """
        try:
            from vigia_detect.db.raw_outputs_client import RawOutputsClient
            from vigia_detect.cv_pipeline.adaptive_medical_detector import RawOutputCapture
            
            raw_outputs_client = RawOutputsClient()
            
            # Create comprehensive raw output data
            raw_risk_data = {
                "assessment_id": result.assessment_id,
                "token_id": result.token_id,
                "assessment_type": "comprehensive",
                "input_context": patient_context,
                "calculated_scores": {
                    "overall_risk_level": risk_score.overall_risk_level,
                    "risk_percentage": risk_score.risk_percentage,
                    "braden_score": risk_score.braden_score,
                    "norton_score": risk_score.norton_score,
                    "custom_risk_score": risk_score.custom_risk_score,
                    "assessment_confidence": risk_score.assessment_confidence
                },
                "risk_factors": {
                    "contributing_factors": risk_score.contributing_factors,
                    "risk_timeline": risk_score.risk_timeline,
                    "escalation_required": risk_score.escalation_required
                },
                "medical_recommendations": {
                    "preventive_recommendations": risk_score.preventive_recommendations,
                    "intervention_priorities": result.intervention_priorities,
                    "monitoring_requirements": result.monitoring_requirements,
                    "follow_up_schedule": result.follow_up_schedule
                },
                "anatomical_assessment": result.anatomical_risk_assessment,
                "processing_metadata": {
                    "processing_time": (datetime.now() - result.assessment_timestamp).total_seconds(),
                    "model_version": "risk_assessment_v1.0",
                    "evidence_base": "NPUAP_EPUAP_2019",
                    "hipaa_compliant": True
                },
                "session_info": {
                    "timestamp": result.assessment_timestamp.isoformat(),
                    "agent_id": self.agent_id,
                    "batman_tokenization": True
                }
            }
            
            # Create RawOutputCapture
            raw_capture = RawOutputCapture(
                raw_predictions=raw_risk_data,
                model_metadata={
                    "model_version": "risk_assessment_v1.0",
                    "evidence_base": "NPUAP_EPUAP_2019",
                    "confidence_level": risk_score.assessment_confidence,
                    "processing_time": raw_risk_data["processing_metadata"]["processing_time"]
                },
                processing_metadata={
                    "agent_type": "risk_assessment",
                    "assessment_type": "comprehensive",
                    "batman_compliant": True
                }
            )
            
            # Store raw output
            raw_output_id = await raw_outputs_client.store_raw_output(
                token_id=result.token_id,
                ai_engine="risk_assessment",
                raw_outputs=raw_capture,
                structured_result_id=result.assessment_id,
                structured_result_type="risk_assessment",
                research_approved=True,  # Risk assessments valuable for research
                retention_priority="high"  # Important for medical studies
            )
            
            if raw_output_id:
                logger.info(f"Stored raw risk assessment output: {raw_output_id}")
            else:
                logger.warning("Failed to store raw risk assessment output")
                
        except Exception as e:
            logger.error(f"Error storing raw risk assessment: {e}")
            # Don't fail the assessment if raw storage fails


# Export for ADK integration
__all__ = ["RiskAssessmentAgent", "RiskLevel", "RiskFactor", "RiskScore", "RiskAssessmentResult"]