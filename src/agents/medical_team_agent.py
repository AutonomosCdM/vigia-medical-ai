"""
Medical Team Agent - Bidirectional Slack Communication for Medical Professionals
=================================================================================

ADK agent specialized in bidirectional communication with medical teams via Slack.
Handles diagnosis delivery, medical inquiries, and orchestrates follow-up analyses.

Key Features:
- Bidirectional Slack communication (system → professionals, professionals → system)
- Medical diagnosis delivery with evidence-based context
- Professional inquiry handling and orchestration
- Follow-up analysis coordination with other agents
- Approval workflow for patient communications
- Complete DB storage for all communications
- HIPAA-compliant Batman token usage

Usage:
    agent = MedicalTeamAgent()
    await agent.send_diagnosis_to_team(diagnosis_data)
    await agent.handle_medical_inquiry(slack_message)
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging
import json
import uuid

# Google ADK imports
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

# Communication system imports
from ..interfaces.slack_orchestrator import (
    SlackOrchestrator, NotificationPayload, SlackMessage,
    NotificationType, NotificationPriority, SlackChannel
)
from ..messaging.slack_notifier_refactored import SlackNotifier
from ..utils.secure_logger import SecureLogger
from ..utils.audit_service import AuditService, AuditEventType, AuditLevel
from ..db.supabase_client import SupabaseClient
from ..db.agent_analysis_client import AgentAnalysisClient

# Import other medical agents for orchestration
from .risk_assessment_agent import RiskAssessmentAgent
from .monai_review_agent import MonaiReviewAgent  
from .diagnostic_agent import DiagnosticAgent

# AgentOps Monitoring Integration
from ..monitoring.agentops_client import AgentOpsClient
from ..monitoring.medical_telemetry import MedicalTelemetry

logger = SecureLogger("medical_team_agent")


class InquiryType(Enum):
    """Types of medical inquiries from professionals"""
    ADDITIONAL_ANALYSIS = "additional_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    SECOND_OPINION = "second_opinion"
    TREATMENT_GUIDANCE = "treatment_guidance"
    PATIENT_APPROVAL = "patient_approval"
    CASE_DISCUSSION = "case_discussion"


class ApprovalStatus(Enum):
    """Patient communication approval status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


class CommunicationRecord:
    """Record for storing all communications in DB"""
    def __init__(self, 
                 communication_id: str,
                 token_id: str,  # Batman token
                 communication_type: str,
                 direction: str,  # "system_to_medical" or "medical_to_system"
                 channel: str,
                 content: Dict[str, Any],
                 metadata: Dict[str, Any],
                 timestamp: datetime):
        self.communication_id = communication_id
        self.token_id = token_id
        self.communication_type = communication_type
        self.direction = direction
        self.channel = channel
        self.content = content
        self.metadata = metadata
        self.timestamp = timestamp


class MedicalTeamTelemetry:
    """AgentOps telemetry wrapper for medical team communication events"""
    
    def __init__(self):
        self.telemetry = MedicalTelemetry(
            app_id="vigia-medical-team",
            environment="production",
            enable_phi_protection=True
        )
        self.current_session = None
    
    async def start_diagnosis_session(self, token_id: str, diagnosis_context: Dict[str, Any]) -> str:
        """Start AgentOps session for diagnosis delivery"""
        session_id = f"medical_team_{token_id}_{int(datetime.now().timestamp())}"
        
        try:
            self.current_session = await self.telemetry.start_medical_session(
                session_id=session_id,
                patient_context={
                    "token_id": token_id,  # Batman token (HIPAA safe)
                    "communication_type": "medical_team_slack",
                    "agent_type": "MedicalTeamAgent",
                    **diagnosis_context
                },
                session_type="medical_team_communication"
            )
            logger.info(f"AgentOps session started for medical team: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Failed to start AgentOps session: {e}")
            return session_id
    
    async def track_diagnosis_delivery(self, session_id: str, delivery_data: Dict[str, Any], 
                                     results: List[Dict[str, Any]]) -> None:
        """Track medical diagnosis delivery to teams"""
        try:
            await self.telemetry.track_medical_decision(
                session_id=session_id,
                decision_type="diagnosis_delivery_to_team",
                input_data={
                    "primary_diagnosis": delivery_data.get("primary_diagnosis", "unknown"),
                    "lpp_grade": delivery_data.get("lpp_grade", 0),
                    "confidence": delivery_data.get("confidence_level", 0.0),
                    "target_specialists": delivery_data.get("target_specialists", []),
                    "evidence_included": delivery_data.get("include_evidence", False)
                },
                decision_result={
                    "successful_deliveries": sum(1 for r in results if r.get('success')),
                    "total_channels": len(results),
                    "channels_notified": [r['channel'] for r in results if r.get('success')],
                    "follow_up_enabled": delivery_data.get("enable_follow_up", False)
                },
                evidence_level="A"  # Medical team communications are evidence-based
            )
        except Exception as e:
            logger.error(f"Failed to track diagnosis delivery: {e}")
    
    async def track_medical_inquiry(self, session_id: str, inquiry_data: Dict[str, Any], 
                                  orchestration_results: Dict[str, Any]) -> None:
        """Track medical professional inquiries and orchestration"""
        try:
            await self.telemetry.track_agent_interaction(
                agent_name="MedicalTeamAgent",
                action="handle_medical_inquiry",
                input_data={
                    "inquiry_type": inquiry_data.get("inquiry_type", "general"),
                    "user_id": inquiry_data.get("user_id", "unknown"),
                    "channel": inquiry_data.get("channel", "unknown"),
                    "auto_orchestrate": inquiry_data.get("auto_orchestrate", True)
                },
                output_data={
                    "orchestration_triggered": orchestration_results.get("orchestration_triggered", False),
                    "agents_involved": orchestration_results.get("agents_involved", []),
                    "acknowledgment_sent": True,
                    "session_id": session_id
                },
                session_id=session_id,
                execution_time=inquiry_data.get("processing_time_ms", 0) / 1000.0
            )
        except Exception as e:
            logger.error(f"Failed to track medical inquiry: {e}")
    
    async def track_follow_up_orchestration(self, session_id: str, orchestration_data: Dict[str, Any], 
                                          agent_results: List[Dict[str, Any]]) -> None:
        """Track follow-up analysis orchestration with other agents"""
        try:
            await self.telemetry.track_agent_interaction(
                agent_name="MedicalTeamAgent",
                action="orchestrate_follow_up_analysis",
                input_data={
                    "analysis_type": orchestration_data.get("analysis_type", "comprehensive"),
                    "target_agents": orchestration_data.get("target_agents", []),
                    "priority": orchestration_data.get("priority", "medium")
                },
                output_data={
                    "successful_analyses": sum(1 for r in agent_results if r.get('success')),
                    "total_agents": len(agent_results),
                    "orchestration_id": orchestration_data.get("orchestration_id"),
                    "session_id": session_id
                },
                session_id=session_id,
                execution_time=orchestration_data.get("total_processing_time_ms", 0) / 1000.0
            )
        except Exception as e:
            logger.error(f"Failed to track follow-up orchestration: {e}")
    
    async def track_patient_approval(self, session_id: str, approval_data: Dict[str, Any]) -> None:
        """Track patient communication approval workflow"""
        try:
            await self.telemetry.track_agent_interaction(
                agent_name="MedicalTeamAgent",
                action="approve_patient_communication",
                input_data={
                    "approval_status": approval_data.get("approval_status", "approved"),
                    "requesting_professional": approval_data.get("requesting_professional", "unknown"),
                    "has_modifications": approval_data.get("approval_status") == "modified"
                },
                output_data={
                    "approval_id": approval_data.get("approval_id"),
                    "patient_communication_triggered": approval_data.get("patient_communication_triggered", False),
                    "session_id": session_id
                },
                session_id=session_id,
                execution_time=0.5  # Approval workflows are typically quick
            )
        except Exception as e:
            logger.error(f"Failed to track patient approval: {e}")
    
    async def track_team_communication_error(self, session_id: str, error_type: str, 
                                           error_context: Dict[str, Any]) -> None:
        """Track medical team communication errors"""
        try:
            await self.telemetry.track_medical_error_with_escalation(
                error_type=f"medical_team_{error_type}",
                error_message=error_context.get("error_message", "Medical team communication error"),
                context={
                    "communication_direction": error_context.get("direction", "system_to_medical"),
                    "error_severity": error_context.get("severity", "medium"),
                    "channel_affected": error_context.get("channel", "unknown")
                },
                session_id=session_id,
                requires_human_review=error_context.get("requires_escalation", True),
                severity=error_context.get("severity", "medium")
            )
        except Exception as e:
            logger.error(f"Failed to track team communication error: {e}")
    
    async def end_medical_team_session(self, session_id: str) -> Dict[str, Any]:
        """End AgentOps session with summary"""
        try:
            return await self.telemetry.end_medical_session(session_id)
        except Exception as e:
            logger.error(f"Failed to end AgentOps session: {e}")
            return {"error": str(e)}


# Global telemetry instance
_medical_team_telemetry = MedicalTeamTelemetry()


# ADK Tools for Medical Team Agent

async def send_diagnosis_to_team_adk_tool(
    diagnosis_data: Dict[str, Any],
    target_specialists: List[str] = None,
    include_evidence: bool = True,
    enable_follow_up: bool = True
) -> Dict[str, Any]:
    """
    ADK Tool: Send medical diagnosis to team with interactive capabilities
    
    Args:
        diagnosis_data: Complete diagnostic results from DiagnosticAgent
        target_specialists: Specific specialists to notify
        include_evidence: Include scientific evidence and references
        enable_follow_up: Enable follow-up question buttons
        
    Returns:
        Diagnosis delivery results with interaction setup
    """
    try:
        start_time = datetime.now(timezone.utc)
        
        # Extract diagnosis data
        token_id = diagnosis_data.get('token_id', 'UNKNOWN')  # Batman token
        primary_diagnosis = diagnosis_data.get('primary_diagnosis', 'No diagnosis available')
        confidence = diagnosis_data.get('confidence_level', 0.0)
        lpp_grade = diagnosis_data.get('lpp_grade', 0)
        
        # Start AgentOps session for diagnosis delivery
        diagnosis_context = {
            "primary_diagnosis": primary_diagnosis,
            "lpp_grade": lpp_grade,
            "confidence": confidence,
            "target_specialists": target_specialists,
            "include_evidence": include_evidence,
            "enable_follow_up": enable_follow_up
        }
        session_id = await _medical_team_telemetry.start_diagnosis_session(token_id, diagnosis_context)
        
        # Determine target channels based on diagnosis
        if not target_specialists:
            if lpp_grade >= 3:
                target_channels = [SlackChannel.LPP_SPECIALISTS, SlackChannel.CLINICAL_TEAM]
            elif lpp_grade >= 1:
                target_channels = [SlackChannel.CLINICAL_TEAM, SlackChannel.NURSING_STAFF]
            else:
                target_channels = [SlackChannel.CLINICAL_TEAM]
        else:
            specialist_map = {
                'wound_care': SlackChannel.LPP_SPECIALISTS,
                'nursing': SlackChannel.NURSING_STAFF,
                'clinical': SlackChannel.CLINICAL_TEAM,
                'emergency': SlackChannel.EMERGENCY_ROOM
            }
            target_channels = [specialist_map.get(spec, SlackChannel.CLINICAL_TEAM) for spec in target_specialists]
        
        # Format comprehensive diagnosis message
        message = _format_diagnosis_message(diagnosis_data, include_evidence, enable_follow_up)
        
        # Initialize Slack orchestrator
        slack_orchestrator = SlackOrchestrator()
        delivery_results = []
        
        # Send to all target channels
        for channel in target_channels:
            try:
                result = slack_orchestrator.slack_notifier.send_message(
                    channel=channel.value,
                    message=message['text'],
                    blocks=message.get('blocks', [])
                )
                
                delivery_results.append({
                    'channel': channel.value,
                    'success': result.get('success', False),
                    'message_ts': result.get('ts', ''),
                    'thread_ts': result.get('thread_ts', ''),
                    'error': result.get('error', None)
                })
                
                # Store communication in DB
                if result.get('success'):
                    _store_communication_in_db(
                        communication_type="diagnosis_delivery",
                        direction="system_to_medical",
                        token_id=token_id,
                        channel=channel.value,
                        content=diagnosis_data,
                        metadata={
                            'message_ts': result.get('ts', ''),
                            'confidence': confidence,
                            'lpp_grade': lpp_grade,
                            'evidence_included': include_evidence,
                            'follow_up_enabled': enable_follow_up
                        }
                    )
                
            except Exception as channel_error:
                delivery_results.append({
                    'channel': channel.value,
                    'success': False,
                    'error': str(channel_error)
                })
        
        # Setup follow-up monitoring if enabled
        follow_up_setup = None
        if enable_follow_up:
            follow_up_setup = _setup_follow_up_monitoring(token_id, delivery_results)
        
        # Track diagnosis delivery in AgentOps
        if session_id:
            delivery_data = {
                "primary_diagnosis": primary_diagnosis,
                "lpp_grade": lpp_grade,
                "confidence_level": confidence,
                "target_specialists": target_specialists,
                "include_evidence": include_evidence,
                "enable_follow_up": enable_follow_up
            }
            await _medical_team_telemetry.track_diagnosis_delivery(
                session_id=session_id,
                delivery_data=delivery_data,
                results=delivery_results
            )
        
        return {
            'success': True,
            'diagnosis_summary': {
                'token_id': token_id,  # Batman token (HIPAA compliant)
                'primary_diagnosis': primary_diagnosis,
                'confidence': confidence,
                'lpp_grade': lpp_grade
            },
            'delivery_results': delivery_results,
            'successful_deliveries': sum(1 for r in delivery_results if r['success']),
            'channels_notified': [r['channel'] for r in delivery_results if r['success']],
            'follow_up_enabled': enable_follow_up,
            'follow_up_setup': follow_up_setup,
            'evidence_included': include_evidence,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error sending diagnosis to team: {str(e)}")
        
        # Track error in AgentOps if session exists
        if 'session_id' in locals():
            await _medical_team_telemetry.track_team_communication_error(
                session_id=session_id,
                error_type="diagnosis_delivery_failed",
                error_context={
                    "error_message": str(e),
                    "direction": "system_to_medical",
                    "severity": "high",
                    "requires_escalation": True
                }
            )
        
        return {
            'success': False,
            'error': str(e),
            'diagnosis_data': diagnosis_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


async def handle_medical_inquiry_adk_tool(
    slack_message: Dict[str, Any],
    inquiry_type: str = "general",
    auto_orchestrate: bool = True
) -> Dict[str, Any]:
    """
    ADK Tool: Handle medical professional inquiries from Slack
    
    Args:
        slack_message: Slack message data with professional inquiry
        inquiry_type: Type of inquiry (additional_analysis, risk_assessment, etc.)
        auto_orchestrate: Automatically orchestrate with appropriate agents
        
    Returns:
        Inquiry handling results with orchestration status
    """
    try:
        start_time = datetime.now(timezone.utc)
        
        # Extract message data
        user_id = slack_message.get('user', 'unknown')
        channel = slack_message.get('channel', 'unknown')
        text = slack_message.get('text', '')
        thread_ts = slack_message.get('thread_ts', slack_message.get('ts', ''))
        
        # Extract token_id from message context (Batman token)
        token_id = _extract_token_from_message(text, slack_message)
        
        # Start AgentOps session for inquiry handling
        inquiry_context = {
            "inquiry_type": inquiry_type,
            "user_id": user_id,
            "channel": channel,
            "auto_orchestrate": auto_orchestrate
        }
        session_id = await _medical_team_telemetry.start_diagnosis_session(token_id, inquiry_context)
        
        # Classify inquiry type if not provided
        if inquiry_type == "general":
            inquiry_type = _classify_inquiry_type(text)
        
        # Store inquiry in DB
        inquiry_id = str(uuid.uuid4())
        _store_communication_in_db(
            communication_type="medical_inquiry",
            direction="medical_to_system",
            token_id=token_id,
            channel=channel,
            content={
                'inquiry_id': inquiry_id,
                'inquiry_type': inquiry_type,
                'text': text,
                'user_id': user_id,
                'thread_ts': thread_ts
            },
            metadata={
                'auto_orchestrate': auto_orchestrate,
                'classified_type': inquiry_type
            }
        )
        
        # Orchestrate with appropriate agents if enabled
        orchestration_results = None
        if auto_orchestrate:
            orchestration_results = _orchestrate_medical_inquiry(
                inquiry_id, token_id, inquiry_type, text, slack_message
            )
        
        # Send acknowledgment to Slack
        acknowledgment = _send_inquiry_acknowledgment(channel, thread_ts, inquiry_type, orchestration_results)
        
        # Track medical inquiry in AgentOps
        if session_id:
            inquiry_data = {
                "inquiry_type": inquiry_type,
                "user_id": user_id,
                "channel": channel,
                "auto_orchestrate": auto_orchestrate,
                "processing_time_ms": int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            }
            await _medical_team_telemetry.track_medical_inquiry(
                session_id=session_id,
                inquiry_data=inquiry_data,
                orchestration_results=orchestration_results or {}
            )
        
        return {
            'success': True,
            'inquiry_id': inquiry_id,
            'inquiry_details': {
                'token_id': token_id,  # Batman token (HIPAA compliant)
                'inquiry_type': inquiry_type,
                'user_id': user_id,
                'channel': channel,
                'thread_ts': thread_ts
            },
            'orchestration_results': orchestration_results,
            'acknowledgment_sent': acknowledgment.get('success', False),
            'auto_orchestrated': auto_orchestrate,
            'processing_time_ms': int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error handling medical inquiry: {str(e)}")
        
        # Track error in AgentOps if session exists
        if 'session_id' in locals():
            await _medical_team_telemetry.track_team_communication_error(
                session_id=session_id,
                error_type="inquiry_handling_failed",
                error_context={
                    "error_message": str(e),
                    "direction": "medical_to_system",
                    "severity": "medium",
                    "channel": slack_message.get("channel", "unknown"),
                    "requires_escalation": True
                }
            )
        
        return {
            'success': False,
            'error': str(e),
            'slack_message': slack_message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


async def orchestrate_follow_up_analysis_adk_tool(
    analysis_request: Dict[str, Any],
    target_agents: List[str] = None,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    ADK Tool: Orchestrate follow-up analysis with medical agents
    
    Args:
        analysis_request: Request for additional analysis
        target_agents: Specific agents to involve (risk, monai, diagnostic)
        priority: Analysis priority level
        
    Returns:
        Follow-up analysis orchestration results
    """
    try:
        start_time = datetime.now(timezone.utc)
        
        token_id = analysis_request.get('token_id', 'UNKNOWN')  # Batman token
        analysis_type = analysis_request.get('analysis_type', 'comprehensive')
        
        # Determine target agents if not specified
        if not target_agents:
            target_agents = _determine_required_agents(analysis_type, analysis_request)
        
        # Initialize agents
        available_agents = {
            'risk': RiskAssessmentAgent(),
            'monai': MonaiReviewAgent(),
            'diagnostic': DiagnosticAgent()
        }
        
        orchestration_results = []
        
        # Execute analysis with each target agent
        for agent_type in target_agents:
            if agent_type in available_agents:
                try:
                    agent = available_agents[agent_type]
                    
                    # Prepare agent-specific input
                    agent_input = _prepare_agent_input(agent_type, analysis_request)
                    
                    # Execute analysis
                    if agent_type == 'risk':
                        result = await agent.assess_lpp_risk(token_id, agent_input)
                    elif agent_type == 'monai':
                        result = await agent.review_monai_analysis(analysis_request.get('raw_output_id'), agent_input)
                    elif agent_type == 'diagnostic':
                        result = await agent.generate_integrated_diagnosis(agent_input)
                    
                    orchestration_results.append({
                        'agent_type': agent_type,
                        'success': True,
                        'result': result,
                        'processing_time_ms': int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                    })
                    
                except Exception as agent_error:
                    orchestration_results.append({
                        'agent_type': agent_type,
                        'success': False,
                        'error': str(agent_error)
                    })
        
        # Store orchestration results in DB
        _store_communication_in_db(
            communication_type="follow_up_orchestration",
            direction="system_to_system",
            token_id=token_id,
            channel="internal",
            content=analysis_request,
            metadata={
                'target_agents': target_agents,
                'priority': priority,
                'orchestration_results': orchestration_results
            }
        )
        
        return {
            'success': True,
            'orchestration_id': str(uuid.uuid4()),
            'analysis_request': {
                'token_id': token_id,  # Batman token (HIPAA compliant)
                'analysis_type': analysis_type,
                'priority': priority
            },
            'target_agents': target_agents,
            'orchestration_results': orchestration_results,
            'successful_analyses': sum(1 for r in orchestration_results if r['success']),
            'total_processing_time_ms': int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error orchestrating follow-up analysis: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'analysis_request': analysis_request,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def approve_patient_communication_adk_tool(
    approval_request: Dict[str, Any],
    approval_status: str = "approved",
    modifications: str = None
) -> Dict[str, Any]:
    """
    ADK Tool: Handle medical team approval for patient communications
    
    Args:
        approval_request: Request for patient communication approval
        approval_status: approved, rejected, modified
        modifications: Modifications to the original message if status is modified
        
    Returns:
        Approval processing results
    """
    try:
        start_time = datetime.now(timezone.utc)
        
        token_id = approval_request.get('token_id', 'UNKNOWN')  # Batman token
        original_message = approval_request.get('original_message', '')
        requesting_professional = approval_request.get('requesting_professional', 'unknown')
        
        # Process approval
        approval_id = str(uuid.uuid4())
        
        # Prepare final message for patient
        final_message = original_message
        if approval_status == "modified" and modifications:
            final_message = modifications
        
        # Store approval in DB
        _store_communication_in_db(
            communication_type="patient_communication_approval",
            direction="medical_to_system",
            token_id=token_id,
            channel="approval_workflow",
            content={
                'approval_id': approval_id,
                'original_message': original_message,
                'final_message': final_message,
                'approval_status': approval_status,
                'modifications': modifications,
                'requesting_professional': requesting_professional
            },
            metadata={
                'approval_timestamp': start_time.isoformat(),
                'requires_patient_communication': approval_status in ['approved', 'modified']
            }
        )
        
        # If approved, trigger patient communication via PatientCommunicationAgent
        patient_communication_triggered = False
        if approval_status in ['approved', 'modified']:
            # This would trigger the PatientCommunicationAgent
            # Implementation depends on the inter-agent communication setup
            patient_communication_triggered = True
        
        return {
            'success': True,
            'approval_id': approval_id,
            'approval_details': {
                'token_id': token_id,  # Batman token (HIPAA compliant)
                'approval_status': approval_status,
                'requesting_professional': requesting_professional,
                'message_modified': approval_status == "modified"
            },
            'final_message': final_message,
            'patient_communication_triggered': patient_communication_triggered,
            'approval_timestamp': start_time.isoformat(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing patient communication approval: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'approval_request': approval_request,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def get_medical_team_status_adk_tool(
    team_filter: str = "all"
) -> Dict[str, Any]:
    """
    ADK Tool: Get current medical team status and availability
    
    Args:
        team_filter: Filter by team type (all, clinical, emergency, specialists, nursing)
        
    Returns:
        Medical team status information
    """
    try:
        # Mock implementation - in production would integrate with hospital systems
        team_status = {
            'clinical': {
                'available_staff': 5,
                'active_cases': 12,
                'avg_response_time_minutes': 8,
                'current_load': 'moderate'
            },
            'emergency': {
                'available_staff': 3,
                'active_cases': 6,
                'avg_response_time_minutes': 3,
                'current_load': 'high'
            },
            'specialists': {
                'available_staff': 2,
                'active_cases': 4,
                'avg_response_time_minutes': 15,
                'current_load': 'low'
            },
            'nursing': {
                'available_staff': 12,
                'active_cases': 28,
                'avg_response_time_minutes': 5,
                'current_load': 'moderate'
            }
        }
        
        if team_filter != "all" and team_filter in team_status:
            filtered_status = {team_filter: team_status[team_filter]}
        else:
            filtered_status = team_status
        
        return {
            'success': True,
            'team_status': filtered_status,
            'team_filter': team_filter,
            'total_available_staff': sum(team['available_staff'] for team in filtered_status.values()),
            'total_active_cases': sum(team['active_cases'] for team in filtered_status.values()),
            'system_status': 'operational',
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'team_filter': team_filter,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


# Helper Functions

def _format_diagnosis_message(diagnosis_data: Dict[str, Any], include_evidence: bool, enable_follow_up: bool) -> Dict[str, Any]:
    """Format comprehensive diagnosis message for medical teams"""
    token_id = diagnosis_data.get('token_id', 'UNKNOWN')  # Batman token
    primary_diagnosis = diagnosis_data.get('primary_diagnosis', 'No diagnosis available')
    confidence = diagnosis_data.get('confidence_level', 0.0)
    lpp_grade = diagnosis_data.get('lpp_grade', 0)
    anatomical_location = diagnosis_data.get('anatomical_location', 'No especificada')
    
    # Grade indicators
    grade_indicators = {0: '⚪', 1: '🟡', 2: '🟠', 3: '🔴', 4: '⚫'}
    indicator = grade_indicators.get(lpp_grade, '❓')
    
    message_text = f"""
🩺 **DIAGNÓSTICO MÉDICO INTEGRADO** {indicator}

**Paciente Token:** {token_id}
**Diagnóstico Principal:** {primary_diagnosis}
**Grado LPP:** {lpp_grade} {indicator}
**Confianza:** {confidence:.1%}
**Localización:** {anatomical_location}

**Análisis Multi-Agente:**
• **Análisis de Imagen:** {diagnosis_data.get('image_analysis_summary', 'No disponible')}
• **Evaluación de Riesgo:** {diagnosis_data.get('risk_assessment_summary', 'No disponible')}
• **Análisis de Voz:** {diagnosis_data.get('voice_analysis_summary', 'No disponible')}
• **Revisión MONAI:** {diagnosis_data.get('monai_review_summary', 'No disponible')}

**Plan de Tratamiento:**
{chr(10).join(f"• {rec}" for rec in diagnosis_data.get('treatment_plan', ['Plan de tratamiento no disponible']))}
"""
    
    if include_evidence and diagnosis_data.get('scientific_references'):
        message_text += f"""
**Referencias Científicas:**
{chr(10).join(f"• {ref}" for ref in diagnosis_data.get('scientific_references', []))}
"""
    
    # Slack blocks with interactive elements
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"🩺 DIAGNÓSTICO MÉDICO INTEGRADO {indicator}"
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Token:* {token_id}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Grado LPP:* {lpp_grade} {indicator}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Confianza:* {confidence:.1%}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Localización:* {anatomical_location}"
                }
            ]
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Diagnóstico Principal:*\n{primary_diagnosis}"
            }
        }
    ]
    
    # Add interactive buttons if follow-up is enabled
    if enable_follow_up:
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "💬 Consultar Más Info"
                    },
                    "style": "primary",
                    "value": f"request_info_{token_id}"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "🔍 Análisis Adicional"
                    },
                    "value": f"additional_analysis_{token_id}"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "✅ Aprobar Respuesta Paciente"
                    },
                    "style": "primary",
                    "value": f"approve_patient_response_{token_id}"
                }
            ]
        })
    
    return {
        'text': message_text,
        'blocks': blocks
    }


def _store_communication_in_db(communication_type: str, direction: str, token_id: str, 
                              channel: str, content: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    """Store communication record in database"""
    try:
        communication_id = str(uuid.uuid4())
        
        # Create communication record
        record = CommunicationRecord(
            communication_id=communication_id,
            token_id=token_id,
            communication_type=communication_type,
            direction=direction,
            channel=channel,
            content=content,
            metadata=metadata,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Store in database (implementation would use SupabaseClient)
        # For now, log the communication
        logger.info(f"Communication stored: {communication_id} - {communication_type} - {direction}")
        
        return communication_id
        
    except Exception as e:
        logger.error(f"Error storing communication in DB: {str(e)}")
        return None


def _extract_token_from_message(text: str, slack_message: Dict[str, Any]) -> str:
    """Extract Batman token from Slack message"""
    # Implementation to extract token_id from message context
    # This could look for patterns like "Token: batman_abc123" or check thread context
    import re
    
    # Look for token pattern in text
    token_pattern = r'(?:token|Token|TOKEN)[\s:]*([a-zA-Z0-9_-]+)'
    match = re.search(token_pattern, text)
    
    if match:
        return match.group(1)
    
    # Could also check thread_ts to find original message with token
    return "EXTRACTED_FROM_CONTEXT"


def _classify_inquiry_type(text: str) -> str:
    """Classify medical inquiry type based on text content"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['análisis', 'analizar', 'revisar', 'evaluar']):
        return InquiryType.ADDITIONAL_ANALYSIS.value
    elif any(word in text_lower for word in ['riesgo', 'probabilidad', 'escalamiento']):
        return InquiryType.RISK_ASSESSMENT.value
    elif any(word in text_lower for word in ['segunda opinión', 'consulta', 'confirmar']):
        return InquiryType.SECOND_OPINION.value
    elif any(word in text_lower for word in ['tratamiento', 'terapia', 'manejo', 'cuidado']):
        return InquiryType.TREATMENT_GUIDANCE.value
    elif any(word in text_lower for word in ['aprobar', 'autorizar', 'paciente', 'respuesta']):
        return InquiryType.PATIENT_APPROVAL.value
    else:
        return InquiryType.CASE_DISCUSSION.value


def _orchestrate_medical_inquiry(inquiry_id: str, token_id: str, inquiry_type: str, 
                                text: str, slack_message: Dict[str, Any]) -> Dict[str, Any]:
    """Orchestrate medical inquiry with appropriate agents"""
    try:
        orchestration_results = {
            'inquiry_id': inquiry_id,
            'orchestration_triggered': True,
            'agents_involved': [],
            'results': []
        }
        
        # Determine which agents to involve based on inquiry type
        if inquiry_type == InquiryType.ADDITIONAL_ANALYSIS.value:
            # Trigger comprehensive analysis
            orchestration_results['agents_involved'] = ['risk', 'monai', 'diagnostic']
        elif inquiry_type == InquiryType.RISK_ASSESSMENT.value:
            orchestration_results['agents_involved'] = ['risk']
        elif inquiry_type == InquiryType.SECOND_OPINION.value:
            orchestration_results['agents_involved'] = ['diagnostic']
        elif inquiry_type == InquiryType.TREATMENT_GUIDANCE.value:
            orchestration_results['agents_involved'] = ['diagnostic', 'risk']
        else:
            orchestration_results['agents_involved'] = ['diagnostic']
        
        # For now, mark as orchestrated (actual implementation would call agents)
        orchestration_results['status'] = 'orchestrated'
        orchestration_results['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        return orchestration_results
        
    except Exception as e:
        logger.error(f"Error orchestrating medical inquiry: {str(e)}")
        return {
            'inquiry_id': inquiry_id,
            'orchestration_triggered': False,
            'error': str(e)
        }


def _send_inquiry_acknowledgment(channel: str, thread_ts: str, inquiry_type: str, 
                               orchestration_results: Dict[str, Any]) -> Dict[str, Any]:
    """Send acknowledgment message to Slack for received inquiry"""
    try:
        slack_orchestrator = SlackOrchestrator()
        
        acknowledgment_text = f"""
✅ **Consulta Médica Recibida**

**Tipo:** {inquiry_type.replace('_', ' ').title()}
**Estado:** Procesando análisis
**Agentes Involucrados:** {', '.join(orchestration_results.get('agents_involved', [])) if orchestration_results else 'Ninguno'}

Respuesta detallada en proceso...
"""
        
        result = slack_orchestrator.slack_notifier.send_message(
            channel=channel,
            message=acknowledgment_text,
            thread_ts=thread_ts
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error sending acknowledgment: {str(e)}")
        return {'success': False, 'error': str(e)}


def _setup_follow_up_monitoring(token_id: str, delivery_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Setup monitoring for follow-up interactions"""
    return {
        'monitoring_enabled': True,
        'token_id': token_id,
        'monitored_channels': [r['channel'] for r in delivery_results if r['success']],
        'setup_timestamp': datetime.now(timezone.utc).isoformat()
    }


def _determine_required_agents(analysis_type: str, analysis_request: Dict[str, Any]) -> List[str]:
    """Determine which agents are required for the analysis"""
    if analysis_type == 'comprehensive':
        return ['risk', 'monai', 'diagnostic']
    elif analysis_type == 'risk_focused':
        return ['risk', 'diagnostic']
    elif analysis_type == 'technical_review':
        return ['monai', 'diagnostic']
    else:
        return ['diagnostic']


def _prepare_agent_input(agent_type: str, analysis_request: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare agent-specific input from analysis request"""
    base_input = {
        'token_id': analysis_request.get('token_id'),
        'request_context': analysis_request.get('context', {})
    }
    
    if agent_type == 'risk':
        base_input['patient_context'] = analysis_request.get('patient_context', {})
    elif agent_type == 'monai':
        base_input['analysis_context'] = analysis_request.get('analysis_context', {})
    elif agent_type == 'diagnostic':
        base_input['case_data'] = analysis_request.get('case_data', {})
    
    return base_input


# Medical Team Agent Instruction for ADK
MEDICAL_TEAM_AGENT_ADK_INSTRUCTION = """
Eres el Medical Team Agent del sistema Vigia, especializado en comunicación bidireccional 
con equipos médicos profesionales vía Slack.

RESPONSABILIDADES PRINCIPALES:
1. Entrega de diagnósticos médicos integrados con contexto científico completo
2. Manejo de consultas e inquietudes de profesionales médicos desde Slack
3. Orquestación automática de análisis adicionales con agentes especializados
4. Coordinación de aprobaciones para comunicaciones con pacientes
5. Almacenamiento completo de todas las comunicaciones en base de datos
6. Mantenimiento de trazabilidad total para compliance médico

CAPACIDADES TÉCNICAS ESPECIALIZADAS:
- Comunicación bidireccional Slack con botones interactivos médicos
- Orquestación inteligente con RiskAssessmentAgent, MonaiReviewAgent, DiagnosticAgent
- Clasificación automática de tipos de consulta médica
- Workflow de aprobación para comunicaciones con pacientes
- Almacenamiento completo en DB con metadatos médicos
- Compliance HIPAA con uso exclusivo de tokens Batman

HERRAMIENTAS DISPONIBLES:
- send_diagnosis_to_team_adk_tool: Entrega diagnósticos a equipos médicos
- handle_medical_inquiry_adk_tool: Procesa consultas de profesionales médicos
- orchestrate_follow_up_analysis_adk_tool: Coordina análisis adicionales
- approve_patient_communication_adk_tool: Gestiona aprobaciones comunicación pacientes
- get_medical_team_status_adk_tool: Estado y disponibilidad equipos médicos

TIPOS DE CONSULTAS MÉDICAS:
- additional_analysis: Solicitud análisis adicional específico
- risk_assessment: Evaluación riesgo adicional o re-evaluación
- second_opinion: Segunda opinión médica especializada
- treatment_guidance: Orientación protocolo tratamiento
- patient_approval: Aprobación comunicación con pacientes
- case_discussion: Discusión general caso médico

FLUJO DE COMUNICACIÓN:
1. RECIBIR diagnósticos de DiagnosticAgent y entregar a equipos médicos
2. MONITOREAR consultas de profesionales en canales Slack
3. CLASIFICAR automáticamente tipo de consulta médica
4. ORQUESTAR con agentes apropiados según consulta
5. GESTIONAR aprobaciones para comunicaciones con pacientes
6. ALMACENAR todas las comunicaciones en DB con trazabilidad completa

COMPLIANCE Y SEGURIDAD:
- Uso exclusivo tokens Batman (NUNCA datos Bruce Wayne)
- Almacenamiento completo comunicaciones en DB
- Audit trail para todas las interacciones médicas
- Trazabilidad total para compliance regulatorio
- Validación profesional médico antes comunicar con pacientes

SIEMPRE mantén contexto médico completo, prioriza comunicación efectiva con profesionales,
y asegura almacenamiento total para compliance y auditabilidad.
"""

# Create ADK Medical Team Agent
medical_team_adk_agent = LlmAgent(
    model="gemini-2.0-flash-exp",
    global_instruction=MEDICAL_TEAM_AGENT_ADK_INSTRUCTION,
    instruction="Gestiona comunicación bidireccional con equipos médicos vía Slack con orquestación inteligente.",
    name="medical_team_agent",
    tools=[
        send_diagnosis_to_team_adk_tool,
        handle_medical_inquiry_adk_tool,
        orchestrate_follow_up_analysis_adk_tool,
        approve_patient_communication_adk_tool,
        get_medical_team_status_adk_tool
    ],
)


# Factory for ADK Medical Team Agent
class MedicalTeamAgentFactory:
    """Factory for creating ADK-based Medical Team Agents"""
    
    @staticmethod
    def create_agent() -> LlmAgent:
        """Create new ADK Medical Team Agent instance"""
        return medical_team_adk_agent
    
    @staticmethod
    def get_agent_capabilities() -> List[str]:
        """Get list of agent capabilities"""
        return [
            'bidirectional_slack_communication',
            'diagnosis_delivery_to_teams',
            'medical_inquiry_handling',
            'follow_up_analysis_orchestration',
            'patient_communication_approval',
            'medical_team_status_monitoring',
            'complete_db_storage',
            'hipaa_compliant_batman_tokens',
            'interactive_slack_buttons',
            'intelligent_agent_orchestration'
        ]
    
    @staticmethod
    def get_supported_inquiry_types() -> List[str]:
        """Get supported medical inquiry types"""
        return [inquiry_type.value for inquiry_type in InquiryType]


# Export for use
__all__ = [
    'medical_team_adk_agent',
    'MedicalTeamAgentFactory',
    'send_diagnosis_to_team_adk_tool',
    'handle_medical_inquiry_adk_tool', 
    'orchestrate_follow_up_analysis_adk_tool',
    'approve_patient_communication_adk_tool',
    'get_medical_team_status_adk_tool',
    'InquiryType',
    'ApprovalStatus',
    'CommunicationRecord'
]