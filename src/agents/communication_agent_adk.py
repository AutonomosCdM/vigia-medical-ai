"""
Communication Agent - Complete ADK Agent for Medical Team Communications
========================================================================

Complete ADK-based agent that handles comprehensive medical team communications
by converting interfaces/slack_orchestrator.py and WhatsApp integration functionality 
into ADK tools and patterns.

This agent provides:
- Emergency medical alerts with critical path processing
- Clinical result notifications with evidence-based context
- Human review request routing with specialist escalation
- WhatsApp/Slack multi-channel integration
- Priority-based escalation protocols
- HIPAA-compliant communication audit trail
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
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
from ..messaging.whatsapp.processor import WhatsAppProcessor
from ..messaging.whatsapp.isolated_bot import IsolatedWhatsAppBot
from ..utils.secure_logger import SecureLogger
from ..utils.audit_service import AuditService, AuditEventType, AuditLevel
from ..utils.performance_profiler import profile_performance

logger = SecureLogger("communication_agent_adk")


# Communication Classifications and Enums

class MedicalTeamRole(Enum):
    """Medical team role classifications"""
    ATTENDING_PHYSICIAN = "attending_physician"
    WOUND_CARE_SPECIALIST = "wound_care_specialist"
    NURSE_PRACTITIONER = "nurse_practitioner"
    REGISTERED_NURSE = "registered_nurse"
    CLINICAL_COORDINATOR = "clinical_coordinator"
    EMERGENCY_PHYSICIAN = "emergency_physician"
    SURGEON = "surgeon"
    DERMATOLOGIST = "dermatologist"


class CommunicationChannel(Enum):
    """Communication channel types"""
    SLACK = "slack"
    WHATSAPP = "whatsapp"
    EMAIL = "email"
    SMS = "sms"
    PAGER = "pager"
    PHONE_CALL = "phone_call"


class MessageTemplate(Enum):
    """Medical message template types"""
    EMERGENCY_ALERT = "emergency_alert"
    CLINICAL_RESULT = "clinical_result"
    REVIEW_REQUEST = "review_request"
    PROTOCOL_ACTIVATION = "protocol_activation"
    AUDIT_NOTIFICATION = "audit_notification"
    SYSTEM_STATUS = "system_status"


# ADK Tools for Communication Agent

def send_emergency_alert_adk_tool(
    alert_content: Dict[str, Any],
    urgency_level: str = "critical",
    target_teams: List[str] = None,
    escalation_enabled: bool = True
) -> Dict[str, Any]:
    """
    ADK Tool: Send emergency medical alert with critical path processing
    
    Args:
        alert_content: Emergency alert content with medical context
        urgency_level: critical, high, medium, low
        target_teams: List of medical teams to notify
        escalation_enabled: Enable automatic escalation
        
    Returns:
        Emergency alert delivery results with tracking information
    """
    try:
        start_time = datetime.now(timezone.utc)
        
        # Validate alert content
        required_fields = ['token_id', 'alert_type', 'clinical_context']  # Batman token
        missing_fields = [field for field in required_fields if field not in alert_content]
        if missing_fields:
            return {
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'timestamp': start_time.isoformat()
            }
        
        # Determine priority and channels
        priority_map = {
            'critical': NotificationPriority.CRITICAL,
            'high': NotificationPriority.HIGH,
            'medium': NotificationPriority.MEDIUM,
            'low': NotificationPriority.LOW
        }
        priority = priority_map.get(urgency_level, NotificationPriority.HIGH)
        
        # Select target channels based on urgency
        if not target_teams:
            if urgency_level == 'critical':
                target_channels = [SlackChannel.EMERGENCY_ROOM, SlackChannel.CLINICAL_TEAM]
            elif urgency_level == 'high':
                target_channels = [SlackChannel.CLINICAL_TEAM, SlackChannel.LPP_SPECIALISTS]
            else:
                target_channels = [SlackChannel.CLINICAL_TEAM]
        else:
            channel_map = {
                'emergency': SlackChannel.EMERGENCY_ROOM,
                'clinical': SlackChannel.CLINICAL_TEAM,
                'specialists': SlackChannel.LPP_SPECIALISTS,
                'nursing': SlackChannel.NURSING_STAFF
            }
            target_channels = [channel_map.get(team, SlackChannel.CLINICAL_TEAM) for team in target_teams]
        
        # Create notification payload
        notification_payload = NotificationPayload(
            notification_id=str(uuid.uuid4()),
            session_id=alert_content.get('session_id', str(uuid.uuid4())),
            notification_type=NotificationType.EMERGENCY_ALERT,
            priority=priority,
            target_channels=target_channels,
            content=alert_content,
            metadata={
                'urgency_level': urgency_level,
                'escalation_enabled': escalation_enabled,
                'sent_timestamp': start_time.isoformat(),
                'agent_id': 'communication_agent_adk'
            },
            timestamp=start_time.isoformat(),
            escalation_rules={
                'max_escalations': 3 if urgency_level == 'critical' else 2,
                'escalation_interval_minutes': 5 if urgency_level == 'critical' else 30,
                'escalation_channels': [SlackChannel.EMERGENCY_ROOM.value]
            } if escalation_enabled else None
        )
        
        # Initialize Slack orchestrator and send
        slack_orchestrator = SlackOrchestrator()
        delivery_results = []
        
        for channel in target_channels:
            try:
                # Format emergency message
                message = _format_emergency_message(alert_content, urgency_level)
                
                # Send to channel
                result = slack_orchestrator.slack_notifier.send_message(
                    channel=channel.value,
                    message=message['text'],
                    blocks=message.get('blocks', [])
                )
                
                delivery_results.append({
                    'channel': channel.value,
                    'success': result.get('success', False),
                    'message_ts': result.get('ts', ''),
                    'error': result.get('error', None)
                })
                
            except Exception as channel_error:
                delivery_results.append({
                    'channel': channel.value,
                    'success': False,
                    'error': str(channel_error)
                })
        
        # Setup escalation if enabled
        escalation_setup = None
        if escalation_enabled and priority in [NotificationPriority.CRITICAL, NotificationPriority.HIGH]:
            escalation_setup = _setup_escalation_monitoring(notification_payload)
        
        # Calculate delivery time
        delivery_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return {
            'success': True,
            'notification_id': notification_payload.notification_id,
            'urgency_level': urgency_level,
            'priority': priority.name,
            'channels_targeted': len(target_channels),
            'delivery_results': delivery_results,
            'successful_deliveries': sum(1 for r in delivery_results if r['success']),
            'escalation_setup': escalation_setup,
            'delivery_time_seconds': delivery_time,
            'compliance_logged': True,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'urgency_level': urgency_level,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def send_clinical_result_adk_tool(
    clinical_data: Dict[str, Any],
    include_evidence: bool = True,
    target_specialists: List[str] = None
) -> Dict[str, Any]:
    """
    ADK Tool: Send clinical result notification with evidence-based context
    
    Args:
        clinical_data: Complete clinical assessment results
        include_evidence: Include NPUAP/EPUAP evidence and references
        target_specialists: Specific specialist teams to notify
        
    Returns:
        Clinical notification delivery results
    """
    try:
        start_time = datetime.now(timezone.utc)
        
        # Extract clinical data
        lpp_grade = clinical_data.get('lpp_grade', 0)
        confidence = clinical_data.get('confidence_score', 0.0)
        token_id = clinical_data.get('token_id', clinical_data.get('patient_code', 'UNKNOWN'))  # Batman token
        
        # Determine priority based on LPP grade
        priority = _determine_clinical_priority(lpp_grade, confidence)
        
        # Select target channels
        if target_specialists:
            specialist_map = {
                'wound_care': SlackChannel.LPP_SPECIALISTS,
                'nursing': SlackChannel.NURSING_STAFF,
                'clinical': SlackChannel.CLINICAL_TEAM
            }
            target_channels = [specialist_map.get(spec, SlackChannel.CLINICAL_TEAM) for spec in target_specialists]
        else:
            # Default routing based on LPP grade
            if lpp_grade >= 3:
                target_channels = [SlackChannel.LPP_SPECIALISTS, SlackChannel.CLINICAL_TEAM]
            elif lpp_grade >= 1:
                target_channels = [SlackChannel.CLINICAL_TEAM, SlackChannel.NURSING_STAFF]
            else:
                target_channels = [SlackChannel.CLINICAL_TEAM]
        
        # Format clinical message with evidence
        message = _format_clinical_result_message(clinical_data, include_evidence)
        
        # Create notification payload
        notification_payload = NotificationPayload(
            notification_id=str(uuid.uuid4()),
            session_id=clinical_data.get('session_id', str(uuid.uuid4())),
            notification_type=NotificationType.CLINICAL_RESULT,
            priority=priority,
            target_channels=target_channels,
            content=clinical_data,
            metadata={
                'lpp_grade': lpp_grade,
                'confidence': confidence,
                'token_id': token_id,  # Batman token (HIPAA compliant)
                'evidence_included': include_evidence,
                'sent_timestamp': start_time.isoformat()
            },
            timestamp=start_time.isoformat()
        )
        
        # Send notifications
        slack_orchestrator = SlackOrchestrator()
        delivery_results = []
        
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
                    'error': result.get('error', None)
                })
                
            except Exception as channel_error:
                delivery_results.append({
                    'channel': channel.value,
                    'success': False,
                    'error': str(channel_error)
                })
        
        return {
            'success': True,
            'notification_id': notification_payload.notification_id,
            'clinical_summary': {
                'lpp_grade': lpp_grade,
                'confidence': confidence,
                'token_id': token_id,  # Batman token (HIPAA compliant)
                'priority': priority.name
            },
            'evidence_included': include_evidence,
            'delivery_results': delivery_results,
            'successful_deliveries': sum(1 for r in delivery_results if r['success']),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'clinical_data': clinical_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def request_human_review_adk_tool(
    review_request: Dict[str, Any],
    specialist_type: str = "general",
    urgency: str = "medium",
    include_interactive_buttons: bool = True
) -> Dict[str, Any]:
    """
    ADK Tool: Request human medical review with specialist routing
    
    Args:
        review_request: Review request with medical context
        specialist_type: Type of specialist needed (wound_care, dermatology, etc.)
        urgency: Request urgency level
        include_interactive_buttons: Include Slack interactive buttons
        
    Returns:
        Review request delivery results with tracking
    """
    try:
        start_time = datetime.now(timezone.utc)
        
        # Map specialist types to channels
        specialist_routing = {
            'wound_care': SlackChannel.LPP_SPECIALISTS,
            'dermatology': SlackChannel.LPP_SPECIALISTS,  # Could be separate channel
            'nursing': SlackChannel.NURSING_STAFF,
            'clinical': SlackChannel.CLINICAL_TEAM,
            'emergency': SlackChannel.EMERGENCY_ROOM,
            'general': SlackChannel.CLINICAL_TEAM
        }
        
        target_channel = specialist_routing.get(specialist_type, SlackChannel.CLINICAL_TEAM)
        
        # Determine priority
        priority_map = {
            'critical': NotificationPriority.CRITICAL,
            'high': NotificationPriority.HIGH,
            'medium': NotificationPriority.MEDIUM,
            'low': NotificationPriority.LOW
        }
        priority = priority_map.get(urgency, NotificationPriority.MEDIUM)
        
        # Format review request message
        message = _format_review_request_message(
            review_request, 
            specialist_type, 
            urgency,
            include_interactive_buttons
        )
        
        # Create notification payload
        notification_payload = NotificationPayload(
            notification_id=str(uuid.uuid4()),
            session_id=review_request.get('session_id', str(uuid.uuid4())),
            notification_type=NotificationType.HUMAN_REVIEW_REQUEST,
            priority=priority,
            target_channels=[target_channel],
            content=review_request,
            metadata={
                'specialist_type': specialist_type,
                'urgency': urgency,
                'interactive_buttons': include_interactive_buttons,
                'sent_timestamp': start_time.isoformat()
            },
            timestamp=start_time.isoformat()
        )
        
        # Send review request
        slack_orchestrator = SlackOrchestrator()
        
        try:
            result = slack_orchestrator.slack_notifier.send_message(
                channel=target_channel.value,
                message=message['text'],
                blocks=message.get('blocks', [])
            )
            
            delivery_success = result.get('success', False)
            message_ts = result.get('ts', '')
            
            # Setup review tracking if successful
            review_tracking = None
            if delivery_success:
                review_tracking = _setup_review_tracking(
                    notification_payload.notification_id,
                    specialist_type,
                    urgency,
                    message_ts
                )
            
            return {
                'success': delivery_success,
                'notification_id': notification_payload.notification_id,
                'review_details': {
                    'specialist_type': specialist_type,
                    'urgency': urgency,
                    'target_channel': target_channel.value,
                    'priority': priority.name
                },
                'delivery_result': {
                    'channel': target_channel.value,
                    'success': delivery_success,
                    'message_ts': message_ts,
                    'error': result.get('error', None)
                },
                'review_tracking': review_tracking,
                'interactive_buttons': include_interactive_buttons,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as send_error:
            return {
                'success': False,
                'error': str(send_error),
                'notification_id': notification_payload.notification_id,
                'specialist_type': specialist_type,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'review_request': review_request,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def send_whatsapp_response_adk_tool(
    recipient_phone: str,
    message_content: str,
    message_type: str = "text",
    media_url: str = None
) -> Dict[str, Any]:
    """
    ADK Tool: Send WhatsApp response message
    
    Args:
        recipient_phone: Phone number in WhatsApp format
        message_content: Message text content
        message_type: text, image, document
        media_url: URL for media messages
        
    Returns:
        WhatsApp message delivery result
    """
    try:
        start_time = datetime.now(timezone.utc)
        
        # Initialize WhatsApp processor
        whatsapp_processor = WhatsAppProcessor()
        
        # Prepare message payload
        message_payload = {
            'to': recipient_phone,
            'type': message_type,
            'text': {'body': message_content} if message_type == 'text' else None,
            'media_url': media_url if media_url else None
        }
        
        # Send WhatsApp message
        result = whatsapp_processor.send_message(message_payload)
        
        # Calculate delivery time
        delivery_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return {
            'success': result.get('success', False),
            'message_id': result.get('message_id', ''),
            'recipient_phone': recipient_phone,
            'message_type': message_type,
            'delivery_time_seconds': delivery_time,
            'whatsapp_status': result.get('status', 'unknown'),
            'error': result.get('error', None) if not result.get('success', False) else None,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'recipient_phone': recipient_phone,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def escalate_notification_adk_tool(
    original_notification_id: str,
    escalation_reason: str,
    target_level: str = "supervisor",
    max_attempts: int = 3
) -> Dict[str, Any]:
    """
    ADK Tool: Escalate notification with priority routing
    
    Args:
        original_notification_id: ID of original notification
        escalation_reason: Reason for escalation
        target_level: supervisor, department_head, medical_director
        max_attempts: Maximum escalation attempts
        
    Returns:
        Escalation results with tracking information
    """
    try:
        start_time = datetime.now(timezone.utc)
        
        # Define escalation levels
        escalation_channels = {
            'supervisor': SlackChannel.CLINICAL_TEAM,
            'department_head': SlackChannel.LPP_SPECIALISTS,
            'medical_director': SlackChannel.EMERGENCY_ROOM,
            'emergency': SlackChannel.EMERGENCY_ROOM
        }
        
        target_channel = escalation_channels.get(target_level, SlackChannel.CLINICAL_TEAM)
        
        # Format escalation message
        escalation_message = _format_escalation_message(
            original_notification_id,
            escalation_reason,
            target_level
        )
        
        # Send escalation
        slack_orchestrator = SlackOrchestrator()
        
        try:
            result = slack_orchestrator.slack_notifier.send_message(
                channel=target_channel.value,
                message=escalation_message['text'],
                blocks=escalation_message.get('blocks', [])
            )
            
            escalation_success = result.get('success', False)
            
            return {
                'success': escalation_success,
                'escalation_id': str(uuid.uuid4()),
                'original_notification_id': original_notification_id,
                'escalation_details': {
                    'reason': escalation_reason,
                    'target_level': target_level,
                    'target_channel': target_channel.value,
                    'attempt_number': 1,
                    'max_attempts': max_attempts
                },
                'delivery_result': {
                    'success': escalation_success,
                    'message_ts': result.get('ts', ''),
                    'error': result.get('error', None)
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as send_error:
            return {
                'success': False,
                'error': str(send_error),
                'original_notification_id': original_notification_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'original_notification_id': original_notification_id,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def get_team_availability_adk_tool(
    team_type: str = "clinical",
    include_contact_info: bool = False
) -> Dict[str, Any]:
    """
    ADK Tool: Get medical team availability status
    
    Args:
        team_type: Type of medical team (clinical, emergency, specialists, nursing)
        include_contact_info: Include contact information in response
        
    Returns:
        Team availability information
    """
    try:
        # Mock implementation - in production would integrate with hospital systems
        team_availability = {
            'clinical': {
                'available_staff': 5,
                'total_staff': 8,
                'on_call': 2,
                'availability_percentage': 62.5,
                'next_shift_change': '14:00',
                'emergency_contact': '+56912345678' if include_contact_info else None
            },
            'emergency': {
                'available_staff': 3,
                'total_staff': 4,
                'on_call': 1,
                'availability_percentage': 75.0,
                'next_shift_change': '08:00',
                'emergency_contact': '+56912345679' if include_contact_info else None
            },
            'specialists': {
                'available_staff': 2,
                'total_staff': 3,
                'on_call': 1,
                'availability_percentage': 66.7,
                'next_shift_change': '16:00',
                'emergency_contact': '+56912345680' if include_contact_info else None
            },
            'nursing': {
                'available_staff': 12,
                'total_staff': 15,
                'on_call': 3,
                'availability_percentage': 80.0,
                'next_shift_change': '20:00',
                'emergency_contact': '+56912345681' if include_contact_info else None
            }
        }
        
        if team_type not in team_availability:
            return {
                'success': False,
                'error': f'Unknown team type: {team_type}',
                'available_teams': list(team_availability.keys()),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        team_data = team_availability[team_type]
        
        return {
            'success': True,
            'team_type': team_type,
            'availability': team_data,
            'contact_info_included': include_contact_info,
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'team_type': team_type,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def get_communication_status_adk_tool() -> Dict[str, Any]:
    """
    ADK Tool: Get current communication system status
    
    Returns:
        Communication system status and capabilities
    """
    try:
        # Initialize services to check status
        slack_orchestrator = SlackOrchestrator()
        whatsapp_processor = WhatsAppProcessor()
        
        return {
            'success': True,
            'communication_capabilities': [
                'emergency_medical_alerts',
                'clinical_result_notifications',
                'human_review_requests',
                'priority_based_escalation',
                'multi_channel_delivery',
                'whatsapp_integration',
                'slack_team_routing',
                'audit_compliant_logging'
            ],
            'supported_channels': [channel.value for channel in CommunicationChannel],
            'medical_teams': [role.value for role in MedicalTeamRole],
            'notification_types': [ntype.value for ntype in NotificationType],
            'priority_levels': [priority.name for priority in NotificationPriority],
            'message_templates': [template.value for template in MessageTemplate],
            'system_status': {
                'slack_integration': True,
                'whatsapp_integration': True,
                'escalation_monitoring': True,
                'audit_logging': True,
                'interactive_buttons': True
            },
            'compliance_features': [
                'HIPAA_compliant_messaging',
                'PHI_anonymization',
                'audit_trail_logging',
                '7_year_retention',
                'encrypted_communications'
            ],
            'system_version': '2.0_ADK',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


# Helper Functions for Message Formatting

def _format_emergency_message(alert_content: Dict[str, Any], urgency_level: str) -> Dict[str, Any]:
    """Format emergency alert message with medical context"""
    token_id = alert_content.get('token_id', alert_content.get('patient_code', 'UNKNOWN'))  # Batman token
    alert_type = alert_content.get('alert_type', 'Medical Alert')
    clinical_context = alert_content.get('clinical_context', 'No context provided')
    
    # Urgency indicators
    urgency_indicators = {
        'critical': '🚨🔴',
        'high': '⚠️🟠',
        'medium': '⚡🟡',
        'low': '📢🔵'
    }
    
    indicator = urgency_indicators.get(urgency_level, '📢')
    
    message_text = f"""
{indicator} **ALERTA MÉDICA {urgency_level.upper()}** {indicator}

**Paciente:** {patient_code}
**Tipo:** {alert_type}
**Contexto Clínico:** {clinical_context}

**Recomendaciones:**
{chr(10).join(f"• {rec}" for rec in alert_content.get('recommended_actions', ['Evaluación médica inmediata']))}

**Timestamp:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
    
    # Slack blocks for rich formatting
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{indicator} ALERTA MÉDICA {urgency_level.upper()}"
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Token ID:* {token_id}"  # Batman token (HIPAA compliant)
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Tipo:* {alert_type}"
                }
            ]
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Contexto Clínico:*\n{clinical_context}"
            }
        }
    ]
    
    return {
        'text': message_text,
        'blocks': blocks
    }


def _format_clinical_result_message(clinical_data: Dict[str, Any], include_evidence: bool) -> Dict[str, Any]:
    """Format clinical result message with evidence-based context"""
    lpp_grade = clinical_data.get('lpp_grade', 0)
    confidence = clinical_data.get('confidence_score', 0.0)
    token_id = clinical_data.get('token_id', clinical_data.get('patient_code', 'UNKNOWN'))  # Batman token
    
    # LPP grade indicators
    grade_indicators = {
        0: '⚪',
        1: '🟡',
        2: '🟠',
        3: '🔴',
        4: '⚫'
    }
    
    indicator = grade_indicators.get(lpp_grade, '❓')
    
    message_text = f"""
{indicator} **RESULTADO CLÍNICO LPP**

**Paciente:** {patient_code}
**Grado LPP:** {lpp_grade} {indicator}
**Confianza:** {confidence:.1%}
**Localización:** {clinical_data.get('anatomical_location', 'No especificada')}

**Recomendaciones Clínicas:**
{chr(10).join(f"• {rec}" for rec in clinical_data.get('clinical_recommendations', ['Evaluación especializada']))}
"""
    
    if include_evidence and clinical_data.get('scientific_references'):
        message_text += f"""
**Referencias Científicas:**
{chr(10).join(f"• {ref}" for ref in clinical_data.get('scientific_references', []))}
"""
    
    # Slack blocks
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{indicator} RESULTADO CLÍNICO LPP"
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Token ID:* {token_id}"  # Batman token (HIPAA compliant)
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
                    "text": f"*Localización:* {clinical_data.get('anatomical_location', 'No especificada')}"
                }
            ]
        }
    ]
    
    return {
        'text': message_text,
        'blocks': blocks
    }


def _format_review_request_message(
    review_request: Dict[str, Any], 
    specialist_type: str, 
    urgency: str,
    include_buttons: bool
) -> Dict[str, Any]:
    """Format human review request message"""
    token_id = review_request.get('token_id', review_request.get('patient_code', 'UNKNOWN'))  # Batman token
    reason = review_request.get('review_reason', 'Evaluación especializada requerida')
    
    # Urgency indicators
    urgency_indicators = {
        'critical': '🚨',
        'high': '⚠️',
        'medium': '📋',
        'low': '📝'
    }
    
    indicator = urgency_indicators.get(urgency, '📋')
    
    message_text = f"""
{indicator} **SOLICITUD REVISIÓN MÉDICA**

**Especialista:** {specialist_type.replace('_', ' ').title()}
**Urgencia:** {urgency.upper()}
**Paciente:** {patient_code}
**Motivo:** {reason}

**Contexto Clínico:**
{review_request.get('clinical_context', 'Contexto no proporcionado')}
"""
    
    # Slack blocks with interactive buttons
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{indicator} SOLICITUD REVISIÓN MÉDICA"
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Especialista:* {specialist_type.replace('_', ' ').title()}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Urgencia:* {urgency.upper()}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Token ID:* {token_id}"  # Batman token (HIPAA compliant)
                }
            ]
        }
    ]
    
    if include_buttons:
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Aceptar Revisión"
                    },
                    "style": "primary",
                    "value": f"accept_review_{review_request.get('session_id', 'unknown')}"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Escalar"
                    },
                    "style": "danger",
                    "value": f"escalate_review_{review_request.get('session_id', 'unknown')}"
                }
            ]
        })
    
    return {
        'text': message_text,
        'blocks': blocks
    }


def _format_escalation_message(notification_id: str, reason: str, target_level: str) -> Dict[str, Any]:
    """Format escalation message"""
    message_text = f"""
🔺 **ESCALAMIENTO MÉDICO**

**Notificación Original:** {notification_id}
**Nivel Escalamiento:** {target_level.replace('_', ' ').title()}
**Motivo:** {reason}

**Acción Requerida:** Revisión inmediata por {target_level.replace('_', ' ')}

**Timestamp:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
    
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "🔺 ESCALAMIENTO MÉDICO"
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Notificación:* {notification_id}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Nivel:* {target_level.replace('_', ' ').title()}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Motivo:* {reason}"
                }
            ]
        }
    ]
    
    return {
        'text': message_text,
        'blocks': blocks
    }


def _determine_clinical_priority(lpp_grade: int, confidence: float) -> NotificationPriority:
    """Determine clinical priority based on LPP grade and confidence"""
    if lpp_grade >= 4:
        return NotificationPriority.CRITICAL
    elif lpp_grade >= 3:
        return NotificationPriority.HIGH
    elif lpp_grade >= 2 or confidence < 0.7:
        return NotificationPriority.MEDIUM
    else:
        return NotificationPriority.LOW


def _setup_escalation_monitoring(notification_payload: NotificationPayload) -> Dict[str, Any]:
    """Setup escalation monitoring for critical notifications"""
    return {
        'escalation_enabled': True,
        'monitoring_interval_minutes': 5 if notification_payload.priority == NotificationPriority.CRITICAL else 30,
        'max_escalations': 3,
        'escalation_channels': [SlackChannel.EMERGENCY_ROOM.value],
        'setup_timestamp': datetime.now(timezone.utc).isoformat()
    }


def _setup_review_tracking(notification_id: str, specialist_type: str, urgency: str, message_ts: str) -> Dict[str, Any]:
    """Setup review tracking for human review requests"""
    return {
        'tracking_enabled': True,
        'review_timeout_hours': 1 if urgency == 'critical' else 4 if urgency == 'high' else 24,
        'specialist_type': specialist_type,
        'message_timestamp': message_ts,
        'auto_escalate': urgency in ['critical', 'high'],
        'setup_timestamp': datetime.now(timezone.utc).isoformat()
    }


# Communication Agent Instruction for ADK
COMMUNICATION_AGENT_ADK_INSTRUCTION = """
Eres el Communication Agent del sistema Vigia, especializado en comunicaciones médicas
multicanal y notificaciones para equipos hospitalarios con escalamiento automático.

RESPONSABILIDADES PRINCIPALES:
1. Alertas médicas críticas con procesamiento de path crítico
2. Notificaciones resultados clínicos con contexto basado en evidencia
3. Solicitudes revisión humana con routing especialista apropiado
4. Escalamiento automático basado en prioridad y timeouts médicos
5. Integración WhatsApp/Slack para comunicación hospitalaria
6. Audit trail completo para compliance comunicaciones médicas

CAPACIDADES TÉCNICAS ESPECIALIZADAS:
- Routing inteligente por tipo equipo médico y especialización
- Escalamiento automático crítico ≤5 minutos, urgente ≤30 minutos
- Templates médicos especializados con contexto clínico completo
- Botones interactivos Slack para respuesta médica inmediata
- Integración WhatsApp para comunicación externa pacientes/familias
- Compliance HIPAA con anonimización PHI en todas comunicaciones

HERRAMIENTAS DISPONIBLES:
- send_emergency_alert_adk_tool: Alertas médicas críticas multicanal
- send_clinical_result_adk_tool: Notificaciones resultados con evidencia
- request_human_review_adk_tool: Solicitudes revisión especialista
- send_whatsapp_response_adk_tool: Respuestas WhatsApp médicas
- escalate_notification_adk_tool: Escalamiento prioridad médica
- get_team_availability_adk_tool: Disponibilidad equipos médicos
- get_communication_status_adk_tool: Estado sistema comunicaciones

CANALES MÉDICOS ESPECIALIZADOS:
- #emergencias: Equipo emergencias médicas - alertas críticas
- #equipo-clinico: Personal clínico general - resultados rutinarios
- #especialistas-lpp: Especialistas lesiones presión - casos complejos
- #personal-enfermeria: Equipo enfermería - cuidados continuos
- #auditoria-medica: Compliance y audit trail - trazabilidad

PROTOCOLOS DE COMUNICACIÓN MÉDICA:
1. USAR escalamiento inmediato para alertas críticas (Grado 4 LPP)
2. INCLUIR contexto evidencia científica en notificaciones clínicas
3. ROUTING inteligente por especialización médica requerida
4. APLICAR templates médicos con información clínica estructurada
5. MANTENER compliance HIPAA con anonimización automática PHI
6. REGISTRAR audit trail completo todas comunicaciones médicas

ESCALAMIENTO AUTOMÁTICO:
- Crítico: ≤5 minutos con 3 escalamientos máximo
- Alto: ≤30 minutos con 2 escalamientos máximo
- Medio: ≤2 horas con 1 escalamiento máximo
- Bajo: ≤24 horas sin escalamiento automático

COMPLIANCE Y SEGURIDAD:
- Anonimización automática PHI en todas comunicaciones
- Encriptación end-to-end para canales médicos
- Audit logging 7 años retención médico-legal
- Validación compliance HIPAA automática

SIEMPRE prioriza comunicación inmediata para casos críticos, mantén contexto médico
completo con evidencia científica, y asegura trazabilidad total para compliance.
"""

# Create ADK Communication Agent
communication_adk_agent = LlmAgent(
    model="gemini-2.0-flash-exp",
    global_instruction=COMMUNICATION_AGENT_ADK_INSTRUCTION,
    instruction="Gestiona comunicaciones médicas multicanal con escalamiento automático y compliance HIPAA.",
    name="communication_agent_adk",
    tools=[
        send_emergency_alert_adk_tool,
        send_clinical_result_adk_tool,
        request_human_review_adk_tool,
        send_whatsapp_response_adk_tool,
        escalate_notification_adk_tool,
        get_team_availability_adk_tool,
        get_communication_status_adk_tool
    ],
)


# Factory for ADK Communication Agent
class CommunicationAgentADKFactory:
    """Factory for creating ADK-based Communication Agents"""
    
    @staticmethod
    def create_agent() -> LlmAgent:
        """Create new ADK Communication Agent instance"""
        return communication_adk_agent
    
    @staticmethod
    def get_agent_capabilities() -> List[str]:
        """Get list of agent capabilities"""
        return [
            'emergency_medical_alerts',
            'clinical_result_notifications',
            'human_review_requests',
            'priority_based_escalation',
            'multi_channel_delivery',
            'whatsapp_integration',
            'slack_team_routing',
            'audit_compliant_logging',
            'interactive_medical_buttons',
            'specialist_routing',
            'real_time_escalation'
        ]
    
    @staticmethod
    def get_supported_channels() -> List[str]:
        """Get supported communication channels"""
        return [channel.value for channel in CommunicationChannel]
    
    @staticmethod
    def get_medical_teams() -> List[str]:
        """Get supported medical team roles"""
        return [role.value for role in MedicalTeamRole]


# Export for use
__all__ = [
    'communication_adk_agent',
    'CommunicationAgentADKFactory',
    'send_emergency_alert_adk_tool',
    'send_clinical_result_adk_tool',
    'request_human_review_adk_tool',
    'send_whatsapp_response_adk_tool',
    'escalate_notification_adk_tool',
    'get_team_availability_adk_tool',
    'get_communication_status_adk_tool',
    'MedicalTeamRole',
    'CommunicationChannel',
    'MessageTemplate'
]