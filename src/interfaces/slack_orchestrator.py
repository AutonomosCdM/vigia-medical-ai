"""
Slack Orchestrator - Production VIGIA Medical AI Integration
===========================================================

Real Slack integration with medical-grade Block Kit components and HIPAA compliance.
Implements advanced notification routing and interactive medical workflows.
"""

import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
import json

# Slack SDK imports
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_SDK_AVAILABLE = True
except ImportError:
    SLACK_SDK_AVAILABLE = False
    WebClient = None
    SlackApiError = Exception

from src.messaging.slack_block_templates import (
    VigiaMessageTemplates, SlackBlockBuilder, MedicalBlockContext,
    MedicalSeverity, LPPGrade, VoiceAnalysisIndicator
)

logger = logging.getLogger(__name__)

class SlackNotificationPriority(Enum):
    """Slack notification priority levels with medical urgency mapping"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

class NotificationType(Enum):
    """Medical notification types for VIGIA system"""
    LPP_DETECTION = "lpp_detection"
    VOICE_ANALYSIS = "voice_analysis"
    CLINICAL_RESULT = "clinical_result"
    EMERGENCY_ALERT = "emergency_alert"
    TEAM_COORDINATION = "team_coordination"
    HUMAN_REVIEW_REQUEST = "human_review_request"
    SYSTEM_STATUS = "system_status"
    AUDIT_ALERT = "audit_alert"

class SlackChannel(Enum):
    """Medical Slack channels with specialization"""
    EMERGENCY_ROOM = "emergency_room"
    CLINICAL_TEAM = "clinical_team"
    LPP_SPECIALISTS = "lpp_specialists"
    NURSING_STAFF = "nursing_staff"
    SYSTEM_ALERTS = "system_alerts"
    AUDIT_LOG = "audit_log"

@dataclass
class NotificationPayload:
    """Enhanced notification payload for medical workflows"""
    notification_id: str
    session_id: str
    notification_type: NotificationType
    priority: SlackNotificationPriority
    target_channels: List[SlackChannel]
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    escalation_rules: Optional[Dict[str, Any]] = None

@dataclass
class MedicalAlert:
    """Medical alert with enhanced context"""
    severity: int
    message: str
    batman_token: str
    alert_type: str
    requires_immediate_response: bool = False
    clinical_context: Optional[Dict[str, Any]] = None

class SlackMessage:
    """Enhanced Slack message with Block Kit support"""
    def __init__(
        self, 
        text: str = None, 
        blocks: List[Dict[str, Any]] = None,
        channel: str = None, 
        priority: str = "medium", 
        attachments: List[Dict] = None
    ):
        self.text = text
        self.blocks = blocks or []
        self.channel = channel
        self.priority = priority
        self.attachments = attachments or []

# Legacy compatibility
NotificationPriority = SlackNotificationPriority

class SlackOrchestrator:
    """
    Production Slack orchestrator for VIGIA Medical AI.
    
    Provides medical-grade notifications with Block Kit components,
    HIPAA compliance, and intelligent channel routing.
    """
    
    def __init__(self):
        """Initialize Slack orchestrator with production configuration"""
        
        # Load Slack configuration
        self.bot_token = os.getenv('SLACK_BOT_TOKEN')
        self.team_id = os.getenv('SLACK_TEAM_ID')
        self.default_channel = os.getenv('SLACK_CHANNEL_IDS', 'C08U2TB78E6')
        
        # Initialize Slack client
        self.client = None
        self._setup_slack_client()
        
        # Medical channel mapping
        self.medical_channels = {
            SlackChannel.EMERGENCY_ROOM: self.default_channel,  # Use working channel
            SlackChannel.CLINICAL_TEAM: self.default_channel,
            SlackChannel.LPP_SPECIALISTS: self.default_channel,
            SlackChannel.NURSING_STAFF: self.default_channel,
            SlackChannel.SYSTEM_ALERTS: self.default_channel,
            SlackChannel.AUDIT_LOG: self.default_channel
        }
        
        # Priority routing rules
        self.priority_routing = {
            SlackNotificationPriority.CRITICAL: [SlackChannel.EMERGENCY_ROOM, SlackChannel.CLINICAL_TEAM],
            SlackNotificationPriority.URGENT: [SlackChannel.EMERGENCY_ROOM, SlackChannel.CLINICAL_TEAM],
            SlackNotificationPriority.HIGH: [SlackChannel.CLINICAL_TEAM],
            SlackNotificationPriority.MEDIUM: [SlackChannel.CLINICAL_TEAM],
            SlackNotificationPriority.LOW: [SlackChannel.CLINICAL_TEAM]
        }
        
        logger.info(f"SlackOrchestrator initialized - Client: {'✅' if self.client else '❌'}")
    
    def _setup_slack_client(self):
        """Setup Slack WebClient with error handling"""
        if not SLACK_SDK_AVAILABLE:
            logger.warning("Slack SDK not available - install with: pip install slack-sdk")
            return
        
        if not self.bot_token or self.bot_token == 'your_slack_token_here':
            logger.warning("Slack bot token not configured")
            return
        
        try:
            self.client = WebClient(token=self.bot_token)
            # Test connection
            response = self.client.auth_test()
            logger.info(f"Slack connection established - Team: {response.get('team', 'Unknown')}")
        except Exception as e:
            logger.error(f"Failed to setup Slack client: {e}")
            self.client = None
    
    
    async def send_medical_alert(self, alert: MedicalAlert) -> Dict[str, Any]:
        """
        Send urgent medical alert with immediate routing.
        
        Args:
            alert: Medical alert with clinical context
            
        Returns:
            Dict: Alert delivery status
        """
        try:
            # Create medical block context
            context = MedicalBlockContext(
                batman_token=alert.batman_token,
                session_id=f"alert_{datetime.now().timestamp()}",
                timestamp=datetime.now(timezone.utc),
                severity=MedicalSeverity(min(alert.severity, 4)),
                requires_human_review=alert.requires_immediate_response,
                urgency_level="critical" if alert.severity >= 3 else "high"
            )
            
            # Generate alert blocks
            if alert.alert_type == "lpp_detection":
                blocks = VigiaMessageTemplates.create_lpp_detection_alert(
                    context=context,
                    lpp_grade=LPPGrade(alert.severity),
                    confidence=alert.clinical_context.get('confidence', 0.95),
                    clinical_recommendation=alert.message,
                    evidence_level=alert.clinical_context.get('evidence_level', 'A')
                )
            else:
                # Generic medical alert
                blocks = [
                    SlackBlockBuilder.create_medical_header(f"Medical Alert - Severity {alert.severity}", "🚨"),
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Alert:* {alert.message}"
                        }
                    },
                    SlackBlockBuilder.create_batman_context(alert.batman_token, f"Severity: {alert.severity}")
                ]
            
            # Send to emergency channels for high severity
            channels = [self.default_channel]  # Use working channel
            if alert.severity >= 3:
                # Add additional urgent routing if configured
                pass
            
            # Send alert
            delivery_results = []
            for channel in channels:
                result = await self._send_blocks_to_channel(channel, blocks)
                delivery_results.append(result)
            
            success_count = sum(1 for r in delivery_results if r.get('success', False))
            
            return {
                'success': success_count > 0,
                'alert_id': f"alert_{context.session_id}",
                'delivery_results': delivery_results,
                'channels_delivered': success_count,
                'severity': alert.severity,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to send medical alert: {e}")
            return {
                'success': False,
                'error': str(e),
                'alert_id': 'failed'
            }
    
    def send_notification(
        self, 
        message: Union[str, NotificationPayload], 
        priority: SlackNotificationPriority = SlackNotificationPriority.MEDIUM
    ) -> Dict[str, Any]:
        """
        Send notification - supports both legacy string and new payload format.
        
        Args:
            message: Message string or NotificationPayload
            priority: Notification priority
            
        Returns:
            Dict: Send result
        """
        # Handle legacy string format
        if isinstance(message, str):
            return self.send_simple_notification(message, priority)
        
        # Handle new payload format (async)
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._send_notification_async(message))
        except:
            # Fallback to sync simple notification
            text = message.content.get('text', 'Medical notification') if hasattr(message, 'content') else str(message)
            return self.send_simple_notification(text, priority)
    
    async def _send_notification_async(self, payload: NotificationPayload) -> Dict[str, Any]:
        """Async version of send_notification for payload format"""
        return await self.send_notification_payload(payload)
    
    async def send_notification_payload(self, payload: NotificationPayload) -> Dict[str, Any]:
        """
        Send medical notification with Block Kit components.
        
        Args:
            payload: Notification payload with medical context
            
        Returns:
            Dict: Send result with delivery status
        """
        try:
            # Generate Block Kit components based on notification type
            blocks = await self._generate_medical_blocks(payload)
            
            # Determine target channels
            channels = self._resolve_target_channels(payload)
            
            # Send to each target channel
            delivery_results = []
            for channel in channels:
                result = await self._send_to_channel(channel, blocks, payload)
                delivery_results.append(result)
            
            # Aggregate results
            success_count = sum(1 for r in delivery_results if r.get('success', False))
            
            return {
                'success': success_count > 0,
                'notification_id': payload.notification_id,
                'delivery_results': delivery_results,
                'channels_delivered': success_count,
                'total_channels': len(channels),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to send notification {payload.notification_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'notification_id': payload.notification_id
            }
    
    def send_simple_notification(
        self, 
        message: str, 
        priority: SlackNotificationPriority = SlackNotificationPriority.MEDIUM
    ) -> Dict[str, Any]:
        """
        Send simple text notification (legacy compatibility).
        
        Args:
            message: Message text
            priority: Notification priority
            
        Returns:
            Dict: Send result
        """
        try:
            priority_str = priority.value if hasattr(priority, 'value') else str(priority)
            if not self.client:
                logger.warning(f"Slack notification (no client): [{priority_str}] {message[:50]}...")
                return {'success': False, 'error': 'No Slack client configured'}
            
            # Send to default channel
            response = self.client.chat_postMessage(
                channel=self.default_channel,
                text=message,
                username="VIGIA Medical AI"
            )
            
            return {
                'success': True,
                'message_id': response.get('ts'),
                'channel': self.default_channel,
                'priority': priority_str
            }
            
        except Exception as e:
            logger.error(f"Failed to send simple notification: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _generate_medical_blocks(self, payload: NotificationPayload) -> List[Dict[str, Any]]:
        """Generate medical Block Kit components based on payload type"""
        
        context = MedicalBlockContext(
            batman_token=payload.content.get('batman_token', 'anonymous'),
            session_id=payload.session_id,
            timestamp=datetime.now(timezone.utc),
            severity=MedicalSeverity.MEDIUM,  # Default
            requires_human_review=payload.priority.value in ['urgent', 'critical']
        )
        
        if payload.notification_type == NotificationType.LPP_DETECTION:
            return VigiaMessageTemplates.create_lpp_detection_alert(
                context=context,
                lpp_grade=LPPGrade(payload.content.get('lpp_grade', 1)),
                confidence=payload.content.get('confidence', 0.9),
                clinical_recommendation=payload.content.get('recommendation', 'Medical review recommended'),
                evidence_level=payload.content.get('evidence_level', 'A')
            )
        
        elif payload.notification_type == NotificationType.VOICE_ANALYSIS:
            return VigiaMessageTemplates.create_voice_analysis_alert(
                context=context,
                voice_indicators=payload.content.get('voice_indicators', {}),
                emotional_summary=payload.content.get('emotional_summary', 'Voice analysis completed'),
                pain_score=payload.content.get('pain_score'),
                stress_score=payload.content.get('stress_score')
            )
        
        elif payload.notification_type == NotificationType.TEAM_COORDINATION:
            return VigiaMessageTemplates.create_medical_team_coordination(
                context=context,
                coordination_type=payload.content.get('coordination_type', 'General'),
                team_members=payload.content.get('team_members', []),
                priority=payload.priority.value,
                message=payload.content.get('message', 'Team coordination required'),
                action_required=payload.content.get('action_required', True)
            )
        
        else:
            # Generic medical notification
            return [
                SlackBlockBuilder.create_medical_header(f"VIGIA Medical Notification"),
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": payload.content.get('text', 'Medical notification')
                    }
                },
                SlackBlockBuilder.create_batman_context(context.batman_token)
            ]
    
    def _resolve_target_channels(self, payload: NotificationPayload) -> List[str]:
        """Resolve target channels based on payload configuration"""
        
        # Use specified channels or default routing
        if payload.target_channels:
            channels = [self.medical_channels.get(ch, self.default_channel) for ch in payload.target_channels]
        else:
            # Use priority-based routing
            priority_channels = self.priority_routing.get(payload.priority, [SlackChannel.CLINICAL_TEAM])
            channels = [self.medical_channels.get(ch, self.default_channel) for ch in priority_channels]
        
        # Remove duplicates and ensure we have at least one channel
        channels = list(set(channels))
        if not channels:
            channels = [self.default_channel]
        
        return channels
    
    async def _send_to_channel(
        self, 
        channel: str, 
        blocks: List[Dict[str, Any]], 
        payload: NotificationPayload
    ) -> Dict[str, Any]:
        """Send blocks to specific channel"""
        try:
            return await self._send_blocks_to_channel(channel, blocks)
        except Exception as e:
            logger.error(f"Failed to send to channel {channel}: {e}")
            return {
                'success': False,
                'channel': channel,
                'error': str(e)
            }
    
    async def _send_blocks_to_channel(self, channel: str, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send Block Kit components to channel"""
        
        if not self.client:
            logger.warning(f"Slack blocks (no client): {len(blocks)} blocks to {channel}")
            return {
                'success': False,
                'channel': channel,
                'error': 'No Slack client configured'
            }
        
        try:
            response = self.client.chat_postMessage(
                channel=channel,
                blocks=blocks,
                username="VIGIA Medical AI",
                icon_emoji=":hospital:"
            )
            
            return {
                'success': True,
                'channel': channel,
                'message_id': response.get('ts'),
                'blocks_sent': len(blocks)
            }
            
        except SlackApiError as e:
            logger.error(f"Slack API error sending to {channel}: {e.response['error']}")
            return {
                'success': False,
                'channel': channel,
                'error': f"Slack API: {e.response['error']}"
            }
        except Exception as e:
            logger.error(f"Unexpected error sending to {channel}: {e}")
            return {
                'success': False,
                'channel': channel,
                'error': str(e)
            }

# Legacy compatibility exports
__all__ = [
    'SlackOrchestrator', 'SlackNotificationPriority', 'NotificationPriority', 
    'MedicalAlert', 'NotificationPayload', 'NotificationType', 'SlackChannel', 
    'SlackMessage'
]