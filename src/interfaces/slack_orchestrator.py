"""
Slack Orchestrator Interface - Mock implementation for compatibility
===================================================================
"""

import logging
from typing import Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)

class SlackNotificationPriority(Enum):
    """Slack notification priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

class MedicalAlert:
    """Medical alert data structure"""
    def __init__(self, severity: int, message: str, patient_id: str = None):
        self.severity = severity
        self.message = message
        self.patient_id = patient_id

class NotificationPayload:
    """Notification payload data structure"""
    def __init__(self, message: str, channel: str = None, priority: str = "medium"):
        self.message = message
        self.channel = channel
        self.priority = priority

class NotificationType(Enum):
    """Notification types"""
    ALERT = "alert"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    MEDICAL = "medical"

class SlackChannel(Enum):
    """Slack channel definitions"""
    ALERTS = "#vigia-alerts"
    MEDICAL = "#medical-team"
    GENERAL = "#general"
    URGENT = "#urgent-care"

class SlackMessage:
    """Slack message data structure"""
    def __init__(self, text: str, channel: str = None, priority: str = "medium", attachments: list = None):
        self.text = text
        self.channel = channel
        self.priority = priority
        self.attachments = attachments or []

# Legacy compatibility alias
NotificationPriority = SlackNotificationPriority

class SlackOrchestrator:
    """Mock Slack orchestrator for compatibility"""
    
    def __init__(self):
        logger.info("SlackOrchestrator initialized (mock mode)")
    
    def send_notification(self, message: str, priority: SlackNotificationPriority = SlackNotificationPriority.MEDIUM) -> Dict[str, Any]:
        """Send Slack notification"""
        logger.info(f"Slack notification [{priority.value}]: {message[:50]}... (mock)")
        return {'success': True, 'message_id': 'mock_msg_id'}
    
    def send_medical_alert(self, alert: MedicalAlert) -> Dict[str, Any]:
        """Send medical alert"""
        logger.info(f"Medical alert severity {alert.severity}: {alert.message[:50]}... (mock)")
        return {'success': True, 'alert_id': 'mock_alert_id'}

__all__ = ['SlackOrchestrator', 'SlackNotificationPriority', 'NotificationPriority', 'MedicalAlert', 'NotificationPayload', 'NotificationType', 'SlackChannel', 'SlackMessage']