"""
Slack Notifier Refactored - Mock implementation for compatibility
================================================================
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SlackNotifierRefactored:
    """Mock refactored Slack notifier for compatibility"""
    
    def __init__(self):
        logger.info("SlackNotifierRefactored initialized (mock mode)")
    
    def send_notification(self, message: str, channel: str = "#vigia-alerts") -> Dict[str, Any]:
        """Send Slack notification"""
        logger.info(f"Slack notification to {channel}: {message[:50]}... (mock)")
        return {'success': True, 'message_id': 'mock_msg_id'}
    
    def send_medical_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send medical alert"""
        logger.info("Sending medical alert (mock)")
        return {'success': True, 'alert_id': 'mock_alert_id'}

# Legacy compatibility
SlackNotifier = SlackNotifierRefactored

__all__ = ['SlackNotifierRefactored', 'SlackNotifier']