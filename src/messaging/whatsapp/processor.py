"""
WhatsApp Processor - Mock implementation for compatibility
=========================================================
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class WhatsAppProcessor:
    """Mock WhatsApp processor for compatibility"""
    
    def __init__(self):
        logger.info("WhatsAppProcessor initialized (mock mode)")
    
    def send_message(self, recipient: str, message: str) -> Dict[str, Any]:
        """Send WhatsApp message"""
        logger.info(f"WhatsApp to {recipient}: {message[:50]}... (mock)")
        return {'success': True, 'message_id': 'mock_wa_msg_id'}
    
    def send_media_message(self, recipient: str, media_url: str, caption: str = None) -> Dict[str, Any]:
        """Send WhatsApp media message"""
        logger.info(f"WhatsApp media to {recipient}: {media_url} (mock)")
        return {'success': True, 'message_id': 'mock_wa_media_id'}

__all__ = ['WhatsAppProcessor']