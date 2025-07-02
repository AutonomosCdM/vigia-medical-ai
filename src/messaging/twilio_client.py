"""
Twilio Client for Vigia Medical System
Handles WhatsApp messaging and media downloading
"""

import os
import sys
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# Configure logging
logger = logging.getLogger('vigia.messaging.twilio_client')

class TwilioClient:
    """
    Twilio client wrapper for medical messaging compliance
    Provides fallback functionality when Twilio is not available
    """
    
    def __init__(self, account_sid: Optional[str] = None, auth_token: Optional[str] = None):
        """Initialize Twilio client with optional credentials."""
        self.account_sid = account_sid or os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = auth_token or os.getenv('TWILIO_AUTH_TOKEN')
        self.client = None
        self.available = False
        
        # Try to initialize Twilio client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Twilio client if credentials are available."""
        try:
            if self.account_sid and self.auth_token:
                from twilio.rest import Client
                self.client = Client(self.account_sid, self.auth_token)
                self.available = True
                logger.info("Twilio client initialized successfully")
            else:
                logger.warning("Twilio credentials not found, running in mock mode")
                self.available = False
        except ImportError as e:
            logger.warning(f"Twilio package not available: {e}")
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize Twilio client: {e}")
            self.available = False
    
    def download_media(self, media_url: str, auth_token: Optional[str] = None) -> Optional[bytes]:
        """
        Download media from Twilio media URL
        
        Args:
            media_url: The Twilio media URL
            auth_token: Optional auth token for request
            
        Returns:
            Media content as bytes or None if failed
        """
        try:
            import requests
            
            # Use provided auth token or instance token
            token = auth_token or self.auth_token
            
            if not token:
                logger.error("No auth token available for media download")
                return None
            
            # Make authenticated request to Twilio
            auth = (self.account_sid, token)
            response = requests.get(media_url, auth=auth, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"Successfully downloaded media from {media_url}")
                return response.content
            else:
                logger.error(f"Failed to download media: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading media: {e}")
            return None
    
    def send_message(self, to: str, body: str, from_: Optional[str] = None) -> bool:
        """
        Send WhatsApp message via Twilio
        
        Args:
            to: Recipient phone number (WhatsApp format)
            body: Message body
            from_: Sender number (optional, uses env var)
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.available:
            logger.warning("Twilio not available, message not sent")
            return False
        
        try:
            from_number = from_ or os.getenv('TWILIO_WHATSAPP_FROM', 'whatsapp:+14155238886')
            
            message = self.client.messages.create(
                body=body,
                from_=from_number,
                to=f"whatsapp:{to}" if not to.startswith('whatsapp:') else to
            )
            
            logger.info(f"Message sent successfully: {message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def send_media_message(self, to: str, body: str, media_url: str, from_: Optional[str] = None) -> bool:
        """
        Send WhatsApp message with media via Twilio
        
        Args:
            to: Recipient phone number
            body: Message body
            media_url: URL of media to send
            from_: Sender number (optional)
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.available:
            logger.warning("Twilio not available, media message not sent")
            return False
        
        try:
            from_number = from_ or os.getenv('TWILIO_WHATSAPP_FROM', 'whatsapp:+14155238886')
            
            message = self.client.messages.create(
                body=body,
                from_=from_number,
                to=f"whatsapp:{to}" if not to.startswith('whatsapp:') else to,
                media_url=[media_url]
            )
            
            logger.info(f"Media message sent successfully: {message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send media message: {e}")
            return False
    
    def validate_webhook(self, url: str, token: str) -> bool:
        """
        Validate Twilio webhook signature
        
        Args:
            url: Webhook URL
            token: Webhook token
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not self.available:
                # In development mode, always return True
                logger.warning("Twilio not available, skipping webhook validation")
                return True
            
            # Add actual webhook validation logic here
            # This is a placeholder implementation
            return bool(url and token)
            
        except Exception as e:
            logger.error(f"Webhook validation error: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if Twilio client is available and configured."""
        return self.available
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get Twilio account information."""
        if not self.available:
            return None
        
        try:
            account = self.client.api.accounts(self.account_sid).fetch()
            return {
                "sid": account.sid,
                "friendly_name": account.friendly_name,
                "status": account.status,
                "type": account.type
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None


# Default client instance
default_client = TwilioClient()

# Convenience functions using default client
def download_media(media_url: str, auth_token: Optional[str] = None) -> Optional[bytes]:
    """Download media using default Twilio client."""
    return default_client.download_media(media_url, auth_token)

def send_message(to: str, body: str, from_: Optional[str] = None) -> bool:
    """Send message using default Twilio client."""
    return default_client.send_message(to, body, from_)

def send_media_message(to: str, body: str, media_url: str, from_: Optional[str] = None) -> bool:
    """Send media message using default Twilio client."""
    return default_client.send_media_message(to, body, media_url, from_)

def is_available() -> bool:
    """Check if default Twilio client is available."""
    return default_client.is_available()