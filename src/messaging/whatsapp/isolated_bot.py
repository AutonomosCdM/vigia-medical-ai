"""
Isolated WhatsApp Bot for Vigia Medical System
==============================================

Isolated WhatsApp bot implementation with session management,
medical workflow handling, and HIPAA-compliant communication.

Key Features:
- Session management with timeouts
- Medical workflow orchestration
- PHI tokenization (Batman tokens)
- Error handling and fallbacks
- Audit trail logging
- Production webhook handling
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..whatsapp_processor import WhatsAppProcessor, MessageType, WhatsAppResponse
from ...core.phi_tokenization_client import PHITokenizationClient
from ...utils.audit_service import AuditService, AuditEventType
from ...utils.secure_logger import SecureLogger

logger = SecureLogger("whatsapp_bot")


class SessionState(Enum):
    """WhatsApp session states"""
    NEW = "new"
    ACTIVE = "active"
    WAITING_IMAGE = "waiting_image"
    PROCESSING = "processing"
    COMPLETED = "completed"
    EXPIRED = "expired"
    ERROR = "error"


class BotCommand(Enum):
    """Supported bot commands"""
    START = "start"
    HELP = "help"
    RESTART = "restart"
    STATUS = "status"
    CANCEL = "cancel"


@dataclass
class WhatsAppSession:
    """WhatsApp session management"""
    session_id: str
    phone_number: str
    batman_token: str  # PHI anonymized token
    state: str = SessionState.NEW.value
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    image_count: int = 0
    message_count: int = 0
    analysis_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session is expired"""
        timeout = timedelta(minutes=timeout_minutes)
        return datetime.now(timezone.utc) - self.last_activity > timeout
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            'session_id': self.session_id,
            'phone_number': self.phone_number,
            'batman_token': self.batman_token,
            'state': self.state,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'image_count': self.image_count,
            'message_count': self.message_count,
            'analysis_results_count': len(self.analysis_results),
            'metadata': self.metadata
        }


class IsolatedWhatsAppBot:
    """
    Isolated WhatsApp bot for medical communications.
    
    Capabilities:
    - Session management with timeout handling
    - Medical workflow orchestration
    - PHI tokenization for HIPAA compliance
    - Multi-step conversation handling
    - Error recovery and fallbacks
    - Comprehensive audit logging
    """
    
    def __init__(
        self,
        whatsapp_processor: Optional[WhatsAppProcessor] = None,
        session_timeout_minutes: int = 30,
        max_images_per_session: int = 5
    ):
        """Initialize isolated WhatsApp bot"""
        self.whatsapp_processor = whatsapp_processor or WhatsAppProcessor()
        self.phi_tokenizer = PHITokenizationClient()
        self.audit_service = AuditService()
        
        # Session management
        self.sessions: Dict[str, WhatsAppSession] = {}
        self.session_timeout_minutes = session_timeout_minutes
        self.max_images_per_session = max_images_per_session
        
        # Bot configuration
        self.bot_config = {
            'welcome_message_delay': 2,  # seconds
            'processing_notification_delay': 5,
            'session_cleanup_interval': 300,  # 5 minutes
            'max_concurrent_sessions': 100
        }
        
        # Command handlers
        self.command_handlers = {
            BotCommand.START.value: self._handle_start_command,
            BotCommand.HELP.value: self._handle_help_command,
            BotCommand.RESTART.value: self._handle_restart_command,
            BotCommand.STATUS.value: self._handle_status_command,
            BotCommand.CANCEL.value: self._handle_cancel_command
        }
        
        # Medical workflow callbacks (to be set by orchestrator)
        self.medical_analysis_callback: Optional[Callable] = None
        self.priority_escalation_callback: Optional[Callable] = None
        
        # Statistics
        self.stats = {
            'total_sessions': 0,
            'active_sessions': 0,
            'completed_analyses': 0,
            'expired_sessions': 0,
            'error_sessions': 0,
            'avg_session_duration': 0.0
        }
        
        # Start session cleanup task
        asyncio.create_task(self._session_cleanup_loop())
    
    async def process_incoming_message(
        self,
        phone_number: str,
        message_content: str,
        message_type: str = MessageType.TEXT.value,
        media_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process incoming WhatsApp message.
        
        Args:
            phone_number: Sender phone number
            message_content: Message text content
            message_type: Type of message (text, image, etc.)
            media_url: URL for media messages
            
        Returns:
            Processing result with session info
        """
        try:
            # Get or create session
            session = await self._get_or_create_session(phone_number)
            
            # Update session activity
            session.update_activity()
            session.message_count += 1
            
            # Check for bot commands
            command = self._extract_command(message_content)
            if command:
                return await self._handle_command(session, command, message_content)
            
            # Process based on message type and session state
            if message_type == MessageType.IMAGE.value:
                return await self._handle_image_message(session, media_url, message_content)
            else:
                return await self._handle_text_message(session, message_content)
                
        except Exception as e:
            logger.error(f"Error processing incoming message: {e}")
            
            # Send error message to user
            if phone_number:
                await self.whatsapp_processor.send_message({
                    'to': phone_number,
                    'type': MessageType.TEXT.value,
                    'text': {'body': 'Lo siento, ocurrió un error. Intenta nuevamente en unos minutos.'},
                    'token_id': 'error'
                })
            
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def send_medical_result(
        self,
        session_id: str,
        analysis_result: Dict[str, Any]
    ) -> WhatsAppResponse:
        """
        Send medical analysis result to patient.
        
        Args:
            session_id: WhatsApp session ID
            analysis_result: Medical analysis results
            
        Returns:
            WhatsApp response status
        """
        try:
            session = self.sessions.get(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            # Update session with result
            session.analysis_results.append(analysis_result)
            session.state = SessionState.COMPLETED.value
            session.update_activity()
            
            # Send result via WhatsApp
            response = await self.whatsapp_processor.send_medical_result(
                recipient=session.phone_number,
                analysis_result=analysis_result,
                token_id=session.batman_token
            )
            
            # Update statistics
            if response.success:
                self.stats['completed_analyses'] += 1
            
            # Log completion
            await self._log_session_event(
                session,
                "medical_result_sent",
                {
                    "analysis_result": {
                        "lpp_grade": analysis_result.get('lpp_grade', 0),
                        "confidence": analysis_result.get('confidence', 0.0),
                        "recommendations_count": len(analysis_result.get('medical_recommendations', []))
                    },
                    "delivery_success": response.success
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to send medical result: {e}")
            return WhatsAppResponse(
                success=False,
                error=str(e),
                timestamp=datetime.now(timezone.utc)
            )
    
    async def _get_or_create_session(self, phone_number: str) -> WhatsAppSession:
        """Get existing session or create new one"""
        # Look for existing active session
        for session in self.sessions.values():
            if (session.phone_number == phone_number and 
                not session.is_expired(self.session_timeout_minutes)):
                return session
        
        # Create new session
        session_id = str(uuid.uuid4())
        
        # Create Batman token for PHI anonymization
        batman_token = await self.phi_tokenizer.create_token_async(
            hospital_mrn=f"whatsapp_{phone_number[-4:]}",  # Use last 4 digits
            patient_data={'phone': phone_number, 'platform': 'whatsapp'}
        )
        
        session = WhatsAppSession(
            session_id=session_id,
            phone_number=phone_number,
            batman_token=batman_token
        )
        
        self.sessions[session_id] = session
        self.stats['total_sessions'] += 1
        self.stats['active_sessions'] += 1
        
        # Send welcome message
        await self._send_welcome_message(session)
        
        # Log session creation
        await self._log_session_event(
            session,
            "session_created",
            {"phone_number_hash": hash(phone_number)}
        )
        
        return session
    
    async def _handle_image_message(
        self,
        session: WhatsAppSession,
        media_url: Optional[str],
        caption: str
    ) -> Dict[str, Any]:
        """Handle incoming image message"""
        try:
            # Check session limits
            if session.image_count >= self.max_images_per_session:
                await self.whatsapp_processor.send_message({
                    'to': session.phone_number,
                    'type': MessageType.TEXT.value,
                    'text': {'body': 'Has alcanzado el límite de imágenes por sesión. Inicia una nueva sesión para continuar.'},
                    'token_id': session.batman_token
                })
                
                return {
                    'success': False,
                    'error': 'Session image limit reached',
                    'session_id': session.session_id
                }
            
            # Update session state
            session.state = SessionState.PROCESSING.value
            session.image_count += 1
            
            # Send processing confirmation
            await self.whatsapp_processor.send_analysis_request_confirmation(
                recipient=session.phone_number,
                token_id=session.batman_token
            )
            
            # Trigger medical analysis (if callback is set)
            if self.medical_analysis_callback and media_url:
                analysis_result = await self.medical_analysis_callback(
                    image_url=media_url,
                    session_id=session.session_id,
                    batman_token=session.batman_token,
                    patient_context={'caption': caption}
                )
                
                # Send result immediately
                await self.send_medical_result(session.session_id, analysis_result)
            else:
                # No callback - send placeholder response
                session.state = SessionState.ERROR.value
                await self.whatsapp_processor.send_message({
                    'to': session.phone_number,
                    'type': MessageType.TEXT.value,
                    'text': {'body': 'Sistema de análisis no disponible temporalmente. Intenta más tarde.'},
                    'token_id': session.batman_token
                })
            
            return {
                'success': True,
                'session_id': session.session_id,
                'batman_token': session.batman_token,
                'image_count': session.image_count,
                'processing_started': True
            }
            
        except Exception as e:
            logger.error(f"Error handling image message: {e}")
            session.state = SessionState.ERROR.value
            self.stats['error_sessions'] += 1
            
            return {
                'success': False,
                'error': str(e),
                'session_id': session.session_id
            }
    
    async def _handle_text_message(
        self,
        session: WhatsAppSession,
        message_content: str
    ) -> Dict[str, Any]:
        """Handle incoming text message"""
        try:
            # Simple responses based on session state
            if session.state == SessionState.NEW.value:
                session.state = SessionState.WAITING_IMAGE.value
                
                await self.whatsapp_processor.send_message({
                    'to': session.phone_number,
                    'type': MessageType.TEXT.value,
                    'text': {'body': '📸 Perfecto! Para comenzar el análisis médico, envía una foto clara de la zona a evaluar.'},
                    'token_id': session.batman_token
                })
                
            elif session.state == SessionState.WAITING_IMAGE.value:
                await self.whatsapp_processor.send_message({
                    'to': session.phone_number,
                    'type': MessageType.TEXT.value,
                    'text': {'body': 'Estoy esperando una imagen para analizar. Envía una foto de la zona médica a evaluar.'},
                    'token_id': session.batman_token
                })
                
            elif session.state == SessionState.PROCESSING.value:
                await self.whatsapp_processor.send_message({
                    'to': session.phone_number,
                    'type': MessageType.TEXT.value,
                    'text': {'body': '⏳ Estoy procesando tu imagen. El análisis estará listo en unos momentos.'},
                    'token_id': session.batman_token
                })
                
            else:
                # Default response
                await self.whatsapp_processor.send_help_message(
                    recipient=session.phone_number,
                    token_id=session.batman_token
                )
            
            return {
                'success': True,
                'session_id': session.session_id,
                'session_state': session.state,
                'response_sent': True
            }
            
        except Exception as e:
            logger.error(f"Error handling text message: {e}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session.session_id
            }
    
    async def _send_welcome_message(self, session: WhatsAppSession):
        """Send welcome message to new session"""
        try:
            await asyncio.sleep(self.bot_config['welcome_message_delay'])
            
            await self.whatsapp_processor.send_welcome_message(
                recipient=session.phone_number,
                token_id=session.batman_token
            )
            
            session.state = SessionState.ACTIVE.value
            
        except Exception as e:
            logger.error(f"Failed to send welcome message: {e}")
    
    def _extract_command(self, message_content: str) -> Optional[str]:
        """Extract bot command from message"""
        content = message_content.lower().strip()
        
        # Check for explicit commands
        if content in ['hola', 'start', 'comenzar', 'empezar']:
            return BotCommand.START.value
        elif content in ['ayuda', 'help', '?']:
            return BotCommand.HELP.value
        elif content in ['reiniciar', 'restart', 'nuevo']:
            return BotCommand.RESTART.value
        elif content in ['estado', 'status']:
            return BotCommand.STATUS.value
        elif content in ['cancelar', 'cancel', 'salir']:
            return BotCommand.CANCEL.value
        
        return None
    
    async def _handle_command(
        self,
        session: WhatsAppSession,
        command: str,
        message_content: str
    ) -> Dict[str, Any]:
        """Handle bot command"""
        try:
            handler = self.command_handlers.get(command)
            if handler:
                return await handler(session, message_content)
            else:
                return await self._handle_unknown_command(session, message_content)
                
        except Exception as e:
            logger.error(f"Error handling command {command}: {e}")
            return {
                'success': False,
                'error': str(e),
                'command': command
            }
    
    async def _handle_start_command(self, session: WhatsAppSession, content: str) -> Dict[str, Any]:
        """Handle start command"""
        session.state = SessionState.ACTIVE.value
        
        await self.whatsapp_processor.send_welcome_message(
            recipient=session.phone_number,
            token_id=session.batman_token
        )
        
        return {'success': True, 'command': 'start', 'session_reset': True}
    
    async def _handle_help_command(self, session: WhatsAppSession, content: str) -> Dict[str, Any]:
        """Handle help command"""
        await self.whatsapp_processor.send_help_message(
            recipient=session.phone_number,
            token_id=session.batman_token
        )
        
        return {'success': True, 'command': 'help'}
    
    async def _handle_restart_command(self, session: WhatsAppSession, content: str) -> Dict[str, Any]:
        """Handle restart command"""
        # Reset session state
        session.state = SessionState.NEW.value
        session.image_count = 0
        session.analysis_results.clear()
        session.update_activity()
        
        await self.whatsapp_processor.send_message({
            'to': session.phone_number,
            'type': MessageType.TEXT.value,
            'text': {'body': '🔄 Sesión reiniciada. ¡Comencemos de nuevo!'},
            'token_id': session.batman_token
        })
        
        return {'success': True, 'command': 'restart', 'session_reset': True}
    
    async def _handle_status_command(self, session: WhatsAppSession, content: str) -> Dict[str, Any]:
        """Handle status command"""
        status_message = f"""📊 **Estado de tu Sesión**

🆔 Sesión: {session.session_id[:8]}...
📸 Imágenes analizadas: {session.image_count}/{self.max_images_per_session}
💬 Mensajes: {session.message_count}
⏰ Activa desde: {session.created_at.strftime('%H:%M')}

Para continuar, envía una nueva imagen médica."""

        await self.whatsapp_processor.send_message({
            'to': session.phone_number,
            'type': MessageType.TEXT.value,
            'text': {'body': status_message},
            'token_id': session.batman_token
        })
        
        return {'success': True, 'command': 'status', 'session_info': session.to_dict()}
    
    async def _handle_cancel_command(self, session: WhatsAppSession, content: str) -> Dict[str, Any]:
        """Handle cancel command"""
        session.state = SessionState.EXPIRED.value
        
        await self.whatsapp_processor.send_message({
            'to': session.phone_number,
            'type': MessageType.TEXT.value,
            'text': {'body': '👋 Sesión finalizada. ¡Gracias por usar VIGIA! Envía "Hola" cuando quieras iniciar nuevamente.'},
            'token_id': session.batman_token
        })
        
        return {'success': True, 'command': 'cancel', 'session_ended': True}
    
    async def _handle_unknown_command(self, session: WhatsAppSession, content: str) -> Dict[str, Any]:
        """Handle unknown command"""
        await self.whatsapp_processor.send_message({
            'to': session.phone_number,
            'type': MessageType.TEXT.value,
            'text': {'body': '❓ No entendí ese comando. Envía "Ayuda" para ver las opciones disponibles.'},
            'token_id': session.batman_token
        })
        
        return {'success': True, 'command': 'unknown'}
    
    async def _session_cleanup_loop(self):
        """Background task to clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(self.bot_config['session_cleanup_interval'])
                await self._cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if session.is_expired(self.session_timeout_minutes):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            session = self.sessions.pop(session_id)
            self.stats['expired_sessions'] += 1
            self.stats['active_sessions'] -= 1
            
            # Send timeout message if recently expired
            if session.last_activity > datetime.now(timezone.utc) - timedelta(hours=1):
                await self.whatsapp_processor.send_message({
                    'to': session.phone_number,
                    'type': MessageType.TEXT.value,
                    'text': {'body': 'Tu sesión ha expirado por inactividad. Envía "Hola" para comenzar nuevamente.'},
                    'token_id': session.batman_token
                })
            
            # Log session cleanup
            await self._log_session_event(
                session,
                "session_expired",
                {"duration_minutes": (session.last_activity - session.created_at).total_seconds() / 60}
            )
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    async def _log_session_event(
        self,
        session: WhatsAppSession,
        event_type: str,
        details: Dict[str, Any]
    ):
        """Log session event for audit trail"""
        try:
            await self.audit_service.log_event(
                event_type=AuditEventType.SESSION_EVENT,
                component="whatsapp_bot",
                action=event_type,
                details={
                    "session_id": session.session_id,
                    "batman_token": session.batman_token,
                    "session_state": session.state,
                    "message_count": session.message_count,
                    "image_count": session.image_count,
                    **details
                }
            )
        except Exception as e:
            logger.warning(f"Session event logging failed: {e}")
    
    def set_medical_analysis_callback(self, callback: Callable):
        """Set medical analysis callback"""
        self.medical_analysis_callback = callback
    
    def set_priority_escalation_callback(self, callback: Callable):
        """Set priority escalation callback"""
        self.priority_escalation_callback = callback
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            'bot_statistics': self.stats,
            'active_sessions': len([s for s in self.sessions.values() if not s.is_expired()]),
            'total_sessions_stored': len(self.sessions),
            'session_timeout_minutes': self.session_timeout_minutes,
            'max_images_per_session': self.max_images_per_session,
            'whatsapp_available': self.whatsapp_processor.is_available()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get bot health status"""
        active_sessions = len([s for s in self.sessions.values() if not s.is_expired()])
        
        return {
            'bot_status': 'healthy',
            'whatsapp_processor_available': self.whatsapp_processor.is_available(),
            'active_sessions': active_sessions,
            'max_concurrent_sessions': self.bot_config['max_concurrent_sessions'],
            'session_capacity_used': (active_sessions / self.bot_config['max_concurrent_sessions']) * 100,
            'medical_callback_configured': self.medical_analysis_callback is not None,
            'last_health_check': datetime.now(timezone.utc).isoformat()
        }


# Factory function
def create_isolated_whatsapp_bot(
    whatsapp_processor: Optional[WhatsAppProcessor] = None,
    session_timeout_minutes: int = 30
) -> IsolatedWhatsAppBot:
    """Create isolated WhatsApp bot instance"""
    return IsolatedWhatsAppBot(whatsapp_processor, session_timeout_minutes)


# Export for use
__all__ = [
    'IsolatedWhatsAppBot',
    'WhatsAppSession',
    'SessionState',
    'BotCommand',
    'create_isolated_whatsapp_bot'
]