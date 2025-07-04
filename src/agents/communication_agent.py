"""
Communication Agent - Agente ADK especializado en comunicaciones médicas
Implementa Agent Development Kit (ADK) para gestión de notificaciones y comunicaciones.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json

from src.agents.base_agent import BaseAgent, AgentMessage, AgentResponse

# Define missing AgentCapability enum
class AgentCapability(Enum):
    """Agent capabilities for medical processing"""
    IMAGE_ANALYSIS = "image_analysis"
    CLINICAL_ASSESSMENT = "clinical_assessment"
    PROTOCOL_CONSULTATION = "protocol_consultation"
    MEDICAL_COMMUNICATION = "medical_communication"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    LPP_DETECTION = "lpp_detection"
    EVIDENCE_BASED_DECISIONS = "evidence_based_decisions"
    HIPAA_COMPLIANCE = "hipaa_compliance"
    EMERGENCY_ALERTS = "emergency_alerts"
    TEAM_COORDINATION = "team_coordination"
    ESCALATION_MANAGEMENT = "escalation_management"
from src.interfaces.slack_orchestrator import (
    SlackOrchestrator, SlackNotificationPriority as NotificationPriority
)
# Import from updated slack orchestrator
from src.interfaces.slack_orchestrator import NotificationPayload, NotificationType, SlackChannel
from src.utils.secure_logger import SecureLogger

# AgentOps Monitoring Integration
from src.monitoring.agentops_client import AgentOpsClient
from src.monitoring.medical_telemetry import MedicalTelemetry
from src.monitoring.adk_wrapper import adk_agent_wrapper

logger = SecureLogger("communication_agent")


class CommunicationType(Enum):
    """Tipos de comunicación médica."""
    EMERGENCY_ALERT = "emergency_alert"
    CLINICAL_NOTIFICATION = "clinical_notification"
    TEAM_COORDINATION = "team_coordination"
    PATIENT_UPDATE = "patient_update"
    SYSTEM_ALERT = "system_alert"
    AUDIT_NOTIFICATION = "audit_notification"
    ESCALATION_REQUEST = "escalation_request"


class CommunicationChannel(Enum):
    """Canales de comunicación disponibles."""
    SLACK = "slack"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    INTERNAL_QUEUE = "internal_queue"


@dataclass
class CommunicationRequest:
    """Solicitud de comunicación médica."""
    communication_type: CommunicationType
    channel: CommunicationChannel
    recipients: List[str]
    subject: str
    content: Dict[str, Any]
    priority: str = "medium"
    language: str = "es"
    requires_acknowledgment: bool = False
    escalation_rules: Optional[Dict[str, Any]] = None


@dataclass
class CommunicationResult:
    """Resultado de envío de comunicación."""
    success: bool
    message_id: str
    channels_sent: List[str]
    delivery_status: Dict[str, Any]
    acknowledgments_received: int
    errors: List[str]
    next_actions: List[str]


class CommunicationAgent(BaseAgent):
    """
    Agent especializado en comunicaciones y notificaciones médicas.
    
    Responsabilidades:
    - Envío de notificaciones médicas a equipos
    - Coordinación de comunicaciones entre especialistas
    - Gestión de escalamientos automáticos
    - Integración con múltiples canales de comunicación
    """
    
    def __init__(self, 
                 agent_id: str = "communication_agent",
                 slack_orchestrator: Optional[SlackOrchestrator] = None):
        """
        Inicializar CommunicationAgent.
        
        Args:
            agent_id: Identificador único del agente
            slack_orchestrator: Orquestador de Slack
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="communication_agent"
        )
        
        # Additional properties for advanced agent
        self.name = "Communication Agent"
        self.description = "Agente especializado en comunicaciones y notificaciones médicas"
        self.capabilities = [
            AgentCapability.MEDICAL_COMMUNICATION,
            AgentCapability.EMERGENCY_ALERTS,
            AgentCapability.TEAM_COORDINATION,
            AgentCapability.ESCALATION_MANAGEMENT
        ]
        self.version = "1.0.0"
        
        # Orquestador de comunicaciones
        self.slack_orchestrator = slack_orchestrator or SlackOrchestrator()
        
        # Configuración de comunicaciones
        self.communication_config = {
            "max_retries": 3,
            "retry_delay": 5.0,  # segundos
            "acknowledgment_timeout": 300,  # 5 minutos
            "escalation_enabled": True,
            "supported_channels": [
                CommunicationChannel.SLACK,
                CommunicationChannel.INTERNAL_QUEUE
            ]
        }
        
        # Tracking de comunicaciones activas
        self.active_communications: Dict[str, CommunicationRequest] = {}
        self.pending_acknowledgments: Dict[str, datetime] = {}
        
        # AgentOps telemetry integration
        self.telemetry = MedicalTelemetry(
            agent_id=self.agent_id,
            agent_type="communication"
        )
        
        logger.audit("communication_agent_initialized", {
            "agent_id": self.agent_id,
            "capabilities": [cap.value for cap in self.capabilities],
            "supported_channels": [ch.value for ch in self.communication_config["supported_channels"]],
            "escalation_enabled": self.communication_config["escalation_enabled"],
            "telemetry_enabled": True
        })
    
    async def initialize(self) -> bool:
        """
        Inicializar el agente y sus dependencias.
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            # Inicializar orquestador de Slack si es necesario
            # (SlackOrchestrator no tiene método initialize explícito)
            
            # Marcar como inicializado
            self.is_initialized = True
            
            logger.audit("communication_agent_ready", {
                "agent_id": self.agent_id,
                "initialization_successful": True,
                "slack_orchestrator_ready": True
            })
            
            return True
            
        except Exception as e:
            logger.error("communication_agent_initialization_failed", {
                "agent_id": self.agent_id,
                "error": str(e)
            })
            return False
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """
        Procesar mensaje y generar comunicación médica.
        
        Args:
            message: Mensaje del agente
            
        Returns:
            AgentResponse: Respuesta del agente
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Validar mensaje
            if not self._validate_message(message):
                return self._create_error_response(
                    message, 
                    "Invalid message format for communication request"
                )
            
            # Extraer solicitud de comunicación
            comm_request = self._extract_communication_request(message)
            
            # Procesar comunicación
            comm_result = await self._process_communication(comm_request)
            
            # Generar respuesta
            response = await self._generate_communication_response(
                message,
                comm_request, 
                comm_result,
                start_time
            )
            
            # Registrar comunicación procesada
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.audit("communication_processed", {
                "agent_id": self.agent_id,
                "session_id": message.session_id,
                "communication_type": comm_request.communication_type.value,
                "channel": comm_request.channel.value,
                "success": comm_result.success,
                "channels_sent": len(comm_result.channels_sent),
                "processing_time": processing_time
            })
            
            return response
            
        except Exception as e:
            logger.error("communication_processing_failed", {
                "agent_id": self.agent_id,
                "session_id": message.session_id,
                "error": str(e)
            })
            
            return self._create_error_response(message, str(e))
    
    async def _process_communication(self, request: CommunicationRequest) -> CommunicationResult:
        """
        Procesar solicitud de comunicación.
        
        Args:
            request: Solicitud de comunicación
            
        Returns:
            CommunicationResult: Resultado del procesamiento
        """
        try:
            # Generar ID único para la comunicación
            message_id = f"{request.communication_type.value}_{datetime.now().timestamp()}"
            
            # Registrar comunicación activa
            self.active_communications[message_id] = request
            
            # Procesar según el canal
            if request.channel == CommunicationChannel.SLACK:
                result = await self._send_slack_communication(request, message_id)
            elif request.channel == CommunicationChannel.INTERNAL_QUEUE:
                result = await self._send_internal_communication(request, message_id)
            else:
                raise ValueError(f"Canal de comunicación no soportado: {request.channel}")
            
            # Configurar seguimiento de acknowledgments si es necesario
            if request.requires_acknowledgment:
                await self._setup_acknowledgment_tracking(message_id)
            
            return result
            
        except Exception as e:
            logger.error("communication_processing_failed", {
                "communication_type": request.communication_type.value,
                "channel": request.channel.value,
                "error": str(e)
            })
            
            return CommunicationResult(
                success=False,
                message_id="",
                channels_sent=[],
                delivery_status={"error": str(e)},
                acknowledgments_received=0,
                errors=[str(e)],
                next_actions=["Reintentar comunicación", "Verificar configuración"]
            )
    
    async def _send_slack_communication(self, 
                                      request: CommunicationRequest, 
                                      message_id: str) -> CommunicationResult:
        """
        Enviar comunicación a través de Slack.
        
        Args:
            request: Solicitud de comunicación
            message_id: ID del mensaje
            
        Returns:
            CommunicationResult: Resultado del envío
        """
        try:
            # Crear payload de notificación Slack
            notification_payload = await self._create_slack_payload(request, message_id)
            
            # Enviar a través del orquestador (sync method)
            slack_result = self.slack_orchestrator.send_notification(notification_payload)
            
            if slack_result["success"]:
                return CommunicationResult(
                    success=True,
                    message_id=message_id,
                    channels_sent=[f"slack_{ch}" for ch in slack_result.get("delivery_results", [])],
                    delivery_status=slack_result,
                    acknowledgments_received=0,
                    errors=[],
                    next_actions=["Monitorear respuestas"] if request.requires_acknowledgment else []
                )
            else:
                return CommunicationResult(
                    success=False,
                    message_id=message_id,
                    channels_sent=[],
                    delivery_status=slack_result,
                    acknowledgments_received=0,
                    errors=["Error enviando a Slack"],
                    next_actions=["Reintentar envío", "Usar canal alternativo"]
                )
                
        except Exception as e:
            logger.error("slack_communication_failed", {
                "message_id": message_id,
                "error": str(e)
            })
            
            return CommunicationResult(
                success=False,
                message_id=message_id,
                channels_sent=[],
                delivery_status={"error": str(e)},
                acknowledgments_received=0,
                errors=[str(e)],
                next_actions=["Verificar configuración Slack", "Usar canal alternativo"]
            )
    
    async def _send_internal_communication(self, 
                                         request: CommunicationRequest, 
                                         message_id: str) -> CommunicationResult:
        """
        Enviar comunicación interna (cola de mensajes).
        
        Args:
            request: Solicitud de comunicación
            message_id: ID del mensaje
            
        Returns:
            CommunicationResult: Resultado del envío
        """
        try:
            # Simular envío a cola interna
            internal_message = {
                "message_id": message_id,
                "communication_type": request.communication_type.value,
                "recipients": request.recipients,
                "subject": request.subject,
                "content": request.content,
                "priority": request.priority,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # En una implementación real, esto enviaría a Redis/RabbitMQ/etc.
            logger.info("internal_communication_queued", {
                "message_id": message_id,
                "recipients": len(request.recipients),
                "priority": request.priority
            })
            
            return CommunicationResult(
                success=True,
                message_id=message_id,
                channels_sent=["internal_queue"],
                delivery_status={"status": "queued", "recipients": len(request.recipients)},
                acknowledgments_received=0,
                errors=[],
                next_actions=["Monitorear cola de mensajes"]
            )
            
        except Exception as e:
            logger.error("internal_communication_failed", {
                "message_id": message_id,
                "error": str(e)
            })
            
            return CommunicationResult(
                success=False,
                message_id=message_id,
                channels_sent=[],
                delivery_status={"error": str(e)},
                acknowledgments_received=0,
                errors=[str(e)],
                next_actions=["Reintentar envío interno"]
            )
    
    async def _create_slack_payload(self, 
                                  request: CommunicationRequest, 
                                  message_id: str) -> NotificationPayload:
        """
        Crear payload de notificación para Slack.
        
        Args:
            request: Solicitud de comunicación
            message_id: ID del mensaje
            
        Returns:
            NotificationPayload: Payload para Slack
        """
        # Mapear tipo de comunicación a tipo de notificación Slack
        notification_type_map = {
            CommunicationType.EMERGENCY_ALERT: NotificationType.EMERGENCY_ALERT,
            CommunicationType.CLINICAL_NOTIFICATION: NotificationType.CLINICAL_RESULT,
            CommunicationType.TEAM_COORDINATION: NotificationType.TEAM_COORDINATION,
            CommunicationType.PATIENT_UPDATE: NotificationType.CLINICAL_RESULT,
            CommunicationType.SYSTEM_ALERT: NotificationType.SYSTEM_STATUS,
            CommunicationType.AUDIT_NOTIFICATION: NotificationType.AUDIT_ALERT,
            CommunicationType.ESCALATION_REQUEST: NotificationType.HUMAN_REVIEW_REQUEST
        }
        
        # Mapear prioridad a enum de Slack
        priority_map = {
            "low": NotificationPriority.LOW,
            "medium": NotificationPriority.MEDIUM,
            "high": NotificationPriority.HIGH,
            "critical": NotificationPriority.CRITICAL
        }
        
        notification_type = notification_type_map.get(
            request.communication_type, 
            NotificationType.CLINICAL_RESULT
        )
        
        priority = priority_map.get(request.priority.lower(), NotificationPriority.MEDIUM)
        
        # Determinar canales objetivo
        target_channels = self._determine_slack_channels(request.communication_type, priority)
        
        return NotificationPayload(
            notification_id=message_id,
            session_id=message_id,  # Usar message_id como session_id
            notification_type=notification_type,
            priority=priority,
            target_channels=target_channels,
            content=request.content,
            metadata={
                "communication_type": request.communication_type.value,
                "original_channel": request.channel.value,
                "recipients": request.recipients,
                "requires_acknowledgment": request.requires_acknowledgment,
                "subject": request.subject
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
            escalation_rules=request.escalation_rules
        )
    
    def _determine_slack_channels(self, 
                                comm_type: CommunicationType, 
                                priority: NotificationPriority) -> List[SlackChannel]:
        """
        Determinar canales Slack apropiados.
        
        Args:
            comm_type: Tipo de comunicación
            priority: Prioridad
            
        Returns:
            List[SlackChannel]: Lista de canales objetivo
        """
        # Mapeo básico por tipo de comunicación
        channel_map = {
            CommunicationType.EMERGENCY_ALERT: [SlackChannel.EMERGENCY_ROOM, SlackChannel.CLINICAL_TEAM],
            CommunicationType.CLINICAL_NOTIFICATION: [SlackChannel.CLINICAL_TEAM, SlackChannel.LPP_SPECIALISTS],
            CommunicationType.TEAM_COORDINATION: [SlackChannel.CLINICAL_TEAM, SlackChannel.NURSING_STAFF],
            CommunicationType.PATIENT_UPDATE: [SlackChannel.CLINICAL_TEAM],
            CommunicationType.SYSTEM_ALERT: [SlackChannel.SYSTEM_ALERTS],
            CommunicationType.AUDIT_NOTIFICATION: [SlackChannel.AUDIT_LOG],
            CommunicationType.ESCALATION_REQUEST: [SlackChannel.CLINICAL_TEAM, SlackChannel.EMERGENCY_ROOM]
        }
        
        base_channels = channel_map.get(comm_type, [SlackChannel.CLINICAL_TEAM])
        
        # Agregar canales adicionales para prioridades altas
        if priority in [NotificationPriority.CRITICAL, NotificationPriority.HIGH]:
            if SlackChannel.EMERGENCY_ROOM not in base_channels:
                base_channels.append(SlackChannel.EMERGENCY_ROOM)
        
        return base_channels
    
    def _extract_communication_request(self, message: AgentMessage) -> CommunicationRequest:
        """
        Extraer solicitud de comunicación del mensaje.
        
        Args:
            message: Mensaje del agente
            
        Returns:
            CommunicationRequest: Solicitud de comunicación
        """
        content = message.content
        
        # Determinar tipo de comunicación
        comm_type = self._classify_communication_type(content.get("text", ""))
        
        # Extraer canal (default: Slack)
        channel_str = content.get("channel", "slack")
        channel = CommunicationChannel(channel_str) if channel_str in [c.value for c in CommunicationChannel] else CommunicationChannel.SLACK
        
        # Extraer destinatarios
        recipients = content.get("recipients", ["equipo_clinico"])
        
        # Extraer asunto
        subject = content.get("subject", content.get("text", "Comunicación médica")[:50])
        
        # Extraer prioridad
        priority = content.get("priority", "medium")
        
        # Extraer idioma
        language = content.get("language", "es")
        
        # Extraer configuración de acknowledgment
        requires_ack = content.get("requires_acknowledgment", False)
        
        # Extraer reglas de escalamiento
        escalation_rules = content.get("escalation_rules")
        
        return CommunicationRequest(
            communication_type=comm_type,
            channel=channel,
            recipients=recipients,
            subject=subject,
            content=content,
            priority=priority,
            language=language,
            requires_acknowledgment=requires_ack,
            escalation_rules=escalation_rules
        )
    
    def _classify_communication_type(self, text: str) -> CommunicationType:
        """
        Clasificar tipo de comunicación basado en el texto.
        
        Args:
            text: Texto a clasificar
            
        Returns:
            CommunicationType: Tipo de comunicación
        """
        text_lower = text.lower()
        
        # Palabras clave por tipo
        type_keywords = {
            CommunicationType.EMERGENCY_ALERT: [
                "emergencia", "emergency", "crítico", "critical", "alerta", "alert"
            ],
            CommunicationType.CLINICAL_NOTIFICATION: [
                "resultado clínico", "clinical result", "diagnóstico", "diagnosis",
                "lpp", "pressure injury"
            ],
            CommunicationType.TEAM_COORDINATION: [
                "coordinación", "coordination", "equipo", "team", "reunión", "meeting"
            ],
            CommunicationType.PATIENT_UPDATE: [
                "paciente", "patient", "actualización", "update", "estado", "status"
            ],
            CommunicationType.SYSTEM_ALERT: [
                "sistema", "system", "error", "fallo", "maintenance", "mantenimiento"
            ],
            CommunicationType.AUDIT_NOTIFICATION: [
                "auditoría", "audit", "compliance", "cumplimiento", "log"
            ],
            CommunicationType.ESCALATION_REQUEST: [
                "escalamiento", "escalation", "escalate", "urgente", "urgent"
            ]
        }
        
        # Buscar coincidencias
        for comm_type, keywords in type_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return comm_type
        
        # Default
        return CommunicationType.CLINICAL_NOTIFICATION
    
    async def _setup_acknowledgment_tracking(self, message_id: str):
        """
        Configurar seguimiento de acknowledgments.
        
        Args:
            message_id: ID del mensaje
        """
        self.pending_acknowledgments[message_id] = datetime.now(timezone.utc)
        
        logger.audit("acknowledgment_tracking_setup", {
            "message_id": message_id,
            "timeout": self.communication_config["acknowledgment_timeout"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def _generate_communication_response(self, 
                                             message: AgentMessage,
                                             request: CommunicationRequest,
                                             result: CommunicationResult,
                                             start_time: datetime) -> AgentResponse:
        """
        Generar respuesta de comunicación.
        
        Args:
            message: Mensaje original
            request: Solicitud de comunicación
            result: Resultado de comunicación
            start_time: Tiempo de inicio
            
        Returns:
            AgentResponse: Respuesta del agente
        """
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        if result.success:
            # Comunicación exitosa
            response_content = {
                "communication_result": asdict(result),
                "status": "sent",
                "channels_delivered": result.channels_sent,
                "delivery_status": result.delivery_status,
                "next_actions": result.next_actions
            }
            
            success = True
            response_text = f"Comunicación enviada a {len(result.channels_sent)} canales"
            
        else:
            # Error en comunicación
            response_content = {
                "communication_result": asdict(result),
                "status": "failed",
                "errors": result.errors,
                "next_actions": result.next_actions,
                "retry_available": True
            }
            
            success = False
            response_text = f"Error enviando comunicación: {', '.join(result.errors)}"
        
        return AgentResponse(
            success=success,
            content=response_content,
            message=response_text,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "session_id": message.session_id,
                "agent_id": self.agent_id,
                "processing_time": processing_time,
                "communication_type": request.communication_type.value,
                "channel": request.channel.value,
                "recipients": len(request.recipients),
                "requires_acknowledgment": request.requires_acknowledgment,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            next_actions=result.next_actions,
            requires_human_review=not result.success
        )
    
    def _validate_message(self, message: AgentMessage) -> bool:
        """
        Validar formato del mensaje.
        
        Args:
            message: Mensaje a validar
            
        Returns:
            bool: True si el mensaje es válido
        """
        if not message.content:
            return False
        
        # Debe tener texto o contenido específico
        text = message.content.get("text", "")
        subject = message.content.get("subject", "")
        
        if not text and not subject:
            return False
        
        return True
    
    def _create_error_response(self, message: AgentMessage, error: str) -> AgentResponse:
        """
        Crear respuesta de error.
        
        Args:
            message: Mensaje original
            error: Descripción del error
            
        Returns:
            AgentResponse: Respuesta de error
        """
        return AgentResponse(
            success=False,
            content={
                "error": error,
                "status": "error",
                "retry_available": True
            },
            message=f"Error en comunicación: {error}",
            timestamp=datetime.now(timezone.utc),
            metadata={
                "session_id": message.session_id,
                "agent_id": self.agent_id,
                "error_occurred": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            next_actions=["Verificar configuración", "Reintentar comunicación"],
            requires_human_review=True
        )
    
    async def receive_acknowledgment(self, message_id: str, user_id: str) -> bool:
        """
        Recibir acknowledgment de mensaje.
        
        Args:
            message_id: ID del mensaje
            user_id: ID del usuario que confirma
            
        Returns:
            bool: True si se procesó correctamente
        """
        try:
            if message_id in self.pending_acknowledgments:
                # Registrar acknowledgment
                logger.audit("acknowledgment_received", {
                    "message_id": message_id,
                    "user_id": user_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                # Remover de pendientes
                del self.pending_acknowledgments[message_id]
                
                return True
            
            return False
            
        except Exception as e:
            logger.error("acknowledgment_processing_failed", {
                "message_id": message_id,
                "user_id": user_id,
                "error": str(e)
            })
            return False
    
    async def shutdown(self):
        """Cerrar agente y liberar recursos."""
        try:
            logger.audit("communication_agent_shutdown", {
                "agent_id": self.agent_id,
                "active_communications": len(self.active_communications),
                "pending_acknowledgments": len(self.pending_acknowledgments),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Limpiar comunicaciones activas
            self.active_communications.clear()
            self.pending_acknowledgments.clear()
            
            self.is_initialized = False
            
        except Exception as e:
            logger.error("communication_agent_shutdown_error", {
                "agent_id": self.agent_id,
                "error": str(e)
            })


# Factory para crear CommunicationAgent
class CommunicationAgentFactory:
    """Factory para crear instancias de CommunicationAgent."""
    
    @staticmethod
    async def create_agent(config: Optional[Dict[str, Any]] = None) -> CommunicationAgent:
        """
        Crear instancia de CommunicationAgent.
        
        Args:
            config: Configuración opcional
            
        Returns:
            CommunicationAgent: Instancia configurada
        """
        config = config or {}
        
        # Crear orquestador de Slack si se especifica
        slack_orchestrator = None
        if config.get("use_slack", True):
            slack_orchestrator = SlackOrchestrator()
        
        # Crear agente
        agent = CommunicationAgent(
            agent_id=config.get("agent_id", "communication_agent"),
            slack_orchestrator=slack_orchestrator
        )
        
        # Inicializar
        await agent.initialize()
        
        return agent